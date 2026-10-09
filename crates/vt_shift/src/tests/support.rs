//! Test support: synthetic vowels with known pitch and formants, an offline
//! f64 pitch tracker, LPC envelopes, a click detector and an engine-like
//! driver for the shifter.

use std::f64::consts::PI;

use realfft::RealFftPlanner;
use vt_core::{EngineKind, LatencyMode, QualityTier, ShiftControl, VoiceRange};

use crate::lpc::levinson;
use crate::{Shifter, ShifterConfig, latency_samples};

pub const SR: f32 = 48_000.0;

/// Male /a/ (Peterson-Barney-like F1-F5 with bandwidths), Hz.
pub const VOWEL_A: [(f64, f64); 5] = [(730.0, 80.0), (1090.0, 90.0), (2440.0, 120.0), (3400.0, 150.0), (4500.0, 200.0)];
/// Female /i/.
pub const VOWEL_I: [(f64, f64); 5] =
    [(310.0, 60.0), (2790.0, 120.0), (3310.0, 150.0), (4200.0, 200.0), (5000.0, 250.0)];

/// Synthetic vowel: a band-limited glottal pulse train (all harmonics in
/// cosine phase, so every period starts with a closure at phase 0, tilted
/// −6 dB/oct above 500 Hz, tapered to zero at `band`) through a cascade of
/// formant resonators. `f0(n)` gives the pitch in Hz at sample n; the phase
/// is integrated per sample, so the true f0 trajectory is exact.
pub fn vowel(n: usize, f0: impl Fn(usize) -> f64, formants: &[(f64, f64)], band: f64) -> Vec<f32> {
    let sr = SR as f64;
    let mut out = vec![0.0f64; n];
    let mut ph = 0.0f64;
    for (i, o) in out.iter_mut().enumerate() {
        let f = f0(i);
        let th = 2.0 * PI * ph;
        // cos(kθ) by the Chebyshev recurrence.
        let c1 = th.cos();
        let (mut cm, mut c) = (1.0f64, c1);
        let mut acc = 0.0;
        let mut k = 1usize;
        loop {
            let fk = k as f64 * f;
            if fk >= band || k > 2000 {
                break;
            }
            acc += source_amp(fk, band) * c;
            let cn = 2.0 * c1 * c - cm;
            cm = c;
            c = cn;
            k += 1;
        }
        *o = acc;
        ph = (ph + f / sr).fract();
    }
    for &(fc, bw) in formants {
        let r = (-PI * bw / sr).exp();
        let a1 = 2.0 * r * (2.0 * PI * fc / sr).cos();
        let a2 = -r * r;
        let g = 1.0 - r;
        let (mut y1, mut y2) = (0.0, 0.0);
        for v in out.iter_mut() {
            let y = g * *v + a1 * y1 + a2 * y2;
            y2 = y1;
            y1 = y;
            *v = y;
        }
    }
    let peak = out.iter().fold(0.0f64, |m, &v| m.max(v.abs())).max(1e-12);
    out.iter().map(|&v| (0.5 * v / peak) as f32).collect()
}

/// Glottal source amplitude of a harmonic at `f` Hz: −6 dB/oct above
/// 500 Hz, raised-cosine taper over the top 20 % of the band.
fn source_amp(f: f64, band: f64) -> f64 {
    let tilt = 1.0 / (1.0 + (f / 500.0).powi(2)).sqrt();
    let x = (f / band - 0.8) / 0.2;
    let taper = if x <= 0.0 { 1.0 } else { 0.5 + 0.5 * (PI * x.min(1.0)).cos() };
    tilt * taper
}

/// Exact spectral envelope (dB) of [`vowel`] at `f` Hz: source amplitude ×
/// the formant cascade's magnitude response.
pub fn vowel_env_db(f: f64, formants: &[(f64, f64)], band: f64) -> f64 {
    let sr = SR as f64;
    let om = 2.0 * PI * f / sr;
    let mut mag = source_amp(f, band);
    for &(fc, bw) in formants {
        let r = (-PI * bw / sr).exp();
        let a1 = 2.0 * r * (2.0 * PI * fc / sr).cos();
        let a2 = -r * r;
        // |1 − a1 e^{−jω} − a2 e^{−2jω}|
        let re = 1.0 - a1 * om.cos() - a2 * (2.0 * om).cos();
        let im = a1 * om.sin() + a2 * (2.0 * om).sin();
        mag *= (1.0 - r) / (re * re + im * im).sqrt();
    }
    20.0 * mag.max(1e-30).log10()
}

/// Amplitudes (dB) of the harmonics `k·f0` of `x` in `[f_lo, f_hi]`, from a
/// Blackman-Harris-windowed DFT evaluated at the exact harmonic frequencies.
/// Returns (frequency, dB) pairs.
pub fn harmonic_amps_db(x: &[f32], f0: f64, f_lo: f64, f_hi: f64) -> Vec<(f64, f64)> {
    let n = x.len();
    let w: Vec<f64> = (0..n)
        .map(|i| {
            let t = 2.0 * PI * i as f64 / n as f64;
            x[i] as f64 * (0.35875 - 0.48829 * t.cos() + 0.14128 * (2.0 * t).cos() - 0.01168 * (3.0 * t).cos())
        })
        .collect();
    let mut out = Vec::new();
    let mut k = (f_lo / f0).ceil().max(1.0) as usize;
    while (k as f64) * f0 <= f_hi {
        let f = k as f64 * f0;
        let om = 2.0 * PI * f / SR as f64;
        // Rotating phasor (re-normalised every 1024 steps).
        let (c, s) = (om.cos(), om.sin());
        let (mut pr, mut pi) = (1.0f64, 0.0f64);
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &v) in w.iter().enumerate() {
            re += v * pr;
            im -= v * pi;
            let (nr, ni) = (pr * c - pi * s, pr * s + pi * c);
            pr = nr;
            pi = ni;
            if i % 1024 == 1023 {
                let m = (pr * pr + pi * pi).sqrt();
                pr /= m;
                pi /= m;
            }
        }
        out.push((f, 10.0 * (re * re + im * im).max(1e-30).log10()));
        k += 1;
    }
    out
}

/// True-envelope log-spectral distance (dB): RMS over the harmonics of `y`
/// (pitch `f0`) in `[f_lo, f_hi]` of their level minus the known envelope at
/// those frequencies, after removing the mean difference. Free of the
/// harmonic-sampling bias of envelope estimators (an ideal shifter scores 0).
pub fn true_env_lsd_db(y: &[f32], f0: f64, formants: &[(f64, f64)], band: f64, f_lo: f64, f_hi: f64) -> f64 {
    let h = harmonic_amps_db(y, f0, f_lo, f_hi);
    let d: Vec<f64> = h.iter().map(|&(f, a)| a - vowel_env_db(f, formants, band)).collect();
    let mean = d.iter().sum::<f64>() / d.len().max(1) as f64;
    (d.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d.len().max(1) as f64).sqrt()
}

/// Observable-envelope LSD (dB): the output's harmonic levels (pitch
/// `f_out`) against the input's harmonic levels (pitch `f_in`) linearly
/// interpolated in dB at the output frequencies, over `[f_lo, f_hi]`, mean
/// removed. The envelope between input harmonics cannot be known from the
/// input, so this is what an ideal formant-preserving shifter can achieve
/// (≈ 0 dB), whatever the pitch.
pub fn observed_env_lsd_db(x: &[f32], f_in: f64, y: &[f32], f_out: f64, f_lo: f64, f_hi: f64) -> f64 {
    let hin = harmonic_amps_db(x, f_in, f_in, f_hi + f_in);
    let hout = harmonic_amps_db(y, f_out, f_lo, f_hi);
    let mut d = Vec::new();
    for &(f, a) in &hout {
        // Bracketing input harmonics.
        let Some(i) = hin.iter().position(|h| h.0 >= f) else { continue };
        let ref_db = if i == 0 {
            hin[0].1
        } else {
            let (f0, a0) = hin[i - 1];
            let (f1, a1) = hin[i];
            a0 + (a1 - a0) * (f - f0) / (f1 - f0)
        };
        d.push(a - ref_db);
    }
    let mean = d.iter().sum::<f64>() / d.len().max(1) as f64;
    (d.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d.len().max(1) as f64).sqrt()
}

/// Steady vowel at `f0` Hz, band-limited to 10 kHz.
pub fn steady(n: usize, f0: f64, formants: &[(f64, f64)]) -> Vec<f32> {
    vowel(n, |_| f0, formants, 10_000.0)
}

/// Deterministic white noise in [−a, a].
pub fn noise(n: usize, a: f32, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            a * (2.0 * ((s >> 40) as f32 / (1u64 << 24) as f32) - 1.0)
        })
        .collect()
}

/// Offline f64 NSDF (McLeod) pitch of one frame: the first key maximum above
/// 0.9 × the highest one, refined by a parabola. `None` if unpitched.
pub fn nsdf_f0(x: &[f32], f_lo: f64, f_hi: f64) -> Option<f64> {
    let sr = SR as f64;
    let n = x.len();
    let t_lo = (sr / f_hi).floor().max(2.0) as usize;
    let t_hi = ((sr / f_lo).ceil() as usize).min(n / 2);
    if t_hi <= t_lo + 2 {
        return None;
    }
    let xs: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let mut nsdf = vec![0.0f64; t_hi + 2];
    for (tau, v) in nsdf.iter_mut().enumerate().skip(t_lo.saturating_sub(1)) {
        let (mut r, mut m) = (0.0, 0.0);
        for j in 0..n - tau {
            r += xs[j] * xs[j + tau];
            m += xs[j] * xs[j] + xs[j + tau] * xs[j + tau];
        }
        *v = if m > 0.0 { 2.0 * r / m } else { 0.0 };
    }
    // Key maxima: the highest point of each positive lobe in range.
    let mut peaks: Vec<(usize, f64)> = Vec::new();
    let mut tau = t_lo;
    while tau <= t_hi {
        if nsdf[tau] > 0.0 {
            let mut best = (tau, nsdf[tau]);
            while tau <= t_hi && nsdf[tau] > 0.0 {
                if nsdf[tau] > best.1 {
                    best = (tau, nsdf[tau]);
                }
                tau += 1;
            }
            peaks.push(best);
        }
        tau += 1;
    }
    let gmax = peaks.iter().fold(0.0f64, |m, p| m.max(p.1));
    if gmax < 0.5 {
        return None;
    }
    let &(k, _) = peaks.iter().find(|p| p.1 >= 0.9 * gmax)?;
    if k == 0 || k + 1 >= nsdf.len() {
        return Some(sr / k as f64);
    }
    let (a, b, c) = (nsdf[k - 1], nsdf[k], nsdf[k + 1]);
    let den = a - 2.0 * b + c;
    let d = if den.abs() > 1e-15 { 0.5 * (a - c) / den } else { 0.0 };
    Some(sr / (k as f64 + d.clamp(-1.0, 1.0)))
}

/// Median NSDF pitch over frames of `frame` samples hopped by `hop`.
pub fn median_f0(x: &[f32], f_lo: f64, f_hi: f64, frame: usize, hop: usize) -> Option<f64> {
    let mut v: Vec<f64> = (0..)
        .map(|i| i * hop)
        .take_while(|&s| s + frame <= x.len())
        .filter_map(|s| nsdf_f0(&x[s..s + frame], f_lo, f_hi))
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

/// Cents from `b` to `a`.
pub fn cents(a: f64, b: f64) -> f64 {
    1200.0 * (a / b).log2()
}

/// LPC envelope in dB on `bins` frequencies spaced `df` Hz from 0, from a
/// Hann-windowed segment (autocorrelation method).
pub fn lpc_env_db(x: &[f32], order: usize, bins: usize, df: f64) -> Vec<f64> {
    let n = x.len();
    let w: Vec<f64> = (0..n).map(|i| x[i] as f64 * (PI * (i as f64 + 0.5) / n as f64).sin().powi(2)).collect();
    let mut r = vec![0.0f64; order + 1];
    for (k, rk) in r.iter_mut().enumerate() {
        *rk = (0..n - k).map(|j| w[j] * w[j + k]).sum();
    }
    r[0] *= 1.0 + 1e-9;
    let mut a = vec![0.0f64; order + 1];
    let err = levinson(&r, &mut a).unwrap_or(1.0).max(1e-30);
    (0..bins)
        .map(|b| {
            let om = 2.0 * PI * b as f64 * df / SR as f64;
            let (mut re, mut im) = (0.0, 0.0);
            for (k, &ak) in a.iter().enumerate() {
                re += ak * (om * k as f64).cos();
                im -= ak * (om * k as f64).sin();
            }
            10.0 * (err / (re * re + im * im).max(1e-30)).log10()
        })
        .collect()
}

/// Log-spectral distance (dB) between two envelopes over `[f_lo, f_hi]`,
/// after removing their mean level difference.
pub fn lsd_db(a: &[f64], b: &[f64], df: f64, f_lo: f64, f_hi: f64) -> f64 {
    let idx: Vec<usize> =
        (0..a.len().min(b.len())).filter(|&i| (i as f64 * df) >= f_lo && (i as f64 * df) <= f_hi).collect();
    let mean = idx.iter().map(|&i| a[i] - b[i]).sum::<f64>() / idx.len() as f64;
    (idx.iter().map(|&i| (a[i] - b[i] - mean).powi(2)).sum::<f64>() / idx.len() as f64).sqrt()
}

/// Click detector: per 256-sample frame (hop 128, Blackman-Harris window),
/// energy above `f_hp` relative to the signal's mean frame energy, in dB;
/// returns the worst frame. Discontinuities (clicks, splices) are broadband
/// and stand out against a band-limited voice.
pub fn hf_click_db(x: &[f32], f_hp: f64) -> f64 {
    hf_click_at(x, f_hp).0
}

/// [`hf_click_db`] plus the start sample of the worst frame.
pub fn hf_click_at(x: &[f32], f_hp: f64) -> (f64, usize) {
    let n = 256;
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let win: Vec<f64> = (0..n)
        .map(|i| {
            let t = 2.0 * PI * i as f64 / n as f64;
            0.35875 - 0.48829 * t.cos() + 0.14128 * (2.0 * t).cos() - 0.01168 * (3.0 * t).cos()
        })
        .collect();
    let mut inb = fft.make_input_vec();
    let mut spec = fft.make_output_vec();
    let k_hp = (f_hp / (SR as f64 / n as f64)).ceil() as usize;
    let mut frames = Vec::new();
    let mut s = 0;
    while s + n <= x.len() {
        for i in 0..n {
            inb[i] = x[s + i] as f64 * win[i];
        }
        fft.process(&mut inb, &mut spec).unwrap();
        let total: f64 = spec.iter().map(|c| c.norm_sqr()).sum();
        let hf: f64 = spec.iter().skip(k_hp).map(|c| c.norm_sqr()).sum();
        frames.push((total, hf));
        s += n / 2;
    }
    let mean_total = frames.iter().map(|f| f.0).sum::<f64>() / frames.len().max(1) as f64;
    let (wi, worst) =
        frames.iter().enumerate().fold((0usize, 0.0f64), |m, (i, f)| if f.1 > m.1 { (i, f.1) } else { m });
    (10.0 * (worst.max(1e-30) / mean_total.max(1e-30)).log10(), wi * n / 2)
}

/// Energy (dB) of `x` in the band `[lo, hi]` Hz, from one Hann-windowed FFT
/// of the whole slice (length rounded down to even).
pub fn band_energy_db(x: &[f32], lo: f64, hi: f64) -> f64 {
    let n = x.len() & !1;
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut inb: Vec<f64> = (0..n).map(|i| x[i] as f64 * (PI * (i as f64 + 0.5) / n as f64).sin().powi(2)).collect();
    let mut spec = fft.make_output_vec();
    fft.process(&mut inb, &mut spec).unwrap();
    let df = SR as f64 / n as f64;
    let e: f64 = spec
        .iter()
        .enumerate()
        .filter(|(k, _)| (*k as f64 * df) >= lo && (*k as f64 * df) <= hi)
        .map(|(_, c)| c.norm_sqr())
        .sum();
    10.0 * e.max(1e-30).log10()
}

/// Configuration for a test render at 48 kHz.
pub fn config(mode: LatencyMode, range: VoiceRange, channels: usize) -> ShifterConfig {
    ShifterConfig {
        sample_rate: SR,
        channels,
        hop: vt_core::hop_for_rate(SR),
        latency_mode: mode,
        voice_range: range,
        latency: latency_samples(SR, mode, range, 10.0, 40.0),
    }
}

/// Builds a shifter with the given engine and tier.
pub fn shifter(mode: LatencyMode, range: VoiceRange, channels: usize, kind: EngineKind, tier: QualityTier) -> Shifter {
    let mut s = Shifter::new(config(mode, range, channels));
    s.set_engine(kind, tier);
    // Skip the engine crossfade so tests see the requested engine from 0.
    s.fade = None;
    s
}

/// Voiced control with ratio `r` and the true period hint.
pub fn voiced(r: f32, period: f32) -> ShiftControl {
    ShiftControl { ratio: r, formant_ratio: 1.0, voiced: true, period_hint: period, confidence: 1.0 }
}

/// Drives `sh` like `vt_engine` does: input split at hop boundaries,
/// `set_control(ctl(hop_index))` at every boundary, and within each hop the
/// host blocks given by `blocks` (cycled; any sizes). Returns the output
/// channels.
pub fn render(
    sh: &mut Shifter,
    input: &[&[f32]],
    analysis: &[f32],
    mut ctl: impl FnMut(usize) -> ShiftControl,
    blocks: &[usize],
) -> Vec<Vec<f32>> {
    let hop = sh.hop;
    let n = analysis.len();
    let mut out: Vec<Vec<f32>> = input.iter().map(|c| c.to_vec()).collect();
    let mut pos = 0usize;
    let mut bi = 0usize;
    while pos < n {
        if pos.is_multiple_of(hop) {
            sh.set_control(&ctl(pos / hop));
        }
        let want = blocks[bi % blocks.len()].max(1);
        bi += 1;
        let to_boundary = hop - pos % hop;
        let len = want.min(to_boundary).min(n - pos);
        let mut chans: Vec<&mut [f32]> = out.iter_mut().map(|c| &mut c[pos..pos + len]).collect();
        sh.process(&analysis[pos..pos + len], &mut chans);
        pos += len;
    }
    out
}

/// Mono render with a fixed control.
pub fn render_mono(sh: &mut Shifter, x: &[f32], ctl: ShiftControl) -> Vec<f32> {
    render(sh, &[x], x, |_| ctl, &[64]).remove(0)
}

/// Best lag of `y` against `x` (y[n] ≈ x[n − lag]) in `[lo, hi]` by
/// normalised cross-correlation over `[from, to)` of y.
pub fn best_lag(x: &[f32], y: &[f32], lo: usize, hi: usize, from: usize, to: usize) -> (usize, f64) {
    let mut best = (lo, f64::MIN);
    for lag in lo..=hi {
        let (mut xy, mut xx, mut yy) = (0.0f64, 0.0f64, 0.0f64);
        for n in from..to {
            let a = x[n - lag] as f64;
            let b = y[n] as f64;
            xy += a * b;
            xx += a * a;
            yy += b * b;
        }
        let c = xy / (xx * yy).sqrt().max(1e-30);
        if c > best.1 {
            best = (lag, c);
        }
    }
    best
}

/// Segmental SNR (dB, mean over 10 ms segments) of `y` against `x` delayed
/// by `lag`, over `[from, to)`.
pub fn seg_snr_db(x: &[f32], y: &[f32], lag: usize, from: usize, to: usize) -> f64 {
    let seg = 480;
    let mut v = Vec::new();
    let mut s = from;
    while s + seg <= to {
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for n in s..s + seg {
            let r = x[n - lag] as f64;
            sig += r * r;
            err += (y[n] as f64 - r).powi(2);
        }
        v.push(10.0 * (sig / err.max(1e-30)).log10().min(100.0));
        s += seg;
    }
    v.iter().sum::<f64>() / v.len().max(1) as f64
}
