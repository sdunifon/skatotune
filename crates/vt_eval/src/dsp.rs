//! Small f64 DSP toolkit for the offline harness (filters, statistics,
//! band-limited interpolation, FFT correlation).
//!
//! Everything here runs offline on whole signals, in f64, and favours accuracy
//! over speed. It deliberately does not depend on `vt_core`'s realtime kernels:
//! the harness must stay an independent yardstick for the code it measures.

use std::f64::consts::PI;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// RBJ-cookbook biquad in transposed direct form II, f64 coefficients and state.
#[derive(Clone, Copy, Debug)]
pub struct Biquad64 {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad64 {
    fn from_raw(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Normalised angular frequency and alpha for the cookbook formulas. The
    /// frequency is clamped just below Nyquist so callers cannot build an
    /// unstable filter at low sample rates.
    fn w0_alpha(sr: f64, f: f64, q: f64) -> (f64, f64) {
        let f = f.clamp(1e-3, 0.4999 * sr);
        let w0 = 2.0 * PI * f / sr;
        (w0, w0.sin() / (2.0 * q.max(1e-3)))
    }

    pub fn lowpass(sr: f64, f: f64, q: f64) -> Self {
        let (w0, alpha) = Self::w0_alpha(sr, f, q);
        let c = w0.cos();
        Self::from_raw(
            (1.0 - c) / 2.0,
            1.0 - c,
            (1.0 - c) / 2.0,
            1.0 + alpha,
            -2.0 * c,
            1.0 - alpha,
        )
    }

    pub fn highpass(sr: f64, f: f64, q: f64) -> Self {
        let (w0, alpha) = Self::w0_alpha(sr, f, q);
        let c = w0.cos();
        Self::from_raw(
            (1.0 + c) / 2.0,
            -(1.0 + c),
            (1.0 + c) / 2.0,
            1.0 + alpha,
            -2.0 * c,
            1.0 - alpha,
        )
    }

    /// Band-pass with 0 dB peak gain at `f`.
    pub fn bandpass(sr: f64, f: f64, q: f64) -> Self {
        let (w0, alpha) = Self::w0_alpha(sr, f, q);
        let c = w0.cos();
        Self::from_raw(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Butterworth Q for the two sections of a 4th-order cascade.
pub const BUTTER4_Q: [f64; 2] = [0.541_196_100_146_197, 1.306_562_964_876_376_8];

/// Zero-phase filtering (forward then backward) with a freshly reset copy of
/// `proto` per pass, so the magnitude response is squared and the phase is zero.
pub fn filtfilt(proto: &[Biquad64], x: &[f64]) -> Vec<f64> {
    let mut y = x.to_vec();
    for pass in 0..2 {
        let mut sections: Vec<Biquad64> = proto.to_vec();
        for s in &mut sections {
            s.reset();
        }
        let run = |v: &mut f64, sections: &mut [Biquad64]| {
            let mut acc = *v;
            for s in sections.iter_mut() {
                acc = s.process(acc);
            }
            *v = acc;
        };
        if pass == 0 {
            for v in y.iter_mut() {
                run(v, &mut sections);
            }
        } else {
            for v in y.iter_mut().rev() {
                run(v, &mut sections);
            }
        }
    }
    y
}

/// [`filtfilt`] on an odd (point-reflected) extension of `x` by `pad` samples
/// at both ends, cropped back: start/end transients of a high-pass are
/// suppressed because the extension continues value and slope (as SciPy's
/// `filtfilt` padding does).
pub fn filtfilt_padded(proto: &[Biquad64], x: &[f64], pad: usize) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return filtfilt(proto, x);
    }
    let pad = pad.min(n - 1);
    let (first, last) = (x[0], x[n - 1]);
    let mut ext = Vec::with_capacity(n + 2 * pad);
    ext.extend((1..=pad).rev().map(|i| 2.0 * first - x[i]));
    ext.extend_from_slice(x);
    ext.extend((1..=pad).map(|i| 2.0 * last - x[n - 1 - i]));
    let y = filtfilt(proto, &ext);
    y[pad..pad + n].to_vec()
}

/// Percentile `p` in [0, 100] of an already **sorted** slice, linearly
/// interpolated between order statistics. NaN for an empty slice.
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        n => {
            let rank = (p.clamp(0.0, 100.0) / 100.0) * (n - 1) as f64;
            let lo = rank.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = rank - lo as f64;
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    }
}

/// Sort a copy (NaNs dropped) and take percentile `p`.
pub fn percentile(values: &[f64], p: f64) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    v.sort_by(f64::total_cmp);
    percentile_sorted(&v, p)
}

pub fn median(values: &[f64]) -> f64 {
    percentile(values, 50.0)
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

pub fn db(power_ratio: f64) -> f64 {
    10.0 * power_ratio.max(1e-300).log10()
}

pub fn next_pow2(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

/// Modified Bessel function of the first kind, order 0 (power series; converges
/// fast for the β ≤ 20 used by Kaiser windows).
pub fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..200 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// Kaiser-windowed sinc interpolator for uniformly sampled sequences.
///
/// Used by the reference tracker to read the correlation function between
/// integer lags (Boersma-style sinc interpolation) and by the alignment helpers
/// to read signals at fractional delays.
#[derive(Clone, Debug)]
pub struct KaiserSinc {
    half_width: usize,
    /// Kaiser window w(r) for r = |d|/half_width in [0, 1], tabulated at
    /// `WINDOW_TABLE` + 1 points (linear interpolation error < 1e-7).
    window: Vec<f64>,
}

const WINDOW_TABLE: usize = 8192;

impl KaiserSinc {
    /// `half_width` taps on each side; β = 8.6 gives about −90 dB sidelobes,
    /// so band-limited content below ~0.45·fs interpolates to ~1e-4 or better.
    pub fn new(half_width: usize, beta: f64) -> Self {
        let inv_i0 = 1.0 / bessel_i0(beta);
        let window = (0..=WINDOW_TABLE)
            .map(|i| {
                let r = i as f64 / WINDOW_TABLE as f64;
                bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) * inv_i0
            })
            .collect();
        Self {
            half_width: half_width.max(1),
            window,
        }
    }

    pub fn half_width(&self) -> usize {
        self.half_width
    }

    #[inline]
    fn kernel(&self, d: f64) -> f64 {
        let r = d.abs() / self.half_width as f64;
        if r >= 1.0 {
            return 0.0;
        }
        let x = r * WINDOW_TABLE as f64;
        let i = x as usize;
        let frac = x - i as f64;
        let w0 = self.window.get(i).copied().unwrap_or(0.0);
        let w1 = self.window.get(i + 1).copied().unwrap_or(0.0);
        let w = w0 + (w1 - w0) * frac;
        let sinc = if d.abs() < 1e-12 {
            1.0
        } else {
            (PI * d).sin() / (PI * d)
        };
        sinc * w
    }

    /// Value of `y` at fractional index `t` (samples outside `y` count as 0).
    pub fn interp(&self, y: &[f64], t: f64) -> f64 {
        if !t.is_finite() {
            return 0.0;
        }
        let base = t.floor();
        let hw = self.half_width as i64;
        let b = base as i64;
        let mut acc = 0.0;
        for m in (b - hw + 1)..=(b + hw) {
            if m < 0 {
                continue;
            }
            let Some(v) = y.get(m as usize) else { continue };
            acc += v * self.kernel(t - m as f64);
        }
        acc
    }
}

/// FFT plans + scratch for linear cross-correlation of real f64 sequences.
pub struct Correlator64 {
    n: usize,
    fwd: Arc<dyn RealToComplex<f64>>,
    inv: Arc<dyn ComplexToReal<f64>>,
    buf_a: Vec<f64>,
    buf_b: Vec<f64>,
    spec_a: Vec<Complex<f64>>,
    spec_b: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    out: Vec<f64>,
}

impl Correlator64 {
    /// Correlator for FFT size `n` (rounded up to a power of two).
    pub fn new(n: usize) -> Self {
        let n = next_pow2(n.max(2));
        let mut planner = RealFftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let scratch_len = fwd.get_scratch_len().max(inv.get_scratch_len());
        Self {
            n,
            buf_a: vec![0.0; n],
            buf_b: vec![0.0; n],
            spec_a: fwd.make_output_vec(),
            spec_b: fwd.make_output_vec(),
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            out: vec![0.0; n],
            fwd,
            inv,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `out[τ] = Σ_j a[j]·b[j+τ]` for τ in `0..out.len()` (linear, zero padded).
    /// Requires `a.len() + out.len() <= n` and `b.len() <= n` to avoid circular
    /// wrap; inputs are truncated to `n` otherwise.
    pub fn cross_correlate(&mut self, a: &[f64], b: &[f64], out: &mut [f64]) {
        let n = self.n;
        self.buf_a.fill(0.0);
        self.buf_b.fill(0.0);
        let la = a.len().min(n);
        let lb = b.len().min(n);
        self.buf_a[..la].copy_from_slice(&a[..la]);
        self.buf_b[..lb].copy_from_slice(&b[..lb]);
        // Lengths are fixed by construction, so these cannot fail; on the
        // impossible error path the output is left zeroed.
        let ok_a =
            self.fwd
                .process_with_scratch(&mut self.buf_a, &mut self.spec_a, &mut self.scratch);
        let ok_b =
            self.fwd
                .process_with_scratch(&mut self.buf_b, &mut self.spec_b, &mut self.scratch);
        if ok_a.is_err() || ok_b.is_err() {
            out.fill(0.0);
            return;
        }
        // conj(A)·B ⇔ Σ a[j] b[j+τ]. DC and Nyquist bins stay exactly real.
        for (sa, sb) in self.spec_a.iter_mut().zip(self.spec_b.iter()) {
            *sa = sa.conj() * sb;
        }
        if let Some(first) = self.spec_a.first_mut() {
            first.im = 0.0;
        }
        if let Some(last) = self.spec_a.last_mut() {
            last.im = 0.0;
        }
        if self
            .inv
            .process_with_scratch(&mut self.spec_a, &mut self.out, &mut self.scratch)
            .is_err()
        {
            out.fill(0.0);
            return;
        }
        let scale = 1.0 / n as f64;
        for (o, v) in out.iter_mut().zip(self.out.iter()) {
            *o = v * scale;
        }
        if out.len() > n {
            out[n..].fill(0.0);
        }
    }
}

/// Windowed-sinc low-pass + integer decimation (zero phase, f64). Cutoff at
/// 0.45·(fs/factor). Used to analyse formant envelopes in a speech-like band.
pub fn decimate(x: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 {
        return x.to_vec();
    }
    let half = 24 * factor;
    let fc = 0.45 / factor as f64; // cycles per input sample
    let beta = 8.0;
    let inv_i0 = 1.0 / bessel_i0(beta);
    let taps: Vec<f64> = (0..=2 * half)
        .map(|i| {
            let d = i as f64 - half as f64;
            let r = d / half as f64;
            let sinc = if d == 0.0 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * d).sin() / (PI * d)
            };
            sinc * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) * inv_i0
        })
        .collect();
    let gain: f64 = taps.iter().sum();
    let out_len = x.len().div_ceil(factor);
    let mut out = Vec::with_capacity(out_len);
    for k in 0..out_len {
        let centre = (k * factor) as i64;
        let mut acc = 0.0;
        for (i, t) in taps.iter().enumerate() {
            let idx = centre + i as i64 - half as i64;
            if idx >= 0
                && let Some(v) = x.get(idx as usize)
            {
                acc += v * t;
            }
        }
        out.push(acc / gain);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles() {
        let v = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(median(&v), 3.0);
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 100.0), 5.0);
        assert!((percentile(&v, 25.0) - 2.0).abs() < 1e-12);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn biquad_dc_and_nyquist_gains() {
        let sr = 48_000.0;
        let run = |mut f: Biquad64, x: &dyn Fn(usize) -> f64| {
            let mut last = 0.0;
            for n in 0..20_000 {
                last = f.process(x(n));
            }
            last
        };
        // Low-pass passes DC, high-pass blocks it.
        assert!((run(Biquad64::lowpass(sr, 1000.0, 0.707), &|_| 1.0) - 1.0).abs() < 1e-9);
        assert!(run(Biquad64::highpass(sr, 1000.0, 0.707), &|_| 1.0).abs() < 1e-9);
    }

    #[test]
    fn sinc_interpolates_bandlimited_sine() {
        let k = KaiserSinc::new(16, 8.6);
        let f = 0.31; // cycles per sample
        let y: Vec<f64> = (0..200).map(|n| (2.0 * PI * f * n as f64).sin()).collect();
        for i in 0..50 {
            let t = 60.0 + i as f64 * 0.137;
            let want = (2.0 * PI * f * t).sin();
            assert!((k.interp(&y, t) - want).abs() < 2e-4, "t={t}");
        }
    }

    #[test]
    fn correlator_matches_direct() {
        let a: Vec<f64> = (0..100).map(|i| ((i * 7919) % 13) as f64 - 6.0).collect();
        let b: Vec<f64> = (0..300)
            .map(|i| ((i * 104_729) % 17) as f64 - 8.0)
            .collect();
        let mut c = Correlator64::new(512);
        let mut out = vec![0.0; 200];
        c.cross_correlate(&a, &b, &mut out);
        for (tau, o) in out.iter().enumerate() {
            let direct: f64 = a
                .iter()
                .enumerate()
                .map(|(j, x)| x * b.get(j + tau).copied().unwrap_or(0.0))
                .sum();
            assert!(
                (o - direct).abs() < 1e-8 * (1.0 + direct.abs()),
                "tau {tau}"
            );
        }
    }

    #[test]
    fn decimate_keeps_passband_and_rejects_stopband() {
        let n = 48_000;
        let tone = |f: f64| -> Vec<f64> {
            (0..n)
                .map(|i| (2.0 * PI * f * i as f64 / 48_000.0).sin())
                .collect()
        };
        let pass = decimate(&tone(1000.0), 4);
        let stop = decimate(&tone(9000.0), 4);
        let mid = |v: &[f64]| rms(&v[1000..v.len() - 1000]);
        assert!((mid(&pass) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3);
        assert!(mid(&stop) < 1e-3);
    }

    #[test]
    fn filtfilt_is_zero_phase() {
        let sr = 1000.0;
        let x: Vec<f64> = (0..4000)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / sr).sin())
            .collect();
        let lp = [Biquad64::lowpass(sr, 50.0, 0.707)];
        let y = filtfilt(&lp, &x);
        // Peak positions coincide (no phase shift) in the steady middle.
        for i in 1000..3000 {
            assert!((y[i] - x[i]).abs() < 1e-3, "i={i}");
        }
    }
}
