//! Test signals with exact ground truth, and a detector runner.
//!
//! * [`Voice`]: band-limited glottal-like harmonic source (−12 dB/octave above
//!   the second harmonic, like a Rosenberg/LF pulse) with a per-sample f0
//!   trajectory, optional jitter, through a cascade of formant resonators
//!   (Klatt-style 2-pole sections, unity DC gain). Harmonics fade out smoothly
//!   below a cutoff so f0 motion never makes a harmonic pop in or out.
//! * Pure and plain harmonic tones, white / pink / sibilant noise, silence.
//! * [`run`]: drives a detector exactly as the engine does (blocks split at
//!   hop boundaries, `analyze` once per hop) and records each estimate with
//!   the newest sample index it saw.

#![allow(dead_code)]

use vt_core::{PitchEstimate, QualityTier, VoiceRange};
use vt_detect::{Alignment, Detector, DetectorConfig};

pub const TAU: f64 = std::f64::consts::TAU;

/// Deterministic xorshift noise in [-1, 1).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Formant set: (frequency Hz, bandwidth Hz) per resonator.
pub type Formants = [(f64, f64); 4];

pub const VOWEL_A: Formants = [(730.0, 90.0), (1090.0, 110.0), (2440.0, 160.0), (3400.0, 250.0)];
pub const VOWEL_E: Formants = [(530.0, 80.0), (1840.0, 100.0), (2480.0, 160.0), (3500.0, 250.0)];
pub const VOWEL_I: Formants = [(270.0, 60.0), (2290.0, 100.0), (3010.0, 160.0), (3700.0, 250.0)];
pub const VOWEL_O: Formants = [(570.0, 80.0), (840.0, 90.0), (2410.0, 160.0), (3400.0, 250.0)];
pub const VOWEL_U: Formants = [(300.0, 60.0), (870.0, 90.0), (2240.0, 160.0), (3300.0, 250.0)];
pub const VOWELS: [Formants; 5] = [VOWEL_A, VOWEL_E, VOWEL_I, VOWEL_O, VOWEL_U];

/// Cascade of 2-pole resonators with unity DC gain (Klatt).
pub struct FormantFilter {
    c: [(f64, f64, f64); 4],
    s: [(f64, f64); 4],
}

impl FormantFilter {
    pub fn new(fs: f64, f: &Formants) -> Self {
        let mut c = [(0.0, 0.0, 0.0); 4];
        for (ci, &(freq, bw)) in c.iter_mut().zip(f.iter()) {
            let freq = freq.min(0.45 * fs);
            let r = (-std::f64::consts::PI * bw / fs).exp();
            let a1 = 2.0 * r * (TAU * freq / fs).cos();
            let a2 = -r * r;
            *ci = (1.0 - a1 - a2, a1, a2);
        }
        Self { c, s: [(0.0, 0.0); 4] }
    }
    pub fn process(&mut self, x: f64) -> f64 {
        let mut y = x;
        for ((b0, a1, a2), (y1, y2)) in self.c.iter().zip(self.s.iter_mut()) {
            let out = b0 * y + a1 * *y1 + a2 * *y2;
            *y2 = *y1;
            *y1 = out;
            y = out;
        }
        y
    }
}

/// Glottal-like source + formants. `f0[i]` is the fundamental at sample i (Hz).
pub struct Voice {
    pub formants: Option<Formants>,
    /// Standard deviation of the relative frequency jitter (e.g. 0.002), an
    /// AR(1) process with a 2 ms correlation time.
    pub jitter: f64,
    /// Gain of the first harmonic relative to the glottal model (1 = normal).
    pub h1_gain: f64,
    /// Harmonics fade out between 0.8 and 1.0 of this frequency.
    pub cutoff_hz: f64,
    /// Output RMS (linear).
    pub rms: f64,
    pub seed: u64,
}

impl Default for Voice {
    fn default() -> Self {
        Self { formants: Some(VOWEL_A), jitter: 0.0, h1_gain: 1.0, cutoff_hz: 8000.0, rms: 0.1, seed: 7 }
    }
}

impl Voice {
    /// Renders the voice for the f0 trajectory (0 Hz = silence).
    pub fn render(&self, fs: f64, f0: &[f64]) -> Vec<f32> {
        self.render_with_truth(fs, f0).0
    }

    /// Renders the voice and returns the true instantaneous f0 (including
    /// jitter) per sample.
    pub fn render_with_truth(&self, fs: f64, f0: &[f64]) -> (Vec<f32>, Vec<f64>) {
        let mut out = vec![0.0f64; f0.len()];
        let mut truth = vec![0.0f64; f0.len()];
        let k_jit = 1.0 / (0.002 * fs);
        let a_jit = self.jitter * (6.0 / k_jit).sqrt();
        let mut phase = 0.0f64; // cycles
        let mut rng = Rng::new(self.seed);
        let mut jit = 0.0f64;
        let mut filt = self.formants.map(|f| FormantFilter::new(fs, &f));
        let cutoff = self.cutoff_hz.min(0.45 * fs);
        for ((o, tr), &f) in out.iter_mut().zip(truth.iter_mut()).zip(f0.iter()) {
            if f <= 0.0 {
                // Keep the filter ringing down naturally.
                *o = filt.as_mut().map_or(0.0, |ff| ff.process(0.0));
                continue;
            }
            jit += k_jit * (rng.next() * a_jit - jit);
            let fi = f * (1.0 + jit);
            *tr = fi;
            phase += fi / fs;
            phase -= phase.floor();
            let kmax = (cutoff / f).floor().max(1.0) as usize;
            // Harmonic sum via complex rotation e^{i k φ}.
            let (s1, c1) = (TAU * phase).sin_cos();
            let (mut sk, mut ck) = (s1, c1);
            let mut x = 0.0;
            for k in 1..=kmax {
                let kf = k as f64;
                let fk = kf * f;
                let fade = ((cutoff - fk) / (0.2 * cutoff)).clamp(0.0, 1.0);
                // −12 dB/oct above k = 2, flat below (glottal-like).
                let mut a = 1.0 / (1.0 + (kf / 2.0).powi(2));
                if k == 1 {
                    a *= self.h1_gain;
                }
                // Fixed phase offsets (glottal pulse shape: −π/2 per harmonic
                // is a sawtooth-like derivative pulse; add a small dispersion).
                let th = -0.5 * std::f64::consts::PI + 0.15 * kf;
                x += a * fade * (sk * th.cos() + ck * th.sin());
                let (ns, nc) = (sk * c1 + ck * s1, ck * c1 - sk * s1);
                sk = ns;
                ck = nc;
            }
            *o = filt.as_mut().map_or(x, |ff| ff.process(x));
        }
        normalise(&mut out, self.rms, f0);
        (out.into_iter().map(|v| v as f32).collect(), truth)
    }
}

/// Scales `x` so its RMS over the voiced samples (f0 > 0) is `rms`.
fn normalise(x: &mut [f64], rms: f64, f0: &[f64]) {
    let (mut e, mut n) = (0.0, 0usize);
    for (&v, &f) in x.iter().zip(f0.iter()) {
        if f > 0.0 {
            e += v * v;
            n += 1;
        }
    }
    if n > 0 && e > 0.0 {
        let g = rms / (e / n as f64).sqrt();
        for v in x.iter_mut() {
            *v *= g;
        }
    }
}

/// Pure sine (amplitude `amp`) following `f0`.
pub fn sine(fs: f64, f0: &[f64], amp: f64) -> Vec<f32> {
    let mut ph = 0.0f64;
    f0.iter()
        .map(|&f| {
            ph += f / fs;
            ph -= ph.floor();
            if f > 0.0 { (amp * (TAU * ph).sin()) as f32 } else { 0.0 }
        })
        .collect()
}

/// Plain harmonic tone: harmonics 1..=k with amplitudes 1/k, below `cutoff`.
pub fn harmonic_tone(fs: f64, f0: &[f64], k: usize, rms: f64) -> Vec<f32> {
    let v = Voice { formants: None, cutoff_hz: 1e9, rms, ..Voice::default() };
    // Reuse Voice's source with a 1/k law instead of the glottal law.
    let mut out = vec![0.0f64; f0.len()];
    let mut ph = 0.0f64;
    for (o, &f) in out.iter_mut().zip(f0.iter()) {
        if f <= 0.0 {
            continue;
        }
        ph += f / fs;
        ph -= ph.floor();
        let kmax = k.min((0.45 * fs / f) as usize).max(1);
        *o = (1..=kmax).map(|h| (TAU * h as f64 * ph + 0.4 * h as f64).sin() / h as f64).sum();
    }
    normalise(&mut out, v.rms, f0);
    out.into_iter().map(|v| v as f32).collect()
}

/// White noise of the given RMS.
pub fn white(n: usize, rms: f64, seed: u64) -> Vec<f32> {
    let mut r = Rng::new(seed);
    let k = rms * 3f64.sqrt();
    (0..n).map(|_| (k * r.next()) as f32).collect()
}

/// Breath-like noise: white noise through a 1.5 kHz-ish low-pass and a gentle
/// tilt, scaled to `rms`.
pub fn breath(fs: f64, n: usize, rms: f64, seed: u64) -> Vec<f32> {
    let w = white(n, 1.0, seed);
    let mut lp = vt_core::filter::Biquad::lowpass(fs as f32, 1500.0, 0.5);
    let mut x: Vec<f64> = w.iter().map(|&v| lp.process(v) as f64).collect();
    let ones = vec![1.0; n];
    normalise(&mut x, rms, &ones);
    x.into_iter().map(|v| v as f32).collect()
}

/// Sibilant: white noise high-passed at 4.5 kHz (4th order), scaled to `rms`.
pub fn sibilant(fs: f64, n: usize, rms: f64, seed: u64) -> Vec<f32> {
    let w = white(n, 1.0, seed);
    let mut a = vt_core::filter::Biquad::highpass(fs as f32, 4500.0, 0.707);
    let mut b = vt_core::filter::Biquad::highpass(fs as f32, 4500.0, 0.707);
    let mut x: Vec<f64> = w.iter().map(|&v| b.process(a.process(v)) as f64).collect();
    let ones = vec![1.0; n];
    normalise(&mut x, rms, &ones);
    x.into_iter().map(|v| v as f32).collect()
}

/// Constant f0 for `secs` seconds.
pub fn steady(fs: f64, f: f64, secs: f64) -> Vec<f64> {
    vec![f; (fs * secs) as usize]
}

/// Vibrato: `f·2^(depth_c/1200·sin(2π rate t))`.
pub fn vibrato(fs: f64, f: f64, rate: f64, depth_c: f64, secs: f64) -> Vec<f64> {
    (0..(fs * secs) as usize).map(|i| f * 2f64.powf(depth_c / 1200.0 * (TAU * rate * i as f64 / fs).sin())).collect()
}

/// Absolute cents of a frequency.
pub fn cents(hz: f64) -> f64 {
    6900.0 + 1200.0 * (hz / 440.0).log2()
}

pub fn config(fs: f32, range: VoiceRange, alignment: Alignment) -> DetectorConfig {
    DetectorConfig { sample_rate: fs, voice_range: range, hop: vt_core::hop_for_rate(fs), alignment }
}

pub fn detector(fs: f32, range: VoiceRange, alignment: Alignment, tier: QualityTier) -> Detector {
    let mut d = Detector::new(config(fs, range, alignment));
    d.set_tier(tier);
    d
}

/// One analysed hop.
#[derive(Clone, Copy, Debug)]
pub struct Hop {
    /// Index of the newest sample pushed before this `analyze()`.
    pub newest: u64,
    pub est: PitchEstimate,
}

/// Runs `x` through `det` with host blocks from `blocks` (cycled), split at
/// hop boundaries like the engine.
pub fn run_blocks(det: &mut Detector, x: &[f32], blocks: &[usize]) -> Vec<Hop> {
    let hop = det.config().hop.max(1);
    let mut out = Vec::with_capacity(x.len() / hop + 1);
    let mut pos = 0usize;
    let mut phase = 0usize;
    let mut bi = 0usize;
    while pos < x.len() {
        let b = blocks[bi % blocks.len()].max(1);
        bi += 1;
        let end = (pos + b).min(x.len());
        while pos < end {
            let take = (hop - phase).min(end - pos);
            det.push(&x[pos..pos + take]);
            pos += take;
            phase += take;
            if phase == hop {
                phase = 0;
                out.push(Hop { newest: pos as u64 - 1, est: det.analyze() });
            }
        }
    }
    out
}

/// [`run_blocks`] with one hop per block.
pub fn run(det: &mut Detector, x: &[f32]) -> Vec<Hop> {
    let hop = det.config().hop;
    run_blocks(det, x, &[hop])
}

/// Truth f0 at a sample index (0 outside).
pub fn truth_at(f0: &[f64], i: u64) -> f64 {
    f0.get(i as usize).copied().unwrap_or(0.0)
}

/// Median of a slice (sorted copy).
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

/// Errors (cents) of voiced estimates against the truth at `sample_pos`,
/// for hops whose newest sample is ≥ `skip`, plus the voiced fraction.
pub fn cents_errors(hops: &[Hop], f0: &[f64], skip: u64) -> (Vec<f64>, f64) {
    let mut errs = Vec::new();
    let mut total = 0usize;
    for h in hops.iter().filter(|h| h.newest >= skip) {
        let t = truth_at(f0, h.est.sample_pos);
        if t <= 0.0 {
            continue;
        }
        total += 1;
        if h.est.voiced {
            errs.push(h.est.cents as f64 - cents(t));
        }
    }
    let frac = if total > 0 { errs.len() as f64 / total as f64 } else { 0.0 };
    (errs, frac)
}
