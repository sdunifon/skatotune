//! Small realtime filters: DC blocker, RBJ biquads, one-pole smoother, and the
//! minimum-phase decimators that feed the pitch detector (R§8.1, R§8.2).
//!
//! All coefficient design happens in constructors (f64 maths, rounded once);
//! `process`/`push` are a handful of multiply-adds with no branches on data.
//! Constructors sanitise their arguments (NaN, out-of-range frequencies) to a
//! working filter rather than panicking.

use core::f64::consts::PI;

/// Sanitised sample rate: non-finite or non-positive rates fall back to 48 kHz.
fn sane_rate(sr: f32) -> f64 {
    if sr.is_finite() && sr > 0.0 { sr as f64 } else { 48_000.0 }
}

// ---------------------------------------------------------------------------
// DC blocker
// ---------------------------------------------------------------------------

/// First-order DC blocker `y = x − x₁ + R·y₁` (R§8.1).
///
/// `R = 0.995` at 48 kHz (a −3 dB corner near 38 Hz); at other rates R is
/// scaled as `0.995^(48000/sr)` so the corner frequency stays put.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DcBlocker {
    r: f32,
    x1: f32,
    y1: f32,
}

impl DcBlocker {
    /// DC blocker for `sample_rate` Hz.
    pub fn new(sample_rate: f32) -> Self {
        let r = 0.995f64.powf(48_000.0 / sane_rate(sample_rate));
        Self { r: r as f32, x1: 0.0, y1: 0.0 }
    }

    /// The pole radius R.
    pub fn coefficient(&self) -> f32 {
        self.r
    }

    /// Filters one sample.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = x - self.x1 + self.r * self.y1;
        self.x1 = x;
        self.y1 = y;
        y
    }

    /// Filters a block in place.
    pub fn process_block(&mut self, xs: &mut [f32]) {
        for x in xs {
            *x = self.process(*x);
        }
    }

    /// Clears the state.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Biquad
// ---------------------------------------------------------------------------

/// Second-order IIR section, transposed direct form II, f32 coefficients and
/// state. Coefficients follow the RBJ Audio-EQ cookbook (normalised, a0 = 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

/// Shared RBJ prelude: (cos ω0, α) for a sanitised (sr, f, q).
fn rbj(sr: f32, f: f32, q: f32) -> (f64, f64) {
    let sr = sane_rate(sr);
    let f = if f.is_finite() { (f as f64).clamp(1e-6 * sr, 0.499 * sr) } else { 0.25 * sr };
    let q = if q.is_finite() && q > 0.0 { (q as f64).clamp(1e-3, 1e3) } else { core::f64::consts::FRAC_1_SQRT_2 };
    let w0 = 2.0 * PI * f / sr;
    (w0.cos(), w0.sin() / (2.0 * q))
}

impl Biquad {
    /// Section from normalised coefficients
    /// `H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²)`.
    pub fn from_coefficients(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self { b0, b1, b2, a1, a2, z1: 0.0, z2: 0.0 }
    }

    /// Pass-through section.
    pub fn identity() -> Self {
        Self::from_coefficients(1.0, 0.0, 0.0, 0.0, 0.0)
    }

    fn from_f64(b: [f64; 3], a: [f64; 3]) -> Self {
        let n = 1.0 / a[0];
        Self::from_coefficients(
            (b[0] * n) as f32,
            (b[1] * n) as f32,
            (b[2] * n) as f32,
            (a[1] * n) as f32,
            (a[2] * n) as f32,
        )
    }

    /// 12 dB/oct low-pass at `f` Hz (Q = 0.7071 is Butterworth).
    pub fn lowpass(sr: f32, f: f32, q: f32) -> Self {
        let (c, alpha) = rbj(sr, f, q);
        Self::from_f64([(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0], [1.0 + alpha, -2.0 * c, 1.0 - alpha])
    }

    /// 12 dB/oct high-pass at `f` Hz.
    pub fn highpass(sr: f32, f: f32, q: f32) -> Self {
        let (c, alpha) = rbj(sr, f, q);
        Self::from_f64([(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0], [1.0 + alpha, -2.0 * c, 1.0 - alpha])
    }

    /// Band-pass centred on `f` Hz with 0 dB peak gain (bandwidth set by `q`).
    pub fn bandpass(sr: f32, f: f32, q: f32) -> Self {
        let (c, alpha) = rbj(sr, f, q);
        Self::from_f64([alpha, 0.0, -alpha], [1.0 + alpha, -2.0 * c, 1.0 - alpha])
    }

    /// Normalised coefficients `[b0, b1, b2, a1, a2]`.
    pub fn coefficients(&self) -> [f32; 5] {
        [self.b0, self.b1, self.b2, self.a1, self.a2]
    }

    /// Filters one sample.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Filters a block in place.
    pub fn process_block(&mut self, xs: &mut [f32]) {
        for x in xs {
            *x = self.process(*x);
        }
    }

    /// Clears the state (coefficients are kept).
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// |H| at `f` Hz (analysis helper, not for the audio thread).
    pub fn magnitude_at(&self, sr: f32, f: f32) -> f32 {
        let s = Section {
            b0: self.b0 as f64,
            b1: self.b1 as f64,
            b2: self.b2 as f64,
            a1: self.a1 as f64,
            a2: self.a2 as f64,
            s1: 0.0,
            s2: 0.0,
        };
        let (re, im) = s.response(2.0 * PI * f as f64 / sane_rate(sr));
        (re * re + im * im).sqrt() as f32
    }
}

// ---------------------------------------------------------------------------
// One-pole smoother
// ---------------------------------------------------------------------------

/// One-pole low-pass / parameter smoother `y += α (x − y)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnePoleLp {
    alpha: f32,
    y: f32,
}

impl OnePoleLp {
    /// Smoother with a 63 % time constant of `tau_samples` per-sample steps
    /// (τ ≤ 0 → no smoothing).
    pub fn new(tau_samples: f32) -> Self {
        Self { alpha: crate::math::onepole_alpha(tau_samples, 1.0), y: 0.0 }
    }

    /// Smoother with an explicit coefficient α ∈ [0, 1].
    pub fn with_alpha(alpha: f32) -> Self {
        let mut s = Self { alpha: 1.0, y: 0.0 };
        s.set_alpha(alpha);
        s
    }

    /// Sets the time constant for updates every `step_samples` samples.
    pub fn set_tau(&mut self, tau_samples: f32, step_samples: f32) {
        self.alpha = crate::math::onepole_alpha(tau_samples, step_samples);
    }

    /// Sets α directly (clamped to [0, 1]; NaN → 1).
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = if alpha.is_nan() { 1.0 } else { alpha.clamp(0.0, 1.0) };
    }

    /// Current coefficient.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// One update toward `x`; returns the new output.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        self.y += self.alpha * (x - self.y);
        self.y
    }

    /// Current output.
    pub fn value(&self) -> f32 {
        self.y
    }

    /// Jumps the output to `y` (e.g. to start a smoother at its target).
    pub fn set_value(&mut self, y: f32) {
        self.y = y;
    }

    /// Output back to 0.
    pub fn reset(&mut self) {
        self.y = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Decimator
// ---------------------------------------------------------------------------

/// f64 TDF-II section used inside the decimator. High-order low-passes with
/// cutoffs down to 0.025·fs_in (factor 16 at 192 kHz) put poles close to z = 1,
/// where f64 coefficients and state keep the response exact; the cost is
/// irrelevant (a few sections per input sample).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Section {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    s1: f64,
    s2: f64,
}

impl Section {
    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Complex response at ω rad/sample.
    fn response(&self, w: f64) -> (f64, f64) {
        let (c1, s1) = (w.cos(), -w.sin());
        let (c2, s2) = ((2.0 * w).cos(), -(2.0 * w).sin());
        let (nr, ni) = (self.b0 + self.b1 * c1 + self.b2 * c2, self.b1 * s1 + self.b2 * s2);
        let (dr, di) = (1.0 + self.a1 * c1 + self.a2 * c2, self.a1 * s1 + self.a2 * s2);
        let d = dr * dr + di * di;
        ((nr * dr + ni * di) / d, (ni * dr - nr * di) / d)
    }

    /// Group delay at DC in samples: for a polynomial Σ c_k z^-k the DC group
    /// delay is Σ k c_k / Σ c_k; the section's is numerator − denominator.
    fn dc_group_delay(&self) -> f64 {
        let num = (self.b1 + 2.0 * self.b2) / (self.b0 + self.b1 + self.b2);
        let den = (self.a1 + 2.0 * self.a2) / (1.0 + self.a1 + self.a2);
        num - den
    }
}

/// Largest supported number of biquad sections (order 16).
const MAX_SECTIONS: usize = 8;
/// Largest supported decimation factor.
const MAX_FACTOR: usize = 32;
/// Designed stopband attenuation (contract: ≥ 60 dB; 5 dB of margin).
const STOPBAND_DB: f64 = 65.0;
/// −3 dB point as a fraction of the output rate (contract: ~0.4·fs_out).
const PASSBAND_FRAC: f64 = 0.4;

/// Minimum-phase IIR anti-alias filter + integer downsampler (R§8.2).
///
/// The filter is an even-order **Chebyshev type II** low-pass (monotonic
/// passband, equiripple stopband) designed by bilinear transform with
/// prewarping: stopband edge exactly at fs_out/2 with 65 dB attenuation, and
/// the smallest order (10-12 for factors 2-16) whose −3 dB point reaches
/// 0.4·fs_out. Pitch detection does not care about phase, so a minimum-phase
/// IIR is used instead of a linear-phase FIR: its low-frequency group delay is
/// only ≈ 1.5-1.8 *output* samples, versus milliseconds for a half-band
/// cascade (R§8.2). (A Chebyshev II was preferred over the elliptic filter
/// research.md mentions as an example: same contract, closed-form design and
/// a flat passband; the order difference costs < 0.1 % of a core.)
///
/// Output timing: the k-th output (k = 0, 1, …) is the filtered value at input
/// sample `k·factor + factor − 1`, i.e. `push` returns `Some` on every
/// `factor`-th call, starting with the `factor`-th.
#[derive(Clone, Debug, PartialEq)]
pub struct Decimator {
    sections: [Section; MAX_SECTIONS],
    n_sections: usize,
    factor: usize,
    phase: usize,
    sr_in: f32,
    group_delay: f32,
}

impl Decimator {
    /// Decimator by `factor` (clamped to 1..=32; 1 = pass-through, no filter)
    /// for input rate `sr_in`.
    pub fn new(sr_in: f32, factor: usize) -> Self {
        let factor = factor.clamp(1, MAX_FACTOR);
        let mut sections = [Section::default(); MAX_SECTIONS];
        let n_sections = if factor > 1 { design_cheby2(factor, &mut sections) } else { 0 };
        let group_delay = sections[..n_sections].iter().map(Section::dc_group_delay).sum::<f64>() as f32;
        Self { sections, n_sections, factor, phase: 0, sr_in: sane_rate(sr_in) as f32, group_delay }
    }

    /// Decimation factor.
    pub fn factor(&self) -> usize {
        self.factor
    }

    /// Output sample rate, `sr_in / factor`.
    pub fn output_rate(&self) -> f32 {
        self.sr_in / self.factor as f32
    }

    /// Filter order (2 × number of biquad sections).
    pub fn order(&self) -> usize {
        2 * self.n_sections
    }

    /// Push one input sample; returns `Some(output)` every `factor` samples.
    #[inline]
    pub fn push(&mut self, x: f32) -> Option<f32> {
        let mut y = x as f64;
        for s in self.sections.iter_mut().take(self.n_sections) {
            y = s.process(y);
        }
        self.phase += 1;
        if self.phase >= self.factor {
            self.phase = 0;
            Some(y as f32)
        } else {
            None
        }
    }

    /// Pushes a block, writing outputs to the front of `out`; returns how many
    /// were produced. Outputs that do not fit in `out` are dropped (size `out`
    /// to at least `input.len() / factor + 1`).
    pub fn push_block(&mut self, input: &[f32], out: &mut [f32]) -> usize {
        let mut n = 0;
        for &x in input {
            if let Some(y) = self.push(x)
                && let Some(slot) = out.get_mut(n)
            {
                *slot = y;
                n += 1;
            }
        }
        n
    }

    /// Approximate group delay in *input* samples at low frequencies (the DC
    /// value; it rises by < 10 % up to 0.1·fs_out). Counts toward detection
    /// lag.
    pub fn group_delay(&self) -> f32 {
        self.group_delay
    }

    /// Group delay in input samples at `freq_hz` (analysis helper).
    pub fn group_delay_at(&self, freq_hz: f32) -> f32 {
        let w = 2.0 * PI * freq_hz as f64 / self.sr_in as f64;
        let dw = 1e-6;
        let mut gd = 0.0;
        for s in &self.sections[..self.n_sections] {
            let (r0, i0) = s.response(w - dw);
            let (r1, i1) = s.response(w + dw);
            // arg(H1 / H0) without unwrapping problems.
            let d = (i1 * r0 - r1 * i0).atan2(r1 * r0 + i1 * i0);
            gd -= d / (2.0 * dw);
        }
        gd as f32
    }

    /// |H| at `freq_hz` of the anti-alias filter (analysis helper).
    pub fn magnitude_at(&self, freq_hz: f32) -> f32 {
        let w = 2.0 * PI * freq_hz as f64 / self.sr_in as f64;
        let mut m = 1.0f64;
        for s in &self.sections[..self.n_sections] {
            let (re, im) = s.response(w);
            m *= (re * re + im * im).sqrt();
        }
        m as f32
    }

    /// Clears filter state and the output phase.
    pub fn reset(&mut self) {
        for s in &mut self.sections {
            s.s1 = 0.0;
            s.s2 = 0.0;
        }
        self.phase = 0;
    }
}

/// Designs the Chebyshev II low-pass for `factor` into `out`; returns the
/// number of sections.
///
/// Analog prototype (stopband edge Ωs, ripple ε = 1/√(10^(As/10) − 1)): the
/// poles are the reciprocals of the Chebyshev-I poles
/// `pₖ = −sinh μ·sin θₖ + j·cosh μ·cos θₖ`, μ = asinh(1/ε)/N,
/// θₖ = π(2k−1)/(2N), scaled by Ωs; the zeros are `±j·Ωs / cos θₖ`. The
/// response is 3 dB down at Ω₃ = Ωs / cosh(acosh(1/ε)/N), which picks N.
fn design_cheby2(factor: usize, out: &mut [Section; MAX_SECTIONS]) -> usize {
    let f = factor as f64;
    // Prewarped analog edges for the bilinear map s = (1 − z⁻¹)/(1 + z⁻¹).
    let ws = (PI * 0.5 / f).tan();
    let wp = (PI * PASSBAND_FRAC / f).tan();
    let inv_eps = (10f64.powf(STOPBAND_DB / 10.0) - 1.0).sqrt();
    let mut n = 2;
    while n < 2 * MAX_SECTIONS && ws / (inv_eps.acosh() / n as f64).cosh() < wp {
        n += 2;
    }
    let mu = inv_eps.asinh() / n as f64;
    let sections = n / 2;
    for (k, sec) in out.iter_mut().enumerate().take(sections) {
        let th = PI * (2 * k + 1) as f64 / (2 * n) as f64;
        // Chebyshev-I prototype pole, then s = Ωs / p.
        let (pr, pi) = (-mu.sinh() * th.sin(), mu.cosh() * th.cos());
        let pm = pr * pr + pi * pi;
        let (sr, si) = (ws * pr / pm, -ws * pi / pm);
        // Bilinear: z = (1 + s) / (1 − s).
        let (nr, ni) = (1.0 + sr, si);
        let (dr, di) = (1.0 - sr, -si);
        let dm = dr * dr + di * di;
        let (zr, zi) = ((nr * dr + ni * di) / dm, (ni * dr - nr * di) / dm);
        // Zero at s = jΩz maps to e^{jφ} with φ = 2·atan(Ωz).
        let wz = ws / th.cos();
        let phi = 2.0 * wz.atan();
        let a1 = -2.0 * zr;
        let a2 = zr * zr + zi * zi;
        let (b1, b2) = (-2.0 * phi.cos(), 1.0);
        // Unity gain at DC per section.
        let g = (1.0 + a1 + a2) / (1.0 + b1 + b2);
        *sec = Section { b0: g, b1: g * b1, b2: g * b2, a1, a2, s1: 0.0, s2: 0.0 };
    }
    sections
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone_gain_db(process: &mut dyn FnMut(f32) -> f32, sr: f32, f: f32) -> f32 {
        // Steady-state RMS ratio after a settling period.
        let n_settle = 8192;
        let n = 16384;
        let mut acc = 0.0f64;
        for i in 0..(n_settle + n) {
            let x = (2.0 * core::f64::consts::PI * f as f64 * i as f64 / sr as f64).sin() as f32;
            let y = process(x);
            if i >= n_settle {
                acc += (y as f64) * (y as f64);
            }
        }
        (10.0 * (acc / n as f64 * 2.0).log10()) as f32
    }

    #[test]
    fn dc_blocker_removes_dc_keeps_audio() {
        let mut dc = DcBlocker::new(48_000.0);
        assert!((dc.coefficient() - 0.995).abs() < 1e-7);
        let mut y = 0.0;
        for _ in 0..48_000 {
            y = dc.process(1.0);
        }
        assert!(y.abs() < 1e-6, "residual DC {y}");
        let mut dc = DcBlocker::new(48_000.0);
        let g = tone_gain_db(&mut |x| dc.process(x), 48_000.0, 440.0);
        assert!(g.abs() < 0.05, "440 Hz gain {g} dB");
        // Corner frequency is rate-independent.
        let mut a = DcBlocker::new(48_000.0);
        let mut b = DcBlocker::new(96_000.0);
        let ga = tone_gain_db(&mut |x| a.process(x), 48_000.0, 30.0);
        let gb = tone_gain_db(&mut |x| b.process(x), 96_000.0, 30.0);
        // (Equal to first order in 1/fs; the residual is 0.09 dB at 30 Hz.)
        assert!((ga - gb).abs() < 0.15, "{ga} vs {gb}");
        assert!(DcBlocker::new(f32::NAN).coefficient() > 0.99);
        a.reset();
        assert_eq!(a.process(0.0), 0.0);
    }

    #[test]
    fn biquad_responses() {
        let sr = 48_000.0;
        let lp = Biquad::lowpass(sr, 1000.0, core::f32::consts::FRAC_1_SQRT_2);
        assert!((lp.magnitude_at(sr, 1.0) - 1.0).abs() < 1e-4);
        assert!((20.0 * lp.magnitude_at(sr, 1000.0).log10() + 3.01).abs() < 0.05);
        assert!(lp.magnitude_at(sr, 10_000.0) < 0.02);
        let hp = Biquad::highpass(sr, 1000.0, core::f32::consts::FRAC_1_SQRT_2);
        assert!(hp.magnitude_at(sr, 10.0) < 1e-3);
        assert!((hp.magnitude_at(sr, 20_000.0) - 1.0).abs() < 0.01);
        let bp = Biquad::bandpass(sr, 2000.0, 4.0);
        assert!((bp.magnitude_at(sr, 2000.0) - 1.0).abs() < 1e-4);
        assert!(bp.magnitude_at(sr, 200.0) < 0.05);
        // Time-domain response agrees with the analytic magnitude.
        let mut lp2 = lp;
        let g = tone_gain_db(&mut |x| lp2.process(x), sr, 3000.0);
        let expect = 20.0 * lp.magnitude_at(sr, 3000.0).log10();
        assert!((g - expect).abs() < 0.05, "{g} vs {expect}");
        lp2.reset();
        assert_eq!(lp2.process(0.0), 0.0);
        // Bad parameters still give a stable, finite filter.
        let mut bad = Biquad::lowpass(f32::NAN, f32::INFINITY, -1.0);
        for _ in 0..1000 {
            assert!(bad.process(1.0).is_finite());
        }
        let mut id = Biquad::identity();
        assert_eq!(id.process(0.3), 0.3);
    }

    #[test]
    fn one_pole_smoother() {
        let mut s = OnePoleLp::new(100.0);
        for _ in 0..100 {
            s.process(1.0);
        }
        assert!((s.value() - (1.0 - (-1.0f32).exp())).abs() < 2e-3);
        let mut inst = OnePoleLp::new(0.0);
        assert_eq!(inst.process(0.7), 0.7);
        let mut w = OnePoleLp::with_alpha(f32::NAN);
        assert_eq!(w.alpha(), 1.0);
        w.set_alpha(2.0);
        assert_eq!(w.alpha(), 1.0);
        w.set_tau(64.0 * 10.0, 64.0);
        assert!((w.alpha() - (1.0 - (-0.1f32).exp())).abs() < 1e-6);
        w.set_value(3.0);
        assert_eq!(w.value(), 3.0);
        w.reset();
        assert_eq!(w.value(), 0.0);
    }

    #[test]
    fn decimator_stopband_and_passband() {
        for &(sr, factor) in &[
            (44_100.0f32, 2usize),
            (48_000.0, 2),
            (48_000.0, 3),
            (48_000.0, 4),
            (48_000.0, 6),
            (96_000.0, 4),
            (96_000.0, 8),
            (192_000.0, 8),
            (192_000.0, 16),
        ] {
            let d = Decimator::new(sr, factor);
            let fs_out = sr / factor as f32;
            // Analytic response: ≥ 60 dB everywhere above fs_out/2.
            let mut worst = f32::MIN;
            for k in 0..=4000 {
                let f = fs_out * 0.5 + (sr * 0.5 - fs_out * 0.5) * k as f32 / 4000.0;
                worst = worst.max(20.0 * d.magnitude_at(f).max(1e-12).log10());
            }
            assert!(worst < -60.0, "sr={sr} F={factor}: stopband peak {worst:.1} dB");
            // Passband: within 3 dB up to 0.4 fs_out, flat at low frequencies.
            let g_edge = 20.0 * d.magnitude_at(0.4 * fs_out).log10();
            assert!(g_edge > -3.01, "sr={sr} F={factor}: {g_edge:.2} dB at 0.4 fs_out");
            let g_low = 20.0 * d.magnitude_at(0.1 * fs_out).log10();
            assert!(g_low.abs() < 0.1, "sr={sr} F={factor}: {g_low:.3} dB at 0.1 fs_out");
            assert!(d.order() <= 2 * MAX_SECTIONS);
        }
    }

    #[test]
    fn decimator_time_domain_aliasing() {
        // A tone just above fs_out/2 must come out ≥ 60 dB down after
        // decimation; a low tone passes at unity.
        let sr = 48_000.0;
        let factor = 3;
        let fs_out = sr / factor as f32;
        let run = |f: f32| {
            let mut d = Decimator::new(sr, factor);
            let mut acc = 0.0f64;
            let mut count = 0usize;
            for i in 0..(48_000 * 2) {
                let x = (2.0 * core::f64::consts::PI * f as f64 * i as f64 / sr as f64).sin() as f32;
                if let Some(y) = d.push(x)
                    && i > 24_000
                {
                    acc += (y as f64) * (y as f64);
                    count += 1;
                }
            }
            10.0 * (acc / count as f64 * 2.0).log10()
        };
        let stop = run(fs_out * 0.5 * 1.02);
        assert!(stop < -60.0, "aliased tone at {stop:.1} dB");
        let stop2 = run(fs_out * 0.9);
        assert!(stop2 < -60.0, "aliased tone at {stop2:.1} dB");
        let pass = run(440.0);
        assert!(pass.abs() < 0.1, "440 Hz at {pass:.2} dB");
    }

    #[test]
    fn decimator_output_timing_and_group_delay() {
        let mut d = Decimator::new(48_000.0, 4);
        let mut outs = Vec::new();
        for i in 0..16 {
            if d.push(0.0).is_some() {
                outs.push(i);
            }
        }
        assert_eq!(outs, vec![3, 7, 11, 15]);
        // Group delay: small (a couple of output samples) and consistent with
        // the numerical phase derivative at low frequency.
        let gd = d.group_delay();
        assert!(gd > 0.5 * 4.0 && gd < 3.0 * 4.0, "gd={gd}");
        let gd_num = d.group_delay_at(50.0);
        assert!((gd_num - gd).abs() / gd < 0.02, "{gd_num} vs {gd}");
        // Measured: delay of a slow Gaussian pulse's centroid through the filter.
        let mut d = Decimator::new(48_000.0, 4);
        let (mut m0, mut m1) = (0.0f64, 0.0f64);
        let (mut i0, mut i1) = (0.0f64, 0.0f64);
        for i in 0..20_000 {
            let t = (i as f64 - 6000.0) / 800.0;
            let x = (-t * t).exp();
            i0 += x;
            i1 += x * i as f64;
            if let Some(y) = d.push(x as f32) {
                m0 += y as f64;
                m1 += y as f64 * i as f64;
            }
        }
        let measured = m1 / m0 - i1 / i0;
        assert!((measured - gd as f64).abs() < 0.5, "measured {measured} vs {gd}");
        // push_block agrees with push.
        let input: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.05).sin()).collect();
        let mut a = Decimator::new(48_000.0, 3);
        let mut b = a.clone();
        let mut out = vec![0.0f32; 400];
        let n = a.push_block(&input, &mut out);
        let ref_out: Vec<f32> = input.iter().filter_map(|&x| b.push(x)).collect();
        assert_eq!(n, ref_out.len());
        assert_eq!(&out[..n], &ref_out[..]);
        // reset restores the initial state.
        a.reset();
        let again: Vec<f32> = input.iter().filter_map(|&x| a.push(x)).collect();
        assert_eq!(again, ref_out);
    }

    #[test]
    fn decimator_factor_one_and_clamping() {
        let mut d = Decimator::new(48_000.0, 1);
        assert_eq!(d.push(0.25), Some(0.25));
        assert_eq!(d.group_delay(), 0.0);
        assert_eq!(Decimator::new(48_000.0, 0).factor(), 1);
        assert_eq!(Decimator::new(48_000.0, 1000).factor(), MAX_FACTOR);
        let mut big = Decimator::new(384_000.0, 32);
        for i in 0..10_000 {
            if let Some(y) = big.push(if i == 0 { 1.0 } else { 0.0 }) {
                assert!(y.is_finite());
            }
        }
        assert!(big.magnitude_at(384_000.0 / 64.0 * 1.01) < 1e-3);
    }
}
