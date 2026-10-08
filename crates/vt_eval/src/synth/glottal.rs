//! Band-limited Liljencrants-Fant (LF) glottal source.
//!
//! The LF model (Fant, Liljencrants & Lin 1985) describes one period of the
//! glottal *flow derivative* (which already includes lip radiation): an
//! exponentially growing sinusoid during the open phase up to the main
//! excitation at `te`, then an exponential return phase of time constant `ta`.
//! We parameterise it with the single "Rd" voice-quality number (Fant 1995):
//! Rd ≈ 0.3 is pressed/tense, ≈ 1.0 modal, ≈ 2.7 breathy.
//!
//! **Band limiting.** Sampling the LF waveform directly aliases badly (its
//! spectrum only falls at −12 dB/oct). Instead one period is evaluated on a
//! dense grid, its Fourier series is taken once, and every output sample is a
//! harmonic sum `Σ_k w(k·f0)·Re(c_k·e^{i2πkφ})` over harmonics below a cutoff,
//! with a raised-cosine taper `w` so harmonics fade in and out smoothly as f0
//! moves. The result is exactly the band-limited LF pulse train for a constant
//! f0 and alias-free for any slowly varying f0 trajectory, because the phase
//! φ(t) = ∫f0 is integrated per sample.

use std::f64::consts::PI;

use realfft::RealFftPlanner;

/// LF timing parameters as fractions of the period T0 (`tc` = 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LfShape {
    /// Instant of maximum glottal flow (zero of the derivative).
    pub tp: f64,
    /// Instant of the main excitation (most negative derivative).
    pub te: f64,
    /// Effective duration of the return phase.
    pub ta: f64,
}

impl LfShape {
    /// Fant's (1995) Rd regression: Rd → (Ra, Rk, Rg) → (tp, te, ta).
    pub fn from_rd(rd: f64) -> Self {
        let rd = rd.clamp(0.3, 2.7);
        let ra = (-1.0 + 4.8 * rd) / 100.0;
        let rk = (22.4 + 11.8 * rd) / 100.0;
        let rg = 1.0 / (4.0 * ((0.11 * rd / (0.5 + 1.2 * rk)) - ra) / rk);
        let tp = 1.0 / (2.0 * rg);
        let te = tp * (1.0 + rk);
        Self { tp, te, ta: ra }
    }
}

/// Solved LF waveform for one normalised period, scaled so the main
/// excitation `|E(te)| = 1`.
#[derive(Clone, Copy, Debug)]
pub struct LfModel {
    shape: LfShape,
    /// Open-phase growth rate (per period).
    alpha: f64,
    /// Return-phase decay rate (per period).
    eps: f64,
    /// Open-phase angular frequency π / tp.
    wg: f64,
    /// Open-phase amplitude after normalisation.
    e0: f64,
}

impl LfModel {
    pub fn new(shape: LfShape) -> Self {
        let LfShape { tp, te, ta } = shape;
        let wg = PI / tp;
        let rest = 1.0 - te;

        // ε solves ε·ta = 1 − exp(−ε(1 − te)) (continuity of the return phase).
        // f is convex with f(0) = 0, so Newton from 1/ta converges to the
        // positive root.
        let mut eps = 1.0 / ta;
        for _ in 0..60 {
            let e = (-eps * rest).exp();
            let f = eps * ta - 1.0 + e;
            let df = ta - rest * e;
            if df.abs() < 1e-300 {
                break;
            }
            let next = eps - f / df;
            if !next.is_finite() || next <= 0.0 {
                break;
            }
            if (next - eps).abs() < 1e-14 * eps {
                eps = next;
                break;
            }
            eps = next;
        }

        // α makes the net flow over the period zero (∫E dt = 0), with E0 = 1.
        let ee_of = |alpha: f64| -(alpha * te).exp() * (wg * te).sin();
        let area = |alpha: f64| {
            let open = ((alpha * te).exp() * (alpha * (wg * te).sin() - wg * (wg * te).cos()) + wg)
                / (alpha * alpha + wg * wg);
            let e_rest = (-eps * rest).exp();
            let ret = -(ee_of(alpha) / (eps * ta)) * ((1.0 - e_rest) / eps - rest * e_rest);
            open + ret
        };
        // Scan for a sign change, then bisect. The area falls with α (the
        // excitation grows faster than the open-phase lobe).
        let mut lo = -40.0;
        let mut hi = lo;
        let mut found = false;
        let mut prev = area(lo);
        let mut a = lo;
        while a < 200.0 {
            let next_a = a + 0.25;
            let v = area(next_a);
            if prev.is_finite() && v.is_finite() && prev > 0.0 && v <= 0.0 {
                lo = a;
                hi = next_a;
                found = true;
                break;
            }
            prev = v;
            a = next_a;
        }
        let alpha = if found {
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if area(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        } else {
            // Not reachable for Rd in [0.3, 2.7]; a plain sinusoidal open phase
            // keeps the source usable rather than silent.
            0.0
        };
        let ee = ee_of(alpha);
        let e0 = if ee.abs() > 1e-300 { 1.0 / ee } else { 1.0 };
        Self { shape, alpha, eps, wg, e0 }
    }

    pub fn shape(&self) -> LfShape {
        self.shape
    }

    /// Flow derivative at normalised time `t` in [0, 1).
    pub fn eval(&self, t: f64) -> f64 {
        let LfShape { te, ta, .. } = self.shape;
        if t <= te {
            self.e0 * (self.alpha * t).exp() * (self.wg * t).sin()
        } else {
            let rest = 1.0 - te;
            -(1.0 / (self.eps * ta)) * ((-self.eps * (t - te)).exp() - (-self.eps * rest).exp())
        }
    }
}

/// Fourier coefficients of one LF period, ready for band-limited synthesis.
#[derive(Clone, Debug)]
pub struct GlottalTable {
    /// `c[k-1]` = complex coefficient (re, im) of harmonic k.
    coeffs: Vec<(f64, f64)>,
    /// RMS of the full-band waveform (for level calibration of added noise).
    rms: f64,
    shape: LfShape,
}

/// Grid size used to take the Fourier series of one period. The LF spectrum
/// falls at least 12 dB/oct, so harmonics near N/2 are ~1e-8 of the
/// fundamental and grid aliasing is negligible.
const GRID: usize = 16_384;

impl GlottalTable {
    /// Coefficients for harmonics 1..=`max_harmonics` (capped at GRID/2 − 1).
    pub fn new(shape: LfShape, max_harmonics: usize) -> Self {
        let model = LfModel::new(shape);
        let mut planner = RealFftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(GRID);
        let mut input: Vec<f64> = (0..GRID).map(|n| model.eval(n as f64 / GRID as f64)).collect();
        let rms = (input.iter().map(|v| v * v).sum::<f64>() / GRID as f64).sqrt();
        let mut spec = fft.make_output_vec();
        let k_max = max_harmonics.clamp(1, GRID / 2 - 1);
        let coeffs = if fft.process(&mut input, &mut spec).is_ok() {
            // x(φ) = Σ_k Re(c_k e^{i2πkφ}) with c_k = 2·X[k]/N for a real signal.
            spec.iter().skip(1).take(k_max).map(|c| (2.0 * c.re / GRID as f64, 2.0 * c.im / GRID as f64)).collect()
        } else {
            vec![(0.0, 0.0); k_max]
        };
        Self { coeffs, rms, shape }
    }

    pub fn shape(&self) -> LfShape {
        self.shape
    }

    pub fn harmonics(&self) -> usize {
        self.coeffs.len()
    }

    /// |c_k|² of harmonic `k` (1-based); 0 outside the table.
    pub fn harmonic_power(&self, k: usize) -> f64 {
        k.checked_sub(1).and_then(|i| self.coeffs.get(i)).map_or(0.0, |(re, im)| re * re + im * im)
    }

    /// Weight of harmonic frequency `fk` in [`GlottalTable::sample`]'s taper.
    pub fn taper(fk: f64, f_lo: f64, f_hi: f64) -> f64 {
        if fk <= f_lo {
            1.0
        } else if fk >= f_hi {
            0.0
        } else {
            0.5 * (1.0 + (PI * (fk - f_lo) / (f_hi - f_lo).max(1e-9)).cos())
        }
    }

    /// RMS of one full-band period (the band-limited version is within a few
    /// percent of this, since almost all energy sits in the low harmonics).
    pub fn rms(&self) -> f64 {
        self.rms
    }

    /// Band-limited source value at phase `phase` (cycles) for fundamental
    /// `f0` Hz. Harmonics at or below `f_lo` get full weight, those between
    /// `f_lo` and `f_hi` a raised-cosine taper, those above `f_hi` none.
    pub fn sample(&self, phase: f64, f0: f64, f_lo: f64, f_hi: f64) -> f64 {
        if !(f0 > 0.0) || !phase.is_finite() {
            return 0.0;
        }
        let k_top = ((f_hi / f0).floor() as usize).min(self.coeffs.len());
        let (s1, c1) = (2.0 * PI * phase).sin_cos();
        // e^{i2πkφ} by repeated complex multiplication: exact to ~k·1e-16.
        let (mut zr, mut zi) = (c1, s1);
        let mut acc = 0.0;
        for (k, &(cr, ci)) in self.coeffs.iter().take(k_top).enumerate() {
            let fk = (k + 1) as f64 * f0;
            let w = if fk <= f_lo { 1.0 } else { Self::taper(fk, f_lo, f_hi) };
            acc += w * (cr * zr - ci * zi);
            let nr = zr * c1 - zi * s1;
            zi = zr * s1 + zi * c1;
            zr = nr;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rd_shapes_are_ordered_and_valid() {
        let mut prev_te = 0.0;
        for rd in [0.3, 0.6, 1.0, 1.5, 2.0, 2.7] {
            let s = LfShape::from_rd(rd);
            assert!(s.tp > 0.2 && s.tp < s.te && s.te < 1.0, "rd {rd}: {s:?}");
            assert!(s.ta > 0.0 && s.ta < 1.0 - s.te, "rd {rd}: {s:?}");
            // Laxer voices close later (larger open quotient).
            assert!(s.te > prev_te);
            prev_te = s.te;
        }
    }

    #[test]
    fn lf_has_zero_net_flow_and_unit_excitation() {
        for rd in [0.5, 1.0, 1.7, 2.5] {
            let m = LfModel::new(LfShape::from_rd(rd));
            let n = 200_000;
            let area: f64 = (0..n).map(|i| m.eval((i as f64 + 0.5) / n as f64)).sum::<f64>() / n as f64;
            assert!(area.abs() < 1e-6, "rd {rd} area {area}");
            let te = m.shape().te;
            assert!((m.eval(te) + 1.0).abs() < 1e-9, "rd {rd}");
            // Continuous at te and back to zero at the period end.
            assert!((m.eval(te + 1e-9) + 1.0).abs() < 1e-6);
            assert!(m.eval(1.0 - 1e-12).abs() < 1e-6);
        }
    }

    #[test]
    fn harmonic_sum_reconstructs_waveform() {
        let shape = LfShape::from_rd(1.0);
        let table = GlottalTable::new(shape, 4000);
        let model = LfModel::new(shape);
        // With a very low f0 every stored harmonic is below the taper.
        for i in 0..64 {
            let phase = i as f64 / 64.0 + 0.003;
            let bl = table.sample(phase, 1.0, 1e9, 2e9);
            assert!((bl - model.eval(phase)).abs() < 2e-3, "phase {phase}: {bl} vs {}", model.eval(phase));
        }
    }

    #[test]
    fn harmonics_above_cutoff_are_absent() {
        // One period of a 1 kHz source at 48 kHz with f_hi = 10 kHz: the DFT of
        // exactly one period must hold no energy above harmonic 10.
        let table = GlottalTable::new(LfShape::from_rd(1.0), 2000);
        let n = 48;
        let x: Vec<f64> = (0..n).map(|i| table.sample(i as f64 / n as f64, 1000.0, 8000.0, 10_000.0)).collect();
        for k in 11..n / 2 {
            let (mut re, mut im) = (0.0, 0.0);
            for (i, v) in x.iter().enumerate() {
                let a = 2.0 * PI * (k * i) as f64 / n as f64;
                re += v * a.cos();
                im -= v * a.sin();
            }
            assert!((re * re + im * im).sqrt() < 1e-9, "harmonic {k} leaked");
        }
    }
}
