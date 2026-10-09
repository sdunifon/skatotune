//! Sub-sample period refinement on an undecimated (or verifier-rate) stream
//! (R§8.3 "full-rate local tracking", R§8.7).
//!
//! Given a seed period P from the chosen candidate, the type-I normalised
//! correlation
//!
//! ```text
//! nsdf(τ) = 1 − Σ_{j<W} (x_j − x_{j+τ})² / (Σ_{j<W} x_j² + Σ_{j<W} x_{j+τ}²),   W = round(P)
//! ```
//!
//! is evaluated at every integer lag in a small band around P on a window
//! that holds the newest W samples (newest-first, so the lagged copy reaches
//! back by τ). The evidence therefore spans ≈ 2P and is centred ≈ P before
//! the window end: the pitch-synchronous, low-information-lag measurement.
//! (The caller may impose a minimum W of a few ms so short periods average
//! more noise; the span is then W + P.)
//! For a periodic signal `nsdf(P) = 1` exactly whatever W is, so the maximum
//! is unbiased in principle; the error comes only from locating it between
//! integer lags.
//!
//! While the signal grows (head louder than its lagged copy: a note's
//! attack) and the two copies have nearly the same shape, the type-I form is
//! replaced by the attack-compensated normalised correlation of
//! `eh::attack_corr` (`r/√(e₀·e_τ)`, faded out when the lagged copy is nearly
//! silent). Type-I reads an amplitude ramp as aperiodicity
//! (its value is capped at `2ρ/(1 + ρ²)` for an RMS ratio ρ, which held low
//! voices below the voicing threshold for a period or more after the first
//! cycles) and, for an exponential envelope `e^{γt}`, is weighted by
//! `sech(γτ)`, which pulls the peak towards shorter lags (sharp) during an
//! attack; the gain-normalised form is exactly envelope-invariant. Steady
//! notes (`e₀ = e_τ`, where both forms agree) and decays are unchanged.
//!
//! * Eco/HQ: 3-point parabola through the integer maximum (R§8.4 formula).
//!   Sub-0.01-cent on long periods; its bias grows on short, sharp peaks
//!   (≈ 0.3-0.5 c at P ≈ 32-38 samples with harmonics to ≈ 0.2 fs).
//! * Max: for periods below [`EXACT_BELOW`] samples the normalised difference
//!   is evaluated *exactly* at fractional lags, the lagged copy being
//!   fractionally delayed by a 16-tap Kaiser-windowed sinc FIR (interpolated
//!   from a 512-phase bank built once, not designed from Bessel series per
//!   evaluation: PERF-1, −40 % on the exact-confirm tail), and its
//!   maximum located by two shrinking 3-point steps (R§8.7 "Max tier: sinc
//!   interpolation … of the lag peak"; interpolating the signal rather than
//!   the lag curve, because the curve's own interpolation error tilts broad
//!   peaks). Longer periods keep the parabola, which is already exact there.
//!
//! If the maximum sits on the band edge the band is shifted (R§8.3 "if the
//! minimum hits the edge of the band, shift the band"), at most twice.

use vt_core::simd;

use crate::hist::Mirror;

/// Smallest refinement window (samples).
const MIN_WINDOW: usize = 12;
/// Fractional-delay FIR half-length (taps each side) for the Max refinement.
const FIR_HALF: usize = 8;
/// Kaiser β for the 16-tap FIR (vt_core uses 9.25 for its 16-tap bank).
const FIR_BETA: f64 = 9.25;
/// Phases of the precomputed fractional-delay bank (taps for μ = k/P are
/// interpolated linearly in μ: delay error ≈ 1e-6 samples).
const FIR_PHASES: usize = 512;
/// Max tier evaluates fractional lags exactly below this period (samples).
const EXACT_BELOW: f64 = 160.0;
/// Band shifts allowed when the maximum lands on an edge.
const MAX_SHIFTS: usize = 2;

/// Result of one refinement.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Refined {
    /// Fractional period, stream samples.
    pub period: f64,
    /// Peak normalised correlation, [-1, 1].
    pub clarity: f32,
    /// Window length used (samples).
    pub window: usize,
}

impl Refined {
    /// Distance from the newest analysed sample back to the centre of the
    /// evidence (samples): the window spans `W + τ` samples.
    pub(crate) fn centre_back(&self) -> f64 {
        0.5 * (self.window as f64 + self.period - 1.0)
    }
}

/// Scratch for refinements of periods up to `max_period` samples.
pub(crate) struct Refiner {
    max_period: usize,
    max_lags: usize,
    /// Kaiser-windowed-sinc fractional-delay taps for μ = k/[`FIR_PHASES`],
    /// k = 0 ..= P, each row normalised to unity DC gain (built once: the
    /// exact peak search used to design 16 taps from Bessel series on each
    /// of its 7 evaluations, 57 % of its cost, PERF-1).
    bank: Box<[[f64; 2 * FIR_HALF]]>,
    buf: Box<[f32]>,
    d: Box<[f32]>,
    /// Lagged-window energy per lag.
    et: Box<[f64]>,
    nsdf: Box<[f32]>,
}

impl Refiner {
    pub(crate) fn new(max_period: usize, max_lags: usize) -> Self {
        let max_period = max_period.max(MIN_WINDOW);
        let max_lags = max_lags.max(8);
        let len = 2 * max_period + 2 * max_lags + 2 * FIR_HALF + 8;
        Self {
            max_period,
            max_lags,
            bank: (0..=FIR_PHASES).map(|k| fir_taps(k as f64 / FIR_PHASES as f64)).collect(),
            buf: vec![0.0; len].into_boxed_slice(),
            d: vec![0.0; max_lags].into_boxed_slice(),
            et: vec![0.0; max_lags].into_boxed_slice(),
            nsdf: vec![0.0; max_lags].into_boxed_slice(),
        }
    }

    /// Samples of history a refinement of `max_period` may read.
    pub(crate) fn span(&self) -> usize {
        self.buf.len()
    }

    /// Refines `seed` (stream samples) against the window ending at stream
    /// index `end` (exclusive), searching ±`band` lags, with a window of
    /// `max(round(seed), min_window)` samples. `exact` selects the Max
    /// fractional-lag search. `None` for an invalid seed or a silent window.
    pub(crate) fn refine(
        &mut self,
        hist: &Mirror,
        end: i64,
        seed: f64,
        band: usize,
        exact: bool,
        min_window: usize,
    ) -> Option<Refined> {
        if !seed.is_finite() || seed < 2.0 {
            return None;
        }
        let w = (seed.round() as usize).max(min_window).clamp(MIN_WINDOW, self.max_period);
        let band = band.clamp(1, (self.max_lags - 1) / 2);
        let mut centre = seed.round() as i64;
        for _ in 0..=MAX_SHIFTS {
            let lo = (centre - band as i64).max(2) as usize;
            let hi = (centre as usize + band).min(self.max_period + self.max_lags / 2);
            if hi <= lo + 1 {
                return None;
            }
            let n = hi - lo + 1;
            // + FIR_HALF + 1 so the fractional evaluation near `hi` has taps.
            let len = (w + hi + FIR_HALF + 1).min(self.buf.len());
            hist.copy_reversed(end, len, &mut self.buf);
            let x = &self.buf[..len];
            let (d, nsdf) = (&mut self.d[..n], &mut self.nsdf[..n]);
            debug_assert!(n <= self.et.len());
            simd::diff_sq(x, w, lo, d);
            let e0 = simd::sum_sq(&x[..w]) as f64;
            if e0 <= 1e-20 * w as f64 {
                return None;
            }
            // Lagged-window energies first (a serial sliding sum), then the
            // correlations, which are independent per lag (PERF-4).
            let mut et = simd::sum_sq(x.get(lo..lo + w).unwrap_or(&[])) as f64;
            let sq = |v: f32| (v as f64) * (v as f64);
            let ets = &mut self.et[..n];
            for (k, e) in ets.iter_mut().enumerate() {
                *e = et;
                let tau = lo + k;
                et =
                    (et - sq(x.get(tau).copied().unwrap_or(0.0)) + sq(x.get(tau + w).copied().unwrap_or(0.0))).max(0.0);
            }
            for ((o, &dv), &e) in nsdf.iter_mut().zip(d.iter()).zip(ets.iter()) {
                *o = norm_corr(e0, e, dv as f64) as f32;
            }
            let mut best = 0;
            for (k, &v) in nsdf.iter().enumerate() {
                if v > nsdf[best] {
                    best = k;
                }
            }
            let (off, val) = if best > 0 && best + 1 < n {
                vt_core::math::parabolic_peak(nsdf[best - 1], nsdf[best], nsdf[best + 1])
            } else {
                (0.0, nsdf[best])
            };
            let mut period = (lo + best) as f64 + off as f64;
            let mut clarity = val;
            let interior = best > 0 && best + 1 < n;
            if exact
                && interior
                && period < EXACT_BELOW
                && let Some((p, v)) = exact_peak(x, w, period, &self.bank)
            {
                period = p;
                clarity = v as f32;
            }
            if interior {
                return Some(Refined { period, clarity: clarity.clamp(-1.0, 1.0), window: w });
            }
            // Edge maximum: the true peak may be just outside the band → shift
            // (bounded: at most MAX_SHIFTS bands). A monotonic curve (smooth
            // input, wrong seed) never yields an interior peak → None.
            if best == 0 && lo > 2 {
                centre = (centre - band as i64).max(2);
            } else {
                centre += band as i64;
            }
        }
        None
    }
}

/// Kaiser-windowed-sinc taps (β = [`FIR_BETA`]) delaying by `j + μ` from
/// tap `FIR_HALF − 1`, normalised to unity DC gain.
fn fir_taps(mu: f64) -> [f64; 2 * FIR_HALF] {
    let mut h = [0.0f64; 2 * FIR_HALF];
    let mut sum = 0.0;
    for (i, hm) in h.iter_mut().enumerate() {
        let u = (i as f64 - (FIR_HALF as f64 - 1.0)) - mu;
        let sinc = if u.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * u).sin() / (std::f64::consts::PI * u) };
        *hm = sinc * vt_core::window::kaiser_at(u / FIR_HALF as f64, FIR_BETA);
        sum += *hm;
    }
    if sum.abs() > 1e-9 {
        for hm in h.iter_mut() {
            *hm /= sum;
        }
    }
    h
}

/// Normalised correlation at a fractional lag `tau`, exact up to the FIR's
/// accuracy: `x` is newest-first, the head is `x[..w]` with energy `e0`
/// ([`head_energy`]), and the lagged copy `x(j + tau)` is a 16-tap
/// Kaiser-windowed-sinc fractional delay interpolated from `bank`. `None`
/// if the taps would leave `x`.
fn nsdf_at(x: &[f32], w: usize, tau: f64, e0: f64, bank: &[[f64; 2 * FIR_HALF]]) -> Option<f64> {
    if !tau.is_finite() || tau < FIR_HALF as f64 {
        return None;
    }
    let base = tau.floor();
    let mu = tau - base;
    let base = base as usize;
    // Taps m = −(H−1) ..= H at positions j + base + m.
    let first = base + 1 - FIR_HALF;
    if first + w + 2 * FIR_HALF > x.len() {
        return None;
    }
    let pos = mu * FIR_PHASES as f64;
    let k = (pos.floor() as usize).min(FIR_PHASES - 1);
    let f = pos - k as f64;
    let (lo, hi) = (bank.get(k)?, bank.get(k + 1)?);
    let mut h = [0.0f64; 2 * FIR_HALF];
    for ((hm, &a), &b) in h.iter_mut().zip(lo.iter()).zip(hi.iter()) {
        *hm = a + f * (b - a);
    }
    let (mut d, mut et) = (0.0f64, 0.0f64);
    let lagged = x.get(first..first + w + 2 * FIR_HALF)?;
    for (taps, &a) in lagged.windows(2 * FIR_HALF).zip(x.iter().take(w)) {
        // Four partial sums: the 16-tap dot is latency-bound as one chain.
        let mut acc = [0.0f64; 4];
        for (t4, h4) in taps.as_chunks::<4>().0.iter().zip(h.as_chunks::<4>().0.iter()) {
            for ((s, &v), &k) in acc.iter_mut().zip(t4.iter()).zip(h4.iter()) {
                *s += v as f64 * k;
            }
        }
        let b = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        let a = a as f64;
        d += (a - b) * (a - b);
        et += b * b;
    }
    Some(norm_corr(e0, et, d))
}

/// `Σ_{j<w} x_j²` in f64, in sample order (the head energy of [`nsdf_at`]).
fn head_energy(x: &[f32], w: usize) -> f64 {
    x.iter().take(w).fold(0.0f64, |e, &a| e + a as f64 * a as f64)
}

/// Normalised correlation of the head (energy `e0`) and its lagged copy
/// (`et`) from their squared difference `d = e0 + et − 2r`, attack-
/// compensated (module docs).
#[inline]
fn norm_corr(e0: f64, et: f64, d: f64) -> f64 {
    crate::eh::attack_corr(e0, et, 0.5 * (e0 + et - d))
}

/// Maximum of the exact fractional-lag correlation near `t0`: two 3-point
/// parabola steps on exact values with spacing 0.2 then 0.04 samples (the
/// parabola's bias shrinks with the square of the spacing). The head energy
/// is computed once, and the final value reuses the last step's centre when
/// the search stopped there.
fn exact_peak(x: &[f32], w: usize, t0: f64, bank: &[[f64; 2 * FIR_HALF]]) -> Option<(f64, f64)> {
    let e0 = head_energy(x, w);
    let at = |t: f64| nsdf_at(x, w, t, e0, bank);
    let mut t = t0;
    let mut centre = (f64::NAN, 0.0);
    for h in [0.2f64, 0.04] {
        let a = at(t - h)?;
        let b = at(t)?;
        let c = at(t + h)?;
        centre = (t, b);
        let curv = a - 2.0 * b + c;
        if curv.is_nan() || curv >= 0.0 {
            break;
        }
        let delta = (0.5 * (a - c) / curv).clamp(-1.0, 1.0);
        t += delta * h;
    }
    if centre.0.to_bits() == t.to_bits() {
        return Some((t, centre.1));
    }
    Some((t, at(t)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harmonic(hist: &mut Mirror, n: usize, period: f64, k: usize) {
        for i in 0..n {
            let ph = std::f64::consts::TAU * i as f64 / period;
            let x: f64 = (1..=k).map(|h| (h as f64 * ph + 0.3 * h as f64).sin() / h as f64).sum();
            hist.push(x as f32);
        }
    }

    #[test]
    fn parabolic_and_exact_refinement_accuracy() {
        // (period, harmonics): harmonics reach 0.2-0.25 of the rate on the
        // short periods, where the parabola is biased.
        let cases = [(31.5f64, 7usize), (34.29, 6), (37.61, 6), (55.3, 9), (90.7, 12), (218.18, 12), (436.4, 20)];
        for &(period, k) in &cases {
            let mut hist = Mirror::new(8192);
            harmonic(&mut hist, 4000, period, k);
            let mut r = Refiner::new(1000, 64);
            let p = r.refine(&hist, hist.len(), period * 1.004, 4, false, 0).unwrap();
            let s = r.refine(&hist, hist.len(), period * 1.004, 4, true, 0).unwrap();
            let cents = |x: f64| 1200.0 * (x / period).log2();
            assert!(cents(p.period).abs() < 0.6, "P={period}: parabolic {} c", cents(p.period));
            assert!(cents(s.period).abs() < 0.02, "P={period}: exact {} c", cents(s.period));
            assert!(p.clarity > 0.99 && s.clarity > 0.999);
            assert!((p.centre_back() - period).abs() < 1.0);
        }
    }

    #[test]
    fn attack_reads_periodic_and_unbiased() {
        // A harmonic tone whose amplitude doubles every period (+6 dB per
        // period, a fast attack). The refinement's correlation must see the
        // shape repeat (clarity ≈ 1; type-I is capped at 2ρ/(1 + ρ²) = 0.8)
        // and not be pulled to a shorter lag by the envelope (type-I's
        // sech(γτ) weighting).
        for &period in &[40.3f64, 87.6, 151.2] {
            let mut hist = Mirror::new(8192);
            let n = (8.0 * period) as usize;
            for i in 0..n {
                let ph = std::f64::consts::TAU * i as f64 / period;
                let x: f64 = (1..=8).map(|h| (h as f64 * ph + 0.3 * h as f64).sin() / h as f64).sum();
                hist.push((x * 2f64.powf(i as f64 / period - 8.0)) as f32);
            }
            let mut r = Refiner::new(1000, 64);
            let cents = |x: f64| 1200.0 * (x / period).log2();
            for exact in [false, true] {
                let p = r.refine(&hist, hist.len(), period * 1.01, 6, exact, 0).unwrap();
                // Type-I read clarity 0.80 and −5…−8 c (sharp) here. The
                // gain-normalised correlation is exact (Max path); the
                // parabola adds its own interpolation error on short periods.
                let limit = if exact { 0.05 } else { 3.0 };
                assert!(p.clarity > 0.99, "P={period} exact={exact}: clarity {}", p.clarity);
                assert!(cents(p.period).abs() < limit, "P={period} exact={exact}: {} c", cents(p.period));
            }
        }
    }

    #[test]
    fn band_shifts_when_seed_is_off() {
        let mut hist = Mirror::new(8192);
        harmonic(&mut hist, 4000, 100.0, 5);
        let mut r = Refiner::new(400, 32);
        // Seed 6 samples low with a ±3 band: two shifts reach the peak.
        let p = r.refine(&hist, hist.len(), 94.0, 3, false, 0).unwrap();
        assert!((p.period - 100.0).abs() < 0.05, "{}", p.period);
    }

    #[test]
    fn interpolated_fir_bank_matches_the_designed_filter() {
        // PERF-1: the exact search interpolates a precomputed tap bank instead
        // of designing 16 Kaiser-sinc taps from Bessel series per evaluation.
        // The interpolated taps must match the designed ones, and the
        // fractional-lag correlation must agree with a direct evaluation.
        let r = Refiner::new(400, 32);
        let mut worst_tap = 0.0f64;
        for i in 0..997 {
            let mu = (i as f64 * 0.618_034) % 1.0;
            let pos = mu * FIR_PHASES as f64;
            let k = (pos.floor() as usize).min(FIR_PHASES - 1);
            let f = pos - k as f64;
            let want = fir_taps(mu);
            for ((&w, &a), &b) in want.iter().zip(r.bank[k].iter()).zip(r.bank[k + 1].iter()) {
                worst_tap = worst_tap.max((a + f * (b - a) - w).abs());
            }
        }
        assert!(worst_tap < 2e-6, "tap error {worst_tap}");
        // Direct evaluation with freshly designed taps (the pre-bank code).
        let mut hist = Mirror::new(4096);
        harmonic(&mut hist, 3000, 37.3, 9);
        let mut x = vec![0.0f32; 400];
        hist.copy_reversed(hist.len(), 400, &mut x);
        let w = 64;
        let e0 = head_energy(&x, w);
        for &tau in &[30.0f64, 37.3, 37.31, 52.77, 74.6] {
            let h = fir_taps(tau - tau.floor());
            let first = tau.floor() as usize + 1 - FIR_HALF;
            let (mut d, mut et) = (0.0f64, 0.0f64);
            for j in 0..w {
                let b: f64 = (0..2 * FIR_HALF).map(|m| x[first + j + m] as f64 * h[m]).sum();
                let a = x[j] as f64;
                d += (a - b) * (a - b);
                et += b * b;
            }
            let want = norm_corr(e0, et, d);
            let got = nsdf_at(&x, w, tau, e0, &r.bank).unwrap();
            assert!((got - want).abs() < 1e-7, "tau {tau}: {got} vs {want}");
        }
    }

    #[test]
    fn invalid_inputs() {
        let hist = Mirror::new(1024);
        let mut r = Refiner::new(100, 16);
        assert!(r.refine(&hist, 0, f64::NAN, 3, false, 0).is_none());
        assert!(r.refine(&hist, 0, 1.0, 3, false, 0).is_none());
        // Silent history.
        assert!(r.refine(&hist, 0, 50.0, 3, true, 0).is_none());
        assert!(r.span() >= 2 * 100);
        assert!(nsdf_at(&[0.0; 10], 4, 3.0, 0.0, &r.bank).is_none());
        assert!(nsdf_at(&[0.0; 10], 4, f64::NAN, 0.0, &r.bank).is_none());
    }
}
