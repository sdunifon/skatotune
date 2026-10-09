//! Pitch-synchronous E/H candidate generator (Hildebrand, US 5,973,252,
//! expired; R§8.3).
//!
//! For every candidate lag L of the decimated search stream two running sums
//! are kept, updated once per decimated sample i:
//!
//! ```text
//! E_i(L) = E_{i−1}(L) + x_i²      − x_{i−2L}²             energy of the last 2L samples
//! H_i(L) = H_{i−1}(L) + x_i·x_{i−L} − x_{i−L}·x_{i−2L}    lag product over the last L pairs
//! E − 2H = Σ_{k=i−L+1..i} (x_k − x_{k−L})²   ⇒   (E − 2H)/E = 1 − NSDF over a 2L window
//! ```
//!
//! This is the patent form; one source printed a variant with `x[n−3L]` and
//! mixed window lengths, which breaks the identity (R§8.3). Because each lag's
//! window is exactly two of its own periods, the measure at the *true* period
//! only looks back about two periods of the sung note: the low-information-
//! lag path of the detector.
//!
//! **Attack gain compensation.** During a note's amplitude attack the newer
//! period is louder than the older one, and `(E − 2H)/E` (type-I) mistakes
//! that loudness change for aperiodicity: with an RMS ratio ρ between the
//! older and newer half it can never fall below `1 − 2ρ/(1 + ρ²)`. At an E2
//! onset (12 ms periods, 20 ms attack) the true period therefore stayed
//! "aperiodic" until the attack was over plus two periods (≈ 35 ms). A third
//! sum `E₁ = Σ x²` over the newer half (computed from the history when the
//! compensation is evaluated: one prefix sum over the lag range, PERF-2)
//! gives the older half's energy `E₂ = E − E₁`, and while the signal *grows* (`E₁ > E₂`) the height uses the
//! gain-invariant normalised cross-correlation (YAAPT/RAPT's NCCF)
//! `cos = H / √(E₁·E₂)` instead, faded out by `f(ρ) = 2u/(1 + u²)`, `u = ρ/ρ₀`,
//! when the older half is nearly silent (ρ < ρ₀), so the first glottal pulse
//! against silence is never "periodic", and blended in only where the two
//! periods have nearly the same shape (`cos` ≥ 0.8 … 0.95, [`attack_corr`]),
//! so growing noise or breath resonances gain nothing. At `E₁ = E₂` both
//! forms are equal, so steady notes and decays (releases keep type-I: no
//! longer voiced tails) are unchanged. The detector enables the compensation
//! only in an onset phase of a Causal analysis (`Detector::analyze`,
//! `ONSET_RISE_DB`): Studio's centred windows already see past the attack.
//!
//! Numerics: the accumulators are f64 (products of f32 samples are exact in
//! f64), and one lag is re-initialised from the history every
//! [`RENORM_STRIDE`] samples (round-robin), so rounding residue or a burst of
//! extreme input can never persist (R§8.3 "f64 accumulators, or periodic
//! re-init"; we do both).
//!
//! Studio (centred analysis): each lag reads its samples `off_L` positions
//! back, `off_L = max(d − L, 0)`, so every lag's 2L window is centred on the
//! same instant `d` samples before the newest (or as close as the data
//! allows). Causal analysis uses `off_L = 0`.

use crate::hist::Mirror;

/// One lag is recomputed from scratch every this many decimated samples.
const RENORM_STRIDE: u32 = 4;
/// Most E/H troughs reported per frame (shortest lags first).
pub(crate) const MAX_EH_TROUGHS: usize = 16;
/// Troughs with `(E − 2H)/E` above this are ignored (anti-correlated or
/// noise-level; the pYIN thresholds never reach them).
const MAX_TROUGH_HEIGHT: f32 = 0.8;
/// Attack gain compensation is complete down to this older/newer RMS ratio
/// (−10.5 dB) and fades out below it (module docs).
const GAIN_COMP_RHO: f64 = 0.3;
/// Gain-invariant similarity range over which the attack compensation is
/// blended in (none below `.0`, full above `.1`; see [`attack_corr`]).
const GAIN_COMP_COS: (f64, f64) = (0.8, 0.95);

/// A local minimum of `(E − 2H)/E`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct EhTrough {
    /// Fractional lag in search-stream samples (parabolic vertex).
    pub lag: f32,
    /// Interpolated minimum height, ≥ 0.
    pub height: f32,
}

/// Running E/H sums for every lag of the search range.
#[derive(Clone, Debug)]
pub(crate) struct EhTracker {
    lag_min: usize,
    /// Per lag (index k ↔ L = lag_min + k): offsets back from the newest sample
    /// of the entering sample, its L-lagged partner and its 2L-lagged partner.
    off0: Box<[usize]>,
    off1: Box<[usize]>,
    off2: Box<[usize]>,
    e: Box<[f64]>,
    h: Box<[f64]>,
    /// Energy of the newer half (the last L samples) per lag, scratch of
    /// [`EhTracker::evaluate`] with attack compensation.
    e1: Box<[f64]>,
    /// Ratio scratch `(E − 2H)/E` per lag, filled by [`EhTracker::evaluate`].
    ratio: Box<[f32]>,
    /// Round-robin re-initialisation state.
    renorm_lag: usize,
    renorm_phase: u32,
    /// History span the tracker needs (samples back from the newest).
    span: usize,
    /// Causal layout (`off_L = 0` for every lag): enables the contiguous /
    /// stride-2 fast path in [`EhTracker::update`].
    causal: bool,
}

impl EhTracker {
    /// Tracker for lags `lag_min ..= lag_max` (search-stream samples). With
    /// `centre_offset = Some(d)` every lag's window is centred `d` samples
    /// before the newest sample (Studio); `None` is causal.
    pub(crate) fn new(lag_min: usize, lag_max: usize, centre_offset: Option<usize>) -> Self {
        let lag_min = lag_min.max(1);
        let lag_max = lag_max.max(lag_min);
        let n = lag_max - lag_min + 1;
        let mut off0 = vec![0usize; n].into_boxed_slice();
        let mut off1 = vec![0usize; n].into_boxed_slice();
        let mut off2 = vec![0usize; n].into_boxed_slice();
        let mut span = 0;
        for (k, ((a, b), c)) in off0.iter_mut().zip(off1.iter_mut()).zip(off2.iter_mut()).enumerate() {
            let l = lag_min + k;
            // Window of lag l covers [newest − off − 2l + 1, newest − off]; its
            // centre newest − off − l + ½ sits at newest − d when off = d − l.
            let off = centre_offset.map_or(0, |d| d.saturating_sub(l));
            *a = off;
            *b = off + l;
            *c = off + 2 * l;
            span = span.max(*c + 1);
        }
        Self {
            lag_min,
            off0,
            off1,
            off2,
            e: vec![0.0; n].into_boxed_slice(),
            h: vec![0.0; n].into_boxed_slice(),
            e1: vec![0.0; n].into_boxed_slice(),
            ratio: vec![1.0; n].into_boxed_slice(),
            renorm_lag: 0,
            renorm_phase: 0,
            span,
            causal: centre_offset.is_none(),
        }
    }

    /// Samples of history (including the newest) the tracker reads.
    pub(crate) fn span(&self) -> usize {
        self.span
    }

    /// Smallest lag.
    #[cfg(test)]
    pub(crate) fn lag_min(&self) -> usize {
        self.lag_min
    }

    /// Number of lags.
    #[cfg(test)]
    pub(crate) fn n_lags(&self) -> usize {
        self.e.len()
    }

    /// Clears the accumulators (the caller clears the history).
    pub(crate) fn reset(&mut self) {
        self.e.fill(0.0);
        self.h.fill(0.0);
        self.e1.fill(0.0);
        self.ratio.fill(1.0);
        self.renorm_lag = 0;
        self.renorm_phase = 0;
    }

    /// Updates every lag for the sample just pushed into `hist` (its newest
    /// sample). Work: 3 reads + 4 flops per lag, plus one bounded
    /// re-initialisation every [`RENORM_STRIDE`] calls.
    pub(crate) fn update(&mut self, hist: &Mirror) {
        let Some(w) = hist.window(hist.len(), self.span) else {
            return;
        };
        let top = self.span - 1;
        let at = |o: usize| w.get(top.wrapping_sub(o)).copied().unwrap_or(0.0) as f64;
        let n = self.e.len();
        let (lmin, lmax) = (self.lag_min, self.lag_min + n - 1);
        if self.causal && 2 * lmax <= top {
            // x_{i−L} for L = lmin..=lmax is w[top − L] (reverse-contiguous),
            // x_{i−2L} is w[top − 2L] (reverse, stride 2): plain zipped
            // iterators, no bounds checks, vectorisable.
            let a = at(0);
            let bs = w.get(top - lmax..=top - lmin).unwrap_or(&[]);
            let cs = w.get(top - 2 * lmax..=top - 2 * lmin).unwrap_or(&[]);
            for ((e, h), (&b, &c)) in
                self.e.iter_mut().zip(self.h.iter_mut()).zip(bs.iter().rev().zip(cs.iter().rev().step_by(2)))
            {
                let (b, c) = (b as f64, c as f64);
                *e += a * a - c * c;
                *h += a * b - b * c;
            }
        } else {
            for (((e, h), &o0), (&o1, &o2)) in self
                .e
                .iter_mut()
                .zip(self.h.iter_mut())
                .zip(self.off0.iter())
                .zip(self.off1.iter().zip(self.off2.iter()))
            {
                let (a, b, c) = (at(o0), at(o1), at(o2));
                *e += a * a - c * c;
                *h += a * b - b * c;
            }
        }
        self.renorm_phase += 1;
        if self.renorm_phase >= RENORM_STRIDE {
            self.renorm_phase = 0;
            let k = self.renorm_lag;
            if let (Some(&o0), Some(e), Some(h)) = (self.off0.get(k), self.e.get_mut(k), self.h.get_mut(k)) {
                let l = self.lag_min + k;
                // E = Σ_{j<2L} x²[newest − o0 − j]; H = Σ_{j<L} x[..−j]·x[..−j−L].
                let (mut se, mut sh) = (0.0f64, 0.0f64);
                for j in 0..2 * l {
                    let v = at(o0 + j);
                    se += v * v;
                }
                for j in 0..l {
                    sh += at(o0 + j) * at(o0 + j + l);
                }
                *e = se;
                *h = sh;
            }
            self.renorm_lag = if k + 1 >= self.e.len() { 0 } else { k + 1 };
        }
    }

    /// Computes `(E − 2H)/E` for every lag (energies below `energy_floor` per
    /// sample count as silence → ratio 1), with `compensate` the attack-
    /// compensated height where the signal grows (module docs), and returns
    /// the per-lag ratios (index k ↔ lag `lag_min + k`). `hist` is the
    /// history the tracker has been updated with (read for `E₁` only).
    pub(crate) fn evaluate(&mut self, hist: &Mirror, energy_floor: f64, compensate: bool) -> &[f32] {
        if compensate {
            self.newer_half_energies(hist);
        }
        for (k, (((r, &e), &h), &e1)) in
            self.ratio.iter_mut().zip(self.e.iter()).zip(self.h.iter()).zip(self.e1.iter()).enumerate()
        {
            let floor = energy_floor * (2 * (self.lag_min + k)) as f64;
            *r = if e <= floor {
                1.0
            } else if compensate {
                height(e, e1, h)
            } else {
                ((e - 2.0 * h).max(0.0) / e) as f32
            };
        }
        &self.ratio
    }

    /// `E₁(L) = Σ_{j<L} x²[newest − off_L − j]` for every lag into `e1`:
    /// one running prefix sum in the causal layout (every window ends at the
    /// newest sample), a direct sum per lag otherwise.
    fn newer_half_energies(&mut self, hist: &Mirror) {
        let Some(w) = hist.window(hist.len(), self.span) else {
            self.e1.fill(0.0);
            return;
        };
        let top = self.span - 1;
        let at = |o: usize| w.get(top.wrapping_sub(o)).copied().unwrap_or(0.0) as f64;
        if self.causal {
            let mut acc = 0.0f64;
            for j in 0..self.lag_min.saturating_sub(1) {
                let v = at(j);
                acc += v * v;
            }
            for (k, e1) in self.e1.iter_mut().enumerate() {
                let v = at(self.lag_min + k - 1);
                acc += v * v;
                *e1 = acc;
            }
        } else {
            for (k, (e1, &o0)) in self.e1.iter_mut().zip(self.off0.iter()).enumerate() {
                let l = self.lag_min + k;
                *e1 = (0..l).map(|j| at(o0 + j)).fold(0.0f64, |s, v| s + v * v);
            }
        }
    }

    /// The ratios from the last [`EhTracker::evaluate`].
    #[cfg(test)]
    pub(crate) fn ratios(&self) -> &[f32] {
        &self.ratio
    }

    /// Local minima of the last evaluated ratios, shortest lag first, below
    /// [`MAX_TROUGH_HEIGHT`], with parabolic sub-sample vertices. Returns the
    /// number written to `out`.
    pub(crate) fn troughs(&self, out: &mut [EhTrough]) -> usize {
        // Interior minima only: `(E − 2H)/E` has no cumulative-mean
        // normalisation, so on smooth (low-pitched) input it is small at the
        // shortest lags and rises from the range edge; that edge is not a
        // period.
        find_troughs(&self.ratio, self.lag_min, MAX_TROUGH_HEIGHT, false, out)
    }

    /// Interpolated minimum of the ratio near fractional lag `lag` (searches
    /// ±`radius` lags for the smallest value, then refines parabolically).
    /// `None` outside the lag range.
    pub(crate) fn min_near(&self, lag: f32, radius: usize) -> Option<EhTrough> {
        min_near(&self.ratio, self.lag_min, lag, radius)
    }
}

/// Periodicity height of one lag from its running sums (`e` > 0): type-I
/// `(E − 2H)/E`, or while the newer half is louder (`E₁ > E₂`) the
/// attack-compensated `1 − f(ρ)·H/√(E₁E₂)` (module docs). Continuous at
/// `E₁ = E₂`, in `[0, 2]` like the type-I form.
#[inline]
fn height(e: f64, e1: f64, h: f64) -> f32 {
    let e1 = e1.clamp(0.0, e);
    (1.0 - attack_corr(e1, e - e1, h)).max(0.0) as f32
}

/// Normalised correlation of a newer segment (energy `e_new`) with an older
/// one (`e_old`), cross term `r`: type-I `2r/(e_new + e_old)`, or while the
/// signal grows (`e_new > e_old`) and the two segments have nearly the same
/// shape the attack-compensated `f(ρ)·r/√(e_new·e_old)`, with `ρ = √(e_old/e_new)`
/// and the fade `f` of the module docs. The compensation is blended in as
/// the gain-invariant similarity `cos = r/√(e_new·e_old)` rises from
/// [`GAIN_COMP_COS`]`.0` to `.1`: a clean glottal attack has `cos ≥ 0.95`
/// within two periods, while noise and breath resonances that merely grow
/// louder stay below (measured on vocadito: compensating every rising
/// segment raised false voicing far from any note from 3.8 % to 4.7 %).
/// Equal to type-I at `e_new = e_old`; in `[−1, 1]`; never below type-I.
/// Shared by the E/H heights and the refinement (`refine.rs`).
#[inline]
pub(crate) fn attack_corr(e_new: f64, e_old: f64, r: f64) -> f64 {
    let (e_new, e_old) = (e_new.max(0.0), e_old.max(0.0));
    let den1 = e_new + e_old;
    let type1 = if den1 > 0.0 { (2.0 * r / den1).clamp(-1.0, 1.0) } else { 0.0 };
    if e_new <= e_old {
        return type1;
    }
    let u = (e_old / e_new).sqrt() / GAIN_COMP_RHO;
    let fade = if u >= 1.0 { 1.0 } else { 2.0 * u / (1.0 + u * u) };
    let den = (e_new * e_old).sqrt();
    let cos = if den > 0.0 { (r / den).clamp(-1.0, 1.0) } else { 0.0 };
    let (c0, c1) = GAIN_COMP_COS;
    let w = ((cos - c0) / (c1 - c0)).clamp(0.0, 1.0);
    (w * fade * cos + (1.0 - w) * type1).max(type1)
}

/// Local minima of `y` (index k ↔ lag `lag0 + k`) below `max_height`, in lag
/// order, with parabolic vertices. With `edges`, edge samples count as minima
/// when lower than their single neighbour (librosa's `is_trough[0]`, right
/// for the cumulative-mean-normalised CMNDF).
pub(crate) fn find_troughs(y: &[f32], lag0: usize, max_height: f32, edges: bool, out: &mut [EhTrough]) -> usize {
    let n = y.len();
    let mut count = 0;
    for k in 0..n {
        if count >= out.len() {
            break;
        }
        let v = y[k];
        if v.is_nan() || v >= max_height {
            continue;
        }
        let left_ok = if k == 0 { edges } else { v < y[k - 1] };
        let right_ok = if k + 1 >= n { edges } else { v <= y[k + 1] };
        if !(left_ok && right_ok) || n < 2 {
            continue;
        }
        let (off, val) =
            if k > 0 && k + 1 < n { vt_core::math::parabolic_peak(y[k - 1], v, y[k + 1]) } else { (0.0, v) };
        out[count] = EhTrough { lag: (lag0 + k) as f32 + off, height: val.max(0.0) };
        count += 1;
    }
    count
}

/// Smallest value of `y` within ±`radius` of fractional lag `lag`, refined
/// parabolically. `None` when the lag is outside `y`'s range.
pub(crate) fn min_near(y: &[f32], lag0: usize, lag: f32, radius: usize) -> Option<EhTrough> {
    if !lag.is_finite() {
        return None;
    }
    let centre = (lag - lag0 as f32).round();
    if centre < -(radius as f32) || centre > (y.len() + radius) as f32 {
        return None;
    }
    let c = centre as i64;
    let lo = (c - radius as i64).max(0) as usize;
    let hi = ((c + radius as i64).max(0) as usize).min(y.len().checked_sub(1)?);
    if lo > hi {
        return None;
    }
    let mut best = lo;
    for k in lo..=hi {
        if y[k] < y[best] {
            best = k;
        }
    }
    let (off, val) = if best > 0 && best + 1 < y.len() {
        vt_core::math::parabolic_peak(y[best - 1], y[best], y[best + 1])
    } else {
        (0.0, y[best])
    };
    Some(EhTrough { lag: (lag0 + best) as f32 + off, height: val.max(0.0) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(tr: &mut EhTracker, hist: &mut Mirror, xs: impl Iterator<Item = f32>) {
        for x in xs {
            hist.push(x);
            tr.update(hist);
        }
    }

    /// Direct sums `(E, E₁, H)` with the window ending `off` samples before
    /// the newest.
    fn direct_sums(xs: &[f32], l: usize, off: usize) -> (f64, f64, f64) {
        let n = xs.len() - off;
        let (mut e, mut e1, mut h) = (0.0f64, 0.0f64, 0.0f64);
        for j in 0..2 * l {
            let v = xs[n - 1 - j] as f64;
            e += v * v;
            if j < l {
                e1 += v * v;
            }
        }
        for j in 0..l {
            h += xs[n - 1 - j] as f64 * xs[n - 1 - j - l] as f64;
        }
        (e, e1, h)
    }

    /// Direct height (type-I, attack-compensated while growing).
    fn direct(xs: &[f32], l: usize, off: usize) -> f32 {
        let (e, e1, h) = direct_sums(xs, l, off);
        height(e, e1, h)
    }

    /// Direct type-I `(E − 2H)/E`.
    fn direct_type1(xs: &[f32], l: usize, off: usize) -> f32 {
        let (e, _, h) = direct_sums(xs, l, off);
        ((e - 2.0 * h) / e) as f32
    }

    #[test]
    fn recursion_matches_direct_sums() {
        let xs: Vec<f32> = (0..3000).map(|i| (i as f32 * 0.21).sin() + 0.4 * (i as f32 * 0.537).cos()).collect();
        for centre in [None, Some(60usize)] {
            let mut tr = EhTracker::new(5, 50, centre);
            let mut hist = Mirror::new(256);
            feed(&mut tr, &mut hist, xs.iter().copied());
            let r = tr.evaluate(&hist, 1e-12, true).to_vec();
            for (k, &got) in r.iter().enumerate() {
                let l = 5 + k;
                let off = centre.map_or(0, |d| d.saturating_sub(l));
                let want = direct(&xs, l, off);
                assert!((got - want).abs() < 1e-5, "L={l} centre={centre:?}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn attack_is_compensated_but_decay_and_silence_are_not() {
        // Harmonic tone, period 40 samples, under exponential amplitude
        // envelopes: growing ×2 per period (−6 dB older half), decaying ×½,
        // and a first period against silence.
        let p = 40usize;
        let shape = |i: usize| {
            let ph = std::f64::consts::TAU * i as f64 / p as f64;
            (ph.sin() + 0.5 * (2.0 * ph + 0.4).sin() + 0.3 * (3.0 * ph + 1.0).sin()) as f32
        };
        let env = |rate: f64| -> Vec<f32> {
            (0..6 * p).map(|i| shape(i) * (rate.powf(i as f64 / p as f64)) as f32).collect()
        };
        let grow = env(2.0);
        // Type-I is limited to 1 − 2ρ/(1 + ρ²) = 0.2 at ρ = ½ …
        assert!((direct_type1(&grow, p, 0) - 0.2).abs() < 0.01, "{}", direct_type1(&grow, p, 0));
        // … the attack-compensated height sees the exact periodicity.
        assert!(direct(&grow, p, 0) < 1e-6, "{}", direct(&grow, p, 0));
        // A wrong lag stays aperiodic.
        assert!(direct(&grow, 3 * p / 4, 0) > 0.5, "{}", direct(&grow, 3 * p / 4, 0));
        // Decay (a release): unchanged type-I.
        let decay = env(0.5);
        assert_eq!(direct(&decay, p, 0), direct_type1(&decay, p, 0));
        assert!(direct(&decay, p, 0) > 0.15);
        // First period against silence: no evidence (fade to ratio 1).
        let mut first = vec![0.0f32; p];
        first.extend((0..p).map(shape));
        assert!(direct(&first, p, 0) > 0.999, "{}", direct(&first, p, 0));
        // Continuity at E₁ = E₂ and the fade corner.
        assert!((height(2.0, 1.0, 0.7) - height(2.0, 1.0 + 1e-9, 0.7)).abs() < 1e-6);
        let rho2 = GAIN_COMP_RHO * GAIN_COMP_RHO;
        let a = height(1.0 + rho2 * (1.0 - 1e-9), 1.0, 0.9 * GAIN_COMP_RHO);
        let b = height(1.0 + rho2 * (1.0 + 1e-9), 1.0, 0.9 * GAIN_COMP_RHO);
        assert!((a - b).abs() < 1e-6, "{a} {b}");
        // White noise growing ×2 per "period" is still aperiodic.
        let mut s = 0x9e37_79b9u32;
        let noise: Vec<f32> = (0..6 * p)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s as f32 / u32::MAX as f32) - 0.5) * 2f32.powf(i as f32 / p as f32)
            })
            .collect();
        assert!(direct(&noise, p, 0) > 0.6, "{}", direct(&noise, p, 0));
    }

    #[test]
    fn finds_the_period_of_a_harmonic_tone() {
        // Period 37.5 samples: first trough near 37.5, octave relative near 75.
        let p = 37.5f32;
        let xs = (0..2000).map(|i| {
            let ph = std::f32::consts::TAU * i as f32 / p;
            ph.sin() + 0.5 * (2.0 * ph).sin() + 0.25 * (3.0 * ph + 1.0).sin()
        });
        let mut tr = EhTracker::new(8, 100, None);
        let mut hist = Mirror::new(256);
        feed(&mut tr, &mut hist, xs);
        tr.evaluate(&hist, 1e-9, true);
        let mut t = [EhTrough::default(); MAX_EH_TROUGHS];
        let n = tr.troughs(&mut t);
        let first = t[..n].iter().find(|t| t.height < 0.1).expect("periodic trough");
        assert!((first.lag - p).abs() < 0.3, "{:?}", &t[..n]);
        assert!(t[..n].iter().any(|t| (t.lag - 2.0 * p).abs() < 0.5 && t.height < 0.1));
        let m = tr.min_near(38.0, 2).unwrap();
        assert!((m.lag - p).abs() < 0.3 && m.height < 0.05);
        assert!(tr.min_near(f32::NAN, 2).is_none());
        assert!(tr.min_near(1e9, 2).is_none());
    }

    #[test]
    fn extreme_input_does_not_persist() {
        // A huge burst destroys f64 precision for a while; round-robin
        // re-initialisation must restore exact sums afterwards.
        let mut tr = EhTracker::new(4, 40, None);
        let mut hist = Mirror::new(128);
        let mut xs: Vec<f32> = (0..200).map(|i| (i as f32 * 0.3).sin()).collect();
        xs.extend((0..10).map(|_| 1e15f32));
        xs.extend((0..1000).map(|i| (i as f32 * 0.3).sin()));
        feed(&mut tr, &mut hist, xs.iter().copied());
        let r = tr.evaluate(&hist, 1e-12, true).to_vec();
        for (k, &got) in r.iter().enumerate() {
            let want = direct(&xs, 4 + k, 0);
            assert!((got - want).abs() < 1e-4, "L={}: {got} vs {want}", 4 + k);
        }
        // Silence after the signal → ratio 1 everywhere (energy floor).
        feed(&mut tr, &mut hist, std::iter::repeat_n(0.0, 400));
        assert!(tr.evaluate(&hist, 1e-10, true).iter().all(|&r| r == 1.0));
        tr.reset();
        assert!(tr.ratios().iter().all(|&r| r == 1.0));
        assert_eq!(tr.lag_min(), 4);
        assert_eq!(tr.n_lags(), 37);
    }
}
