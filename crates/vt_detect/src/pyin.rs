//! pYIN-style candidate probabilities (R§8.4).
//!
//! pYIN (Mauch & Dixon 2014; we follow librosa's `pyin`, ISC) turns one
//! difference-function frame into a *distribution* over period candidates
//! instead of a single YIN pick:
//!
//! 1. A prior over 100 thresholds `s = 0.01 … 1.00` (Beta(2, 18) in pYIN).
//! 2. For each threshold, the troughs whose height is below it are ranked by
//!    lag, and a Boltzmann prior (λ = 2) over that rank shares the threshold's
//!    weight among them, so the *first* (shortest-lag) qualifying trough gets
//!    ≈ 86 % of it. This is YIN's "first dip below threshold" rule made soft.
//! 3. A threshold that no trough passes gives `no_trough_prob = 0.01` of its
//!    weight to the global minimum.
//!
//! The probabilities sum to the frame's voiced probability (≤ 1).
//!
//! The same routine scores two kinds of "heights" here: type-I CMNDF troughs
//! of the long verifier frame, and `(E−2H)/E = 1 − NSDF` minima of the
//! pitch-synchronous E/H generator (and of the short verifier frame). Only the
//! threshold prior differs: its mean follows the Tracking knob (R§8.3, R§8.4
//! "recalibrate the Beta prior").

/// Number of thresholds (librosa default).
pub(crate) const N_THRESHOLDS: usize = 100;
/// Most troughs scored per frame. Troughs are supplied in lag order and the
/// Boltzmann prior gives a rank-32 trough ≈ e⁻⁶⁴ of the first one's weight, so
/// the cap never changes a meaningful probability.
pub(crate) const MAX_TROUGHS: usize = 32;
/// Boltzmann (rank) prior parameter λ (librosa default).
const BOLTZMANN: f64 = 2.0;
/// Weight given to the global minimum by thresholds no trough passes.
const NO_TROUGH_PROB: f32 = 0.01;

/// Prior over the 100 thresholds: Beta(2, b) probability mass per bin.
#[derive(Clone, Debug)]
pub(crate) struct ThresholdPrior {
    /// `w[k]` is the prior mass of threshold `s = (k + 1) / 100`.
    w: [f32; N_THRESHOLDS],
    /// Prefix sums `cum[k] = Σ_{k' < k} w[k']`.
    cum: [f32; N_THRESHOLDS + 1],
    /// Mean of the Beta distribution currently loaded.
    mean: f32,
}

impl ThresholdPrior {
    /// Beta(2, b) prior with the given mean `2 / (2 + b)` (clamped to
    /// [0.02, 0.5]). pYIN's Beta(2, 18) is `mean = 0.1`.
    pub(crate) fn with_mean(mean: f32) -> Self {
        let mut p = Self { w: [0.0; N_THRESHOLDS], cum: [0.0; N_THRESHOLDS + 1], mean: f32::NAN };
        p.set_mean(mean);
        p
    }

    /// Reloads the prior for a new mean (closed form, no allocation; skipped
    /// when the mean is unchanged).
    pub(crate) fn set_mean(&mut self, mean: f32) {
        let mean = if mean.is_finite() { mean.clamp(0.02, 0.5) } else { 0.1 };
        if mean == self.mean {
            return;
        }
        self.mean = mean;
        let b = 2.0 / mean as f64 - 2.0;
        // Beta(2, b) CDF in closed form: I_x(2, b) = 1 − (1 − x)^b (1 + b x).
        let cdf = |x: f64| 1.0 - (1.0 - x).powf(b) * (1.0 + b * x);
        let mut prev = 0.0;
        let mut acc = 0.0f64;
        self.cum[0] = 0.0;
        for (k, w) in self.w.iter_mut().enumerate() {
            let c = cdf((k + 1) as f64 / N_THRESHOLDS as f64);
            let m = (c - prev).max(0.0);
            *w = m as f32;
            prev = c;
            acc += m;
            self.cum[k + 1] = acc as f32;
        }
    }

    /// Mean of the loaded prior.
    #[cfg(test)]
    pub(crate) fn mean(&self) -> f32 {
        self.mean
    }
}

/// Precomputed Boltzmann rank prior `pmf(r; λ, N) = (1 − e^−λ) e^(−λ r) / (1 − e^(−λ N))`.
#[derive(Clone, Debug)]
pub(crate) struct RankPrior {
    /// `e^(−λ r)` for r < MAX_TROUGHS.
    decay: [f32; MAX_TROUGHS],
    /// `(1 − e^−λ) / (1 − e^(−λ N))` for N = 1..=MAX_TROUGHS (index N − 1).
    norm: [f32; MAX_TROUGHS],
}

impl RankPrior {
    pub(crate) fn new() -> Self {
        let mut decay = [0.0; MAX_TROUGHS];
        let mut norm = [0.0; MAX_TROUGHS];
        let head = 1.0 - (-BOLTZMANN).exp();
        for r in 0..MAX_TROUGHS {
            decay[r] = (-BOLTZMANN * r as f64).exp() as f32;
            norm[r] = (head / (1.0 - (-BOLTZMANN * (r + 1) as f64).exp())) as f32;
        }
        Self { decay, norm }
    }
}

/// Index of the first threshold `s_k = (k + 1)/100` that `h` is below
/// (`N_THRESHOLDS` = never): trough `h` takes part in thresholds `k ≥ first_below(h)`.
fn first_below(h: f32) -> usize {
    if h.is_nan() {
        return N_THRESHOLDS;
    }
    let s = |k: usize| (k + 1) as f32 / N_THRESHOLDS as f32;
    let mut k = (h * N_THRESHOLDS as f32).floor().clamp(0.0, N_THRESHOLDS as f32) as usize;
    // Exact against the same float comparison `h < s_k` (≤ 2 steps).
    while k > 0 && h < s(k - 1) {
        k -= 1;
    }
    while k < N_THRESHOLDS && h >= s(k) {
        k += 1;
    }
    k
}

/// Assigns pYIN probabilities to troughs.
///
/// `heights` are trough heights **in lag order** (shortest lag first; at most
/// [`MAX_TROUGHS`] are used). Writes one probability per trough into
/// `probs[..heights.len()]` and returns their sum (the voiced probability,
/// ≤ 1).
///
/// Equivalent to the per-threshold loop of librosa, evaluated per *segment*:
/// the set of troughs below a threshold only changes at the troughs' own
/// heights, so the thresholds between two consecutive heights share one rank
/// assignment and their prior mass is a prefix-sum difference. Bounded work:
/// O(T²) for T ≤ 32 troughs.
pub(crate) fn trough_probabilities(
    heights: &[f32],
    prior: &ThresholdPrior,
    rank: &RankPrior,
    probs: &mut [f32],
) -> f32 {
    let n = heights.len().min(MAX_TROUGHS).min(probs.len());
    let (heights, probs) = (&heights[..n], &mut probs[..n]);
    probs.fill(0.0);
    if n == 0 {
        return 0.0;
    }
    let mut first = [N_THRESHOLDS; MAX_TROUGHS];
    let mut gmin = 0;
    for (i, (&h, f)) in heights.iter().zip(first.iter_mut()).enumerate() {
        *f = first_below(h);
        if h < heights[gmin] || heights[gmin].is_nan() {
            gmin = i;
        }
    }
    let first = &first[..n];
    let start0 = first.iter().copied().min().unwrap_or(N_THRESHOLDS);
    // Thresholds below every trough: their mass goes to the global minimum.
    let no_trough_mass = prior.cum[start0];
    let mut start = start0;
    while start < N_THRESHOLDS {
        let next = first.iter().copied().filter(|&k| k > start).min().unwrap_or(N_THRESHOLDS);
        let wseg = prior.cum[next] - prior.cum[start];
        let active = first.iter().filter(|&&k| k <= start).count();
        if wseg > 0.0 && active > 0 {
            let norm = rank.norm[active - 1] * wseg;
            let mut r = 0;
            for (p, &k) in probs.iter_mut().zip(first.iter()) {
                if k <= start {
                    *p += norm * rank.decay[r];
                    r += 1;
                }
            }
        }
        start = next;
    }
    probs[gmin] += NO_TROUGH_PROB * no_trough_mass;
    probs.iter().sum::<f32>().min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_prior_matches_pyin_default() {
        let p = ThresholdPrior::with_mean(0.1);
        let total: f32 = p.w.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "mass {total}");
        // Beta(2,18) mode is 1/18 ≈ 0.056: the 5th-6th thresholds dominate.
        let kmax = (0..N_THRESHOLDS).max_by(|&a, &b| p.w[a].total_cmp(&p.w[b])).unwrap();
        assert!((4..=6).contains(&kmax), "mode at k={kmax}");
        // Mean of the discretised prior ≈ 0.1.
        let mean: f32 = p.w.iter().enumerate().map(|(k, w)| w * (k as f32 + 0.5) / 100.0).sum();
        assert!((mean - 0.1).abs() < 0.005, "mean {mean}");
        assert_eq!(p.mean(), 0.1);
        // Clamping and NaN.
        assert_eq!(ThresholdPrior::with_mean(f32::NAN).mean(), 0.1);
        assert_eq!(ThresholdPrior::with_mean(5.0).mean(), 0.5);
    }

    #[test]
    fn first_deep_trough_gets_most_mass() {
        let prior = ThresholdPrior::with_mean(0.1);
        let rank = RankPrior::new();
        let mut probs = [0.0f32; 4];
        // A clean period trough (0.02) after a shallow formant trough (0.4)
        // and before its sub-octave (0.03).
        let vp = trough_probabilities(&[0.4, 0.02, 0.03, 0.9], &prior, &rank, &mut probs);
        // Thresholds 0.01 and 0.02 (≈ 5 % of the prior) pass no trough.
        assert!(vp > 0.93 && vp <= 1.0, "vp {vp}");
        assert!(probs[1] > 0.8, "{probs:?}");
        assert!(probs[2] < 0.15 && probs[2] > 0.0, "{probs:?}");
        assert!(probs[0] < 0.05, "{probs:?}");
        assert_eq!(probs[3], 0.0);
        // Noise-like frame: every trough is high → low voiced probability,
        // all of it on the global minimum.
        let vp = trough_probabilities(&[0.8, 0.7, 0.85], &prior, &rank, &mut probs);
        assert!(vp < 0.02, "vp {vp}");
        assert!(probs[1] > 100.0 * (probs[0] + probs[2]), "{probs:?}");
        // Empty input.
        assert_eq!(trough_probabilities(&[], &prior, &rank, &mut probs), 0.0);
    }

    /// librosa's per-threshold loop, as a reference.
    fn reference(heights: &[f32], prior: &ThresholdPrior, rank: &RankPrior) -> Vec<f32> {
        let mut probs = vec![0.0f32; heights.len()];
        let gmin = (0..heights.len()).min_by(|&a, &b| heights[a].total_cmp(&heights[b])).unwrap();
        let mut none = 0.0;
        for (k, &w) in prior.w.iter().enumerate() {
            let s = (k + 1) as f32 / 100.0;
            let below = heights.iter().filter(|&&h| h < s).count();
            if below == 0 {
                none += w;
                continue;
            }
            let mut r = 0;
            for (p, &h) in probs.iter_mut().zip(heights) {
                if h < s {
                    *p += rank.norm[below - 1] * w * rank.decay[r];
                    r += 1;
                }
            }
        }
        probs[gmin] += NO_TROUGH_PROB * none;
        probs
    }

    #[test]
    fn segment_form_matches_per_threshold_loop() {
        let rank = RankPrior::new();
        let mut s = 0x2545_f491u32;
        for case in 0..500 {
            let n = 1 + case % 20;
            let heights: Vec<f32> = (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    // Include exact threshold values and out-of-range heights.
                    match s % 7 {
                        0 => (s % 100) as f32 / 100.0,
                        1 => 1.2,
                        _ => (s >> 8) as f32 / (1u32 << 24) as f32 * 0.8,
                    }
                })
                .collect();
            for mean in [0.05f32, 0.1, 0.2] {
                let prior = ThresholdPrior::with_mean(mean);
                let mut got = vec![0.0f32; n];
                trough_probabilities(&heights, &prior, &rank, &mut got);
                let want = reference(&heights, &prior, &rank);
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() < 1e-5, "{heights:?}: {got:?} vs {want:?}");
                }
            }
        }
    }

    #[test]
    fn relaxed_prior_accepts_breathier_troughs() {
        let rank = RankPrior::new();
        let mut probs = [0.0f32; 1];
        let strict = trough_probabilities(&[0.2], &ThresholdPrior::with_mean(0.05), &rank, &mut probs);
        let relaxed = trough_probabilities(&[0.2], &ThresholdPrior::with_mean(0.2), &rank, &mut probs);
        assert!(relaxed > strict + 0.3, "relaxed {relaxed} strict {strict}");
    }
}
