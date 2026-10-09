//! Per-hop candidate set: merging of the three evidence sources, probability
//! combination and octave checks (R§8.3-R§8.5).
//!
//! Sources, each with its own pYIN-style distribution over its troughs:
//!
//! * **fast**: E/H minima of the pitch-synchronous generator (2 periods of
//!   the sung note; reacts within a few periods to onsets, glides, vibrato);
//! * **long**: type-I CMNDF troughs of the W = 2.2/f_min verifier frame (the
//!   robust octave/voicing evidence; slow to react);
//! * **short** (HQ/Max): weighted type-I minima of the W/2 frame.
//!
//! Troughs within ±[`MERGE_CENTS`] are one candidate. The observation
//! probability is the mixture of the sources' distributions, weighted by each
//! source's voiced probability, scaled to the largest voiced probability:
//! `p_c = (Σ_s p_{s,c}) · max_s vp_s / Σ_s vp_s`. Taking the *largest*
//! voiced probability lets the fast path declare an onset before the long
//! frame has filled; the voicing state machine (fast clarity + gate) stops the
//! long frame from keeping an offset voiced.
//!
//! Octave checks then move probability between octave-related candidates
//! (period ratio 2 ± 4 %). Periodicity is compared as `g = 1 − nsdf` on one
//! common measure: the full-rate pitch-synchronous refinement when both
//! candidates have it (short periods, where decimated troughs are too
//! coarse: at 1400 Hz and 16 kHz a trough between samples reads 10× too
//! shallow), else the long verifier frame, plus the E/H heights.
//!
//! * **octave-up error** (the shorter lag a is really half the period b, the
//!   F1 ≈ 2·F0 case, where `g(a) ≈ 2·E_odd/E`): b must be *clearly* more
//!   periodic, `g(b) + 0.005 < ½ g(a)` (Hildebrand: "prefer the larger lag
//!   only if its (E−2H)/E ≤ that of the smaller lag + margin", made strict so
//!   a true period, whose double is never twice as periodic, cannot lose to
//!   its sub-octave). The prime-harmonic score can veto the move only when it
//!   strongly prefers a *and* the time-domain measures are not decisive
//!   (`g(b) < 0.2·g(a)` on every measure overrides it: the score is itself
//!   fooled by an F1-boosted 2nd harmonic).
//! * **sub-octave error** (b is twice the true period a): a is at least as
//!   periodic as b within a margin (Praat's octave cost, favouring the higher
//!   f) and, when available, its harmonic score is higher.
//! * **rough phonation** (period doubling: 2T slightly more periodic than T,
//!   but T is heard): when a is strongly periodic with a clearly positive
//!   harmonic score, b is too low for a score of its own and b is not
//!   decisively more periodic, a wins both rules above ([`rough_pair`]).
//!
//! A **third-harmonic check** (ratio 3) applies the octave-up test to a
//! candidate at 3·f against one at f (a dominant 3rd harmonic, F1 ≈ 3·f0).

/// Most candidates per hop.
pub(crate) const MAX_CANDS: usize = 12;
/// Troughs closer than this (cents) are merged into one candidate.
const MERGE_CENTS: f32 = 40.0;
/// Octave relation tolerance (cents around ±1200).
const OCTAVE_TOL_CENTS: f32 = 70.0;
/// A twelfth (frequency ratio 3), cents.
const TWELFTH_CENTS: f32 = 1901.955;
/// Share of probability moved by a decided octave check.
const OCTAVE_TRANSFER: f32 = 0.9;
/// Absolute margin of the octave-up rule (`g(b) + margin < ½ g(a)`).
const UP_MARGIN: f32 = 0.005;
/// `g(b) < DECISIVE_RATIO · g(a)` on the time-domain measures makes the
/// lower candidate b decisively more periodic than a (octave or twelfth).
const DECISIVE_RATIO: f32 = 0.2;
/// Rough-phonation test ([`rough_pair`]): minimum prime-harmonic score of
/// the upper candidate (the score's scale: in contested octave pairs on
/// vocadito the right upper candidate scored a median of 22, an octave-up
/// error 6) …
const ROUGH_MIN_HARM: f32 = 12.0;
/// … and its largest long-frame and E/H heights (it must be clearly
/// periodic itself).
const ROUGH_MAX_G_LONG: f32 = 0.3;
const ROUGH_MAX_H_FAST: f32 = 0.35;
/// Time-domain rough-phonation test ([`rough_pair_td`], Eco): the upper
/// candidate's largest long-frame and E/H heights …
const ROUGH_TD_MAX_G: f32 = 0.3;
/// … and the share of them the lower candidate must at least reach (it is
/// not decisively more periodic).
const ROUGH_TD_RATIO: f32 = 0.2;
/// "Unknown" value for the periodicity heights (1 − nsdf, E/H ratio).
pub(crate) const UNKNOWN_HEIGHT: f32 = 2.0;

/// Which analysis produced a trough.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    Fast,
    Long,
    Short,
}

/// One pitch candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cand {
    /// Pitch, absolute cents.
    pub cents: f32,
    /// Seed period, full-rate samples.
    pub period: f32,
    /// Refinement half-band, full-rate samples.
    pub band: f32,
    /// Seed came from the E/H generator (newest evidence).
    pub fast_seed: bool,
    pub p_fast: f32,
    pub p_long: f32,
    pub p_short: f32,
    /// `1 − nsdf` on the long frame at this period.
    pub g_long: f32,
    /// The same with attack compensation (voicing-onset check only).
    pub g_onset: f32,
    /// `1 − nsdf` of the full-rate pitch-synchronous refinement (short
    /// periods only, where decimated troughs are too coarse to compare).
    pub g_full: f32,
    /// `(E − 2H)/E` minimum near this period.
    pub h_fast: f32,
    /// Prime-harmonic score (NaN when unavailable).
    pub harm: f32,
    /// Final observation probability.
    pub p: f32,
}

impl Default for Cand {
    fn default() -> Self {
        Self {
            cents: 0.0,
            period: 0.0,
            band: 0.0,
            fast_seed: false,
            p_fast: 0.0,
            p_long: 0.0,
            p_short: 0.0,
            g_long: UNKNOWN_HEIGHT,
            g_onset: UNKNOWN_HEIGHT,
            g_full: UNKNOWN_HEIGHT,
            h_fast: UNKNOWN_HEIGHT,
            harm: f32::NAN,
            p: 0.0,
        }
    }
}

/// Fixed-capacity candidate list.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CandSet {
    c: [Cand; MAX_CANDS],
    n: usize,
}

impl Default for CandSet {
    fn default() -> Self {
        Self { c: [Cand::default(); MAX_CANDS], n: 0 }
    }
}

impl CandSet {
    pub(crate) fn clear(&mut self) {
        self.n = 0;
    }

    pub(crate) fn as_slice(&self) -> &[Cand] {
        &self.c[..self.n]
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [Cand] {
        &mut self.c[..self.n]
    }

    /// Index of the candidate within ±`tol` cents of `cents`, if any.
    pub(crate) fn find(&self, cents: f32, tol: f32) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.as_slice().iter().enumerate() {
            let d = (c.cents - cents).abs();
            if d <= tol && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Adds a trough from `src` with probability `p`. Merges into an existing
    /// candidate within ±40 c; when full, replaces the least probable
    /// candidate if `p` is larger. Returns the candidate index.
    pub(crate) fn add(&mut self, src: Source, cents: f32, period: f32, band: f32, p: f32) -> Option<usize> {
        if !(cents.is_finite() && period.is_finite() && period > 0.0) {
            return None;
        }
        let p = if p.is_finite() { p.max(0.0) } else { 0.0 };
        let idx = match self.find(cents, MERGE_CENTS) {
            Some(i) => i,
            None if self.n < MAX_CANDS => {
                self.n += 1;
                self.c[self.n - 1] = Cand { cents, period, band, ..Cand::default() };
                self.n - 1
            }
            None => {
                let (i, weakest) = self
                    .as_slice()
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, c.p_fast + c.p_long + c.p_short))
                    .min_by(|a, b| a.1.total_cmp(&b.1))?;
                if weakest >= p {
                    return None;
                }
                self.c[i] = Cand { cents, period, band, ..Cand::default() };
                i
            }
        };
        let c = &mut self.c[idx];
        match src {
            Source::Fast => {
                c.p_fast += p;
                // E/H describes the newest audio: prefer it as the seed.
                if !c.fast_seed {
                    c.period = period;
                    c.band = band;
                    c.cents = cents;
                    c.fast_seed = true;
                }
            }
            Source::Long => c.p_long += p,
            Source::Short => c.p_short += p,
        }
        Some(idx)
    }

    /// Combines the three sources into `p` (see module docs). Returns the
    /// combined voiced probability.
    pub(crate) fn combine(&mut self, vp_fast: f32, vp_long: f32, vp_short: f32) -> f32 {
        let vps = [vp_fast, vp_long, vp_short].map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 });
        let vmax = vps.iter().copied().fold(0.0f32, f32::max);
        let vsum: f32 = vps.iter().sum();
        let scale = if vsum > 0.0 { vmax / vsum } else { 0.0 };
        let mut total = 0.0;
        for c in self.as_mut_slice() {
            c.p = (c.p_fast + c.p_long + c.p_short) * scale;
            total += c.p;
        }
        total.min(1.0)
    }

    /// Octave checks (see module docs). Moves probability between octave
    /// pairs; total probability is unchanged.
    #[cfg(test)]
    pub(crate) fn octave_checks(&mut self) {
        self.octave_checks_with(false);
    }

    /// [`CandSet::octave_checks`]; `time_domain_rough` also applies the
    /// rough-phonation rule without a harmonic score ([`rough_pair_td`],
    /// Eco).
    pub(crate) fn octave_checks_with(&mut self, time_domain_rough: bool) {
        let n = self.n;
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                let (ca, cb) = (self.c[a], self.c[b]);
                // b one octave below a (period twice a's).
                if ((ca.cents - cb.cents) - 1200.0).abs() > OCTAVE_TOL_CENTS {
                    continue;
                }
                // One common periodicity measure for the pair.
                let full = known(ca.g_full) && known(cb.g_full);
                let (ga, gb) = if full { (ca.g_full, cb.g_full) } else { (ca.g_long, cb.g_long) };
                let g_known = known(ga) && known(gb);
                let t_up = g_known && gb + UP_MARGIN < 0.5 * ga;
                // E/H heights are search-rate measures: only without full-rate ones.
                let f_up = !full && known(ca.h_fast) && known(cb.h_fast) && cb.h_fast + UP_MARGIN < 0.5 * ca.h_fast;
                let harm_known = ca.harm.is_finite() && cb.harm.is_finite();
                // The time-domain measures are decisive when b is several
                // times more periodic than a on every measure available
                // (`g(b) < k·g(a)`); the harmonic score may then not veto. It
                // is fooled by a strong second harmonic (F1 ≈ 2·f0, weak odd
                // harmonics) exactly in the case the octave-up rule exists for
                // (vocadito: 69 % of the vetoes were wrong, at a median
                // g(b)/g(a) of 0.04 against 0.70 for the right ones).
                let decisive = known(ga)
                    && known(gb)
                    && gb < DECISIVE_RATIO * ga
                    && (full || (known(ca.h_fast) && known(cb.h_fast) && cb.h_fast < DECISIVE_RATIO * ca.h_fast));
                let rough = rough_pair(&ca, &cb) || (time_domain_rough && rough_pair_td(&ca, &cb));
                let harm_veto =
                    (harm_known && !decisive && (cb.harm <= 0.0 || ca.harm > 1.5 * cb.harm)) || rough;
                if ca.p > 0.0 && (t_up || f_up) && !harm_veto {
                    // a is an octave-up error: hand its mass to b.
                    let m = OCTAVE_TRANSFER * ca.p;
                    self.c[a].p -= m;
                    self.c[b].p += m;
                    continue;
                }
                let sub_t = !known(gb) || (known(ga) && ga <= gb + 0.02);
                let sub_fast = full || !known(cb.h_fast) || ca.h_fast <= cb.h_fast + 0.02;
                let sub_harm = !harm_known || ca.harm > cb.harm;
                let a_measured = known(ga) || known(ca.h_fast);
                if cb.p > ca.p && a_measured && ((sub_t && sub_fast && sub_harm) || rough) {
                    // b is a sub-octave of the true period a.
                    let m = OCTAVE_TRANSFER * cb.p;
                    self.c[b].p -= m;
                    self.c[a].p += m;
                }
            }
        }
    }

    /// Indices of the candidates in octave pairs whose full-rate measures
    /// would trigger the octave-up rule (`g_full(b) + margin < ½ g_full(a)`)
    /// against an upper candidate `a` that holds at least as much pYIN mass
    /// as `b` (the move would change the winner; the octave relative of a
    /// true pitch, which always satisfies the rule, normally holds none), at
    /// most `out.len()`, pair members adjacent. The caller re-measures them at
    /// the exact fractional lag before [`CandSet::octave_checks`]: the
    /// parabolic vertex under-reads a sharp NSDF peak between integer lags
    /// (a half-integer period with partials near Nyquist), which makes the
    /// true period look aperiodic next to its integer-lag sub-octave.
    pub(crate) fn octave_up_suspects(&self, out: &mut [usize]) -> usize {
        let mut k = 0;
        let mass = |c: &Cand| c.p_fast + c.p_long + c.p_short;
        for (a, ca) in self.as_slice().iter().enumerate() {
            for (b, cb) in self.as_slice().iter().enumerate() {
                if a == b || ((ca.cents - cb.cents) - 1200.0).abs() > OCTAVE_TOL_CENTS || mass(ca) < mass(cb) {
                    continue;
                }
                if !(known(ca.g_full) && known(cb.g_full) && cb.g_full + UP_MARGIN < 0.5 * ca.g_full) {
                    continue;
                }
                for i in [a, b] {
                    if k < out.len() && !out[..k].contains(&i) {
                        out[k] = i;
                        k += 1;
                    }
                }
            }
        }
        k
    }

    /// Third-harmonic check (R§8.5): a candidate at 3·f whose period T/3 is
    /// clearly *less* periodic than the candidate at f is the classic YIN
    /// harmonic error (a dominant third harmonic, F1 ≈ 3·f0, gives the
    /// shortest-lag trough, which pYIN's rank prior favours): hand its mass
    /// to f. Same strict test as the octave-up rule; the reverse (a sub-
    /// harmonic f/3 of a true pitch) is never more periodic than the pitch,
    /// so no reverse rule is needed.
    pub(crate) fn third_harmonic_checks(&mut self) {
        let n = self.n;
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                let (ca, cb) = (self.c[a], self.c[b]);
                if ((ca.cents - cb.cents) - TWELFTH_CENTS).abs() > OCTAVE_TOL_CENTS || ca.p <= 0.0 {
                    continue;
                }
                let full = known(ca.g_full) && known(cb.g_full);
                let (ga, gb) = if full { (ca.g_full, cb.g_full) } else { (ca.g_long, cb.g_long) };
                let t_up = known(ga) && known(gb) && gb + UP_MARGIN < 0.5 * ga;
                let f_up = known(ca.h_fast) && known(cb.h_fast) && cb.h_fast + UP_MARGIN < 0.5 * ca.h_fast;
                let harm_veto = (ca.harm.is_finite()
                    && cb.harm.is_finite()
                    && (cb.harm <= 0.0 || ca.harm > 1.5 * cb.harm))
                    || rough_pair(&ca, &cb);
                if t_up && f_up && !harm_veto {
                    let m = OCTAVE_TRANSFER * ca.p;
                    self.c[a].p -= m;
                    self.c[b].p += m;
                }
            }
        }
    }

    /// Scales every probability (gate / sibilance attenuation).
    pub(crate) fn scale(&mut self, k: f32) {
        for c in self.as_mut_slice() {
            c.p *= k;
        }
    }

    /// Index of the most probable candidate.
    pub(crate) fn best(&self) -> Option<usize> {
        self.as_slice().iter().enumerate().max_by(|x, y| x.1.p.total_cmp(&y.1.p)).map(|(i, _)| i)
    }
}

/// Rough-phonation test for a candidate `a` over its sub-harmonic `b`
/// (period 2T or 3T): `a` is strongly periodic on its own (long-frame and
/// E/H heights small), its prime-harmonic score is clearly positive (the
/// half harmonics, i.e. the subharmonics, are weak), `b` is too low for a
/// harmonic score of its own, and `b` is *not* decisively more periodic
/// (`g(b) ≥ k·g(a)` on both measures). Period-doubled ("rough", vocal-fry-
/// like) phonation alternates slightly between cycles, so 2T reads a little
/// more periodic than T, while listeners (and the vocadito annotators) hear
/// T. Calibrated on vocadito octave pairs where `b` held the mass: truth T
/// had median `g(b)/g(a)` 0.45 (10th percentile 0.25) and harmonic score 23;
/// truth 2T had 0.04 and 15.
fn rough_pair(ca: &Cand, cb: &Cand) -> bool {
    !cb.harm.is_finite()
        && ca.harm.is_finite()
        && ca.harm >= ROUGH_MIN_HARM
        && known(ca.g_long)
        && ca.g_long <= ROUGH_MAX_G_LONG
        && known(ca.h_fast)
        && ca.h_fast <= ROUGH_MAX_H_FAST
        && known(cb.g_long)
        && cb.g_long >= DECISIVE_RATIO * ca.g_long
        && known(cb.h_fast)
        && cb.h_fast >= DECISIVE_RATIO * ca.h_fast
}

/// Rough-phonation test without a spectrum (Eco has no harmonic score,
/// DESIGN §3.2): `a` is strongly periodic on every time-domain measure
/// (long frame, E/H and, when both have it, full rate) and `b` (2T) is not
/// decisively more periodic on any of them. Without the harmonic score's
/// evidence that `a`'s half harmonics are weak the periodicity limits are
/// stricter than [`rough_pair`]'s.
fn rough_pair_td(ca: &Cand, cb: &Cand) -> bool {
    let full = known(ca.g_full) && known(cb.g_full);
    !ca.harm.is_finite()
        && !cb.harm.is_finite()
        && known(ca.g_long)
        && ca.g_long <= ROUGH_TD_MAX_G
        && known(ca.h_fast)
        && ca.h_fast <= ROUGH_TD_MAX_G
        && known(cb.g_long)
        && cb.g_long >= ROUGH_TD_RATIO * ca.g_long
        && known(cb.h_fast)
        && cb.h_fast >= ROUGH_TD_RATIO * ca.h_fast
        && (!full || cb.g_full >= ROUGH_TD_RATIO * ca.g_full)
}

fn known(h: f32) -> bool {
    h.is_finite() && h < UNKNOWN_HEIGHT
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(cands: &[(f32, f32, f32, f32)]) -> CandSet {
        // (cents, p_long, g_long, h_fast)
        let mut s = CandSet::default();
        for &(cents, p, g, h) in cands {
            let i = s.add(Source::Long, cents, 100.0, 1.0, p).unwrap();
            s.c[i].g_long = g;
            s.c[i].h_fast = h;
        }
        s.combine(0.0, 0.9, 0.0);
        s
    }

    #[test]
    fn merge_and_combine() {
        let mut s = CandSet::default();
        s.add(Source::Long, 5000.0, 100.0, 1.0, 0.6);
        s.add(Source::Fast, 5020.0, 99.0, 2.0, 0.8);
        s.add(Source::Short, 4990.0, 101.0, 1.0, 0.3);
        assert_eq!(s.as_slice().len(), 1);
        let c = s.as_slice()[0];
        assert!(c.fast_seed && c.period == 99.0 && c.cents == 5020.0);
        let vp = s.combine(0.8, 0.6, 0.3);
        assert!((vp - 0.8).abs() < 1e-6, "{vp}");
        assert!((s.as_slice()[0].p - 0.8).abs() < 1e-6);
        assert!(s.add(Source::Long, f32::NAN, 1.0, 1.0, 0.1).is_none());
    }

    #[test]
    fn capacity_keeps_the_most_probable() {
        let mut s = CandSet::default();
        for i in 0..MAX_CANDS {
            s.add(Source::Long, 3000.0 + 100.0 * i as f32, 100.0, 1.0, 0.01 * (i + 1) as f32);
        }
        assert!(s.add(Source::Long, 9000.0, 100.0, 1.0, 0.001).is_none());
        let i = s.add(Source::Long, 9000.0, 100.0, 1.0, 0.5).unwrap();
        assert_eq!(s.as_slice()[i].cents, 9000.0);
        assert!(s.find(3000.0, 1.0).is_none(), "weakest was replaced");
    }

    #[test]
    fn octave_up_error_is_corrected() {
        // a = 2·f0 (F1 boosted second harmonic): wins pYIN but is clearly less
        // periodic than b = f0 on both measures.
        let mut s = set(&[(6200.0, 0.6, 0.15, 0.12), (5000.0, 0.3, 0.01, 0.01)]);
        s.octave_checks();
        let best = s.best().unwrap();
        assert_eq!(s.as_slice()[best].cents, 5000.0);
        let total: f32 = s.as_slice().iter().map(|c| c.p).sum();
        assert!((total - 0.9).abs() < 1e-5);
    }

    #[test]
    fn true_period_never_loses_to_its_sub_octave() {
        // Correct case: a = f0 (true), b = f0/2 equally periodic.
        let mut s = set(&[(5000.0, 0.7, 0.03, 0.02), (3800.0, 0.2, 0.025, 0.02)]);
        s.octave_checks();
        assert_eq!(s.as_slice()[s.best().unwrap()].cents, 5000.0);
        // Sub-octave holding most mass is moved back up.
        let mut s = set(&[(5000.0, 0.2, 0.03, 0.02), (3800.0, 0.7, 0.04, 0.03)]);
        s.octave_checks();
        assert_eq!(s.as_slice()[s.best().unwrap()].cents, 5000.0);
    }

    #[test]
    fn full_rate_measure_overrides_coarse_decimated_troughs() {
        // 1400 Hz tone: decimated troughs make the true pitch look 10× less
        // periodic than its sub-octave; full-rate measures say both are clean.
        let mut s = set(&[(7700.0, 0.8, 0.03, 0.045), (6500.0, 0.15, 0.004, 0.004)]);
        s.c[0].g_full = 0.0005;
        s.c[1].g_full = 0.0004;
        s.octave_checks();
        assert_eq!(s.as_slice()[s.best().unwrap()].cents, 7700.0);
        // Without the full-rate values the coarse troughs would mislead.
        let mut s = set(&[(7700.0, 0.8, 0.03, 0.045), (6500.0, 0.15, 0.004, 0.004)]);
        s.octave_checks();
        assert_eq!(s.as_slice()[s.best().unwrap()].cents, 6500.0);
    }

    /// A candidate as the detector measured it on vocadito: frequency,
    /// per-source pYIN mass (fast, long, short), long-frame and full-rate
    /// heights, E/H height, harmonic score (NaN: none).
    #[allow(clippy::too_many_arguments)]
    fn measured(s: &mut CandSet, hz: f32, pf: f32, pl: f32, ps: f32, gl: f32, gf: f32, h: f32, harm: f32) {
        let cents = 6900.0 + 1200.0 * (hz / 440.0).log2();
        let period = 44_100.0 / hz;
        let mut i = None;
        for (src, p) in [(Source::Fast, pf), (Source::Long, pl), (Source::Short, ps)] {
            i = s.add(src, cents, period, 1.0, p);
        }
        let c = &mut s.c[i.unwrap()];
        c.g_long = gl;
        c.g_full = gf;
        c.h_fast = h;
        c.harm = harm;
    }

    fn decide(s: &mut CandSet) -> f32 {
        let (f, l, sh) = s.as_slice().iter().fold((0.0, 0.0, 0.0), |a, c| (a.0 + c.p_fast, a.1 + c.p_long, a.2 + c.p_short));
        s.combine(f.min(1.0), l.min(1.0), sh.min(1.0));
        s.octave_checks();
        s.third_harmonic_checks();
        let b = s.best().unwrap();
        let c = s.as_slice()[b];
        44_100.0 / c.period
    }

    #[test]
    fn rough_phonation_keeps_the_heard_pitch() {
        // DSP-5, vocadito track 34 at 8.80 s (annotated 194 Hz): a period-
        // doubled voice whose 2T (97 Hz) is a little more periodic than T and
        // holds most of the pYIN mass. Without the rough-phonation rule the
        // E/H octave-up rule handed T's mass to 2T (no harmonic score below
        // ~125 Hz to veto it), and nothing moved it back.
        let mut s = CandSet::default();
        measured(&mut s, 361.0, 0.0, 0.0, 0.0, 0.58, 0.64, 0.64, 5.73);
        measured(&mut s, 192.0, 0.11, 0.24, 0.10, 0.14, 0.24, 0.25, 26.47);
        measured(&mut s, 128.0, 0.0, 0.0, 0.0, 0.55, UNKNOWN_HEIGHT, 0.66, 0.12);
        measured(&mut s, 97.0, 0.76, 0.38, 0.0, 0.07, UNKNOWN_HEIGHT, 0.05, f32::NAN);
        let f = decide(&mut s);
        assert!((f - 192.0).abs() < 2.0, "chose {f} Hz");
        // A true low pitch with a strong second harmonic is *decisively* more
        // periodic at 2T (g(2T) < 0.2·g(T)): it keeps its octave.
        let mut s = CandSet::default();
        measured(&mut s, 192.0, 0.11, 0.24, 0.10, 0.14, 0.24, 0.25, 26.47);
        measured(&mut s, 97.0, 0.76, 0.38, 0.0, 0.02, UNKNOWN_HEIGHT, 0.01, f32::NAN);
        let f = decide(&mut s);
        assert!((f - 97.0).abs() < 2.0, "chose {f} Hz");
    }

    #[test]
    fn eco_keeps_the_heard_pitch_of_a_rough_voice_without_a_harmonic_score() {
        // Task 2: the DSP-5 case (vocadito track 34) as Eco measures it, with
        // no harmonic score. The HQ/Max rules alone hand T's mass to 2T (the
        // E/H octave-up rule, nothing vetoes it): Eco read the track an octave
        // low on 24 % of its frames. The time-domain rough rule keeps T.
        let eco = |s: &mut CandSet, rough_td: bool| {
            let (f, l, sh) = s.as_slice().iter().fold((0.0, 0.0, 0.0), |a, c| (a.0 + c.p_fast, a.1 + c.p_long, a.2 + c.p_short));
            s.combine(f.min(1.0), l.min(1.0), sh.min(1.0));
            s.octave_checks_with(rough_td);
            s.third_harmonic_checks();
            44_100.0 / s.as_slice()[s.best().unwrap()].period
        };
        let rough = || {
            let mut s = CandSet::default();
            measured(&mut s, 361.0, 0.0, 0.0, 0.0, 0.58, 0.64, 0.64, f32::NAN);
            measured(&mut s, 192.0, 0.11, 0.24, 0.0, 0.14, 0.24, 0.25, f32::NAN);
            measured(&mut s, 128.0, 0.0, 0.0, 0.0, 0.55, UNKNOWN_HEIGHT, 0.66, f32::NAN);
            measured(&mut s, 97.0, 0.76, 0.38, 0.0, 0.07, UNKNOWN_HEIGHT, 0.05, f32::NAN);
            s
        };
        let f = eco(&mut rough(), false);
        assert!((f - 97.0).abs() < 2.0, "without the rule: {f} Hz");
        let f = eco(&mut rough(), true);
        assert!((f - 192.0).abs() < 2.0, "chose {f} Hz");
        // A true low pitch, decisively more periodic at 2T, keeps 2T in Eco.
        let mut s = CandSet::default();
        measured(&mut s, 192.0, 0.11, 0.24, 0.0, 0.14, 0.24, 0.25, f32::NAN);
        measured(&mut s, 97.0, 0.76, 0.38, 0.0, 0.02, UNKNOWN_HEIGHT, 0.01, f32::NAN);
        let f = eco(&mut s, true);
        assert!((f - 97.0).abs() < 2.0, "chose {f} Hz");
    }

    #[test]
    fn decisive_periodicity_overrides_the_harmonic_veto() {
        // Vocadito track 2 at 21.58 s (annotated 413 Hz, F1 near 2·f0): both
        // full-rate measures say 413 Hz is several times more periodic than
        // 828 Hz, but the harmonic score (fooled by the strong second
        // harmonic) vetoed the octave-up correction for 0.46 s.
        let mut s = CandSet::default();
        measured(&mut s, 828.0, 0.84, 0.77, 0.84, 0.03, 0.02, 0.03, 18.77);
        measured(&mut s, 414.0, 0.14, 0.20, 0.14, 0.004, 0.002, 0.01, 9.82);
        measured(&mut s, 275.0, 0.02, 0.01, 0.01, 0.04, 0.03, 0.02, 4.58);
        let f = decide(&mut s);
        assert!((f - 414.0).abs() < 2.0, "chose {f} Hz");
    }

    #[test]
    fn dominant_third_harmonic_hands_its_mass_to_the_fundamental() {
        // Vocadito track 21 at 11.22 s (annotated 158 Hz): the third harmonic
        // (F1 ≈ 3·f0) gives the shortest-lag E/H trough and most of the pYIN
        // mass; only octave relations were checked, so it was kept.
        let mut s = CandSet::default();
        measured(&mut s, 473.0, 0.66, 0.59, 0.67, 0.07, 0.05, 0.07, 30.90);
        measured(&mut s, 240.0, 0.13, 0.07, 0.08, 0.07, 0.07, 0.06, 18.37);
        measured(&mut s, 158.0, 0.18, 0.27, 0.25, 0.02, UNKNOWN_HEIGHT, 0.01, 23.60);
        measured(&mut s, 79.0, 0.02, 0.01, 0.0, 0.05, UNKNOWN_HEIGHT, 0.01, f32::NAN);
        let f = decide(&mut s);
        assert!((f - 158.0).abs() < 2.0, "chose {f} Hz");
    }

    #[test]
    fn octave_up_suspects_are_the_full_rate_pairs_that_would_move() {
        let mut s = set(&[(7700.0, 0.8, 0.03, 0.04), (6500.0, 0.15, 0.004, 0.004), (5000.0, 0.05, 0.01, 0.01)]);
        s.c[0].g_full = 0.3;
        s.c[1].g_full = 0.001;
        let mut out = [0usize; 4];
        let k = s.octave_up_suspects(&mut out);
        assert_eq!(&out[..k], &[0, 1]);
        // The rule would not change the winner (the upper one holds less
        // mass): no re-measurement.
        s.c[0].p_long = 0.1;
        assert_eq!(s.octave_up_suspects(&mut out), 0);
        s.c[0].p_long = 0.8;
        s.c[0].g_full = 0.001;
        assert_eq!(s.octave_up_suspects(&mut out), 0);
    }

    #[test]
    fn harmonic_score_vetoes_weak_single_measure() {
        // The time-domain measure suggests an octave-up error but the harmonic
        // score strongly says the higher candidate is right: no transfer.
        let mut s = set(&[(6200.0, 0.6, 0.15, 0.02), (5000.0, 0.3, 0.01, 0.02)]);
        s.c[0].harm = 1.0;
        s.c[1].harm = 0.5;
        s.octave_checks();
        assert_eq!(s.as_slice()[s.best().unwrap()].cents, 6200.0);
        s.scale(0.5);
        assert!(s.as_slice().iter().map(|c| c.p).sum::<f32>() < 0.5);
        s.clear();
        assert!(s.best().is_none());
    }
}
