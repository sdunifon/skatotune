//! Causal pitch-mark tracker (R§8.13, Live; also used by Studio in v1).
//!
//! Pitch marks `m_j` are fractional input positions one period apart, at the
//! same phase of every glottal cycle. PSOLA cuts its grains around them and
//! the control law reads the instantaneous input pitch from their spacing
//! (`f_in = fs / (m_{j+1} − m_j)`, R§7.6.2 A).
//!
//! **Chaining.** From the last mark `m_j` and the predicted period `T`, the
//! next mark is searched at lags τ ∈ [0.9·T, 1.1·T] (±20 % for two periods
//! after a seed or a weak period) by normalised cross-correlation of
//! `x[m_j − W_b, m_j + W_f)` with `x[m_j + τ − W_b, m_j + τ + W_f)`,
//! W_b = 0.6·T, W_f = 0.1·T. `T` is the last period plus half its last change
//! (capped at 4 %) when the last two changes agree in sign: a fast glide
//! moves the period by up to ≈ 9 % per period, and jitter (alternating sign)
//! gets no trend. Long periods are searched coarse-to-fine (every
//! `T/256`-th lag on a box-decimated copy first, then every full-rate lag
//! around the best), and the peak is refined to a fraction of a sample (NCC
//! at ±½ sample from a half-sample-shifted template, then a parabola), so
//! marks are fractional (R§8.16: rounding to whole samples would add ±9 c of
//! jitter at 500 Hz). The search runs the moment the newest input reaches
//! `m_j + τ_max + W_f`, so a mark becomes available `≈ 0.2·T` after it (the
//! `d_mark` term of the latency formula).
//!
//! **HQ/Max: competing peaks (R§7.2.2 multi-hypothesis marks).** The search
//! keeps the best and the runner-up NCC peak (at least 3 % of a period
//! apart). Eco takes the higher; HQ/Max take the one with the lower cost
//! `(1 − NCC) + λ·|semitones from T|`, λ = 0.1. On real singing 5 % of steps
//! have a runner-up within 0.05 of the best; picking by height alone there
//! makes the mark spacing jump by tens of cents and back (vocadito: spikes
//! of more than 30 c on 8.9 % of marks, 8.4 % with the cost). The choice is
//! causal and costs no latency (a deferred Viterbi decision would not fit
//! the Live mark-delay budget).
//!
//! **Anchor.** NCC chaining is relative, so its phase can drift slowly as the
//! waveform changes. Every second period the LP residual (order `2 + fs/1000`,
//! capped at 50; autocorrelation method on a 2·T Hann window, at most 2048
//! samples) is computed around the previous mark; if its peak (the glottal
//! closure) is salient and more than 0.15·T away, the next mark is pulled
//! towards it by at most 0.015·T per period, so marks never jump and the
//! deadband adds no jitter.
//!
//! **Seeding.** When the control says voiced and gives a period hint, a seed
//! mark is placed on the residual peak (or the largest sample) of the newest
//! period and verified by one backward NCC step. Marks on the closure keep
//! the formant envelope best; marks later in the cycle (its energy centre)
//! keep pulse peaks higher at 330-440 Hz but cost 0.4-0.7 dB of envelope
//! error on average and up to 6 dB at 440 Hz (measured on two synthetic voice
//! families), so the closure is kept. The chain is then extended *backwards*
//! as far as the output still needs (D plus 1.5 periods), so the PSOLA
//! engine has marks for content that is about to be output even though
//! voicing was confirmed later than it began. Marks therefore carry the time
//! they became known (`known_at`); the engines only use marks known at the
//! output sample being computed, which keeps the output causal and identical
//! for every host block size.
//!
//! **Losing lock.** A period whose best NCC is below 0.5 is extrapolated; two
//! in a row unlock the tracker (R§8.13 "unvoiced if NCC_max < 0.5").
//!
//! **Detector disagreement and the octave guard (R§8.13).** The detector's
//! period is compared with the marks *at the instant it describes*: the
//! newest input in Live, `newest − D` in Studio, whose centred detector lags
//! by D (comparing a centred hint with the newest marks would flag every
//! interval above 300 c as a mismatch for D after the leap and re-seed every
//! two hops). A disagreement of more than 300 c for two hops is then checked
//! against the waveform, cycle by cycle (`MarkTracker::periodicity`):
//! of the two periods the shorter is kept unless the longer correlates
//! better by 0.1 (the detector's own octave rule), on two consecutive
//! checks. A hint an octave low therefore never pulls the marks to 2·T,
//! where every grain would hold two pulses whose spacing PSOLA cannot shift
//! (doubling). Only a supported hint re-seeds, with the correction factor it
//! implies applied to the newest mark spacing (in Studio the hint is D old).
//! A seed takes T/2 instead of the hint T only when the waveform clearly
//! repeats at T/2, and a widened chaining step whose period grew by 40 % is
//! checked for having slipped to two periods (a fast upward glide can walk
//! the ±20 % range off its edge).
//!
//! **Phase continuity.** A re-seed while the previous run is still alive
//! starts on that run's newest chained mark, so the new marks keep the
//! glottal phase the engines are using. A fresh seed never replaces the
//! previous run's chained marks (Studio's voicing lags by D, so it re-seeds
//! on release tails while the PSOLA is still in the note) and, if that run
//! is recent, is moved onto the same waveform phase by a long-lag NCC. A
//! switch between runs then never moves the grains by a fraction of a period.

use skato_core::ring::Ring;
use skato_core::simd;
use skato_core::window::kaiser_at;

use crate::latency::{MAX_HALF_WIDTH, SEARCH_LOCKED, SEARCH_WIDE, WB_FRAC, WF_FRAC};
use crate::lpc::levinson;

/// One pitch mark.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Mark {
    /// Absolute fractional input position.
    pub pos: f64,
    /// Measured period at this mark: the NCC lag from the previous mark of the
    /// same run (for the first mark of a run, the lag to its neighbour).
    pub period: f64,
    /// `Ring::written()` count at which the mark became available.
    pub known_at: u64,
    /// Unique id, increasing with `pos`.
    pub id: u64,
    /// Tracker run (incremented by every seed); spacing across runs is
    /// meaningless.
    pub run: u32,
    /// Best NCC of the step that produced the mark (0 for an extrapolated
    /// mark).
    pub score: f32,
}

/// Capacity of the mark store. Enough for 50 ms of marks of a 1.4 kHz voice
/// plus the backward chain, at any sample rate.
const MARK_CAP: usize = 512;

/// Fixed-capacity, position-sorted store of marks (oldest first). Its
/// `known_at` values are non-decreasing, so "marks known at time t" is always
/// a prefix.
pub(crate) struct MarkStore {
    buf: Box<[Mark]>,
    head: usize,
    len: usize,
    next_id: u64,
    /// Bumped on every change (invalidates cached lookups).
    version: u64,
}

impl MarkStore {
    fn new() -> Self {
        let empty = Mark { pos: 0.0, period: 0.0, known_at: 0, id: 0, run: 0, score: 0.0 };
        Self { buf: vec![empty; MARK_CAP].into_boxed_slice(), head: 0, len: 0, next_id: 0, version: 0 }
    }

    /// Number of stored marks.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Mark at logical index `i` (0 = oldest).
    #[inline]
    pub fn get(&self, i: usize) -> Option<&Mark> {
        if i < self.len { self.buf.get((self.head + i) % MARK_CAP) } else { None }
    }

    /// Newest mark.
    #[inline]
    pub fn newest(&self) -> Option<&Mark> {
        self.len.checked_sub(1).and_then(|i| self.get(i))
    }

    /// Appends a mark (assigning its id); drops the oldest when full.
    fn push(&mut self, mut m: Mark) -> u64 {
        m.id = self.next_id;
        self.next_id += 1;
        self.version += 1;
        if self.len == MARK_CAP {
            self.head = (self.head + 1) % MARK_CAP;
            self.len -= 1;
        }
        if let Some(slot) = self.buf.get_mut((self.head + self.len) % MARK_CAP) {
            *slot = m;
        }
        self.len += 1;
        m.id
    }

    /// Drops the newest marks whose position is ≥ `pos`.
    fn truncate_from(&mut self, pos: f64) {
        while self.newest().is_some_and(|m| m.pos >= pos) {
            self.len -= 1;
            self.version += 1;
        }
    }

    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
        self.version += 1;
    }

    /// Smallest logical index whose mark does not satisfy `pred` (marks
    /// satisfying it must form a prefix).
    fn partition_point(&self, pred: impl Fn(&Mark) -> bool) -> usize {
        self.partition_point_in(self.len, pred)
    }

    /// `partition_point` over the first `limit` marks.
    fn partition_point_in(&self, limit: usize, pred: impl Fn(&Mark) -> bool) -> usize {
        let (mut lo, mut hi) = (0usize, limit.min(self.len));
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.get(mid).is_some_and(&pred) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Number of marks available once `written` input samples exist.
    #[inline]
    pub fn known_len(&self, written: u64) -> usize {
        self.partition_point(|m| m.known_at <= written)
    }

    /// First index below `limit` whose position is ≥ `x` (`limit` if none).
    #[inline]
    pub fn lower_bound(&self, x: f64, limit: usize) -> usize {
        self.partition_point_in(limit, |m| m.pos < x)
    }

    /// Logical index of the mark with id `id`.
    pub fn find_id(&self, id: u64) -> Option<usize> {
        let i = self.partition_point(|m| m.id < id);
        self.get(i).filter(|m| m.id == id).map(|_| i)
    }
}

/// Direction of an NCC search relative to the template mark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dir {
    Forward,
    Backward,
}

/// The next forward search, run once the newest input reaches `need`.
#[derive(Clone, Copy, Debug)]
struct Pending {
    /// Mark the search starts from.
    from: f64,
    /// Lag range in samples.
    lo: f64,
    hi: f64,
    /// Expected period (centre of the range; used to extrapolate on a miss).
    centre: f64,
    /// `Ring::written()` needed to run the search.
    need: u64,
    /// The range is the widened one (onset, miss or edge hit).
    wide: bool,
}

/// Seed / backward-chain acceptance threshold (stricter than chaining, so a
/// seed on breath or noise is unlikely).
const SEED_SCORE: f32 = 0.6;
/// Below this NCC a forward step is a miss (R§8.13).
const MISS_SCORE: f32 = 0.5;
/// Consecutive misses that unlock the tracker.
const MAX_MISSES: u32 = 2;
/// Largest backward chain (bounded work at a seed).
const MAX_BACK: usize = 160;
/// Forward searches run per `advance` call at most (a 256-sample chunk holds
/// at most a handful of periods; the cap only bounds pathological input).
const MAX_STEPS: usize = 64;
/// Detector disagreement (cents) that counts towards a re-seed.
const MISMATCH_CENTS: f64 = 300.0;
/// Octave guard: a longer candidate period must out-correlate a shorter one
/// by this much to be preferred (R§8.13 "confirm T against 2·T").
const OCTAVE_MARGIN: f32 = 0.1;
/// Seed octave guard: the half period must correlate within this of the
/// hint's period (and at least [`SEED_SCORE`]), or at least
/// [`SEED_OCTAVE_MIN`] when the hint's period cannot be measured yet.
const SEED_OCTAVE_MARGIN: f32 = 0.03;
const SEED_OCTAVE_MIN: f32 = 0.9;
/// HQ/Max chaining (R§7.2.2 multi-hypothesis marks): NCC a competing
/// peak must gain per semitone it lies further from the predicted period.
/// On real singing 5 % of chaining steps have a second NCC peak within 0.05
/// of the best and at least 3 % of a period away; picking by NCC alone
/// there jumps the period by ≥ 50 c and back.
const CONTINUITY_PER_ST: f64 = 0.1;
/// Largest period trend the next search is centred on, fraction of T per
/// period (vibrato moves the period by 2-4 % per period).
const TREND_MAX: f64 = 0.04;
/// A widened step whose period grew this much over the last locked step's
/// is checked for an octave slip (two periods instead of one).
const OCTAVE_SLIP: f64 = 1.4;
/// Lag tolerance of the octave guard's periodicity measure, fraction of the lag.
const OCTAVE_LAG_TOL: f64 = 0.03;
/// Largest correction factor a re-seed transfers onto the newest marks (cents).
const TRANSFER_MAX_CENTS: f64 = 1300.0;
/// Residual peak / RMS needed to trust it as a glottal closure.
const ANCHOR_SALIENCE: f32 = 3.0;
/// Anchor deadband and maximum pull per period, as fractions of T.
const ANCHOR_DEADBAND: f64 = 0.15;
const ANCHOR_STEP: f64 = 0.015;
/// Anchor search half-range around the mark (R§8.13: ±0.3·T).
const ANCHOR_RANGE: f64 = 0.3;

/// Causal NCC pitch-mark tracker with LP-residual anchoring.
pub(crate) struct MarkTracker {
    sr: f64,
    /// Plausible period range (voice range widened by 15 %).
    t_lo: f64,
    t_hi: f64,
    /// Reported latency; the backward chain covers D + 1.5 periods.
    d: f64,
    /// How far behind the newest input the detector's period hint is
    /// measured: D in Studio (centred detector), 0 in the Live modes.
    hint_delay: f64,
    pub store: MarkStore,
    locked: bool,
    run: u32,
    pending: Option<Pending>,
    misses: u32,
    /// Remaining forward steps that use the wide (±20 %) search.
    wide_steps: u32,
    /// Consecutive hops whose detector hint disagreed with the marks.
    mismatch_hops: u32,
    /// Consecutive octave-guard checks that supported the disagreeing hint.
    guard_votes: u32,
    /// Previous measured period of the chain and the change into it (0 after
    /// a miss or a seed).
    last_tau: f64,
    last_delta: f64,
    /// Period of the last locked (not widened) step, or the seed's: the
    /// reference the chain's octave check compares a widened step with.
    ref_tau: f64,
    /// Largest `known_at` handed out (keeps the prefix property).
    last_known: u64,
    /// LP order for the anchor.
    order: usize,
    region: Box<[f32]>,
    /// Box-decimated copy of `region` for the coarse lag pass.
    dec: Box<[f32]>,
    /// NCC per lag of the current search (peak candidates).
    scan: Box<[f32]>,
    /// Choose between competing NCC peaks by period continuity (HQ/Max).
    continuity: bool,
    /// Half-sample interpolation filter (16-tap Kaiser-windowed sinc, the
    /// same kernel as `skato_core`'s Sinc16 at phase ½).
    half: [f32; 16],
    /// Template region and its half-sample-shifted copy (refinement).
    tsrc: Box<[f32]>,
    tmpl: Box<[f32]>,
    lpc_win: Box<[f32]>,
    resid_src: Box<[f32]>,
    r: Box<[f64]>,
    a: Box<[f64]>,
    /// Prediction-error filter reversed, f32 (residual by dot products).
    arev: Box<[f32]>,
    back: Box<[(f64, f64, f32)]>,
    /// Forward steps since the last anchor check.
    anchor_tick: u32,
    /// `period_at` memo: (store version, valid from, valid until, result).
    memo: (u64, u64, u64, Option<f64>),
    pub seeds: u64,
    pub losses: u64,
    pub marks: u64,
    pub anchors: u64,
}

impl MarkTracker {
    /// Tracker for periods within `[t_min, t_max]` samples (widened by 15 %)
    /// at sample rate `sr`, feeding an output delayed by `d` samples, with
    /// period hints that describe the input `hint_delay` samples before the
    /// newest one.
    pub fn new(sr: f64, t_min: f64, t_max: f64, d: f64, hint_delay: f64) -> Self {
        let t_lo = (t_min / 1.15).max(4.0);
        let t_hi = (t_max * 1.15).max(t_lo * 2.0);
        // Largest region: the octave guard's search, a window of one long
        // period (≤ 1.2·t_hi, a seed's wide step) at lags up to 1.03× that,
        // plus the coarse step's rounding and neighbours. (A chaining search
        // needs W_b + W_f + τ_max(1+wide) ≈ 1.95·t_hi.)
        let long = (1.0 + SEARCH_WIDE) * t_hi;
        let region_len =
            ((2.0 + OCTAVE_LAG_TOL) * long + 0.05 * t_hi).ceil() as usize + 2 * coarse_step(long) as usize + 16;
        let order = ((2.0 + sr / 1000.0).round() as usize).clamp(8, 50);
        let lpc_len = ((2.0 * t_hi).ceil() as usize).min(2 * LPC_HALF_WIN_MAX) + 4;
        let resid_len = ((2.0 * ANCHOR_RANGE + 1.0) * t_hi).ceil() as usize + order + 8;
        Self {
            sr,
            t_lo,
            t_hi,
            d,
            hint_delay: if hint_delay.is_finite() { hint_delay.max(0.0) } else { 0.0 },
            store: MarkStore::new(),
            locked: false,
            run: 0,
            pending: None,
            misses: 0,
            wide_steps: 0,
            mismatch_hops: 0,
            guard_votes: 0,
            last_tau: 0.0,
            last_delta: 0.0,
            ref_tau: 0.0,
            last_known: 0,
            order,
            region: vec![0.0; region_len].into_boxed_slice(),
            dec: vec![0.0; region_len / 2 + 2].into_boxed_slice(),
            // Widest chaining range: ±20 % of the longest period (+ slack).
            scan: vec![0.0; (2.0 * SEARCH_WIDE * long).ceil() as usize + 16].into_boxed_slice(),
            continuity: false,
            half: half_sample_filter(),
            tsrc: vec![0.0; ((WB_FRAC + WF_FRAC) * t_hi).ceil() as usize + 2 * HALF_TAPS + 8].into_boxed_slice(),
            tmpl: vec![0.0; ((WB_FRAC + WF_FRAC) * t_hi).ceil() as usize + 8].into_boxed_slice(),
            lpc_win: vec![0.0; lpc_len].into_boxed_slice(),
            resid_src: vec![0.0; resid_len.max(lpc_len)].into_boxed_slice(),
            r: vec![0.0; order + 1].into_boxed_slice(),
            a: vec![0.0; order + 1].into_boxed_slice(),
            arev: vec![0.0; order + 1].into_boxed_slice(),
            back: vec![(0.0, 0.0, 0.0); MAX_BACK].into_boxed_slice(),
            anchor_tick: 0,
            memo: (u64::MAX, 0, 0, None),
            seeds: 0,
            losses: 0,
            marks: 0,
            anchors: 0,
        }
    }

    pub fn reset(&mut self) {
        self.store.clear();
        self.locked = false;
        self.pending = None;
        self.misses = 0;
        self.wide_steps = 0;
        self.mismatch_hops = 0;
        self.guard_votes = 0;
        self.last_tau = 0.0;
        self.last_delta = 0.0;
        self.ref_tau = 0.0;
        self.last_known = 0;
        self.anchor_tick = 0;
        self.memo = (u64::MAX, 0, 0, None);
    }

    /// Enables the HQ/Max choice between competing NCC peaks by period
    /// continuity (Eco: the highest peak).
    pub fn set_continuity(&mut self, on: bool) {
        self.continuity = on;
    }

    /// Whether the tracker is currently chaining marks.
    #[cfg(test)]
    pub fn locked(&self) -> bool {
        self.locked
    }

    /// Newest measured period while locked (after a missed period, the
    /// extrapolated mark's: the corrector does better continuing the last
    /// period through a miss than falling back to the detector, e.g. on
    /// breathy or reverberant onsets).
    pub fn period(&self) -> Option<f64> {
        if self.locked { self.store.newest().map(|m| m.period).filter(|&p| p > 0.0) } else { None }
    }

    /// Period of the newest mark known once `written` input samples exist,
    /// if that mark is recent (less than two longest periods old, i.e. the
    /// chain is still running at that sample). This is the per-sample view:
    /// unlike [`Self::period`] it does not depend on how far ahead of the
    /// output sample the searches of a chunk have already run, which keeps
    /// the output block-size independent.
    pub fn period_at(&mut self, written: u64) -> Option<f64> {
        let (ver, from, until, cached) = self.memo;
        if ver == self.store.version && written >= from && written < until {
            return cached;
        }
        // The answer holds from the newest known mark's `known_at` until the
        // next mark becomes known or the newest one ages out.
        let k = self.store.known_len(written);
        let newest = k.checked_sub(1).and_then(|i| self.store.get(i)).copied();
        let next_known = self.store.get(k).map_or(u64::MAX, |m| m.known_at);
        let (res, from, until) = match newest {
            Some(m) => {
                let age_out = (m.known_at as f64 + 2.0 * self.t_hi).ceil() as u64;
                if m.period > 0.0 && written < age_out {
                    (Some(m.period), m.known_at, next_known.min(age_out))
                } else {
                    (None, age_out.max(m.known_at), next_known)
                }
            }
            None => (None, 0, next_known),
        };
        self.memo = (self.store.version, from, until, res);
        res
    }

    /// Instantaneous input pitch in absolute cents from the newest mark
    /// spacing, while locked.
    pub fn mark_cents(&self) -> Option<f32> {
        let p = self.period()?;
        let c = skato_core::hz_to_cents((self.sr / p) as f32);
        c.is_finite().then_some(c)
    }

    /// Control-hop update (called at every hop boundary, when the analysis
    /// ring holds exactly the input up to that boundary).
    pub fn on_control(&mut self, ring: &Ring, voiced: bool, hint: f64) {
        if !voiced {
            self.unlock(false);
            self.mismatch_hops = 0;
            self.guard_votes = 0;
            return;
        }
        let hint_ok = hint.is_finite() && hint >= self.t_lo && hint <= self.t_hi;
        let mut seed_period = hint;
        let mut confirmed = false;
        if self.locked {
            let n = ring.written() as f64;
            // The marks' period at the instant the hint describes.
            // (None until the marks cover that instant, e.g. right after a
            // Studio onset whose backward chain stopped at the note start.)
            let reach = 1.5 * hint.max(self.store.newest().map_or(0.0, |m| m.period));
            let (Some(at), Some(newest)) = (self.mark_near(n - self.hint_delay, reach), self.store.newest().copied())
            else {
                self.mismatch_hops = 0;
                return;
            };
            if !(hint_ok && at.period > 0.0 && (1200.0 * (at.period / hint).log2()).abs() > MISMATCH_CENTS) {
                self.mismatch_hops = 0;
                self.guard_votes = 0;
                return;
            }
            self.mismatch_hops += 1;
            if self.mismatch_hops < 2 {
                return;
            }
            self.mismatch_hops = 0;
            // Octave guard: keep the marks unless the waveform supports the
            // hint's period at least as well (shorter wins ties), on two
            // consecutive checks (cycle-to-cycle irregularity can swing one).
            if !self.hint_supported(ring, at.pos, at.period, hint) {
                self.guard_votes = 0;
                return;
            }
            self.guard_votes += 1;
            if self.guard_votes < 2 {
                return;
            }
            self.guard_votes = 0;
            // Re-seed with the correction the hint implies, applied to the
            // newest spacing (they differ when the pitch moved since `at`).
            let max = (TRANSFER_MAX_CENTS / 1200.0).exp2();
            seed_period = (newest.period * (hint / at.period).clamp(1.0 / max, max)).clamp(self.t_lo, self.t_hi);
            confirmed = true;
            self.unlock(false);
        }
        self.mismatch_hops = 0;
        if hint_ok {
            // A re-seed the guard just confirmed keeps its period (running the
            // guard again could pick the other octave and ping-pong); a fresh
            // seed runs the guard itself.
            self.seed(ring, seed_period, !confirmed);
        }
    }

    /// The stored mark nearest to position `x`, if one lies within `reach`.
    fn mark_near(&self, x: f64, reach: f64) -> Option<Mark> {
        let j = self.store.lower_bound(x, self.store.len);
        let best = [j.checked_sub(1), Some(j)]
            .into_iter()
            .flatten()
            .filter_map(|i| self.store.get(i).copied())
            .min_by(|a, b| (a.pos - x).abs().total_cmp(&(b.pos - x).abs()))?;
        ((best.pos - x).abs() <= reach).then_some(best)
    }

    /// Cycle-wise periodicity at `lag`: the normalised correlation of the
    /// input with itself `lag` samples earlier over windows of one `short`
    /// cycle, `[a − 0.15·T, a + 0.85·T]` for marks `a` at and before `m` (one
    /// `short` period apart, back over two `long` periods: 2-4 windows),
    /// averaged, each the best over ±3 % of the lag. Marks sit at the same
    /// phase of every cycle (near the closure when seeded there), so each
    /// window holds one pulse and its ringing and the per-window
    /// normalisation ignores cycle-to-cycle amplitude changes: shimmer, or a
    /// diplophonic voice whose pulses alternate in strength, which a window
    /// straddling two pulses would mistake for a doubled period (the period
    /// at which PSOLA doubles). Spanning the long period still exposes a
    /// harmonic lock (windows of a long cycle that differ). Windows that
    /// cannot be measured (silent, e.g. before the note) are left out; `None`
    /// if none can. Uses no input at or after `limit`.
    fn periodicity(&mut self, ring: &Ring, m: f64, short: f64, long: f64, lag: f64, limit: u64) -> Option<f32> {
        let t_win = short / (WB_FRAC + WF_FRAC);
        // `search` puts its window at [c − W_b, c + W_f]; centre it so the
        // window starts 0.15·T before the mark.
        let off = 0.85 * short - wf(t_win) as f64;
        // Latest whole-period step back from `m` whose window (and the
        // refinement kernel) is inside the input.
        let room = limit as f64 - (0.85 * short + HALF_TAPS as f64 + 1.0);
        let back = ((m - room) / short).ceil().max(0.0);
        let a = m - back * short;
        let (lo, hi) = (lag * (1.0 - OCTAVE_LAG_TOL), lag * (1.0 + OCTAVE_LAG_TOL));
        let segs = (2.0 * long / short).round().clamp(2.0, 4.0) as usize;
        let (wb, w) = (((WB_FRAC * t_win).round() as i64).max(2), short.round() as i64);
        // Windows that cannot be measured are left out: a silent window, or
        // one whose lagged copy is mostly silent (before the note).
        let (mut sum, mut count) = (0.0f32, 0u32);
        for k in 0..segs {
            let c = a - k as f64 * short + off;
            let start = c.round() as i64 - wb;
            let (e_now, e_lag) =
                (self.energy(ring, start, w, limit), self.energy(ring, start - lag.round() as i64, w, limit));
            if !(e_now > 0.0 && e_lag >= 0.05 * e_now) {
                continue;
            }
            if let Some((_, v, _)) = self.search(ring, c, t_win, lo, hi, Dir::Backward, limit, None) {
                sum += v;
                count += 1;
            }
        }
        (count > 0).then(|| sum / count as f32)
    }

    /// Energy of the input over `[start, start + len)`, 0 if any of it is
    /// unavailable or at or after `limit`.
    fn energy(&mut self, ring: &Ring, start: i64, len: i64, limit: u64) -> f64 {
        let end = start + len;
        if start < 0 || len <= 0 || !ring.contains(start as u64) || end as u64 > limit.min(ring.written()) {
            return 0.0;
        }
        match self.region.get_mut(..len as usize) {
            Some(buf) => {
                ring.copy_to(start as u64, buf);
                simd::sum_sq(buf) as f64
            }
            None => 0.0,
        }
    }

    /// R§8.13 octave guard: whether the waveform near `m` supports the
    /// `short` period over the `long` one (the detector's rule: the shorter
    /// unless the longer correlates better by [`OCTAVE_MARGIN`], cycle-wise).
    /// When the longer lag cannot be measured (it reaches back before the
    /// note) a well-correlated short period wins; `None` if neither can be
    /// measured. Uses no input at or after `limit`.
    fn prefers_short(&mut self, ring: &Ring, m: f64, short: f64, long: f64, limit: u64) -> Option<bool> {
        let a = self.periodicity(ring, m, short, long, short, limit);
        let b = self.periodicity(ring, m, short, long, long, limit);
        match (a, b) {
            (Some(a), Some(b)) => Some(a >= b - OCTAVE_MARGIN),
            (Some(a), None) => Some(a >= MISS_SCORE),
            (None, Some(_)) => Some(false),
            (None, None) => None,
        }
    }

    /// Whether the octave guard supports the detector's `hint` over the
    /// marks' period near `m`. If the input cannot be measured the marks are
    /// kept: if they are wrong the chain soon misses, and the fresh seed that
    /// follows runs the guard.
    fn hint_supported(&mut self, ring: &Ring, m: f64, marks: f64, hint: f64) -> bool {
        let (short, long) = (marks.min(hint), marks.max(hint));
        self.prefers_short(ring, m, short, long, ring.written()).is_some_and(|short_wins| short_wins == (hint < marks))
    }

    /// Seed period: `t/2` instead of the hint `t` only when the waveform
    /// near `m` repeats at `t/2` (nearly) as well as at `t` and well in
    /// absolute terms, so a hint an octave low does not seed two-pulse
    /// grains. The guard is stricter than when locked: here it overrides a
    /// hint that may be right, and voices with weak odd harmonics correlate
    /// well at half their period (on real singing the margin-only rule
    /// seeded such voices at T/2 and lost lock, and did not lower the
    /// octave-error rate). It never doubles the hint.
    fn guard_seed_period(&mut self, ring: &Ring, m: f64, t: f64) -> f64 {
        let (half, n) = (0.5 * t, ring.written());
        if half < self.t_lo {
            return t;
        }
        // The half period must correlate at least SEED_SCORE either way, so
        // the (costly) periodicity at `t` is only measured when it can
        // matter (PERF-5: 2-4 NCC searches saved on most seeds).
        let a = match self.periodicity(ring, m, half, t, half, n) {
            Some(a) if a >= SEED_SCORE => a,
            _ => return t,
        };
        match self.periodicity(ring, m, half, t, t, n) {
            Some(b) if a >= b - SEED_OCTAVE_MARGIN => half,
            // The hint's lag reaches back before the note: a clearly periodic
            // half period wins.
            None if a >= SEED_OCTAVE_MIN => half,
            _ => t,
        }
    }

    fn unlock(&mut self, lost: bool) {
        if self.locked && lost {
            self.losses += 1;
        }
        self.locked = false;
        self.pending = None;
        self.misses = 0;
    }

    /// Runs every forward search whose input is complete.
    pub fn advance(&mut self, ring: &Ring) {
        for _ in 0..MAX_STEPS {
            let Some(p) = self.pending else { return };
            if p.need > ring.written() {
                return;
            }
            self.step(ring, p);
        }
    }

    /// One forward chaining step.
    fn step(&mut self, ring: &Ring, p: Pending) {
        let prior = self.continuity.then_some(p.centre);
        let found = self.search(ring, p.from, p.centre, p.lo, p.hi, Dir::Forward, p.need, prior);
        let (tau, score) = match found {
            Some((tau, score, edge)) if score >= MISS_SCORE => {
                self.misses = 0;
                self.wide_steps = if edge { self.wide_steps.max(1) } else { self.wide_steps.saturating_sub(1) };
                if p.wide && tau >= OCTAVE_SLIP * self.ref_tau {
                    self.confirm_period(ring, &p, tau, score)
                } else {
                    (tau, score)
                }
            }
            _ => {
                self.misses += 1;
                if self.misses >= MAX_MISSES {
                    self.unlock(true);
                    return;
                }
                self.wide_steps = 2;
                (p.centre, 0.0)
            }
        };
        let mut pos = p.from + tau;
        // The anchor only corrects slow drift (deadband 0.15·T), so checking
        // every second period is enough and halves its cost.
        self.anchor_tick = self.anchor_tick.wrapping_add(1);
        if score > 0.0 && self.anchor_tick.is_multiple_of(2) {
            pos += self.anchor_pull(ring, p.from, tau);
        }
        self.publish(pos, tau, p.need, score);
        // Centre the next search on the period trend (half the last change,
        // capped) when the last two changes agree in sign: a fast glide moves
        // the period by up to ≈ 9 % per period, close to the ±10 % range, and
        // a miss there can send the chain off the wrong edge; cycle-to-cycle
        // jitter alternates in sign and gets no trend.
        let delta = if score > 0.0 && self.last_tau > 0.0 { tau - self.last_tau } else { 0.0 };
        let next = if delta * self.last_delta > 0.0 {
            tau + (0.5 * delta).clamp(-TREND_MAX * tau, TREND_MAX * tau)
        } else {
            tau
        };
        self.last_delta = delta;
        self.last_tau = if score > 0.0 { tau } else { 0.0 };
        if score > 0.0 && !p.wide {
            self.ref_tau = tau;
        }
        self.schedule(pos, next);
    }

    /// R§8.13 octave guard on the chain. A widened step is one the chain was
    /// unsure of (onset, miss, edge hit); on a fast glide it can walk off the
    /// range edge and settle on two periods, where the NCC is as high as on
    /// one. When such a step's period has grown by [`OCTAVE_SLIP`] over the
    /// last locked one, and the half period, searched from the same mark,
    /// correlates nearly as well (the shorter-period rule), take it.
    /// Uses no input past the step's trigger, so it stays block-size
    /// independent.
    fn confirm_period(&mut self, ring: &Ring, p: &Pending, tau: f64, score: f32) -> (f64, f32) {
        let half = 0.5 * tau;
        if half < self.t_lo {
            return (tau, score);
        }
        // Cycle-wise periodicity at τ and τ/2 around the new mark (input up
        // to the trigger only), then the half period re-searched from the
        // same mark as an ordinary locked step.
        if self.prefers_short(ring, p.from + tau, half, tau, p.need) != Some(true) {
            return (tau, score);
        }
        let (lo, hi) = (half * (1.0 - SEARCH_LOCKED), half * (1.0 + SEARCH_LOCKED));
        match self.search(ring, p.from, half, lo, hi, Dir::Forward, p.need, None) {
            Some((t2, s2, _)) if s2 >= MISS_SCORE => (t2, s2),
            _ => (tau, score),
        }
    }

    /// Correction (≤ 0.015·T) pulling the chain towards the residual peak
    /// near the previous mark `m`, or 0.
    fn anchor_pull(&mut self, ring: &Ring, m: f64, t: f64) -> f64 {
        if t < 32.0 {
            // Very short periods: the residual cannot separate closures from
            // harmonics; the NCC chain alone is accurate there.
            return 0.0;
        }
        let half = (ANCHOR_RANGE * t).round() as i64;
        let c = m.round() as i64;
        // 2·T Hann window around the mark, capped: a few hundred samples per
        // predictor coefficient are plenty, and long periods at high sample
        // rates would otherwise dominate the per-period cost.
        let win = (t.round() as i64).min(LPC_HALF_WIN_MAX as i64);
        let Some((peak, sal)) = self.residual_peak(ring, c - win, 2 * win, c - half, c + half, t) else {
            return 0.0;
        };
        if sal < ANCHOR_SALIENCE {
            return 0.0;
        }
        let d = peak - m;
        if d.abs() <= ANCHOR_DEADBAND * t {
            return 0.0;
        }
        self.anchors += 1;
        d.clamp(-ANCHOR_STEP * t, ANCHOR_STEP * t)
    }

    fn publish(&mut self, pos: f64, period: f64, need: u64, score: f32) {
        let known_at = need.max(self.last_known);
        self.last_known = known_at;
        self.store.push(Mark { pos, period, known_at, id: 0, run: self.run, score });
        self.marks += 1;
    }

    /// Plans the forward search from mark `from` with expected period `t`.
    fn schedule(&mut self, from: f64, t: f64) {
        let t = t.clamp(self.t_lo, self.t_hi);
        let frac = if self.wide_steps > 0 { SEARCH_WIDE } else { SEARCH_LOCKED };
        let lo = (t * (1.0 - frac)).max(self.t_lo * 0.9);
        let hi = (t * (1.0 + frac)).min(self.t_hi * 1.1);
        let wf = wf(t);
        // Region end (exclusive): template anchor + last lag + neighbour + W_f.
        let need = from.round() as i64 + hi.floor() as i64 + 1 + wf;
        let wide = self.wide_steps > 0;
        self.pending =
            if need > 0 && hi > lo { Some(Pending { from, lo, hi, centre: t, need: need as u64, wide }) } else { None };
    }

    /// Seeds a new run around the newest period using the detector `hint`;
    /// `guard` lets the octave guard choose among `hint/2`, `hint`, `2·hint`.
    fn seed(&mut self, ring: &Ring, hint: f64, guard: bool) {
        let n = ring.written() as i64;
        let wf0 = wf(hint);
        let tw = hint.ceil() as i64;
        // Enough history for the octave guard (a window of 2·T at a lag of
        // 2·T, plus W_f) and one wide backward step.
        let need_hist = ((2.0 + 2.0 * (1.0 + OCTAVE_LAG_TOL) + SEARCH_WIDE) * hint).ceil() as i64 + 2 * tw + wf0 + 16;
        if n < need_hist || ((n - need_hist) as u64) < ring.oldest() {
            return;
        }
        // The newest full period whose NCC window (plus the refinement
        // kernel) is already available.
        let s_hi = n - wf0 - 2 - MAX_HALF_WIDTH as i64;
        let s_lo = s_hi - tw;
        // Phase continuity: start on the previous run's newest good mark if
        // it is still recent (a re-seed during a note), else on a closure.
        let reach = (n - wf(2.0 * hint) - 2) as f64;
        let cont = (0..self.store.len)
            .rev()
            .filter_map(|i| self.store.get(i))
            .take_while(|m| m.pos >= s_lo as f64 - 1.25 * hint)
            .find(|m| m.score >= MISS_SCORE && m.pos <= reach)
            .map(|m| m.pos);
        // A fresh run never replaces the previous run's chained marks (only
        // its extrapolated ones): the engines may still be reading them.
        // Studio's voicing describes `newest − D`, so its tracker keeps
        // re-seeding on a note's release tail while the PSOLA is still in
        // the note.
        let keep = match cont {
            Some(_) => None,
            None => (0..self.store.len).rev().filter_map(|i| self.store.get(i)).find(|m| m.score >= MISS_SCORE),
        }
        .map(|m| m.pos);
        let m0 = match cont {
            Some(p) => p,
            None => match self.seed_position(ring, n, s_lo, s_hi, hint) {
                // A run the engines may still be reading: put the new run on
                // the same waveform phase (whole "periods" of waveform
                // similarity from its last chained mark), so the switch
                // between runs does not move the grains.
                Some(p) => match keep.filter(|&k| k >= n as f64 - self.d - 3.0 * hint) {
                    Some(k) => self.align(ring, k, p, hint).unwrap_or(p),
                    None => p,
                },
                None => return,
            },
        };
        let t0 = if guard { self.guard_seed_period(ring, m0, hint) } else { hint };
        // Verify periodicity with one wide backward step.
        let Some((tau0, score0, edge0)) = self.search(
            ring,
            m0,
            t0,
            t0 * (1.0 - SEARCH_WIDE),
            t0 * (1.0 + SEARCH_WIDE),
            Dir::Backward,
            n as u64,
            None,
        ) else {
            return;
        };
        if score0 < SEED_SCORE {
            return;
        }
        // Backward chain over the span the output still needs.
        let stop = n as f64 - self.d - 1.5 * tau0;
        let mut count = 0usize;
        let (mut cur, mut tau, mut score) = (m0, tau0, score0);
        loop {
            let prev = cur - tau;
            if keep.is_some_and(|k| prev <= k + 0.5 * tau) {
                break;
            }
            if let Some(slot) = self.back.get_mut(count) {
                *slot = (prev, tau, score);
                count += 1;
            } else {
                break;
            }
            if prev < stop || prev - self.t_hi * 2.0 < ring.oldest() as f64 {
                break;
            }
            let t = tau.clamp(self.t_lo, self.t_hi);
            match self.search(
                ring,
                prev,
                t,
                t * (1.0 - SEARCH_LOCKED),
                t * (1.0 + SEARCH_LOCKED),
                Dir::Backward,
                n as u64,
                None,
            ) {
                Some((nt, ns, _)) if ns >= SEED_SCORE => {
                    cur = prev;
                    tau = nt;
                    score = ns;
                }
                _ => break,
            }
        }
        // Publish as a new run: oldest first, replacing any newer marks of
        // an earlier run.
        self.run = self.run.wrapping_add(1);
        let oldest = if count > 0 { self.back.get(count - 1).map_or(m0, |b| b.0) } else { m0 };
        let cut = oldest - 0.5 * tau0;
        self.store.truncate_from(keep.map_or(cut, |k| cut.max(k + 0.25 * tau0)));
        let known = (n as u64).max(self.last_known);
        self.last_known = known;
        for i in (0..count).rev() {
            let Some(&(pos, _, sc)) = self.back.get(i) else { continue };
            // A mark's period is its spacing to the previous (older) mark;
            // the oldest one takes the spacing to its newer neighbour.
            let period = self
                .back
                .get(i + 1)
                .filter(|_| i + 1 < count)
                .map_or_else(|| self.back.get(i).map_or(tau0, |b| b.1), |older| pos - older.0);
            self.store.push(Mark { pos, period, known_at: known, id: 0, run: self.run, score: sc });
        }
        self.store.push(Mark { pos: m0, period: tau0, known_at: known, id: 0, run: self.run, score: score0 });
        self.marks += count as u64 + 1;
        self.seeds += 1;
        self.locked = true;
        self.misses = 0;
        // R§8.13 widens the search at onsets because the detector's period is
        // only a hint. The seed's ±20 % backward step is that onset search:
        // an interior NCC peak there is a measured period, so the forward
        // chain starts locked. Wide forward steps would publish the first
        // marks 0.1·T later than the latency budget's d_mark assumes, which
        // holds back the PSOLA onset by about two periods. A peak on the
        // range edge (period not bracketed) keeps two wide steps, and any
        // later step that ends on its edge widens the next one.
        self.wide_steps = if edge0 { 2 } else { 0 };
        self.last_tau = 0.0;
        self.last_delta = 0.0;
        self.ref_tau = tau0;
        self.schedule(m0, tau0);
    }

    /// The position within ±½·`t` of `near` whose surrounding waveform best
    /// matches the one around mark `from` (NCC over the chaining window
    /// W_b + W_f of period `t`; long periods coarse-to-fine like
    /// [`Self::search`]), if it matches as well as a chained step would
    /// (≥ [`MISS_SCORE`]). Unlike `search`, the lag may span many periods:
    /// the template and the candidates are copied separately.
    fn align(&mut self, ring: &Ring, from: f64, near: f64, t: f64) -> Option<f64> {
        if !(from.is_finite() && near.is_finite() && t.is_finite() && t >= 2.0) {
            return None;
        }
        let wb = ((WB_FRAC * t).round() as i64).max(2);
        let wf = wf(t);
        let w = (wb + wf) as usize;
        let written = ring.written() as i64;
        let i = from.round() as i64;
        let c = near.round() as i64;
        let half = (0.5 * t).round() as i64;
        let (q_lo, q_hi) = (c - half, (c + half).min(written - wf - 1));
        let (t_start, c_start) = (i - wb, q_lo - wb);
        let c_len = (q_hi - q_lo + 1).max(0) as usize + w;
        if q_hi <= q_lo + 2
            || t_start < 0
            || c_start < 0
            || i + wf > written
            || !ring.contains(t_start as u64)
            || !ring.contains(c_start as u64)
            || c_len > self.region.len()
            || w > self.tmpl.len()
        {
            return None;
        }
        let tmpl = self.tmpl.get_mut(..w)?;
        ring.copy_to(t_start as u64, tmpl);
        let region = self.region.get_mut(..c_len)?;
        ring.copy_to(c_start as u64, region);
        let (tmpl, region): (&[f32], &[f32]) = (tmpl, region);
        let et = simd::sum_sq(tmpl) as f64;
        if et <= 1e-12 * w as f64 {
            return None;
        }
        let ncc = |k: i64| -> f32 {
            let Some(cw) = (k >= 0).then(|| region.get(k as usize..k as usize + w)).flatten() else { return -1.0 };
            let ec = simd::sum_sq(cw) as f64;
            if ec <= 1e-12 * w as f64 { 0.0 } else { (simd::dot(tmpl, cw) as f64 / (et * ec).sqrt()) as f32 }
        };
        let last = q_hi - q_lo;
        // Coarse pass every T/64-th candidate (full-rate windows; an NCC
        // peak of a voice is several times wider), then every candidate
        // around the winner.
        let step = ((t / 64.0) as i64).max(1);
        let mut coarse = (0i64, f32::MIN);
        let mut k = 0;
        while k <= last {
            let v = ncc(k);
            if v > coarse.1 {
                coarse = (k, v);
            }
            k += step;
        }
        let mut best = coarse;
        for k in (coarse.0 - step).max(0)..=(coarse.0 + step).min(last) {
            let v = ncc(k);
            if v > best.1 {
                best = (k, v);
            }
        }
        let (kb, vb) = best;
        if vb < MISS_SCORE {
            return None;
        }
        let d = if kb > 0 && kb < last {
            skato_core::math::parabolic_peak(ncc(kb - 1), vb, ncc(kb + 1)).0 as f64
        } else {
            0.0
        };
        Some((q_lo + kb) as f64 + d)
    }

    /// Seed position for a fresh run: the strongest closure in `[s_lo, s_hi]`
    /// (one period of `t0`), by the LP residual when it is salient, else the
    /// largest |x|.
    fn seed_position(&mut self, ring: &Ring, n: i64, s_lo: i64, s_hi: i64, t0: f64) -> Option<f64> {
        let tw = s_hi - s_lo;
        let lw = (2 * tw).min(2 * LPC_HALF_WIN_MAX as i64);
        match self.residual_peak(ring, n - lw, lw, s_lo, s_hi, t0) {
            Some((p, sal)) if sal >= ANCHOR_SALIENCE => Some(p),
            _ => self.abs_peak(ring, s_lo, s_hi),
        }
    }

    /// Position of the largest |x| in `[lo, hi]` (whole samples).
    fn abs_peak(&mut self, ring: &Ring, lo: i64, hi: i64) -> Option<f64> {
        if lo < 0 || hi < lo {
            return None;
        }
        let len = (hi - lo + 1) as usize;
        let buf = self.region.get_mut(..len)?;
        ring.copy_to(lo as u64, buf);
        let (i, v) = buf.iter().enumerate().fold(
            (0usize, 0.0f32),
            |acc, (i, &x)| {
                if x.abs() > acc.1 { (i, x.abs()) } else { acc }
            },
        );
        (v > 1e-9).then_some((lo + i as i64) as f64)
    }

    /// LP-residual peak in `[s_lo, s_hi]`, with the predictor fitted on the
    /// Hann-windowed `[w_start, w_start + w_len)`. Returns the peak position
    /// and its salience (peak / RMS of the residual over the search range).
    fn residual_peak(
        &mut self,
        ring: &Ring,
        w_start: i64,
        w_len: i64,
        s_lo: i64,
        s_hi: i64,
        t: f64,
    ) -> Option<(f64, f32)> {
        let p = self.order.min((t / 2.0) as usize).max(4);
        if w_start < 0 || w_len < 4 * p as i64 || s_hi <= s_lo || s_lo - (p as i64) < 0 {
            return None;
        }
        // Predictor from the windowed autocorrelation.
        let wl = w_len as usize;
        let win = self.lpc_win.get_mut(..wl)?;
        ring.copy_to(w_start as u64, win);
        // Hann sin²(π (i + ½) / len), with sin generated by a rotation
        // (exact to ~1e-13 over the window; no libm call per sample).
        let th = core::f64::consts::PI / wl as f64;
        let (c1, s1) = (th.cos(), th.sin());
        let (mut sn, mut cs) = ((0.5 * th).sin(), (0.5 * th).cos());
        for v in win.iter_mut() {
            *v *= (sn * sn) as f32;
            (sn, cs) = (sn * c1 + cs * s1, cs * c1 - sn * s1);
        }
        let r = self.r.get_mut(..=p)?;
        for (k, rk) in r.iter_mut().enumerate() {
            let (a, b) = (win.get(..wl - k)?, win.get(k..)?);
            *rk = simd::dot(a, b) as f64;
        }
        // Slight white-noise correction keeps high orders well conditioned.
        r[0] *= 1.0 + 1e-5;
        let a = self.a.get_mut(..=p)?;
        levinson(r, a)?;
        // Residual over the search range (unwindowed input).
        let src_len = (s_hi - s_lo + 1) as usize + p;
        let src = self.resid_src.get_mut(..src_len)?;
        ring.copy_to((s_lo - p as i64) as u64, src);
        // e[n] = Σ_k a[k]·x[n−k] = dot(reversed a, x[n−p ..= n]).
        let arev = self.arev.get_mut(..=p)?;
        for (dst, &ak) in arev.iter_mut().rev().zip(a.iter()) {
            *dst = ak as f32;
        }
        let mut best = (0usize, 0.0f32);
        let mut energy = 0.0f64;
        for n in 0..=(s_hi - s_lo) as usize {
            let e = simd::dot(arev, src.get(n..n + p + 1).unwrap_or(&[]));
            energy += (e as f64) * (e as f64);
            if e.abs() > best.1 {
                best = (n, e.abs());
            }
        }
        let count = (s_hi - s_lo + 1) as f64;
        let rms = (energy / count).sqrt() as f32;
        if rms <= 1e-12 || !rms.is_finite() {
            return None;
        }
        Some(((s_lo + best.0 as i64) as f64, best.1 / rms))
    }

    /// Best NCC lag in `[lo, hi]` from mark `m` (period `t` sets the window),
    /// searching later (`Forward`) or earlier (`Backward`) input, reading no
    /// input at or after index `limit`. Returns (fractional lag, peak NCC,
    /// peak-at-edge).
    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        ring: &Ring,
        m: f64,
        t: f64,
        lo: f64,
        hi: f64,
        dir: Dir,
        limit: u64,
        prior: Option<f64>,
    ) -> Option<(f64, f32, bool)> {
        if !(m.is_finite() && t.is_finite() && lo.is_finite() && hi.is_finite()) {
            return None;
        }
        let wb = ((WB_FRAC * t).round() as i64).max(2);
        let wf = wf(t);
        let w = (wb + wf) as usize;
        let k_lo = (lo.ceil() as i64).max(2);
        let k_hi = hi.floor() as i64;
        if k_hi < k_lo {
            return None;
        }
        let step = coarse_step(t);
        // Backward searches put the template at offset `e_hi`; round it up to
        // a multiple of the coarse step so the decimated template is aligned
        // (the region simply starts a little earlier).
        let e_hi = match dir {
            Dir::Forward => k_hi + 1,
            Dir::Backward => (k_hi + 1 + step - 1) / step * step,
        };
        let i = m.round() as i64;
        let len = e_hi as usize + w;
        let start = match dir {
            Dir::Forward => i - wb,
            Dir::Backward => i - e_hi - wb,
        };
        if start < 0 || len > self.region.len() {
            return None;
        }
        let end = start as u64 + len as u64;
        let limit = limit.min(ring.written());
        if !ring.contains(start as u64) || end > limit {
            return None;
        }
        let region = self.region.get_mut(..len)?;
        ring.copy_to(start as u64, region);
        let region: &[f32] = region;
        let t_off = match dir {
            Dir::Forward => 0usize,
            Dir::Backward => e_hi as usize,
        };
        let tmpl = region.get(t_off..t_off + w)?;
        let et = simd::sum_sq(tmpl) as f64;
        if et <= 1e-12 * w as f64 {
            return None;
        }
        let ncc = |k: i64| -> f32 {
            let off = match dir {
                Dir::Forward => k,
                Dir::Backward => e_hi - k,
            };
            if off < 0 {
                return -1.0;
            }
            let Some(c) = region.get(off as usize..off as usize + w) else { return -1.0 };
            let ec = simd::sum_sq(c) as f64;
            if ec <= 1e-12 * w as f64 {
                return 0.0;
            }
            (simd::dot(tmpl, c) as f64 / (et * ec).sqrt()) as f32
        };
        // Long periods: a coarse pass every `step` lags on a box-decimated
        // copy (cost ÷ step²), then every full-rate lag around its winner.
        // The best and the runner-up local maximum (at least 3 % of a period
        // apart) are kept; with a period `prior` (HQ/Max chaining, R§7.2.2),
        // the one with the lower cost (1 − NCC) + λ·|semitones from the
        // prior| wins, otherwise the best.
        let sep = ((0.03 * t).round() as i64).max(2);
        let scan = &mut self.scan;
        let mut top: [(i64, f32); 2] = [(k_lo, f32::MIN), (k_lo, f32::MIN)];
        if step > 1 {
            let su = step as usize;
            let nd = (len / su).min(self.dec.len());
            let dec = self.dec.get_mut(..nd)?;
            for (q, d) in dec.iter_mut().enumerate() {
                *d = region.get(q * su..q * su + su).map_or(0.0, |c| c.iter().sum());
            }
            let dec: &[f32] = dec;
            let wd = w / su;
            let td = t_off / su;
            let tmpl_d = dec.get(td..td + wd)?;
            let etd = simd::sum_sq(tmpl_d) as f64;
            let q0 = (k_lo + step - 1) / step;
            let nq = ((k_hi / step - q0 + 1).max(0) as usize).min(scan.len());
            // Decimated offsets of lags q0, q0 + 1, ...: ascending forward,
            // descending backward (scored in offset order, then reversed).
            let scores = scan.get_mut(..nq)?;
            match dir {
                Dir::Forward => ncc_run(tmpl_d, etd, dec, q0, 0.0, scores),
                Dir::Backward => {
                    ncc_run(tmpl_d, etd, dec, e_hi / step - (q0 + nq as i64 - 1), 0.0, scores);
                    scores.reverse();
                }
            }
            let coarse = local_maxima(scores, (sep / step).max(1));
            // Refine each coarse candidate at full rate around it.
            let floor = 1e-12 * w as f64;
            for (slot, &(j, v)) in top.iter_mut().zip(coarse.iter()) {
                if v == f32::MIN {
                    continue;
                }
                let c = (q0 + j as i64) * step;
                let (a, b) = ((c - step).max(k_lo), (c + step).min(k_hi));
                let Some(fine) = scan.get_mut(..(b - a + 1).max(0) as usize) else { continue };
                match dir {
                    Dir::Forward => ncc_run(tmpl, et, region, a, floor, fine),
                    Dir::Backward => {
                        ncc_run(tmpl, et, region, e_hi - b, floor, fine);
                        fine.reverse();
                    }
                }
                for (k, &v) in (a..=b).zip(fine.iter()) {
                    if v > slot.1 {
                        *slot = (k, v);
                    }
                }
            }
            if top[1].1 > top[0].1 {
                top.swap(0, 1);
            }
        } else {
            let n = (k_hi - k_lo + 1) as usize;
            let scores = scan.get_mut(..n)?;
            match dir {
                Dir::Forward => ncc_run(tmpl, et, region, k_lo, 1e-12 * w as f64, scores),
                Dir::Backward => {
                    ncc_run(tmpl, et, region, e_hi - k_hi, 1e-12 * w as f64, scores);
                    scores.reverse();
                }
            }
            let lm = local_maxima(scores, sep);
            top = [(k_lo + lm[0].0 as i64, lm[0].1), (k_lo + lm[1].0 as i64, lm[1].1)];
            if top[0].1 == f32::MIN {
                top[0] = (k_lo, ncc(k_lo));
            }
        }
        let best = match prior {
            Some(pr) if pr > 0.0 && top[1].1 > f32::MIN => {
                let cost =
                    |(k, v): (i64, f32)| (1.0 - v as f64) + CONTINUITY_PER_ST * 12.0 * (k as f64 / pr).log2().abs();
                if cost(top[1]) < cost(top[0]) { top[1] } else { top[0] }
            }
            _ => top[0],
        };
        let (kb, vb) = best;
        let edge = kb == k_lo || kb == k_hi;
        let (vm1, vp1) = (ncc(kb - 1), ncc(kb + 1));
        // Sub-sample refinement. A parabola through integer lags is biased by
        // up to ~0.2 sample on sharp (pulse-like) NCC peaks, which would be
        // period jitter. Halve the spacing instead: NCC at kb ± ½ comes from
        // the *template* shifted by half a sample (a fixed 16-tap sinc FIR;
        // one pass gives both shifts, since −½ is +½ one sample earlier)
        // against the integer candidate window. The template is older than
        // the candidate, so this needs no input beyond the search region.
        let c_off = match dir {
            Dir::Forward => kb,
            Dir::Backward => e_hi - kb,
        };
        let Some(cand) = region.get(c_off.max(0) as usize..c_off.max(0) as usize + w) else {
            return Some((kb as f64, vb.min(1.0), edge));
        };
        let t0 = i - wb;
        // Samples the refinement may touch: a forward search must not use
        // input past its trigger (`need`), nor any search past `limit`, or
        // its result would depend on the host block size.
        let limit = match dir {
            Dir::Forward => end,
            Dir::Backward => limit,
        };
        let src_start = t0 - HALF_TAPS as i64 / 2;
        let src_len = w + HALF_TAPS;
        if src_start < 0
            || !ring.contains(src_start as u64)
            || src_start as u64 + src_len as u64 > limit
            || src_len > self.tsrc.len()
            || w + 1 > self.tmpl.len()
        {
            let (d, peak) = skato_core::math::parabolic_peak(vm1, vb, vp1);
            let peak = if peak.is_finite() { peak.min(1.0) } else { vb.min(1.0) };
            return Some((kb as f64 + d as f64, peak, edge));
        }
        let tsrc = self.tsrc.get_mut(..src_len)?;
        ring.copy_to(src_start as u64, tsrc);
        // z[j] = x(t0 + j − ½) for j = 0..=w (taps t0 + j − 8 ..= t0 + j + 7).
        let z = self.tmpl.get_mut(..=w)?;
        for (j, v) in z.iter_mut().enumerate() {
            *v = simd::dot(&self.half, tsrc.get(j..j + HALF_TAPS).unwrap_or(&[]));
        }
        let z: &[f32] = z;
        let eb = simd::sum_sq(cand) as f64;
        let frac_ncc = |t: &[f32]| -> f32 {
            let ea = simd::sum_sq(t) as f64;
            if ea * eb <= 0.0 { 0.0 } else { (simd::dot(t, cand) as f64 / (ea * eb).sqrt()) as f32 }
        };
        // Template positions t0 + j − ½ (`early`) and t0 + j + ½ (`late`). A
        // lag of kb + δ pairs the candidate with the template read δ earlier
        // (forward) or later (backward).
        let (early, late) = (z.get(..w)?, z.get(1..=w)?);
        let (plus, minus) = match dir {
            Dir::Forward => (early, late),
            Dir::Backward => (late, early),
        };
        let (vmh, vph) = (frac_ncc(minus), frac_ncc(plus));
        // Centre the half-spacing parabola on the best of the three middle
        // points so both of its neighbours exist.
        let (c, y) = if vmh > vb && vmh >= vph {
            (-0.5, [vm1, vmh, vb])
        } else if vph > vb {
            (0.5, [vb, vph, vp1])
        } else {
            (0.0, [vmh, vb, vph])
        };
        let (d, peak) = skato_core::math::parabolic_peak(y[0], y[1], y[2]);
        let peak = if peak.is_finite() { peak.min(1.0) } else { vb.min(1.0) };
        Some((kb as f64 + c + 0.5 * d as f64, peak, edge))
    }
}

/// Normalised cross-correlation of `tmpl` (energy `et`) with the windows
/// `x[o..o + tmpl.len()]` for the consecutive offsets `o0, o0 + 1, …` into
/// `out`. The window energy is updated in O(1) per offset, `E(o + 1) = E(o)
/// − x[o]² + x[o + w]²`, instead of a fresh Σx² per lag, which was half the
/// cost of every mark search (PERF-5: seeds at 192 kHz run 10-20 searches of
/// 2-3 k-sample windows in one block). Windows reaching outside `x` score −1,
/// windows with energy ≤ `floor` score 0.
fn ncc_run(tmpl: &[f32], et: f64, x: &[f32], o0: i64, floor: f64, out: &mut [f32]) {
    let w = tmpl.len();
    let mut e: Option<f64> = None;
    for (j, slot) in out.iter_mut().enumerate() {
        let o = o0 + j as i64;
        let Some(c) = (o >= 0).then(|| x.get(o as usize..o as usize + w)).flatten() else {
            *slot = -1.0;
            e = None;
            continue;
        };
        let ec = match e {
            Some(prev) => {
                let (a, b) =
                    (x.get(o as usize - 1).copied().unwrap_or(0.0) as f64, c.last().copied().unwrap_or(0.0) as f64);
                (prev - a * a + b * b).max(0.0)
            }
            None => simd::sum_sq(c) as f64,
        };
        e = Some(ec);
        *slot = if ec <= floor || et <= 0.0 { 0.0 } else { (simd::dot(tmpl, c) as f64 / (et * ec).sqrt()) as f32 };
    }
}

/// The highest value of `v` and the highest local maximum at least `sep`
/// indices away from it, as (index, value) (`f32::MIN` when there is none).
/// The ends of the range count as local maxima (the true peak may lie
/// beyond).
fn local_maxima(v: &[f32], sep: i64) -> [(usize, f32); 2] {
    let mut best = (0usize, f32::MIN);
    for (i, &x) in v.iter().enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    let mut second = (0usize, f32::MIN);
    for (i, &x) in v.iter().enumerate() {
        let left = i.checked_sub(1).and_then(|j| v.get(j)).is_none_or(|&l| x >= l);
        let right = v.get(i + 1).is_none_or(|&r| x >= r);
        if left && right && (i as i64 - best.0 as i64).abs() >= sep && x > second.1 {
            second = (i, x);
        }
    }
    [best, second]
}

/// Coarse lag step for period `t`: about 1/256 of the period (whole
/// samples), so a period is always searched with ≥ 256 coarse lags' worth of
/// resolution before the full-rate pass.
fn coarse_step(t: f64) -> i64 {
    ((t / 256.0) as i64).max(1)
}

/// Largest half-window of the anchor's LP analysis, samples.
const LPC_HALF_WIN_MAX: usize = 1024;

/// Taps of the half-sample refinement filter.
const HALF_TAPS: usize = 16;

/// 16-tap Kaiser-windowed sinc (β = 9.25, as `skato_core`'s Sinc16) that
/// evaluates a signal half-way between its 8th and 9th taps, normalised to
/// unit DC gain.
fn half_sample_filter() -> [f32; HALF_TAPS] {
    let mut h = [0.0f64; HALF_TAPS];
    let half = (HALF_TAPS / 2) as f64;
    for (k, v) in h.iter_mut().enumerate() {
        let u = k as f64 - (half - 0.5);
        let x = core::f64::consts::PI * u;
        *v = x.sin() / x * kaiser_at(u / half, 9.25);
    }
    let sum: f64 = h.iter().sum();
    h.map(|v| (v / sum) as f32)
}

/// NCC window after the mark for period `t` (≥ 1 sample).
fn wf(t: f64) -> i64 {
    ((WF_FRAC * t).round() as i64).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Band-limited pulse train through one resonance: a crude vowel with a
    /// sharp closure at every multiple of `period`, starting at `phase`.
    fn pulse_vowel(n: usize, period: f64, phase: f64) -> Vec<f32> {
        let f0 = 1.0 / period;
        let kmax = (0.4 / f0) as usize;
        let mut src = vec![0.0f64; n];
        for (i, s) in src.iter_mut().enumerate() {
            let ph = 2.0 * core::f64::consts::PI * f0 * (i as f64 - phase);
            *s = (1..=kmax).map(|k| (k as f64 * ph).cos() / (k as f64).sqrt()).sum::<f64>();
        }
        // Resonance at 0.03·fs (≈ 1.4 kHz at 48 kHz), bandwidth 0.003·fs.
        let r = (-core::f64::consts::PI * 0.003f64).exp();
        let th = 2.0 * core::f64::consts::PI * 0.03;
        let (mut y1, mut y2) = (0.0, 0.0);
        src.iter()
            .map(|&x| {
                let y = x + 2.0 * r * th.cos() * y1 - r * r * y2;
                y2 = y1;
                y1 = y;
                (y * 0.01) as f32
            })
            .collect()
    }

    fn ring_of(x: &[f32]) -> Ring {
        let mut r = Ring::new(1 << 16);
        r.extend(x);
        r
    }

    #[test]
    fn store_prefix_and_ids() {
        let mut s = MarkStore::new();
        for i in 0..600u64 {
            s.push(Mark { pos: i as f64 * 10.0, period: 10.0, known_at: i / 2, id: 0, run: 0, score: 1.0 });
        }
        assert_eq!(s.len(), MARK_CAP);
        assert_eq!(s.get(0).unwrap().id, 600 - MARK_CAP as u64);
        assert_eq!(s.known_len(1000), MARK_CAP);
        assert_eq!(s.known_len(0), 0);
        let k = s.known_len(200);
        assert!(s.get(k - 1).unwrap().known_at <= 200 && s.get(k).unwrap().known_at > 200);
        assert_eq!(s.find_id(500), Some(500 - (600 - MARK_CAP)));
        assert_eq!(s.find_id(3), None);
        assert_eq!(s.find_id(10_000), None);
        s.truncate_from(5000.0);
        assert!(s.newest().unwrap().pos < 5000.0);
        s.clear();
        assert_eq!(s.len(), 0);
        assert!(s.newest().is_none());
    }

    #[test]
    fn locks_and_measures_period() {
        let sr = 48_000.0;
        for &period in &[48_000.0 / 110.0, 48_000.0 / 220.0, 48_000.0 / 440.0, 48_000.0 / 900.0] {
            let x = pulse_vowel(24_000, period, 37.3);
            let mut tr = MarkTracker::new(sr, 34.0, 873.0, 200.0, 0.0);
            let mut ring = Ring::new(1 << 16);
            for (h, chunk) in x.chunks(64).enumerate() {
                ring.extend(chunk);
                tr.advance(&ring);
                if h > 0 {
                    tr.on_control(&ring, true, period * 1.03); // hint 3 % off
                }
            }
            assert!(tr.locked(), "period {period}");
            assert_eq!(tr.seeds, 1);
            // Mark spacing equals the period, and marks sit on the closures
            // (positions ≡ phase mod period) without drifting.
            let n = tr.store.len();
            assert!(n > 20);
            let mut worst_spacing = 0.0f64;
            let mut worst_phase = 0.0f64;
            // Skip the synthetic signal's start-up (the resonance and the
            // backward chain see the zero history before sample 0).
            for i in 1..n {
                let (a, b) = (tr.store.get(i - 1).unwrap(), tr.store.get(i).unwrap());
                if a.pos < 2000.0 {
                    continue;
                }
                worst_spacing = worst_spacing.max((b.pos - a.pos - period).abs());
                let ph = (b.pos - 37.3).rem_euclid(period);
                worst_phase = worst_phase.max(ph.min(period - ph));
            }
            assert!(worst_spacing < 0.005, "period {period}: spacing error {worst_spacing}");
            assert!(worst_phase < 0.1 * period, "period {period}: phase {worst_phase}");
            let cents = tr.mark_cents().unwrap();
            let expect = skato_core::hz_to_cents((sr / period) as f32);
            assert!((cents - expect).abs() < 0.5, "{cents} vs {expect}");
        }
    }

    #[test]
    fn marks_are_known_after_the_mark_delay() {
        let period = 218.18;
        let x = pulse_vowel(20_000, period, 11.0);
        let mut tr = MarkTracker::new(48_000.0, 34.0, 600.0, 220.0, 0.0);
        let mut ring = Ring::new(1 << 16);
        for chunk in x.chunks(64) {
            ring.extend(chunk);
            tr.advance(&ring);
            tr.on_control(&ring, true, period);
        }
        let mut forward = 0;
        for i in 0..tr.store.len() {
            let m = tr.store.get(i).unwrap();
            let delay = m.known_at as f64 - m.pos;
            // Past the backward chain and the two wide steps after the seed.
            if m.score > 0.0 && i >= 10 {
                forward += 1;
                // Locked forward marks: d_mark ≤ 0.2·T + 2.
                assert!(delay <= crate::latency::mark_delay(period) + 1e-9, "delay {delay}");
            }
            if i > 0 {
                assert!(tr.store.get(i - 1).unwrap().known_at <= m.known_at);
                assert!(tr.store.get(i - 1).unwrap().pos < m.pos);
            }
        }
        assert!(forward > 50);
    }

    #[test]
    fn backward_chain_covers_the_output_span() {
        let period = 300.0;
        let x = pulse_vowel(30_000, period, 5.0);
        let ring = ring_of(&x);
        let d = 2000.0;
        let mut tr = MarkTracker::new(48_000.0, 34.0, 873.0, d, 0.0);
        tr.on_control(&ring, true, period);
        assert!(tr.locked());
        let first = tr.store.get(0).unwrap().pos;
        assert!(first < 30_000.0 - d - period, "first mark {first}");
        // Every mark of the seed is known at the seed time.
        for i in 0..tr.store.len() {
            assert_eq!(tr.store.get(i).unwrap().known_at, 30_000);
        }
    }

    #[test]
    fn noise_does_not_lock_and_silence_is_safe() {
        let mut s = 0x1234_5678u64;
        let noise: Vec<f32> = (0..20_000)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        let mut tr = MarkTracker::new(48_000.0, 34.0, 873.0, 200.0, 0.0);
        let mut ring = Ring::new(1 << 16);
        for chunk in noise.chunks(64) {
            ring.extend(chunk);
            tr.advance(&ring);
            tr.on_control(&ring, true, 218.0);
        }
        // Noise may briefly seed by chance but never stays locked for long.
        assert!(tr.store.len() < 40, "{} marks on noise", tr.store.len());
        let mut tr = MarkTracker::new(48_000.0, 34.0, 873.0, 200.0, 0.0);
        let silence = Ring::new(1 << 12);
        tr.on_control(&silence, true, 218.0);
        tr.advance(&silence);
        assert!(!tr.locked());
        assert!(tr.mark_cents().is_none());
        // Garbage hints are ignored.
        let ring = ring_of(&pulse_vowel(20_000, 218.0, 0.0));
        tr.on_control(&ring, true, f64::NAN);
        tr.on_control(&ring, true, 1e9);
        tr.on_control(&ring, true, -5.0);
        assert!(!tr.locked());
    }

    #[test]
    fn unlocks_when_unvoiced_and_reseeds_on_octave_jump() {
        // An octave jump up: the ±10 % chain keeps finding the old period
        // (two new periods per mark), so only the detector hint can fix it.
        let period = 218.0;
        let mut x = pulse_vowel(200 * 64, period, 0.0);
        x.extend(pulse_vowel(300 * 64, period / 2.0, 0.0));
        let mut tr = MarkTracker::new(48_000.0, 34.0, 873.0, 200.0, 0.0);
        let mut ring = Ring::new(1 << 16);
        for (h, chunk) in x.chunks(64).enumerate() {
            ring.extend(chunk);
            tr.advance(&ring);
            let hint = if h < 200 { period } else { period / 2.0 };
            tr.on_control(&ring, h < 280, hint);
            if h == 199 {
                assert!(tr.locked());
                assert_eq!(tr.seeds, 1);
            }
            if h == 270 {
                assert!(tr.locked());
                let p = tr.period().unwrap();
                assert!((p - period / 2.0).abs() < 1.0, "period after jump {p}");
            }
        }
        assert!(tr.seeds >= 2, "seeds {}", tr.seeds);
        // The final unvoiced hops unlocked.
        assert!(!tr.locked());
        assert!(tr.mark_cents().is_none());
    }

    /// DSP-2 / R§8.13 octave guard: a detector hint an octave low (2·T) never
    /// puts the marks at 2·T, whether it arrives at the seed or as a glitch
    /// (or a long run) while the tracker is locked at T. At 2·T every grain
    /// would hold two pulses whose spacing PSOLA cannot shift.
    #[test]
    fn octave_low_hints_do_not_pull_marks_to_two_periods() {
        for &period in &[48_000.0 / 110.0, 48_000.0 / 200.0, 48_000.0 / 330.0] {
            // Seeded with 2·T from the start.
            let x = pulse_vowel(30_000, period, 7.0);
            let mut tr = MarkTracker::new(48_000.0, 34.0, 873.0, 200.0, 0.0);
            let mut ring = Ring::new(1 << 16);
            for chunk in x.chunks(64) {
                ring.extend(chunk);
                tr.advance(&ring);
                tr.on_control(&ring, true, 2.0 * period);
            }
            let p = tr.period().expect("locked");
            assert!((p / period - 1.0).abs() < 0.01, "T={period:.1}: seeded on a 2·T hint, marks at {p:.1}");
            // Locked at T, then 2·T glitches of 1-30 hops and a sustained run.
            for &k in &[1usize, 2, 3, 8, 30, 300] {
                let mut tr = MarkTracker::new(48_000.0, 34.0, 873.0, 200.0, 0.0);
                let mut ring = Ring::new(1 << 16);
                for (h, chunk) in x.chunks(64).enumerate() {
                    ring.extend(chunk);
                    tr.advance(&ring);
                    let glitch = (100..100 + k).contains(&h);
                    tr.on_control(&ring, true, if glitch { 2.0 * period } else { period });
                    if tr.seeds > 0 {
                        let p = tr.period().expect("locked");
                        assert!((p / period - 1.0).abs() < 0.01, "T={period:.1} glitch {k}: hop {h} marks at {p:.1}");
                    }
                }
                assert!(tr.seeds == 1 && tr.locked(), "T={period:.1} glitch {k}: {} seeds", tr.seeds);
            }
        }
    }

    /// HQ/Max chaining (R§7.2.2): between two NCC peaks of similar height
    /// the period-continuity cost picks the one nearer the predicted period;
    /// without a prediction (Eco) the higher one wins.
    #[test]
    fn continuity_resolves_competing_peaks() {
        // Two trains of short bumps, 15 % apart in period and of similar
        // strength: from any position the NCC has a sharp peak at each lag.
        let (ta, tb) = (190.0, 219.0);
        let n = 30_000;
        let mut x = vec![0.0f32; n];
        for (period, amp, phase) in [(ta, 1.0f32, 0.0), (tb, 0.9, 61.0)] {
            let mut c = phase;
            while c < n as f64 {
                for d in -4i64..=4 {
                    let i = c.round() as i64 + d;
                    if (0..n as i64).contains(&i) {
                        x[i as usize] += amp * (0.5 + 0.5 * (core::f64::consts::PI * d as f64 / 5.0).cos()) as f32;
                    }
                }
                c += period;
            }
        }
        let ring = ring_of(&x);
        let mut tr = MarkTracker::new(48_000.0, 34.0, 873.0, 200.0, 0.0);
        let (lo, hi) = (180.0, 232.0);
        let mut checked = 0;
        for k in 0..60 {
            let m = 8_000.0 + 97.3 * k as f64;
            let at = |tr: &mut MarkTracker, prior| tr.search(&ring, m, 205.0, lo, hi, Dir::Forward, n as u64, prior);
            let (Some(free), Some(near_a), Some(near_b)) =
                (at(&mut tr, None), at(&mut tr, Some(ta)), at(&mut tr, Some(tb)))
            else {
                continue;
            };
            // Positions where each prior found its own peak, of similar height.
            if (near_a.0 - ta).abs() < 2.0 && (near_b.0 - tb).abs() < 2.0 {
                checked += 1;
                assert!((near_a.1 - near_b.1).abs() < CONTINUITY_PER_ST as f32 * 12.0 * 0.21 + 0.01);
                // Without a prior the higher peak wins.
                let hi = if near_a.1 >= near_b.1 { near_a.0 } else { near_b.0 };
                assert!((free.0 - hi).abs() < 0.5, "m={m}: free {:.1} vs higher peak {hi:.1}", free.0);
            }
        }
        assert!(checked >= 10, "only {checked} of 60 positions resolved both peaks");
    }

    #[test]
    fn local_maxima_finds_best_and_runner_up() {
        let v = [0.1, 0.5, 0.2, 0.3, 0.9, 0.4, 0.45, 0.2, 0.8, 0.1];
        let [(i0, v0), (i1, v1)] = local_maxima(&v, 2);
        assert_eq!((i0, v0), (4, 0.9));
        assert_eq!((i1, v1), (8, 0.8));
        // The runner-up must be `sep` away from the best.
        assert_eq!(local_maxima(&v, 4)[1], (8, 0.8));
        assert_eq!(local_maxima(&v, 5)[1].1, f32::MIN);
        // Range ends count as maxima; an empty range has none.
        assert_eq!(local_maxima(&[0.9, 0.1, 0.2], 1)[1], (2, 0.2));
        assert_eq!(local_maxima(&[], 1)[0].1, f32::MIN);
    }

    /// The octave guard still lets the tracker follow a real octave jump
    /// down (and up, see above) when the detector reports it.
    #[test]
    fn follows_a_real_octave_drop() {
        let period = 150.0;
        let mut x = pulse_vowel(200 * 64, period, 0.0);
        x.extend(pulse_vowel(300 * 64, 2.0 * period, 0.0));
        let mut tr = MarkTracker::new(48_000.0, 34.0, 873.0, 200.0, 0.0);
        let mut ring = Ring::new(1 << 16);
        for (h, chunk) in x.chunks(64).enumerate() {
            ring.extend(chunk);
            tr.advance(&ring);
            tr.on_control(&ring, true, if h < 200 { period } else { 2.0 * period });
        }
        let p = tr.period().expect("locked");
        assert!((p / (2.0 * period) - 1.0).abs() < 0.01, "after the drop: {p:.1}");
    }
}

#[cfg(test)]
mod filter_tests {
    use super::*;

    #[test]
    fn half_sample_filter_interpolates() {
        let h = half_sample_filter();
        assert!((h.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // Symmetric about the half-sample point.
        for k in 0..8 {
            assert!((h[k] - h[15 - k]).abs() < 1e-7);
        }
        // Midpoint of a band-limited sine within -80 dB up to 0.25 fs.
        for &f in &[0.01f64, 0.05, 0.12, 0.25] {
            let x: Vec<f32> =
                (0..16).map(|n| (2.0 * core::f64::consts::PI * f * n as f64 + 0.3).sin() as f32).collect();
            let y = simd::dot(&h, &x) as f64;
            let exact = (2.0 * core::f64::consts::PI * f * 7.5 + 0.3).sin();
            assert!((y - exact).abs() < 1e-4, "f={f}: {y} vs {exact}");
        }
    }

    #[test]
    fn period_at_follows_known_marks() {
        let mut tr = MarkTracker::new(48_000.0, 34.0, 600.0, 200.0, 0.0);
        tr.store.push(Mark { pos: 100.0, period: 50.0, known_at: 120, id: 0, run: 1, score: 1.0 });
        tr.store.push(Mark { pos: 151.0, period: 51.0, known_at: 170, id: 0, run: 1, score: 1.0 });
        assert_eq!(tr.period_at(100), None);
        assert_eq!(tr.period_at(120), Some(50.0));
        assert_eq!(tr.period_at(169), Some(50.0));
        assert_eq!(tr.period_at(170), Some(51.0));
        // Ages out two longest periods after the newest mark became known.
        let age = 170 + (2.0 * tr.t_hi).ceil() as u64;
        assert_eq!(tr.period_at(age - 1), Some(51.0));
        assert_eq!(tr.period_at(age), None);
        // A new mark invalidates the memo.
        tr.store.push(Mark { pos: 202.0, period: 52.0, known_at: 200, id: 0, run: 1, score: 1.0 });
        assert_eq!(tr.period_at(201), Some(52.0));
    }
}
