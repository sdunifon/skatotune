//! Pitch-synchronous overlap-add (TD-PSOLA): the Natural engine, causal in
//! Live/Live+ (L1/L2) and lookahead in Studio (S1). R§5.3, R§7.4, R§8.13-8.16,
//! R§8.18-8.19.
//!
//! ### Time base
//!
//! Output sample `t` carries input content from about `u = t − D`. Synthesis
//! marks `s_k` live on that `u` axis; grain k reads the input around its source
//! mark `m_j`:
//!
//! `y(u) = Σ_k w_k(u) · x(m_j + γ·(u − s_k))`            (R§7.5, R§8.16)
//!
//! with fractional positions everywhere and the read low-passed for rate γ
//! (formant read). The content offset `e_k = m_j − s_k` is the PSOLA "wander"
//! (R§2.1): the content delay is `D − e_k`.
//!
//! ### Synthesis marks and boundaries (R§8.14, R§8.15)
//!
//! `s_{k+1} = s_k + T_s`, `T_s = T_a/β` with `T_a` the input period at grain
//! k's source mark (the spacing to the next mark once it is known) and β the
//! pitch ratio, sampled once per synthesis period: the newest commanded ratio
//! (the end value of the control's per-hop ramp, which the per-sample value
//! trails by up to a hop). Each boundary between grain k and k+1 is decided as
//! late as possible, from the marks known at that output sample:
//!
//! * **Symmetric** (Hann halves, R§8.15), whenever the next grain's mark can
//!   be known in time, `h + d_mark(T_a) + h_interp ≤ D` with
//!   `h = min(T_s, T_a/γ)`, and the halves are at most 3 ms long (longer
//!   ones are committed lazily, below). Decided when the output reaches `s_k`; grain k
//!   falls over `[s_k, s_k + h]` and grain k+1 rises over `[s_{k+1} − h,
//!   s_{k+1}]`. Shifting up, `h = T_s` and the halves are exactly
//!   complementary (window sum 1, so β = 1 is transparent). Shifting down,
//!   `h = T_a/γ < T_s` leaves the classic TD-PSOLA gap instead of reaching the
//!   neighbouring pulse (R§8.15 "never extend a grain past the neighbouring
//!   pulse"). R§8.15 writes `ρ·T_a/γ` with ρ ≈ 0.9; ρ = 1 is used because
//!   ρ < 1 would break the window sum at β = 1 (an amplitude dip every period,
//!   so no transparency), and the Hann tail is already ≈ 0 at the neighbour.
//! * **Lazy symmetric** when the symmetric halves would be longer than 3 ms
//!   (Studio and Live+ below ≈ 330 Hz): the crossfade is `max(h/2, 3 ms)`,
//!   committed like an asymmetric boundary at `s_{k+1} − L`, so the period
//!   it shapes carries the ratio commanded near its middle instead of at
//!   `s_k` (ON-3: the realised shift lagged the command by ≈ T/2 + one hop,
//!   up to 8 ms at 82 Hz; see `lazy_rise`).
//! * **Asymmetric short-rise** otherwise (low voices in Live): the next grain
//!   rises over `L_b` just before `s_{k+1}`, grain k is flat until then and
//!   falls complementarily. `L_b` is at least the research's
//!   `min(1 ms, T_s/4)` and otherwise as long as the look-ahead left after the
//!   mark delay allows (see `asym_rise`, also for its cost in heard latency,
//!   DSP-9). The fall is committed only when the
//!   output reaches `s_{k+1} − L_b`, by which time the next mark (and so the
//!   exact `T_a`) is known, which keeps `e` stable at β = 1. If the grain would
//!   run past the next closure (`T_s > T_a/γ`, shifting down), it falls early
//!   at `s_k + T_a/γ` and leaves a gap. The window sum is 1 or 0 everywhere,
//!   so no normalisation is needed.
//!
//! ### Grain source choice (R§8.14)
//!
//! Among marks known at the output sample, the next grain keeps the mark
//! sequence (`j+1`) while `e` stays inside a keep window, otherwise takes the
//! mark nearest to `s_{k+1}`: the Studio nearest-mark rule. Keeping the
//! sequence, `e` drifts by `T_a − T_s` per grain (up when shifting up). The
//! window ends at `½·T` on the side `e` drifts towards and at `(½ + 0.1)·T`
//! on the other, so a steady drift cycles `e` over `[−½T, ½T]` (mean 0: the
//! Studio content delay is centred on D, R§7.2.1) while a ratio hovering at
//! a repeat/skip threshold cannot flip-flop: the drift direction only
//! changes once `e` has moved 0.1·T back from its extreme. (A symmetric
//! `±0.6·T` window biased the mean by ±0.1·T with the correction's
//! direction, 2 ms at 55 Hz.) In Live the marks ahead of `s` are not known
//! yet, so the rule degrades to "latest known mark" and `e` wanders in
//! `(−T, 0]`: the content delay averages D + T/2 (D + 0.45·T measured on
//! steady vowels), the "heard" latency (R§2.1). If the sequence's next mark is not known yet and the latest
//! known mark is more than a period back (the tracker published it late),
//! the grain takes the *predicted* next mark instead (head mark + its
//! period; never twice in a row), which keeps `e` and is transparent at
//! β = 1 (SN-9). Every choice
//! satisfies the causality bound `e ≤ D − h_interp − 1 − (1−γ)·rise`, and the
//! grain extent for γ > 1 is clamped so its last read stays causal.
//!
//! **HQ/Max multi-candidate choice (R§7.2.2).** Where a grain leaves the mark
//! sequence (a repeat, a skip, a new tracker run), HQ/Max score the two or
//! three nearest causal marks by the overlap correlation of their grain
//! with the outgoing one over the crossfade, less a small penalty for
//! content offset, instead of taking the nearest; Eco takes the nearest. On
//! near-periodic material every candidate matches about equally and the
//! choice is the nearest; it matters only for irregular cycles.
//!
//! ### Voicing (R§8.18)
//!
//! Idle, the engine outputs exactly `x[t − D]`. At a voiced onset it starts on
//! a mark with `s_0 = m_0` (e = 0), cross-fading from the dry signal over the
//! first grain's rise (the dry signal acts as the previous grain, so at γ = 1
//! the start is seamless); from then on β is the control's ratio (the control
//! law decides how fast the correction starts). When the tracker's newest
//! period is clean (NCC ≥ 0.9) and agrees with the detector, the first grain
//! may sit on the newest mark at or just before the content position (ON-4):
//! it reads the dry signal itself, and its boundary to the next grain, the
//! first place β acts, comes up to a period sooner than waiting for a mark
//! ahead of the content (a low voice in Live waited 7-10 ms). When the control turns
//! unvoiced it cross-fades to the dry signal over 4 ms (equal gain: the
//! signals are correlated) and goes idle. If the marks run out (the tracker
//! lost the voice) the next grain is the dry signal itself. When an engine
//! switch fades over from Classic, the onset instead starts at Classic's
//! content offset (lead-in read at that offset too), so the 10 ms engine
//! crossfade sums two aligned copies (DSP-7).

use vt_core::ring::Ring;

use crate::latency::mark_delay;
use crate::marks::MarkStore;
use crate::read::{RcTable, ReadStats, Reader};

/// Channels processed with identical timing.
pub(crate) const MAX_CH: usize = 2;

/// Grain slots (at most three overlap; the rest is headroom).
const MAX_GRAINS: usize = 8;

/// Shortest synthesis period, samples (β is clamped, this only guards maths).
const MIN_TS: f64 = 2.0;

/// Keep-window hysteresis, fraction of T: the window reaches (½ + this)·T on
/// the side opposite the drift, and the drift direction changes only after
/// `e` moved back this far from its extreme.
const HYSTERESIS: f64 = 0.1;

/// Shortest crossfade of a lazily committed symmetric boundary, ms (see
/// `Psola::lazy_rise`).
const LAZY_RISE_MS: f64 = 3.0;

/// Points of the overlap correlation scoring a grain candidate (HQ/Max).
const OLC_POINTS: usize = 64;
/// Overlap-correlation penalty per period of content offset: of two
/// candidates matching equally, the nearer one wins.
const OLC_OFFSET_PENALTY: f64 = 0.1;

/// An onset on a past mark needs the mark's verifying NCC at least this
/// high, and its period within this many cents of the control's hint.
const ONSET_PAST_SCORE: f32 = 0.9;
const ONSET_PAST_AGREE_CENTS: f64 = 25.0;

/// Shortest first crossfade of an onset on a past mark, as a fraction of
/// the crossfade the scheduler would use there.
const ONSET_MIN_RISE: f64 = 0.5;

/// Largest read skew between an onset grain on a past mark and the lead-in
/// it cross-fades from (γ ≠ 1), as a fraction of the period.
const ONSET_MAX_SKEW: f64 = 0.1;

/// Content offset beyond which the marks have "run out", in periods.
const E_LOST: f64 = 1.25;

/// Wet → dry fade when the control turns unvoiced, ms (R§8.18: 3-5 ms).
pub(crate) const STOP_FADE_MS: f64 = 4.0;

/// Everything the engine reads for one output sample.
pub(crate) struct Ctx<'a> {
    pub rings: &'a [Ring],
    /// The mono analysis signal (grain choices depend on it only, so every
    /// channel gets the same schedule).
    pub analysis: &'a Ring,
    pub marks: &'a MarkStore,
    pub reader: Reader<'a>,
    pub rc: &'a RcTable,
    /// Pitch ratio β (output f0 / input f0) where a grain boundary fixes the
    /// next synthesis period: the value the current control ramp ends at,
    /// i.e. the newest commanded ratio. (The per-sample ramp value lags it by
    /// up to a hop; PSOLA samples β once per synthesis period, R§8.14.)
    pub beta_target: f64,
    /// Formant ratio γ at this sample.
    pub gamma: f64,
    /// Control says the input is pitched.
    pub voiced: bool,
    /// Period to fall back on when a grain has no measured one.
    pub fallback_period: f64,
    /// The control's period hint (0 when none).
    pub period_hint: f64,
    /// HQ/Max: when a grain leaves the mark sequence, choose among the
    /// nearest marks by overlap correlation (R§7.2.2).
    pub hq: bool,
}

#[derive(Clone, Copy, Debug)]
struct Grain {
    /// Centre on the output (`u = t − D`) axis.
    s: f64,
    /// Source position read at `u = s`.
    m: f64,
    gamma: f64,
    rise_start: f64,
    rise_len: f64,
    /// `+∞` until the boundary to the next grain is committed.
    fall_start: f64,
    fall_len: f64,
    /// Source mark (id, run); `None` for a dry grain (reads `x[u]` exactly).
    src: Option<(u64, u32)>,
    /// Input period at the source mark.
    period: f64,
    /// `period` is the measured spacing to the next mark.
    period_final: bool,
}

impl Grain {
    #[inline]
    fn weight(&self, u: f64, rc: &RcTable) -> f32 {
        if u < self.rise_start {
            return 0.0;
        }
        if u < self.rise_start + self.rise_len {
            return rc.rc((u - self.rise_start) / self.rise_len);
        }
        if u < self.fall_start {
            return 1.0;
        }
        if u < self.fall_start + self.fall_len {
            return 1.0 - rc.rc((u - self.fall_start) / self.fall_len);
        }
        0.0
    }

    fn fall_end(&self) -> f64 {
        self.fall_start + self.fall_len
    }

    /// Longest extent after the centre for which a read at rate γ > 1 stays
    /// causal: `m + γ(u − s) + h_interp + 1 ≤ u + D`.
    fn causal_extent(&self, d: f64, hw: f64) -> f64 {
        if self.src.is_none() || self.gamma <= 1.0 {
            return f64::INFINITY;
        }
        ((d - hw - 1.0 - (self.m - self.s)) / (self.gamma - 1.0)).max(1.0)
    }
}

/// Scheduling state of the newest ("head") grain.
#[derive(Clone, Copy, Debug)]
enum Phase {
    /// Boundary to the next grain is decided when the output reaches the
    /// head's centre.
    AwaitCentre,
    /// Asymmetric boundary: the fall is committed lazily.
    AsymPending { lb: f64 },
    /// The head fell early (shift down past the next closure); the next
    /// centre is still open.
    AwaitNextCentre { lb: f64 },
    /// The next grain is created when the output reaches its rise.
    AwaitCreate { s: f64, rise_start: f64, rise_len: f64 },
}

/// Diagnostics of the PSOLA engine.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PsolaStats {
    pub grains: u64,
    pub repeats: u64,
    pub skips: u64,
    pub dry_grains: u64,
    /// Grains on a predicted mark (see `Psola::predicted_grain`).
    pub predicted: u64,
    /// Grains whose content offset is more than a period back (`e < −T`).
    pub beyond_period: u64,
    pub onsets: u64,
    pub symmetric: u64,
    pub asymmetric: u64,
    /// Content offset e (samples) of the newest real grain, its range and
    /// the sum over all grains (`e_sum / grains` is the mean offset).
    pub e_last: f64,
    pub e_min: f64,
    pub e_max: f64,
    pub e_sum: f64,
    /// Source position and centre of the newest grain (tests check its
    /// phase and the synthesis periods).
    pub m_last: f64,
    pub s_last: f64,
    pub read: ReadStats,
}

pub(crate) struct Psola {
    d: f64,
    /// `D` as an integer (the dry path is an exact integer delay).
    d_int: i64,
    hw: f64,
    rise: f64,
    /// Shortest crossfade of a lazily committed symmetric boundary, samples.
    lazy_min: f64,
    stop_step: f32,
    active: bool,
    grains: [Grain; MAX_GRAINS],
    n: usize,
    phase: Phase,
    prev_src: Option<(u64, u32)>,
    /// Dry signal fading out under the first grain's rise: (start, len).
    dry_lead: Option<(f64, f64)>,
    /// Content offset of the lead-in signal and of the first grain: 0 for a
    /// voiced onset (the lead is the exact dry signal); the outgoing Classic
    /// engine's offset when an engine switch starts the PSOLA (see
    /// [`Psola::align_next_onset`]).
    lead_e: f64,
    /// Offset the next onset should start at (engine switch), if any.
    align: Option<f64>,
    /// Wet → dry fade progress (0 = wet).
    stop: f32,
    /// Drift of `e` when keeping the mark sequence: direction (−1, 0, +1),
    /// the drift accumulated since the onset (no re-centring jumps), and its
    /// extreme in the current direction.
    drift_dir: i8,
    drift_e: f64,
    drift_ext: f64,
    lost: bool,
    /// The head grain sits on a predicted mark.
    virt_head: bool,
    pub stats: PsolaStats,
}

const EMPTY: Grain = Grain {
    s: 0.0,
    m: 0.0,
    gamma: 1.0,
    rise_start: 0.0,
    rise_len: 0.0,
    fall_start: f64::INFINITY,
    fall_len: 0.0,
    src: None,
    period: 0.0,
    period_final: false,
};

impl Psola {
    /// Engine for latency `d` samples at sample rate `sr`; `hw` is the
    /// interpolation half-width budgeted in D.
    pub fn new(sr: f64, d: usize, hw: usize) -> Self {
        Self {
            d: d as f64,
            d_int: d as i64,
            hw: hw as f64,
            rise: crate::latency::rise_samples(sr),
            lazy_min: LAZY_RISE_MS * 1e-3 * sr,
            stop_step: (1.0 / (STOP_FADE_MS * 1e-3 * sr).max(1.0)) as f32,
            active: false,
            grains: [EMPTY; MAX_GRAINS],
            n: 0,
            phase: Phase::AwaitCentre,
            prev_src: None,
            dry_lead: None,
            lead_e: 0.0,
            align: None,
            stop: 0.0,
            drift_dir: 0,
            drift_e: 0.0,
            drift_ext: 0.0,
            lost: false,
            virt_head: false,
            stats: PsolaStats::default(),
        }
    }

    /// Back to idle (dry output); keeps the statistics.
    pub fn reset(&mut self) {
        self.active = false;
        self.n = 0;
        self.phase = Phase::AwaitCentre;
        self.prev_src = None;
        self.dry_lead = None;
        self.lead_e = 0.0;
        self.align = None;
        self.stop = 0.0;
        self.lost = false;
        self.virt_head = false;
        self.drift_dir = 0;
        self.drift_e = 0.0;
        self.drift_ext = 0.0;
    }

    /// Makes the next onset start at content offset `e` (samples) instead
    /// of 0, with the lead-in read at that offset too. Used when an engine
    /// switch fades from Classic, whose content runs `e` ahead of the dry
    /// signal: an equal-gain crossfade between two copies a fraction of a
    /// period apart would partly cancel (DSP-7). Cleared by [`Self::reset`]
    /// and by the onset that uses it.
    pub fn align_next_onset(&mut self, e: f64) {
        self.align = e.is_finite().then_some(e);
    }

    /// Input position the engine reads for output content position `u`
    /// (the most heavily weighted grain's), or `None` while idle.
    pub fn content_pos(&self, u: f64, rc: &RcTable) -> Option<f64> {
        if !self.active {
            return None;
        }
        let mut best: Option<(f32, f64)> = None;
        for g in self.grains.iter().take(self.n) {
            let w = g.weight(u, rc);
            let pos = if g.src.is_some() { g.m + g.gamma * (u - g.s) } else { u };
            if w > 0.0 && best.is_none_or(|(bw, _)| w > bw) {
                best = Some((w, pos));
            }
        }
        match (best, self.dry_lead) {
            (Some((w, pos)), _) if w >= 0.5 => Some(pos),
            (_, Some(_)) => Some(u + self.lead_e),
            (b, None) => b.map(|(_, pos)| pos),
        }
    }

    #[inline]
    fn head(&self) -> Option<&Grain> {
        self.n.checked_sub(1).and_then(|i| self.grains.get(i))
    }

    #[inline]
    fn head_mut(&mut self) -> Option<&mut Grain> {
        self.n.checked_sub(1).and_then(|i| self.grains.get_mut(i))
    }

    /// One output sample for every channel. `t` is the absolute index of the
    /// newest input sample (the output sample being computed).
    pub fn tick(&mut self, t: u64, cx: &Ctx, out: &mut [f32; MAX_CH]) {
        let nch = cx.rings.len().min(MAX_CH);
        let ui = t as i64 - self.d_int;
        let mut dry = [0.0f32; MAX_CH];
        if ui >= 0 {
            for (d, r) in dry.iter_mut().zip(cx.rings).take(nch) {
                *d = r.get(ui as u64);
            }
        }
        if !self.active && cx.voiced && ui >= 0 {
            self.try_onset(ui as f64, t, cx);
        }
        if !self.active {
            *out = dry;
            return;
        }
        let u = ui as f64;
        for _ in 0..6 {
            if !self.schedule(u, t, cx) {
                break;
            }
        }
        // Overlap-add (grain-major per sample; fixed grain order keeps the
        // summation order, and so the output, block-size independent).
        let mut acc = [0.0f32; MAX_CH];
        let hw = cx.reader.half_width();
        // Grains by reference (a grain is ~100 bytes; copying each one per
        // sample was measurable, PERF-6).
        let live = self.n.min(MAX_GRAINS);
        for g in self.grains.iter().take(live) {
            let w = g.weight(u, cx.rc);
            if w == 0.0 {
                continue;
            }
            if g.src.is_none() {
                for (a, &d) in acc.iter_mut().zip(&dry).take(nch) {
                    *a += w * d;
                }
                continue;
            }
            let pos = g.m + g.gamma * (u - g.s);
            self.stats.read.note(pos, hw, t);
            let rate = g.gamma as f32;
            for (a, r) in acc.iter_mut().zip(cx.rings).take(nch) {
                *a += w * cx.reader.read(r, pos, rate);
            }
        }
        if let Some((rs, rl)) = self.dry_lead {
            let wd = if u < rs {
                1.0
            } else if u < rs + rl {
                1.0 - cx.rc.rc((u - rs) / rl)
            } else {
                self.dry_lead = None;
                0.0
            };
            if self.lead_e == 0.0 {
                for (a, &d) in acc.iter_mut().zip(&dry).take(nch) {
                    *a += wd * d;
                }
            } else if wd > 0.0 {
                let pos = u + self.lead_e;
                self.stats.read.note(pos, hw, t);
                for (a, r) in acc.iter_mut().zip(cx.rings).take(nch) {
                    *a += wd * cx.reader.read(r, pos, 1.0);
                }
            }
        }
        // Wet ↔ dry fade for voicing changes.
        let want = cx.voiced && !self.lost;
        self.stop = if want { (self.stop - self.stop_step).max(0.0) } else { (self.stop + self.stop_step).min(1.0) };
        if self.stop <= 0.0 {
            *out = acc;
        } else if self.stop >= 1.0 {
            *out = dry;
            if !want {
                self.reset();
                return;
            }
        } else {
            let w = cx.rc.rc(self.stop as f64);
            for ch in 0..nch {
                out[ch] = acc[ch] + w * (dry[ch] - acc[ch]);
            }
        }
        self.retire(u + 1.0);
    }

    /// Drops grains that have finished before `u_next` (never the head).
    /// Compacts only when a grain is dropped (most samples drop none).
    fn retire(&mut self, u_next: f64) {
        let older = self.n.saturating_sub(1).min(MAX_GRAINS);
        if self.grains.iter().take(older).all(|g| u_next < g.fall_end()) {
            return;
        }
        let mut k = 0;
        for i in 0..self.n {
            let Some(&g) = self.grains.get(i) else { break };
            let keep = i + 1 == self.n || u_next < g.fall_end();
            if keep {
                if let Some(slot) = self.grains.get_mut(k) {
                    *slot = g;
                }
                k += 1;
            }
        }
        self.n = k;
    }

    /// Longest rise a grain on a mark of period `t` can have so that its mark
    /// is known when the rise starts: `D − d_mark(T) − h_interp`. The latency
    /// formula sizes this for the lowest note of the range, so higher notes
    /// have room to spare.
    fn rise_budget(&self, t: f64) -> f64 {
        self.d - mark_delay(t) - self.hw
    }

    /// Rise of an asymmetric boundary: the research's short rise
    /// (`min(1 ms, T_s/4)`, R§8.15) at least, but as much of the remaining
    /// look-ahead budget as is available (minus 2 samples; the lazy commit
    /// copes with the period estimate if the next mark arrives late), up to
    /// the symmetric half `h`. This is a continuum between the short-rise
    /// grain of a low voice and the symmetric Hann grain: a 220 Hz voice in
    /// Live gets a ≈ 3 ms rise instead of 1 ms, which keeps the spectral
    /// smearing of the splice close to the symmetric case.
    ///
    /// It is not free in heard latency (DSP-9): with the whole look-ahead
    /// spent on the rise, no mark after the grain centre is known when a
    /// grain is created, so the content delay averages D + ≈ 0.45·T below
    /// ≈ 250 Hz. Measured alternatives (mean content delay under ±35 c, 48
    /// kHz, against formant error):
    ///
    /// | rise rule | Live 82 / 110 / 165 / 220 Hz | Live+ 82 / 110 Hz | formant cost |
    /// |---|---|---|---|
    /// | whole budget (kept) | 11.3 / 8.5 / 7.3 / 6.5 ms | 15.5 / 14.0 ms | – |
    /// | budget − T/2, ≥ 2 ms | 9.9 / 8.1 / 6.1 / 5.2 ms | 11.1 / 10.1 ms | suite envelope LSD Live HQ +0.4-1.1 dB (low_male 0.9 → 2.0, melody_male 1.2 → 2.1); melody_male onset 2 settles 23 → 43 ms |
    /// | ≤ 1 ms | 8.9 / 7.3 / 5.1 / 4.5 ms | 8.9 / 9.8 ms | observed envelope error at 220 Hz 1.4-2.8 → 4.3-6.0 dB |
    ///
    /// (On vocadito the 2 ms variant lowered Live HQ hard-tune onset
    /// glitches 16.0 → 14.5 % and the amount-0 wander 2.1 → 1.0 periods, but
    /// the synthetic envelope and settling costs above kept the whole-budget
    /// rise.)
    fn asym_rise(&self, avail: f64, t_eff: f64, h: f64) -> f64 {
        let short = self.rise.min(0.25 * t_eff).max(1.0);
        (avail - 2.0).min(h).max(short)
    }

    /// Crossfade of a boundary that could be symmetric (`h ≤` the rise
    /// budget): `h` itself when it is short, else `max(h/2, 3 ms)`, committed
    /// lazily like an asymmetric boundary (ON-3). A symmetric boundary fixes
    /// the next synthesis period when the output reaches the head's centre,
    /// with β of that instant, so the period after it carries a ratio up to
    /// a period old: the realised shift lagged the commanded one by
    /// ≈ T/2 + one hop (Studio: 8 ms at 82 Hz, 5 ms at 147 Hz), which was
    /// most of Studio's settling time on real singing. Committed at
    /// `s_{k+1} − L` instead, the decision sits near the middle of the period
    /// it shapes. The crossfade never gets shorter than 3 ms, the rise Live
    /// already uses at 220 Hz; ≥ 330 Hz the halves are shorter than that and
    /// stay symmetric. Measured: vocadito Studio HQ hard chromatic settle
    /// median / p90 35.1 / 83.1 → 22.4 / 54.2 ms, output p95 22.0 → 20.1 c,
    /// envelope LSD 0.75 → 0.65 dB (in key: p95 14.1 → 10.2 c). The cost is
    /// on the synthetic voices, where the shorter, flat-topped grain smears
    /// the envelope a little more: suite envelope LSD Studio HQ +0.15-0.5 dB
    /// (melody_male 0.68 → 1.09, vibrato 1.10 → 1.64; Live's asymmetric
    /// grains score 1.2-1.9 there), the crate's observed-envelope error at
    /// 220 Hz up to +1.5 dB (+1 st), unchanged at 100 and 440 Hz.
    fn lazy_rise(&self, h: f64) -> f64 {
        h.min((0.5 * h).max(self.lazy_min))
    }

    /// Updates the head grain's period with the measured spacing to the next
    /// mark of its run once that mark is known.
    fn refresh_period(&mut self, t: u64, marks: &MarkStore) {
        let Some(g) = self.head() else { return };
        if g.period_final {
            return;
        }
        let Some((id, run)) = g.src else { return };
        let spacing = marks.find_id(id).and_then(|i| {
            let (a, b) = (marks.get(i)?, marks.get(i + 1)?);
            (b.known_at <= t + 1 && b.run == run).then_some(b.pos - a.pos)
        });
        if let (Some(sp), Some(g)) = (spacing, self.head_mut())
            && sp > 0.5 * g.period
            && sp < 2.0 * g.period
        {
            g.period = sp;
            g.period_final = true;
        }
    }

    /// Advances the head grain's scheduling by at most one step. Returns
    /// whether anything changed (the caller loops, bounded).
    fn schedule(&mut self, u: f64, t: u64, cx: &Ctx) -> bool {
        match self.phase {
            Phase::AwaitCentre => {
                let Some(&h) = self.head() else { return false };
                if u < h.s {
                    return false;
                }
                self.refresh_period(t, cx.marks);
                let Some(&h) = self.head() else { return false };
                let beta = cx.beta_target;
                let ts = (h.period / beta).max(MIN_TS);
                let hh = ts.min(h.period / h.gamma).min(h.causal_extent(self.d, self.hw));
                let avail = self.rise_budget(h.period);
                let lazy = self.lazy_rise(hh);
                if hh <= avail && lazy < hh {
                    // Long symmetric halves: commit lazily (ON-3, see
                    // `lazy_rise`).
                    self.stats.asymmetric += 1;
                    self.phase = Phase::AsymPending { lb: lazy };
                } else if hh <= avail {
                    self.stats.symmetric += 1;
                    if let Some(g) = self.head_mut() {
                        g.fall_start = h.s;
                        g.fall_len = hh;
                    }
                    // Coincident halves share the exact same numbers.
                    let rise_start = if hh >= ts { h.s } else { h.s + ts - hh };
                    self.phase = Phase::AwaitCreate { s: rise_start + hh, rise_start, rise_len: hh };
                } else {
                    self.stats.asymmetric += 1;
                    let lb = self.asym_rise(avail, ts.min(h.period / h.gamma), hh);
                    self.phase = Phase::AsymPending { lb };
                }
                true
            }
            Phase::AsymPending { lb } => {
                self.refresh_period(t, cx.marks);
                let Some(&h) = self.head() else { return false };
                let beta = cx.beta_target;
                let ts = (h.period / beta).max(MIN_TS);
                let end_content = h.s + (h.period / h.gamma).min(h.causal_extent(self.d, self.hw));
                let s_next = h.s + ts;
                if s_next <= end_content {
                    if u < s_next - lb {
                        return false;
                    }
                    // The fall starts at its exact fractional time, which
                    // is at most one sample in the past (an integer `u` just
                    // crossed it). Only a β jump could put it further back;
                    // clamp to the previous sample so both halves stay
                    // continuous (and after the head's own rise).
                    let fs = (s_next - lb).max(u - 1.0).max(h.rise_start + h.rise_len);
                    if let Some(g) = self.head_mut() {
                        g.fall_start = fs;
                        g.fall_len = lb;
                    }
                    self.phase = Phase::AwaitCreate { s: fs + lb, rise_start: fs, rise_len: lb };
                } else {
                    let fl = lb.min(end_content - h.s).max(1.0);
                    if u < end_content - fl {
                        return false;
                    }
                    let fs = (end_content - fl).max(u - 1.0).max(h.rise_start + h.rise_len);
                    if let Some(g) = self.head_mut() {
                        g.fall_start = fs;
                        g.fall_len = fl;
                    }
                    self.phase = Phase::AwaitNextCentre { lb };
                }
                true
            }
            Phase::AwaitNextCentre { lb } => {
                let Some(&h) = self.head() else { return false };
                let beta = cx.beta_target;
                // The next rise may overlap the early fall (a dip in the
                // window sum, like symmetric TD-PSOLA shifting down) but must
                // not start before it, or the sum would exceed 1.
                let s_next = (h.s + (h.period / beta).max(MIN_TS)).max(h.fall_end());
                if u < s_next - lb {
                    return false;
                }
                let rs = (s_next - lb).max(u - 1.0);
                self.phase = Phase::AwaitCreate { s: rs + lb, rise_start: rs, rise_len: lb };
                true
            }
            Phase::AwaitCreate { s, rise_start, rise_len } => {
                if u < rise_start {
                    return false;
                }
                self.create(s, rise_start, rise_len, t, cx);
                self.phase = Phase::AwaitCentre;
                true
            }
        }
    }

    /// Creates the next grain centred at `s`, choosing its source mark among
    /// the marks known at output sample `t`.
    fn create(&mut self, s: f64, rise_start: f64, rise_len: f64, t: u64, cx: &Ctx) {
        let gamma = cx.gamma;
        let t_ref = self.head().map_or(cx.fallback_period, |g| g.period);
        let t_ref = if t_ref.is_finite() && t_ref > 1.0 { t_ref } else { cx.fallback_period.max(16.0) };
        // Causality: the centre read and (γ < 1) the rise-start read.
        let e_hi = self.d - self.hw - 1.0 - ((1.0 - gamma) * (s - rise_start)).max(0.0);
        let marks = cx.marks;
        let known = marks.known_len(t + 1);
        // Drift of `e` since the previous grain if the sequence is kept.
        if let Some(h) = self.head().filter(|h| h.src.is_some()) {
            self.track_drift(h.period - (s - h.s), t_ref);
        }
        let (lo, hi) = self.keep_window(t_ref);
        let mut pick = None;
        // Keep the mark sequence while the offset stays inside the window.
        if let Some((pid, prun)) = self.prev_src
            && let Some(i) = marks.find_id(pid + 1)
            && i < known
            && let Some(m) = marks.get(i)
            && m.run == prun
        {
            let e = m.pos - s;
            if e <= e_hi && (lo..=hi).contains(&e) {
                pick = Some(i);
            }
        }
        if pick.is_none() {
            // Nearest known mark: the last one at or before `s` or the first
            // one after it (marks are sorted by position).
            let j = marks.lower_bound(s, known);
            let mut best: Option<(usize, f64)> = None;
            for i in [j.checked_sub(1), Some(j)].into_iter().flatten() {
                let Some(m) = marks.get(i).filter(|_| i < known) else { continue };
                let e = m.pos - s;
                if e <= e_hi && best.is_none_or(|(_, be)| e.abs() < be.abs()) {
                    best = Some((i, e));
                }
            }
            pick = best.map(|b| b.0);
            if cx.hq
                && let Some(i) = self.best_overlap(j, s, rise_start, rise_len, e_hi, t_ref, known, t, cx)
            {
                pick = Some(i);
            }
        }
        let chosen = pick.and_then(|i| marks.get(i).map(|m| (i, *m))).filter(|(_, m)| m.pos - s >= -E_LOST * t_ref);
        if let Some(g) = self.predicted_grain(
            chosen.map(|(_, m)| m.pos - s),
            s,
            rise_start,
            rise_len,
            e_hi,
            (lo, hi),
            t_ref,
            known,
            cx,
        ) {
            self.push(g);
            return;
        }
        self.virt_head = false;
        let grain = match chosen {
            Some((i, m)) => {
                self.lost = false;
                let after = marks.get(i + 1).filter(|n| n.known_at <= t + 1 && n.run == m.run).map(|n| n.pos - m.pos);
                let (period, fin) = match after {
                    Some(sp) if sp > 0.5 * m.period && sp < 2.0 * m.period => (sp, true),
                    _ => (m.period, false),
                };
                let period = if period.is_finite() && period > 1.0 { period } else { t_ref };
                if let Some((pid, prun)) = self.prev_src
                    && prun == m.run
                {
                    if m.id == pid {
                        self.stats.repeats += 1;
                    } else if m.id > pid + 1 {
                        self.stats.skips += 1;
                    }
                }
                self.prev_src = Some((m.id, m.run));
                let e = m.pos - s;
                self.note_offset(e);
                if e < -t_ref {
                    self.stats.beyond_period += 1;
                }
                self.stats.m_last = m.pos;
                self.stats.s_last = s;
                Grain {
                    s,
                    m: m.pos,
                    gamma,
                    rise_start,
                    rise_len,
                    fall_start: f64::INFINITY,
                    fall_len: 0.0,
                    src: Some((m.id, m.run)),
                    period,
                    period_final: fin,
                }
            }
            None => {
                // The marks ran out: continue with the dry signal itself and
                // let the engine fade out.
                self.lost = true;
                self.stats.dry_grains += 1;
                Grain {
                    s,
                    m: s,
                    gamma: 1.0,
                    rise_start,
                    rise_len,
                    fall_start: f64::INFINITY,
                    fall_len: 0.0,
                    src: None,
                    period: t_ref,
                    period_final: true,
                }
            }
        };
        self.push(grain);
    }

    /// SN-9 / ON-4: the grain on the *predicted* next mark of the head's run
    /// (`m_head + T`) when that mark is not known yet and the best known
    /// mark would put the content offset more than a period back (`e <
    /// −T`, or no mark at all): the tracker published it late (a widened
    /// search after a seed or a miss, or a period shrinking faster than the
    /// latency margin covers), and falling back to the latest known mark
    /// jumped the content delay back by up to 1.25 periods and forward again
    /// a grain later. The predicted grain continues the sequence: at β = 1
    /// it reads exactly what the head grain would (`e` unchanged, so
    /// transparent), and at β ≠ 1 its period error is the period's change
    /// over one cycle. Never two in a row (if the tracker has lost the voice
    /// the marks really have run out); the next grain takes the real mark
    /// after the predicted one (ids of a run are consecutive).
    #[allow(clippy::too_many_arguments)]
    fn predicted_grain(
        &mut self,
        e_known: Option<f64>,
        s: f64,
        rise_start: f64,
        rise_len: f64,
        e_hi: f64,
        (lo, hi): (f64, f64),
        t_ref: f64,
        known: usize,
        cx: &Ctx,
    ) -> Option<Grain> {
        if self.virt_head || e_known.is_some_and(|e| e >= -t_ref) {
            return None;
        }
        let (pid, prun) = self.prev_src?;
        let h = *self.head().filter(|h| h.src == Some((pid, prun)))?;
        if cx.marks.find_id(pid + 1).is_some_and(|i| i < known && cx.marks.get(i).is_some_and(|m| m.run == prun)) {
            return None;
        }
        let m = h.m + h.period;
        let e = m - s;
        if !(m.is_finite() && e <= e_hi && (lo..=hi).contains(&e)) {
            return None;
        }
        self.virt_head = true;
        self.lost = false;
        self.prev_src = Some((pid + 1, prun));
        self.stats.predicted += 1;
        self.note_offset(e);
        self.stats.m_last = m;
        self.stats.s_last = s;
        Some(Grain {
            s,
            m,
            gamma: cx.gamma,
            rise_start,
            rise_len,
            fall_start: f64::INFINITY,
            fall_len: 0.0,
            src: Some((pid + 1, prun)),
            period: h.period,
            period_final: false,
        })
    }

    /// Updates the drift direction with the change `de` of `e` the next grain
    /// would see on the kept sequence (DSP-8).
    fn track_drift(&mut self, de: f64, t: f64) {
        if !de.is_finite() {
            return;
        }
        self.drift_e += de;
        let (e, rev) = (self.drift_e, HYSTERESIS * t);
        match self.drift_dir {
            1 if e < self.drift_ext - rev => (self.drift_dir, self.drift_ext) = (-1, e),
            -1 if e > self.drift_ext + rev => (self.drift_dir, self.drift_ext) = (1, e),
            0 if (e - self.drift_ext).abs() > rev => self.drift_dir = if e > self.drift_ext { 1 } else { -1 },
            1 => self.drift_ext = self.drift_ext.max(e),
            -1 => self.drift_ext = self.drift_ext.min(e),
            _ => {}
        }
    }

    /// Range of `e` over which the mark sequence is kept: `½·T` on the side
    /// `e` drifts towards, `(½ + h)·T` on the other (both before a direction
    /// is established).
    fn keep_window(&self, t: f64) -> (f64, f64) {
        let (near, far) = (0.5 * t, (0.5 + HYSTERESIS) * t);
        match self.drift_dir {
            1 => (-far, near),
            -1 => (-near, far),
            _ => (-far, far),
        }
    }

    /// HQ/Max multi-candidate grain choice (R§7.2.2): among the known marks
    /// around `s` (logical indices `j − 2 ..= j + 1`, causal and within
    /// ±(½ + h)·T of `s`), the one whose grain best continues the outgoing
    /// grain over the crossfade `[rise_start, rise_start + rise_len)`: the
    /// highest normalised overlap correlation, less a small penalty for
    /// content offset. Used where a grain leaves the mark sequence (repeat,
    /// skip, or a new tracker run), so a period whose waveform differs (an
    /// irregular cycle) is not spliced in where a better-matching one is
    /// available. Reads whole samples of the analysis signal already written;
    /// `None` without an outgoing grain or a scorable candidate.
    #[allow(clippy::too_many_arguments)]
    fn best_overlap(
        &self,
        j: usize,
        s: f64,
        rise_start: f64,
        rise_len: f64,
        e_hi: f64,
        t_ref: f64,
        known: usize,
        t: u64,
        cx: &Ctx,
    ) -> Option<usize> {
        let h = self.head().filter(|h| h.src.is_some())?;
        let ring = cx.analysis;
        let tol = (0.5 + HYSTERESIS) * t_ref;
        let stride = (rise_len / OLC_POINTS as f64).max(1.0);
        let newest = t as f64;
        let mut best: Option<(usize, f64)> = None;
        for i in j.saturating_sub(2)..(j + 2).min(known) {
            let Some(m) = cx.marks.get(i) else { continue };
            let e = m.pos - s;
            if e > e_hi || e.abs() > tol {
                continue;
            }
            let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
            for k in 0..OLC_POINTS {
                let u = rise_start + k as f64 * stride;
                if u >= rise_start + rise_len {
                    break;
                }
                let (pa, pb) = (h.m + h.gamma * (u - h.s), m.pos + cx.gamma * (u - s));
                if pa < 0.0 || pb < 0.0 || pa > newest || pb > newest {
                    continue;
                }
                let (a, b) = (ring.get(pa.round() as u64) as f64, ring.get(pb.round() as u64) as f64);
                ab += a * b;
                aa += a * a;
                bb += b * b;
            }
            if aa * bb <= 1e-20 {
                continue;
            }
            let score = ab / (aa * bb).sqrt() - OLC_OFFSET_PENALTY * e.abs() / t_ref;
            if best.is_none_or(|(_, b)| score > b) {
                best = Some((i, score));
            }
        }
        best.map(|b| b.0)
    }

    fn note_offset(&mut self, e: f64) {
        let st = &mut self.stats;
        if st.grains == 0 || e < st.e_min {
            st.e_min = e;
        }
        if st.grains == 0 || e > st.e_max {
            st.e_max = e;
        }
        st.e_last = e;
        st.e_sum += e;
        st.grains += 1;
    }

    fn push(&mut self, g: Grain) {
        if self.n == MAX_GRAINS {
            // Cannot happen with ≤ 3 overlapping grains; drop the oldest.
            self.grains.copy_within(1.., 0);
            self.n -= 1;
        }
        if let Some(slot) = self.grains.get_mut(self.n) {
            *slot = g;
            self.n += 1;
        }
    }

    /// Period of onset candidate mark `m` (logical index `i`): the measured
    /// spacing to the next mark of its run when that is known at `t + 1`
    /// (and plausible), else the mark's own period; and whether it is final.
    fn onset_period(&self, m: &crate::marks::Mark, i: usize, t: u64, marks: &MarkStore) -> (f64, bool) {
        let after = marks.get(i + 1).filter(|n| n.known_at <= t + 1 && n.run == m.run).map(|n| n.pos - m.pos);
        let p = after.filter(|&sp| sp > 0.5 * m.period && sp < 2.0 * m.period).unwrap_or(m.period);
        (p, after.is_some())
    }

    /// The research's short rise, `min(1 ms, T/4)` (R§8.15), at least one sample.
    fn short_rise(&self, p: f64) -> f64 {
        self.rise.min(0.25 * p).max(1.0)
    }

    /// Starts on the earliest known mark far enough ahead for the first
    /// grain's rise (R§8.18: s_0 = m_0, e = 0). β follows the control from
    /// the first grain: R§8.18 starts at β = 1 and "lets the correction curve
    /// move β", which the control law does (its onset snap is deliberate for
    /// hard-tune settings; slower retune speeds start from no correction). An
    /// extra two-period ratio ramp here delayed hard-tune onsets by up to
    /// ≈ 25 ms for an 82 Hz voice. After an engine switch from Classic the
    /// onset starts at Classic's content offset instead (`align`).
    fn try_onset(&mut self, u: f64, t: u64, cx: &Ctx) {
        let marks = cx.marks;
        let known = marks.known_len(t + 1);
        // Content offset of the first grain: 0, or the outgoing Classic
        // engine's offset at an engine switch (kept causal).
        let e0 = self.align.map_or(0.0, |e| e.min(self.d - self.hw - 1.0));
        let j = marks.lower_bound(u + e0, known);
        if self.onset_on_past_mark(u, e0, j, known, t, cx) {
            return;
        }
        // Earliest known mark whose grain (centred at `m − e0`) has its first
        // rise at or after `u` (a few candidates at most: the rise is at
        // most one period).
        let mut cand = None;
        for i in j..known.min(j + 4) {
            let Some(m) = marks.get(i) else { break };
            let (p, fin) = self.onset_period(m, i, t, marks);
            if !(p.is_finite() && p > 1.0) {
                continue;
            }
            let s = m.pos - e0;
            let h0 = p.min(p / cx.gamma);
            let avail = self.rise_budget(p);
            let r0 = if h0 <= avail { h0 } else { self.asym_rise(avail, h0, h0) };
            // γ < 1 reads the rise start later than its centre: keep it causal.
            let fits = |r: f64| e0 <= self.d - self.hw - 1.0 - ((1.0 - cx.gamma) * r).max(0.0);
            if s >= u + r0 && fits(r0) {
                cand = Some((*m, s, r0, p, fin));
                break;
            }
            // Otherwise start on this mark with the rise that still fits, if
            // that is at least the short rise (R§8.15), instead of waiting up
            // to a period for a fresher mark. With e = 0 the first grain
            // reads the dry samples themselves at γ = 1, so the shorter
            // cross-fade from the dry signal costs nothing there.
            if s - u >= self.short_rise(p) && fits(s - u) {
                cand = Some((*m, s, s - u, p, fin));
                break;
            }
        }
        let Some((m, s, r0, p, fin)) = cand else { return };
        self.start(&m, s, s - r0, r0, p, fin, e0, Phase::AwaitCentre, cx);
    }

    /// Onset on the newest known mark at or just before the content position
    /// (ON-4). Its grain is centred at `s_0 = m − e0 ≤ u + short rise`, partly
    /// in the past: with `e0 = 0` and γ = 1 it reads exactly the dry signal
    /// (`x(m + u − s_0) = x(u)`), so starting it is seamless and only its
    /// boundary to the next grain matters. That boundary, one synthesis
    /// period `T/β` after `s_0`, is the first place β acts, and it is decided
    /// lazily (as an asymmetric boundary) with its rise starting no earlier
    /// than now. Waiting instead for a mark ahead of `u` (the rule below) cost
    /// up to a period: in Live, below ≈ 250 Hz, the next mark is only known
    /// `d_mark` after it, so a low voice waited 7-10 ms (≈ one period at
    /// 82 Hz) between the control turning voiced and the first corrected
    /// period. Returns whether it started.
    fn onset_on_past_mark(&mut self, u: f64, e0: f64, j: usize, known: usize, t: u64, cx: &Ctx) -> bool {
        let marks = cx.marks;
        // The later of the last mark before `u + e0` and the first one after
        // it, if that one is still within a short rise of `u`.
        let mut pick = None;
        for i in [j.checked_sub(1), Some(j)].into_iter().flatten() {
            let Some(m) = marks.get(i).filter(|_| i < known) else { continue };
            let (p, fin) = self.onset_period(m, i, t, marks);
            if !(p.is_finite() && p > 1.0) {
                continue;
            }
            let s0 = m.pos - e0;
            if s0 < u + self.short_rise(p) {
                pick = Some((*m, s0, p, fin));
            }
        }
        let Some((m, s0, p, fin)) = pick else { return false };
        // Committing the first boundary a period early relies on the newest
        // period: only when the tracker measured it cleanly and agrees with
        // the detector (on real singing the first cycles after a seed are
        // often still unsettled, and an early start then spliced them; the
        // established start on the next mark is safer there).
        let agree = cx.period_hint > 0.0 && (1200.0 * (p / cx.period_hint).log2()).abs() <= ONSET_PAST_AGREE_CENTS;
        if m.score < ONSET_PAST_SCORE || !agree {
            return false;
        }
        // The rise from the lead-in: at γ = 1 and e0 = 0 the grain is the
        // lead-in itself, bit for bit (`m + (u − m) = u` exactly), so one
        // sample. At γ ≠ 1 the grain reads `(γ − 1)(u − s0)` away from the
        // lead-in, so a real crossfade: only while that is small.
        let gamma = cx.gamma;
        let r0 = if gamma == 1.0 && e0 == 0.0 { 1.0 } else { self.short_rise(p) };
        let lim = self.d - self.hw - 1.0;
        if ((gamma - 1.0) * (u - s0)).abs() > ONSET_MAX_SKEW * p || e0 + ((1.0 - gamma) * (s0 - u)).max(0.0) > lim {
            return false;
        }
        // The boundary to grain 1: the rise the scheduler would give it,
        // shortened so it starts after grain 0 has risen, and over by the
        // end of grain 0's causal extent.
        let ts = (p / cx.beta_target).max(MIN_TS);
        let ext = (p / gamma).min(if gamma > 1.0 { ((lim - e0) / (gamma - 1.0)).max(1.0) } else { f64::INFINITY });
        let hh = ts.min(ext);
        let avail = self.rise_budget(p);
        let normal = if hh <= avail { self.lazy_rise(hh) } else { self.asym_rise(avail, hh, hh) };
        let s1 = s0 + ts;
        let lb = normal.min(s1 - u - r0);
        // Much shorter than the scheduler's own crossfade it would splice
        // audibly (two copies a fraction of a period apart); then the next
        // mark is the better start.
        if lb.is_nan() || lb < self.short_rise(p).max(ONSET_MIN_RISE * normal) || s0 + ext - lb < u + r0 {
            return false;
        }
        self.stats.asymmetric += 1;
        self.start(&m, s0, u, r0, p, fin, e0, Phase::AsymPending { lb }, cx);
        true
    }

    /// Activates the engine with grain 0 on mark `m`, centred at `s`, rising
    /// over `[rise_start, rise_start + r0]` while the lead-in (the dry signal
    /// read at content offset `e0`) fades out.
    #[allow(clippy::too_many_arguments)]
    fn start(
        &mut self,
        m: &crate::marks::Mark,
        s: f64,
        rise_start: f64,
        r0: f64,
        p: f64,
        fin: bool,
        e0: f64,
        phase: Phase,
        cx: &Ctx,
    ) {
        self.active = true;
        self.stop = 0.0;
        self.lost = false;
        self.virt_head = false;
        self.n = 0;
        self.phase = phase;
        self.prev_src = Some((m.id, m.run));
        self.dry_lead = Some((rise_start, r0));
        self.lead_e = e0;
        self.align = None;
        self.stats.onsets += 1;
        self.note_offset(e0);
        self.stats.m_last = m.pos;
        self.stats.s_last = s;
        self.push(Grain {
            s,
            m: m.pos,
            gamma: cx.gamma,
            rise_start,
            rise_len: r0,
            fall_start: f64::INFINITY,
            fall_len: 0.0,
            src: Some((m.id, m.run)),
            period: p,
            period_final: fin,
        });
    }
}
