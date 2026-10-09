//! Classic engine (L0): resample + whole-cycle repeat/drop, the Auto-Tune
//! patent method (US 5,973,252, expired; R§8.17).
//!
//! A fractional read pointer `p` advances by the pitch ratio r per output
//! sample, so the output is the input resampled by r (formants move with the
//! pitch: +1 st ≈ +5.9 %, the intended hard-tune character, R§7.5.6). The
//! pointer's delay `t − p` must stay inside a window:
//!
//! * shifting up (r > 1) the pointer gains on the input; when its delay falls
//!   below the floor it jumps back by a whole number of periods (a cycle is
//!   repeated),
//! * shifting down it falls behind; past `floor + J` it jumps forward by whole
//!   periods (cycles are dropped).
//!
//! The floor is D (Live-Min reports ≈ 1 ms; in Live/Live+ the engine is
//! padded to the mode's D, R§8.17 "Latency"), or `D − J/2` in Studio so the
//! mean content delay matches the compensated D. The jump `J = k·T` is the
//! smallest whole number of periods at least as long as the crossfade
//! (R§8.17), and, shifting up, at least 1.5× what the pointer gains during
//! the crossfade (otherwise a large ratio would need a new splice before the
//! last one finished).
//!
//! Tiers (R§7.4): Eco reads with the cubic and splices whole mark periods
//! under a short 0.75 ms equal-gain raised-cosine crossfade. (R§7.4's Eco is
//! the patent form with no crossfade at all; its hard splices clicked in the
//! evaluation suite, 6-22 per scenario, so every tier now fades.) HQ/Max read
//! with the 16-tap sinc (anti-aliased at rate r when shifting up) and splice
//! with a 1.25 ms crossfade (zita-at1 uses ≈ 1.3 ms); the jump length is
//! refined by a local NCC search over ±4 % of a period on the analysis
//! signal, so the splice lands on the matching phase even when the period
//! moved since it was measured. Crossfades are shortened when a large upward
//! ratio would otherwise let the outgoing pointer overtake the input, which
//! is what lets Live-Min stay at 1 ms; and if the ratio starts rising during
//! a fade (so the *incoming* pointer would overtake the input), the fade ends
//! early and the next splice starts at once.
//!
//! Unvoiced input (or no period) fades to the exact dry `x[t − D]`; a voiced
//! onset restarts the pointer at `t − D`, so ratio 1 is bit-exact.

use vt_core::ring::Ring;
use vt_core::simd;

use crate::psola::MAX_CH;
use crate::read::{RcTable, ReadStats, Reader};

/// HQ splice crossfade, ms (R§8.17: 1-1.5 ms).
const XFADE_MS: f64 = 1.25;
/// Eco splice crossfade, ms (short: Eco skips the NCC refinement, and the
/// fade only has to hide the residual waveform mismatch).
const ECO_XFADE_MS: f64 = 0.75;
/// Wet ↔ dry fade at voicing changes, ms.
const MIX_FADE_MS: f64 = 4.0;
/// Splice refinement range, fraction of a period.
const REFINE_FRAC: f64 = 0.04;
/// Splice refinement window before the splice point, ms.
const REFINE_WIN_MS: f64 = 1.0;

/// What the engine reads for one output sample.
pub(crate) struct ClassicCtx<'a> {
    pub rings: &'a [Ring],
    pub analysis: &'a Ring,
    pub reader: Reader<'a>,
    pub rc: &'a RcTable,
    /// Read-pointer speed (pitch ratio).
    pub ratio: f64,
    /// Largest ratio the current control ramp will reach (sizes crossfades
    /// so the outgoing pointer stays causal while the ratio still rises).
    pub ratio_bound: f64,
    pub voiced: bool,
    /// Local input period in samples, 0 if unknown.
    pub period: f64,
    /// HQ/Max: sinc reads, 1.25 ms crossfades and refined splices. Eco:
    /// cubic reads, 0.75 ms crossfades, unrefined whole mark periods.
    pub hq: bool,
}

#[derive(Clone, Copy, Debug)]
struct Xfade {
    /// Outgoing pointer.
    from: f64,
    i: u32,
    len: u32,
}

/// Diagnostics of the Classic engine.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ClassicStats {
    pub repeats: u64,
    pub drops: u64,
    pub resets: u64,
    pub read: ReadStats,
}

pub(crate) struct Classic {
    d: f64,
    d_int: i64,
    /// Smallest pointer delay that keeps every kernel causal (+1 guard).
    safe: f64,
    studio: bool,
    xf_len: f64,
    eco_xf_len: f64,
    mix_step: f32,
    refine_win: usize,
    active: bool,
    p: f64,
    xf: Option<Xfade>,
    mix: f32,
    last_period: f64,
    seg_a: Box<[f32]>,
    seg_b: Box<[f32]>,
    pub stats: ClassicStats,
}

impl Classic {
    /// Engine for latency `d` at `sr`; `hw` is the largest interpolation
    /// half-width; `studio` centres the delay window on D.
    pub fn new(sr: f64, d: usize, hw: usize, studio: bool) -> Self {
        let refine_win = (REFINE_WIN_MS * 1e-3 * sr).round().max(8.0) as usize;
        // Window + look-ahead part (≤ 4·crossfade) + refinement range.
        let seg = refine_win + (4.0 * XFADE_MS * 1e-3 * sr).ceil() as usize + 2 * 64 + 16;
        Self {
            d: d as f64,
            d_int: d as i64,
            safe: hw as f64 + 2.0,
            studio,
            xf_len: (XFADE_MS * 1e-3 * sr).round(),
            eco_xf_len: (ECO_XFADE_MS * 1e-3 * sr).round(),
            mix_step: (1.0 / (MIX_FADE_MS * 1e-3 * sr).max(1.0)) as f32,
            refine_win,
            active: false,
            p: 0.0,
            xf: None,
            mix: 0.0,
            last_period: 0.0,
            seg_a: vec![0.0; seg].into_boxed_slice(),
            seg_b: vec![0.0; seg].into_boxed_slice(),
            stats: ClassicStats::default(),
        }
    }

    pub fn reset(&mut self) {
        self.active = false;
        self.xf = None;
        self.mix = 0.0;
        self.last_period = 0.0;
    }

    /// Starts fully wet with the read pointer at input position `p` for
    /// output sample `t` (kept causal). Used when an engine switch fades
    /// from the PSOLA: starting at the dry position instead would cross-fade
    /// two copies a fraction of a period apart, which partly cancel (DSP-7).
    pub fn start_at(&mut self, p: f64, t: u64) {
        if !p.is_finite() {
            return;
        }
        self.active = true;
        self.p = p.min(t as f64 - self.safe);
        self.xf = None;
        self.mix = 1.0;
    }

    /// Input position read for output sample `t` (the dominant pointer of a
    /// splice crossfade; the dry position while mostly dry), or `None` while
    /// idle.
    pub fn content_pos(&self, t: u64, rc: &RcTable) -> Option<f64> {
        if !self.active {
            return None;
        }
        let p = match self.xf {
            Some(x) if rc.rc((x.i as f64 + 0.5) / x.len as f64) < 0.5 => x.from,
            _ => self.p,
        };
        Some(if self.mix >= 0.5 { p } else { t as f64 - self.d })
    }

    /// One output sample for every channel; `t` is the newest input index.
    pub fn tick(&mut self, t: u64, cx: &ClassicCtx, out: &mut [f32; MAX_CH]) {
        let nch = cx.rings.len().min(MAX_CH);
        let ui = t as i64 - self.d_int;
        let mut dry = [0.0f32; MAX_CH];
        if ui >= 0 {
            for (d, r) in dry.iter_mut().zip(cx.rings).take(nch) {
                *d = r.get(ui as u64);
            }
        }
        let period = if cx.period.is_finite() && cx.period > 2.0 { cx.period } else { 0.0 };
        if period > 0.0 {
            self.last_period = period;
        }
        let want = cx.voiced && period > 0.0;
        if !self.active {
            if !(want && ui >= 0) {
                *out = dry;
                return;
            }
            // Onset: start exactly at the dry position (ratio 1 is bit-exact).
            self.active = true;
            self.p = ui as f64;
            self.xf = None;
            self.mix = 0.0;
        }
        let r = if cx.ratio.is_finite() { cx.ratio.clamp(0.25, 4.0) } else { 1.0 };
        let rb = if cx.ratio_bound.is_finite() { cx.ratio_bound.clamp(r, 4.0) } else { r };
        // The incoming pointer gains (r − 1) per sample. A ratio that starts
        // rising during a crossfade (the next hop's control) can make it
        // overtake the input before the fade ends: end the fade and splice
        // now (a whole-period repeat of a periodic signal, so the fade's
        // remaining weight is the only discontinuity).
        if self.xf.is_some() && (t as f64 - self.p) < self.safe {
            self.xf = None;
        }
        if self.xf.is_none() {
            self.maybe_splice(t, r, rb, want, cx);
        }
        // Last resort (no period known, or no whole-period jump fits): restart
        // at the dry position without a fade rather than read unwritten input.
        if t as f64 - self.p < self.safe - 1.0 {
            self.xf = None;
            self.p = t as f64 - self.d;
            self.stats.resets += 1;
        }
        let hw = cx.reader.half_width();
        let rate = r as f32;
        let mut y = [0.0f32; MAX_CH];
        self.stats.read.note(self.p, hw, t);
        for (v, ring) in y.iter_mut().zip(cx.rings).take(nch) {
            *v = cx.reader.read(ring, self.p, rate);
        }
        // Safety net: a control change during a crossfade can speed the
        // outgoing pointer up beyond what the crossfade was sized for; end
        // the crossfade rather than read unwritten input.
        if self.xf.is_some_and(|x| t as f64 - x.from < self.safe) {
            self.xf = None;
        }
        if let Some(mut x) = self.xf {
            self.stats.read.note(x.from, hw, t);
            let w = cx.rc.rc((x.i as f64 + 0.5) / x.len as f64);
            for (v, ring) in y.iter_mut().zip(cx.rings).take(nch) {
                let old = cx.reader.read(ring, x.from, rate);
                *v = old + w * (*v - old);
            }
            x.from += r;
            x.i += 1;
            self.xf = (x.i < x.len).then_some(x);
        }
        self.p += r;
        self.mix = if want { (self.mix + self.mix_step).min(1.0) } else { (self.mix - self.mix_step).max(0.0) };
        if self.mix >= 1.0 {
            *out = y;
        } else if self.mix <= 0.0 {
            *out = dry;
            if !want {
                self.reset();
            }
        } else {
            let w = cx.rc.rc(self.mix as f64);
            for ch in 0..nch {
                out[ch] = dry[ch] + w * (y[ch] - dry[ch]);
            }
        }
    }

    /// Jumps the read pointer by whole periods when its delay leaves the
    /// window (and keeps it causal when no period is known).
    fn maybe_splice(&mut self, t: u64, r: f64, rb: f64, want: bool, cx: &ClassicCtx) {
        let delay = t as f64 - self.p;
        let period = if want { cx.period } else { self.last_period };
        let xn = if cx.hq { self.xf_len } else { self.eco_xf_len };
        if period <= 2.0 {
            // No period at all: only guard causality / runaway, by restarting
            // at the dry position.
            if delay < self.safe + (r - 1.0).max(0.0) || delay > self.d + 4096.0 {
                self.restart(t, xn, rb);
            }
            return;
        }
        // Whole periods covering the crossfade. Shifting up, the incoming
        // pointer also gains (r − 1) per sample during the crossfade, so the
        // jump must exceed that gain too or it would need another splice
        // right away.
        let span = if r > 1.0 { xn.max(1.5 * (rb - 1.0) * xn) } else { xn };
        let k = if span > 0.0 { (span / period).ceil().max(1.0) } else { 1.0 };
        let jump = k * period;
        let floor = if self.studio { (self.d - 0.5 * jump).max(self.safe + 1.0) } else { self.d.max(self.safe + 1.0) };
        if r > 1.0 && delay < floor {
            // Repeat cycle(s). The outgoing pointer keeps gaining (r − 1) per
            // sample during the crossfade, so shorten it to stay causal.
            let j = self.refine(cx, t, jump, period, true, rb * xn);
            let x = if xn > 0.0 { xn.min(((delay - self.safe) / (rb - 1.0)).floor()) } else { 0.0 };
            self.start_xfade(x);
            self.p -= j;
            self.stats.repeats += 1;
        } else if delay > floor + jump + if r < 1.0 { 0.0 } else { 2.0 * jump } {
            // Drop whole cycles back to the floor (shifting down; or a period
            // change left the pointer far behind), but no closer than what the
            // incoming pointer gains during the crossfade if the ratio is
            // ramping above 1.
            let room = delay - self.safe - 1.0 - (rb - 1.0).max(0.0) * xn;
            let n = ((delay - floor) / period).floor().min((room / period).floor());
            if n >= 1.0 {
                let j = self.refine(cx, t, n * period, period, false, rb * xn);
                if delay - j >= self.safe + 1.0 + (rb - 1.0).max(0.0) * xn {
                    self.start_xfade(xn);
                    self.p += j;
                    self.stats.drops += 1;
                }
            }
        }
    }

    fn start_xfade(&mut self, len: f64) {
        self.xf = (len >= 2.0).then_some(Xfade { from: self.p, i: 0, len: len as u32 });
    }

    /// Moves the pointer to the dry position `t − D` (crossfaded).
    fn restart(&mut self, t: u64, xn: f64, r: f64) {
        let delay = t as f64 - self.p;
        let x = if r > 1.0 { xn.min(((delay - self.safe) / (r - 1.0)).floor()) } else { xn };
        self.start_xfade(x.max(0.0));
        self.p = t as f64 - self.d;
        self.stats.resets += 1;
    }

    /// Jump length near `j` (whole periods) that best continues the waveform
    /// at the pointer, by NCC on the analysis signal (HQ/Max only).
    fn refine(&mut self, cx: &ClassicCtx, t: u64, j: f64, period: f64, back: bool, ahead: f64) -> f64 {
        if !cx.hq {
            return j;
        }
        let dlt = ((REFINE_FRAC * period).round() as i64).clamp(1, 64);
        let ip = self.p.round() as i64;
        let ji = j.round() as i64;
        let w = self.refine_win as i64;
        // Look-ahead part of the window: what the crossfade will play, as far
        // as it is already written (kernel-safe) for both pointers.
        let delay_new = if back { t as i64 - (ip - ji) } else { t as i64 - (ip + ji) };
        let delay_old = t as i64 - ip;
        let f = (ahead.ceil() as i64).min(delay_old.min(delay_new) - self.safe as i64 - dlt - 1).max(0);
        let len = (w + f) as usize;
        let b_len = len + 2 * dlt as usize + 2;
        let a_start = ip - w;
        let b_start = if back { ip - ji - dlt - 1 - w } else { ip + ji - dlt - 1 - w };
        if a_start < 0 || b_start < 0 || b_len > self.seg_b.len() || len > self.seg_a.len() {
            return j;
        }
        let (Some(a), Some(b)) = (self.seg_a.get_mut(..len), self.seg_b.get_mut(..b_len)) else { return j };
        cx.analysis.copy_to(a_start as u64, a);
        cx.analysis.copy_to(b_start as u64, b);
        let ea = simd::sum_sq(a) as f64;
        if ea <= 1e-12 {
            return j;
        }
        // Candidate c ∈ [0, 2·dlt + 2] corresponds to offset k = c − dlt − 1.
        let ncc = |c: usize| -> f32 {
            let Some(s) = b.get(c..c + len) else { return -1.0 };
            let eb = simd::sum_sq(s) as f64;
            if eb <= 1e-12 { 0.0 } else { (simd::dot(a, s) as f64 / (ea * eb).sqrt()) as f32 }
        };
        let mut best = (dlt as usize + 1, f32::MIN);
        for c in 1..=(2 * dlt as usize + 1) {
            let v = ncc(c);
            if v > best.1 {
                best = (c, v);
            }
        }
        let (c, v) = best;
        let (frac, _) = vt_core::math::parabolic_peak(ncc(c - 1), v, ncc(c + 1));
        let k = c as f64 - dlt as f64 - 1.0 + frac as f64;
        // Candidate c holds the material `ji − k` samples back (or `ji + k`
        // ahead) of the pointer.
        let refined = ji as f64 + if back { -k } else { k };
        // Keep the pointer's fractional phase: only the integer part of j was
        // rounded.
        let refined = refined + (j - ji as f64);
        if refined > 0.5 * j && refined < 1.5 * j { refined } else { j }
    }
}
