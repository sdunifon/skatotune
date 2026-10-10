//! Back end, R§7.6.2 F-L: note decision, error shaping, the retune filter on
//! the correction, Humanize, vibrato scaling, synthetic vibrato and the
//! output shift. Runs on the frame that is being output now.

use skato_core::{MidiMode, PitchFrame, Settings, ShiftControl, frame_flags};

use super::front::Frame;
use super::{Corrector, MEDIAN_CAP, STUDIO_ADVANCE_HOPS, STUDIO_ADVANCE_PERIODS, STUDIO_LATE_ONSET};
use crate::ControlOutput;
use crate::note_hmm::P_SWITCH;
use crate::rules::past_midpoint;
use crate::shape::{beyond_snap_max, flex_weight, snap_min};
use crate::targets::TargetSource;

/// Automatic (scale) correction is clamped to ±600 c (R§7.6.2 G).
const AUTO_CLAMP_C: f32 = 600.0;
/// MIDI targets may be further away: clamp and default Snap Max 1200 c.
const MIDI_CLAMP_C: f32 = 1200.0;
const SHIFT_CLAMP_C: f32 = 2400.0;
/// The transition window ends once the correction is this close to the error.
const TRANSITION_END_C: f32 = 5.0;
/// Humanize "stable" test (R§7.6.2 I).
const HUMANIZE_NEAR_C: f32 = 40.0;
const HUMANIZE_SLOPE_MAX: f32 = 200.0;
/// HQ/Max: abandon the current note when its posterior drops below this even
/// if no single other note has reached `P_SWITCH` (design choice).
const P_ABANDON: f32 = 0.05;

/// Back-end state (correction filter, current note, smoothed knobs).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BackState {
    pub(super) note: Option<usize>,
    /// Correction c, cents (the one-pole state).
    c: f32,
    trans: bool,
    t_note: u32,
    /// Humanize sustain gate g_s, 0-1.
    g_s: f32,
    /// Voiced hops waited in the onset grace.
    grace: u32,
    /// Studio: hops after an onset during which the onset note is kept.
    onset_hold: u32,
    /// Voiced hops the current note keeps its target although a scale change
    /// removed it (the key-change hold; 0 = none).
    pub(super) key_hold: u32,
    /// Voiced weight of the output shift (5 ms ramp to Transpose when unvoiced).
    u: f32,
    /// Last voiced shift before Transpose (c + vibrato scaling + LFO).
    s_core: f32,
    bend_c: f32,
    detune: f32,
    amount: f32,
    depth: f32,
    transpose: f32,
    formant: f32,
    lfo_depth: f32,
    pub(super) midi_armed: bool,
    pub(super) last_voiced: bool,
    was_pass: bool,
    /// The note has changed since the phrase onset (ends the ON-5 attack).
    pub(super) switched: bool,
    /// Follow singer's tuning: the grid offset latched when the current note
    /// started (its target keeps it for the whole note).
    pub(super) r_tgt: f32,
}

impl BackState {
    /// Jump the smoothed knobs to their targets (no glide at start-up).
    pub(super) fn prime(&mut self, s: &Settings) {
        self.amount = s.amount;
        self.depth = s.vibrato_depth;
        self.transpose = s.transpose_st;
        self.formant = s.formant_st;
        self.lfo_depth = s.synth_vibrato.depth_cents;
        self.detune = s.detune_cents;
    }

    /// Offset of the note grid: Detune + smoothed bend (R§7.6.1).
    #[inline]
    pub(super) fn offset(&self) -> f32 {
        self.detune + self.bend_c
    }
}

/// Outcome of the note decision for one hop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Change {
    Keep,
    /// First note of a phrase (onset rules apply, no transition window).
    Onset(usize),
    /// Target change (transition time constant applies).
    Switch(usize),
}

impl Corrector {
    /// Per-hop smoothing of continuous knobs and the MIDI bend.
    pub(super) fn smooth_params(&mut self) {
        let a = self.d.alpha_param;
        let s = &self.s;
        let be = &mut self.be;
        be.amount += a * (s.amount - be.amount);
        be.depth += a * (s.vibrato_depth - be.depth);
        be.transpose += a * (s.transpose_st - be.transpose);
        be.formant += a * (s.formant_st - be.formant);
        be.lfo_depth += a * (s.synth_vibrato.depth_cents - be.lfo_depth);
        be.detune = s.detune_cents;
        let bend_target = if s.midi_mode != MidiMode::Off { self.midi.bend() * s.bend_range_st * 100.0 } else { 0.0 };
        be.bend_c += self.d.alpha_bend * (bend_target - be.bend_c);
    }

    pub(super) fn back_end(&mut self, f: &Frame) -> ControlOutput {
        // Self-healing filter state: a non-finite correction would latch in
        // the one-pole (`c += α(e − c)` and `c *= 1 − α` keep a NaN forever)
        // and silently disable correction until reset (RT-1). Nothing should
        // produce one, but restart from 0 rather than trust that.
        if !self.be.c.is_finite() {
            self.be.c = 0.0;
        }
        if !self.be.s_core.is_finite() {
            self.be.s_core = 0.0;
        }
        let base = self.be.offset();
        // Follow singer's tuning: decisions on the singer's grid, and the
        // target on it too, with the offset the note started with. Not for
        // MIDI targets (the offset is 0 for them, and so is the target's).
        let follow = self.d.follow;
        let offset = if follow { base + f.r_dec } else { base };
        let scale_t = self.targets.source() == TargetSource::Scale;
        let t_off = if follow && scale_t { base + self.be.r_tgt } else { base };
        // A fresh onset also clears the note, in case a Studio lead change
        // skipped the frame that carried the forget.
        if f.forget || (f.onset && self.be.note.is_some()) {
            self.be.note = None;
            self.be.key_hold = 0;
            self.be.onset_hold = 0;
            self.be.trans = false;
            self.be.g_s = 0.0;
            self.rules.reset();
            self.pc_len = 0;
        }

        if !f.voiced {
            // B. Hold (30 ms) keeps c; then release c → 0 with τ_rel = 40 ms.
            // The output shift ramps to Transpose over 5 ms either way (L).
            if f.t_unv > self.d.hold_hops {
                self.be.c *= 1.0 - self.d.alpha_release;
                if self.be.c.abs() < 1e-6 {
                    self.be.c = 0.0;
                }
                self.be.g_s = 0.0;
            }
            self.be.u = (self.be.u - self.d.ramp_step).max(0.0);
            self.be.last_voiced = false;
            self.be.grace = 0;
            let s_total = self.compose(self.be.u * self.be.s_core);
            let target = self.be.note.map(|n| 100.0 * n as f32 + t_off);
            let mut flags = 0;
            if self.be.trans {
                flags |= frame_flags::TRANSITION;
            }
            if f.sibilant {
                flags |= frame_flags::SIBILANT;
            }
            return self.finish(f, s_total, target, flags);
        }

        let source = self.targets.source();
        let pass = source == TargetSource::PassThrough;
        let midi_t = source == TargetSource::Midi;
        let mut snap = false;
        if pass {
            // MIDI Target with nothing held: pass through untuned (c → 0, 50 ms).
            self.be.note = None;
            self.be.key_hold = 0;
            self.be.trans = false;
            self.be.was_pass = true;
            self.be.midi_armed = false;
        } else {
            match self.decide(f, offset) {
                Change::Keep => {}
                Change::Onset(n) => {
                    self.be.r_tgt = f.r_dec;
                    self.start_note(n, false);
                    snap = self.d.snap_onset;
                    self.be.switched = false;
                }
                Change::Switch(n) => {
                    self.be.r_tgt = f.r_dec;
                    self.start_note(n, true);
                    self.be.switched = true;
                }
            }
            self.be.was_pass = false;
            if self.be.midi_armed && self.be.note.is_some() {
                snap = true;
                self.be.midi_armed = false;
            }
        }

        // G. Error.
        let note = if pass { None } else { self.be.note };
        let t_off = if follow && scale_t { base + self.be.r_tgt } else { base };
        let target = note.map(|n| 100.0 * n as f32 + t_off);
        let bypass = note.is_some_and(|n| self.targets.is_bypass(n) && !midi_t);
        // SN-3: in Studio the shift reaches the output a fraction of a hop
        // plus a fraction of a period after the boundary that applies it
        // (the shifter's ratio ramp and per-period ratio sampling), so the
        // pitch that sets it is read from the frame that much newer; the
        // lookahead already holds it. Decisions stay on this frame.
        let fp = self.pitch_frame(f);
        let e = match note {
            Some(n) => self.error(&fp, n, t_off, midi_t),
            None => 0.0,
        };
        // A non-finite error is dropped (no correction this hop) rather than
        // fed to the one-pole (RT-1).
        let e = if e.is_finite() { e } else { 0.0 };

        // H/I. Retune one-pole on the correction (held by the octave guard).
        if !f.guard {
            let alpha = if pass {
                self.d.alpha_midi_release
            } else {
                let base = if self.be.trans { self.d.tau_tr_ms } else { self.d.tau_ms };
                self.clock.alpha(base + self.be.g_s * self.d.humanize_ms)
            };
            self.be.c += alpha * (e - self.be.c);
            if snap {
                self.be.c = e;
            }
            self.be.t_note = self.be.t_note.saturating_add(1);
            if self.be.trans && (self.be.t_note as f32 >= self.d.trans_end_hops || (e - self.be.c).abs() < TRANSITION_END_C) {
                self.be.trans = false;
            }
            self.humanize(f, target);
        }

        // J/K. Vibrato scaling (also on bypassed notes) and synthetic vibrato.
        let s_vib = if pass { 0.0 } else { (self.be.depth - 1.0) * fp.v };
        let s_lfo = if note.is_some() { self.lfo.step(&self.s.synth_vibrato, self.be.lfo_depth, self.clock.hop_s) } else { 0.0 };

        // L. Output. The voiced weight ramps in over 5 ms, except where the
        // correction snaps and the PSOLA renders: the shifter fades from the
        // dry signal under its first grain itself, so a second ramp only
        // delayed the full shift by 3-4 ms at every hard-tune onset (ON-9).
        // Classic (and Live-Min) splices need the ramp.
        self.be.s_core = self.be.c + s_vib + s_lfo;
        self.be.u = if self.d.natural && self.d.snap_onset && (snap || !self.be.last_voiced) {
            1.0
        } else {
            (self.be.u + self.d.ramp_step).min(1.0)
        };
        self.be.last_voiced = true;
        let s_total = self.compose(self.be.u * self.be.s_core);
        let mut flags = frame_flags::VOICED;
        if pass || bypass {
            flags |= frame_flags::BYPASSED;
        }
        if self.be.trans {
            flags |= frame_flags::TRANSITION;
        }
        if midi_t && note.is_some() {
            flags |= frame_flags::MIDI;
        }
        if f.w > 0.0 {
            flags |= frame_flags::VIBRATO;
        }
        if f.sibilant {
            flags |= frame_flags::SIBILANT;
        }
        self.finish(f, s_total, target, flags)
    }

    /// The Labs learned note hold applies to this hop: on, an HQ/Max tier,
    /// and scale targets (MIDI targets are the singer's intent already).
    fn note_hold(&self) -> bool {
        self.d.note_hold && self.targets.source() == TargetSource::Scale
    }

    /// F. Note decision for one voiced frame: the control law's rule, then
    /// (learned note hold) the learned veto on a proposed switch away from an
    /// allowed held note. The veto can only turn a switch into a Keep.
    fn decide(&mut self, f: &Frame, offset: f32) -> Change {
        let change = self.decide_rule(f, offset);
        let out = match change {
            Change::Switch(m) => match self.be.note {
                Some(cur)
                    if self.note_hold()
                        && cur != m
                        && self.targets.is_allowed(cur)
                        && !self.intent.keeps_switch(self.lead_now(), cur, m, offset, &self.targets) =>
                {
                    Change::Keep
                }
                _ => change,
            },
            _ => change,
        };
        #[cfg(test)]
        if let Some(log) = self.decisions.as_mut() {
            log.push((change, out));
        }
        out
    }

    /// The control law's note decision (the rule the learned veto judges).
    fn decide_rule(&mut self, f: &Frame, offset: f32) -> Change {
        let rel = f.p_dec - offset;
        let lead = self.lead_now();
        let hmm = self.d.hmm_on;
        let Some(cur) = self.be.note else {
            // Onset. Live: wait out the grace period unless the retune is
            // fast; Studio with lookahead: decide immediately from the future.
            let immediate = lead > 0 || self.be.midi_armed || self.be.was_pass || self.be.grace >= self.d.grace_hops;
            if !immediate {
                self.be.grace = self.be.grace.saturating_add(1);
                return Change::Keep;
            }
            let pick = if lead > 0 {
                // Studio onset (every tier): nearest to the median decision
                // pitch over the first 40 ms, applied from the first sample.
                // The Viterbi path is not used here: its attack states cannot
                // absorb a scoop of more than ~75 c, so it would start on the
                // scoop note.
                self.be.onset_hold = self.d.studio_onset_hops.min(lead) as u32;
                let m = if self.note_hold() {
                    // Learned note hold: the newest third of the visible
                    // onset window (where a scoop has landed).
                    let n = self.d.studio_onset_hops.min(lead);
                    let k = ((n as f32) * STUDIO_LATE_ONSET).round().max(1.0) as usize;
                    self.lookahead_median_from(k.min(lead), k.min(lead), offset)
                } else {
                    self.lookahead_median(self.d.studio_onset_hops, offset)
                };
                self.targets.nearest(m.unwrap_or(rel))
            } else if hmm {
                self.hmm.argmax(rel).map(|(n, _)| n).or_else(|| self.targets.nearest(rel))
            } else {
                self.targets.nearest(rel)
            };
            return match pick {
                Some(n) if self.be.was_pass => Change::Switch(n),
                Some(n) => Change::Onset(n),
                None => Change::Keep,
            };
        };

        if !self.targets.is_allowed(cur) {
            if self.be.key_hold > 0 {
                // A key change removed the note being sung: it keeps its
                // target until the next note starts or the hold runs out.
                self.be.key_hold -= 1;
                return Change::Keep;
            }
            // Scale or MIDI change removed the current note.
            let pick = if hmm { self.hmm.argmax(rel).map(|(n, _)| n) } else { None };
            return pick.or_else(|| self.targets.nearest(rel)).map_or(Change::Keep, Change::Switch);
        }
        if self.be.onset_hold > 0 {
            // Studio: the onset note already accounts for these frames.
            self.be.onset_hold -= 1;
            return Change::Keep;
        }
        if hmm {
            if lead > 0 { self.decide_hmm_studio(cur, lead, offset) } else { self.decide_hmm_live(cur, f, rel, offset) }
        } else {
            // Eco rules. Studio: the tolerance is judged on the median of the
            // next 30 ms (replacing that much dwell), applied at the crossing.
            let tol_c = self.s.tolerance_cents;
            let pick = if lead > 0 {
                let win = self.d.studio_switch_hops;
                let med = self.lookahead_median(win, offset).unwrap_or(rel);
                let tol_hops = self.d.tol_hops.saturating_sub(win.min(lead) as u32);
                // The median already rejects jitter, so the crossing test uses
                // the unsmoothed pitch (no 4 ms pre-smoother lag).
                let raw = if self.s.ignore_vibrato { f.ref_c } else { f.p };
                self.rules.decide_ahead(cur, raw - offset, med, &self.targets, tol_c, tol_hops)
            } else {
                self.rules.decide(cur, rel, &self.targets, tol_c, self.d.tol_hops)
            };
            pick.map_or(Change::Keep, Change::Switch)
        }
    }

    /// The unsmoothed decision pitch of a frame relative to the note grid
    /// (what the HMM is fed, see the front end).
    fn raw_rel(&self, f: &Frame, offset: f32) -> f32 {
        (if self.s.ignore_vibrato { f.ref_c } else { f.p }) - offset
    }

    /// HQ/Max: Waves' OR rule (R§7.6.2 F) as a gate on the note HMM's
    /// decisions (CL-2): a switch from `cur` to `m` is allowed only once the
    /// pitch is past the midpoint by more than T_c (cents part), or has been
    /// past it continuously for T_t (time part, `ahead` more hops known from
    /// the Studio lookahead), or simply past it when both are off. The HMM's
    /// evidence (capture regions, time slope, near-band slope) decides *when*
    /// within that, so the time tolerance can no longer switch a pitch that
    /// is still inside the cents band (it used to after a fraction of T_t).
    fn or_gate(&self, cur: usize, m: usize, raw: f32, ahead: u32) -> bool {
        let x = past_midpoint(cur, m, raw);
        let tc = self.s.tolerance_cents;
        let need = self.d.hq_dwell_hops;
        let cents_ok = tc > 0.0 && x > tc;
        let time_ok = need > 0 && x > 0.0 && self.rules.dwell_toward(m).saturating_add(ahead) >= need;
        let both_off = tc <= 0.0 && need == 0 && x > 0.0;
        cents_ok || time_ok || both_off
    }

    /// HQ/Max, Live: forward posterior, switch when a new note's mass
    /// exceeds `P_SWITCH` (or the current one's falls below `P_ABANDON`) and
    /// the OR-rule gate is open.
    fn decide_hmm_live(&mut self, cur: usize, f: &Frame, rel: f32, offset: f32) -> Change {
        let raw = self.raw_rel(f, offset);
        self.rules.observe(cur, raw, &self.targets);
        match self.hmm.argmax(rel) {
            Some((m, pm))
                if m != cur && (pm > P_SWITCH || self.hmm.prob(cur) < P_ABANDON) && self.or_gate(cur, m, raw, 0) =>
            {
                if self.release_freeze(cur, m) { Change::Keep } else { Change::Switch(m) }
            }
            _ => Change::Keep,
        }
    }

    /// Labs learned note hold: a downward switch waits while the level falls
    /// (`crate::release`). Counted when it holds a switch back.
    fn release_freeze(&mut self, cur: usize, m: usize) -> bool {
        let frozen = self.note_hold() && self.release.blocks(cur, m);
        if frozen {
            self.intent.stats.frozen += 1;
        }
        frozen
    }

    /// The frame whose pitch sets this hop's shift (SN-3): `f` itself in
    /// Live; in Studio the frame `round(STUDIO_ADVANCE_HOPS +
    /// STUDIO_ADVANCE_PERIODS·T / hop)` hops newer, when the lookahead holds
    /// it and it is a voiced, unguarded continuation of the same note.
    fn pitch_frame(&self, f: &Frame) -> Frame {
        let lead = self.lead_now();
        if lead == 0 || f.period <= 0.0 || !f.period.is_finite() {
            return *f;
        }
        let hop = self.cfg.hop.max(1) as f32;
        let adv = ((STUDIO_ADVANCE_HOPS * hop + STUDIO_ADVANCE_PERIODS * f.period) / hop).round();
        let adv = if adv >= 1.0 { (adv as usize).min(lead) } else { 0 };
        if adv == 0 {
            return *f;
        }
        let g = self.frame_at(lead - adv);
        if g.voiced && !g.onset && !g.guard && !g.forget {
            Frame { voiced: f.voiced, onset: f.onset, forget: f.forget, guard: f.guard, t_unv: f.t_unv, ..g }
        } else {
            *f
        }
    }

    /// HQ/Max, Studio with `lead` hops of lookahead: the fixed-lag Viterbi
    /// note, gated by the OR rule judged with the lookahead (R§7.6.2 F
    /// "Studio": the time tolerance's dwell counts the future hops too, so it
    /// costs only what the lookahead cannot see). The decision is evaluated
    /// `g = τ_tr/2` ahead of the frame being output (clamped to the
    /// lookahead), so a transition glide is centred on the singer's crossing
    /// (R§7.6.3 "Timing"); at τ_tr = 0 it lands on the crossing itself, and
    /// never before it: inside the band both notes are equally likely and the
    /// backtrace could otherwise place the switch anywhere in it, retuning the
    /// delayed audio before the singer moves.
    fn decide_hmm_studio(&mut self, cur: usize, lead: usize, offset: f32) -> Change {
        let g = self.d.glide_hops.min(lead);
        let age = lead - g;
        let fr = self.frame_at(age);
        if !fr.voiced {
            return Change::Keep;
        }
        let raw = self.raw_rel(&fr, offset);
        self.rules.observe(cur, raw, &self.targets);
        let Some(m) = self.hmm.viterbi_note(age).filter(|&m| m != cur && self.targets.is_allowed(m)) else {
            return Change::Keep;
        };
        // Hops after `fr` that stay past the midpoint toward `m`.
        let ahead = (0..age)
            .rev()
            .map(|a| self.frame_at(a))
            .take_while(|n| n.voiced && past_midpoint(cur, m, self.raw_rel(n, offset)) > 0.0)
            .count();
        // SN-1/SN-5: the crossing must also persist through the lookahead
        // it can see (up to STUDIO_PERSIST_MS); one that comes back within
        // it is a wobble, not a new note. Real singers wander across
        // midpoints within a note, and every such switch is a 100 c output
        // step (vocadito Studio HQ hard tune: warble episodes −31 %).
        let persists = ahead >= self.d.studio_persist_hops.min(age);
        if persists && self.or_gate(cur, m, raw, ahead as u32) && !self.release_freeze(cur, m) {
            Change::Switch(m)
        } else {
            Change::Keep
        }
    }

    fn start_note(&mut self, n: usize, transition: bool) {
        self.be.note = Some(n);
        self.be.key_hold = 0;
        self.be.trans = transition;
        self.be.t_note = 0;
        self.be.g_s = 0.0;
        self.be.grace = 0;
        self.lfo.note_on();
        self.rules.reset();
        self.pc_len = 0;
    }

    /// Median of the decision pitch (relative to the grid) over the delayed
    /// frame and up to `span` frames after it.
    fn lookahead_median(&mut self, span: usize, offset: f32) -> Option<f32> {
        self.lookahead_median_from(self.lead_now(), span, offset)
    }

    /// Median of the decision pitch over the frame `age` hops old and up to
    /// `span` newer frames.
    fn lookahead_median_from(&mut self, age: usize, span: usize, offset: f32) -> Option<f32> {
        let n = span.min(age);
        let mut k = 0;
        for j in 0..=n {
            let fr = self.frame_at(age - j);
            if fr.voiced && k < MEDIAN_CAP {
                self.scratch[k] = fr.p_dec - offset;
                k += 1;
            }
        }
        if k == 0 {
            return None;
        }
        let vals = self.scratch.get_mut(..k)?;
        let mid = k / 2;
        vals.select_nth_unstable_by(mid, f32::total_cmp);
        vals.get(mid).copied()
    }

    /// G. `e = w_flex · snap(Amount · (pitch(n*) + bend − ref))`, clamped.
    ///
    /// Vibrato Depth g_v (J) scales the sung vibrato *before* correction:
    /// the output is `p + c + (g_v − 1)·v`, i.e. the correction acts on the
    /// source `p' = ref_c + g_v·v`. With Preserve on the reference is the
    /// vibrato-free `ref_c`, so the output keeps `g_v·v` around the corrected
    /// centre. With Preserve off the reference must be `p'` itself, not the
    /// raw `p`: the retune filter then removes the *scaled* vibrato as fast as
    /// τ says (none of it survives at τ = 0, `g_v·HP_τ(v)` at τ > 0). Using
    /// `p` would cancel the original vibrato in `c` and then add `(g_v − 1)·v`
    /// on top, i.e. Depth 0 % would output a full-extent *inverted* vibrato
    /// (CL-1).
    fn error(&self, f: &Frame, n: usize, offset: f32, midi_t: bool) -> f32 {
        let target = 100.0 * n as f32 + offset;
        if self.targets.is_bypass(n) && !midi_t {
            return 0.0; // Bypass: legal note, never corrected (MIDI overrides).
        }
        let reference = if self.s.preserve_vibrato { f.ref_c } else { f.p + (self.be.depth - 1.0) * f.v };
        let e1 = self.be.amount * (target - reference);
        let mut e2 = snap_min(e1, self.s.snap_min_cents);
        let dev = f.p_dec - target;
        let snap_max = if self.s.snap_max_st > 0.0 {
            100.0 * self.s.snap_max_st
        } else if midi_t {
            MIDI_CLAMP_C
        } else {
            0.0
        };
        if beyond_snap_max(dev, snap_max) {
            e2 = 0.0;
        }
        let gap = self.targets.gap(n, if dev >= 0.0 { 1 } else { -1 });
        let clamp = if midi_t { MIDI_CLAMP_C } else { AUTO_CLAMP_C };
        (flex_weight(dev.abs(), gap, self.s.flex_tune) * e2).clamp(-clamp, clamp)
    }

    /// I. Humanize sustain gate: g_s rises over 150 ms once a note has been
    /// stable for 150 ms, and falls over 20 ms otherwise.
    fn humanize(&mut self, f: &Frame, target: Option<f32>) {
        let cap = self.pc_hist.len();
        if cap == 0 {
            return;
        }
        self.pc_head = (self.pc_head + 1) % cap;
        if let Some(slot) = self.pc_hist.get_mut(self.pc_head) {
            *slot = f.p_c;
        }
        self.pc_len = (self.pc_len + 1).min(cap);
        let slope = if self.pc_len >= 2 {
            let oldest = self.pc_hist.get((self.pc_head + cap - (self.pc_len - 1)) % cap).copied().unwrap_or(f.p_c);
            (f.p_c - oldest) / ((self.pc_len - 1) as f32 * self.clock.hop_s)
        } else {
            0.0
        };
        let stable = target.is_some_and(|t| (f.p_c - t).abs() < HUMANIZE_NEAR_C && slope.abs() < HUMANIZE_SLOPE_MAX);
        self.be.g_s = if stable && self.be.t_note > self.d.hum_sustain_hops {
            (self.be.g_s + self.d.hum_rise).min(1.0)
        } else {
            (self.be.g_s - self.d.hum_fall).max(0.0)
        };
    }

    /// Total shift: voiced part + Transpose, clamped to ±2400 c.
    fn compose(&self, voiced_part: f32) -> f32 {
        (100.0 * self.be.transpose + voiced_part).clamp(-SHIFT_CLAMP_C, SHIFT_CLAMP_C)
    }

    /// Test hook: corrupt the one-pole state (RT-1 self-healing test).
    #[cfg(test)]
    pub(crate) fn poison_filter_state_for_test(&mut self) {
        self.be.c = f32::NAN;
        self.be.s_core = f32::NAN;
    }

    fn finish(&self, f: &Frame, s_total: f32, target: Option<f32>, flags: u8) -> ControlOutput {
        let mut formant_ratio = (self.be.formant / 12.0).exp2();
        if self.s.formant_follows_transpose {
            formant_ratio *= (self.be.transpose / 12.0).exp2();
        }
        let vib = f.voiced && f.w > 0.0;
        ControlOutput {
            shift: ShiftControl {
                ratio: (s_total / 1200.0).exp2(),
                formant_ratio,
                voiced: f.voiced,
                period_hint: f.period,
                confidence: f.confidence,
            },
            shift_cents: s_total,
            frame: PitchFrame {
                sample_pos: 0,
                detected: if f.voiced { f.p } else { 0.0 },
                centre: if f.voiced { f.p_c } else { 0.0 },
                target: target.unwrap_or(0.0),
                output: if f.voiced { f.p + s_total } else { 0.0 },
                confidence: f.confidence,
                vib_rate: if vib { f.vib_rate } else { 0.0 },
                vib_extent: if vib { f.vib_extent } else { 0.0 },
                flags,
            },
        }
    }
}
