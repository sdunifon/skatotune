//! [`Corrector`]: per-hop orchestration of the control law.
//!
//! Each hop runs a *front end* (R§7.6.2 A-E: source select, voicing FSM,
//! octave guard, pre-smoothing, centre/vibrato split, decision pitch and the
//! note HMM) on the newest estimate, stores the result in a short frame ring,
//! and runs the *back end* (F-L: note decision, error shaping, retune filter,
//! vibrato scaling, LFO, output) on the frame that is being output now.
//!
//! In Live modes both ends see the same frame. In Studio, if the estimates
//! lead the audio (see [`CorrectorConfig::lookahead_hops`]), the back end
//! runs `lead` frames behind the front end, so fixed-lag decisions (Viterbi
//! backtrace, lookahead medians, onset notes) can use the frames in between.
//!
//! The Labs options hook in at three places: the front end records every hop
//! for the learned veto and the release freeze and feeds the singer tuning
//! tracker; the HQ/Max decisions consult the release freeze; and `decide`
//! hands every proposed switch away from a held note to the learned veto.

mod back;
mod front;

use skato_core::{
    EngineKind, LatencyMode, MidiEvent, MidiMode, NoteState, PitchEstimate, QualityTier, ReferenceMode, Settings, SynthVibrato,
};

use crate::clock::{HopClock, finite_or, sane};
use crate::intent::{NoteHoldStats, NoteIntent};
use crate::lfo::SynthLfo;
use crate::midi::MidiState;
use crate::note_hmm::{MAX_JUMP_HQ, MAX_JUMP_MAX, MAX_LAG, NoteHmm};
use crate::reftrack::RefTracker;
use crate::release::{self, ReleaseFreeze};
use crate::rules::RuleDecider;
use crate::targets::{TargetSet, TargetSource};
use crate::vibrato::VibratoTracker;
use crate::{ControlOutput, CorrectorConfig};

use back::BackState;
#[cfg(test)]
pub(crate) use back::Change;
use front::{Frame, FrontState};

// Timing constants of R§7.6.2 / R§7.6.4, in ms unless noted. Converted to
// hops in `derive()`.
const HOLD_MS: f32 = 30.0;
const RELEASE_MS: f32 = 40.0;
const FORGET_MS: f32 = 100.0;
const ONSET_GRACE_MS: f32 = 20.0;
/// Consecutive voiced hops needed for an onset (R§7.6.2 B).
const ONSET_HOPS: u32 = 2;
const OCTAVE_GUARD_MS: f32 = 20.0;
const PRESMOOTH_MS: f32 = 4.0;
const BEND_SMOOTH_MS: f32 = 5.0;
/// Smoothing of continuous knobs (Amount, Vibrato Depth, Transpose, Formant,
/// LFO depth) so automation and steps do not click. Time-constant knobs are
/// never smoothed (R§7.6.1). Design choice.
const PARAM_SMOOTH_MS: f32 = 10.0;
const UNVOICED_RAMP_MS: f32 = 5.0;
const MIDI_RELEASE_MS: f32 = 50.0;
const HUMANIZE_SUSTAIN_MS: f32 = 150.0;
const HUMANIZE_RISE_MS: f32 = 150.0;
const HUMANIZE_FALL_MS: f32 = 20.0;
const HUMANIZE_SLOPE_MS: f32 = 20.0;
const STUDIO_SWITCH_WINDOW_MS: f32 = 30.0;
const STUDIO_ONSET_WINDOW_MS: f32 = 40.0;
/// Onset snap (`c = e`) below this retune time, or with the Classic engine.
const SNAP_ONSET_TAU_MS: f32 = 10.0;
/// No onset grace at or below this retune time.
const GRACE_TAU_MS: f32 = 5.0;
/// HQ/Max: shortest dwell the time tolerance may open the OR-rule gate on,
/// ms. The dwell runs on the unsmoothed pitch, so this keeps 1-3 hop detector
/// glitches from switching the note however small the Tolerance time is (the
/// HMM's evidence cap does the same for the cents part).
const HQ_MIN_DWELL_MS: f32 = 5.0;
/// HQ/Max: how long the pitch must stay past the current note's cents band
/// before the HMM's near-band slope applies (CL-2). A held off-centre note
/// then switches about 20 ms later (1 c past the band; sooner further out)
/// instead of after 0.57 s. Brief excursions past the band, the main cause of
/// hard-tune warble on real singing (vocadito: flips shorter than 40 ms),
/// keep the quadratic tail's hysteresis: applying the slope at once raised
/// Live HQ hard-tune warble episodes by 37 % on vocadito.
const NEAR_PERSIST_MS: f32 = 25.0;
/// Phrase restart (ON-1): a voiced estimate that jumps more than this from
/// the last accepted pitch starts a fresh note (onset decision, no octave
/// guard, note HMM reset) when it comes early in a phrase or after a gap.
/// A 300 c hop-to-hop step is no scoop or glide; measured on vocadito (Live
/// HQ hard tune, chromatic) it took onset glitches from 14.5 % to 8.0 %.
const RESTART_JUMP_C: f32 = 300.0;
/// ... within this long after the phrase started (voiced runs joined across
/// gaps no longer than the 30 ms hold form one phrase),
const RESTART_WINDOW_MS: f32 = 60.0;
/// ... or after an unvoiced gap at least this long.
const RESTART_GAP_MS: f32 = 15.0;
/// Attack of a phrase's first note (ON-5, Live HQ/Max): for this long after
/// the onset, a pitch that has moved at least `ATTACK_MOVE_C` from where the
/// phrase started toward the next note (a scoop into the note) switches
/// without the near-band persistence. melody_male #0 (a slow −75 c scoop
/// ending 0-6 c past the band) settled in 31.6 ms with it, 15.6 ms without;
/// vocadito warble +1.8 % (chromatic). Later notes, and Studio (onset from
/// the lookahead median), keep the persistence.
const ATTACK_MS: f32 = 150.0;
/// A scale change that removes the note being sung keeps that note's target
/// until the next note starts, for at most this long, then glides to the new
/// scale with the transition (docs/research/host-key-scale.md §3.3: a key
/// change from MIDI Scale, Auto-Key or an edit never yanks a held note).
const KEY_CHANGE_HOLD_MS: f32 = 200.0;
const ATTACK_MOVE_C: f32 = 20.0;
/// Studio (SN-3): the shifter applies a new ratio some time after the
/// boundary that hands it over (its linear ramp over the next hop; PSOLA
/// fixes the ratio once per synthesis period), and Studio's content is
/// centred on D, so nothing cancels that lag: the pitch that sets the shift
/// is read `STUDIO_ADVANCE_HOPS·hop + STUDIO_ADVANCE_PERIODS·T` later in the
/// lookahead. Coupled to `skato_shift`'s ratio timing; measured on vocadito
/// (Studio HQ hard tune, chromatic, settle median / p90 ms):
///
/// | advance | skato_shift v1 (ratio at the head grain's centre) | skato_shift with ON-3 (ramp target, lazy commit) |
/// |---|---|---|
/// | none | 34.5 / 82.6 | 19.9 / 50.1 |
/// | 0.5·hop + 0.25·T (chosen) | 24.0 / 60.0 | 17.1 / 41.4 |
/// | hop + 0.5·T | 18.7 / 47.2 | 19.2 / 49.8 |
///
/// Re-measure if the shifter's ratio timing changes again.
const STUDIO_ADVANCE_HOPS: f32 = 0.5;
const STUDIO_ADVANCE_PERIODS: f32 = 0.25;
/// Studio (SN-1/SN-5): a note switch also needs the crossing to persist
/// through this much of the visible lookahead (all of it at the default
/// 40 ms: 11-21 ms for Alto/Tenor and Soprano).
const STUDIO_PERSIST_MS: f32 = 30.0;
/// Most voiced frames a lookahead median looks at.
const MEDIAN_CAP: usize = 64;
/// Learned note hold, Studio: the onset note is read from the newest third of
/// the visible onset window instead of all of it (where a scoop has landed;
/// note-intent prototypes B and C found this independently: onset glitches
/// on vocadito DEV 1.1 → 0.5 %, not confirmed on TEST).
const STUDIO_LATE_ONSET: f32 = 0.33;
/// Snap Min values below this (cents) are treated as 0 (off). Nothing audible
/// changes below a few hundredths of a cent, and an exponentially decaying
/// automation or modulation tail would otherwise hand the dead zone values
/// down to the subnormal range (RT-1). The plugin's Snap Max has the same kind
/// of jitter guard.
const SNAP_MIN_FLUSH_C: f32 = 0.05;

/// Settings-dependent constants, all in hop units (R§8.0: recomputed on any
/// settings change).
#[derive(Clone, Copy, Debug, Default)]
struct Derived {
    tau_ms: f32,
    tau_tr_ms: f32,
    /// Humanize addition at full sustain, `300 ms·(H/100)²` (R§7.6.2 I).
    humanize_ms: f32,
    trans_end_hops: f32,
    grace_hops: u32,
    snap_onset: bool,
    hold_hops: u32,
    forget_hops: u32,
    octave_hops: u32,
    alpha_release: f32,
    alpha_midi_release: f32,
    alpha_pre: f32,
    alpha_bend: f32,
    alpha_param: f32,
    ramp_step: f32,
    hum_sustain_hops: u32,
    hum_rise: f32,
    hum_fall: f32,
    tol_hops: u32,
    /// HQ/Max time-tolerance dwell (0 = Time off).
    hq_dwell_hops: u32,
    near_persist_hops: u32,
    hmm_on: bool,
    max_jump: usize,
    studio_switch_hops: usize,
    studio_onset_hops: usize,
    glide_hops: usize,
    restart_window_hops: u32,
    restart_gap_hops: u32,
    attack_hops: u32,
    /// Voiced hops a note removed by a scale change keeps its target.
    key_hold_hops: u32,
    studio_persist_hops: usize,
    /// The PSOLA engine renders the output (it fades in from the dry
    /// signal itself at a voiced onset, ON-9).
    natural: bool,
    /// Labs learned note hold in effect (on, and an HQ/Max tier).
    note_hold: bool,
    /// Labs follow singer's tuning.
    follow: bool,
}

fn derive(clock: &HopClock, s: &Settings, mode: LatencyMode) -> Derived {
    let tau = s.retune_ms;
    let tau_tr = s.transition_ms.unwrap_or(tau);
    let h = s.humanize / 100.0;
    Derived {
        tau_ms: tau,
        tau_tr_ms: tau_tr,
        humanize_ms: 300.0 * h * h,
        trans_end_hops: 3.0 * clock.hops(tau_tr),
        grace_hops: if tau <= GRACE_TAU_MS { 0 } else { clock.whole_hops(ONSET_GRACE_MS, 1) },
        snap_onset: tau <= SNAP_ONSET_TAU_MS || s.engine == EngineKind::Classic,
        hold_hops: clock.whole_hops(HOLD_MS, 1),
        forget_hops: clock.whole_hops(FORGET_MS, 2),
        octave_hops: clock.whole_hops(OCTAVE_GUARD_MS, 1),
        alpha_release: clock.alpha(RELEASE_MS),
        alpha_midi_release: clock.alpha(MIDI_RELEASE_MS),
        alpha_pre: clock.alpha(PRESMOOTH_MS),
        alpha_bend: clock.alpha(BEND_SMOOTH_MS),
        alpha_param: clock.alpha(PARAM_SMOOTH_MS),
        ramp_step: (clock.hop_ms / UNVOICED_RAMP_MS).min(1.0),
        hum_sustain_hops: clock.whole_hops(HUMANIZE_SUSTAIN_MS, 0),
        hum_rise: (clock.hop_ms / HUMANIZE_RISE_MS).min(1.0),
        hum_fall: (clock.hop_ms / HUMANIZE_FALL_MS).min(1.0),
        tol_hops: if s.tolerance_ms > 0.0 { clock.whole_hops(s.tolerance_ms, 1) } else { 0 },
        near_persist_hops: clock.whole_hops(NEAR_PERSIST_MS, 1),
        hq_dwell_hops: if s.tolerance_ms > 0.0 {
            clock.whole_hops(s.tolerance_ms, 1).max(clock.whole_hops(HQ_MIN_DWELL_MS, 1))
        } else {
            0
        },
        hmm_on: s.tier != QualityTier::Eco,
        max_jump: if s.tier == QualityTier::Max { MAX_JUMP_MAX } else { MAX_JUMP_HQ },
        studio_switch_hops: clock.whole_hops(STUDIO_SWITCH_WINDOW_MS, 0) as usize,
        studio_onset_hops: clock.whole_hops(STUDIO_ONSET_WINDOW_MS, 0) as usize,
        glide_hops: (0.5 * clock.hops(tau_tr)).round().max(0.0) as usize,
        restart_window_hops: clock.whole_hops(RESTART_WINDOW_MS, 0),
        restart_gap_hops: clock.whole_hops(RESTART_GAP_MS, 1),
        attack_hops: clock.whole_hops(ATTACK_MS, 0),
        key_hold_hops: clock.whole_hops(KEY_CHANGE_HOLD_MS, 0),
        studio_persist_hops: clock.whole_hops(STUDIO_PERSIST_MS, 0) as usize,
        natural: s.engine == EngineKind::Natural && mode != LatencyMode::LiveMin,
        note_hold: s.note_hold && s.tier != QualityTier::Eco,
        follow: s.reference == ReferenceMode::FollowSinger,
    }
}

/// Clamp every field to its documented range; NaN/Inf fall back to the
/// default, so nothing downstream ever sees a non-finite setting.
fn sanitize(s: &Settings) -> Settings {
    let d = Settings::default();
    let sv = &s.synth_vibrato;
    let dv = SynthVibrato::default();
    Settings {
        key: s.key % 12,
        detune_cents: sane(s.detune_cents, -100.0, 100.0, 0.0),
        retune_ms: sane(s.retune_ms, 0.0, 1000.0, d.retune_ms),
        transition_ms: s.transition_ms.map(|t| sane(t, 0.0, 1000.0, d.retune_ms)),
        humanize: sane(s.humanize, 0.0, 100.0, 0.0),
        flex_tune: sane(s.flex_tune, 0.0, 100.0, 0.0),
        amount: sane(s.amount, 0.0, 1.0, 1.0),
        tolerance_cents: sane(s.tolerance_cents, 0.0, 45.0, d.tolerance_cents),
        tolerance_ms: sane(s.tolerance_ms, 0.0, 1000.0, 0.0),
        snap_min_cents: flush_below(sane(s.snap_min_cents, 0.0, 100.0, 0.0), SNAP_MIN_FLUSH_C),
        snap_max_st: sane(s.snap_max_st, 0.0, 48.0, 0.0),
        vibrato_depth: sane(s.vibrato_depth, -1.0, 2.0, 1.0),
        synth_vibrato: SynthVibrato {
            enabled: sv.enabled,
            rate_hz: sane(sv.rate_hz, 0.1, 10.0, dv.rate_hz),
            depth_cents: sane(sv.depth_cents, 0.0, 100.0, dv.depth_cents),
            delay_ms: sane(sv.delay_ms, 0.0, 10_000.0, dv.delay_ms),
            ramp_ms: sane(sv.ramp_ms, 0.0, 10_000.0, dv.ramp_ms),
        },
        transpose_st: sane(s.transpose_st, -24.0, 24.0, 0.0),
        formant_st: sane(s.formant_st, -24.0, 24.0, 0.0),
        tracking: sane(s.tracking, 0.0, 100.0, d.tracking),
        gate_db: finite_or(s.gate_db, d.gate_db),
        bend_range_st: sane(s.bend_range_st, 0.0, 24.0, d.bend_range_st),
        ..*s
    }
}

/// `x`, or 0 when `x < floor` (a knob value too small to mean anything).
#[inline]
fn flush_below(x: f32, floor: f32) -> f32 {
    if x < floor { 0.0 } else { x }
}

/// The pitch-correction control law (see the crate docs).
pub struct Corrector {
    cfg: CorrectorConfig,
    clock: HopClock,
    /// Studio lookahead capacity in hops (0 in Live modes).
    studio_lag: usize,
    s: Settings,
    d: Derived,
    primed: bool,
    hop_index: u64,
    midi: MidiState,
    targets: TargetSet,
    targets_dirty: bool,
    /// The scale (note states) changed since the targets were last built:
    /// the next rebuild may start the key-change hold.
    scale_changed: bool,
    hmm: NoteHmm,
    vib: VibratoTracker,
    rules: RuleDecider,
    lfo: SynthLfo,
    fe: FrontState,
    be: BackState,
    ring: Vec<Frame>,
    ring_head: usize,
    ring_len: usize,
    lead: usize,
    /// Recent centre values for the Humanize slope (20 ms).
    pc_hist: Vec<f32>,
    pc_head: usize,
    pc_len: usize,
    scratch: [f32; MEDIAN_CAP],
    /// Learned note hold: the switch veto (history, features, model) and the
    /// release freeze's level ring. Both record every hop, so switching the
    /// option on mid-phrase finds the context in place.
    intent: NoteIntent,
    release: ReleaseFreeze,
    /// Follow singer's tuning (idle while the option is off).
    rt: RefTracker,
    /// Tests: every voiced hop's (rule decision, final decision), when on.
    #[cfg(test)]
    pub(crate) decisions: Option<Vec<(Change, Change)>>,
}

impl Corrector {
    pub fn new(cfg: CorrectorConfig) -> Self {
        let clock = HopClock::new(cfg.sample_rate, cfg.hop);
        let studio_lag = if cfg.latency_mode == LatencyMode::Studio { cfg.lookahead_hops.min(MAX_LAG) } else { 0 };
        let s = Settings::default();
        let d = derive(&clock, &s, cfg.latency_mode);
        let mut hmm = NoteHmm::new(clock.hop_ms, studio_lag);
        hmm.set_params(s.tolerance_cents, s.tolerance_ms, d.max_jump);
        let pc_len = clock.whole_hops(HUMANIZE_SLOPE_MS, 1) as usize + 1;
        let mut c = Self {
            cfg,
            clock,
            studio_lag,
            s,
            d,
            primed: false,
            hop_index: 0,
            midi: MidiState::new(),
            targets: TargetSet::default(),
            targets_dirty: true,
            scale_changed: false,
            hmm,
            vib: VibratoTracker::new(clock.hop_s, d.hmm_on),
            rules: RuleDecider::default(),
            lfo: SynthLfo::default(),
            fe: FrontState::default(),
            be: BackState::default(),
            ring: vec![Frame::default(); studio_lag + 1],
            ring_head: 0,
            ring_len: 0,
            lead: 0,
            pc_hist: vec![0.0; pc_len],
            pc_head: 0,
            pc_len: 0,
            scratch: [0.0; MEDIAN_CAP],
            intent: NoteIntent::new(clock.hop_ms, cfg.latency_mode),
            release: ReleaseFreeze::new(clock.whole_hops(release::WINDOW_MS, 1)),
            rt: RefTracker::new(clock.hop_s),
            #[cfg(test)]
            decisions: None,
        };
        c.be.prime(&c.s);
        c.rebuild_targets();
        c
    }

    /// Clear all signal state (notes, filters, MIDI). Settings are kept.
    pub fn reset(&mut self) {
        self.hop_index = 0;
        self.midi.reset();
        self.hmm.reset();
        self.vib.reset();
        self.rules.reset();
        self.lfo.reset();
        self.fe = FrontState::default();
        self.be = BackState::default();
        self.be.prime(&self.s);
        self.ring_head = 0;
        self.ring_len = 0;
        self.lead = 0;
        self.pc_head = 0;
        self.pc_len = 0;
        self.intent.reset();
        self.release.reset();
        self.rt.reset();
        self.targets_dirty = true;
        self.rebuild_targets();
    }

    /// Apply the current parameter values (call every block; RT-safe). All
    /// hop-unit constants are re-derived when anything changed.
    pub fn set_settings(&mut self, settings: &Settings) {
        let s = sanitize(settings);
        if self.primed && s == self.s {
            return;
        }
        let old = self.s;
        self.s = s;
        self.d = derive(&self.clock, &s, self.cfg.latency_mode);
        if s.midi_mode == MidiMode::Learn && old.midi_mode != MidiMode::Learn {
            self.midi.arm_learn();
        }
        if s.note_states != old.note_states || s.midi_mode != old.midi_mode || s.midi_all_octaves != old.midi_all_octaves {
            self.targets_dirty = true;
        }
        if self.primed && s.note_states != old.note_states {
            self.scale_changed = true;
        }
        self.hmm.set_params(s.tolerance_cents, s.tolerance_ms, self.d.max_jump);
        if s.reference != old.reference {
            // Follow singer's tuning starts (or stops) from A440; a note that
            // is already sounding keeps the A440 grid until the next one.
            self.rt.reset();
            self.be.r_tgt = 0.0;
        }
        if s.tier != old.tier {
            self.vib.set_kalman(self.d.hmm_on);
            if self.d.hmm_on && old.tier == QualityTier::Eco {
                // The HMM was idle in Eco: continue from the current note.
                match self.be.note {
                    Some(n) => self.hmm.seed(n),
                    None => self.hmm.reset(),
                }
            }
        }
        if !self.primed {
            self.be.prime(&s);
            self.primed = true;
        }
    }

    /// Feed a MIDI event (applied at the next hop; in Studio, delayed to line
    /// up with the delayed audio, R§7.6.3).
    pub fn midi(&mut self, ev: MidiEvent) {
        let delay = self.studio_lag.saturating_sub(self.d.glide_hops) as u64;
        let learn = self.s.midi_mode == MidiMode::Learn;
        self.midi.push(ev, self.hop_index + delay, self.hop_index, learn);
        if self.midi.take_changed() {
            self.targets_dirty = true;
        }
    }

    /// One control hop. `mark_cents` = instantaneous input pitch from the PSOLA mark tracker when
    /// it is locked (Live modes, R§7.6.2 A), else None. It is accepted for API stability and
    /// ignored: the correction follows the detector (SN-2, see the front end).
    pub fn process_hop(&mut self, est: &PitchEstimate, mark_cents: Option<f32>) -> ControlOutput {
        self.hop_index = self.hop_index.wrapping_add(1);
        let learn = self.s.midi_mode == MidiMode::Learn;
        self.midi.apply_due(self.hop_index, learn);
        if self.midi.take_changed() {
            self.targets_dirty = true;
        }
        if self.midi.take_note_on() && self.midi_targeting() && !self.be.last_voiced {
            // Note-on during an unvoiced gap: snap to it at the voiced onset.
            self.be.midi_armed = true;
        }
        if self.targets_dirty {
            self.rebuild_targets();
        }
        if std::mem::take(&mut self.scale_changed)
            && self.targets.source() == TargetSource::Scale
            && self.be.note.is_some_and(|n| !self.targets.is_allowed(n))
        {
            // A key change removed the note being sung: keep its target until
            // the next note starts, capped (KEY_CHANGE_HOLD_MS).
            self.be.key_hold = self.d.key_hold_hops;
        }
        self.smooth_params();
        self.update_lead(est);
        let frame = self.front_end(est, mark_cents);
        self.push_frame(frame);
        let delayed = self.frame_at(self.lead_now());
        self.back_end(&delayed)
    }

    /// Studio lookahead currently applied, in hops: the control (and the
    /// telemetry frame) returned by the latest `process_hop` describes the
    /// estimate passed this many hops earlier, so its output position is
    /// `est.sample_pos − lookahead_lead()·hop + D`. Always 0 in Live modes and
    /// when the detector is centred on the output.
    pub fn lookahead_lead(&self) -> usize {
        self.lead_now()
    }

    /// Effective scale after MIDI Learn edits (for the GUI keyboard), one state per pitch class.
    pub fn learned_states(&self) -> Option<[NoteState; 12]> {
        if self.s.midi_mode == MidiMode::Learn { self.midi.learned_states() } else { None }
    }

    /// What the learned note hold did since the last reset: switches the
    /// model judged, switches it held back, and downward switches the
    /// release freeze held back (diagnostics and evaluation).
    pub fn note_hold_stats(&self) -> NoteHoldStats {
        self.intent.stats
    }

    /// Offline tools (the learned veto's training data): the veto's features
    /// of the frame decided by the latest `process_hop`, as
    /// (decided pitch r in cents, the 58 features, lookahead hops used,
    /// segment age in hops or `u32::MAX`). None when that frame is unvoiced.
    /// Costs what a veto does; not for the audio thread.
    #[doc(hidden)]
    pub fn note_hold_features(&mut self) -> Option<(f32, [f32; crate::intent::FEATURES], usize, u32)> {
        let age = self.lead_now();
        self.intent.features_at(age)
    }

    /// Offset of the singer's tuning from A440 that follow singer's tuning
    /// applies to new decisions, cents (0 while the option is off or the
    /// targets are not a scale).
    pub fn singer_offset(&self) -> f32 {
        self.adaptive_offset()
    }

    /// Decision offset of the singer tuning tracker for this hop (0 unless
    /// the option is on and the targets come from a scale).
    fn adaptive_offset(&self) -> f32 {
        if self.d.follow && self.targets.source() == TargetSource::Scale { self.rt.offset() } else { 0.0 }
    }

    /// Tests: run as if the embedded model were missing or corrupt.
    #[cfg(test)]
    pub(crate) fn drop_model_for_test(&mut self) {
        self.intent = NoteIntent::without_model(self.clock.hop_ms);
    }

    /// MIDI notes currently decide the targets (Target, or Auto with notes held).
    fn midi_targeting(&self) -> bool {
        match self.s.midi_mode {
            MidiMode::Target => true,
            MidiMode::Auto => self.midi.any_held(),
            MidiMode::Off | MidiMode::Learn => false,
        }
    }

    fn rebuild_targets(&mut self) {
        if self.midi_targeting() {
            let held = self.midi.held_effective();
            self.targets.set_midi(&held, self.s.midi_all_octaves);
        } else {
            let states = match self.s.midi_mode {
                MidiMode::Learn => self.midi.learned_states().unwrap_or(self.s.note_states),
                _ => self.s.note_states,
            };
            self.targets.set_scale(&states);
        }
        self.hmm.set_targets(&self.targets);
        self.targets_dirty = false;
    }

    /// Studio: how many hops the newest estimate leads the output (latched
    /// only while unvoiced so a running note is never re-timed).
    fn update_lead(&mut self, est: &PitchEstimate) {
        if self.studio_lag == 0 {
            self.lead = 0;
            return;
        }
        if est.info_lag.is_finite() && est.info_lag > 0.0 {
            let lag_hops = (est.info_lag / self.cfg.hop.max(1) as f32).ceil();
            let lag_hops = if lag_hops < self.studio_lag as f32 { lag_hops as usize } else { self.studio_lag };
            let target = self.studio_lag - lag_hops;
            if target != self.lead && !self.be.last_voiced {
                self.lead = target;
            }
        }
    }

    /// Lookahead actually available this hop (the ring may still be filling).
    fn lead_now(&self) -> usize {
        self.lead.min(self.ring_len.saturating_sub(1))
    }

    fn push_frame(&mut self, f: Frame) {
        let cap = self.ring.len();
        if cap == 0 {
            return;
        }
        self.ring_head = (self.ring_head + 1) % cap;
        if let Some(slot) = self.ring.get_mut(self.ring_head) {
            *slot = f;
        }
        self.ring_len = (self.ring_len + 1).min(cap);
    }

    /// The frame `age` hops older than the newest (clamped to what exists).
    fn frame_at(&self, age: usize) -> Frame {
        let cap = self.ring.len().max(1);
        let age = age.min(self.ring_len.saturating_sub(1));
        let idx = (self.ring_head + cap - age) % cap;
        self.ring.get(idx).copied().unwrap_or_default()
    }
}
