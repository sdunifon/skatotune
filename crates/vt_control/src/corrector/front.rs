//! Front end, R§7.6.2 A-E: everything that only *analyses* the input pitch.
//! Runs on the newest estimate every hop and produces a [`Frame`].

use vt_core::{LatencyMode, PitchEstimate, QualityTier, hz_to_cents};

use super::{ATTACK_MOVE_C, Corrector, ONSET_HOPS, RESTART_JUMP_C};
use crate::note_hmm::HmmObs;
use crate::rules::past_midpoint;

/// A pitch move larger than this is held back by the octave guard (R§7.6.2 B).
const OCTAVE_JUMP_C: f32 = 700.0;
/// Estimates within this of each other count as "consistent" for the guard
/// (design choice).
const OCTAVE_CONSISTENT_C: f32 = 100.0;

/// Analysis results for one hop, handed to the back end (possibly later).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Frame {
    /// FSM-confirmed voicing.
    pub voiced: bool,
    /// First voiced frame of a fresh note (after reset or forget, or a
    /// phrase restart).
    pub onset: bool,
    /// The note was forgotten at this frame (unvoiced > 100 ms).
    pub forget: bool,
    /// Hops unvoiced so far (0 when voiced).
    pub t_unv: u32,
    /// Octave guard holding: keep the correction as it is.
    pub guard: bool,
    /// Source pitch p (after the octave guard), cents.
    pub p: f32,
    /// Pre-smoothed analysis pitch p_s.
    pub p_s: f32,
    /// Analysis centre `w·centre + (1 − w)·p_s` (decision, Humanize, GUI).
    pub p_c: f32,
    /// Correction reference with vibrato preserved: `w·centre + (1 − w)·p`.
    pub ref_c: f32,
    /// Vibrato component `v = p − ref_c = w·(p − centre)`.
    pub v: f32,
    /// Decision pitch (centre when Ignore Vibrato, else p_s).
    pub p_dec: f32,
    pub w: f32,
    pub vib_rate: f32,
    pub vib_extent: f32,
    pub confidence: f32,
    pub period: f32,
    pub sibilant: bool,
}

/// Front-end state: voicing FSM, octave guard and pre-smoother.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct FrontState {
    /// Consecutive raw-voiced hops.
    run: u32,
    voiced: bool,
    t_unv: u32,
    /// A note has been sung since the last forget.
    has_note: bool,
    /// The vibrato split was dropped for the current gap (hold expired).
    released: bool,
    /// Last accepted pitch (octave guard reference).
    p_held: f32,
    guard_cand: f32,
    guard_count: u32,
    p_s: f32,
    /// Consecutive voiced hops the decision pitch has been past the current
    /// note's cents band (HQ/Max near-band persistence, CL-2).
    near_run: u32,
    /// Hops since the current phrase started (0: no phrase). Voiced runs
    /// separated by gaps no longer than the hold form one phrase.
    phrase_t: u32,
    /// HMM decision pitch (relative to the grid) at the phrase onset or
    /// restart, and voiced hops since then (ON-5 attack).
    onset_rel: f32,
    t_onset: u32,
}

impl Corrector {
    pub(super) fn front_end(&mut self, est: &PitchEstimate, _mark: Option<f32>) -> Frame {
        // Sanitise: the detector should give cents, but accept Hz alone and
        // treat NaN/Inf/negative as unvoiced. The correction refers to the
        // input at `sample_pos` (DESIGN §2.2), so in Live Max, whose `cents`
        // are extrapolated to the newest sample (vt_detect's R§8.10 lag
        // compensation), the pitch at `sample_pos` is taken from `f0_hz`.
        // The extrapolation predicts past the audio being corrected and adds
        // the slope estimate's noise (vocadito, Live Max: fine error p95
        // 32.6 c against 22.6 c in HQ); the mark pitch used to mask it.
        let f0_ok = est.f0_hz.is_finite() && est.f0_hz > 0.0;
        let extrapolated = self.s.tier == QualityTier::Max && self.cfg.latency_mode != LatencyMode::Studio;
        let p_det = if f0_ok && extrapolated {
            hz_to_cents(est.f0_hz)
        } else if est.cents.is_finite() && est.cents > 0.0 {
            est.cents
        } else if f0_ok {
            hz_to_cents(est.f0_hz)
        } else {
            f32::NAN
        };
        let raw_voiced = est.voiced && !est.sibilant && p_det.is_finite() && p_det > 0.0;

        // A. Pitch source: the detector alone. R§7.6.2 A's PSOLA mark pitch
        // is not used (SN-2). It is a single-cycle NCC lag with 2.3x the
        // detector's error on real singing (7.3 vs 3.3 c RMS against the
        // offline reference tracker on vocadito) and no lag advantage, so as
        // the correction reference it put each new cycle's jitter into the
        // ratio: vocadito Live HQ hard tune, output median 2.0 → 1.0 c and
        // settle median 39.1 → 31.1 ms without it. `mark_cents` stays in the
        // API (DESIGN §3.3) but is ignored.

        // B. Voicing FSM. The detector already applies its own gate and
        // hysteresis; here a 2-hop onset confirmation, hold/release/forget.
        self.fe.run = if raw_voiced { self.fe.run.saturating_add(1) } else { 0 };
        let gap = self.fe.t_unv;
        let voiced = if self.fe.voiced { raw_voiced } else { self.fe.run >= ONSET_HOPS };
        // Phrase restart (ON-1): a pitch jump early in a phrase, or across a
        // gap, starts a fresh note instead of a note change. Pre-onset false
        // voicing (breath, consonant or reverb resonances, a gross first
        // estimate) used to leave its note in place across the short gap
        // before the real note, which then waited for the HMM and the octave
        // guard while the stale note's error sat at the ±600 c clamp
        // (vocadito: a third of the Live hard-tune onset glitches).
        let restart = voiced
            && self.fe.has_note
            && (p_det - self.fe.p_held).abs() > RESTART_JUMP_C
            && (self.fe.phrase_t <= self.d.restart_window_hops || gap >= self.d.restart_gap_hops);
        let onset = voiced && (!self.fe.has_note || restart);
        if restart {
            self.hmm.reset();
            self.vib.reset();
            self.fe.guard_count = 0;
            self.fe.near_run = 0;
        }
        let mut forget = false;
        if voiced {
            self.fe.t_unv = 0;
            self.fe.released = false;
            self.fe.phrase_t = self.fe.phrase_t.saturating_add(1);
        } else {
            self.fe.t_unv = self.fe.t_unv.saturating_add(1);
            if self.fe.t_unv > self.d.hold_hops && !self.fe.released {
                // Past the hold: the vibrato component goes to 0.
                self.fe.released = true;
                self.vib.reset();
            }
            // The phrase ends with the hold; within it the clock runs on.
            self.fe.phrase_t = if self.fe.t_unv > self.d.hold_hops || self.fe.phrase_t == 0 {
                0
            } else {
                self.fe.phrase_t.saturating_add(1)
            };
            if self.fe.has_note && self.fe.t_unv > self.d.forget_hops {
                self.fe.has_note = false;
                forget = true;
                self.vib.reset();
                self.hmm.reset();
            }
        }
        self.fe.voiced = voiced;

        let mut f = Frame {
            voiced,
            onset,
            forget,
            t_unv: self.fe.t_unv,
            confidence: if est.confidence.is_finite() { est.confidence.clamp(0.0, 1.0) } else { 0.0 },
            period: if est.period.is_finite() { est.period.max(0.0) } else { 0.0 },
            sibilant: est.sibilant,
            ..Frame::default()
        };

        if !voiced {
            if self.d.hmm_on {
                self.fe.near_run = 0;
                self.hmm.step(&HmmObs { rel: 0.0, confidence: 0.0, voiced: false, near: false });
            }
            return f;
        }

        // Octave guard: a jump of more than 700 c is only accepted after
        // 20 ms of consistent estimates; until then hold p (and the back end
        // holds c). It applies within an established phrase, across gaps
        // shorter than the restart gap; earlier in a phrase, or after a
        // longer gap, such a jump restarts the note instead (above).
        let mut p = p_det;
        let mut restart = onset;
        if !onset && gap <= self.d.hold_hops && (p - self.fe.p_held).abs() > OCTAVE_JUMP_C {
            if self.fe.guard_count > 0 && (p - self.fe.guard_cand).abs() < OCTAVE_CONSISTENT_C {
                self.fe.guard_count += 1;
            } else {
                self.fe.guard_cand = p;
                self.fe.guard_count = 1;
            }
            if self.fe.guard_count >= self.d.octave_hops {
                self.fe.guard_count = 0;
                self.fe.p_held = p;
                restart = true;
            } else {
                p = self.fe.p_held;
                f.guard = true;
            }
        } else {
            self.fe.guard_count = 0;
            self.fe.p_held = p;
        }
        self.fe.has_note = true;

        // C. Pre-smoothing (4 ms) of the *analysis* pitch. It feeds the
        // centre/vibrato split and the note decision; the correction
        // reference uses p itself (see ref_c), so the retune filter alone
        // defines the step response and the residual-vibrato curve (R§7.6.2 H).
        if restart || !self.fe.p_s.is_finite() {
            self.fe.p_s = p;
            if !onset {
                self.vib.reset();
            }
        } else {
            self.fe.p_s += self.d.alpha_pre * (p - self.fe.p_s);
        }
        let p_s = self.fe.p_s;

        // D. Centre/vibrato split.
        let vo = self.vib.update(p_s);
        let w = vo.w;
        f.p = p;
        f.p_s = p_s;
        f.p_c = w * vo.centre + (1.0 - w) * p_s;
        f.ref_c = w * vo.centre + (1.0 - w) * p;
        f.v = p - f.ref_c;
        f.w = w;
        f.vib_rate = vo.rate_hz;
        f.vib_extent = vo.extent;

        // E. Decision pitch (+ note HMM evidence in HQ/Max).
        f.p_dec = if self.s.ignore_vibrato { f.p_c } else { p_s };
        if self.d.hmm_on {
            // The HMM integrates evidence over time itself, with a per-hop
            // cap so a one-hop outlier cannot flip the note. It is therefore
            // fed the *unsmoothed* decision pitch: pre-smoothing would smear a
            // one-hop glitch over several hops and defeat the cap (measured: a
            // single +600 c hop flipped the note through p_s, not through p).
            let raw = if self.s.ignore_vibrato { f.ref_c } else { p };
            let rel = raw - self.be.offset();
            // Near-band persistence (CL-2): the pitch has stayed past the
            // current note's cents band (x > T_c toward the nearest other
            // candidate) for NEAR_PERSIST_MS. No note yet: no hysteresis.
            // ON-5: in the attack of a phrase's first note (Live), a pitch
            // that has moved from where the phrase started toward the other
            // note (a scoop into the note) needs no persistence: the scoop
            // note is then left as soon as the HMM evidence allows.
            if onset {
                self.fe.onset_rel = rel;
                self.fe.t_onset = 0;
            } else {
                self.fe.t_onset = self.fe.t_onset.saturating_add(1);
            }
            let attack = self.fe.t_onset <= self.d.attack_hops && self.lead == 0 && !self.be.switched;
            let near = match self.be.note {
                None => true,
                Some(cur) => {
                    let (past, scoop) = match self.targets.nearest(rel) {
                        Some(k) if k != cur => {
                            let dir = if k > cur { 1.0 } else { -1.0 };
                            (
                                past_midpoint(cur, k, rel) > self.s.tolerance_cents,
                                attack && (rel - self.fe.onset_rel) * dir >= ATTACK_MOVE_C,
                            )
                        }
                        _ => (false, false),
                    };
                    self.fe.near_run = if past { self.fe.near_run.saturating_add(1) } else { 0 };
                    self.fe.near_run >= if scoop { 1 } else { self.d.near_persist_hops }
                }
            };
            let obs = HmmObs { rel, confidence: f.confidence, voiced: true, near };
            self.hmm.step(&obs);
        }
        f
    }
}
