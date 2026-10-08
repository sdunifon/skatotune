//! Synthetic telemetry for checking the GUI without audio (feature
//! `demo-telemetry`).
//!
//! A looping sung phrase with the things that make real vocals hard to read:
//! per-note detuning, scoops into notes (and one overshoot from above), slow
//! drift, delayed-onset vibrato, note-end falls and breaths between phrases.
//! The "corrected output" runs a small version of the R§7.6.2 control law on
//! top (nearest legal note from the current scale, one-pole retune on the
//! correction, vibrato kept around the corrected centre, transpose), using
//! the live parameter values, so turning Retune, Vibrato or the scale editor
//! visibly changes the demo.

use std::f32::consts::TAU;
use std::time::Instant;

use vt_core::{NoteState, PitchFrame, frame_flags};

use super::history::{PitchHistory, control_hop};
use crate::params::ParamValues;

/// One sung note of the loop.
#[derive(Clone, Copy, Debug)]
struct DemoNote {
    start: f64,
    dur: f64,
    /// Intended note in C major (shifted by the key at render time).
    midi: i32,
    /// How far off the singer is, cents.
    detune: f32,
    /// Scoop at the onset, cents (negative = from below).
    scoop: f32,
    vibrato: bool,
}

/// (note, duration s, breath after s, detune c, scoop c, vibrato)
const PHRASE: [(i32, f64, f64, f32, f32, bool); 15] = [
    (64, 0.45, 0.06, 18.0, -60.0, false),
    (67, 0.45, 0.06, -22.0, -80.0, false),
    (69, 1.30, 0.40, 12.0, -90.0, true),
    (67, 0.40, 0.05, -15.0, 45.0, false),
    (64, 0.40, 0.05, 25.0, -50.0, false),
    (62, 0.40, 0.05, -18.0, -40.0, false),
    (60, 1.40, 0.65, 10.0, -110.0, true),
    (62, 0.35, 0.05, -12.0, -45.0, false),
    (64, 0.35, 0.05, 20.0, -35.0, false),
    (65, 0.35, 0.05, -25.0, -40.0, false),
    (67, 0.95, 0.20, 8.0, -70.0, true),
    (72, 0.50, 0.05, -20.0, -150.0, false),
    (71, 0.40, 0.05, 15.0, -40.0, false),
    (69, 0.40, 0.05, -10.0, -35.0, false),
    (67, 1.70, 0.90, 14.0, -80.0, true),
];

/// Demo frame generator, driven by wall-clock time.
pub struct DemoSource {
    notes: Vec<DemoNote>,
    loop_s: f64,
    started: Option<Instant>,
    next_pos: u64,
    /// Correction state (cents) of the simulated control law.
    correction: f32,
    target_note: Option<i32>,
    target_changed_at: u64,
    unvoiced_since: Option<u64>,
    noise: u32,
}

impl Default for DemoSource {
    fn default() -> Self {
        let mut t = 0.4;
        let notes = PHRASE
            .iter()
            .map(|&(midi, dur, gap, detune, scoop, vibrato)| {
                let n = DemoNote {
                    start: t,
                    dur,
                    midi,
                    detune,
                    scoop,
                    vibrato,
                };
                t += dur + gap;
                n
            })
            .collect();
        Self {
            notes,
            loop_s: t + 0.3,
            started: None,
            next_pos: 0,
            correction: 0.0,
            target_note: None,
            target_changed_at: 0,
            unvoiced_since: Some(0),
            noise: 0x1234_5678,
        }
    }
}

/// The sung (input) pitch at loop time `t`: `(centre, vibrato, vib extent,
/// vibrato active, confidence)`, or `None` during breaths.
fn sung(note: &DemoNote, local: f64, t: f64, key_shift: i32) -> (f32, f32, f32, bool, f32) {
    let l = local as f32;
    let base = ((note.midi + key_shift) * 100) as f32 + note.detune;
    let scoop = note.scoop * (-l / 0.07).exp();
    let drift = 6.0 * (TAU * 0.7 * t as f32 + note.midi as f32).sin();
    let tail = (note.dur - local) as f32;
    let fall = if tail < 0.08 {
        -45.0 * (1.0 - tail / 0.08).powi(2)
    } else {
        0.0
    };
    let centre = base + scoop + drift + fall;
    let vib_start = 0.25;
    let (vib, extent, active) = if note.vibrato && l > vib_start {
        let extent = 38.0 * ((l - vib_start) / 0.3).min(1.0);
        let v = extent * (TAU * 5.6 * (l - vib_start)).sin();
        (v, extent, extent > 4.0)
    } else {
        (0.0, 0.0, false)
    };
    // Confidence dips at the onset (transient) and wobbles a little.
    let conf =
        (0.55 + 0.4 * (l / 0.03).min(1.0) + 0.04 * (TAU * 1.3 * t as f32).sin()).clamp(0.0, 1.0);
    (centre, vib, extent, active, conf)
}

/// Nearest legal note (On or Bypass) to `cents`; chromatic if none is legal.
fn nearest_legal(cents: f32, states: &[NoteState; 12]) -> i32 {
    let n0 = (cents / 100.0).round() as i32;
    let legal = |n: i32| {
        states
            .get(n.rem_euclid(12) as usize)
            .is_some_and(|s| *s != NoteState::Remove)
    };
    if !(0..12).any(legal) {
        return n0;
    }
    // Search outward (bounded: at most 6 semitones each way).
    (0..=6)
        .flat_map(|d| [n0 - d, n0 + d])
        .filter(|&n| legal(n))
        .min_by(|a, b| {
            let da = (cents - (*a * 100) as f32).abs();
            let db = (cents - (*b * 100) as f32).abs();
            da.total_cmp(&db)
        })
        .unwrap_or(n0)
}

impl DemoSource {
    /// Restart the loop at `now` (positions start again at 0).
    pub fn restart(&mut self, now: Instant) {
        self.started = Some(now);
        self.next_pos = 0;
        self.correction = 0.0;
        self.target_note = None;
        self.unvoiced_since = Some(0);
    }

    fn jitter(&mut self) -> f32 {
        // xorshift32: deterministic, allocation-free noise.
        self.noise ^= self.noise << 13;
        self.noise ^= self.noise >> 17;
        self.noise ^= self.noise << 5;
        (self.noise as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    /// Generate every frame up to wall-clock `now` into `out`. Returns the
    /// newest frame position, if any frame was produced.
    pub fn generate(
        &mut self,
        now: Instant,
        sample_rate: f32,
        values: &ParamValues,
        states: &[NoteState; 12],
        out: &mut PitchHistory,
    ) -> Option<u64> {
        let started = *self.started.get_or_insert(now);
        let sr = if sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        let hop = control_hop(sr);
        let until = (now.saturating_duration_since(started).as_secs_f64() * f64::from(sr)) as u64;
        let mut newest = None;
        // Bounded: after a stall, skip ahead rather than generating seconds of frames.
        if until.saturating_sub(self.next_pos) > 4096 * hop {
            self.next_pos = until.saturating_sub(4096 * hop);
        }
        while self.next_pos <= until {
            let pos = self.next_pos;
            let frame = self.frame_at(pos, sr, hop, values, states);
            out.push(frame);
            newest = Some(pos);
            self.next_pos += hop;
        }
        newest
    }

    fn frame_at(
        &mut self,
        pos: u64,
        sr: f32,
        hop: u64,
        v: &ParamValues,
        states: &[NoteState; 12],
    ) -> PitchFrame {
        let t = pos as f64 / f64::from(sr);
        let tl = t % self.loop_s;
        let key = i32::from(v.key.pitch_class());
        let key_shift = if key > 6 { key - 12 } else { key };
        let Some(note) = self
            .notes
            .iter()
            .find(|n| tl >= n.start && tl < n.start + n.dur)
            .copied()
        else {
            let since = *self.unvoiced_since.get_or_insert(pos);
            // Forget the note after 100 ms of silence (R§7.6.2 B).
            if pos.saturating_sub(since) as f32 > 0.1 * sr {
                self.correction = 0.0;
                self.target_note = None;
            }
            return PitchFrame {
                sample_pos: pos,
                ..PitchFrame::default()
            };
        };
        self.unvoiced_since = None;
        let (centre, vib, extent, vib_active, conf) = sung(&note, tl - note.start, t, key_shift);
        let detected = centre + vib + 1.5 * self.jitter();

        // Note decision on the vibrato-free pitch (Ignore Vibrato on).
        let decision = if v.ignore_vibrato { centre } else { detected };
        let n = nearest_legal(decision, states);
        if self.target_note != Some(n) {
            self.target_note = Some(n);
            self.target_changed_at = pos;
        }
        let target = (n * 100) as f32 + v.detune_cents;
        let bypassed = states.get(n.rem_euclid(12) as usize) == Some(&NoteState::Bypass);

        // One-pole retune on the correction (R§7.6.2 H).
        let reference = if v.preserve_vibrato { centre } else { detected };
        let error = if bypassed {
            0.0
        } else {
            (target - reference) * (v.amount_pct * 0.01).clamp(0.0, 1.0)
        };
        let since_change = pos.saturating_sub(self.target_changed_at) as f32 / sr * 1000.0;
        let tau_tr = if v.transition_link {
            v.retune_ms
        } else {
            v.transition_ms
        };
        let in_transition = since_change < 3.0 * tau_tr.max(1.0);
        let tau_ms = if in_transition { tau_tr } else { v.retune_ms };
        let alpha = if tau_ms <= 0.0 {
            1.0
        } else {
            1.0 - (-(hop as f32) / (tau_ms * 0.001 * sr)).exp()
        };
        self.correction += alpha * (error - self.correction);
        let depth = (v.vibrato_depth_pct * 0.01).clamp(-1.0, 2.0);
        let vib_scale = if v.preserve_vibrato {
            (depth - 1.0) * vib
        } else {
            0.0
        };
        let output = detected + self.correction + vib_scale + 100.0 * v.transpose_st.round();

        let mut flags = frame_flags::VOICED;
        if vib_active {
            flags |= frame_flags::VIBRATO;
        }
        if in_transition && tau_tr > 0.0 {
            flags |= frame_flags::TRANSITION;
        }
        if bypassed {
            flags |= frame_flags::BYPASSED;
        }
        PitchFrame {
            sample_pos: pos,
            detected,
            centre,
            target,
            output,
            confidence: conf,
            vib_rate: if vib_active { 5.6 } else { 0.0 },
            vib_extent: if vib_active { extent } else { 0.0 },
            flags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn generates_voiced_and_unvoiced_frames_in_order() {
        let mut demo = DemoSource::default();
        let mut h = PitchHistory::default();
        h.ensure_allocated();
        let t0 = Instant::now();
        demo.restart(t0);
        let states = [NoteState::On; 12];
        let v = ParamValues::default();
        let mut last = None;
        for k in 1..=300 {
            if let Some(p) = demo.generate(
                t0 + Duration::from_millis(k * 20),
                48_000.0,
                &v,
                &states,
                &mut h,
            ) {
                assert!(last.is_none_or(|l| p > l));
                last = Some(p);
            }
        }
        let voiced = (0..h.len())
            .filter(|&i| h.get(i).is_some_and(|f| f.flags & frame_flags::VOICED != 0))
            .count();
        assert!(voiced > h.len() / 2, "mostly sung: {voiced}/{}", h.len());
        assert!(voiced < h.len(), "with breaths");
        let vib = (0..h.len())
            .filter(|&i| {
                h.get(i)
                    .is_some_and(|f| f.flags & frame_flags::VIBRATO != 0)
            })
            .count();
        assert!(vib > 0);
    }

    #[test]
    fn hard_tune_output_sits_on_the_target() {
        let mut demo = DemoSource::default();
        let mut h = PitchHistory::default();
        h.ensure_allocated();
        let t0 = Instant::now();
        demo.restart(t0);
        let v = ParamValues {
            retune_ms: 0.0,
            vibrato_depth_pct: 0.0,
            ..ParamValues::default()
        };
        let states = [NoteState::On; 12];
        demo.generate(t0 + Duration::from_secs(6), 48_000.0, &v, &states, &mut h);
        let mut checked = 0;
        for i in 0..h.len() {
            let Some(f) = h.get(i) else { continue };
            if f.flags & frame_flags::VOICED != 0 {
                // Output = detected + (target - centre) + (0 - 1)·vibrato = target + jitter.
                assert!(
                    (f.output - f.target).abs() < 3.0,
                    "output {} target {}",
                    f.output,
                    f.target
                );
                checked += 1;
            }
        }
        assert!(checked > 1000);
    }

    #[test]
    fn removed_notes_are_never_targets() {
        let mut states = [NoteState::On; 12];
        states[9] = NoteState::Remove; // A
        let n = nearest_legal(6905.0, &states); // A4 + 5 c
        assert_ne!(n.rem_euclid(12), 9);
        assert!(n == 68 || n == 70);
        assert_eq!(nearest_legal(6905.0, &[NoteState::Remove; 12]), 69);
    }
}
