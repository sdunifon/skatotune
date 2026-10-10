//! Replay the control law over a cached detector stream, hop by hop, with
//! the engine's settings sequence (`Engine::new` applies the defaults, a
//! render resets the engine and applies the run's settings). The telemetry
//! is bit-identical to an engine render's; the proxy output (input reference
//! track plus the applied shift) stands in for the shifter.

use skato_control::{Corrector, CorrectorConfig, NoteHoldStats};
use skato_core::{PitchFrame, Settings, frame_flags};

use super::cache::{EstHeader, EstStream};
use crate::reference::{PitchTrack, RefFrame};

/// One replayed hop.
#[derive(Clone, Copy, Debug)]
pub struct HopRec {
    /// Telemetry frame as the engine emits it (`sample_pos` filled in:
    /// `est.sample_pos − lead·hop + D`).
    pub frame: PitchFrame,
    pub shift_cents: f32,
    /// `ShiftControl::voiced`.
    pub voiced: bool,
    /// Target note (A440 grid index), None when there is no target.
    pub note: Option<i32>,
}

/// A replayed run of one file.
pub struct Run {
    pub h: EstHeader,
    pub settings: Settings,
    pub legal: [bool; 12],
    pub recs: Vec<HopRec>,
    /// What the learned note hold did.
    pub hold: NoteHoldStats,
}

/// Output sample at which control `k` is applied.
pub fn applied_at(h: &EstHeader, k: usize) -> u64 {
    ((k + 1 + usize::from(h.pipelined)) * h.hop as usize) as u64
}

/// Input time (s) of the start of the content control `k` is applied to.
pub fn content_time(h: &EstHeader, k: usize) -> f64 {
    (applied_at(h, k) as f64 - f64::from(h.latency)) / f64::from(h.sample_rate)
}

/// Target note of a telemetry frame on the A440 grid (`detune` removed).
fn target_note(frame: &PitchFrame, detune: f32) -> Option<i32> {
    let has = frame.flags & frame_flags::VOICED != 0 && frame.target.is_finite() && frame.target > 0.0;
    has.then(|| ((frame.target - detune) / 100.0).round() as i32)
}

/// Replay `settings` over `stream`.
pub fn replay(stream: &EstStream, settings: &Settings) -> Run {
    let h = stream.h;
    let hop = h.hop as usize;
    let mut c = Corrector::new(CorrectorConfig {
        sample_rate: h.sample_rate,
        hop,
        latency_mode: h.mode,
        lookahead_hops: h.lookahead_hops as usize,
    });
    c.set_settings(&Settings::default());
    c.reset();
    c.set_settings(settings);
    let mut recs = Vec::with_capacity(stream.est.len());
    for est in &stream.est {
        let o = c.process_hop(est, None);
        let lead = (c.lookahead_lead() as u64).saturating_mul(hop as u64);
        let mut frame = o.frame;
        frame.sample_pos = est.sample_pos.saturating_sub(lead).saturating_add(u64::from(h.latency));
        recs.push(HopRec { frame, shift_cents: o.shift_cents, voiced: o.shift.voiced, note: target_note(&frame, settings.detune_cents) });
    }
    Run { h, settings: *settings, legal: crate::scenarios::legal_classes(settings), recs, hold: c.note_hold_stats() }
}

impl Run {
    pub fn hop_s(&self) -> f64 {
        f64::from(self.h.hop) / f64::from(self.h.sample_rate)
    }

    /// Input time (s) of the middle of control `k`'s content.
    pub fn t_mid(&self, k: usize) -> f64 {
        content_time(&self.h, k) + 0.5 * self.hop_s()
    }

    /// The control index whose content covers input time `t` (s).
    pub fn k_at(&self, t: f64) -> Option<usize> {
        let o = t * f64::from(self.h.sample_rate) + f64::from(self.h.latency);
        let k = (o / f64::from(self.h.hop)).floor() as i64 - 1 - i64::from(self.h.pipelined);
        (k >= 0 && (k as usize) < self.recs.len()).then_some(k as usize)
    }

    /// Telemetry as a render collects it: one frame per applied control (the
    /// Studio pipeline never applies the last analysis' control).
    pub fn telemetry(&self) -> Vec<PitchFrame> {
        let n = self.recs.len().saturating_sub(usize::from(self.h.pipelined));
        self.recs[..n].iter().map(|r| r.frame).collect()
    }

    /// Shift (cents) acting on output sample `o`: control k is applied at
    /// `applied_at(k)` and ramps linearly from the previous control over one hop.
    pub fn shift_at_output(&self, o: f64) -> f64 {
        let hop = f64::from(self.h.hop);
        let k = (o / hop).floor() as i64 - 1 - i64::from(self.h.pipelined);
        if k < 0 || self.recs.is_empty() {
            return 0.0;
        }
        let k = (k as usize).min(self.recs.len() - 1);
        let a = applied_at(&self.h, k) as f64;
        let cur = f64::from(self.recs[k].shift_cents);
        let prev = if k > 0 { f64::from(self.recs[k - 1].shift_cents) } else { 0.0 };
        let j = (o - a).clamp(0.0, hop - 1.0);
        prev + (cur - prev) * (j + 1.0) / hop
    }

    /// Shift acting on input sample position `pos`.
    pub fn shift_at_input(&self, pos: f64) -> f64 {
        self.shift_at_output(pos + f64::from(self.h.latency))
    }

    /// Proxy for the corrected output's pitch track: the input track plus the
    /// shift acting on each frame's content (no shifter artefacts).
    pub fn proxy_track(&self, input: &PitchTrack) -> PitchTrack {
        let frames = input
            .frames
            .iter()
            .map(|f| {
                let s = self.shift_at_input(f.pos as f64);
                RefFrame { cents: if f.voiced { f.cents + s } else { f.cents }, ..*f }
            })
            .collect();
        PitchTrack { sample_rate: input.sample_rate, hop: input.hop, frames }
    }

    /// Control `k` is voiced and has a target.
    pub fn target_voiced(&self, k: usize) -> bool {
        let f = &self.recs[k].frame;
        f.flags & frame_flags::VOICED != 0 && self.recs[k].note.is_some()
    }
}
