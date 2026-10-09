//! Pitch-correction control law for Skatotune (DESIGN §3.3, R§7.6).
//!
//! Once per control hop (64 samples at 44.1/48 kHz) the engine hands the
//! [`Corrector`] the detector's [`PitchEstimate`] (and, in Live modes, the
//! PSOLA mark tracker's pitch, which is no longer used). The corrector decides which note
//! the singer is aiming for and how far to move the pitch, and returns a
//! [`ShiftControl`] (pitch ratio, formant ratio, voicing) for the shifter plus
//! a [`PitchFrame`] for the GUI. Pure control maths: no audio passes through.
//!
//! The per-hop pipeline follows R§7.6.2 step by step:
//!
//! | Step | What | Where |
//! |---|---|---|
//! | A | pitch source: the detector (the pitch at `sample_pos`; not the PSOLA mark pitch, SN-2) | [`Corrector`] front end |
//! | B | voicing FSM (2-hop onset, 30 ms hold, 40 ms release, 100 ms forget), phrase restarts (ON-1) and the 20 ms octave guard | front end |
//! | C | 4 ms pre-smoothing of the analysis pitch | front end |
//! | D | centre/vibrato split (extrema tracker; + AR(2) Kalman in HQ/Max) | `vibrato` |
//! | E | decision pitch and candidate notes (scale On/Bypass, MIDI held notes) | `targets` |
//! | F | note decision: rules (Eco), MonoNote-style HMM gated by the OR-tolerance (HQ/Max; no near-band persistence in a scoop into the first note, ON-5), fixed-lag in Studio (switches only on crossings that persist through the lookahead) | `rules`, `note_hmm` |
//! | G | error: Amount, Snap Min/Max, Flex-Tune, Bypass, clamp | `shape` |
//! | H/I | retune one-pole on the correction, transition τ, Humanize | back end |
//! | J/K | vibrato depth scaling, synthetic vibrato | back end, `lfo` |
//! | L | total shift S → ratio 2^(S/1200) (Studio: from the pitch the shifter will be playing, SN-3), unvoiced ramp to Transpose (none at PSOLA hard-tune onsets, ON-9) | back end |
//!
//! Conventions (R§7.6.1): pitch in absolute cents (MIDI × 100); every time
//! constant is converted to hops from the actual sample rate and hop and
//! re-derived whenever the settings change; `α(τ) = 1 − exp(−Δ/τ)`.
//!
//! Realtime: everything is allocated in [`Corrector::new`]; `set_settings`,
//! `midi` and `process_hop` never allocate, lock, log or panic, and every loop
//! is bounded (at most 128 notes × the jump band, or the lookahead length).

mod clock;
mod corrector;
mod lfo;
mod midi;
mod note_hmm;
mod rules;
mod shape;
mod targets;
mod vibrato;

#[cfg(test)]
mod tests;

pub use corrector::Corrector;
pub use skato_core::{LatencyMode, MidiEvent, PitchEstimate, PitchFrame, Settings, ShiftControl};

/// Static configuration of a [`Corrector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectorConfig {
    pub sample_rate: f32,
    pub hop: usize,
    pub latency_mode: LatencyMode,
    /// Studio: control hops of lookahead available to fixed-lag decisions (0 in Live modes).
    ///
    /// The corrector treats this as the reported latency in hops (`D / hop`)
    /// and measures how much of it is real lookahead from each estimate's
    /// `info_lag`: an estimate that describes the input `info_lag` samples
    /// before the newest sample leads the audio being output by
    /// `lookahead_hops − ceil(info_lag / hop)` hops. Decisions use that many
    /// hops of future estimates, and the control returned for each hop is the
    /// one for the frame that is being output (see `Corrector::process_hop`).
    /// With the detector centred on the output (`Alignment::Centered { delay: D }`)
    /// the lead would be 0 and Studio would decide causally, so `skato_engine`
    /// centres the detector `k` hops later than the output (`skato_engine::studio_split`)
    /// and passes `⌈D/hop⌉` (one less with its Studio pipeline), which makes
    /// the lead exactly `k` (`k − 1`). MIDI is delayed by `lookahead_hops`
    /// minus half the transition time so it lines up with the audio.
    pub lookahead_hops: usize,
}

/// Result of one control hop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlOutput {
    pub shift: ShiftControl,
    /// Total shift S in cents (correction + vibrato scaling + LFO + transpose).
    pub shift_cents: f32,
    /// Telemetry for this hop; `sample_pos` is left 0 (the engine fills it).
    pub frame: PitchFrame,
}
