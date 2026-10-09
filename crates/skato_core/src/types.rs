//! Shared data types that flow between the DSP crates.
//!
//! This module is the *contract* between `skato_detect`, `skato_control`, `skato_shift`,
//! `skato_engine`, the plugin and the evaluation harness. Everything here is plain
//! `Copy` data so it can cross the audio thread without allocation.
//!
//! Pitch is expressed in **absolute cents**: `cents = 6900 + 1200·log2(f / 440)`,
//! i.e. MIDI note number × 100 (A4 = 6900). Sample positions are `u64` counts of
//! input samples since the engine was reset; fractional positions are `f64`.

/// Lowest-level structural choice: how much latency the plugin reports to the
/// host. Changing it changes the reported latency, so it is non-automatable and
/// forces the host to restart the plugin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum LatencyMode {
    /// ~1 ms. Classic resample + cycle-splice engine only (hard-tune character).
    LiveMin,
    /// ~3-5 ms depending on voice range. Causal PSOLA (formant-preserving).
    #[default]
    Live,
    /// Up to `EngineConfig::live_plus_ms` (5-10 ms). Lets PSOLA use symmetric
    /// two-period grains for more voices.
    LivePlus,
    /// `EngineConfig::studio_ms` (20-50 ms) of lookahead, DAW-compensated.
    Studio,
}

/// Voice-range preset. Sets the lowest/highest f0 the detector searches, which
/// in turn sets analysis windows and the PSOLA lookahead, so it is part of the
/// latency configuration (non-automatable).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum VoiceRange {
    LowMale,
    #[default]
    AltoTenor,
    Soprano,
    Wide,
}

impl VoiceRange {
    /// Lowest fundamental searched, in Hz.
    pub const fn f_min(self) -> f32 {
        match self {
            VoiceRange::LowMale => 55.0,
            VoiceRange::AltoTenor => 80.0,
            VoiceRange::Soprano => 140.0,
            VoiceRange::Wide => 55.0,
        }
    }

    /// Highest fundamental searched, in Hz.
    pub const fn f_max(self) -> f32 {
        match self {
            VoiceRange::LowMale => 600.0,
            VoiceRange::AltoTenor => 1000.0,
            VoiceRange::Soprano => 1400.0,
            VoiceRange::Wide => 1400.0,
        }
    }
}

/// Quality / CPU tier. Never changes the reported latency; it only decides how
/// much work the detector, note decision and shifter do inside the same budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum QualityTier {
    Eco,
    #[default]
    Hq,
    Max,
}

/// Which pitch-shifting engine renders the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum EngineKind {
    /// Resample + whole-cycle repeat/drop (Auto-Tune patent method). Formants
    /// move with pitch; the classic hard-tune sound.
    Classic,
    /// Pitch-synchronous overlap-add (PSOLA). Formant-preserving.
    #[default]
    Natural,
}

/// Per-pitch-class behaviour of a scale note.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum NoteState {
    /// A legal target: pitch near it is corrected to it.
    #[default]
    On,
    /// A legal note that is never corrected (sung pitch passes through), but it
    /// still "captures" pitch so neighbouring notes don't pull it.
    Bypass,
    /// Not a target at all: pitch is corrected to the nearest *other* legal note.
    Remove,
}

/// Built-in scales. Intervals are semitones above the key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum ScaleKind {
    #[default]
    Chromatic,
    Major,
    NaturalMinor,
    HarmonicMinor,
    MelodicMinor,
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Locrian,
    MajorPentatonic,
    MinorPentatonic,
    Blues,
}

impl ScaleKind {
    pub const ALL: [ScaleKind; 13] = [
        ScaleKind::Chromatic,
        ScaleKind::Major,
        ScaleKind::NaturalMinor,
        ScaleKind::HarmonicMinor,
        ScaleKind::MelodicMinor,
        ScaleKind::Dorian,
        ScaleKind::Phrygian,
        ScaleKind::Lydian,
        ScaleKind::Mixolydian,
        ScaleKind::Locrian,
        ScaleKind::MajorPentatonic,
        ScaleKind::MinorPentatonic,
        ScaleKind::Blues,
    ];

    /// Semitone offsets from the key that belong to the scale.
    pub const fn intervals(self) -> &'static [u8] {
        match self {
            ScaleKind::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            ScaleKind::Major => &[0, 2, 4, 5, 7, 9, 11],
            ScaleKind::NaturalMinor => &[0, 2, 3, 5, 7, 8, 10],
            ScaleKind::HarmonicMinor => &[0, 2, 3, 5, 7, 8, 11],
            ScaleKind::MelodicMinor => &[0, 2, 3, 5, 7, 9, 11],
            ScaleKind::Dorian => &[0, 2, 3, 5, 7, 9, 10],
            ScaleKind::Phrygian => &[0, 1, 3, 5, 7, 8, 10],
            ScaleKind::Lydian => &[0, 2, 4, 6, 7, 9, 11],
            ScaleKind::Mixolydian => &[0, 2, 4, 5, 7, 9, 10],
            ScaleKind::Locrian => &[0, 1, 3, 5, 6, 8, 10],
            ScaleKind::MajorPentatonic => &[0, 2, 4, 7, 9],
            ScaleKind::MinorPentatonic => &[0, 3, 5, 7, 10],
            ScaleKind::Blues => &[0, 3, 5, 6, 7, 10],
        }
    }

    /// Effective note states for every pitch class (index 0 = C) when this
    /// scale is played in `key` (0 = C .. 11 = B): scale notes are `On`,
    /// everything else `Remove`.
    pub fn note_states(self, key: u8) -> [NoteState; 12] {
        let mut states = [NoteState::Remove; 12];
        for &iv in self.intervals() {
            states[((key as usize) + iv as usize) % 12] = NoteState::On;
        }
        states
    }
}

/// How MIDI input chooses target notes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum MidiMode {
    /// MIDI is ignored.
    #[default]
    Off,
    /// Held notes are the only targets. With nothing held, audio passes untuned.
    Target,
    /// Each note-on adds its pitch class to the scale (the scale is cleared by
    /// the first note-on after entering this mode).
    Learn,
    /// Held notes are the targets while any is held, otherwise the scale.
    Auto,
}

/// Which input channel feeds the pitch analysis for stereo input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum AnalysisChannel {
    Left,
    Right,
    /// (L + R) / 2. Beware phase cancellation on stereo-miked sources.
    #[default]
    Mix,
}

/// Structural engine configuration. Anything here may change buffer sizes or the
/// reported latency, so it is only applied by building a new engine (in the
/// plugin's `activate()`), never on the audio thread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineConfig {
    pub sample_rate: f32,
    /// Largest host block the engine must accept in one `process()` call.
    pub max_block: usize,
    /// 1 (mono) or 2 (stereo). Both channels get identical pitch processing.
    pub channels: usize,
    pub latency_mode: LatencyMode,
    pub voice_range: VoiceRange,
    /// Reported latency for `LatencyMode::LivePlus`, 5-10 ms.
    pub live_plus_ms: f32,
    /// Reported latency for `LatencyMode::Studio`, 20-50 ms.
    pub studio_ms: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            max_block: 4096,
            channels: 2,
            latency_mode: LatencyMode::Live,
            voice_range: VoiceRange::AltoTenor,
            live_plus_ms: 10.0,
            studio_ms: 40.0,
        }
    }
}

/// Synthetic (LFO) vibrato added after correction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthVibrato {
    pub enabled: bool,
    /// 0.1-10 Hz.
    pub rate_hz: f32,
    /// Peak deviation in cents, 0-100.
    pub depth_cents: f32,
    /// Time after a note onset before the vibrato starts, ms.
    pub delay_ms: f32,
    /// Fade-in time once it starts, ms.
    pub ramp_ms: f32,
}

impl Default for SynthVibrato {
    fn default() -> Self {
        Self { enabled: false, rate_hz: 5.5, depth_cents: 30.0, delay_ms: 250.0, ramp_ms: 300.0 }
    }
}

/// Every runtime-adjustable setting, copied from the plugin parameters once per
/// host block. Realtime-safe to apply at any time (no allocation, no latency
/// change). Units are given per field; see `docs/DESIGN.md` §Control law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub tier: QualityTier,
    pub engine: EngineKind,

    // --- Targets -------------------------------------------------------------
    /// Key, 0 = C .. 11 = B.
    pub key: u8,
    pub scale: ScaleKind,
    /// Effective state of every pitch class (index 0 = C). The plugin resolves
    /// scale membership + per-note overrides into this array.
    pub note_states: [NoteState; 12],
    /// Global detune of every target, cents (±100).
    pub detune_cents: f32,

    // --- Correction behaviour -----------------------------------------------
    /// Retune speed: 63% time constant of the correction one-pole, ms (0-400).
    /// 0 = instant (hard-tune).
    pub retune_ms: f32,
    /// Time constant used right after the target note changes, ms (0-800).
    /// `None` = linked to `retune_ms`.
    pub transition_ms: Option<f32>,
    /// Humanize 0-100: slows correction on the sustained part of long notes.
    pub humanize: f32,
    /// Flex-tune 0-100: 0 corrects everything; higher values only correct pitch
    /// that is already close to the target.
    pub flex_tune: f32,
    /// Correction amount 0-1 (scales the correction error).
    pub amount: f32,
    /// Hysteresis past the midpoint between notes before switching, cents
    /// (0-40). 0 = off.
    pub tolerance_cents: f32,
    /// Time the pitch must stay past the midpoint before switching, ms
    /// (0-300). 0 = off. OR-combined with `tolerance_cents`.
    pub tolerance_ms: f32,
    /// Soft dead zone: errors below this many cents are mostly left alone (0-50).
    pub snap_min_cents: f32,
    /// Don't correct pitch farther than this from the target, semitones.
    /// 0 = off.
    pub snap_max_st: f32,

    // --- Vibrato ------------------------------------------------------------
    /// Correct the vibrato-free centre of the note and keep the sung vibrato
    /// (DVA-style). Off = classic Auto-Tune behaviour.
    pub preserve_vibrato: bool,
    /// Decide the target note from the vibrato-free centre.
    pub ignore_vibrato: bool,
    /// Scale of the sung vibrato, -1.0..2.0 (1.0 = unchanged, 0 = removed,
    /// negative = inverted).
    pub vibrato_depth: f32,
    pub synth_vibrato: SynthVibrato,

    // --- Output -------------------------------------------------------------
    /// Shift applied after correction, semitones (±12).
    pub transpose_st: f32,
    /// Formant shift, semitones (±12, cleanest within ±3).
    pub formant_st: f32,
    /// Also move formants by `transpose_st`.
    pub formant_follows_transpose: bool,

    // --- Detection ----------------------------------------------------------
    /// Tracking 0-100: 0 = relaxed (accepts breathy/noisy pitch), 100 = choosy.
    pub tracking: f32,
    /// Input level below which the signal counts as silence, dBFS.
    pub gate_db: f32,
    pub analysis_channel: AnalysisChannel,

    // --- MIDI ---------------------------------------------------------------
    pub midi_mode: MidiMode,
    /// MIDI targets apply in every octave (pitch-class matching).
    pub midi_all_octaves: bool,
    /// Pitch-bend range, semitones (0-12).
    pub bend_range_st: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            tier: QualityTier::Hq,
            engine: EngineKind::Natural,
            key: 0,
            scale: ScaleKind::Chromatic,
            note_states: [NoteState::On; 12],
            detune_cents: 0.0,
            retune_ms: 20.0,
            transition_ms: None,
            humanize: 0.0,
            flex_tune: 0.0,
            amount: 1.0,
            tolerance_cents: 8.0,
            tolerance_ms: 0.0,
            snap_min_cents: 0.0,
            snap_max_st: 0.0,
            preserve_vibrato: true,
            ignore_vibrato: true,
            vibrato_depth: 1.0,
            synth_vibrato: SynthVibrato::default(),
            transpose_st: 0.0,
            formant_st: 0.0,
            formant_follows_transpose: false,
            tracking: 50.0,
            gate_db: -55.0,
            analysis_channel: AnalysisChannel::Mix,
            midi_mode: MidiMode::Off,
            midi_all_octaves: false,
            bend_range_st: 2.0,
        }
    }
}

/// A MIDI event, already decoded by the plugin wrapper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MidiEvent {
    NoteOn { note: u8, velocity: f32 },
    NoteOff { note: u8 },
    /// Normalised pitch bend, -1.0..=1.0.
    PitchBend { value: f32 },
    /// Sustain pedal (CC64) down/up.
    Sustain { down: bool },
    /// CC120/CC123 or a host panic: release everything.
    AllNotesOff,
}

/// One pitch analysis result, produced by `skato_detect` once per control hop.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PitchEstimate {
    /// Input sample position the estimate describes (after accounting for the
    /// detector's own information lag, i.e. the centre of the evidence).
    pub sample_pos: u64,
    /// How many samples older than the newest analysed sample `sample_pos` is.
    pub info_lag: f32,
    /// Smoothed fundamental, Hz. 0.0 when unvoiced.
    pub f0_hz: f32,
    /// Same as `f0_hz` in absolute cents. 0.0 when unvoiced.
    pub cents: f32,
    /// Full-rate fractional period in samples (`sample_rate / f0_hz`), 0.0 when
    /// unvoiced. Seeds the PSOLA pitch-mark tracker.
    pub period: f32,
    /// Periodicity / clarity 0-1 of the chosen candidate.
    pub confidence: f32,
    /// Probability the frame is voiced, 0-1 (before the voicing state machine).
    pub voiced_prob: f32,
    /// Final voiced decision after gate, hysteresis and hold.
    pub voiced: bool,
    /// High-frequency/noisy frame (s, sh, t, breath).
    pub sibilant: bool,
    /// Short-term input level, dBFS.
    pub level_db: f32,
}

/// What the control law hands to the shifter for the next hop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShiftControl {
    /// Pitch ratio (output f0 / input f0) to reach by the end of the next hop.
    /// The shifter ramps linearly from the previous value.
    pub ratio: f32,
    /// Formant scale factor (1.0 = preserve).
    pub formant_ratio: f32,
    /// Whether the shifter should treat the input as pitched.
    pub voiced: bool,
    /// Detector's period estimate in input samples (0.0 if unknown).
    pub period_hint: f32,
    /// Detector confidence 0-1.
    pub confidence: f32,
}

impl Default for ShiftControl {
    fn default() -> Self {
        Self { ratio: 1.0, formant_ratio: 1.0, voiced: false, period_hint: 0.0, confidence: 0.0 }
    }
}

/// Bit flags carried in [`PitchFrame::flags`].
pub mod frame_flags {
    pub const VOICED: u8 = 1 << 0;
    /// The current target note is bypassed (pitch passes uncorrected).
    pub const BYPASSED: u8 = 1 << 1;
    /// A note transition is in progress (transition time constant active).
    pub const TRANSITION: u8 = 1 << 2;
    /// The target comes from a held MIDI note.
    pub const MIDI: u8 = 1 << 3;
    /// Vibrato is detected and the centre/vibrato split is engaged.
    pub const VIBRATO: u8 = 1 << 4;
    pub const SIBILANT: u8 = 1 << 5;
}

/// One GUI telemetry record, pushed once per control hop. Times are in *output*
/// samples so the graph lines up with what is heard.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PitchFrame {
    /// Output sample position this frame describes.
    pub sample_pos: u64,
    /// Detected input pitch, absolute cents (0.0 when unvoiced).
    pub detected: f32,
    /// Vibrato-free centre of the input pitch, cents.
    pub centre: f32,
    /// Target note pitch including detune and bend, cents (0.0 when none).
    pub target: f32,
    /// Output pitch after correction, vibrato scaling and transpose, cents.
    pub output: f32,
    /// Detector confidence 0-1.
    pub confidence: f32,
    /// Detected vibrato rate, Hz (0.0 when none).
    pub vib_rate: f32,
    /// Detected vibrato half-extent, cents (0.0 when none).
    pub vib_extent: f32,
    /// `frame_flags` bits.
    pub flags: u8,
}

/// Conversions between Hz and absolute cents (exact; see `crate::math` for
/// fast approximations used in hot loops).
pub fn hz_to_cents(hz: f32) -> f32 {
    6900.0 + 1200.0 * (hz / 440.0).log2()
}

pub fn cents_to_hz(cents: f32) -> f32 {
    440.0 * ((cents - 6900.0) / 1200.0).exp2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cents_round_trip() {
        for hz in [55.0_f32, 110.0, 261.6256, 440.0, 1400.0] {
            let c = hz_to_cents(hz);
            assert!((cents_to_hz(c) - hz).abs() / hz < 1e-5);
        }
        assert!((hz_to_cents(440.0) - 6900.0).abs() < 1e-3);
        assert!((hz_to_cents(261.6256) - 6000.0).abs() < 0.01);
    }

    #[test]
    fn scale_states() {
        let s = ScaleKind::Major.note_states(2); // D major: D E F# G A B C#
        let on: Vec<usize> = (0..12).filter(|&i| s[i] == NoteState::On).collect();
        assert_eq!(on, vec![1, 2, 4, 6, 7, 9, 11]);
    }
}
