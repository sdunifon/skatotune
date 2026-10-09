//! Plugin parameters and their mapping onto the engine's [`Settings`].
//!
//! There are three layers here, deliberately kept apart:
//!
//! 1. [`SkatotuneParams`]: the nice-plug parameter objects the host sees. Their
//!    `#[id]`s are part of every saved project and preset, so they are stable
//!    from day one and must never be renamed (DESIGN §3.6).
//! 2. [`ParamValues`]: a plain `Copy` snapshot of every parameter's current
//!    value, read once per host block with [`SkatotuneParams::values`]. Reading
//!    is a handful of atomic loads; nothing allocates.
//! 3. [`ParamValues::to_settings`] / [`ParamValues::structural`]: pure
//!    functions that resolve the snapshot into the engine's runtime
//!    [`Settings`] and the structural (latency-defining) configuration.
//!
//! Keeping the resolution logic on the plain snapshot makes it testable
//! without a host: note-override resolution, the transition link, "Snap Max
//! = off" and the unit conversions (percent → fraction, ...) are all covered by
//! unit tests and by `tests/settings_mapping.rs`.
//!
//! Defaults come from R§7.6.4 and match `skato_core::Settings::default()`; one
//! [`ParamValues::default`] feeds both the parameter objects and the tests so
//! they cannot drift apart.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

use nice_plug::prelude::*;
use skato_core::{
    AnalysisChannel, EngineConfig, EngineKind, LatencyMode, MidiMode, NoteState, QualityTier,
    ScaleKind, Settings, SynthVibrato, VoiceRange,
};

use crate::labs::LabsSettings;

/// Note names used for keys, note overrides and the GUI (sharps only: the
/// default egui fonts have no `♯`/`♭` glyphs).
pub const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

// ---------------------------------------------------------------------------
// Choice enums
//
// nice-plug's `Enum` derive cannot be implemented for the skato_core types
// (orphan rule), so each choice gets a local mirror with stable variant ids
// and a lossless conversion.
// ---------------------------------------------------------------------------

/// Latency mode (structural, non-automatable). Mirrors [`LatencyMode`].
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyModeChoice {
    #[id = "live-min"]
    #[name = "Live-Min"]
    LiveMin,
    #[id = "live"]
    #[name = "Live"]
    Live,
    #[id = "live-plus"]
    #[name = "Live+"]
    LivePlus,
    #[id = "studio"]
    #[name = "Studio"]
    Studio,
}

impl From<LatencyModeChoice> for LatencyMode {
    fn from(c: LatencyModeChoice) -> Self {
        match c {
            LatencyModeChoice::LiveMin => LatencyMode::LiveMin,
            LatencyModeChoice::Live => LatencyMode::Live,
            LatencyModeChoice::LivePlus => LatencyMode::LivePlus,
            LatencyModeChoice::Studio => LatencyMode::Studio,
        }
    }
}

/// Voice range (structural, non-automatable). Mirrors [`VoiceRange`].
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceRangeChoice {
    #[id = "low-male"]
    #[name = "Low Male"]
    LowMale,
    #[id = "alto-tenor"]
    #[name = "Alto/Tenor"]
    AltoTenor,
    #[id = "soprano"]
    #[name = "Soprano"]
    Soprano,
    #[id = "wide"]
    #[name = "Wide"]
    Wide,
}

impl From<VoiceRangeChoice> for VoiceRange {
    fn from(c: VoiceRangeChoice) -> Self {
        match c {
            VoiceRangeChoice::LowMale => VoiceRange::LowMale,
            VoiceRangeChoice::AltoTenor => VoiceRange::AltoTenor,
            VoiceRangeChoice::Soprano => VoiceRange::Soprano,
            VoiceRangeChoice::Wide => VoiceRange::Wide,
        }
    }
}

/// Quality tier. Mirrors [`QualityTier`].
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierChoice {
    #[id = "eco"]
    #[name = "Eco"]
    Eco,
    #[id = "hq"]
    #[name = "HQ"]
    Hq,
    #[id = "max"]
    #[name = "Max"]
    Max,
}

impl From<TierChoice> for QualityTier {
    fn from(c: TierChoice) -> Self {
        match c {
            TierChoice::Eco => QualityTier::Eco,
            TierChoice::Hq => QualityTier::Hq,
            TierChoice::Max => QualityTier::Max,
        }
    }
}

/// Shifting engine. Mirrors [`EngineKind`].
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineChoice {
    #[id = "classic"]
    #[name = "Classic"]
    Classic,
    #[id = "natural"]
    #[name = "Natural"]
    Natural,
}

impl From<EngineChoice> for EngineKind {
    fn from(c: EngineChoice) -> Self {
        match c {
            EngineChoice::Classic => EngineKind::Classic,
            EngineChoice::Natural => EngineKind::Natural,
        }
    }
}

/// Key (tonic) of the scale, C .. B.
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyChoice {
    #[id = "c"]
    #[name = "C"]
    C,
    #[id = "c#"]
    #[name = "C#"]
    Cs,
    #[id = "d"]
    #[name = "D"]
    D,
    #[id = "d#"]
    #[name = "D#"]
    Ds,
    #[id = "e"]
    #[name = "E"]
    E,
    #[id = "f"]
    #[name = "F"]
    F,
    #[id = "f#"]
    #[name = "F#"]
    Fs,
    #[id = "g"]
    #[name = "G"]
    G,
    #[id = "g#"]
    #[name = "G#"]
    Gs,
    #[id = "a"]
    #[name = "A"]
    A,
    #[id = "a#"]
    #[name = "A#"]
    As,
    #[id = "b"]
    #[name = "B"]
    B,
}

impl KeyChoice {
    /// Pitch class, 0 = C .. 11 = B.
    pub fn pitch_class(self) -> u8 {
        // Declaration order is chromatic, so the variant index is the pitch class.
        self.to_index() as u8
    }
}

/// Built-in scale. Mirrors [`ScaleKind`].
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleChoice {
    #[id = "chromatic"]
    #[name = "Chromatic"]
    Chromatic,
    #[id = "major"]
    #[name = "Major"]
    Major,
    #[id = "natural-minor"]
    #[name = "Natural Minor"]
    NaturalMinor,
    #[id = "harmonic-minor"]
    #[name = "Harmonic Minor"]
    HarmonicMinor,
    #[id = "melodic-minor"]
    #[name = "Melodic Minor"]
    MelodicMinor,
    #[id = "dorian"]
    #[name = "Dorian"]
    Dorian,
    #[id = "phrygian"]
    #[name = "Phrygian"]
    Phrygian,
    #[id = "lydian"]
    #[name = "Lydian"]
    Lydian,
    #[id = "mixolydian"]
    #[name = "Mixolydian"]
    Mixolydian,
    #[id = "locrian"]
    #[name = "Locrian"]
    Locrian,
    #[id = "major-pentatonic"]
    #[name = "Major Pentatonic"]
    MajorPentatonic,
    #[id = "minor-pentatonic"]
    #[name = "Minor Pentatonic"]
    MinorPentatonic,
    #[id = "blues"]
    #[name = "Blues"]
    Blues,
}

impl From<ScaleChoice> for ScaleKind {
    fn from(c: ScaleChoice) -> Self {
        match c {
            ScaleChoice::Chromatic => ScaleKind::Chromatic,
            ScaleChoice::Major => ScaleKind::Major,
            ScaleChoice::NaturalMinor => ScaleKind::NaturalMinor,
            ScaleChoice::HarmonicMinor => ScaleKind::HarmonicMinor,
            ScaleChoice::MelodicMinor => ScaleKind::MelodicMinor,
            ScaleChoice::Dorian => ScaleKind::Dorian,
            ScaleChoice::Phrygian => ScaleKind::Phrygian,
            ScaleChoice::Lydian => ScaleKind::Lydian,
            ScaleChoice::Mixolydian => ScaleKind::Mixolydian,
            ScaleChoice::Locrian => ScaleKind::Locrian,
            ScaleChoice::MajorPentatonic => ScaleKind::MajorPentatonic,
            ScaleChoice::MinorPentatonic => ScaleKind::MinorPentatonic,
            ScaleChoice::Blues => ScaleKind::Blues,
        }
    }
}

/// Per-pitch-class override. `Scale` defers to scale membership (On when the
/// pitch class is in the scale, Remove otherwise); the others force a state.
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NoteOverride {
    #[default]
    #[id = "scale"]
    #[name = "Scale"]
    Scale,
    #[id = "on"]
    #[name = "On"]
    On,
    #[id = "bypass"]
    #[name = "Bypass"]
    Bypass,
    #[id = "remove"]
    #[name = "Remove"]
    Remove,
}

impl NoteOverride {
    /// The GUI keyboard's click cycle: Scale → On → Bypass → Remove → Scale.
    pub fn next(self) -> Self {
        match self {
            NoteOverride::Scale => NoteOverride::On,
            NoteOverride::On => NoteOverride::Bypass,
            NoteOverride::Bypass => NoteOverride::Remove,
            NoteOverride::Remove => NoteOverride::Scale,
        }
    }
}

/// Analysis input for stereo signals. Mirrors [`AnalysisChannel`].
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisChoice {
    #[id = "left"]
    #[name = "Left"]
    Left,
    #[id = "right"]
    #[name = "Right"]
    Right,
    #[id = "mix"]
    #[name = "Mix (L+R)"]
    Mix,
}

impl From<AnalysisChoice> for AnalysisChannel {
    fn from(c: AnalysisChoice) -> Self {
        match c {
            AnalysisChoice::Left => AnalysisChannel::Left,
            AnalysisChoice::Right => AnalysisChannel::Right,
            AnalysisChoice::Mix => AnalysisChannel::Mix,
        }
    }
}

/// MIDI target mode. Mirrors [`MidiMode`].
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidiModeChoice {
    #[id = "off"]
    #[name = "Off"]
    Off,
    #[id = "target"]
    #[name = "Target"]
    Target,
    #[id = "learn"]
    #[name = "Learn"]
    Learn,
    #[id = "auto"]
    #[name = "Auto"]
    Auto,
}

impl From<MidiModeChoice> for MidiMode {
    fn from(c: MidiModeChoice) -> Self {
        match c {
            MidiModeChoice::Off => MidiMode::Off,
            MidiModeChoice::Target => MidiMode::Target,
            MidiModeChoice::Learn => MidiMode::Learn,
            MidiModeChoice::Auto => MidiMode::Auto,
        }
    }
}

// ---------------------------------------------------------------------------
// Plain snapshot
// ---------------------------------------------------------------------------

/// Plain snapshot of every parameter value, in the parameters' own units.
/// `Copy`, so taking one per host block costs nothing but the atomic loads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamValues {
    // Structural (changing these rebuilds the engine and changes latency).
    pub latency_mode: LatencyModeChoice,
    pub voice_range: VoiceRangeChoice,
    /// Live+ latency, ms (5-10).
    pub live_plus_ms: f32,
    /// Studio latency, ms (20-50).
    pub studio_ms: f32,

    pub tier: TierChoice,
    pub engine: EngineChoice,
    pub key: KeyChoice,
    pub scale: ScaleChoice,
    /// Override per pitch class, index 0 = C.
    pub note_overrides: [NoteOverride; 12],

    /// Retune speed τ, ms (0-400).
    pub retune_ms: f32,
    /// Note transition linked to the retune speed.
    pub transition_link: bool,
    /// Note transition τ when unlinked, ms (0-800).
    pub transition_ms: f32,
    /// 0-100.
    pub humanize: f32,
    /// 0-100.
    pub flex_tune: f32,
    /// Correction amount, percent (0-100).
    pub amount_pct: f32,
    /// ±100 cents.
    pub detune_cents: f32,
    /// ±12 semitones, whole steps.
    pub transpose_st: f32,
    /// ±12 semitones.
    pub formant_st: f32,
    pub formant_follow: bool,
    pub preserve_vibrato: bool,
    pub ignore_vibrato: bool,
    /// Vibrato depth, percent (-100..200).
    pub vibrato_depth_pct: f32,
    /// 0-40 cents.
    pub tolerance_cents: f32,
    /// 0-300 ms.
    pub tolerance_ms: f32,
    /// 0-50 cents.
    pub snap_min_cents: f32,
    /// 0-12 semitones, 0 = off.
    pub snap_max_st: f32,
    /// 0-100.
    pub tracking: f32,
    /// dBFS.
    pub gate_db: f32,
    pub analysis_channel: AnalysisChoice,

    pub lfo_on: bool,
    pub lfo_rate_hz: f32,
    pub lfo_depth_cents: f32,
    pub lfo_delay_ms: f32,
    pub lfo_ramp_ms: f32,

    pub midi_mode: MidiModeChoice,
    pub midi_all_octaves: bool,
    /// Pitch-bend range, semitones (0-12).
    pub bend_range_st: f32,

    /// Output gain, dB. Applied by the plugin after the engine (smoothed).
    pub output_gain_db: f32,
}

impl Default for ParamValues {
    /// R§7.6.4 recommended defaults.
    fn default() -> Self {
        Self {
            latency_mode: LatencyModeChoice::Live,
            voice_range: VoiceRangeChoice::AltoTenor,
            live_plus_ms: 10.0,
            studio_ms: 40.0,
            tier: TierChoice::Hq,
            engine: EngineChoice::Natural,
            key: KeyChoice::C,
            scale: ScaleChoice::Chromatic,
            note_overrides: [NoteOverride::Scale; 12],
            retune_ms: 20.0,
            transition_link: true,
            // R§7.6.4: "120 when unlinked".
            transition_ms: 120.0,
            humanize: 0.0,
            flex_tune: 0.0,
            amount_pct: 100.0,
            detune_cents: 0.0,
            transpose_st: 0.0,
            formant_st: 0.0,
            formant_follow: false,
            preserve_vibrato: true,
            ignore_vibrato: true,
            vibrato_depth_pct: 100.0,
            tolerance_cents: 8.0,
            tolerance_ms: 0.0,
            snap_min_cents: 0.0,
            snap_max_st: 0.0,
            tracking: 50.0,
            gate_db: -55.0,
            analysis_channel: AnalysisChoice::Mix,
            lfo_on: false,
            lfo_rate_hz: 5.5,
            lfo_depth_cents: 30.0,
            lfo_delay_ms: 250.0,
            lfo_ramp_ms: 300.0,
            midi_mode: MidiModeChoice::Off,
            midi_all_octaves: false,
            bend_range_st: 2.0,
            output_gain_db: 0.0,
        }
    }
}

/// Snap Min below this many cents is off (RT-1). The knob's bottom position
/// after a host's float round trip (normalized 1e-6 is 0.00005 c) must reach
/// the control law as exactly 0, never as a microscopic dead zone. Same
/// threshold as `skato_control`'s own flush, so both layers agree.
pub const SNAP_MIN_OFF_BELOW_CENTS: f32 = 0.05;

/// Clamp to `[lo, hi]`, mapping NaN/Inf to `fallback` (Settings must never
/// carry non-finite values into the control law).
fn clean(x: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        fallback
    }
}

/// Effective state of every pitch class: the override when set, otherwise scale
/// membership (On inside the scale, Remove outside). DESIGN §3.6.
pub fn resolve_note_states(
    scale: ScaleKind,
    key: u8,
    overrides: &[NoteOverride; 12],
) -> [NoteState; 12] {
    let from_scale = scale.note_states(key % 12);
    let mut out = [NoteState::On; 12];
    for ((dst, &ovr), &scale_state) in out.iter_mut().zip(overrides).zip(&from_scale) {
        *dst = match ovr {
            NoteOverride::Scale => scale_state,
            NoteOverride::On => NoteState::On,
            NoteOverride::Bypass => NoteState::Bypass,
            NoteOverride::Remove => NoteState::Remove,
        };
    }
    out
}

impl ParamValues {
    /// Resolve the runtime engine settings. Pure and allocation-free; called
    /// once per host block on the audio thread.
    pub fn to_settings(&self) -> Settings {
        let key = self.key.pitch_class();
        let scale: ScaleKind = self.scale.into();
        let snap_max = clean(self.snap_max_st, 0.0, 12.0, 0.0);
        let snap_min = clean(self.snap_min_cents, 0.0, 50.0, 0.0);
        Settings {
            tier: self.tier.into(),
            engine: self.engine.into(),
            key,
            scale,
            note_states: resolve_note_states(scale, key, &self.note_overrides),
            detune_cents: clean(self.detune_cents, -100.0, 100.0, 0.0),
            retune_ms: clean(self.retune_ms, 0.0, 400.0, 20.0),
            transition_ms: if self.transition_link {
                None
            } else {
                Some(clean(self.transition_ms, 0.0, 800.0, 120.0))
            },
            humanize: clean(self.humanize, 0.0, 100.0, 0.0),
            flex_tune: clean(self.flex_tune, 0.0, 100.0, 0.0),
            amount: clean(self.amount_pct * 0.01, 0.0, 1.0, 1.0),
            tolerance_cents: clean(self.tolerance_cents, 0.0, 40.0, 8.0),
            tolerance_ms: clean(self.tolerance_ms, 0.0, 300.0, 0.0),
            snap_min_cents: if snap_min < SNAP_MIN_OFF_BELOW_CENTS {
                0.0
            } else {
                snap_min
            },
            // The knob's bottom position reads "Off"; anything below a hair
            // above zero is treated as off so a host's float jitter cannot
            // turn on a 0.001 st limit that would disable all correction.
            snap_max_st: if snap_max < 0.05 { 0.0 } else { snap_max },
            preserve_vibrato: self.preserve_vibrato,
            ignore_vibrato: self.ignore_vibrato,
            vibrato_depth: clean(self.vibrato_depth_pct * 0.01, -1.0, 2.0, 1.0),
            synth_vibrato: SynthVibrato {
                enabled: self.lfo_on,
                rate_hz: clean(self.lfo_rate_hz, 0.1, 10.0, 5.5),
                depth_cents: clean(self.lfo_depth_cents, 0.0, 100.0, 30.0),
                delay_ms: clean(self.lfo_delay_ms, 0.0, 2000.0, 250.0),
                ramp_ms: clean(self.lfo_ramp_ms, 0.0, 2000.0, 300.0),
            },
            // Transpose is a whole-semitone control (DESIGN §3.6, step 1).
            transpose_st: clean(self.transpose_st, -12.0, 12.0, 0.0).round(),
            formant_st: clean(self.formant_st, -12.0, 12.0, 0.0),
            formant_follows_transpose: self.formant_follow,
            tracking: clean(self.tracking, 0.0, 100.0, 50.0),
            gate_db: clean(self.gate_db, -100.0, 0.0, -55.0),
            analysis_channel: self.analysis_channel.into(),
            midi_mode: self.midi_mode.into(),
            midi_all_octaves: self.midi_all_octaves,
            bend_range_st: clean(self.bend_range_st, 0.0, 12.0, 2.0),
        }
    }

    /// The latency-defining part of the configuration.
    pub fn structural(&self) -> Structural {
        Structural {
            latency_mode: self.latency_mode.into(),
            voice_range: self.voice_range.into(),
            live_plus_ms: clean(self.live_plus_ms, 5.0, 10.0, 10.0),
            studio_ms: clean(self.studio_ms, 20.0, 50.0, 40.0),
        }
    }
}

/// Structural configuration: everything that sets buffer sizes or the reported
/// latency. Only applied by building a new engine in `activate()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Structural {
    pub latency_mode: LatencyMode,
    pub voice_range: VoiceRange,
    pub live_plus_ms: f32,
    pub studio_ms: f32,
}

impl Structural {
    /// Full engine configuration for a host setup.
    pub fn engine_config(
        &self,
        sample_rate: f32,
        max_block: usize,
        channels: usize,
    ) -> EngineConfig {
        EngineConfig {
            sample_rate,
            max_block: max_block.max(1),
            channels: channels.clamp(1, 2),
            latency_mode: self.latency_mode,
            voice_range: self.voice_range,
            live_plus_ms: self.live_plus_ms,
            studio_ms: self.studio_ms,
        }
    }

    /// Whether two configurations need the same engine. Only the latency
    /// setting of the *selected* mode counts: editing the Studio latency while
    /// in Live mode must not restart the plugin.
    pub fn same_engine(&self, other: &Structural) -> bool {
        // Parameter values are step-snapped, but a host can still hand us a
        // value that went through a float round trip; compare with slack.
        let close = |a: f32, b: f32| (a - b).abs() < 1e-3;
        self.latency_mode == other.latency_mode
            && self.voice_range == other.voice_range
            && match self.latency_mode {
                LatencyMode::LivePlus => close(self.live_plus_ms, other.live_plus_ms),
                LatencyMode::Studio => close(self.studio_ms, other.studio_ms),
                LatencyMode::LiveMin | LatencyMode::Live => true,
            }
    }
}

// ---------------------------------------------------------------------------
// Presets (R§7.6.4)
// ---------------------------------------------------------------------------

/// Factory correction presets. A preset only touches correction-behaviour
/// parameters; key, scale, latency and detection settings are left alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    HardTune,
    Natural,
    LiveSafe,
    Classic,
}

impl Preset {
    pub const ALL: [Preset; 4] = [
        Preset::HardTune,
        Preset::Natural,
        Preset::LiveSafe,
        Preset::Classic,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Preset::HardTune => "Hard Tune",
            Preset::Natural => "Natural",
            Preset::LiveSafe => "Live Safe",
            Preset::Classic => "Classic",
        }
    }

    /// One-line description for tooltips.
    pub fn description(self) -> &'static str {
        match self {
            Preset::HardTune => {
                "Instant retune and transitions, vibrato flattened: the robotic effect"
            }
            Preset::Natural => {
                "25 ms retune, humanize and flex-tune, vibrato kept: transparent correction"
            }
            Preset::LiveSafe => "15 ms retune with modest tolerance: stable on stage",
            Preset::Classic => {
                "Classic engine, near-instant retune, raw-pitch targeting (Auto-Tune 5 style)"
            }
        }
    }

    /// `base` with this preset's settings applied (R§7.6.4 preset table).
    pub fn apply(self, base: &ParamValues) -> ParamValues {
        let mut v = *base;
        // Shared baseline: full amount, no dead zone/limit, no humanize/flex
        // unless the preset asks for it.
        v.amount_pct = 100.0;
        v.snap_min_cents = 0.0;
        v.snap_max_st = 0.0;
        v.humanize = 0.0;
        v.flex_tune = 0.0;
        v.vibrato_depth_pct = 100.0;
        v.transition_link = true;
        v.preserve_vibrato = true;
        v.ignore_vibrato = true;
        match self {
            Preset::HardTune => {
                // τ = 0, τ_tr = 0 (linked), Ignore Vibrato on, Depth 0 %,
                // Tolerance 5 c / 0 ms. Onset snap follows from τ <= 10 ms in
                // the control law (R§7.6.2 F).
                v.retune_ms = 0.0;
                v.vibrato_depth_pct = 0.0;
                v.tolerance_cents = 5.0;
                v.tolerance_ms = 0.0;
            }
            Preset::Natural => {
                v.engine = EngineChoice::Natural;
                v.retune_ms = 25.0;
                v.humanize = 30.0;
                v.flex_tune = 20.0;
                v.tolerance_cents = 15.0;
                v.tolerance_ms = 30.0;
            }
            Preset::LiveSafe => {
                v.engine = EngineChoice::Natural;
                v.retune_ms = 15.0;
                v.tolerance_cents = 10.0;
                v.tolerance_ms = 0.0;
            }
            Preset::Classic => {
                // Classic engine, τ in 0-5 ms (3 ms chosen so the very first
                // hop is not a hard step), instant jump on note change
                // (unlinked τ_tr = 0), formant off, Preserve Vibrato off and
                // targeting from the raw pitch.
                v.engine = EngineChoice::Classic;
                v.retune_ms = 3.0;
                v.transition_link = false;
                v.transition_ms = 0.0;
                v.preserve_vibrato = false;
                v.ignore_vibrato = false;
                v.formant_st = 0.0;
                v.formant_follow = false;
                v.tolerance_cents = 8.0;
                v.tolerance_ms = 0.0;
            }
        }
        v
    }

    /// Whether `v` currently has exactly this preset's settings.
    pub fn matches(self, v: &ParamValues) -> bool {
        self.apply(v) == *v
    }
}

// ---------------------------------------------------------------------------
// nice-plug parameter objects
// ---------------------------------------------------------------------------

type ValueToString = Arc<dyn Fn(f32) -> String + Send + Sync>;
type StringToValue = Arc<dyn Fn(&str) -> Option<f32> + Send + Sync>;

/// Parse the leading number of a string such as `"20 ms"`, `"-3st"` or `"+5 c"`.
pub fn parse_leading_f32(s: &str) -> Option<f32> {
    let t = s.trim();
    let end = t
        .char_indices()
        .find(|&(i, c)| !(c.is_ascii_digit() || c == '.' || (i == 0 && (c == '-' || c == '+'))))
        .map_or(t.len(), |(i, _)| i);
    t.get(..end)?.parse().ok()
}

fn s2v_number() -> StringToValue {
    Arc::new(parse_leading_f32)
}

/// `"20 ms"`, with one decimal below 10 ms so short retune times are readable.
fn v2s_ms() -> ValueToString {
    Arc::new(|v| {
        if v < 9.95 {
            format!("{v:.1}")
        } else {
            format!("{v:.0}")
        }
    })
}

fn v2s_digits(digits: usize) -> ValueToString {
    Arc::new(move |v| format!("{v:.digits$}"))
}

/// Explicit sign for bipolar values: `"+5"`, `"-3"`, `"0"`.
fn v2s_signed(digits: usize) -> ValueToString {
    Arc::new(move |v| {
        if v.abs() < 0.5 * 10f32.powi(-(digits as i32)) {
            format!("{:.digits$}", 0.0)
        } else {
            format!("{v:+.digits$}")
        }
    })
}

/// Snap Max reads "Off" at zero. The unit is part of the string (the
/// parameter has no unit) so the off state does not read "Off st".
fn v2s_snap_max() -> ValueToString {
    Arc::new(|v| {
        if v < 0.05 {
            "Off".to_string()
        } else {
            format!("{v:.1} st")
        }
    })
}

fn s2v_snap_max() -> StringToValue {
    Arc::new(|s| {
        if s.trim().eq_ignore_ascii_case("off") {
            Some(0.0)
        } else {
            parse_leading_f32(s)
        }
    })
}

/// A plain linear float parameter with a numeric formatter and parser.
fn linear(
    name: &str,
    default: f32,
    min: f32,
    max: f32,
    unit: &'static str,
    digits: usize,
) -> FloatParam {
    FloatParam::new(name, default, FloatRange::Linear { min, max })
        .with_unit(unit)
        .with_value_to_string(v2s_digits(digits))
        .with_string_to_value(s2v_number())
}

/// A time-constant style parameter, skewed so the musically busy short end
/// gets most of the knob travel.
fn skewed_ms(name: &str, default: f32, max: f32, skew: f32) -> FloatParam {
    FloatParam::new(
        name,
        default,
        FloatRange::Skewed {
            min: 0.0,
            max,
            factor: FloatRange::skew_factor(skew),
        },
    )
    .with_unit(" ms")
    .with_value_to_string(v2s_ms())
    .with_string_to_value(s2v_number())
}

/// A bipolar parameter displayed with an explicit sign.
fn bipolar(name: &str, default: f32, range: f32, unit: &'static str, digits: usize) -> FloatParam {
    FloatParam::new(
        name,
        default,
        FloatRange::Linear {
            min: -range,
            max: range,
        },
    )
    .with_unit(unit)
    .with_value_to_string(v2s_signed(digits))
    .with_string_to_value(s2v_number())
}

fn note_param(pc: usize, default: NoteOverride) -> EnumParam<NoteOverride> {
    let name = NOTE_NAMES.get(pc).copied().unwrap_or("?");
    EnumParam::new(format!("Note {name}"), default)
}

/// Every host-visible parameter. Field ids are stable forever (DESIGN §3.6):
/// never rename an `#[id]`, only add new ones.
#[derive(Params)]
pub struct SkatotuneParams {
    // --- Structural (non-automatable; changing them restarts the plugin) ----
    #[id = "latency_mode"]
    pub latency_mode: EnumParam<LatencyModeChoice>,
    #[id = "voice_range"]
    pub voice_range: EnumParam<VoiceRangeChoice>,
    #[id = "live_plus_ms"]
    pub live_plus_ms: FloatParam,
    #[id = "studio_ms"]
    pub studio_ms: FloatParam,

    // --- Engine ---------------------------------------------------------------
    /// Automation is discouraged (tier changes crossfade), so it is
    /// non-automatable like R§9.5 suggests.
    #[id = "tier"]
    pub tier: EnumParam<TierChoice>,
    #[id = "engine"]
    pub engine: EnumParam<EngineChoice>,

    // --- Targets ----------------------------------------------------------------
    #[id = "key"]
    pub key: EnumParam<KeyChoice>,
    #[id = "scale"]
    pub scale: EnumParam<ScaleChoice>,
    #[id = "note_0"]
    pub note_0: EnumParam<NoteOverride>,
    #[id = "note_1"]
    pub note_1: EnumParam<NoteOverride>,
    #[id = "note_2"]
    pub note_2: EnumParam<NoteOverride>,
    #[id = "note_3"]
    pub note_3: EnumParam<NoteOverride>,
    #[id = "note_4"]
    pub note_4: EnumParam<NoteOverride>,
    #[id = "note_5"]
    pub note_5: EnumParam<NoteOverride>,
    #[id = "note_6"]
    pub note_6: EnumParam<NoteOverride>,
    #[id = "note_7"]
    pub note_7: EnumParam<NoteOverride>,
    #[id = "note_8"]
    pub note_8: EnumParam<NoteOverride>,
    #[id = "note_9"]
    pub note_9: EnumParam<NoteOverride>,
    #[id = "note_10"]
    pub note_10: EnumParam<NoteOverride>,
    #[id = "note_11"]
    pub note_11: EnumParam<NoteOverride>,
    #[id = "detune_cents"]
    pub detune_cents: FloatParam,

    // --- Correction ---------------------------------------------------------------
    #[id = "retune_ms"]
    pub retune_ms: FloatParam,
    #[id = "transition_link"]
    pub transition_link: BoolParam,
    #[id = "transition_ms"]
    pub transition_ms: FloatParam,
    #[id = "humanize"]
    pub humanize: FloatParam,
    #[id = "flex_tune"]
    pub flex_tune: FloatParam,
    #[id = "amount"]
    pub amount: FloatParam,
    #[id = "tolerance_cents"]
    pub tolerance_cents: FloatParam,
    #[id = "tolerance_ms"]
    pub tolerance_ms: FloatParam,
    #[id = "snap_min_cents"]
    pub snap_min_cents: FloatParam,
    #[id = "snap_max_st"]
    pub snap_max_st: FloatParam,

    // --- Vibrato --------------------------------------------------------------------
    #[id = "preserve_vibrato"]
    pub preserve_vibrato: BoolParam,
    #[id = "ignore_vibrato"]
    pub ignore_vibrato: BoolParam,
    #[id = "vibrato_depth"]
    pub vibrato_depth: FloatParam,
    #[id = "lfo_on"]
    pub lfo_on: BoolParam,
    #[id = "lfo_rate"]
    pub lfo_rate: FloatParam,
    #[id = "lfo_depth"]
    pub lfo_depth: FloatParam,
    #[id = "lfo_delay"]
    pub lfo_delay: FloatParam,
    #[id = "lfo_ramp"]
    pub lfo_ramp: FloatParam,

    // --- Output -----------------------------------------------------------------------
    #[id = "transpose_st"]
    pub transpose_st: FloatParam,
    #[id = "formant_st"]
    pub formant_st: FloatParam,
    #[id = "formant_follow"]
    pub formant_follow: BoolParam,
    #[id = "output_gain_db"]
    pub output_gain_db: FloatParam,

    // --- Detection ----------------------------------------------------------------------
    #[id = "tracking"]
    pub tracking: FloatParam,
    #[id = "gate_db"]
    pub gate_db: FloatParam,
    #[id = "analysis_channel"]
    pub analysis_channel: EnumParam<AnalysisChoice>,

    // --- MIDI -------------------------------------------------------------------------------
    #[id = "midi_mode"]
    pub midi_mode: EnumParam<MidiModeChoice>,
    #[id = "midi_all_octaves"]
    pub midi_all_octaves: BoolParam,
    #[id = "bend_range"]
    pub bend_range: FloatParam,

    // --- Saved state that is not a parameter --------------------------------------------
    /// Scale learned through MIDI Learn ([`crate::learn`] word layout). Saved
    /// with the project; engine resets and rebuilds never clear it.
    #[persist = "learned-scale"]
    pub learned_scale: AtomicU32,
    /// Editor experiments switched in the LABS menu ([`crate::labs`]). Saved
    /// with the project; GUI-only, never read on the audio thread.
    #[persist = "labs"]
    pub labs: RwLock<LabsSettings>,
}

impl Default for SkatotuneParams {
    fn default() -> Self {
        Self::with_values(&ParamValues::default())
    }
}

impl SkatotuneParams {
    /// Build the parameter objects with `d` as their defaults.
    pub fn with_values(d: &ParamValues) -> Self {
        let n = &d.note_overrides;
        Self {
            latency_mode: EnumParam::new("Latency Mode", d.latency_mode).non_automatable(),
            voice_range: EnumParam::new("Voice Range", d.voice_range).non_automatable(),
            live_plus_ms: linear("Live+ Latency", d.live_plus_ms, 5.0, 10.0, " ms", 1)
                .with_step_size(0.5)
                .non_automatable(),
            studio_ms: linear("Studio Latency", d.studio_ms, 20.0, 50.0, " ms", 0)
                .with_step_size(1.0)
                .non_automatable(),

            tier: EnumParam::new("Quality", d.tier).non_automatable(),
            engine: EnumParam::new("Engine", d.engine),

            key: EnumParam::new("Key", d.key),
            scale: EnumParam::new("Scale", d.scale),
            note_0: note_param(0, n[0]),
            note_1: note_param(1, n[1]),
            note_2: note_param(2, n[2]),
            note_3: note_param(3, n[3]),
            note_4: note_param(4, n[4]),
            note_5: note_param(5, n[5]),
            note_6: note_param(6, n[6]),
            note_7: note_param(7, n[7]),
            note_8: note_param(8, n[8]),
            note_9: note_param(9, n[9]),
            note_10: note_param(10, n[10]),
            note_11: note_param(11, n[11]),
            detune_cents: bipolar("Detune", d.detune_cents, 100.0, " c", 0).with_step_size(1.0),

            // Log-like skew: 20 ms sits near the middle of the knob.
            retune_ms: skewed_ms("Retune Speed", d.retune_ms, 400.0, -2.0),
            transition_link: BoolParam::new("Transition Link", d.transition_link),
            transition_ms: skewed_ms("Note Transition", d.transition_ms, 800.0, -2.0),
            humanize: linear("Humanize", d.humanize, 0.0, 100.0, "", 0),
            flex_tune: linear("Flex-Tune", d.flex_tune, 0.0, 100.0, "", 0),
            amount: linear("Amount", d.amount_pct, 0.0, 100.0, " %", 0),
            tolerance_cents: linear("Tolerance", d.tolerance_cents, 0.0, 40.0, " c", 0),
            tolerance_ms: skewed_ms("Tolerance Time", d.tolerance_ms, 300.0, -1.0),
            snap_min_cents: linear("Snap Min", d.snap_min_cents, 0.0, 50.0, " c", 0),
            snap_max_st: FloatParam::new(
                "Snap Max",
                d.snap_max_st,
                FloatRange::Linear {
                    min: 0.0,
                    max: 12.0,
                },
            )
            .with_step_size(0.5)
            .with_value_to_string(v2s_snap_max())
            .with_string_to_value(s2v_snap_max()),

            preserve_vibrato: BoolParam::new("Preserve Vibrato", d.preserve_vibrato),
            ignore_vibrato: BoolParam::new("Ignore Vibrato", d.ignore_vibrato),
            vibrato_depth: linear("Vibrato Depth", d.vibrato_depth_pct, -100.0, 200.0, " %", 0),
            lfo_on: BoolParam::new("Synth Vibrato", d.lfo_on),
            lfo_rate: FloatParam::new(
                "Synth Vibrato Rate",
                d.lfo_rate_hz,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 10.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_unit(" Hz")
            .with_value_to_string(v2s_digits(2))
            .with_string_to_value(s2v_number()),
            lfo_depth: linear(
                "Synth Vibrato Depth",
                d.lfo_depth_cents,
                0.0,
                100.0,
                " c",
                0,
            ),
            lfo_delay: skewed_ms("Synth Vibrato Delay", d.lfo_delay_ms, 2000.0, -1.5),
            lfo_ramp: skewed_ms("Synth Vibrato Ramp", d.lfo_ramp_ms, 2000.0, -1.5),

            transpose_st: bipolar("Transpose", d.transpose_st, 12.0, " st", 0).with_step_size(1.0),
            formant_st: bipolar("Formant", d.formant_st, 12.0, " st", 1).with_step_size(0.1),
            formant_follow: BoolParam::new("Formant Follows Transpose", d.formant_follow),
            output_gain_db: FloatParam::new(
                "Output Gain",
                d.output_gain_db,
                FloatRange::Linear {
                    min: -24.0,
                    max: 12.0,
                },
            )
            .with_step_size(0.1)
            .with_unit(" dB")
            .with_value_to_string(v2s_signed(1))
            .with_string_to_value(s2v_number())
            // Smoothed in dB (perceptually even) to avoid zipper noise.
            .with_smoother(SmoothingStyle::Linear(20.0)),

            tracking: linear("Tracking", d.tracking, 0.0, 100.0, "", 0),
            gate_db: linear("Gate", d.gate_db, -90.0, -20.0, " dB", 0),
            analysis_channel: EnumParam::new("Analysis Channel", d.analysis_channel),

            midi_mode: EnumParam::new("MIDI Mode", d.midi_mode),
            midi_all_octaves: BoolParam::new("MIDI All Octaves", d.midi_all_octaves),
            bend_range: linear("Bend Range", d.bend_range_st, 0.0, 12.0, " st", 0)
                .with_step_size(1.0),
            learned_scale: AtomicU32::new(0),
            labs: RwLock::new(LabsSettings::default()),
        }
    }

    /// The scale learned through MIDI Learn, if any (On = learned, the rest
    /// Remove). Atomic load; safe on any thread.
    pub fn learned_states(&self) -> Option<[NoteState; 12]> {
        crate::learn::unpack(self.learned_scale.load(Ordering::Relaxed))
    }

    /// Forget the learned scale: correction returns to Key/Scale and the note
    /// overrides. Atomic store; safe on any thread.
    pub fn clear_learned_scale(&self) {
        self.learned_scale.store(0, Ordering::Relaxed);
    }

    /// The current Labs switches (GUI thread; a short read lock).
    pub fn labs(&self) -> LabsSettings {
        match self.labs.read() {
            Ok(l) => *l,
            Err(poisoned) => *poisoned.into_inner(),
        }
    }

    /// Change the Labs switches (GUI thread). Saved with the project.
    pub fn set_labs(&self, labs: LabsSettings) {
        match self.labs.write() {
            Ok(mut l) => *l = labs,
            Err(poisoned) => *poisoned.into_inner() = labs,
        }
    }

    /// The override parameter of pitch class `pc` (0 = C; wraps modulo 12).
    pub fn note_override(&self, pc: usize) -> &EnumParam<NoteOverride> {
        match pc % 12 {
            0 => &self.note_0,
            1 => &self.note_1,
            2 => &self.note_2,
            3 => &self.note_3,
            4 => &self.note_4,
            5 => &self.note_5,
            6 => &self.note_6,
            7 => &self.note_7,
            8 => &self.note_8,
            9 => &self.note_9,
            10 => &self.note_10,
            _ => &self.note_11,
        }
    }

    /// Snapshot every parameter value (atomic loads only; RT-safe).
    pub fn values(&self) -> ParamValues {
        ParamValues {
            latency_mode: self.latency_mode.value(),
            voice_range: self.voice_range.value(),
            live_plus_ms: self.live_plus_ms.value(),
            studio_ms: self.studio_ms.value(),
            tier: self.tier.value(),
            engine: self.engine.value(),
            key: self.key.value(),
            scale: self.scale.value(),
            note_overrides: std::array::from_fn(|pc| self.note_override(pc).value()),
            retune_ms: self.retune_ms.value(),
            transition_link: self.transition_link.value(),
            transition_ms: self.transition_ms.value(),
            humanize: self.humanize.value(),
            flex_tune: self.flex_tune.value(),
            amount_pct: self.amount.value(),
            detune_cents: self.detune_cents.value(),
            transpose_st: self.transpose_st.value(),
            formant_st: self.formant_st.value(),
            formant_follow: self.formant_follow.value(),
            preserve_vibrato: self.preserve_vibrato.value(),
            ignore_vibrato: self.ignore_vibrato.value(),
            vibrato_depth_pct: self.vibrato_depth.value(),
            tolerance_cents: self.tolerance_cents.value(),
            tolerance_ms: self.tolerance_ms.value(),
            snap_min_cents: self.snap_min_cents.value(),
            snap_max_st: self.snap_max_st.value(),
            tracking: self.tracking.value(),
            gate_db: self.gate_db.value(),
            analysis_channel: self.analysis_channel.value(),
            lfo_on: self.lfo_on.value(),
            lfo_rate_hz: self.lfo_rate.value(),
            lfo_depth_cents: self.lfo_depth.value(),
            lfo_delay_ms: self.lfo_delay.value(),
            lfo_ramp_ms: self.lfo_ramp.value(),
            midi_mode: self.midi_mode.value(),
            midi_all_octaves: self.midi_all_octaves.value(),
            bend_range_st: self.bend_range.value(),
            output_gain_db: self.output_gain_db.value(),
        }
    }
}

#[cfg(test)]
// Tests read best as "defaults, then change what matters".
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::labs::CorrectionTheme;
    use skato_core::NoteState::{Bypass, On, Remove};

    #[test]
    fn defaults_match_core_settings() {
        // The plugin's defaults are the R§7.6.4 table, which skato_core's
        // Settings::default() also encodes.
        assert_eq!(ParamValues::default().to_settings(), Settings::default());
    }

    #[test]
    fn param_objects_report_the_snapshot_defaults() {
        let p = SkatotuneParams::default();
        assert_eq!(p.values(), ParamValues::default());
    }

    #[test]
    fn scale_override_uses_scale_membership() {
        let mut v = ParamValues::default();
        v.key = KeyChoice::D;
        v.scale = ScaleChoice::Major; // D E F# G A B C#
        let s = v.to_settings();
        assert_eq!(s.key, 2);
        assert_eq!(s.scale, ScaleKind::Major);
        let expected = [
            Remove, On, On, Remove, On, Remove, On, On, Remove, On, Remove, On,
        ];
        assert_eq!(s.note_states, expected);
    }

    #[test]
    fn explicit_overrides_win_over_the_scale() {
        let mut v = ParamValues::default();
        v.scale = ScaleChoice::Major; // C major
        v.note_overrides[1] = NoteOverride::On; // C# forced on although out of scale
        v.note_overrides[4] = NoteOverride::Bypass; // E in scale, bypassed
        v.note_overrides[7] = NoteOverride::Remove; // G in scale, removed
        let s = v.to_settings();
        assert_eq!(s.note_states[0], On);
        assert_eq!(s.note_states[1], On);
        assert_eq!(s.note_states[3], Remove);
        assert_eq!(s.note_states[4], Bypass);
        assert_eq!(s.note_states[7], Remove);
        assert_eq!(s.note_states[9], On);
    }

    #[test]
    fn transition_link_maps_to_none() {
        let mut v = ParamValues::default();
        v.transition_ms = 300.0;
        v.transition_link = true;
        assert_eq!(v.to_settings().transition_ms, None);
        v.transition_link = false;
        assert_eq!(v.to_settings().transition_ms, Some(300.0));
    }

    #[test]
    fn snap_max_zero_is_off() {
        let mut v = ParamValues::default();
        v.snap_max_st = 0.0;
        assert_eq!(v.to_settings().snap_max_st, 0.0);
        v.snap_max_st = 0.01; // float jitter from a host must not enable it
        assert_eq!(v.to_settings().snap_max_st, 0.0);
        v.snap_max_st = 2.5;
        assert_eq!(v.to_settings().snap_max_st, 2.5);
    }

    #[test]
    fn unit_conversions() {
        let mut v = ParamValues::default();
        v.amount_pct = 40.0;
        v.vibrato_depth_pct = -50.0;
        v.transpose_st = 3.4;
        let s = v.to_settings();
        assert!((s.amount - 0.4).abs() < 1e-6);
        assert!((s.vibrato_depth + 0.5).abs() < 1e-6);
        assert_eq!(s.transpose_st, 3.0);
    }

    #[test]
    fn non_finite_and_out_of_range_values_are_sanitized() {
        let mut v = ParamValues::default();
        v.retune_ms = f32::NAN;
        v.amount_pct = 1e9;
        v.gate_db = f32::NEG_INFINITY;
        v.vibrato_depth_pct = 900.0;
        let s = v.to_settings();
        assert_eq!(s.retune_ms, 20.0);
        assert_eq!(s.amount, 1.0);
        assert_eq!(s.gate_db, -55.0);
        assert_eq!(s.vibrato_depth, 2.0);
    }

    #[test]
    fn synth_vibrato_fields_map_through() {
        let mut v = ParamValues::default();
        v.lfo_on = true;
        v.lfo_rate_hz = 6.0;
        v.lfo_depth_cents = 45.0;
        v.lfo_delay_ms = 100.0;
        v.lfo_ramp_ms = 50.0;
        let lfo = v.to_settings().synth_vibrato;
        assert_eq!(
            lfo,
            SynthVibrato {
                enabled: true,
                rate_hz: 6.0,
                depth_cents: 45.0,
                delay_ms: 100.0,
                ramp_ms: 50.0
            }
        );
    }

    #[test]
    fn structural_comparison_ignores_irrelevant_latency_fields() {
        let mut a = ParamValues::default();
        let mut b = a;
        b.studio_ms = 25.0; // not the active mode
        assert!(a.structural().same_engine(&b.structural()));
        a.latency_mode = LatencyModeChoice::Studio;
        b.latency_mode = LatencyModeChoice::Studio;
        assert!(!a.structural().same_engine(&b.structural()));
        b.studio_ms = a.studio_ms;
        b.voice_range = VoiceRangeChoice::Soprano;
        assert!(!a.structural().same_engine(&b.structural()));
    }

    #[test]
    fn engine_config_carries_the_structural_fields() {
        let mut v = ParamValues::default();
        v.latency_mode = LatencyModeChoice::LivePlus;
        v.live_plus_ms = 7.5;
        v.voice_range = VoiceRangeChoice::LowMale;
        let cfg = v.structural().engine_config(44_100.0, 512, 1);
        assert_eq!(cfg.latency_mode, LatencyMode::LivePlus);
        assert_eq!(cfg.voice_range, VoiceRange::LowMale);
        assert_eq!(cfg.live_plus_ms, 7.5);
        assert_eq!(cfg.sample_rate, 44_100.0);
        assert_eq!(cfg.max_block, 512);
        assert_eq!(cfg.channels, 1);
        // Channels and block size are clamped to what the engine supports.
        let cfg = v.structural().engine_config(48_000.0, 0, 8);
        assert_eq!(cfg.max_block, 1);
        assert_eq!(cfg.channels, 2);
    }

    #[test]
    fn presets_apply_the_research_table() {
        let base = ParamValues::default();
        let hard = Preset::HardTune.apply(&base);
        assert_eq!(hard.retune_ms, 0.0);
        assert!(hard.transition_link);
        assert_eq!(hard.vibrato_depth_pct, 0.0);
        assert_eq!(hard.tolerance_cents, 5.0);
        assert_eq!(hard.to_settings().transition_ms, None);

        let natural = Preset::Natural.apply(&hard);
        assert_eq!(natural.retune_ms, 25.0);
        assert_eq!(natural.humanize, 30.0);
        assert_eq!(natural.flex_tune, 20.0);
        assert_eq!(natural.vibrato_depth_pct, 100.0);
        assert_eq!(
            (natural.tolerance_cents, natural.tolerance_ms),
            (15.0, 30.0)
        );

        let classic = Preset::Classic.apply(&natural);
        assert_eq!(classic.engine, EngineChoice::Classic);
        assert_eq!(classic.to_settings().transition_ms, Some(0.0));
        assert!(!classic.preserve_vibrato);
        assert_eq!(classic.formant_st, 0.0);

        // Presets leave key/scale/latency alone.
        let mut keyed = base;
        keyed.key = KeyChoice::Fs;
        keyed.latency_mode = LatencyModeChoice::Studio;
        let applied = Preset::LiveSafe.apply(&keyed);
        assert_eq!(applied.key, KeyChoice::Fs);
        assert_eq!(applied.latency_mode, LatencyModeChoice::Studio);
        assert_eq!(applied.retune_ms, 15.0);
    }

    #[test]
    fn preset_matching() {
        let v = Preset::Natural.apply(&ParamValues::default());
        assert!(Preset::Natural.matches(&v));
        assert!(!Preset::HardTune.matches(&v));
        let mut tweaked = v;
        tweaked.retune_ms = 26.0;
        assert!(!Preset::Natural.matches(&tweaked));
    }

    #[test]
    fn note_override_cycle_visits_every_state() {
        let mut o = NoteOverride::Scale;
        let mut seen = Vec::new();
        for _ in 0..4 {
            seen.push(o);
            o = o.next();
        }
        assert_eq!(o, NoteOverride::Scale);
        assert_eq!(
            seen,
            [
                NoteOverride::Scale,
                NoteOverride::On,
                NoteOverride::Bypass,
                NoteOverride::Remove
            ]
        );
    }

    #[test]
    fn parsing_and_formatting() {
        assert_eq!(parse_leading_f32("20 ms"), Some(20.0));
        assert_eq!(parse_leading_f32(" -3.5st"), Some(-3.5));
        assert_eq!(parse_leading_f32("+5 c"), Some(5.0));
        assert_eq!(parse_leading_f32("abc"), None);
        let p = SkatotuneParams::default();
        assert_eq!(p.snap_max_st.normalized_value_to_string(0.0, false), "Off");
        assert_eq!(p.retune_ms.to_string(), "20 ms");
        assert_eq!(p.detune_cents.to_string(), "0 c");
        assert_eq!(p.snap_max_st.string_to_normalized_value("off"), Some(0.0));
    }

    #[test]
    fn note_override_accessor_matches_ids() {
        let p = SkatotuneParams::default();
        for (pc, note) in NOTE_NAMES.iter().enumerate() {
            assert_eq!(p.note_override(pc).name(), format!("Note {note}"));
        }
    }

    #[test]
    fn param_ids_are_unique_and_avoid_reserved_hashes() {
        // Same hash nice-plug uses for CLAP/VST3 parameter ids
        // (nice-plug wrapper::util::hash_param_id).
        fn hash(id: &str) -> u32 {
            let mut h: u32 = 0;
            for b in id.bytes() {
                h = h.wrapping_mul(31).wrapping_add(b as u32);
            }
            h & !(1 << 31)
        }
        // VST3 MIDI CC parameters occupy the top 130 * 16 ids below 2^31.
        let midi_start = (1u32 << 31) - 130 * 16;
        let p = SkatotuneParams::default();
        let map = p.param_map();
        let mut hashes: Vec<u32> = map.iter().map(|(id, _, _)| hash(id)).collect();
        assert_eq!(map.len(), 49, "the full parameter set");
        assert!(hashes.iter().all(|&h| h < midi_start));
        hashes.sort_unstable();
        hashes.dedup();
        assert_eq!(hashes.len(), map.len(), "parameter id hash collision");
    }

    /// The Labs switches are saved with the project as the `labs` field and
    /// come back on load; they are not host parameters.
    #[test]
    fn labs_are_saved_with_the_project() {
        let p = SkatotuneParams::default();
        assert_eq!(p.labs(), LabsSettings::default());
        let mine = LabsSettings {
            correction_ribbon: false,
            heat_line: true,
            correction_theme: CorrectionTheme::Viridis,
        };
        p.set_labs(mine);
        p.learned_scale
            .store(crate::learn::pack(&[0, 4, 7]), Ordering::Relaxed);
        let saved = p.serialize_fields();
        assert!(saved.contains_key("labs"), "{saved:?}");
        assert!(saved.contains_key("learned-scale"), "{saved:?}");
        let restored = SkatotuneParams::default();
        restored.deserialize_fields(&saved);
        assert_eq!(restored.labs(), mine);
        assert!(
            !p.param_map().iter().any(|(id, _, _)| id.contains("lab")),
            "never a host parameter"
        );

        // A project saved before Labs existed has no `labs` entry at all:
        // it loads, and the switches keep their defaults.
        let mut pre_labs = saved.clone();
        pre_labs.remove("labs");
        let old_project = SkatotuneParams::default();
        old_project.deserialize_fields(&pre_labs);
        assert_eq!(old_project.labs(), LabsSettings::default());
        assert_eq!(
            old_project.learned_scale.load(Ordering::Relaxed),
            p.learned_scale.load(Ordering::Relaxed),
            "the rest of the state still loads"
        );

        // A save from before some switch existed gets that switch's default.
        let mut older = saved.clone();
        older.insert("labs".into(), r#"{"heat_line":true}"#.into());
        restored.deserialize_fields(&older);
        assert_eq!(
            restored.labs(),
            LabsSettings {
                correction_ribbon: true,
                heat_line: true,
                correction_theme: CorrectionTheme::Teal,
            }
        );

        // A save from a newer version with a theme this one lacks: the
        // switches load, the theme is Teal.
        let mut newer = saved;
        newer.insert(
            "labs".into(),
            r#"{"correction_ribbon":false,"heat_line":true,"correction_theme":"plasma"}"#.into(),
        );
        restored.deserialize_fields(&newer);
        assert_eq!(
            restored.labs(),
            LabsSettings {
                correction_ribbon: false,
                heat_line: true,
                correction_theme: CorrectionTheme::Teal,
            }
        );
    }
}
