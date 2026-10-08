//! End-to-end check of the parameter → `Settings` mapping through the real
//! nice-plug parameter objects: build a parameter set whose values are a
//! chosen snapshot, read it back the way `process()` does, and resolve it.

// Tests read best as "defaults, then change what matters".
#![allow(clippy::field_reassign_with_default)]

use nice_plug::prelude::*;
use vocal_tune::params::{
    KeyChoice, LatencyModeChoice, NoteOverride, ParamValues, Preset, ScaleChoice, VocalTuneParams,
    VoiceRangeChoice,
};
use vt_core::{LatencyMode, NoteState, ScaleKind, Settings};

/// Parameters whose current values are `v` (nice-plug starts every parameter
/// at its default), resolved exactly like the audio thread does.
fn resolve(v: &ParamValues) -> Settings {
    let params = VocalTuneParams::with_values(v);
    let read_back = params.values();
    assert_eq!(
        &read_back, v,
        "parameter objects must report the values they were built with"
    );
    read_back.to_settings()
}

#[test]
fn note_overrides_resolve_against_the_scale() {
    let mut v = ParamValues::default();
    v.key = KeyChoice::A;
    v.scale = ScaleChoice::NaturalMinor; // A B C D E F G
    v.note_overrides[8] = NoteOverride::On; // G#: the harmonic-minor leading tone
    v.note_overrides[7] = NoteOverride::Remove; // G
    v.note_overrides[0] = NoteOverride::Bypass; // C: sung freely
    let s = resolve(&v);
    assert_eq!(s.key, 9);
    assert_eq!(s.scale, ScaleKind::NaturalMinor);
    use NoteState::{Bypass, On, Remove};
    //                  C       C#      D   D#      E   F   F#      G       G#  A   A#      B
    let expected = [
        Bypass, Remove, On, Remove, On, On, Remove, Remove, On, On, Remove, On,
    ];
    assert_eq!(s.note_states, expected);
}

#[test]
fn scale_only_states_match_vt_core() {
    for (i, scale) in [ScaleChoice::Major, ScaleChoice::Blues, ScaleChoice::Dorian]
        .into_iter()
        .enumerate()
    {
        let mut v = ParamValues::default();
        v.scale = scale;
        v.key = [KeyChoice::C, KeyChoice::Fs, KeyChoice::B][i];
        let s = resolve(&v);
        assert_eq!(
            s.note_states,
            ScaleKind::from(scale).note_states(v.key.pitch_class())
        );
    }
}

#[test]
fn transition_link_and_snap_off() {
    let mut v = ParamValues::default();
    v.transition_ms = 250.0;
    assert_eq!(resolve(&v).transition_ms, None, "linked by default");
    v.transition_link = false;
    assert_eq!(resolve(&v).transition_ms, Some(250.0));

    assert_eq!(resolve(&v).snap_max_st, 0.0, "Snap Max defaults to off");
    v.snap_max_st = 3.0;
    assert_eq!(resolve(&v).snap_max_st, 3.0);
}

#[test]
fn percentages_and_whole_semitones() {
    let mut v = ParamValues::default();
    v.amount_pct = 75.0;
    v.vibrato_depth_pct = 150.0;
    v.transpose_st = -5.0;
    let s = resolve(&v);
    assert!((s.amount - 0.75).abs() < 1e-6);
    assert!((s.vibrato_depth - 1.5).abs() < 1e-6);
    assert_eq!(s.transpose_st, -5.0);
}

#[test]
fn presets_resolve_to_research_settings() {
    let hard = resolve(&Preset::HardTune.apply(&ParamValues::default()));
    assert_eq!(hard.retune_ms, 0.0);
    assert_eq!(hard.transition_ms, None); // linked to τ = 0
    assert_eq!(hard.vibrato_depth, 0.0);
    assert_eq!(hard.tolerance_cents, 5.0);

    let natural = resolve(&Preset::Natural.apply(&ParamValues::default()));
    assert_eq!(
        (natural.retune_ms, natural.humanize, natural.flex_tune),
        (25.0, 30.0, 20.0)
    );
    assert_eq!(
        (natural.tolerance_cents, natural.tolerance_ms),
        (15.0, 30.0)
    );
}

#[test]
fn structural_config_and_latency() {
    let mut v = ParamValues::default();
    v.latency_mode = LatencyModeChoice::Studio;
    v.studio_ms = 30.0;
    v.voice_range = VoiceRangeChoice::LowMale;
    let params = VocalTuneParams::with_values(&v);
    let cfg = params
        .values()
        .structural()
        .engine_config(48_000.0, 1024, 2);
    assert_eq!(cfg.latency_mode, LatencyMode::Studio);
    assert_eq!(cfg.studio_ms, 30.0);
    // The plugin reports whatever the engine formula says for this config.
    let latency = vt_engine::latency_samples(&cfg);
    assert!(latency > 0);
}

#[test]
fn structural_params_are_not_automatable() {
    let p = VocalTuneParams::default();
    for flags in [
        p.latency_mode.flags(),
        p.voice_range.flags(),
        p.live_plus_ms.flags(),
        p.studio_ms.flags(),
        p.tier.flags(),
    ] {
        assert!(flags.contains(ParamFlags::NON_AUTOMATABLE));
    }
    assert!(!p.retune_ms.flags().contains(ParamFlags::NON_AUTOMATABLE));
}

#[test]
fn float_defaults_survive_a_normalized_round_trip() {
    let p = VocalTuneParams::default();
    let floats: [&FloatParam; 8] = [
        &p.retune_ms,
        &p.transition_ms,
        &p.vibrato_depth,
        &p.detune_cents,
        &p.gate_db,
        &p.lfo_rate,
        &p.output_gain_db,
        &p.studio_ms,
    ];
    for f in floats {
        let d = f.default_plain_value();
        let back = f.preview_plain(f.preview_normalized(d));
        assert!(
            (back - d).abs() < 1e-3 * d.abs().max(1.0),
            "{}: {d} -> {back}",
            f.name()
        );
    }
}
