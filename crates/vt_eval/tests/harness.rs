//! Harness self-tests through the public `Engine` API.
//!
//! These hold for the stub engine (a pure delay line) *and* encode contract
//! properties the real engine must keep (DESIGN §2.1, §3.5, §4): measured
//! latency equals the reported latency for every mode × range × rate,
//! block-size invariance, determinism, reset, stereo identity, NaN/Inf
//! handling and telemetry bookkeeping.

use vt_core::{EngineConfig, LatencyMode, QualityTier, Settings, VoiceRange};
use vt_engine::Engine;
use vt_eval::render::{self, BlockPattern, RenderOptions};
use vt_eval::scenarios;
use vt_eval::suite::{self, MODES};
use vt_eval::synth::{Consonant, Melody, Note, SignalBuilder, Vibrato, Voice, Vowel};

/// Control hop per DESIGN §2.2.
fn hop_for(sr: f32) -> usize {
    if sr <= 50_000.0 {
        64
    } else if sr <= 100_000.0 {
        128
    } else {
        256
    }
}

/// ~1.6 s of varied material: notes with vibrato, consonants, a breath and a
/// noise burst, so voiced/unvoiced switching and grain scheduling are covered.
fn material(sr: f32) -> Vec<f32> {
    SignalBuilder::new(sr)
        .seed(77)
        .silence(0.05)
        .melody(
            Melody::new(Voice::male())
                .note(Note::midi(52.0, 0.35).detune(25.0).scoop(-80.0))
                .note(Note::midi(55.0, 0.3).detune(-15.0).overshoot(30.0))
                .note(
                    Note::midi(57.0, 0.45)
                        .gap(0.12)
                        .consonant(Consonant::S, 0.1, -30.0)
                        .vibrato(Vibrato::new(6.0, 50.0)),
                ),
        )
        .consonant(Consonant::Breath, 0.15, -40.0)
        .noise_burst(0.05, -24.0)
        .melody(
            Melody::new(Voice::female()).note(Note::midi(69.0, 0.25).detune(-30.0).vowel(Vowel::I)),
        )
        .silence(0.05)
        .build()
        .audio
}

fn opts(sr: f32, mode: LatencyMode) -> RenderOptions {
    let config = EngineConfig {
        sample_rate: sr,
        max_block: 4096,
        channels: 1,
        latency_mode: mode,
        ..EngineConfig::default()
    };
    RenderOptions::new(sr, mode, QualityTier::Hq).with_config(config)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn measured_latency_equals_reported_for_every_mode_range_and_rate() {
    let rows = suite::latency_sweep(&[44_100.0, 48_000.0, 96_000.0]);
    assert_eq!(rows.len(), 3 * 4 * 4);
    for r in &rows {
        let cfg = EngineConfig {
            sample_rate: r.sample_rate,
            latency_mode: r.mode,
            voice_range: r.range,
            ..EngineConfig::default()
        };
        assert_eq!(r.reported, vt_engine::latency_samples(&cfg), "{r:?}");
        assert!(r.ok(), "latency mismatch: {r:?}");
    }
}

#[test]
fn output_is_block_size_invariant() {
    for (sr, modes) in [
        (48_000.0, MODES.to_vec()),
        (44_100.0, vec![LatencyMode::Live]),
        (96_000.0, vec![LatencyMode::Studio]),
    ] {
        let x = material(sr);
        for mode in modes {
            let base = opts(sr, mode);
            let reference = render::render(&x, &base.clone().with_blocks(BlockPattern::Fixed(64)));
            for p in BlockPattern::invariance_set() {
                let o = render::render(&x, &base.clone().with_blocks(p.clone()));
                let d = max_diff(o.raw(0), reference.raw(0));
                assert!(
                    d <= 1e-6,
                    "{sr} Hz {mode:?} blocks {}: max diff {d}",
                    p.label()
                );
            }
        }
    }
}

#[test]
fn renders_are_deterministic_and_reset_restores_state() {
    let sr = 48_000.0;
    let x = material(sr);
    for mode in MODES {
        let o = opts(sr, mode).with_blocks(BlockPattern::Random {
            seed: 3,
            min: 1,
            max: 700,
        });
        let a = render::render(&x, &o);
        let b = render::render(&x, &o);
        assert_eq!(a.raw(0), b.raw(0), "{mode:?}: fresh engines differ");
        let mut engine = Engine::new(o.config);
        let first = render::render_with(&mut engine, &[&x], &o);
        let again = render::render_with(&mut engine, &[&x], &o);
        assert_eq!(
            first.raw(0),
            again.raw(0),
            "{mode:?}: reset() did not restore the initial state"
        );
        assert_eq!(first.raw(0), a.raw(0));
    }
}

#[test]
fn stereo_channels_get_identical_processing() {
    let sr = 48_000.0;
    let x = material(sr);
    for mode in MODES {
        let mut o = opts(sr, mode);
        o.config.channels = 2;
        let r = render::render(&x, &o);
        assert_eq!(r.channels.len(), 2);
        assert_eq!(
            r.raw(0),
            r.raw(1),
            "{mode:?}: L/R differ for identical input"
        );
    }
}

#[test]
fn nan_and_inf_in_silence_are_treated_as_silence() {
    let sr = 48_000.0;
    let clean = material(sr);
    let mut dirty = clean.clone();
    // The material starts with 50 ms of silence: poison part of it.
    for (i, v) in dirty.iter_mut().enumerate().take(2000).skip(200) {
        *v = match i % 3 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            _ => f32::NEG_INFINITY,
        };
    }
    for mode in MODES {
        let o = opts(sr, mode);
        let a = render::render(&clean, &o);
        let b = render::render(&dirty, &o);
        assert!(
            b.raw(0).iter().all(|v| v.is_finite()),
            "{mode:?}: non-finite output"
        );
        assert_eq!(
            a.raw(0),
            b.raw(0),
            "{mode:?}: NaN/Inf in silence changed the output"
        );
    }
}

#[test]
fn nan_inside_a_note_recovers() {
    let sr = 48_000.0;
    let sig = SignalBuilder::new(sr)
        .silence(0.05)
        .melody(Melody::new(Voice::male()).note(Note::midi(55.0, 0.5).detune(20.0)))
        .silence(0.4)
        .melody(Melody::new(Voice::male()).note(Note::midi(57.0, 0.5).detune(-20.0)))
        .silence(0.05)
        .build();
    let mut dirty = sig.audio.clone();
    let start = sig.notes[0].start + 9600;
    for v in &mut dirty[start..start + 480] {
        *v = f32::NAN;
    }
    let last = sig.notes[1];
    for mode in MODES {
        let o = opts(sr, mode);
        let clean = render::render(&sig.audio, &o);
        let r = render::render(&dirty, &o);
        assert!(
            r.raw(0).iter().all(|v| v.is_finite()),
            "{mode:?}: non-finite output"
        );
        // The second note (after a 400 ms gap) must come out as it would have.
        let (a, b) = (
            &clean.aligned(0)[last.start..last.end],
            &r.aligned(0)[last.start..last.end],
        );
        let e: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let p: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
        assert!(
            p > 0.0 && e / p < 0.01,
            "{mode:?}: no recovery after NaN (error {:.1} dB)",
            10.0 * (e / p).log10()
        );
    }
}

#[test]
fn extreme_inputs_do_not_panic_and_stay_finite_when_representable() {
    let sr = 48_000.0;
    let edge = scenarios::build("edge", sr)
        .map(|s| s.signal.audio)
        .unwrap_or_default();
    let loud: Vec<f32> = (0..9600)
        .map(|i| if i % 2 == 0 { 1.0e6 } else { -1.0e6 })
        .collect();
    let huge: Vec<f32> = (0..4800)
        .map(|i| if i % 7 < 3 { 1.0e30 } else { -1.0e30 })
        .collect();
    for mode in MODES {
        let o = opts(sr, mode);
        assert!(
            render::render(&edge, &o)
                .raw(0)
                .iter()
                .all(|v| v.is_finite()),
            "{mode:?}: edge scenario"
        );
        assert!(
            render::render(&loud, &o)
                .raw(0)
                .iter()
                .all(|v| v.is_finite()),
            "{mode:?}: ±1e6 input"
        );
        // Values near f32::MAX may overflow internal energy sums; only
        // panics are forbidden here (recovery is a quality gate).
        let _ = render::render(&huge, &o);
    }
}

/// One frame per applied control, none dropped. The Studio pipeline (RT-2)
/// applies hop h's control at the boundary closing hop h + 1, so the first hop
/// after a reset carries no frame (`render::pipeline_hops`); every later hop
/// does, and pipelined frames are exactly one hop apart. Studio without a
/// control lead runs unpipelined and has one frame per hop like Live.
#[test]
fn telemetry_is_one_frame_per_hop_without_drops() {
    let studio_short_low = |sr: f32| {
        // Low Male at Studio 20 ms leaves the control law no lead: unpipelined.
        let mut o = opts(sr, LatencyMode::Studio);
        o.config.voice_range = VoiceRange::LowMale;
        o.config.studio_ms = 20.0;
        o
    };
    for sr in [44_100.0f32, 48_000.0, 96_000.0] {
        let x = material(sr);
        let hop = hop_for(sr) as u64;
        let mut cases: Vec<(String, RenderOptions)> = MODES
            .iter()
            .map(|&m| (format!("{m:?}"), opts(sr, m)))
            .collect();
        cases.push(("Studio 20 ms LowMale".into(), studio_short_low(sr)));
        let mut pipelined_seen = 0;
        for (label, o) in cases {
            let o = o.with_blocks(BlockPattern::Random {
                seed: 9,
                min: 1,
                max: 2000,
            });
            let skip = render::pipeline_hops(&o.config);
            if label.starts_with("Studio 20") {
                assert_eq!(skip, 0, "{sr} {label} should be unpipelined");
            }
            pipelined_seen += skip;
            let r = render::render(&x, &o);
            let total = r.raw(0).len() as u64;
            assert_eq!(r.stats.hops, total / hop, "{sr} {label}");
            assert_eq!(
                r.telemetry.len() as u64,
                r.stats.hops - skip,
                "{sr} {label}: frames vs hops (pipeline {skip})"
            );
            assert_eq!(r.stats.telemetry_dropped, 0);
            assert!(
                r.telemetry
                    .iter()
                    .all(|f| f.sample_pos <= total + r.latency as u64)
            );
            if skip > 0 {
                // Pipelined Studio frames land on the boundary where their
                // control takes effect (once past the warm-up, where the
                // position saturates at D). Live frames carry the estimate's
                // evidence time, which is not hop-periodic.
                let lat = u64::from(r.latency);
                assert!(
                    r.telemetry
                        .windows(2)
                        .filter(|w| w[0].sample_pos > lat)
                        .all(|w| w[1].sample_pos == w[0].sample_pos + hop),
                    "{sr} {label}: frames not exactly one hop apart"
                );
            }
        }
        assert!(
            pipelined_seen >= 1,
            "{sr}: default Studio should be pipelined"
        );
    }
}

#[test]
fn reported_latency_is_independent_of_tier_and_settings() {
    let sr = 48_000.0;
    for mode in MODES {
        for range in suite::RANGES {
            let cfg = EngineConfig {
                sample_rate: sr,
                latency_mode: mode,
                voice_range: range,
                ..EngineConfig::default()
            };
            let d = vt_engine::latency_samples(&cfg);
            let mut e = Engine::new(cfg);
            assert_eq!(e.latency_samples(), d);
            for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
                e.set_settings(&Settings {
                    tier,
                    ..Settings::default()
                });
                assert_eq!(e.latency_samples(), d, "{mode:?} {range:?} {tier:?}");
            }
        }
    }
    // Live+ and Studio follow their configured milliseconds.
    let studio = EngineConfig {
        sample_rate: sr,
        latency_mode: LatencyMode::Studio,
        voice_range: VoiceRange::Wide,
        studio_ms: 50.0,
        ..EngineConfig::default()
    };
    assert!(
        vt_engine::latency_samples(&studio)
            >= vt_engine::latency_samples(&EngineConfig {
                studio_ms: 20.0,
                ..studio
            })
    );
}
