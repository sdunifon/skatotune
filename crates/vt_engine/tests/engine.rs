//! Engine contract tests (DESIGN §3.5, §2.1, §2.2): block-size invariance,
//! measured latency == reported, NaN/Inf recovery, telemetry timing, MIDI
//! segmentation, analysis-channel selection, MIDI Learn forwarding and an
//! end-to-end correction check. The zero-allocation proof lives in
//! `tests/no_alloc.rs` (it needs its own global allocator).

mod common;

use common::{Rng, blocks, cents, engine, invariance_patterns, noise_bursts, pitch_hz, render, silence, vowel};
use vt_core::{
    AnalysisChannel, EngineConfig, EngineKind, LatencyMode, MidiEvent, MidiMode, NoteState, PitchFrame, QualityTier,
    ScaleKind, Settings, VoiceRange, frame_flags,
};
use vt_engine::{Engine, TimedMidi};

const MODES: [LatencyMode; 4] = [LatencyMode::LiveMin, LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio];
const RANGES: [VoiceRange; 4] = [VoiceRange::LowMale, VoiceRange::AltoTenor, VoiceRange::Soprano, VoiceRange::Wide];

fn cfg(sr: f32, mode: LatencyMode, channels: usize) -> EngineConfig {
    EngineConfig { sample_rate: sr, max_block: 4096, channels, latency_mode: mode, ..EngineConfig::default() }
}

/// Live-Min only runs Classic; everything else defaults to PSOLA.
fn settings(mode: LatencyMode) -> Settings {
    Settings {
        engine: if mode == LatencyMode::LiveMin { EngineKind::Classic } else { EngineKind::Natural },
        retune_ms: 0.0,
        ..Settings::default()
    }
}

/// ~1.6 s of varied material at `sr`: detuned notes with a glide and
/// vibrato, a noise burst (unvoiced), a gap and a second voice.
fn material(sr: f32) -> Vec<f32> {
    let mut x = silence(sr, 0.05);
    x.extend(vowel(sr, 0.45, 1.0, |t| 196.0 * 2f64.powf((25.0 + 60.0 * (t / 0.45)) / 1200.0)));
    x.extend(vowel(sr, 0.4, 1.0, |t| {
        220.0 * 2f64.powf((-20.0 + 45.0 * (std::f64::consts::TAU * 6.0 * t).sin()) / 1200.0)
    }));
    x.extend(noise_bursts(sr, 0.12, 0.06, 1.0, 0.05, 7));
    x.extend(silence(sr, 0.1));
    x.extend(vowel(sr, 0.35, 0.7, |_| 330.0 * 2f64.powf(-30.0 / 1200.0)));
    x.extend(silence(sr, 0.05));
    x
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

// ---------------------------------------------------------------------------
// Block-size invariance (DESIGN §2.1 rule 6, §3.5)
// ---------------------------------------------------------------------------

#[test]
fn output_is_block_size_invariant_for_every_mode() {
    let sr = 48_000.0;
    let x = material(sr);
    // MIDI at fixed absolute positions: Auto mode, a held note in the middle.
    let midi = [
        (20_011u64, MidiEvent::NoteOn { note: 57, velocity: 0.8 }),
        (20_011, MidiEvent::PitchBend { value: 0.25 }),
        (41_000, MidiEvent::NoteOff { note: 57 }),
    ];
    for mode in MODES {
        let s = Settings { midi_mode: MidiMode::Auto, ..settings(mode) };
        let c = cfg(sr, mode, 2);
        let total = x.len() + vt_engine::latency_samples(&c) as usize;
        let reference = render(&mut engine(c, &s), &x, &blocks(total, Some(64), 0), &midi);
        for p in invariance_patterns() {
            let out = render(&mut engine(c, &s), &x, &blocks(total, p, 0xb10c), &midi);
            for ch in 0..2 {
                let d = max_diff(&out[ch], &reference[ch]);
                assert!(d <= 1e-6, "{mode:?} blocks {p:?} ch {ch}: max diff {d}");
            }
        }
    }
}

#[test]
fn block_size_invariance_at_other_rates() {
    for (sr, mode) in
        [(44_100.0f32, LatencyMode::Live), (96_000.0, LatencyMode::Studio), (192_000.0, LatencyMode::Live)]
    {
        let x = material(sr);
        let c = cfg(sr, mode, 1);
        let s = settings(mode);
        let total = x.len() + vt_engine::latency_samples(&c) as usize;
        let reference = render(&mut engine(c, &s), &x, &blocks(total, Some(64), 0), &[]);
        for p in [Some(1), Some(7), Some(480), None] {
            let out = render(&mut engine(c, &s), &x, &blocks(total, p, 3), &[]);
            let d = max_diff(&out[0], &reference[0]);
            assert!(d <= 1e-6, "{sr} Hz {mode:?} blocks {p:?}: max diff {d}");
        }
    }
}

/// One `process()` call far larger than `max_block` (and than any internal
/// buffer) equals a 64-sample render.
#[test]
fn blocks_larger_than_max_block_are_handled() {
    let sr = 48_000.0;
    let x = material(sr);
    let c = EngineConfig { max_block: 256, ..cfg(sr, LatencyMode::Live, 1) };
    let s = settings(LatencyMode::Live);
    let total = x.len() + vt_engine::latency_samples(&c) as usize;
    let a = render(&mut engine(c, &s), &x, &blocks(total, Some(64), 0), &[]);
    let b = render(&mut engine(c, &s), &x, &[total], &[]);
    assert_eq!(a[0], b[0]);
}

/// MIDI is applied at the hop boundary that closes the hop containing it,
/// whatever block the event arrives in: moving an event within its hop
/// changes nothing (bit-identical output), moving it into the next hop
/// moves the target change one control hop later. (Checked on the
/// telemetry: the PSOLA fixes the ratio once per synthesis period, so a
/// one-hop shift of a ratio step need not change the audio.)
#[test]
fn midi_is_quantised_to_the_enclosing_hop() {
    let sr = 48_000.0;
    let x = vowel(sr, 0.8, 1.0, |_| 220.0 * 2f64.powf(30.0 / 1200.0));
    let c = cfg(sr, LatencyMode::Live, 1);
    let s = Settings { midi_mode: MidiMode::Target, ..settings(LatencyMode::Live) };
    let total = x.len() + vt_engine::latency_samples(&c) as usize;
    let hop = vt_core::hop_for_rate(sr) as u64;
    let k = 300; // hop index of the event
    let at = |pos: u64| vec![(pos, MidiEvent::NoteOn { note: 58, velocity: 1.0 })];
    // Output and the hop of the first telemetry frame with the MIDI target.
    let run = |blk: usize, pos: u64| {
        let mut e = engine(c, &s);
        e.set_telemetry_enabled(true);
        let mut rx = e.take_telemetry().expect("telemetry");
        let out = render(&mut e, &x, &blocks(total, Some(blk), 0), &at(pos));
        let frames: Vec<PitchFrame> = std::iter::from_fn(|| rx.pop().ok()).collect();
        (out, frames.iter().position(|f| f.target == 5800.0))
    };
    let (first, t_first) = run(100, k * hop);
    let (last, t_last) = run(37, k * hop + hop - 1);
    assert_eq!(first[0], last[0], "events inside one hop must act at the same boundary");
    assert_eq!(t_first, t_last);
    let (_, t_next) = run(100, (k + 1) * hop);
    assert!(t_first.is_some(), "the MIDI target never applied");
    assert_eq!(t_next, t_first.map(|i| i + 1), "an event in the next hop acts one hop later");
}

/// Engine/tier changes start at the next hop boundary, so a change requested
/// anywhere inside the same hop gives identical output.
#[test]
fn engine_switch_is_hop_aligned() {
    let sr = 48_000.0;
    let x = material(sr);
    let c = cfg(sr, LatencyMode::Live, 1);
    let s = settings(LatencyMode::Live);
    let switched = Settings { engine: EngineKind::Classic, tier: QualityTier::Eco, ..s };
    let run = |split: usize| {
        let mut e = engine(c, &s);
        let mut buf = x.clone();
        buf.extend(silence(sr, 0.01));
        let (a, b) = buf.split_at_mut(split);
        e.process(&mut [a], &[]);
        e.set_settings(&switched);
        e.process(&mut [b], &[]);
        buf
    };
    let hop = vt_core::hop_for_rate(sr);
    let base = 300 * hop;
    let a = run(base + 1);
    let b = run(base + hop - 3);
    assert_eq!(a, b);
    // A change one hop later is a different render.
    let c2 = run(base + hop + 1);
    assert_ne!(a, c2);
}

// ---------------------------------------------------------------------------
// Latency (DESIGN §3.5, §4)
// ---------------------------------------------------------------------------

/// Lag in `0..=max_lag` maximising the normalised cross-correlation of the
/// output against the input, and its value.
fn best_lag(x: &[f32], y: &[f32], max_lag: usize) -> (usize, f64) {
    let ex: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
    let mut best = (0, f64::MIN);
    for lag in 0..=max_lag {
        let (mut r, mut ey) = (0.0f64, 0.0f64);
        for (i, xv) in x.iter().enumerate() {
            if let Some(&yv) = y.get(i + lag) {
                r += *xv as f64 * yv as f64;
                ey += (yv as f64).powi(2);
            }
        }
        let c = if ex > 0.0 && ey > 0.0 { r / (ex * ey).sqrt() } else { 0.0 };
        if c > best.1 {
            best = (lag, c);
        }
    }
    best
}

#[test]
fn measured_latency_equals_reported_for_every_mode_range_and_rate() {
    for sr in [44_100.0f32, 48_000.0, 96_000.0] {
        let x = noise_bursts(sr, 0.5, 0.02, 0.1, 0.3, 11);
        for mode in MODES {
            for range in RANGES {
                let c = EngineConfig { voice_range: range, ..cfg(sr, mode, 1) };
                let d = vt_engine::latency_samples(&c) as usize;
                let mut e = engine(c, &settings(mode));
                assert_eq!(e.latency_samples() as usize, d);
                let out = render(&mut e, &x, &blocks(x.len() + d + 64, Some(97), 0), &[]);
                let (lag, corr) = best_lag(&x, &out[0], d + 64);
                assert_eq!(lag, d, "{sr} Hz {mode:?} {range:?}: measured {lag} vs reported {d} (corr {corr:.4})");
                assert!(corr > 0.999, "{sr} Hz {mode:?} {range:?}: correlation {corr}");
                assert_eq!(e.stats().voiced_hops, 0, "noise bursts must stay unvoiced");
            }
        }
    }
}

#[test]
fn latency_delegates_to_the_shifter_formula() {
    for sr in [44_100.0f32, 48_000.0, 88_200.0, 96_000.0, 192_000.0] {
        for mode in MODES {
            for range in RANGES {
                for (lp, st) in [(5.0, 20.0), (7.5, 33.0), (10.0, 50.0)] {
                    let c = EngineConfig { voice_range: range, live_plus_ms: lp, studio_ms: st, ..cfg(sr, mode, 2) };
                    let want = vt_shift::latency_samples(sr, mode, range, lp, st) as u32;
                    assert_eq!(vt_engine::latency_samples(&c), want);
                }
            }
        }
    }
    // Studio follows its milliseconds; Live+ never goes below Live.
    let studio =
        |ms| vt_engine::latency_samples(&EngineConfig { studio_ms: ms, ..cfg(48_000.0, LatencyMode::Studio, 2) });
    assert!(studio(50.0) > studio(20.0));
    let live = vt_engine::latency_samples(&cfg(48_000.0, LatencyMode::Live, 2));
    let plus =
        vt_engine::latency_samples(&EngineConfig { live_plus_ms: 5.0, ..cfg(48_000.0, LatencyMode::LivePlus, 2) });
    assert!(plus >= live);
}

#[test]
fn latency_does_not_depend_on_runtime_settings() {
    let c = cfg(48_000.0, LatencyMode::Live, 2);
    let mut e = Engine::new(c);
    let d = e.latency_samples();
    for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
        for kind in [EngineKind::Classic, EngineKind::Natural] {
            e.set_settings(&Settings { tier, engine: kind, ..Settings::default() });
            assert_eq!(e.latency_samples(), d);
        }
    }
}

// ---------------------------------------------------------------------------
// NaN / Inf / extreme input (DESIGN §2.1 rule 3)
// ---------------------------------------------------------------------------

#[test]
fn nan_inside_a_note_recovers() {
    let sr = 48_000.0;
    let mut clean = silence(sr, 0.05);
    clean.extend(vowel(sr, 0.5, 1.0, |_| 196.0 * 2f64.powf(20.0 / 1200.0)));
    clean.extend(silence(sr, 0.4));
    let second = clean.len();
    clean.extend(vowel(sr, 0.5, 1.0, |_| 220.0 * 2f64.powf(-25.0 / 1200.0)));
    clean.extend(silence(sr, 0.05));
    let end = second + (0.5 * sr) as usize;
    let mut dirty = clean.clone();
    let poison = (0.25 * sr) as usize;
    for (i, v) in dirty[poison..poison + 480].iter_mut().enumerate() {
        *v = match i % 3 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            _ => f32::NEG_INFINITY,
        };
    }
    for mode in MODES {
        let c = cfg(sr, mode, 2);
        let d = vt_engine::latency_samples(&c) as usize;
        let sizes = blocks(clean.len() + d, Some(128), 0);
        let a = render(&mut engine(c, &settings(mode)), &clean, &sizes, &[]);
        let b = render(&mut engine(c, &settings(mode)), &dirty, &sizes, &[]);
        assert!(b.iter().flatten().all(|v| v.is_finite()), "{mode:?}: non-finite output");
        // The second note (after a 400 ms gap) comes out as it would have.
        let (pa, pb) = (&a[0][second + d..end + d], &b[0][second + d..end + d]);
        let e: f64 = pa.iter().zip(pb).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let p: f64 = pa.iter().map(|v| (*v as f64).powi(2)).sum();
        assert!(p > 0.0 && e / p < 1e-4, "{mode:?}: no recovery ({:.1} dB)", 10.0 * (e / p).log10());
    }
}

#[test]
fn nan_and_inf_in_silence_equal_silence() {
    let sr = 48_000.0;
    let mut clean = silence(sr, 0.1);
    clean.extend(vowel(sr, 0.3, 1.0, |_| 247.0));
    let mut dirty = clean.clone();
    for (i, v) in dirty[100..4000].iter_mut().enumerate() {
        *v = if i % 2 == 0 { f32::NAN } else { f32::NEG_INFINITY };
    }
    for mode in MODES {
        let c = cfg(sr, mode, 1);
        let sizes = blocks(clean.len() + 4000, None, 5);
        let a = render(&mut engine(c, &settings(mode)), &clean, &sizes, &[]);
        let b = render(&mut engine(c, &settings(mode)), &dirty, &sizes, &[]);
        assert_eq!(a[0], b[0], "{mode:?}");
    }
}

#[test]
fn absurd_magnitudes_never_produce_non_finite_output() {
    let sr = 48_000.0;
    let mut x: Vec<f32> = (0..4800).map(|i| if i % 3 == 0 { 1.0e30 } else { -1.0e30 }).collect();
    x.extend((0..4800).map(|i| if i % 2 == 0 { f32::MAX } else { f32::MIN }));
    x.extend(silence(sr, 0.4));
    let tail = x.len();
    x.extend(vowel(sr, 0.4, 1.0, |_| 200.0));
    for mode in MODES {
        let c = cfg(sr, mode, 2);
        let out = render(&mut engine(c, &settings(mode)), &x, &blocks(x.len() + 3000, Some(512), 0), &[]);
        assert!(out.iter().flatten().all(|v| v.is_finite()), "{mode:?}");
        // After the garbage and the gap the voice is processed normally.
        let late = &out[0][tail + 4800..];
        assert!(late.iter().any(|v| v.abs() > 0.05), "{mode:?}: no output after recovery");
    }
}

// ---------------------------------------------------------------------------
// Telemetry and stats (DESIGN §2.2, §3.5)
// ---------------------------------------------------------------------------

#[test]
fn one_telemetry_frame_per_hop_only_while_enabled() {
    for sr in [44_100.0f32, 48_000.0, 96_000.0, 192_000.0] {
        let hop = vt_core::hop_for_rate(sr) as u64;
        let c = cfg(sr, LatencyMode::Live, 1);
        let mut e = engine(c, &settings(LatencyMode::Live));
        let mut rx = e.take_telemetry().expect("consumer");
        assert!(e.take_telemetry().is_none(), "telemetry is handed out once");
        let x = material(sr);
        let mut pos = 0usize;
        let mut frames = 0u64;
        let mut enabled_hops = 0u64;
        let mut buf = x.clone();
        // Disabled for the first half, enabled for the second.
        for n in blocks(x.len(), None, 2) {
            let on = pos >= x.len() / 2;
            e.set_telemetry_enabled(on);
            let before = e.stats().hops;
            e.process(&mut [&mut buf[pos..pos + n]], &[]);
            if on {
                enabled_hops += e.stats().hops - before;
            }
            pos += n;
            while rx.pop().is_ok() {
                frames += 1;
            }
        }
        let st = e.stats();
        assert_eq!(st.hops, x.len() as u64 / hop, "{sr}");
        assert_eq!(frames, enabled_hops, "{sr}");
        assert!(enabled_hops > 0 && enabled_hops < st.hops);
        assert_eq!(st.telemetry_dropped, 0);
        assert!(st.voiced_hops > 0 && st.voiced_hops < st.hops);
    }
}

#[test]
fn full_telemetry_ring_drops_and_counts() {
    let sr = 48_000.0;
    let mut e = engine(cfg(sr, LatencyMode::Live, 1), &settings(LatencyMode::Live));
    let _rx = e.take_telemetry();
    e.set_telemetry_enabled(true);
    let hop = vt_core::hop_for_rate(sr);
    let mut buf = vec![0.0f32; hop * 9000];
    e.process(&mut [&mut buf[..]], &[]);
    let st = e.stats();
    assert_eq!(st.hops, 9000);
    assert_eq!(st.telemetry_dropped, 9000 - 8192);
}

/// Telemetry timestamps are output time: the frame describes the input that
/// is heard when its control takes effect. In Studio that is exactly the
/// newest input sample at the hop boundary where the control is applied
/// (every frame one hop after the previous one, the last on the final
/// boundary); in Live it is ahead of the newest sample by D minus the
/// detector's information lag.
#[test]
fn telemetry_positions_are_output_time() {
    let sr = 48_000.0;
    let hop = vt_core::hop_for_rate(sr) as u64;
    let x = vowel(sr, 0.6, 1.0, |_| 220.0 * 2f64.powf(30.0 / 1200.0));
    for mode in [LatencyMode::Live, LatencyMode::Studio] {
        let c = cfg(sr, mode, 1);
        let d = vt_engine::latency_samples(&c) as u64;
        let mut e = engine(c, &settings(mode));
        let mut rx = e.take_telemetry().expect("consumer");
        e.set_telemetry_enabled(true);
        let mut buf = x.clone();
        e.process(&mut [&mut buf[..]], &[]);
        let mut k = 0u64;
        let mut voiced = 0;
        let mut last_pos = None;
        while let Ok(f) = rx.pop() {
            k += 1;
            let newest = k * hop - 1;
            // (Early frames are clamped at D until the detector has D of history.)
            if mode == LatencyMode::Studio && newest > 2 * d {
                assert_eq!((f.sample_pos + 1) % hop, 0, "frame {k}: {}", f.sample_pos);
                if let Some(p) = last_pos {
                    assert_eq!(f.sample_pos, p + hop, "frame {k}");
                }
                last_pos = Some(f.sample_pos);
            }
            if mode == LatencyMode::Live {
                // info_lag is at most a few periods of the lowest note.
                let t_max = (sr / VoiceRange::AltoTenor.f_min()) as u64;
                assert!(
                    f.sample_pos <= newest + d && f.sample_pos + 3 * t_max >= newest + d.min(newest),
                    "frame {k}: {}",
                    f.sample_pos
                );
            }
            if f.flags & frame_flags::VOICED != 0 {
                voiced += 1;
                // Chromatic, retune 0: detected ≈ +30 c over A3, target A3.
                assert!((f.target - 5700.0).abs() < 0.5, "target {}", f.target);
                if voiced > 40 {
                    assert!((f.detected - 5730.0).abs() < 8.0, "{mode:?} detected {}", f.detected);
                    assert!((f.output - 5700.0).abs() < 1.0, "{mode:?} output {}", f.output);
                }
            }
        }
        assert!(voiced > 200, "{mode:?}: {voiced} voiced frames");
        if mode == LatencyMode::Studio {
            let last_boundary = (x.len() as u64 / hop) * hop - 1;
            assert_eq!(last_pos, Some(last_boundary));
            assert!(d > 0);
        }
    }
}

// ---------------------------------------------------------------------------
// End-to-end correction
// ---------------------------------------------------------------------------

/// A steady vowel 35 c sharp of A3 comes out on A3 in every mode and engine
/// (chromatic, retune 0); with Amount 0 it keeps its pitch.
#[test]
fn detuned_vowel_is_corrected() {
    let sr = 48_000.0;
    let x = vowel(sr, 1.0, 1.0, |_| 220.0 * 2f64.powf(35.0 / 1200.0));
    let cases = [
        (LatencyMode::LiveMin, EngineKind::Classic),
        (LatencyMode::Live, EngineKind::Natural),
        (LatencyMode::Live, EngineKind::Classic),
        (LatencyMode::LivePlus, EngineKind::Natural),
        (LatencyMode::Studio, EngineKind::Natural),
    ];
    for (mode, kind) in cases {
        for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
            let c = cfg(sr, mode, 1);
            let d = vt_engine::latency_samples(&c) as usize;
            let s = Settings { engine: kind, tier, ..settings(mode) };
            let out = render(&mut engine(c, &s), &x, &blocks(x.len() + d, Some(256), 0), &[]);
            let seg = &out[0][d + 24_000..d + 43_200];
            let p = cents(pitch_hz(seg, sr, 150.0, 400.0).expect("pitched output"));
            assert!((p - 5700.0).abs() < 3.0, "{mode:?} {kind:?} {tier:?}: output at {p:.1} c");
        }
        let c = cfg(sr, mode, 1);
        let d = vt_engine::latency_samples(&c) as usize;
        let s = Settings { engine: kind, amount: 0.0, ..settings(mode) };
        let out = render(&mut engine(c, &s), &x, &blocks(x.len() + d, Some(256), 0), &[]);
        let p = cents(pitch_hz(&out[0][d + 24_000..d + 43_200], sr, 150.0, 400.0).expect("pitched"));
        assert!((p - 5735.0).abs() < 3.0, "{mode:?} {kind:?} amount 0: {p:.1} c");
    }
}

/// The scale decides the target: in C major a vowel 40 c below F is pulled
/// to F; with F removed it goes to the nearest other legal note (E).
#[test]
fn scale_and_note_states_reach_the_control_law() {
    let sr = 48_000.0;
    let f = 349.228 * 2f64.powf(-40.0 / 1200.0); // F4 − 40 c
    let x = vowel(sr, 0.9, 1.0, move |_| f);
    let c = cfg(sr, LatencyMode::Studio, 1);
    let d = vt_engine::latency_samples(&c) as usize;
    let major = Settings {
        key: 0,
        scale: ScaleKind::Major,
        note_states: ScaleKind::Major.note_states(0),
        ..settings(LatencyMode::Studio)
    };
    let mut no_f = major;
    no_f.note_states[5] = NoteState::Remove;
    for (s, want) in [(major, 6500.0), (no_f, 6400.0)] {
        let out = render(&mut engine(c, &s), &x, &blocks(x.len() + d, Some(128), 0), &[]);
        let p = cents(pitch_hz(&out[0][d + 19_200..d + 38_400], sr, 200.0, 600.0).expect("pitched"));
        assert!((p - want).abs() < 3.0, "want {want}, got {p:.1}");
    }
}

// ---------------------------------------------------------------------------
// Channels and the analysis tap
// ---------------------------------------------------------------------------

#[test]
fn analysis_channel_selects_the_pitch_source() {
    let sr = 48_000.0;
    let voice = vowel(sr, 0.6, 1.0, |_| 210.0);
    let quiet = vec![0.0f32; voice.len()];
    let c = cfg(sr, LatencyMode::Live, 2);
    let d = vt_engine::latency_samples(&c) as usize;
    for (ch, voiced) in [(AnalysisChannel::Left, true), (AnalysisChannel::Mix, true), (AnalysisChannel::Right, false)] {
        let s = Settings { analysis_channel: ch, ..settings(LatencyMode::Live) };
        let mut e = engine(c, &s);
        let (mut l, mut r) = (voice.clone(), quiet.clone());
        l.extend(silence(sr, 0.01));
        r.extend(silence(sr, 0.01));
        e.process(&mut [&mut l[..], &mut r[..]], &[]);
        assert_eq!(e.stats().voiced_hops > 0, voiced, "{ch:?}");
        if !voiced {
            // Unvoiced analysis: both channels are the exact delayed input.
            for i in d..voice.len() {
                assert_eq!(l[i].to_bits(), voice[i - d].to_bits(), "{ch:?} at {i}");
            }
        }
    }
}

#[test]
fn stereo_channels_share_one_grain_schedule() {
    let sr = 48_000.0;
    let x = material(sr);
    for mode in MODES {
        let c = cfg(sr, mode, 2);
        let out = render(&mut engine(c, &settings(mode)), &x, &blocks(x.len() + 3000, None, 1), &[]);
        assert_eq!(out[0], out[1], "{mode:?}");
    }
}

#[test]
fn channel_mismatches_are_tolerated() {
    let sr = 48_000.0;
    let x = material(sr);
    // Mono engine, stereo buffers: the extra channel is left untouched.
    let mut e = engine(cfg(sr, LatencyMode::Live, 1), &settings(LatencyMode::Live));
    let (mut l, mut r) = (x.clone(), x.clone());
    e.process(&mut [&mut l[..], &mut r[..]], &[]);
    assert_eq!(r, x);
    assert_ne!(l, x);
    // Stereo engine, one buffer; and buffers of different lengths.
    let mut e = engine(cfg(sr, LatencyMode::Live, 2), &settings(LatencyMode::Live));
    let mut m = x.clone();
    e.process(&mut [&mut m[..]], &[]);
    let mut short = vec![0.1f32; 100];
    let mut long = vec![0.1f32; 300];
    e.process(&mut [&mut short[..], &mut long[..]], &[]);
    assert!(long[100..].iter().all(|v| *v == 0.1), "only the common prefix is processed");
    // No channels at all: nothing happens, nothing panics.
    e.process(&mut [], &[TimedMidi { timing: 0, event: MidiEvent::AllNotesOff }]);
    // Absurd configs are clamped.
    let e = Engine::new(EngineConfig { channels: 9, ..cfg(sr, LatencyMode::Live, 2) });
    assert_eq!(e.config().channels, 9, "config() reports what was asked for");
    let mut e = Engine::new(EngineConfig { channels: 0, ..cfg(sr, LatencyMode::Live, 1) });
    let mut b = vec![0.2f32; 64];
    e.process(&mut [&mut b[..]], &[]);
    assert_eq!(e.stats().hops, 1);
}

// ---------------------------------------------------------------------------
// MIDI Learn, determinism, reset
// ---------------------------------------------------------------------------

#[test]
fn learned_states_are_forwarded() {
    let sr = 48_000.0;
    let mut e = engine(cfg(sr, LatencyMode::Live, 1), &Settings { midi_mode: MidiMode::Learn, ..Settings::default() });
    assert_eq!(e.learned_states(), None);
    let midi = [
        TimedMidi { timing: 3, event: MidiEvent::NoteOn { note: 64, velocity: 1.0 } },
        TimedMidi { timing: 70, event: MidiEvent::NoteOn { note: 67, velocity: 1.0 } },
    ];
    let mut buf = vec![0.0f32; 256];
    e.process(&mut [&mut buf[..]], &midi);
    let states = e.learned_states().expect("learned scale");
    for (pc, st) in states.iter().enumerate() {
        let want = if pc == 4 || pc == 7 { NoteState::On } else { NoteState::Remove };
        assert_eq!(*st, want, "pitch class {pc}");
    }
    e.set_settings(&Settings::default());
    assert_eq!(e.learned_states(), None, "only reported in Learn mode");
}

#[test]
fn renders_are_deterministic_and_reset_restores_state() {
    let sr = 48_000.0;
    let x = material(sr);
    for mode in MODES {
        let c = cfg(sr, mode, 1);
        let s = settings(mode);
        let sizes = blocks(x.len() + 3000, None, 9);
        let mut e = engine(c, &s);
        let a = render(&mut e, &x, &sizes, &[]);
        let stats = e.stats();
        e.reset();
        assert_eq!(e.stats(), vt_engine::EngineStats::default());
        e.set_settings(&s);
        let b = render(&mut e, &x, &sizes, &[]);
        assert_eq!(a, b, "{mode:?}: reset did not restore the initial state");
        assert_eq!(e.stats(), stats);
        let c2 = render(&mut engine(c, &s), &x, &sizes, &[]);
        assert_eq!(a, c2, "{mode:?}: fresh engines differ");
    }
}

/// Settings changes mid-stream (tier, engine, retune, scale, MIDI mode) stay
/// finite and never break the stream.
#[test]
fn settings_can_change_every_block() {
    let sr = 48_000.0;
    let x = material(sr);
    let mut rng = Rng::new(42);
    for mode in MODES {
        let mut e = engine(cfg(sr, mode, 2), &settings(mode));
        let (mut l, mut r) = (x.clone(), x.clone());
        let mut pos = 0;
        for n in blocks(x.len(), None, 4) {
            let s = Settings {
                tier: [QualityTier::Eco, QualityTier::Hq, QualityTier::Max][rng.range(0, 2)],
                engine: if rng.range(0, 1) == 0 { EngineKind::Classic } else { EngineKind::Natural },
                retune_ms: rng.range(0, 100) as f32,
                scale: ScaleKind::ALL[rng.range(0, 12)],
                midi_mode: [MidiMode::Off, MidiMode::Target, MidiMode::Learn, MidiMode::Auto][rng.range(0, 3)],
                analysis_channel: [AnalysisChannel::Left, AnalysisChannel::Right, AnalysisChannel::Mix]
                    [rng.range(0, 2)],
                tracking: rng.range(0, 100) as f32,
                ..Settings::default()
            };
            e.set_settings(&s);
            e.process(&mut [&mut l[pos..pos + n], &mut r[pos..pos + n]], &[]);
            pos += n;
        }
        assert!(l.iter().chain(&r).all(|v| v.is_finite()), "{mode:?}");
    }
}

// ---------------------------------------------------------------------------
// Studio lookahead (CL-3) and vibrato depth with Preserve off (CL-1)
// ---------------------------------------------------------------------------

/// Telemetry frames of a render (enabled from the start).
fn telemetry(c: EngineConfig, s: &Settings, x: &[f32]) -> Vec<vt_core::PitchFrame> {
    let mut e = engine(c, s);
    let mut rx = e.take_telemetry().expect("consumer");
    e.set_telemetry_enabled(true);
    let d = vt_engine::latency_samples(&c) as usize;
    let mut frames = Vec::new();
    let mut buf = x.to_vec();
    buf.extend(std::iter::repeat_n(0.0, d));
    for chunk in buf.chunks_mut(128) {
        e.process(&mut [chunk], &[]);
        while let Ok(f) = rx.pop() {
            frames.push(f);
        }
    }
    frames
}

/// With D split between detector and corrector, Studio estimates still
/// describe the audio being output: the detected pitch of a step changes at
/// the step's output time (within the detector's residual step lag), for
/// every voice range and rate. A wrong lead would move it by whole hops.
#[test]
fn studio_estimates_line_up_with_the_audio() {
    for sr in [44_100.0f32, 48_000.0, 96_000.0] {
        for (range, f0) in [
            (VoiceRange::LowMale, 110.0),
            (VoiceRange::AltoTenor, 220.0),
            (VoiceRange::Soprano, 440.0),
            (VoiceRange::Wide, 220.0),
        ] {
            let c = EngineConfig { voice_range: range, ..cfg(sr, LatencyMode::Studio, 1) };
            let d = vt_engine::latency_samples(&c) as f64;
            let step_s = 0.6;
            let x = vowel(sr, 1.2, 1.0, |t| f0 * if t < step_s { 1.0 } else { 2f64.powf(300.0 / 1200.0) });
            let frames = telemetry(c, &settings(LatencyMode::Studio), &x);
            let mid = cents(f0) + 150.0;
            let hit = frames
                .iter()
                .find(|f| f.detected as f64 > mid && (f.sample_pos as f64 - d) / sr as f64 > 0.3)
                .expect("step seen");
            let lag_ms = ((hit.sample_pos as f64 - d) / sr as f64 - step_s) * 1e3;
            assert!(
                (-1.5..=4.0).contains(&lag_ms),
                "{sr} {range:?}: detected step {lag_ms:.2} ms after the audio step"
            );
        }
    }
}

/// CL-3: Studio decides each frame with lookahead, so a note change switches
/// the target at the singer's crossing (Live, deciding causally, switches
/// later), and a short onset scoop starts on the intended note instead of
/// the scoop's.
#[test]
fn studio_note_switches_use_the_lookahead() {
    let sr = 48_000.0;
    let hop = vt_core::hop_for_rate(sr) as f64;
    // Legato step A3 + 20 c → A#3 + 20 c at 0.6 s, retune 0, chromatic.
    let step_s = 0.6;
    let x = vowel(sr, 1.2, 1.0, |t| 220.0 * 2f64.powf(if t < step_s { 20.0 } else { 120.0 } / 1200.0));
    let switch_ms = |mode: LatencyMode| {
        let c = cfg(sr, mode, 1);
        let d = vt_engine::latency_samples(&c) as f64;
        let frames = telemetry(c, &settings(mode), &x);
        let f = frames
            .iter()
            .find(|f| (f.target - 5800.0).abs() < 0.5 && (f.sample_pos as f64 - d) / sr as f64 > 0.3)
            .expect("switch");
        ((f.sample_pos as f64 - d) / sr as f64 - step_s) * 1e3
    };
    let studio = switch_ms(LatencyMode::Studio);
    let live = switch_ms(LatencyMode::Live);
    let hop_ms = hop / sr as f64 * 1e3;
    // (Measured: Studio 0.0 ms, Live 5.0 ms; Studio without lookahead 4.0 ms.)
    assert!(studio.abs() <= hop_ms, "Studio switched {studio:.2} ms from the step");
    assert!(live > studio + 2.0 * hop_ms, "Live {live:.2} ms vs Studio {studio:.2} ms");

    // Onset with a 12 ms scoop up from G#3 into A3: Studio's onset note comes
    // from the lookahead median, so it starts on A3 and never switches.
    let mut x = silence(sr, 0.2);
    x.extend(vowel(sr, 0.6, 1.0, |t| 220.0 * 2f64.powf((-100.0 * (1.0 - t / 0.012).max(0.0)) / 1200.0)));
    let frames = telemetry(cfg(sr, LatencyMode::Studio, 1), &settings(LatencyMode::Studio), &x);
    let voiced: Vec<_> = frames.iter().filter(|f| f.target != 0.0).collect();
    let changes = voiced.windows(2).filter(|w| (w[0].target - w[1].target).abs() > 1.0).count();
    assert_eq!((voiced.first().map(|f| f.target), changes), (Some(5700.0), 0));
}

/// CL-1, end to end: Preserve Vibrato off with Vibrato Depth 0 % at retune 0
/// hard-tunes the vibrato flat. It used to come out as a full-extent
/// anti-phase vibrato (the correction cancelled the vibrato and the depth
/// scaling then subtracted it again).
#[test]
fn preserve_off_depth_zero_outputs_no_vibrato() {
    let sr = 48_000.0;
    let x =
        vowel(sr, 1.6, 1.0, |t| 220.0 * 2f64.powf((25.0 + 40.0 * (std::f64::consts::TAU * 6.0 * t).sin()) / 1200.0));
    for (tier, depth) in [(QualityTier::Hq, 0.0f32), (QualityTier::Eco, 0.0), (QualityTier::Hq, 2.0)] {
        let c = cfg(sr, LatencyMode::Live, 1);
        let d = vt_engine::latency_samples(&c) as usize;
        let s = Settings { tier, preserve_vibrato: false, vibrato_depth: depth, ..settings(LatencyMode::Live) };
        let out = render(&mut engine(c, &s), &x, &blocks(x.len() + d, Some(128), 0), &[]);
        let win = (0.025 * sr) as usize;
        let pitches: Vec<f64> = (0..80)
            .filter_map(|i| {
                let a = d + (0.6 * sr) as usize + i * (0.01 * sr) as usize;
                pitch_hz(&out[0][a..a + win], sr, 150.0, 400.0).map(cents)
            })
            .collect();
        assert!(pitches.len() > 70);
        let (lo, hi) = pitches.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        assert!(hi - lo < 16.0, "{tier:?} depth {depth}: output vibrato {:.1} c peak to peak", hi - lo);
        assert!((0.5 * (lo + hi) - 5700.0).abs() < 4.0, "{tier:?} depth {depth}: centre {:.1}", 0.5 * (lo + hi));
    }
}

// ---------------------------------------------------------------------------
// Voice range at runtime (HI-3)
// ---------------------------------------------------------------------------

/// HI-3: `set_voice_range` applies a range whose reported latency is the
/// same without rebuilding the engine (Live-Min, Live+ and Studio at their
/// defaults: every range; Live: Low Male ↔ Wide), and refuses (returns
/// false, changes nothing) when D would change, so the plugin knows a
/// restart is needed. It used to need a host restart in every case.
#[test]
fn voice_range_changes_that_keep_the_latency_apply_at_runtime() {
    for sr in [44_100.0f32, 48_000.0, 96_000.0] {
        for mode in MODES {
            for from in RANGES {
                for to in RANGES {
                    let c = EngineConfig { voice_range: from, ..cfg(sr, mode, 2) };
                    let d = vt_engine::latency_samples(&c);
                    let same = vt_engine::latency_samples(&EngineConfig { voice_range: to, ..c }) == d;
                    let mut e = Engine::new(c);
                    let ok = e.set_voice_range(to);
                    assert_eq!(ok, same, "{sr} {mode:?} {from:?} → {to:?}");
                    assert_eq!(e.config().voice_range, if ok { to } else { from });
                    assert_eq!(e.latency_samples(), d);
                    // Every switch the latency allows is available.
                    if mode != LatencyMode::Live {
                        assert!(ok, "{sr} {mode:?} {from:?} → {to:?}");
                    }
                }
            }
        }
    }
}

/// A C2 singer (65 Hz) on an Alto/Tenor engine (f_min 80 Hz) is not
/// corrected; switching to Low Male mid-stream corrects them, the output
/// fades out and back in around the switch (no step: the largest
/// sample-to-sample change stays within the signal's own), and telemetry
/// positions keep increasing.
#[test]
fn switching_the_voice_range_mid_stream_takes_effect_cleanly() {
    let sr = 48_000.0;
    let f0 = 65.406 * 2f64.powf(30.0 / 1200.0);
    let x = vowel(sr, 3.0, 1.0, |_| f0);
    for mode in [LatencyMode::Studio, LatencyMode::LivePlus] {
        let c = EngineConfig { voice_range: VoiceRange::AltoTenor, ..cfg(sr, mode, 1) };
        let d = vt_engine::latency_samples(&c) as usize;
        let mut e = engine(c, &Settings { retune_ms: 0.0, ..settings(mode) });
        e.set_telemetry_enabled(true);
        let mut rx = e.take_telemetry().expect("telemetry");
        let mut y = x.clone();
        y.resize(x.len() + d, 0.0);
        let switch_at = 64 * 750; // 1 s
        let mut pos = 0;
        for chunk in y.chunks_mut(64) {
            if pos == switch_at {
                assert!(e.set_voice_range(VoiceRange::LowMale));
            }
            e.process(&mut [chunk], &[]);
            pos += 64;
        }
        assert_eq!(e.config().voice_range, VoiceRange::LowMale);
        // Before: uncorrected (+30 c); after: on C2.
        let before = cents(pitch_hz(&y[d + 24_000..d + 43_200], sr, 50.0, 200.0).expect("pitched"));
        assert!((before - 3630.0).abs() < 5.0, "{mode:?}: before {before:.1} c");
        let after = cents(pitch_hz(&y[d + 100_000..d + 136_000], sr, 50.0, 200.0).expect("pitched"));
        assert!((after - 3600.0).abs() < 5.0, "{mode:?}: after {after:.1} c");
        // The transition is a fade, not a step.
        let max_step = |s: &[f32]| s.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        let normal = max_step(&y[d + 24_000..d + 43_200]);
        let around = max_step(&y[switch_at..switch_at + 2 * d + 4_800]);
        assert!(around <= 1.2 * normal, "{mode:?}: step {around} vs {normal}");
        // Telemetry keeps the stream's clock: no frame after the switch is
        // placed back near the start of the stream.
        let frames: Vec<u64> = std::iter::from_fn(|| rx.pop().ok()).map(|f| f.sample_pos).collect();
        let hop = vt_core::hop_for_rate(sr);
        let after = &frames[switch_at / hop + 1..];
        let lowest = after.iter().copied().min().unwrap_or(0);
        assert!(lowest + 2 * d as u64 > switch_at as u64, "{mode:?}: frame at {lowest} after the switch at {switch_at}");
        assert!(after.last().is_some_and(|&p| p as usize + d + 4 * hop > y.len()), "{mode:?}: {:?}", after.last());
    }
}

/// A voice-range switch requested at the same sample position gives the
/// same output whatever the host's block sizes (DESIGN §2.1 rule 6).
#[test]
fn voice_range_switches_are_block_size_invariant() {
    let sr = 48_000.0;
    let x = vowel(sr, 1.5, 1.0, |t| 110.0 * 2f64.powf((20.0 + 30.0 * (t * 3.0).sin()) / 1200.0));
    let c = EngineConfig { voice_range: VoiceRange::Wide, ..cfg(sr, LatencyMode::Studio, 1) };
    let d = vt_engine::latency_samples(&c) as usize;
    let at = 64 * 300;
    let run = |blk: usize| {
        let mut e = engine(c, &settings(LatencyMode::Studio));
        let mut y = x.clone();
        y.resize(x.len() + d, 0.0);
        let mut pos = 0;
        for chunk in y.chunks_mut(blk) {
            if pos == at {
                assert!(e.set_voice_range(VoiceRange::AltoTenor));
            }
            e.process(&mut [chunk], &[]);
            pos += chunk.len();
        }
        y
    };
    let a = run(64);
    for blk in [16usize, 32, 192] {
        assert_eq!(a, run(blk), "block {blk}");
    }
}

/// Voice-range requests that arrive mid-transition (a change cancelled
/// while fading out, another one while fading back in) keep the gain
/// continuous and end on the last requested range.
#[test]
fn overlapping_voice_range_changes_stay_click_free() {
    let sr = 48_000.0;
    let x = vowel(sr, 3.0, 1.0, |t| 196.0 * 2f64.powf((15.0 + 20.0 * (t * 5.0).sin()) / 1200.0));
    let c = EngineConfig { voice_range: VoiceRange::AltoTenor, ..cfg(sr, LatencyMode::Studio, 1) };
    let d = vt_engine::latency_samples(&c) as usize;
    let mut e = engine(c, &settings(LatencyMode::Studio));
    let mut y = x.clone();
    y.resize(x.len() + d, 0.0);
    let requests = [
        (48_000usize, VoiceRange::LowMale),
        (48_000 + 96, VoiceRange::AltoTenor), // cancelled 2 ms into the fade-out
        (72_000, VoiceRange::Wide),
        (72_000 + 240 + d / 2, VoiceRange::Soprano), // during the fade-in wait
        (100_000, VoiceRange::LowMale),
        (100_000 + 240 + d + 96, VoiceRange::Wide), // 2 ms into the fade-in
    ];
    let mut pos = 0;
    for chunk in y.chunks_mut(32) {
        for &(at, r) in &requests {
            if pos == at / 32 * 32 {
                assert!(e.set_voice_range(r));
            }
        }
        e.process(&mut [chunk], &[]);
        pos += 32;
    }
    assert_eq!(e.config().voice_range, VoiceRange::Wide);
    let max_step = |s: &[f32]| s.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
    let normal = max_step(&y[d + 4_800..d + 43_200]);
    let around = max_step(&y[d + 43_200..d + 120_000]);
    assert!(around <= 1.2 * normal, "step {around} vs {normal}");
    assert!(y.iter().all(|v| v.is_finite()));
}
