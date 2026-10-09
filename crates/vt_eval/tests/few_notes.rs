//! End-to-end check with only one or two legal notes: steady tones 0.5 to 10
//! semitones away from the enabled notes must come out on the nearest enabled
//! note (any octave), in every engine, for pure sines and harmonic tones.
//! With the other notes on Bypass instead of Remove, the tone must pass
//! through unshifted (the documented Bypass semantics).

use vt_core::{EngineKind, LatencyMode, NoteState, QualityTier, Settings};
use vt_eval::reference;
use vt_eval::render::{RenderOptions, render};

const SR: f32 = 48_000.0;
/// C4 in absolute cents.
const C4: f64 = 6000.0;

/// A steady tone at `cents` with a 10 ms fade-in: a pure sine, or a
/// band-limited harmonic tone with a 1/k^1.2 spectrum.
fn tone(cents: f64, secs: f64, harmonic: bool) -> Vec<f32> {
    let f0 = 440.0 * 2f64.powf((cents - 6900.0) / 1200.0);
    let n = (secs * f64::from(SR)) as usize;
    let fade = 0.01 * f64::from(SR);
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            phase += std::f64::consts::TAU * f0 / f64::from(SR);
            let s = if harmonic {
                let mut s = 0.0;
                let mut k = 1.0;
                while k * f0 < 6000.0 {
                    s += (k * phase).sin() / k.powf(1.2);
                    k += 1.0;
                }
                0.25 * s
            } else {
                0.4 * phase.sin()
            };
            (s * (i as f64 / fade).min(1.0)) as f32
        })
        .collect()
}

/// Median output pitch (cents) over 0.6-1.4 s of the latency-aligned output,
/// and the share of those frames that are voiced.
fn steady_output_cents(y: &[f32]) -> (f64, f64) {
    let track = reference::track(y, SR);
    let (lo, hi) = ((0.6 * SR) as usize, (1.4 * SR) as usize);
    let window: Vec<_> = track.frames.iter().filter(|f| f.pos >= lo && f.pos < hi).collect();
    let mut voiced: Vec<f64> = window.iter().filter(|f| f.voiced).map(|f| f.cents).collect();
    voiced.sort_by(f64::total_cmp);
    let median = voiced.get(voiced.len() / 2).copied().unwrap_or(f64::NAN);
    (median, voiced.len() as f64 / window.len().max(1) as f64)
}

/// Nearest enabled note to `cents` across octaves.
fn nearest_enabled(cents: f64, enabled: &[usize]) -> f64 {
    (2..=8)
        .flat_map(|oct| enabled.iter().map(move |&pc| 1200.0 * f64::from(oct) + 100.0 * pc as f64))
        .min_by(|a, b| (a - cents).abs().total_cmp(&(b - cents).abs()))
        .unwrap_or(f64::NAN)
}

fn states(enabled: &[usize], others: NoteState) -> [NoteState; 12] {
    let mut s = [others; 12];
    for &pc in enabled {
        s[pc] = NoteState::On;
    }
    s
}

const CONFIGS: [(LatencyMode, QualityTier, EngineKind); 4] = [
    (LatencyMode::Live, QualityTier::Hq, EngineKind::Natural),
    (LatencyMode::Live, QualityTier::Eco, EngineKind::Natural),
    (LatencyMode::LiveMin, QualityTier::Hq, EngineKind::Classic),
    (LatencyMode::Studio, QualityTier::Hq, EngineKind::Natural),
];

/// Offsets in semitones above C4. 6.0 is skipped for the one-note set because
/// the tritone is equidistant from both Cs.
const OFFSETS: [f64; 11] = [0.5, 1.0, 2.0, 3.0, 3.4, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0];

fn hard_tune(states: [NoteState; 12], tier: QualityTier, engine: EngineKind) -> Settings {
    Settings {
        tier,
        engine,
        note_states: states,
        retune_ms: 0.0,
        vibrato_depth: 0.0,
        tolerance_cents: 0.0,
        ..Settings::default()
    }
}

#[test]
fn far_off_tones_snap_to_the_few_enabled_notes() {
    let mut failures = Vec::new();
    for enabled in [&[0usize][..], &[0, 7][..]] {
        for harmonic in [false, true] {
            for (mode, tier, engine) in CONFIGS {
                for off in OFFSETS {
                    let input = C4 + 100.0 * off;
                    let expect = nearest_enabled(input, enabled);
                    let settings = hard_tune(states(enabled, NoteState::Remove), tier, engine);
                    let opts = RenderOptions::new(SR, mode, tier).with_settings(settings);
                    let rendered = render(&tone(input, 1.5, harmonic), &opts);
                    let (out, voiced) = steady_output_cents(rendered.aligned(0));
                    let ok = (out - expect).abs() < 5.0 && voiced > 0.95;
                    if !ok {
                        failures.push(format!(
                            "{mode:?} {tier:?} {engine:?} enabled {enabled:?} harmonic {harmonic}: \
                             in {input:.0} c, expected {expect:.0} c, got {out:.1} c ({:.0} % voiced)",
                            voiced * 100.0
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} misses:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn bypassed_neighbours_pass_tones_through_unshifted() {
    let enabled = [0usize, 7];
    for (mode, tier, engine) in CONFIGS {
        for off in [1.0, 2.0, 4.0, 5.0, 9.0] {
            let input = C4 + 100.0 * off;
            let settings = hard_tune(states(&enabled, NoteState::Bypass), tier, engine);
            let opts = RenderOptions::new(SR, mode, tier).with_settings(settings);
            let rendered = render(&tone(input, 1.5, true), &opts);
            let (out, _) = steady_output_cents(rendered.aligned(0));
            assert!(
                (out - input).abs() < 5.0,
                "{mode:?} {tier:?}: a tone near a bypassed note must pass unshifted \
                 (in {input:.0} c, out {out:.1} c)"
            );
        }
    }
}
