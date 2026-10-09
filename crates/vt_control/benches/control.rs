//! `Corrector::process_hop` cost per tier (DESIGN §3.3).
//!
//! The input is a pre-generated sung phrase (notes with scoops, 5.5 Hz
//! vibrato, short unvoiced gaps) so every stage of the control law is
//! exercised: voicing FSM, vibrato tracker (+ Kalman in HQ/Max), note rules
//! or HMM (forward in Live, forward + fixed-lag Viterbi in Studio). One
//! iteration = one control hop (1.33 ms of audio at 48 kHz).

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use vt_control::{Corrector, CorrectorConfig};
use vt_core::{LatencyMode, PitchEstimate, QualityTier, ScaleKind, Settings, cents_to_hz};

const SR: f32 = 48_000.0;
const HOP: usize = 64;

/// A 4 s phrase: C major notes, each with a scoop, vibrato and a short gap.
fn phrase() -> Vec<PitchEstimate> {
    let hop_s = HOP as f32 / SR;
    let notes = [60.0f32, 62.0, 64.0, 65.0, 67.0, 65.0, 64.0, 62.0];
    let n = (4.0 / hop_s) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 * hop_s;
            let k = ((t / 0.5) as usize) % notes.len();
            let tn = t % 0.5;
            if tn > 0.46 {
                return PitchEstimate { info_lag: 640.0, ..PitchEstimate::default() };
            }
            let scoop = if tn < 0.03 { -80.0 * (1.0 - tn / 0.03) } else { 0.0 };
            let vib = if tn > 0.15 { 35.0 * (2.0 * std::f32::consts::PI * 5.5 * t).sin() } else { 0.0 };
            let cents = 100.0 * notes[k] + 15.0 + scoop + vib;
            let f0 = cents_to_hz(cents);
            PitchEstimate {
                sample_pos: (i * HOP) as u64,
                info_lag: 640.0,
                f0_hz: f0,
                cents,
                period: SR / f0,
                confidence: 0.9,
                voiced_prob: 0.95,
                voiced: true,
                sibilant: false,
                level_db: -18.0,
            }
        })
        .collect()
}

fn bench_process_hop(c: &mut Criterion) {
    let input = phrase();
    let mut group = c.benchmark_group("process_hop");
    for (mode, look, label) in [(LatencyMode::Live, 0, "live"), (LatencyMode::Studio, 30, "studio")] {
        for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
            let cfg = CorrectorConfig { sample_rate: SR, hop: HOP, latency_mode: mode, lookahead_hops: look };
            let mut corr = Corrector::new(cfg);
            let settings = Settings {
                tier,
                note_states: ScaleKind::Major.note_states(0),
                humanize: 30.0,
                flex_tune: 10.0,
                ..Settings::default()
            };
            corr.set_settings(&settings);
            let mut i = 0usize;
            group.bench_function(BenchmarkId::new(label, format!("{tier:?}")), |b| {
                b.iter(|| {
                    let est = &input[i % input.len()];
                    i = i.wrapping_add(1);
                    black_box(corr.process_hop(black_box(est), None))
                })
            });
        }
    }
    group.finish();

    // The per-block settings call when nothing changed (the common case).
    let cfg = CorrectorConfig { sample_rate: SR, hop: HOP, latency_mode: LatencyMode::Live, lookahead_hops: 0 };
    let mut corr = Corrector::new(cfg);
    let s = Settings::default();
    corr.set_settings(&s);
    c.bench_function("set_settings_unchanged", |b| b.iter(|| corr.set_settings(black_box(&s))));
}

criterion_group!(benches, bench_process_hop);
criterion_main!(benches);
