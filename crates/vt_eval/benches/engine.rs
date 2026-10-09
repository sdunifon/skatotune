//! Criterion benches of `Engine::process` (DESIGN §3.7): host buffers of
//! 32/64/128/512 samples × latency modes × quality tiers at 48 kHz, stereo,
//! on realistic sung material (melody with consonants, so voicing changes and
//! grain scheduling are exercised rather than a steady state).
//!
//! `cargo bench -p vt_eval --bench engine` (filter e.g. `-- live/hq`).
//! Throughput is reported in samples (frames) per second; the per-iteration
//! time is directly comparable with the buffer period (e.g. 667 µs at 32).
//! For tail latencies (p99.9, max) use `vt_eval rt-bench`.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use vt_core::{EngineConfig, Settings, VoiceRange};
use vt_engine::Engine;
use vt_eval::render::default_engine;
use vt_eval::scenarios;
use vt_eval::suite::{MODES, TIERS, mode_name, tier_name};

const SR: f32 = 48_000.0;
const BUFFERS: [usize; 4] = [32, 64, 128, 512];

fn bench_engine(c: &mut Criterion) {
    let material = scenarios::build("melody_male", SR)
        .map(|s| s.signal.audio)
        .unwrap_or_else(|| vec![0.0; SR as usize]);
    for mode in MODES {
        for tier in TIERS {
            let mut group =
                c.benchmark_group(format!("engine/{}/{}", mode_name(mode), tier_name(tier)));
            for buffer in BUFFERS {
                group.throughput(Throughput::Elements(buffer as u64));
                group.bench_with_input(
                    BenchmarkId::from_parameter(buffer),
                    &buffer,
                    |b, &buffer| {
                        let cfg = EngineConfig {
                            sample_rate: SR,
                            max_block: buffer,
                            channels: 2,
                            latency_mode: mode,
                            voice_range: VoiceRange::AltoTenor,
                            ..EngineConfig::default()
                        };
                        let mut engine = Engine::new(cfg);
                        engine.set_settings(&Settings {
                            tier,
                            engine: default_engine(mode),
                            retune_ms: 10.0,
                            ..scenarios::settings_for(0, vt_core::ScaleKind::Major)
                        });
                        let mut left = vec![0.0f32; buffer];
                        let mut right = vec![0.0f32; buffer];
                        let mut pos = 0usize;
                        let n = material.len().max(1);
                        b.iter(|| {
                            // Refill from the looping material (a copy of `buffer`
                            // samples: negligible next to the engine's work).
                            for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
                                let v = material.get((pos + i) % n).copied().unwrap_or(0.0);
                                *l = v;
                                *r = v;
                            }
                            pos = (pos + buffer) % n;
                            let mut io = [left.as_mut_slice(), right.as_mut_slice()];
                            engine.process(black_box(&mut io), &[]);
                            black_box(&io);
                        });
                    },
                );
            }
            group.finish();
        }
    }
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = config();
    targets = bench_engine
}
criterion_main!(benches);
