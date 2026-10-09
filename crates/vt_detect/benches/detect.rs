//! Detector benchmarks (DESIGN §3.2): one control hop of work, `push(hop)` +
//! `analyze()`, per tier × voice range × alignment at 48 kHz on a vowel with
//! vibrato (the detector keeps running across iterations, so every iteration
//! is a real steady-state hop). Also: `push` alone (the per-sample path) and
//! the heaviest configurations at 192 kHz.
//!
//! The 48 kHz hop is 64 samples = 1.333 ms, so `time / 1333 µs` is the share
//! of one core.
//!
//! Run: `cargo bench -p vt_detect --bench detect -- --warm-up-time 1 --measurement-time 2`

#[path = "../tests/common/mod.rs"]
mod common;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use vt_core::{QualityTier, VoiceRange};
use vt_detect::{Alignment, Detector, DetectorConfig};

const RANGES: [(VoiceRange, &str, f64); 4] = [
    (VoiceRange::LowMale, "low_male", 110.0),
    (VoiceRange::AltoTenor, "alto_tenor", 220.0),
    (VoiceRange::Soprano, "soprano", 523.0),
    (VoiceRange::Wide, "wide", 220.0),
];
const TIERS: [(QualityTier, &str); 3] = [(QualityTier::Eco, "eco"), (QualityTier::Hq, "hq"), (QualityTier::Max, "max")];

fn signal(fs: f64, f: f64) -> Vec<f32> {
    let f0 = common::vibrato(fs, f, 5.5, 40.0, 1.0);
    common::Voice { jitter: 0.002, ..common::Voice::default() }.render(fs, &f0)
}

fn detector(fs: f32, range: VoiceRange, alignment: Alignment, tier: QualityTier) -> Detector {
    let mut d = Detector::new(DetectorConfig {
        sample_rate: fs,
        voice_range: range,
        hop: vt_core::hop_for_rate(fs),
        alignment,
    });
    d.set_tier(tier);
    d
}

/// One steady-state hop per iteration, cycling through `x`.
fn bench_hops(
    c: &mut Criterion,
    group: &str,
    fs: f32,
    configs: &[(VoiceRange, &str, f64)],
    tiers: &[(QualityTier, &str)],
) {
    let mut g = c.benchmark_group(group);
    let hop = vt_core::hop_for_rate(fs);
    let studio = Alignment::Centered { delay: (0.04 * fs) as usize };
    for &(range, rname, f) in configs {
        let x = signal(fs as f64, f);
        for &(tier, tname) in tiers {
            for (al, aname) in [(Alignment::Causal, "live"), (studio, "studio")] {
                let mut d = detector(fs, range, al, tier);
                // Warm the detector up on 0.3 s of signal.
                for chunk in x[..(0.3 * fs as f64) as usize].chunks_exact(hop) {
                    d.push(chunk);
                    d.analyze();
                }
                let mut pos = 0usize;
                g.bench_function(BenchmarkId::new(format!("{tname}/{aname}"), rname), |b| {
                    b.iter(|| {
                        if pos + hop > x.len() {
                            pos = 0;
                        }
                        d.push(black_box(&x[pos..pos + hop]));
                        pos += hop;
                        black_box(d.analyze())
                    })
                });
            }
        }
    }
    g.finish();
}

fn bench_48k(c: &mut Criterion) {
    bench_hops(c, "hop_48k", 48_000.0, &RANGES, &TIERS);
}

fn bench_192k(c: &mut Criterion) {
    // Heaviest cases: lowest voices (longest windows) at the highest rate.
    bench_hops(c, "hop_192k", 192_000.0, &RANGES[..1], &TIERS[1..]);
}

fn bench_push(c: &mut Criterion) {
    // Per-sample path only (conditioning, 3 decimators, E/H recursion over
    // all lags), one 48 kHz hop per iteration.
    let mut g = c.benchmark_group("push_only_48k");
    for &(range, rname, f) in &RANGES {
        let x = signal(48_000.0, f);
        let mut d = detector(48_000.0, range, Alignment::Causal, QualityTier::Hq);
        let mut pos = 0usize;
        g.bench_function(rname, |b| {
            b.iter(|| {
                if pos + 64 > x.len() {
                    pos = 0;
                }
                d.push(black_box(&x[pos..pos + 64]));
                pos += 64;
            })
        });
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(40).warm_up_time(std::time::Duration::from_millis(500)).measurement_time(std::time::Duration::from_millis(1200));
    targets = bench_48k, bench_push, bench_192k
}
criterion_main!(benches);
