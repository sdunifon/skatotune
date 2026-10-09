//! `Shifter::process` benchmarks (DESIGN §3.4): 64 / 128 / 512-sample stereo
//! blocks at 48 kHz for every engine × tier × latency mode, in steady state
//! on a voiced vowel with a vibrato-like ratio (so grains repeat, splices
//! happen and the mark tracker runs), with hop-aligned `set_control` calls
//! as the engine makes them.
//!
//! Run: `cargo bench -p skato_shift --bench shift -- --warm-up-time 0.5 --measurement-time 1`
//!
//! The real-time budget of a block is `n / 48 kHz` (1.33 ms at 64); the
//! throughput line (elements = samples) gives the per-sample cost.

use std::f64::consts::PI;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use skato_core::{EngineKind, LatencyMode, QualityTier, ShiftControl, VoiceRange};
use skato_shift::{Shifter, ShifterConfig, latency_samples};

const SR: f32 = 48_000.0;
const HOP: usize = 64;

/// One second of a 196 Hz vowel-like pulse train through two resonances.
fn vowel() -> Vec<f32> {
    let n = 48_000;
    let f0 = 196.0f64;
    let mut src: Vec<f64> = (0..n)
        .map(|i| {
            let ph = 2.0 * PI * f0 * i as f64 / SR as f64;
            (1..40).map(|k| (k as f64 * ph).cos() / k as f64).sum::<f64>()
        })
        .collect();
    for (fc, bw) in [(730.0f64, 80.0f64), (1090.0, 90.0)] {
        let r = (-PI * bw / SR as f64).exp();
        let a1 = 2.0 * r * (2.0 * PI * fc / SR as f64).cos();
        let (mut y1, mut y2) = (0.0, 0.0);
        for v in src.iter_mut() {
            let y = (1.0 - r) * *v + a1 * y1 - r * r * y2;
            y2 = y1;
            y1 = y;
            *v = y;
        }
    }
    let peak = src.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    src.iter().map(|v| (0.5 * v / peak) as f32).collect()
}

struct Rig {
    sh: Shifter,
    x: Vec<f32>,
    pos: usize,
    l: Vec<f32>,
    r: Vec<f32>,
}

impl Rig {
    fn new(mode: LatencyMode, kind: EngineKind, tier: QualityTier, block: usize) -> Self {
        let cfg = ShifterConfig {
            sample_rate: SR,
            channels: 2,
            hop: HOP,
            latency_mode: mode,
            voice_range: VoiceRange::AltoTenor,
            latency: latency_samples(SR, mode, VoiceRange::AltoTenor, 10.0, 40.0),
        };
        let mut sh = Shifter::new(cfg);
        sh.set_engine(kind, tier);
        let mut rig = Self { sh, x: vowel(), pos: 0, l: vec![0.0; block], r: vec![0.0; block] };
        // Warm up: lock the tracker, start the engine, pass the crossfades.
        for _ in 0..(24_000 / block) {
            rig.block();
        }
        rig
    }

    /// One host block, split at hop boundaries like the engine does.
    fn block(&mut self) {
        let n = self.l.len();
        let mut done = 0;
        while done < n {
            if self.pos.is_multiple_of(HOP) {
                let t = self.pos as f64 / SR as f64;
                let st = 2.0 + (2.0 * PI * 6.0 * t).sin();
                self.sh.set_control(&ShiftControl {
                    ratio: 2f32.powf(st as f32 / 12.0),
                    formant_ratio: 1.0,
                    voiced: true,
                    period_hint: SR / 196.0,
                    confidence: 1.0,
                });
            }
            let len = (HOP - self.pos % HOP).min(n - done);
            let start = self.pos % self.x.len();
            let len = len.min(self.x.len() - start);
            let a = &self.x[start..start + len];
            self.l[done..done + len].copy_from_slice(a);
            self.r[done..done + len].copy_from_slice(a);
            let (l, r) = (&mut self.l[done..done + len], &mut self.r[done..done + len]);
            self.sh.process(a, &mut [l, r]);
            done += len;
            self.pos += len;
        }
        black_box((&self.l, &self.r));
    }
}

fn bench_process(c: &mut Criterion) {
    let modes = [LatencyMode::LiveMin, LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio];
    let tiers = [QualityTier::Eco, QualityTier::Hq, QualityTier::Max];
    for kind in [EngineKind::Classic, EngineKind::Natural] {
        let mut g = c.benchmark_group(format!("process_{kind:?}").to_lowercase());
        g.warm_up_time(Duration::from_millis(300));
        g.measurement_time(Duration::from_millis(800));
        g.sample_size(20);
        for mode in modes {
            if kind == EngineKind::Natural && mode == LatencyMode::LiveMin {
                continue; // Live-Min is Classic only.
            }
            for tier in tiers {
                for block in [64usize, 128, 512] {
                    let mut rig = Rig::new(mode, kind, tier, block);
                    g.throughput(Throughput::Elements(block as u64));
                    let id = BenchmarkId::new(format!("{mode:?}/{tier:?}"), block);
                    g.bench_function(id, |b| b.iter(|| rig.block()));
                }
            }
        }
        g.finish();
    }
}

criterion_group!(benches, bench_process);
criterion_main!(benches);
