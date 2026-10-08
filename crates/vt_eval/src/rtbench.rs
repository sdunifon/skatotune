//! Tail-latency benchmark of `Engine::process` (DESIGN §3.7, §4; R§10.7).
//!
//! Means hide what matters for audio: a single slow callback is a dropout. Each
//! configuration processes several seconds of realistic material (sung
//! melody with consonants and breaths, so voicing changes and grain
//! scheduling are exercised) block by block, timing every `process()` call,
//! and reports p50 / p99 / p99.9 / max against the buffer period.

use std::hint::black_box;
use std::time::Instant;

use vt_core::{EngineConfig, EngineKind, LatencyMode, QualityTier, Settings, VoiceRange};
use vt_engine::Engine;

use crate::render::default_engine;
use crate::scenarios;
use crate::suite::{MODES, TIERS};

#[derive(Clone, Debug, PartialEq)]
pub struct RtBenchOptions {
    pub sample_rate: f32,
    /// Seconds of audio timed per configuration (after warm-up).
    pub seconds: f64,
    pub buffers: Vec<usize>,
    pub modes: Vec<LatencyMode>,
    pub tiers: Vec<QualityTier>,
    pub channels: usize,
}

impl Default for RtBenchOptions {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            seconds: 10.0,
            buffers: vec![32, 64, 128, 256, 512],
            modes: MODES.to_vec(),
            tiers: TIERS.to_vec(),
            channels: 2,
        }
    }
}

/// Exit-code budget: p99.9 must stay below this fraction of the buffer period
/// at the small buffers (DESIGN §3.7).
pub const BUDGET_FRACTION: f64 = 0.10;
/// Buffers the exit-code budget applies to.
pub const GATED_BUFFERS: [usize; 3] = [32, 64, 128];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtBenchRow {
    pub mode: LatencyMode,
    pub tier: QualityTier,
    pub engine: EngineKind,
    pub buffer: usize,
    pub blocks: usize,
    pub p50_us: f64,
    pub p99_us: f64,
    pub p999_us: f64,
    pub max_us: f64,
    /// Buffer period, µs.
    pub period_us: f64,
}

impl RtBenchRow {
    pub fn p999_fraction(&self) -> f64 {
        self.p999_us / self.period_us
    }

    /// Within the exit-code budget (always true for ungated buffer sizes).
    pub fn within_budget(&self) -> bool {
        !GATED_BUFFERS.contains(&self.buffer) || self.p999_fraction() <= BUDGET_FRACTION
    }

    /// The stricter DESIGN §4 CPU gate, where it applies: Live HQ < 5 % at
    /// 128 and < 10 % at 32; Max < 25 %.
    pub fn design_gate(&self) -> Option<bool> {
        let f = self.p999_fraction();
        match (self.mode, self.tier, self.buffer) {
            (LatencyMode::Live, QualityTier::Hq, 128) => Some(f < 0.05),
            (LatencyMode::Live, QualityTier::Hq, 32) => Some(f < 0.10),
            (_, QualityTier::Max, _) => Some(f < 0.25),
            _ => None,
        }
    }
}

/// Run every configuration. `progress` is called after each one.
pub fn run(opts: &RtBenchOptions, progress: &mut dyn FnMut(&RtBenchRow)) -> Vec<RtBenchRow> {
    let sr = opts.sample_rate;
    // Realistic material: the male melody (voiced/unvoiced changes, consonants).
    let material = scenarios::build("melody_male", sr).map(|s| s.signal.audio).unwrap_or_else(|| vec![0.0; sr as usize]);
    let warmup = (0.5 * sr as f64) as usize;
    let timed = (opts.seconds * sr as f64) as usize;
    let mut rows = Vec::new();
    for &mode in &opts.modes {
        for &tier in &opts.tiers {
            for &buffer in &opts.buffers {
                let buffer = buffer.max(1);
                let cfg = EngineConfig {
                    sample_rate: sr,
                    max_block: buffer,
                    channels: opts.channels.max(1),
                    latency_mode: mode,
                    voice_range: VoiceRange::AltoTenor,
                    ..EngineConfig::default()
                };
                let settings = Settings { tier, engine: default_engine(mode), retune_ms: 10.0, ..scenarios::settings_for(0, vt_core::ScaleKind::Major) };
                let mut engine = Engine::new(cfg);
                engine.set_settings(&settings);
                let mut bufs: Vec<Vec<f32>> = vec![vec![0.0; buffer]; cfg.channels];
                let mut io: Vec<&mut [f32]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
                let mut times: Vec<u64> = Vec::with_capacity(timed / buffer + 1);
                let mut pos = 0usize;
                let mut done = 0usize;
                while done < warmup + timed {
                    for b in io.iter_mut() {
                        for (i, v) in b.iter_mut().enumerate() {
                            *v = material.get((pos + i) % material.len().max(1)).copied().unwrap_or(0.0);
                        }
                    }
                    let t0 = Instant::now();
                    engine.process(black_box(&mut io), &[]);
                    let dt = t0.elapsed().as_nanos() as u64;
                    if done >= warmup {
                        times.push(dt);
                    }
                    black_box(&io);
                    pos = (pos + buffer) % material.len().max(1);
                    done += buffer;
                }
                times.sort_unstable();
                let pct = |p: f64| -> f64 {
                    if times.is_empty() {
                        return 0.0;
                    }
                    let idx = ((p / 100.0) * (times.len() - 1) as f64).round() as usize;
                    times[idx.min(times.len() - 1)] as f64 / 1000.0
                };
                let row = RtBenchRow {
                    mode,
                    tier,
                    engine: settings.engine,
                    buffer,
                    blocks: times.len(),
                    p50_us: pct(50.0),
                    p99_us: pct(99.0),
                    p999_us: pct(99.9),
                    max_us: times.last().map_or(0.0, |t| *t as f64 / 1000.0),
                    period_us: buffer as f64 / sr as f64 * 1e6,
                };
                progress(&row);
                rows.push(row);
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_bench_produces_rows() {
        let opts = RtBenchOptions {
            seconds: 0.05,
            buffers: vec![64],
            modes: vec![LatencyMode::Live],
            tiers: vec![QualityTier::Hq],
            ..RtBenchOptions::default()
        };
        let rows = run(&opts, &mut |_| {});
        assert_eq!(rows.len(), 1);
        let r = rows[0];
        assert!(r.blocks >= 30 && r.p50_us <= r.p99_us && r.p99_us <= r.p999_us && r.p999_us <= r.max_us);
        assert!((r.period_us - 1333.33).abs() < 0.1);
    }
}
