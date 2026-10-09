//! Shared helpers for the engine integration tests: deterministic test
//! signals, a block-pattern renderer over the public API, and an offline f64
//! pitch estimator for checking corrected output.
//!
//! `skato_eval` has a far more complete harness, but it depends on this crate, so
//! the engine's own tests carry a small self-contained toolkit instead.

#![allow(dead_code)]

use skato_core::EngineConfig;
use skato_engine::{Engine, TimedMidi};

/// Small deterministic xorshift generator (uniform in [-1, 1)).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [-1, 1).
    pub fn bipolar(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// Uniform integer in `lo..=hi`.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as usize
    }
}

/// A vowel-like voiced tone: harmonics of `f0_of(t)` up to 5 kHz with a
/// 1/k spectral tilt and two broad formant bumps (≈ 700 Hz and 1.2 kHz),
/// phase integrated in f64, with 10 ms raised-cosine fades.
pub fn vowel(sr: f32, secs: f64, amp: f32, f0_of: impl Fn(f64) -> f64) -> Vec<f32> {
    let n = (secs * sr as f64) as usize;
    let fade = (0.01 * sr as f64) as usize;
    let mut phase = 0.0f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sr as f64;
        let f0 = f0_of(t);
        phase += f0 / sr as f64;
        phase -= phase.floor();
        let kmax = ((5000.0 / f0) as usize).max(1);
        let mut y = 0.0f64;
        for k in 1..=kmax {
            let fk = k as f64 * f0;
            let formant =
                1.0 + 3.0 * (-((fk - 700.0) / 250.0).powi(2)).exp() + 2.0 * (-((fk - 1200.0) / 300.0).powi(2)).exp();
            y += formant / k as f64 * (std::f64::consts::TAU * k as f64 * phase).sin();
        }
        let env = if i < fade {
            0.5 - 0.5 * (std::f64::consts::PI * i as f64 / fade as f64).cos()
        } else if i + fade > n {
            0.5 - 0.5 * (std::f64::consts::PI * (n - i) as f64 / fade as f64).cos()
        } else {
            1.0
        };
        out.push((y * env * 0.2) as f32 * amp);
    }
    out
}

/// White-noise bursts of `burst` seconds every `period` seconds (aperiodic,
/// so the engine treats them as unvoiced and passes them as a pure delay).
pub fn noise_bursts(sr: f32, secs: f64, burst: f64, period: f64, amp: f32, seed: u64) -> Vec<f32> {
    let n = (secs * sr as f64) as usize;
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let t = i as f64 / sr as f64;
            if (t % period) < burst { amp * rng.bipolar() } else { 0.0 }
        })
        .collect()
}

pub fn silence(sr: f32, secs: f64) -> Vec<f32> {
    vec![0.0; (secs * sr as f64) as usize]
}

/// Block sizes covering exactly `total` samples for a named pattern:
/// `Some(n)` = fixed n, `None` = seeded random sizes in 1..=1500.
pub fn blocks(total: usize, fixed: Option<usize>, seed: u64) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    let mut done = 0;
    while done < total {
        let want = fixed.unwrap_or_else(|| rng.range(1, 1500)).max(1);
        let n = want.min(total - done);
        out.push(n);
        done += n;
    }
    out
}

/// The DESIGN §2.1 invariance set: {1, 7, 32, 64, 128, 480, 4096, random}.
pub fn invariance_patterns() -> Vec<Option<usize>> {
    vec![Some(1), Some(7), Some(32), Some(64), Some(128), Some(480), Some(4096), None]
}

/// Renders `input` (duplicated onto every configured channel) through
/// `engine` with the given block sizes and MIDI events at absolute input
/// positions, followed by `latency` samples of silence so the tail comes out.
/// Returns one output vector per channel.
pub fn render(
    engine: &mut Engine,
    input: &[f32],
    sizes: &[usize],
    midi: &[(u64, skato_core::MidiEvent)],
) -> Vec<Vec<f32>> {
    let chans = engine.config().channels.clamp(1, 2);
    let total: usize = sizes.iter().sum();
    let mut bufs: Vec<Vec<f32>> = (0..chans)
        .map(|_| {
            let mut v = vec![0.0f32; total];
            let n = input.len().min(total);
            v[..n].copy_from_slice(&input[..n]);
            v
        })
        .collect();
    let mut pos = 0usize;
    let mut next = 0usize;
    let mut block_midi = Vec::new();
    for &n in sizes {
        block_midi.clear();
        while let Some(&(t, ev)) = midi.get(next) {
            if t as usize >= pos + n {
                break;
            }
            block_midi.push(TimedMidi { timing: (t as usize).saturating_sub(pos) as u32, event: ev });
            next += 1;
        }
        let mut io: Vec<&mut [f32]> = bufs.iter_mut().map(|b| &mut b[pos..pos + n]).collect();
        engine.process(&mut io, &block_midi);
        pos += n;
    }
    bufs
}

/// Fresh engine, reset and configured with `settings`.
pub fn engine(cfg: EngineConfig, settings: &skato_core::Settings) -> Engine {
    let mut e = Engine::new(cfg);
    e.reset();
    e.set_settings(settings);
    e
}

/// Fundamental of `x` in Hz by a normalised squared-difference function
/// (type-II NSDF over a fixed window) with parabolic peak interpolation, in
/// f64. Searches `f_lo..f_hi`; picks the first peak within 0.9 of the global
/// maximum (the usual MPM rule against octave errors).
pub fn pitch_hz(x: &[f32], sr: f32, f_lo: f64, f_hi: f64) -> Option<f64> {
    let tau_min = (sr as f64 / f_hi).floor() as usize;
    let tau_max = (sr as f64 / f_lo).ceil() as usize;
    if x.len() < 2 * tau_max + 2 {
        return None;
    }
    let w = x.len() - tau_max - 1;
    let x: Vec<f64> = x.iter().map(|v| *v as f64).collect();
    let nsdf = |tau: usize| -> f64 {
        let (mut r, mut m) = (0.0, 0.0);
        for j in 0..w {
            r += x[j] * x[j + tau];
            m += x[j] * x[j] + x[j + tau] * x[j + tau];
        }
        if m > 0.0 { 2.0 * r / m } else { 0.0 }
    };
    let vals: Vec<f64> = (0..=tau_max + 1).map(|t| if t + 1 >= tau_min { nsdf(t) } else { 0.0 }).collect();
    let peak = (tau_min..=tau_max).map(|t| vals[t]).fold(f64::MIN, f64::max);
    if peak < 0.5 {
        return None;
    }
    let tau = (tau_min.max(1)..=tau_max)
        .find(|&t| vals[t] >= 0.9 * peak && vals[t] >= vals[t - 1] && vals[t] >= vals[t + 1])?;
    let (a, b, c) = (vals[tau - 1], vals[tau], vals[tau + 1]);
    let den = a - 2.0 * b + c;
    let off = if den.abs() > 1e-12 { 0.5 * (a - c) / den } else { 0.0 };
    Some(sr as f64 / (tau as f64 + off))
}

/// Absolute cents of a frequency.
pub fn cents(hz: f64) -> f64 {
    6900.0 + 1200.0 * (hz / 440.0).log2()
}
