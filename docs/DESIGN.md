# Vocal Tune v1: build contract

This is the **implementation contract** for v1. It turns `docs/research.md` (the evidence and the full algorithm details; cited below as **R§x.y**) and `docs/rust-performance.md` (Rust optimisation research; cited as **P§…**) into concrete crates, Rust APIs, tests and gates.

When this file and `research.md` disagree, **this file wins**, because it records the v1 scope decisions. When this file is silent on an algorithmic detail, follow `research.md`.

---

## 1. Scope

### 1.1 In v1

| Area | What ships |
|---|---|
| Formats | VST3 (Ableton Live 12) + CLAP (Bitwig), standalone app (feature `standalone`) |
| Latency modes | Live-Min (~1 ms, Classic engine), Live (~3-5 ms, causal PSOLA), Live+ (5-10 ms, user-set), Studio (20-50 ms, lookahead PSOLA). Constant reported latency per mode + voice range |
| Quality tiers | Eco / HQ / Max inside every mode, never changing latency |
| Detector | Decimated pitch-synchronous E/H candidates (R§8.3), FFT NSDF + type-I CMNDF verifier with pYIN-style probabilities (R§8.4), octave checks (R§8.5), causal pitch HMM (R§8.6; fixed-lag smoothing in Studio), full-rate refinement (R§8.7), voicing/sibilance FSM (R§8.8), mean-pitch Kalman with innovation gating (R§8.10) |
| Shifter | L0 Classic (R§8.17), L1/L2 causal PSOLA (R§8.13-8.16, 8.18), S1 Studio PSOLA (nearest-mark, lookahead, R§8.14). Formant shift by grain resampling (R§7.5) |
| Control law | All of R§7.6: note decision (rules in Eco, MonoNote-style HMM in HQ/Max), retune one-pole on the correction, transition τ, Humanize, Flex-Tune, Snap Min/Max, centre/vibrato split + Vibrato Depth, synthetic vibrato, Transpose/Detune, MIDI Target/Learn/Auto + bend |
| GUI | egui: scrolling piano-roll graph (detected, centre, target, output), cents meter, scale/keyboard editor, knobs, mode/tier selectors, CPU + latency readout |
| Perf | NEON (aarch64) + AVX2/FMA (x86, runtime-dispatched) + scalar kernels, vDSP FFT on macOS, fast math, benches proving every optimisation, tail-latency gate, zero-allocation gate |
| Eval | Synthetic singing harness with exact ground truth, objective tuning/settling/jitter/latency metrics, block-size invariance |

### 1.2 Deferred (hooks only, documented as future work)

- Neural octave/voicing arbiter (R§8.9, R§8.22): needs a training pipeline and licensed singing data. Keep a `Max`-tier hook in the detector (`candidates()` accessor) so it can be added without API changes.
- fastF0ArMl harmonic re-scoring, harmonic-chirp scoring, SWIPE′ second opinion.
- S2 spectral Studio engine, per-grain true-envelope re-colouring beyond ±3 st (formant knob still works to ±12 st via grain resampling; clean within ±3 st).
- SEDREAMS/ZFF GCI marks with DP path (Studio uses the Live NCC mark tracker with lookahead in v1).
- Key detection, sidechain chord prior, worker-thread pipelining (everything runs inline on the audio thread in v1; the whole chain is a few % of one core).

---

## 2. Workspace and conventions

```
crates/vt_core     shared types (done: src/types.rs), math, ring, interp, filter, window, simd, fft
crates/vt_detect   Detector
crates/vt_control  Corrector (control law), scale + MIDI state
crates/vt_shift    Shifter (Classic, PSOLA engines, mark tracker), latency formula
crates/vt_engine   Engine: block-size agnostic orchestration, telemetry, stats
crates/vocal_tune  nice-plug plugin + egui editor + standalone bin
crates/vt_eval     synth signals, reference pitch tracker, metrics, render, CLI, suite tests, engine benches
xtask              nice-plug-xtask bundler
```

- Toolchain: Rust **1.99.0** (pinned in `rust-toolchain.toml`), edition 2024. Stable only.
- Build: `cargo build`, `cargo test`, `cargo bench -p <crate>`, `cargo xtask bundle vocal_tune --release`.
- **Parallel agents:** when several agents build at once, each uses its own target dir to avoid lock contention: `CARGO_TARGET_DIR=target/agent-<crate> cargo test -p <crate>`.
- Style: match the existing code. Module-level `//!` docs explain *what and why*; item docs on public APIs; comments for non-obvious maths (cite R§ sections). No dead code, no `println!` in library code.

### 2.1 Realtime rules (apply to every `process`/`push`/`analyze`/`process_hop` path)

1. **No allocation** after construction. Everything is allocated in `new()`, sized for the worst case (max tier, max block size, max sample rate in config).
2. **No locks, no syscalls, no logging, no `format!`**, no `Arc` drops, no channels other than `rtrb`.
3. **No panics.** No `unwrap()`/`expect()`/unchecked indexing that can fail on bad input. Use `get()`, clamping, saturating maths, `debug_assert!`. Treat NaN/Inf input as silence (sanitize at the engine boundary).
4. **Bounded work per call.** Fixed caps on every search, beam, candidate list and loop.
5. **Sample positions** are `u64` (absolute) or `f64` (fractional). Never `f32` positions.
6. **Determinism:** same input + same block sizes ⇒ bit-identical output. Different block sizes ⇒ identical output within 1e-6 (only control-rate boundaries matter, and they are hop-aligned, not block-aligned).

### 2.2 Hop and timing

- The **control hop** is `hop = 64` samples at 44.1/48 kHz, `128` at 88.2/96 kHz, `256` at 176.4/192 kHz (`vt_core::hop_for_rate(sr)`). It is the same for every tier and mode.
- The engine splits every host block at hop boundaries. Per segment: push samples into the detector, run the shifter on the segment. At each hop boundary: `detector.analyze()` → `corrector.process_hop()` → `shifter.set_control()` (the ratio then ramps linearly across the next hop). This adds **no** latency; control just updates at hop boundaries.
- Absolute input sample index `n` counts from `reset()`. Output sample `n` carries input content from about `n − D` (plus PSOLA content wander, R§2.1).
- Telemetry `PitchFrame::sample_pos` is **output time**: `estimate.sample_pos + D`.

---

## 3. Crate contracts

All public signatures below are binding. Implementers may add private items and extra public helpers, but must not change these signatures without updating this file.

### 3.1 `vt_core`

`types` is written. Add these modules:

```rust
pub mod math {
    /// 2^x. Max relative error < 1e-6 for x in [-60, 60]; saturates outside.
    pub fn fast_exp2(x: f32) -> f32;
    /// log2(x) for x > 0 (normal range). Max abs error < 1e-5. Returns -126.0 for x <= 0.
    pub fn fast_log2(x: f32) -> f32;
    /// tanh. Max abs error < 1e-4.
    pub fn fast_tanh(x: f32) -> f32;
    pub fn cents_to_ratio(cents: f32) -> f32;      // fast_exp2(cents / 1200)
    pub fn ratio_to_cents(ratio: f32) -> f32;      // 1200 * fast_log2(ratio)
    pub fn hz_to_cents_fast(hz: f32) -> f32;
    pub fn cents_to_hz_fast(cents: f32) -> f32;
    /// One-pole smoothing coefficient α = 1 − exp(−step/τ); 1.0 when τ <= 0.
    pub fn onepole_alpha(tau_samples: f32, step_samples: f32) -> f32;
    pub fn db_to_lin(db: f32) -> f32;
    pub fn lin_to_db(lin: f32) -> f32;             // floor at -160 dB
    /// Parabolic peak through (−1, ym1), (0, y0), (1, yp1): returns (offset in [-1, 1], peak value).
    pub fn parabolic_peak(ym1: f32, y0: f32, yp1: f32) -> (f32, f32);
}
/// Control hop in samples for a sample rate (see §2.2).
pub fn hop_for_rate(sample_rate: f32) -> usize;

pub mod ring {
    /// Power-of-two circular buffer addressed by absolute sample index.
    pub struct Ring { /* buf: Box<[f32]>, mask, written: u64 */ }
    impl Ring {
        pub fn new(min_capacity: usize) -> Self;          // rounds up to a power of two
        pub fn capacity(&self) -> usize;
        pub fn written(&self) -> u64;                     // total pushed; newest index = written-1
        pub fn push(&mut self, x: f32);
        pub fn extend(&mut self, xs: &[f32]);
        /// Sample at absolute index; 0.0 if not yet written or already overwritten (never panics).
        pub fn get(&self, index: u64) -> f32;
        /// Copy `out.len()` samples starting at absolute `start` (zeros where unavailable).
        pub fn copy_to(&self, start: u64, out: &mut [f32]);
        pub fn clear(&mut self);
    }
}

pub mod interp {
    use crate::ring::Ring;
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum InterpQuality { Cubic, Sinc8, Sinc16 }
    /// 4-point cubic Hermite read at fractional absolute position.
    pub fn cubic(ring: &Ring, pos: f64) -> f32;
    /// Kaiser-windowed polyphase sinc tables (several cutoffs so reads at rate γ > 1 can be
    /// anti-aliased, R§8.16). Built in `new()`, read allocation-free.
    pub struct SincBank { /* ... */ }
    impl SincBank {
        pub fn new(taps: usize /* 8 or 16 */, phases: usize /* >= 256 */) -> Self;
        pub fn taps(&self) -> usize;
        pub fn half_width(&self) -> usize;                // taps / 2
        /// Read at `pos` with a low-pass appropriate for reading at `rate` samples per output
        /// sample (rate <= 1.0 → full band).
        pub fn read(&self, ring: &Ring, pos: f64, rate: f32) -> f32;
    }
    /// Convenience dispatcher used by the engines.
    pub struct Interpolator { /* quality + SincBank(s) */ }
    impl Interpolator {
        pub fn new() -> Self;                             // builds Sinc8 and Sinc16 banks
        pub fn read(&self, q: InterpQuality, ring: &Ring, pos: f64, rate: f32) -> f32;
        pub fn half_width(q: InterpQuality) -> usize;     // samples of lookahead the read needs
    }
}

pub mod filter {
    pub struct DcBlocker { /* y = x − x1 + R·y1, R = 0.995 at 48 kHz scaled with rate */ }
    pub struct Biquad { /* TDF-II, f32 coefficients, f64 state not required */ }
    impl Biquad {
        pub fn lowpass(sr: f32, f: f32, q: f32) -> Self;
        pub fn highpass(sr: f32, f: f32, q: f32) -> Self;
        pub fn bandpass(sr: f32, f: f32, q: f32) -> Self;
        pub fn process(&mut self, x: f32) -> f32;
        pub fn reset(&mut self);
    }
    pub struct OnePoleLp { /* … */ }
    /// Minimum-phase IIR anti-alias filter + integer downsampler (R§8.2). Cascade of biquads,
    /// cutoff ~0.4·fs_out, ≥ 60 dB stopband above fs_out/2.
    pub struct Decimator { /* … */ }
    impl Decimator {
        pub fn new(sr_in: f32, factor: usize) -> Self;
        /// Push one input sample; returns Some(output) every `factor` samples.
        pub fn push(&mut self, x: f32) -> Option<f32>;
        /// Approximate group delay in *input* samples at low frequencies (counts toward detection lag).
        pub fn group_delay(&self) -> f32;
        pub fn reset(&mut self);
    }
}

pub mod window {
    pub fn hann(out: &mut [f32]);                        // periodic Hann
    pub fn kaiser(out: &mut [f32], beta: f32);
    /// Equal-power and equal-gain raised-cosine fade curves sampled at t in [0, 1].
    pub fn raised_cosine(t: f32) -> f32;
}

pub mod simd {
    //! Hot kernels with three implementations each: `scalar` (portable, written so LLVM
    //! auto-vectorises it: chunked multi-accumulators), `neon` (aarch64, core::arch
    //! intrinsics) and `avx2` (x86_64, #[target_feature(enable = "avx2,fma")], selected at
    //! runtime with is_x86_feature_detected!). The top-level functions dispatch to the best
    //! available implementation (selection cached in an atomic on first use).
    pub fn dot(a: &[f32], b: &[f32]) -> f32;             // over min(len)
    pub fn sum_sq(a: &[f32]) -> f32;
    /// y[i] += a * x[i]
    pub fn axpy(y: &mut [f32], a: f32, x: &[f32]);
    /// y[i] += x[i] * w[i]   (windowed overlap-add)
    pub fn mul_acc(y: &mut [f32], x: &[f32], w: &[f32]);
    /// Direct (type-I) difference function: out[k] = Σ_{j<w} (x[j] − x[j+τ])², τ = tau0 + k.
    /// Requires x.len() >= w + tau0 + out.len() − 1.
    pub fn diff_sq(x: &[f32], w: usize, tau0: usize, out: &mut [f32]);
    pub mod scalar { /* same fns */ }
    #[cfg(target_arch = "aarch64")] pub mod neon { /* same fns */ }
    #[cfg(target_arch = "x86_64")] pub mod avx2 { /* same fns, unsafe + target_feature */ }
    /// For tests/benches: which backend the dispatcher picked.
    pub fn active_backend() -> &'static str;
}

pub mod fft {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum FftBackend { RealFft, Vdsp /* macOS Accelerate; falls back to RealFft elsewhere */ }
    impl FftBackend { pub fn best() -> Self; }       // Vdsp on macOS, RealFft otherwise
    /// FFT-based linear correlation and spectra for a fixed power-of-two size. All scratch is
    /// preallocated; every method is allocation-free.
    pub struct Correlator { /* … */ }
    impl Correlator {
        pub fn new(n: usize, backend: FftBackend) -> Self;
        pub fn len(&self) -> usize;
        pub fn backend(&self) -> FftBackend;
        /// out[τ] = Σ_j a[j]·b[j+τ] (linear, zero-padded) for τ in 0..out.len().
        /// Requires a.len() <= n, b.len() <= n, out.len() <= n/2 and a.len() + out.len() <= n is not
        /// required (zero padding avoids wrap as long as b.len() + a.len() <= n).
        pub fn cross_correlate(&mut self, a: &[f32], b: &[f32], out: &mut [f32]);
        /// out[τ] = Σ_j x[j]·x[j+τ].
        pub fn autocorrelate(&mut self, x: &[f32], out: &mut [f32]);
        /// |X[k]|² for k in 0..=n/2 of zero-padded x (out.len() == n/2 + 1).
        pub fn power_spectrum(&mut self, x: &[f32], out: &mut [f32]);
    }
}
```

Tests (in-crate): every SIMD backend vs an f64 reference at lengths {0,1,3,15,16,17,31,33,63,100,1024,2047,2048,4096}; fast-math error bounds over dense sweeps; ring wrap/`get` out-of-range; sinc interpolation of band-limited sines (error < -80 dB for Sinc16 at f < 0.4·fs); decimator stopband; Correlator vs direct O(n²) for both backends (rel error < 1e-4).

Benches (`crates/vt_core/benches/kernels.rs`, criterion): scalar vs neon (vs avx2) for each kernel at 256/1024/4096; `diff_sq` direct vs `Correlator` path for W=1024/τ=512 and W=2048/τ=1024; RealFft vs Vdsp at n = 512..8192; fast vs std exp2/log2/tanh; cubic vs Sinc8 vs Sinc16 per 512 reads.

### 3.2 `vt_detect`

Implements R§4.5 / R§7.3 / R§8.1-8.10 for v1 (no neural, no fastF0ArMl).

```rust
use vt_core::{PitchEstimate, QualityTier, VoiceRange};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Alignment {
    /// Live modes: analysis uses only past samples; estimate describes ~newest − info_lag.
    Causal,
    /// Studio: the estimate describes `newest − delay`; windows centred there (W ≤ 2·delay).
    Centered { delay: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectorConfig {
    pub sample_rate: f32,
    pub voice_range: VoiceRange,
    pub hop: usize,
    pub alignment: Alignment,
}

pub struct Detector { /* … */ }
impl Detector {
    pub fn new(cfg: DetectorConfig) -> Self;           // allocates for QualityTier::Max sizes
    pub fn config(&self) -> &DetectorConfig;
    pub fn reset(&mut self);
    pub fn set_tier(&mut self, tier: QualityTier);    // RT-safe; may soft-reset trackers
    /// Tracking 0-100 and gate dBFS (see Settings).
    pub fn set_tracking(&mut self, tracking: f32, gate_db: f32);
    /// Feed any number of new input samples (mono analysis signal). Cheap per-sample work
    /// only (conditioning, decimation, E/H recursion).
    pub fn push(&mut self, x: &[f32]);
    /// Run the per-hop analysis. Called by the engine exactly once every `hop` pushed samples.
    pub fn analyze(&mut self) -> PitchEstimate;
    /// Up to 5 (period_samples, score) candidates from the last analyze(), best first (hook for a
    /// future neural arbiter and for tests).
    pub fn candidates(&self) -> &[(f32, f32)];
}
```

Tier behaviour (R§7.2.2, simplified): **Eco** verifier at ~16 kHz, 20-cent HMM bins, parabolic refinement only above 400 Hz; **HQ** verifier at ~24 kHz, 10-cent bins, two-resolution NSDF, full-rate refinement every hop; **Max** HQ + 5-cent bins + detector-lag compensation (R§8.10) applied to `cents`. Studio (`Centered`) adds fixed-lag HMM smoothing over `delay / hop` frames.

Tests: synthetic glottal vowels and pure/harmonic tones at 55-1400 Hz, all sample rates (44.1/48/96 kHz): median error < 1 c on steady tones (< 3 c above 1 kHz), no octave errors on a vowel set with strong F1 near 2·f0, step-response lag at 110/220/440 Hz reported and asserted below 1.5·W/2, voicing F1 ≥ 0.97 on vowel/silence/noise/sibilant sequences, vibrato (6 Hz, ±50 c) tracked with RMS error < 15 c in Live HQ, block-size independence.

Benches: `analyze()` + `push(hop)` per tier at 48 kHz, each voice range, both alignments.

### 3.3 `vt_control`

Implements R§7.6 completely (including MIDI R§7.6.3 and defaults R§7.6.4).

```rust
use vt_core::{LatencyMode, MidiEvent, PitchEstimate, PitchFrame, Settings, ShiftControl};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectorConfig {
    pub sample_rate: f32,
    pub hop: usize,
    pub latency_mode: LatencyMode,
    /// Studio: control hops of lookahead available to fixed-lag decisions (0 in Live modes).
    pub lookahead_hops: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlOutput {
    pub shift: ShiftControl,
    /// Total shift S in cents (correction + vibrato scaling + LFO + transpose).
    pub shift_cents: f32,
    /// Telemetry for this hop; `sample_pos` is left 0 (the engine fills it).
    pub frame: PitchFrame,
}

pub struct Corrector { /* … */ }
impl Corrector {
    pub fn new(cfg: CorrectorConfig) -> Self;
    pub fn reset(&mut self);
    pub fn set_settings(&mut self, s: &Settings);     // every block; RT-safe
    pub fn midi(&mut self, ev: MidiEvent);
    /// One control hop. `mark_cents` = instantaneous input pitch from the PSOLA mark tracker when
    /// it is locked (Live modes, R§7.6.2 A), else None.
    pub fn process_hop(&mut self, est: &PitchEstimate, mark_cents: Option<f32>) -> ControlOutput;
    /// Effective scale after MIDI Learn edits (for the GUI keyboard), one state per pitch class.
    pub fn learned_states(&self) -> Option<[vt_core::NoteState; 12]>;
}
```

Note decision: **Eco** = rules (fat1 bias + Waves OR-tolerance, R§7.6.2 F); **HQ/Max** = MonoNote-style HMM forward filter (R§8.12); **Studio** = fixed-lag decisions using `lookahead_hops`.

Tests (pure control law, no audio): step response matches the one-pole definition (63% at τ); τ = 0 hard-snaps; residual vibrato ratio vs τ matches the R§7.6.2 H table within 0.03; PreserveVibrato + τ = 0 keeps a 6 Hz ±40 c vibrato around the corrected centre (extent within ±10% after engagement) while Depth 0 removes it; Flex-Tune weight curve; Humanize slows only sustained notes; hysteresis prevents flip-flop on a pitch hovering ±5 c around a midpoint; Bypass/Remove semantics; MIDI Target/Learn/Auto, all-octaves, bend, sustain, all-notes-off; transpose/detune; unvoiced hold/release/forget timings; octave guard.

### 3.4 `vt_shift`

Implements R§5.3 / R§7.4 / R§8.13-8.19 for v1.

```rust
use vt_core::{EngineKind, LatencyMode, QualityTier, ShiftControl, VoiceRange};

/// Reported latency D in samples for a configuration. Single source of truth (the engine and
/// plugin call this). Must satisfy every engine's lookahead needs at every tier in that mode
/// (use the largest interpolation kernel), R§7.2.1.
pub fn latency_samples(sample_rate: f32, mode: LatencyMode, range: VoiceRange,
                       live_plus_ms: f32, studio_ms: f32) -> usize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShifterConfig {
    pub sample_rate: f32,
    pub channels: usize,      // 1 or 2
    pub hop: usize,
    pub latency_mode: LatencyMode,
    pub voice_range: VoiceRange,
    pub latency: usize,       // = latency_samples(..)
}

pub struct Shifter { /* Classic + PSOLA engines, mark tracker, rings per channel */ }
impl Shifter {
    pub fn new(cfg: ShifterConfig) -> Self;
    pub fn reset(&mut self);
    pub fn latency(&self) -> usize;
    /// RT-safe. Engine changes crossfade over ~10 ms. In LiveMin only Classic is allowed (the
    /// request is ignored otherwise).
    pub fn set_engine(&mut self, kind: EngineKind, tier: QualityTier);
    /// Called at every hop boundary; the pitch ratio ramps linearly to `ctl.ratio` over `hop`
    /// samples.
    pub fn set_control(&mut self, ctl: &ShiftControl);
    /// In-place processing of n samples. `analysis` is the mono analysis signal (len n) used
    /// for pitch marks; `io[ch]` holds input and is overwritten with output (all len n).
    pub fn process(&mut self, analysis: &[f32], io: &mut [&mut [f32]]);
    /// Instantaneous input pitch (cents) from mark spacing when the mark tracker is locked.
    pub fn mark_cents(&self) -> Option<f32>;
}
```

Requirements: identical grain timing and ratio on both channels (marks come from `analysis`); unvoiced input → exact delayed dry `x[t−D]` with equal-gain crossfades (R§8.18); ratio 1.0 + formant 1.0 + voiced must be near-transparent (output vs `x[t−D]`, allowing the PSOLA content wander: segmental SNR ≥ 25 dB on a steady vowel after aligning by the measured wander); Classic padded to D in Live/Live+/Studio.

Tests: fixed-ratio shifts (±1, ±3, ±7, ±12 st) of synthetic vowels at 100/220/440 Hz: output f0 within 3 c of target (offline reference tracker in the test), formant envelope preserved (LPC-envelope log-spectral distance between input and output < 3 dB for PSOLA; Classic expected to move formants); time-varying ratio (vibrato-like ratio sweeps) without clicks (energy above 8 kHz at grain rate below threshold); unvoiced noise passes as exact delay; latency formula covers lookahead (assert no read beyond the newest written sample, via debug counters); block-size independence; stereo channels get identical timing.

Benches: `process()` of 64/128/512 samples per engine × tier × mode at 48 kHz.

### 3.5 `vt_engine`

```rust
use vt_core::{EngineConfig, MidiEvent, PitchFrame, Settings};

/// Reported latency for a config (delegates to vt_shift::latency_samples).
pub fn latency_samples(cfg: &EngineConfig) -> u32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedMidi { pub timing: u32, pub event: MidiEvent }

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EngineStats { pub hops: u64, pub voiced_hops: u64, pub telemetry_dropped: u64 }

pub struct Engine { /* Detector, Corrector, Shifter, analysis scratch, telemetry producer */ }
impl Engine {
    pub fn new(cfg: EngineConfig) -> Self;
    pub fn config(&self) -> &EngineConfig;
    pub fn latency_samples(&self) -> u32;
    pub fn reset(&mut self);
    pub fn set_settings(&mut self, s: &Settings);
    /// Process a host block in place. Any length (internally chunked to max_block). MIDI events
    /// must be sorted by `timing` (sample offset within this block); each is applied at its
    /// sample offset rounded to the enclosing hop boundary.
    pub fn process(&mut self, io: &mut [&mut [f32]], midi: &[TimedMidi]);
    /// The GUI side of the telemetry ring (capacity ≥ 4096 frames). Returns Some once.
    pub fn take_telemetry(&mut self) -> Option<rtrb::Consumer<PitchFrame>>;
    /// Telemetry is only produced while enabled (editor open). Cheap atomic-free flag set by
    /// the plugin each block.
    pub fn set_telemetry_enabled(&mut self, on: bool);
    pub fn stats(&self) -> EngineStats;
    /// Effective scale after MIDI Learn (forwarded from Corrector).
    pub fn learned_states(&self) -> Option<[vt_core::NoteState; 12]>;
}
```

Responsibilities: sanitize NaN/Inf; build the analysis signal from `Settings::analysis_channel`; split at hop boundaries and MIDI timings; per-hop pipeline (§2.2); Studio `Alignment::Centered { delay: D }` and `lookahead_hops = D / hop`; fill `PitchFrame::sample_pos = est.sample_pos + D`; drop telemetry when the ring is full (count it).

Tests: zero allocations in `process()` after construction (separate test binary with a counting `#[global_allocator]`, including a negative control); block-size invariance over {1, 7, 32, 64, 128, 480, 4096, random}; measured latency == reported for every mode × range at 44.1/48/96 kHz (cross-correlation on unvoiced noise bursts); NaN input recovers.

### 3.6 `vocal_tune` (plugin)

- nice-plug `Plugin` + `ClapPlugin` + `Vst3Plugin`, exported with `nice_export_clap!` / `nice_export_vst3!`. Stable `CLAP_ID = "com.vocaltune.vocal-tune"`, fixed `VST3_CLASS_ID`, stable param `#[id]`s from day one.
- `AUDIO_IO_LAYOUTS`: stereo→stereo (first/default), mono→mono, mono→stereo (upmix). `MIDI_INPUT = MidiConfig::Basic`. `SAMPLE_ACCURATE_AUTOMATION = false`.
- `activate()`: build `Engine` from `EngineConfig` (sample rate, max block, channels, latency mode, voice range, live+/studio ms), report `latency_samples`. Changing a structural param (latency mode, voice range, live+/studio ms) calls `context.set_latency_samples()` once and requests a restart; the engine is rebuilt in the next `activate()`.
- Params (all with sensible ranges/skews/units/formatters; non-automatable where noted):
  - Structural (non-automatable): `latency_mode`, `voice_range`, `live_plus_ms` (5-10), `studio_ms` (20-50).
  - Runtime: `tier` (non-automatable), `engine` (Classic/Natural), `key`, `scale`, 12 note overrides `note_<pc>` ∈ {Scale, On, Bypass, Remove}, `retune_ms` (0-400, skewed, default 20), `transition_link` (bool), `transition_ms` (0-800), `humanize`, `flex_tune`, `amount` (%), `detune_cents`, `transpose_st` (±12, step 1), `formant_st` (±12), `formant_follow`, `preserve_vibrato`, `ignore_vibrato`, `vibrato_depth` (-100..200 %), `tolerance_cents`, `tolerance_ms`, `snap_min_cents`, `snap_max_st` (0 = off), `tracking`, `gate_db`, `analysis_channel`, synthetic vibrato (`lfo_on`, `lfo_rate`, `lfo_depth`, `lfo_delay`, `lfo_ramp`), MIDI (`midi_mode`, `midi_all_octaves`, `bend_range`), `output_gain_db`, `presets` are GUI actions writing params.
  - Settings mapping resolves `note_states[pc]` = override, or scale membership (On/Remove) when the override is `Scale`.
- `process()`: copy params → `Settings` (no allocation), convert nice-plug note events to `TimedMidi` into a fixed-capacity array (cap 512; drop extra), `engine.process()`, apply output gain, measure CPU with `Instant` (EMA + peak into `AtomicF32`s for the GUI).
- Editor (nice-plug-egui), fixed 1100×720 logical px, dark theme:
  - Header: name, latency mode / voice range / tier / engine selectors, reported latency (ms), CPU meter.
  - Left: key + scale selectors, presets (Hard Tune / Natural / Live Safe / Classic, R§7.6.4), a vertical 12-note keyboard showing each pitch class's effective state (click cycles Scale → On → Bypass → Remove), MIDI-learned notes highlighted.
  - Centre: **pitch graph** (R§8.23): 6 s scrolling history, piano-roll lanes with scale notes shaded and labelled, target note bars, detected trace (thin, alpha = confidence, gaps when unvoiced), centre line, corrected output trace (thick, accent colour), vibrato band, live cents meter (input deviation from target, and applied correction), note name of the current target. Smooth scrolling from interpolated output time; vertical auto-follow with manual zoom/scroll. History buffers preallocated; drawing reuses buffers.
  - Bottom: knob row (Retune, Transition, Humanize, Flex-Tune, Amount, Vibrato, Formant, Transpose, Detune, Tracking) and an "Advanced" collapsible (tolerance, snap, gate, LFO, MIDI, analysis channel, output gain).
  - The GUI never blocks the audio thread: telemetry via the engine's rtrb consumer handed over in `activate()` through a `triple_buffer`/`Mutex` on the **GUI side only**; scalar meters via atomics.
- Standalone: `src/main.rs` behind feature `standalone` (`nice_export_standalone`), with a device picker like `~/code/aiudio/src/main.rs` if cpal is available through nice-plug's standalone deps; otherwise plain CLI flags.

### 3.7 `vt_eval`

- `synth`: band-limited glottal source (Rosenberg/LF-like, f0 trajectory per sample, jitter, shimmer, aspiration noise) through formant resonators (vowel presets a/e/i/o/u, male and female sets) → `Signal { audio, f0_truth (Hz per sample, 0 = unvoiced), target_cents (intended in-tune note per sample, NaN if none) }`. Builders: `Melody` (notes, durations, per-note detune, scoops, overshoot, OU drift, vibrato rate/extent, legato/gaps), sibilant and breath bursts, silence, noise bursts, pulse trains, sweeps (A2-A3-A2), ±40 c steps.
- `reference`: offline, non-causal, high-accuracy f0 tracker in f64 (centred NSDF with long window, parabolic + continuity) for measuring rendered output. Validated against `f0_truth` in its own tests (median < 0.5 c on clean synthetic vowels).
- `render`: run `Engine` over a `Signal` with a block-size pattern; returns output + telemetry frames.
- `metrics`: tuning error vs target on stable voiced frames (exclude ±30 ms around note boundaries; vibrato notes measured on the centre), % within 5/10/25 c, settling time, added jitter (>12 Hz), voicing F1 of telemetry vs truth, measured latency (xcorr on noise bursts), click detector, formant LSD.
- CLI `vt_eval`: `synth <scenario> <out.wav>`, `render <in.wav> <out.wav> [--mode --range --tier --engine --retune --key --scale ...]`, `suite [--out out/]` (runs all scenarios × modes × tiers, writes WAVs + CSV + a Markdown table to `out/report.md`), `latency`, `rt-bench` (tail-latency histogram per mode/tier/buffer: p50/p99/p99.9/max, exit 1 if p99.9 > 10 % of the buffer period at 32/64/128).
- Integration tests (`tests/`): the quality gates of §4 at a smaller scale so `cargo test` stays < ~60 s.
- Benches (`benches/engine.rs`, criterion): `Engine::process` at 32/64/128/512 buffers × modes × tiers at 48 kHz.

---

## 4. Gates (v1 definition of done)

| Gate | Target |
|---|---|
| Tuning (Live HQ, detuned melody, retune 0-20 ms) | median \|error\| < 5 c, p95 < 15 c on stable voiced frames |
| Tuning (Studio HQ) | median < 3 c |
| Settling (Live, retune 0) | < 30 ms to stay within 10 c after onset |
| Vibrato preservation | PreserveVibrato on, τ = 0: output vibrato extent within ±15 % of input, centre error median < 8 c |
| Added jitter | < 3 c RMS (>12 Hz) on steady notes |
| Voicing F1 (detector vs truth) | ≥ 0.97 |
| Latency | measured == reported (±1 sample) for every mode × range × sample rate |
| Transparency | amount 0: output == input delayed by D (Classic exact; PSOLA within the wander tolerance) |
| RT safety | zero allocations in `process()`; no panics on NaN/extreme input; block-size invariance |
| CPU (M4 Max, 48 kHz) | Live HQ p99.9 block time < 5 % of the buffer period at 128; < 10 % at 32; Max < 25 % |
| Perf proofs | each SIMD/FFT/fast-math optimisation shows a statistically significant speedup in criterion vs its scalar/std baseline (numbers recorded in `docs/PERFORMANCE.md`) |
| Plugin | builds/bundles VST3 + CLAP; clap-validator passes; loads in Bitwig and Ableton |
