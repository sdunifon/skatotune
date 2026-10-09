# Vocal Tune: Rust Performance and Real-Time Engineering

**Date:** 2026-10-08
**Status:** performance architecture v1. Companion to [`research.md`](research.md), the architecture baseline.
**Scope:** how to make the DSP fast in Rust (algorithms, SIMD, codegen, numerics, build settings) and keep it real-time safe (memory, threads, latency), plus the test plan that proves both.

**Where this comes from**

- Two research reports (SIMD/codegen/numerics, and real-time systems), each checked by an adversarial verifier. Wherever a verifier corrected or refuted a claim, the corrected version is used here. Section 8 lists the corrections.
- Spot checks made while writing:
  - the nice-plug 0.4.2 sources in the local cargo registry;
  - three new probes on this machine (exact-size FFT YIN with realfft and vDSP, phase-interpolated polyphase SNR, fixed-phase FIR grain read);
  - a scratch crate (kept outside the repository) that compiles the main code patterns in this document and tests them (7/7 pass).
- "(measured, M4 Max)" means measured on the dev machine: Apple M4 Max, 12 P-cores + 4 E-cores, macOS 27.2, release build, f32. Other jobs were running during most runs (load average 5-62), so absolute numbers carry about ±10% noise. Comparisons made inside one run are reliable.
- "(estimate)" means an engineering estimate. "(design choice)" means our own decision.
- **Budget machine.** Release budgets are set against a base M1, which is slower than the M4 Max, as in research.md §10.7. Gates on the M4 Max are therefore tighter.

**What this document changes in research.md** (update that file once these are accepted)

| research.md | Was | Now | Why |
|---|---|---|---|
| §9.2 `wide` as the explicit SIMD crate; fearless_simd "only when shipping x86" | `wide` 1.7.1 | **fearless_simd 1.1.0** for all portable kernels, with wide logical vectors (f32x16) | It matches hand-written NEON on the M4 and dispatches at runtime on x86. `wide` is compile-time only, so a default x86 build is stuck at SSE2 (§2.2) |
| §8.4 `N = next_pow2(W + τ_max)` | power of two | **N = smallest f·2^k ≥ W+τ_max, f ∈ {1,3,5,15}** | 11-25% faster, same accuracy (measured, §6.2) |
| §9.4 rule 5 "FTZ only needed on x86" and the implied assumption that the framework sets it | | **nice-plug 0.4.2 does *not* set FTZ by default** (opt-in feature `unsafe_flush_denormals`). We set our own guard in process() and in every worker task | Gives the same numerics on every thread and in vt_eval, and is required for x86 (§3.6) |
| §9.6 "No real-time worker threads of our own" | inline only | Inline core stays authoritative. **Optional assist jobs** run on a process-wide pool of real-time worker threads, each joined to its own Apple work interval, pipelined one hop behind and never waited on | Under load, RT workers missed 0 of 1500 deadlines, against 2-3% for normal or QoS workers (measured). This is what delivers "quality scales with CPU" (§4.5) |
| §8.16 "8-16-tap Kaiser-windowed sinc" (table phase handling unspecified) | | **Interpolate linearly between table phases.** A truncated phase index costs 20-50 dB of SNR | Truncated: −55 to −83 dB. Interpolated: −106 to −121 dB, at about zero extra cost (measured, §6.4) |
| §10.7 CPU budgets | p99 per kernel and block | Adds the paced per-call WCET/jitter matrix (32/64/128/512), RTF gates, load and E-core axes, and pass/fail tables (§5) | |

---

## Contents

1. Executive summary
2. Technique register (adopt / consider / reject)
3. Recipes: toolchain, profiles, SIMD, hot kernels, FFT, denormals, fast math, codegen hygiene
4. Real-time architecture rules
5. Performance and RT-safety test plan
6. Local measurements (M4 Max)
7. Risks and open questions
8. Corrections applied from verification
- Appendix A: sources

---

## 1. Executive summary

### 1.1 Strategy

Work in this order: **algorithm first, then portable SIMD, then threads, and prove every step with a gate.**

1. **Algorithm.** The per-hop analysis cost is dominated by the difference function and autocorrelation.
   - Compute them by FFT at an **exact mixed-radix size** (N = W+τ_max, rounded up to f·2^k).
   - Share the forward transform between the NSDF and the type-I CMNDF.
   - On the M4 this takes a 1024/512 analysis from 304 µs (naive) to 4.3 µs (realfft) or 3.4 µs (Apple vDSP), a 70-90x gain that no SIMD trick can match.
   - A whole prototype block of 128 samples costs 5-6 µs, about 0.2% of one core (measured).
2. **Portable SIMD.**
   - Write each hot kernel once, generically over `fearless_simd::Simd` with wide logical vectors (`f32x16`).
   - Detect the SIMD level once in `activate()` and dispatch with `dispatch!`.
   - Keep a scalar f64 oracle beside every kernel.
   - On macOS, put Accelerate (vDSP DFT, BLAS) behind the same traits.
   - Hand-written `core::arch` NEON is allowed only where a benchmark shows more than 10% on a kernel that matters to the budget. Today no kernel qualifies.
3. **Codegen discipline.**
   - Use multi-accumulator or wide-vector reductions, because Rust never reorders a strict f32 sum by itself.
   - Never call bare `f32::mul_add` in code compiled for baseline x86: it becomes a `fmaf` libcall that runs 26-28x slower.
   - Slice once per kernel so bounds checks disappear. Pad buffers to vector multiples so hot kernels have no scalar tails.
   - Build with fat LTO and codegen-units = 1. Do not ship `target-cpu=native`, and do not use PGO (0% measured gain).
4. **Real-time core.**
   - The audio thread does a bounded, allocation-free, lock-free and syscall-free amount of work per sample, driven by a fixed internal hop that is independent of the host block size.
   - It talks to the GUI and the workers only through `rtrb`, `triple_buffer` and atomics.
   - Everything is preallocated and pre-faulted.
5. **Spare CPU becomes quality, not latency.**
   - Heavier analysis (harmonic re-scoring, the neural octave/voicing arbiter, second-opinion trackers, Studio envelope work) runs as **assist jobs** on a process-wide pool of real-time threads. On macOS each thread joins its own `AudioWorkIntervalCreate` work interval.
   - Live mode consumes results one hop later; Studio consumes them within its lookahead.
   - A late result is never waited for: the inline estimate stands, and a demote-only governor sheds assist work under load.
   - Latency is fixed per mode and never depends on CPU.
6. **Proof.** Every optimization ships with:
   - an equivalence test against the f64 oracle at every SIMD level;
   - an error-bound or SNR test;
   - a scalar-vs-SIMD A/B benchmark.

   On top of those, the build is gated on: zero allocations, a clean RealtimeSanitizer run, exact latency, bit-identical output across host block patterns, paced per-call WCET histograms at 32/64/128/512 samples against fixed budgets, real-time-factor limits, worker deadline misses under realistic load, and x86 fallback paths run under Rosetta.

### 1.2 Key decisions

| Area | Decision |
|---|---|
| Toolchain | Rust **1.99.0** pinned (already in `rust-toolchain.toml`). This brings algebraic float ops and NEON `vdotq_s32` (1.98), `array_windows` (1.94), `as_chunks` (1.88), safe intrinsics (1.87) and safe `#[target_feature]` (1.86). Add `x86_64-apple-darwin` to the toolchain targets for Rosetta tests. |
| SIMD layer | **fearless_simd 1.1.0** + fearless_simd_macros 0.1.0. Kernels use `f32x16` (four q-registers on NEON, two ymm on AVX2, one zmm on AVX-512). Detect the `Level` once per instance. |
| FFT | `vt_core::fft::RealFft` trait. **vDSP DFT** (`vDSP_DFT_zrop`) on macOS, **realfft 3.5.0** (`process_with_scratch` only) elsewhere. Sizes are f·2^k with f ∈ {1,3,5,15}, k ≥ 4. phastft is rejected for now. |
| FMA | Fused on aarch64. On x86, fused only inside FMA-enabled paths (fearless `.mul_add()` on AVX2 or multiversion clones), never bare `f32::mul_add`. |
| Numerics | Fast exp2/log2/tanh with tested bounds (errors ≤ 0.001 cent). f64 for sample positions and energy accumulators. Our own `FtzGuard` (FZ on aarch64, FTZ+DAZ on x86) in process() and in every worker task. |
| Interpolation | 16-tap Blackman-Harris polyphase, 512 phases, **linear interpolation between phases** (c = c0 + a·dc). |
| Build | `opt-level=3`, `lto="fat"`, `codegen-units=1`, `debug="line-tables-only"`, `panic="unwind"` (already in Cargo.toml). An x86-64-v2 floor plus runtime dispatch when x86 ships. |
| Audio thread | Allocate in `activate()` (cached by configuration), pre-fault, fixed hop, rtrb/triple_buffer/atomics only, no locks, logging, `thread::current()` or channels. |
| Workers | One refcounted, process-wide pool. Each worker: `audio_thread_priority` 0.38.0 promotion, its own work interval on macOS, rtrb + `Thread::unpark` wake-ups, never spins. |
| Pipeline | A sample ring owns the job windows. Results are drained to the newest. Late results fall back to the inline estimate. The assist runs inline, with the same one-hop lag, in offline or faster-than-real-time rendering. |
| Proof | divan kernel A/B tables; criterion block baselines plus a `VT_SIMD=fallback` A/B; gungraun instruction counts on Linux CI; paced hdrhistogram WCET harness; RTF harness; RTSan; counting-allocator and page-fault tests; exact-latency and block-invariance tests; Rosetta SSE4.2 and AVX runs. |

### 1.3 Headline numbers (measured, M4 Max)

| What | Before | After | Gain |
|---|---|---|---|
| Type-I difference W=1024, L=512 | 304 µs naive | 4.33-4.47 µs realfft N=1536; 3.39-3.44 µs vDSP N=1536 | 70-90x |
| Same, W=2048, L=1024 | 1.24 ms naive | 9.13 µs realfft N=3072; 6.16-6.20 µs vDSP N=3072 | 136-200x |
| Prototype block, 128 samples (YIN + CMNDF + decision + interpolation) | 305 µs (11.4% of 2667 µs) | 6.3 µs realfft / 4.86 µs vDSP (pow-2 N; 0.18-0.24%) | 48-63x |
| Dot product, 2048 elements | 909 ns iterator sum | 80.9-82.4 ns fearless `f32x16` x2 | 11x, equal to hand NEON (83.6 ns) |
| exp2 / log2 / tanh, 512 values | 543 / 734 / 801 ns std | 184 / 237 / 153 ns | 2.95x / 3.1x / 5.2x |
| Polyphase SNR at a semitone ratio | −55 to −83 dB, truncated phase | −106 to −121 dB, interpolated phase | +23-51 dB at no measurable cost |
| Worker deadline misses, 16 busy threads | 48/1500 normal priority, 31/1500 QoS | 0/1500 time-constraint | — |
| Worker compute p99, idle machine | 2.36-2.45 ms RT only | 0.43-0.73 ms RT + own work interval | 3.2-5.7x |
| Studio-size block tail, p99.9 | 63% of budget naive | 1.0-1.1% FFT (P-core). Under background QoS (E-cores): 40%, max 33 ms | — |

The last row is the main warning in this document. Hot-loop P-core numbers say little about a thread the scheduler is allowed to demote. That is why the worker architecture (§4.5) and the load and E-core axes of the test plan (§5.6) exist.

---

## 2. Technique register

Verdicts: **Adopt** = do it now. **Consider** = keep as an option; adopt when a benchmark or a feature needs it. **Reject** = do not do it, with the reason given.

"Stable" means it builds on the pinned 1.99.0 toolchain. The minimum version is given where it matters, because the machine default is still 1.89. Versions are current as of 2026-10-08 (crates.io).

### 2.1 Algorithms and libraries

| Technique | Verdict | Expected / measured gain | Stable | Crate / version | Portability | Source |
|---|---|---|---|---|---|---|
| FFT-based difference function / autocorrelation (`d = e0 + eτ − 2r`) | **Adopt** | 70-200x vs naive; 3-5x vs NEON direct form (measured) | yes | realfft 3.5.0, rustfft 6.4.1 | all | [realfft docs](https://docs.rs/realfft/3.5.0/realfft/) |
| Exact mixed-radix N = f·2^k ≥ W+τ_max, f ∈ {1,3,5,15} | **Adopt** | realfft −22 to −25%, vDSP −11 to −20%; error unchanged (measured) | yes | same | all; also valid vDSP zrop lengths (f·2^n, n ≥ 4) | MacOSX27.0 SDK `vDSP.h`; §6.2 |
| One forward FFT of the frame shared by NSDF (`IFFT(abs(A)^2)`) and type-I cross term (`IFFT(conj(A)·B)`) | **Adopt** | 4 transforms instead of 5 per hop (estimate: about −15%) | yes | — | all | research.md §8.4 |
| Apple vDSP DFT (`vDSP_DFT_zrop_CreateSetup` / `vDSP_DFT_Execute`) via about 40 lines of FFI | **Adopt** (macOS) | Raw FFT 2.0-2.9x faster than realfft for N = 512-8192. At YIN level 1.3-1.5x. Whole block −26% (Live) / −31% (Studio), p < 0.01 (measured) | yes (FFI) | none (system framework) | macOS/iOS | [Apple](https://developer.apple.com/documentation/accelerate/vdsp_dft_zrop_createsetup), [forum](https://developer.apple.com/forums/thread/23321) |
| Legacy `vDSP_fft_zrip` | Reject | Erratic (681 ns at 1024, though once faster at 8192) | yes | — | macOS | §6.3 |
| phastft (pure-Rust, fearless_simd based) | Reject for now | 1.36-1.51x faster than realfft, but it depends on **fearless_simd 0.5.0**: two SIMD stacks that cannot share a `Level`. Its c2r helpers allocate | yes (MSRV 1.88) | phastft 0.4.1 | all, power-of-two sizes only | [crates.io](https://crates.io/crates/phastft) |
| Direct-form NEON YIN with lag blocking | Reject | 16.1 µs vs 3.4-4.3 µs FFT; moot | yes | — | aarch64 | §6.2 |
| Sliding / incremental correlation for tiny hops | Consider | O(H·L) per hop, likely 1-2 µs with SIMD for H=64, L=512 (estimate). The E/H candidate tracker (research.md §8.3) already is one | yes | — | all | — |
| Polyphase windowed sinc (16 taps, 512 phases) with **linear interpolation between phases** | **Adopt** | SNR −106 to −121 dB vs −55 to −83 dB truncated; 310-349 ns vs 304-419 ns per 512 outputs (measured) | yes | — | all | [CCRMA resample](https://ccrma.stanford.edu/~jos/resample/); §6.4 |
| Fixed-phase FIR across outputs for γ = 1 grains | Reject | 1.0-1.1x (measured); a second code path for nothing | yes | — | all | §6.4 |
| Grain-major loop order for PSOLA overlap-add | Consider | 44-54 vs 62-74 ns per output (verifier rerun). The original AoS vs SoA result did **not** reproduce. Re-measure with real kernels | yes | — | all | §6.6 |
| Accelerate `cblas_sgemv` for neural mat-vec | Consider (worker only) | 768×256: 1.60-1.70 µs vs 9.5 µs NEON vs 54.8 µs naive. Single-threaded (cpu/wall = 1.00); consistent with the matrix unit, which is **shared per cluster** | yes (FFI) | none | macOS | [Apple BLAS](https://developer.apple.com/documentation/accelerate/blas) |
| Accelerate BNNSGraph | Consider (worker only) | Not measured. Apple positions it for audio models, with a single-threaded option and no allocation at execution | yes (FFI) | none | macOS | [WWDC24 10211](https://developer.apple.com/videos/play/wwdc2024/10211/) |
| int8 weights + `vdotq_s32` | Consider | 4x fewer weight bytes for a bandwidth-bound GRU/CNN (estimate). `dotprod` is in the apple-m1 baseline, so no detection is needed on macOS | **1.98+** | core::arch | aarch64; x86 VNNI or AVX2 `maddubs` via dispatch | [vdotq_s32](https://doc.rust-lang.org/stable/core/arch/aarch64/fn.vdotq_s32.html) |
| SME / AMX intrinsics directly | Reject | Detection and intrinsics are unstable; reach the matrix unit through Accelerate | no | — | — | — |

### 2.2 SIMD and codegen

| Technique | Verdict | Expected / measured gain | Stable | Crate / version | Portability | Source |
|---|---|---|---|---|---|---|
| **fearless_simd** generic kernels, `#[fearless_simd_macros::simd]`, `dispatch!`, `Level` detected once | **Adopt** | Dot 2048: 80.9-82.4 ns with `f32x16` x2 accumulators, against 90.6 ns with `f32x4` x4, 83.6 ns hand NEON and 909 ns naive. Under Rosetta SSE4.2: 99 ns, best of all crates | yes (MSRV 1.89) | fearless_simd 1.1.0 (2026-10-06), fearless_simd_macros 0.1.0 | NEON; SSE2, SSE4.2, AVX2(+FMA), AVX-512 at runtime; WASM; fallback | [docs.rs](https://docs.rs/fearless_simd/1.1.0/fearless_simd/), [GitHub](https://github.com/linebender/fearless_simd) |
| Wide logical vectors (`f32x16` or `S::f32s`) instead of `f32x4` | **Adopt** | Matches hand NEON on M4. On AVX2 it emits ymm `vfmadd` where the `f32x4` pattern emits xmm only (disassembly checked) | yes | same | all | verification of the SIMD report |
| `Level::fallback()` (dev feature `force_support_fallback`) for tests and A/B benches | **Adopt** | Runs every kernel's fallback path natively on the Mac | yes | same | all | fearless_simd `src/lib.rs` |
| Hand-written `core::arch::aarch64` NEON | Consider (by exception) | 0-3% over fearless `f32x16` for dot; earlier 2.4x claims compared against scalar code. Allowed only with a benchmark showing more than 10% on a budget-relevant kernel, plus an equivalence test | yes (safe-call rule 1.87) | core::arch | aarch64 only | [1.87 notes](https://blog.rust-lang.org/2025/05/15/Rust-1.87.0/) |
| FMA policy: no bare `f32::mul_add` in code compiled for baseline x86 | **Adopt** | Avoids a 26-28x slowdown (`jmpq *fmaf@GOTPCREL` per element or lane, measured under Rosetta). On M4, `mul_add` is 10-15% faster than mul+add | yes | — | x86 hazard; aarch64 always fused | [f32::mul_add](https://doc.rust-lang.org/std/primitive.f32.html#method.mul_add) |
| Reduction-friendly plain-Rust shapes (`[[f32;4];K]` accumulators, `as_chunks`) | **Adopt** for warm code | 909 → 98-113 ns (full `fmla.4s`). `[f32;16]` accumulators gave only 2-lane `fadd.2s` (117 ns) | yes (`as_chunks` 1.88) | — | all; codegen fragile across LLVM versions | [as_chunks](https://doc.rust-lang.org/std/primitive.slice.html#method.as_chunks) |
| Algebraic float ops (`algebraic_add` / `algebraic_mul`) | Consider for warm code | 909 → 118 ns from a one-line fold. Results are non-deterministic, so they need tolerance tests (§3.3.4) | **1.98+** | std | all | [1.98 notes](https://blog.rust-lang.org/2026/08/20/Rust-1.98.0/) |
| multiversion | Consider | x86 clones for plain-Rust warm loops, plus `match_target!` for FMA: 96-107 ns on M4, 99 ns under Rosetta SSE4.2 | yes (MSRV 1.86) | multiversion 0.9.0 | all | [docs.rs](https://docs.rs/multiversion/0.9.0/multiversion/) |
| pulp | Reject as primary | 91 ns on M4, but no SSE tier (pre-AVX2 x86 runs scalar), and strict `mul_add_f32s` on its fallback calls `fmaf` (16x slower). Only relevant if we adopt faer | yes | pulp 0.22.3 | NEON, V3, V4, scalar | [docs.rs](https://docs.rs/pulp/0.22.3/pulp/) |
| wide | Reject as primary | 90.8 ns on M4, but compile-time cfg only: one shipped x86 binary never uses AVX2 | yes (MSRV 1.89) | wide 1.7.1 | compile-time only | [docs.rs](https://docs.rs/wide/1.7.1/wide/) |
| `std::simd` (portable_simd) | Reject | 95.4 ns, no better than stable crates; nightly only (tracking issue open, needs-rfc) | **no** | std | — | [#86656](https://github.com/rust-lang/rust/issues/86656) |
| macerator / simdeez | Reject | macerator needs Rust 1.94 and is about 9.6% documented; simdeez has low adoption | — | 0.5.1 / 3.0.1 | x86-centric | [crates.io](https://crates.io/crates/macerator) |
| Bounds-check elimination: slice once at entry, `chunks_exact`/`as_chunks`, zip; `core::hint::assert_unchecked` only with a `debug_assert!` twin | **Adopt** | An indexed loop with bounds checks took 1195 ns vs 909 ns for an iterator (a `panic_bounds_check` branch inside the loop) | yes (1.81) | — | all | [assert_unchecked](https://doc.rust-lang.org/core/hint/fn.assert_unchecked.html) |
| Buffers padded to multiples of 16 floats; no scalar tails in hot kernels | **Adopt** (design choice) | Removes tail cost. More importantly, it removes fused-vs-unfused differences between body and tail, which would break bit-exact block invariance (§3.3.3) | yes | — | all | — |
| Software prefetch | Reject | Hardware prefetchers handle linear streams; the `_prefetch` intrinsic is unstable (`asm!("prfm …")` works) | asm only | — | per arch | [#117217](https://github.com/rust-lang/rust/issues/117217) |

### 2.3 Numerics

| Technique | Verdict | Expected / measured gain | Stable | Crate / version | Portability | Source |
|---|---|---|---|---|---|---|
| Fast exp2 (Cephes deg-6), log2 (atanh series), tanh (Padé 7/6) | **Adopt** | 2.95x / 3.1x / 5.2x. Max errors: exp2 rel 9.7e-8 (0.00017 cent), log2 abs 5.8e-7 (0.0007 cent), tanh abs 9.6e-5 | yes (`round_ties_even` 1.77) | — | all | [Cephes](https://www.netlib.org/cephes/); §3.7 |
| Windows, sinc and other tables precomputed off the audio thread and shared across instances | **Adopt** | Removes transcendentals from hot loops; a shared immutable `Arc` improves cache use with many instances (estimate) | yes | — | all | — |
| `FtzGuard`: FPCR.FZ on aarch64, MXCSR FTZ+DAZ (0x8040) on x86, scoped, in process() and every worker task | **Adopt** | No subnormal penalty on M4 (332 vs 332 ns; reruns 418/403/378 ns). Needed for x86 and for identical results across threads and in offline renders. **Spawned threads do not inherit FZ** (tested) | yes (`asm!`) | own 40 lines (or no_denormals 0.3.0) | aarch64, x86_64 | [nice-plug util.rs](https://codeberg.org/RustAudio/nice-plug); §3.6 |
| nice-plug `unsafe_flush_denormals` feature | Consider | Sets FZ on aarch64 and only FTZ (no DAZ) on x86, around process() only. Our guard supersedes it | yes | nice-plug 0.4.2 | aarch64, x86 | local source check |
| Denormal-safe recursions independent of the FP environment (explicit flush below 1e-20 in feedback paths) | **Adopt** for IIR/feedback | Deterministic on x86 whatever the thread's FP state (estimate) | yes | — | all | — |
| f64 sample positions and energy running sums | **Adopt** | FFT-YIN error stays at 5e-7 to 1.1e-6 of signal energy (measured). f32 positions cause audible period jitter (research.md §8.16) | yes | — | all | — |

### 2.4 Build and toolchain

| Technique | Verdict | Expected / measured gain | Stable | Notes / source |
|---|---|---|---|---|
| `opt-level=3`, `lto="fat"`, `codegen-units=1`, `debug="line-tables-only"` | **Adopt** (in Cargo.toml) | 4-11% vs the default release (within the ±10% noise floor; not re-verified) | yes | [Cargo profiles](https://doc.rust-lang.org/cargo/reference/profiles.html) |
| `panic="abort"` | Reject | −5.8% binary size, identical speed. `unwind` lets worker threads `catch_unwind` and degrade; since Rust 1.81 a panic escaping `extern "C"` aborts anyway | yes | [1.81 notes](https://blog.rust-lang.org/2024/09/05/Rust-1.81.0/) |
| `target-cpu=native` | Reject | 0% (pipeline 6.203 vs 6.201 µs). Risks SIGILL on older Macs. The aarch64-apple-darwin default is already `apple-m1`, which includes dotprod, fp16, fhm and rdm | yes | [codegen options](https://doc.rust-lang.org/rustc/codegen-options/index.html#target-cpu) |
| x86-64-v2 floor plus runtime dispatch for v3/v4 | **Adopt** when x86 ships | Covers CPUs since about 2009; Rosetta supports it | yes | — |
| PGO (cargo-pgo 0.3.0) | Reject | 0% on the SIMD and FFT kernels (5.59 vs 5.54-5.66 µs). Revisit only if branchy control code dominates a profile | yes | [cargo-pgo](https://github.com/Kobzol/cargo-pgo) |
| Rust 1.99.0 pinned | **Adopt** (done) | Needed by egui 0.36, algebraic ops and `vdotq_s32` | — | [1.98 notes](https://blog.rust-lang.org/2026/08/20/Rust-1.98.0/) |

### 2.5 Real-time plumbing

| Technique | Verdict | Evidence | Stable | Crate / version | Source |
|---|---|---|---|---|---|
| Allocate in `activate()`, cached by (sr, max block, mode, voice range, tier); immutable tables shared process-wide | **Adopt** | RTSan flags realloc/free/calloc otherwise; 20k random blocks gave 0 violations | yes | — | nice-plug `plugin.rs` |
| Pre-fault (one write per 16 KB page) and `mlock` buffers larger than 64 KB | **Adopt** | First touch after `vec![0.0; n]`: +230-250 µs at 4 MB, +800 µs at 16 MB, which neither assert_no_alloc nor RTSan sees | yes (libc) | libc | verifier probe `review/pf` |
| Fixed-capacity collections | **Adopt** | Allocation-free by construction | yes | arrayvec 0.7.8, heapless 0.9.3 | crates.io |
| `sort_unstable` / `select_nth_unstable` only | **Adopt** | `slice::sort` (driftsort) can allocate scratch | yes | std | [std docs](https://doc.rust-lang.org/std/primitive.slice.html#method.sort_unstable) |
| Power-of-two const-generic ring with mask indexing and `wrapping_*` | **Adopt** | Mask 3.3-4.5 ns vs branch-wrap 5.6-9.8 ns vs modulo 20.5-24.3 ns (4 taps). The const assert is post-monomorphization: `cargo build` catches it, `cargo check` does not | yes | — | §3.4 |
| rtrb SPSC rings (jobs, results, GUI frames) | **Adopt** | push+pop 2.65 ns; RTSan clean | yes | rtrb 0.4.0 (2026-08-17) | [docs.rs](https://docs.rs/rtrb/0.4.0/rtrb/) |
| triple_buffer for full-state GUI snapshots | **Adopt** | 2 KB publish 43.5 ns, read 0.22 ns; RTSan clean. Always write the full state | yes (MSRV 1.86) | triple_buffer 9.0.0 | [docs.rs](https://docs.rs/triple_buffer/9.0.0/triple_buffer/struct.Input.html) |
| `AtomicF32` meters, Relaxed, one store per block | **Adopt** | 0.22 ns store vs 4.27 ns per-sample `fetch_max` | yes | atomic_float 1.1.0 | crates.io |
| `CachePadded` (128 B on aarch64 and x86_64) on cross-thread indices and flags | **Adopt** | 1.6-2.8x on contended counters; `hw.cachelinesize` = 128 on M4 | yes | crossbeam-utils 0.8.23 | [cache_padded.rs](https://github.com/crossbeam-rs/crossbeam/blob/master/crossbeam-utils/src/cache_padded.rs) |
| Deferred free: `basedrop` `SharedCell`, or a return rtrb | **Adopt** | basedrop drop and `SharedCell::get` were RTSan-clean; `Arc` last drop and `ArcSwap::store` were flagged | yes | basedrop 0.1.3 | [docs.rs](https://docs.rs/basedrop/0.1.3/basedrop/) |
| std/parking_lot mutexes, crossbeam/std channels, ArcSwap, `println!`, `thread::current()`, nice-plug `execute_background`/`execute_gui` on the audio thread | **Reject** | All flagged by RTSan (malloc, pthread_mutex, cond_signal, write); parking_lot `try_lock` unlock with a waiter takes the slow path | — | parking_lot 0.12.5, crossbeam-channel 0.5.17, arc-swap 1.9.2 | §6.8 |
| `#![no_std]` (without `alloc`) DSP kernel modules | Consider | Makes `Vec`/`format!` impossible in kernels. realfft needs `alloc`, so FFT plans stay in the shell | yes | libm if needed | [embedded book](https://docs.rust-embedded.org/book/intro/no-std.html) |

### 2.6 Threads and scheduling

| Technique | Verdict | Evidence | Stable | Crate / version | Source |
|---|---|---|---|---|---|
| Worker wake-up: rtrb push + `Thread::unpark`; worker `park()`s when empty | **Adopt** | Signal cost on the audio thread p50 0.3-1.0 µs, max ≤ 6.5 µs. Wake latency to an RT worker p99.9 4.5-7.4 µs under full load. Implemented as a dispatch semaphore on Darwin | yes | std | [darwin.rs](https://github.com/rust-lang/rust/blob/master/library/std/src/sys/sync/thread_parking/darwin.rs) |
| Busy-spinning workers | **Reject** | A spinning RT worker observed only 703-1373 of 2950 wakeups. xnu cancels the RT deadline when the quantum (= computation value) expires, and demotes long-running RT threads | — | — | [xnu sched_prim.c](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/sched_prim.c) |
| Real-time priority for workers (THREAD_TIME_CONSTRAINT / MMCSS / rtkit) | **Adopt** | With 16 busy threads: 0/1500 misses vs 48/1500 (normal) and 31/1500 (QoS user-interactive) | yes | audio_thread_priority 0.38.0 (2026-09-17) | [docs.rs](https://docs.rs/audio_thread_priority/0.38.0/audio_thread_priority/) |
| Own work interval per worker (`AudioWorkIntervalCreate` + `os_workgroup_join` + `interval_start/finish` per job) | **Adopt** (macOS) | p99 compute 2.36-2.45 ms → 0.43-0.73 ms at low load. 1491 → 565 µs with 16 RT threads at 90% duty. **No gain** with 12 RT threads at 70% duty (434 vs 416 µs). Creation ≈ 0.55 ms after the CoreAudio HAL is up, 68-91 ms in a process that never touched CoreAudio | yes (FFI) | none | [Apple](https://developer.apple.com/documentation/audiotoolbox/adding-asynchronous-real-time-threads-to-audio-workgroups) |
| Joining the host's device workgroup from VST3/CLAP | **Reject** | No API exists (only AU has a render-context observer); Steinberg declined to add one | — | — | [Blue Cat](https://www.bluecataudio.com/Blog/preview/realtime-audio-multicore-issues-for-apple-silicon-end-of-the-story/) |
| Process-wide refcounted worker pool, sticky instance-to-worker assignment | **Adopt** (design choice) | Per-instance RT threads do not scale to 16 vocal tracks; `os_workgroup_max_parallel_threads` = 12 on M4 Max | yes | std `OnceLock`/`Weak` | [Apple](https://developer.apple.com/documentation/os/os_workgroup_max_parallel_threads) |
| One-hop pipeline: ring-owned windows, drain-to-newest results, never wait | **Adopt** | Audio-thread cost of a 0.4 ms job: max 24 µs offloaded vs 425 µs inline. The verifier's simulation of the report's original pattern: one late job made 90 of 99 hops late; this design makes it 1 (tested, §4.5.4) | yes | rtrb 0.4.0 | §4.5 |
| Demote-only quality governor | **Adopt** (design choice) | Needed so "quality scales with CPU" adapts instead of just counting late jobs | yes | — | research.md §1.5 |
| CLAP thread-pool extension (`request_exec`) | Consider later | Fork-join on the host's RT threads; Bitwig/REAPER only, not in nice-plug, blocks inside process() | — | clack (git) | [thread-pool.h](https://github.com/free-audio/clap/blob/main/include/clap/ext/thread-pool.h) |

### 2.7 Verification tooling

| Tool | Verdict | Use | Stable | Version |
|---|---|---|---|---|
| divan | **Adopt** | Kernel A/B tables (scalar vs SIMD vs backend). Always pass `--sample-size`: the first benchmark of a run was miscalibrated otherwise (40.4 ns shown, true value 14.4 ns) | yes | 0.1.21 (last release 2025-04-10) |
| criterion | **Adopt** | Block-level benchmarks with `--save-baseline`/`--baseline`; `noise_threshold(0.03)`, `significance_level(0.01)` | yes (MSRV 1.86) | 0.8.2 (in workspace) |
| gungraun (successor of iai-callgrind) | **Adopt** | Deterministic instruction counts on Linux CI | yes (MSRV 1.88) | 0.20.0 |
| hdrhistogram | **Adopt** | Per-call WCET histograms; `record()` does not allocate within its bounds | yes (MSRV 1.88) | 7.6.0 |
| rtsan-standalone (LLVM RealtimeSanitizer) | **Adopt** | Locks, syscalls and allocations in the harness and the plugin; also sees C `malloc` inside Accelerate, which a Rust counting allocator cannot | yes (MSRV 1.77) | 0.3.0 |
| nightly `-Zsanitizer=realtime` | Consider | Same runtime through rustc (merged 2025-11-08). The installed nightly (2025-08-31) predates it, so pin a newer nightly | nightly | — |
| assert_no_alloc via nice-plug `assert_process_allocs` | **Adopt** | Debug plugin builds; installs `nice_assert_no_alloc::AllocDisabler` as the global allocator | yes | nice-assert-no-alloc 1.0.0 |
| Counting `#[global_allocator]` test binaries (DSP crates only) | **Adopt** | Zero allocations in `Engine::process`, with a negative control | yes | std |
| loom / shuttle | **Adopt** for hand-rolled atomics (pool registration, shutdown) | Enumerate interleavings | yes | loom 0.7.2, shuttle 0.9.5 |
| Miri, ThreadSanitizer | Consider (nightly CI) | UB and races in non-FFI ring and pipeline code | nightly | — |
| no-panic + clippy `disallowed-methods`/`disallowed-macros` | **Adopt** | Panic-freedom proof for hot kernels; a ban list for the audio-path crates | yes | no-panic 0.1.37 |
| cargo-show-asm / llvm-objdump | **Adopt** | CI disassembly checks: no `fmaf` and no `panic_bounds_check` in hot loops; full-width `fmla`/ymm present | yes | cargo-show-asm 0.2.63 |
| samply, Instruments (cargo-instruments), os_signpost, tracy-client | **Adopt** | Profiling inside the harness and inside DAWs | yes | samply 0.13.1, cargo-instruments 0.4.17, tracy-client 0.19 |
| Rosetta 2 (SSE4.2), `ROSETTA_ADVERTISE_AVX=1` (AVX2/FMA), Intel SDE (AVX-512), native x86 CI | **Adopt** | Exercise every dispatch level for correctness. Rosetta is useless for x86 performance numbers, and Apple says general Rosetta support ends after macOS 27 | yes | — |

---

## 3. Recipes

### 3.1 Toolchain, profiles and config

`rust-toolchain.toml` already pins 1.99.0. Add the x86 target for Rosetta tests. It is Tier 2 with host tools since 1.90, so rustup ships a prebuilt std and `-Zbuild-std` is unnecessary:

```toml
[toolchain]
channel = "1.99.0"
profile = "minimal"
components = ["rustfmt", "clippy"]
targets = ["aarch64-apple-darwin", "x86_64-apple-darwin"]
```

The workspace `Cargo.toml` profiles are already right: `release` uses opt 3, fat LTO, cgu 1 and line tables; `bench` inherits release; `test` uses opt-level 2 with debug assertions on, so overflow checks fire. Keep them as they are, with these additions:

```toml
# Ship builds: same codegen, symbols stripped at packaging (keep the dSYM for crash symbolication).
[profile.dist]
inherits = "release"
split-debuginfo = "packed"
strip = "symbols"

# panic stays "unwind" (the default): workers can catch_unwind and degrade gracefully.
# Panics escaping extern "C" abort anyway (Rust ≥ 1.81), so process() must not panic (§3.8).
```

Add to `[workspace.dependencies]` (the versions are the ones checked above):

```toml
fearless_simd = "1.1.0"
fearless_simd_macros = "0.1.0"
basedrop = "0.1.3"
crossbeam-utils = "0.8.23"
arrayvec = "0.7.8"
audio_thread_priority = { version = "0.38.0", default-features = false }  # drops dbus on Linux; revisit for Linux
libc = "0.2"
# dev / bench
divan = "0.1.21"
hdrhistogram = "7.6.0"
rtsan-standalone = { version = "0.3.0", default-features = false }        # no prebuilt download; see §5.9
no-panic = "0.1.37"
loom = "0.7.2"
```

In dev-dependencies, enable `fearless_simd = { version = "1.1.0", features = ["force_support_fallback"] }` so tests and benches can force `Level::fallback()`.

When x86 ships, add to `.cargo/config.toml` (never `target-cpu=native`):

```toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-cpu=x86-64-v2"]
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=x86-64-v2"]
[target.x86_64-apple-darwin]
rustflags = ["-C", "target-cpu=x86-64-v2"]   # Rosetta implements SSE4.2/POPCNT
```

### 3.2 Where kernels live

All kernels go in `vt_core`, which has no plugin dependencies. Every kernel module exposes the same three layers, so the tests and the A/B benchmarks can reach every variant:

```
crates/vt_core/src/
  simd/mod.rs        Kernels { level: fearless_simd::Level, ... }  (built in activate())
  simd/dot.rs        pub fn reference(a,b) -> f64    // f64 oracle, obviously correct
                     pub fn scalar(a,b) -> f32       // plain Rust, no SIMD types
                     #[simd] pub fn simd<S: Simd>(s, a, b) -> f32
                     #[cfg(target_arch="aarch64")] pub fn neon(..)  // only if it earns its place (> 10%)
  simd/interp.rs     polyphase table + interp_one / interp_block
  simd/window.rs     window multiply, overlap-add
  simd/math.rs       fast_exp2/log2/tanh (+ vector forms)
  fft/mod.rs         trait RealFft; fn fft_len_at_least(min) -> usize
  fft/realfft.rs     portable backend (process_with_scratch only)
  fft/vdsp.rs        #[cfg(target_os = "macos")] backend
  rt/ftz.rs          FtzGuard
  rt/ring.rs         Ring<const N>
  rt/pipeline.rs     AudioEnd / WorkerEnd (§4.5.4)
  rt/pool.rs         process-wide worker pool
  rt/workgroup.rs    #[cfg(target_os = "macos")] work-interval FFI
  rt/prefault.rs     prefault + mlock helpers
```

Hot code calls the dispatched entry point (`dispatch!(self.level, s => dot::simd(s, a, b))`). `Level` comes from `Level::new()`, called once in `activate()`. On aarch64 it returns the static NEON level; on x86 it reads a cached runtime detection. Test and bench builds may override it from `VT_SIMD=fallback` (§5.3).

Dispatch **once per block or per hop**, never per sample. Write the inner loop inside the `#[simd]` function so that every helper inlines into the target-feature context. Mark helpers `#[inline(always)]` when they are not `#[simd]` themselves.

### 3.3 SIMD abstraction and kernel patterns

#### 3.3.1 Dot product and reductions

This pattern is compiled and tested in the scratch crate at every SIMD level, at lengths 0-4096 and operand misalignments 0-3. It runs at 80.9-82.4 ns for 2048 elements on the M4.

```rust
use fearless_simd::{Simd, SimdBase, SimdFloat, f32x16};

/// Two f32x16 accumulators: 8 q-registers on NEON, 2x ymm on AVX2, zmm on AVX-512.
#[fearless_simd_macros::simd]
pub fn dot<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);              // slice once: no bounds checks in the loop
    let z = f32x16::splat(simd, 0.0);
    let (mut acc0, mut acc1) = (z, z);
    let (ca, cb) = (a.chunks_exact(32), b.chunks_exact(32));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        acc0 = f32x16::from_slice(simd, &x[..16]).mul_add(f32x16::from_slice(simd, &y[..16]), acc0);
        acc1 = f32x16::from_slice(simd, &x[16..]).mul_add(f32x16::from_slice(simd, &y[16..]), acc1);
    }
    let mut s = (acc0 + acc1).reduce_sum();
    for (x, y) in ra.iter().zip(rb) { s += x * y; }   // remove by padding buffers (§3.3.3)
    s
}
// activate():  self.level = fearless_simd::Level::new();
// hot path:    fearless_simd::dispatch!(self.level, s => dot(s, a, b))
```

Rules:

- **`.mul_add()` in fearless_simd is fused only where the level has FMA.** On SSE4.2 it is literally `a * b + c`, so it never becomes a libcall. **Do not use `mul_add_precise`** on x86 paths: on levels without FMA it is emulated in software.
- **Size accumulators for the M4, not the M1.**
  - Measured on this M4 P-core: dependent FMLA latency is about 3.1 cycles and FADD about 2.1, with 4 FMA pipes. Published Firestorm (M1) figures say 4 cycles.
  - A load-bound kernel such as dot (two loads per FMA) needs about 8 independent q-register chains (two `f32x16`).
  - A compute-bound kernel (polynomials, mat-vec with reused inputs) needs up to about 12 (three `f32x16`).
- **Plain-Rust warm code** (CMNDF post-processing, energies, GUI decimation) must not use strict-order iterator sums. The plain `fold` compiled to `fmul.4s` followed by a serial scalar `fadd` chain: 909 ns. Use `[[f32; 4]; K]` accumulators or `as_chunks::<16>()` (98-113 ns, full `fmla.4s`), or `algebraic_*` ops (§3.3.4). Avoid `[f32; 16]` accumulators: on this toolchain the SLP vectorizer produced only 2-lane code (117 ns).

#### 3.3.2 FMA policy (cross-platform hazard)

| Where | Use | Never |
|---|---|---|
| fearless_simd kernels | `.mul_add()` (fused if the level has FMA) | `.mul_add_precise()` on non-FMA levels |
| plain Rust on aarch64 | `f32::mul_add` freely (fmadd) | — |
| plain Rust compiled for x86 | `a * b + c`, or `multiversion` with `match_target!{ "x86_64+fma" => mul_add, _ => a*b+c }` | bare `f32::mul_add`: a `fmaf` libcall per element (26-28x slower, measured) |
| pulp (if ever used) | `mul_add_e_f32s` | `mul_add_f32s` (calls `fmaf` on the scalar fallback) |

CI enforces this: the disassembly of every hot x86 function must contain no `fmaf` (§5.13).

#### 3.3.3 Tails, padding and determinism

A kernel whose SIMD body uses FMA while its scalar tail uses mul+add computes a given sample differently depending on whether it lands in the body or the tail. If the body/tail split depends on the host block size, bit-exact block-size invariance (§5.12) breaks. Rules:

- Hot kernels operate on **internal frames whose lengths are multiples of 16** (hop, window, grain buffers padded with zeros). There is no scalar tail at all.
- Where a tail is unavoidable, compute it with the same `S` level using a zero-padded 16-lane temporary, not with scalar code.
- Per-sample paths (grain overlap-add) are processed in fixed internal chunks, independent of host block boundaries.

#### 3.3.4 Algebraic float ops (stable since 1.98)

`x.algebraic_mul(y)` / `algebraic_add` let LLVM reassociate and contract (909 → 118 ns from a one-line fold). Results are allowed to differ between compilations and between inlined copies of the same code. Use them only in warm code that has **tolerance-based tests**, and put each such kernel behind an `#[inline(never)]` function so that every call site runs the same instructions (this keeps the bit-exact invariance test meaningful). Never use them in compensated sums or in denormal-sensitive recursions.

#### 3.3.5 Hand-written NEON (exception path)

Since 1.87, intrinsics without pointer arguments are safe to call inside `#[target_feature]` or NEON-baseline code; loads and stores stay `unsafe`. Add a NEON variant only when **all** of these hold:

1. divan shows more than 10% over the fearless `f32x16` version;
2. the kernel matters to the budget (shows up in the block profile);
3. there is an equivalence test against the f64 oracle;
4. a fearless or scalar version remains for other targets.

Measured today: the dot product is 0-3% faster, interpolation and YIN are moot. So nothing qualifies.

### 3.4 Hot kernels

#### 3.4.1 Difference function and autocorrelation (per hop)

Sizes: choose N once per (mode, voice range, sample rate) in `activate()`:

```rust
/// Smallest N >= min_len with N = f·2^k, f ∈ {1,3,5,15}, k >= 4.
/// Valid for vDSP_DFT_zrop and fast (radix 2/3/5) for realfft.
pub fn fft_len_at_least(min_len: usize) -> usize {
    [1usize, 3, 5, 15]
        .into_iter()
        .map(|f| { let mut n = f << 4; while n < min_len { n <<= 1; } n })
        .min()
        .unwrap()
}
// W=1024, L=512 -> 1536; W=2048, L=1024 -> 3072; W+L=1056 -> 1280
```

Circular cross-correlation needs only N ≥ W+L−1, because j+τ ≤ W+L−2 never wraps.

Per hop, share the forward transform of the frame `a = x[0..W)`:

```text
A = rfft(a ⧺ 0…)                 B = rfft(x[0..W+L) ⧺ 0…)
acf[τ]   = irfft(abs(A)^2)[τ] / N      -> NSDF (MPM) and its m[τ] recurrence
r[τ]     = irfft(conj(A)·B)[τ] / N     -> type-I cross term
d[τ]     = e0 + eτ − 2·r[τ]            e0 = Σ_{j<W} x_j²,  eτ = Σ_{j<W} x_{j+τ}²  (f64)
cmndf[τ] = d[τ]·τ / Σ_{j=1..τ} d[j]
```

Pattern for the realfft backend (compiled and measured in the probe):

```rust
// all buffers come from activate(); nothing here allocates
self.r2c.process_with_scratch(&mut self.a, &mut self.sa, &mut self.scratch)?;
self.r2c.process_with_scratch(&mut self.b, &mut self.sb, &mut self.scratch)?;
for (p, q) in self.sa.iter_mut().zip(&self.sb) { *p = p.conj() * q; }
self.sa[0].im = 0.0;
if n % 2 == 0 { self.sa[n / 2].im = 0.0; }
self.c2r.process_with_scratch(&mut self.sa, &mut self.a, &mut self.scratch)?;
// energies in f64: cancellation near the period dip scales with signal energy
let mut e0 = 0f64; for v in &x[..w] { e0 += (*v as f64) * (*v as f64); }
let mut et = e0;
for t in 0..l {
    if t > 0 { let (o, i) = (x[t - 1] as f64, x[t + w - 1] as f64); et += i * i - o * o; }
    d[t] = (e0 + et) as f32 - 2.0 * self.a[t] / n as f32;
}
```

Notes:

- **`process()` on realfft/rustfft allocates scratch on every call** (realfft `src/lib.rs` calls `make_scratch_vec()`). Only `process_with_scratch` is allowed on the audio thread.
- The error relative to signal energy is 5e-7 to 1.1e-6 for both backends and sizes (measured). The f0 estimate agrees with the direct method within 0.5 cent (test). On very quiet input, the CMNDF threshold must be checked against an energy floor (silence gate first).
- The two f64 energy loops are strictly sequential and are probably a visible share of a 3-4 µs kernel (estimate). If profiling confirms it, compute e0 with a multi-accumulator `f64x4` sum, or maintain prefix sums of x² per input sample with an exact recompute every hop.
- At the verifier rates of research.md §8.2 (16-24 kHz after decimation), the windows are much shorter (for example W+τ_max ≈ 1056 → N = 1280 for Alto/Tenor at 24 kHz). Benchmark each (range, rate) cell in `vt_detect/benches` instead of extrapolating from the 48 kHz figures.

#### 3.4.2 Windowed overlap-add (PSOLA grains)

- Grain windows come from precomputed tables (Hann/raised-cosine halves at the needed lengths) or an incremental recurrence. Never call `cos()` per sample.
- Output accumulation `out[t] += w[t] · g[t]` is a contiguous `f32x16` multiply-add over the grain span, which is trivially vectorized.
- Loop order: **grain-major** (for each active grain, its whole span) measured 44-54 ns per output vs 62-74 ns sample-major in the verifier's reruns. The per-sample summation order across grains is then fixed by the grain order, which keeps results block-size invariant. Re-measure once the real grain read exists.
- Asymmetric windows and the "normalise by the summed window" path (research.md §8.15): keep a running window-sum buffer in the same pass, then apply one vectorized `x * recip(sum)` pass. Use an approximate reciprocal plus one Newton step only if a test shows the error is below −100 dB.

#### 3.4.3 Polyphase windowed-sinc interpolation (fractional reads)

16 taps, 512 phases, Blackman-Harris. The table is 32 KB for c0 plus 32 KB for the deltas, which fits the 128 KB P-core L1D but not twice over in the 64 KB E-core L1D. It is built once per process and shared. **Interpolate between adjacent phases.** Compiled, tested and measured:

```rust
pub struct Polyphase { c0: Box<[[f32; 16]]>, dc: Box<[[f32; 16]]> }  // dc[p] = c0[p+1] - c0[p]

/// Output aligned to x[i + 7 + frac]. One f32x16 FMA for the coefficients, one multiply, one reduction.
#[fearless_simd_macros::simd]
pub fn interp_one<S: Simd>(simd: S, t: &Polyphase, x: &[f32; 16], frac: f64) -> f32 {
    let pf = frac * 512.0;
    let p = (pf as usize).min(511);
    let a = f32x16::splat(simd, (pf - p as f64) as f32);
    let c = f32x16::from_slice(simd, &t.dc[p]).mul_add(a, f32x16::from_slice(simd, &t.c0[p]));
    (f32x16::from_slice(simd, x) * c).reduce_sum()
}
```

- SNR at a semitone ratio, f32 and both SIMD levels: −106.1 dB (500 Hz), −110.8 (1 kHz), −110.5 (4 kHz), −121.3 (8 kHz), −107.6 (12 kHz). Truncating the phase index gives −55 to −83 dB, and rounding it gains only 6 dB.
- Cost per 512 outputs: 304-419 ns truncated vs 310-349 ns interpolated, the same within noise. The kernel is load-bound, so the extra FMA is free.
- Positions are f64; `frac` is computed in f64 and cast once.
- When γ > 1, the kernel must be a low-passed sinc scaled by 1/γ (research.md §8.16). Use a second table family indexed by a quantized γ (for example 16 steps) and interpolate between phases in the same way.
- A fixed-phase FIR-across-outputs variant for γ = 1 was measured at 1.0-1.1x and is rejected: keep one code path.

#### 3.4.4 Dot products and mat-vec for the neural arbiter (worker thread)

- Portable: rows × `dot` (above), blocked as 4 output rows × `f32x16` columns so each input load is reused 4 times (the compute-bound case, so 3 accumulators per row block).
- macOS: `cblas_sgemv` (row-major = 101, no-trans = 111) or BNNSGraph, on a worker only.
  - Warm it up once in `activate()`, because first calls may initialize lazily.
  - Never `setenv("VECLIB_MAXIMUM_THREADS")` from a plugin; it changes the whole DAW process. The measured sgemv was single-threaded anyway.
  - Expect contention on the per-cluster matrix unit from other plugins.
  - Accelerate's internal `malloc` calls are invisible to a Rust counting allocator. Check them with RTSan (§5.9).
- int8 (if the model allows): `vdotq_s32` (stable since 1.98; `dotprod` is in the apple-m1 baseline, so no detection is needed on macOS). On x86, AVX-VNNI/AVX2 via dispatch.
- A hand-written GRU(128) step already measured 10.8 µs median (research.md §8.22). Speed is not the constraint; allocation and tail latency are. The assist runs on a worker and never on the audio thread (§4.5).

#### 3.4.5 Cepstrum / envelope (Studio HQ, Live Max)

Reuse the `RealFft` backend. Log-magnitude via vectorized `fast_log2` (scale by ln 2 / log2 10 as needed; the error budget is 5.8e-7 absolute in log2). Keep the lifter and iteration count fixed per configuration so the cost per grain is bounded.

### 3.5 FFT backend

```rust
/// Real FFT used by the analysis kernels. `new()` allocates and creates setups (off the audio
/// thread); every other method is allocation-free.
pub trait RealFft: Send {
    fn len(&self) -> usize;
    /// Fills acf[..m] = autocorrelation of a and xc[..l] = Σ_j a[j]·b[j+τ], sharing FFT(a).
    fn acf_and_xcorr(&mut self, a: &[f32], b: &[f32], acf: &mut [f32], xc: &mut [f32]);
}
#[cfg(target_os = "macos")] pub type DefaultFft = VdspFft;      // vDSP_DFT_zrop
#[cfg(not(target_os = "macos"))] pub type DefaultFft = RealFftBackend;
```

vDSP specifics (all handled and tested in the prototype):

- Input is converted to split-complex even/odd form with `vDSP_ctoz` (requires even W and L).
- Forward output is scaled by 2 and packs Nyquist in `imag[0]`.
- The inverse setup is created from the forward one: `vDSP_DFT_zrop_CreateSetup(fwd, n, -1)`. For `conj(A)·B` the result is `out/(4N)`.
- Setups allocate, so create them in `activate()` and destroy them in `Drop`.
- Lengths are 2^n, or f·2^n with f ∈ {3,5,15} and n ≥ 4.

Required tests:

- `vdsp_matches_realfft`: spectra within 1e-2 absolute and correlations within 1e-6 of signal energy, at every size in use.
- Difference function vs naive f64 for both backends.

### 3.6 Denormals

Facts:

- The M4 shows no subnormal penalty (normal 418 ns, subnormal 403 ns, subnormal with FZ 378 ns).
- **std::thread::spawn on macOS does not inherit FPCR.FZ** (parent FPCR = 0x1000000, child 0x0; the snippet test confirms it).
- **nice-plug 0.4.2 sets FTZ around process() only when the `unsafe_flush_denormals` feature is enabled.** It is not a default feature, and on x86 it sets FTZ without DAZ (`src/wrapper/util.rs`, `Cargo.toml` `[features]`).
- Host-owned threads (a CLAP thread pool) must not be left modified, so use a scoped guard per task.

Recipe (compiled and tested: flushes, restores, and is not inherited):

```rust
/// FTZ for the current thread while in scope; restores the previous state on drop.
/// aarch64: FPCR.FZ (bit 24). x86_64: MXCSR FTZ (bit 15) | DAZ (bit 6).
pub struct FtzGuard { old: u64, _not_send: core::marker::PhantomData<*const ()> }
impl FtzGuard {
    #[inline]
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        let old = unsafe {
            let v: u64;
            core::arch::asm!("mrs {}, fpcr", out(reg) v);
            if v & (1 << 24) == 0 { core::arch::asm!("msr fpcr, {}", in(reg) v | (1 << 24)); }
            v
        };
        #[cfg(target_arch = "x86_64")]
        let old = unsafe {
            let mut v: u32 = 0;
            core::arch::asm!("stmxcsr [{}]", in(reg) &mut v);
            if v & 0x8040 != 0x8040 { core::arch::asm!("ldmxcsr [{}]", in(reg) &(v | 0x8040)); }
            v as u64
        };
        Self { old, _not_send: core::marker::PhantomData }
    }
}
// Drop restores `old` with msr fpcr / ldmxcsr.
```

Use it:

- at the top of `Plugin::process()`;
- around every worker job;
- in vt_eval renders;
- in every benchmark and test harness (so the offline and plugin numerics match).

Also put explicit flushes (`if s.abs() < 1e-20 { s = 0.0 }`, or a tiny DC offset) in the few recursive paths (DC blocker, one-pole smoothers, envelope followers). Changing the FP environment is technically UB according to the Rust docs (LLVM assumes the default environment), but it is standard audio practice. The explicit flushes make the x86 behaviour deterministic either way.

### 3.7 Fast-math approximations

The code is in `simdbench/src/lib.rs` (`fast_exp2`, `fast_log2`, `fast_tanh`). Each function is branch-light and vectorizes inside `#[simd]` loops; std `exp2f`/`log2f`/`tanhf` are opaque libm calls.

| Function | Method | Domain / clamp | Max error (tested) | In cents | Speed (512 values) |
|---|---|---|---|---|---|
| `fast_exp2` | round-to-nearest split, Cephes degree-6 polynomial, exponent built from bits | clamp to [−126, 126] | rel 9.7e-8 | 0.00017 c | 184 vs 543 ns (2.95x) |
| `fast_log2` | exponent extraction, mantissa folded to [1/√2, √2), atanh series in s = (m−1)/(m+1) | x > 0, normal floats only (guard x ≤ 0 and subnormals) | abs 5.8e-7 | 0.0007 c | 237 vs 734 ns (3.1x) |
| `fast_tanh` | Padé [7/6], clamped | clamp to ±4.97 | abs 9.6e-5 | — | 153 vs 801 ns (5.2x) |
| sigmoid | `0.5 + 0.5·fast_tanh(x/2)` | as tanh | abs 4.8e-5 | — | as tanh |

CI asserts the bounds: exp2 relative < 1e-6, log2 absolute < 1e-5, tanh absolute < 1e-3. Pitch conversions (Hz ↔ cents, semitone ratios, the correction curve) use these. Anything that feeds the GUI tolerates them easily.

### 3.8 Codegen and panic hygiene

- **Slice once** at kernel entry (`let (a, b) = (&a[..n], &b[..n]);`), then iterate with `zip`, `chunks_exact` or `as_chunks`. Check that the disassembly of the hot loop has no `panic_bounds_check`.
- `core::hint::assert_unchecked` / `get_unchecked` are allowed only with a `debug_assert!` twin and a `// SAFETY:` line, and only where the disassembly shows the check was not removed otherwise.
- **No panics in process().** nice-plug has no `catch_unwind`, and a panic escaping `extern "C"` aborts the DAW. In hot kernels:
  - Use `#[no_panic]` (no-panic 0.1.37) under a `cfg(feature = "no-panic-check")` release CI build. It proves at link time that the function cannot panic.
  - Use saturating or wrapping arithmetic on indices and `debug_assert!` for invariants.
- **clippy ban list** for the audio-path crates (`vt_detect`, `vt_control`, `vt_shift`, `vt_engine`, and `vt_core::{simd,rt}`) through `clippy.toml`:
  - `disallowed-methods`: `std::sync::Mutex::lock`, `std::sync::RwLock::{read,write}`, `std::thread::current`, `slice::sort`, `slice::sort_by`, `std::time::SystemTime::now`, `Vec::with_capacity`/`Vec::new`. Construction code opts out with `#[allow]` in clearly named `setup`/`new` functions.
  - `disallowed-macros`: `println`, `eprintln`, `format`, `vec`.
  - Unlike the post-monomorphization const assert, these lints show up in `cargo check` and rust-analyzer.
- Run the `test` profile with debug assertions on (as configured). In the prototype, a debug-mode overflow check caught a real FIFO index-wrap bug.

---

## 4. Real-time architecture rules

### 4.1 Audio-thread contract

| Allowed in `process()` / `reset()` | Forbidden |
|---|---|
| Arithmetic on preallocated, pre-faulted buffers | Any allocation or free (`Vec` growth, `Box::new`, last `Arc` drop, `collect`, `to_vec`, `format!`) |
| `rtrb` push/pop/chunks; `triple_buffer` publish/read; atomics; `basedrop::SharedCell::get` | `std::sync::Mutex`, `RwLock`, parking_lot (including `try_lock`, whose unlock with a waiter signals a condvar), `Condvar` |
| `Thread::unpark()` on a worker (dispatch semaphore, ≤ 6.5 µs max). It is not intercepted by RTSan, so treat it as "measured bounded" rather than proven | crossbeam/std channels (their senders lock a waker mutex when the receiver sleeps); nice-plug `execute_background`/`execute_gui` |
| `Instant::now()` (12.8 ns) or `mach_absolute_time()` (4.5 ns) | `println!`/logging, file or socket I/O, `thread::current()`, lazily initialized thread-locals, `ArcSwap::load` on a fresh thread |
| `FtzGuard` | `sleep`, `yield`, spinning on another thread's progress |
| `ctx.set_latency_samples()` **only** on an explicit mode switch | Calls into tract/rten/ort/candle or other inference runtimes (they allocate or use thread pools) |

Hosts may call `process()` from different engine threads. Anything that is initialized lazily per thread is a hazard, even when it is clean on the first thread you tested.

### 4.2 Allocation-free design with nice-plug 0.4.2

Facts checked in the nice-plug 0.4.2 sources (they differ from upstream nih-plug, which the real-time report studied):

- The lifecycle is `Plugin::activate()` (it replaced `initialize()`), `reset()`, `process()` and `deactivate()`. **`reset()` may be called at any time from the audio thread** (trait docs), so it must be allocation-free and bounded. It clears state; it does not resize.
- Both wrappers hold the plugin in a `try_lock::TryLock` (an atomic swap that never blocks). The CLAP process callback returns `CLAP_PROCESS_ERROR` instead of waiting if the lock is taken. That is not a stall, but it does drop a block.
- `set_state_inner` deserializes parameters (wrapped in `permit_alloc`) and schedules `Task::StateChanged` for the GUI. It does **not** re-run `activate()` while audio runs. The plugin therefore has to react to parameter changes inside `process()` (smoothers, table swaps) rather than relying on re-activation.
  - Persistent non-parameter fields are written on the main thread. Never read them from `process()`; publish their effect through `triple_buffer` or `SharedCell`.
- `assert_process_allocs` (debug only) installs `nice_assert_no_alloc::AllocDisabler` as the crate's `#[global_allocator]`, so any binary that links `vocal_tune` with that feature cannot declare a second one (§5.8).

Rules:

1. **`activate()` allocates, keyed by configuration.**
   - Compute a key from (sample rate, `max_buffer_size`, latency mode, voice range, tier). Reallocate only when it changes; otherwise only call `reset()`.
   - Build large immutable tables (sinc tables, windows, NN weights) once per process in a `OnceLock<Arc<…>>` and share them across instances.
   - Hosts deactivate and reactivate on every latency change, so a fast `activate()` shortens the mode-switch dropout.
2. **Size for the worst case:**
   - `max_buffer_size` from `BufferConfig`;
   - the largest window and hop of the configuration;
   - fixed capacities for candidate lists (`ArrayVec<_, 16>`), the grain pool (≤ 4 active), the MIDI stack (16) and the Viterbi beam.
3. **Pre-fault and lock.** After allocating, write one value per 16 KB page (4 KB on x86) of every buffer larger than 64 KB, and `mlock` NN weights and activations. Never call `mlockall` (it affects the whole DAW).
4. **Swaps.** New scale tables, models or presets are built on the GUI or a background thread and installed through `basedrop::SharedCell::replace`, or sent over rtrb. Old objects go back over a return ring to be freed off the audio thread, or are collected by `Collector::collect()` on a GUI timer.
5. **Std idioms.** Use `sort_unstable`/`select_nth_unstable` for medians and candidate ranking. No `collect`, `to_vec`, `extend` beyond capacity, or `format!`.

### 4.3 Fixed internal hop

- The input goes into a power-of-two ring (`Ring<const N>`, `N ≥ 2^16`, compiled and tested). A hop accumulator triggers analysis at sample-exact positions regardless of host block boundaries, from a 1-sample block to 4096.
- Analysis hops are fixed per configuration (Live: 64 or 128 at 48 kHz; Studio per research.md §7.2). Sample positions are `u64` counts; fractional positions are `f64`.
- **Work spreading.** At a 32-sample buffer one callback in eight carries the per-hop work at hop 256. In the prototype, the 32-frame max (5.6% of 667 µs, 37 µs) equalled the 64-frame max (2.4% of 1333 µs, 32 µs): the hop work landed in a single call.
  - First move heavy, lag-tolerant work to assist jobs (§4.5).
  - Then stagger the remaining independent inline stages across the hop: decimate and FFT at hop phase 0, peak picking and HMM at phase ½.
  - Do not split an FFT across callbacks unless the gate in §5.5 fails.
- The `max_buffer_size` block path and the 1-sample path run the same code. Tests use block sizes {1, 7, 32, 64, 128, 480, 512, 4096, random}.

### 4.4 Lock-free GUI and data channels

| Data | Channel | Sizing / rule |
|---|---|---|
| `PitchFrame` history (research.md §8.23), 375-750 frames/s at hop 128-64 @ 48 kHz | `rtrb::RingBuffer<PitchFrame>` | Capacity 4096 (≥ 5 s if the GUI stalls). Push only while the editor is open (an `AtomicBool`). If full, drop the frame, never block. The GUI drains at 60 fps |
| Latest state (current pitch and target, scale mask, CPU stats, governor state) | `triple_buffer` | Write the **full** snapshot every hop or block; `input_buffer_mut()` does not hold the last published value |
| Meters (in/out peak, correction amount) | `AtomicF32`, Relaxed | Reduce locally, then make one store per block |
| GUI → audio commands | `rtrb` | Pop a bounded number per block (for example ≤ 32) |
| Large objects | `basedrop::SharedCell` or rtrb + return rtrb | §4.2 rule 4 |

- The rtrb `Consumer` lives in the editor, behind a `Mutex` that only the GUI thread takes on open and close. The audio thread never touches that mutex.
- Repaint requests go through an atomic flag.
- Producer/consumer indices, sequence numbers and flags shared across threads sit on separate 128-byte lines (`CachePadded`).
- Orderings: Release/Acquire for publish and consume, Relaxed for meters, SeqCst only for Dekker-style flag pairs. The design below needs none, because park/unpark carries the wake token.

### 4.5 Worker threads: making quality scale with CPU

#### 4.5.1 Division of labour

- **Inline core (authoritative).** Everything that sets the exact period, the pitch marks, the grain timing and the output samples runs inline on the audio thread, within the per-call budgets of §5.5, for every tier. If every worker vanished, the plugin would still be correct at the tier's base quality.
- **Assist jobs (advisory).** Work that only *biases* decisions and tolerates one hop of extra lag runs on workers:
  - harmonic-model re-scoring;
  - the neural octave/voicing arbiter (≤ 0.25 ms per 5 ms hop, research.md §10.7);
  - second-opinion trackers (SWIPE′ in Studio);
  - key detection;
  - Studio envelope estimation and fixed-lag smoothing.
- **Live mode** consumes the result for hop k at hop k+1, which adds 1.33-2.67 ms of *detection* lag to those decisions only. **Studio mode** consumes results anywhere within its lookahead, so its deadline is post time + (D − margin) and its pipeline depth is ⌈D/hop⌉ + 1.
- Reported latency never depends on workers.

#### 4.5.2 The pool

- **One pool per process**, refcounted.
  - The first `activate()` starts it, the last instance's `Drop` stops and joins it.
  - Threads must never outlive the last instance, because the host may unload the dylib.
  - Bitwig's "Together" and "By plug-in" modes share one process; "Individually" gives each instance its own pool.
- **Size.** Use about `clamp(hw.perflevel0.physicalcpu / 4, 1, 3)` workers: 1 on a base M1, 3 on the M4 Max (design choice, validated by the RT-load test T6). The DAW's own engine threads also want the P-cores; the work-interval guidance caps a group at `os_workgroup_max_parallel_threads` (12 here).
- **Assignment.** Each instance gets its own SPSC rings (§4.5.4), attached to one worker, chosen as the least loaded at `activate()`.
  - Registration travels to the worker over a per-worker rtrb control ring. Its producer is guarded by a mutex that only main-thread code takes.
  - On unregistration, the worker returns the ring ends over a reclaim ring so they are freed on the main thread, not on the RT worker.
- **Per-worker setup, before signalling "ready"**, in this order (macOS):

```rust
// 1. Real-time policy: period = constraint = hop duration, computation = half (capped at 50 ms).
let _rt = audio_thread_priority::promote_current_thread_to_real_time(hop_frames, sr as u32).ok();
// 2. Own work interval (stable FFI; AudioToolbox + libSystem). ~0.55 ms once CoreAudio's HAL is up,
//    68-91 ms in a process that never touched CoreAudio (e.g. a Bitwig sandbox): do it at pool start.
let wg = unsafe { AudioWorkIntervalCreate(c"vt-assist".as_ptr(), OS_CLOCK_MACH_ABSOLUTE_TIME /*32*/, null_mut()) };
let mut tok = WgToken { sig: 0, opaque: [0; 36] };          // LP64 os_workgroup_join_token_s
let joined = unsafe { os_workgroup_join(wg, &mut tok) } == 0; // only RT threads may join; join+leave on this thread
ready.store(true, Release);                                   // audio side posts nothing until ready (else 16-19 startup misses)
// 3. Per job (one primary thread per interval calls start/finish):
//    os_workgroup_interval_start(wg, now, job.posted_at + hop_ticks, null_mut()); run(job); os_workgroup_interval_finish(wg, null_mut());
// 4. On exit: os_workgroup_leave(wg, &mut tok); os_release(wg);
```

- The FFI declarations are in the prototype (`rtsys/src/lib.rs`, module `rt`). Read the timebase from `mach_timebase_info`; do not hard-code 125/3 ns.
- **Keep each job well under the "computation" value** (half the period). xnu caps an RT thread's quantum at that value and cancels its RT deadline when the quantum expires. Split big jobs into sub-jobs.
- **Never spin**: the worker `park()`s when its rings are empty. A spinning RT worker lost 53-76% of its wakeups.
- Wrap each job in `FtzGuard` (threads do not inherit FZ) and in `catch_unwind`. A panicking job disables assists for that instance and is reported in the GUI.
- **Windows (later):** `audio_thread_priority` uses the MMCSS "Audio" task. Also opt workers out of power throttling (`SetThreadInformation(ThreadPowerThrottling)`), the Windows counterpart of the work interval on hybrid Intel CPUs.
- **Linux (later):** under rtkit, RLIMIT_RTTIME (often 200 ms) sends SIGXCPU and then SIGKILL to an RT thread that never blocks, which kills the DAW. A worker that parks every hop is safe; a stuck worker is fatal.

#### 4.5.3 The governor (demote-only)

- Each instance tracks, over a sliding 2 s window: assist late ratio, worker compute p99 (from `posted_at` / `done_at` stamps carried in the result) and audio-thread call p99.
- **Demote** one assist level (Max → HQ assists → none) when the late ratio exceeds 1%, or the compute p99 exceeds 50% of the hop, or the audio-thread p99 exceeds its §5.5 budget. Demotion only changes which advisory inputs are consulted, with crossfaded decision weights, so it has no audible discontinuity and no latency change.
- **Promote** only on user action. Proposal: also allow it automatically after 10 s of headroom, and only during unvoiced or silent segments. research.md §1.5 allows only demotion, so this needs sign-off.
- Show the state ("Max (assist on)", "Max (assist shed: CPU)") next to the tier selector, together with the per-machine tier recommendation from the RTF harness (§5.6).

#### 4.5.4 Pipeline protocol

The verifier showed that the obvious protocol (pop one result, accept it if `seq` matches, overwrite `slots[seq % N]`) never resynchronizes after a single late result (90 of 99 hops late in simulation). It also lets the audio thread overwrite a slot the worker is still reading, which is a data race and UB.

The design below makes the **sample ring the slot allocator**. A window that is in the ring belongs to the worker until the worker commits the read, so there is nothing to race on and no `UnsafeCell`. It is compiled and tested in the scratch crate: one late job costs exactly one late hop, and a threaded smoke test with 2000 hops and an RT-style park/unpark loop passes.

```rust
pub struct AudioEnd {
    samples: rtrb::Producer<f32>,      // window samples; capacity = max_window × depth
    jobs: rtrb::Producer<Job>,         // Job { seq, len, posted_at }
    results: rtrb::Consumer<Analysis>, // Analysis { seq, …, done_at }
    worker: Option<std::thread::Thread>,
    seq: u64, pub newest: Analysis, pub late: u64, pub dropped: u64, pub max_age: u64,
}
impl AudioEnd {
    /// Once per hop on the audio thread. Never blocks, never allocates.
    pub fn on_hop(&mut self, window: &[f32]) -> Analysis {
        let mut got = false;
        while let Ok(r) = self.results.pop() { self.newest = r; got = true; }  // drain to newest
        if !got || self.seq.saturating_sub(self.newest.seq) > self.max_age {
            self.late += 1;                                                   // keep the inline/previous estimate
        }
        if self.samples.slots() >= window.len() && self.jobs.slots() >= 1 {
            self.seq += 1;
            self.samples.push_entire_slice(window).expect("checked");          // one memcpy (~0.3-0.5 µs for 8 KB)
            let _ = self.jobs.push(Job { seq: self.seq, len: window.len() as u32, posted_at: now_ticks() });
            if let Some(w) = &self.worker { w.unpark(); }
        } else {
            self.dropped += 1;                                                // worker far behind: shed, don't wait
        }
        self.newest
    }
}
// Worker: while let Ok(job) = jobs.pop() { samples.pop_entire_slice(&mut scratch[..len]); results.push(analyse(…)); }
//         then park(). park/unpark tokens make "push, then unpark" vs "check empty, then park" lossless.
```

Rendering modes:

- **Offline** (nice-plug `ProcessMode::Offline`, CLAP `clap.render`): run the assist **inline on the audio thread, but apply its result with the same one-hop lag** as real time. Real-time output with `late == 0` is then bit-identical to an offline render.
- **Faster-than-real-time export reported as real time** (Ableton's VST3 export reportedly keeps `kRealtime`): detect it by comparing host sample-clock progress against `mach_absolute_time` over 0.5 s windows (ratio > 1.2), then switch to the offline behaviour. Also offer a "Render: wait for analysis" toggle.
- **Hand-rolled atomics** (pool registration, ready and stop flags) go behind a `cfg(loom)` shim and are model-checked with loom or shuttle (§5.10). The rings themselves are rtrb's responsibility.

### 4.6 Latency reporting

- Latency is constant per latency mode (and voice range), and reported from `activate()` with `ctx.set_latency_samples(…)`. Tiers, engines, worker state and governor state never change it (research.md §9.5).
- Mode switch: call `set_latency_samples(n)` once from `process()`. nice-plug swaps an atomic and schedules a GUI task:
  - **CLAP**: a lock-free `ArrayQueue` plus `host->request_callback()`, then `request_restart()`.
  - **VST3**: `restartComponent(kLatencyChanged)` on the GUI thread. When the editor is open, that path takes an `RwLock` read on the plug view, sends an Objective-C `isMainThread` message and goes through the run loop.

  Both paths are acceptable only for this rare, explicit user action. In RTSan harness builds, wrap it in `scoped_disabler!`.
- Telemetry (`PitchFrame.sample_pos`) is stamped in **output time** (input position + D), so the GUI lines up with what is heard. Ableton does not delay-compensate graphics, or monitored tracks when "Reduced Latency When Monitoring" is on.
- Ableton is VST3-only (Live 12 has no CLAP), so every Live behaviour goes through nice-plug's VST3 wrapper. CLAP features (the thread pool, `clap.render`) are Bitwig/REAPER extras.

---

## 5. Performance and RT-safety test plan

### 5.1 Principles and suite layout

- Every optimization arrives with three things: an **equivalence test** (vs the f64 oracle, at every SIMD level), an **error or SNR bound**, and an **A/B benchmark** against its scalar variant. A SIMD variant that cannot show at least 2x over its scalar variant in the A/B bench is deleted, so the codebase carries no dead SIMD.
- Wall-clock gates run on the M4 Max (dev, pre-merge) and the base M1 (release sign-off). Shared CI runners gate on deterministic instruction counts (gungraun) and on correctness only.
- Every perf run records machine state (hw.model, macOS version, toolchain, `uptime` load average, power source). A run with a 1-minute load above 4 is marked "noisy" and is not a valid gate result.

```
crates/vt_core/
  benches/kernels.rs        divan: every kernel × variant (reference, scalar, simd@Level::new, simd@fallback, neon?) × sizes
  benches/fft.rs            divan: realfft vs vDSP × N ∈ {512…8192} ∪ {3·2^k, 5·2^k}
  tests/equivalence.rs      all variants vs f64 oracle; levels {new, fallback}; lengths and offsets
  tests/numerics.rs         fast-math bounds, interpolator SNR, FFT backend agreement, FTZ guard and inheritance
  tests/no_alloc.rs         counting #[global_allocator] (Rust allocations), negative control
  tests/pipeline.rs         late-recovery, ring ownership, threaded smoke; cfg(loom) model of pool registration
crates/vt_detect/benches/detect.rs   divan: per-hop verifier, E/H tracker, HMM per (voice range, rate)
crates/vt_shift/benches/shift.rs     divan: grain read + OLA per block; Classic splice
crates/vt_engine/
  benches/block.rs          criterion: Engine::process per block, mode × tier × {32,64,128,512}; baselines; VT_SIMD A/B
  benches/iai.rs            gungraun (Linux CI): instruction counts for the same cells
  tests/latency.rs          sample-accurate latency matrix (§5.11)
  tests/invariance.rs       block-size invariance and offline/realtime determinism (§5.12)
  tests/no_alloc.rs         20k random blocks + parameter/mode/preset sweeps under the counting allocator
  tests/page_faults.rs      zero page reclaims during steady-state processing
  examples/rt_jitter.rs     paced WCET/jitter harness (hdrhistogram) → JSON, exit code = gate (§5.5)
  examples/rtf.rs           real-time-factor harness (§5.6)
  examples/pipeline_stress.rs  worker pool under the load matrix (§5.7)
crates/vt_eval/src/bin/host_sim.rs   clack-host 0.2.0 harness that loads the built .clap (RTSan, wrapper coverage)
perf/gates.toml             committed pass/fail budgets (the tables below)
perf/baselines/*.json       committed reference results per machine
xtask: `cargo xtask perf [--quick]` runs T5-T7 and compares them with gates.toml and baselines
```

### 5.2 Correctness gates for optimized code (T0)

`cargo test --release` on aarch64-apple-darwin, plus the x86 runs of §5.13. These must pass before any benchmark result counts.

| Test | Pass condition |
|---|---|
| Every kernel variant vs f64 oracle; `Level::new()` **and** `Level::fallback()`; lengths {0,1,3,15,16,17,31,32,33,63,100,1024,2047,2048,4096}; operand offsets 0-3 | dot: abs error ≤ 1e-6·Σ abs(aᵢbᵢ) + 1e-6. Difference function: ≤ 1e-5 × signal energy. Interpolation and OLA: ≤ 1e-6 abs |
| Whole-pipeline f0: every backend and level, after warm-up | within 0.5 cent of each other |
| Fast math | exp2 rel < 1e-6; log2 abs < 1e-5; tanh abs < 1e-3; inputs ±0, subnormal, ±inf, NaN handled as specified |
| Interpolator SNR, semitone ratio, 500 Hz-12 kHz | < −100 dB (today −106 to −121 dB) |
| vDSP vs realfft | spectra within 1e-2 abs; correlations within 1e-6 × energy |
| FTZ | `FtzGuard` flushes a subnormal product and restores on drop; a spawned thread does *not* inherit it |
| Ring | wraparound and copy_latest correctness; `Ring::<4000>` fails `cargo build` (compile-fail test via a trybuild-style doctest) |
| Pipeline | one late job adds exactly 1 to `late`; the window received equals the window posted, for 10⁵ random-length jobs under a threaded run |

### 5.3 Scalar-vs-SIMD A/B micro-benchmarks (T1, divan 0.1.21)

```
cargo bench -p vt_core --bench kernels -- --sample-count 300 --sample-size 200   # --sample-size 20 for >10 µs groups
cargo bench -p vt_core --bench fft     -- --sample-count 300 --sample-size 200
```

Each group lists `reference`, `scalar`, `simd_fallback`, `simd` (and `neon`, `vdsp` where present) side by side. Because the Fallback level may itself be auto-vectorized by LLVM, the explicit `scalar` variant stays as the true scalar baseline.

Pass thresholds on the M4 Max, derived from today's medians plus margin:

| Kernel | Gate |
|---|---|
| dot 2048 | `simd` ≤ 100 ns and ≥ 8x `scalar`-iterator (today 81 vs 909 ns) |
| Difference function W=1024, L=512 (N=1536) | realfft ≤ 6 µs; vDSP ≤ 4.5 µs; ≥ 40x naive (today 4.3 / 3.4 vs 304 µs) |
| Difference function W=2048, L=1024 (N=3072) | realfft ≤ 12 µs; vDSP ≤ 8.5 µs (today 9.1 / 6.2 µs) |
| Exact-N vs pow-2 N | exact ≤ pow-2 for both backends (today −11 to −25%) |
| Polyphase, 512 outputs | ≤ 0.5 µs; phase-interpolated ≤ 1.15x truncated (today 0.31-0.35 µs) |
| exp2 / log2 / tanh | ≥ 2.5x std each |
| FFT backends | each within 15% of the table in §6.3 |
| Any SIMD kernel | ≥ 2x its `scalar` variant, or the SIMD variant is deleted |

### 5.4 Block-level regression and A/B (T2, criterion 0.8.2; gungraun 0.20.0)

```
cargo bench -p vt_engine --bench block -- --save-baseline main               # on main
cargo bench -p vt_engine --bench block -- --baseline main                    # on the branch
VT_SIMD=fallback cargo bench -p vt_engine --bench block -- --baseline main   # proves the SIMD gain at block level
VT_FFT=realfft   cargo bench -p vt_engine --bench block -- --baseline main   # proves the vDSP gain (macOS)
```

- Configuration: `noise_threshold(0.03)`, `significance_level(0.01)`; cells for mode × tier × {32, 64, 128, 512} at 48 kHz, fed with looped vocal recordings.
- **Fail** a PR on a regression over 5% at p < 0.01. A 3.6% drift was observed on untouched code under load, so 3% is too tight on a developer machine.
- An optimization PR must show improvement at p < 0.01. For reference, the vDSP switch measured −26.3% (Live) and −31.3% (Studio).
- **Linux CI**: gungraun instruction counts for the same cells. Fail on more than 2% growth. These counts are deterministic, so they are the gate on shared runners.

### 5.5 Per-process() WCET and jitter histograms (T3)

Harness `vt_engine/examples/rt_jitter.rs`:

- A thread promoted with `audio_thread_priority` (period = block/sr), optionally joined to its own work interval to mimic a host I/O thread.
- **Paced** with `mach_wait_until` on absolute deadlines. A hot loop is optimistic: the same job ran 1.6-2.4x slower at the median when paced.
- Drives the exact `Engine::process` over looped vocals plus edge cases (silence, noise, sweeps, mode-switch-free automation). Every call is wrapped in `FtzGuard`.
- Records per-call wall time into a preallocated `hdrhistogram::Histogram::<u64>::new_with_bounds(1, 100_000_000, 3)`. It also counts overruns (call > period), `late`/`dropped` from the pipeline, and governor demotions.
- Writes JSON (p50/p90/p99/p99.9/p99.99/max) and a log-bucket CSV for plotting, then exits non-zero on any gate failure.
- 10 minutes per cell for sign-off, 60 s for `--quick`.

Budgets at 48 kHz (period: 32 = 667 µs, 64 = 1333 µs, 128 = 2667 µs, 512 = 10667 µs), per instance, as **% of period**. The same percentages apply to every buffer size:

| Configuration | Machine | p50 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| Live-Min / Live / Live+, Eco and HQ | M4 Max | ≤ 2% | ≤ 5% | ≤ 8% | ≤ 15% |
| Live Max, Studio (all tiers) | M4 Max | ≤ 4% | ≤ 10% | ≤ 15% | ≤ 25% |
| Live Eco/HQ | base M1 (sign-off) | ≤ 4% | ≤ 10% | ≤ 15% | ≤ 30% |
| Live Max, Studio | base M1 (sign-off) | ≤ 8% | ≤ 25% | ≤ 30% | ≤ 50% |

The M1 p99 limits are research.md §10.7's budgets. In µs, for Live HQ on the M4: p99 ≤ 33 / 67 / 133 / 533 µs and max ≤ 100 / 200 / 400 / 1600 µs for 32 / 64 / 128 / 512 frames.

Additional gates:

- **Work spreading.** In µs, p99.9(32-frame calls) ≤ 0.75 × p99.9(64-frame calls). If the work were spread per sample the ratio would be about 0.5; the prototype's ratio was about 1.0 (concentrated). Calibrate after the first real implementation.
- **Signal cost.** Audio-thread time spent in `on_hop` posting (memcpy + push + unpark): max ≤ 10 µs.
- **Load matrix.** Each cell runs under:
  - (a) an idle machine;
  - (b) 16 default-QoS spinner threads;
  - (c) 12 RT threads at 70% duty;
  - (d) 16 RT threads at 90% duty (oversubscribed).

  Under (b)-(d): 0 overruns, and p99.9 ≤ 2x the idle limit.
- **E-core stress (informational, plus one gate).** Run under `taskpolicy -b` (background QoS forces E-cores at about 1.4 GHz). Today's prototype went from p99.9 1% to 40% with a 33 ms max there, which is scheduling, not compute. Report wall time, and separately thread CPU time (`clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID)` in a separate run, because it is a syscall). Gate: Eco tier CPU-time p99.9 ≤ 40% of period, so the lowest tier survives if a host ever schedules us on an E-core.
- **Cold cache.** A variant that touches 32 MB between calls, to simulate other plugins evicting the sinc table, FFT twiddles and NN weights. Gate: max ≤ 1.5x the hot max.
- **First block.** The first call after `activate()` + `reset()` is ≤ 1.5x the steady-state p99 (pre-faulting works).

### 5.6 Real-time factor (T4)

`vt_engine/examples/rtf.rs`: an unpaced offline render of the evaluation corpus (≥ 60 s of vocals, 48 kHz, block 128) through `Engine::process`, single thread. RTF = processing time / audio duration.

| Cell | M4 Max gate (initial) | Notes |
|---|---|---|
| Live Eco | ≤ 0.015 | the prototype block is about 0.002-0.0025 |
| Live HQ | ≤ 0.03 | research.md: "a few percent of one core" |
| Live Max (inline part) | ≤ 0.05 | assists excluded |
| Studio HQ / Max | ≤ 0.04 / ≤ 0.06 | |
| Assist CPU, all worker threads (Max tiers) | ≤ 0.15 | thread CPU time |
| 16 instances, interleaved on one thread | ≤ 16 × single × 1.2 | cache-pressure scaling |
| Eco on E-core (thread CPU time under `taskpolicy -b`) | ≤ 0.05 | |

- **Ratchet rule.** Once the real DSP exists, replace each initial gate with the measured value × 1.25 and commit it in `perf/gates.toml`. A gate only moves down, unless a quality change justifies the extra cost.
- The same harness writes the per-machine **tier recommendation** shown in the GUI (research.md §9.6): the highest tier whose RTF × 4 stays under 25%.

### 5.7 Worker pipeline under load (T5)

`examples/pipeline_stress.rs` uses the production assist jobs, runs 10 minutes per cell over the §5.5 load matrix, and tests with and without the work interval. Gates:

- deadline misses (`late`) == 0 in every RT configuration, 1 to 16 instances;
- worker compute p99 ≤ 25% of the hop; post-to-done p99.9 ≤ 50% of the hop;
- idle machine, work interval on: compute p99 ≤ 0.5x the RT-only value (measured 0.43-0.73 vs 2.36-2.45 ms). A regression here means macOS behaviour changed;
- startup: 0 misses (the ready barrier holds);
- 16 instances: the governor never demotes on an idle M4 Max at the Max tier (design target);
- forced-late test (sleep injected in the worker): output stays latency-aligned, `late` counts, and the governor demotes within 2 s;
- multi-instance teardown: the pool joins every thread when the last instance drops (no thread outlives the dylib).

### 5.8 Allocation and page-fault tests (T6)

- **DSP crates** (`vt_core` … `vt_engine`): a test binary with a counting `#[global_allocator]` armed by a thread-local flag around `process()` only. Construction happens outside the armed region. The negative control (one deliberate `vec!` must count exactly 1) runs first.
  - Coverage: 20k blocks of random size 1..=4096, every parameter swept, scale changes, Live/Studio toggles (rebuilding through `activate()` outside the armed region), preset swaps through `SharedCell`, GUI attach and detach.
  - Pass: 0 allocations. The prototype passed, and its negative control proved the counter works.
- **Plugin crate**: debug builds with `--features assert_process_allocs` abort on any allocation inside the wrapped process call, wrapper included. Drive them with `host_sim` (clack-host loads the `.clap`), and use the same builds for manual DAW testing. Do **not** add a second `#[global_allocator]` to a binary that links `vocal_tune` with that feature; it will not link.
- **C allocations** (Accelerate, BNNS, system libraries) are invisible to both of the above, so RTSan (§5.9) is the check for them.
- **Page faults**: around 10 s of steady-state processing after warm-up, read `getrusage(RUSAGE_SELF).ru_minflt` (macOS "page reclaims") in a single-threaded harness. Pass: delta == 0. Negative control: an un-prefaulted 16 MB buffer must move the counter.

### 5.9 RealtimeSanitizer (T7)

- **Stable route**: rtsan-standalone 0.3.0, `default-features = false`.
  - Mark with `#[rtsan_standalone::nonblocking]`: the engine's process entry, the worker job function, and the harness's call into the plugin's CLAP `process`.
  - Workarounds on macOS for 0.3.0, all verified:
    - the build script reads `RTSAN_LIB_PATH` (the docs say `RTSAN_LIBRARY_PATH`);
    - the copied dylib's ad-hoc signature is invalidated (SIGKILL, exit 137);
    - the rpath does not reach the final binary.

```
RTSAN_ENABLE=1 \
RTSAN_LIB_PATH=/opt/homebrew/opt/llvm/lib/clang/20/lib/darwin/libclang_rt.rtsan_osx_dynamic.dylib \
  cargo build -p vt_eval --bin host_sim --features rtsan
RTSAN_OPTIONS=halt_on_error=true \
DYLD_LIBRARY_PATH=/opt/homebrew/opt/llvm/lib/clang/20/lib/darwin \
  ./target/debug/host_sim --scenario full --secs 60
```

- **Scenario** (60 s, scripted): random block sizes; automation every block; a mode switch; a preset load during processing; GUI consumer attach/detach; a forced-late worker; state save/restore during processing; 4 instances sharing the pool.
- **Pass**: 0 reports with `halt_on_error=true`. The only allowed `scoped_disabler!` is around `set_latency_samples` on a mode switch.
- **Negative control**: a debug-only feature that allocates once in `process()` must be reported. This proves that interception reaches code inside the plugin dylib.
- **Nightly route** (optional): `-Zsanitizer=realtime` with `#[sanitize(realtime = "nonblocking")]` on a pinned nightly dated after 2025-11-08. The installed nightly (2025-08-31) predates the feature.
- **Blind spots**: RTSan cannot see mach traps (a `dispatch_semaphore_signal` with a waiter) or page faults; §5.5 and §5.8 cover those. Linux CI runs the same harness with the static runtime. There is no Windows support.

### 5.10 Concurrency verification (T8)

- **loom 0.7.2 or shuttle 0.9.5** for every hand-rolled atomic protocol: pool registration and unregistration, ready and stop flags, governor state. Put std atomics behind a `cfg(loom)` alias module.
- **Miri** (nightly, CI) on `Ring`, the pipeline's non-FFI logic and the table builders. FFI and `asm!` are excluded.
- **ThreadSanitizer** (nightly, `-Zsanitizer=thread -Zbuild-std`) on the threaded pipeline smoke test.

### 5.11 Sample-accurate latency verification (T9)

`vt_engine/tests/latency.rs` covers every latency mode × voice range × sample rate {44.1, 48, 88.2, 96 kHz} × 20 seeds × random host block sizes 1..=max:

1. **Unvoiced / bypass path.** A white-noise burst at a random offset. The output must equal the input delayed by exactly D samples (bit-exact, residual ≤ −120 dBFS), and the cross-correlation argmax must equal D.
2. **Voiced, correction off (β = 1).** A synthetic LF-pulse vowel with known glottal-closure instants: output pulse positions = input GCI + D within ±0.5 sample (fractional marks).
3. **Voiced with correction.** The content-delay distribution stays within [D − T, D + T] (research.md: wander ≤ T), and voiced/unvoiced transitions never shift alignment.
4. **Classic in Live/Studio** is padded to exactly D.
5. **Reporting.** `Engine::latency_samples()` == the value passed to `set_latency_samples` == the value the host sees. Check it with `host_sim`, which queries `clap_plugin_latency.get()` after `activate()`, and again after a mode switch and restart.
6. **Workers.** With assists disabled, forced late or on time, alignment is identical (workers never move audio).

Pass: 0-sample tolerance on items 1, 4 and 5. In the DAW, run the research.md §10.6 loopback test and a null test (dry vs plugin in bypass-identity, PDC on). Pass: residual ≤ −120 dBFS, and the latency readouts in Ableton and Bitwig equal D.

### 5.12 Block-size invariance and determinism (T10)

- An offline render with fixed 64-sample blocks vs blocks {1, 7, 32, 480, 512, 4096, random 1..4096} must be **bit-identical**. This relies on §3.3.3: no body/tail FMA splits, and no `algebraic_*` code inlined differently at different call sites.
- The offline mode (inline assist with one-hop lag) must be bit-identical to the real-time pipeline whenever `late == 0`.
- The same render on the `Level::fallback()` path must match within the §5.2 tolerances (not bit-exact, since FMA is unfused there).

### 5.13 Cross-architecture (T11)

- **Rosetta, SSE4.2** (no AVX): `cargo test --release --target x86_64-apple-darwin`. This exercises the SSE2/SSE4.2 dispatch levels, the multiversion fallbacks and the non-FMA paths.
- **Rosetta with `ROSETTA_ADVERTISE_AVX=1`** (undocumented; verified on macOS 27.2): fearless_simd selects AVX2 and the AVX2/FMA paths run. Correctness only: translated AVX2 is *slower* than SSE4.2 under Rosetta.
- **Relative-performance gate under Rosetta**: no hot kernel more than 3x slower than its mul+add autovec reference. This catches the `fmaf` trap (28x) and pulp-style scalar fallbacks (16x).
- **Disassembly checks** (cargo-show-asm 0.2.63, on `#[inline(never)]` concrete entry points of each hot kernel):
  - x86_64: no `fmaf` call and no `panic_bounds_check` in the loop; `vfmadd…ps ymm` present in the AVX2 clone.
  - aarch64: `fmla.4s` / `fmla v…4s` present, no `.2s` arithmetic in the hot loop, no `panic_bounds_check`.
- **Native x86 CI** (GitHub windows-latest, ubuntu-latest): tests with the default and the x86-64-v3 builds, plus Intel SDE for AVX-512. Apple says general Rosetta support ends after macOS 27, so native x86 CI is the long-term plan.

### 5.14 In-DAW soak and validators (T12)

- Ableton Live 12 (VST3) and Bitwig (CLAP and VST3; "Within Bitwig" plus one sandboxed mode), at a 64-sample buffer @ 48 kHz. 8 instances on vocal tracks, 30 minutes of transport loops with automation and one preset load.
- The plugin's debug panel shows `late`, `dropped`, max/p99 `process()` ns, overruns and governor state.
- Pass:
  - `late == 0` at HQ;
  - no audible or recorded dropouts;
  - stable Live CPU meter and Bitwig DSP meter;
  - an Instruments System Trace shows workers on P-cores, inside their work interval, and never fail-safe demoted.
- os_signpost intervals around `process()` and around each worker job make this visible on the Instruments timeline.
- Validators: clap-validator 0.4.1, the Steinberg VST3 validator, and pluginval at strictness ≥ 10 (with `--rtcheck` on a develop build). All must be green.

### 5.15 Measurement hygiene

- Record the load average before every run (it ranged from 5 to 62 during the research). Compare medians within a single run, and interleave A/B runs at least twice.
- Pass `--sample-size` to divan. Use the paced harness, not hot loops, for tail numbers. Run on mains power, with Low Power Mode off.
- Use `debug = "line-tables-only"` builds with samply or the Instruments Time Profiler on a headless `vt_eval` render to find glue code once the kernels are fast. Example: in the prototype, the vDSP even/odd split loop dominated until it was replaced by `vDSP_ctoz`.

### 5.16 What runs where

| Gate | Per PR (macOS arm64 CI) | Pre-merge on the M4 (`cargo xtask perf --quick`) | Nightly | Release sign-off |
|---|---|---|---|---|
| T0 correctness, T6 alloc + page faults, T9 latency, T10 invariance | yes | yes | yes | yes |
| clippy ban list, no-panic build, disassembly checks | yes | — | yes | yes |
| T1 divan A/B, T2 criterion baselines | — | yes | — | yes |
| T2 gungraun instruction counts | Linux CI | — | — | — |
| T3 WCET matrix, T4 RTF, T5 pipeline | — | quick cells | full matrix on the M4 | full matrix on the M4 **and a base M1** |
| T7 RTSan (`host_sim`), T8 loom/Miri/TSan | — | — | yes | yes |
| T11 Rosetta (SSE4.2 and AVX), native x86 | — | — | yes | yes |
| T12 DAW soak and validators | — | — | — | yes |

---

## 6. Local measurements (M4 Max)

Unless noted, everything here is single-threaded, f32, built with fat LTO and codegen-units = 1. The reports used divan medians; the new probes use best-of-40 batch means. Load averages were 5-62 during the research and 8.9-13.4 during the new probes. The raw outputs came from scratch benchmark crates kept outside the repository (a SIMD kernel bench, a real-time threading probe, verifier probes and the documentation probes); the kernel benchmarks that matter were later rebuilt inside the repository (`crates/vt_core/benches/kernels.rs`, see `docs/PERFORMANCE.md`).

### 6.1 Dot product, 2048 elements (256 elements in parentheses)

| Variant | Time | Codegen |
|---|---|---|
| naive iterator sum | 909 ns (70 ns) | `fmul.4s`, then a serial scalar `fadd` chain |
| indexed loop | 1195 ns | `bl panic_bounds_check` in the loop |
| `[f32;16]` accumulators | 117 ns; 107 ns with `mul_add` | only 2-lane `.2s` (SLP) |
| `[[f32;4];4]` accumulators / `as_chunks` | 98 / 113 ns | `fmla.4s` |
| multiversion plain / with `match_target` FMA | 123 / 96-107 ns | |
| wide f32x4 | 90.8 ns | |
| pulp (level detected per call / cached) | 91.6 / 91.4 ns | |
| fearless_simd f32x4 × 4 accumulators | 90.6 ns | |
| **fearless_simd f32x16 × 2 accumulators** | **80.2-82.4 ns** (verifier); **80.9-81.5 ns** (new probe) | 8 q-registers |
| hand NEON 4 / 8 accumulators | 95.6 / 83.6-83.7 ns (12.2 ns at 256) | |
| nightly std::simd f32x16 / algebraic fold | 95.4 / 118 ns | |

### 6.2 Difference function (type-I, W / L)

| Variant | 1024 / 512 | 2048 / 1024 |
|---|---|---|
| naive | 304 µs | 1.241 ms |
| auto-vectorized direct | 32.5 µs | 129 µs |
| NEON direct, 4 lags blocked | 16.1 µs | 64.8 µs |
| realfft, N = next pow-2 (2048 / 4096) | 5.57-5.97 µs | 11.8-12.5 µs |
| **realfft, N = W+L (1536 / 3072)** | **4.33-4.47 µs** | **9.13 µs** |
| vDSP DFT, N = pow-2 | 3.86-3.87 µs | 7.66-7.73 µs |
| **vDSP DFT, N = W+L** | **3.39-3.44 µs** | **6.16-6.20 µs** |
| error / signal energy (all FFT variants) | 4.8e-7 to 1.1e-6 | 5.0e-7 to 1.1e-6 |

The exact-N and vDSP rows are new probes, run in two interleaved rounds. vDSP's advantage at YIN level (1.3-1.5x) is smaller than its raw FFT advantage (2-2.9x), because the energy loops and the spectrum multiply are shared costs.

### 6.3 FFT backends (r2c forward), N = 512 / 1024 / 2048 / 4096 / 8192

| Backend | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| realfft 3.5.0 | 373 ns | 702 ns | 1.73 µs | 3.58 µs | 8.2 µs |
| phastft 0.4.1 | 246 ns | 577 ns | 1.22 µs | 2.59 µs | 5.66 µs |
| vDSP ctoz + DFT_zrop | 171 ns | 357 ns | 944 ns | 1.59 µs | 2.91 µs |
| vDSP ctoz + fft_zrip (legacy) | 162 ns | 681 ns | 977 ns | 2.13 µs | 4.18 µs (one run 2.53) |
| rustfft complex, same N | 562 ns | 1.29 µs | 2.48 µs | 6.2 µs | 14.1 µs |

The verifier's rerun gave vDSP DFT 1.97-2.86x and phastft 1.36-1.51x faster than realfft.

### 6.4 Interpolation and grain read

| Measurement | Result |
|---|---|
| 16-tap polyphase, 512 outputs (report) | scalar 1.46 µs; autovec 0.73 µs; NEON 0.61 µs |
| fearless f32x16, 512 outputs (new probe) | truncated phase 304-419 ns; **phase-interpolated 310-349 ns** |
| SNR re signal, semitone ratio, 500 / 1k / 4k / 8k / 12k Hz | truncated: −82.6 dB at 500 Hz, −64.6 at 4 kHz, −55.0 at 12 kHz. Rounded: +6 dB. **Phase-interpolated (f32, NEON and fallback): −106.1 / −110.8 / −110.5 / −121.3 / −107.6 dB** |
| 512-sample grain, sinc read + window + OLA (new probe) | per-output polyphase dot 412-460 ns vs fixed-phase FIR across outputs 404-412 ns (1.0-1.1x; rejected) |

### 6.5 Fast math, denormals, mat-vec

- Per 512 values:
  - exp2: std 543 ns, fast 184 ns (max rel error 9.7e-8);
  - log2: std 734 ns, fast 237 ns (max abs error 5.8e-7);
  - tanh: std 801 ns, fast 153 ns (max abs error 9.6e-5).
- Denormals, 4-stage feedback filter, 256 steps: normal 332 ns, subnormal 332 ns, subnormal + FZ 317 ns. Reruns: 418/375/327 and 418/403/378 ns. In the RT report: an IIR bank at 0.52 ns per filter-sample in all cases, and a NEON loop at 291 ns per 4096 elements either way. **No penalty on M4.**
- FPCR inheritance: parent 0x1000000, child thread 0x0, for both `thread::spawn` and `Builder`.
- GRU-sized mat-vec, 768×256: naive 54.8 µs; NEON 8 accumulators 9.2-9.5 µs (about 85 GB/s); Accelerate sgemv 1.60-1.70 µs (about 460 GB/s effective, single-threaded). With background QoS (E-cores) sgemv rose to a 16.9 µs median.
- M4 microarchitecture (verifier, calibrated against a 1-cycle integer add chain at about 4.3-4.4 GHz): FMLA through the accumulator about 3.1 cycles, scalar FADD about 2.1 cycles.

### 6.6 Prototype pipeline and tail latency

- Full 128-sample block (YIN + CMNDF + decision + interpolation; budget 2667 µs):
  - **Live (1024/512):** scalar 304.8 µs; autovec 32.8 µs; NEON 16.8 µs; realfft 6.33 µs (reverified 6.41 µs); vDSP 4.86 µs (−26.3%, p < 0.01).
  - **Studio (2048/1024):** scalar 1.243 ms; NEON 65.5 µs; realfft 13.4 µs; vDSP 9.48 µs (−31.3%).
- Tail, Studio configuration, 43,500 blocks, unpaced:

| Variant | p50 | p99.9 | p99.9 / budget | max |
|---|---|---|---|---|
| naive | 1255 µs | 1685 µs | 63% | — |
| autovec | 131 µs | 167 µs | 6.3% | — |
| NEON | 66 µs | 96.5 µs | 3.6% | — |
| FFT (P-core) | 14.1 µs | 29.8 µs | 1.1% (rerun 0.99-1.05%) | 69.5 µs |
| FFT under `taskpolicy -b` (E-cores) | 43.5 µs | 1076 µs | 40% | 33.1 ms |

- Paced jitter, inline YIN-2048 per 256-sample hop (RT report), as % of period:

| Block | p99 | max |
|---|---|---|
| 32 | 1.89% | 5.61% |
| 64 | 0.96% | 2.43% |
| 128 | 0.51% | 1.23% |
| random 1..512 | 0.79% | 3.07% |

- Grain overlap-add, 16 grains (verifier reruns; the original AoS 44.1 / SoA 107.0 / grain-major 60.2 ns per sample did not reproduce): AoS 62-74, SoA 63-68, grain-major 44-54 ns per output sample.

### 6.7 Threads, wake-ups, work intervals

- Wake latency (simulated 64-frame host RT thread signals a worker; 3000 iterations; p50 / p99.9 / max in µs):
  - idle machine: park + normal worker 3.5 / 17.3 / 19.6; park + RT 2.2 / 7.4 / 7.5; crossbeam channel + RT 4.2 / 9.1 / 9.6; condvar + RT 5.8 / 10.2 / 10.8;
  - 16 busy threads: park + normal 2.2 / 1301 / 1329; QoS user-interactive 8.5 / 1325 / 1328; **park + RT 1.4 / 4.5 / 5.5**; spin-then-park 1.5 / 4.8 / 5.5;
  - signal cost: unpark p50 0.3-1.0 µs, max ≤ 6.46 µs;
  - a spinning RT worker saw only 1373/2950 (idle) and 703/2950 (loaded) wakeups.
- Pipeline: one 2048-sample job per 256 hop, due 5.33 ms later, 1500 jobs:
  - 16 busy default-QoS threads: normal priority 48 misses (max 8.0 ms late); QoS user-interactive 31 misses; RT 0 misses (p99 712 µs, max 726 µs); RT + work interval 0 misses (p99 400 µs, max 442 µs).
  - Idle, RT only vs RT + work interval: p50 929-943 vs 388-390 µs and p99 2444-2452 vs 428-433 µs (report). The verifier's rerun at load average about 5: p50 1029 vs 645-656 µs and p99 2360 vs 727 µs.
  - 12 RT load threads at 70% duty: p99 434 vs 416 µs, 0 misses either way (work interval irrelevant).
  - 16 RT load threads at 90% duty: p99 1491 vs 565 µs, max 2469 vs 1504 µs, 0 misses either way.
  - Small job, idle: normal p99 418 µs vs RT + work interval 54.5 µs.
  - Audio-thread cost per 64-frame block: offloaded max 24 µs vs inline max 425 µs.
- Work-interval costs:
  - `AudioWorkIntervalCreate`: 68-91 ms first call in a cold process (it is CoreAudio initialization; later calls 4-8 µs); 0.55 ms first call once the HAL is up;
  - `os_workgroup_join` 1.5-2.1 µs; interval start + finish 1.44-1.68 µs hot;
  - `os_workgroup_max_parallel_threads` = 12;
  - without a ready barrier, 16-19 startup misses.

### 6.8 Plumbing micro-benchmarks and RTSan matrix

- ns per op:

| Operation | ns |
|---|---|
| rtrb push+pop | 2.65 |
| rtrb 64-f32 chunk write+read | 10.89 |
| crossbeam ArrayQueue | 3.57 |
| triple_buffer one-field publish / 2 KB publish / read | 0.94 / 43.46 / 0.22 |
| AtomicF32 store / AtomicU32 fetch_max | 0.22 / 4.27 |
| arc_swap warm load | 1.90 |
| mach_absolute_time / Instant::now | 4.48 / 12.78 |
| unpark with no waiter | 0.56 |
| ring indexing, 4 taps: mask / branch-wrap / modulo | 3.3-4.5 / 5.6-9.8 / 20.5-24.3 |
| false sharing, 2 × 50M increments: adjacent vs CachePadded | 368-566 vs 158-345 ms |

- **RTSan clean:** Vec push within capacity; rtrb push; `Thread::unpark` to a parked thread; `Instant::now`; `mach_absolute_time`; basedrop Shared drop; `SharedCell::get` (also on a new thread); drop of a replaced basedrop value; triple_buffer publish; `ArcSwap::load` on an already-warmed thread; parking_lot uncontended `try_lock`.
- **RTSan flagged:**
  - Vec growth (realloc); Box/Arc last drop (free);
  - std `Mutex` first lock (malloc + pthread_mutex);
  - crossbeam and std mpsc `try_send` to a blocked receiver;
  - `ArcSwap::load` on a new thread (posix_memalign); `ArcSwap::store`;
  - parking_lot contended lock; parking_lot unlock with a parked waiter (pthread_cond_signal);
  - `println!`.
- First-touch page faults: +230-250 µs at 4 MB, +800 µs at 16 MB on the first pass.

### 6.9 Build settings and x86 under Rosetta

- Fat LTO + cgu=1 vs the default release: phastft 2048 1.387 → 1.228 µs; realfft 2048 1.911 → 1.754 µs; YIN-FFT 6.83 → 6.31 µs; interpolation 674 → 646 ns.
- `target-cpu=native`: 0% (6.203 vs 6.201 µs).
- PGO: 0% (5.59 vs 5.54/5.66 µs).
- `panic=abort`: 974,136 → 917,176 bytes, identical p50 of 14.17 µs.
- Rosetta, SSE4.2 only, dot 2048:
  - autovec with mul+add 213-224 ns; **with `mul_add` 5.81-5.96 µs (fmaf libcalls)**;
  - pulp `mul_add_f32s` 5.84-5.9 µs; pulp `mul_add_e` 324-360 ns (scalar);
  - fearless 99 ns; wide 98 ns; multiversion `match_target` 98-99 ns;
  - interpolation scalar 2.7 µs vs 26.4 µs with `mul_add`.
- `ROSETTA_ADVERTISE_AVX=1` exposes avx, avx2, fma, f16c and bmi2 (not avx512f), and all tests pass on the AVX2 levels. Translated AVX2 is slower: fearless 181 vs 99 ns, multiversion FMA clone 366 vs 98 ns.

---

## 7. Risks and open questions

| Risk / question | Mitigation |
|---|---|
| fearless_simd 1.0 is 2-3 weeks old; minor-version churn | Pin `=1.1.0` in the lockfile and wrap it in `vt_core::simd` so an API change touches one module. The equivalence tests catch semantic drift. |
| Work-interval benefit is workload-dependent and undocumented in plugin contexts (Apple documents it for AU/app threads) | T5 runs with and without it under the RT-load matrix. It is a measured win at low load and under oversubscription and neutral otherwise, never a loss. Revisit if a CLAP workgroup extension appears. |
| Process-wide RT threads in a DAW with many plugins | Pool size of 1-3, park every hop, governor sheds work, refcounted shutdown. The 16-instance soak in T12. |
| Ableton VST3 exports may report real time | Faster-than-real-time detection plus the "wait for analysis" toggle (§4.5.4). Verify in Live 12 early. |
| Accelerate (vDSP, BLAS, BNNS) internals are not documented as real-time safe | vDSP only on setups made in `activate()`. BLAS/BNNS on workers only. RTSan on all of them in `host_sim`. |
| RTSan dylib interposition may not reach code inside a host-loaded plugin | The negative control in §5.9 proves coverage or exposes the gap. If there is a gap, fall back to RTSan on `vt_engine` directly plus assert_process_allocs in the bundle. |
| Base M1 numbers are unknown | Every gate has an M1 column; buy or borrow a base M1 for sign-off (research.md already treats it as the budget machine). |
| Rosetta's general support ends after macOS 27 | Native x86 CI (Windows and Linux) plus Intel SDE are the permanent path. |
| `algebraic_*` non-determinism vs bit-exact tests | Restricted to `#[inline(never)]` warm kernels with tolerance tests. |
| Governor auto-promotion conflicts with research.md §1.5 (demote only) | Ship demote-only. Auto-promotion is a proposal pending sign-off. |

---

## 8. Corrections applied from verification

From the adversarial verifications:

1. Current stable is **1.99.0** (2026-10-01), not "1.98 or later". 1.98 also stabilized the NEON dot-product intrinsics. Cite `array_windows` from the official 1.94 post.
2. fearless_simd `f32x4` patterns compile to xmm-only FMAs on AVX2. Use `f32x16`/`S::f32s`. With two `f32x16` accumulators, fearless matches hand NEON (80-82 vs 83.6 ns), so **there is no current case for hand NEON**.
3. `mul_add_precise` is emulated in software on non-FMA levels; "always fused" describes its semantics, not the hardware.
4. `Level::new()` costs nothing on aarch64 (a static); on x86 it reads cached detection (cost not measured natively). Detect once anyway.
5. phastft depends on fearless_simd **0.5.0**: two SIMD stacks.
6. The vDSP zrop length rule is 2^n or f·2^n with f ∈ {3,5,15}, n ≥ 4. Legacy `fft_zrip` was not uniformly worse.
7. **Exact-size FFT** (N = W+L) is about 25% faster with realfft, at the same accuracy.
8. **Polyphase "512 phases ≈ −100 dB" was wrong** for truncated phase indices (−55 to −83 dB). Interpolating between phases gives −106 to −121 dB.
9. The tail gate measured only P-cores at boost clock. Under background QoS the p99.9 rose to 40% of budget with a 33 ms max. Worker scheduling (RT policy plus work interval) and E-core/load test axes are required.
10. Firestorm (M1) latency figures do not apply to the M4. Measured: FMLA about 3.1 cycles, FADD about 2.1, which gives about 12 accumulators for compute-bound kernels.
11. Rosetta **can** expose AVX2/FMA (`ROSETTA_ADVERTISE_AVX=1`), so AVX2 paths can be correctness-tested locally.
12. x86_64-apple-darwin is Tier 2 with host tools from 1.90: `rustup target add` works on stable, and `-Zbuild-std` is unnecessary.
13. `AudioWorkIntervalCreate`'s 94-146 ms was one-time CoreAudio initialization. Once the HAL is up the first call takes 0.55 ms and later calls 3-9 µs.
14. The work-interval gain reproduced as 1.6x/3.2x (p50/p99), not 2.4x/5.7x. It disappeared under moderate RT load and returned (2.6x p99) under RT oversubscription.
15. The spin-miss mechanism is better explained by xnu's per-quantum RT deadline cancellation than by the 1 s fail-safe. The advice (never spin) stands.
16. The original pipeline code pattern had two bugs (no resynchronization; slot overwrite race). It is replaced by the ring-owned protocol (§4.5.4).
17. Offline determinism cannot rely on hosts reporting offline mode (Ableton VST3).
18. Pre-touching is not free above about 1 MB (page faults are invisible to assert_no_alloc and RTSan).
19. The AoS/SoA result did not reproduce. Grain-major order was fastest in reruns.
20. Masking vs branch-wrap is about 1.7x, not 5.3x (5.3x was against modulo).
21. The power-of-two const assert is post-monomorphization: `cargo check` does not catch it.
22. A clean RTSan result for `unpark` means "not intercepted" (mach trap), not "proven bounded".
23. The prototype's latency and bit-identity tests used a delay line, so they validated the harness, not the DSP. The real tests (§5.11, §5.12) must run on the real engine.

From spot checks made while writing this document (nice-plug 0.4.2 sources in `~/.cargo/registry`, plus new probes):

24. **nice-plug 0.4.2 enables FTZ only with the non-default `unsafe_flush_denormals` feature**, and on x86 it sets FTZ without DAZ. The reports' "the framework sets FTZ around process()" does not hold for our framework by default.
25. **nice-plug 0.4.2 uses `activate()`** (not `initialize()`). It holds the plugin in a non-blocking `try_lock::TryLock`; CLAP `process` returns `CLAP_PROCESS_ERROR` if the lock is taken. `set_state_inner` does **not** re-run `activate()`. The real-time verifier's "per-block parking_lot Mutex + set_state → initialize under the lock" hazard is an upstream nih-plug behaviour and does not apply in that form. The underlying rule (keep `activate()` cheap, build heavy objects once) still holds, for mode-switch restarts.
26. nice-plug's `reset()` may be called from the audio thread at any time (trait docs).
27. vDSP's gain from exact-size N is smaller than realfft's: −11% (N=1536) and −20% (N=3072), against −22% and −25%. At YIN level vDSP is 1.3-1.5x faster than realfft, not 2-2.8x.
28. A fixed-phase FIR grain read gives no speedup over the per-output polyphase dot (1.0-1.1x), so keep one code path.

---

## Appendix A: Sources

**Rust and toolchain**

- Rust 1.98.0 (algebraic float ops, NEON dot-product intrinsics): https://blog.rust-lang.org/2026/08/20/Rust-1.98.0/
- Rust 1.94.0 (`array_windows`, NEON fp16 intrinsics): https://blog.rust-lang.org/2026/03/05/Rust-1.94.0/
- Rust 1.89.0 (last Tier-1 release of x86_64-apple-darwin): https://blog.rust-lang.org/2025/08/07/Rust-1.89.0/
- Rust 1.87.0 (safe intrinsic calls): https://blog.rust-lang.org/2025/05/15/Rust-1.87.0/
- Rust 1.86.0 (`#[target_feature]` on safe fns): https://blog.rust-lang.org/2025/04/03/Rust-1.86.0/
- Rust 1.81.0 (panics in `extern "C"` abort): https://blog.rust-lang.org/2024/09/05/Rust-1.81.0/
- portable_simd tracking issue: https://github.com/rust-lang/rust/issues/86656
- float_algebraic tracking issue: https://github.com/rust-lang/rust/issues/136469
- RealtimeSanitizer in rustc (PR #147935): https://github.com/rust-lang/rust/pull/147935; unstable book: https://doc.rust-lang.org/nightly/unstable-book/compiler-flags/sanitizer.html
- aarch64 prefetch intrinsic tracking issue: https://github.com/rust-lang/rust/issues/117217
- `f32::mul_add`: https://doc.rust-lang.org/std/primitive.f32.html#method.mul_add
- `slice::as_chunks`: https://doc.rust-lang.org/std/primitive.slice.html#method.as_chunks; `sort_unstable`: https://doc.rust-lang.org/std/primitive.slice.html#method.sort_unstable
- `assert_unchecked`: https://doc.rust-lang.org/core/hint/fn.assert_unchecked.html
- `vdotq_s32`: https://doc.rust-lang.org/stable/core/arch/aarch64/fn.vdotq_s32.html
- target-cpu: https://doc.rust-lang.org/rustc/codegen-options/index.html#target-cpu; Cargo profiles: https://doc.rust-lang.org/cargo/reference/profiles.html
- std thread parking on Darwin: https://github.com/rust-lang/rust/blob/master/library/std/src/sys/sync/thread_parking/darwin.rs

**Crates**

- fearless_simd: https://docs.rs/fearless_simd/1.1.0/fearless_simd/, https://github.com/linebender/fearless_simd
- realfft: https://docs.rs/realfft/3.5.0/realfft/; RustFFT changelog: https://github.com/ejmahler/RustFFT/blob/master/CHANGELOG.md
- phastft: https://crates.io/crates/phastft, https://github.com/smu160/PhastFT
- pulp: https://docs.rs/pulp/0.22.3/pulp/
- wide: https://docs.rs/wide/1.7.1/wide/
- multiversion: https://docs.rs/multiversion/0.9.0/multiversion/
- macerator: https://crates.io/crates/macerator; simdeez: https://crates.io/crates/simdeez
- no_denormals: https://docs.rs/no_denormals/0.3.0/no_denormals/
- rtrb: https://docs.rs/rtrb/0.4.0/rtrb/, https://github.com/mgeier/rtrb/releases
- triple_buffer: https://docs.rs/triple_buffer/9.0.0/triple_buffer/struct.Input.html
- basedrop: https://docs.rs/basedrop/0.1.3/basedrop/
- crossbeam-utils CachePadded: https://github.com/crossbeam-rs/crossbeam/blob/master/crossbeam-utils/src/cache_padded.rs
- audio_thread_priority: https://docs.rs/audio_thread_priority/0.38.0/audio_thread_priority/, https://github.com/mozilla/audio_thread_priority (Linux rtkit: `src/rt_linux.rs`)
- hdrhistogram: https://crates.io/crates/hdrhistogram
- criterion: https://crates.io/crates/criterion; divan: https://github.com/nvzqz/divan; gungraun: https://github.com/gungraun/gungraun
- rtsan-standalone: https://docs.rs/rtsan-standalone/0.3.0/rtsan_standalone/, https://github.com/realtime-sanitizer/rtsan-standalone-rs; Clang RTSan: https://clang.llvm.org/docs/RealtimeSanitizer.html
- assert_no_alloc: https://crates.io/crates/assert_no_alloc
- loom: https://crates.io/crates/loom; shuttle: https://crates.io/crates/shuttle; no-panic: https://crates.io/crates/no-panic
- arrayvec: https://crates.io/crates/arrayvec; heapless: https://crates.io/crates/heapless; atomic_float: https://crates.io/crates/atomic_float
- cargo-pgo: https://github.com/Kobzol/cargo-pgo; samply: https://crates.io/crates/samply
- clack (CLAP host and thread pool): https://github.com/prokopyl/clack
- nice-plug: https://codeberg.org/RustAudio/nice-plug (checked locally: `nice-plug-0.4.2/src/wrapper/{util.rs, clap/wrapper.rs, vst3/inner.rs, vst3/wrapper.rs}`, `nice-plug-core-0.4.2/src/plugin.rs`, `nice-plug-0.4.2/Cargo.toml`)

**Apple**

- vDSP DFT setup: https://developer.apple.com/documentation/accelerate/vdsp_dft_zrop_createsetup; forum (DFT vs FFT): https://developer.apple.com/forums/thread/23321
- BLAS: https://developer.apple.com/documentation/accelerate/blas; BNNSGraph (WWDC24 10211): https://developer.apple.com/videos/play/wwdc2024/10211/
- Asynchronous real-time threads and workgroups: https://developer.apple.com/documentation/audiotoolbox/adding-asynchronous-real-time-threads-to-audio-workgroups
- Understanding audio workgroups: https://developer.apple.com/documentation/audiotoolbox/understanding-audio-workgroups
- `os_workgroup_max_parallel_threads`: https://developer.apple.com/documentation/os/os_workgroup_max_parallel_threads
- AU auxiliary threads (render context observer): https://developer.apple.com/documentation/audiotoolbox/adding-audio-unit-auxiliary-real-time-threads-to-audio-workgroups
- WWDC20 "Meet Audio Workgroups": https://developer.apple.com/videos/play/wwdc2020/10224/
- xnu scheduler: https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/sched_prim.c, https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/priority.c
- Firestorm SIMD tables (M1 reference only): https://dougallj.github.io/applecpu/firestorm-simd.html
- Rosetta AVX behaviour (Go runtime notes): https://go.googlesource.com/go/+/refs/heads/master/src/internal/cpu/cpu_x86_darwin.go

**Plugin formats and hosts**

- CLAP latency: https://github.com/free-audio/clap/blob/main/include/clap/ext/latency.h; CLAP thread pool: https://github.com/free-audio/clap/blob/main/include/clap/ext/thread-pool.h
- Blue Cat Audio on Apple Silicon multicore dropouts and workgroups: https://www.bluecataudio.com/Blog/preview/realtime-audio-multicore-issues-for-apple-silicon-end-of-the-story/
- JUCE forum, workgroups from non-AU plugins: https://forum.juce.com/t/joining-apple-audio-workgroup-from-non-juce-thread-spawned-inside-of-juce-plugin/58041/2
- JUCE forum, Ableton VST3 `isNonRealtime` not set: https://forum.juce.com/t/ableton-vst3-isnonrealtime-not-set/46912
- Ableton, P-cores only since 11.3: https://help.ableton.com/hc/en-us/articles/5266527910812; delay compensation exceptions: https://help.ableton.com/hc/en-us/articles/209072409
- Bitwig plug-in hosting modes: https://www.bitwig.com/userguide/latest/vst_plug-in_handling_and_options/
- Polyphase resampling (J.O. Smith): https://ccrma.stanford.edu/~jos/resample/; Cephes: https://www.netlib.org/cephes/
