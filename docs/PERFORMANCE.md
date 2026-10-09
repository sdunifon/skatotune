# Vocal Tune: performance evidence

This file records the evidence behind DESIGN §4's two performance gates:

- **CPU** (M4 Max, 48 kHz): Live HQ p99.9 block time < 5 % of the buffer period at 128 and < 10 % at 32; Max tiers < 25 %.
- **Perf proofs**: every SIMD, FFT and fast-math optimisation shows a statistically significant speedup in criterion against its scalar or std baseline.

It also gives a quick view of plugin latency (§1), because "latency" can mean two different things: the delay the host compensates, and the CPU time per block.

All numbers come from fresh runs on 2026-10-08 (21:42-22:56 EDT). The machine was shared with other agents, so §2 explains how load was handled. P§ refers to `docs/rust-performance.md`, R§ to `docs/research.md`.

**Final merged tree (2026-10-09, 00:05-00:36 EDT).** §10 re-measures the engine, the tail latency and the vt_detect exact-size adoption on the tree as shipped, with no other agent running. Where §10 and the sections below differ, §10 is the current state; §3-§7 remain the kernel-level evidence.

**Hardening pass (2026-10-09, 05:47-06:03 EDT).** §11 re-measures the engine and the tail latency after the hardening pass (vt_shift PERF-5/6, vt_detect PERF-1-4, the Studio split analysis in vt_engine), and A/Bs the split analysis against the one-chunk Studio analysis on the same tree. **§11 is the current state**; the §0 summary below is updated to it.

## 0. Summary

| Gate / question | Result |
|---|---|
| Plugin latency D (what the host compensates) | Live, Alto/Tenor, 48 kHz: **4.44 ms**. Average over the vocadito corpus: Live 3.96 ms, Live-Min 1.02 ms, Live+ 10 ms, Studio 40 ms. Measured equals reported in all 48 mode × range × rate rows (§1) |
| CPU: Live HQ p99.9 @128 (< 5 %) | **2.38 %** (median of 3 rt-bench runs, §11.2; 2.98 % in §5). PASS |
| CPU: Live HQ p99.9 @32 (< 10 %) | **5.16 %** (5.99 % in §5). PASS |
| CPU: Max tiers p99.9 (< 25 %) | worst **7.33 %** (Studio Max @32; 10.79 % in §5, 8.84 % in §10). PASS, and every configuration is inside the rt-bench exit budget of 10 % |
| Mean cost, Live HQ | 9.2 µs per 32-sample block = **1.4 % of real time**. Studio Max: 2.8 % (§11.1) |
| Studio work spreading (new) | The Studio analysis runs in stages over the next hop (§11.3): at 192 kHz/32 the Studio Max p99.9 block fell from 30.1 % to 22.0 % of the deadline, at 96 kHz/32 from 13.7 % to 9.9 % |
| Perf proofs | SIMD `dot`, `sum_sq`, `diff_sq`, the FFT difference function, vDSP and fast math are all significantly faster than their baselines (in every round, with disjoint 95 % CIs). NEON `axpy`/`mul_acc` gain only 0-8 % over the auto-vectorised scalar loop and nothing at 4096 (memory-bound; documented exception, §3.1) |
| New: exact FFT sizes (`FftSizing::Fast`) | realfft is **25-27 % faster** at 768/1536/3072 than at the next power of two. vDSP gains −5 to −36 %, but is *slower* at 384 than at 512 (§3.4) |
| New: vDSP backend fixes | Spectral product vectorised: vDSP difference function **−40 %** at n = 2048/4096. Split buffers 128-byte aligned: vDSP results are now **bit-identical across instances**, which they were not before (§3.5) |
| Rosetta (x86_64) | vt_core 66 + 2 tests pass under Rosetta 2. With `ROSETTA_ADVERTISE_AVX=1` the dispatcher selects AVX2 and the same tests pass. No native-x86 run (§6) |

## 1. Latency at a glance

**Plugin latency D** is what the host compensates, and what a performer monitoring through the plugin hears. It is fixed per mode and voice range and never depends on CPU load or buffer size. `vt_eval latency` checks that the measured value equals the reported one for every row (all 48 rows pass, ±0 samples):

| Mode | Low Male / Wide | Alto/Tenor | Soprano |
|---|---|---|---|
| Live-Min | 48 samples = **1.00 ms** | 1.00 ms | 1.00 ms |
| Live (default) | 281 = **5.85 ms** | 213 = **4.44 ms** | 149 = **3.10 ms** |
| Live+ (default 10 ms) | 480 = 10.0 ms | 10.0 ms | 10.0 ms |
| Studio (default 40 ms) | 1920 = 40.0 ms | 40.0 ms | 40.0 ms |

Values are at 48 kHz. At 44.1 kHz Live is 259 / 197 / 138 samples (5.87 / 4.47 / 3.13 ms); at 96 kHz the sample counts double and the times stay the same. `vt_shift/src/latency.rs` documents 4.6 ms and 3.2 ms for Live, but the formula gives 4.44 ms and 3.10 ms. That file belongs to the vt_shift owner.

**On real singing** (vocadito, 40 tracks, voice range chosen per file; `out/vocadito/report.md`):

- The corpus averages 3.96 ms in Live, 1.02 ms in Live-Min, 10 ms in Live+ and 40 ms in Studio.
- On top of D, the correction needs time to act. Pitch is detected a median 12.9 ms after the annotated note start in Live HQ (9.3 ms in Studio HQ). A hard-tuned note is settled a median 27.1 ms after the start (Live HQ; 17.1 ms in Studio HQ). Before the hardening pass these were 11.2 / 7.9 ms and 39.1 / 35.1 ms.

**CPU time is a separate quantity.** It decides whether a block finishes before its deadline, not how late the audio is. Live HQ uses 9.2 µs of a 667 µs 32-sample period (§11).

## 2. Machine, build and method

- **Machine:** Apple M4 Max with 16 cores (12 P + 4 E), 128 GB, 128-byte cache lines, macOS 27.2 (26B5091g).
- **Toolchain:** Rust 1.99.0 (LLVM 23.1.1), `aarch64-apple-darwin`.
- **Build profile:** `bench` = `release` (opt-level 3, `lto = "fat"`, `codegen-units = 1`, `debug = "line-tables-only"`).
- **Harnesses:**
  - criterion 0.8.2. Kernels use `--warm-up-time 1 --measurement-time 2`. The engine bench uses its own config: 30 samples, 0.5 s warm-up, 2 s measurement.
  - `vt_eval rt-bench`: an unpaced loop over 10 s of sung material (`melody_male`, Alto/Tenor, stereo, retune 10 ms) that records the wall time of every `Engine::process` call.
- **Load:** other agents were building and running a 16-thread evaluation. The 1-minute load average ranged from 6.7 to 62 during the runs, against 16 cores. Mitigations:
  - **Repeated rounds, best median kept.** The kernel suites ran 3-5 times (`r1`-`r5`) and the engine suite twice (`e1`, `e2`). Each table shows the **lowest criterion median across rounds**, since load only adds time, together with the round count.
  - **Significance.** "k/n" in the tables is the number of rounds, out of n, in which the optimised variant B had the lower median *and* the two 95 % bootstrap CIs of the medians were disjoint. A result is called significant when B was faster in at least n − 1 rounds and k is a majority. This allows one round lost to a load spike; the losing rounds are the ones whose CIs were wider than about ±10 %.
  - **Tail runs gated on low load.** rt-bench was rerun 5 times, each run starting only when the load average was below 8. §5 reports the per-configuration **median across the 5 runs**. A single run's max is dominated by OS preemption on this shared machine (§5).
  - **Fixed snapshot.** Other owners were editing vt_detect, vt_control, vt_shift, vt_engine and vt_eval during the session. The engine and rt-bench numbers come from a single build of the tree at **22:08**. vt_core matches the final state described here.

## 3. Kernel evidence (`crates/vt_core/benches/kernels.rs`)

Times are criterion medians: best across rounds r1, r2 and r4 for the SIMD, fast-math, interpolation and decimator groups, which did not change during the session. The FFT groups use r1-r5 for realfft and r3a-r5 for vDSP, because vDSP changed in this pass (§3.5).

### 3.1 SIMD kernels: scalar vs NEON (vs naive)

`naive` is a plain iterator fold, which leaves a serial `fadd` chain. `scalar` is the portable backend: chunked multi-accumulators that LLVM auto-vectorises. `neon` is hand-written `core::arch` code with FMA. `algebraic` is the rejected alternative, a fold with `f32::algebraic_add`/`algebraic_mul` (§7).

| Kernel | n | naive | algebraic | scalar | NEON | NEON vs scalar | NEON vs naive |
|---|---:|---:|---:|---:|---:|---|---:|
| dot | 256 | 77.7 ns | 12.0 ns | 12.2 ns | **11.3 ns** | 1.08× (3/3) | 6.9× |
| dot | 1024 | 417 ns | 44.9 ns | 44.7 ns | **42.6 ns** | 1.05× (3/3) | 9.8× |
| dot | 4096 | 1916 ns | 205 ns | 177 ns | **168 ns** | 1.05× (3/3) | 11.4× |
| sum_sq | 256 | 62.7 ns | 7.0 ns | 9.3 ns | **7.9 ns** | 1.18× (3/3) | 7.9× |
| sum_sq | 1024 | 417 ns | 36.1 ns | 37.0 ns | **24.1 ns** | 1.53× (3/3) | 17.3× |
| sum_sq | 4096 | 1911 ns | 185 ns | 168 ns | **93.2 ns** | 1.80× (3/3) | 20.5× |
| axpy | 256 / 1024 / 4096 | – | – | 15.0 / 50.6 / 183 ns | 14.7 / 47.0 / 183 ns | 1.02× (1/3) / 1.08× (3/3) / 1.00× (0/3) | – |
| mul_acc | 256 / 1024 / 4096 | – | – | 19.1 / 69.2 / 262 ns | 18.8 / 68.5 / 265 ns | 1.02× (2/3) / 1.01× (2/3) / 0.99× (1/3) | – |
| diff_sq (W = n, n/2 lags) | 256 | – | – | 1.87 µs | **1.07 µs** | 1.74× (3/3) | – |
| diff_sq | 1024 | – | – | 26.8 µs | **17.1 µs** | 1.57× (3/3) | – |
| diff_sq | 4096 | – | – | 395 µs | **264 µs** | 1.50× (3/3) | – |

- **The reductions are the real gain.** `dot` and `sum_sq` are 7-20× faster than the naive fold, from the accumulator layout. NEON's FMA adds 5-8 % on `dot` (load-bound) and 18-80 % on `sum_sq`.
- **NEON `axpy` and `mul_acc` show no meaningful speedup.** They gain 1-8 % at 256-1024 (`axpy` 1024 and `mul_acc` 256/1024 meet the significance rule above), and nothing at 4096 (0/3 and 1/3). They are memory/store-bound, and LLVM already vectorises the scalar loop. They are kept for FMA's single rounding, not for speed. P§1.1's rule (hand NEON only for more than 10 % on a budget-relevant kernel) would not admit them. This is the one exception to the perf-proof gate.

### 3.2 Type-I difference function: direct vs FFT

`d(τ) = Σ_{j<W} (x_j − x_{j+τ})²` for the two P§3.4.1 configurations. The FFT rows use `Correlator::diff_sq`: one cross-correlation plus f64 sliding energies.

| Variant | W = 1024, τ = 512 | W = 2048, τ = 1024 |
|---|---:|---:|
| naive (P§6.2, earlier probe) | 304 µs | 1241 µs |
| direct, scalar | 25.8 µs | 100.9 µs |
| direct, NEON (`simd::diff_sq`) | 16.4 µs | 65.6 µs |
| FFT realfft, n = 2048 / 4096 (power of two) | 5.85 µs | 12.60 µs |
| **FFT realfft, n = 1536 / 3072 (`FftSizing::Fast`)** | **4.47 µs** | **9.67 µs** |
| FFT vDSP, n = 2048 / 4096 | 3.61 µs | 7.18 µs |
| **FFT vDSP, n = 1536 / 3072** | **3.32 µs** | **5.92 µs** |

- The FFT route is 3.7× (realfft) to 4.9× (vDSP) faster than direct NEON at W = 1024. At W = 2048 it is 6.8-11.1× faster, and about 90× / 210× faster than naive. NEON direct is 1.5-1.6× faster than scalar direct. Every FFT-vs-direct and NEON-vs-scalar pair is 6/6 (realfft rounds) or 4/4 (vDSP rounds). Fast size vs power of two is 5/6 and 6/6 for realfft, and 3/4 and 4/4 for vDSP.
- The error relative to signal energy is about 1e-6 (unit tests `diff_sq_via_correlator_matches_direct` and `fast_size_diff_sq_matches_direct_and_pow2`, tolerance 1e-5). That is why `simd::diff_sq` stays the exact path for small windows.

### 3.3 FFT backends: realfft vs vDSP

`autocorrelate` is one forward and one inverse transform plus the spectrum product; `power_spectrum` is one forward transform plus |X|². Sizes 768, 1536, 3072 and 6144 are `3·2^k`, the sizes `FftSizing::Fast` produces.

| n | realfft acf | vDSP acf | vDSP speedup | realfft power spectrum | vDSP power spectrum | vDSP speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 784 ns | 380 ns | 2.06× | 405 ns | 193 ns | 2.10× |
| 768 | 1215 ns | 751 ns | 1.62× | 624 ns | 432 ns | 1.45× |
| 1024 | 1645 ns | 834 ns | 1.97× | 846 ns | 401-559 ns ¹ | 1.5-2.1× |
| 1536 | 2621 ns | 1842 ns | 1.42× | 1331 ns | 931 ns | 1.43× |
| 2048 | 3561 ns | 2015 ns | 1.77× | 1851 ns | 1073 ns | 1.72× |
| 3072 | 5620 ns | 2925 ns | 1.92× | 2876 ns | 1513 ns | 1.90× |
| 4096 | 7616 ns | 3827 ns | 1.99× | 3929 ns | 2060 ns | 1.91× |
| 6144 | 12.08 µs | 5.17 µs | 2.34× | 6.31 µs | 2.83 µs | 2.23× |
| 8192 | 16.67 µs | 7.08 µs | 2.36× | 8.90 µs | 3.88 µs | 2.29× |

¹ vDSP's 1024-point power spectrum measured 401 ns in r1, with the old unaligned buffers, and 559-1049 ns in the four later rounds, with wide CIs. An isolated A/B showed the same forward-transform cost at 1024 for aligned and unaligned buffers (333 vs 327 ns, best of 200), and the new |X|² loop is 4 % faster than the old one. The spread is therefore attributed to load, but it is the least certain entry in this file.

vDSP was faster than realfft at every size, in every round after the fix (3/3).

### 3.4 Exact FFT sizes vs the next power of two (new: `FftSizing::Fast`)

The cross-correlation the detector's verifier runs every hop, `Correlator::cross_correlate` of the W-sample head against the W + τ_max + 1 buffer, was timed at the 48 kHz detector cells. Geometry follows `vt_detect`: W = 2.2/f_min at 24 kHz (HQ) or 16 kHz (Eco), τ_max = ⌈1.06·rate/f_min⌉.

| Cell | Buffer | n: pow2 → fast | realfft pow2 | realfft fast | Δ | vDSP pow2 | vDSP fast | Δ |
|---|---:|---|---:|---:|---|---:|---:|---|
| HQ Low Male (W 960, τ 463) | 1424 | 2048 → 1536 | 5.45 µs | **4.02 µs** | **−26 %** (5/6) | 2.80 µs | **2.47 µs** | −12 % (4/4) |
| HQ Soprano (377, 182) | 560 | 1024 → 640 | 2.49 µs | 2.43 µs | −2 % (n.s., 4/6) | 1.60 µs | **1.03 µs** | −36 % (3/4) |
| Eco Alto/Tenor (440, 212) | 653 | 1024 → 768 | 2.47 µs | **1.82 µs** | **−27 %** (6/6) | 1.62 µs | 1.39 µs | −14 % (3/4) |
| Eco Soprano (251, 122) | 374 | 512 → 384 | 1.18 µs | **0.86 µs** | **−27 %** (4/6) | **0.55 µs** | 0.70 µs | **+28 % (slower, 0/4)** |
| Short Low Male (480, 287) | 768 | 1024 → 768 | 2.47 µs | **1.81 µs** | **−27 %** (5/6) | 1.57 µs | 1.48 µs | −5 % (4/4) |
| W 1024, τ 511 | 1536 | 2048 → 1536 | 5.46 µs | **3.97 µs** | **−27 %** (4/6) | 2.83 µs | 2.51 µs | −11 % (3/4) |
| W 2048, τ 1023 | 3072 | 4096 → 3072 | 10.98 µs | **8.21 µs** | **−25 %** (5/6) | 5.76 µs | **4.30 µs** | −25 % (4/4) |
| HQ Alto/Tenor (660, 318) | 979 | 1024 either way | – | – | 0 | – | – | 0 |

The raw transforms (§3.3) show the same pattern:

- **realfft:** every 3·2^k size is 26-29 % cheaper than the power of two above it. 5·2^7 = 640 gains almost nothing over 1024.
- **vDSP:** −9 to −27 % from 768 up; the 768-point power spectrum is compared against the uncertain 1024 entry (¹). vDSP is slower at 384 (the cross-correlation table above).

**API, for the vt_detect owner** (additive; `Correlator::new` keeps rounding up to a power of two):

```rust
pub enum FftSizing { PowerOfTwo /* default, = Correlator::new */, Fast }
impl FftSizing { pub fn len_for(self, min_len: usize) -> usize; }
pub fn fast_len(min_len: usize) -> usize;               // smallest f·2^k ≥ min_len, f ∈ {1,3,5,15}, k ≥ 4
impl Correlator {
    pub fn with_sizing(min_len: usize, backend: FftBackend, sizing: FftSizing) -> Self;
}
```

**Adopting it in vt_detect** is a one-line change per frame: `Correlator::with_sizing(len, FFT_BACKEND, FftSizing::Fast)` in `LongFrame::new` and `ShortFrame::new`.

- The frames already size their spectra from `corr.len()`.
- The short-frame trim in `Detector::new` uses `next_power_of_two`. Using `fast_len` there instead, and capping the Low Male `ts_max` at 287 rather than 288, keeps that buffer at 768 instead of 960.
- **Estimated gain** on the current realfft backend, from the table: about 3 µs per hop for HQ Low Male (long frame plus its power spectrum plus two short-frame correlations), about 15 % of that cell's 22 µs, and about 0.7 µs for Eco Alto/Tenor. HQ Alto/Tenor is unchanged.
- This is an estimate: it was not measured inside the detector. If vDSP is adopted, keep the power of two for buffers ≤ 512.

**Adopted (final pass, §10.3).** vt_detect uses 3·2^k only (`verifier::correlation_len`): the 5·2^k and 15·2^k sizes gained nothing or lost inside the detector (Eco Low Male at 960 instead of 1024: +6 % per hop). The harmonic-score spectrum keeps its power-of-two plan, so its bin interpolation is unchanged.

### 3.5 vDSP backend fixes made in this pass

Two defects were found while measuring §3.3-3.4. Both are fixed in `crates/vt_core/src/fft/vdsp.rs`.

1. **The spectral product did not vectorise.** `conj(A)·B` and |A|² iterated `.skip(1)` over a nested `zip` of four slices, which LLVM left as a slow scalar loop. Run in isolation it took 1.7-2.3 µs at n = 1536-2048, against 0.13-0.14 µs over flat sub-slices: about a third of every vDSP correlation. Before the fix, vDSP was *slower* than realfft for `diff_sq` at n = 2048 (6.04 vs 5.85 µs).

   | Benchmark (vDSP) | before (r1) | after (best of r3a-r5) | change |
   |---|---:|---:|---:|
   | diff_sq W1024, n = 2048 | 6.04 µs | 3.61 µs | −40 % |
   | diff_sq W1024, n = 1536 | 4.82 µs | 3.32 µs | −31 % |
   | diff_sq W2048, n = 4096 | 12.66 µs | 7.18 µs | −43 % |
   | diff_sq W2048, n = 3072 | 10.54 µs | 5.92 µs | −44 % |
   | cross_correlate HQ Low Male, n = 2048 | 5.20 µs | 2.80 µs | −46 % |
   | autocorrelate n = 2048 / 4096 | 2.37 / 4.75 µs | 2.01 / 3.83 µs | −15 % / −19 % |

2. **The result depended on buffer alignment, which broke determinism.**
   - **What failed:** Accelerate picks differently rounded code paths depending on the alignment of the split-complex buffers. Over 600 plans fed the same input, there were exactly two bit patterns, one per alignment class of the imaginary buffer (0 vs 32 mod 64). The `Box<[f32]>` buffers landed wherever the allocator put them, so two engine instances, or two sessions, could disagree in the last bits. That breaks DESIGN §2.1 rule 6. It is also why `vt_detect` pinned `FftBackend::RealFft` (see the `FFT_BACKEND` comment in `crates/vt_detect/src/verifier.rs`).
   - **Fix:** the split buffers are now `Aligned` blocks on 128-byte boundaries.
   - **Tests:** the new `split_buffers_are_line_aligned` failed 3/3 runs without the fix. `results_do_not_depend_on_the_instance` (200 instances per size, both backends) is a property test: in its final form it failed in 1 of 4 runs without the fix, because it depends on where the allocator places buffers. Both pass with the fix.
   - **Side effect:** aligned buffers are also faster at n ≥ 2048. Forward transform, best of 200: 508 vs 702-723 ns at 2048 and 1104 vs 1350-1440 ns at 4096. Below that they are equal.
   - **What this means for vt_detect:** its stated reason for avoiding vDSP no longer holds. Its cross-correlation would run 1.95× faster on vDSP (HQ Low Male: 5.45 → 2.80 µs at 2048, 2.47 µs at 1536). Repeated calls on one instance were already bit-identical: 0 mismatches in 1.2 M calls across 12 threads at load 38. realfft gave identical bits across 2000 instances.

### 3.6 Fast math vs std (1024 values per iteration)

| Function | std | fast | Speedup | Max error (unit tests) |
|---|---:|---:|---:|---|
| exp2 | 1127 ns | 341 ns | **3.30×** (3/3) | relative 9.7e-8 |
| log2 | 1371 ns | 319 ns | **4.30×** (3/3) | absolute 3.9e-6 (full range), 3.6e-7 on [2^-8, 2^8] |
| tanh | 1502 ns | 442 ns | **3.40×** (3/3) | absolute 3.9e-7 |

### 3.7 Interpolation qualities (512 fractional reads, step 1.0371)

| Quality | Time | Per read | Accuracy (band-limited sines, unit tests) |
|---|---:|---:|---|
| Cubic Hermite | 853 ns | 1.7 ns | no anti-aliasing |
| Sinc8 (256 phases) | 1738 ns | 3.4 ns | – |
| Sinc16 (512 phases) | 2670 ns | 5.2 ns | −90 dB up to 0.1·fs, −83 dB up to 0.3·fs (−44 dB at 0.35·fs: the 16-tap limit) |
| Sinc16 at rate 1.5 (anti-alias table) | 2624 ns | 5.1 ns | cutoff table chosen by threshold compares (was 4.14 µs with a `log2` per read, vt_core report) |

### 3.8 Decimator: minimum-phase IIR vs linear-phase FIR

The detector's streams were timed per 1024 input samples. The baseline is a Kaiser-windowed FIR with the same specification (65 dB from fs_out/2, transition from 0.4·fs_out; tap count from Kaiser's formula), evaluated only at the kept outputs (polyphase cost).

| Stream | IIR (Chebyshev II) | FIR | IIR group delay | FIR delay |
|---|---:|---:|---:|---:|
| 48 kHz ÷ 2 (HQ, 24 kHz) | order 10: 5.21 µs | 81 taps: 1.95 µs | 2.9 samples = 0.06 ms | 40 samples = 0.83 ms |
| 48 kHz ÷ 3 (Eco, 16 kHz) | order 12: 5.58 µs | 121 taps: 1.90 µs | 4.9 = 0.10 ms | 60 = 1.25 ms |
| 96 kHz ÷ 4 | order 12: 5.87 µs | 161 taps: 1.96 µs | 6.8 = 0.07 ms | 80 = 0.83 ms |
| 192 kHz ÷ 8 | order 12: 5.97 µs | 319 taps: 2.00 µs | 14.2 = 0.07 ms | 159 = 0.83 ms |

- **The IIR is not a speed optimisation.** It is 2.7-3.0× more expensive than the polyphase FIR, because its f64 biquad chain runs every input sample.
- **It is a latency choice.** It saves 0.8-1.2 ms of detection lag, which adds directly to how late the correction acts (§1).
- **Its cost is negligible:** about 0.67 µs per 64-sample hop for the ÷2 and ÷3 streams together, 0.05 % of the hop.

## 4. Engine-level cost (`cargo bench -p vt_eval --bench engine`)

`Engine::process` at 48 kHz, stereo, Alto/Tenor, `melody_male`, retune 10 ms. Each cell is the best of runs e1 and e2: the median time per call, with the share of the buffer period in parentheses (32 = 667 µs, 64 = 1333 µs, 128 = 2667 µs, 512 = 10667 µs). Runs e1 and e2 agreed within about 3 %, apart from a few loaded cells in e1.

| Mode | Tier | 32 | 64 | 128 | 512 |
|---|---|---:|---:|---:|---:|
| Live-Min | Eco | 6.3 µs (0.95 %) | 13.0 µs (0.97 %) | 26.3 µs (0.99 %) | 106 µs (0.99 %) |
| Live-Min | HQ | 10.3 µs (1.54 %) | 20.3 µs (1.52 %) | 40.6 µs (1.52 %) | 163 µs (1.52 %) |
| Live-Min | Max | 11.4 µs (1.71 %) | 22.2 µs (1.67 %) | 44.4 µs (1.66 %) | 177 µs (1.66 %) |
| Live | Eco | 6.6 µs (0.99 %) | 13.2 µs (0.99 %) | 26.3 µs (0.99 %) | 105 µs (0.98 %) |
| Live | **HQ** | **10.0 µs (1.51 %)** | 20.6 µs (1.54 %) | **41.4 µs (1.55 %)** | 163 µs (1.53 %) |
| Live | Max | 11.3 µs (1.70 %) | 22.5 µs (1.69 %) | 45.5 µs (1.71 %) | 186 µs (1.75 %) |
| Live+ | Eco | 6.9 µs (1.03 %) | 13.6 µs (1.02 %) | 27.1 µs (1.02 %) | 108 µs (1.01 %) |
| Live+ | HQ | 10.3 µs (1.54 %) | 20.5 µs (1.53 %) | 42.0 µs (1.58 %) | 167 µs (1.56 %) |
| Live+ | Max | 11.8 µs (1.78 %) | 24.0 µs (1.80 %) | 48.0 µs (1.80 %) | 191 µs (1.79 %) |
| Studio | Eco | 9.1 µs (1.36 %) | 18.1 µs (1.35 %) | 36.1 µs (1.35 %) | 144 µs (1.35 %) |
| Studio | HQ | 15.5 µs (2.32 %) | 30.5 µs (2.29 %) | 60.9 µs (2.28 %) | 246 µs (2.30 %) |
| Studio | Max | 20.8 µs (3.12 %) | 39.4 µs (2.96 %) | 81.7 µs (3.06 %) | 328 µs (3.07 %) |

- Cost scales linearly with the buffer, so the share of real time is constant per configuration. The engine has no per-call overhead worth measuring.
- **Live modes cost about 20-25 % more than in the v1 integration report.** Live HQ was 8.26 µs at 32, 33.0 µs at 128 and 131 µs at 512; it is now 10.0 / 41.4 / 163 µs. Studio is unchanged or cheaper (Studio HQ: 15.36 vs 15.5 µs; Studio Max: 22.4 vs 20.8 µs).
  - The 22:08 snapshot contains other owners' in-flight changes to vt_detect, vt_control and vt_shift, made between 21:59 and 22:08.
  - Two runs agree with each other. A quieter run of a 22:50 snapshot (§7.1) gave 9.66 µs at 32 and 38.5 µs at 128, so load explains about 4-7 % of the increase, not all of it.
  - **The regression agent should re-run `engine/live/*` after the merge** to see whether the increase is real.

## 5. Tail latency (`vt_eval rt-bench`) against the DESIGN §4 CPU gate

Wall time of every `process()` call over 10 s of audio, as a percentage of the buffer period. Each cell is the **median across 5 runs**, each started at a 1-minute load average below 8 (22:40-22:43). The max column shows the median and the worst of the 5 runs.

| Mode | Tier | Buffer | p50 | p99 | p99.9 | max: median / worst | DESIGN §4 gate |
|---|---|---:|---:|---:|---:|---:|---|
| Live-Min | Eco | 32 | 0.48 % | 2.91 % | 4.07 % | 7.1 % / 11.4 % | |
| Live-Min | Eco | 64 | 0.99 % | 1.80 % | 2.80 % | 4.4 % / 8.8 % | |
| Live-Min | Eco | 128 | 1.01 % | 1.66 % | 2.10 % | 2.5 % / 4.8 % | |
| Live-Min | Eco | 512 | 1.06 % | 1.58 % | 1.92 % | 1.9 % / 2.7 % | |
| Live-Min | HQ | 32 | 0.60 % | 4.02 % | 5.73 % | 11.4 % / 50.4 % | |
| Live-Min | HQ | 64 | 1.57 % | 2.40 % | 3.35 % | 6.5 % / 11.6 % | |
| Live-Min | HQ | 128 | 1.59 % | 2.27 % | 2.96 % | 4.4 % / 17.5 % | |
| Live-Min | HQ | 512 | 1.64 % | 2.09 % | 2.40 % | 2.5 % / 7.6 % | |
| Live-Min | Max | 32 | 0.84 % | 5.04 % | 6.54 % | 13.8 % / 24.7 % | < 25 %: PASS |
| Live-Min | Max | 64 | 1.78 % | 2.92 % | 3.92 % | 7.0 % / 149 % | < 25 %: PASS |
| Live-Min | Max | 128 | 1.80 % | 2.83 % | 4.25 % | 5.6 % / 160 % | < 25 %: PASS |
| Live-Min | Max | 512 | 1.84 % | 2.61 % | 2.88 % | 3.0 % / 3.0 % | < 25 %: PASS |
| Live | Eco | 32 | 0.49 % | 2.97 % | 4.53 % | 11.1 % / 13.7 % | |
| Live | Eco | 64 | 1.03 % | 1.86 % | 2.98 % | 5.9 % / 6.1 % | |
| Live | Eco | 128 | 1.05 % | 1.73 % | 2.41 % | 4.1 % / 4.8 % | |
| Live | Eco | 512 | 1.09 % | 1.63 % | 1.99 % | 2.0 % / 2.1 % | |
| **Live** | **HQ** | **32** | 0.61 % | 4.09 % | **5.99 %** | 12.9 % / 14.4 % | **< 10 %: PASS** |
| Live | HQ | 64 | 1.62 % | 2.48 % | 3.45 % | 7.0 % / 8.0 % | |
| **Live** | **HQ** | **128** | 1.65 % | 2.32 % | **2.98 %** | 3.7 % / 4.9 % | **< 5 %: PASS** |
| Live | HQ | 512 | 1.69 % | 2.18 % | 2.45 % | 2.5 % / 2.8 % | |
| Live | Max | 32 | 0.85 % | 5.05 % | 6.51 % | 16.5 % / 23.4 % | < 25 %: PASS |
| Live | Max | 64 | 1.82 % | 2.95 % | 5.42 % | 10.1 % / 229 % | < 25 %: PASS |
| Live | Max | 128 | 1.85 % | 2.84 % | 3.82 % | 5.7 % / 11.4 % | < 25 %: PASS |
| Live | Max | 512 | 1.89 % | 2.59 % | 2.87 % | 2.9 % / 7.4 % | < 25 %: PASS |
| Live+ | Eco | 32 | 0.49 % | 2.97 % | 4.84 % | 12.7 % / 347 % | |
| Live+ | Eco | 64 | 1.04 % | 1.86 % | 3.15 % | 6.7 % / 7.1 % | |
| Live+ | Eco | 128 | 1.07 % | 1.72 % | 2.43 % | 3.7 % / 86 % | |
| Live+ | Eco | 512 | 1.11 % | 1.62 % | 1.99 % | 2.0 % / 64 % | |
| Live+ | HQ | 32 | 0.63 % | 4.14 % | 6.36 % | 13.7 % / 3199 % | |
| Live+ | HQ | 64 | 1.68 % | 2.65 % | 4.00 % | 8.5 % / 438 % | |
| Live+ | HQ | 128 | 1.74 % | 2.40 % | 3.96 % | 5.0 % / 1119 % | |
| Live+ | HQ | 512 | 1.76 % | 2.24 % | 2.61 % | 2.7 % / 11.0 % | |
| Live+ | Max | 32 | 0.86 % | 5.07 % | 6.71 % | 14.1 % / 5546 % | < 25 %: PASS |
| Live+ | Max | 64 | 1.85 % | 2.92 % | 4.46 % | 8.2 % / 268 % | < 25 %: PASS |
| Live+ | Max | 128 | 1.87 % | 2.85 % | 3.43 % | 5.6 % / 892 % | < 25 %: PASS |
| Live+ | Max | 512 | 1.91 % | 2.60 % | 2.85 % | 2.9 % / 96 % | < 25 %: PASS |
| Studio | Eco | 32 | 1.10 % | 3.34 % | 5.64 % | 20.3 % / 5938 % | |
| Studio | Eco | 64 | 1.38 % | 2.07 % | 4.54 % | 6.4 % / 819 % | |
| Studio | Eco | 128 | 1.42 % | 1.92 % | 3.77 % | 5.8 % / 343 % | |
| Studio | Eco | 512 | 1.44 % | 1.85 % | 2.19 % | 2.5 % / 32 % | |
| Studio | HQ | 32 | 1.80 % | 5.55 % | 8.28 % | 18.7 % / 532 % | |
| Studio | HQ | 64 | 2.48 % | 3.25 % | 5.10 % | 9.2 % / 3409 % | |
| Studio | HQ | 128 | 2.50 % | 3.19 % | 4.67 % | 6.0 % / 90 % | |
| Studio | HQ | 512 | 2.55 % | 2.98 % | 3.41 % | 3.6 % / 358 % | |
| Studio | Max | 32 | 3.31 % | 7.54 % | **10.79 %** | 18.2 % / 1525 % | < 25 %: PASS |
| Studio | Max | 64 | 3.40 % | 4.32 % | 8.87 % | 10.4 % / 384 % | < 25 %: PASS |
| Studio | Max | 128 | 3.43 % | 4.13 % | 6.40 % | 7.0 % / 13.2 % | < 25 %: PASS |
| Studio | Max | 512 | 3.42 % | 3.89 % | 4.29 % | 4.4 % / 5.2 % | < 25 %: PASS |

**Verdict.** Every DESIGN §4 CPU gate passes with margin: Live HQ 2.98 % @128 and 5.99 % @32; Max worst 10.79 %. The P§5.5 per-instance budgets for the M4 also hold at the median: p99.9 ≤ 8 % for Live Eco/HQ and ≤ 15 % for Live Max and Studio.

**Caveats:**

- **The rt-bench exit budget (p99.9 ≤ 10 % at 32/64/128, DESIGN §3.7) is marginal for Studio Max @32.** The median is 10.79 %. The 22:29 run at load 13 gave 9.48 %, and the v1 integration measured 9.63 %. The §4 gate for Max is 25 %, so that gate holds. The exit-code budget is the stricter of the two and sits right at the edge.
- **p99 at 32 is about 7× p50** (Live HQ: 4.09 % vs 0.61 %). At 48 kHz the 64-sample hop analysis lands in every other 32-sample block. P§5.5's "work spreading" gate (p99.9 at 32 ≤ 0.75 × p99.9 at 64) fails here: 39.9 µs vs 46.0 µs = 0.87. That gate is informational.
- **The `max` column is not a property of the engine.** rt-bench runs on a normal-priority thread, unpaced. Runs that overlapped the other agents' 16-thread jobs had the thread preempted for up to 126 ms. Those were 4 of the 5 quiet runs plus the three 60 s runs at 22:32-22:35 (load 6.7 → 59), whose p99.9 reached 412 % of the period. P§1.3 and §5.5 predicted exactly this.
- **Real hosts run `process()` on a time-constraint thread.** A sign-off number needs P§5.5's paced harness with RT priority. That is a recommendation for the vt_eval owner: add an RT-promoted, paced mode to rt-bench.

## 6. x86_64 / Rosetta status

- **Build.** The installed toolchain has no x86_64 standard library for 1.99.0, and adding it means a download. The tests were therefore built with the installed nightly 1.91 (2025-08-31) and `-Zbuild-std`, offline:

  ```
  CARGO_TARGET_DIR=target/owner-core-perf-x86 cargo +nightly test -p vt_core \
      --target x86_64-apple-darwin -Zbuild-std --offline --ignore-rust-version
  ```

- **Default Rosetta 2** (no AVX advertised): the dispatcher reports `active_backend() = "scalar"`. All 66 unit tests and 2 integration tests pass, including vDSP (Accelerate exists on x86_64 macOS) and the zero-allocation proof.
- **`ROSETTA_ADVERTISE_AVX=1`:** `is_x86_feature_detected!("avx2")` and `("fma")` are true, the dispatcher reports `"avx2"`, and the same 66 + 2 tests pass.
  - This is the first time the in-crate AVX2 equivalence tests (every contract length, against an f64 reference) have run through the real dispatcher.
  - The vt_core implementer could only call the AVX2 kernels directly, because default Rosetta hides AVX2.
- **vt_engine:** all 35 tests, including the MXCSR FTZ/DAZ guard, passed as x86_64 under Rosetta in the v1 integration. They were not re-run here.
- **Not done:**
  - native x86 hardware, never tested;
  - x86 performance numbers: Rosetta is not representative (P§5.13), and translated AVX2 is slower than SSE;
  - P§5.13's relative-performance and disassembly gates.

## 7. Rejected optimisations

| Candidate | Evidence | Verdict |
|---|---|---|
| Algebraic float ops (`f32::algebraic_add/mul`, stable since 1.98) instead of explicit accumulators | Fresh (§3.1). `dot`: equal to scalar at 256-1024, **16 % slower** at 4096. `sum_sq`: 25 % faster than scalar at 256, equal at 1024, 10 % slower at 4096, and 1.5-2× slower than NEON from 1024 up. LLVM may also reassociate differently per call site or inlining, which threatens bit-exact block-size invariance (DESIGN §2.1 rule 6, P§5.12) | Rejected: no consistent gain, and a determinism risk |
| PGO (cargo-pgo) | P§6.9: 0 % (5.59 vs 5.54-5.66 µs on the analysis pipeline). Not re-run: `llvm-profdata` for LLVM 23 (rustc 1.99.0) is not installed. The available ones are LLVM 20 (Homebrew and the 1.89 toolchain), whose profile format does not match, and installing `llvm-tools` is a download | Rejected (P§ evidence) |
| `-C target-cpu=native` (apple-m4) | Fresh A/B from one source snapshot, two builds, interleaved rounds; see §7.1. P§6.9 also measured 0 %. The default `apple-m1` target already includes dotprod, fp16, fhm and rdm. A native build risks SIGILL on older Macs | Rejected |
| Hand-written NEON for memory-bound kernels (`axpy`, `mul_acc`) | Fresh (§3.1): 0.99-1.08×, not significant at 4096 | Not worth writing. The existing NEON versions stay for FMA rounding but are not a speed claim, and no new hand-NEON memory-bound kernels should be added |
| vDSP at small exact sizes (≤ 512, e.g. 384) | §3.4: 28 % slower than 512 | Don't use `FftSizing::Fast` with vDSP below 768 |
| Linear-phase FIR decimator | §3.8: 2.7-3× cheaper, but 0.8-1.25 ms more detection lag | Rejected: latency matters more than 0.05 % of a hop |
| phastft (pure-Rust FFT) | P§2.1: 1.36-1.51× over realfft, but depends on fearless_simd 0.5 (a second SIMD stack), and its c2r helpers allocate | Rejected for now |
| Legacy `vDSP_fft_zrip` | P§6.3: erratic (681 ns at 1024) | Rejected |
| `panic = "abort"` | P§6.9: −5.8 % binary size, identical speed; `unwind` lets workers degrade gracefully | Rejected |
| fearless_simd generic kernels (P§1.2's plan) | Not adopted by the implementation. vt_core uses hand `core::arch` NEON plus an auto-vectorised scalar backend plus AVX2. P§6.1 measured fearless `f32x16` equal to hand NEON on `dot` (80-82 vs 84 ns), so there is no speed case either way | Deviation from P§, recorded. No action |

### 7.1 `target-cpu=native` A/B

Both binaries were built from one copy of the tree, taken at 22:50, with identical flags apart from `RUSTFLAGS="-C target-cpu=native"` (resolved to `apple-m4`). They ran alternately: default, native, default, native (22:52-22:56, load 6-13). Each cell shows the medians of both rounds; the last column compares the best of each.

| Benchmark | default (r1 / r2) | native (r1 / r2) | native vs default |
|---|---:|---:|---:|
| engine/live/hq/32 | 9.66 / 9.66 µs | 9.67 / 9.68 µs | +0.2 % |
| engine/live/hq/128 | 38.55 / 38.53 µs | 38.63 / 38.67 µs | +0.3 % |
| engine/live/max/32 | 10.98 / 11.01 µs | 11.00 / 11.01 µs | +0.2 % |
| engine/live/max/128 | 43.85 / 44.00 µs | 43.95 / 43.85 µs | −0.0 % |
| engine/studio/hq/32 | 14.97 / 14.96 µs | 14.89 / 14.95 µs | −0.5 % |
| engine/studio/hq/128 | 59.82 / 59.92 µs | 59.64 / 59.68 µs | −0.3 % |
| engine/studio/max/32 | 20.48 / 20.53 µs | 20.43 / 20.53 µs | −0.3 % |
| engine/studio/max/128 | 82.45 / 82.10 µs | 81.84 / 81.92 µs | −0.3 % |
| dot/neon/4096 | 178.6 / 178.7 ns | 178.7 / 178.6 ns | 0.0 % |
| sum_sq/neon/4096 | 94.6 / 94.8 ns | 94.6 / 94.6 ns | 0.0 % |
| diff_sq direct NEON, W1024 | 18.08 / 16.67 µs | 16.67 / 16.66 µs | −0.1 % |
| diff_sq realfft n1536, W1024 | 4.69 / 4.68 µs | 4.69 / 4.69 µs | +0.1 % |
| fast_math/exp2_fast | 360.2 / 359.9 ns | 359.5 / 359.3 ns | −0.2 % |
| fast_math/tanh_fast | 481.0 / 480.7 ns | 473.6 / 473.0 ns | −1.6 % |
| interp_512_reads/sinc16 | 2.77 / 2.77 µs | 2.75 / 2.75 µs | −1.0 % |

All differences are within ±1.6 %, mostly within ±0.5 %; the vDSP rows (−2 to −4 %) are left out because Accelerate is not recompiled. **`target-cpu=native` gains nothing here**, which confirms P§6.9.

This build is newer than the 22:08 snapshot of §4, and it ran at a lower load. Here Live HQ costs 9.66 µs at 32 and 38.5 µs at 128, against 10.0 and 41.4 µs in §4, so part of §4's increase over v1 is load (about 4-7 %).

## 8. Reproduce

Use a private target dir when other agents share the machine. Check `sysctl -n vm.loadavg` before timing anything, and repeat any run whose load average exceeds about half the core count.

```sh
cd vocal_tune   # the repository root
export CARGO_TARGET_DIR=target/perf

# Kernels: every group, saved as a named baseline (≈ 8 min)
cargo bench -p vt_core --bench kernels -- --warm-up-time 1 --measurement-time 2 --noplot --save-baseline r1
# Only the FFT / exact-size / difference-function groups (≈ 4.5 min)
cargo bench -p vt_core --bench kernels -- --warm-up-time 1 --measurement-time 2 --noplot --save-baseline r2 \
    '^(fft|fft_exact_vs_pow2|diff_sq_direct_vs_fft)/'
# Engine::process per mode × tier × buffer (≈ 2.5 min)
cargo bench -p vt_eval --bench engine -- --noplot --save-baseline e1
# Tail latency (≈ 10 s per run; repeat 5×, only when the load is low)
cargo run --release -p vt_eval -- rt-bench --buffers 32,64,128,512 --seconds 10
# Plugin latency D, measured vs reported, every mode × range × rate
cargo run --release -p vt_eval -- latency
# vt_core under Rosetta (scalar dispatch, then AVX2 dispatch)
CARGO_TARGET_DIR=target/perf-x86 cargo +nightly test -p vt_core --target x86_64-apple-darwin -Zbuild-std --offline --ignore-rust-version
ROSETTA_ADVERTISE_AVX=1 CARGO_TARGET_DIR=target/perf-x86 cargo +nightly test -p vt_core --target x86_64-apple-darwin -Zbuild-std --offline --ignore-rust-version
# target-cpu A/B: build both from the same tree, then run the two bench binaries alternately
RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=target/perf-native cargo bench -p vt_eval --bench engine --no-run
```

- **Where the numbers are.** Criterion stores each result in `$CARGO_TARGET_DIR/criterion/<group>/<bench>/<baseline>/estimates.json`; this file quotes `median.point_estimate` and its `confidence_interval`. To compare two saved runs, use `--baseline <name>` in place of `--save-baseline`.
- **Running bench binaries directly.** `target/perf/release/deps/kernels-*` and `engine-*` accept `--bench` plus the same flags, so a run cannot trigger a rebuild from a tree that other people are editing. Set `CRITERION_HOME` to choose the output directory.

## 9. Follow-ups for other owners

1. **vt_detect:**
   - Adopt `FftSizing::Fast` (§3.4). Expect about −26 % per long-frame and short-frame correlation for Low Male, Soprano Eco and Alto/Tenor Eco; HQ Alto/Tenor is unchanged.
   - Re-evaluate `FftBackend::Vdsp` now that it is deterministic (§3.5): about 1.95× faster correlations. Keep the power of two for buffers ≤ 512.
   - The `FFT_BACKEND` comment in `verifier.rs` describes the alignment bug that §3.5 fixes.
2. **vt_eval:** give rt-bench an RT-priority (time-constraint), paced mode (P§5.5), so that tails measure the engine rather than the scheduler. Studio Max @32 sits at the 10 % exit budget.
3. **Regression agent:** re-run `cargo bench -p vt_eval --bench engine -- 'engine/live'` on the merged tree. Live modes are 20-25 % above the v1 integration figures (§4).
4. **vt_shift:** the doc comment in `latency.rs` says 4.6 ms / 3.2 ms; the formula gives 4.44 / 3.10 ms (§1).

   *Status after the final pass (§10):* 1 adopted for realfft at 3·2^k sizes (vDSP still not adopted); 3 done (§10.1); 4 done by the vt_shift owner (`latency.rs` now says 3.10 / 4.44 / 5.85 ms). 2 is open.

## 10. Final merged tree (2026-10-09)

Measured on the merged tree after every owner had finished, by the regression agent, with no other agent on the machine. The 1-minute load average was 4-10 during the timed runs (a DAW and desktop apps run in the background), except for a transient spike to 62 at the start of the final vt_detect bench run in §10.3; its unchanged cells show no effect from it. Same machine, toolchain and profile as §2.

### 10.1 Engine cost (`cargo bench -p vt_eval --bench engine -- 'engine/' --save-baseline final_merged`)

Criterion median per `Engine::process` call (48 kHz, stereo, Alto/Tenor, `melody_male`, retune 10 ms) and its share of the buffer period.

| Mode | Tier | 32 | 64 | 128 | 512 |
|---|---|---:|---:|---:|---:|
| Live-Min | Eco | 5.1 µs (0.77 %) | 10.2 µs | 20.6 µs | 82.6 µs |
| Live-Min | HQ | 8.5 µs (1.28 %) | 17.0 µs | 34.1 µs | 136 µs |
| Live-Min | Max | 9.8 µs (1.47 %) | 19.6 µs | 38.9 µs | 156 µs |
| Live | Eco | 5.4 µs (0.81 %) | 10.7 µs | 21.4 µs | 85.6 µs |
| Live | **HQ** | **8.9 µs (1.33 %)** | 17.6 µs | **35.5 µs (1.33 %)** | 143 µs |
| Live | Max | 10.2 µs (1.53 %) | 20.3 µs | 40.5 µs | 162 µs |
| Live+ | Eco | 5.4 µs (0.82 %) | 10.8 µs | 21.7 µs | 86.0 µs |
| Live+ | HQ | 9.1 µs (1.36 %) | 17.9 µs | 36.0 µs | 143 µs |
| Live+ | Max | 10.3 µs (1.55 %) | 20.5 µs | 41.0 µs | 165 µs |
| Studio | Eco | 7.7 µs (1.16 %) | 15.2 µs | 30.6 µs | 121 µs |
| Studio | HQ | 13.7 µs (2.06 %) | 27.6 µs | 55.1 µs | 219 µs |
| Studio | Max | 18.8 µs (2.82 %) | 37.6 µs | 74.6 µs | 301 µs |

Against the v1 integration report (Live HQ 8.26 / 33.0 / 131 µs at 32 / 128 / 512) the merged Live HQ costs **+8-9 %**, not the +20-25 % of the 22:08 mid-edit snapshot (§4). That matches the owners' own A/Bs of the features added since v1 (vt_detect: +0-15 % per hop for the octave and onset work, partly paid back by §10.3). Studio is cheaper than at v1 (HQ 15.36 → 13.7 µs, Max 22.4 → 18.8 µs at 32).

### 10.2 Tail latency (`vt_eval rt-bench --seconds 5`, 5 runs, median)

| Config | p50 | p99.9 | p99.9 % of period (min-max over 5 runs) | DESIGN §4 gate |
|---|---:|---:|---:|---|
| Live HQ @32 | 4.3 µs | 34.9 µs | **5.24 %** (5.04-5.29) | < 10 %: PASS |
| Live HQ @128 | 36.2 µs | 67.9 µs | **2.55 %** (2.45-2.69) | < 5 %: PASS |
| Live Max @32 | 5.2 µs | 40.1 µs | 6.02 % (5.34-6.39) | < 25 %: PASS |
| Studio HQ @32 | 10.8 µs | 45.8 µs | 6.87 % (6.72-7.17) | |
| Studio Max @32 (worst Max tier) | 19.5 µs | 59.0 µs | **8.84 %** (8.51-9.34) | < 25 %: PASS |

Every configuration (4 modes × 3 tiers × 32-512) was inside the 10 % rt-bench exit budget in all 5 runs. Studio Max @32, which sat at 10.79 % in §5, is now 8.84 %: the Studio analysis no longer shares a host block with the hop-boundary control update (RT-2 pipeline, vt_engine).

### 10.3 vt_detect exact FFT sizes (`cargo bench -p vt_detect --bench detect -- hop_48k`)

`push(64) + analyze()` per 48 kHz hop, criterion median, before (power-of-two plans) → after (3·2^k plans where they fit, `verifier::correlation_len`). Run back to back on the same tree; the unchanged cells moved by +0.4 to +1.5 %, which is the noise floor of this comparison.

| Cell | Before | After | Δ |
|---|---:|---:|---:|
| HQ Live Low Male (long 2048 → 1536, short 1024 → 768) | 22.84 µs | 20.37 µs | **−10.8 %** |
| HQ Live Wide | 24.30 µs | 21.90 µs | **−9.9 %** |
| Max Live Low Male | 24.60 µs | 22.09 µs | −10.2 % |
| HQ Studio Low Male | 33.36 µs | 30.98 µs | −7.1 % |
| HQ Live Soprano (long 1024 → 768, short 512 → 384) | 12.73 µs | 11.68 µs | **−8.2 %** |
| Eco Live Alto/Tenor (1024 → 768) | 10.33 µs | 9.37 µs | **−9.3 %** |
| Eco Live Soprano (512 → 384) | 6.85 µs | 6.61 µs | −3.6 % |
| HQ Live Alto/Tenor (1024 either way) | 15.34 µs | 15.40 µs | +0.4 % (unchanged) |
| Eco Live Low Male (1024 either way) | 10.45 µs | 10.61 µs | +1.5 % (unchanged) |

Accuracy: on vocadito (`vt_eval vocadito --conditions detect`, 7 configs × 40 files, 44.1 kHz) every metric of every one of the 280 detector rows (all columns of `detector.csv` except render time) is identical before and after. The only geometry change is at 48/96/192 kHz for Low Male / Wide, whose short frame keeps lags up to 287 instead of 288 (24 kHz verifier rate); the quality gates are unchanged by it.


## 11. Hardening pass (2026-10-09)

Measured by the regression agent after every owner of the hardening pass had finished, with no other agent running (1-minute load average 4.4-7.3: a DAW and desktop apps in the background). Same machine, toolchain and profile as §2. Owners' own A/Bs for their changes (vt_shift PERF-5/PERF-6(1), vt_detect PERF-1-4) are in their crates' docs; this section measures the merged result and the one engine change made at integration, the Studio split analysis.

### 11.1 Engine cost (`cargo bench -p vt_eval --bench engine -- 'engine/' --baseline final_merged`)

Criterion median per `Engine::process` call (48 kHz, stereo, Alto/Tenor, `melody_male`, retune 10 ms), against the §10.1 baseline. "sig." = criterion's change test p < 0.05.

| Mode | Tier | 32 | 64 | 128 | 512 | Change vs §10.1 |
|---|---|---:|---:|---:|---:|---|
| Live-Min | Eco | 4.81 µs | 9.70 µs | 19.3 µs | 77.3 µs | −6 to −7 % (sig.) |
| Live-Min | HQ | 8.18 µs | 16.4 µs | 32.9 µs | 132 µs | −3 to −4 % (n.s.) |
| Live-Min | Max | 9.47 µs | 18.8 µs | 37.7 µs | 151 µs | −4 % (n.s.) |
| Live | Eco | 5.06 µs | 10.1 µs | 20.1 µs | 80.5 µs | −6 % (sig.) |
| Live | **HQ** | **9.19 µs (1.38 %)** | 17.2 µs | **34.1 µs (1.28 %)** | 136 µs | 0 to −4 % (n.s.) |
| Live | Max | 9.71 µs | 19.4 µs | 38.8 µs | 155 µs | −5 % (n.s., p ≈ 0.1) |
| Live+ | Eco | 5.04 µs | 10.1 µs | 20.2 µs | 80.3 µs | −7 % (sig.) |
| Live+ | HQ | 8.50 µs | 17.0 µs | 33.8 µs | 135 µs | −6 % (sig.) |
| Live+ | Max | 9.68 µs | 19.3 µs | 38.7 µs | 154 µs | −6.5 % (sig.) |
| Studio | Eco | 7.32 µs | 14.6 µs | 29.4 µs | 117 µs | −4 to −5 % (sig.) |
| Studio | HQ | 13.5 µs (2.03 %) | 27.0 µs | 54.1 µs | 216 µs | −2 to −3 % (n.s.) |
| Studio | Max | 18.4 µs (2.76 %) | 37.0 µs | 73.8 µs | 295 µs | −2 % (n.s.) |

The hardening pass added work (onset rules, predicted grains, the Studio pitch-frame advance) and removed more (the shifter's per-grain copies, the detector's octave-confirm FIR design and E/H running sums): every cell is equal or cheaper than at v1 close. The split analysis does not change the mean cost per call (it moves work between calls).

### 11.2 Tail latency (`vt_eval rt-bench --seconds 5`, 3 runs, median)

| Config | p50 | p99.9 | p99.9 % of period | v1 close (§10.2) | DESIGN §4 gate |
|---|---:|---:|---:|---:|---|
| Live HQ @32 | 3.8 µs | 34.4 µs | **5.16 %** | 5.24 % | < 10 %: PASS |
| Live HQ @128 | 35.5 µs | 63.4 µs | **2.38 %** | 2.55 % | < 5 %: PASS |
| Live Max @32 | 5.3 µs | 37.4 µs | 5.61 % | 6.02 % | < 25 %: PASS |
| Studio Eco @32 | 7.5 µs | 23.4 µs | 3.51 % | 4.34 % | |
| Studio HQ @32 | 13.4 µs | 35.7 µs | 5.36 % | 6.87 % | |
| Studio Max @32 (worst Max tier) | 19.6 µs | 48.9 µs | **7.33 %** | 8.84 % | < 25 %: PASS |
| Studio Max @64 | 38.1 µs | 65.7 µs | 4.93 % | 4.95 % | |

All 60 configurations (4 modes × 3 tiers × 32-512) were inside the 10 % rt-bench exit budget in all 3 runs. The Studio @32 p50 rose (HQ 10.8 → 13.4 µs) and its p99.9 fell: at 48 kHz with 32-sample blocks the analysis used to fill every other block, and now both blocks of a hop carry about half of it. P§5.5's informational work-spreading ratio, p99.9(32) / p99.9(64) in µs with target ≤ 0.75, is now met in Studio (HQ 0.68, Max 0.74; were 0.86 and 0.89) and still not in Live (0.85), whose analysis must finish at the boundary.

### 11.3 Studio split analysis (vt_engine, PERF-7)

Studio now begins each hop's detector analysis at the boundary (`Detector::begin_analysis`, a snapshot of the stream positions) and runs its four stages before samples hop/4, 3·hop/8, hop/2 and 3·hop/4 of the next hop, then the control law before 7·hop/8; the shifter update (which re-seeds the mark tracker at a voiced onset) lands before sample 0. Before, the whole analysis ran at hop/2 with the new samples held back, and the control law at 3·hop/4. Output is bit-identical (336 output + telemetry hashes, and the engine test `studio_split_analysis_matches_a_one_shot_analysis`).

Method: a scratch tool replays a deterministic engine 9 times over 2.5 s of sung material (ten notes from 98 to 440 Hz, scoops and vibrato, stereo), keeps the minimum time per block (which removes OS noise; this is the per-block cost the engine itself needs), and reports the p99.9 and the maximum of those minima after the first 0.5 s. "Before" is the same tree with the pre-integration `vt_engine/src/lib.rs`, built with the same profile, measured right after. Studio, 40 ms:

| Rate / block | Config | p99.9 of the block period | Worst block (share of the period) |
|---|---|---|---|
| 48 kHz / 32 | Max Wide | 6.4 % → **5.8 %** | 43.5 → 39.0 µs (6.5 → 5.9 %) |
| 48 kHz / 32 | Max Low Male | 5.6 % → **5.0 %** | 37.6 → 33.3 µs |
| 48 kHz / 32 | HQ Wide | 4.5 % → **4.1 %** | 30.0 → 29.5 µs |
| 48 kHz / 64 | Max Wide | 4.5 % → 4.6 % | 60.5 → 63.0 µs (one block per hop: no change expected) |
| 96 kHz / 32 | Max Wide | 13.7 % → **9.9 %** | 46.4 → 33.7 µs (13.9 → 10.1 %) |
| 96 kHz / 32 | Max Low Male | 12.2 % → **8.4 %** | 41.1 → 30.5 µs |
| 96 kHz / 32 | HQ Wide | 10.4 % → **7.2 %** | 35.3 → 30.8 µs |
| 96 kHz / 64 | Max Wide | 7.1 % → 6.7 % | 47.7 → 45.0 µs |
| 96 kHz / 64 | HQ Wide | 5.5 % → 6.5 % | 38.5 → 45.3 µs |
| 192 kHz / 32 | Max Wide | 30.0 % → **22.0 %** | 52.9 → 47.1 µs (31.7 → 28.3 %) |
| 192 kHz / 32 | Max Low Male | 26.9 % → **18.8 %** | 47.6 → 45.8 µs |
| 192 kHz / 32 | HQ Wide | 23.7 % → **15.5 %** | 47.6 → 47.0 µs |
| 192 kHz / 64 | Max Wide | 16.0 % → **11.9 %** | 54.3 → 48.7 µs |
| 192 kHz / 64 | HQ Wide | 12.9 % → **8.8 %** | 47.7 → 49.4 µs |
| 192 kHz / 128 | Max Wide | 8.5 % → 9.7 % | 59.1 → 72.0 µs (8.9 → 10.8 %) |
| 192 kHz / 128 | HQ Wide | 8.2 % → 9.8 % | 57.6 → 72.6 µs (8.6 → 10.9 %) |

Live is unchanged (its analysis stays at the boundary): Live HQ p99.9 2.8 / 6.5 / 15.9 % at 48 / 96 / 192 kHz with 32-sample blocks in both builds.

- **Where it helps:** with four or eight blocks per hop the p99.9 falls by a quarter to a third. The worst block at 192 kHz/32 is now the shifter's mark re-seed at a voiced onset (≈ 45 µs), not the analysis, so it moves less.
- **Where it costs:** with two blocks per hop at 96 and 192 kHz the first block carries the re-seed together with Frames and Checks, so its worst case rises (192 kHz/128: 57.6 → 72.6 µs, still 10.9 % of a 667 µs period). At 48 kHz/32, also two blocks per hop, the same split is the better one.
- **Placement was measured**, in eighths of a hop for (Frames, Checks, Track, Output, control law):
  - [1, 2, 4, 6, 7] put Frames with the re-seed at four blocks per hop (192 kHz/64 worst block 59.5 µs).
  - [2, 4, 5, 6, 7] keeps the two-block case lighter (192 kHz/128 67.6 µs) but pairs Checks with Track at four blocks per hop (192 kHz/64 p99.9 13.1 % against 11.9 %) and is worse at 48 kHz/32 (Max Wide 40.8 µs).
  - [2, 3, 4, 6, 7] was kept (`vt_engine::PIPELINE_EIGHTHS`).
- **Determinism across builds:** the old and new engines, built separately, gave identical hashes on all 336 configurations, so the cross-build drift the vt_control/vt_engine owner saw in a detector-only hash (an unused padding field changed it) did not show up here.

## 12. End-to-end SIMD and FFT-backend A/B (2026-10-09)

Kernel speedups (§3) do not say how much the whole plugin gains. Three builds of the same commit (`704d95d`) were benchmarked with `cargo bench -p vt_eval --bench engine`, 128-sample blocks, 48 kHz stereo, two interleaved rounds (load 7-12 on 16 cores), median of the two rounds:

| Build | Live Eco | Live HQ | Studio Max |
|---|---:|---:|---:|
| **Shipping**: hand NEON kernels + realfft in the detector | **21.6 µs** | **36.6 µs** | **81.6 µs** |
| `simd::best = scalar` (auto-vectorised scalar kernels, still realfft) | 22.7 µs (+5 %) | 37.9 µs (+3.5 %) | 83.0 µs (+1.7 %) |
| Detector FFT on vDSP instead of realfft (NEON kernels) | 24.4 µs (+13 %) | 37.9 µs (+3.6 %) | 90.1 µs (+10 %) |

- **Hand-written NEON is worth 2-5 % of the whole engine.** The "scalar" backend is not SIMD-free: LLVM auto-vectorises its chunked accumulator loops into NEON, which is where most of the kernel gain over naive code (§3.1: 7-20×) comes from. On aarch64 a truly SIMD-free build is not possible (NEON is part of the base ABI).
- **vDSP loses end to end** even though it wins every isolated FFT (§3.3). The detector's decimated-rate FFTs are small (exact sizes of a few hundred points), where vDSP is slower (§3.4), and vDSP once produced instance-dependent last bits. The detector therefore stays on realfft (`vt_detect/src/verifier.rs`, `FFT_BACKEND`).
- The engine uses about 1.4 % of one M4 Max core in Live HQ, so the remaining CPU is better spent on quality (Max tier, worker threads) than on further kernel tuning.
