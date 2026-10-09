//! Kernel benchmarks for `skato_core` (DESIGN §3.1). Each group pits an
//! optimisation against its baseline so the speedup is measured, not assumed:
//!
//! * `dot`, `sum_sq`, `axpy`, `mul_acc`, `diff_sq`: scalar vs NEON (vs AVX2)
//!   at 256 / 1024 / 4096 (plus a naive iterator baseline for the reductions).
//!   `dot` and `sum_sq` also carry an `algebraic` variant (a plain fold with
//!   `f32::algebraic_add`/`algebraic_mul`, the rejected alternative to
//!   explicit accumulators).
//! * `diff_sq_direct_vs_fft`: direct type-I difference vs the `Correlator`
//!   path for W=1024/τ=512 and W=2048/τ=1024, at the power-of-two size and at
//!   the exact fast size (1536 / 3072).
//! * `fft`: RealFft vs Vdsp at n = 512 … 8192 plus the fast sizes 768, 1536,
//!   3072, 6144 (autocorrelation = forward + inverse, and power spectrum =
//!   forward only).
//! * `fft_exact_vs_pow2`: the detector's cross-correlation (W-sample head
//!   against the W + τ_max buffer, τ_max + 1 lags) at `FftSizing::PowerOfTwo`
//!   vs `FftSizing::Fast` for the 48 kHz detector cells, both backends.
//! * `fast_math`: fast vs std exp2 / log2 / tanh over 1024-element arrays.
//! * `interp_512_reads`: cubic vs Sinc8 vs Sinc16, 512 reads per iteration.
//! * `decimator`: the minimum-phase IIR `Decimator` vs a linear-phase Kaiser
//!   FIR polyphase decimator of the same stopband (65 dB from fs_out/2,
//!   passband to 0.4·fs_out), per 1024 input samples.
//!
//! Run: `cargo bench -p skato_core --bench kernels -- --warm-up-time 1 --measurement-time 2`

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use skato_core::fft::{Correlator, FftBackend, FftSizing};
use skato_core::filter::Decimator;
use skato_core::interp::{InterpQuality, Interpolator};
use skato_core::math::{fast_exp2, fast_log2, fast_tanh};
use skato_core::ring::Ring;
use skato_core::simd;

const SIZES: [usize; 3] = [256, 1024, 4096];

fn signal(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            0.5 * ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) + 0.5 * (i as f32 * 0.05).sin()
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
fn avx2_ok() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

fn bench_reductions(c: &mut Criterion) {
    let mut g = c.benchmark_group("dot");
    for &n in &SIZES {
        let a = signal(n, 1);
        let b = signal(n, 2);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("naive_iter", n), &n, |bch, _| {
            bch.iter(|| black_box(&a).iter().zip(black_box(&b)).map(|(x, y)| x * y).sum::<f32>())
        });
        g.bench_with_input(BenchmarkId::new("algebraic", n), &n, |bch, _| {
            bch.iter(|| {
                black_box(&a).iter().zip(black_box(&b)).fold(0.0f32, |s, (x, y)| s.algebraic_add(x.algebraic_mul(*y)))
            })
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
            bch.iter(|| simd::scalar::dot(black_box(&a), black_box(&b)))
        });
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        g.bench_with_input(BenchmarkId::new("neon", n), &n, |bch, _| {
            bch.iter(|| simd::neon::dot(black_box(&a), black_box(&b)))
        });
        #[cfg(target_arch = "x86_64")]
        if avx2_ok() {
            g.bench_with_input(BenchmarkId::new("avx2", n), &n, |bch, _| {
                // SAFETY: AVX2 + FMA detected above.
                bch.iter(|| unsafe { simd::avx2::dot(black_box(&a), black_box(&b)) })
            });
        }
    }
    g.finish();

    let mut g = c.benchmark_group("sum_sq");
    for &n in &SIZES {
        let a = signal(n, 3);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("naive_iter", n), &n, |bch, _| {
            bch.iter(|| black_box(&a).iter().map(|x| x * x).sum::<f32>())
        });
        g.bench_with_input(BenchmarkId::new("algebraic", n), &n, |bch, _| {
            bch.iter(|| black_box(&a).iter().fold(0.0f32, |s, x| s.algebraic_add(x.algebraic_mul(*x))))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
            bch.iter(|| simd::scalar::sum_sq(black_box(&a)))
        });
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        g.bench_with_input(BenchmarkId::new("neon", n), &n, |bch, _| bch.iter(|| simd::neon::sum_sq(black_box(&a))));
        #[cfg(target_arch = "x86_64")]
        if avx2_ok() {
            g.bench_with_input(BenchmarkId::new("avx2", n), &n, |bch, _| {
                // SAFETY: AVX2 + FMA detected above.
                bch.iter(|| unsafe { simd::avx2::sum_sq(black_box(&a)) })
            });
        }
    }
    g.finish();
}

fn bench_elementwise(c: &mut Criterion) {
    let mut g = c.benchmark_group("axpy");
    for &n in &SIZES {
        let x = signal(n, 4);
        let mut y = signal(n, 5);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
            bch.iter(|| simd::scalar::axpy(black_box(&mut y), black_box(1e-3), black_box(&x)))
        });
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        g.bench_with_input(BenchmarkId::new("neon", n), &n, |bch, _| {
            bch.iter(|| simd::neon::axpy(black_box(&mut y), black_box(1e-3), black_box(&x)))
        });
        #[cfg(target_arch = "x86_64")]
        if avx2_ok() {
            g.bench_with_input(BenchmarkId::new("avx2", n), &n, |bch, _| {
                // SAFETY: AVX2 + FMA detected above.
                bch.iter(|| unsafe { simd::avx2::axpy(black_box(&mut y), black_box(1e-3), black_box(&x)) })
            });
        }
    }
    g.finish();

    let mut g = c.benchmark_group("mul_acc");
    for &n in &SIZES {
        let x = signal(n, 6);
        let w = signal(n, 7);
        let mut y = signal(n, 8);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
            bch.iter(|| simd::scalar::mul_acc(black_box(&mut y), black_box(&x), black_box(&w)))
        });
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        g.bench_with_input(BenchmarkId::new("neon", n), &n, |bch, _| {
            bch.iter(|| simd::neon::mul_acc(black_box(&mut y), black_box(&x), black_box(&w)))
        });
        #[cfg(target_arch = "x86_64")]
        if avx2_ok() {
            g.bench_with_input(BenchmarkId::new("avx2", n), &n, |bch, _| {
                // SAFETY: AVX2 + FMA detected above.
                bch.iter(|| unsafe { simd::avx2::mul_acc(black_box(&mut y), black_box(&x), black_box(&w)) })
            });
        }
    }
    g.finish();
}

fn bench_diff_sq(c: &mut Criterion) {
    // Kernel sizes: window W = n, n/2 lags.
    let mut g = c.benchmark_group("diff_sq");
    for &n in &SIZES {
        let lags = n / 2;
        let x = signal(n + lags, 9);
        let mut out = vec![0.0f32; lags];
        g.throughput(Throughput::Elements((n * lags) as u64));
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
            bch.iter(|| simd::scalar::diff_sq(black_box(&x), n, 0, black_box(&mut out)))
        });
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        g.bench_with_input(BenchmarkId::new("neon", n), &n, |bch, _| {
            bch.iter(|| simd::neon::diff_sq(black_box(&x), n, 0, black_box(&mut out)))
        });
        #[cfg(target_arch = "x86_64")]
        if avx2_ok() {
            g.bench_with_input(BenchmarkId::new("avx2", n), &n, |bch, _| {
                // SAFETY: AVX2 + FMA detected above.
                bch.iter(|| unsafe { simd::avx2::diff_sq(black_box(&x), n, 0, black_box(&mut out)) })
            });
        }
    }
    g.finish();

    // Detector configurations: direct difference vs the FFT path.
    let mut g = c.benchmark_group("diff_sq_direct_vs_fft");
    for &(w, tau) in &[(1024usize, 512usize), (2048, 1024)] {
        let x = signal(w + tau, 10);
        let mut out = vec![0.0f32; tau];
        let label = format!("W{w}_tau{tau}");
        g.bench_function(BenchmarkId::new("direct_scalar", &label), |bch| {
            bch.iter(|| simd::scalar::diff_sq(black_box(&x), w, 0, black_box(&mut out)))
        });
        g.bench_function(BenchmarkId::new("direct_best", &label), |bch| {
            bch.iter(|| simd::diff_sq(black_box(&x), w, 0, black_box(&mut out)))
        });
        // `tau` lags need n ≥ w + tau − 1: 2048 / 4096 as a power of two,
        // 1536 / 3072 at the fast size.
        for sizing in [FftSizing::PowerOfTwo, FftSizing::Fast] {
            for be in [FftBackend::RealFft, FftBackend::Vdsp] {
                let mut corr = Correlator::with_sizing(w + tau, be, sizing);
                if be == FftBackend::Vdsp && corr.backend() != FftBackend::Vdsp {
                    continue; // no vDSP on this platform
                }
                let name = format!("fft_{}_n{}", backend_tag(be), corr.len());
                g.bench_function(BenchmarkId::new(name, &label), |bch| {
                    bch.iter(|| corr.diff_sq(black_box(&x), w, 0, black_box(&mut out)))
                });
            }
        }
    }
    g.finish();
}

fn backend_tag(be: FftBackend) -> &'static str {
    match be {
        FftBackend::RealFft => "realfft",
        FftBackend::Vdsp => "vdsp",
    }
}

fn bench_fft(c: &mut Criterion) {
    let mut g = c.benchmark_group("fft");
    // Powers of two and the fast sizes between them (3·2^8 … 3·2^11).
    for &n in &[512usize, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192] {
        let x = signal(n / 2, 11);
        let mut out = vec![0.0f32; n / 2 + 1];
        g.throughput(Throughput::Elements(n as u64));
        for be in [FftBackend::RealFft, FftBackend::Vdsp] {
            let mut corr = Correlator::with_sizing(n, be, FftSizing::Fast);
            debug_assert_eq!(corr.len(), n);
            if be == FftBackend::Vdsp && corr.backend() != FftBackend::Vdsp {
                continue;
            }
            let tag = backend_tag(be);
            g.bench_with_input(BenchmarkId::new(format!("{tag}_autocorrelate"), n), &n, |bch, _| {
                bch.iter(|| corr.autocorrelate(black_box(&x), black_box(&mut out)))
            });
            g.bench_with_input(BenchmarkId::new(format!("{tag}_power_spectrum"), n), &n, |bch, _| {
                bch.iter(|| corr.power_spectrum(black_box(&x), black_box(&mut out)))
            });
        }
    }
    g.finish();

    // The verifier's per-hop cross term at 48 kHz (skato_detect LongFrame /
    // ShortFrame geometry: W = 2.2/f_min at 24 kHz (HQ) or 16 kHz (Eco),
    // τ_max = 1.06·rate/f_min; buffer W + τ_max + 1), plus the P§3.4.1 shapes.
    let mut g = c.benchmark_group("fft_exact_vs_pow2");
    for &(label, w, tau) in &[
        ("hq_lowmale_W960_tau463", 960usize, 463usize),
        ("hq_soprano_W377_tau182", 377, 182),
        ("eco_altotenor_W440_tau212", 440, 212),
        ("eco_soprano_W251_tau122", 251, 122),
        ("short_lowmale_W480_tau287", 480, 287),
        ("W1024_tau512", 1024, 511),
        ("W2048_tau1024", 2048, 1023),
    ] {
        let len = w + tau + 1;
        let x = signal(len, 13);
        let mut out = vec![0.0f32; tau + 1];
        for sizing in [FftSizing::PowerOfTwo, FftSizing::Fast] {
            for be in [FftBackend::RealFft, FftBackend::Vdsp] {
                let mut corr = Correlator::with_sizing(len, be, sizing);
                if be == FftBackend::Vdsp && corr.backend() != FftBackend::Vdsp {
                    continue;
                }
                let name = format!(
                    "{}_{}_n{}",
                    backend_tag(be),
                    if sizing == FftSizing::Fast { "fast" } else { "pow2" },
                    corr.len()
                );
                g.bench_function(BenchmarkId::new(name, label), |bch| {
                    bch.iter(|| corr.cross_correlate(black_box(&x[..w]), black_box(&x), black_box(&mut out)))
                });
            }
        }
    }
    g.finish();
}

fn bench_fast_math(c: &mut Criterion) {
    let n = 1024;
    let xs_exp: Vec<f32> = (0..n).map(|i| -20.0 + 40.0 * i as f32 / n as f32).collect();
    let xs_log: Vec<f32> = (0..n).map(|i| 1e-3 + 1e3 * (i as f32 / n as f32)).collect();
    let xs_tanh: Vec<f32> = (0..n).map(|i| -4.0 + 8.0 * i as f32 / n as f32).collect();
    let mut out = vec![0.0f32; n];
    let mut g = c.benchmark_group("fast_math");
    g.throughput(Throughput::Elements(n as u64));
    macro_rules! pair {
        ($name:literal, $xs:expr, $f:expr) => {
            g.bench_function($name, |bch| {
                bch.iter(|| {
                    for (o, &x) in out.iter_mut().zip(black_box(&$xs)) {
                        *o = $f(x);
                    }
                    black_box(&out);
                })
            });
        };
    }
    pair!("exp2_fast", xs_exp, fast_exp2);
    pair!("exp2_std", xs_exp, f32::exp2);
    pair!("log2_fast", xs_log, fast_log2);
    pair!("log2_std", xs_log, f32::log2);
    pair!("tanh_fast", xs_tanh, fast_tanh);
    pair!("tanh_std", xs_tanh, f32::tanh);
    g.finish();
}

fn bench_interp(c: &mut Criterion) {
    let mut ring = Ring::new(8192);
    ring.extend(&signal(8192, 12));
    let it = Interpolator::new();
    // 512 reads advancing by a non-integer step, as a PSOLA grain / Classic
    // read pointer would.
    let positions: Vec<f64> = (0..512).map(|k| 1000.0 + k as f64 * 1.0371).collect();
    let mut g = c.benchmark_group("interp_512_reads");
    g.throughput(Throughput::Elements(512));
    for (name, q, rate) in [
        ("cubic", InterpQuality::Cubic, 1.0f32),
        ("sinc8", InterpQuality::Sinc8, 1.0),
        ("sinc16", InterpQuality::Sinc16, 1.0),
        ("sinc16_rate1.5", InterpQuality::Sinc16, 1.5),
    ] {
        g.bench_function(name, |bch| {
            bch.iter(|| {
                let mut acc = 0.0f32;
                for &p in black_box(&positions) {
                    acc += it.read(q, &ring, p, rate);
                }
                black_box(acc)
            })
        });
    }
    g.finish();
}

/// Linear-phase baseline for `decimator`: a Kaiser-windowed sinc low-pass with
/// the IIR's specification (65 dB at and above fs_out/2, transition from
/// 0.4·fs_out), evaluated only at the kept output instants (polyphase cost:
/// taps / factor multiply-adds per input sample) on a contiguous history.
struct FirDecimator {
    taps: Vec<f32>,
    hist: Vec<f32>,
    factor: usize,
    phase: usize,
}

impl FirDecimator {
    fn new(factor: usize) -> Self {
        // Kaiser's estimate: N = (A − 7.95) / (14.36·Δf) + 1, Δf = 0.1 / factor
        // (normalised to fs_in); β = 0.1102·(A − 8.7).
        let atten = 65.0f64;
        let n = (((atten - 7.95) / (14.36 * 0.1 / factor as f64)).ceil() as usize + 1) | 1;
        let fc = 0.45 / factor as f64; // cutoff mid-transition, cycles/sample
        let mut win = vec![0.0f32; n];
        skato_core::window::kaiser(&mut win, (0.1102 * (atten - 8.7)) as f32);
        let mid = (n - 1) as f64 / 2.0;
        let taps: Vec<f32> = win
            .iter()
            .enumerate()
            .map(|(i, &wv)| {
                let t = i as f64 - mid;
                let s = if t == 0.0 {
                    2.0 * fc
                } else {
                    (2.0 * std::f64::consts::PI * fc * t).sin() / (std::f64::consts::PI * t)
                };
                (s * wv as f64) as f32
            })
            .collect();
        Self { hist: vec![0.0; n + 1024], taps, factor, phase: 0 }
    }

    /// Filters `input` (≤ 1024 samples), writing kept outputs; returns the count.
    fn push_block(&mut self, input: &[f32], out: &mut [f32]) -> usize {
        let n = self.taps.len();
        // `hist` = the previous n − 1 samples, then this block.
        let keep = n - 1;
        let total = keep + input.len();
        let Some(hist) = self.hist.get_mut(..total) else { return 0 };
        hist[keep..].copy_from_slice(input);
        let mut produced = 0;
        for i in 0..input.len() {
            self.phase += 1;
            if self.phase == self.factor {
                self.phase = 0;
                if let Some(o) = out.get_mut(produced) {
                    *o = simd::dot(&hist[i..i + n], &self.taps);
                    produced += 1;
                }
            }
        }
        hist.copy_within(total - keep..total, 0);
        produced
    }
}

fn bench_decimator(c: &mut Criterion) {
    let x = signal(1024, 14);
    let mut out = vec![0.0f32; 1024];
    let mut g = c.benchmark_group("decimator");
    g.throughput(Throughput::Elements(1024));
    // The detector's streams: 48 kHz → 24 kHz (HQ), 16 kHz (Eco); 96 kHz →
    // 24 kHz; 192 kHz → 24 kHz.
    for &(label, sr, factor) in &[
        ("48k_div2", 48_000.0f32, 2usize),
        ("48k_div3", 48_000.0, 3),
        ("96k_div4", 96_000.0, 4),
        ("192k_div8", 192_000.0, 8),
    ] {
        let mut iir = Decimator::new(sr, factor);
        g.bench_function(BenchmarkId::new(format!("iir_cheby2_order{}", iir.order()), label), |bch| {
            bch.iter(|| iir.push_block(black_box(&x), black_box(&mut out)))
        });
        let mut fir = FirDecimator::new(factor);
        g.bench_function(BenchmarkId::new(format!("fir_kaiser_{}taps", fir.taps.len()), label), |bch| {
            bch.iter(|| fir.push_block(black_box(&x), black_box(&mut out)))
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_reductions,
    bench_elementwise,
    bench_diff_sq,
    bench_fft,
    bench_fast_math,
    bench_interp,
    bench_decimator
);
criterion_main!(benches);
