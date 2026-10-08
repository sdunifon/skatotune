//! Kernel benchmarks for `vt_core` (DESIGN §3.1). Each group pits an
//! optimisation against its baseline so the speedup is measured, not assumed:
//!
//! * `dot`, `sum_sq`, `axpy`, `mul_acc`, `diff_sq`: scalar vs NEON (vs AVX2)
//!   at 256 / 1024 / 4096 (plus a naive iterator baseline for the reductions).
//! * `diff_sq_direct_vs_fft`: direct type-I difference vs the `Correlator`
//!   path for W=1024/τ=512 and W=2048/τ=1024.
//! * `fft`: RealFft vs Vdsp at n = 512 … 8192 (autocorrelation = forward +
//!   inverse, and power spectrum = forward only).
//! * `fast_math`: fast vs std exp2 / log2 / tanh over 1024-element arrays.
//! * `interp_512_reads`: cubic vs Sinc8 vs Sinc16, 512 reads per iteration.
//!
//! Run: `cargo bench -p vt_core --bench kernels -- --warm-up-time 1 --measurement-time 2`

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use vt_core::fft::{Correlator, FftBackend};
use vt_core::interp::{InterpQuality, Interpolator};
use vt_core::math::{fast_exp2, fast_log2, fast_tanh};
use vt_core::ring::Ring;
use vt_core::simd;

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
        let n = (w + tau).next_power_of_two();
        for be in [FftBackend::RealFft, FftBackend::Vdsp] {
            let mut corr = Correlator::new(n, be);
            let name = match corr.backend() {
                FftBackend::RealFft => "fft_realfft",
                FftBackend::Vdsp => "fft_vdsp",
            };
            if be == FftBackend::Vdsp && corr.backend() != FftBackend::Vdsp {
                continue; // no vDSP on this platform
            }
            g.bench_function(BenchmarkId::new(name, &label), |bch| {
                bch.iter(|| corr.diff_sq(black_box(&x), w, 0, black_box(&mut out)))
            });
        }
    }
    g.finish();
}

fn bench_fft(c: &mut Criterion) {
    let mut g = c.benchmark_group("fft");
    for &n in &[512usize, 1024, 2048, 4096, 8192] {
        let x = signal(n / 2, 11);
        let mut out = vec![0.0f32; n / 2 + 1];
        g.throughput(Throughput::Elements(n as u64));
        for be in [FftBackend::RealFft, FftBackend::Vdsp] {
            let mut corr = Correlator::new(n, be);
            if be == FftBackend::Vdsp && corr.backend() != FftBackend::Vdsp {
                continue;
            }
            let tag = if be == FftBackend::Vdsp { "vdsp" } else { "realfft" };
            g.bench_with_input(BenchmarkId::new(format!("{tag}_autocorrelate"), n), &n, |bch, _| {
                bch.iter(|| corr.autocorrelate(black_box(&x), black_box(&mut out)))
            });
            g.bench_with_input(BenchmarkId::new(format!("{tag}_power_spectrum"), n), &n, |bch, _| {
                bch.iter(|| corr.power_spectrum(black_box(&x), black_box(&mut out)))
            });
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

criterion_group!(benches, bench_reductions, bench_elementwise, bench_diff_sq, bench_fft, bench_fast_math, bench_interp);
criterion_main!(benches);
