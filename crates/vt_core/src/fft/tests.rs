//! Both backends against direct O(n²) correlation / DFT in f64.

use super::*;

fn backends() -> [FftBackend; 2] {
    [FftBackend::RealFft, FftBackend::Vdsp]
}

fn signal(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let noise = ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            0.5 * noise + 0.5 * (i as f32 * 0.043).sin()
        })
        .collect()
}

fn direct_xcorr(a: &[f32], b: &[f32], lags: usize) -> Vec<f64> {
    (0..lags)
        .map(|t| a.iter().enumerate().map(|(j, &x)| x as f64 * b.get(j + t).copied().unwrap_or(0.0) as f64).sum())
        .collect()
}

fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| v as f64 * v as f64).sum()
}

/// max |got − ref| normalised by the Cauchy-Schwarz bound √(Σa²·Σb²).
fn rel_err(got: &[f32], reference: &[f64], a: &[f32], b: &[f32]) -> f64 {
    let norm = (energy(a) * energy(b)).sqrt().max(1e-30);
    got.iter().zip(reference).map(|(&g, &r)| (g as f64 - r).abs()).fold(0.0, f64::max) / norm
}

#[test]
fn best_backend_and_fallbacks() {
    if cfg!(target_os = "macos") {
        assert_eq!(FftBackend::best(), FftBackend::Vdsp);
        assert_eq!(Correlator::new(2048, FftBackend::Vdsp).backend(), FftBackend::Vdsp);
    } else {
        assert_eq!(FftBackend::best(), FftBackend::RealFft);
        assert_eq!(Correlator::new(2048, FftBackend::Vdsp).backend(), FftBackend::RealFft);
    }
    assert_eq!(Correlator::new(2048, FftBackend::RealFft).backend(), FftBackend::RealFft);
    // Sizes round up to powers of two.
    assert_eq!(Correlator::new(1000, FftBackend::RealFft).len(), 1024);
    assert_eq!(Correlator::new(0, FftBackend::Vdsp).len(), 2);
    // Every small size works with either request (vDSP may decline tiny
    // lengths; the fallback must still be correct).
    for n in [2usize, 4, 8, 16, 32] {
        for be in backends() {
            let mut c = Correlator::new(n, be);
            let x = signal(n / 2, 3);
            let mut out = vec![0.0f32; n / 2 + 1];
            c.autocorrelate(&x, &mut out);
            let r = direct_xcorr(&x, &x, n / 2 + 1);
            assert!(rel_err(&out, &r, &x, &x) < 1e-5, "n={n} {:?}", c.backend());
        }
    }
}

#[test]
fn cross_correlation_matches_direct() {
    for n in [64usize, 512, 1024, 2048, 4096, 8192] {
        for be in backends() {
            let mut c = Correlator::new(n, be);
            // Detector-shaped call: a = W-sample head, b = W + τmax buffer.
            for &(la, lb, lags) in
                &[(n / 2, n / 2 + n / 4, n / 4 + 1), (n / 4, n, n / 2), (n / 2 + 1, n / 2, n / 2), (1, n, 3)]
            {
                let a = signal(la, 1 + n as u64);
                let b = signal(lb, 2 + n as u64);
                let mut out = vec![f32::NAN; lags];
                c.cross_correlate(&a, &b, &mut out);
                let r = direct_xcorr(&a, &b, lags);
                let e = rel_err(&out, &r, &a, &b);
                assert!(e < 1e-4, "{be:?} n={n} la={la} lb={lb}: rel err {e:e}");
            }
        }
    }
}

#[test]
fn autocorrelation_matches_direct() {
    for n in [64usize, 512, 2048, 8192] {
        for be in backends() {
            let mut c = Correlator::new(n, be);
            let x = signal(n / 2, 7 + n as u64);
            let mut out = vec![0.0f32; n / 2 + 1];
            c.autocorrelate(&x, &mut out);
            let r = direct_xcorr(&x, &x, n / 2 + 1);
            let e = rel_err(&out, &r, &x, &x);
            assert!(e < 1e-4, "{be:?} n={n}: rel err {e:e}");
            // r[0] is the energy.
            assert!((out[0] as f64 - energy(&x)).abs() / energy(&x) < 1e-5);
        }
    }
}

#[test]
fn power_spectrum_matches_direct_dft() {
    for n in [64usize, 256, 1024] {
        for be in backends() {
            let mut c = Correlator::new(n, be);
            let x = signal(n - 3, 11 + n as u64); // zero-padded by 3
            let mut out = vec![0.0f32; n / 2 + 1];
            c.power_spectrum(&x, &mut out);
            let e_tot = energy(&x) * n as f64; // Parseval scale of |X|²
            for (k, &got) in out.iter().enumerate() {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (j, &v) in x.iter().enumerate() {
                    let ph = -2.0 * core::f64::consts::PI * (j * k) as f64 / n as f64;
                    re += v as f64 * ph.cos();
                    im += v as f64 * ph.sin();
                }
                let r = re * re + im * im;
                assert!((got as f64 - r).abs() / e_tot < 1e-6, "{be:?} n={n} k={k}: {got} vs {r}");
            }
        }
    }
    // Unit sine on bin 8 of n = 256 → |X|² = (n/2)² there.
    for be in backends() {
        let n = 256;
        let mut c = Correlator::new(n, be);
        let x: Vec<f32> = (0..n).map(|i| (2.0 * core::f32::consts::PI * 8.0 * i as f32 / n as f32).sin()).collect();
        let mut out = vec![0.0f32; n / 2 + 1];
        c.power_spectrum(&x, &mut out);
        assert!((out[8] / (128.0 * 128.0) - 1.0).abs() < 1e-4, "{be:?}: {}", out[8]);
        assert!(out[0] < 1e-3 && out[n / 2] < 1e-3);
    }
}

#[test]
fn backends_agree() {
    let n = 2048;
    let mut r = Correlator::new(n, FftBackend::RealFft);
    let mut v = Correlator::new(n, FftBackend::Vdsp);
    let a = signal(1024, 21);
    let b = signal(1536, 22);
    let mut o1 = vec![0.0f32; 513];
    let mut o2 = vec![0.0f32; 513];
    r.cross_correlate(&a, &b, &mut o1);
    v.cross_correlate(&a, &b, &mut o2);
    let norm = (energy(&a) * energy(&b)).sqrt();
    for (x, y) in o1.iter().zip(&o2) {
        assert!(((x - y) as f64).abs() / norm < 1e-5);
    }
    // Repeated calls give identical results (no state leaks between calls).
    let mut o3 = vec![0.0f32; 513];
    v.cross_correlate(&a, &b, &mut o3);
    assert_eq!(o2, o3);
    r.cross_correlate(&a, &b, &mut o3);
    assert_eq!(o1, o3);
}

#[test]
fn diff_sq_via_correlator_matches_direct() {
    // The two detector configurations from the contract.
    for &(w, tmax, n) in &[(1024usize, 512usize, 2048usize), (2048, 1024, 4096)] {
        let x = signal(w + tmax, 31);
        let mut direct = vec![0.0f32; tmax + 1];
        crate::simd::diff_sq(&x, w, 0, &mut direct[..tmax]);
        for be in backends() {
            let mut c = Correlator::new(n, be);
            let mut fast = vec![0.0f32; tmax];
            c.diff_sq(&x, w, 0, &mut fast);
            let e0 = energy(&x[..w]);
            for (k, (&f, &d)) in fast.iter().zip(&direct).enumerate() {
                let scale = e0 + energy(&x[k..k + w]);
                assert!(((f - d) as f64).abs() / scale < 1e-5, "{be:?} w={w} k={k}: {f} vs {d}");
            }
            // tau0 > 0 and a short tail of zero-padded lags.
            let mut fast2 = vec![0.0f32; 40];
            let mut direct2 = vec![0.0f32; 40];
            c.diff_sq(&x, w, tmax - 20, &mut fast2);
            crate::simd::diff_sq(&x, w, tmax - 20, &mut direct2);
            for (k, (&f, &d)) in fast2.iter().zip(&direct2).enumerate() {
                assert!(((f - d) as f64).abs() / (2.0 * e0) < 1e-5, "{be:?} tail k={k}: {f} vs {d}");
            }
        }
    }
    // Too-large request falls back to the direct kernel (still correct).
    let x = signal(600, 5);
    let mut c = Correlator::new(256, FftBackend::best());
    let mut a = vec![0.0f32; 100];
    let mut b = vec![0.0f32; 100];
    c.diff_sq(&x, 400, 3, &mut a);
    crate::simd::diff_sq(&x, 400, 3, &mut b);
    assert_eq!(a, b);
}

#[test]
fn out_of_contract_lengths_never_panic() {
    for be in backends() {
        let mut c = Correlator::new(64, be);
        let big = signal(500, 1);
        let mut out = vec![f32::NAN; 200];
        c.cross_correlate(&big, &big, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out[64..].iter().all(|&v| v == 0.0));
        c.autocorrelate(&big, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        let mut spec = vec![f32::NAN; 100];
        c.power_spectrum(&big, &mut spec);
        assert!(spec[33..].iter().all(|&v| v == 0.0));
        let mut short = vec![0.0f32; 5];
        c.power_spectrum(&[], &mut short);
        assert!(short.iter().all(|&v| v == 0.0));
        c.cross_correlate(&[], &[], &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        let mut odd = vec![0.0f32; 7];
        let x = signal(9, 2);
        c.autocorrelate(&x, &mut odd);
        let r = direct_xcorr(&x, &x, 7);
        assert!(rel_err(&odd, &r, &x, &x) < 1e-5);
        c.diff_sq(&[], 10, 0, &mut odd);
        assert!(odd.iter().all(|&v| v == 0.0));
        c.diff_sq(&x, 0, 0, &mut []);
    }
}
