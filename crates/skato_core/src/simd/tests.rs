//! Equivalence tests: every backend (and the dispatcher) against an f64
//! reference at the lengths the contract lists.

use super::*;

const LENGTHS: [usize; 14] = [0, 1, 3, 15, 16, 17, 31, 33, 63, 100, 1024, 2047, 2048, 4096];

/// One backend's kernels as plain function pointers.
struct Backend {
    name: &'static str,
    dot: fn(&[f32], &[f32]) -> f32,
    sum_sq: fn(&[f32]) -> f32,
    axpy: fn(&mut [f32], f32, &[f32]),
    mul_acc: fn(&mut [f32], &[f32], &[f32]),
    diff_sq: fn(&[f32], usize, usize, &mut [f32]),
}

#[cfg(target_arch = "x86_64")]
mod avx2_safe {
    //! Safe shims; only called after `has_avx2()` returned true.
    use super::super::{avx2, best};
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        assert!(best::has_avx2());
        // SAFETY: feature checked above.
        unsafe { avx2::dot(a, b) }
    }
    pub fn sum_sq(a: &[f32]) -> f32 {
        assert!(best::has_avx2());
        // SAFETY: feature checked above.
        unsafe { avx2::sum_sq(a) }
    }
    pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
        assert!(best::has_avx2());
        // SAFETY: feature checked above.
        unsafe { avx2::axpy(y, a, x) }
    }
    pub fn mul_acc(y: &mut [f32], x: &[f32], w: &[f32]) {
        assert!(best::has_avx2());
        // SAFETY: feature checked above.
        unsafe { avx2::mul_acc(y, x, w) }
    }
    pub fn diff_sq(x: &[f32], w: usize, tau0: usize, out: &mut [f32]) {
        assert!(best::has_avx2());
        // SAFETY: feature checked above.
        unsafe { avx2::diff_sq(x, w, tau0, out) }
    }
}

fn backends() -> Vec<Backend> {
    let mut v = vec![
        Backend {
            name: "scalar",
            dot: scalar::dot,
            sum_sq: scalar::sum_sq,
            axpy: scalar::axpy,
            mul_acc: scalar::mul_acc,
            diff_sq: scalar::diff_sq,
        },
        Backend { name: "dispatch", dot, sum_sq, axpy, mul_acc, diff_sq },
    ];
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    v.push(Backend {
        name: "neon",
        dot: neon::dot,
        sum_sq: neon::sum_sq,
        axpy: neon::axpy,
        mul_acc: neon::mul_acc,
        diff_sq: neon::diff_sq,
    });
    #[cfg(target_arch = "x86_64")]
    if best::has_avx2() {
        v.push(Backend {
            name: "avx2",
            dot: avx2_safe::dot,
            sum_sq: avx2_safe::sum_sq,
            axpy: avx2_safe::axpy,
            mul_acc: avx2_safe::mul_acc,
            diff_sq: avx2_safe::diff_sq,
        });
    }
    v
}

/// Deterministic pseudo-random signal in [-1, 1] with some structure.
fn signal(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let noise = ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            0.6 * noise + 0.4 * (i as f32 * 0.071).sin()
        })
        .collect()
}

/// Rounding tolerance for an f32 sum of `n` terms with absolute sum `abs`.
fn tol(n: usize, abs: f64) -> f64 {
    // Summation error is ≤ ~(n / lanes + log2 lanes)·eps·Σ|terms| for the
    // multi-accumulator schemes; use a generous constant over √n growth.
    (4.0 + (n as f64).sqrt()) * 1.2e-7 * abs + 1e-30
}

#[test]
fn dispatcher_reports_a_backend() {
    let b = active_backend();
    assert!(["neon", "avx2", "scalar"].contains(&b));
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    assert_eq!(b, "neon");
}

#[test]
fn dot_matches_f64_reference() {
    for be in backends() {
        for &n in &LENGTHS {
            for extra in [0usize, 5] {
                let a = signal(n, 1);
                let b = signal(n + extra, 2); // unequal lengths: min() semantics
                let (r, abs) = a.iter().zip(&b).fold((0.0f64, 0.0f64), |(r, s), (&x, &y)| {
                    (r + x as f64 * y as f64, s + (x as f64 * y as f64).abs())
                });
                let got = (be.dot)(&a, &b) as f64;
                assert!((got - r).abs() <= tol(n, abs), "{} dot n={n}: {got} vs {r}", be.name);
                let got2 = (be.dot)(&b, &a) as f64;
                assert!((got2 - r).abs() <= tol(n, abs), "{} dot (swapped) n={n}", be.name);
            }
        }
    }
}

#[test]
fn sum_sq_matches_f64_reference() {
    for be in backends() {
        for &n in &LENGTHS {
            let a = signal(n, 3);
            let r: f64 = a.iter().map(|&x| x as f64 * x as f64).sum();
            let got = (be.sum_sq)(&a) as f64;
            assert!((got - r).abs() <= tol(n, r), "{} sum_sq n={n}: {got} vs {r}", be.name);
        }
    }
}

#[test]
fn axpy_matches_f64_reference() {
    for be in backends() {
        for &n in &LENGTHS {
            for extra in [0usize, 3] {
                let x = signal(n + extra, 4);
                let y0 = signal(n, 5);
                let a = -0.37f32;
                let mut y = y0.clone();
                (be.axpy)(&mut y, a, &x);
                for i in 0..n {
                    let r = y0[i] as f64 + a as f64 * x[i] as f64;
                    let e = (y[i] as f64 - r).abs();
                    assert!(
                        e <= 2.5e-7 * (y0[i].abs() as f64 + (a * x[i]).abs() as f64) + 1e-30,
                        "{} axpy n={n} i={i}",
                        be.name
                    );
                }
                // Longer y than x: the tail of y is untouched.
                let mut ylong = signal(n + 4, 6);
                let before = ylong.clone();
                (be.axpy)(&mut ylong, 2.0, &x[..n]);
                assert_eq!(&ylong[n..], &before[n..], "{} axpy tail", be.name);
            }
        }
    }
}

#[test]
fn mul_acc_matches_f64_reference() {
    for be in backends() {
        for &n in &LENGTHS {
            let x = signal(n, 7);
            let w = signal(n + 2, 8);
            let y0 = signal(n + 1, 9);
            let mut y = y0.clone();
            (be.mul_acc)(&mut y, &x, &w);
            for i in 0..n {
                let r = y0[i] as f64 + x[i] as f64 * w[i] as f64;
                let e = (y[i] as f64 - r).abs();
                assert!(
                    e <= 2.5e-7 * (y0[i].abs() as f64 + (x[i] * w[i]).abs() as f64) + 1e-30,
                    "{} mul_acc n={n} i={i}",
                    be.name
                );
            }
            // y is longer than x: the extra element is untouched.
            assert_eq!(y[n], y0[n], "{} mul_acc tail", be.name);
        }
    }
}

fn diff_ref(x: &[f32], w: usize, tau: usize) -> (f64, f64) {
    let mut d = 0.0f64;
    for j in 0..w.min(x.len()) {
        let b = x.get(j + tau).copied().unwrap_or(0.0) as f64;
        let e = x[j] as f64 - b;
        d += e * e;
    }
    (d, d)
}

#[test]
fn diff_sq_matches_f64_reference() {
    for be in backends() {
        for &w in &LENGTHS {
            for &(tau0, n_out) in &[(0usize, 0usize), (0, 1), (1, 3), (2, 5), (7, 8), (11, 33), (3, 64)] {
                let x = signal(w + tau0 + n_out.max(1) - 1, 10 + w as u64);
                let mut out = vec![f32::NAN; n_out];
                (be.diff_sq)(&x, w, tau0, &mut out);
                for (k, &got) in out.iter().enumerate() {
                    let (r, abs) = diff_ref(&x, w, tau0 + k);
                    assert!(
                        (got as f64 - r).abs() <= tol(w, abs),
                        "{} diff_sq w={w} tau0={tau0} k={k}: {got} vs {r}",
                        be.name
                    );
                }
            }
        }
    }
}

#[test]
fn diff_sq_out_of_range_lags_are_zero_padded() {
    for be in backends() {
        // x too short for every requested lag: no panic, zero-padded semantics.
        let x = signal(100, 11);
        for &(w, tau0, n_out) in
            &[(64usize, 30usize, 20usize), (200, 0, 5), (10, 95, 10), (0, 0, 4), (50, usize::MAX - 2, 3)]
        {
            let mut out = vec![f32::NAN; n_out];
            (be.diff_sq)(&x, w, tau0, &mut out);
            for (k, &got) in out.iter().enumerate() {
                let tau = tau0.saturating_add(k);
                let mut r = 0.0f64;
                for j in 0..w.min(x.len()) {
                    let b = j.checked_add(tau).and_then(|i| x.get(i)).copied().unwrap_or(0.0) as f64;
                    r += (x[j] as f64 - b).powi(2);
                }
                assert!(
                    (got as f64 - r).abs() <= tol(w, r) + 1e-6,
                    "{} padded w={w} tau0={tau0} k={k}: {got} vs {r}",
                    be.name
                );
            }
        }
        // Empty x.
        let mut out = [1.0f32; 3];
        (be.diff_sq)(&[], 16, 0, &mut out);
        assert_eq!(out, [0.0; 3]);
    }
}

#[test]
fn diff_sq_periodic_signal_has_zero_at_period() {
    // A signal with exact period 50 gives d(50) == 0 (to rounding).
    let x: Vec<f32> = (0..2048).map(|i| ((i % 50) as f32 / 50.0 - 0.5) * 2.0).collect();
    for be in backends() {
        let mut out = vec![0.0f32; 100];
        (be.diff_sq)(&x, 1024, 1, &mut out);
        assert!(out[49] < 1e-6, "{}: d(50) = {}", be.name, out[49]);
        assert!(out[24] > 10.0);
    }
}

#[test]
fn backends_agree_closely_on_large_inputs() {
    // Cross-check backends against each other (tighter than vs f64 because
    // all use similar multi-accumulator orders).
    let a = signal(4096, 12);
    let b = signal(4096, 13);
    let bes = backends();
    let r0 = (bes[0].dot)(&a, &b);
    for be in &bes[1..] {
        let r = (be.dot)(&a, &b);
        assert!((r - r0).abs() < 1e-3, "{} vs scalar: {r} vs {r0}", be.name);
    }
}

#[test]
fn helpers() {
    assert_eq!(valid_lags(100, 64, 0, 50), 37);
    assert_eq!(valid_lags(100, 64, 30, 50), 7);
    assert_eq!(valid_lags(100, 64, 37, 50), 0);
    assert_eq!(valid_lags(100, usize::MAX, 1, 5), 0);
    assert_eq!(head(&[1.0, 2.0, 3.0], 2), &[1.0, 2.0]);
    assert_eq!(head(&[1.0], 5), &[1.0]);
}
