//! Window functions and fade curves.
//!
//! These are evaluated at construction time (tables, analysis windows) or per
//! crossfade sample; all computation is done in f64 and rounded once.

use core::f64::consts::PI;

/// Fills `out` with a *periodic* Hann window, `w[n] = 0.5 − 0.5·cos(2πn/N)`
/// with `N = out.len()`. Periodic (rather than symmetric) so that windows hopped
/// by N/2 sum exactly to 1 and the DFT of the window has the textbook 3-bin
/// main lobe.
pub fn hann(out: &mut [f32]) {
    let n = out.len() as f64;
    for (i, w) in out.iter_mut().enumerate() {
        *w = (0.5 - 0.5 * (2.0 * PI * i as f64 / n).cos()) as f32;
    }
}

/// Fills `out` with a symmetric Kaiser window of shape parameter `beta`:
/// `w[n] = I0(β·√(1 − (2n/(N−1) − 1)²)) / I0(β)`. A single-sample window is 1.
/// Negative or NaN `beta` is treated as 0 (rectangular).
pub fn kaiser(out: &mut [f32], beta: f32) {
    let len = out.len();
    if len == 1 {
        out[0] = 1.0;
        return;
    }
    let beta = if beta.is_finite() { (beta as f64).max(0.0) } else { 0.0 };
    let norm = 1.0 / bessel_i0(beta);
    let m = (len - 1) as f64;
    for (i, w) in out.iter_mut().enumerate() {
        let r = 2.0 * i as f64 / m - 1.0;
        *w = (bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) * norm) as f32;
    }
}

/// Continuous Kaiser window on u ∈ [−1, 1] (0 outside), used by the sinc
/// tables, which evaluate the window at fractional tap positions.
pub fn kaiser_at(u: f64, beta: f64) -> f64 {
    if u.is_nan() || u.abs() > 1.0 {
        return 0.0;
    }
    bessel_i0(beta * (1.0 - u * u).max(0.0).sqrt()) / bessel_i0(beta)
}

/// Modified Bessel function of the first kind, order 0, by its power series
/// `Σ ((x/2)^k / k!)²`. Converges for every finite x; the loop is capped.
pub fn bessel_i0(x: f64) -> f64 {
    let half = 0.5 * x;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for k in 1..500 {
        let r = half / k as f64;
        term *= r * r;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// Equal-gain raised-cosine fade-in, `0.5 − 0.5·cos(π t)` for t ∈ [0, 1]
/// (clamped; NaN → 0). `raised_cosine(t) + raised_cosine(1 − t) = 1`, the right
/// crossfade for *correlated* signals such as the PSOLA output and the delayed
/// dry signal (R§8.18). For uncorrelated signals use [`equal_power`].
#[inline]
pub fn raised_cosine(t: f32) -> f32 {
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    (0.5 - 0.5 * (core::f32::consts::PI * t).cos()).clamp(0.0, 1.0)
}

/// Equal-power fade-in, `sin(π t / 2)` = `√raised_cosine(t)` for t ∈ [0, 1]
/// (clamped; NaN → 0). `equal_power(t)² + equal_power(1 − t)² = 1`, the right
/// crossfade for *uncorrelated* signals.
#[inline]
pub fn equal_power(t: f32) -> f32 {
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    (core::f32::consts::FRAC_PI_2 * t).sin().clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_is_periodic_and_sums_to_one() {
        let n = 64;
        let mut w = vec![0.0f32; n];
        hann(&mut w);
        assert_eq!(w[0], 0.0);
        assert!((w[n / 2] - 1.0).abs() < 1e-7);
        // Periodic symmetry: w[k] == w[N - k].
        for k in 1..n {
            assert!((w[k] - w[n - k]).abs() < 1e-6);
        }
        // 50 % overlap-add is exactly flat.
        for k in 0..n / 2 {
            assert!((w[k] + w[k + n / 2] - 1.0).abs() < 1e-6);
        }
        let mut empty: [f32; 0] = [];
        hann(&mut empty);
    }

    #[test]
    fn kaiser_shape() {
        let mut w = vec![0.0f32; 33];
        kaiser(&mut w, 8.0);
        assert!((w[16] - 1.0).abs() < 1e-7);
        for k in 0..33 {
            assert!((w[k] - w[32 - k]).abs() < 1e-7);
        }
        // Edge value 1/I0(β).
        assert!((w[0] as f64 - 1.0 / bessel_i0(8.0)).abs() < 1e-7);
        // β = 0 is rectangular.
        kaiser(&mut w, 0.0);
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-7));
        let mut one = [0.0f32];
        kaiser(&mut one, 5.0);
        assert_eq!(one[0], 1.0);
        kaiser(&mut w, f32::NAN);
        assert!(w.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bessel_i0_reference_values() {
        // Abramowitz & Stegun table 9.8 values.
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-15);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-13);
        assert!((bessel_i0(5.0) - 27.239_871_823_604_44).abs() < 1e-10);
        assert!((bessel_i0(10.0) - 2_815.716_628_466_254).abs() < 1e-8);
        assert_eq!(kaiser_at(1.5, 5.0), 0.0);
        assert!((kaiser_at(0.0, 5.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn fades_are_complementary() {
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            assert!((raised_cosine(t) + raised_cosine(1.0 - t) - 1.0).abs() < 1e-6);
            let (a, b) = (equal_power(t), equal_power(1.0 - t));
            assert!((a * a + b * b - 1.0).abs() < 1e-6);
            assert!((raised_cosine(t).sqrt() - a).abs() < 1e-6);
        }
        assert_eq!(raised_cosine(0.0), 0.0);
        assert_eq!(raised_cosine(1.0), 1.0);
        assert_eq!(raised_cosine(-3.0), 0.0);
        assert_eq!(raised_cosine(7.0), 1.0);
        assert_eq!(raised_cosine(f32::NAN), 0.0);
        assert_eq!(equal_power(f32::NAN), 0.0);
        assert_eq!(equal_power(2.0), 1.0);
    }
}
