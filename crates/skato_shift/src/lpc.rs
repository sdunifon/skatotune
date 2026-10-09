//! Linear prediction by the autocorrelation method (Levinson-Durbin), used to
//! compute the LP residual whose peaks anchor the pitch marks on glottal
//! closures (R§8.13 "Anchor").

/// Solves the normal equations for predictor coefficients `a[1..=p]` from
/// autocorrelation `r[0..=p]` (Levinson-Durbin, f64). On return `a[0] = 1`
/// and the prediction-error filter is `e[n] = Σ_k a[k]·x[n−k]`.
///
/// `a.len()` must equal `r.len()`; `p = r.len() − 1`. Returns the final
/// prediction-error power, or `None` if the recursion becomes unstable
/// (non-positive error, e.g. for silence or a numerically singular frame), in
/// which case `a` is left as a valid but truncated predictor.
pub(crate) fn levinson(r: &[f64], a: &mut [f64]) -> Option<f64> {
    let p = r.len().checked_sub(1)?;
    if a.len() != r.len() {
        return None;
    }
    a.fill(0.0);
    a[0] = 1.0;
    let mut err = *r.first()?;
    if !(err > 0.0 && err.is_finite()) {
        return None;
    }
    for i in 1..=p {
        // Reflection coefficient k_i = −(r[i] + Σ_{j<i} a[j]·r[i−j]) / err.
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / err;
        if !k.is_finite() || k.abs() >= 1.0 {
            return None;
        }
        // Symmetric in-place update a[j] += k·a[i−j] for j < i.
        for j in 1..=i / 2 {
            let (aj, aij) = (a[j], a[i - j]);
            a[j] = aj + k * aij;
            a[i - j] = aij + k * aj;
        }
        a[i] = k;
        err *= 1.0 - k * k;
        if err <= 0.0 {
            return None;
        }
    }
    Some(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_ar2_process() {
        // x[n] = 1.3 x[n−1] − 0.6 x[n−2] + w: the predictor must find
        // a = [1, −1.3, 0.6] from the exact autocorrelation.
        let (a1, a2) = (1.3f64, -0.6f64);
        // Yule-Walker: r1 = a1 r0 + a2 r1, r2 = a1 r1 + a2 r0.
        let r0 = 1.0;
        let r1 = a1 * r0 / (1.0 - a2);
        let r2 = a1 * r1 + a2 * r0;
        let mut a = [0.0; 3];
        let e = levinson(&[r0, r1, r2], &mut a).unwrap();
        assert!((a[1] + 1.3).abs() < 1e-12 && (a[2] - 0.6).abs() < 1e-12, "{a:?}");
        assert!(e > 0.0 && e < r0);
    }

    #[test]
    fn rejects_degenerate_input() {
        let mut a = [0.0; 3];
        assert!(levinson(&[0.0, 0.0, 0.0], &mut a).is_none());
        assert!(levinson(&[f64::NAN, 0.0, 0.0], &mut a).is_none());
        assert!(levinson(&[1.0, 1.0, 1.0], &mut a).is_none()); // |k| = 1
        let mut short = [0.0; 2];
        assert!(levinson(&[1.0, 0.5, 0.2], &mut short).is_none());
        assert!(levinson(&[], &mut []).is_none());
    }
}
