//! The frontier yardstick: is a change better, or just another setting of
//! the same trade-off?
//!
//! The baseline is swept over Tolerance (hard tune, 0-45 c) and its points
//! are joined piecewise linearly in Tolerance order. Its frontier is the
//! Pareto envelope of that curve (the baseline may use any Tolerance):
//!
//! | envelope | definition |
//! |---|---|
//! | `p95_F(a)` | lowest p95 over curve points with note agreement ≥ a |
//! | `agr_F(x)` | highest note agreement over curve points with p95 ≤ x |
//! | `wrb_F(a)` | lowest warble over curve points with note agreement ≥ a |
//! | `acc_F(w)` | highest frame accuracy over curve points with warble ≤ w |
//! | `note_F(x)` | highest per-note accuracy over curve points with p95 ≤ x |
//!
//! The gaps of a system point are positive when it beats every Tolerance of
//! the baseline: `dP95@agr = p95_F(agreeP) − p95`, `dAgr@p95 = agreeP −
//! agr_F(p95)`, `dWrb@agr = wrb_F(agreeP) − warble`, `dAcc@wrb = acc −
//! acc_F(warble)`, `dNote@p95 = note − note_F(p95)`. A system value outside
//! the baseline's range is clamped to the range's end and the gap is marked
//! clamped (e.g. more agreement than the baseline reaches at any Tolerance).
//! This is the note-intent study's `frontier.py`.

use super::metrics::NoteMetrics;

/// One point of the baseline's Tolerance sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvePoint {
    pub tolerance: f32,
    pub agree: f64,
    pub p95: f64,
    pub warble: f64,
    pub acc: f64,
    pub note: f64,
}

impl CurvePoint {
    pub fn from_metrics(tolerance: f32, m: &NoteMetrics) -> Self {
        CurvePoint {
            tolerance,
            agree: m.get("pub_agree"),
            p95: m.get("p95"),
            warble: m.get("warble_min"),
            acc: m.get("acc_agree"),
            note: m.get("note_dom"),
        }
    }
}

/// A gap and whether the system value had to be clamped to the curve's range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gap {
    pub value: f64,
    pub clamped: bool,
}

/// The five gaps of a system point (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gaps {
    pub p95_at_agree: Gap,
    pub agree_at_p95: Gap,
    pub warble_at_agree: Gap,
    pub acc_at_warble: Gap,
    pub note_at_p95: Gap,
}

impl Gaps {
    pub fn all(&self) -> [Gap; 5] {
        [self.p95_at_agree, self.agree_at_p95, self.warble_at_agree, self.acc_at_warble, self.note_at_p95]
    }

    pub const NAMES: [&'static str; 5] = ["dP95@agr", "dAgr@p95", "dWrb@agr", "dAcc@wrb", "dNote@p95"];
}

/// Best `value` over the piecewise-linear curve (sorted by Tolerance)
/// restricted to `key ≥ c` (`at_least`) or `key ≤ c`; lowest when
/// `want_min`, else highest. Returns the value and whether `c` was clamped
/// to the curve's range.
fn envelope(pts: &[(f64, f64)], c: f64, want_min: bool, at_least: bool) -> Gap {
    let lo = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let hi = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let mut c = c;
    let mut clamped = false;
    if at_least && c > hi {
        c = hi;
        clamped = true;
    }
    if !at_least && c < lo {
        c = lo;
        clamped = true;
    }
    let ok = |k: f64| if at_least { k >= c } else { k <= c };
    let mut best: Option<f64> = None;
    let mut take = |v: f64| {
        if best.is_none_or(|b| if want_min { v < b } else { v > b }) {
            best = Some(v);
        }
    };
    for w in pts.windows(2) {
        let ((c0, v0), (c1, v1)) = (w[0], w[1]);
        for (ca, va) in [(c0, v0), (c1, v1)] {
            if ok(ca) {
                take(va);
            }
        }
        // The segment crosses the constraint: its point at c.
        if (c0 - c) * (c1 - c) < 0.0 {
            take(v0 + (v1 - v0) * (c - c0) / (c1 - c0));
        }
    }
    if let [(k, v)] = pts
        && ok(*k)
    {
        take(*v);
    }
    Gap { value: best.unwrap_or(f64::NAN), clamped }
}

/// Gaps of a system point to the baseline curve (sorted by Tolerance).
pub fn gaps(curve: &[CurvePoint], sys: &CurvePoint) -> Gaps {
    let line = |k: fn(&CurvePoint) -> f64, v: fn(&CurvePoint) -> f64| -> Vec<(f64, f64)> { curve.iter().map(|p| (k(p), v(p))).collect() };
    let gap = |f: Gap, d: f64| Gap { value: d, clamped: f.clamped };
    let pf = envelope(&line(|p| p.agree, |p| p.p95), sys.agree, true, true);
    let af = envelope(&line(|p| p.p95, |p| p.agree), sys.p95, false, false);
    let wf = envelope(&line(|p| p.agree, |p| p.warble), sys.agree, true, true);
    let accf = envelope(&line(|p| p.warble, |p| p.acc), sys.warble, false, false);
    let nf = envelope(&line(|p| p.p95, |p| p.note), sys.p95, false, false);
    Gaps {
        p95_at_agree: gap(pf, pf.value - sys.p95),
        agree_at_p95: gap(af, sys.agree - af.value),
        warble_at_agree: gap(wf, wf.value - sys.warble),
        acc_at_warble: gap(accf, sys.acc - accf.value),
        note_at_p95: gap(nf, sys.note - nf.value),
    }
}

/// Mean of each gap over several system points, unclamped gaps only.
pub fn mean_gaps(all: &[Gaps]) -> [f64; 5] {
    let mut out = [f64::NAN; 5];
    for (i, o) in out.iter_mut().enumerate() {
        let v: Vec<f64> = all.iter().map(|g| g.all()[i]).filter(|g| !g.clamped && g.value.is_finite()).map(|g| g.value).collect();
        if !v.is_empty() {
            *o = v.iter().sum::<f64>() / v.len() as f64;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(tolerance: f32, agree: f64, p95: f64, warble: f64, acc: f64, note: f64) -> CurvePoint {
        CurvePoint { tolerance, agree, p95, warble, acc, note }
    }

    /// Raising Tolerance trades agreement for p95 along the baseline curve.
    fn curve() -> Vec<CurvePoint> {
        vec![pt(0.0, 90.0, 30.0, 60.0, 70.0, 82.0), pt(10.0, 86.0, 22.0, 40.0, 72.0, 80.0), pt(20.0, 80.0, 16.0, 25.0, 71.0, 76.0)]
    }

    #[test]
    fn a_point_on_the_curve_has_zero_gaps() {
        let c = curve();
        let g = gaps(&c, &c[1]);
        for x in g.all() {
            assert!(x.value.abs() < 1e-9 && !x.clamped, "{g:?}");
        }
    }

    #[test]
    fn a_better_point_has_positive_gaps() {
        let c = curve();
        // Agreement 86 (the 10 c point) with the p95 of the 20 c point.
        let g = gaps(&c, &pt(8.0, 86.0, 16.0, 30.0, 73.0, 80.0));
        assert!((g.p95_at_agree.value - 6.0).abs() < 1e-9);
        assert!((g.agree_at_p95.value - 6.0).abs() < 1e-9);
        assert!((g.warble_at_agree.value - 10.0).abs() < 1e-9);
        // Interpolated: at warble 30 the curve reaches 71 + (72 − 71)·(30 − 25)/(40 − 25).
        assert!((g.acc_at_warble.value - (73.0 - (71.0 + 1.0 / 3.0))).abs() < 1e-9);
        assert!((g.note_at_p95.value - 4.0).abs() < 1e-9);
        assert_eq!(mean_gaps(&[g, g])[0], g.p95_at_agree.value);
    }

    #[test]
    fn values_beyond_the_curve_are_clamped() {
        let c = curve();
        let g = gaps(&c, &pt(8.0, 95.0, 10.0, 10.0, 80.0, 90.0));
        assert!(g.p95_at_agree.clamped && g.agree_at_p95.clamped);
        assert!(mean_gaps(&[g])[0].is_nan());
    }
}
