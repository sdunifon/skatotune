//! Real-singing gates on vocadito (EV-4): the synthetic DESIGN §4 gates are
//! idealised and were partly used to tune the HMM and vibrato constants, so
//! these check the engine on real takes it was never tuned on.
//!
//! Two kinds of threshold, because the annotation itself caps what can be
//! measured (the offline reference tracker scores voicing F1 0.94 and fine
//! p95 24 c against it, mostly annotation conventions):
//!
//! - **Detector, relative:** the engine's detector (Live HQ, amount 0) must
//!   stay within a fixed margin of the offline f64 reference tracker scored
//!   the same way on the same frames (RPA 50 c, octave and other gross
//!   errors, voicing F1).
//! - **Correction, regression guards** per voice range (hard tune,
//!   chromatic): stable-frame output p95, onset settling p90 and the
//!   onset-glitch rate may not get worse than the values measured when the
//!   gate was added (plus a margin). They are far from the synthetic §4
//!   targets (p95 < 15 c, settling < 30 ms); the report prints both so the
//!   gap stays visible. Tighten the guards as the engine improves.

use vt_core::{LatencyMode, QualityTier, VoiceRange};

use super::measure::{CorrCounts, DetCounts};
use super::{Condition, Outcome, RunConfig};
use crate::suite::{Gate, range_name};

/// The configuration the gates apply to.
pub fn gate_config() -> RunConfig {
    RunConfig::new(LatencyMode::Live, QualityTier::Hq)
}

/// Detector margins against the offline reference tracker.
pub const RPA50_MARGIN: f64 = 0.02;
pub const OCTAVE_MARGIN: f64 = 0.005;
pub const GROSS_OTHER_MARGIN: f64 = 0.01;
pub const VOICING_F1_MARGIN: f64 = 0.015;

/// Hard-tune regression guards per voice range: (range, stable-frame output
/// p95 c, onset settle p90 ms, onset glitch %), Live HQ, 256-sample blocks,
/// all 40 files. Re-baselined after the 2026-10-09 hardening pass (merged
/// tree): low male 30.2 c / 68 ms / 10.8 %, alto/tenor 29.8 c / 66 ms /
/// 5.2 %, soprano 23.4 c / 54 ms / 6.6 % (renders are deterministic), down
/// from 33.1 / 88 / 13.5, 31.2 / 103 / 16.0 and 25.9 / 76 / 11.9 at v1
/// close. The guards add about 4 c, 20 ms and 6 points (the low-male group
/// has only 5 files). While the crates were changed in parallel the same
/// numbers moved by up to 3 c, 17 ms and 8 points. They catch gross
/// regressions only; the synthetic DESIGN §4 targets are printed alongside.
pub const HARD_TUNE_GUARDS: [(VoiceRange, f64, f64, f64); 3] = [
    (VoiceRange::LowMale, 34.0, 88.0, 17.0),
    (VoiceRange::AltoTenor, 34.0, 86.0, 11.0),
    (VoiceRange::Soprano, 27.0, 74.0, 13.0),
];

/// Synthetic DESIGN §4 targets, printed next to the real-singing numbers.
const DESIGN_P95_C: f64 = 15.0;
const DESIGN_SETTLE_MS: f64 = 30.0;

fn pooled_det(o: &Outcome, cfg: RunConfig) -> Option<DetCounts> {
    let mut c = DetCounts::default();
    let mut any = false;
    for d in o.det.iter().filter(|d| d.config == cfg) {
        c.merge(&d.counts);
        any = true;
    }
    any.then_some(c)
}

fn pooled_corr(o: &Outcome, cfg: RunConfig, range: Option<VoiceRange>) -> Option<CorrCounts> {
    let mut c = CorrCounts::default();
    let mut any = false;
    for r in o.corr.iter().filter(|r| {
        r.config == cfg
            && r.condition == Condition::HardChromatic
            && range.is_none_or(|g| r.range == g)
    }) {
        c.merge(&r.counts);
        any = true;
    }
    any.then_some(c)
}

/// Evaluate the real-singing gates (None where the run lacks the needed
/// config or condition: `pass: None`).
pub fn gates(o: &Outcome) -> Vec<Gate> {
    let cfg = gate_config();
    let mut g = Vec::new();
    let mut reference = DetCounts::default();
    for f in &o.files {
        reference.merge(&f.reference);
    }
    g.push(match pooled_det(o, cfg) {
        Some(e) => {
            let ok = e.rpa50() >= reference.rpa50() - RPA50_MARGIN
                && e.octave_rate() <= reference.octave_rate() + OCTAVE_MARGIN
                && e.gross_other_rate() <= reference.gross_other_rate() + GROSS_OTHER_MARGIN
                && e.f1() >= reference.f1() - VOICING_F1_MARGIN;
            Gate {
                name: "Real singing: detector vs offline reference (Live HQ)",
                target: "RPA50 ≥ ref − 2 pt; octave ≤ ref + 0.5 pt; other gross ≤ ref + 1 pt; voicing F1 ≥ ref − 0.015",
                value: format!(
                    "RPA50 {:.1} % (ref {:.1}); octave {:.2} % (ref {:.2}); other gross {:.2} % (ref {:.2}); F1 {:.3} (ref {:.3})",
                    100.0 * e.rpa50(),
                    100.0 * reference.rpa50(),
                    100.0 * e.octave_rate(),
                    100.0 * reference.octave_rate(),
                    100.0 * e.gross_other_rate(),
                    100.0 * reference.gross_other_rate(),
                    e.f1(),
                    reference.f1()
                ),
                pass: Some(ok),
            }
        }
        None => Gate {
            name: "Real singing: detector vs offline reference (Live HQ)",
            target: "within margins of the offline reference",
            value: "not run (needs live_hq + detect)".into(),
            pass: None,
        },
    });
    for (range, p95_max, settle_max, glitch_max) in HARD_TUNE_GUARDS {
        let name = match range {
            VoiceRange::LowMale => "Real singing: hard tune, low male (Live HQ, chromatic)",
            VoiceRange::AltoTenor => "Real singing: hard tune, alto/tenor (Live HQ, chromatic)",
            _ => "Real singing: hard tune, soprano (Live HQ, chromatic)",
        };
        let target = "regression guards (see vocadito::gates); synthetic §4 targets p95 < 15 c, settling < 30 ms";
        g.push(match pooled_corr(o, cfg, Some(range)) {
            Some(c) if c.stable > 0 => {
                let (p95, settle, glitch) = (
                    c.out.percentile(95.0),
                    c.settle_p90(),
                    c.onset_glitch_pct(),
                );
                Gate {
                    name,
                    target,
                    value: format!(
                        "{}: output p95 {:.1} c (guard {p95_max}, §4 {DESIGN_P95_C}); settle p90 {:.0} ms (guard {settle_max}, §4 {DESIGN_SETTLE_MS}); onset glitches {:.1} % (guard {glitch_max})",
                        range_name(range),
                        p95,
                        settle,
                        glitch
                    ),
                    // No checked onset → no glitch rate (NaN): not a failure.
                    pass: Some(
                        p95 <= p95_max
                            && settle <= settle_max
                            && (glitch.is_nan() || glitch <= glitch_max),
                    ),
                }
            }
            _ => Gate {
                name,
                target,
                value: format!("not run ({} files with live_hq + hard_chromatic)", range_name(range)),
                pass: None,
            },
        });
    }
    g
}
