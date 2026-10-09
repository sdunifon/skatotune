//! Centre/vibrato split in the full pipeline (R§7.6.2 D, J; DESIGN §3.3).

use super::*;
use vt_core::{QualityTier, frame_flags};

const RATE: f32 = 6.0;
const EXTENT: f32 = 40.0;
/// Sung centre: 25 c sharp of C4.
const CENTRE: f32 = 6025.0;

fn sung(i: usize) -> PitchEstimate {
    est(CENTRE + EXTENT * (2.0 * std::f32::consts::PI * RATE * i as f32 * HOP_S).sin())
}

/// Output vibrato (half-extent) and output centre over the last second of a
/// 2.5 s vibrato note, for τ = 0.
fn render(tier: QualityTier, preserve: bool, depth: f32) -> (f32, f32, Vec<ControlOutput>) {
    render_tau(tier, preserve, depth, 0.0)
}

fn render_tau(tier: QualityTier, preserve: bool, depth: f32, tau: f32) -> (f32, f32, Vec<ControlOutput>) {
    let mut c = live(|s| {
        s.retune_ms = tau;
        s.tier = tier;
        s.preserve_vibrato = preserve;
        s.vibrato_depth = depth;
    });
    let n = hops(2500.0);
    let outs = run(&mut c, n, sung);
    let tail: Vec<f32> = outs[n - hops(1000.0)..].iter().map(|o| o.frame.output).collect();
    (half_range(tail.iter().copied()), mean(tail.iter().copied()), outs)
}

#[test]
fn preserve_vibrato_at_tau_zero_keeps_the_vibrato_around_the_corrected_centre() {
    for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
        let (extent, centre, outs) = render(tier, true, 1.0);
        assert!((extent / EXTENT - 1.0).abs() < 0.10, "{tier:?}: output extent {extent}");
        assert!((centre - 6000.0).abs() < 3.0, "{tier:?}: output centre {centre}");
        // Engaged within ~1.5 vibrato cycles, and reported for the GUI.
        let engaged = outs.iter().position(|o| o.frame.flags & frame_flags::VIBRATO != 0).unwrap_or(usize::MAX);
        assert!((engaged as f32) * HOP_MS < 400.0, "{tier:?}: engaged after {} ms", engaged as f32 * HOP_MS);
        let last = outs[outs.len() - 1].frame;
        assert!((last.vib_rate - RATE).abs() < 0.3 && (last.vib_extent - EXTENT).abs() < 4.0, "{last:?}");
        // The vibrato is kept *in phase* with the input (output − 6000 tracks
        // input − centre).
        let err = outs[outs.len() - hops(1000.0)..]
            .iter()
            .map(|o| ((o.frame.output - 6000.0) - (o.frame.detected - CENTRE)).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 6.0, "{tier:?}: vibrato shape error {err}");
    }
}

#[test]
fn depth_zero_removes_vibrato_and_other_depths_scale_it() {
    for tier in [QualityTier::Eco, QualityTier::Hq] {
        let (extent, centre, _) = render(tier, true, 0.0);
        assert!(extent < 0.1 * EXTENT, "{tier:?}: depth 0 left {extent} c");
        assert!((centre - 6000.0).abs() < 1.0);
        let (extent, _, _) = render(tier, true, 2.0);
        assert!((extent / (2.0 * EXTENT) - 1.0).abs() < 0.1, "{tier:?}: depth 200% gave {extent}");
        // Inverted: the output moves against the input.
        let (extent, _, outs) = render(tier, true, -1.0);
        assert!((extent / EXTENT - 1.0).abs() < 0.12, "{tier:?}: depth -100% gave {extent}");
        let o = outs[outs.len() - 1].frame;
        let (inp, out) = (o.detected - CENTRE, o.output - 6000.0);
        assert!(inp * out <= 0.0 || inp.abs() < 5.0, "not inverted: in {inp} out {out}");
    }
}

#[test]
fn classic_hard_tune_flattens_vibrato() {
    // Preserve off at τ = 0 is Auto-Tune 5 behaviour: vibrato removed.
    let (extent, centre, outs) = render(QualityTier::Hq, false, 1.0);
    assert!(extent < 0.5, "extent {extent}");
    assert!((centre - 6000.0).abs() < 0.5);
    // Even before the tracker engages the output sits exactly on the target.
    for o in &outs[hops(10.0)..hops(150.0)] {
        assert!((o.frame.output - o.frame.target).abs() < 1e-3);
    }
}

/// Output pitch minus its mean over the last second.
fn tail_deviation(outs: &[ControlOutput]) -> Vec<f32> {
    let tail: Vec<f32> = outs[outs.len() - hops(1000.0)..].iter().map(|o| o.frame.output).collect();
    let m = mean(tail.iter().copied());
    tail.iter().map(|x| x - m).collect()
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        ab += f64::from(*x) * f64::from(*y);
        aa += f64::from(*x) * f64::from(*x);
        bb += f64::from(*y) * f64::from(*y);
    }
    (ab / (aa * bb).sqrt().max(1e-12)) as f32
}

/// CL-1: with Preserve off, Vibrato Depth scales the sung vibrato before
/// the correction, so it scales whatever vibrato survives the retune filter
/// and never inverts it. (It used to cancel the vibrato in the correction and
/// add `(depth − 1)·v` on top: Depth 0 % gave a full-extent anti-phase
/// vibrato at τ = 0, and Depth 200 % gave 1× instead of 2×.)
#[test]
fn preserve_off_depth_scales_the_surviving_vibrato_without_inverting_it() {
    for tier in [QualityTier::Eco, QualityTier::Hq] {
        // τ = 0: the hard tune removes all of it, whatever the depth.
        for depth in [0.0f32, 0.5, 1.0, 2.0] {
            let (extent, centre, _) = render(tier, false, depth);
            assert!(extent < 0.5, "{tier:?} depth {depth}: {extent} c of vibrato left at τ = 0");
            assert!((centre - 6000.0).abs() < 0.5, "{tier:?} depth {depth}: centre {centre}");
        }
        // τ = 20 ms: the high-passed residual is scaled by the depth and
        // stays in phase with the 100 % residual.
        let (_, _, outs) = render_tau(tier, false, 1.0, 20.0);
        let base = tail_deviation(&outs);
        let base_extent = half_range(base.iter().copied());
        assert!(base_extent > 0.4 * EXTENT, "{tier:?}: residual {base_extent}");
        for depth in [0.0f32, 0.5, 2.0] {
            let (_, centre, outs) = render_tau(tier, false, depth, 20.0);
            assert!((centre - 6000.0).abs() < 2.0, "{tier:?} depth {depth}: centre {centre}");
            let y = tail_deviation(&outs);
            let extent = half_range(y.iter().copied());
            if depth == 0.0 {
                assert!(extent < 0.15 * base_extent, "{tier:?} depth 0: {extent} c left (residual {base_extent})");
            } else {
                let r = extent / base_extent;
                assert!((r / depth - 1.0).abs() < 0.15, "{tier:?} depth {depth}: {extent} c vs residual {base_extent}");
                let rho = correlation(&y, &base);
                assert!(rho > 0.95, "{tier:?} depth {depth}: correlation {rho} with the 100 % residual");
            }
        }
    }
}

#[test]
fn first_cycle_is_flattened_before_engagement() {
    // Causal limitation documented in R§8.11: at τ = 0 with Preserve on, the
    // first cycle is corrected flat until the tracker engages.
    let (_, _, outs) = render(QualityTier::Hq, true, 1.0);
    for o in &outs[hops(10.0)..hops(120.0)] {
        assert_eq!(o.frame.flags & frame_flags::VIBRATO, 0);
        assert!((o.frame.output - o.frame.target).abs() < 1e-3);
    }
    // The note decision holds C through first-cycle excursions that stay
    // inside the tolerance band: a 6015 ± 40 c vibrato peaks 5 c past the
    // C/C# midpoint, inside the default 8 c. (Excursions *past* the band are
    // switches by the OR rule in every tier, CL-2; that is what Ignore
    // Vibrato and the tolerance are for once the tracker has engaged.)
    for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
        let mut c = live(|s| {
            s.retune_ms = 0.0;
            s.tier = tier;
        });
        let outs = run(&mut c, hops(2500.0), |i| {
            est(6015.0 + EXTENT * (2.0 * std::f32::consts::PI * RATE * i as f32 * HOP_S).sin())
        });
        assert_eq!(target_changes(&outs), 0, "{tier:?}");
    }
}

#[test]
fn vibrato_scaling_applies_on_bypassed_notes() {
    let mut c = live(|s| {
        s.note_states = [vt_core::NoteState::Bypass; 12];
        s.vibrato_depth = 0.0;
        s.retune_ms = 0.0;
    });
    let n = hops(2500.0);
    let outs = run(&mut c, n, sung);
    let tail = &outs[n - hops(1000.0)..];
    // Not corrected (centre stays sharp) but the vibrato is removed.
    assert!((mean(tail.iter().map(|o| o.frame.output)) - CENTRE).abs() < 3.0);
    assert!(half_range(tail.iter().map(|o| o.frame.output)) < 0.1 * EXTENT);
    assert_ne!(tail[0].frame.flags & frame_flags::BYPASSED, 0);
}
