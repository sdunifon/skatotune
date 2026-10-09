//! Voicing FSM, unvoiced ramp, octave guard, output mapping (R§7.6.2 A, B, L).

use super::*;
use vt_core::{QualityTier, frame_flags};

fn last(outs: &[ControlOutput]) -> ControlOutput {
    outs[outs.len() - 1]
}

#[test]
fn transpose_formant_and_ratio_mapping() {
    let mut c = live(|s| {
        s.retune_ms = 0.0;
        s.transpose_st = 12.0;
        s.formant_st = -3.0;
        s.formant_follows_transpose = true;
    });
    let outs = hold(&mut c, 6020.0, hops(200.0));
    let o = last(&outs);
    // S = correction (−20) + 1200; r = 2^(S/1200).
    assert!((o.shift_cents - 1180.0).abs() < 1e-2);
    assert!((o.shift.ratio - (1180.0f32 / 1200.0).exp2()).abs() < 1e-5);
    assert!((o.frame.output - 7200.0).abs() < 1e-2);
    // γ = 2^(formant/12)·2^(transpose/12) when following.
    assert!((o.shift.formant_ratio - (9.0f32 / 12.0).exp2()).abs() < 1e-4);
    assert!(o.shift.voiced);

    // Unvoiced: S → 100·Transpose over 5 ms, ShiftControl.voiced = false.
    let outs = run(&mut c, hops(20.0), |_| unvoiced());
    assert!(!outs[0].shift.voiced);
    let ramp = hops(5.0);
    assert!(outs[0].shift_cents > 1180.0 && outs[0].shift_cents < 1200.0);
    for o in &outs[ramp..] {
        assert!((o.shift_cents - 1200.0).abs() < 1e-3);
        assert_eq!(o.frame.detected, 0.0);
        assert_eq!(o.frame.flags & frame_flags::VOICED, 0);
    }
    // Without follow the formant ratio ignores the transpose.
    let mut c = live(|s| {
        s.transpose_st = 12.0;
        s.formant_st = -3.0;
    });
    let o = c.process_hop(&est(6000.0), None);
    assert!((o.shift.formant_ratio - (-3.0f32 / 12.0).exp2()).abs() < 1e-4);
}

#[test]
fn transpose_changes_glide_without_steps() {
    let mut s = Settings { retune_ms: 0.0, ..Settings::default() };
    let mut c = live(|x| *x = s);
    hold(&mut c, 6000.0, hops(100.0));
    s.transpose_st = 7.0;
    c.set_settings(&s);
    let outs = hold(&mut c, 6000.0, hops(100.0));
    // 10 ms parameter smoothing: no single-hop jump larger than ~15 %.
    let max_step = outs.windows(2).map(|w| (w[1].shift_cents - w[0].shift_cents).abs()).fold(0.0f32, f32::max);
    assert!(max_step < 0.15 * 700.0, "step {max_step}");
    assert!((last(&outs).shift_cents - 700.0).abs() < 0.1);
}

/// Settle on C4 + 20 c with slow retune, then a gap of `gap_ms`, then sing
/// again; returns the shift right after re-voicing (after the 5 ms ramp).
fn shift_after_gap(gap_ms: f32) -> (f32, ControlOutput) {
    let mut c = live(|s| s.retune_ms = 400.0);
    hold(&mut c, 6020.0, hops(4000.0)); // c ≈ −20 after 10 τ
    run(&mut c, hops(gap_ms), |_| unvoiced());
    let outs = hold(&mut c, 6020.0, 1 + hops(5.0) + 1);
    (last(&outs).shift_cents, last(&outs))
}

#[test]
fn unvoiced_hold_release_and_forget() {
    // Within the 30 ms hold the correction is kept.
    let (s, _) = shift_after_gap(25.0);
    assert!((s + 20.0).abs() < 0.3, "hold: {s}");
    // After the hold it releases toward 0 with τ_rel = 40 ms.
    let (s, o) = shift_after_gap(70.0);
    let expect = -20.0 * (-(70.0 - 30.0) / 40.0f32).exp();
    assert!((s - expect).abs() < 2.0, "release: {s} vs {expect}");
    assert_ne!(o.frame.target, 0.0, "note still remembered before 100 ms");
    // After 100 ms the note is forgotten: the onset grace (20 ms) runs again
    // with no target.
    let (s, o) = shift_after_gap(150.0);
    assert!(s.abs() < 1.0, "forget: {s}");
    assert_eq!(o.frame.target, 0.0);
}

#[test]
fn forget_timing_is_100ms() {
    let mut c = live(|s| s.retune_ms = 0.0);
    hold(&mut c, 6000.0, hops(200.0));
    let outs = run(&mut c, hops(150.0), |_| unvoiced());
    let forgotten = outs.iter().position(|o| o.frame.target == 0.0).unwrap_or(usize::MAX);
    let ms = forgotten as f32 * HOP_MS;
    assert!((ms - 100.0).abs() <= 2.0 * HOP_MS, "forgot after {ms} ms");
}

#[test]
fn single_unvoiced_blip_needs_two_hops_to_revoice() {
    let mut c = live(|_| {});
    hold(&mut c, 6000.0, hops(100.0));
    let o = c.process_hop(&unvoiced(), None);
    assert!(!o.shift.voiced);
    // One voiced hop is not an onset when coming from silence...
    let mut c = live(|_| {});
    let o = c.process_hop(&est(6000.0), None);
    assert!(!o.shift.voiced);
    let o = c.process_hop(&est(6000.0), None);
    assert!(o.shift.voiced);
    // ...and sibilant frames count as unvoiced.
    let o = c.process_hop(&PitchEstimate { sibilant: true, ..est(6000.0) }, None);
    assert!(!o.shift.voiced);
    assert_ne!(o.frame.flags & frame_flags::SIBILANT, 0);
}

#[test]
fn octave_guard_holds_until_20ms_of_consistent_estimates() {
    for tier in [QualityTier::Eco, QualityTier::Hq] {
        let mut c = live(|s| {
            s.retune_ms = 20.0;
            s.tier = tier;
        });
        let settled = last(&hold(&mut c, 6010.0, hops(300.0)));
        // A 10 ms octave error: the shift and the target do not move.
        let outs = hold(&mut c, 7210.0, hops(10.0));
        for o in &outs {
            assert!((o.shift_cents - settled.shift_cents).abs() < 0.01, "{tier:?}: shift moved {}", o.shift_cents);
            assert_eq!(o.frame.target, 6000.0);
        }
        let outs = hold(&mut c, 6010.0, hops(100.0));
        assert_eq!(target_changes(&outs), 0);
        // A real octave leap is accepted after 20 ms.
        let outs = hold(&mut c, 7210.0, hops(300.0));
        let k = outs.iter().position(|o| o.frame.target == 7200.0).unwrap_or(usize::MAX);
        let ms = k as f32 * HOP_MS;
        assert!((18.0..40.0).contains(&ms), "{tier:?}: accepted after {ms} ms");
        assert!((last(&outs).frame.output - 7200.0).abs() < 0.05);
    }
}

/// SN-2: the correction follows the detector, not the PSOLA mark pitch.
/// The mark pitch (a single-cycle period) carries each new cycle's jitter
/// into the ratio; it used to replace the detector whenever the two agreed
/// within 50 c, which doubled the hard-tune residual on real singing.
#[test]
fn mark_pitch_never_replaces_the_detector() {
    for mode in [vt_core::LatencyMode::Live, vt_core::LatencyMode::LiveMin, vt_core::LatencyMode::LivePlus] {
        let mut c = with_config(config(mode, 0), |s| s.retune_ms = 0.0);
        // Detector says +20 c, marks wander around +30 c (agreeing within 50 c).
        let outs: Vec<_> =
            (0..hops(100.0)).map(|i| c.process_hop(&est(6020.0), Some(6030.0 + (i % 3) as f32 * 4.0))).collect();
        // (Past the onset confirmation and the Classic engine's 5 ms ramp.)
        for o in &outs[8..] {
            assert!((o.frame.detected - 6020.0).abs() < 1e-3, "{mode:?}: detected {}", o.frame.detected);
            assert!((o.shift_cents + 20.0).abs() < 1e-3, "{mode:?}: shift {}", o.shift_cents);
        }
    }
}

#[test]
fn time_constants_are_in_ms_at_every_rate() {
    // Same behaviour in milliseconds at 44.1, 48, 96 and 192 kHz.
    for (sr, hop) in [(44_100.0f32, 64usize), (48_000.0, 64), (96_000.0, 128), (192_000.0, 256)] {
        let hop_ms = hop as f32 / sr * 1000.0;
        let cfg = CorrectorConfig { sample_rate: sr, hop, latency_mode: vt_core::LatencyMode::Live, lookahead_hops: 0 };
        let mut s = Settings { retune_ms: 30.0, ..Settings::default() };
        let mut c = with_config(cfg, |x| *x = s);
        let n = (400.0 / hop_ms) as usize;
        run(&mut c, n, |_| est(6020.0));
        s.detune_cents = 30.0;
        c.set_settings(&s);
        let outs = run(&mut c, n, |_| est(6020.0));
        // Time to 63 %: 30 ms.
        let k = outs.iter().position(|o| (o.shift_cents + 20.0) / 30.0 >= 0.632).unwrap_or(usize::MAX);
        let ms = (k + 1) as f32 * hop_ms;
        assert!((ms - 30.0).abs() <= hop_ms, "{sr}: 63 % after {ms} ms");
        // Forget after 100 ms.
        let outs = run(&mut c, n, |_| unvoiced());
        let k = outs.iter().position(|o| o.frame.target == 0.0).unwrap_or(usize::MAX);
        let ms = k as f32 * hop_ms;
        assert!((ms - 100.0).abs() <= 2.0 * hop_ms, "{sr}: forgot after {ms} ms");
    }
}

/// Live Max: the detector's `cents` are extrapolated to the newest sample
/// (R§8.10 lag compensation) but the correction applies to the input at
/// `sample_pos`, whose pitch is `f0_hz`. The corrector used to take the
/// extrapolated `cents` (masked by the mark pitch until SN-2), which added
/// the slope estimate's noise and over-shot the audio being corrected
/// (vocadito Live Max hard tune: warble +38 %, p95 30.2 c vs 27.2 c).
#[test]
fn live_max_corrects_the_pitch_at_sample_pos_not_the_extrapolation() {
    let extrapolated = |c: f32| PitchEstimate { cents: c + 20.0, ..est(c) };
    for (mode, tier, follows_f0) in [
        (vt_core::LatencyMode::Live, QualityTier::Max, true),
        (vt_core::LatencyMode::LivePlus, QualityTier::Max, true),
        (vt_core::LatencyMode::Live, QualityTier::Hq, false),
        (vt_core::LatencyMode::Studio, QualityTier::Max, false),
    ] {
        let mut c = with_config(config(mode, 0), |s| {
            s.retune_ms = 0.0;
            s.tier = tier;
        });
        let o = last(&run(&mut c, hops(100.0), |_| extrapolated(6010.0)));
        let p = if follows_f0 { 6010.0 } else { 6030.0 };
        assert!((o.frame.detected - p).abs() < 0.01, "{mode:?} {tier:?}: detected {}", o.frame.detected);
        assert!((o.shift_cents + (p - 6000.0)).abs() < 0.01, "{mode:?} {tier:?}: shift {}", o.shift_cents);
    }
}
