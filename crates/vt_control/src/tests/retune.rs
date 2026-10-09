//! Retune filter on the correction (R§7.6.2 H) and Humanize (I).

use super::*;
use vt_core::QualityTier;

/// One-pole step response after k hops: fraction of the step covered.
fn onepole_fraction(tau_ms: f32, k: usize) -> f32 {
    let a = 1.0 - (-HOP_MS / tau_ms).exp();
    1.0 - (1.0 - a).powi(k as i32)
}

#[test]
fn target_step_is_a_one_pole_with_63_percent_at_tau() {
    for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
        let mut s = Settings { retune_ms: 20.0, tier, ..Settings::default() };
        let mut c = live(|x| *x = s);
        // Settle 20 c sharp of C4: correction → −20 c.
        let outs = hold(&mut c, 6020.0, hops(400.0));
        assert!((outs[outs.len() - 1].shift_cents + 20.0).abs() < 0.05);
        // Detune +30 c moves the target instantly: e steps from −20 to +10.
        s.detune_cents = 30.0;
        c.set_settings(&s);
        let outs = hold(&mut c, 6020.0, hops(200.0));
        for (k, o) in outs.iter().enumerate().take(60) {
            let expect = -20.0 + 30.0 * onepole_fraction(20.0, k + 1);
            assert!((o.shift_cents - expect).abs() < 0.02, "{tier:?} hop {k}: {} vs {expect}", o.shift_cents);
        }
        // τ = 20 ms is exactly 15 hops at 48 kHz/64: 63.2 % there.
        let frac = (outs[14].shift_cents + 20.0) / 30.0;
        assert!((frac - 0.632).abs() < 0.005, "{tier:?}: {frac}");
    }
}

#[test]
fn input_step_within_a_note_decays_with_tau() {
    let mut c = live(|s| s.retune_ms = 30.0);
    hold(&mut c, 6010.0, hops(400.0));
    let outs = hold(&mut c, 6040.0, hops(200.0));
    // Output deviation from the target decays from +30 c like exp(−t/τ).
    let k = hops(30.0);
    let dev = outs[k - 1].frame.output - 6000.0;
    let expect = 30.0 * (1.0 - onepole_fraction(30.0, k));
    assert!((dev - expect).abs() < 0.1, "{dev} vs {expect}");
    assert!((dev / 30.0 - (-1.0f32).exp()).abs() < 0.02);
    assert!((outs[outs.len() - 1].frame.output - 6000.0).abs() < 0.06);
}

#[test]
fn tau_zero_hard_snaps() {
    let mut c = live(|s| s.retune_ms = 0.0);
    // Onset is snapped (no grace at τ = 0); after the 5 ms voiced ramp the
    // output sits exactly on the note whatever the input does.
    let outs = run(&mut c, hops(300.0), |i| est(6000.0 + 35.0 * ((i as f32) * 0.37).sin()));
    let after_ramp = hops(5.0) + 3;
    for o in &outs[after_ramp..] {
        assert!((o.frame.output - 6000.0).abs() < 1e-3, "output {}", o.frame.output);
    }
}

#[test]
fn residual_vibrato_matches_the_r_7_6_2_h_table() {
    // τ (ms) → residual of a 5.5 Hz vibrato with ref = p_s (Preserve off).
    let table = [(5.0, 0.17), (10.0, 0.33), (20.0, 0.57), (30.0, 0.72), (50.0, 0.87), (100.0, 0.96), (200.0, 0.99)];
    for (tau, expect) in table {
        let mut c = live(|s| {
            s.retune_ms = tau;
            s.preserve_vibrato = false;
        });
        let n = hops(3000.0);
        let outs = run(&mut c, n, |i| est(6000.0 + 30.0 * (2.0 * std::f32::consts::PI * 5.5 * i as f32 * HOP_S).sin()));
        let tail = &outs[n - hops(1000.0)..];
        let residual = half_range(tail.iter().map(|o| o.frame.output)) / 30.0;
        assert!((residual - expect).abs() < 0.03, "τ {tau}: residual {residual}, table {expect}");
    }
}

#[test]
fn humanize_slows_only_sustained_notes() {
    // Fraction of a +20 c detune step covered 20 ms after the step.
    fn covered(humanize: f32, note_ms: f32) -> f32 {
        let mut s = Settings { retune_ms: 10.0, humanize, ..Settings::default() };
        let mut c = live(|x| *x = s);
        run(&mut c, 5, |_| unvoiced());
        hold(&mut c, 6000.0, hops(note_ms));
        let before = c.process_hop(&est(6000.0), None).shift_cents;
        s.detune_cents = 20.0;
        c.set_settings(&s);
        let outs = hold(&mut c, 6000.0, hops(20.0));
        (outs[outs.len() - 1].shift_cents - before) / 20.0
    }
    let fast = covered(0.0, 1000.0);
    assert!((fast - onepole_fraction(10.0, hops(20.0))).abs() < 0.02, "fast {fast}");
    // Long sustained note at Humanize 100: τ_eff ≈ 10 + 300 ms.
    let slow = covered(100.0, 1000.0);
    assert!(slow < 0.1, "sustained note still fast: {slow}");
    // Humanize 50 adds 75 ms.
    let mid = covered(50.0, 1000.0);
    assert!((mid - onepole_fraction(85.0, hops(20.0))).abs() < 0.03, "H50 {mid}");
    // A short note (100 ms, before the 150 ms sustain gate) keeps the fast τ.
    let short = covered(100.0, 100.0);
    assert!((short - fast).abs() < 0.02, "short note slowed: {short} vs {fast}");
}

#[test]
fn transition_time_constant_applies_after_a_note_change() {
    let mut c = live(|s| {
        s.retune_ms = 0.0;
        s.transition_ms = Some(60.0);
        s.tier = QualityTier::Eco;
        // C major: no C# between C and D, so the jump is a single switch.
        s.note_states = vt_core::ScaleKind::Major.note_states(0);
    });
    hold(&mut c, 6000.0, hops(200.0));
    // Jump a whole tone (sung 10 c flat): the correction glides with τ_tr.
    let outs = hold(&mut c, 6190.0, hops(400.0));
    let first = outs.iter().position(|o| o.frame.target == 6200.0).unwrap_or(usize::MAX);
    assert!(first <= 3, "switch at {first}");
    assert_ne!(outs[first].frame.flags & vt_core::frame_flags::TRANSITION, 0);
    // e jumps from −190 to +10; after τ_tr the correction covers 63 %.
    let k = first + hops(60.0) - 1;
    let frac = (outs[k].shift_cents - (-190.0)) / 200.0;
    assert!((frac - 0.632).abs() < 0.03, "frac {frac}");
    // The window closes once within 5 c; then τ = 0 snaps the rest.
    let last = outs[outs.len() - 1];
    assert_eq!(last.frame.flags & vt_core::frame_flags::TRANSITION, 0);
    assert!((last.frame.output - 6200.0).abs() < 1e-3);
}
