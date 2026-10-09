//! Note starts on real-singing patterns: phrase restarts after pre-onset
//! false voicing (ON-1), the output ramp at hard-tune onsets (ON-9) and
//! scoops into the first note (ON-5).

use super::*;
use vt_core::{EngineKind, LatencyMode, QualityTier};

/// The vocadito "Hard Tune" condition (τ 0, tolerance 5 c, depth 0).
fn hard_tune(s: &mut Settings) {
    s.retune_ms = 0.0;
    s.tolerance_cents = 5.0;
    s.tolerance_ms = 0.0;
    s.vibrato_depth = 0.0;
    s.humanize = 0.0;
    s.flex_tune = 0.0;
}

/// `pre` hops of a false voiced estimate at `pre_c` (breath, consonant
/// resonance or a gross first estimate) after silence, `gap` unvoiced
/// hops, then the real note at `note_c`; returns the outputs of the real
/// note.
fn pre_voiced_onset(tier: QualityTier, pre_c: f32, pre: usize, gap: usize, note_c: f32) -> Vec<ControlOutput> {
    let mut c = live(|s| {
        hard_tune(s);
        s.tier = tier;
    });
    run(&mut c, 120, |_| unvoiced());
    hold(&mut c, pre_c, pre);
    run(&mut c, gap, |_| unvoiced());
    hold(&mut c, note_c, 60)
}

/// ON-1: a note started on pre-onset false voicing must not survive the
/// short gap before the real note. It used to: the real note was treated as
/// a switch away from the stale note, held by the octave guard (p kept for
/// 20 ms) and then corrected with the stale note's error at the ±600 c
/// clamp until the HMM switched (vocadito, `onset_diag ctlunit`: target
/// 3900 for 21 hops, then −600 c for 6 hops).
#[test]
fn pre_onset_false_voicing_does_not_carry_its_note_into_the_real_note() {
    for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
        for (pre_c, pre, gap, note_c) in [
            (3854.0, 20, 3, 4920.0),  // 1 @ 0.668 s: 1066 c below, 4 ms gap
            (3854.0, 20, 15, 4920.0), // 20 ms gap
            (3854.0, 20, 30, 4920.0), // 40 ms gap (past the hold)
            (8960.0, 3, 8, 6398.0),   // 2 @ 9.474 s: an octave-up first estimate
            (5800.0, 30, 6, 5490.0),  // 310 c above, inside the octave guard's range
        ] {
            let outs = pre_voiced_onset(tier, pre_c, pre, gap, note_c);
            let want = 100.0 * (note_c / 100.0).round();
            // FSM onset confirmation (2 hops), then the onset decision.
            let first = outs.iter().position(|o| o.shift.voiced).unwrap_or(usize::MAX);
            assert!(first <= 2, "{tier:?} {pre_c}→{note_c}: voiced only at hop {first}");
            for (i, o) in outs.iter().enumerate().skip(first) {
                assert_eq!(o.frame.target, want, "{tier:?} {pre_c}→{note_c} gap {gap}: hop {i} target");
                assert!(o.shift_cents.abs() <= 50.0, "{tier:?} {pre_c}→{note_c} gap {gap}: hop {i} shift {}", o.shift_cents);
            }
        }
    }
}

/// ON-1 keeps the octave guard where it belongs: within an established
/// phrase a short octave error is still held, also right after a short
/// dropout.
#[test]
fn octave_spikes_inside_an_established_phrase_are_still_held() {
    for tier in [QualityTier::Eco, QualityTier::Hq] {
        let mut c = live(|s| {
            hard_tune(s);
            s.tier = tier;
        });
        hold(&mut c, 6010.0, hops(300.0));
        run(&mut c, 2, |_| unvoiced());
        let outs = hold(&mut c, 7210.0, hops(10.0));
        for o in outs.iter().filter(|o| o.shift.voiced) {
            assert_eq!(o.frame.target, 6000.0, "{tier:?}");
        }
        let outs = hold(&mut c, 6010.0, hops(50.0));
        assert_eq!(target_changes(&outs), 0, "{tier:?}");
    }
}

/// ON-9: with the Natural engine at a snapping (hard-tune) onset the full
/// shift applies from the first voiced hop; the PSOLA fades in from the dry
/// signal under its first grain itself. It used to ramp in over 5 ms
/// (0, 12, 25, 37, 46 c at 48 kHz), delaying the full shift by 3-4 ms.
/// Classic, and Live-Min (always Classic), keep the ramp.
#[test]
fn hard_tune_onsets_apply_the_full_shift_at_once_with_psola_only() {
    let first_shift = |mode: LatencyMode, engine: EngineKind| {
        let mut c = with_config(config(mode, 0), |s| {
            hard_tune(s);
            s.engine = engine;
        });
        run(&mut c, 50, |_| unvoiced());
        let outs = hold(&mut c, 6046.0, 10);
        let k = outs.iter().position(|o| o.shift.voiced).unwrap_or(usize::MAX);
        (outs[k].shift_cents, outs[k + 1].shift_cents)
    };
    let (s0, s1) = first_shift(LatencyMode::Live, EngineKind::Natural);
    assert!((s0 + 46.0).abs() < 1e-3 && (s1 + 46.0).abs() < 1e-3, "natural: {s0} {s1}");
    for (mode, engine) in [(LatencyMode::Live, EngineKind::Classic), (LatencyMode::LiveMin, EngineKind::Natural)] {
        let (s0, _) = first_shift(mode, engine);
        assert!(s0 > -40.0 && s0 < -1.0, "{mode:?} {engine:?}: first shift {s0} (ramp expected)");
    }
    // Re-voicing after a short gap (note kept): full shift at once too.
    let mut c = live(hard_tune);
    hold(&mut c, 6046.0, hops(100.0));
    run(&mut c, hops(10.0), |_| unvoiced());
    let outs = hold(&mut c, 6046.0, 5);
    let k = outs.iter().position(|o| o.shift.voiced).unwrap_or(usize::MAX);
    assert!((outs[k].shift_cents + 46.0).abs() < 1e-3, "re-voiced: {}", outs[k].shift_cents);
}

/// A sung scoop into a note after silence: exponential (τ 60 ms) from
/// `scoop` cents below the final pitch `6000 + end`.
fn scoop_into(end: f32, scoop: f32) -> impl Fn(usize) -> f32 {
    move |i| 6000.0 + end + scoop * (-(i as f32 * HOP_MS) / 60.0).exp()
}

/// ms from the moment the pitch passes the midpoint + tolerance (5900 + 50
/// + 8) until the target becomes C4, for a scoop that starts on B3.
fn scoop_switch_late_ms(tier: QualityTier, end: f32, scoop: f32) -> f32 {
    let mut c = live(|s| {
        s.retune_ms = 0.0;
        s.tier = tier;
    });
    run(&mut c, 50, |_| unvoiced());
    let p = scoop_into(end, scoop);
    let outs = run(&mut c, hops(400.0), |i| est(p(i)));
    assert_eq!(outs[hops(5.0)].frame.target, 5900.0, "{tier:?}: onset should take the scoop's note");
    let crossed = (0..outs.len()).find(|&i| p(i) > 5958.0).unwrap_or(usize::MAX);
    let switched = outs.iter().position(|o| o.frame.target == 6000.0).unwrap_or(usize::MAX);
    (switched as f32 - crossed as f32) * HOP_MS
}

/// ON-5: a scoop into the phrase's first note that ends only just past the
/// tolerance band (melody_male #0: −75 c scoop, the pitch 0-6 c past the
/// 8 c band) leaves the scoop note as soon as the HMM evidence allows. It
/// used to wait out the 25 ms near-band persistence first (CL-2), which is
/// meant for held notes drifting past the band, and settled in 31.6 ms.
#[test]
fn a_slow_scoop_into_the_first_note_switches_without_the_persistence() {
    for tier in [QualityTier::Hq, QualityTier::Max] {
        for (end, scoop) in [(-34.0, -50.0), (-36.0, -40.0), (-32.0, -60.0)] {
            let late = scoop_switch_late_ms(tier, end, scoop);
            assert!(late < 20.0, "{tier:?} end {end} scoop {scoop}: switched {late:.1} ms after the band");
        }
    }
}

/// ...while a held note that drifts just past the band later in the phrase
/// still needs the persistence (warble protection, CL-2).
#[test]
fn a_held_note_drifting_past_the_band_keeps_the_persistence() {
    for tier in [QualityTier::Hq, QualityTier::Max] {
        let mut c = live(|s| {
            s.retune_ms = 0.0;
            s.tier = tier;
        });
        hold(&mut c, 5990.0, hops(300.0));
        let outs = hold(&mut c, 6059.0, hops(200.0));
        let switched = outs.iter().position(|o| o.frame.target == 6100.0).unwrap_or(usize::MAX);
        assert!((switched as f32) * HOP_MS >= 24.0, "{tier:?}: switched after {} ms", switched as f32 * HOP_MS);
    }
}
