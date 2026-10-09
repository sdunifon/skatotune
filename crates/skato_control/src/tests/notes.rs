//! Note decision, candidates, error shaping (R§7.6.2 E-G).

use super::*;
use skato_core::{NoteState, QualityTier, ScaleKind, frame_flags};

const TIERS: [QualityTier; 3] = [QualityTier::Eco, QualityTier::Hq, QualityTier::Max];

fn hover(rate_hz: f32) -> impl FnMut(usize) -> PitchEstimate {
    move |i| est(6050.0 + 5.0 * (2.0 * std::f32::consts::PI * rate_hz * i as f32 * HOP_S).sin())
}

#[test]
fn hysteresis_prevents_flip_flop_around_a_midpoint() {
    for tier in TIERS {
        for rate in [0.5, 2.0, 7.0] {
            let mut c = live(|s| {
                s.tier = tier;
                s.retune_ms = 0.0;
            });
            hold(&mut c, 6000.0, hops(200.0));
            let outs = run(&mut c, hops(4000.0), hover(rate));
            assert_eq!(target_changes(&outs), 0, "{tier:?} at {rate} Hz flip-flopped");
            assert_eq!(outs[0].frame.target, 6000.0);
        }
    }
}

#[test]
fn without_tolerance_the_rules_do_flip() {
    // Sanity check that the hover test above is meaningful.
    let mut c = live(|s| {
        s.tier = QualityTier::Eco;
        s.tolerance_cents = 0.0;
        s.tolerance_ms = 0.0;
    });
    hold(&mut c, 6000.0, hops(200.0));
    let outs = run(&mut c, hops(4000.0), hover(2.0));
    assert!(target_changes(&outs) >= 4);
}

#[test]
fn real_note_changes_switch_quickly() {
    for tier in TIERS {
        let mut c = live(|s| {
            s.tier = tier;
            s.retune_ms = 0.0;
        });
        hold(&mut c, 6000.0, hops(300.0));
        let outs = hold(&mut c, 6105.0, hops(200.0));
        let k = outs.iter().position(|o| o.frame.target == 6100.0).unwrap_or(usize::MAX);
        assert!((k as f32) * HOP_MS < 15.0, "{tier:?}: switched after {} ms", k as f32 * HOP_MS);
        assert!((outs[outs.len() - 1].frame.output - 6100.0).abs() < 1e-3);
        assert_eq!(target_changes(&outs), 1, "{tier:?}");
    }
}

/// Legato step from C4 to `6050 + x` (x cents past the C/C# midpoint) under
/// the given tolerance; ms from the step until the target becomes C#, or
/// None if it never does within 1 s.
fn or_switch_ms(tier: QualityTier, tol_c: f32, tol_ms: f32, x: f32) -> Option<f32> {
    let mut c = live(|s| {
        s.tier = tier;
        s.retune_ms = 0.0;
        s.tolerance_cents = tol_c;
        s.tolerance_ms = tol_ms;
    });
    hold(&mut c, 6000.0, hops(300.0));
    let outs = hold(&mut c, 6050.0 + x, hops(1000.0));
    outs.iter().position(|o| o.frame.target == 6100.0).map(|k| k as f32 * HOP_MS)
}

/// CL-2: every tier follows Waves' OR-tolerance (R§7.6.2 F): no switch
/// before x > T_c (cents part) or x > 0 held for T_t (time part). Eco applies
/// the rule directly; HQ/Max use it as a gate on the note HMM, whose evidence
/// decides when within it (R§8.12 "map ... calibrate"), so a pitch held
/// just past the band switches within ~40 ms. HQ/Max used to let the time
/// term switch pitches still inside the cents band (35 c inside a 40 c band
/// after 13 ms, not 100 ms; 6 c inside 15 c / 30 ms after 13 ms) and held a
/// pitch 1 c past the band for 0.57 s.
#[test]
fn every_tier_follows_the_or_tolerance() {
    type Range = Option<(f32, f32)>;
    // (T_c, T_t, x past the midpoint, Eco ms, HQ/Max ms); None = never.
    let cases: [(f32, f32, f32, Range, Range); 10] = [
        (5.0, 0.0, 6.0, Some((0.0, 25.0)), Some((0.0, 45.0))), // Hard Tune: 1 c past the band
        (8.0, 0.0, 9.0, Some((0.0, 25.0)), Some((0.0, 45.0))), // default: 1 c past
        (8.0, 0.0, 12.0, Some((0.0, 15.0)), Some((0.0, 30.0))), // default: 4 c past
        (8.0, 0.0, 5.0, None, None),                           // inside the band, Time off
        (15.0, 30.0, 6.0, Some((28.0, 45.0)), Some((28.0, 45.0))), // Natural: inside → T_t
        (15.0, 30.0, 25.0, Some((0.0, 20.0)), Some((0.0, 20.0))), // Natural: past the band
        (40.0, 100.0, 35.0, Some((95.0, 115.0)), Some((95.0, 115.0))),
        (40.0, 100.0, 1.0, Some((95.0, 120.0)), Some((95.0, 160.0))),
        (0.0, 30.0, 20.0, Some((28.0, 45.0)), Some((28.0, 45.0))), // Cents off: T_t only
        (0.0, 0.0, 3.0, Some((0.0, 30.0)), Some((0.0, 40.0))),     // both off: past the midpoint
    ];
    for tier in TIERS {
        for (tc, tt, x, eco, hq) in cases {
            let want = if tier == QualityTier::Eco { eco } else { hq };
            let got = or_switch_ms(tier, tc, tt, x);
            match (want, got) {
                (None, None) => {}
                (Some((lo, hi)), Some(ms)) if ms >= lo && ms <= hi => {}
                _ => panic!("{tier:?} {tc} c / {tt} ms, {x} c past the midpoint: switched {got:?} ms, want {want:?}"),
            }
        }
    }
}

/// HQ/Max: the time tolerance's dwell (the OR-rule gate) runs on the
/// unsmoothed pitch with a 5 ms floor, so short detector glitches inside the
/// cents band (where only the time part can open the gate) never switch,
/// however small the Tolerance time and however fast the evidence.
#[test]
fn hq_time_tolerance_ignores_short_glitches() {
    for tier in [QualityTier::Hq, QualityTier::Max] {
        for glitch in [55.0, 70.0, 90.0] {
            for len in [1usize, 2, 3] {
                let mut c = live(|s| {
                    s.tier = tier;
                    s.retune_ms = 0.0;
                    s.tolerance_cents = 45.0;
                    s.tolerance_ms = 1.0;
                });
                hold(&mut c, 6000.0, hops(300.0));
                let outs = run(&mut c, 200, |i| est(if i < len { 6000.0 + glitch } else { 6000.0 }));
                assert_eq!(target_changes(&outs), 0, "{tier:?}: +{glitch} c for {len} hops");
            }
        }
    }
}

/// A scoop that starts nearer the lower neighbour commits to it at the onset
/// (nearest note), but must switch promptly once the rising pitch is past the
/// midpoint plus the tolerance, also while the note is still in its attack.
#[test]
fn scoop_from_the_lower_neighbour_switches_past_the_tolerance() {
    for tier in TIERS {
        let mut c = live(|s| {
            s.tier = tier;
            s.retune_ms = 0.0;
        });
        // B3 + 20 c rising at 0.5 c/ms towards C4, then held 10 c flat of C4.
        let p = |i: usize| (5920.0 + 0.5 * i as f32 * HOP_MS).min(5990.0);
        let outs = run(&mut c, hops(400.0), |i| est(p(i)));
        assert_eq!(outs[hops(10.0)].frame.target, 5900.0, "{tier:?}: onset should take the nearest note");
        // Midpoint 5950 + 8 c default tolerance; allow 5 c of evidence.
        let crossed = (0..outs.len()).find(|&i| p(i) > 5963.0).unwrap_or(usize::MAX);
        let switched = outs.iter().position(|o| o.frame.target == 6000.0).unwrap_or(usize::MAX);
        let late_ms = (switched as f32 - crossed as f32) * HOP_MS;
        assert!(late_ms < 15.0, "{tier:?}: switched {late_ms:.1} ms after passing midpoint + tolerance");
        assert_eq!(target_changes(&outs), 1, "{tier:?}");
    }
}

#[test]
fn bypass_and_remove_semantics() {
    for tier in TIERS {
        let mut states = ScaleKind::Major.note_states(0);
        states[4] = NoteState::Bypass; // E
        let mut c = live(|s| {
            s.tier = tier;
            s.retune_ms = 0.0;
            s.note_states = states;
        });
        // E + 20 c: E is legal but bypassed, so the pitch passes uncorrected
        // (and is not pulled to F or D either).
        let outs = hold(&mut c, 6420.0, hops(300.0));
        let last = outs[outs.len() - 1];
        assert!(last.shift_cents.abs() < 1e-3, "{tier:?}: bypass corrected by {}", last.shift_cents);
        assert_eq!(last.frame.target, 6400.0);
        assert_ne!(last.frame.flags & frame_flags::BYPASSED, 0);
        // C# is Remove in C major: C# + 10 c goes to D (nearest legal).
        let mut c = live(|s| {
            s.tier = tier;
            s.retune_ms = 0.0;
            s.note_states = states;
        });
        let outs = hold(&mut c, 6110.0, hops(300.0));
        assert!((outs[outs.len() - 1].frame.output - 6200.0).abs() < 1e-3, "{tier:?}");
        // C# − 10 c goes to C.
        let mut c = live(|s| {
            s.tier = tier;
            s.retune_ms = 0.0;
            s.note_states = states;
        });
        let outs = hold(&mut c, 6090.0, hops(300.0));
        assert!((outs[outs.len() - 1].frame.output - 6000.0).abs() < 1e-3, "{tier:?}");
    }
}

#[test]
fn all_remove_falls_back_to_chromatic() {
    let mut c = live(|s| {
        s.note_states = [NoteState::Remove; 12];
        s.retune_ms = 0.0;
    });
    let outs = hold(&mut c, 6130.0, hops(200.0));
    assert!((outs[outs.len() - 1].frame.output - 6100.0).abs() < 1e-3);
}

#[test]
fn flex_tune_corrects_only_near_the_target() {
    for (sung, corrected) in [(6010.0, true), (6018.0, false), (6030.0, false), (5985.0, true)] {
        let mut c = live(|s| {
            s.flex_tune = 50.0; // chromatic: R = 25 c, full weight inside 15 c
            s.retune_ms = 0.0;
        });
        let outs = hold(&mut c, sung, hops(200.0));
        let out = outs[outs.len() - 1].frame.output;
        if corrected {
            assert!((out - 6000.0).abs() < 1e-3, "{sung}: {out}");
        } else {
            // 18 c is in the taper (partial), 30 c is outside (untouched).
            assert!((out - 6000.0).abs() > 1.0, "{sung}: {out}");
        }
    }
    let mut c = live(|s| {
        s.flex_tune = 50.0;
        s.retune_ms = 0.0;
    });
    let outs = hold(&mut c, 6030.0, hops(200.0));
    assert!((outs[outs.len() - 1].frame.output - 6030.0).abs() < 1e-3);
}

#[test]
fn amount_snap_min_and_snap_max() {
    let mut c = live(|s| {
        s.amount = 0.5;
        s.retune_ms = 0.0;
    });
    let outs = hold(&mut c, 6030.0, hops(200.0));
    assert!((outs[outs.len() - 1].frame.output - 6015.0).abs() < 1e-2);
    // Snap Min 20 c: a 10 c error keeps 80 % of itself (x² / (1 + x²) = 0.2).
    let mut c = live(|s| {
        s.snap_min_cents = 20.0;
        s.retune_ms = 0.0;
    });
    let outs = hold(&mut c, 6010.0, hops(200.0));
    assert!((outs[outs.len() - 1].frame.output - 6008.0).abs() < 1e-2);
    // Snap Max 0.3 st: 40 c off is too far to correct.
    let mut c = live(|s| {
        s.snap_max_st = 0.3;
        s.retune_ms = 0.0;
        s.tolerance_cents = 0.0;
    });
    let outs = hold(&mut c, 6040.0, hops(200.0));
    assert!((outs[outs.len() - 1].frame.output - 6040.0).abs() < 1e-3);
}

#[test]
fn onset_grace_delays_the_decision_unless_retune_is_fast() {
    let mut c = live(|s| s.retune_ms = 20.0);
    let outs = hold(&mut c, 6030.0, hops(100.0));
    // Two hops of onset confirmation, then 20 ms of grace with no target.
    let first = outs.iter().position(|o| o.frame.target != 0.0).unwrap_or(usize::MAX);
    assert!((first as f32 - (1.0 + hops(20.0) as f32)).abs() <= 1.0, "first target at hop {first}");
    let mut c = live(|s| s.retune_ms = 3.0);
    let outs = hold(&mut c, 6030.0, hops(100.0));
    let first = outs.iter().position(|o| o.frame.target != 0.0).unwrap_or(usize::MAX);
    assert_eq!(first, 1);
}

#[test]
fn detune_shifts_every_target() {
    let mut c = live(|s| {
        s.detune_cents = 50.0;
        s.retune_ms = 0.0;
    });
    let outs = hold(&mut c, 6070.0, hops(200.0));
    let last = outs[outs.len() - 1].frame;
    assert_eq!(last.target, 6050.0);
    assert!((last.output - 6050.0).abs() < 1e-3);
}

#[test]
fn hmm_ignores_short_detector_glitches() {
    // HQ/Max integrate evidence with a per-hop cap: glitches of up to two
    // hops (2.7 ms), anywhere below the 700 c octave guard, never flip the note.
    for tier in [QualityTier::Hq, QualityTier::Max] {
        for glitch in [150.0, 300.0, 600.0, 690.0] {
            for len in [1, 2] {
                let mut c = live(|s| {
                    s.tier = tier;
                    s.retune_ms = 0.0;
                });
                hold(&mut c, 6000.0, hops(300.0));
                let outs = run(&mut c, 200, |i| est(if i < len { 6000.0 + glitch } else { 6000.0 }));
                assert_eq!(target_changes(&outs), 0, "{tier:?}: +{glitch} c for {len} hops");
            }
        }
    }
}
