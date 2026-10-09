//! Studio: lookahead derived from `info_lag`, fixed-lag decisions, onset
//! median, MIDI alignment (R§7.6.2 F, R§7.6.3 "Timing").

use super::*;
use vt_core::{LatencyMode, MidiEvent, QualityTier};

/// 40 ms of latency at 48 kHz / 64 = 30 hops.
const L: usize = 30;

fn studio(edit: impl FnOnce(&mut Settings)) -> Corrector {
    with_config(config(LatencyMode::Studio, L), edit)
}

/// An estimate whose evidence is `lag_hops` hops old (detector delay).
fn est_lag(cents: f32, lag_hops: usize) -> PitchEstimate {
    PitchEstimate { info_lag: (lag_hops * HOP) as f32, ..est(cents) }
}

fn silence_lag(lag_hops: usize) -> PitchEstimate {
    PitchEstimate { info_lag: (lag_hops * HOP) as f32, ..unvoiced() }
}

#[test]
fn detector_centred_on_the_output_means_no_extra_delay() {
    // DESIGN §3.5 literally: Centered { delay: D } → info_lag = D → lead 0.
    let mut c = studio(|s| s.retune_ms = 0.0);
    run(&mut c, 40, |_| silence_lag(L));
    let outs = run(&mut c, 200, |i| est_lag(if i < 100 { 6000.0 } else { 6200.0 }, L));
    assert_eq!(outs[100].frame.detected, 6200.0);
    assert_eq!(outs[99].frame.detected, 6000.0);
    assert_eq!(c.lookahead_lead(), 0);
}

#[test]
fn estimates_that_lead_the_audio_are_delayed_to_line_up() {
    // Detector centred 10 hops behind the newest sample: 20 hops of lookahead.
    let mut c = studio(|s| s.retune_ms = 0.0);
    run(&mut c, 40, |_| silence_lag(10));
    let outs = run(&mut c, 300, |i| est_lag(if i < 100 { 6000.0 } else { 6200.0 }, 10));
    assert_eq!(outs[119].frame.detected, 6000.0);
    assert_eq!(outs[120].frame.detected, 6200.0);
    assert_eq!(c.lookahead_lead(), 20);
}

#[test]
fn fixed_lag_switches_line_up_with_the_pitch_change() {
    for tier in [QualityTier::Eco, QualityTier::Hq, QualityTier::Max] {
        // Live: the switch comes a few hops after the pitch crosses.
        let mut c = live(|s| {
            s.retune_ms = 0.0;
            s.tier = tier;
        });
        let outs = run(&mut c, 300, |i| est(if i < 150 { 6000.0 } else { 6100.0 }));
        let live_switch = outs.iter().position(|o| o.frame.target == 6100.0).unwrap_or(usize::MAX);
        assert!(live_switch > 150, "{tier:?}");

        // Studio with 20 hops of lookahead: the output frame for the first
        // hop at the new pitch already has the new target, so the (delayed)
        // audio is corrected from the right sample with no dwell.
        let mut c = studio(|s| {
            s.retune_ms = 0.0;
            s.tier = tier;
        });
        run(&mut c, 40, |_| silence_lag(10));
        let outs = run(&mut c, 400, |i| est_lag(if i < 150 { 6000.0 } else { 6100.0 }, 10));
        let step = outs.iter().position(|o| o.frame.detected == 6100.0).unwrap_or(usize::MAX);
        let switch = outs.iter().position(|o| o.frame.target == 6100.0).unwrap_or(usize::MAX);
        assert_eq!(step, 170, "{tier:?}");
        assert!(switch <= step + 1, "{tier:?}: switch {switch} vs pitch step {step} (live lag {})", live_switch - 150);
        assert!(switch + 4 >= step, "{tier:?}: switched too early ({switch} vs {step})");
        assert_eq!(target_changes(&outs), 1, "{tier:?}");
        assert!((outs[outs.len() - 1].frame.output - 6100.0).abs() < 1e-3);
    }
}

/// Studio HQ/Max judge the OR-tolerance on the lookahead (CL-2, R§7.6.2 F
/// "Studio"): the time part needs only the dwell the lookahead cannot see
/// (here T_t − 20 hops), and a pitch held just past the cents band switches
/// as promptly as in Live (it used to take 0.57 s).
#[test]
fn studio_hq_or_tolerance_uses_the_lookahead() {
    for tier in [QualityTier::Hq, QualityTier::Max] {
        let switch_after_step = |tol_c: f32, tol_ms: f32, x: f32| {
            let mut c = studio(|s| {
                s.retune_ms = 0.0;
                s.tier = tier;
                s.tolerance_cents = tol_c;
                s.tolerance_ms = tol_ms;
            });
            run(&mut c, 40, |_| silence_lag(10));
            let outs = run(&mut c, 600, |i| est_lag(if i < 150 { 6000.0 } else { 6050.0 + x }, 10));
            let step = outs.iter().position(|o| o.frame.detected > 6040.0).unwrap_or(usize::MAX);
            let switch = outs.iter().position(|o| o.frame.target == 6100.0).unwrap_or(usize::MAX);
            assert_eq!(step, 170, "{tier:?}");
            (switch as f32 - step as f32) * HOP_MS
        };
        // Default 8 c / 0 ms, 1 c past the band.
        let ms = switch_after_step(8.0, 0.0, 9.0);
        assert!((0.0..=45.0).contains(&ms), "{tier:?}: near-band switch {ms} ms after the step");
        // 40 c / 100 ms, inside the band: 100 ms minus the 26.7 ms lookahead.
        let ms = switch_after_step(40.0, 100.0, 35.0);
        assert!((65.0..=80.0).contains(&ms), "{tier:?}: dwell switch {ms} ms after the step");
    }
}

/// With a transition time, Studio switches τ_tr/2 before the crossing so the
/// glide is centred on it (R§7.6.3 "Timing" applied to pitch-driven
/// transitions); at τ_tr = 0 it lands on the crossing.
#[test]
fn studio_transitions_are_centred_on_the_crossing() {
    for tier in [QualityTier::Hq, QualityTier::Max] {
        let mut c = studio(|s| {
            s.retune_ms = 20.0;
            s.tier = tier;
        });
        run(&mut c, 40, |_| silence_lag(10));
        let outs = run(&mut c, 400, |i| est_lag(if i < 150 { 6000.0 } else { 6100.0 }, 10));
        let step = outs.iter().position(|o| o.frame.detected == 6100.0).unwrap_or(usize::MAX);
        let switch = outs.iter().position(|o| o.frame.target == 6100.0).unwrap_or(usize::MAX);
        // τ_tr = 20 ms = 15 hops: the switch comes 8 hops (τ_tr/2) early.
        assert_eq!(step, 170, "{tier:?}");
        assert!(switch + 9 >= step && switch + 7 <= step, "{tier:?}: switch {switch}, crossing {step}");
        assert_eq!(target_changes(&outs), 1, "{tier:?}");
    }
}

/// A 10 ms scoop from 150 c flat into C4 at the onset.
fn scoop(i: usize) -> f32 {
    let t = i as f32 * HOP_MS;
    if t < 10.0 { 5850.0 + 15.0 * t } else { 6000.0 }
}

#[test]
fn studio_onset_note_comes_from_the_lookahead_median() {
    for tier in [QualityTier::Eco, QualityTier::Hq] {
        // Live at τ = 0 decides on the scoop and then corrects.
        let mut c = live(|s| {
            s.retune_ms = 0.0;
            s.tier = tier;
        });
        let outs = run(&mut c, 200, |i| est(scoop(i)));
        assert!(target_changes(&outs) >= 1, "{tier:?}: live did not see the scoop");

        // Studio sees the next ~27 ms and starts on C4 directly.
        let mut c = studio(|s| {
            s.retune_ms = 0.0;
            s.tier = tier;
        });
        run(&mut c, 40, |_| silence_lag(10));
        let outs = run(&mut c, 300, |i| est_lag(scoop(i), 10));
        assert_eq!(target_changes(&outs), 0, "{tier:?}");
        let first = outs.iter().find(|o| o.frame.target != 0.0).map(|o| o.frame.target);
        assert_eq!(first, Some(6000.0), "{tier:?}");
    }
}

#[test]
fn studio_hysteresis_holds_with_lookahead() {
    for tier in [QualityTier::Eco, QualityTier::Hq] {
        let mut c = studio(|s| s.tier = tier);
        run(&mut c, 40, |_| silence_lag(10));
        run(&mut c, hops(200.0), |_| est_lag(6000.0, 10));
        let outs = run(&mut c, hops(3000.0), |i| {
            est_lag(6050.0 + 5.0 * (2.0 * std::f32::consts::PI * 2.0 * i as f32 * HOP_S).sin(), 10)
        });
        assert_eq!(target_changes(&outs), 0, "{tier:?}");
    }
}

#[test]
fn studio_midi_is_delayed_by_the_latency_minus_half_the_transition() {
    for (tau_tr, expect_delay) in [(0.0f32, L), (20.0, L - 8)] {
        let mut c = studio(|s| {
            s.midi_mode = vt_core::MidiMode::Target;
            s.retune_ms = 0.0;
            s.transition_ms = Some(tau_tr);
        });
        run(&mut c, 50, |_| est_lag(6030.0, L));
        c.midi(MidiEvent::NoteOn { note: 62, velocity: 1.0 });
        let outs = run(&mut c, 100, |_| est_lag(6030.0, L));
        let k = outs.iter().position(|o| o.frame.target == 6200.0).unwrap_or(usize::MAX);
        assert_eq!(k + 1, expect_delay, "τ_tr {tau_tr}");
    }
}

/// SN-3: Studio reads the pitch that sets the shift `round(0.5 + T/(4·hop))`
/// hops later in the lookahead, because the shifter applies a ratio a
/// fraction of a hop (its linear ramp) plus a fraction of a period (PSOLA
/// fixes the ratio once per synthesis period) after the boundary. It used to
/// correct each frame with its own pitch, so on moving pitch the output was
/// the input plus the shift from 3-6 ms earlier (vocadito Studio HQ hard
/// tune: settle median 35 ms, output median 2.1 c). Decisions stay on the
/// frame itself; Live has no lookahead and is unchanged.
#[test]
fn studio_shift_uses_the_pitch_the_shifter_will_be_playing() {
    // A slow rise inside C4: 5975 → 6025 c at 0.4 c per hop.
    let p = |i: usize| (5975.0 + 0.4 * i.saturating_sub(40) as f32).min(6025.0);
    for tier in [QualityTier::Eco, QualityTier::Hq] {
        let mut c = studio(|s| {
            s.retune_ms = 0.0;
            s.tier = tier;
        });
        run(&mut c, 40, |_| silence_lag(10));
        let outs = run(&mut c, 260, |i| est_lag(p(i), 10));
        // period ≈ 183 samples at C4: advance round(0.5 + 45.8 / 64) = 1 hop.
        let mut checked = 0;
        for k in 80..200 {
            let (o, later) = (outs[k], outs[k + 1]);
            assert_eq!(o.frame.target, 6000.0, "{tier:?} hop {k}");
            assert!(
                (o.shift_cents + (later.frame.detected - 6000.0)).abs() < 1e-3,
                "{tier:?} hop {k}: shift {} vs pitch a hop later {}",
                o.shift_cents,
                later.frame.detected
            );
            checked += 1;
        }
        assert!(checked > 100);
    }
    // Live: each hop's own pitch.
    let mut c = live(|s| s.retune_ms = 0.0);
    let outs = run(&mut c, 260, |i| est(p(i)));
    for o in &outs[80..200] {
        assert!((o.shift_cents + (o.frame.detected - 6000.0)).abs() < 1e-3);
    }
}

/// SN-1/SN-5 (Studio): a crossing that comes back within the visible
/// lookahead is a wobble, not a note change, and keeps the note. It used to
/// switch there and back (a 100 c output step each way); real singers
/// wander across midpoints inside a note all the time (vocadito Studio HQ
/// hard tune: warble episodes 582 → 400). A longer excursion still switches,
/// on the frame where it starts.
#[test]
fn studio_ignores_crossings_that_return_within_the_lookahead() {
    for tier in [QualityTier::Hq, QualityTier::Max] {
        for len in [3usize, 6, 12] {
            let mut c = studio(|s| {
                s.retune_ms = 0.0;
                s.tier = tier;
            });
            run(&mut c, 40, |_| silence_lag(10));
            let outs = run(&mut c, 400, |i| est_lag(if (200..200 + len).contains(&i) { 6075.0 } else { 6040.0 }, 10));
            assert_eq!(target_changes(&outs), 0, "{tier:?}: {len}-hop excursion switched");
        }
        // Longer than the lookahead (20 hops here): a real move.
        let mut c = studio(|s| {
            s.retune_ms = 0.0;
            s.tier = tier;
        });
        run(&mut c, 40, |_| silence_lag(10));
        let outs = run(&mut c, 400, |i| est_lag(if i >= 200 { 6075.0 } else { 6040.0 }, 10));
        let step = outs.iter().position(|o| o.frame.detected > 6060.0).unwrap_or(usize::MAX);
        let switch = outs.iter().position(|o| o.frame.target == 6100.0).unwrap_or(usize::MAX);
        assert!(switch <= step + 1, "{tier:?}: switch {switch} vs step {step}");
        assert_eq!(target_changes(&outs), 1, "{tier:?}");
    }
}
