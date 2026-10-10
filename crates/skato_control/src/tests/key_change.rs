//! Key changes while singing (docs/research/host-key-scale.md §3.3): a scale
//! change that removes the note being sung keeps that note's target until
//! the next note starts, for at most 200 ms, then glides to the new scale.

use skato_core::{LatencyMode, QualityTier, ScaleKind, Settings};

use super::*;

const TIERS: [QualityTier; 3] = [QualityTier::Eco, QualityTier::Hq, QualityTier::Max];

fn c_major(tier: QualityTier) -> impl FnOnce(&mut Settings) {
    move |s: &mut Settings| {
        s.tier = tier;
        s.retune_ms = 0.0;
        s.note_states = ScaleKind::Major.note_states(0);
    }
}

/// Switch `c` to C natural minor (E removed, E♭ and F legal).
fn to_c_minor(c: &mut Corrector, tier: QualityTier) {
    let mut s = Settings::default();
    c_major(tier)(&mut s);
    s.note_states = ScaleKind::NaturalMinor.note_states(0);
    c.set_settings(&s);
}

#[test]
fn a_key_change_keeps_the_sung_note_for_at_most_200_ms() {
    for (mode, lookahead) in [(LatencyMode::Live, 0), (LatencyMode::Studio, hops(40.0))] {
        for tier in TIERS {
            let mut c = with_config(config(mode, lookahead), c_major(tier));
            // E − 10 c: E in C major.
            let before = hold(&mut c, 6390.0, hops(300.0));
            assert_eq!(before[before.len() - 1].frame.target, 6400.0, "{mode:?} {tier:?}");
            // The key changes mid-note: E is gone, E♭ is nearest.
            to_c_minor(&mut c, tier);
            let after = hold(&mut c, 6390.0, hops(400.0));
            let kept = after.iter().take_while(|o| o.frame.target == 6400.0).count();
            let kept_ms = kept as f32 * HOP_MS;
            assert!(
                (195.0..=215.0).contains(&kept_ms),
                "{mode:?} {tier:?}: the removed note kept its target {kept_ms:.0} ms"
            );
            let last = after[after.len() - 1].frame;
            assert_eq!(last.target, 6300.0, "{mode:?} {tier:?}: then the new scale's nearest note");
        }
    }
}

#[test]
fn the_next_note_after_a_key_change_takes_the_new_scale_at_once() {
    for tier in TIERS {
        let mut c = live(c_major(tier));
        hold(&mut c, 6390.0, hops(300.0));
        to_c_minor(&mut c, tier);
        hold(&mut c, 6390.0, hops(50.0));
        // A breath ends the note; the next one starts in the new scale.
        run(&mut c, hops(150.0), |_| unvoiced());
        let next = hold(&mut c, 6390.0, hops(200.0));
        let first = next.iter().find(|o| o.frame.target > 0.0).map(|o| o.frame.target);
        assert_eq!(first, Some(6300.0), "{tier:?}: no hold for a new note");
    }
}

#[test]
fn a_key_change_that_keeps_the_sung_note_changes_nothing() {
    for tier in TIERS {
        let mut c = live(c_major(tier));
        hold(&mut c, 6710.0, hops(300.0)); // G + 10 c
        to_c_minor(&mut c, tier); // G stays
        let after = hold(&mut c, 6710.0, hops(300.0));
        assert!(after.iter().all(|o| o.frame.target == 6700.0), "{tier:?}");
    }
}
