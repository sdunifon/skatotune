//! The note-intent harness on real singing: the learned note hold on the
//! held-out TEST singers of vocadito reproduces the audited recommendation
//! (`docs/research/note-intent.md` §5.4) in Live HQ hard chromatic at the
//! plugin's default Tolerance (8 c): frame accuracy up, warble and the
//! stable-frame p95 down.
//!
//! Needs the vocadito dataset (CC BY 4.0; not in the repository) at
//! `data/vocadito` or `$SKATO_VOCADITO_DIR`; without it the test prints why
//! and passes. About 10 s in release (it builds its own detector cache).
//!
//! A change to the detector or the control law's front end that moves these
//! numbers needs the models retrained (`crates/skato_control/models/README.md`)
//! and the recorded values below updated in the same change.

use std::path::{Path, PathBuf};

use skato_core::{ReferenceMode, Settings};
use skato_eval::note_intent::cache::{self, Config};
use skato_eval::note_intent::data::Split;
use skato_eval::note_intent::{self, NoteIntentOptions, System, replay};
use skato_eval::render::{self, BlockPattern, RenderOptions};
use skato_eval::vocadito::{Condition, RunConfig, condition_settings, data as vdata};

fn real_dataset() -> Option<PathBuf> {
    let root = std::env::var_os("SKATO_VOCADITO_DIR")
        .map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/vocadito"), PathBuf::from);
    if root.join("Annotations").is_dir() && root.join("Audio/vocadito_10.wav").is_file() {
        Some(root)
    } else {
        eprintln!(
            "skipping the note-intent harness check: the vocadito dataset is not at {} \
             (download it to data/vocadito, or set SKATO_VOCADITO_DIR)",
            root.display()
        );
        None
    }
}

#[test]
fn learned_note_hold_reproduces_the_recommendation_on_test() {
    let Some(data) = real_dataset() else { return };
    let tmp = std::env::temp_dir().join(format!("skato_note_intent_{}", std::process::id()));
    let opts = NoteIntentOptions {
        data,
        out: tmp.join("out"),
        cache: Some(tmp.join("cache")),
        split: Split::Test,
        configs: vec![Config::LIVE_HQ],
        conditions: vec![Condition::HardChromatic],
        systems: vec![System::NoteHold],
        tolerance: 8.0,
        sweep: Vec::new(),
        ..NoteIntentOptions::default()
    };
    let o = note_intent::run(&opts, &|_| {}).expect("harness run");
    let _ = std::fs::remove_dir_all(&tmp);
    assert_eq!(o.files.len(), 12);
    assert_eq!(o.singers.len(), 9);
    let get = |s: System| o.main_point(s, Config::LIVE_HQ, Condition::HardChromatic).expect("point").pooled.clone();
    let (base, hold) = (get(System::Baseline), get(System::NoteHold));
    let m = |x: &skato_eval::note_intent::metrics::NoteMetrics, k: &str| x.get(k);
    eprintln!(
        "TEST Live HQ hard chromatic, 8 c: frame acc {:.1} → {:.1} %, warble {:.1} → {:.1} /min, p95 {:.1} → {:.1} c, \
         per-note {:.1} → {:.1} %, agreeP {:.1} → {:.1} %",
        m(&base, "acc_agree"),
        m(&hold, "acc_agree"),
        m(&base, "warble_min"),
        m(&hold, "warble_min"),
        m(&base, "p95"),
        m(&hold, "p95"),
        m(&base, "note_dom"),
        m(&hold, "note_dom"),
        m(&base, "pub_agree"),
        m(&hold, "pub_agree"),
    );
    // The direction of the recommendation, with margins (it found +3.3
    // points, −26 /min and −5.5 c).
    assert!(m(&hold, "acc_agree") > m(&base, "acc_agree") + 1.5, "frame accuracy must rise");
    assert!(m(&hold, "warble_min") < 0.7 * m(&base, "warble_min"), "warble must fall by 30 % or more");
    assert!(m(&hold, "p95") < m(&base, "p95") - 2.0, "the stable p95 must fall by 2 c or more");
    // And the recorded values themselves (§5.4), so a detector or front-end
    // change that would need a retrain shows up here.
    for (x, key, want, tol) in [
        (&base, "acc_agree", 70.5, 0.5),
        (&base, "warble_min", 50.0, 1.0),
        (&base, "p95", 25.6, 0.5),
        (&hold, "acc_agree", 73.8, 0.5),
        (&hold, "warble_min", 24.0, 1.0),
        (&hold, "p95", 20.1, 0.5),
        (&hold, "note_dom", 76.5, 0.5),
    ] {
        let got = m(x, key);
        assert!((got - want).abs() <= tol, "{key}: {got:.2}, recorded {want} ± {tol}");
    }
}

/// The harness measures what the plugin does: replaying the cached detector
/// stream with both Labs options on gives the engine's own telemetry, frame
/// for frame and bit for bit (a real render, 256-sample blocks), in Live HQ
/// and Studio HQ.
#[test]
fn the_replay_equals_an_engine_render_with_the_options_on() {
    let Some(data) = real_dataset() else { return };
    let meta = vdata::load_metadata(&data);
    let track = vdata::load_track(&data, 10, &meta).expect("vocadito_10");
    let range = vdata::choose_range(track.min_f0());
    for cfg in [Config::LIVE_HQ, Config::STUDIO_HQ] {
        let mut s = condition_settings(Condition::HardChromatic, RunConfig::new(cfg.mode, cfg.tier), None).expect("settings");
        s.note_hold = true;
        s.reference = ReferenceMode::FollowSinger;
        let det = Settings { tier: cfg.tier, ..Settings::default() };
        let stream = cache::run_detector(&track.audio, track.sample_rate, cfg, range, &det);
        let run = replay::replay(&stream, &s);
        assert!(run.hold.proposals > 0, "{}: the veto ran", cfg.label());
        let opts = RenderOptions {
            config: cfg.engine_config(track.sample_rate, range),
            settings: s,
            blocks: BlockPattern::Fixed(256),
            telemetry: true,
            midi: Vec::new(),
            extra_tail: 0,
        };
        let rendered = render::render(&track.audio, &opts);
        let replayed = run.telemetry();
        assert_eq!(replayed.len(), rendered.telemetry.len(), "{}", cfg.label());
        for (k, (a, b)) in replayed.iter().zip(&rendered.telemetry).enumerate() {
            assert_eq!(format!("{a:?}"), format!("{b:?}"), "{} frame {k}", cfg.label());
        }
    }
}
