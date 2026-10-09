//! End-to-end check of the vocadito evaluation on a synthetic, vocadito-shaped
//! dataset written to a temp directory: the telemetry time mapping, the
//! annotation comparison and the correction measures must agree with the
//! synthesiser's exact truth.

use std::fmt::Write as _;
use std::path::Path;

use vt_core::{LatencyMode, QualityTier};
use vt_eval::synth::{Melody, Note, SignalBuilder, Vibrato, Voice};
use vt_eval::vocadito::{self, Condition, RunConfig, VocaditoOptions};
use vt_eval::wav;

const SR: f32 = 44_100.0;

fn write_dataset(root: &Path) {
    let sig = SignalBuilder::new(SR)
        .seed(5)
        .silence(0.3)
        .melody(
            Melody::new(Voice::male())
                .note(Note::midi(50.0, 0.5).detune(30.0))
                .note(Note::midi(53.0, 0.45).detune(-25.0))
                .note(
                    Note::midi(55.0, 0.6)
                        .gap(0.2)
                        .detune(20.0)
                        .vibrato(Vibrato::new(5.5, 40.0)),
                )
                .note(Note::midi(52.0, 0.5).detune(-35.0)),
        )
        .silence(0.3)
        .build();
    wav::write_wav(&root.join("Audio/vocadito_1.wav"), &[&sig.audio], SR).unwrap();
    let mut f0 = String::new();
    let mut n = 0usize;
    while n < sig.len() {
        let _ = writeln!(f0, "{},{}", n as f64 / SR as f64, sig.f0_truth[n]);
        n += 256;
    }
    std::fs::create_dir_all(root.join("Annotations/F0")).unwrap();
    std::fs::write(root.join("Annotations/F0/vocadito_1_f0.csv"), f0).unwrap();
    let mut notes = String::new();
    for s in &sig.notes {
        let hz = 440.0 * ((s.target_cents - 6900.0) / 1200.0).exp2();
        let _ = writeln!(
            notes,
            "{:.6},{hz:.3},{:.6}",
            s.start as f64 / SR as f64,
            (s.end - s.start) as f64 / SR as f64
        );
    }
    std::fs::create_dir_all(root.join("Annotations/Notes")).unwrap();
    std::fs::write(
        root.join("Annotations/Notes/vocadito_1_notesA1.csv"),
        &notes,
    )
    .unwrap();
    std::fs::write(
        root.join("Annotations/Notes/vocadito_1_notesA2.csv"),
        &notes,
    )
    .unwrap();
    std::fs::write(
        root.join("vocadito_metadata.csv"),
        "track_id,singer_id,average_pitch,language\n1,SX,52,none\n",
    )
    .unwrap();
}

#[test]
fn synthetic_dataset_scores_like_its_truth() {
    let root = std::env::temp_dir().join(format!("vt_eval_vocadito_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    write_dataset(&root);
    let opts = VocaditoOptions {
        data: root.clone(),
        out: root.join("out"),
        configs: vec![
            RunConfig::new(LatencyMode::Live, QualityTier::Hq),
            RunConfig::new(LatencyMode::Studio, QualityTier::Hq),
        ],
        conditions: vec![Condition::Detect, Condition::HardChromatic],
        threads: 2,
        wavs: false,
        ..VocaditoOptions::default()
    };
    let o = vocadito::run(&opts, &|_| {}).unwrap();
    assert_eq!(o.files.len(), 1);
    let file = &o.files[0];
    assert_eq!(file.range, vt_core::VoiceRange::AltoTenor);
    // The offline reference tracker reproduces the truth.
    assert!(
        file.reference.rpa50() > 0.97,
        "ref rpa {}",
        file.reference.rpa50()
    );
    assert!(
        file.reference_lag_ms.abs() <= 2.0,
        "ref lag {}",
        file.reference_lag_ms
    );
    for d in &o.det {
        let c = &d.counts;
        assert!(d.telemetry_ok && d.finite);
        // Telemetry mapped with sample_pos − D lines up with the truth.
        assert!(
            d.best_lag_ms.abs() <= 4.0,
            "{}: lag {}",
            d.config.label(),
            d.best_lag_ms
        );
        assert!(c.rpa50() > 0.93, "{}: rpa {}", d.config.label(), c.rpa50());
        assert!(c.octave_rate() < 0.01);
        assert!(c.f1() > 0.93, "{}: f1 {}", d.config.label(), c.f1());
        assert!(c.fine.percentile(50.0) < 5.0);
        // Amount 0 is a pass-through delayed by D.
        let m = d.measured.expect("measured latency");
        assert!(
            (m.lag as i64 - i64::from(d.latency)).abs() <= 2,
            "{m:?} vs {}",
            d.latency
        );
    }
    for r in &o.corr {
        let c = &r.counts;
        assert!(c.stable > 500, "{}", c.stable);
        if r.condition == Condition::Detect {
            // Amount-0 pass-through: the output measures like the input.
            let (i, o) = (c.inp.percentile(50.0), c.out.percentile(50.0));
            assert!((i - o).abs() < 2.0, "passthrough {i} vs {o}");
            assert_eq!(c.shift_segments + c.jumps + c.onset_glitches, 0);
            continue;
        }
        // The input is detuned by 20-35 c; hard-tune puts it on the grid.
        assert!(c.inp.percentile(50.0) > 15.0);
        assert!(
            c.out.percentile(50.0) < 3.0,
            "{}: out {}",
            r.config.label(),
            c.out.percentile(50.0)
        );
        assert!(c.out.within(10.0) > 90.0);
        assert_eq!(c.nonfinite + c.blowups, 0);
        assert_eq!(c.notes, 4);
        assert_eq!(c.notes_agree_out, 4);
    }
    vocadito::report::write_all(&opts.out, &opts, &o).unwrap();
    let md = std::fs::read_to_string(opts.out.join("report.md")).unwrap();
    assert!(md.contains("## Detector on real singing"));
    assert!(md.contains("live_hq_natural"));
    let _ = std::fs::remove_dir_all(&root);
}

/// The real vocadito dataset, if it has been downloaded (it is not in the
/// repository: CC BY 4.0, see `vt_eval help`): `data/vocadito` in the
/// workspace, or `$VT_VOCADITO_DIR`.
fn real_dataset() -> Option<std::path::PathBuf> {
    let root = std::env::var_os("VT_VOCADITO_DIR").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/vocadito"),
        std::path::PathBuf::from,
    );
    if root.join("Audio/vocadito_10.wav").is_file() {
        Some(root)
    } else {
        eprintln!(
            "skipping: the vocadito dataset is not at {} (download it to data/vocadito to run the real-singing checks)",
            root.display()
        );
        None
    }
}

/// Fast smoke test on three short real takes (one per voice range: 10 alto/
/// tenor, 20 soprano, 26 low male): the whole chain runs on real singing,
/// telemetry lines up, the output stays finite, the amount-0 render is a
/// pass-through delayed by D (plus a bounded content wander), and hard tune
/// moves the pitch onto the grid. Quality is gated by
/// `real_singing_gates` (opt-in, all 40 files).
#[test]
fn real_vocadito_smoke() {
    let Some(data) = real_dataset() else { return };
    let out = std::env::temp_dir().join(format!("vt_eval_vocadito_smoke_{}", std::process::id()));
    let opts = VocaditoOptions {
        data,
        out: out.clone(),
        configs: vec![RunConfig::new(LatencyMode::Live, QualityTier::Hq)],
        conditions: vec![Condition::Detect, Condition::HardChromatic],
        files: vec![10, 20, 26],
        threads: 3,
        wavs: false,
        ..VocaditoOptions::default()
    };
    let o = vocadito::run(&opts, &|_| {}).unwrap();
    assert_eq!(o.files.len(), 3);
    let ranges: Vec<_> = o.files.iter().map(|f| (f.id, f.range)).collect();
    for (id, want) in [
        (10, vt_core::VoiceRange::AltoTenor),
        (20, vt_core::VoiceRange::Soprano),
        (26, vt_core::VoiceRange::LowMale),
    ] {
        assert!(ranges.contains(&(id, want)), "{ranges:?}");
    }
    assert_eq!(o.det.len(), 3);
    for d in &o.det {
        let c = &d.counts;
        eprintln!(
            "vocadito_{}: rpa50 {:.3} f1 {:.3} lag {:+.0} ms, D {} measured {:?}, content median {:.2} ms (max |wander| {:.2} periods)",
            d.file,
            c.rpa50(),
            c.f1(),
            d.best_lag_ms,
            d.latency,
            d.measured.map(|m| m.lag),
            d.content.delay_ms(50.0),
            d.content.max_abs_wander_periods()
        );
        assert!(d.telemetry_ok && d.finite, "vocadito_{}", d.file);
        assert!(c.rpa50() > 0.85, "vocadito_{}: rpa50 {}", d.file, c.rpa50());
        assert!(c.f1() > 0.85, "vocadito_{}: voicing F1 {}", d.file, c.f1());
        // Sanity only (the synthetic test above checks the mapping to
        // ±4 ms against exact truth): a time-base error such as forgetting
        // to subtract D, or a Studio-sized offset, would exceed this.
        assert!(
            d.best_lag_ms.abs() <= 12.0,
            "vocadito_{}: telemetry offset {}",
            d.file,
            d.best_lag_ms
        );
        // The whole-take cross-correlation can lock onto a neighbouring
        // glottal cycle on strongly pitched audio, so it only has to land
        // within one period of the range's lowest pitch; the voiced content
        // delay below is the precise check.
        let m = d.measured.expect("measured delay");
        let one_period = 44_100.0 / f64::from(d.range.f_min());
        assert!(
            (m.lag as f64 - f64::from(d.latency)).abs() <= one_period,
            "vocadito_{}: measured {} vs D {}",
            d.file,
            m.lag,
            d.latency
        );
        assert!(
            d.content.segments > 50,
            "vocadito_{}: {:?}",
            d.file,
            d.content.segments
        );
        // Content arrives D plus at most a few periods late.
        let med = d.content.delay_ms(50.0);
        assert!(
            (med - d.content.reported_ms()).abs() < 1.0,
            "vocadito_{}: content median {med} ms vs D {} ms",
            d.file,
            d.content.reported_ms()
        );
    }
    for r in &o.corr {
        let c = &r.counts;
        assert_eq!(c.nonfinite + c.blowups, 0, "vocadito_{}", r.file);
        assert!(
            c.stable > 500,
            "vocadito_{}: {} stable frames",
            r.file,
            c.stable
        );
        if r.condition == Condition::HardChromatic {
            let (i, out50) = (c.inp.percentile(50.0), c.out.percentile(50.0));
            assert!(
                out50 < 0.5 * i,
                "vocadito_{}: hard tune {out50} c vs input {i} c",
                r.file
            );
            // Sanity, not quality (that is `real_singing_gates`): file 10
            // has 70 % of its stable frames within 10 c today.
            assert!(c.out.within(10.0) > 50.0, "vocadito_{}", r.file);
        }
    }
    vocadito::report::write_all(&opts.out, &opts, &o).unwrap();
    let md = std::fs::read_to_string(opts.out.join("report.md")).unwrap();
    assert!(md.contains("## Real-singing gates") && md.contains("Voiced content delay"));
    let _ = std::fs::remove_dir_all(&out);
}

/// Real-singing gates (EV-4) on all 40 vocadito files, Live HQ: detector
/// within margins of the offline reference tracker, hard-tune regression
/// guards per voice range (`vt_eval::vocadito::gates`). Opt-in because it
/// renders 13.6 min of audio (about 1-2 min in release):
///
/// ```text
/// cargo test -p vt_eval --release --test vocadito -- --ignored --nocapture
/// ```
#[test]
#[ignore = "real-data gate: needs data/vocadito and ~1-2 min in release"]
fn real_singing_gates() {
    let Some(data) = real_dataset() else { return };
    let opts = VocaditoOptions {
        data,
        out: std::env::temp_dir().join(format!("vt_eval_vocadito_gates_{}", std::process::id())),
        configs: vec![vocadito::gates::gate_config()],
        conditions: vec![Condition::Detect, Condition::HardChromatic],
        wavs: false,
        ..VocaditoOptions::default()
    };
    let o = vocadito::run(&opts, &|_| {}).unwrap();
    assert_eq!(o.files.len(), 40, "expected the full vocadito set");
    let gates = vocadito::gates::gates(&o);
    for g in &gates {
        eprintln!("[{:?}] {}: {}", g.pass, g.name, g.value);
    }
    assert!(
        gates.iter().all(|g| g.pass == Some(true)),
        "real-singing gate failed (see above)"
    );
}
