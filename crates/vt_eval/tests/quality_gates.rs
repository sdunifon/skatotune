//! DESIGN §4 quality gates at test scale, run against the real engine:
//!
//! ```text
//! cargo test -p vt_eval --release --test quality_gates -- --nocapture
//! ```
//!
//! Each test prints its measurements so a failure is diagnosable without
//! re-running the suite. The full matrix (every mode and tier) is
//! `vt_eval suite`, which evaluates the same gates into `out/report.md`
//! strictly. Here, the few known misses of the current engine are listed
//! explicitly (`KNOWN_SLOW_ONSETS`, `KNOWN_HELDOUT_*`) with a regression
//! guard each and a printed "KNOWN DESIGN §4 FAILURE" line; every other
//! measurement must meet the DESIGN target, and the suite still reports the
//! gate as FAIL. The CPU gate only runs in release builds, and exclusively
//! (no other gate test renders while it times `process()`).

use std::sync::{RwLock, RwLockReadGuard};

use vt_core::{EngineKind, LatencyMode, QualityTier};
use vt_eval::render::{self, BlockPattern, RenderOptions};
use vt_eval::scenarios::{self, Scenario};
use vt_eval::suite::{self, RunResult};
use vt_eval::{reference, rtbench};

const SR: f32 = 48_000.0;

/// The CPU gate times `process()` and must not share the machine with the
/// other gate tests' renders (cargo runs the tests of this binary in
/// parallel): every test holds this lock shared, the CPU gate exclusively.
static CPU_QUIET: RwLock<()> = RwLock::new(());

fn shared() -> RwLockReadGuard<'static, ()> {
    CPU_QUIET.read().unwrap_or_else(|e| e.into_inner())
}

fn scenario(name: &str) -> Scenario {
    scenarios::build(name, SR).unwrap_or_else(|| panic!("unknown scenario {name}"))
}

fn run(name: &str, mode: LatencyMode, tier: QualityTier, retune_ms: f32) -> RunResult {
    let s = scenario(name);
    let in_track =
        (s.measures.vibrato || s.measures.jitter).then(|| reference::track(&s.signal.audio, SR));
    let (r, _) = suite::evaluate(
        &s,
        in_track.as_ref(),
        mode,
        tier,
        retune_ms,
        &BlockPattern::Fixed(128),
    );
    assert!(r.finite, "{name}: non-finite output");
    r
}

const MELODIES: [&str; 4] = ["melody_male", "melody_female", "low_male", "high_soprano"];

#[test]
fn gate_tuning_live_hq() {
    let _quiet = shared();
    for name in MELODIES {
        for retune in [0.0, 20.0] {
            let t = run(name, LatencyMode::Live, QualityTier::Hq, retune)
                .tuning
                .expect("tuning measured");
            eprintln!("{name} Live HQ rt{retune}: {t:?}");
            assert!(
                t.frames > 100,
                "{name}: too few stable voiced output frames ({t:?})"
            );
            assert!(
                t.median_abs < 5.0,
                "{name} rt{retune}: median |error| {:.2} c ≥ 5 c",
                t.median_abs
            );
            assert!(
                t.p95_abs < 15.0,
                "{name} rt{retune}: p95 |error| {:.2} c ≥ 15 c",
                t.p95_abs
            );
        }
    }
}

#[test]
fn gate_tuning_studio_hq() {
    let _quiet = shared();
    for name in MELODIES {
        let t = run(name, LatencyMode::Studio, QualityTier::Hq, 0.0)
            .tuning
            .expect("tuning measured");
        eprintln!("{name} Studio HQ: {t:?}");
        assert!(t.frames > 100);
        assert!(
            t.median_abs < 3.0,
            "{name}: Studio median |error| {:.2} c ≥ 3 c",
            t.median_abs
        );
    }
}

/// Onsets that miss the DESIGN §4 settling gate (< 30 ms, every onset) with
/// the current engine: (scenario, onset index, regression guard in s, cause).
///
/// This is **not** a pass of the DESIGN gate. `vt_eval suite` reports the
/// strict gate (every onset < 30 ms) and shows FAIL while this list is not
/// empty. Each guard is the value measured when the entry was last set (Live
/// HQ, retune 0, 48 kHz, 128-sample blocks) plus 5 ms, so the onset cannot
/// get much worse unnoticed; every onset *not* listed must meet 30 ms. When
/// a listed onset meets 30 ms the test says so: delete its entry then. The
/// per-onset breakdown printed by the test (decision point, detection,
/// target switch, convergence) is what the causes are read from.
///
/// Empty since the 2026-10-09 hardening pass (integrated tree): every one of
/// the 18 onsets meets 30 ms, worst 27.0 ms (low_male #2). The three former
/// entries and what fixed them:
///
/// - low_male #0 / #2 (E2, 82 Hz): 33.0 / 35.0 → 26.0 / 27.0 ms. vt_shift
///   ON-4 starts a Live PSOLA onset on the newest known pitch mark instead of
///   waiting up to a period for one ahead of the content (and ON-3 applies
///   the newest commanded ratio); vt_control ON-9 applies a hard-tune shift
///   at once at a PSOLA onset.
/// - melody_male #0 (C3 after a slow −75 c scoop): 31.6 → 15.6 ms. vt_control
///   ON-5 drops the 25 ms near-band persistence (CL-2) once the first note of
///   a phrase has scooped ≥ 20 c toward the next one.
const KNOWN_SLOW_ONSETS: &[(&str, usize, f64, &str)] = &[];

fn ms(v: Option<f64>) -> String {
    v.map_or("-".into(), |x| format!("{:.1} ms", x * 1e3))
}

/// DESIGN §4 "Settling (Live, retune 0): < 30 ms to stay within 10 c after
/// onset", on every onset (EV-1: no median; an onset that never settles
/// fails). The clock starts at the note-decision point
/// (`metrics::settling_with`).
#[test]
fn gate_settling_live_retune0() {
    let _quiet = shared();
    let mut failures = Vec::new();
    let mut known_failing = 0;
    for name in [
        "melody_male",
        "melody_female",
        "low_male",
        "high_soprano",
        "steps",
    ] {
        let s = run(name, LatencyMode::Live, QualityTier::Hq, 0.0)
            .settling
            .expect("settling measured");
        assert!(!s.onsets.is_empty(), "{name}: no onset evaluated");
        for (i, o) in s.onsets.iter().enumerate() {
            eprintln!(
                "{name} onset {i}: settle {} (decision point {} into the note; detected {}, target switch {}, convergence {})",
                o.settle_s
                    .map_or("never".into(), |v| format!("{:.1} ms", v * 1e3)),
                ms(Some(o.clock_s)),
                ms(o.detect_s),
                ms(o.switch_s),
                ms(o.convergence_s()),
            );
            let Some(v) = o.settle_s else {
                failures.push(format!("{name} onset {i} never settled within 10 c"));
                continue;
            };
            match KNOWN_SLOW_ONSETS.iter().find(|k| k.0 == name && k.1 == i) {
                Some(&(_, _, guard, cause)) => {
                    if v < suite::SETTLING_LIMIT_S {
                        eprintln!(
                            "  {name} onset {i} now meets the 30 ms gate: remove it from KNOWN_SLOW_ONSETS"
                        );
                    } else {
                        known_failing += 1;
                        eprintln!(
                            "  KNOWN DESIGN §4 FAILURE ({cause}): {:.1} ms ≥ 30 ms; regression guard {:.1} ms",
                            v * 1e3,
                            guard * 1e3
                        );
                    }
                    if v >= guard {
                        failures.push(format!(
                            "{name} onset {i}: {:.1} ms ≥ its regression guard {:.1} ms ({cause})",
                            v * 1e3,
                            guard * 1e3
                        ));
                    }
                }
                None if v >= suite::SETTLING_LIMIT_S => failures.push(format!(
                    "{name} onset {i}: settling {:.1} ms ≥ 30 ms",
                    v * 1e3
                )),
                None => {}
            }
        }
    }
    if known_failing == 0 {
        eprintln!("settling: every onset meets DESIGN §4 (< 30 ms)");
    } else {
        eprintln!(
            "settling: {known_failing} known onsets still miss DESIGN §4 (see KNOWN_SLOW_ONSETS; `vt_eval suite` reports FAIL)"
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn gate_vibrato_preservation() {
    let _quiet = shared();
    // Scenario settings keep PreserveVibrato on and depth 100 %; retune 0.
    // Both the delayed, faded-in vibrato and vibrato from the note onset.
    for name in suite::VIBRATO_SCENARIOS {
        let v = run(name, LatencyMode::Live, QualityTier::Hq, 0.0)
            .vibrato
            .expect("vibrato measured");
        for n in &v.notes {
            eprintln!("{name} note @{}: {n:?}", n.start);
        }
        assert!(
            !v.notes.is_empty(),
            "{name}: no vibrato note could be measured in the output"
        );
        assert!(
            v.worst_extent_ratio_error <= 0.15,
            "{name}: extent off by {:.1} %",
            v.worst_extent_ratio_error * 100.0
        );
        assert!(
            v.centre_error_median < 8.0,
            "{name}: centre median error {:.2} c",
            v.centre_error_median
        );
    }
}

/// EV-3: the preservation gate's window starts after the vibrato is
/// developed, so it never sees the causal tracker engage. On vibrato that
/// starts with the note, the output must reach 80 % of the input extent
/// within 2.5 periods (R§8.11: engage after about one cycle, ramp over half
/// a period) and the first two periods must not be exaggerated. Their
/// flattening (early ratio < 1) is the documented causal limit and is only
/// printed.
#[test]
fn gate_vibrato_engagement() {
    let _quiet = shared();
    let v = run(
        suite::ENGAGEMENT_SCENARIO,
        LatencyMode::Live,
        QualityTier::Hq,
        0.0,
    )
    .vibrato
    .expect("vibrato measured");
    assert_eq!(v.engagement.len(), 5, "every vibrato note measured");
    for e in &v.engagement {
        eprintln!(
            "vibrato from onset @{}: {:.2} Hz, engaged after {:?} periods, first two periods out/in extent {:.1}/{:.1} c",
            e.note_start,
            e.rate_hz,
            e.engagement_periods(),
            e.out_early,
            e.in_early
        );
    }
    let worst = v.worst_engagement_periods();
    assert!(
        worst <= suite::ENGAGEMENT_LIMIT_PERIODS,
        "vibrato engaged only after {worst:.2} periods"
    );
    let (lo, hi) = v.early_ratio_range();
    eprintln!("early extent ratio {lo:.2}-{hi:.2} (first cycle flattened by the causal tracker)");
    assert!(
        hi <= suite::EARLY_RATIO_MAX,
        "early vibrato exaggerated: ratio {hi:.2}"
    );
}

#[test]
fn gate_added_jitter() {
    let _quiet = shared();
    for retune in [0.0, 20.0] {
        let j = run("steady", LatencyMode::Live, QualityTier::Hq, retune)
            .jitter
            .expect("jitter measured");
        eprintln!("steady rt{retune}: {j:?}");
        assert!(j.frames > 500);
        assert!(
            j.added_rms < 3.0,
            "added jitter {:.2} c RMS ≥ 3 c",
            j.added_rms
        );
    }
}

#[test]
fn gate_voicing_f1() {
    let _quiet = shared();
    for name in ["consonants", "melody_male", "melody_female"] {
        let v = run(name, LatencyMode::Live, QualityTier::Hq, 20.0)
            .voicing
            .expect("voicing measured");
        eprintln!("{name}: {v:?}");
        assert!(v.f1 >= 0.97, "{name}: voicing F1 {:.3} < 0.97", v.f1);
    }
}

#[test]
fn gate_transparency_amount0() {
    let _quiet = shared();
    let configs: Vec<(LatencyMode, EngineKind)> = suite::MODES
        .iter()
        .flat_map(|&mode| {
            let engines: &[EngineKind] = if mode == LatencyMode::LiveMin {
                &[EngineKind::Classic]
            } else {
                &[EngineKind::Classic, EngineKind::Natural]
            };
            engines.iter().map(move |&e| (mode, e))
        })
        .collect();
    // The seven checks are independent renders: run them side by side.
    let checks: Vec<suite::ModeCheck> = std::thread::scope(|scope| {
        let handles: Vec<_> = configs
            .iter()
            .map(|&(mode, e)| scope.spawn(move || suite::mode_check(SR, mode, e)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .expect("mode check panicked")
                    .expect("melody scenario")
            })
            .collect()
    });
    for c in &checks {
        let (mode, e) = (c.mode, c.engine);
        eprintln!("{c:?}");
        match e {
            EngineKind::Classic => assert!(
                c.transparency_max_diff <= 1e-6,
                "{mode:?} Classic not exact: {:.2e}",
                c.transparency_max_diff
            ),
            EngineKind::Natural => assert!(
                c.transparency_snr_db >= 25.0,
                "{mode:?} PSOLA seg. SNR {:.1} dB < 25",
                c.transparency_snr_db
            ),
        }
        assert!(
            c.block_invariance_max_diff <= 1e-6,
            "{mode:?} {e:?}: block invariance {:.2e}",
            c.block_invariance_max_diff
        );
        // R§10.3: voiced content wander within one sung period.
        let d = &c.content_delay;
        assert!(
            d.segments > 0 && d.max_abs_wander_periods() <= 1.0,
            "{mode:?} {e:?}: content wander {d:?}"
        );
    }
}

#[test]
fn gate_recovers_after_overflowing_input() {
    let _quiet = shared();
    // 100 ms of ±1e30 (finite, but its square overflows f32), 400 ms of
    // silence, then a note: the note must come out as from a fresh engine.
    let s = scenario("melody_male");
    let note = &s.signal.audio[..(1.5 * SR) as usize];
    let mut x: Vec<f32> = (0..4800)
        .map(|i| if i % 3 == 0 { 1.0e30 } else { -1.0e30 })
        .collect();
    x.extend(std::iter::repeat_n(0.0, 19_200));
    let offset = x.len();
    x.extend_from_slice(note);
    for mode in suite::MODES {
        let o = RenderOptions::new(SR, mode, QualityTier::Hq);
        let clean = render::render(note, &o);
        let r = render::render(&x, &o);
        let tail = &r.aligned(0)[offset..];
        assert!(
            tail.iter().all(|v| v.is_finite()),
            "{mode:?}: output still non-finite after the gap"
        );
        // Allow the first 300 ms to settle trackers, then compare.
        let skip = (0.3 * SR) as usize;
        let (a, b) = (&clean.aligned(0)[skip..], &tail[skip..]);
        let e: f64 = a.iter().zip(b).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let p: f64 = a.iter().map(|v| (*v as f64).powi(2)).sum();
        assert!(
            e / p < 0.01,
            "{mode:?}: did not recover ({:.1} dB error)",
            10.0 * (e / p).log10()
        );
    }
}

#[test]
fn gate_cpu_tail_latency() {
    let _quiet = CPU_QUIET.write().unwrap_or_else(|e| e.into_inner());
    if cfg!(debug_assertions) {
        eprintln!(
            "gate_cpu_tail_latency skipped: timing gates need a release build (cargo test --release -p vt_eval --test quality_gates)"
        );
        return;
    }
    let opts = rtbench::RtBenchOptions {
        seconds: 5.0,
        buffers: vec![32, 128],
        modes: vec![LatencyMode::Live],
        tiers: vec![QualityTier::Hq, QualityTier::Max],
        ..rtbench::RtBenchOptions::default()
    };
    let rows = rtbench::run(&opts, &mut |r| eprintln!("{r:?}"));
    for r in rows {
        if let Some(ok) = r.design_gate() {
            assert!(
                ok,
                "{:?} {:?} @{}: p99.9 {:.1} % of the period",
                r.mode,
                r.tier,
                r.buffer,
                r.p999_fraction() * 100.0
            );
        }
    }
}

/// Held-out tuning runs (EV-4, `scenarios::HELDOUT`) that miss DESIGN §4's
/// p95 < 15 c with the current engine: (scenario, retune ms, p95 guard c,
/// cause), regression guards like [`KNOWN_SLOW_ONSETS`]. `heldout_room` at
/// retune 0 measured 22.6, 40.0 and 34.7 c across the parallel changes of
/// 2026-10-08 (92 % of frames within 10 c), 39.2 c at v1 close and 34.0 c
/// after the 2026-10-09 hardening pass (guard tightened 45 → 40 c); its
/// retune-20 run and both pentatonic runs meet the target.
const KNOWN_HELDOUT_TUNING: &[(&str, f32, f64, &str)] = &[(
    "heldout_room",
    0.0,
    40.0,
    "room reverb and breathy/voiced onsets: hard tune off-note on 5-8 % of frames",
)];

/// Held-out settling guard: at most this many onsets of the held-out
/// scenarios may miss 30 ms, none by more than [`HELDOUT_WORST_S`], and
/// every onset must settle. Aggregate rather than per onset because which
/// voiced-consonant onset is slow moved between engine versions (2026-10-08:
/// 4-5 of 11 onsets ≥ 30 ms, worst 82-89 ms, all /m z v l/ or breathy
/// onsets: detection 10-41 ms, then 20-76 ms of convergence; the owners'
/// snapshots of the 2026-10-09 hardening pass measured 3-6). After that
/// pass: 3 of 11 (heldout_pentatonic #4 36 ms, heldout_room #1 32 ms and #2
/// 85 ms; the room onsets are measurement-limited: a perfectly in-tune
/// resynthesis in the same room also measures 96 / 78 ms). Guard tightened
/// 6 → 5.
const HELDOUT_SLOW_MAX: usize = 5;
const HELDOUT_WORST_S: f64 = 0.120;

/// The DESIGN §4 Live targets on scenarios never used to tune the engine
/// (EV-4): R§10.2's ±80 c detune in a pentatonic scale, voiced-consonant
/// onsets, hard and soft attacks, a breathy voice in a room over a noise
/// floor. Every miss beyond the guards above fails the test.
#[test]
fn gate_heldout_scenarios() {
    let _quiet = shared();
    let mut failures = Vec::new();
    let mut slow = 0usize;
    for name in scenarios::HELDOUT {
        for retune in [0.0f32, 20.0] {
            let r = run(name, LatencyMode::Live, QualityTier::Hq, retune);
            let t = r.tuning.expect("tuning measured");
            eprintln!("{name} rt{retune}: {t:?}");
            assert!(t.frames > 1000, "{name}: too few stable frames");
            let known = KNOWN_HELDOUT_TUNING
                .iter()
                .find(|k| k.0 == name && k.1 == retune);
            match known {
                Some(&(_, _, guard, cause)) => {
                    eprintln!(
                        "  KNOWN DESIGN §4 MISS on held-out data ({cause}): p95 {:.1} c; regression guard {guard} c",
                        t.p95_abs
                    );
                    if t.p95_abs >= guard || t.median_abs >= 5.0 {
                        failures.push(format!(
                            "{name} rt{retune}: p95 {:.1} c / median {:.1} c beyond guard {guard} c",
                            t.p95_abs, t.median_abs
                        ));
                    }
                }
                None if t.median_abs >= 5.0 || t.p95_abs >= 15.0 => failures.push(format!(
                    "{name} rt{retune}: median {:.2} c, p95 {:.2} c (target < 5 / < 15)",
                    t.median_abs, t.p95_abs
                )),
                None => {}
            }
            if retune != 0.0 {
                continue;
            }
            let s = r.settling.expect("settling measured");
            for (i, o) in s.onsets.iter().enumerate() {
                let v = o.settle_s.unwrap_or(f64::INFINITY);
                eprintln!(
                    "  {name} onset {i}: settle {} (decision point {}, detected {}, switch {}, convergence {})",
                    ms(o.settle_s),
                    ms(Some(o.clock_s)),
                    ms(o.detect_s),
                    ms(o.switch_s),
                    ms(o.convergence_s())
                );
                if v >= suite::SETTLING_LIMIT_S {
                    slow += 1;
                }
                if v >= HELDOUT_WORST_S {
                    failures.push(format!(
                        "{name} onset {i}: settling {} beyond the held-out guard {:.0} ms",
                        o.settle_s
                            .map_or("never".into(), |x| format!("{:.1} ms", x * 1e3)),
                        HELDOUT_WORST_S * 1e3
                    ));
                }
            }
            if name == "heldout_pentatonic" {
                let v = r.voicing.expect("voicing measured");
                eprintln!("  voicing {v:?}");
                if v.f1 < 0.97 {
                    failures.push(format!("{name}: voicing F1 {:.3} < 0.97", v.f1));
                }
            }
        }
    }
    eprintln!(
        "held-out settling: {slow} onsets ≥ 30 ms (KNOWN DESIGN §4 misses on held-out data; guard ≤ {HELDOUT_SLOW_MAX})"
    );
    if slow > HELDOUT_SLOW_MAX {
        failures.push(format!(
            "{slow} held-out onsets ≥ 30 ms (guard {HELDOUT_SLOW_MAX})"
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
