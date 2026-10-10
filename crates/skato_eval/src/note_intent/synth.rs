//! The synthetic scenarios (exact ground truth) with the Labs options off
//! and on (`skato_eval note-intent synth`): the DESIGN §4 control numbers
//! two ways.
//!
//! - **Proxy**, as the note-intent study measured them: the cached detector
//!   stream replayed through the control law, and the input's reference track
//!   plus the applied shift as the output (no shifter). Comparable with the
//!   audited synthetic table.
//! - **Engine**: a real render through `skato_engine` with the option set,
//!   measured with the reference tracker on the output, as the suite does.
//!
//! Per row: stable-frame tuning median and p95 (the suite's rule), settling
//! (the slowest onset and how many take 30 ms or more) and target accuracy
//! (frames inside a note whose target is the true note). The study's
//! no-new-failure rule judges each row: a p95 under 15 c stays under 15 c,
//! a higher one rises at most 2 c, and no more onsets settle in 30 ms or more.

use std::fmt::Write as _;

use skato_core::{Settings, frame_flags};

use super::System;
use super::cache::{self, Config};
use super::replay;
use crate::metrics::{self, TuningOptions};
use crate::reference::{PitchTrack, ReferenceConfig, ReferenceTracker};
use crate::render::BlockPattern;
use crate::scenarios::{self, Scenario};
use crate::suite;
use crate::synth::{NoteKind, Signal};

/// The scenarios the note-intent study gated on.
pub const SCENARIOS: &[&str] = &[
    "melody_male",
    "melody_female",
    "low_male",
    "high_soprano",
    "steps",
    "vibrato",
    "vibrato_onset",
    "heldout_pentatonic",
    "heldout_room",
];

/// Sample rate of the scenarios.
pub const RATE: f32 = 48_000.0;

/// The measures of one way of rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Measures {
    pub tune_median: f64,
    pub tune_p95: f64,
    pub settle_max_ms: f64,
    /// Onsets settling in 30 ms or more (or never).
    pub slow_onsets: usize,
    pub onsets: usize,
    /// Frames inside a note whose target is the true note, %.
    pub acc_target: f64,
}

/// One scenario × config × retune × system.
#[derive(Clone, Debug)]
pub struct SynthRow {
    pub scenario: &'static str,
    pub config: Config,
    pub retune_ms: f32,
    pub system: System,
    pub proxy: Measures,
    pub engine: Measures,
}

fn settle(s: Option<&metrics::SettlingStats>) -> (f64, usize, usize) {
    match s {
        Some(s) => {
            let ms: Vec<f64> = s.per_note.iter().map(|x| x.map_or(f64::INFINITY, |v| v * 1000.0)).collect();
            (1000.0 * s.max_s, ms.iter().filter(|v| **v >= 30.0).count(), ms.len())
        }
        None => (f64::NAN, 0, 0),
    }
}

/// Target accuracy: `(pos, note)` per control (`pos` = input sample of its content).
fn target_accuracy(sig: &Signal, controls: impl Iterator<Item = (usize, Option<i32>)>) -> f64 {
    let (mut f, mut ok) = (0u64, 0u64);
    for (p, note) in controls {
        let tgt = sig.target_cents.get(p).copied().unwrap_or(f32::NAN);
        let in_note = sig.notes.iter().any(|n| n.kind == NoteKind::Note && p >= n.start && p < n.end);
        if in_note && tgt.is_finite() {
            f += 1;
            ok += u64::from(note == Some((f64::from(tgt) / 100.0).round() as i32));
        }
    }
    super::metrics::pct(ok, f)
}

fn tuning_options(scn: &Scenario, settings: &Settings) -> TuningOptions {
    TuningOptions {
        include_sweeps: scn.measures.sweep,
        stable_window_s: 2.0 * f64::from(settings.retune_ms.max(0.0)) * 1e-3,
        legal: Some(scenarios::legal_classes(&scn.settings)),
        legal_tolerance_cents: f64::from(settings.tolerance_cents),
        ..TuningOptions::default()
    }
}

/// The proxy measures of one run.
fn proxy(scn: &Scenario, stream: &cache::EstStream, in_track: &PitchTrack, settings: &Settings) -> Measures {
    let sig = &scn.signal;
    let run = replay::replay(stream, settings);
    let out = run.proxy_track(in_track);
    let m = scn.measures;
    let tuning = m.tuning.then(|| metrics::tuning(sig, &out, &tuning_options(scn, settings)));
    let tel = run.telemetry();
    let settling = m.settling.then(|| metrics::settling_with(sig, &out, &suite::settling_options(settings, m.settling_legato), Some((&tel, run.h.latency))));
    let (settle_max_ms, slow_onsets, onsets) = settle(settling.as_ref());
    let sr = f64::from(sig.sample_rate);
    let controls = (0..run.recs.len()).filter_map(|k| {
        let p = (run.t_mid(k) * sr).round();
        (p >= 0.0).then(|| (p as usize, if run.target_voiced(k) { run.recs[k].note } else { None }))
    });
    Measures {
        tune_median: tuning.as_ref().map_or(f64::NAN, |t| t.median_abs),
        tune_p95: tuning.as_ref().map_or(f64::NAN, |t| t.p95_abs),
        settle_max_ms,
        slow_onsets,
        onsets,
        acc_target: target_accuracy(sig, controls),
    }
}

/// The real-engine measures of one run.
fn engine(scn: &Scenario, in_track: &PitchTrack, cfg: Config, settings: Settings) -> Measures {
    let (r, rendered) = suite::evaluate_with(scn, Some(in_track), cfg.mode, &BlockPattern::Fixed(128), settings);
    let (settle_max_ms, slow_onsets, onsets) = settle(r.settling.as_ref());
    let d = u64::from(rendered.latency);
    let controls = rendered.telemetry.iter().filter_map(|f| {
        let has = f.flags & frame_flags::VOICED != 0 && f.target.is_finite() && f.target > 0.0;
        let note = has.then(|| ((f.target - settings.detune_cents) / 100.0).round() as i32);
        f.sample_pos.checked_sub(d).map(|p| (p as usize, note))
    });
    Measures {
        tune_median: r.tuning.map_or(f64::NAN, |t| t.median_abs),
        tune_p95: r.tuning.map_or(f64::NAN, |t| t.p95_abs),
        settle_max_ms,
        slow_onsets,
        onsets,
        acc_target: target_accuracy(&scn.signal, controls),
    }
}

/// Every scenario × config × retune × system.
pub fn run(cfgs: &[Config], systems: &[System], threads: usize) -> Vec<SynthRow> {
    let scns: Vec<Scenario> = SCENARIOS.iter().filter_map(|n| scenarios::build(n, RATE)).collect();
    let tracks: Vec<PitchTrack> =
        super::parallel(scns.len(), threads, |i| ReferenceTracker::new(RATE, ReferenceConfig::default()).track(&scns[i].signal.audio));
    let mut jobs: Vec<(usize, Config)> = Vec::new();
    for i in 0..scns.len() {
        for &c in cfgs {
            jobs.push((i, c));
        }
    }
    let rows = super::parallel(jobs.len(), threads, |j| {
        let (i, cfg) = jobs[j];
        let scn = &scns[i];
        let det = Settings { tier: cfg.tier, ..scn.settings };
        let stream = cache::run_detector(&scn.signal.audio, RATE, cfg, scn.voice_range, &det);
        let mut out = Vec::new();
        for &retune in scn.retunes {
            for &system in systems {
                let mut s = suite::run_settings(scn, cfg.mode, cfg.tier, retune);
                system.apply(&mut s);
                out.push(SynthRow {
                    scenario: scn.name,
                    config: cfg,
                    retune_ms: retune,
                    system,
                    proxy: proxy(scn, &stream, &tracks[i], &s),
                    engine: engine(scn, &tracks[i], cfg, s),
                });
            }
        }
        out
    });
    rows.into_iter().flatten().collect()
}

/// The study's no-new-failure rule for `on` against `off`.
pub fn passes(off: &Measures, on: &Measures) -> bool {
    let tune_ok = if !off.tune_p95.is_finite() {
        true
    } else if off.tune_p95 < 15.0 {
        on.tune_p95 < 15.0
    } else {
        on.tune_p95 <= off.tune_p95 + 2.0
    };
    tune_ok && on.slow_onsets <= off.slow_onsets
}

fn f(v: f64) -> String {
    if v.is_finite() { format!("{v:.2}") } else { "-".into() }
}

/// A Markdown table: each system against the baseline, proxy and engine.
pub fn report(rows: &[SynthRow]) -> String {
    let mut s = String::from(
        "| scenario | config | retune ms | system | p95 c proxy | p95 c engine | target acc % proxy | target acc % engine | settle max ms proxy | slow onsets proxy | settle max ms engine | slow onsets engine | rule proxy / engine |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for r in rows.iter().filter(|r| r.system != System::Baseline) {
        let Some(b) =
            rows.iter().find(|b| b.system == System::Baseline && b.scenario == r.scenario && b.config == r.config && b.retune_ms == r.retune_ms)
        else {
            continue;
        };
        let pair = |a: f64, z: f64| format!("{} → {}", f(a), f(z));
        let ok = |p: bool| if p { "pass" } else { "FAIL" };
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} → {} | {} | {} → {} | {} / {} |",
            r.scenario,
            r.config.label(),
            r.retune_ms,
            r.system.label(),
            pair(b.proxy.tune_p95, r.proxy.tune_p95),
            pair(b.engine.tune_p95, r.engine.tune_p95),
            pair(b.proxy.acc_target, r.proxy.acc_target),
            pair(b.engine.acc_target, r.engine.acc_target),
            pair(b.proxy.settle_max_ms, r.proxy.settle_max_ms),
            b.proxy.slow_onsets,
            r.proxy.slow_onsets,
            pair(b.engine.settle_max_ms, r.engine.settle_max_ms),
            b.engine.slow_onsets,
            r.engine.slow_onsets,
            ok(passes(&b.proxy, &r.proxy)),
            ok(passes(&b.engine, &r.engine)),
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(p95: f64, slow: usize) -> Measures {
        Measures { tune_p95: p95, slow_onsets: slow, ..Measures::default() }
    }

    #[test]
    fn the_no_new_failure_rule() {
        assert!(passes(&m(5.0, 0), &m(14.9, 0)));
        assert!(!passes(&m(5.0, 0), &m(15.1, 0)));
        assert!(passes(&m(20.0, 1), &m(22.0, 1)));
        assert!(!passes(&m(20.0, 1), &m(22.5, 1)));
        assert!(!passes(&m(5.0, 0), &m(5.0, 1)));
        assert!(passes(&m(f64::NAN, 0), &m(f64::NAN, 0)));
    }
}
