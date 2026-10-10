//! Note-intent harness: how well the note decision picks the note a singer
//! means, on real singing (vocadito), with the Labs options off and on
//! (`docs/research/note-intent.md`).
//!
//! The detector's per-hop stream for every vocadito file is computed once
//! and cached under `out/` ([`cache`]; a detector change rebuilds it). The
//! control law is then replayed over it hop by hop ([`replay`]; the telemetry
//! is bit-identical to an engine render) with each system's settings, and
//! scored against both annotators' notes ([`metrics`]): frame accuracy where
//! the annotators agree, per-note accuracy, warble per minute, the
//! stable-frame p95 and note agreement of a proxy output (input plus the
//! applied shift; there is no shifter), an onset-glitch proxy and the legato
//! decision latency. The [`frontier`] yardstick compares each system with
//! the Pareto envelope of the baseline's Tolerance sweep.
//!
//! The files are split once and for all into DEV (28 files, 20 singers) and
//! TEST (12 files, 9 singers) by singer ([`data::SPLITS_TXT`]): models and
//! choices are made on DEV, and TEST is read once per release.
//!
//! `skato_eval note-intent` drives [`run`] and writes `report.md` and
//! `points.csv`; its `dump` / `dump-synth` write the learned veto's training
//! data ([`train`]) and `convert-model` the embedded model format ([`model`]).

pub mod cache;
pub mod data;
pub mod frontier;
pub mod metrics;
pub mod model;
pub mod replay;
pub mod synth;
pub mod train;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use skato_core::{ReferenceMode, Settings};

use crate::vocadito::{Condition, RunConfig, condition_settings};
use cache::{Config, Fingerprints};
use data::{FileCtx, Split, Splits};
use frontier::{CurvePoint, Gaps};
use metrics::NoteMetrics;

/// Run `f(i)` for `i in 0..n` on `threads` workers; results in index order.
pub fn parallel<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<T>>> = Mutex::new((0..n).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, n.max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let v = f(i);
                    if let Ok(mut s) = slots.lock()
                        && let Some(slot) = s.get_mut(i)
                    {
                        *slot = Some(v);
                    }
                }
            });
        }
    });
    slots.into_inner().unwrap_or_default().into_iter().flatten().collect()
}

/// A note-decision system: the baseline or a Labs option.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum System {
    /// Both Labs options off (the control law as shipped).
    Baseline,
    /// Learned note hold on.
    NoteHold,
    /// Follow singer's tuning on.
    Follow,
}

impl System {
    pub fn label(self) -> &'static str {
        match self {
            System::Baseline => "baseline",
            System::NoteHold => "note_hold",
            System::Follow => "follow",
        }
    }

    pub fn parse(s: &str) -> Option<System> {
        [System::Baseline, System::NoteHold, System::Follow].into_iter().find(|x| x.label() == s)
    }

    /// The system's settings on top of a condition's.
    pub fn apply(self, s: &mut Settings) {
        s.note_hold = self == System::NoteHold;
        s.reference = if self == System::Follow { ReferenceMode::FollowSinger } else { ReferenceMode::A440 };
    }
}

/// What to evaluate.
#[derive(Clone, Debug)]
pub struct NoteIntentOptions {
    /// The vocadito dataset (`Audio/`, `Annotations/`, metadata).
    pub data: PathBuf,
    /// Where `report.md` and `points.csv` go.
    pub out: PathBuf,
    /// Detector-stream cache (default `<out>/cache`).
    pub cache: Option<PathBuf>,
    pub split: Split,
    /// Restrict to these files (within the split).
    pub files: Option<Vec<u32>>,
    pub configs: Vec<Config>,
    pub conditions: Vec<Condition>,
    /// Systems compared with the baseline.
    pub systems: Vec<System>,
    /// Hard-tune Tolerance of the comparison, cents (the plugin default is 8;
    /// Natural keeps its own 15 c).
    pub tolerance: f32,
    /// The baseline's Tolerance sweep for the frontier (hard tune; empty: no
    /// frontier). The other systems are swept too, for the sweep-mean gaps.
    pub sweep: Vec<f32>,
    pub threads: usize,
}

impl Default for NoteIntentOptions {
    fn default() -> Self {
        Self {
            data: PathBuf::from("data/vocadito"),
            out: PathBuf::from("out/note-intent"),
            cache: None,
            split: Split::Test,
            files: None,
            configs: vec![Config::LIVE_HQ, Config::STUDIO_HQ],
            conditions: vec![Condition::HardChromatic, Condition::HardKey, Condition::NaturalKey],
            systems: vec![System::NoteHold],
            tolerance: 8.0,
            sweep: vec![0.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 45.0],
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        }
    }
}

impl NoteIntentOptions {
    pub fn cache_dir(&self) -> PathBuf {
        self.cache.clone().unwrap_or_else(|| self.out.join("cache"))
    }
}

/// One evaluated point: a system × config × condition × Tolerance, pooled
/// over the files, and per singer.
#[derive(Clone, Debug)]
pub struct Point {
    pub system: System,
    pub config: Config,
    pub condition: Condition,
    pub tolerance: f32,
    pub pooled: NoteMetrics,
    pub per_singer: BTreeMap<String, NoteMetrics>,
}

/// Frontier gaps of a system in one config × condition.
#[derive(Clone, Debug)]
pub struct FrontierRow {
    pub system: System,
    pub config: Config,
    pub condition: Condition,
    /// At the comparison's Tolerance.
    pub at: Gaps,
    /// Mean of the unclamped gaps over the system's own sweep.
    pub sweep_mean: [f64; 5],
}

/// Everything a run produced.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub split: Split,
    pub files: Vec<u32>,
    pub singers: Vec<String>,
    pub minutes: f64,
    pub tolerance: f32,
    pub points: Vec<Point>,
    pub frontier: Vec<FrontierRow>,
    /// Files whose cache had to be (re)built.
    pub rebuilt: usize,
}

impl Outcome {
    /// The pooled point of a system × config × condition at a Tolerance.
    pub fn point(&self, system: System, config: Config, condition: Condition, tolerance: f32) -> Option<&Point> {
        self.points.iter().find(|p| p.system == system && p.config == config && p.condition == condition && p.tolerance == tolerance)
    }

    /// The comparison point (the run's Tolerance for hard tune, the
    /// condition's own for Natural).
    pub fn main_point(&self, system: System, config: Config, condition: Condition) -> Option<&Point> {
        self.point(system, config, condition, tolerance_for(condition, self.tolerance))
    }
}

/// Natural keeps the condition's own Tolerance (15 c / 30 ms); hard tune
/// takes `t`.
fn is_hard(c: Condition) -> bool {
    matches!(c, Condition::HardChromatic | Condition::HardKey)
}

fn tolerance_for(c: Condition, t: f32) -> f32 {
    if is_hard(c) { t } else { 15.0 }
}

/// Settings of a job (None when the condition needs a key the file lacks).
pub fn job_settings(ctx: &FileCtx, cfg: Config, cond: Condition, system: System, tolerance: f32) -> Option<Settings> {
    let mut s = condition_settings(cond, RunConfig::new(cfg.mode, cfg.tier), ctx.key.as_ref())?;
    if is_hard(cond) {
        s.tolerance_cents = tolerance;
    }
    system.apply(&mut s);
    Some(s)
}

/// Evaluate `jobs` (system, config, condition, Tolerance) on `ctxs`.
pub fn evaluate(
    ctxs: &[FileCtx],
    cache: &Path,
    fp: &Fingerprints,
    jobs: &[(System, Config, Condition, f32)],
    threads: usize,
) -> Result<Vec<Point>, String> {
    let work: Vec<(usize, usize)> = (0..ctxs.len()).flat_map(|f| (0..jobs.len()).map(move |j| (f, j))).collect();
    let res = parallel(work.len(), threads, |w| -> Result<Option<(usize, usize, NoteMetrics)>, String> {
        let (fi, ji) = work[w];
        let ctx = &ctxs[fi];
        let (system, cfg, cond, tol) = jobs[ji];
        let Some(s) = job_settings(ctx, cfg, cond, system, tol) else { return Ok(None) };
        let stream = cache::load_stream(cache, fp, cfg, ctx.id)?;
        let run = replay::replay(&stream, &s);
        Ok(Some((fi, ji, metrics::score(ctx, &run))))
    });
    let mut points: Vec<Point> = jobs
        .iter()
        .map(|&(system, config, condition, tolerance)| Point {
            system,
            config,
            condition,
            tolerance,
            pooled: NoteMetrics::default(),
            per_singer: BTreeMap::new(),
        })
        .collect();
    for r in res {
        if let Some((fi, ji, m)) = r? {
            let p = &mut points[ji];
            p.pooled.merge(&m);
            p.per_singer.entry(ctxs[fi].singer.clone()).or_default().merge(&m);
        }
    }
    Ok(points)
}

/// Load the contexts of `files` (parallel).
pub fn load_contexts(data: &Path, cache: &Path, files: &[u32], splits: &Splits, threads: usize) -> Result<Vec<FileCtx>, String> {
    parallel(files.len(), threads, |i| FileCtx::load(data, cache, files[i], splits)).into_iter().collect()
}

/// Run the whole evaluation (see the module docs).
pub fn run(opts: &NoteIntentOptions, progress: &(dyn Fn(&str) + Sync)) -> Result<Outcome, String> {
    let splits = Splits::committed();
    let mut files = splits.files(opts.split);
    if let Some(only) = &opts.files {
        files.retain(|f| only.contains(f));
    }
    if files.is_empty() {
        return Err("no files to evaluate".into());
    }
    if !opts.data.join("Annotations").is_dir() {
        return Err(format!("{}: no vocadito dataset (Annotations/ missing)", opts.data.display()));
    }
    let cache_dir = opts.cache_dir();
    progress("fingerprinting the detector");
    let fp = Fingerprints::compute(&opts.configs);
    let rebuilt = cache::prepare(&opts.data, &cache_dir, &fp, &files, &opts.configs, opts.threads, progress)?;
    let ctxs = load_contexts(&opts.data, &cache_dir, &files, &splits, opts.threads)?;
    let mut jobs = Vec::new();
    let mut systems = vec![System::Baseline];
    systems.extend(opts.systems.iter().copied().filter(|s| *s != System::Baseline));
    for &cfg in &opts.configs {
        for &cond in &opts.conditions {
            let mut tols = vec![tolerance_for(cond, opts.tolerance)];
            if is_hard(cond) {
                tols.extend(opts.sweep.iter().copied());
            }
            tols.sort_by(f32::total_cmp);
            tols.dedup();
            for &system in &systems {
                for &t in &tols {
                    jobs.push((system, cfg, cond, t));
                }
            }
        }
    }
    progress(&format!("replaying {} jobs on {} files", jobs.len(), ctxs.len()));
    let points = evaluate(&ctxs, &cache_dir, &fp, &jobs, opts.threads)?;
    let mut out = Outcome {
        split: opts.split,
        files: files.clone(),
        singers: splits.by_singer(&files).into_keys().collect(),
        minutes: ctxs.iter().map(|c| c.dur_s).sum::<f64>() / 60.0,
        tolerance: opts.tolerance,
        points,
        frontier: Vec::new(),
        rebuilt,
    };
    out.frontier = frontier_rows(&out, &systems);
    Ok(out)
}

/// Frontier gaps of every non-baseline system in every hard-tune config ×
/// condition that has a baseline sweep.
fn frontier_rows(o: &Outcome, systems: &[System]) -> Vec<FrontierRow> {
    let mut rows = Vec::new();
    let mut keys: Vec<(Config, Condition)> = o.points.iter().map(|p| (p.config, p.condition)).collect();
    keys.dedup();
    let mut seen = Vec::new();
    for (cfg, cond) in keys {
        if !is_hard(cond) || seen.contains(&(cfg, cond)) {
            continue;
        }
        seen.push((cfg, cond));
        let sweep = |s: System| -> Vec<CurvePoint> {
            let mut v: Vec<CurvePoint> = o
                .points
                .iter()
                .filter(|p| p.system == s && p.config == cfg && p.condition == cond)
                .map(|p| CurvePoint::from_metrics(p.tolerance, &p.pooled))
                .collect();
            v.sort_by(|a, b| a.tolerance.total_cmp(&b.tolerance));
            v
        };
        let base = sweep(System::Baseline);
        if base.len() < 2 {
            continue;
        }
        for &system in systems.iter().filter(|s| **s != System::Baseline) {
            let own = sweep(system);
            let Some(at) = own.iter().find(|p| p.tolerance == o.tolerance) else { continue };
            let all: Vec<Gaps> = own.iter().map(|p| frontier::gaps(&base, p)).collect();
            rows.push(FrontierRow { system, config: cfg, condition: cond, at: frontier::gaps(&base, at), sweep_mean: frontier::mean_gaps(&all) });
        }
    }
    rows
}

/// Columns of the main table: (metric, header, decimals).
const MAIN_COLS: &[(&str, &str)] = &[
    ("acc_agree", "frame acc %"),
    ("note_dom", "note %"),
    ("warble_min", "warble /min"),
    ("p95", "p95 c"),
    ("pub_agree", "agreeP %"),
    ("glitch_tgt", "glitchT %"),
    ("lat_leg_med", "legato lat med ms"),
    ("lat_leg_p90", "legato lat p90 ms"),
    ("lat_missed", "missed %"),
    ("switch_min", "switches /min"),
];

fn f1(v: f64) -> String {
    if v.is_finite() { format!("{v:.1}") } else { "-".into() }
}

fn signed(v: f64) -> String {
    if v.is_finite() { format!("{v:+.1}") } else { "-".into() }
}

/// The Markdown report.
pub fn report(o: &Outcome, opts: &NoteIntentOptions) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# Note intent on real singing (vocadito)\n");
    let _ = writeln!(
        s,
        "Split **{}**: {} files ({}), {} singers, {:.1} min. Hard tune at Tolerance {} c (Natural at its own 15 c). Systems: {}. \
         Replay of the cached detector stream through the control law; p95 and note agreement are measured on a proxy output \
         (input + shift, no shifter). Definitions: `skato_eval::note_intent::metrics`.\n",
        o.split.label(),
        o.files.len(),
        o.files.iter().map(u32::to_string).collect::<Vec<_>>().join(", "),
        o.singers.len(),
        o.minutes,
        o.tolerance,
        std::iter::once(System::Baseline).chain(opts.systems.iter().copied()).map(System::label).collect::<Vec<_>>().join(", "),
    );
    let mut keys: Vec<(Config, Condition)> = Vec::new();
    for p in &o.points {
        if !keys.contains(&(p.config, p.condition)) {
            keys.push((p.config, p.condition));
        }
    }
    let _ = writeln!(s, "## At the comparison Tolerance\n");
    let _ = write!(s, "| config | condition | system |");
    for (_, h) in MAIN_COLS {
        let _ = write!(s, " {h} |");
    }
    let _ = writeln!(s, "\n|{}", "---|".repeat(3 + MAIN_COLS.len()));
    for &(cfg, cond) in &keys {
        let base = o.main_point(System::Baseline, cfg, cond);
        for p in o.points.iter().filter(|p| p.config == cfg && p.condition == cond && p.tolerance == tolerance_for(cond, o.tolerance)) {
            let _ = write!(s, "| {} | {} | {} |", cfg.label(), cond.label(), p.system.label());
            for (m, _) in MAIN_COLS {
                let v = p.pooled.get(m);
                match base {
                    Some(b) if p.system != System::Baseline => {
                        let _ = write!(s, " {} ({}) |", f1(v), signed(v - b.pooled.get(m)));
                    }
                    _ => {
                        let _ = write!(s, " {} |", f1(v));
                    }
                }
            }
            let _ = writeln!(s);
        }
    }
    if let Some(p) = o.points.first() {
        let _ = writeln!(
            s,
            "\nInter-annotator ceiling ({} {}): frames {} %, notes {} %.\n",
            p.config.label(),
            p.condition.label(),
            f1(p.pooled.get("ceil_frames")),
            f1(p.pooled.get("ceil_notes"))
        );
    }
    if !o.frontier.is_empty() {
        let _ = writeln!(s, "## Frontier yardstick\n");
        let _ = writeln!(
            s,
            "Distance to the Pareto envelope of the baseline's Tolerance sweep ({} c); positive beats every Tolerance of the \
             baseline, `*` = clamped to the sweep's range. At the comparison Tolerance, and the mean over the system's own sweep \
             (unclamped gaps only).\n",
            opts.sweep.iter().map(|t| format!("{t}")).collect::<Vec<_>>().join("/")
        );
        let _ = write!(s, "| config | condition | system | at |");
        for n in Gaps::NAMES {
            let _ = write!(s, " {n} |");
        }
        let _ = writeln!(s, "\n|{}", "---|".repeat(4 + Gaps::NAMES.len()));
        for r in &o.frontier {
            let _ = write!(s, "| {} | {} | {} | {} c |", r.config.label(), r.condition.label(), r.system.label(), o.tolerance);
            for g in r.at.all() {
                let _ = write!(s, " {}{} |", signed(g.value), if g.clamped { "*" } else { "" });
            }
            let _ = write!(s, "\n| {} | {} | {} | sweep mean |", r.config.label(), r.condition.label(), r.system.label());
            for v in r.sweep_mean {
                let _ = write!(s, " {} |", signed(v));
            }
            let _ = writeln!(s);
        }
        let _ = writeln!(s);
    }
    // Per singer, for every non-baseline system at the comparison point.
    for &(cfg, cond) in keys.iter().filter(|(_, c)| is_hard(*c)) {
        let Some(base) = o.main_point(System::Baseline, cfg, cond) else { continue };
        for p in o.points.iter().filter(|p| p.system != System::Baseline && p.config == cfg && p.condition == cond && p.tolerance == o.tolerance) {
            let _ = writeln!(s, "## Per singer: {} vs baseline, {} {}\n", p.system.label(), cfg.label(), cond.label());
            let _ = writeln!(s, "| singer | Δ frame acc | Δ note % | Δ warble /min | Δ p95 c | Δ agreeP |\n|---|---|---|---|---|---|");
            let mut better = [0usize; 4];
            for (singer, m) in &p.per_singer {
                let Some(b) = base.per_singer.get(singer) else { continue };
                let d = |k: &str| m.get(k) - b.get(k);
                let row = [d("acc_agree"), d("note_dom"), d("warble_min"), d("p95"), d("pub_agree")];
                better[0] += usize::from(row[0] > 0.0);
                better[1] += usize::from(row[1] >= 0.0);
                better[2] += usize::from(row[2] < 0.0);
                better[3] += usize::from(row[3] < 0.0);
                let _ = writeln!(s, "| {singer} | {} | {} | {} | {} | {} |", signed(row[0]), signed(row[1]), signed(row[2]), signed(row[3]), signed(row[4]));
            }
            let n = p.per_singer.len();
            let _ = writeln!(
                s,
                "\nFrame accuracy higher for {}/{n} singers, per-note accuracy not lower for {}/{n}, warble lower for {}/{n}, p95 lower for {}/{n}.\n",
                better[0], better[1], better[2], better[3]
            );
        }
    }
    let hold: Vec<&Point> = o.points.iter().filter(|p| p.system == System::NoteHold && p.tolerance == tolerance_for(p.condition, o.tolerance)).collect();
    if !hold.is_empty() {
        let _ = writeln!(s, "## Learned note hold activity\n");
        let _ = writeln!(s, "| config | condition | switches judged, % of hops | held back, % of judged | release freezes |\n|---|---|---|---|---|");
        for p in hold {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {:.0} |",
                p.config.label(),
                p.condition.label(),
                f1(p.pooled.get("hold_proposals_pct")),
                f1(p.pooled.get("hold_vetoed_pct")),
                p.pooled.get("hold_frozen")
            );
        }
        let _ = writeln!(s);
    }
    s
}

/// Every point's scalars, one CSV row each.
pub fn points_csv(o: &Outcome) -> String {
    let names: Vec<&str> = NoteMetrics::default().scalars().into_iter().map(|x| x.0).collect();
    let mut s = format!("system,config,condition,tolerance,{}\n", names.join(","));
    for p in &o.points {
        let vals: Vec<String> = p.pooled.scalars().into_iter().map(|(_, v)| if v.is_finite() { format!("{v:.4}") } else { String::new() }).collect();
        let _ = writeln!(s, "{},{},{},{},{}", p.system.label(), p.config.label(), p.condition.label(), p.tolerance, vals.join(","));
    }
    s
}

/// Write `report.md` and `points.csv` to `opts.out`.
pub fn write_all(o: &Outcome, opts: &NoteIntentOptions) -> Result<(), String> {
    std::fs::create_dir_all(&opts.out).map_err(|e| format!("{}: {e}", opts.out.display()))?;
    let w = |name: &str, text: String| std::fs::write(opts.out.join(name), text).map_err(|e| format!("{name}: {e}"));
    w("report.md", report(o, opts))?;
    w("points.csv", points_csv(o))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systems_set_exactly_their_option() {
        let mut s = Settings::default();
        System::NoteHold.apply(&mut s);
        assert!(s.note_hold && s.reference == ReferenceMode::A440);
        System::Follow.apply(&mut s);
        assert!(!s.note_hold && s.reference == ReferenceMode::FollowSinger);
        System::Baseline.apply(&mut s);
        assert_eq!(s, Settings::default());
        for sys in [System::Baseline, System::NoteHold, System::Follow] {
            assert_eq!(System::parse(sys.label()), Some(sys));
        }
    }

    #[test]
    fn parallel_keeps_order() {
        assert_eq!(parallel(10, 3, |i| i * i), (0..10).map(|i| i * i).collect::<Vec<_>>());
        assert!(parallel(0, 3, |i| i).is_empty());
    }
}
