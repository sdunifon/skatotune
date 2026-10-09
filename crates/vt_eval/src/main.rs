//! `vt_eval` command line: synthesise test signals, render WAVs through the
//! engine, run the evaluation suite, check latency and benchmark callbacks.
//!
//! ```text
//! vt_eval synth <scenario|list> <out.wav> [--sr 48000]
//! vt_eval render <in.wav> <out.wav> [--mode live] [--range altotenor] [--tier hq] [--engine natural]
//!                [--retune 20] [--key C] [--scale major] [...]   (see `vt_eval help`)
//! vt_eval track <in.wav> <out.csv>
//! vt_eval suite [--out out/] [--sr 48000] [--scenario a,b] [--modes ..] [--tiers ..] [--wavs hq] [--quick]
//! vt_eval latency [--sr 44100,48000,96000] [--no-content]
//! vt_eval rt-bench [--sr 48000] [--seconds 10] [--buffers 32,64,128,256,512] [--modes ..] [--tiers ..]
//! vt_eval vocadito [--data data/vocadito] [--out out/vocadito] [--modes ..] [--tiers ..] [--engine ..]
//!                  [--files 1,2] [--conditions ..] [--range auto] [--listen ..] [--dump ..] [--no-wavs]
//! ```
//!
//! Exit codes: 0 success, 1 a gate failed (latency mismatch, rt-bench over
//! budget, suite or real-singing gate failure), 2 usage or I/O error.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use vt_core::{
    EngineConfig, EngineKind, LatencyMode, QualityTier, ScaleKind, Settings, VoiceRange,
};
use vt_eval::reference::{ReferenceConfig, ReferenceTracker};
use vt_eval::render::{self, BlockPattern, RenderOptions};
use vt_eval::rtbench::{self, RtBenchOptions};
use vt_eval::suite::{self, SuiteOptions, WavPolicy};
use vt_eval::vocadito::{self, Condition, RunConfig, VocaditoOptions};
use vt_eval::{scenarios, wav};

const USAGE: &str = "\
vt_eval: Vocal Tune evaluation harness

USAGE:
  vt_eval synth <scenario|list> <out.wav> [--sr 48000]
      Write a synthetic scenario (plus <out>.truth.csv with f0/target at 1 ms).
  vt_eval render <in.wav> <out.wav> [OPTIONS]
      Process a WAV through the engine (latency-compensated unless --no-compensate).
      --mode livemin|live|liveplus|studio   --range lowmale|altotenor|soprano|wide
      --tier eco|hq|max   --engine classic|natural   --retune MS   --transition MS
      --key C..B|0-11   --scale chromatic|major|minor|harmonicminor|melodicminor|dorian|
            phrygian|lydian|mixolydian|locrian|majorpentatonic|minorpentatonic|blues
      --amount 0-100   --humanize 0-100   --flex 0-100   --depth -100..200   --detune C
      --transpose ST   --formant ST   --tolerance C   --tracking 0-100   --gate DB
      --live-plus-ms MS   --studio-ms MS   --block N   --no-preserve-vibrato   --no-compensate
  vt_eval track <in.wav> <out.csv>
      Offline reference pitch track (1 ms frames) of the first channel.
  vt_eval suite [--out out] [--sr 48000] [--scenario a,b] [--modes live,studio] [--tiers hq]
                [--wavs none|hq|all] [--block N] [--threads N] [--quick]
      All scenarios x modes x tiers: WAVs, out/results.csv and out/report.md.
  vt_eval latency [--sr 44100,48000,96000] [--no-content]
      Measured vs reported latency for every mode x voice range (exit 1 on mismatch),
      then the latency gauge: the delay of voiced content (D + PSOLA wander) per
      mode x engine x voice range at 48 kHz, i.e. what a performer hears.
  vt_eval rt-bench [--sr 48000] [--seconds 10] [--buffers 32,64,128,256,512] [--modes ..] [--tiers ..]
      process() tail latency per config; exit 1 if p99.9 > 10% of the period at 32/64/128.
  vt_eval vocadito [--data data/vocadito] [--out out/vocadito] [--modes live,studio,livemin]
                   [--tiers eco,hq,max] [--engine classic|natural] [--files 1,2,3]
                   [--conditions detect,hard_chromatic,natural_chromatic,hard_key,natural_key]
                   [--range auto|lowmale|altotenor|soprano|wide] [--listen 26,1,9,2] [--no-wavs]
                   [--block 256] [--threads N] [--dump 1,2]
      Real-singing evaluation on the vocadito dataset (CC BY 4.0; download it to
      data/vocadito: Audio/vocadito_N.wav, Annotations/F0, Annotations/Notes, metadata):
      detector vs the F0 annotation (offline reference tracker as baseline), voiced
      content delay (D + PSOLA wander) on the amount-0 render, and hard-tune / Natural
      correction measured with the reference tracker. Default matrix: Live and Studio x
      Eco/HQ/Max (Natural) plus Live-Min HQ (Classic) x all five conditions; --modes /
      --tiers select a full grid instead, --engine overrides Natural/Classic (not in
      Live-Min), --range forces one voice range (default: per file from the annotation).
      --files limits the tracks, --listen writes listening WAVs for those tracks,
      --dump writes 1 ms per-frame CSVs. Writes report.md, files.csv, detector.csv,
      correction.csv, segments.csv (failure spans) under --out. Prints the real-singing
      gates (Live HQ; exit 1 if one fails). Full run: about 4 min on 16 threads.
";

/// Flags that take no value.
const SWITCHES: &[&str] = &[
    "no-compensate",
    "no-preserve-vibrato",
    "quick",
    "help",
    "no-wavs",
    "no-content",
];

struct Args {
    pos: Vec<String>,
    flags: HashMap<String, String>,
}

impl Args {
    fn parse(raw: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut pos = Vec::new();
        let mut flags = HashMap::new();
        let mut it = raw.peekable();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--") {
                if let Some((k, v)) = name.split_once('=') {
                    flags.insert(k.to_string(), v.to_string());
                } else if SWITCHES.contains(&name) {
                    flags.insert(name.to_string(), String::new());
                } else {
                    let v = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
                    flags.insert(name.to_string(), v);
                }
            } else {
                pos.push(a);
            }
        }
        Ok(Self { pos, flags })
    }

    fn has(&self, k: &str) -> bool {
        self.flags.contains_key(k)
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(|s| s.as_str())
    }

    fn num(&self, k: &str) -> Result<Option<f32>, String> {
        self.get(k)
            .map(|v| {
                v.parse::<f32>()
                    .map_err(|_| format!("--{k}: not a number: {v}"))
            })
            .transpose()
    }

    fn list<T>(
        &self,
        k: &str,
        parse: fn(&str) -> Result<T, String>,
    ) -> Result<Option<Vec<T>>, String> {
        self.get(k)
            .map(|v| v.split(',').filter(|s| !s.is_empty()).map(parse).collect())
            .transpose()
    }
}

fn parse_mode(s: &str) -> Result<LatencyMode, String> {
    Ok(
        match s.to_ascii_lowercase().replace(['-', '_', '+'], "").as_str() {
            "livemin" | "min" => LatencyMode::LiveMin,
            "live" => LatencyMode::Live,
            "liveplus" | "plus" => LatencyMode::LivePlus,
            "studio" => LatencyMode::Studio,
            _ => return Err(format!("unknown mode {s}")),
        },
    )
}

fn parse_tier(s: &str) -> Result<QualityTier, String> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "eco" => QualityTier::Eco,
        "hq" => QualityTier::Hq,
        "max" => QualityTier::Max,
        _ => return Err(format!("unknown tier {s}")),
    })
}

fn parse_range(s: &str) -> Result<VoiceRange, String> {
    Ok(
        match s.to_ascii_lowercase().replace(['-', '_', '/'], "").as_str() {
            "lowmale" | "low" => VoiceRange::LowMale,
            "altotenor" | "alto" | "tenor" => VoiceRange::AltoTenor,
            "soprano" => VoiceRange::Soprano,
            "wide" => VoiceRange::Wide,
            _ => return Err(format!("unknown voice range {s}")),
        },
    )
}

fn parse_engine(s: &str) -> Result<EngineKind, String> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "classic" => EngineKind::Classic,
        "natural" | "psola" => EngineKind::Natural,
        _ => return Err(format!("unknown engine {s}")),
    })
}

fn parse_key(s: &str) -> Result<u8, String> {
    if let Ok(n) = s.parse::<u8>() {
        return if n < 12 {
            Ok(n)
        } else {
            Err(format!("key {n} out of range 0-11"))
        };
    }
    const NAMES: [(&str, u8); 17] = [
        ("c", 0),
        ("c#", 1),
        ("db", 1),
        ("d", 2),
        ("d#", 3),
        ("eb", 3),
        ("e", 4),
        ("f", 5),
        ("f#", 6),
        ("gb", 6),
        ("g", 7),
        ("g#", 8),
        ("ab", 8),
        ("a", 9),
        ("a#", 10),
        ("bb", 10),
        ("b", 11),
    ];
    let l = s.to_ascii_lowercase();
    NAMES
        .iter()
        .find(|(n, _)| *n == l)
        .map(|(_, k)| *k)
        .ok_or_else(|| format!("unknown key {s}"))
}

fn parse_scale(s: &str) -> Result<ScaleKind, String> {
    let l = s.to_ascii_lowercase().replace(['-', '_', ' '], "");
    let name = |k: ScaleKind| format!("{k:?}").to_ascii_lowercase();
    if l == "minor" {
        return Ok(ScaleKind::NaturalMinor);
    }
    ScaleKind::ALL
        .iter()
        .copied()
        .find(|k| name(*k) == l)
        .ok_or_else(|| format!("unknown scale {s}"))
}

fn parse_usize(s: &str) -> Result<usize, String> {
    s.parse().map_err(|_| format!("not a count: {s}"))
}

fn parse_rate(s: &str) -> Result<f32, String> {
    s.parse::<f32>()
        .ok()
        .filter(|v| *v >= 8000.0 && *v <= 384_000.0)
        .ok_or_else(|| format!("bad sample rate {s}"))
}

fn main() -> ExitCode {
    let mut raw = std::env::args().skip(1);
    let Some(cmd) = raw.next() else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let args = match Args::parse(raw) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let result = match cmd.as_str() {
        "synth" => cmd_synth(&args),
        "render" => cmd_render(&args),
        "track" => cmd_track(&args),
        "suite" => cmd_suite(&args),
        "latency" => cmd_latency(&args),
        "rt-bench" | "rtbench" => cmd_rt_bench(&args),
        "vocadito" => cmd_vocadito(&args),
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(true)
        }
        other => Err(format!("unknown command {other}").into()),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

type CmdResult = Result<bool, Box<dyn std::error::Error>>;

fn cmd_synth(args: &Args) -> CmdResult {
    let name = args
        .pos
        .first()
        .ok_or("synth: missing scenario (or `list`)")?;
    if name == "list" {
        for n in scenarios::NAMES {
            let d = scenarios::build(n, 48_000.0).map_or("", |s| s.description);
            println!("{n:<14} {d}");
        }
        return Ok(true);
    }
    let out = PathBuf::from(args.pos.get(1).ok_or("synth: missing output path")?);
    let sr = args
        .get("sr")
        .map(parse_rate)
        .transpose()?
        .unwrap_or(48_000.0);
    let scn = scenarios::build(name, sr)
        .ok_or_else(|| format!("unknown scenario {name} (try `vt_eval synth list`)"))?;
    wav::write_wav(&out, &[&scn.signal.audio], sr)?;
    let csv = out.with_extension("truth.csv");
    wav::write_truth_csv(&csv, &scn.signal)?;
    println!(
        "wrote {} ({:.2} s) and {}",
        out.display(),
        scn.signal.duration_s(),
        csv.display()
    );
    println!(
        "suggested: --key {} --scale {:?} --range {:?}",
        scn.settings.key, scn.settings.scale, scn.voice_range
    );
    Ok(true)
}

fn render_settings(args: &Args, mode: LatencyMode) -> Result<Settings, Box<dyn std::error::Error>> {
    let key = args.get("key").map(parse_key).transpose()?.unwrap_or(0);
    let scale = args
        .get("scale")
        .map(parse_scale)
        .transpose()?
        .unwrap_or(ScaleKind::Chromatic);
    let mut s = scenarios::settings_for(key, scale);
    s.engine = render::default_engine(mode);
    if let Some(t) = args.get("tier") {
        s.tier = parse_tier(t)?;
    }
    if let Some(e) = args.get("engine") {
        s.engine = parse_engine(e)?;
    }
    let set = |v: Option<f32>, slot: &mut f32| {
        if let Some(v) = v {
            *slot = v;
        }
    };
    set(args.num("retune")?, &mut s.retune_ms);
    if let Some(t) = args.num("transition")? {
        s.transition_ms = Some(t);
    }
    set(args.num("humanize")?, &mut s.humanize);
    set(args.num("flex")?, &mut s.flex_tune);
    if let Some(a) = args.num("amount")? {
        s.amount = (a / 100.0).clamp(0.0, 1.0);
    }
    if let Some(d) = args.num("depth")? {
        s.vibrato_depth = (d / 100.0).clamp(-1.0, 2.0);
    }
    set(args.num("detune")?, &mut s.detune_cents);
    set(args.num("transpose")?, &mut s.transpose_st);
    set(args.num("formant")?, &mut s.formant_st);
    set(args.num("tolerance")?, &mut s.tolerance_cents);
    set(args.num("tracking")?, &mut s.tracking);
    set(args.num("gate")?, &mut s.gate_db);
    if args.has("no-preserve-vibrato") {
        s.preserve_vibrato = false;
    }
    Ok(s)
}

fn cmd_render(args: &Args) -> CmdResult {
    let input = PathBuf::from(args.pos.first().ok_or("render: missing input path")?);
    let output = PathBuf::from(args.pos.get(1).ok_or("render: missing output path")?);
    let (chans, sr) = wav::read_wav(&input)?;
    let mode = args
        .get("mode")
        .map(parse_mode)
        .transpose()?
        .unwrap_or(LatencyMode::Live);
    let range = args
        .get("range")
        .map(parse_range)
        .transpose()?
        .unwrap_or(VoiceRange::AltoTenor);
    let channels = chans.len().clamp(1, 2);
    let mut config = EngineConfig {
        sample_rate: sr,
        max_block: 4096,
        channels,
        latency_mode: mode,
        voice_range: range,
        ..EngineConfig::default()
    };
    if let Some(v) = args.num("live-plus-ms")? {
        config.live_plus_ms = v.clamp(5.0, 10.0);
    }
    if let Some(v) = args.num("studio-ms")? {
        config.studio_ms = v.clamp(20.0, 50.0);
    }
    let block = args
        .get("block")
        .map(parse_usize)
        .transpose()?
        .unwrap_or(256);
    let opts = RenderOptions {
        config,
        settings: render_settings(args, mode)?,
        blocks: BlockPattern::Fixed(block),
        telemetry: false,
        midi: Vec::new(),
        extra_tail: 0,
    };
    let inputs: Vec<&[f32]> = chans.iter().take(channels).map(|c| c.as_slice()).collect();
    let r = render::render_multi(&inputs, &opts);
    let outs: Vec<&[f32]> = (0..channels)
        .map(|c| {
            if args.has("no-compensate") {
                r.raw(c)
            } else {
                r.aligned(c)
            }
        })
        .collect();
    wav::write_wav(&output, &outs, sr)?;
    println!(
        "rendered {} -> {} ({} ch, {} Hz, mode {}, reported latency {} samples = {:.2} ms{})",
        input.display(),
        output.display(),
        channels,
        sr,
        suite::mode_name(mode),
        r.latency,
        r.latency as f64 / sr as f64 * 1000.0,
        if args.has("no-compensate") {
            ", uncompensated"
        } else {
            ", compensated"
        }
    );
    Ok(true)
}

fn cmd_track(args: &Args) -> CmdResult {
    let input = PathBuf::from(args.pos.first().ok_or("track: missing input path")?);
    let output = PathBuf::from(args.pos.get(1).ok_or("track: missing output csv")?);
    let (chans, sr) = wav::read_wav(&input)?;
    let x = chans.first().ok_or("track: empty file")?;
    let t = ReferenceTracker::new(sr, ReferenceConfig::default()).track(x);
    let mut s = String::from("time_s,voiced,f0_hz,cents,clarity,level_dbfs\n");
    for f in &t.frames {
        s.push_str(&format!(
            "{:.4},{},{:.4},{},{:.4},{:.1}\n",
            f.pos as f64 / sr as f64,
            u8::from(f.voiced),
            f.f0_hz,
            if f.voiced {
                format!("{:.3}", f.cents)
            } else {
                String::new()
            },
            f.clarity,
            f.level_dbfs
        ));
    }
    std::fs::write(&output, s)?;
    println!(
        "tracked {} frames ({} voiced) -> {}",
        t.frames.len(),
        t.voiced_count(),
        output.display()
    );
    Ok(true)
}

fn cmd_suite(args: &Args) -> CmdResult {
    let mut opts = SuiteOptions::default();
    if let Some(o) = args.get("out") {
        opts.out_dir = PathBuf::from(o);
    }
    if let Some(sr) = args.get("sr") {
        opts.sample_rate = parse_rate(sr)?;
    }
    if let Some(s) = args.get("scenario") {
        opts.scenarios = s.split(',').map(str::to_string).collect();
        if let Some(bad) = opts
            .scenarios
            .iter()
            .find(|n| !scenarios::NAMES.contains(&n.as_str()))
        {
            return Err(format!("unknown scenario {bad}").into());
        }
    }
    if let Some(m) = args.list("modes", parse_mode)? {
        opts.modes = m;
    }
    if let Some(t) = args.list("tiers", parse_tier)? {
        opts.tiers = t;
    }
    if args.has("quick") {
        opts.tiers = vec![QualityTier::Hq];
    }
    if let Some(w) = args.get("wavs") {
        opts.wavs = match w {
            "none" => WavPolicy::None,
            "hq" => WavPolicy::Hq,
            "all" => WavPolicy::All,
            _ => return Err(format!("--wavs: expected none|hq|all, got {w}").into()),
        };
    }
    if let Some(b) = args.get("block") {
        opts.blocks = BlockPattern::Fixed(parse_usize(b)?);
    }
    if let Some(t) = args.get("threads") {
        opts.threads = parse_usize(t)?;
    }
    std::fs::create_dir_all(&opts.out_dir)?;
    let t0 = std::time::Instant::now();
    let (results, checks) = suite::run_suite(&opts, &|line| eprintln!("{line}"))?;
    let gates = suite::gates(&results, &checks);
    let gauge = suite::latency_gauge(opts.sample_rate, &opts.modes, opts.threads);
    let csv = opts.out_dir.join("results.csv");
    let report = opts.out_dir.join("report.md");
    suite::write_csv(&csv, &results)?;
    suite::write_report(&report, &opts, &results, &checks, &gates, &gauge)?;
    println!(
        "\n{} renders in {:.1} s",
        results.len(),
        t0.elapsed().as_secs_f64()
    );
    for g in &gates {
        let r = match g.pass {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => " n/a",
        };
        println!("[{r}] {}: {} (target {})", g.name, g.value, g.target);
    }
    println!(
        "\nwrote {}, {} and WAVs under {}",
        csv.display(),
        report.display(),
        opts.out_dir.join("wav").display()
    );
    Ok(gates.iter().all(|g| g.pass != Some(false)))
}

fn cmd_latency(args: &Args) -> CmdResult {
    let rates = args
        .list("sr", parse_rate)?
        .unwrap_or_else(|| vec![44_100.0, 48_000.0, 96_000.0]);
    let rows = suite::latency_sweep(&rates);
    println!(
        "{:>7} {:>9} {:>10} {:>9} {:>9} {:>7}  result",
        "rate", "mode", "range", "reported", "measured", "peak"
    );
    let mut ok = true;
    for r in &rows {
        let (m, p) = r.measured.map_or(("-".to_string(), "-".to_string()), |m| {
            (m.lag.to_string(), format!("{:.4}", m.peak))
        });
        println!(
            "{:>7} {:>9} {:>10} {:>9} {:>9} {:>7}  {}",
            r.sample_rate,
            suite::mode_name(r.mode),
            suite::range_name(r.range),
            r.reported,
            m,
            p,
            if r.ok() { "ok" } else { "MISMATCH" }
        );
        ok &= r.ok();
    }
    println!(
        "{}",
        if ok {
            "all measured latencies match the reported ones (±1 sample)"
        } else {
            "LATENCY MISMATCH"
        }
    );
    if !args.has("no-content") {
        let sr = if rates.contains(&48_000.0) {
            48_000.0
        } else {
            rates.first().copied().unwrap_or(48_000.0)
        };
        print_gauge(sr, &suite::latency_gauge(sr, &suite::MODES, 0));
    }
    Ok(ok)
}

/// The latency gauge: reported D and the delay of voiced content (D + PSOLA
/// wander) that a performer monitoring through the plugin hears.
fn print_gauge(sr: f32, rows: &[suite::GaugeRow]) {
    println!(
        "\nvoiced content delay at {sr} Hz (amount 0, HQ; D + PSOLA wander, R§10.3):\n{:>9} {:>8} {:>10} {:>14} {:>7} {:>7} {:>7} {:>7} {:>9}",
        "mode", "engine", "range", "scenario", "D ms", "mean", "median", "p95", "max |w|/T"
    );
    for g in rows {
        let c = &g.content;
        println!(
            "{:>9} {:>8} {:>10} {:>14} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>9.2}",
            suite::mode_name(g.mode),
            suite::engine_name(g.engine),
            suite::range_name(g.range),
            g.scenario,
            c.reported_ms(),
            c.mean_delay_ms(),
            c.delay_ms(50.0),
            c.delay_ms(95.0),
            c.max_abs_wander_periods()
        );
    }
}

fn cmd_rt_bench(args: &Args) -> CmdResult {
    let mut opts = RtBenchOptions::default();
    if let Some(sr) = args.get("sr") {
        opts.sample_rate = parse_rate(sr)?;
    }
    if let Some(s) = args.num("seconds")? {
        opts.seconds = s.max(0.1) as f64;
    }
    if let Some(b) = args.list("buffers", parse_usize)? {
        opts.buffers = b;
    }
    if let Some(m) = args.list("modes", parse_mode)? {
        opts.modes = m;
    }
    if let Some(t) = args.list("tiers", parse_tier)? {
        opts.tiers = t;
    }
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: debug build; timings are meaningless (use `cargo run --release -p vt_eval -- rt-bench`)"
        );
    }
    println!(
        "{:>9} {:>4} {:>8} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8}  budget",
        "mode", "tier", "engine", "buffer", "p50 µs", "p99 µs", "p99.9 µs", "max µs", "p99.9 %"
    );
    let rows = rtbench::run(&opts, &mut |r| {
        let gate = match r.design_gate() {
            Some(true) => " (§4 gate ok)",
            Some(false) => " (§4 gate FAIL)",
            None => "",
        };
        println!(
            "{:>9} {:>4} {:>8} {:>6} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>7.2}%  {}{}",
            suite::mode_name(r.mode),
            suite::tier_name(r.tier),
            suite::engine_name(r.engine),
            r.buffer,
            r.p50_us,
            r.p99_us,
            r.p999_us,
            r.max_us,
            r.p999_fraction() * 100.0,
            if r.within_budget() { "ok" } else { "OVER" },
            gate
        );
    });
    let ok = rows.iter().all(|r| r.within_budget());
    println!(
        "{}",
        if ok {
            format!(
                "p99.9 within {:.0}% of the buffer period at {:?}",
                rtbench::BUDGET_FRACTION * 100.0,
                rtbench::GATED_BUFFERS
            )
        } else {
            "OVER BUDGET".to_string()
        }
    );
    Ok(ok)
}

fn parse_condition(s: &str) -> Result<Condition, String> {
    Condition::parse(s).ok_or_else(|| {
        format!(
            "unknown condition {s} (expected {})",
            Condition::ALL.map(|c| c.label()).join(", ")
        )
    })
}

fn parse_u32(s: &str) -> Result<u32, String> {
    s.parse().map_err(|_| format!("not a track id: {s}"))
}

fn cmd_vocadito(args: &Args) -> CmdResult {
    let mut opts = VocaditoOptions::default();
    if let Some(d) = args.get("data") {
        opts.data = PathBuf::from(d);
    }
    if let Some(o) = args.get("out") {
        opts.out = PathBuf::from(o);
    }
    let modes = args.list("modes", parse_mode)?;
    let tiers = args.list("tiers", parse_tier)?;
    if modes.is_some() || tiers.is_some() {
        let modes = modes
            .unwrap_or_else(|| vec![LatencyMode::Live, LatencyMode::Studio, LatencyMode::LiveMin]);
        let tiers =
            tiers.unwrap_or_else(|| vec![QualityTier::Eco, QualityTier::Hq, QualityTier::Max]);
        opts.configs = modes
            .iter()
            .flat_map(|m| tiers.iter().map(move |t| RunConfig::new(*m, *t)))
            .collect();
    }
    if let Some(e) = args.get("engine") {
        let e = parse_engine(e)?;
        for c in &mut opts.configs {
            if c.mode != LatencyMode::LiveMin {
                c.engine = e;
            }
        }
    }
    if let Some(f) = args.list("files", parse_u32)? {
        opts.files = f;
    }
    if let Some(c) = args.list("conditions", parse_condition)? {
        opts.conditions = c;
    }
    if let Some(l) = args.list("listen", parse_u32)? {
        opts.listen = l;
    }
    if let Some(d) = args.list("dump", parse_u32)? {
        opts.dump = d;
    }
    if let Some(r) = args.get("range")
        && r != "auto"
    {
        opts.range = Some(parse_range(r)?);
    }
    if let Some(b) = args.get("block") {
        opts.blocks = BlockPattern::Fixed(parse_usize(b)?);
    }
    if let Some(t) = args.get("threads") {
        opts.threads = parse_usize(t)?;
    }
    opts.wavs = !args.has("no-wavs");
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: debug build; use `cargo run --release -p vt_eval -- vocadito` (much faster)"
        );
    }
    let outcome = vocadito::run(&opts, &|line| eprintln!("{line}"))?;
    vocadito::report::write_all(&opts.out, &opts, &outcome)?;
    println!(
        "\n{} detector + {} correction renders over {} tracks in {:.1} s",
        outcome.det.len(),
        outcome.corr.len(),
        outcome.files.len(),
        outcome.elapsed_s
    );
    println!(
        "wrote {}/{{report.md, files.csv, detector.csv, correction.csv, segments.csv}}{}",
        opts.out.display(),
        if opts.wavs { " and listen/*.wav" } else { "" }
    );
    let gates = vocadito::gates::gates(&outcome);
    for g in &gates {
        let r = match g.pass {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => " n/a",
        };
        println!("[{r}] {}: {}", g.name, g.value);
    }
    Ok(gates.iter().all(|g| g.pass != Some(false)))
}
