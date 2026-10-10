//! `skato_eval` command line: synthesise test signals, render WAVs through the
//! engine, run the evaluation suite, check latency and benchmark callbacks.
//!
//! ```text
//! skato_eval synth <scenario|list> <out.wav> [--sr 48000]
//! skato_eval render <in.wav> <out.wav> [--mode live] [--range altotenor] [--tier hq] [--engine natural]
//!                [--retune 20] [--key C] [--scale major] [...]   (see `skato_eval help`)
//! skato_eval track <in.wav> <out.csv>
//! skato_eval suite [--out out/] [--sr 48000] [--scenario a,b] [--modes ..] [--tiers ..] [--wavs hq] [--quick]
//! skato_eval latency [--sr 44100,48000,96000] [--no-content]
//! skato_eval rt-bench [--sr 48000] [--seconds 10] [--buffers 32,64,128,256,512] [--modes ..] [--tiers ..]
//!                  [--note-hold] [--follow-singer]
//! skato_eval vocadito [--data data/vocadito] [--out out/vocadito] [--modes ..] [--tiers ..] [--engine ..]
//!                  [--files 1,2] [--conditions ..] [--range auto] [--listen ..] [--dump ..] [--no-wavs]
//! skato_eval note-intent [--split test] [--configs live_hq,studio_hq] [--conditions ..] [--systems note_hold]
//!                  [--tolerance 8] [--sweep 0,3,5,..|none] [--data ..] [--out out/note-intent] [--cache ..]
//! skato_eval note-intent synth [--configs live_hq,studio_hq] [--systems note_hold] [--out out/note-intent]
//! skato_eval note-intent dump --configs live_hq --split dev --out <file.f32>
//! skato_eval note-intent dump-synth --configs live_hq --n 400 --seed0 50000 --out <file.f32>
//! skato_eval note-intent convert-model <in.ldm> <out.skgb>
//! skato_eval autokey [--per-key 4] [--seconds 60] [--no-sweep] [--data ..] [--out out/autokey] [--cache ..]
//! ```
//!
//! Exit codes: 0 success, 1 a gate failed (latency mismatch, rt-bench over
//! budget, suite or real-singing gate failure), 2 usage or I/O error.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use skato_core::{
    EngineConfig, EngineKind, LatencyMode, QualityTier, ScaleKind, Settings, VoiceRange,
};
use skato_eval::reference::{ReferenceConfig, ReferenceTracker};
use skato_eval::render::{self, BlockPattern, RenderOptions};
use skato_eval::rtbench::{self, RtBenchOptions};
use skato_eval::suite::{self, SuiteOptions, WavPolicy};
use skato_eval::autokey::{self, AutoKeyOptions};
use skato_eval::note_intent::{self, NoteIntentOptions, System};
use skato_eval::vocadito::{self, Condition, RunConfig, VocaditoOptions};
use skato_eval::{scenarios, wav};

const USAGE: &str = "\
skato_eval: Skatotune evaluation harness

USAGE:
  skato_eval synth <scenario|list> <out.wav> [--sr 48000]
      Write a synthetic scenario (plus <out>.truth.csv with f0/target at 1 ms).
  skato_eval render <in.wav> <out.wav> [OPTIONS]
      Process a WAV through the engine (latency-compensated unless --no-compensate).
      --mode livemin|live|liveplus|studio   --range lowmale|altotenor|soprano|wide
      --tier eco|hq|max   --engine classic|natural   --retune MS   --transition MS
      --key C..B|0-11   --scale chromatic|major|minor|harmonicminor|melodicminor|dorian|
            phrygian|lydian|mixolydian|locrian|majorpentatonic|minorpentatonic|blues
      --amount 0-100   --humanize 0-100   --flex 0-100   --depth -100..200   --detune C
      --transpose ST   --formant ST   --tolerance C   --tracking 0-100   --gate DB
      --live-plus-ms MS   --studio-ms MS   --block N   --no-preserve-vibrato   --no-compensate
  skato_eval track <in.wav> <out.csv>
      Offline reference pitch track (1 ms frames) of the first channel.
  skato_eval suite [--out out] [--sr 48000] [--scenario a,b] [--modes live,studio] [--tiers hq]
                [--wavs none|hq|all] [--block N] [--threads N] [--quick]
      All scenarios x modes x tiers: WAVs, out/results.csv and out/report.md.
  skato_eval latency [--sr 44100,48000,96000] [--no-content]
      Measured vs reported latency for every mode x voice range (exit 1 on mismatch),
      then the latency gauge: the delay of voiced content (D + PSOLA wander) per
      mode x engine x voice range at 48 kHz, i.e. what a performer hears.
  skato_eval rt-bench [--sr 48000] [--seconds 10] [--buffers 32,64,128,256,512] [--modes ..] [--tiers ..]
                  [--note-hold] [--follow-singer] [--wav FILE]
      process() tail latency per config; exit 1 if p99.9 > 10% of the period at 32/64/128.
      --note-hold / --follow-singer switch the Labs options on; --wav times a recording
      (first channel, at its own sample rate) instead of the synthetic melody.
  skato_eval vocadito [--data data/vocadito] [--out out/vocadito] [--modes live,studio,livemin]
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
  skato_eval note-intent [--split test|dev|all] [--configs live_hq,studio_hq] [--files 7,9]
                   [--conditions hard_chromatic,hard_key,natural_key] [--systems note_hold,follow]
                   [--tolerance 8] [--sweep 0,3,5,8,12,16,20,25,30,40,45|none]
                   [--data data/vocadito] [--out out/note-intent] [--cache DIR] [--threads N]
      Note-intent harness: replays the cached detector stream of every vocadito file
      through the control law with the Labs options off (baseline) and on, on the fixed
      DEV/TEST split, and scores the note decisions against both annotators: frame and
      per-note accuracy, warble, stable p95 and note agreement of a proxy output, onset
      glitches, legato decision latency, and the gaps to the baseline's Tolerance
      frontier. The detector streams are cached under <out>/cache (rebuilt when the
      detector changes). Writes report.md and points.csv under --out.
  skato_eval note-intent synth [--configs live_hq,studio_hq] [--systems note_hold] [--out out/note-intent]
      The synthetic scenarios with the Labs options off and on: tuning p95, target accuracy
      and settling, on the study's proxy and through the engine. Writes synth.md under --out.
  skato_eval note-intent dump --configs live_hq --split dev --out FILE.f32 [--data ..] [--cache ..]
  skato_eval note-intent dump-synth --configs live_hq [--n 400] [--seed0 50000] --out FILE.f32
      Training data for the learned veto (see crates/skato_control/models/README.md):
      per-hop features and annotations of the split's vocadito files, or of synthetic lines.
  skato_eval note-intent convert-model <in.ldm> <out.skgb>
      Convert a trained candidate tree ensemble (LDMODEL text) to the embedded format.
  skato_eval autokey [--per-key 4] [--seconds 60] [--no-sweep] [--data data/vocadito] [--out out/autokey]
                     [--cache out/note-intent/cache] [--threads N]
      Auto-Key: tune on DEV (synthetic melodies in 36 keys + vocadito DEV), then report TEST once
      against classic Krumhansl-Schmuckler key finding. Writes report.md under --out.
";

/// Flags that take no value.
const SWITCHES: &[&str] = &[
    "no-compensate",
    "no-preserve-vibrato",
    "quick",
    "help",
    "no-wavs",
    "no-content",
    "note-hold",
    "follow-singer",
    "no-sweep",
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
        "note-intent" => cmd_note_intent(&args),
        "autokey" | "auto-key" => cmd_autokey(&args),
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
        .ok_or_else(|| format!("unknown scenario {name} (try `skato_eval synth list`)"))?;
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
    opts.note_hold = args.has("note-hold");
    opts.follow_singer = args.has("follow-singer");
    if let Some(path) = args.get("wav") {
        let (chans, sr) = wav::read_wav(std::path::Path::new(path))?;
        opts.material = chans.into_iter().next();
        opts.sample_rate = sr;
    }
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: debug build; timings are meaningless (use `cargo run --release -p skato_eval -- rt-bench`)"
        );
    }
    println!(
        "{:>9} {:>4} {:>8} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8} {:>6}  budget",
        "mode", "tier", "engine", "buffer", "p50 µs", "p99 µs", "p99.9 µs", "max µs", "p99.9 %", "vetos"
    );
    let rows = rtbench::run(&opts, &mut |r| {
        let gate = match r.design_gate() {
            Some(true) => " (§4 gate ok)",
            Some(false) => " (§4 gate FAIL)",
            None => "",
        };
        println!(
            "{:>9} {:>4} {:>8} {:>6} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>7.2}% {:>6}  {}{}",
            suite::mode_name(r.mode),
            suite::tier_name(r.tier),
            suite::engine_name(r.engine),
            r.buffer,
            r.p50_us,
            r.p99_us,
            r.p999_us,
            r.max_us,
            r.p999_fraction() * 100.0,
            r.veto_runs,
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
            "warning: debug build; use `cargo run --release -p skato_eval -- vocadito` (much faster)"
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

fn parse_config(s: &str) -> Result<note_intent::cache::Config, String> {
    note_intent::cache::Config::parse(s)
        .ok_or_else(|| format!("unknown config {s} (expected live_hq, studio_hq, live_eco, ...)"))
}

fn parse_system(s: &str) -> Result<System, String> {
    System::parse(s).ok_or_else(|| format!("unknown system {s} (expected note_hold or follow)"))
}

fn parse_cents(s: &str) -> Result<f32, String> {
    s.parse::<f32>()
        .ok()
        .filter(|v| (0.0..=45.0).contains(v))
        .ok_or_else(|| format!("bad Tolerance {s} (0-45 c)"))
}

fn cmd_autokey(args: &Args) -> CmdResult {
    let d = AutoKeyOptions::default();
    let opts = AutoKeyOptions {
        data: args.get("data").map_or(d.data.clone(), PathBuf::from),
        out: args.get("out").map_or(d.out.clone(), PathBuf::from),
        cache: args.get("cache").map(PathBuf::from),
        per_key: args.get("per-key").map(parse_usize).transpose()?.unwrap_or(d.per_key),
        dur_s: args.get("seconds").map(|s| s.parse::<f32>()).transpose()?.unwrap_or(d.dur_s),
        threads: args.get("threads").map(parse_usize).transpose()?.unwrap_or(d.threads),
        no_sweep: args.has("no-sweep"),
    };
    let report = autokey::run(&opts, &|m: &str| eprintln!("{m}"))?;
    print!("{report}");
    println!("wrote {}", opts.out.join("report.md").display());
    Ok(true)
}

fn cmd_note_intent(args: &Args) -> CmdResult {
    if args.pos.first().map(String::as_str) == Some("convert-model") {
        let input = PathBuf::from(args.pos.get(1).ok_or("convert-model: missing input .ldm")?);
        let output = PathBuf::from(args.pos.get(2).ok_or("convert-model: missing output .skgb")?);
        let text = std::fs::read_to_string(&input)?;
        let bytes = note_intent::model::convert_ldm(&text)?;
        let m = skato_control::intent::gbt::GbtModel::parse(&bytes)?;
        std::fs::write(&output, &bytes)?;
        println!(
            "wrote {} ({} bytes): {} trees, {} nodes, {} features, checksum {:#018x}",
            output.display(),
            bytes.len(),
            m.trees(),
            m.nodes(),
            m.features(),
            m.checksum()
        );
        return Ok(true);
    }
    let sub = args.pos.first().map(String::as_str);
    if sub == Some("synth") {
        let cfgs = args.list("configs", parse_config)?.unwrap_or_else(|| {
            vec![note_intent::cache::Config::LIVE_HQ, note_intent::cache::Config::STUDIO_HQ]
        });
        let mut systems = vec![System::Baseline];
        systems.extend(args.list("systems", parse_system)?.unwrap_or_else(|| vec![System::NoteHold]));
        let threads = args.get("threads").map(parse_usize).transpose()?.unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(4, |n| n.get())
        });
        let out = PathBuf::from(args.get("out").unwrap_or("out/note-intent"));
        let rows = note_intent::synth::run(&cfgs, &systems, threads);
        let md = format!(
            "# Synthetic scenarios, Labs options off → on\n\nProxy = the note-intent study's replay (input + shift, no shifter); \
             engine = a real render. `rule`: a p95 under 15 c stays under, a higher one rises ≤ 2 c, no new onset ≥ 30 ms.\n\n{}",
            note_intent::synth::report(&rows)
        );
        std::fs::create_dir_all(&out)?;
        std::fs::write(out.join("synth.md"), &md)?;
        print!("{md}");
        println!("wrote {}", out.join("synth.md").display());
        return Ok(true);
    }
    if matches!(sub, Some("dump" | "dump-synth")) {
        let cfgs = args.list("configs", parse_config)?.unwrap_or_else(|| vec![note_intent::cache::Config::LIVE_HQ]);
        let [cfg] = cfgs.as_slice() else {
            return Err("dump: one --configs value".into());
        };
        let out = PathBuf::from(args.get("out").ok_or("dump: --out FILE.f32")?);
        let threads = args.get("threads").map(parse_usize).transpose()?.unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(4, |n| n.get())
        });
        if sub == Some("dump") {
            let defaults = NoteIntentOptions::default();
            let data = args.get("data").map_or(defaults.data.clone(), PathBuf::from);
            let cache = args.get("cache").map_or_else(|| defaults.cache_dir(), PathBuf::from);
            let split = note_intent::data::Split::parse(args.get("split").unwrap_or("dev"))
                .ok_or("--split: expected dev|test|all")?;
            let n = note_intent::train::dump(&data, &cache, *cfg, split, &out, threads)?;
            println!("dumped {n} rows x {} columns to {}", note_intent::train::NCOL, out.display());
        } else {
            let n = args.get("n").map(parse_usize).transpose()?.unwrap_or(400);
            let seed0 = args.get("seed0").map(|s| s.parse::<u64>()).transpose()?.unwrap_or(50_000);
            let (rows, secs) = note_intent::train::dump_synth(*cfg, n, seed0, &out, threads)?;
            println!("synthetic: {n} lines, {secs:.0} s of audio, {rows} rows to {}", out.display());
        }
        return Ok(true);
    }
    if let Some(p) = sub {
        return Err(format!("note-intent: unknown argument {p}").into());
    }
    let mut opts = NoteIntentOptions::default();
    if let Some(d) = args.get("data") {
        opts.data = PathBuf::from(d);
    }
    if let Some(o) = args.get("out") {
        opts.out = PathBuf::from(o);
    }
    if let Some(c) = args.get("cache") {
        opts.cache = Some(PathBuf::from(c));
    }
    if let Some(s) = args.get("split") {
        opts.split = note_intent::data::Split::parse(s)
            .ok_or_else(|| format!("--split: expected dev|test|all, got {s}"))?;
    }
    if let Some(f) = args.list("files", parse_u32)? {
        opts.files = Some(f);
    }
    if let Some(c) = args.list("configs", parse_config)? {
        opts.configs = c;
    }
    if let Some(c) = args.list("conditions", parse_condition)? {
        if c.contains(&Condition::Detect) {
            return Err("--conditions: detect is not a note-decision condition".into());
        }
        opts.conditions = c;
    }
    if let Some(s) = args.list("systems", parse_system)? {
        opts.systems = s;
    }
    if let Some(t) = args.get("tolerance") {
        opts.tolerance = parse_cents(t)?;
    }
    match args.get("sweep") {
        Some("none") => opts.sweep.clear(),
        Some(_) => opts.sweep = args.list("sweep", parse_cents)?.unwrap_or_default(),
        None => {}
    }
    if let Some(t) = args.get("threads") {
        opts.threads = parse_usize(t)?.max(1);
    }
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: debug build; use `cargo run --release -p skato_eval -- note-intent` (much faster)"
        );
    }
    let t0 = std::time::Instant::now();
    let outcome = note_intent::run(&opts, &|line| eprintln!("{line}"))?;
    note_intent::write_all(&outcome, &opts)?;
    print!("{}", note_intent::report(&outcome, &opts));
    println!(
        "{} files, {} points in {:.1} s ({} cache files rebuilt); wrote {}/{{report.md, points.csv}}",
        outcome.files.len(),
        outcome.points.len(),
        t0.elapsed().as_secs_f64(),
        outcome.rebuilt,
        opts.out.display()
    );
    Ok(true)
}
