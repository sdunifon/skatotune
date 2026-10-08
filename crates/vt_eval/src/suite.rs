//! The evaluation suite: every scenario × latency mode × quality tier (×
//! retune speed) rendered through the engine, measured, and summarised as a
//! CSV plus a Markdown report with the DESIGN §4 gates.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use vt_core::{EngineConfig, EngineKind, LatencyMode, QualityTier, Settings, VoiceRange};

use crate::metrics::{self, ClickStats, JitterStats, LatencyMeasurement, LsdStats, SettlingStats, TuningOptions, TuningStats, VibratoStats, VoicingStats};
use crate::reference::{PitchTrack, ReferenceConfig, ReferenceTracker};
use crate::render::{self, BlockPattern, RenderOptions, Rendered, default_engine};
use crate::scenarios::{self, Scenario};
use crate::wav::{self, Result};

pub const MODES: [LatencyMode; 4] = [LatencyMode::LiveMin, LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::Studio];
pub const TIERS: [QualityTier; 3] = [QualityTier::Eco, QualityTier::Hq, QualityTier::Max];
pub const RANGES: [VoiceRange; 4] = [VoiceRange::LowMale, VoiceRange::AltoTenor, VoiceRange::Soprano, VoiceRange::Wide];

pub fn mode_name(m: LatencyMode) -> &'static str {
    match m {
        LatencyMode::LiveMin => "livemin",
        LatencyMode::Live => "live",
        LatencyMode::LivePlus => "liveplus",
        LatencyMode::Studio => "studio",
    }
}

pub fn tier_name(t: QualityTier) -> &'static str {
    match t {
        QualityTier::Eco => "eco",
        QualityTier::Hq => "hq",
        QualityTier::Max => "max",
    }
}

pub fn range_name(r: VoiceRange) -> &'static str {
    match r {
        VoiceRange::LowMale => "lowmale",
        VoiceRange::AltoTenor => "altotenor",
        VoiceRange::Soprano => "soprano",
        VoiceRange::Wide => "wide",
    }
}

pub fn engine_name(e: EngineKind) -> &'static str {
    match e {
        EngineKind::Classic => "classic",
        EngineKind::Natural => "natural",
    }
}

/// Engine configuration for a scenario run (mono: metrics only need one
/// channel, and both channels get identical processing by contract).
pub fn engine_config(sample_rate: f32, mode: LatencyMode, range: VoiceRange) -> EngineConfig {
    EngineConfig { sample_rate, max_block: 4096, channels: 1, latency_mode: mode, voice_range: range, ..EngineConfig::default() }
}

/// Settings for a scenario run.
pub fn run_settings(scn: &Scenario, mode: LatencyMode, tier: QualityTier, retune_ms: f32) -> Settings {
    Settings { tier, engine: default_engine(mode), retune_ms, ..scn.settings }
}

/// Measurements of one render.
#[derive(Clone, Debug, PartialEq)]
pub struct RunResult {
    pub scenario: &'static str,
    pub mode: LatencyMode,
    pub tier: QualityTier,
    pub engine: EngineKind,
    pub retune_ms: f32,
    pub latency: u32,
    pub measured_latency: Option<LatencyMeasurement>,
    pub tuning: Option<TuningStats>,
    pub settling: Option<SettlingStats>,
    pub vibrato: Option<VibratoStats>,
    pub jitter: Option<JitterStats>,
    pub voicing: Option<VoicingStats>,
    pub clicks: ClickStats,
    pub lsd: Option<LsdStats>,
    /// Every output sample finite.
    pub finite: bool,
    /// Wall time of the render alone, s.
    pub render_s: f64,
}

/// Render one scenario configuration and measure everything that applies.
/// `in_track` is the reference track of the input (computed once per scenario).
pub fn evaluate(scn: &Scenario, in_track: Option<&PitchTrack>, mode: LatencyMode, tier: QualityTier, retune_ms: f32, blocks: &BlockPattern) -> (RunResult, Rendered) {
    let sig = &scn.signal;
    let opts = RenderOptions {
        config: engine_config(sig.sample_rate, mode, scn.voice_range),
        settings: run_settings(scn, mode, tier, retune_ms),
        blocks: blocks.clone(),
        telemetry: true,
        midi: Vec::new(),
        extra_tail: 0,
    };
    let t0 = Instant::now();
    let r = render::render(&sig.audio, &opts);
    let render_s = t0.elapsed().as_secs_f64();
    let out = r.aligned(0);
    let finite = r.raw(0).iter().all(|v| v.is_finite());
    let m = scn.measures;
    let needs_pitch = m.tuning || m.settling || m.vibrato || m.jitter || m.sweep;
    let out_track = needs_pitch.then(|| ReferenceTracker::new(sig.sample_rate, ReferenceConfig::default()).track(out));
    let tuning = out_track.as_ref().filter(|_| m.tuning || m.sweep).map(|t| {
        metrics::tuning(sig, t, &TuningOptions { include_sweeps: m.sweep, ..TuningOptions::default() })
    });
    let settling = out_track.as_ref().filter(|_| m.settling).map(|t| metrics::settling(sig, t, 10.0, 0.02, m.settling_legato));
    let vibrato = match (&out_track, in_track) {
        (Some(o), Some(i)) if m.vibrato => Some(metrics::vibrato(sig, i, o, 0.03)),
        _ => None,
    };
    let jitter = match (&out_track, in_track) {
        (Some(o), Some(i)) if m.jitter => Some(metrics::jitter(sig, i, o, 12.0)),
        _ => None,
    };
    let voicing = m.voicing.then(|| metrics::voicing_f1(sig, &r.telemetry, r.latency, 0.02));
    let measured_latency = m.latency.then(|| metrics::measure_latency(&sig.audio, r.raw(0), r.latency as usize + (0.06 * sig.sample_rate as f64) as usize)).flatten();
    let clicks = metrics::clicks(&sig.audio, out, sig.sample_rate, 0.02);
    let lsd = (m.tuning && !sig.notes.is_empty()).then(|| metrics::formant_lsd(sig, out));
    let result = RunResult {
        scenario: scn.name,
        mode,
        tier,
        engine: opts.settings.engine,
        retune_ms,
        latency: r.latency,
        measured_latency,
        tuning,
        settling,
        vibrato,
        jitter,
        voicing,
        clicks,
        lsd,
        finite,
        render_s,
    };
    (result, r)
}

/// Which outputs the suite saves as WAV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WavPolicy {
    None,
    /// Inputs plus every HQ-tier output (default).
    Hq,
    All,
}

#[derive(Clone, Debug)]
pub struct SuiteOptions {
    pub out_dir: PathBuf,
    pub sample_rate: f32,
    /// Scenario names (empty = all).
    pub scenarios: Vec<String>,
    pub modes: Vec<LatencyMode>,
    pub tiers: Vec<QualityTier>,
    pub wavs: WavPolicy,
    pub blocks: BlockPattern,
    /// Worker threads (0 = available parallelism).
    pub threads: usize,
}

impl Default for SuiteOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("out"),
            sample_rate: 48_000.0,
            scenarios: Vec::new(),
            modes: MODES.to_vec(),
            tiers: TIERS.to_vec(),
            wavs: WavPolicy::Hq,
            blocks: BlockPattern::Fixed(128),
            threads: 0,
        }
    }
}

/// Extra checks run once per mode (HQ): amount-0 transparency and
/// block-size invariance.
#[derive(Clone, Debug, PartialEq)]
pub struct ModeCheck {
    pub mode: LatencyMode,
    pub engine: EngineKind,
    /// max |output − input delayed by D| with amount 0.
    pub transparency_max_diff: f64,
    /// Segmental SNR of the amount-0 output vs the input after aligning each
    /// segment by its content wander, dB.
    pub transparency_snr_db: f64,
    /// Largest difference between block patterns {1, 7, 32, 64, 128, 480,
    /// 4096, random} and 64-sample blocks.
    pub block_invariance_max_diff: f64,
}

/// Transparency at amount 0 and block-size invariance for one mode/engine.
pub fn mode_check(sample_rate: f32, mode: LatencyMode, engine: EngineKind) -> Option<ModeCheck> {
    let scn = scenarios::build("melody_male", sample_rate)?;
    let sig = &scn.signal;
    let mut settings = run_settings(&scn, mode, QualityTier::Hq, 20.0);
    settings.engine = engine;
    let config = engine_config(sample_rate, mode, scn.voice_range);
    let amount0 = RenderOptions::new(sample_rate, mode, QualityTier::Hq)
        .with_config(config)
        .with_settings(Settings { amount: 0.0, ..settings });
    let r = render::render(&sig.audio, &amount0);
    let out = r.aligned(0);
    let skip = (0.1 * sample_rate as f64) as usize;
    let max_diff = out.iter().zip(sig.audio.iter()).skip(skip).map(|(a, b)| (a - b).abs() as f64).fold(0.0, f64::max);
    // PSOLA content wander is within one input period: allow up to 1/f_min.
    let max_wander = sample_rate as f64 / scn.voice_range.f_min() as f64;
    let (snr, _) = metrics::aligned_segmental_snr(&sig.audio, out, (0.02 * sample_rate as f64) as usize, -2.0, max_wander);

    // Block-size invariance on a 2 s excerpt (block size 1 is slow on a real engine).
    let excerpt = &sig.audio[..sig.audio.len().min((2.0 * sample_rate) as usize)];
    let base = RenderOptions::new(sample_rate, mode, QualityTier::Hq).with_config(config).with_settings(settings);
    let reference = render::render(excerpt, &base.clone().with_blocks(BlockPattern::Fixed(64)));
    let mut inv = 0.0f64;
    for p in BlockPattern::invariance_set() {
        let o = render::render(excerpt, &base.clone().with_blocks(p));
        let d = o.raw(0).iter().zip(reference.raw(0)).map(|(a, b)| (a - b).abs() as f64).fold(0.0, f64::max);
        inv = inv.max(d);
    }
    Some(ModeCheck { mode, engine, transparency_max_diff: max_diff, transparency_snr_db: snr, block_invariance_max_diff: inv })
}

/// One measured-vs-reported latency comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatencyRow {
    pub sample_rate: f32,
    pub mode: LatencyMode,
    pub range: VoiceRange,
    pub reported: u32,
    pub measured: Option<LatencyMeasurement>,
}

impl LatencyRow {
    /// Measured within ±1 sample of reported, with a clean correlation peak.
    pub fn ok(&self) -> bool {
        self.measured.is_some_and(|m| (m.lag as i64 - self.reported as i64).abs() <= 1 && m.peak > 0.9)
    }
}

/// Measured vs reported latency for every mode × voice range at each rate
/// (DESIGN §3.5 / §4), by cross-correlation on noise bursts (R§10.6).
pub fn latency_sweep(sample_rates: &[f32]) -> Vec<LatencyRow> {
    let mut rows = Vec::new();
    for &sr in sample_rates {
        let Some(scn) = scenarios::build("noise_bursts", sr) else { continue };
        for mode in MODES {
            for range in RANGES {
                let opts = RenderOptions::new(sr, mode, QualityTier::Hq).with_config(engine_config(sr, mode, range));
                let r = render::render(&scn.signal.audio, &opts);
                let max_lag = r.latency as usize + (0.06 * sr as f64) as usize;
                let measured = metrics::measure_latency(&scn.signal.audio, r.raw(0), max_lag);
                rows.push(LatencyRow { sample_rate: sr, mode, range, reported: r.latency, measured });
            }
        }
    }
    rows
}

/// Run the whole suite, writing WAVs as configured. `progress` receives one
/// line per finished render.
pub fn run_suite(opts: &SuiteOptions, progress: &(dyn Fn(&str) + Sync)) -> Result<(Vec<RunResult>, Vec<ModeCheck>)> {
    let names: Vec<&str> = if opts.scenarios.is_empty() {
        scenarios::NAMES.to_vec()
    } else {
        scenarios::NAMES.iter().copied().filter(|n| opts.scenarios.iter().any(|s| s == n)).collect()
    };
    let wav_dir = opts.out_dir.join("wav");
    let built: Vec<Scenario> = names.iter().filter_map(|n| scenarios::build(n, opts.sample_rate)).collect();
    // Input tracks once per scenario (needed for vibrato and jitter baselines).
    let in_tracks: Vec<Option<PitchTrack>> = built
        .iter()
        .map(|s| (s.measures.vibrato || s.measures.jitter).then(|| ReferenceTracker::new(opts.sample_rate, ReferenceConfig::default()).track(&s.signal.audio)))
        .collect();
    if opts.wavs != WavPolicy::None {
        for s in &built {
            wav::write_wav(&wav_dir.join(format!("{}__input.wav", s.name)), &[&s.signal.audio], opts.sample_rate)?;
            wav::write_truth_csv(&wav_dir.join(format!("{}__truth.csv", s.name)), &s.signal)?;
        }
    }

    let mut jobs = Vec::new();
    for (si, s) in built.iter().enumerate() {
        for &mode in &opts.modes {
            for &tier in &opts.tiers {
                for &rt in s.retunes {
                    jobs.push((si, mode, tier, rt));
                }
            }
        }
    }
    let threads = if opts.threads == 0 { std::thread::available_parallelism().map_or(4, |n| n.get()) } else { opts.threads };
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<RunResult>>> = Mutex::new(vec![None; jobs.len()]);
    let error: Mutex<Option<String>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(si, mode, tier, rt)) = jobs.get(i) else { break };
                let scn = &built[si];
                let (res, rendered) = evaluate(scn, in_tracks[si].as_ref(), mode, tier, rt, &opts.blocks);
                let save = match opts.wavs {
                    WavPolicy::All => true,
                    WavPolicy::Hq => tier == QualityTier::Hq,
                    WavPolicy::None => false,
                };
                if save {
                    let path = wav_dir.join(format!("{}__{}_{}_rt{}.wav", scn.name, mode_name(mode), tier_name(tier), rt));
                    if let Err(e) = wav::write_wav(&path, &[rendered.aligned(0)], opts.sample_rate)
                        && let Ok(mut slot) = error.lock()
                    {
                        *slot = Some(format!("{}: {e}", path.display()));
                    }
                }
                progress(&format!("{:>14} {:>8} {:>3} rt{:<3} {}", scn.name, mode_name(mode), tier_name(tier), rt, short_summary(&res)));
                if let Ok(mut v) = results.lock()
                    && let Some(slot) = v.get_mut(i)
                {
                    *slot = Some(res);
                }
            });
        }
    });
    if let Some(e) = error.into_inner().ok().flatten() {
        return Err(e.into());
    }
    let results: Vec<RunResult> = results.into_inner().map_err(|_| "worker panicked")?.into_iter().flatten().collect();

    let mut checks = Vec::new();
    for &mode in &opts.modes {
        let engines: &[EngineKind] = if mode == LatencyMode::LiveMin { &[EngineKind::Classic] } else { &[EngineKind::Classic, EngineKind::Natural] };
        for &e in engines {
            checks.extend(mode_check(opts.sample_rate, mode, e));
            progress(&format!("{:>14} {:>8} {:>3} {}", "mode-check", mode_name(mode), "hq", engine_name(e)));
        }
    }
    Ok((results, checks))
}

fn short_summary(r: &RunResult) -> String {
    let mut s = String::new();
    if let Some(t) = &r.tuning {
        let _ = write!(s, "tune med {:.1} p95 {:.1} ", t.median_abs, t.p95_abs);
    }
    if let Some(v) = &r.voicing {
        let _ = write!(s, "F1 {:.3} ", v.f1);
    }
    if let Some(m) = &r.measured_latency {
        let _ = write!(s, "lat {}/{} ", m.lag, r.latency);
    }
    s
}

/// One DESIGN §4 gate evaluated over suite results.
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub name: &'static str,
    pub target: &'static str,
    /// Measured value(s), human readable.
    pub value: String,
    /// None when the suite did not include the needed runs.
    pub pass: Option<bool>,
}

const TUNING_SCENARIOS: [&str; 4] = ["melody_male", "melody_female", "low_male", "high_soprano"];

fn worst<'a>(rs: impl Iterator<Item = &'a RunResult>, f: impl Fn(&RunResult) -> Option<f64>) -> Option<(f64, String)> {
    rs.filter_map(|r| f(r).map(|v| (v, format!("{} {}/{} rt{}", r.scenario, mode_name(r.mode), tier_name(r.tier), r.retune_ms))))
        .max_by(|a, b| a.0.total_cmp(&b.0))
}

/// Evaluate the §4 gates that the suite covers.
pub fn gates(results: &[RunResult], checks: &[ModeCheck]) -> Vec<Gate> {
    let sel = |mode: LatencyMode, scenarios: &'static [&'static str]| {
        results.iter().filter(move |r| r.mode == mode && r.tier == QualityTier::Hq && scenarios.contains(&r.scenario))
    };
    let mut g = Vec::new();
    let nan_fail = |v: f64| v.is_nan();

    let live = sel(LatencyMode::Live, &TUNING_SCENARIOS).filter(|r| r.retune_ms <= 20.0);
    let med = worst(live.clone(), |r| r.tuning.map(|t| t.median_abs));
    let p95 = worst(live, |r| r.tuning.map(|t| t.p95_abs));
    g.push(match (med, p95) {
        (Some((m, ms)), Some((p, ps))) => Gate {
            name: "Tuning (Live HQ, retune 0-20 ms)",
            target: "median < 5 c, p95 < 15 c",
            value: format!("worst median {m:.2} c ({ms}); worst p95 {p:.2} c ({ps})"),
            pass: Some(m < 5.0 && p < 15.0 && !nan_fail(m) && !nan_fail(p)),
        },
        _ => Gate { name: "Tuning (Live HQ, retune 0-20 ms)", target: "median < 5 c, p95 < 15 c", value: "not run".into(), pass: None },
    });

    let studio = worst(sel(LatencyMode::Studio, &TUNING_SCENARIOS), |r| r.tuning.map(|t| t.median_abs));
    g.push(match studio {
        Some((m, s)) => Gate { name: "Tuning (Studio HQ)", target: "median < 3 c", value: format!("worst median {m:.2} c ({s})"), pass: Some(m < 3.0 && !m.is_nan()) },
        None => Gate { name: "Tuning (Studio HQ)", target: "median < 3 c", value: "not run".into(), pass: None },
    });

    let settle = worst(sel(LatencyMode::Live, &["melody_male", "melody_female", "low_male", "high_soprano"]).filter(|r| r.retune_ms == 0.0), |r| {
        r.settling.as_ref().map(|s| if s.settled == 0 { f64::INFINITY } else { s.median_s })
    });
    g.push(match settle {
        Some((v, s)) => Gate {
            name: "Settling (Live HQ, retune 0)",
            target: "median onset settling < 30 ms (within 10 c for 20 ms)",
            value: format!("worst median {:.1} ms ({s})", v * 1000.0),
            pass: Some(v < 0.03),
        },
        None => Gate { name: "Settling (Live HQ, retune 0)", target: "< 30 ms", value: "not run".into(), pass: None },
    });

    let vib: Vec<&RunResult> = sel(LatencyMode::Live, &["vibrato"]).filter(|r| r.retune_ms == 0.0).collect();
    g.push(match vib.first().and_then(|r| r.vibrato.as_ref()) {
        Some(v) if !v.notes.is_empty() => Gate {
            name: "Vibrato preservation (Live HQ, τ = 0)",
            target: "extent within ±15 %, centre median < 8 c",
            value: format!("worst extent error {:.1} %, centre median {:.2} c", v.worst_extent_ratio_error * 100.0, v.centre_error_median),
            pass: Some(v.worst_extent_ratio_error <= 0.15 && v.centre_error_median < 8.0),
        },
        Some(_) => Gate { name: "Vibrato preservation (Live HQ, τ = 0)", target: "±15 %, < 8 c", value: "no vibrato measured in output".into(), pass: Some(false) },
        None => Gate { name: "Vibrato preservation (Live HQ, τ = 0)", target: "±15 %, < 8 c", value: "not run".into(), pass: None },
    });

    let jit = worst(sel(LatencyMode::Live, &["steady"]), |r| r.jitter.map(|j| if j.frames == 0 { f64::INFINITY } else { j.added_rms }));
    g.push(match jit {
        Some((v, s)) => Gate { name: "Added jitter (Live HQ, steady notes)", target: "< 3 c RMS (> 12 Hz)", value: format!("worst {v:.2} c ({s})"), pass: Some(v < 3.0) },
        None => Gate { name: "Added jitter (Live HQ, steady notes)", target: "< 3 c RMS", value: "not run".into(), pass: None },
    });

    let f1 = worst(sel(LatencyMode::Live, &["consonants", "melody_male", "melody_female"]), |r| r.voicing.map(|v| -v.f1));
    g.push(match f1 {
        Some((v, s)) => Gate { name: "Voicing F1 (Live HQ telemetry vs truth)", target: ">= 0.97", value: format!("worst {:.3} ({s})", -v), pass: Some(-v >= 0.97) },
        None => Gate { name: "Voicing F1 (Live HQ telemetry vs truth)", target: ">= 0.97", value: "not run".into(), pass: None },
    });

    let lat: Vec<&RunResult> = results.iter().filter(|r| r.scenario == "noise_bursts").collect();
    let lat_bad = lat.iter().filter(|r| !r.measured_latency.is_some_and(|m| (m.lag as i64 - r.latency as i64).abs() <= 1)).count();
    g.push(Gate {
        name: "Latency (suite rate, scenario voice range)",
        target: "measured == reported ± 1 sample",
        value: if lat.is_empty() { "not run (see `vt_eval latency`)".into() } else { format!("{} of {} runs off", lat_bad, lat.len()) },
        pass: (!lat.is_empty()).then_some(lat_bad == 0),
    });

    if !checks.is_empty() {
        let classic = checks.iter().filter(|c| c.engine == EngineKind::Classic).map(|c| c.transparency_max_diff).fold(0.0, f64::max);
        let psola = checks.iter().filter(|c| c.engine == EngineKind::Natural).map(|c| c.transparency_snr_db).fold(f64::INFINITY, f64::min);
        g.push(Gate {
            name: "Transparency (amount 0)",
            target: "Classic exact (≤ 1e-6); PSOLA seg. SNR ≥ 25 dB after wander alignment",
            value: format!("Classic max diff {classic:.2e}; PSOLA min SNR {psola:.1} dB"),
            pass: Some(classic <= 1e-6 && (psola >= 25.0 || psola.is_infinite())),
        });
        let inv = checks.iter().map(|c| c.block_invariance_max_diff).fold(0.0, f64::max);
        let finite = results.iter().all(|r| r.finite);
        g.push(Gate {
            name: "RT safety: block-size invariance, finite output",
            target: "≤ 1e-6 across {1,7,32,64,128,480,4096,random}; no NaN/Inf",
            value: format!("max diff {inv:.2e}; all outputs finite: {finite}"),
            pass: Some(inv <= 1e-6 && finite),
        });
    }
    g.push(Gate {
        name: "CPU (p99.9 block time)",
        target: "Live HQ < 5 % @128, < 10 % @32; Max < 25 %",
        value: "run `vt_eval rt-bench` (release build)".into(),
        pass: None,
    });
    g
}

fn opt(v: Option<f64>, prec: usize) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.prec$}"),
        _ => String::new(),
    }
}

/// One CSV row per render.
pub fn write_csv(path: &Path, results: &[RunResult]) -> Result<()> {
    let mut s = String::from(
        "scenario,mode,tier,engine,retune_ms,latency,measured_latency,latency_peak,tune_frames,tune_unvoiced,tune_median_c,tune_p95_c,tune_mean_c,within5_pct,within10_pct,within25_pct,settle_median_ms,settle_p90_ms,settle_max_ms,settle_unsettled,vib_extent_err_pct,vib_centre_median_c,vib_rate_err_hz,jitter_added_c,jitter_out_c,jitter_in_c,voicing_f1,voicing_precision,voicing_recall,clicks,click_max_excess_db,lsd_mean_db,finite,render_ms\n",
    );
    for r in results {
        let t = r.tuning;
        let st = r.settling.as_ref();
        let v = r.vibrato.as_ref();
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.1}",
            r.scenario,
            mode_name(r.mode),
            tier_name(r.tier),
            engine_name(r.engine),
            r.retune_ms,
            r.latency,
            r.measured_latency.map_or(String::new(), |m| m.lag.to_string()),
            opt(r.measured_latency.map(|m| m.peak), 4),
            t.map_or(String::new(), |t| t.frames.to_string()),
            t.map_or(String::new(), |t| t.unvoiced.to_string()),
            opt(t.map(|t| t.median_abs), 3),
            opt(t.map(|t| t.p95_abs), 3),
            opt(t.map(|t| t.mean), 3),
            opt(t.map(|t| t.within_5), 1),
            opt(t.map(|t| t.within_10), 1),
            opt(t.map(|t| t.within_25), 1),
            opt(st.map(|s| s.median_s * 1000.0), 1),
            opt(st.map(|s| s.p90_s * 1000.0), 1),
            opt(st.map(|s| s.max_s * 1000.0), 1),
            st.map_or(String::new(), |s| (s.per_note.len() - s.settled).to_string()),
            opt(v.map(|v| v.worst_extent_ratio_error * 100.0), 1),
            opt(v.map(|v| v.centre_error_median), 2),
            opt(v.map(|v| v.worst_rate_error_hz), 3),
            opt(r.jitter.map(|j| j.added_rms), 3),
            opt(r.jitter.map(|j| j.out_rms), 3),
            opt(r.jitter.map(|j| j.in_rms), 3),
            opt(r.voicing.map(|v| v.f1), 4),
            opt(r.voicing.map(|v| v.precision), 4),
            opt(r.voicing.map(|v| v.recall), 4),
            r.clicks.clicks,
            opt(Some(r.clicks.max_excess_db), 1),
            opt(r.lsd.map(|l| l.mean_db), 2),
            r.finite,
            r.render_s * 1000.0
        );
    }
    std::fs::write(path, s)?;
    Ok(())
}

/// Markdown report: gates first, then one table per scenario.
pub fn write_report(path: &Path, opts: &SuiteOptions, results: &[RunResult], checks: &[ModeCheck], gates: &[Gate]) -> Result<()> {
    let mut s = String::new();
    let _ = writeln!(s, "# Vocal Tune evaluation report\n");
    let _ = writeln!(
        s,
        "Sample rate {} Hz, block pattern `{}`, {} renders. Generated by `vt_eval suite`; metrics per DESIGN §3.7/§4 and R§10.3. \
         Pitch is measured on the latency-compensated output with the offline reference tracker.\n",
        opts.sample_rate,
        opts.blocks.label(),
        results.len()
    );
    let _ = writeln!(s, "## Gates (DESIGN §4)\n\n| Gate | Target | Measured | Result |\n|---|---|---|---|");
    for g in gates {
        let res = match g.pass {
            Some(true) => "PASS",
            Some(false) => "**FAIL**",
            None => "n/a",
        };
        let _ = writeln!(s, "| {} | {} | {} | {} |", g.name, g.target, g.value, res);
    }
    if !checks.is_empty() {
        let _ = writeln!(s, "\n## Per-mode checks (HQ, melody_male)\n\n| Mode | Engine | Amount-0 max diff | Amount-0 seg. SNR (dB) | Block invariance max diff |\n|---|---|---|---|---|");
        for c in checks {
            let _ = writeln!(
                s,
                "| {} | {} | {:.2e} | {:.1} | {:.2e} |",
                mode_name(c.mode),
                engine_name(c.engine),
                c.transparency_max_diff,
                c.transparency_snr_db,
                c.block_invariance_max_diff
            );
        }
    }
    let mut names: Vec<&str> = results.iter().map(|r| r.scenario).collect();
    names.dedup();
    for name in names {
        let desc = scenarios::build(name, opts.sample_rate).map_or("", |s| s.description);
        let _ = writeln!(s, "\n## {name}\n\n{desc}\n");
        let _ = writeln!(
            s,
            "| Mode | Tier | Engine | Retune | D | Median c | p95 c | ≤5/10/25 c % | Settle med/max ms | Vib ext err % | Jitter+ c | F1 | Clicks | LSD dB |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        );
        for r in results.iter().filter(|r| r.scenario == name) {
            let t = r.tuning;
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                mode_name(r.mode),
                tier_name(r.tier),
                engine_name(r.engine),
                r.retune_ms,
                r.latency,
                r.measured_latency.map_or(String::new(), |m| format!(" (meas. {})", m.lag)),
                opt(t.map(|t| t.median_abs), 2),
                opt(t.map(|t| t.p95_abs), 2),
                t.map_or(String::new(), |t| format!("{:.0}/{:.0}/{:.0}", t.within_5, t.within_10, t.within_25)),
                r.settling.as_ref().map_or(String::new(), |st| format!("{}/{}", opt(Some(st.median_s * 1000.0), 1), opt(Some(st.max_s * 1000.0), 1))),
                opt(r.vibrato.as_ref().map(|v| v.worst_extent_ratio_error * 100.0), 1),
                opt(r.jitter.map(|j| j.added_rms), 2),
                opt(r.voicing.map(|v| v.f1), 3),
                r.clicks.clicks,
                opt(r.lsd.map(|l| l.mean_db), 2),
            );
        }
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, s)?;
    Ok(())
}
