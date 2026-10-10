//! Detector-stream and reference-track caches.
//!
//! The detector never sees anything from the control law, so its per-hop
//! `PitchEstimate` stream depends only on the audio, the engine layout (rate,
//! latency mode and voice range, which set the alignment) and the detector
//! settings (tier, tracking, gate). [`run_detector`] reproduces exactly what
//! `skato_engine::Engine` hands `Corrector::process_hop` in a
//! `render::render_with` run: the same construction, settings and reset
//! sequence, the input padded with D samples of silence, and one `analyze()`
//! per full hop (the Studio pipeline's split analysis is bit-identical to a
//! one-shot analysis at the boundary, a `skato_engine` test).
//!
//! Each file carries a fingerprint of the detector (its output on a fixed
//! synthetic signal in that configuration) or of the reference tracker, so a
//! change to either rebuilds the cache instead of replaying stale streams.
//! Files are little-endian; see `write_est` and `write_track`.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use skato_core::{EngineConfig, LatencyMode, PitchEstimate, QualityTier, Settings, VoiceRange};
use skato_detect::{Alignment, Detector, DetectorConfig};

use crate::reference::{PitchTrack, RefFrame, ReferenceConfig, ReferenceTracker};
use crate::vocadito::data as vdata;

const EST_MAGIC: &[u8; 8] = b"SKNIEST1";
const TRK_MAGIC: &[u8; 8] = b"SKNITRK1";

/// A latency mode × quality tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Config {
    pub mode: LatencyMode,
    pub tier: QualityTier,
}

impl Config {
    pub const LIVE_HQ: Config = Config { mode: LatencyMode::Live, tier: QualityTier::Hq };
    pub const STUDIO_HQ: Config = Config { mode: LatencyMode::Studio, tier: QualityTier::Hq };

    /// `live_hq`, `studio_max`, `liveplus_eco`, `livemin_hq`, ...
    pub fn parse(s: &str) -> Option<Config> {
        let (m, t) = s.split_once('_')?;
        let mode = match m {
            "live" => LatencyMode::Live,
            "studio" => LatencyMode::Studio,
            "liveplus" => LatencyMode::LivePlus,
            "livemin" => LatencyMode::LiveMin,
            _ => return None,
        };
        let tier = match t {
            "eco" => QualityTier::Eco,
            "hq" => QualityTier::Hq,
            "max" => QualityTier::Max,
            _ => return None,
        };
        Some(Config { mode, tier })
    }

    pub fn label(&self) -> String {
        let m = match self.mode {
            LatencyMode::Live => "live",
            LatencyMode::Studio => "studio",
            LatencyMode::LivePlus => "liveplus",
            LatencyMode::LiveMin => "livemin",
        };
        let t = match self.tier {
            QualityTier::Eco => "eco",
            QualityTier::Hq => "hq",
            QualityTier::Max => "max",
        };
        format!("{m}_{t}")
    }

    pub fn engine_config(&self, sample_rate: f32, range: VoiceRange) -> EngineConfig {
        EngineConfig { sample_rate, max_block: 4096, channels: 1, latency_mode: self.mode, voice_range: range, ..EngineConfig::default() }
    }
}

/// How the engine is laid out for one stream (everything replay needs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EstHeader {
    pub sample_rate: f32,
    pub hop: u32,
    /// Reported latency D, samples.
    pub latency: u32,
    /// `CorrectorConfig::lookahead_hops` the engine passes.
    pub lookahead_hops: u32,
    /// Studio pipeline (controls applied one hop after their analysis).
    pub pipelined: bool,
    /// Studio control lead from `studio_split` (0 in Live).
    pub lead: u32,
    /// Studio detector delay (`Alignment::Centered`), samples (0 in Live).
    pub det_delay: u32,
    pub range: VoiceRange,
    pub mode: LatencyMode,
    pub tier: QualityTier,
    /// Input length (before the D samples of padding).
    pub n_input: u64,
}

/// A cached detector stream.
#[derive(Clone, Debug)]
pub struct EstStream {
    pub h: EstHeader,
    pub est: Vec<PitchEstimate>,
}

fn range_code(r: VoiceRange) -> u8 {
    match r {
        VoiceRange::LowMale => 0,
        VoiceRange::AltoTenor => 1,
        VoiceRange::Soprano => 2,
        VoiceRange::Wide => 3,
    }
}

fn range_from(c: u8) -> VoiceRange {
    match c {
        0 => VoiceRange::LowMale,
        2 => VoiceRange::Soprano,
        3 => VoiceRange::Wide,
        _ => VoiceRange::AltoTenor,
    }
}

fn mode_code(m: LatencyMode) -> u8 {
    match m {
        LatencyMode::LiveMin => 0,
        LatencyMode::Live => 1,
        LatencyMode::LivePlus => 2,
        LatencyMode::Studio => 3,
    }
}

fn mode_from(c: u8) -> LatencyMode {
    match c {
        0 => LatencyMode::LiveMin,
        2 => LatencyMode::LivePlus,
        3 => LatencyMode::Studio,
        _ => LatencyMode::Live,
    }
}

fn tier_code(t: QualityTier) -> u8 {
    match t {
        QualityTier::Eco => 0,
        QualityTier::Hq => 1,
        QualityTier::Max => 2,
    }
}

fn tier_from(c: u8) -> QualityTier {
    match c {
        0 => QualityTier::Eco,
        2 => QualityTier::Max,
        _ => QualityTier::Hq,
    }
}

/// Engine layout for a config (mirrors `skato_engine::Engine::new`).
pub fn layout(cfg: Config, sample_rate: f32, range: VoiceRange, n_input: usize) -> (EstHeader, Alignment) {
    let ecfg = cfg.engine_config(sample_rate, range);
    let hop = skato_core::hop_for_rate(sample_rate).max(1);
    let d = skato_engine::latency_samples(&ecfg) as usize;
    let studio = cfg.mode == LatencyMode::Studio;
    let (alignment, pipelined, lead, det_delay) = if studio {
        let (delay, lead) = skato_engine::studio_split(sample_rate, hop, range, d);
        (Alignment::Centered { delay }, lead >= 1, lead, delay)
    } else {
        (Alignment::Causal, false, 0, 0)
    };
    let lookahead_hops = if studio { d.div_ceil(hop) - usize::from(pipelined) } else { 0 };
    (
        EstHeader {
            sample_rate,
            hop: hop as u32,
            latency: d as u32,
            lookahead_hops: lookahead_hops as u32,
            pipelined,
            lead: lead as u32,
            det_delay: det_delay as u32,
            range,
            mode: cfg.mode,
            tier: cfg.tier,
            n_input: n_input as u64,
        },
        alignment,
    )
}

/// Run the detector exactly as the engine does in a render (see the module
/// docs). `settings` supplies the tier, tracking and gate.
pub fn run_detector(audio: &[f32], sample_rate: f32, cfg: Config, range: VoiceRange, settings: &Settings) -> EstStream {
    let (h, alignment) = layout(cfg, sample_rate, range, audio.len());
    let hop = h.hop as usize;
    let mut det = Detector::new(DetectorConfig { sample_rate, voice_range: range, hop, alignment });
    // Engine::new applies the default settings; render_with resets the
    // engine (and the detector) and applies the run's.
    let defaults = Settings::default();
    det.set_tier(defaults.tier);
    det.set_tracking(defaults.tracking, defaults.gate_db);
    det.reset();
    det.set_tier(settings.tier);
    det.set_tracking(settings.tracking, settings.gate_db);
    let total = audio.len() + h.latency as usize;
    let mut buf = vec![0.0f32; total];
    buf[..audio.len()].copy_from_slice(audio);
    for x in buf.iter_mut() {
        *x = if x.is_finite() { x.clamp(-1.0e4, 1.0e4) } else { 0.0 };
    }
    let est = buf.chunks_exact(hop).map(|c| {
        det.push(c);
        det.analyze()
    });
    EstStream { h, est: est.collect() }
}

/// FNV-1a 64 over a byte stream.
#[derive(Clone, Copy)]
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn eat(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// Fixed signal the fingerprints are taken on: 2 s of the synthetic male
/// melody at 44.1 kHz (onsets, legato, consonants, silence).
fn fingerprint_signal() -> Vec<f32> {
    let sr = 44_100.0;
    let mut a = crate::scenarios::build("melody_male", sr).map(|s| s.signal.audio).unwrap_or_default();
    a.truncate((2.0 * sr) as usize);
    a
}

/// Fingerprint of the detector in `cfg`: a hash of its whole stream on the
/// fixed signal. Any change to the detector's output changes it.
pub fn detector_fingerprint(cfg: Config) -> u64 {
    let audio = fingerprint_signal();
    let s = Settings { tier: cfg.tier, ..Settings::default() };
    let st = run_detector(&audio, 44_100.0, cfg, VoiceRange::AltoTenor, &s);
    let mut h = Fnv::new();
    h.eat(&encode_est(&st, 0));
    h.0
}

/// Fingerprint of the offline reference tracker (its track of the fixed signal).
pub fn reference_fingerprint() -> u64 {
    let audio = fingerprint_signal();
    let mut a = audio;
    a.truncate(44_100 / 2);
    let t = ReferenceTracker::new(44_100.0, ReferenceConfig::default()).track(&a);
    let mut h = Fnv::new();
    h.eat(&encode_track(&t, a.len(), 0));
    h.0
}

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.i..self.i + n).ok_or("truncated cache file")?;
        self.i += n;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N], String> {
        self.take(N)?.try_into().map_err(|_| "truncated cache file".to_string())
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.arr::<1>()?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.arr()?))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.arr()?))
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        f.write_all(bytes).map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

fn read_all(path: &Path) -> Result<Vec<u8>, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut b = Vec::new();
    f.read_to_end(&mut b).map_err(|e| e.to_string())?;
    Ok(b)
}

/// 64-byte header (with the fingerprint), then 40 bytes per estimate.
fn encode_est(s: &EstStream, fingerprint: u64) -> Vec<u8> {
    let mut w = Vec::with_capacity(64 + 40 * s.est.len());
    w.extend_from_slice(EST_MAGIC);
    let h = &s.h;
    for v in [h.hop, h.latency, h.lookahead_hops, u32::from(h.pipelined), h.lead, h.det_delay] {
        w.extend_from_slice(&v.to_le_bytes());
    }
    w.extend_from_slice(&h.sample_rate.to_le_bytes());
    w.extend_from_slice(&[range_code(h.range), mode_code(h.mode), tier_code(h.tier), 0]);
    w.extend_from_slice(&h.n_input.to_le_bytes());
    w.extend_from_slice(&(s.est.len() as u64).to_le_bytes());
    w.extend_from_slice(&fingerprint.to_le_bytes());
    w.resize(64, 0);
    for e in &s.est {
        w.extend_from_slice(&e.sample_pos.to_le_bytes());
        for v in [e.info_lag, e.f0_hz, e.cents, e.period, e.confidence, e.voiced_prob, e.level_db] {
            w.extend_from_slice(&v.to_le_bytes());
        }
        w.extend_from_slice(&[u8::from(e.voiced), u8::from(e.sibilant), 0, 0]);
    }
    w
}

/// Read a stream; `None` when its fingerprint is not `fingerprint` (stale).
pub fn read_est(path: &Path, fingerprint: u64) -> Result<Option<EstStream>, String> {
    let b = read_all(path)?;
    let mut r = Rd { b: &b, i: 0 };
    if r.take(8)? != EST_MAGIC {
        return Ok(None);
    }
    let hop = r.u32()?;
    let latency = r.u32()?;
    let lookahead_hops = r.u32()?;
    let pipelined = r.u32()? != 0;
    let lead = r.u32()?;
    let det_delay = r.u32()?;
    let sample_rate = r.f32()?;
    let codes = r.arr::<4>()?;
    let n_input = r.u64()?;
    let n = r.u64()? as usize;
    if r.u64()? != fingerprint || b.len() != 64 + 40 * n {
        return Ok(None);
    }
    r.i = 64;
    let mut est = Vec::with_capacity(n);
    for _ in 0..n {
        let sample_pos = r.u64()?;
        let mut v = [0.0f32; 7];
        for x in v.iter_mut() {
            *x = r.f32()?;
        }
        let flags = r.arr::<4>()?;
        est.push(PitchEstimate {
            sample_pos,
            info_lag: v[0],
            f0_hz: v[1],
            cents: v[2],
            period: v[3],
            confidence: v[4],
            voiced_prob: v[5],
            level_db: v[6],
            voiced: flags[0] != 0,
            sibilant: flags[1] != 0,
        });
    }
    let h = EstHeader {
        sample_rate,
        hop,
        latency,
        lookahead_hops,
        pipelined,
        lead,
        det_delay,
        range: range_from(codes[0]),
        mode: mode_from(codes[1]),
        tier: tier_from(codes[2]),
        n_input,
    };
    Ok(Some(EstStream { h, est }))
}

fn encode_track(t: &PitchTrack, n_input: usize, fingerprint: u64) -> Vec<u8> {
    let mut w = Vec::with_capacity(48 + 49 * t.frames.len());
    w.extend_from_slice(TRK_MAGIC);
    w.extend_from_slice(&t.sample_rate.to_le_bytes());
    w.extend_from_slice(&(t.hop as u64).to_le_bytes());
    w.extend_from_slice(&(n_input as u64).to_le_bytes());
    w.extend_from_slice(&(t.frames.len() as u64).to_le_bytes());
    w.extend_from_slice(&fingerprint.to_le_bytes());
    for f in &t.frames {
        w.extend_from_slice(&(f.pos as u64).to_le_bytes());
        w.push(u8::from(f.voiced));
        for v in [f.f0_hz, f.cents, f.cents_raw, f.clarity, f.level_dbfs] {
            w.extend_from_slice(&v.to_le_bytes());
        }
    }
    w
}

/// Read a reference track and the input length; `Err` when missing or stale
/// (`prepare` builds it).
pub fn read_track(path: &Path) -> Result<(PitchTrack, usize), String> {
    read_track_checked(path, None)?.ok_or_else(|| format!("{}: stale or invalid cache file", path.display()))
}

fn read_track_checked(path: &Path, fingerprint: Option<u64>) -> Result<Option<(PitchTrack, usize)>, String> {
    let b = read_all(path)?;
    let mut r = Rd { b: &b, i: 0 };
    if r.take(8)? != TRK_MAGIC {
        return Ok(None);
    }
    let sample_rate = r.f64()?;
    let hop = r.u64()? as usize;
    let n_input = r.u64()? as usize;
    let n = r.u64()? as usize;
    let fp = r.u64()?;
    if fingerprint.is_some_and(|f| f != fp) || b.len() != 48 + 49 * n {
        return Ok(None);
    }
    let mut frames = Vec::with_capacity(n);
    for _ in 0..n {
        let pos = r.u64()? as usize;
        let voiced = r.u8()? != 0;
        frames.push(RefFrame { pos, voiced, f0_hz: r.f64()?, cents: r.f64()?, cents_raw: r.f64()?, clarity: r.f64()?, level_dbfs: r.f64()? });
    }
    Ok(Some((PitchTrack { sample_rate, hop, frames }, n_input)))
}

pub fn track_path(cache: &Path, id: u32) -> PathBuf {
    cache.join("input").join(format!("vocadito_{id}.trk"))
}

pub fn est_path(cache: &Path, cfg: Config, id: u32) -> PathBuf {
    cache.join(cfg.label()).join(format!("vocadito_{id}.est"))
}

/// Fingerprints of the detector per config and of the reference tracker.
#[derive(Clone, Debug)]
pub struct Fingerprints {
    pub detector: Vec<(Config, u64)>,
    pub reference: u64,
}

impl Fingerprints {
    pub fn compute(cfgs: &[Config]) -> Fingerprints {
        Fingerprints { detector: cfgs.iter().map(|&c| (c, detector_fingerprint(c))).collect(), reference: reference_fingerprint() }
    }

    pub fn detector(&self, cfg: Config) -> u64 {
        self.detector.iter().find(|(c, _)| *c == cfg).map_or(0, |x| x.1)
    }
}

/// Load the cached stream of `id` in `cfg`.
pub fn load_stream(cache: &Path, fp: &Fingerprints, cfg: Config, id: u32) -> Result<EstStream, String> {
    let p = est_path(cache, cfg, id);
    read_est(&p, fp.detector(cfg))?.ok_or_else(|| format!("{}: stale cache file", p.display()))
}

/// Build whatever is missing or stale in the cache for `files` × `cfgs`
/// (the reference track of each input and one detector stream per config).
/// Returns how many files had to be (re)built.
pub fn prepare(
    data: &Path,
    cache: &Path,
    fp: &Fingerprints,
    files: &[u32],
    cfgs: &[Config],
    threads: usize,
    progress: &(dyn Fn(&str) + Sync),
) -> Result<usize, String> {
    let meta = vdata::load_metadata(data);
    let built = super::parallel(files.len(), threads, |i| -> Result<bool, String> {
        let id = files[i];
        let need_ref = !matches!(read_track_checked(&track_path(cache, id), Some(fp.reference)), Ok(Some(_)));
        let need: Vec<Config> =
            cfgs.iter().copied().filter(|&c| !matches!(read_est(&est_path(cache, c, id), fp.detector(c)), Ok(Some(_)))).collect();
        if !need_ref && need.is_empty() {
            return Ok(false);
        }
        let t0 = std::time::Instant::now();
        let track = vdata::load_track(data, id, &meta).map_err(|e| format!("vocadito_{id}: {e}"))?;
        let sr = track.sample_rate;
        if need_ref {
            let rt = ReferenceTracker::new(sr, ReferenceConfig::default()).track(&track.audio);
            write_atomic(&track_path(cache, id), &encode_track(&rt, track.audio.len(), fp.reference))?;
        }
        let range = vdata::choose_range(track.min_f0());
        for cfg in need {
            let s = Settings { tier: cfg.tier, ..Settings::default() };
            let st = run_detector(&track.audio, sr, cfg, range, &s);
            write_atomic(&est_path(cache, cfg, id), &encode_est(&st, fp.detector(cfg)))?;
        }
        progress(&format!("cached vocadito_{id} ({:.1} s)", t0.elapsed().as_secs_f64()));
        Ok(true)
    });
    let mut n = 0;
    for b in built {
        n += usize::from(b?);
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configs_round_trip_their_labels() {
        for s in ["live_hq", "studio_hq", "live_eco", "studio_max", "liveplus_hq", "livemin_eco"] {
            assert_eq!(Config::parse(s).map(|c| c.label()).as_deref(), Some(s));
        }
        assert_eq!(Config::parse("live"), None);
        assert_eq!(Config::parse("studio_ultra"), None);
    }

    /// A stream survives the cache bit for bit, and a fingerprint mismatch
    /// reads as stale.
    #[test]
    fn streams_round_trip_and_stale_files_are_detected() {
        let audio: Vec<f32> = (0..4410).map(|i| 0.3 * (i as f32 * 0.05).sin()).collect();
        let cfg = Config::STUDIO_HQ;
        let st = run_detector(&audio, 44_100.0, cfg, VoiceRange::AltoTenor, &Settings::default());
        assert!(st.h.pipelined && st.h.lead >= 1);
        let dir = std::env::temp_dir().join(format!("skato_ni_cache_{}", std::process::id()));
        let p = est_path(&dir, cfg, 1);
        write_atomic(&p, &encode_est(&st, 42)).unwrap();
        let back = read_est(&p, 42).unwrap().unwrap();
        assert_eq!(back.h, st.h);
        assert_eq!(back.est.len(), st.est.len());
        for (a, b) in back.est.iter().zip(&st.est) {
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
        }
        assert!(read_est(&p, 43).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_detector_fingerprint_is_deterministic_and_per_config() {
        let a = detector_fingerprint(Config::LIVE_HQ);
        assert_eq!(a, detector_fingerprint(Config::LIVE_HQ));
        assert_ne!(a, detector_fingerprint(Config::STUDIO_HQ));
    }
}
