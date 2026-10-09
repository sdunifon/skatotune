//! Pitch shifting for Skatotune: the reported-latency formula, the pitch-mark
//! tracker and the two engines (DESIGN §3.4; R§5.3, R§7.4, R§8.13-8.19).
//!
//! | engine | where | what |
//! |---|---|---|
//! | Classic (L0) | every mode; the only one in Live-Min | resample + whole-cycle repeat/drop; formants move with pitch |
//! | Natural, causal PSOLA (L1/L2) | Live, Live+ | short-rise or symmetric grains on causal NCC marks; formant-preserving |
//! | Natural, Studio PSOLA (S1) | Studio | the same grains with D of lookahead and nearest-mark grain choice |
//!
//! ### Quality tiers (R§7.2.2)
//!
//! | | Eco | HQ / Max |
//! |---|---|---|
//! | reads | 4-point cubic | 16-tap Kaiser sinc (anti-aliased at rate r or γ) |
//! | marks | NCC chain, highest peak | + competing NCC peaks resolved by period continuity |
//! | grains | nearest mark when leaving the sequence | best of the nearest 2-3 by overlap correlation |
//! | Classic splices | whole mark periods, 0.75 ms crossfade | NCC-refined, 1.25 ms crossfade |
//!
//! Max behaves as HQ here (per-grain envelope re-colouring is deferred,
//! DESIGN §1.2).
//!
//! ### Timing
//!
//! Every input sample is written to a per-channel ring addressed by absolute
//! index; output sample `t` is produced right after input sample `t` arrives
//! and carries content from about `t − D` (D = [`latency_samples`]). All
//! positions are `f64`. The pitch marks come from the mono `analysis` signal
//! and one grain schedule is applied to every channel, so stereo images stay
//! intact (R§7.1 invariants).
//!
//! Work is strictly sample-accurate and independent of the host block size:
//! input is processed in internal chunks of ≤ 256 samples; mark searches run
//! at the exact sample their input becomes complete and every mark records
//! when it became known; each output sample only uses marks and input that
//! existed at that sample. Combined with hop-aligned [`Shifter::set_control`]
//! calls (DESIGN §2.2) the output is bit-identical for any block sizes.
//!
//! ### Unvoiced input and transitions (R§8.18)
//!
//! Unvoiced (or unlocked) input is output as the exact delayed dry signal
//! `x[t − D]`. Voiced onsets start the PSOLA on a mark with zero content
//! offset (on the newest mark at or before the content when the tracker and
//! the detector agree on a clean period, so the first corrected period comes
//! up to a period sooner), cross-fading from the dry signal under the first
//! grain's rise, and follow the commanded ratio from the first grain
//! boundary on; voiced → unvoiced cross-fades to the dry signal over 4 ms
//! with equal gain (the two are correlated). Engine changes cross-fade over
//! 10 ms (a change requested mid-fade waits for it), with the incoming
//! engine started at the outgoing one's content position; tier changes
//! cross-fade the interpolation kernels over one hop.
//!
//! ### Heard latency
//!
//! D is what the host compensates. The Natural engine's grains wander around
//! it by up to a period ([`ShifterDiagnostics::mean_content_delay`] reports
//! the mean): Studio is centred on D; Live can only use marks already
//! detected and averages about D + 0.45·T below ≈ 250 Hz (Alto/Tenor, 48 kHz:
//! D = 4.44 ms, heard ≈ 6.5 ms at 220 Hz and 8.7 ms at 110 Hz; on the
//! vocadito recordings 5.95 ms on average); Live+ averages D where its
//! look-ahead covers half a period (≥ 220 Hz at 10 ms) and D + 0.45·T below.
//! See `psola::asym_rise` for the measured trade-off with grain quality
//! (DSP-9). Classic's content delay is D plus up to a few periods of splice
//! window.
//!
//! ### Realtime
//!
//! Everything is allocated in [`Shifter::new`] (rings, the ~370 KB sinc
//! tables, mark store, scratch). `process`, `set_control` and `set_engine`
//! never allocate, lock or panic, and every loop is bounded. Non-finite input
//! samples are treated as silence and non-finite controls as neutral. In the
//! Live modes the mark tracker's part of a voiced control update (a seed is
//! the costliest step of the shifter, tens of µs at 192 kHz) runs at the start
//! of the next `process` call, so it never shares a host block with the hop's
//! pitch analysis (PERF-5).

mod classic;
mod latency;
mod lpc;
mod marks;
mod psola;
mod read;

#[cfg(test)]
mod tests;

pub use latency::latency_samples;

use skato_core::interp::{InterpQuality, Interpolator};
use skato_core::ring::Ring;
use skato_core::{EngineKind, LatencyMode, QualityTier, ShiftControl, VoiceRange};

use crate::classic::{Classic, ClassicCtx};
use crate::latency::{MAX_HALF_WIDTH, sanitize_rate, t_max, t_min};
use crate::marks::MarkTracker;
use crate::psola::{MAX_CH, Psola};
use crate::read::{RcTable, Reader, quality_for};

/// Structural configuration of a [`Shifter`]. Built once per engine
/// activation; changing any field means building a new shifter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShifterConfig {
    pub sample_rate: f32,
    /// 1 or 2 (other values are clamped).
    pub channels: usize,
    /// Control hop in samples (`skato_core::hop_for_rate`).
    pub hop: usize,
    pub latency_mode: LatencyMode,
    pub voice_range: VoiceRange,
    /// Reported latency, `= latency_samples(..)` for this mode and range.
    pub latency: usize,
}

/// Counters for tests, the evaluation harness and telemetry (extra public
/// helper; not part of the DESIGN contract). All counts are cumulative since
/// construction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShifterDiagnostics {
    /// Fractional reads made by the engines.
    pub reads: u64,
    /// Reads whose interpolation kernel reached past the newest written input
    /// sample. Always 0: the scheduling guarantees causality.
    pub reads_past_newest: u64,
    /// Largest `floor(pos) + kernel half-width − newest index` over all reads
    /// (≤ 0 when causal; `i64::MIN` before the first read).
    pub max_read_ahead: i64,
    /// PSOLA grains created from pitch marks.
    pub grains: u64,
    /// Grains that reused the previous grain's mark (upward shifts) or
    /// skipped marks (downward shifts).
    pub grain_repeats: u64,
    pub grain_skips: u64,
    /// Grains that fell back to the dry signal because the marks ran out.
    pub dry_grains: u64,
    /// Grains placed on a predicted mark because the next mark was published
    /// too late (SN-9; never two in a row).
    pub predicted_grains: u64,
    /// Grains whose content offset is more than a period behind
    /// (`e < −T`, T the previous grain's period): extra delay a listener
    /// hears as a jump.
    pub grains_beyond_period: u64,
    /// Unvoiced → voiced PSOLA starts.
    pub psola_onsets: u64,
    /// Grain boundaries decided as symmetric Hann halves vs. asymmetric
    /// short-rise ones.
    pub symmetric_boundaries: u64,
    pub asymmetric_boundaries: u64,
    /// Content offset `e = m − s` (samples) of the newest PSOLA grain, and
    /// its range; the content delay is `D − e` (R§2.1 "wander").
    pub content_offset: f64,
    pub content_offset_min: f64,
    pub content_offset_max: f64,
    /// Sum of the content offsets of all PSOLA grains (see
    /// [`ShifterDiagnostics::mean_content_delay`]).
    pub content_offset_sum: f64,
    /// Classic splices (cycles repeated / dropped) and pointer restarts.
    pub classic_repeats: u64,
    pub classic_drops: u64,
    pub classic_resets: u64,
    /// Mark tracker: marks published, seeds, losses of lock, anchor pulls.
    pub marks: u64,
    pub seeds: u64,
    pub lock_losses: u64,
    pub anchor_corrections: u64,
    /// Engine changes requested (and honoured).
    pub engine_switches: u64,
}

impl ShifterDiagnostics {
    /// Mean content delay of the PSOLA grains in samples, `D − mean(e)`: the
    /// latency a listener hears from the Natural engine, as opposed to the
    /// reported (compensated) D. In Live it averages about D + 0.45·T (the
    /// grains can only use marks already detected, R§2.1); in Studio it is
    /// centred on D. `None` before the first grain. `d` is the shifter's D.
    pub fn mean_content_delay(&self, d: usize) -> Option<f64> {
        (self.grains > 0).then(|| d as f64 - self.content_offset_sum / self.grains as f64)
    }
}

/// Internal processing chunk: bounds how far a block's tail can be ahead of
/// the sample being synthesised, so ring sizes never depend on the host block.
const CHUNK: usize = 256;

/// Engine-change crossfade, ms (DESIGN §3.4).
const ENGINE_FADE_MS: f64 = 10.0;

/// Linear per-sample ramp of a control value across one hop.
#[derive(Clone, Copy, Debug)]
struct Ramp {
    value: f64,
    target: f64,
    step: f64,
    left: usize,
}

impl Ramp {
    const fn new(v: f64) -> Self {
        Self { value: v, target: v, step: 0.0, left: 0 }
    }

    fn set(&mut self, target: f64, over: usize) {
        self.target = target;
        self.left = over.max(1);
        self.step = (target - self.value) / self.left as f64;
    }

    /// Value for the next sample.
    #[inline]
    fn tick(&mut self) -> f64 {
        if self.left > 0 {
            self.left -= 1;
            self.value = if self.left == 0 { self.target } else { self.value + self.step };
        }
        self.value
    }
}

#[derive(Clone, Copy, Debug)]
struct EngineFade {
    from: EngineKind,
    pos: u32,
    len: u32,
}

/// The pitch shifter: input rings, mark tracker, Classic and PSOLA engines.
pub struct Shifter {
    cfg: ShifterConfig,
    d: usize,
    channels: usize,
    hop: usize,
    interp: Interpolator,
    rc: RcTable,
    rings: Box<[Ring]>,
    analysis: Ring,
    tracker: MarkTracker,
    psola: Psola,
    classic: Classic,
    ratio: Ramp,
    gamma: Ramp,
    voiced: bool,
    period_hint: f64,
    engine: EngineKind,
    tier: QualityTier,
    fade: Option<EngineFade>,
    /// Engine requested while a fade was running; started when it ends.
    pending: Option<EngineKind>,
    fade_len: u32,
    q: InterpQuality,
    q_prev: InterpQuality,
    q_pos: u32,
    q_len: u32,
    switches: u64,
    /// Live modes: the tracker's update for the last control is still to
    /// run (see [`Shifter::set_control`]).
    tracker_due: bool,
    /// Defer that update to the next `process` call (Live modes).
    defer_tracker: bool,
}

impl Shifter {
    /// Builds a shifter (allocates everything; not for the audio thread).
    /// Starts with the Natural engine at HQ (Classic in Live-Min).
    pub fn new(cfg: ShifterConfig) -> Self {
        let sr = sanitize_rate(cfg.sample_rate);
        let channels = cfg.channels.clamp(1, MAX_CH);
        let hop = cfg.hop.max(1);
        let d = cfg.latency;
        let (tmin, tmax) = (t_min(sr, cfg.voice_range), t_max(sr, cfg.voice_range));
        // History needed behind the newest sample: D, the content offset and
        // grain reach (a few periods), the tracker's regions and backward
        // chain, one chunk. Powers of two; 64 K minimum (R§8.0).
        let need = d + (8.0 * tmax * 1.15) as usize + CHUNK + 1024;
        let cap = (2 * need).max(1 << 16).next_power_of_two();
        let studio = cfg.latency_mode == LatencyMode::Studio;
        let tier = QualityTier::Hq;
        let q = quality_for(tier);
        Self {
            cfg,
            d,
            channels,
            hop,
            interp: Interpolator::new(),
            rc: RcTable::new(),
            rings: (0..channels).map(|_| Ring::new(cap)).collect(),
            analysis: Ring::new(cap),
            // Studio's detector is centred on `newest − D` (DESIGN §3.5), so
            // its period hint describes the input D samples back.
            tracker: {
                let mut t = MarkTracker::new(sr, tmin, tmax, d as f64, if studio { d as f64 } else { 0.0 });
                t.set_continuity(tier != QualityTier::Eco);
                t
            },
            psola: Psola::new(sr, d, MAX_HALF_WIDTH),
            classic: Classic::new(sr, d, MAX_HALF_WIDTH, studio),
            ratio: Ramp::new(1.0),
            gamma: Ramp::new(1.0),
            voiced: false,
            period_hint: 0.0,
            engine: if cfg.latency_mode == LatencyMode::LiveMin { EngineKind::Classic } else { EngineKind::Natural },
            tier,
            fade: None,
            pending: None,
            fade_len: (ENGINE_FADE_MS * 1e-3 * sr).round().max(1.0) as u32,
            q,
            q_prev: q,
            q_pos: 0,
            q_len: 0,
            switches: 0,
            tracker_due: false,
            defer_tracker: cfg.latency_mode != LatencyMode::Studio,
        }
    }

    /// The configuration this shifter was built with.
    pub fn config(&self) -> &ShifterConfig {
        &self.cfg
    }

    /// Clears all signal state (rings, marks, grains, ramps). Keeps the
    /// selected engine and tier.
    pub fn reset(&mut self) {
        for r in self.rings.iter_mut() {
            r.clear();
        }
        self.analysis.clear();
        self.tracker.reset();
        self.psola.reset();
        self.classic.reset();
        self.ratio = Ramp::new(1.0);
        self.gamma = Ramp::new(1.0);
        self.voiced = false;
        self.period_hint = 0.0;
        if let Some(k) = self.pending.take() {
            self.engine = k;
        }
        self.fade = None;
        self.q_prev = self.q;
        self.q_pos = 0;
        self.q_len = 0;
        self.tracker_due = false;
    }

    /// Reported latency D in samples (the configured value).
    pub fn latency(&self) -> usize {
        self.d
    }

    /// The engine selected (the one being faded to, or queued; Classic is
    /// forced in Live-Min).
    pub fn engine(&self) -> EngineKind {
        self.pending.unwrap_or(self.engine)
    }

    /// RT-safe engine / tier selection. Engine changes cross-fade over
    /// ~10 ms; a change requested while a fade is running is applied when it
    /// ends (reversing a fade half-way would put a corner in the crossfade
    /// weight between two differently delayed signals, an audible tick).
    /// Tier changes cross-fade the interpolation kernel over one hop (and do
    /// reverse: the kernels differ only by interpolation error). In Live-Min
    /// only Classic is allowed (a Natural request is ignored).
    pub fn set_engine(&mut self, kind: EngineKind, tier: QualityTier) {
        let kind = if self.cfg.latency_mode == LatencyMode::LiveMin { EngineKind::Classic } else { kind };
        let q = quality_for(tier);
        if q != self.q {
            if q == self.q_prev && self.q_pos < self.q_len {
                // Reverse a kernel fade in progress.
                self.q_pos = self.q_len - self.q_pos;
            } else {
                self.q_pos = 0;
            }
            self.q_prev = self.q;
            self.q = q;
            self.q_len = self.hop as u32;
        }
        self.tier = tier;
        self.tracker.set_continuity(tier != QualityTier::Eco);
        if kind == self.engine() {
            return;
        }
        self.switches += 1;
        if self.fade.is_some() {
            // Queue it (or cancel a queued change back to the fade target).
            self.pending = (kind != self.engine).then_some(kind);
        } else {
            self.start_fade(kind, self.analysis.written());
        }
    }

    /// Starts a crossfade from the current engine to `kind`, effective from
    /// output sample `t`. The incoming engine starts at the content position
    /// the outgoing one is playing (DSP-7): both engines wander around the
    /// dry position by up to a period, and an equal-gain crossfade of two
    /// copies a fraction of a period apart dips by 2-7 dB. While the outgoing
    /// engine is idle the incoming one starts cold (its onset logic aligns it
    /// with the dry signal).
    fn start_fade(&mut self, kind: EngineKind, t: u64) {
        let u = t as f64 - self.d as f64;
        match kind {
            EngineKind::Classic => {
                self.classic.reset();
                if let Some(pos) = self.psola.content_pos(u, &self.rc) {
                    self.classic.start_at(pos, t);
                }
            }
            EngineKind::Natural => {
                self.psola.reset();
                if let Some(pos) = self.classic.content_pos(t, &self.rc) {
                    self.psola.align_next_onset(pos - u);
                }
            }
        }
        self.fade = Some(EngineFade { from: self.engine, pos: 0, len: self.fade_len });
        self.engine = kind;
    }

    /// Control update at a hop boundary: the pitch and formant ratios ramp
    /// linearly to the new values over the next `hop` samples; `voiced` and
    /// `period_hint` drive the mark tracker and the voiced/dry decision.
    ///
    /// In the Live modes the tracker's part of a voiced update (which at an
    /// onset seeds a new mark run: tens of µs at 192 kHz) runs at the start
    /// of the next [`Shifter::process`] call instead, so it does not land in
    /// the host block that also holds the hop's pitch analysis (PERF-5). The
    /// analysis ring does not change in between, so the output is identical;
    /// only [`Shifter::mark_cents`] read between the two calls still shows
    /// the previous state (skato_engine reads it before `set_control`). Studio
    /// runs it here: skato_engine hands Studio controls over before the first
    /// sample after the boundary, and its pipelined analysis stages start a
    /// quarter hop later, so with blocks of up to a quarter hop the update
    /// already has a callback of its own.
    pub fn set_control(&mut self, ctl: &ShiftControl) {
        self.run_tracker_update();
        let ratio = if ctl.ratio.is_finite() && ctl.ratio > 0.0 { (ctl.ratio as f64).clamp(0.25, 4.0) } else { 1.0 };
        let gamma = if ctl.formant_ratio.is_finite() && ctl.formant_ratio > 0.0 {
            (ctl.formant_ratio as f64).clamp(0.5, 2.0)
        } else {
            1.0
        };
        self.ratio.set(ratio, self.hop);
        self.gamma.set(gamma, self.hop);
        self.voiced = ctl.voiced;
        self.period_hint =
            if ctl.period_hint.is_finite() && ctl.period_hint > 0.0 { ctl.period_hint as f64 } else { 0.0 };
        self.tracker_due = true;
        // An unvoiced update only unlocks (cheap); run it now so the mark
        // pitch is withdrawn with the control.
        if !self.defer_tracker || !self.voiced {
            self.run_tracker_update();
        }
    }

    /// The mark tracker's update for the last control, if still due.
    fn run_tracker_update(&mut self) {
        if self.tracker_due {
            self.tracker_due = false;
            self.tracker.on_control(&self.analysis, self.voiced, self.period_hint);
        }
    }

    /// In-place processing of n samples. `analysis` is the mono analysis
    /// signal (length n) used for pitch marks; `io[ch]` holds the input and is
    /// overwritten with the output. All slices should have length n; only the
    /// common prefix is processed, and channels beyond the configured count
    /// are left untouched (missing ones are treated as silence).
    pub fn process(&mut self, analysis: &[f32], io: &mut [&mut [f32]]) {
        self.run_tracker_update();
        let nch = io.len().min(self.channels);
        let n = io.iter().take(nch).fold(analysis.len(), |n, c| n.min(c.len()));
        let mut off = 0;
        while off < n {
            let len = (n - off).min(CHUNK);
            let base = self.analysis.written();
            for &x in analysis.get(off..off + len).unwrap_or(&[]) {
                self.analysis.push(clean(x));
            }
            for (c, ring) in self.rings.iter_mut().enumerate() {
                match io.get(c).filter(|_| c < nch).and_then(|ch| ch.get(off..off + len)) {
                    Some(xs) => xs.iter().for_each(|&x| ring.push(clean(x))),
                    None => (0..len).for_each(|_| ring.push(0.0)),
                }
            }
            self.tracker.advance(&self.analysis);
            for i in 0..len {
                let y = self.tick(base + i as u64);
                for (c, ch) in io.iter_mut().take(nch).enumerate() {
                    if let (Some(o), Some(&v)) = (ch.get_mut(off + i), y.get(c)) {
                        *o = v;
                    }
                }
            }
            off += len;
        }
    }

    /// Instantaneous input pitch (absolute cents) from the newest pitch-mark
    /// spacing while the mark tracker is locked (R§7.6.2 A).
    pub fn mark_cents(&self) -> Option<f32> {
        self.tracker.mark_cents()
    }

    /// Cumulative counters (see [`ShifterDiagnostics`]).
    pub fn diagnostics(&self) -> ShifterDiagnostics {
        let ps = &self.psola.stats;
        let cs = &self.classic.stats;
        let mut read = ps.read;
        read.merge(&cs.read);
        ShifterDiagnostics {
            reads: read.reads,
            reads_past_newest: read.past_newest,
            max_read_ahead: read.max_ahead,
            grains: ps.grains,
            grain_repeats: ps.repeats,
            grain_skips: ps.skips,
            dry_grains: ps.dry_grains,
            predicted_grains: ps.predicted,
            grains_beyond_period: ps.beyond_period,
            psola_onsets: ps.onsets,
            symmetric_boundaries: ps.symmetric,
            asymmetric_boundaries: ps.asymmetric,
            content_offset: ps.e_last,
            content_offset_min: ps.e_min,
            content_offset_max: ps.e_max,
            content_offset_sum: ps.e_sum,
            classic_repeats: cs.repeats,
            classic_drops: cs.drops,
            classic_resets: cs.resets,
            marks: self.tracker.marks,
            seeds: self.tracker.seeds,
            lock_losses: self.tracker.losses,
            anchor_corrections: self.tracker.anchors,
            engine_switches: self.switches,
        }
    }

    /// One output sample per channel for input index `t` (already written).
    fn tick(&mut self, t: u64) -> [f32; MAX_CH] {
        let beta = self.ratio.tick();
        let gamma = self.gamma.tick();
        let blend = if self.q_pos < self.q_len {
            self.q_pos += 1;
            self.q_pos as f32 / self.q_len as f32
        } else {
            1.0
        };
        let reader = Reader { interp: &self.interp, q: self.q, q_prev: self.q_prev, blend };
        let period = self.tracker.period_at(t + 1).unwrap_or(self.period_hint);
        let pcx = psola::Ctx {
            rings: &self.rings,
            analysis: &self.analysis,
            marks: &self.tracker.store,
            reader,
            rc: &self.rc,
            beta_target: self.ratio.target,
            gamma,
            voiced: self.voiced,
            fallback_period: if period > 0.0 { period } else { 256.0 },
            period_hint: self.period_hint,
            hq: self.tier != QualityTier::Eco,
        };
        let ccx = ClassicCtx {
            rings: &self.rings,
            analysis: &self.analysis,
            reader,
            rc: &self.rc,
            ratio: beta,
            ratio_bound: beta.max(self.ratio.target),
            voiced: self.voiced,
            period,
            hq: self.tier != QualityTier::Eco,
        };
        let mut out = [0.0f32; MAX_CH];
        run(self.engine, &mut self.psola, &mut self.classic, &pcx, &ccx, t, &mut out);
        if let Some(mut f) = self.fade {
            let mut from = [0.0f32; MAX_CH];
            run(f.from, &mut self.psola, &mut self.classic, &pcx, &ccx, t, &mut from);
            f.pos += 1;
            let w = self.rc.rc(f.pos as f64 / f.len as f64);
            for (o, &a) in out.iter_mut().zip(&from) {
                *o = a + w * (*o - a);
            }
            if f.pos >= f.len {
                match f.from {
                    EngineKind::Classic => self.classic.reset(),
                    EngineKind::Natural => self.psola.reset(),
                }
                self.fade = None;
                if let Some(k) = self.pending.take()
                    && k != self.engine
                {
                    self.start_fade(k, t + 1);
                }
            } else {
                self.fade = Some(f);
            }
        }
        out
    }
}

/// Runs one engine for one sample.
#[inline]
fn run(
    kind: EngineKind,
    psola: &mut Psola,
    classic: &mut Classic,
    pcx: &psola::Ctx,
    ccx: &ClassicCtx,
    t: u64,
    out: &mut [f32; MAX_CH],
) {
    match kind {
        EngineKind::Natural => psola.tick(t, pcx, out),
        EngineKind::Classic => classic.tick(t, ccx, out),
    }
}

/// Non-finite samples count as silence (DESIGN §2.1 rule 3).
#[inline]
fn clean(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}
