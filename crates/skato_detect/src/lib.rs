//! Realtime monophonic pitch detection for Skatotune (DESIGN §3.2; R§4.5,
//! R§7.3, R§8.0-R§8.10).
//!
//! ## Pipeline
//!
//! ```text
//! push(x) per sample:  sanitise → DC blocker ─┬─ full-rate history (refinement, level, ZCR)
//!                                             ├─ 4 kHz high-pass history (sibilance)
//!                                             ├─ ÷F decimator → ~16 kHz verifier history (Eco)
//!                                             └─ ÷F decimator → ~24 kHz verifier history (HQ/Max)
//!                                                  └─ ÷2 → ~12 kHz → E/H recursion (all lags)
//! analyze() per hop (= begin_analysis + 4 stages, see "Split analysis"):
//!   begin   E/H minima ──────────┐  (pitch-synchronous, 2 periods of the sung note)
//!   Frames  long type-I CMNDF ───┼─ pYIN probabilities per source → merged candidates (≤ 12)
//!           short weighted (HQ) ─┘
//!   Checks  → octave checks (full-rate / NSDF / E/H / prime-harmonic, rough
//!             phonation; Eco: time-domain rough phonation) and third-harmonic checks
//!   Track   → pitch HMM (forward; Studio: fixed-lag smoothing)  → chosen candidate
//!   Output  → refinement (pitch-synchronous window; parabolic, Max: exact fractional lag)
//!           → voicing FSM (gate + hysteresis, sibilance, clarity, release tails,
//!             onset checks, onset/hold)
//!           → mean Kalman (innovation gating; Max: lag compensation of `cents`)
//! ```
//!
//! **Onsets.** A note's attack makes each period louder than the last, which
//! type-I periodicity reads as aperiodicity. The refinement (always) and,
//! in a Live onset phase (the level ≥ 9 dB above its minimum of the last
//! 40 ms), the E/H heights compare periods gain-invariantly where their
//! shapes match (`eh::attack_corr`), so low voices are confirmed about two
//! periods into the note instead of after the attack. A voiced segment that
//! starts after silence must also start on a candidate the newest frame
//! supports (≥ 80 % of its best probability) and that the long frame finds
//! periodic at all (first-formant ringing of the first pulses is periodic
//! over 4 ms, but not across glottal cycles). Vocadito (Live HQ, 40 files):
//! median voicing-onset delay 9.3 → 6.7 ms with unchanged voicing precision
//! and fewer false voicings far from notes (3.8 → 3.7 %).
//!
//! **What may become voiced** (the corrector decides a note from the first
//! voiced hops, so a wrong or stale voiced stretch before a note becomes a
//! wrong note): the onset hops must agree in pitch
//! (`ONSET_CONSISTENT_CENTS`), an onset after silence must start inside the
//! preset's range (`ONSET_MAX_ABOVE_RANGE_CENTS`) and, above 150 Hz, needs a
//! third confirming hop (`ONSET_EXTRA_HOP_ABOVE_HZ`); a segment younger than
//! 12 ms is not held (`MIN_HOLD_AGE_S`); and a release tail more than 18 dB
//! under its note is unvoiced (`TAIL_DB`, a relative silence threshold
//! against a running peak, R§8.8). Inside a phrase a hop as loud as its
//! segment is held even when its periodicity collapses in a fast legato
//! leap (`STEADY_HOLD_DB`). Vocadito, Live HQ (detector alone, 763 note
//! starts): voiced hops in the gap before a note on 44 → 21 % of the starts,
//! a > 300 c estimate before the first good one on 18 → 9 %, voiced hops far
//! from any note 8.6 → 3.0 %, voicing F1 0.938 → 0.954, first estimate
//! within 50 c 10.7 → 12.2 ms after the start (median), RPA50 −0.5 points.
//! Through Live HQ hard tune: onset glitches 13.9 → 7.6 %, warble episodes
//! 909 → 738.
//!
//! The decimated analyses decide *which* period is sung (octave, voicing);
//! the reported pitch comes from the refinement on the undecimated signal.
//!
//! ## Timing ("information lag")
//!
//! The refinement compares the newest W = max(P, 4 ms) samples with the
//! signal one period earlier, so its evidence spans W + P and is centred
//! (W + P)/2 before the newest sample. In [`Alignment::Causal`] that distance
//! is `info_lag` (plus the decimator delay when Eco refines on its 16 kHz
//! stream) and `sample_pos = newest − info_lag`: ≈ one period below 250 Hz,
//! ≈ 2-3 ms above. The constant-velocity Kalman adds no lag on ramps (vibrato
//! slopes, glides).
//!
//! Measured step response (Alto/Tenor, 48 kHz; time from the step until the
//! estimate crosses the midpoint for good, against the newest sample):
//!
//! | step (all tiers) | 110 Hz (P 9.1 ms) | 220 Hz (P 4.5 ms) | 440 Hz (P 2.3 ms) |
//! |---|---|---|---|
//! | reported `info_lag` | 8.5-11 ms | 4.0-5.4 ms | 2.9-3.5 ms |
//! | ±40 c | 10.7-12.0 ms | 6.7 ms | 4.0-5.3 ms |
//! | ±200 c | 12.0-13.3 ms | 8.0-12.0 ms | 5.3 ms |
//! | ±500 c | 12.0-17.3 ms | 8.0-12.0 ms | 6.7-8.0 ms |
//!
//! i.e. `info_lag` + 1.5-2.5 ms for small steps (Kalman settling) and up to
//! ≈ 1.4 *new* periods for leaps (the new period must fill the
//! pitch-synchronous window); all below DESIGN's 1.5·W/2 = 20.6 ms. Vibrato
//! (6 Hz, ±50 c) error is 1.6-3.5 c RMS against `sample_pos`, 6-15 c against
//! the newest sample; Max's lag compensation brings the latter to 2.5-8.8 c.
//!
//! In [`Alignment::Centered`] (Studio) the estimate describes exactly
//! `newest − delay`: observation frames are centred `delay − L·hop` back, the
//! pitch HMM is smoothed over a fixed lag of L hops (L ≤ 8, the delay beyond
//! what the long frame needs for centring), and the refinement and voicing
//! features are centred on the output instant. Measured residual step lag
//! against `sample_pos`: 0.7-2.7 ms.
//!
//! ## Tiers (DESIGN §3.2)
//!
//! | | Eco | HQ | Max |
//! |---|---|---|---|
//! | verifier rate | ~16 kHz | ~24 kHz | ~24 kHz |
//! | verifier | long CMNDF; rough-phonation rule on the time-domain measures | + short weighted frame, + prime-harmonic octave score (rough phonation on it) | as HQ |
//! | HMM bins | 20 c | 10 c | 5 c |
//! | refinement | full rate above 400 Hz, 16 kHz stream below | full rate, parabolic | full rate; exact fractional lag below 160-sample periods |
//! | `cents` | Kalman mean | Kalman mean | + lag compensation `v·info_lag`, clipped ±25 c (Causal) |
//!
//! ## Cost (Apple M-series, 48 kHz, 64-sample hop = 1.33 ms; `benches/detect.rs`)
//!
//! `push(64) + analyze()` per hop (2026-10-09, 3·2^k correlation sizes): Eco
//! 6.6-11.8 µs, HQ 11.7-21.9 µs, Max 17.2-24.2 µs in Live; Studio 12.4-19.1 /
//! 22.8-35.4 / 40.8-50.2 µs (the fixed-lag backward passes).
//! p99.9 ≤ 115 µs. `push` alone: 2-4 µs per hop.
//!
//! Tail after PERF-1..4 (per-hop minimum of 9 replays, `analyze` alone,
//! before → after, µs mean / p99.9): Alto/Tenor melody Live Eco 5.2 / 17.7 →
//! 5.1 / 11.8, Live HQ 10.5 / 23.7 → 10.2 / 17.4, Studio Max 26.6 / 38.5 →
//! 26.3 / 33.0; a jittered 300 Hz pulse train (exact octave confirms on 6 %
//! of hops) Live HQ 11.4 / 36.3 → 10.6 / 19.5. `push` −0.3 µs per hop (E/H
//! newer-half energies on demand).
//!
//! ## Split analysis (PERF-7)
//!
//! [`Detector::analyze`] is [`Detector::begin_analysis`] at the hop boundary
//! (snapshot of the stream positions and the E/H periodicity, ≈ 0.5 µs)
//! followed by [`ANALYSIS_STEPS`] stages run by
//! [`Detector::step_analysis`]: `Frames` (long and short verifier frames),
//! `Checks` (octave measures, full-rate checks, candidate probabilities),
//! `Track` (pitch HMM; Studio's fixed-lag backward passes) and `Output`
//! (refinement, voicing, Kalman, estimate). An engine may run the stages in
//! later callbacks while it pushes up to one more hop of samples (skato_engine's
//! Studio pipeline runs them a quarter, three eighths, a half and three
//! quarters of a hop after the boundary): every
//! stage reads the histories at the boundary's positions (each history keeps
//! a hop of slack), so the estimate is bit-identical to a one-shot
//! `analyze` at the boundary. Per-stage cost, 48 kHz Alto/Tenor melody,
//! mean / p99.9 µs: begin 0.5 / 1.8, Frames 6.3 / 8.4, Checks 1.7 / 8.5,
//! Track 1.7 / 1.8 (Live) or 8.9 / 9.8 (Studio HQ) or 17.9 / 20.0 (Studio
//! Max), Output 0.4 / 1.1.
//!
//! Every buffer is allocated in [`Detector::new`] for the Max tier; `push`,
//! `analyze` (and its stages), `set_tier`, `set_tracking` and `reset` never
//! allocate, lock or panic, and do bounded work (DESIGN §2.1). Output is
//! bit-identical for the same input whatever the host block sizes, provided
//! `analyze` runs at the same sample counts (the FFTs use realfft, see
//! `verifier::FFT_BACKEND`).

mod cands;
mod eh;
mod hist;
mod hmm;
mod kalman;
mod pyin;
mod refine;
mod verifier;
mod voicing;

use skato_core::filter::{Biquad, DcBlocker, Decimator};
use skato_core::math::{cents_to_hz_fast, hz_to_cents_fast};
use skato_core::{PitchEstimate, QualityTier, VoiceRange};

use cands::{CandSet, MAX_CANDS, Source};
use eh::{EhTracker, EhTrough, MAX_EH_TROUGHS};
use hist::Mirror;
use hmm::{MAX_OBS, ObsEntry, PitchHmm};
use kalman::{MeanKalman, Q_LIVE, Q_STUDIO};
use pyin::{MAX_TROUGHS, RankPrior, ThresholdPrior, trough_probabilities};
use refine::{Refined, Refiner};
use verifier::{LONG_WINDOW_PERIODS, LongFrame, ShortFrame};
use voicing::{Features, GATE_HYST_DB, Gate, VoicingFsm};

/// Where the analysis windows sit relative to the newest input sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Alignment {
    /// Live modes: analysis uses only past samples; estimate describes ~newest − info_lag.
    Causal,
    /// Studio: the estimate describes `newest − delay`; windows centred there (W ≤ 2·delay).
    Centered { delay: usize },
}

/// Structural detector configuration (fixed for the detector's lifetime).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectorConfig {
    pub sample_rate: f32,
    pub voice_range: VoiceRange,
    /// Control hop in samples (`skato_core::hop_for_rate`); `analyze` is called
    /// once per hop. 0 selects `hop_for_rate(sample_rate)` (and
    /// [`Detector::config`] then reports the resolved hop).
    pub hop: usize,
    pub alignment: Alignment,
}

// ---------------------------------------------------------------------------
// Constants (each traced to research.md)
// ---------------------------------------------------------------------------

/// Search (E/H) stream target rate, Hz (R§8.2: 8-12 kHz, ≥ 12 kHz for
/// soprano). Derived from the ~24 kHz verifier stream by a second decimator
/// (cascade: the anti-alias filter then runs at 24 kHz instead of the host
/// rate, 8× cheaper at 192 kHz).
const SEARCH_RATE: f32 = 12_000.0;
/// Verifier target rates, Hz (DESIGN §3.2: Eco ~16 kHz, HQ/Max ~24 kHz).
const ECO_RATE: f32 = 16_000.0;
const HQ_RATE: f32 = 24_000.0;
/// Lag ranges extend ±6 % (≈ 1 semitone) past the preset so pitches at the
/// preset edges are interior minima.
const LAG_MARGIN: f32 = 1.06;
/// HMM range margin beyond the preset, cents.
const HMM_MARGIN_CENTS: f32 = 110.0;
/// Fixed-lag smoothing depth cap, hops (Studio): 8 hops ≈ 10.7-11.6 ms. The
/// benefit of longer lags is unproven (R§8.6 "evidence gap") and each hop of
/// lag costs one backward HMM pass per analysis; the rest of the delay
/// centres the windows (measured: centring alone leaves ≤ 3 ms of step lag).
const MAX_FIXED_LAG: usize = 8;
/// Level / sibilance window, seconds (R§8.8 "10 ms RMS").
const LEVEL_WINDOW_S: f32 = 0.010;
/// Voicing hold, seconds (R§8.8: 20-40 ms; short end, see `voicing`).
const HOLD_S: f32 = 0.020;
/// Hops of consecutive raw voicing before an onset (R§8.8).
const ONSET_HOPS: u32 = 2;
/// The onset hops must agree in pitch: a raw hop more than this from the
/// previous one restarts the confirmation. Breath and consonant resonances
/// before a note read a different pitch every hop (vocadito: the first
/// voiced hops of a note start sat 300-3900 c off on 18 % of onsets).
const ONSET_CONSISTENT_CENTS: f32 = 50.0;
/// A voicing onset after silence above this f0 needs one more confirming
/// hop (3 instead of [`ONSET_HOPS`]): the resonances that opened notes on a
/// wrong pitch were nearly all well above it (vocadito: formant and
/// fricative resonances at 440-1500 Hz), and one hop is a small share of
/// such an onset, while at low f0 (12 ms periods) a lost hop compounds (an
/// extra hop for every onset made the synthetic 82 Hz onset settle 14 ms
/// later). Vocadito, Live HQ hard tune: onset glitches 8.4 → 7.6 % against a
/// 400 Hz limit, 7.8 % at 250 Hz; detector onset delay +1.4 ms.
const ONSET_EXTRA_HOP_ABOVE_HZ: f32 = 150.0;
/// A voiced segment may hold through weak hops only once it has been
/// raw-voiced this long: a 1-3 hop segment is more likely a resonance than
/// a note, and holding it published a stale pitch for the whole hold.
/// Vocadito, Live HQ hard tune: onset glitches 12.7 → 9.8 % (8 ms: 11.2 %;
/// 16 ms: 9.6 % with slower settles; only for segments after silence:
/// 11.5 %).
const MIN_HOLD_AGE_S: f32 = 0.012;
/// Inside a held-voiced segment, a hop whose level stays within this of
/// the segment's reference level ([`TAIL_DB`]) is holdable even when its
/// periodicity collapses: in fast legato leaps the period changes inside
/// the pitch-synchronous window (clarity 0.2-0.3 for 2-6 ms), and each
/// unvoiced hop restarted the correction downstream (vocadito: 8.7 such
/// dropouts per voiced minute, 74 % at note boundaries). A new attack
/// (level rise) is never held this way: it may be the next note. Synthetic
/// legato leaps (6 shapes × 3 seeds): 34 → 3 dropouts; vocadito Live HQ hard
/// tune settle p90 −1.7 ms.
const STEADY_HOLD_DB: f32 = 6.0;
/// Eco refines on the 16 kHz stream below this f0 (DESIGN §3.2).
const ECO_FULL_RATE_ABOVE_HZ: f32 = 400.0;
/// Shortest refinement window, seconds: one period, but at least this, so
/// high notes average more than a ~1 ms cycle of noise (lag cost: the
/// evidence centre moves from P to (W + P)/2 behind the newest sample).
const MIN_REFINE_WINDOW_S: f32 = 0.004;
/// Refinement scratch: most lags evaluated per refinement.
const MAX_REFINE_LAGS: usize = 160;
/// Refinement half-band: at least this fraction of the seed period …
const REFINE_BAND_REL: f64 = 0.03;
/// … but at most this many lags each side (bounds the work at 192 kHz, where
/// ±64 lags is still ±1.7 % of a 55 Hz period).
const MAX_REFINE_BAND: usize = 64;
/// Max-tier lag compensation clip, cents (R§8.10).
const LAG_COMP_CLIP_CENTS: f64 = 25.0;
/// Sibilant frames keep this share of their candidate probability.
const SIBILANT_ATTENUATION: f32 = 0.1;
/// Input clamp (≈ +36 dBFS): finite but absurd input cannot wreck the f64
/// accumulators' precision for long.
const INPUT_CLAMP: f32 = 64.0;
/// Number of candidates exposed by [`Detector::candidates`].
const OUT_CANDS: usize = 5;
/// Candidates with periods below this (seconds) get a full-rate periodicity
/// check for the octave decisions …
const FULL_CHECK_S: f32 = 0.006;
/// … capped at this many full-rate samples (bounds the cost at 192 kHz) …
const FULL_CHECK_MAX_SAMPLES: f32 = 400.0;
/// … for at most this many candidates per hop; at most this many of them
/// are re-measured at the exact fractional lag when they would decide an
/// octave-up move.
const MAX_FULL_CHECKS: usize = 6;
const MAX_EXACT_CONFIRMS: usize = 4;
/// Onset phase: the observation level has risen by at least this much over
/// the quietest 10 ms level of the last [`ONSET_WINDOW_S`] (a note attack,
/// not the ±5 dB fluctuation of noise). Only then do the E/H heights use the
/// attack-compensated form (`eh` module docs).
const ONSET_RISE_DB: f32 = 9.0;
/// Level history for the onset phase, seconds.
const ONSET_WINDOW_S: f32 = 0.040;
/// A voiced segment that starts after silence must start on a candidate the
/// newest frame supports: its observation probability at least this share
/// of the frame's best. Otherwise the HMM posterior, built during the
/// breath or consonant before the note, can open it on a stale or harmonic
/// candidate for the first hops (vocadito: such onsets are the corrector's
/// worst glitches).
const ONSET_AGREE: f32 = 0.8;
/// … and on a candidate the long verifier frame (W = 2.2/f_min) finds
/// periodic at all: `1 − nsdf` (attack-compensated, `Cand::g_onset`) at
/// most this. The first-formant ringing of the first glottal pulses of a
/// low note is periodic over a 4 ms refinement window (clarity 0.84-0.97 at
/// F1 ≈ 750 Hz under a 98 Hz /a/), but not across the glottal cycles the
/// long frame spans, while the sung pitch passes even with the frame still
/// partly in the silence before the note. Vocadito (Live HQ): false voicing
/// far from notes 3.81 → 3.67 %, bad note starts 21 → 20, median onset
/// +0.5 ms (0.5 rejects a little more but delays scooped onsets by up to
/// 7 ms; 0.6 lets the 98 Hz /a/ ringing through).
const ONSET_MAX_G_LONG: f32 = 0.55;
/// A voicing onset after silence must start at most this far above the
/// preset's top pitch (`VoiceRange::f_max`, refined pitch): the lag ranges
/// extend 6 % past it so the edge is an interior minimum, but before a note
/// breath and fricative resonances sit just there (vocadito, Live HQ: 8 of
/// the 14 onsets that opened on a pitch near the top of the range opened
/// above it, by 7-190 c). Inside a voiced segment the margin stays usable.
const ONSET_MAX_ABOVE_RANGE_CENTS: f32 = 20.0;
/// `Cand::g_long` at or above this is "unknown".
const UNKNOWN_G: f32 = 1.5;
/// Release tails: a hop whose output-instant level is more than this below
/// the reference level of the voiced segment (its loudest hop, decaying at
/// [`TAIL_DECAY_DB_PER_S`], restarted by a new attack) is not voiced. A
/// decaying note tail or room tail keeps its pitch periodic for a long time
/// (vocadito: voiced down to −25 dB under the note, through gaps of 100-200
/// ms, so the control law carried the old note into the next onset). This
/// is R§8.8's relative silence threshold against a running peak. Live HQ on
/// vocadito: voiced hops where the annotation is unvoiced −35 %,
/// annotated-voiced hops lost 0.4 % (0.08 % inside notes); hard-tune onset
/// glitches 13.9 → 12.7 %, warble episodes 909 → 755 (15 dB: 12.1 %, 699,
/// but more dropouts and a higher Studio p95; 22 dB: 12.7 %, 799).
const TAIL_DB: f32 = 18.0;
/// Decay of the release-tail reference, dB/s (a long decrescendo lowers it;
/// 10 and 5 dB/s cut more tails but lengthened the slowest settles).
const TAIL_DECAY_DB_PER_S: f32 = 20.0;

/// Records `level_db` in a ring of recent hop levels and reports whether it
/// is at least [`ONSET_RISE_DB`] above the quietest of them (an attack).
fn level_rise(ring: &mut [f32], head: &mut usize, level_db: f32) -> bool {
    let level = if level_db.is_finite() { level_db } else { -160.0 };
    let floor = ring.iter().copied().fold(f32::INFINITY, f32::min);
    if let Some(slot) = ring.get_mut(*head) {
        *slot = level;
    }
    *head = (*head + 1) % ring.len().max(1);
    level - floor >= ONSET_RISE_DB
}

/// Sanitised sample rate.
fn sane_rate(sr: f32) -> f32 {
    if sr.is_finite() && sr >= 8_000.0 { sr.min(768_000.0) } else { 48_000.0 }
}

/// Rate of the HQ verifier stream for host rate `fs`.
fn rate_h_of(fs: f32) -> f32 {
    fs / ((fs / HQ_RATE).round() as usize).max(1) as f32
}

/// Integer lag range `[⌊rate/(f_max·m)⌋, ⌈rate·m/f_min⌉]` at a stream rate.
fn lag_range(rate: f32, f_min: f32, f_max: f32) -> (usize, usize) {
    let lo = ((rate / (f_max * LAG_MARGIN)).floor() as usize).max(2);
    let hi = ((rate * LAG_MARGIN / f_min).ceil() as usize).max(lo + 2);
    (lo, hi)
}

/// One decimated analysis stream.
struct Stream {
    dec: Decimator,
    hist: Mirror,
    factor: usize,
    rate: f32,
    /// Decimator group delay, input samples.
    gd: f64,
}

impl Stream {
    fn new(fs: f32, target: f32, capacity: usize) -> Self {
        let factor = ((fs / target).round() as usize).max(1);
        let dec = Decimator::new(fs, factor);
        let gd = dec.group_delay() as f64;
        Self { dec, hist: Mirror::new(capacity), factor, rate: fs / factor as f32, gd }
    }

    /// A stream decimated further from `parent`'s output (fed with the
    /// parent's samples). `factor` and `gd` stay relative to the host rate:
    /// the k-th output is parent sample `k·f2 + f2 − 1`, i.e. host time
    /// `k·F + F − 1 − gd` with `F = F_parent·f2`, `gd = gd_parent + gd₂·F_parent`.
    fn cascade(parent_factor: usize, parent_rate: f32, parent_gd: f64, target: f32, capacity: usize) -> Self {
        let f2 = ((parent_rate / target).round() as usize).max(1);
        let dec = Decimator::new(parent_rate, f2);
        let gd = parent_gd + dec.group_delay() as f64 * parent_factor as f64;
        let factor = parent_factor * f2;
        Self { dec, hist: Mirror::new(capacity), factor, rate: parent_rate / f2 as f32, gd }
    }

    /// Full-rate time (input sample index) stream sample `i` represents.
    fn time_of(&self, i: f64) -> f64 {
        i * self.factor as f64 + (self.factor - 1) as f64 - self.gd
    }

    /// Exclusive stream end index whose last sample represents full-rate time
    /// `t`, clamped to the `len` samples available at the hop boundary.
    fn end_at(&self, t: f64, len: i64) -> i64 {
        let i = ((t + self.gd - (self.factor - 1) as f64) / self.factor as f64).round();
        if i.is_finite() { (i as i64 + 1).clamp(0, len) } else { len }
    }

    fn reset(&mut self) {
        self.dec.reset();
        self.hist.clear();
    }
}

/// Studio geometry (all in full-rate samples).
#[derive(Clone, Copy, Debug)]
struct Studio {
    delay: usize,
    /// Observation centre offset `delay − lag_frames·hop`.
    d_obs: usize,
    lag_frames: usize,
}

/// Tracking-derived parameters (R§8.3 "one knob … maps onto eps, the NSDF
/// clarity threshold, the pYIN threshold prior and the HMM switch
/// probability"). Mapping is a design choice, tuned by listening.
#[derive(Clone, Copy, Debug)]
struct TrackingParams {
    /// E/H periodicity threshold, also the mean of the fast threshold prior.
    eps: f32,
    /// Clarity needed for a voiced hop.
    clarity: f32,
    /// Mean of the long-frame (CMNDF) threshold prior.
    long_mean: f32,
    /// HMM voicing switch probability at pYIN's hop.
    switch_ref: f32,
}

impl TrackingParams {
    /// `tracking` 0 = relaxed (breathy/noisy pitch accepted) … 100 = choosy.
    fn new(tracking: f32) -> Self {
        let t = if tracking.is_finite() { (tracking / 100.0).clamp(0.0, 1.0) } else { 0.5 };
        Self {
            // 0.40 → 0.14 → 0.05, geometric (R§8.3 range 0.05-0.40).
            eps: 0.40 * (0.05f32 / 0.40).powf(t),
            // 0.5 (relaxed) … 0.7 (choosy) (R§8.8).
            clarity: 0.5 + 0.2 * t,
            // Beta prior mean 0.2 → 0.1 (pYIN's Beta(2,18)) → 0.05.
            long_mean: 0.1 * 2f32.powf(1.0 - 2.0 * t),
            // Relaxed voices stay voiced through breathy frames: lower switch
            // probability. 0.005 → 0.01 (librosa) → 0.02.
            switch_ref: 0.01 * 2f32.powf(2.0 * t - 1.0),
        }
    }
}

/// Stages of [`Detector::step_analysis`].
pub const ANALYSIS_STEPS: usize = 4;

/// Next stage of the analysis in flight.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Stage {
    #[default]
    Idle,
    /// Long and short verifier frames.
    Frames,
    /// Octave measures, full-rate checks, candidate probabilities.
    Checks,
    /// Pitch HMM forward step (Studio: fixed-lag smoothing), candidate choice.
    Track,
    /// Output features, refinement, voicing, Kalman, estimate.
    Output,
}

/// The analysis in flight: the hop boundary's stream positions and what one
/// stage hands to the next.
#[derive(Clone, Copy, Debug, Default)]
struct Job {
    stage: Stage,
    /// Newest sample index at the hop boundary …
    newest: i64,
    /// … and the history lengths then (full rate, Eco and HQ verifier streams).
    full_len: i64,
    eco_len: i64,
    hq_len: i64,
    /// HQ/Max analysis (tier at the boundary).
    hq: bool,
    obs_feat: Features,
    obs_open: bool,
    vp_fast: f32,
    vp_long: f32,
    vp_short: f32,
    long_ok: bool,
    /// Combined voiced probability of the frame.
    vp: f32,
    /// `Track` results: the chosen candidate of [`Detector::job_set`] and
    /// the HMM voiced probability.
    chosen: Option<usize>,
    hmm_vp: f32,
}

/// Realtime pitch detector (see the crate docs).
pub struct Detector {
    cfg: DetectorConfig,
    fs: f32,
    hop_s: f32,
    tier: QualityTier,
    tracking: f32,
    gate_db: f32,
    params: TrackingParams,
    prior_long: ThresholdPrior,
    prior_fast: ThresholdPrior,
    rank: RankPrior,
    hmm_lo: f32,
    hmm_hi: f32,
    /// `VoiceRange::f_max` in cents.
    range_top_cents: f32,
    // --- per-sample state ---
    dc: DcBlocker,
    hp_a: Biquad,
    hp_b: Biquad,
    full: Mirror,
    hp: Mirror,
    search: Stream,
    eco: Stream,
    hq: Stream,
    eh: EhTracker,
    n_pushed: u64,
    // --- per-hop analysis ---
    long_eco: LongFrame,
    long_hq: LongFrame,
    short_hq: ShortFrame,
    refine_full: Refiner,
    refine_eco: Refiner,
    hmm: PitchHmm,
    gate: Gate,
    fsm: VoicingFsm,
    kalman: MeanKalman,
    studio: Option<Studio>,
    level_len: usize,
    /// Observation levels (dBFS) of the last [`ONSET_WINDOW_S`], one per hop (ring).
    levels: Box<[f32]>,
    level_head: usize,
    /// The same at the output instant (Studio: `delay` back; Live: equal).
    out_levels: Box<[f32]>,
    out_level_head: usize,
    /// Release-tail reference level, dBFS ([`TAIL_DB`]).
    tail_ref_db: f32,
    /// Pitch of the last raw-voiced hop while an onset is confirmed.
    onset_cents: Option<f32>,
    /// Candidate sets of the last `lag_frames + 1` frames (ring; Causal: 1).
    frames: Box<[CandSet]>,
    frame_head: usize,
    // --- scratch ---
    troughs: [EhTrough; MAX_TROUGHS],
    heights: [f32; MAX_TROUGHS],
    probs: [f32; MAX_TROUGHS],
    obs: [ObsEntry; MAX_OBS],
    out: [(f32, f32); OUT_CANDS],
    n_out: usize,
    last_info_lag: f32,
    job: Job,
    /// Candidates of the observation frame in flight (`Frames`, `Checks`).
    job_frame: CandSet,
    /// The frame the estimate in flight describes (`Track`, `Output`).
    job_set: CandSet,
}

impl Detector {
    /// Builds a detector; allocates every buffer for `QualityTier::Max`.
    pub fn new(cfg: DetectorConfig) -> Self {
        let fs = sane_rate(cfg.sample_rate);
        let hop = if cfg.hop == 0 { skato_core::hop_for_rate(fs) } else { cfg.hop.min(16_384) };
        let cfg = DetectorConfig { hop, ..cfg };
        let hop_s = hop as f32 / fs;
        let (f_min, f_max) = (cfg.voice_range.f_min(), cfg.voice_range.f_max());

        // Stream geometry (capacities filled in below).
        let fac = |target: f32| ((fs / target).round() as usize).max(1);
        let f_search = fac(HQ_RATE) * ((rate_h_of(fs) / SEARCH_RATE).round() as usize).max(1);
        let rate_s = fs / f_search as f32;
        let rate_e = fs / fac(ECO_RATE) as f32;
        let rate_h = fs / fac(HQ_RATE) as f32;
        let (l_min, l_max) = lag_range(rate_s, f_min, f_max);
        let w_of = |rate: f32| (LONG_WINDOW_PERIODS / f_min * rate).round() as usize;
        let (we, wh) = (w_of(rate_e), w_of(rate_h));
        let (te_min, te_max) = lag_range(rate_e, f_min, f_max);
        let (th_min, th_max) = lag_range(rate_h, f_min, f_max);
        // Short frame: W/2 with lags up to 0.5-0.6·W/2 (R§8.4), trimmed so the
        // buffer fits the correlation size for W/2·1.5
        // (`verifier::correlation_len`: halves the FFT for Alto/Tenor at
        // 24 kHz, 512 instead of 1024, and gives Low Male / Wide 768 instead
        // of 1024 at one lag less, 287 instead of 288).
        let ws = wh / 2;
        let n_short = verifier::correlation_len(ws + ws / 2 + 1);
        let ts_max = th_max.min((0.6 * ws as f32) as usize).min(n_short.saturating_sub(ws + 1)).max(th_min + 2);

        // Studio: observation centre and fixed lag.
        let studio = match cfg.alignment {
            Alignment::Causal => None,
            Alignment::Centered { delay } => {
                let half = |w: usize, t: usize, f: usize| ((w + t) as f64 * 0.5 * f as f64).ceil() as usize;
                let d_min = half(wh, th_max, fac(HQ_RATE)).max(half(we, te_max, fac(ECO_RATE)));
                let lag_frames = if delay > d_min { ((delay - d_min) / hop).min(MAX_FIXED_LAG) } else { 0 };
                Some(Studio { delay, d_obs: delay - lag_frames * hop, lag_frames })
            }
        };
        let delay = studio.map_or(0, |s| s.delay);
        let d_obs = studio.map_or(0, |s| s.d_obs);

        let eh_centre = studio.map(|s| (s.d_obs as f64 / f_search as f64).round() as usize);
        let eh = EhTracker::new(l_min, l_max, eh_centre);
        // Eco has no harmonic score (DESIGN §3.2), so no spectrum plan.
        let long_eco = LongFrame::new(we, te_min, te_max, false);
        let long_hq = LongFrame::new(wh, th_min, th_max, true);
        let short_hq = ShortFrame::new(ws, th_min, ts_max);
        let p_max_full = (fs * LAG_MARGIN / f_min).ceil() as usize + 2;
        let refine_full = Refiner::new(p_max_full, MAX_REFINE_LAGS);
        let p_max_eco = (rate_e * LAG_MARGIN / f_min).ceil() as usize + 2;
        let refine_eco = Refiner::new(p_max_eco, MAX_REFINE_LAGS);

        let level_len = ((LEVEL_WINDOW_S * fs).round() as usize).max(16);
        // Every history keeps one hop more than the analysis reads, so a split
        // analysis (`begin_analysis` … `finish_analysis`) can finish with a
        // hop of newer samples pushed.
        let full_cap = refine_full.span() + delay + level_len + 64 + hop;
        let stream_cap = |frames: usize, rate: f32, refine_span: usize| {
            let back = ((d_obs.max(delay) + hop) as f32 * rate / fs).ceil() as usize;
            frames.max(refine_span) + back + 64
        };
        let eco = Stream::new(fs, ECO_RATE, stream_cap(long_eco.span(), rate_e, refine_eco.span()));
        let hq = Stream::new(fs, HQ_RATE, stream_cap(long_hq.span().max(short_hq.span()), rate_h, 0));
        let search = Stream::cascade(hq.factor, hq.rate, hq.gd, SEARCH_RATE, eh.span() + 8);
        debug_assert_eq!(search.factor, f_search);

        let hmm_lo = hz_to_cents_fast(f_min) - HMM_MARGIN_CENTS;
        let hmm_hi = hz_to_cents_fast(f_max) + HMM_MARGIN_CENTS;
        let max_bins = ((hmm_hi - hmm_lo) / 5.0).ceil() as usize + 2;
        let lag_frames = studio.map_or(0, |s| s.lag_frames);
        let hmm = PitchHmm::new(max_bins, lag_frames);

        let params = TrackingParams::new(50.0);
        let mut d = Self {
            cfg,
            fs,
            hop_s,
            tier: QualityTier::Hq,
            tracking: 50.0,
            gate_db: -55.0,
            params,
            prior_long: ThresholdPrior::with_mean(params.long_mean),
            prior_fast: ThresholdPrior::with_mean(params.eps),
            rank: RankPrior::new(),
            hmm_lo,
            hmm_hi,
            range_top_cents: hz_to_cents_fast(f_max),
            dc: DcBlocker::new(fs),
            hp_a: Biquad::highpass(fs, 4000.0, std::f32::consts::FRAC_1_SQRT_2),
            hp_b: Biquad::highpass(fs, 4000.0, std::f32::consts::FRAC_1_SQRT_2),
            full: Mirror::new(full_cap),
            hp: Mirror::new(level_len + delay + 64 + hop),
            search,
            eco,
            hq,
            eh,
            n_pushed: 0,
            long_eco,
            long_hq,
            short_hq,
            refine_full,
            refine_eco,
            hmm,
            gate: Gate::default(),
            fsm: VoicingFsm::new(ONSET_HOPS, (HOLD_S / hop_s).ceil() as u32, (MIN_HOLD_AGE_S / hop_s).ceil() as u32),
            kalman: MeanKalman::new(if studio.is_some() { Q_STUDIO } else { Q_LIVE }),
            studio,
            level_len,
            levels: vec![-160.0; ((ONSET_WINDOW_S / hop_s).ceil() as usize).clamp(2, 4096)].into_boxed_slice(),
            level_head: 0,
            out_levels: vec![-160.0; ((ONSET_WINDOW_S / hop_s).ceil() as usize).clamp(2, 4096)].into_boxed_slice(),
            out_level_head: 0,
            tail_ref_db: -160.0,
            onset_cents: None,
            frames: vec![CandSet::default(); lag_frames + 1].into_boxed_slice(),
            frame_head: 0,
            troughs: [EhTrough::default(); MAX_TROUGHS],
            heights: [0.0; MAX_TROUGHS],
            probs: [0.0; MAX_TROUGHS],
            obs: [ObsEntry::default(); MAX_OBS],
            out: [(0.0, 0.0); OUT_CANDS],
            n_out: 0,
            last_info_lag: 0.0,
            job: Job::default(),
            job_frame: CandSet::default(),
            job_set: CandSet::default(),
        };
        d.relayout();
        d.reset();
        d
    }

    /// The configuration this detector was built with (hop resolved).
    pub fn config(&self) -> &DetectorConfig {
        &self.cfg
    }

    /// Clears all history and tracker state (as if newly constructed with the
    /// current tier and tracking settings).
    pub fn reset(&mut self) {
        self.dc.reset();
        self.hp_a.reset();
        self.hp_b.reset();
        self.full.clear();
        self.hp.clear();
        self.search.reset();
        self.eco.reset();
        self.hq.reset();
        self.eh.reset();
        self.n_pushed = 0;
        self.hmm.reset();
        self.gate.reset();
        self.fsm.reset();
        self.kalman.reset();
        for f in self.frames.iter_mut() {
            f.clear();
        }
        self.frame_head = 0;
        self.levels.fill(-160.0);
        self.level_head = 0;
        self.out_levels.fill(-160.0);
        self.out_level_head = 0;
        self.tail_ref_db = -160.0;
        self.onset_cents = None;
        self.n_out = 0;
        self.last_info_lag = self.nominal_info_lag();
        self.job = Job::default();
        self.job_frame.clear();
        self.job_set.clear();
    }

    /// Selects the quality tier. RT-safe: re-lays out the HMM bins (soft reset
    /// of the pitch HMM; the other trackers keep running). Both verifier
    /// streams are always fed, so the switch has no data gap.
    pub fn set_tier(&mut self, tier: QualityTier) {
        if tier != self.tier {
            self.tier = tier;
            self.relayout();
        }
    }

    /// Tracking 0-100 (0 relaxed … 100 choosy) and gate in dBFS.
    pub fn set_tracking(&mut self, tracking: f32, gate_db: f32) {
        let tracking = if tracking.is_finite() { tracking.clamp(0.0, 100.0) } else { 50.0 };
        self.gate_db = if gate_db.is_finite() { gate_db.clamp(-140.0, 0.0) } else { -55.0 };
        if tracking != self.tracking {
            self.tracking = tracking;
            self.params = TrackingParams::new(tracking);
            self.prior_long.set_mean(self.params.long_mean);
            self.prior_fast.set_mean(self.params.eps);
            self.hmm.set_switch(self.params.switch_ref, self.hop_s);
        }
    }

    /// Feeds new input samples (mono analysis signal). Per sample: sanitise,
    /// DC-block, store, high-pass for sibilance, decimate, and run the E/H
    /// recursion on each new search-rate sample.
    pub fn push(&mut self, x: &[f32]) {
        for &v in x {
            let v = if v.is_finite() { v.clamp(-INPUT_CLAMP, INPUT_CLAMP) } else { 0.0 };
            let y = self.dc.process(v);
            self.full.push(y);
            let h = self.hp_b.process(self.hp_a.process(y));
            self.hp.push(h);
            if let Some(e) = self.eco.dec.push(y) {
                self.eco.hist.push(e);
            }
            if let Some(q) = self.hq.dec.push(y) {
                self.hq.hist.push(q);
                if let Some(s) = self.search.dec.push(q) {
                    self.search.hist.push(s);
                    self.eh.update(&self.search.hist);
                }
            }
            self.n_pushed += 1;
        }
    }

    /// Up to 5 `(period_samples, score)` candidates from the last
    /// [`Detector::analyze`], best first. `score` is the candidate's
    /// observation probability after octave checks (the evidence of that
    /// frame alone, before the HMM). The chosen candidate carries its refined
    /// period. In Studio these are the candidates of the frame the estimate
    /// describes.
    pub fn candidates(&self) -> &[(f32, f32)] {
        &self.out[..self.n_out]
    }

    /// Runs the per-hop analysis (call once per `hop` pushed samples, at the
    /// hop boundary). Equivalent to [`Detector::begin_analysis`] followed by
    /// [`Detector::finish_analysis`].
    pub fn analyze(&mut self) -> PitchEstimate {
        self.begin_analysis();
        self.finish_analysis()
    }

    /// Starts the analysis of the hop that ends with the newest pushed
    /// sample: call it exactly at the hop boundary, before any newer sample
    /// is pushed. It snapshots the stream positions and the E/H periodicity
    /// of the boundary (cheap, ≈ 1 µs); the rest of the analysis runs in
    /// [`Detector::step_analysis`] / [`Detector::finish_analysis`], which may
    /// be called later, with up to one hop of newer samples pushed in
    /// between: every stage reads the histories at the boundary's positions,
    /// so the estimate is bit-identical to [`Detector::analyze`] at the
    /// boundary whatever the split (the engine can spread one hop's analysis
    /// over several callbacks, PERF-7). An unfinished analysis is completed
    /// (and its estimate dropped) when the next one begins.
    pub fn begin_analysis(&mut self) {
        if self.job.stage != Stage::Idle {
            let _ = self.finish_analysis();
        }
        let newest = self.n_pushed as i64 - 1;
        let d_obs = self.studio.map_or(0, |s| s.d_obs) as i64;
        let obs_feat = self.features_at(newest - d_obs, newest);
        let obs_open = obs_feat.level_db >= self.gate_db - GATE_HYST_DB;
        // Attack compensation of the E/H heights in Live only: Studio's
        // centred windows already see past the attack, and compensating
        // there only added false starts (vocadito).
        let onset_phase = self.onset_phase(obs_feat.level_db) && self.studio.is_none();
        let mut job = Job {
            stage: Stage::Frames,
            newest,
            full_len: self.full.len(),
            eco_len: self.eco.hist.len(),
            hq_len: self.hq.hist.len(),
            hq: self.tier != QualityTier::Eco,
            obs_feat,
            obs_open,
            ..Job::default()
        };
        self.job_frame.clear();
        if obs_open {
            // The E/H sums advance with every pushed sample: read them now.
            job.vp_fast = self.cands_fast(onset_phase);
        }
        self.job = job;
    }

    /// Runs the next stage of the analysis begun by
    /// [`Detector::begin_analysis`] and returns the estimate after the last
    /// one ([`ANALYSIS_STEPS`] stages; `None` before, and when no analysis is
    /// in flight). Stages: verifier frames; octave measures and candidate
    /// probabilities; pitch HMM; refinement, voicing and output.
    pub fn step_analysis(&mut self) -> Option<PitchEstimate> {
        match self.job.stage {
            Stage::Idle => None,
            Stage::Frames => {
                if self.job.obs_open {
                    let mut job = self.job;
                    self.cands_frames(&mut job);
                    self.job = job;
                }
                self.job.stage = Stage::Checks;
                None
            }
            Stage::Checks => {
                if self.job.obs_open {
                    let mut job = self.job;
                    self.cands_checks(&mut job);
                    self.job = job;
                }
                self.job.stage = Stage::Track;
                None
            }
            Stage::Track => {
                self.track();
                self.job.stage = Stage::Output;
                None
            }
            Stage::Output => {
                self.job.stage = Stage::Idle;
                Some(self.output())
            }
        }
    }

    /// Runs every remaining stage of the analysis in flight and returns its
    /// estimate (with none in flight, analyses the newest sample as
    /// [`Detector::analyze`] does).
    pub fn finish_analysis(&mut self) -> PitchEstimate {
        if self.job.stage == Stage::Idle {
            self.begin_analysis();
        }
        loop {
            if let Some(est) = self.step_analysis() {
                return est;
            }
        }
    }

    /// Stages of the analysis in flight not yet run (0 when idle).
    pub fn analysis_steps_left(&self) -> usize {
        match self.job.stage {
            Stage::Idle => 0,
            Stage::Frames => 4,
            Stage::Checks => 3,
            Stage::Track => 2,
            Stage::Output => 1,
        }
    }

    /// Stage `Track`: pitch HMM forward step (Studio: fixed-lag smoothing)
    /// and the choice of candidate.
    fn track(&mut self) {
        let k = self.hmm.observation(self.job_frame.as_slice().iter().map(|c| (c.cents, c.p)), &mut self.obs);
        let fwd_vp = self.hmm.forward(&self.obs[..k], self.job.vp);
        let frames_len = self.frames.len();
        if let Some(slot) = self.frames.get_mut(self.frame_head) {
            *slot = self.job_frame;
        }
        let newest_frame = self.frame_head;
        self.frame_head = (self.frame_head + 1) % frames_len;
        let age = self.hmm.smooth();
        let post = self.hmm.posterior();
        let hmm_vp: f32 = if age == 0 { fwd_vp } else { post.iter().sum::<f32>().min(1.0) };
        let set = self
            .frames
            .get((newest_frame + frames_len - age.min(frames_len - 1)) % frames_len)
            .copied()
            .unwrap_or_default();
        let mut chosen: Option<usize> = None;
        let mut best_mass = 0.0f32;
        for (i, c) in set.as_slice().iter().enumerate() {
            let m = PitchHmm::mass_near(post, self.hmm.bin_of(c.cents), 2);
            if m > best_mass {
                best_mass = m;
                chosen = Some(i);
            }
        }
        if chosen.is_none() {
            chosen = set.best().filter(|&i| set.as_slice().get(i).is_some_and(|c| c.p > 0.0));
        }
        self.job_set = set;
        self.job.chosen = chosen;
        self.job.hmm_vp = hmm_vp;
    }

    /// Stage `Output`: features at the output instant, gate, refinement,
    /// voicing, mean Kalman and the published estimate.
    fn output(&mut self) -> PitchEstimate {
        let Job { newest, chosen, hmm_vp, obs_feat, .. } = self.job;
        let set = self.job_set;
        let d_obs = self.studio.map_or(0, |s| s.d_obs) as i64;

        // ---- 3. Output instant: features, gate, refinement ----------------------
        let out_back = self.studio.map_or(0, |s| s.delay) as i64;
        let feat = if out_back == d_obs { obs_feat } else { self.features_at(newest - out_back, newest) };
        let gate_open = self.gate.update(feat.level_db, self.gate_db);
        let sibilant = gate_open && feat.sibilant();
        let cand = chosen.and_then(|i| set.as_slice().get(i)).copied();
        let refined = cand.and_then(|c| self.refine(c.period, c.band, newest - out_back));
        let clarity = match (refined, cand) {
            (Some((r, _)), _) => r.clarity,
            (None, Some(c)) if c.g_long < 1.5 => 1.0 - c.g_long,
            _ => 0.0,
        }
        .clamp(0.0, 1.0);

        // ---- 4. Voicing -----------------------------------------------------------
        let theta = self.params.clarity;
        let mut raw = gate_open && !sibilant && hmm_vp >= 0.5 && clarity >= theta && refined.is_some();
        if raw && self.fsm.after_silence() && let Some(c) = cand {
            // Onset agreement (see ONSET_AGREE).
            let p_max = set.as_slice().iter().map(|x| x.p).fold(0.0f32, f32::max);
            if c.p < ONSET_AGREE * p_max {
                raw = false;
            }
            // Onset periodicity (see ONSET_MAX_G_LONG).
            if c.g_onset < UNKNOWN_G && c.g_onset > ONSET_MAX_G_LONG {
                raw = false;
            }
            // Onset inside the range (see ONSET_MAX_ABOVE_RANGE_CENTS).
            let z = refined.map_or(c.cents, |(r, _)| hz_to_cents_fast((self.fs as f64 / r.period) as f32));
            if z > self.range_top_cents + ONSET_MAX_ABOVE_RANGE_CENTS {
                raw = false;
            }
        }
        // Release tails (see TAIL_DB).
        let level = feat.level_db;
        let rise = level_rise(&mut self.out_levels, &mut self.out_level_head, level);
        self.tail_ref_db = (self.tail_ref_db - TAIL_DECAY_DB_PER_S * self.hop_s).max(-200.0);
        if rise {
            self.tail_ref_db = level;
        }
        let tail = level < self.tail_ref_db - TAIL_DB;
        raw &= !tail;
        // Onset consistency (see ONSET_CONSISTENT_CENTS) and the extra
        // confirmation of high onsets (ONSET_EXTRA_HOP_ABOVE_HZ).
        if raw {
            let z = refined.map(|(r, _)| hz_to_cents_fast((self.fs as f64 / r.period) as f32));
            let high = refined.is_some_and(|(r, _)| self.fs as f64 / r.period > ONSET_EXTRA_HOP_ABOVE_HZ as f64);
            self.fsm.set_onset_extra(u32::from(high));
            if self.fsm.confirming()
                && let (Some(a), Some(b)) = (z, self.onset_cents)
                && (a - b).abs() > ONSET_CONSISTENT_CENTS
            {
                self.fsm.restart_onset();
            }
            self.onset_cents = z;
        }
        // Holdable: still somewhat periodic, or as loud as the segment and no
        // new attack (see STEADY_HOLD_DB); never a sibilant, silence or a
        // release tail.
        let steady = !rise && level >= self.tail_ref_db - STEADY_HOLD_DB;
        let holdable = gate_open && !sibilant && !tail && (clarity >= 0.5 * theta || steady);
        let (voiced, onset) = self.fsm.update(raw, holdable);
        if voiced && level > self.tail_ref_db {
            self.tail_ref_db = level;
        }
        let q = (0.5 + (clarity - theta) / (2.0 * (1.0 - theta).max(1e-3))).clamp(0.0, 1.0);
        let voiced_prob =
            if gate_open { hmm_vp.min(q) * if sibilant { SIBILANT_ATTENUATION } else { 1.0 } } else { 0.0 };

        // ---- 5. Mean Kalman -------------------------------------------------------
        let dt = self.hop_s as f64;
        let mut info_lag = self.last_info_lag;
        let mut period_full = 0.0f64;
        if let Some((r, lag)) = refined {
            period_full = r.period;
            info_lag = lag;
        }
        if voiced {
            if raw && period_full > 0.0 {
                let z = hz_to_cents_fast((self.fs as f64 / period_full) as f32) as f64;
                if onset || !self.kalman.is_init() {
                    self.kalman.start(z, clarity);
                } else {
                    self.kalman.update(z, clarity, dt, self.studio.is_some());
                }
            } else {
                // Holding: coast on the last estimate with a decaying slope.
                self.kalman.coast(dt, 0.7);
            }
        } else {
            self.kalman.reset();
        }
        if self.studio.is_some() {
            info_lag = out_back as f32;
        }
        if voiced {
            self.last_info_lag = info_lag;
        }

        // ---- 6. Output ------------------------------------------------------------------
        self.publish_candidates(&set, chosen, period_full);
        let mut est = PitchEstimate {
            sample_pos: (newest - info_lag.round() as i64).max(0) as u64,
            info_lag,
            f0_hz: 0.0,
            cents: 0.0,
            period: 0.0,
            confidence: clarity,
            voiced_prob,
            voiced,
            sibilant,
            level_db: feat.level_db,
        };
        if voiced && self.kalman.is_init() {
            let p = self.kalman.pitch();
            let f0 = cents_to_hz_fast(p as f32);
            if f0.is_finite() && f0 > 0.0 {
                est.f0_hz = f0;
                est.period = self.fs / f0;
                est.cents = p as f32;
                if self.tier == QualityTier::Max && self.studio.is_none() {
                    // R§8.10 lag compensation: predict to the newest sample.
                    let ahead = (self.kalman.velocity() * info_lag as f64 / self.fs as f64)
                        .clamp(-LAG_COMP_CLIP_CENTS, LAG_COMP_CLIP_CENTS);
                    est.cents = (p + ahead) as f32;
                }
            } else {
                est.voiced = false;
            }
        }
        est
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// The candidate set of the newest frame (tests / debugging).
    #[cfg(test)]
    fn debug_frame(&self) -> CandSet {
        let n = self.frames.len();
        self.frames.get((self.frame_head + n - 1) % n).copied().unwrap_or_default()
    }

    /// Records this hop's observation level and reports whether the signal is
    /// in an onset phase (level ≥ [`ONSET_RISE_DB`] above the quietest level
    /// of the last [`ONSET_WINDOW_S`]).
    fn onset_phase(&mut self, level_db: f32) -> bool {
        level_rise(&mut self.levels, &mut self.level_head, level_db)
    }

    /// Re-lays out the HMM for the current tier (soft reset).
    fn relayout(&mut self) {
        let bin = match self.tier {
            QualityTier::Eco => 20.0,
            QualityTier::Hq => 10.0,
            QualityTier::Max => 5.0,
        };
        self.hmm.layout(self.hmm_lo, self.hmm_hi, bin, self.hop_s);
        self.hmm.set_switch(self.params.switch_ref, self.hop_s);
        for f in self.frames.iter_mut() {
            f.clear();
        }
    }

    /// Nominal info lag before any voiced frame: one period at the range's
    /// geometric centre (Causal) or the Studio delay.
    fn nominal_info_lag(&self) -> f32 {
        match self.studio {
            Some(s) => s.delay as f32,
            None => {
                let f = (self.cfg.voice_range.f_min() * self.cfg.voice_range.f_max()).sqrt();
                self.fs / f
            }
        }
    }

    /// Level / sibilance features of the 10 ms window centred on full-rate
    /// time `t` (Causal: `t` = newest, window right-aligned), with `newest`
    /// the newest sample of the hop being analysed.
    fn features_at(&self, t: i64, newest: i64) -> Features {
        let half = (self.level_len / 2) as i64;
        let end = if t >= newest { newest + 1 } else { (t + half + 1).min(newest + 1) };
        voicing::measure(&self.full, &self.hp, end, self.level_len, self.fs)
    }

    /// Per-sample power floor 20 dB under the closing gate.
    fn silence_floor(&self) -> f64 {
        let a = skato_core::math::db_to_lin(self.gate_db - GATE_HYST_DB - 20.0) as f64;
        a * a
    }

    /// Candidates, part 1 (at the hop boundary): E/H minima at the search
    /// rate. Returns the fast source's voiced probability.
    fn cands_fast(&mut self, onset_phase: bool) -> f32 {
        let fs = self.fs;
        let silence = self.silence_floor();
        let set = &mut self.job_frame;
        set.clear();
        let (rate_s, f_s) = (self.search.rate, self.search.factor as f32);
        self.eh.evaluate(&self.search.hist, silence, onset_phase);
        let mut eh_tr = [EhTrough::default(); MAX_EH_TROUGHS];
        let n_eh = self.eh.troughs(&mut eh_tr);
        for (h, t) in self.heights.iter_mut().zip(eh_tr.iter().take(n_eh)) {
            *h = t.height;
        }
        let vp_fast = trough_probabilities(&self.heights[..n_eh], &self.prior_fast, &self.rank, &mut self.probs);
        let band_fast = 0.75 * f_s + 3.0;
        for (t, &p) in eh_tr.iter().take(n_eh).zip(self.probs.iter()) {
            let f = rate_s / t.lag;
            set.add(Source::Fast, hz_to_cents_fast(f), fs / f, band_fast, p);
        }
        vp_fast
    }

    /// Candidates, part 2: the long type-I CMNDF frame and (HQ/Max) the
    /// short weighted frame of the observation instant.
    fn cands_frames(&mut self, job: &mut Job) {
        let fs = self.fs;
        let silence = self.silence_floor();
        let causal = self.studio.is_none();
        let hq = job.hq;
        let t_obs = job.newest - self.studio.map_or(0, |s| s.d_obs) as i64;
        let set = &mut self.job_frame;

        // --- long: type-I CMNDF on the verifier stream --------------------
        let (stream, frame, len) =
            if hq { (&self.hq, &mut self.long_hq, job.hq_len) } else { (&self.eco, &mut self.long_eco, job.eco_len) };
        let rate_v = stream.rate;
        let f_v = stream.factor as f32;
        let end = if causal {
            len
        } else {
            // Centre the frame (for its mid lag) on t_obs.
            let mid = 0.5 * (frame.window_len() as f64 + 0.5 * frame.tau_max() as f64) * stream.factor as f64;
            stream.end_at(t_obs as f64 + mid, len)
        };
        job.vp_long = 0.0;
        job.long_ok = frame.analyze(&stream.hist, end, hq, silence);
        if job.long_ok {
            let n_t = frame.troughs(&mut self.troughs);
            for (h, t) in self.heights.iter_mut().zip(self.troughs.iter().take(n_t)) {
                *h = t.height;
            }
            job.vp_long = trough_probabilities(&self.heights[..n_t], &self.prior_long, &self.rank, &mut self.probs);
            let band_v = (0.75 * f_v + 3.0).max(2.0);
            for (t, &p) in self.troughs.iter().take(n_t).zip(self.probs.iter()) {
                if p <= 1e-4 {
                    continue;
                }
                let f = rate_v / t.lag;
                set.add(Source::Long, hz_to_cents_fast(f), fs / f, band_v, p);
            }
        }

        // --- short: weighted type-I frame (HQ/Max) -------------------------
        job.vp_short = 0.0;
        if hq {
            let end = if causal {
                job.hq_len
            } else {
                let s = &self.short_hq;
                let mid = 0.5 * (s.window_len() as f64 + 0.5 * s.tau_max() as f64) * self.hq.factor as f64;
                self.hq.end_at(t_obs as f64 + mid, job.hq_len)
            };
            if self.short_hq.analyze(&self.hq.hist, end, silence) {
                let n_t = self.short_hq.troughs(&mut self.troughs);
                for (h, t) in self.heights.iter_mut().zip(self.troughs.iter().take(n_t)) {
                    *h = t.height;
                }
                job.vp_short =
                    trough_probabilities(&self.heights[..n_t], &self.prior_fast, &self.rank, &mut self.probs);
                let band_v = 0.75 * self.hq.factor as f32 + 3.0;
                for (t, &p) in self.troughs.iter().take(n_t).zip(self.probs.iter()) {
                    if p <= 1e-4 {
                        continue;
                    }
                    let f = self.hq.rate / t.lag;
                    set.add(Source::Short, hz_to_cents_fast(f), fs / f, band_v, p);
                }
            }
        }
    }

    /// Candidates, part 3: octave relatives, per-candidate measures for the
    /// octave checks, the combined probabilities and the octave and
    /// third-harmonic checks; sets `job.vp` (sibilance-attenuated).
    fn cands_checks(&mut self, job: &mut Job) {
        let fs = self.fs;
        let causal = self.studio.is_none();
        let hq = job.hq;
        let t_obs = job.newest - self.studio.map_or(0, |s| s.d_obs) as i64;
        let rate_s = self.search.rate;
        let rate_v = if hq { self.hq.rate } else { self.eco.rate };
        let long_ok = job.long_ok;
        let want_onset = !causal || self.fsm.after_silence();
        let set = &mut self.job_frame;

        // --- octave relatives of the two strongest candidates (R§8.3) -------
        let mut ranked = [(0usize, 0.0f32); 2];
        for (i, c) in set.as_slice().iter().enumerate() {
            let s = c.p_fast + c.p_long + c.p_short;
            if s > ranked[0].1 {
                ranked[1] = ranked[0];
                ranked[0] = (i, s);
            } else if s > ranked[1].1 {
                ranked[1] = (i, s);
            }
        }
        for &(i, s) in &ranked {
            if s <= 0.0 {
                continue;
            }
            let Some(c) = set.as_slice().get(i).copied() else { continue };
            for (dc, k) in [(1200.0f32, 0.5f32), (-1200.0, 2.0)] {
                let cents = c.cents + dc;
                if cents < self.hmm_lo || cents > self.hmm_hi || set.find(cents, 70.0).is_some() {
                    continue;
                }
                set.add(Source::Long, cents, c.period * k, c.band * k.max(1.0), 0.0);
            }
        }

        // --- per-candidate measures for the octave checks -------------------
        let frame: &LongFrame = if hq { &self.long_hq } else { &self.long_eco };
        // The onset periodicity (`want_onset`) is read only for a voicing
        // onset after silence; in Live the frame is used at once, so it is
        // only needed when the voicing FSM is after silence now (PERF-3).
        // Studio uses the frame hops later and keeps it for every frame.
        for c in set.as_mut_slice() {
            let f = fs / c.period;
            if long_ok && let Some((_, v)) = frame.nsdf_near(rate_v / f, 2) {
                c.g_long = 1.0 - v;
            }
            if want_onset && long_ok && let Some((_, v)) = frame.onset_nsdf_near(rate_v / f, 2) {
                c.g_onset = 1.0 - v;
            }
            if let Some(m) = self.eh.min_near(rate_s / f, 1) {
                c.h_fast = m.height;
            }
            if hq && let Some(h) = frame.harmonic_score(f / rate_v) {
                c.harm = h;
            }
        }
        // Full-rate periodicity of short-period candidates, most probable
        // first (bounded: ≤ MAX_FULL_CHECKS refinements of ≤ 400-sample
        // windows per hop).
        let max_p = (FULL_CHECK_S * fs).min(FULL_CHECK_MAX_SAMPLES);
        let end = if causal { job.full_len } else { (t_obs + 1).min(job.full_len) };
        let mut order = [0usize; MAX_CANDS];
        let m = set.as_slice().len();
        for (i, o) in order.iter_mut().enumerate().take(m) {
            *o = i;
        }
        let weight = |i: usize| set.as_slice().get(i).map_or(0.0, |c| c.p_fast + c.p_long + c.p_short);
        // In-place, allocation-free (≤ 12 elements).
        order[..m].sort_unstable_by(|&x, &y| weight(y).total_cmp(&weight(x)));
        let mut checks = 0;
        for &i in &order[..m] {
            let Some(c) = set.as_mut_slice().get_mut(i) else { continue };
            if checks >= MAX_FULL_CHECKS || c.period >= max_p {
                continue;
            }
            if let Some(g) = Self::full_rate_g(&mut self.refine_full, &self.full, fs, c, end, causal, false, job.full_len)
            {
                c.g_full = g;
            }
            checks += 1;
        }
        // Confirm would-be octave-up moves at the exact fractional lag
        // (bounded: ≤ MAX_EXACT_CONFIRMS candidates per hop).
        let mut suspects = [0usize; MAX_EXACT_CONFIRMS];
        let k = set.octave_up_suspects(&mut suspects);
        for &i in &suspects[..k] {
            let Some(c) = set.as_mut_slice().get_mut(i) else { continue };
            if let Some(g) = Self::full_rate_g(&mut self.refine_full, &self.full, fs, c, end, causal, true, job.full_len)
            {
                c.g_full = g;
            }
        }
        let mut vp = set.combine(job.vp_fast, job.vp_long, job.vp_short);
        set.octave_checks_with(!hq);
        set.third_harmonic_checks();
        if job.obs_feat.sibilant() {
            set.scale(SIBILANT_ATTENUATION);
            vp *= SIBILANT_ATTENUATION;
        }
        job.vp = vp;
    }

    /// `1 − clarity` of the full-rate pitch-synchronous measure at a
    /// candidate's period, for the octave comparisons; `exact` evaluates the
    /// peak at the fractional lag (`refine::Refiner`'s Max path) instead of
    /// the parabolic vertex. `full_len` is the full-rate history length at
    /// the hop boundary.
    #[allow(clippy::too_many_arguments)]
    fn full_rate_g(
        refiner: &mut Refiner,
        full: &Mirror,
        fs: f32,
        c: &cands::Cand,
        end: i64,
        causal: bool,
        exact: bool,
        full_len: i64,
    ) -> Option<f32> {
        // Studio: centre the pitch-synchronous span (W + P, centre (W + P)/2
        // before its end) on the observation instant.
        let min_w = (MIN_REFINE_WINDOW_S * fs).round() as usize;
        let half = (0.5 * ((c.period as f64).round().max(min_w as f64) + c.period as f64)).round() as i64;
        let e = if causal { end } else { (end + half).min(full_len) };
        let band = ((c.band as f64).max(0.012 * c.period as f64).ceil() as usize).max(2);
        // (Attack-compensated like the final refinement: with type-I here,
        // vocadito bad note starts rose 20 → 22, as during an attack type-I
        // penalises the longer of two octave lags more.)
        let r = refiner.refine(full, e, c.period as f64, band, exact, min_w)?;
        Some((1.0 - r.clarity).max(0.0))
    }

    /// Refines a candidate's period at the output instant (full-rate time
    /// `t_out`). Returns the refinement and its information lag (full-rate
    /// samples from the newest sample back to the evidence centre).
    fn refine(&mut self, seed_full: f32, band_full: f32, t_out: i64) -> Option<(Refined, f32)> {
        if !(seed_full.is_finite() && seed_full > 1.0) {
            return None;
        }
        let newest = self.job.newest;
        let f0 = self.fs / seed_full;
        let use_eco = self.tier == QualityTier::Eco && f0 < ECO_FULL_RATE_ABOVE_HZ;
        let sinc = self.tier == QualityTier::Max;
        // Band: the seed's uncertainty, but at least ±3 % (≈ ±50 c) so the
        // 1-period window follows vibrato, scoops and small steps by itself,
        // without waiting for the decimated candidate to move.
        let causal = self.studio.is_none();
        if use_eco {
            let s = &self.eco;
            let fac = s.factor as f64;
            let seed = seed_full as f64 / fac;
            let band = ((band_full as f64 / fac).max(REFINE_BAND_REL * seed).ceil() as usize).clamp(2, MAX_REFINE_BAND);
            let min_w = (MIN_REFINE_WINDOW_S * s.rate).round() as usize;
            let end = if causal {
                self.job.eco_len
            } else {
                // Studio: centre the W + P span on the output instant.
                let w = seed.round().max(min_w as f64);
                s.end_at(t_out as f64 + 0.5 * (w + seed) * fac, self.job.eco_len)
            };
            let r = self.refine_eco.refine(&s.hist, end, seed, band, sinc, min_w)?;
            // Evidence centre in full-rate time (stream units × factor, plus
            // the decimator's group delay via `time_of`).
            let centre_t = s.time_of((end - 1) as f64 - r.centre_back());
            let lag = (newest as f64 - centre_t) as f32;
            let r = Refined { period: r.period * fac, window: (r.window as f64 * fac).round() as usize, ..r };
            Some((r, lag.max(0.0)))
        } else {
            let seed = seed_full as f64;
            let band = ((band_full as f64).max(REFINE_BAND_REL * seed).ceil() as usize).clamp(2, MAX_REFINE_BAND);
            let min_w = (MIN_REFINE_WINDOW_S * self.fs).round() as usize;
            let end = if causal {
                self.job.full_len
            } else {
                // Studio: centre the W + P span on the output instant.
                let half = (0.5 * (seed.round().max(min_w as f64) + seed)).ceil() as i64;
                (t_out + half + 1).min(self.job.full_len)
            };
            let r = self.refine_full.refine(&self.full, end, seed, band, sinc, min_w)?;
            let lag = ((newest - (end - 1)) as f64 + r.centre_back()) as f32;
            Some((r, lag.max(0.0)))
        }
    }

    /// Fills the `candidates()` output from a frame's set.
    fn publish_candidates(&mut self, set: &CandSet, chosen: Option<usize>, refined_period: f64) {
        let mut items = [(0.0f32, -1.0f32); MAX_CANDS];
        for (slot, (i, c)) in items.iter_mut().zip(set.as_slice().iter().enumerate()) {
            let period = if Some(i) == chosen && refined_period > 0.0 { refined_period as f32 } else { c.period };
            *slot = (period, c.p);
        }
        let m = set.as_slice().len();
        // Insertion sort by score, descending (≤ 12 items).
        for i in 1..m {
            let mut j = i;
            while j > 0 && items[j].1 > items[j - 1].1 {
                items.swap(j, j - 1);
                j -= 1;
            }
        }
        self.n_out = m.min(OUT_CANDS);
        self.out[..self.n_out].copy_from_slice(&items[..self.n_out]);
    }
}

#[cfg(test)]
extern crate self as skato_detect;

#[cfg(test)]
mod tests;
