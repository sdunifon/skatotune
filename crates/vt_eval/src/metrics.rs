//! Objective metrics for rendered outputs (DESIGN §3.7, §4; R§10.3).
//!
//! All pitch metrics compare reference-tracker frames of the
//! latency-compensated output against the signal's ground truth. Because the
//! output is aligned by the reported latency D, an output frame at sample n
//! describes the same note as input sample n (PSOLA content wander of up to
//! one period is far inside the ±30 ms boundary exclusion).

use vt_core::{PitchFrame, frame_flags};

use crate::dsp::{
    BUTTER4_Q, Biquad64, Correlator64, KaiserSinc, decimate, filtfilt_padded, mean, next_pow2,
    percentile_sorted, rms,
};
use crate::reference::PitchTrack;
use crate::synth::{NoteKind, NoteSpan, Signal};

/// Summary of an error distribution in cents.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TuningStats {
    /// Frames that entered the statistics.
    pub frames: usize,
    /// Stable truth frames where the output had no pitch (counted separately).
    pub unvoiced: usize,
    /// Frames skipped because the *input* was too far from its note for the
    /// target to be unambiguous (deep scoops), see [`TuningOptions`].
    pub ambiguous: usize,
    /// Frames skipped because the *input* centre was still moving (scoop
    /// tails, legato glides, overshoot), see [`TuningOptions::stable_window_s`].
    pub moving: usize,
    pub median_abs: f64,
    pub p95_abs: f64,
    /// Signed mean (bias), cents.
    pub mean: f64,
    /// Percentage of frames within 5 / 10 / 25 c.
    pub within_5: f64,
    pub within_10: f64,
    pub within_25: f64,
}

impl TuningStats {
    pub fn from_errors(errors: &[f64], unvoiced: usize) -> Self {
        let mut abs: Vec<f64> = errors
            .iter()
            .filter(|e| e.is_finite())
            .map(|e| e.abs())
            .collect();
        if abs.is_empty() {
            return Self {
                unvoiced,
                median_abs: f64::NAN,
                p95_abs: f64::NAN,
                mean: f64::NAN,
                ..Self::default()
            };
        }
        abs.sort_by(f64::total_cmp);
        let n = abs.len() as f64;
        let within = |c: f64| 100.0 * abs.iter().filter(|e| **e <= c).count() as f64 / n;
        Self {
            frames: abs.len(),
            unvoiced,
            ambiguous: 0,
            moving: 0,
            median_abs: percentile_sorted(&abs, 50.0),
            p95_abs: percentile_sorted(&abs, 95.0),
            mean: mean(
                &errors
                    .iter()
                    .copied()
                    .filter(|e| e.is_finite())
                    .collect::<Vec<_>>(),
            ),
            within_5: within(5.0),
            within_10: within(10.0),
            within_25: within(25.0),
        }
    }
}

/// Options for [`tuning`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TuningOptions {
    /// Exclude this much around every note boundary, s (R§10.3: 30 ms).
    pub boundary_guard_s: f64,
    /// Also score sweeps against their per-sample nearest-semitone target
    /// (the "staircase" a hard-tuned glide should become), excluding the
    /// guard around every target change.
    pub include_sweeps: bool,
    /// Skip frames whose input centre pitch is farther than this from the
    /// intended note. R§10.3 scores *stable* frames; at the start of a deep
    /// scoop (the R§10.2 protocol uses −50…−150 c) the sung pitch is nearer
    /// the neighbouring semitone, so snapping there first is legitimate note
    /// decision, not a tuning error. Onset behaviour is scored by
    /// [`settling`] instead. 50 c = half a semitone (unambiguous for every
    /// scale).
    pub ambiguity_cents: f64,
    /// Score the output pitch averaged over this window, s (vibrato notes
    /// use one vibrato period instead). The reference tracker resolves single
    /// glottal cycles, so a voice's own period jitter (0.2 % ≈ 3.5 c per
    /// cycle) would otherwise dominate the error of a perfect corrector;
    /// perceived pitch integrates over tens of ms, and the R§10.3 reference
    /// trackers (pYIN, CREPE) use 64-93 ms frames. 20 ms is stricter than
    /// both while removing the cycle-to-cycle floor. 0 disables it.
    pub smoothing_s: f64,
    /// A frame is *stable* (R§10.3 "voiced stable frames") only if the
    /// input's vibrato-free centre stayed within [`Self::stable_cents`] of
    /// its current value over this many seconds before it; other frames are
    /// counted as `moving` and skipped. 0 (the default) disables it.
    ///
    /// The suite sets it to twice the retune time constant: a frame within
    /// 2·τ of an input movement is by design still converging (R§7.6.2 H),
    /// so it measures the retune speed rather than tuning accuracy. At
    /// retune 0 nothing is excluded, so note decisions inside transitions
    /// stay fully scored. Without it, an ideal R§7.6 corrector (zero
    /// detection lag, instant decisions, one-pole on the correction) at the
    /// default retune of 20 ms scores p95 = 19.3 c on `melody_male`: the
    /// scenarios' onset scoops (decaying with τ = 60 ms), 80 ms legato
    /// glides and +30 c overshoot bumps reach far past the 30 ms boundary
    /// guard, so the DESIGN §4 "retune 0-20 ms, p95 < 15 c" gate could not
    /// be met by a correct implementation. With 40 ms / 10 c the ideal
    /// corrector scores p95 = 4.0 c (melody_male) and 3.6 c (melody_female);
    /// slow drift (σ 10 c, τ 1 s moves ~3 c in 40 ms) stays in.
    pub stable_window_s: f64,
    /// See [`Self::stable_window_s`], cents.
    pub stable_cents: f64,
    /// Legal pitch classes (C = 0) of the scenario's scale. When set, the
    /// ambiguity limit on the input's side is half the gap to the
    /// neighbouring legal note ([`Self::ambiguity_cents`] per semitone of
    /// gap) minus [`Self::legal_tolerance_cents`] (the neighbour's capture
    /// region reaches that far past the midpoint, as in the engine's note
    /// HMM), but never less than [`Self::ambiguity_cents`]: next to a
    /// whole-tone neighbour a pitch 80 c off is still unambiguously its note
    /// (R§10.2's ±80 c detune needs this), while a semitone neighbour keeps
    /// the original limit. None (the default) keeps the fixed limit.
    pub legal: Option<[bool; 12]>,
    /// The note-decision tolerance used with [`Self::legal`], cents.
    pub legal_tolerance_cents: f64,
}

impl Default for TuningOptions {
    fn default() -> Self {
        Self {
            boundary_guard_s: 0.03,
            include_sweeps: false,
            ambiguity_cents: 50.0,
            smoothing_s: 0.02,
            stable_window_s: 0.0,
            stable_cents: 10.0,
            legal: None,
            legal_tolerance_cents: 0.0,
        }
    }
}

/// Whether the input at `pos` sits within `limit` cents of `target` (unknown
/// centre counts as unambiguous). With `legal`, the limit on the input's
/// side is `limit` per semitone of gap to the neighbouring legal note minus
/// `tol`, but at least `limit` (see [`TuningOptions::legal`]).
fn unambiguous(
    sig: &Signal,
    pos: usize,
    target: f64,
    limit: f64,
    legal: Option<(&[bool; 12], f64)>,
) -> bool {
    sig.centre_cents.get(pos).is_none_or(|c| {
        let c = f64::from(*c);
        let lim = match legal {
            Some((l, tol)) => {
                (limit * neighbour_semitones(target, c, Some(l)) - tol.max(0.0)).max(limit)
            }
            None => limit,
        };
        !c.is_finite() || (c - target).abs() <= lim
    })
}

/// Whether the input centre at `pos` stayed within `cents` of its current
/// value over the preceding `window` samples (unknown centre counts as
/// stable). Checked every 16 samples, which is far finer than any sung
/// pitch movement.
fn input_stable(sig: &Signal, pos: usize, window: usize, cents: f64) -> bool {
    let Some(now) = sig
        .centre_cents
        .get(pos)
        .map(|c| *c as f64)
        .filter(|c| c.is_finite())
    else {
        return true;
    };
    let lo = pos.saturating_sub(window);
    (lo..=pos).step_by(16).all(|q| {
        sig.centre_cents
            .get(q)
            .is_none_or(|c| !c.is_finite() || (*c as f64 - now).abs() <= cents)
    })
}

/// Mean of the piecewise-linear interpolant of `vals` over `width` frames
/// centred on frame `k` (trapezoidal rule); None if any covered frame is
/// missing (NaN) or out of range. A window exactly one vibrato period long
/// cancels a sinusoidal vibrato (residual O((2π/width)²/12)), leaving its centre.
fn boxcar(vals: &[f64], k: usize, width: f64) -> Option<f64> {
    let h = 0.5 * width.max(1.0);
    let (a, b) = (k as f64 - h, k as f64 + h);
    if a < 0.0 {
        return None;
    }
    let (ia, ib) = (a.floor() as usize, b.ceil() as usize);
    let window = vals.get(ia..=ib)?;
    if window.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let at = |x: f64| -> f64 {
        let i = x.floor() as usize;
        let f = x - i as f64;
        let v0 = vals[i];
        if f > 0.0 {
            v0 + (vals[i + 1] - v0) * f
        } else {
            v0
        }
    };
    let mut acc = 0.0;
    let mut x0 = a;
    while x0 < b {
        let x1 = (x0.floor() + 1.0).min(b);
        acc += 0.5 * (at(x0) + at(x1)) * (x1 - x0);
        x0 = x1;
    }
    Some(acc / (b - a))
}

/// Cents per frame (NaN when unvoiced).
fn cents_series(track: &PitchTrack) -> Vec<f64> {
    track
        .frames
        .iter()
        .map(|f| if f.voiced { f.cents } else { f64::NAN })
        .collect()
}

/// Frame index range strictly inside `[a, b)` samples.
fn frames_in(track: &PitchTrack, a: usize, b: usize) -> std::ops::Range<usize> {
    let hop = track.hop.max(1);
    let lo = a.div_ceil(hop);
    let hi = b.div_ceil(hop).min(track.frames.len());
    lo..hi.max(lo)
}

fn guard_samples(sig: &Signal, s: f64) -> usize {
    (s * sig.sample_rate as f64).round() as usize
}

/// Tuning error of the output against the intended notes on stable voiced
/// frames: ±guard around every note boundary excluded, sweeps skipped,
/// vibrato notes measured on their centre (one-period boxcar of the output).
pub fn tuning(sig: &Signal, out: &PitchTrack, opts: &TuningOptions) -> TuningStats {
    let guard = guard_samples(sig, opts.boundary_guard_s);
    let cents = cents_series(out);
    let hop_s = out.hop as f64 / out.sample_rate;
    let mut errors = Vec::new();
    let mut unvoiced = 0;
    let mut ambiguous = 0;
    let mut moving = 0;
    let stable_window = guard_samples(sig, opts.stable_window_s);
    for note in sig
        .notes
        .iter()
        .filter(|n| n.kind == NoteKind::Note && n.target_cents.is_finite())
    {
        for k in frames_in(out, note.start + guard, note.end.saturating_sub(guard)) {
            let pos = k * out.hop;
            if !sig.is_voiced(pos) {
                continue;
            }
            if !unambiguous(
                sig,
                pos,
                note.target_cents,
                opts.ambiguity_cents,
                opts.legal.as_ref().map(|l| (l, opts.legal_tolerance_cents)),
            ) {
                ambiguous += 1;
                continue;
            }
            if stable_window > 0 && !input_stable(sig, pos, stable_window, opts.stable_cents) {
                moving += 1;
                continue;
            }
            let Some(c) = cents.get(k).copied().filter(|c| c.is_finite()) else {
                unvoiced += 1;
                continue;
            };
            let smoothed = || {
                if opts.smoothing_s > 0.0 {
                    boxcar(&cents, k, opts.smoothing_s / hop_s).unwrap_or(c)
                } else {
                    c
                }
            };
            let value = match note
                .vibrato
                .filter(|v| v.rate_hz > 0.0 && v.extent_cents > 0.0)
            {
                Some(v) => {
                    // Before the vibrato starts: plain pitch. Once fully
                    // developed: the one-period centre. During the fade-in
                    // (and a possible fade-out before a slur) a one-period
                    // mean does not cancel a sinusoid of changing extent, so
                    // the centre is undefined there and the frame is skipped.
                    let period = 1.0 / v.rate_hz;
                    let t = (pos - note.start) as f64 / out.sample_rate;
                    let to_end = (note.end - pos) as f64 / out.sample_rate;
                    if t < v.delay_s - 0.5 * period {
                        smoothed()
                    } else if t >= v.delay_s + v.ramp_s + 0.5 * period
                        && to_end > 0.5 * period + VIBRATO_FADE_S
                    {
                        match boxcar(&cents, k, period / hop_s) {
                            Some(m) => m,
                            None => continue,
                        }
                    } else {
                        continue;
                    }
                }
                None => smoothed(),
            };
            errors.push(value - note.target_cents);
        }
    }
    if opts.include_sweeps {
        for note in sig.notes.iter().filter(|n| n.kind == NoteKind::Sweep) {
            // Positions where the per-sample target changes.
            let changes: Vec<usize> = (note.start + 1..note.end)
                .filter(|&i| sig.target_cents.get(i) != sig.target_cents.get(i - 1))
                .collect();
            for k in frames_in(out, note.start + guard, note.end.saturating_sub(guard)) {
                let pos = k * out.hop;
                let i = changes.partition_point(|c| *c < pos);
                let near = changes.get(i).is_some_and(|c| c - pos <= guard)
                    || (i > 0 && changes.get(i - 1).is_some_and(|c| pos - c <= guard));
                let target = sig.target_cents.get(pos).copied().unwrap_or(f32::NAN) as f64;
                if near || !target.is_finite() || !sig.is_voiced(pos) {
                    continue;
                }
                match cents.get(k).copied().filter(|c| c.is_finite()) {
                    Some(c) => errors.push(c - target),
                    None => unvoiced += 1,
                }
            }
        }
    }
    TuningStats {
        ambiguous,
        moving,
        ..TuningStats::from_errors(&errors, unvoiced)
    }
}

/// Longest vibrato fade-out the synthesiser applies before a slurred note, s.
const VIBRATO_FADE_S: f64 = 0.06;

/// Centred smoothing of the output pitch used by [`settling`], s.
pub const SETTLING_SMOOTHING_S: f64 = 0.01;

/// Where the time after one evaluated onset went (DESIGN §4 settling, split
/// as the EV-1 review asked). Every `*_s` time is measured from the
/// settling clock start ([`Self::clock_s`] after the note start); negative
/// values mean the event came before the note decision was due.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OnsetTiming {
    /// Note start, samples.
    pub note_start: usize,
    /// Settling clock start relative to the note start, s: when the sung
    /// (vibrato-free) pitch first passed the decision point (midpoint to the
    /// neighbouring legal note plus the note-decision tolerance), see
    /// [`SettlingOptions`]. 0 when the note is decidable from its first
    /// sample.
    pub clock_s: f64,
    /// When the engine's detected pitch (telemetry, input time) first passed
    /// the same decision point: detection latency. None without telemetry or
    /// if it never did.
    pub detect_s: Option<f64>,
    /// When the engine's target (telemetry, input time) first became this
    /// note: decision latency (includes detection). None without telemetry
    /// or if it never did.
    pub switch_s: Option<f64>,
    /// When the output had settled (the DESIGN §4 quantity); None if it
    /// never settled within the note (counted as a failure).
    pub settle_s: Option<f64>,
}

impl OnsetTiming {
    /// Post-switch convergence: settle − switch, s.
    pub fn convergence_s(&self) -> Option<f64> {
        Some(self.settle_s? - self.switch_s?)
    }

    /// Decision lag beyond detection: switch − detect, s.
    pub fn decision_s(&self) -> Option<f64> {
        Some(self.switch_s? - self.detect_s?)
    }
}

/// Settling after note onsets.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SettlingStats {
    /// Seconds from each evaluated onset until the output stays within
    /// tolerance for the hold time; None if it never did within the note.
    pub per_note: Vec<Option<f64>>,
    pub settled: usize,
    /// Onsets whose output never settled within the note. They count as
    /// failures: `median_s`/`p90_s`/`max_s` treat them as +∞ (so `max_s` is
    /// infinite whenever one onset never settles).
    pub unsettled: usize,
    pub median_s: f64,
    pub p90_s: f64,
    /// Worst onset, s (the DESIGN §4 gate applies to every onset).
    pub max_s: f64,
    /// Per-onset breakdown, same order as `per_note`.
    pub onsets: Vec<OnsetTiming>,
}

/// Options for [`settling_with`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SettlingOptions {
    /// Output tolerance around the target, cents (R§10.3: 10 c).
    pub tol_cents: f64,
    /// How long the output must stay inside the tolerance, s (R§10.3: 20 ms).
    pub hold_s: f64,
    /// Also evaluate legato onsets (step tests); otherwise only notes after a
    /// voicing gap.
    pub include_legato: bool,
    /// Legal pitch classes (C = 0) of the scenario's scale; None = chromatic.
    /// Sets the note-decision midpoints used for the clock start.
    pub legal: Option<[bool; 12]>,
    /// The engine's note-decision tolerance (`Settings::tolerance_cents`),
    /// cents. A corrector must stay on the neighbouring note until the sung
    /// pitch is this far past the midpoint (R§7.6.2 F; the HMM's capture
    /// region, vt_control `note_hmm`), so the settling clock starts only
    /// then. 0 with `legal: None` reproduces the original 50 c convention.
    pub decision_tolerance_cents: f64,
}

impl Default for SettlingOptions {
    fn default() -> Self {
        Self {
            tol_cents: 10.0,
            hold_s: 0.02,
            include_legato: false,
            legal: None,
            decision_tolerance_cents: 0.0,
        }
    }
}

/// Distance from `target` to the nearest other legal note on the side of
/// `pitch`, in semitones (1 when `legal` is None or holds no other note).
fn neighbour_semitones(target: f64, pitch: f64, legal: Option<&[bool; 12]>) -> f64 {
    let Some(legal) = legal else {
        return 1.0;
    };
    let base = (target / 100.0).round() as i64;
    let dir: i64 = if pitch >= target { 1 } else { -1 };
    (1..=12)
        .find(|s| legal[(base + dir * s).rem_euclid(12) as usize])
        .map_or(1.0, |s| s as f64)
}

/// Whether `pitch` is past the decision point towards `target`: within half
/// the gap to the neighbouring legal note minus the decision tolerance.
fn decision_due(target: f64, pitch: f64, opts: &SettlingOptions) -> bool {
    let half_gap = 50.0 * neighbour_semitones(target, pitch, opts.legal.as_ref());
    let limit = (half_gap - opts.decision_tolerance_cents.clamp(0.0, 45.0)).max(1.0);
    (pitch - target).abs() <= limit
}

/// Percentile of an ascending slice that may end in +∞ (unsettled onsets):
/// interpolation towards an infinite neighbour gives +∞ instead of NaN.
fn percentile_with_inf(sorted: &[f64], p: f64) -> f64 {
    if sorted.len() < 2 {
        return percentile_sorted(sorted, p);
    }
    let rank = (p.clamp(0.0, 100.0) / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = rank - lo as f64;
    if frac == 0.0 {
        sorted[lo]
    } else if sorted[hi].is_infinite() {
        f64::INFINITY
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Time until the output stays within `tol_cents` of the target for `hold_s`
/// (R§10.3: 10 c for 20 ms) after each onset. Evaluates notes that start after
/// a voicing gap (or every note when `include_legato`), without vibrato.
///
/// The clock starts at the note onset or, for a deep scoop, when the sung
/// pitch first comes within 50 c of the note (before that the target is
/// ambiguous, see [`TuningOptions::ambiguity_cents`]). Onsets that never
/// settle count as failures (+∞ in the percentiles). [`settling_with`] adds
/// scale-aware decision points and the per-onset latency breakdown.
pub fn settling(
    sig: &Signal,
    out: &PitchTrack,
    tol_cents: f64,
    hold_s: f64,
    include_legato: bool,
) -> SettlingStats {
    settling_with(
        sig,
        out,
        &SettlingOptions {
            tol_cents,
            hold_s,
            include_legato,
            ..SettlingOptions::default()
        },
        None,
    )
}

/// [`settling`] with explicit options and, optionally, the engine's telemetry
/// (`frames`, reported latency D) for the per-onset breakdown
/// ([`OnsetTiming`]: detection, decision, convergence).
///
/// The clock starts when the sung (vibrato-free) pitch first passes the
/// decision point: half the gap to the neighbouring legal note (50 c in a
/// chromatic scale, 100 c next to a whole-tone neighbour) minus
/// [`SettlingOptions::decision_tolerance_cents`]. Before that a correct
/// corrector still outputs the neighbouring note (that is what the tolerance
/// is for), so charging that time as settling would measure the singer's
/// scoop, not the engine (EV-1: on `melody_male` onset 0 it was 27 ms of
/// correct tolerance behaviour). Notes decidable from their first sample
/// start the clock at the onset, as before.
///
/// The output pitch is read through a centred 10 ms moving average: the raw
/// 1 ms track of a real voice (jitter, shimmer, breath) wanders by up to
/// ±10 c per cycle, which would trip a 10 c tolerance on a perfectly tuned
/// output, while a centred window hardly moves a monotone convergence.
pub fn settling_with(
    sig: &Signal,
    out: &PitchTrack,
    opts: &SettlingOptions,
    telemetry: Option<(&[PitchFrame], u32)>,
) -> SettlingStats {
    let raw = cents_series(out);
    let width = SETTLING_SMOOTHING_S * out.sample_rate / out.hop as f64;
    let cents: Vec<f64> = (0..raw.len())
        .map(|k| boxcar(&raw, k, width).unwrap_or(raw[k]))
        .collect();
    let hold = ((opts.hold_s * out.sample_rate) / out.hop as f64).ceil() as usize;
    let sr = sig.sample_rate as f64;
    // Telemetry in input time (sample_pos − D), voiced frames only.
    let tel: Vec<(usize, f64, f64)> = telemetry.map_or_else(Vec::new, |(frames, latency)| {
        frames
            .iter()
            .filter(|f| f.flags & frame_flags::VOICED != 0)
            .filter_map(|f| {
                let p = f.sample_pos.checked_sub(u64::from(latency))?;
                Some((p as usize, f64::from(f.detected), f64::from(f.target)))
            })
            .collect()
    });
    let mut per_note = Vec::new();
    let mut onsets = Vec::new();
    for note in sig.notes.iter().filter(|n| {
        n.kind == NoteKind::Note
            && n.vibrato.is_none()
            && n.target_cents.is_finite()
            && (opts.include_legato || !n.legato_in)
    }) {
        let target = note.target_cents;
        let Some(t_ref) = (note.start..note.end).find(|&p| {
            sig.centre_cents
                .get(p)
                .is_none_or(|c| !c.is_finite() || decision_due(target, f64::from(*c), opts))
        }) else {
            continue;
        };
        let since = |p: usize| (p as f64 - t_ref as f64) / sr;
        let range = frames_in(out, t_ref, note.end);
        let ok = |k: usize| {
            cents
                .get(k)
                .is_some_and(|c| (c - target).abs() <= opts.tol_cents)
        };
        let mut found = None;
        for k in range.clone() {
            if k + hold >= range.end {
                break;
            }
            if (k..=k + hold).all(ok) {
                found = Some((k * out.hop).saturating_sub(t_ref) as f64 / sr);
                break;
            }
        }
        let in_note = || {
            tel.iter()
                .filter(|(p, _, _)| *p >= note.start && *p < note.end)
        };
        let detect_s = in_note()
            .find(|(_, d, _)| d.is_finite() && *d > 0.0 && decision_due(target, *d, opts))
            .map(|(p, _, _)| since(*p));
        let switch_s = in_note()
            .find(|(_, _, t)| (t - target).abs() <= 1.0)
            .map(|(p, _, _)| since(*p));
        per_note.push(found);
        onsets.push(OnsetTiming {
            note_start: note.start,
            clock_s: (t_ref - note.start) as f64 / sr,
            detect_s,
            switch_s,
            settle_s: found,
        });
    }
    let mut t: Vec<f64> = per_note
        .iter()
        .map(|v| v.unwrap_or(f64::INFINITY))
        .collect();
    t.sort_by(f64::total_cmp);
    let settled = per_note.iter().flatten().count();
    SettlingStats {
        settled,
        unsettled: per_note.len() - settled,
        median_s: percentile_with_inf(&t, 50.0),
        p90_s: percentile_with_inf(&t, 90.0),
        max_s: t.last().copied().unwrap_or(f64::NAN),
        per_note,
        onsets,
    }
}

/// Vibrato measured on one note.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VibratoNote {
    pub start: usize,
    /// Half peak-to-peak extent (√2·RMS of the detrended contour), cents.
    pub in_extent: f64,
    pub out_extent: f64,
    pub in_rate_hz: f64,
    pub out_rate_hz: f64,
    /// Median |output centre − target|, cents.
    pub centre_error: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VibratoStats {
    /// Developed-vibrato measures (from vibrato start + fade-in + half a
    /// period; the DESIGN §4 preservation gate).
    pub notes: Vec<VibratoNote>,
    /// max |out_extent / in_extent − 1| over the notes.
    pub worst_extent_ratio_error: f64,
    /// Median |output centre − target| over all evaluated frames, cents.
    pub centre_error_median: f64,
    /// max |out_rate − in_rate|, Hz.
    pub worst_rate_error_hz: f64,
    /// How the output vibrato *starts* on every vibrato note (EV-3: the
    /// developed-vibrato window above starts ~430 ms into a delayed vibrato
    /// and never sees the causal tracker's engagement).
    pub engagement: Vec<VibratoEngagement>,
}

impl VibratoStats {
    /// Slowest engagement over the notes, in vibrato periods (+∞ if a note
    /// never reached [`ENGAGED_RATIO`]; NaN without vibrato notes).
    pub fn worst_engagement_periods(&self) -> f64 {
        self.engagement
            .iter()
            .map(|e| e.engagement_periods().unwrap_or(f64::INFINITY))
            .fold(f64::NAN, f64::max)
    }

    /// (min, max) early-window extent ratio over the notes (NaN if none).
    pub fn early_ratio_range(&self) -> (f64, f64) {
        let r = self.engagement.iter().filter_map(|e| e.early_ratio());
        r.clone()
            .fold((f64::NAN, f64::NAN), |(lo, hi), v| (lo.min(v), hi.max(v)))
    }
}

/// Output/input extent ratio at which the output vibrato counts as engaged.
pub const ENGAGED_RATIO: f64 = 0.8;

/// Vibrato onset behaviour of one note (R§8.11: a causal extrema tracker
/// engages after about one cycle, so Live at retune 0 flattens the first
/// vibrato cycle; "document it and measure it").
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VibratoEngagement {
    pub note_start: usize,
    /// Vibrato start (note start + delay), samples.
    pub vibrato_start: usize,
    pub rate_hz: f64,
    /// Input and output extent over the first two vibrato periods
    /// (√2·RMS about the window mean), cents; NaN if not measurable.
    pub in_early: f64,
    pub out_early: f64,
    /// Time from the vibrato start until the output extent reaches
    /// [`ENGAGED_RATIO`] of the input's (one-period window, reported at its
    /// centre) and stays there for another period, s. None if it never did.
    pub engagement_s: Option<f64>,
}

impl VibratoEngagement {
    /// Output/input extent over the first two periods.
    pub fn early_ratio(&self) -> Option<f64> {
        let r = self.out_early / self.in_early;
        r.is_finite().then_some(r)
    }

    /// [`Self::engagement_s`] in vibrato periods.
    pub fn engagement_periods(&self) -> Option<f64> {
        Some(self.engagement_s? * self.rate_hz)
    }
}

/// Extent (√2·RMS about the mean) of `cents` over frames `range`; None if
/// fewer than 80 % of them are voiced.
fn window_extent(cents: &[f64], range: std::ops::Range<usize>) -> Option<f64> {
    let n = range.len();
    let v: Vec<f64> = cents
        .get(range)?
        .iter()
        .copied()
        .filter(|c| c.is_finite())
        .collect();
    if n == 0 || 5 * v.len() < 4 * n {
        return None;
    }
    let m = mean(&v);
    Some(std::f64::consts::SQRT_2 * rms(&v.iter().map(|c| c - m).collect::<Vec<_>>()))
}

/// [`VibratoEngagement`] for every vibrato note: the early (first two
/// periods) extent ratio and the time until the output vibrato reaches
/// [`ENGAGED_RATIO`] of the input's. Windows are whole periods with only
/// the mean removed, so they need no data from before the vibrato start
/// (a centred detrend would need half a period before it).
pub fn vibrato_engagement(
    sig: &Signal,
    in_track: &PitchTrack,
    out_track: &PitchTrack,
    guard_s: f64,
) -> Vec<VibratoEngagement> {
    let sr = sig.sample_rate as f64;
    let cin = cents_series(in_track);
    let cout = cents_series(out_track);
    let step = (0.002 * sr) as usize;
    let mut out = Vec::new();
    for note in sig.notes.iter().filter(|n| n.kind == NoteKind::Note) {
        let Some(v) = note
            .vibrato
            .filter(|v| v.rate_hz > 0.0 && v.extent_cents > 0.0)
        else {
            continue;
        };
        let period = (sr / v.rate_hz).round() as usize;
        let vs = note.start + (v.delay_s * sr).round() as usize;
        let end = note.end.saturating_sub(guard_samples(sig, guard_s));
        if vs + 2 * period > end {
            continue;
        }
        let early = frames_in(out_track, vs, vs + 2 * period);
        let in_early = window_extent(&cin, early.clone()).unwrap_or(f64::NAN);
        let out_early = window_extent(&cout, early).unwrap_or(f64::NAN);
        // Ratio of one-period windows starting every `step` samples.
        let ratios: Vec<(usize, Option<f64>)> = (vs..end.saturating_sub(period))
            .step_by(step.max(1))
            .map(|a| {
                let w = frames_in(out_track, a, a + period);
                let r = match (window_extent(&cin, w.clone()), window_extent(&cout, w)) {
                    (Some(i), Some(o)) if i > 1.0 => Some(o / i),
                    _ => None,
                };
                (a, r)
            })
            .collect();
        let per_period = period.div_ceil(step.max(1));
        let engaged = (0..ratios.len()).find(|&i| {
            let stay = ratios.get(i..(i + per_period + 1).min(ratios.len()));
            stay.is_some_and(|w| w.iter().all(|(_, r)| r.is_some_and(|r| r >= ENGAGED_RATIO)))
                && i + per_period < ratios.len()
        });
        out.push(VibratoEngagement {
            note_start: note.start,
            vibrato_start: vs,
            rate_hz: v.rate_hz,
            in_early,
            out_early,
            engagement_s: engaged.map(|i| (ratios[i].0 + period / 2 - vs) as f64 / sr),
        });
    }
    out
}

/// Extent and rate of the detrended contour (cents minus one-period boxcar).
fn vibrato_shape(
    cents: &[f64],
    range: std::ops::Range<usize>,
    period_frames: f64,
    hop_s: f64,
) -> Option<(f64, f64, Vec<f64>)> {
    let mut dev = Vec::new();
    let mut times = Vec::new();
    let mut centres = Vec::new();
    for k in range {
        let (Some(c), Some(m)) = (
            cents.get(k).copied().filter(|c| c.is_finite()),
            boxcar(cents, k, period_frames),
        ) else {
            continue;
        };
        dev.push(c - m);
        times.push(k as f64 * hop_s);
        centres.push(m);
    }
    if dev.len() < 8 {
        return None;
    }
    let extent = std::f64::consts::SQRT_2 * rms(&dev);
    // Upward zero crossings, linearly interpolated, with hysteresis: a
    // crossing only counts after the contour has been below −h, so per-cycle
    // tracker noise (a few cents) near zero cannot add spurious crossings.
    let h = (0.25 * extent).max(3.0);
    let mut ups = Vec::new();
    let mut armed = false;
    for i in 1..dev.len() {
        if dev[i] < -h {
            armed = true;
        }
        if armed && dev[i - 1] < 0.0 && dev[i] >= 0.0 && (times[i] - times[i - 1]) < 2.5 * hop_s {
            let f = dev[i - 1] / (dev[i - 1] - dev[i]);
            ups.push(times[i - 1] + f * (times[i] - times[i - 1]));
            armed = false;
        }
    }
    let rate = match (ups.first(), ups.last()) {
        (Some(a), Some(b)) if ups.len() >= 2 && b > a => (ups.len() - 1) as f64 / (b - a),
        _ => f64::NAN,
    };
    Some((extent, rate, centres))
}

/// Vibrato preservation: output extent and rate vs the input's (both measured
/// by the same tracker), and the output centre's tuning, on every vibrato note.
pub fn vibrato(
    sig: &Signal,
    in_track: &PitchTrack,
    out_track: &PitchTrack,
    guard_s: f64,
) -> VibratoStats {
    let hop_s = out_track.hop as f64 / out_track.sample_rate;
    let cin = cents_series(in_track);
    let cout = cents_series(out_track);
    let mut notes = Vec::new();
    let mut all_centre_err = Vec::new();
    for note in sig.notes.iter().filter(|n| n.kind == NoteKind::Note) {
        let Some(v) = note
            .vibrato
            .filter(|v| v.rate_hz > 0.0 && v.extent_cents > 0.0)
        else {
            continue;
        };
        let sr = sig.sample_rate as f64;
        let period = 1.0 / v.rate_hz;
        let a = note.start + ((v.delay_s + v.ramp_s + 0.5 * period) * sr) as usize;
        let b = note
            .end
            .saturating_sub(((guard_s + 0.5 * period) * sr) as usize);
        if b <= a {
            continue;
        }
        let range = frames_in(out_track, a, b);
        let pf = period / hop_s;
        let (Some((ie, ir, _)), Some((oe, or, oc))) = (
            vibrato_shape(&cin, range.clone(), pf, hop_s),
            vibrato_shape(&cout, range, pf, hop_s),
        ) else {
            continue;
        };
        let mut errs: Vec<f64> = oc.iter().map(|c| (c - note.target_cents).abs()).collect();
        all_centre_err.extend(errs.iter().copied());
        errs.sort_by(f64::total_cmp);
        notes.push(VibratoNote {
            start: note.start,
            in_extent: ie,
            out_extent: oe,
            in_rate_hz: ir,
            out_rate_hz: or,
            centre_error: percentile_sorted(&errs, 50.0),
        });
    }
    all_centre_err.sort_by(f64::total_cmp);
    VibratoStats {
        engagement: vibrato_engagement(sig, in_track, out_track, guard_s),
        worst_extent_ratio_error: notes
            .iter()
            .map(|n| (n.out_extent / n.in_extent - 1.0).abs())
            .fold(f64::NAN, f64::max),
        worst_rate_error_hz: notes
            .iter()
            .map(|n| (n.out_rate_hz - n.in_rate_hz).abs())
            .fold(f64::NAN, f64::max),
        centre_error_median: percentile_sorted(&all_centre_err, 50.0),
        notes,
    }
}

/// High-frequency pitch fluctuation on steady notes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JitterStats {
    pub frames: usize,
    /// RMS of the high-passed output deviation from the target, cents.
    pub out_rms: f64,
    /// Same for the input (the singer's own jitter + tracker floor).
    pub in_rms: f64,
    /// √max(0, out² − in²): what the engine added.
    pub added_rms: f64,
}

/// Added jitter (R§10.3): RMS of the zero-phase high-passed (> `cutoff_hz`)
/// output pitch deviation on steady notes (no vibrato, ±60 ms from the
/// boundaries), against the same measure on the input.
pub fn jitter(
    sig: &Signal,
    in_track: &PitchTrack,
    out_track: &PitchTrack,
    cutoff_hz: f64,
) -> JitterStats {
    let frame_rate = out_track.sample_rate / out_track.hop as f64;
    let guard = guard_samples(sig, 0.06);
    let trim = (0.03 * frame_rate).ceil() as usize;
    let min_run = (0.1 * frame_rate).ceil() as usize;
    let hp = [Biquad64::highpass(
        frame_rate,
        cutoff_hz,
        std::f64::consts::FRAC_1_SQRT_2,
    )];
    let cin = cents_series(in_track);
    let cout = cents_series(out_track);
    let (mut so, mut si, mut n) = (0.0, 0.0, 0usize);
    for note in sig
        .notes
        .iter()
        .filter(|n| n.kind == NoteKind::Note && n.vibrato.is_none() && n.target_cents.is_finite())
    {
        let range = frames_in(
            out_track,
            note.start + guard,
            note.end.saturating_sub(guard),
        );
        // Split into runs where both tracks are voiced.
        let mut k = range.start;
        while k < range.end {
            let both = |i: usize| {
                cin.get(i).is_some_and(|c| c.is_finite())
                    && cout.get(i).is_some_and(|c| c.is_finite())
            };
            if !both(k) {
                k += 1;
                continue;
            }
            let s = k;
            while k < range.end && both(k) {
                k += 1;
            }
            if k - s < min_run {
                continue;
            }
            let eo: Vec<f64> = cout[s..k].iter().map(|c| c - note.target_cents).collect();
            let ei: Vec<f64> = cin[s..k].iter().map(|c| c - note.target_cents).collect();
            let pad = (3.0 * frame_rate / cutoff_hz).ceil() as usize;
            let (fo, fi) = (
                filtfilt_padded(&hp, &eo, pad),
                filtfilt_padded(&hp, &ei, pad),
            );
            for (a, b) in fo
                .iter()
                .zip(fi.iter())
                .skip(trim)
                .take((k - s).saturating_sub(2 * trim))
            {
                so += a * a;
                si += b * b;
                n += 1;
            }
        }
    }
    if n == 0 {
        return JitterStats {
            frames: 0,
            out_rms: f64::NAN,
            in_rms: f64::NAN,
            added_rms: f64::NAN,
        };
    }
    let (o, i) = ((so / n as f64).sqrt(), (si / n as f64).sqrt());
    JitterStats {
        frames: n,
        out_rms: o,
        in_rms: i,
        added_rms: (o * o - i * i).max(0.0).sqrt(),
    }
}

/// Binary voicing agreement (positive class = voiced).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VoicingStats {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub tn: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl VoicingStats {
    fn from_counts(tp: usize, fp: usize, fn_: usize, tn: usize) -> Self {
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            f64::NAN
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            f64::NAN
        };
        let f1 = if tp > 0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        Self {
            tp,
            fp,
            fn_,
            tn,
            precision,
            recall,
            f1,
        }
    }
}

/// Voicing F1 of the engine's telemetry against the truth. Telemetry
/// timestamps are output time, so input time = `sample_pos − latency`; frames
/// within `guard_s` of a truth voicing transition are skipped.
pub fn voicing_f1(
    sig: &Signal,
    telemetry: &[PitchFrame],
    latency: u32,
    guard_s: f64,
) -> VoicingStats {
    let guard = guard_samples(sig, guard_s);
    let transitions = sig.voicing_transitions();
    let near = |p: usize| {
        let i = transitions.partition_point(|t| *t < p);
        let after = transitions.get(i).is_some_and(|t| t - p <= guard);
        let before = i > 0 && transitions.get(i - 1).is_some_and(|t| p - t <= guard);
        after || before
    };
    let (mut tp, mut fp, mut fn_, mut tn) = (0, 0, 0, 0);
    for f in telemetry {
        let Some(p) = f.sample_pos.checked_sub(latency as u64).map(|p| p as usize) else {
            continue;
        };
        if p >= sig.len() || near(p) {
            continue;
        }
        let truth = sig.is_voiced(p);
        let pred = f.flags & frame_flags::VOICED != 0;
        match (truth, pred) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    VoicingStats::from_counts(tp, fp, fn_, tn)
}

/// Voicing F1 of a reference track against the truth (same conventions).
pub fn track_voicing_f1(sig: &Signal, track: &PitchTrack, guard_s: f64) -> VoicingStats {
    let frames: Vec<PitchFrame> = track
        .frames
        .iter()
        .map(|f| PitchFrame {
            sample_pos: f.pos as u64,
            flags: if f.voiced { frame_flags::VOICED } else { 0 },
            ..PitchFrame::default()
        })
        .collect();
    voicing_f1(sig, &frames, 0, guard_s)
}

/// Result of a cross-correlation delay measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatencyMeasurement {
    /// Integer lag with the highest normalised correlation, samples.
    pub lag: usize,
    /// Parabolically refined lag, samples.
    pub lag_fine: f64,
    /// Normalised correlation at the peak (1 = exact delayed copy).
    pub peak: f64,
}

/// Delay of `output` relative to `input` (R§10.6): normalised cross-correlation
/// over lags 0..=`max_lag`. Use aperiodic material (noise bursts) so the peak
/// is unique.
pub fn measure_latency(
    input: &[f32],
    output: &[f32],
    max_lag: usize,
) -> Option<LatencyMeasurement> {
    let a: Vec<f64> = input
        .iter()
        .map(|v| if v.is_finite() { *v as f64 } else { 0.0 })
        .collect();
    let b: Vec<f64> = output
        .iter()
        .map(|v| if v.is_finite() { *v as f64 } else { 0.0 })
        .collect();
    let e_a: f64 = a.iter().map(|v| v * v).sum();
    if e_a <= 0.0 || b.is_empty() {
        return None;
    }
    let n = next_pow2(a.len() + b.len() + max_lag + 2);
    let mut corr = Correlator64::new(n);
    let mut r = vec![0.0; max_lag + 1];
    corr.cross_correlate(&a, &b, &mut r);
    let mut prefix = vec![0.0; b.len() + 1];
    for (i, v) in b.iter().enumerate() {
        prefix[i + 1] = prefix[i] + v * v;
    }
    let norm: Vec<f64> = r
        .iter()
        .enumerate()
        .map(|(l, rv)| {
            let lo = l.min(b.len());
            let hi = (l + a.len()).min(b.len());
            let e_b = prefix[hi] - prefix[lo];
            if e_b > 0.0 {
                rv / (e_a * e_b).sqrt()
            } else {
                0.0
            }
        })
        .collect();
    let (lag, peak) = norm
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.total_cmp(y.1))
        .map(|(i, v)| (i, *v))?;
    let lag_fine = match (
        lag.checked_sub(1).and_then(|i| norm.get(i)),
        norm.get(lag + 1),
    ) {
        (Some(&ym1), Some(&yp1)) => {
            let den = ym1 - 2.0 * peak + yp1;
            if den.abs() > 1e-15 {
                lag as f64 + (0.5 * (ym1 - yp1) / den).clamp(-0.5, 0.5)
            } else {
                lag as f64
            }
        }
        _ => lag as f64,
    };
    Some(LatencyMeasurement {
        lag,
        lag_fine,
        peak,
    })
}

/// High-frequency click/buzz detector (R§10.3: energy above 8 kHz at grain rate).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClickStats {
    pub frames: usize,
    /// Frames whose >8 kHz energy exceeds the input's by more than 10 dB.
    pub clicks: usize,
    /// Largest excess over the input's local >8 kHz energy, dB.
    pub max_excess_db: f64,
}

/// Compare >8 kHz energy of the aligned output with the input's, frame by
/// frame (2 ms frames, 1 ms hop). The input reference is the maximum over
/// ±`wander_s`, so PSOLA's content wander is not mistaken for clicks.
pub fn clicks(
    input: &[f32],
    output_aligned: &[f32],
    sample_rate: f32,
    wander_s: f64,
) -> ClickStats {
    click_scan(input, output_aligned, sample_rate, wander_s, |_, _| {})
}

/// [`clicks`] plus where they are: the centre time (s) and excess (dB) of
/// every frame counted as a click, in order.
pub fn click_frames(
    input: &[f32],
    output_aligned: &[f32],
    sample_rate: f32,
    wander_s: f64,
) -> (ClickStats, Vec<(f64, f64)>) {
    let sr = sample_rate as f64;
    let hop = (0.001 * sr).max(1.0).floor();
    let centre = 0.001 * sr; // half of the 2 ms frame, samples
    let mut at = Vec::new();
    let stats = click_scan(input, output_aligned, sample_rate, wander_s, |k, excess| {
        at.push(((k as f64 * hop + centre) / sr, excess));
    });
    (stats, at)
}

/// Shared body of [`clicks`] / [`click_frames`]: `on_click(frame, excess_db)`
/// is called for every frame counted as a click.
fn click_scan(
    input: &[f32],
    output_aligned: &[f32],
    sample_rate: f32,
    wander_s: f64,
    mut on_click: impl FnMut(usize, f64),
) -> ClickStats {
    let sr = sample_rate as f64;
    let hf = |x: &[f32]| -> Vec<f64> {
        let mut f: Vec<Biquad64> = BUTTER4_Q
            .iter()
            .map(|q| Biquad64::highpass(sr, 8000.0, *q))
            .collect();
        let hp: Vec<f64> = x
            .iter()
            .map(|v| {
                f.iter_mut()
                    .fold(if v.is_finite() { *v as f64 } else { 0.0 }, |a, s| {
                        s.process(a)
                    })
            })
            .collect();
        let len = (0.002 * sr) as usize;
        let hop = (0.001 * sr).max(1.0) as usize;
        (0..hp.len().saturating_sub(len) / hop)
            .map(|k| {
                let seg = &hp[k * hop..k * hop + len];
                10.0 * (seg.iter().map(|v| v * v).sum::<f64>() / len as f64)
                    .max(1e-20)
                    .log10()
            })
            .collect()
    };
    let ein = hf(input);
    let eout = hf(output_aligned);
    let reach = (wander_s * 1000.0).ceil() as usize;
    let mut stats = ClickStats {
        max_excess_db: f64::NEG_INFINITY,
        ..ClickStats::default()
    };
    for (k, eo) in eout.iter().enumerate() {
        let lo = k.saturating_sub(reach);
        let hi = (k + reach + 1).min(ein.len());
        let reference = ein
            .get(lo..hi)
            .map_or(-200.0, |s| s.iter().cloned().fold(-200.0, f64::max));
        let excess = eo - reference.max(-90.0);
        stats.frames += 1;
        stats.max_excess_db = stats.max_excess_db.max(excess);
        if excess > 10.0 && *eo > -70.0 {
            stats.clicks += 1;
            on_click(k, excess);
        }
    }
    stats
}

/// LPC-envelope log-spectral distance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LsdStats {
    pub frames: usize,
    pub mean_db: f64,
    pub median_db: f64,
}

/// Autocorrelation-method LPC by Levinson-Durbin; returns (a[0..=p], error).
fn lpc(frame: &[f64], order: usize) -> Option<(Vec<f64>, f64)> {
    let n = frame.len();
    let r: Vec<f64> = (0..=order)
        .map(|k| {
            frame
                .iter()
                .zip(frame.iter().skip(k))
                .map(|(a, b)| a * b)
                .sum::<f64>()
        })
        .collect();
    let r0 = *r.first()?;
    if r0 <= 1e-20 || n <= order {
        return None;
    }
    // Light lag window (white-noise correction) for numerical robustness.
    let mut r = r;
    r[0] *= 1.0 + 1e-9;
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;
    let mut err = r[0];
    for i in 1..=order {
        let acc: f64 = (1..i).map(|j| a[j] * r[i - j]).sum::<f64>() + r[i];
        let k = -acc / err;
        let prev = a.clone();
        for j in 1..i {
            a[j] = prev[j] + k * prev[i - j];
        }
        a[i] = k;
        err *= 1.0 - k * k;
        if err <= 0.0 {
            return None;
        }
    }
    Some((a, err))
}

/// LPC envelope power (dB) of `a`/`err` on frequencies `freqs` (Hz).
fn lpc_env_db(a: &[f64], err: f64, freqs: &[f64], sr: f64) -> Vec<f64> {
    freqs
        .iter()
        .map(|f| {
            let w = 2.0 * std::f64::consts::PI * f / sr;
            let (mut re, mut im) = (0.0, 0.0);
            for (k, c) in a.iter().enumerate() {
                re += c * (w * k as f64).cos();
                im -= c * (w * k as f64).sin();
            }
            10.0 * (err / (re * re + im * im).max(1e-30)).log10()
        })
        .collect()
}

/// Formant-envelope log-spectral distance between input and aligned output
/// (DESIGN §3.4: < 3 dB for PSOLA). Both are decimated to ~12 kHz, then on
/// 40 ms Hann frames (20 ms hop) fully inside notes (±30 ms excluded) an
/// order-14 LPC envelope is compared on 64 log-spaced points over 100 Hz-5 kHz,
/// gain-normalised (only the shape counts).
pub fn formant_lsd(sig: &Signal, output_aligned: &[f32]) -> LsdStats {
    let sr = sig.sample_rate as f64;
    let factor = ((sr / 12_000.0).round() as usize).max(1);
    let dsr = sr / factor as f64;
    let to64 = |x: &[f32]| -> Vec<f64> {
        x.iter()
            .map(|v| if v.is_finite() { *v as f64 } else { 0.0 })
            .collect()
    };
    let xi = decimate(&to64(&sig.audio), factor);
    let xo = decimate(&to64(output_aligned), factor);
    let flen = (0.04 * dsr) as usize;
    let fhop = (0.02 * dsr) as usize;
    let window: Vec<f64> = (0..flen)
        .map(|i| {
            (std::f64::consts::PI * (i as f64 + 0.5) / flen as f64)
                .sin()
                .powi(2)
        })
        .collect();
    let freqs: Vec<f64> = (0..64)
        .map(|i| 100.0 * (50.0f64).powf(i as f64 / 63.0))
        .collect();
    let guard = guard_samples(sig, 0.03);
    let prep = |x: &[f64], s: usize| -> Vec<f64> {
        (0..flen)
            .map(|i| {
                let cur = x.get(s + i).copied().unwrap_or(0.0);
                let prev = if s + i > 0 {
                    x.get(s + i - 1).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                (cur - 0.97 * prev) * window[i]
            })
            .collect()
    };
    let mut lsd = Vec::new();
    let mut s = 0usize;
    while s + flen <= xi.len().min(xo.len()) {
        let (a, b) = (s * factor, (s + flen) * factor);
        let inside = sig
            .notes
            .iter()
            .any(|n: &NoteSpan| a >= n.start + guard && b + guard <= n.end);
        if inside
            && let (Some((ai, ei)), Some((ao, eo))) =
                (lpc(&prep(&xi, s), 14), lpc(&prep(&xo, s), 14))
        {
            let di = lpc_env_db(&ai, ei, &freqs, dsr);
            let dout = lpc_env_db(&ao, eo, &freqs, dsr);
            let diff: Vec<f64> = di.iter().zip(dout.iter()).map(|(p, q)| p - q).collect();
            let m = mean(&diff);
            lsd.push(rms(&diff.iter().map(|d| d - m).collect::<Vec<_>>()));
        }
        s += fhop;
    }
    lsd.sort_by(f64::total_cmp);
    LsdStats {
        frames: lsd.len(),
        mean_db: mean(&lsd),
        median_db: percentile_sorted(&lsd, 50.0),
    }
}

/// Segmental SNR of `test` against `reference` after aligning each segment by
/// the best fractional delay in [`min_lag`, `max_lag`] samples (PSOLA content
/// wander, DESIGN §3.4). Segments of `seg_len` samples with reference energy
/// below −60 dBFS are skipped; per-segment SNR is clamped to [−20, 100] dB.
/// Returns (mean SNR in dB, segments used).
pub fn aligned_segmental_snr(
    reference: &[f32],
    test: &[f32],
    seg_len: usize,
    min_lag: f64,
    max_lag: f64,
) -> (f64, usize) {
    let r: Vec<f64> = reference.iter().map(|v| *v as f64).collect();
    let t: Vec<f64> = test.iter().map(|v| *v as f64).collect();
    let sinc = KaiserSinc::new(16, 8.6);
    let mut snrs = Vec::new();
    let seg_len = seg_len.max(16);
    let mut s = 0;
    while s + seg_len <= t.len().min(r.len()) {
        let e_ref: f64 = r[s..s + seg_len].iter().map(|v| v * v).sum::<f64>() / seg_len as f64;
        if e_ref < 1e-6 {
            s += seg_len;
            continue;
        }
        let a = SegmentAlignment::find(&r, &t[s..s + seg_len], s, (min_lag, max_lag), 0.0, &sinc);
        snrs.push(a.snr_db.clamp(-20.0, 100.0));
        s += seg_len;
    }
    (mean(&snrs), snrs.len())
}

/// Best fractional alignment of one test segment against a reference.
struct SegmentAlignment {
    /// `test[i] ≈ reference[start + i − lag]`, samples.
    lag: f64,
    /// Normalised correlation at `lag`.
    corr: f64,
    /// SNR of the test segment against the shifted reference, dB.
    snr_db: f64,
}

impl SegmentAlignment {
    /// Integer scan of plain shifts over `lags` (min, max), then a
    /// golden-section refinement of the sinc-interpolated score around the
    /// best integer lag. Lags scoring within `tie` of the best count as
    /// equally good and the one nearest zero wins: on strongly periodic
    /// audio a lag one period away correlates almost as well as the true
    /// one, and the nearer choice never invents a multi-period wander.
    fn find(
        r: &[f64],
        seg: &[f64],
        start: usize,
        lags: (f64, f64),
        tie: f64,
        sinc: &KaiserSinc,
    ) -> Self {
        let (min_lag, max_lag) = lags;
        let n = seg.len();
        let shifted = |lag: f64| -> Vec<f64> {
            (0..n)
                .map(|i| sinc.interp(r, (start + i) as f64 - lag))
                .collect()
        };
        let score = |lag: f64| -> f64 {
            let y = shifted(lag);
            let num: f64 = y.iter().zip(seg).map(|(a, b)| a * b).sum();
            let den = (y.iter().map(|a| a * a).sum::<f64>()
                * seg.iter().map(|b| b * b).sum::<f64>())
            .sqrt();
            if den > 0.0 { num / den } else { 0.0 }
        };
        let lo = min_lag.floor() as i64;
        let hi = max_lag.ceil() as i64;
        let scores: Vec<(i64, f64)> = (lo..=hi)
            .map(|l| {
                let (mut num, mut ey, mut et) = (0.0, 0.0, 0.0);
                for (i, b) in seg.iter().enumerate() {
                    let idx = (start + i) as i64 - l;
                    let a = if idx < 0 {
                        0.0
                    } else {
                        r.get(idx as usize).copied().unwrap_or(0.0)
                    };
                    num += a * b;
                    ey += a * a;
                    et += b * b;
                }
                let v = if ey * et > 0.0 {
                    num / (ey * et).sqrt()
                } else {
                    0.0
                };
                (l, v)
            })
            .collect();
        let best_v = scores.iter().fold(f64::NEG_INFINITY, |m, s| m.max(s.1));
        let best_l = scores
            .iter()
            .filter(|s| s.1 >= best_v - tie)
            .min_by_key(|s| s.0.abs())
            .map_or(0.0, |s| s.0 as f64);
        let best_at = scores
            .iter()
            .find(|s| s.0 as f64 == best_l)
            .map_or(0.0, |s| s.1);
        if best_at >= 1.0 - 1e-12 {
            // An exact integer-delay copy (Classic engines, unvoiced
            // pass-through): nothing to refine.
            let shift = |i: usize| {
                let idx = (start + i) as i64 - best_l as i64;
                if idx < 0 {
                    0.0
                } else {
                    r.get(idx as usize).copied().unwrap_or(0.0)
                }
            };
            let sig_e: f64 = (0..n).map(|i| shift(i).powi(2)).sum();
            let noise: f64 = (0..n).map(|i| (shift(i) - seg[i]).powi(2)).sum();
            return Self {
                lag: best_l,
                corr: best_at,
                snr_db: if noise > 0.0 {
                    10.0 * (sig_e / noise).log10()
                } else {
                    100.0
                },
            };
        }
        let g = (5f64.sqrt() - 1.0) / 2.0;
        let (mut a, mut b) = (best_l - 1.0, best_l + 1.0);
        for _ in 0..30 {
            let c = b - g * (b - a);
            let d = a + g * (b - a);
            if score(c) > score(d) {
                b = d;
            } else {
                a = c;
            }
        }
        let lag = 0.5 * (a + b);
        let y = shifted(lag);
        let sig_e: f64 = y.iter().map(|v| v * v).sum();
        let noise: f64 = y.iter().zip(seg).map(|(p, q)| (p - q) * (p - q)).sum();
        let snr_db = if noise > 0.0 {
            10.0 * (sig_e / noise).log10()
        } else {
            100.0
        };
        Self {
            lag,
            corr: score(lag),
            snr_db,
        }
    }
}

/// Content delay of voiced audio (R§10.3 / R§10.6 "content-wander
/// distribution"): how late the output carries each piece of sung input.
///
/// The reported latency D is exact for unvoiced audio (measured by
/// [`measure_latency`] on noise bursts), but PSOLA builds voiced output from
/// whole input periods, so its content arrives `D + wander` late, with the
/// wander bounded by about one period (R§2.1, R§10.3 target "wander ≤ T").
/// This is what a performer monitoring through the plugin actually hears on
/// sustained notes.
#[derive(Clone, Default, PartialEq)]
pub struct ContentDelayStats {
    /// Reported latency D, samples.
    pub reported: u32,
    pub sample_rate: f64,
    /// Voiced segments measured (output a clean copy of the input: peak
    /// correlation ≥ [`CONTENT_MIN_CORR`]).
    pub segments: usize,
    /// Voiced segments rejected (no clean copy found).
    pub rejected: usize,
    /// Wander = content delay − D per segment, samples (positive = later
    /// than D).
    pub wander: Vec<f64>,
    /// Wander divided by the segment's sung period (|·| ≤ 1 is the R§10.3
    /// target).
    pub wander_periods: Vec<f64>,
}

/// Peak correlation a voiced segment needs to count as a delayed copy.
pub const CONTENT_MIN_CORR: f64 = 0.9;

/// Correlation difference within which two candidate content delays count
/// as equally good (the one nearer D wins).
const CONTENT_TIE: f64 = 1e-3;

impl std::fmt::Debug for ContentDelayStats {
    /// A summary: the per-segment vectors run to hundreds of values.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentDelayStats")
            .field("reported_ms", &self.reported_ms())
            .field("segments", &self.segments)
            .field("rejected", &self.rejected)
            .field("median_ms", &self.delay_ms(50.0))
            .field("p95_ms", &self.delay_ms(95.0))
            .field("max_abs_wander_periods", &self.max_abs_wander_periods())
            .finish()
    }
}

impl ContentDelayStats {
    fn ms(&self, samples: f64) -> f64 {
        samples / self.sample_rate * 1e3
    }

    /// Reported latency D, ms.
    pub fn reported_ms(&self) -> f64 {
        self.ms(f64::from(self.reported))
    }

    /// Percentile `p` of the content delay D + wander, ms (NaN if nothing
    /// was measured).
    pub fn delay_ms(&self, p: f64) -> f64 {
        self.ms(f64::from(self.reported) + crate::dsp::percentile(&self.wander, p))
    }

    /// Mean content delay D + wander, ms.
    pub fn mean_delay_ms(&self) -> f64 {
        self.ms(f64::from(self.reported) + mean(&self.wander))
    }

    /// Largest |wander|, ms.
    pub fn max_abs_wander_ms(&self) -> f64 {
        self.ms(self.wander.iter().fold(f64::NAN, |m, w| m.max(w.abs())))
    }

    /// Largest |wander| in sung periods.
    pub fn max_abs_wander_periods(&self) -> f64 {
        self.wander_periods
            .iter()
            .fold(f64::NAN, |m, w| m.max(w.abs()))
    }
}

/// Measure [`ContentDelayStats`] on voiced 20 ms segments ≥ 30 ms inside the
/// notes of `sig`, comparing the *raw* (uncompensated) output with the input
/// over delays `D ± max_wander_s` (fractional, sinc-refined). Meaningful when
/// the output is a delayed copy of the input, i.e. on an amount-0 render;
/// with a pitch correction applied the segments stop correlating and are
/// counted as rejected.
pub fn content_delay(
    sig: &Signal,
    output_raw: &[f32],
    latency: u32,
    max_wander_s: f64,
) -> ContentDelayStats {
    let sr = sig.sample_rate as f64;
    let r: Vec<f64> = sig.audio.iter().map(|v| f64::from(*v)).collect();
    let t: Vec<f64> = output_raw
        .iter()
        .map(|v| if v.is_finite() { f64::from(*v) } else { 0.0 })
        .collect();
    let sinc = KaiserSinc::new(16, 8.6);
    let seg_len = (0.02 * sr).round() as usize;
    let guard = guard_samples(sig, 0.03);
    let d = f64::from(latency);
    let w = (max_wander_s * sr).ceil();
    let mut stats = ContentDelayStats {
        reported: latency,
        sample_rate: sr,
        ..ContentDelayStats::default()
    };
    for note in sig.notes.iter().filter(|n| n.kind == NoteKind::Note) {
        let mut s = note.start + guard;
        while s + seg_len + guard <= note.end {
            let voiced = (s..s + seg_len).all(|i| sig.is_voiced(i));
            let e_in = r[s..s + seg_len].iter().map(|v| v * v).sum::<f64>() / seg_len as f64;
            let out_start = s + latency as usize;
            if !voiced || e_in < 1e-6 || out_start + seg_len + w as usize > t.len() {
                s += seg_len;
                continue;
            }
            // Test segment = output at input time s (+ D), aligned against
            // the input: test[i] ≈ input[s + i − wander].
            let seg: Vec<f64> = t[out_start..out_start + seg_len].to_vec();
            let a = SegmentAlignment::find(&r, &seg, s, (-w.min(d), w), CONTENT_TIE, &sinc);
            if a.corr >= CONTENT_MIN_CORR {
                let f0: Vec<f64> = sig.f0_truth[s..s + seg_len]
                    .iter()
                    .map(|f| f64::from(*f))
                    .collect();
                let period = sr / crate::dsp::median(&f0).max(1.0);
                stats.segments += 1;
                stats.wander.push(a.lag);
                stats.wander_periods.push(a.lag / period);
            } else {
                stats.rejected += 1;
            }
            s += seg_len;
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::RefFrame;
    use crate::synth::{Vibrato, cents_to_hz64};

    /// Signal with two notes (truth only) and a synthetic track built from a
    /// closure, so metrics can be checked against known answers.
    fn fixture(f: impl Fn(f64) -> Option<f64>) -> (Signal, PitchTrack) {
        let sr = 48_000.0f32;
        let n = 96_000;
        let mut sig = Signal::from_audio(vec![0.0; n], sr);
        let notes = [(4800usize, 43_200usize, 5700.0), (48_000, 91_200, 5900.0)];
        for &(a, b, c) in &notes {
            for i in a..b {
                sig.f0_truth[i] = cents_to_hz64(c) as f32;
                sig.target_cents[i] = c as f32;
            }
            sig.notes.push(NoteSpan {
                start: a,
                end: b,
                target_cents: c,
                detune_cents: 0.0,
                vibrato: None,
                legato_in: false,
                kind: NoteKind::Note,
            });
        }
        let hop = 48;
        let frames = (0..n / hop)
            .map(|k| {
                let pos = k * hop;
                let c = f(pos as f64 / sr as f64).filter(|_| sig.is_voiced(pos));
                RefFrame {
                    pos,
                    voiced: c.is_some(),
                    f0_hz: c.map_or(0.0, cents_to_hz64),
                    cents: c.unwrap_or(f64::NAN),
                    cents_raw: c.unwrap_or(f64::NAN),
                    clarity: 1.0,
                    level_dbfs: -20.0,
                }
            })
            .collect();
        (
            sig,
            PitchTrack {
                sample_rate: sr as f64,
                hop,
                frames,
            },
        )
    }

    fn target_at(t: f64) -> f64 {
        if t < 0.95 { 5700.0 } else { 5900.0 }
    }

    #[test]
    fn tuning_constant_offset() {
        let (sig, track) = fixture(|t| Some(target_at(t) + 4.0));
        let s = tuning(&sig, &track, &TuningOptions::default());
        assert!((s.median_abs - 4.0).abs() < 1e-9 && (s.mean - 4.0).abs() < 1e-9);
        assert_eq!(s.within_5, 100.0);
        assert_eq!(s.within_10, 100.0);
        // Boundary guard: notes of 0.8 s and 0.9 s, minus 2 × 30 ms each, at 1 ms frames.
        assert!(
            (s.frames as f64 - 1580.0).abs() <= 4.0,
            "frames {}",
            s.frames
        );
    }

    #[test]
    fn deep_scoop_frames_are_ambiguous_not_errors() {
        let (mut sig, track) = fixture(|t| Some(target_at(t)));
        // First note sung 80 c flat for its first 200 ms.
        for i in 4800..14_400 {
            sig.centre_cents[i] = 5620.0;
        }
        for i in 14_400..43_200 {
            sig.centre_cents[i] = 5700.0;
        }
        let s = tuning(&sig, &track, &TuningOptions::default());
        assert!(
            (s.ambiguous as i64 - 170).abs() <= 2,
            "ambiguous {}",
            s.ambiguous
        );
        let st = settling(&sig, &track, 10.0, 0.02, false);
        // Output is on target throughout: settles immediately from the
        // unambiguous point (and at the onset for the second note).
        assert!(
            st.per_note.iter().all(|t| t.is_some_and(|t| t < 0.002)),
            "{st:?}"
        );
    }

    #[test]
    fn ambiguity_follows_the_scale_when_given() {
        // The first note (A, 5700) is sung 80 c flat for 200 ms; the output
        // is on A throughout. In C major A's lower neighbour is G (200 c):
        // 80 c flat is still unambiguously A and must be scored, while the
        // chromatic default treats it as ambiguous (G# would be nearer).
        let (mut sig, track) = fixture(|t| Some(target_at(t)));
        for i in 4800..14_400 {
            sig.centre_cents[i] = 5620.0;
        }
        for i in 14_400..43_200 {
            sig.centre_cents[i] = 5700.0;
        }
        let chromatic = tuning(&sig, &track, &TuningOptions::default());
        let c_major = vt_core::ScaleKind::Major
            .note_states(0)
            .map(|s| s == vt_core::NoteState::On);
        let in_key = tuning(
            &sig,
            &track,
            &TuningOptions {
                legal: Some(c_major),
                legal_tolerance_cents: 8.0,
                ..TuningOptions::default()
            },
        );
        assert!(chromatic.ambiguous > 160, "{chromatic:?}");
        assert_eq!(in_key.ambiguous, 0, "{in_key:?}");
        assert_eq!(in_key.frames, chromatic.frames + chromatic.ambiguous);
    }

    #[test]
    fn moving_input_frames_are_excluded_when_asked() {
        // The first note glides 100 c up into its target over its first
        // 60 ms; the output follows 20 ms late, like a retune lag.
        let glide = |t: f64, lag: f64| {
            if t < 0.95 {
                5600.0 + 100.0 * ((t - 0.1 - lag) / 0.06).clamp(0.0, 1.0)
            } else {
                5900.0
            }
        };
        let (mut sig, track) = fixture(move |t| Some(glide(t, 0.02)));
        for i in 4800..43_200 {
            sig.centre_cents[i] = glide(i as f64 / 48_000.0, 0.0) as f32;
        }
        for i in 48_000..91_200 {
            sig.centre_cents[i] = 5900.0;
        }
        let all = tuning(&sig, &track, &TuningOptions::default());
        assert_eq!(all.moving, 0);
        assert!(
            all.within_10 < 100.0,
            "the lag shows without the window: {all:?}"
        );
        let stable = tuning(
            &sig,
            &track,
            &TuningOptions {
                stable_window_s: 0.04,
                ..TuningOptions::default()
            },
        );
        // From the guard end (0.130 s) until the input 40 ms earlier was
        // within 10 c of its final value (0.194 s) the frames are "moving";
        // everything else is on target.
        assert!(
            (stable.moving as i64 - 64).abs() <= 2,
            "moving {}",
            stable.moving
        );
        assert_eq!(stable.within_5, 100.0, "{stable:?}");
        assert_eq!(stable.frames + stable.moving, all.frames);
    }

    #[test]
    fn tuning_measures_vibrato_on_centre() {
        let (mut sig, track) = fixture(|t| {
            Some(target_at(t) + 3.0 + 60.0 * (2.0 * std::f64::consts::PI * 5.5 * t).sin())
        });
        for n in &mut sig.notes {
            n.vibrato = Some(Vibrato::immediate(5.5, 60.0));
        }
        let s = tuning(&sig, &track, &TuningOptions::default());
        assert!((s.median_abs - 3.0).abs() < 0.2, "median {}", s.median_abs);
        assert!(s.p95_abs < 3.5, "p95 {}", s.p95_abs);
    }

    #[test]
    fn settling_finds_decay_time() {
        // Output starts 100 c off and decays with τ = 10 ms: within 10 c after
        // τ·ln(10) ≈ 23 ms.
        let (sig, track) = fixture(|t| {
            let (start, tgt) = if t < 0.95 {
                (0.1, 5700.0)
            } else {
                (1.0, 5900.0)
            };
            Some(tgt - 100.0 * (-(t - start) / 0.01).exp())
        });
        let s = settling(&sig, &track, 10.0, 0.02, false);
        assert_eq!(s.settled, 2);
        assert!((s.median_s - 0.023).abs() < 0.002, "median {}", s.median_s);
    }

    #[test]
    fn unsettled_onsets_count_as_failures() {
        // EV-1: the second note's output stays 30 c sharp, so it never
        // settles. It must not vanish from the statistics.
        let (sig, track) = fixture(|t| {
            if t < 0.95 {
                Some(5700.0 - 100.0 * (-(t - 0.1) / 0.01).exp())
            } else {
                Some(5930.0)
            }
        });
        let s = settling(&sig, &track, 10.0, 0.02, false);
        assert_eq!((s.settled, s.unsettled), (1, 1), "{s:?}");
        assert!(s.per_note[0].is_some() && s.per_note[1].is_none());
        assert!(s.max_s.is_infinite(), "worst onset {}", s.max_s);
        assert!(s.median_s.is_infinite() && s.p90_s.is_infinite(), "{s:?}");
        assert_eq!(s.onsets.len(), 2);
        assert_eq!(s.onsets[1].settle_s, None);
    }

    /// Centre pitch of the first fixture note rising from `from` cents
    /// (relative to its 5700 c target) at 1 c/ms from 0.1 s, then held.
    fn scooped(sig: &mut Signal, from: f64) {
        for i in 4800..43_200 {
            let t = i as f64 / 48_000.0;
            sig.centre_cents[i] = (5700.0 + (from + 1000.0 * (t - 0.1)).min(0.0)) as f32;
        }
        for i in 48_000..91_200 {
            sig.centre_cents[i] = 5900.0;
        }
    }

    #[test]
    fn settling_clock_starts_at_the_note_decision_point() {
        // EV-1: a −100 c scoop. A correct corrector (tolerance 8 c) stays on
        // the lower neighbour until the sung pitch is 8 c past the midpoint
        // (5658 c, 0.158 s), then snaps. That is not settling time.
        let centre = |t: f64| 5600.0 + (1000.0 * (t - 0.1)).clamp(0.0, 100.0);
        let (mut sig, track) = fixture(move |t| {
            Some(if t >= 0.95 {
                5900.0
            } else if centre(t) < 5658.0 {
                5600.0
            } else {
                5700.0
            })
        });
        scooped(&mut sig, -100.0);
        let opts = SettlingOptions {
            decision_tolerance_cents: 8.0,
            ..SettlingOptions::default()
        };
        let new = settling_with(&sig, &track, &opts, None);
        assert!((new.onsets[0].clock_s - 0.058).abs() < 0.001, "{new:?}");
        // Only the 10 ms centred output smoothing remains (~4 ms).
        let first = new.per_note[0].unwrap();
        assert!(first < 0.006, "settle {first}");
        // The original 50 c convention charges the 8 ms tolerance band too.
        let old = settling(&sig, &track, 10.0, 0.02, false);
        let first_old = old.per_note[0].unwrap();
        assert!((first_old - first - 0.008).abs() < 0.0015, "{old:?}");
    }

    #[test]
    fn decision_point_follows_the_scale() {
        // In C major the note below A (5700) is G, 200 c away: the midpoint
        // is 100 c, so a −150 c scoop is decidable 92 c below A (tolerance
        // 8 c), 50 ms before the chromatic 42 c point.
        let (mut sig, track) = fixture(|t| Some(target_at(t)));
        scooped(&mut sig, -150.0);
        let c_major = vt_core::ScaleKind::Major.note_states(0);
        let opts = |legal: Option<[bool; 12]>| SettlingOptions {
            legal,
            decision_tolerance_cents: 8.0,
            ..SettlingOptions::default()
        };
        let in_key = settling_with(
            &sig,
            &track,
            &opts(Some(c_major.map(|s| s == vt_core::NoteState::On))),
            None,
        );
        let chromatic = settling_with(&sig, &track, &opts(None), None);
        assert!(
            (in_key.onsets[0].clock_s - 0.058).abs() < 0.001,
            "{in_key:?}"
        );
        assert!(
            (chromatic.onsets[0].clock_s - 0.108).abs() < 0.001,
            "{chromatic:?}"
        );
        // A note decidable from its first sample starts the clock there.
        assert_eq!(in_key.onsets[1].clock_s, 0.0);
    }

    #[test]
    fn onset_breakdown_reads_engine_telemetry() {
        // Telemetry (output time = input time + D): the detected pitch
        // follows the sung centre, the target switches to A at 0.170 s.
        let (mut sig, track) = fixture(|t| Some(target_at(t)));
        scooped(&mut sig, -100.0);
        let latency = 100u32;
        let frames: Vec<PitchFrame> = (0..2000u64)
            .filter_map(|k| {
                let pos = (k * 48) as usize;
                let c = sig
                    .centre_cents
                    .get(pos)
                    .copied()
                    .filter(|c| c.is_finite())?;
                let target = if pos >= 43_200 {
                    5900.0
                } else if pos < 8160 {
                    5600.0
                } else {
                    5700.0
                };
                Some(PitchFrame {
                    sample_pos: pos as u64 + u64::from(latency),
                    detected: c,
                    target,
                    flags: frame_flags::VOICED,
                    ..PitchFrame::default()
                })
            })
            .collect();
        let opts = SettlingOptions {
            decision_tolerance_cents: 8.0,
            ..SettlingOptions::default()
        };
        let s = settling_with(&sig, &track, &opts, Some((&frames, latency)));
        let o = s.onsets[0];
        assert!(o.detect_s.is_some_and(|d| d.abs() <= 0.001), "{o:?}");
        assert!(
            o.switch_s.is_some_and(|w| (w - 0.012).abs() <= 0.001),
            "{o:?}"
        );
        assert!(o.decision_s().is_some_and(|d| (d - 0.012).abs() <= 0.002));
        assert!(o.convergence_s().is_some());
    }

    #[test]
    fn vibrato_extent_ratio() {
        let make = |ext: f64| {
            let (mut sig, track) = fixture(move |t| {
                Some(target_at(t) + ext * (2.0 * std::f64::consts::PI * 6.0 * t).sin())
            });
            for n in &mut sig.notes {
                n.vibrato = Some(Vibrato::immediate(6.0, 50.0));
            }
            (sig, track)
        };
        let (sig, tin) = make(50.0);
        let (_, tout) = make(45.0);
        let v = vibrato(&sig, &tin, &tout, 0.03);
        assert_eq!(v.notes.len(), 2);
        assert!(
            (v.notes[0].in_extent - 50.0).abs() < 0.5,
            "{:?}",
            v.notes[0]
        );
        assert!((v.worst_extent_ratio_error - 0.1).abs() < 0.01);
        assert!((v.notes[0].in_rate_hz - 6.0).abs() < 0.05);
        assert!(v.centre_error_median < 0.5);
    }

    #[test]
    fn vibrato_engagement_sees_a_flattened_first_cycle() {
        // EV-3: vibrato from the note onset; the output vibrato is absent for
        // 1.25 periods (a causal tracker waiting for three extrema), then
        // ramps in over half a period (R§8.11).
        let rate = 6.0;
        let p = 1.0 / rate;
        let start_of = |t: f64| if t < 0.95 { 0.1 } else { 1.0 };
        let gain = move |t: f64| ((t - start_of(t) - 1.25 * p) / (0.5 * p)).clamp(0.0, 1.0);
        let vib =
            move |t: f64| 50.0 * (2.0 * std::f64::consts::PI * rate * (t - start_of(t))).sin();
        let make = |g: &dyn Fn(f64) -> f64| {
            let (mut sig, track) = fixture(|t| Some(target_at(t) + g(t) * vib(t)));
            for n in &mut sig.notes {
                n.vibrato = Some(Vibrato::immediate(rate, 50.0));
            }
            (sig, track)
        };
        let (sig, tin) = make(&|_| 1.0);
        let (_, tout) = make(&gain);
        let v = vibrato(&sig, &tin, &tout, 0.03);
        assert_eq!(v.engagement.len(), 2);
        for e in &v.engagement {
            assert_eq!(e.vibrato_start, e.note_start);
            assert!((e.in_early - 50.0).abs() < 2.0, "{e:?}");
            // First two periods: silent for 1.25, ramping for 0.5, full for
            // 0.25 of them, so well under half the input's extent.
            assert!(e.early_ratio().is_some_and(|r| r < 0.55), "{e:?}");
            // 80 % of the extent in a one-period window, at its centre.
            let k = e.engagement_periods().expect("engages");
            assert!((1.5..2.1).contains(&k), "engaged after {k} periods");
        }
        assert!(v.worst_engagement_periods() < 2.1);
        // An output that never engages is reported as never (+∞).
        let (_, flat) = make(&|_| 0.0);
        let never = vibrato(&sig, &tin, &flat, 0.03);
        assert!(never.engagement.iter().all(|e| e.engagement_s.is_none()));
        assert!(never.worst_engagement_periods().is_infinite());
    }

    #[test]
    fn content_delay_of_a_wandering_copy() {
        // Output = input delayed by D on the first note and by D + 7.5
        // samples (fractional) on the second: the PSOLA content wander.
        let sr = 48_000.0f32;
        let sig = crate::synth::SignalBuilder::new(sr)
            .silence(0.05)
            .melody(
                crate::synth::Melody::new(crate::synth::Voice::male())
                    .note(crate::synth::Note::midi(50.0, 0.5).detune(0.0))
                    .note(crate::synth::Note::midi(55.0, 0.5).detune(0.0).gap(0.1)),
            )
            .silence(0.05)
            .build();
        let d = 213usize;
        let split = sig.notes[1].start - 2000;
        let x: Vec<f64> = sig.audio.iter().map(|v| f64::from(*v)).collect();
        let s = KaiserSinc::new(16, 8.6);
        let mut out = vec![0.0f32; x.len() + d + 64];
        for (n, o) in out.iter_mut().enumerate() {
            let src = n as f64 - d as f64 - if n >= split + d { 7.5 } else { 0.0 };
            *o = s.interp(&x, src) as f32;
        }
        let c = content_delay(&sig, &out, d as u32, 0.0125);
        assert!(c.segments > 30 && c.rejected == 0, "{c:?}");
        let first: Vec<f64> = c.wander.iter().copied().filter(|w| w.abs() < 1.0).collect();
        let second: Vec<f64> = c
            .wander
            .iter()
            .copied()
            .filter(|w| w.abs() >= 1.0)
            .collect();
        assert!(first.iter().all(|w| w.abs() < 0.05), "{first:?}");
        assert!(second.iter().all(|w| (w - 7.5).abs() < 0.05), "{second:?}");
        assert!((c.reported_ms() - 213.0 / 48.0).abs() < 1e-9);
        assert!((c.max_abs_wander_ms() - 7.5 / 48.0).abs() < 0.002);
        // 7.5 samples at G3 (196 Hz, 245 samples per period).
        assert!((c.max_abs_wander_periods() - 7.5 / 244.9).abs() < 0.002);
    }

    #[test]
    fn vibrato_rate_is_robust_to_cycle_noise() {
        // ±3 c alternating noise on a 6.2 Hz vibrato would double-count
        // zero crossings without hysteresis.
        let noise = |t: f64| {
            if ((t * 1000.0).round() as i64) % 2 == 0 {
                3.0
            } else {
                -3.0
            }
        };
        let (mut sig, track) = fixture(move |t| {
            Some(target_at(t) + 55.0 * (2.0 * std::f64::consts::PI * 6.2 * t).sin() + noise(t))
        });
        for n in &mut sig.notes {
            n.vibrato = Some(Vibrato::immediate(6.2, 55.0));
        }
        let v = vibrato(&sig, &track, &track, 0.03);
        for n in &v.notes {
            assert!((n.in_rate_hz - 6.2).abs() < 0.05, "{n:?}");
        }
    }

    #[test]
    fn jitter_added_by_high_frequency_wobble() {
        let (sig, tin) = fixture(|t| Some(target_at(t)));
        let (_, tout) = fixture(|t| {
            Some(
                target_at(t)
                    + 4.0 * (2.0 * std::f64::consts::PI * 40.0 * t).sin()
                    + 20.0 * (2.0 * std::f64::consts::PI * 1.0 * t).sin(),
            )
        });
        let j = jitter(&sig, &tin, &tout, 12.0);
        // 4 c peak at 40 Hz → 2.83 c RMS; the 1 Hz drift is rejected.
        assert!((j.added_rms - 2.83).abs() < 0.15, "{j:?}");
        assert!(j.in_rms < 1e-6);
    }

    #[test]
    fn voicing_counts() {
        let (sig, _) = fixture(|_| None);
        let latency = 100u32;
        let frames: Vec<PitchFrame> = (0..2000u64)
            .map(|k| {
                let pos = k * 48;
                let voiced = sig.is_voiced(pos as usize);
                PitchFrame {
                    sample_pos: pos + latency as u64,
                    flags: if voiced { frame_flags::VOICED } else { 0 },
                    ..PitchFrame::default()
                }
            })
            .collect();
        let v = voicing_f1(&sig, &frames, latency, 0.02);
        assert_eq!(v.fp + v.fn_, 0);
        assert!((v.f1 - 1.0).abs() < 1e-12);
        let flipped: Vec<PitchFrame> = frames
            .iter()
            .map(|f| PitchFrame {
                flags: frame_flags::VOICED,
                ..*f
            })
            .collect();
        let v2 = voicing_f1(&sig, &flipped, latency, 0.02);
        assert!(v2.fp > 0 && v2.precision < 1.0);
    }

    #[test]
    fn latency_of_delayed_noise() {
        let mut r = crate::rng::Rng::new(5);
        let x: Vec<f32> = (0..20_000)
            .map(|i| {
                if (2000..6000).contains(&i) {
                    r.gaussian() as f32 * 0.1
                } else {
                    0.0
                }
            })
            .collect();
        for d in [0usize, 1, 48, 1234] {
            let mut y = vec![0.0f32; x.len() + d];
            y[d..].copy_from_slice(&x);
            let m = measure_latency(&x, &y, 3000).unwrap();
            assert_eq!(m.lag, d);
            assert!(m.peak > 0.999);
        }
    }

    #[test]
    fn click_detector_flags_injected_click() {
        let sr = 48_000.0;
        let x: Vec<f32> = (0..24_000)
            .map(|i| (0.3 * (2.0 * std::f64::consts::PI * 220.0 * i as f64 / sr).sin()) as f32)
            .collect();
        let clean = clicks(&x, &x, sr as f32, 0.02);
        assert_eq!(clean.clicks, 0);
        let mut y = x.clone();
        y[12_000] += 0.5;
        let c = clicks(&x, &y, sr as f32, 0.02);
        assert!(c.clicks >= 1 && c.max_excess_db > 20.0, "{c:?}");
    }

    #[test]
    fn lsd_zero_for_identical_and_positive_for_formant_shift() {
        use crate::synth::{Melody, Note, SignalBuilder, Voice, Vowel};
        let sr = 48_000.0;
        let mk = |v: Vowel| {
            SignalBuilder::new(sr)
                .silence(0.05)
                .melody(Melody::new(Voice::male()).note(Note::midi(50.0, 0.8).vowel(v).detune(0.0)))
                .build()
        };
        let a = mk(Vowel::A);
        let same = formant_lsd(&a, &a.audio);
        assert!(same.frames > 10 && same.mean_db < 1e-6);
        let other = formant_lsd(&a, &mk(Vowel::I).audio);
        assert!(other.mean_db > 4.0, "{other:?}");
    }

    #[test]
    fn segmental_snr_tolerates_fractional_wander() {
        let sr = 48_000.0;
        let x: Vec<f32> = (0..9600)
            .map(|i| {
                (0.3 * (2.0 * std::f64::consts::PI * 300.0 * i as f64 / sr).sin()
                    + 0.1 * (2.0 * std::f64::consts::PI * 1700.0 * i as f64 / sr).sin())
                    as f32
            })
            .collect();
        // Delay by 3.4 samples with a sinc interpolator.
        let s = KaiserSinc::new(16, 8.6);
        let xd: Vec<f64> = x.iter().map(|v| *v as f64).collect();
        let y: Vec<f32> = (0..x.len())
            .map(|i| s.interp(&xd, i as f64 - 3.4) as f32)
            .collect();
        let (snr, n) = aligned_segmental_snr(&x, &y, 960, -2.0, 10.0);
        assert!(n >= 8 && snr > 50.0, "snr {snr}");
    }
}
