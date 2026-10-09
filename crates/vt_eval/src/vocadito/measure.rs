//! Measurements on real singing against the vocadito annotations.
//!
//! Conventions:
//!
//! - **Time** is always *input* time in seconds. Engine telemetry is stamped
//!   in output samples (`sample_pos` = evidence time + D), so it is mapped
//!   back with `sample_pos − D` ([`TimedTrack::from_telemetry`]). The
//!   reference tracker runs on the latency-compensated output, which is
//!   already in input time.
//! - **Detector metrics** follow mir_eval's melody conventions on the
//!   annotation grid (≈ 5.8 ms): the estimate is sampled at each annotation
//!   frame (nearest frame); RPA/RCA count over annotation-voiced frames with an
//!   unvoiced estimate counting as wrong; voicing precision/recall/F1 over all
//!   frames.
//! - **Correction metrics** use the reference tracker's 1 ms frames. "Stable"
//!   frames are annotated voiced, at least 30 ms inside an annotator-1 note,
//!   voiced in the input reference track, and not inside a fast input move
//!   (|Δ| > 40 c over 20 ms of the 10 ms-smoothed input pitch). "In tune" is
//!   the distance from the 20 ms-smoothed pitch to the nearest legal note.

use vt_core::{PitchFrame, frame_flags};

use super::data::AnnNote;
use crate::dsp::median;
use crate::reference::PitchTrack;
use crate::synth::hz_to_cents64;

// ---------------------------------------------------------------------------
// Histogram (pooling error distributions across files without keeping frames)

const HIST_STEP: f64 = 0.1;
const HIST_BINS: usize = 2000; // 0-200 c

/// Histogram of absolute errors in cents (0.1 c bins to 200 c, exact beyond).
#[derive(Clone, Debug, PartialEq)]
pub struct Hist {
    bins: Vec<u64>,
    over: Vec<f64>,
    n: u64,
}

impl Default for Hist {
    fn default() -> Self {
        Self {
            bins: vec![0; HIST_BINS],
            over: Vec::new(),
            n: 0,
        }
    }
}

impl Hist {
    pub fn add(&mut self, v: f64) {
        if !v.is_finite() {
            return;
        }
        let v = v.abs();
        let i = (v / HIST_STEP) as usize;
        if i < HIST_BINS {
            self.bins[i] += 1;
        } else {
            self.over.push(v);
        }
        self.n += 1;
    }

    pub fn merge(&mut self, o: &Hist) {
        for (a, b) in self.bins.iter_mut().zip(&o.bins) {
            *a += b;
        }
        self.over.extend_from_slice(&o.over);
        self.n += o.n;
    }

    pub fn len(&self) -> u64 {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Percentile `p` (0-100); bin centres inside the binned range.
    pub fn percentile(&self, p: f64) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        let rank = ((p.clamp(0.0, 100.0) / 100.0) * (self.n - 1) as f64).round() as u64;
        let mut acc = 0u64;
        for (i, c) in self.bins.iter().enumerate() {
            acc += c;
            if acc > rank {
                return (i as f64 + 0.5) * HIST_STEP;
            }
        }
        let mut over = self.over.clone();
        over.sort_by(f64::total_cmp);
        over.get((rank - acc) as usize)
            .or(over.last())
            .copied()
            .unwrap_or(f64::NAN)
    }

    /// Percentage of values ≤ `c` (bin resolution 0.1 c).
    pub fn within(&self, c: f64) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        let k = ((c / HIST_STEP).round() as usize).min(HIST_BINS);
        let below: u64 = self.bins[..k].iter().sum::<u64>()
            + self.over.iter().filter(|v| **v <= c).count() as u64;
        100.0 * below as f64 / self.n as f64
    }
}

// ---------------------------------------------------------------------------
// Timed pitch tracks

/// A pitch track at (roughly regular) input times.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimedTrack {
    pub t: Vec<f64>,
    pub voiced: Vec<bool>,
    /// Absolute cents, NaN when unvoiced.
    pub cents: Vec<f64>,
    /// Lookups farther than this from every frame return None, s.
    pub max_gap_s: f64,
}

impl TimedTrack {
    fn with_gap(mut self) -> Self {
        let n = self.t.len();
        let step = if n >= 2 {
            (self.t[n - 1] - self.t[0]) / (n - 1) as f64
        } else {
            0.01
        };
        self.max_gap_s = 0.6 * step.max(1e-4);
        self
    }

    /// Engine telemetry mapped to input time (`sample_pos − latency`).
    pub fn from_telemetry(frames: &[PitchFrame], latency: u32, sample_rate: f64) -> Self {
        let mut tr = TimedTrack::default();
        for f in frames {
            let Some(p) = f.sample_pos.checked_sub(u64::from(latency)) else {
                continue;
            };
            let v =
                f.flags & frame_flags::VOICED != 0 && f.detected.is_finite() && f.detected > 0.0;
            tr.t.push(p as f64 / sample_rate);
            tr.voiced.push(v);
            tr.cents
                .push(if v { f64::from(f.detected) } else { f64::NAN });
        }
        tr.with_gap()
    }

    /// The target note of engine telemetry (voiced frames with a target).
    pub fn targets_from_telemetry(frames: &[PitchFrame], latency: u32, sample_rate: f64) -> Self {
        let mut tr = TimedTrack::default();
        for f in frames {
            let Some(p) = f.sample_pos.checked_sub(u64::from(latency)) else {
                continue;
            };
            let v = f.flags & frame_flags::VOICED != 0 && f.target.is_finite() && f.target > 0.0;
            tr.t.push(p as f64 / sample_rate);
            tr.voiced.push(v);
            tr.cents
                .push(if v { f64::from(f.target) } else { f64::NAN });
        }
        tr.with_gap()
    }

    pub fn from_reference(track: &PitchTrack) -> Self {
        let mut tr = TimedTrack::default();
        for f in &track.frames {
            tr.t.push(f.pos as f64 / track.sample_rate);
            tr.voiced.push(f.voiced && f.cents.is_finite());
            tr.cents.push(if f.voiced { f.cents } else { f64::NAN });
        }
        tr.with_gap()
    }

    pub fn from_annotation(f0: &[(f64, f64)]) -> Self {
        let mut tr = TimedTrack::default();
        for &(t, hz) in f0 {
            tr.t.push(t);
            tr.voiced.push(hz > 0.0);
            tr.cents.push(if hz > 0.0 {
                hz_to_cents64(hz)
            } else {
                f64::NAN
            });
        }
        tr.with_gap()
    }

    pub fn len(&self) -> usize {
        self.t.len()
    }

    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    /// Index of the frame nearest to `t` (None outside the track).
    pub fn nearest(&self, t: f64) -> Option<usize> {
        let i = self.t.partition_point(|x| *x < t);
        let mut best = None;
        let mut bd = f64::INFINITY;
        for j in [i.wrapping_sub(1), i] {
            if let Some(tj) = self.t.get(j) {
                let d = (tj - t).abs();
                if d < bd {
                    bd = d;
                    best = Some(j);
                }
            }
        }
        best.filter(|_| bd <= self.max_gap_s)
    }

    pub fn voiced_at(&self, t: f64) -> Option<bool> {
        self.nearest(t).map(|i| self.voiced[i])
    }

    /// Voicing onsets `(time, unvoiced gap before it)` and offsets (time of
    /// the first unvoiced frame after a voiced run).
    pub fn transitions(&self) -> (Vec<(f64, f64)>, Vec<f64>) {
        let mut onsets = Vec::new();
        let mut offsets = Vec::new();
        let mut prev = false;
        let mut last_off = self.t.first().copied().unwrap_or(0.0);
        for (i, &v) in self.voiced.iter().enumerate() {
            if v && !prev {
                onsets.push((self.t[i], self.t[i] - last_off));
            } else if !v && prev {
                offsets.push(self.t[i]);
                last_off = self.t[i];
            }
            prev = v;
        }
        (onsets, offsets)
    }
}

// ---------------------------------------------------------------------------
// Detector vs annotation

/// One annotation frame against the estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameCmp {
    pub t: f64,
    pub ref_v: bool,
    pub ref_c: f64,
    /// None when the estimate does not cover this time.
    pub est_v: Option<bool>,
    pub est_c: f64,
}

impl FrameCmp {
    /// Estimate − annotation, cents (NaN unless both voiced).
    pub fn diff(&self) -> f64 {
        if self.ref_v && self.est_v == Some(true) {
            self.est_c - self.ref_c
        } else {
            f64::NAN
        }
    }
}

/// Sample `est` at every annotation frame, shifted by `lag_s` (compare
/// `ann(t)` with `est(t + lag)`; a positive best lag means the estimate's
/// timestamps run late).
pub fn compare_frames(ann: &TimedTrack, est: &TimedTrack, lag_s: f64) -> Vec<FrameCmp> {
    (0..ann.len())
        .map(|i| {
            let j = est.nearest(ann.t[i] + lag_s);
            FrameCmp {
                t: ann.t[i],
                ref_v: ann.voiced[i],
                ref_c: ann.cents[i],
                est_v: j.map(|j| est.voiced[j]),
                est_c: j.map_or(f64::NAN, |j| est.cents[j]),
            }
        })
        .collect()
}

/// Octave-folded difference in (−600, 600].
fn fold(d: f64) -> f64 {
    d - 1200.0 * (d / 1200.0).round()
}

/// Pooled detector statistics (counts, so files can be merged).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DetCounts {
    pub ref_voiced: u64,
    pub ref_unvoiced: u64,
    pub hit50: u64,
    pub hit25: u64,
    pub chroma50: u64,
    /// Frames voiced in both.
    pub both: u64,
    pub octave_up: u64,
    pub octave_down: u64,
    /// Both voiced, off by more than 50 c, not an octave error.
    pub gross_other: u64,
    pub tp: u64,
    pub fp: u64,
    pub fn_: u64,
    pub tn: u64,
    /// Voicing counts skipping frames within 20 ms of an annotated transition.
    pub g_tp: u64,
    pub g_fp: u64,
    pub g_fn: u64,
    /// |Δ| on both-voiced frames within 50 c.
    pub fine: Hist,
    /// |Δ| on all both-voiced frames.
    pub all: Hist,
    /// Detector voicing onset − annotated onset, ms (onsets after ≥ 100 ms
    /// of annotated silence).
    pub onset_delay_ms: Vec<f64>,
    /// Onsets not voiced within 300 ms.
    pub onsets_missed: u64,
    /// Onsets the estimate was already voiced 50 ms before (gap bridged).
    pub onsets_bridged: u64,
    /// Onsets checked for spikes, and those where a voiced estimate within
    /// 100 ms was > 500 c from the onset's annotated pitch.
    pub onsets_checked: u64,
    pub onset_spikes: u64,
    /// Spikes whose bad frames reach ≥ 0.85 × the tracker's f_max (the
    /// estimate pinned at the top of its search range).
    pub onset_spikes_fmax: u64,
}

fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

fn f1(tp: u64, fp: u64, fn_: u64) -> f64 {
    let p = ratio(tp, tp + fp);
    let r = ratio(tp, tp + fn_);
    if tp == 0 { 0.0 } else { 2.0 * p * r / (p + r) }
}

impl DetCounts {
    pub fn merge(&mut self, o: &DetCounts) {
        self.ref_voiced += o.ref_voiced;
        self.ref_unvoiced += o.ref_unvoiced;
        self.hit50 += o.hit50;
        self.hit25 += o.hit25;
        self.chroma50 += o.chroma50;
        self.both += o.both;
        self.octave_up += o.octave_up;
        self.octave_down += o.octave_down;
        self.gross_other += o.gross_other;
        self.tp += o.tp;
        self.fp += o.fp;
        self.fn_ += o.fn_;
        self.tn += o.tn;
        self.g_tp += o.g_tp;
        self.g_fp += o.g_fp;
        self.g_fn += o.g_fn;
        self.fine.merge(&o.fine);
        self.all.merge(&o.all);
        self.onset_delay_ms.extend_from_slice(&o.onset_delay_ms);
        self.onsets_missed += o.onsets_missed;
        self.onsets_bridged += o.onsets_bridged;
        self.onsets_checked += o.onsets_checked;
        self.onset_spikes += o.onset_spikes;
        self.onset_spikes_fmax += o.onset_spikes_fmax;
    }

    pub fn onset_spike_pct(&self) -> f64 {
        100.0 * ratio(self.onset_spikes, self.onsets_checked)
    }

    pub fn rpa50(&self) -> f64 {
        ratio(self.hit50, self.ref_voiced)
    }
    pub fn rpa25(&self) -> f64 {
        ratio(self.hit25, self.ref_voiced)
    }
    pub fn rca50(&self) -> f64 {
        ratio(self.chroma50, self.ref_voiced)
    }
    /// Octave errors as a fraction of both-voiced frames.
    pub fn octave_rate(&self) -> f64 {
        ratio(self.octave_up + self.octave_down, self.both)
    }
    pub fn gross_other_rate(&self) -> f64 {
        ratio(self.gross_other, self.both)
    }
    pub fn precision(&self) -> f64 {
        ratio(self.tp, self.tp + self.fp)
    }
    pub fn recall(&self) -> f64 {
        ratio(self.tp, self.tp + self.fn_)
    }
    pub fn f1(&self) -> f64 {
        f1(self.tp, self.fp, self.fn_)
    }
    pub fn f1_guarded(&self) -> f64 {
        f1(self.g_tp, self.g_fp, self.g_fn)
    }
    pub fn onset_delay_median(&self) -> f64 {
        median(&self.onset_delay_ms)
    }
    pub fn onset_delay_p90(&self) -> f64 {
        crate::dsp::percentile(&self.onset_delay_ms, 90.0)
    }
}

/// Count everything over the compared frames. `transitions` are annotated
/// voicing transition times (for the guarded voicing F1).
pub fn det_counts(cmp: &[FrameCmp], transitions: &[f64], guard_s: f64) -> DetCounts {
    let mut c = DetCounts::default();
    for f in cmp {
        let Some(ev) = f.est_v else { continue };
        let i = transitions.partition_point(|t| *t < f.t);
        let near = transitions.get(i).is_some_and(|t| t - f.t <= guard_s)
            || (i > 0 && f.t - transitions[i - 1] <= guard_s);
        match (f.ref_v, ev) {
            (true, true) => {
                c.tp += 1;
                if !near {
                    c.g_tp += 1;
                }
            }
            (false, true) => {
                c.fp += 1;
                if !near {
                    c.g_fp += 1;
                }
            }
            (true, false) => {
                c.fn_ += 1;
                if !near {
                    c.g_fn += 1;
                }
            }
            (false, false) => c.tn += 1,
        }
        if !f.ref_v {
            c.ref_unvoiced += 1;
            continue;
        }
        c.ref_voiced += 1;
        if !ev {
            continue;
        }
        let d = f.est_c - f.ref_c;
        if !d.is_finite() {
            continue;
        }
        let a = d.abs();
        c.both += 1;
        c.all.add(a);
        if a <= 50.0 {
            c.hit50 += 1;
            c.fine.add(a);
        }
        if a <= 25.0 {
            c.hit25 += 1;
        }
        let folded = fold(d).abs();
        if folded <= 50.0 {
            c.chroma50 += 1;
        }
        if a > 50.0 {
            if a > 600.0 && folded <= 50.0 {
                if d > 0.0 {
                    c.octave_up += 1;
                } else {
                    c.octave_down += 1;
                }
            } else {
                c.gross_other += 1;
            }
        }
    }
    c
}

/// Timestamp offset of `est` against the annotation: the lag (ms, in
/// `−range_ms..=range_ms`) minimising the mean of min(|Δ|, 50 c) over
/// both-voiced frames. Returns (lag_ms, score at that lag, score at 0).
pub fn best_lag(ann: &TimedTrack, est: &TimedTrack, range_ms: i32) -> (f64, f64, f64) {
    let score = |lag_ms: i32| -> f64 {
        let cmp = compare_frames(ann, est, f64::from(lag_ms) * 1e-3);
        let (mut s, mut n) = (0.0, 0usize);
        for f in &cmp {
            let d = f.diff();
            if d.is_finite() {
                s += d.abs().min(50.0);
                n += 1;
            }
        }
        if n < 50 { f64::INFINITY } else { s / n as f64 }
    };
    let s0 = score(0);
    let mut best = (0.0, s0);
    for l in -range_ms..=range_ms {
        let s = score(l);
        if s < best.1 {
            best = (f64::from(l), s);
        }
    }
    (best.0, best.1, s0)
}

/// Detector voicing-onset delays against annotated onsets that follow at
/// least `min_gap_s` of annotated silence: (delays ms, missed, bridged).
pub fn onset_delays(
    onsets: &[(f64, f64)],
    est: &TimedTrack,
    min_gap_s: f64,
) -> (Vec<f64>, u64, u64) {
    let (mut d, mut missed, mut bridged) = (Vec::new(), 0u64, 0u64);
    for &(t0, gap) in onsets {
        if gap < min_gap_s {
            continue;
        }
        let lo = t0 - 0.05;
        let i = est.t.partition_point(|x| *x < lo);
        if i >= est.len() || est.t[i] - lo > 0.01 {
            continue; // not covered
        }
        if est.voiced[i] {
            bridged += 1;
            continue;
        }
        match (i..est.len())
            .take_while(|j| est.t[*j] <= t0 + 0.3)
            .find(|j| est.voiced[*j])
        {
            Some(j) => d.push((est.t[j] - t0) * 1000.0),
            None => missed += 1,
        }
    }
    (d, missed, bridged)
}

/// Gross estimate errors right at voicing onsets: for every annotated onset
/// after ≥ `min_gap_s` of silence, the voiced estimate frames in
/// [onset − 50 ms, onset + 100 ms] are compared with the median annotated
/// pitch of the onset's first 60 ms; any frame off by > 500 c makes the onset
/// a spike (kind `onset_spike_fmax` when a bad frame reaches 0.85 ×
/// `f_max_hz`, the top of the tracker's search range, else `onset_spike`).
/// Returns (onsets checked, spike spans with median Δ).
pub fn onset_spikes(
    onsets: &[(f64, f64)],
    ann: &TimedTrack,
    est: &TimedTrack,
    min_gap_s: f64,
    f_max_hz: f64,
) -> (u64, Vec<RawSegment>) {
    let top = hz_to_cents64(0.85 * f_max_hz);
    let mut checked = 0u64;
    let mut segs = Vec::new();
    for &(t0, gap) in onsets {
        if gap < min_gap_s {
            continue;
        }
        let a0 = ann.t.partition_point(|x| *x < t0);
        let early: Vec<f64> = (a0..ann.len())
            .take_while(|i| ann.t[*i] <= t0 + 0.06)
            .map(|i| ann.cents[i])
            .filter(|c| c.is_finite())
            .collect();
        let reference = median(&early);
        if !reference.is_finite() {
            continue;
        }
        let e0 = est.t.partition_point(|x| *x < t0 - 0.05);
        if e0 >= est.len() {
            continue;
        }
        checked += 1;
        let bad: Vec<usize> = (e0..est.len())
            .take_while(|j| est.t[*j] <= t0 + 0.1)
            .filter(|j| est.voiced[*j] && (est.cents[*j] - reference).abs() > 500.0)
            .collect();
        if let (Some(&a), Some(&b)) = (bad.first(), bad.last()) {
            let step = est.max_gap_s / 0.6;
            let at_top = bad.iter().any(|j| est.cents[*j] >= top);
            segs.push(RawSegment {
                kind: if at_top {
                    "onset_spike_fmax"
                } else {
                    "onset_spike"
                },
                start_s: est.t[a],
                end_s: est.t[b] + step,
                value: median(
                    &bad.iter()
                        .map(|j| est.cents[*j] - reference)
                        .collect::<Vec<_>>(),
                ),
            });
        }
    }
    (checked, segs)
}

// ---------------------------------------------------------------------------
// Segments (failure analysis)

/// A time span flagged by one of the detectors below, before context tags.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawSegment {
    pub kind: &'static str,
    pub start_s: f64,
    pub end_s: f64,
    /// Kind-specific magnitude: median Δ cents (pitch kinds), duration ms
    /// (voicing kinds), dB excess (clicks), correction cents (shifts).
    pub value: f64,
}

/// Runs of indices where `pred` holds, merging holes of ≤ `max_hole`
/// indices; inclusive `(first, last)`.
pub fn runs(n: usize, pred: impl Fn(usize) -> bool, max_hole: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        if !pred(i) {
            continue;
        }
        match out.last_mut() {
            Some(last) if i - last.1 <= max_hole + 1 => last.1 = i,
            _ => out.push((i, i)),
        }
    }
    out
}

/// Detector failure spans from a frame comparison: octave errors (≥ 2
/// frames), other gross errors (> 100 c, ≥ 3 frames), voicing misses and
/// false voicing (≥ 50 ms).
pub fn detector_segments(cmp: &[FrameCmp]) -> Vec<RawSegment> {
    let mut segs = Vec::new();
    let n = cmp.len();
    let step = if n >= 2 {
        (cmp[n - 1].t - cmp[0].t) / (n - 1) as f64
    } else {
        0.0058
    };
    let span = |a: usize, b: usize| (cmp[a].t, cmp[b].t + step);
    let med_diff = |a: usize, b: usize| {
        median(
            &(a..=b)
                .map(|i| cmp[i].diff())
                .filter(|d| d.is_finite())
                .collect::<Vec<_>>(),
        )
    };
    let is_oct = |i: usize| {
        let d = cmp[i].diff();
        d.is_finite() && d.abs() > 600.0 && fold(d).abs() <= 50.0
    };
    for (a, b) in runs(n, is_oct, 1) {
        if b > a {
            let (s, e) = span(a, b);
            segs.push(RawSegment {
                kind: "octave_err",
                start_s: s,
                end_s: e,
                value: med_diff(a, b),
            });
        }
    }
    let is_gross = |i: usize| {
        let d = cmp[i].diff();
        d.is_finite() && d.abs() > 100.0 && !is_oct(i)
    };
    for (a, b) in runs(n, is_gross, 1) {
        if b >= a + 2 {
            let (s, e) = span(a, b);
            segs.push(RawSegment {
                kind: "gross_err",
                start_s: s,
                end_s: e,
                value: med_diff(a, b),
            });
        }
    }
    for (kind, pred) in [
        (
            "voicing_miss",
            Box::new(|i: usize| cmp[i].ref_v && cmp[i].est_v == Some(false))
                as Box<dyn Fn(usize) -> bool>,
        ),
        (
            "false_voicing",
            Box::new(|i: usize| !cmp[i].ref_v && cmp[i].est_v == Some(true)),
        ),
    ] {
        for (a, b) in runs(n, pred, 1) {
            let (s, e) = span(a, b);
            if e - s >= 0.05 {
                segs.push(RawSegment {
                    kind,
                    start_s: s,
                    end_s: e,
                    value: (e - s) * 1000.0,
                });
            }
        }
    }
    segs
}

// ---------------------------------------------------------------------------
// Correction measures

/// Distance-to-grid helper: nearest legal note (absolute cents) to `c`.
pub fn nearest_legal(c: f64, legal: &[bool; 12]) -> f64 {
    if !c.is_finite() {
        return f64::NAN;
    }
    let base = (c / 100.0).floor() as i64;
    let mut best = f64::NAN;
    let mut bd = f64::INFINITY;
    for m in base - 12..=base + 13 {
        if legal[m.rem_euclid(12) as usize] {
            let d = (c - 100.0 * m as f64).abs();
            if d < bd {
                bd = d;
                best = 100.0 * m as f64;
            }
        }
    }
    best
}

/// Centred mean over ±`half` frames of the finite values, NaN when the
/// centre is NaN or fewer than `half + 1` values are finite.
pub fn smooth(c: &[f64], half: usize) -> Vec<f64> {
    let n = c.len();
    let mut sum = vec![0.0; n + 1];
    let mut cnt = vec![0usize; n + 1];
    for (i, v) in c.iter().enumerate() {
        let ok = v.is_finite();
        sum[i + 1] = sum[i] + if ok { *v } else { 0.0 };
        cnt[i + 1] = cnt[i] + usize::from(ok);
    }
    (0..n)
        .map(|k| {
            if !c[k].is_finite() {
                return f64::NAN;
            }
            let lo = k.saturating_sub(half);
            let hi = (k + half + 1).min(n);
            let m = cnt[hi] - cnt[lo];
            if m > half {
                (sum[hi] - sum[lo]) / m as f64
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Cents per reference frame (NaN when unvoiced).
pub fn track_cents(t: &PitchTrack) -> Vec<f64> {
    t.frames
        .iter()
        .map(|f| if f.voiced { f.cents } else { f64::NAN })
        .collect()
}

/// Everything about the input that every correction render is compared with.
#[derive(Clone, Debug)]
pub struct InputSide {
    pub sample_rate: f64,
    /// Reference-track hop, samples.
    pub hop: usize,
    pub in_c: Vec<f64>,
    /// 10 ms and 20 ms centred means of `in_c`.
    pub in_s10: Vec<f64>,
    pub in_s20: Vec<f64>,
    /// Annotated voiced at each reference frame, and the annotated pitch
    /// (cents, NaN when unvoiced).
    pub ann_v: Vec<bool>,
    pub ann_c: Vec<f64>,
    /// At least 30 ms inside an annotator-1 note.
    pub interior: Vec<bool>,
    pub stable: Vec<bool>,
    /// Stable frames in the first 100 ms of a note's interior.
    pub early: Vec<bool>,
    /// Stable frames whose input wobbles (vibrato or unsteady pitch): std of
    /// the 10 ms-smoothed input over ±100 ms > 15 c.
    pub wobbly: Vec<bool>,
    /// Interior, annotated-voiced, input-voiced frames rejected as moving.
    pub moving: u64,
    /// Per A1 note: (first frame, last frame, annotated cents) of its interior.
    pub note_frames: Vec<(usize, usize, f64)>,
    /// Onsets to settle on: (annotated voicing onset, end of its A1 note), s.
    pub settle_onsets: Vec<(f64, f64)>,
}

const GUARD_S: f64 = 0.03;

impl InputSide {
    pub fn new(
        in_track: &PitchTrack,
        ann: &TimedTrack,
        notes: &[AnnNote],
        onsets: &[(f64, f64)],
    ) -> Self {
        let sr = in_track.sample_rate;
        let hop = in_track.hop.max(1);
        let n = in_track.frames.len();
        let time = |k: usize| (k * hop) as f64 / sr;
        let frame_of = |t: f64| ((t * sr) / hop as f64).max(0.0);
        let in_c = track_cents(in_track);
        let in_s10 = smooth(&in_c, 5);
        let in_s20 = smooth(&in_c, 10);
        let ann_c: Vec<f64> = (0..n)
            .map(|k| ann.nearest(time(k)).map_or(f64::NAN, |i| ann.cents[i]))
            .collect();
        let ann_v: Vec<bool> = ann_c.iter().map(|c| c.is_finite()).collect();
        let mut interior = vec![false; n];
        let mut note_frames = Vec::new();
        for nt in notes {
            let a = frame_of(nt.start_s + GUARD_S).ceil() as usize;
            let b = frame_of(nt.end_s() - GUARD_S).floor() as usize;
            if a > b || a >= n {
                continue;
            }
            let b = b.min(n - 1);
            interior[a..=b].iter_mut().for_each(|x| *x = true);
            note_frames.push((a, b, nt.cents()));
        }
        let moving_at = |k: usize| -> bool {
            let (Some(a), Some(b)) = (
                k.checked_sub(10).and_then(|i| in_s10.get(i)),
                in_s10.get(k + 10),
            ) else {
                return true;
            };
            !(a.is_finite() && b.is_finite()) || (b - a).abs() > 40.0
        };
        let mut moving = 0u64;
        let stable: Vec<bool> = (0..n)
            .map(|k| {
                if !(interior[k] && ann_v[k] && in_c[k].is_finite()) {
                    return false;
                }
                if moving_at(k) {
                    moving += 1;
                    false
                } else {
                    true
                }
            })
            .collect();
        let mut early = vec![false; n];
        for &(a, b, _) in &note_frames {
            let e = (a + ((0.1 * sr) / hop as f64).round() as usize).min(b);
            early[a..=e].copy_from_slice(&stable[a..=e]);
        }
        let half = ((0.1 * sr) / hop as f64).round() as usize;
        let wobbly: Vec<bool> = (0..n)
            .map(|k| {
                if !stable[k] {
                    return false;
                }
                let w: Vec<f64> = in_s10[k.saturating_sub(half)..(k + half + 1).min(n)]
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();
                if w.len() < half {
                    return false;
                }
                let m = w.iter().sum::<f64>() / w.len() as f64;
                (w.iter().map(|v| (v - m).powi(2)).sum::<f64>() / w.len() as f64).sqrt() > 15.0
            })
            .collect();
        let settle_onsets = onsets
            .iter()
            .filter(|(_, gap)| *gap >= 0.1)
            .filter_map(|&(t0, _)| {
                notes
                    .iter()
                    .find(|nt| {
                        t0 >= nt.start_s - 0.05 && t0 <= nt.start_s + 0.1 && nt.dur_s >= 0.15
                    })
                    .map(|nt| (t0, nt.end_s()))
            })
            .collect();
        Self {
            sample_rate: sr,
            hop,
            in_c,
            in_s10,
            in_s20,
            ann_v,
            ann_c,
            interior,
            stable,
            early,
            wobbly,
            moving,
            note_frames,
            settle_onsets,
        }
    }

    pub fn time(&self, k: usize) -> f64 {
        (k * self.hop) as f64 / self.sample_rate
    }

    pub fn frame_at(&self, t: f64) -> usize {
        ((t * self.sample_rate) / self.hop as f64).round().max(0.0) as usize
    }

    pub fn len(&self) -> usize {
        self.in_c.len()
    }

    pub fn is_empty(&self) -> bool {
        self.in_c.is_empty()
    }
}

/// Correction statistics of one render (counts so files can be pooled).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CorrCounts {
    /// Stable frames (input side).
    pub stable: u64,
    /// Stable frames where the output had no pitch.
    pub dropouts: u64,
    pub moving: u64,
    /// Distance to the nearest legal note: output (stable ∧ output voiced)
    /// and input (all stable frames).
    pub out: Hist,
    pub inp: Hist,
    /// Output distance split: first 100 ms of notes, wobbly input (vibrato
    /// or unsteady), and the steady rest.
    pub out_early: Hist,
    pub out_wobbly: Hist,
    pub out_steady: Hist,
    /// Note-level distance (median pitch over the note's stable frames).
    pub note_out: Vec<f64>,
    pub note_in: Vec<f64>,
    /// Notes scored, and how many landed on the annotator's note (rounded to
    /// the nearest legal note).
    pub notes: u64,
    pub notes_agree_out: u64,
    pub notes_agree_in: u64,
    /// Onset settling (within 10 c of a legal note for 20 ms), ms.
    pub settle_ms: Vec<f64>,
    pub settle_missed: u64,
    pub click_frames: u64,
    pub click_events: u64,
    pub click_max_db: f64,
    pub dur_s: f64,
    /// Formant LSD: sum of per-file means × frames, frames.
    pub lsd_sum: f64,
    pub lsd_frames: u64,
    pub shift_segments: u64,
    pub jumps: u64,
    /// Warble episodes (target A→B→A flips within 60 ms of each other) and
    /// the individual flips in them.
    pub flickers: u64,
    pub flicker_flips: u64,
    pub dropout_runs: u64,
    /// Settle onsets, and those whose first 100 ms had ≥ 5 ms of output with
    /// no pitch or > 150 c away from the input while the input was voiced.
    pub onsets_checked: u64,
    pub onset_glitches: u64,
    pub nonfinite: u64,
    pub blowups: u64,
    /// Largest output/input peak ratio, dB.
    pub peak_ratio_db: f64,
}

impl CorrCounts {
    pub fn merge(&mut self, o: &CorrCounts) {
        self.stable += o.stable;
        self.dropouts += o.dropouts;
        self.moving += o.moving;
        self.out.merge(&o.out);
        self.inp.merge(&o.inp);
        self.out_early.merge(&o.out_early);
        self.out_wobbly.merge(&o.out_wobbly);
        self.out_steady.merge(&o.out_steady);
        self.note_out.extend_from_slice(&o.note_out);
        self.note_in.extend_from_slice(&o.note_in);
        self.notes += o.notes;
        self.notes_agree_out += o.notes_agree_out;
        self.notes_agree_in += o.notes_agree_in;
        self.settle_ms.extend_from_slice(&o.settle_ms);
        self.settle_missed += o.settle_missed;
        self.click_frames += o.click_frames;
        self.click_events += o.click_events;
        self.click_max_db = self.click_max_db.max(o.click_max_db);
        self.dur_s += o.dur_s;
        self.lsd_sum += o.lsd_sum;
        self.lsd_frames += o.lsd_frames;
        self.shift_segments += o.shift_segments;
        self.jumps += o.jumps;
        self.flickers += o.flickers;
        self.flicker_flips += o.flicker_flips;
        self.dropout_runs += o.dropout_runs;
        self.onsets_checked += o.onsets_checked;
        self.onset_glitches += o.onset_glitches;
        self.nonfinite += o.nonfinite;
        self.blowups += o.blowups;
        self.peak_ratio_db = self.peak_ratio_db.max(o.peak_ratio_db);
    }

    pub fn dropout_pct(&self) -> f64 {
        100.0 * ratio(self.dropouts, self.stable)
    }
    pub fn lsd_mean(&self) -> f64 {
        if self.lsd_frames == 0 {
            f64::NAN
        } else {
            self.lsd_sum / self.lsd_frames as f64
        }
    }
    pub fn clicks_per_min(&self) -> f64 {
        if self.dur_s <= 0.0 {
            f64::NAN
        } else {
            self.click_events as f64 / (self.dur_s / 60.0)
        }
    }
    pub fn agree_out_pct(&self) -> f64 {
        100.0 * ratio(self.notes_agree_out, self.notes)
    }
    pub fn agree_in_pct(&self) -> f64 {
        100.0 * ratio(self.notes_agree_in, self.notes)
    }
    pub fn settle_median(&self) -> f64 {
        median(&self.settle_ms)
    }
    pub fn onset_glitch_pct(&self) -> f64 {
        100.0 * ratio(self.onset_glitches, self.onsets_checked)
    }
    pub fn settle_p90(&self) -> f64 {
        crate::dsp::percentile(&self.settle_ms, 90.0)
    }
}

/// Note index of a legal-note pitch (cents / 100, rounded).
fn note_index(c: f64) -> Option<i64> {
    c.is_finite().then(|| (c / 100.0).round() as i64)
}

/// Pitch-domain correction measures of one render: in-tune distances,
/// note-level agreement, settling, and the pitch failure spans
/// (`gross_shift`, `out_jump`, `dropout`, `flicker`).
pub fn correction_pitch(
    inp: &InputSide,
    out_track: &PitchTrack,
    targets: &TimedTrack,
    legal: &[bool; 12],
) -> (CorrCounts, Vec<RawSegment>) {
    let mut c = CorrCounts::default();
    let mut segs = Vec::new();
    let n = inp.len().min(out_track.frames.len());
    // A pitch artifact needs the input track to agree with the annotation
    // (otherwise the reference tracker itself erred on the input, e.g. an
    // octave flip on a period-doubled voice) and the output to disagree.
    let in_ok = |k: usize| inp.ann_c[k].is_finite() && (inp.in_c[k] - inp.ann_c[k]).abs() <= 150.0;
    let out_c = track_cents(out_track);
    let out_s10 = smooth(&out_c, 5);
    let out_s20 = smooth(&out_c, 10);
    let dist = |v: f64| (v - nearest_legal(v, legal)).abs();

    c.moving = inp.moving;
    for k in 0..n {
        if !inp.stable[k] {
            continue;
        }
        c.stable += 1;
        c.inp.add(dist(inp.in_s20[k]));
        if out_c[k].is_finite() {
            let v = if out_s20[k].is_finite() {
                out_s20[k]
            } else {
                out_c[k]
            };
            let d = dist(v);
            c.out.add(d);
            if inp.early[k] {
                c.out_early.add(d);
            } else if inp.wobbly[k] {
                c.out_wobbly.add(d);
            } else {
                c.out_steady.add(d);
            }
        } else {
            c.dropouts += 1;
        }
    }

    // Note level.
    for &(a, b, ann_c) in &inp.note_frames {
        let b = b.min(n.saturating_sub(1));
        if a > b {
            continue;
        }
        let ks: Vec<usize> = (a..=b).filter(|k| inp.stable[*k]).collect();
        if ks.len() < 20 {
            continue;
        }
        let vin: Vec<f64> = ks.iter().map(|k| inp.in_c[*k]).collect();
        let vout: Vec<f64> = ks
            .iter()
            .map(|k| out_c[*k])
            .filter(|v| v.is_finite())
            .collect();
        if 2 * vout.len() < ks.len() {
            continue;
        }
        let (mi, mo) = (median(&vin), median(&vout));
        let want = note_index(nearest_legal(ann_c, legal));
        c.notes += 1;
        c.note_in.push(dist(mi));
        c.note_out.push(dist(mo));
        if note_index(nearest_legal(mo, legal)) == want {
            c.notes_agree_out += 1;
        }
        if note_index(nearest_legal(mi, legal)) == want {
            c.notes_agree_in += 1;
        }
    }

    // Settling after onsets: 10 ms-smoothed output within 10 c of one legal
    // note for 20 ms.
    let hold = ((0.02 * inp.sample_rate) / inp.hop as f64).round() as usize;
    for &(t0, t_end) in &inp.settle_onsets {
        let k0 = inp.frame_at(t0);
        let k_end = inp.frame_at(t_end.min(t0 + 0.5)).min(n);
        let mut found = None;
        let mut k = k0;
        while k + hold < k_end {
            let v = out_s10[k];
            if v.is_finite() {
                let l = nearest_legal(v, legal);
                if (k..=k + hold).all(|j| out_s10[j].is_finite() && (out_s10[j] - l).abs() <= 10.0)
                {
                    found = Some(k);
                    break;
                }
            }
            k += 1;
        }
        match found {
            Some(k) => c.settle_ms.push((inp.time(k) - t0) * 1000.0),
            None => c.settle_missed += 1,
        }
    }

    let hop_s = inp.hop as f64 / inp.sample_rate;
    let span = |a: usize, b: usize| (inp.time(a), inp.time(b) + hop_s);
    // Onset glitches: output silent/garbled or far from the input in the
    // first 100 ms of a note while the input is voiced.
    let min_bad = ((0.005 * inp.sample_rate) / inp.hop as f64).ceil() as usize;
    for &(t0, _) in &inp.settle_onsets {
        let k0 = inp.frame_at(t0);
        let k1 = inp.frame_at(t0 + 0.1).min(n);
        c.onsets_checked += 1;
        let bad: Vec<usize> = (k0..k1)
            .filter(|k| inp.ann_v[*k] && inp.in_c[*k].is_finite())
            .filter(|k| {
                !out_c[*k].is_finite()
                    || (in_ok(*k)
                        && (out_c[*k] - inp.in_c[*k]).abs() > 150.0
                        && (out_c[*k] - inp.ann_c[*k]).abs() > 150.0)
            })
            .collect();
        if bad.len() >= min_bad.max(1) {
            c.onset_glitches += 1;
            let (s0, e0) = span(bad[0], bad[bad.len() - 1]);
            segs.push(RawSegment {
                kind: "onset_glitch",
                start_s: s0,
                end_s: e0,
                value: bad.len() as f64 * hop_s * 1000.0,
            });
        }
    }
    // Gross shifts: |output − input| > 150 c (no legal correction needs more
    // than 100 c) for ≥ 15 ms while the input is not leaping.
    let fast_in = |k: usize| match (
        k.checked_sub(10).and_then(|i| inp.in_s10.get(i)),
        inp.in_s10.get(k + 10),
    ) {
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (b - a).abs() > 100.0,
        _ => false,
    };
    let shift = |k: usize| out_c[k] - inp.in_c[k];
    let is_shift = |k: usize| {
        in_ok(k)
            && out_c[k].is_finite()
            && shift(k).abs() > 150.0
            && (out_c[k] - inp.ann_c[k]).abs() > 150.0
            && !fast_in(k)
    };
    let min_len = ((0.015 * inp.sample_rate) / inp.hop as f64).ceil() as usize;
    for (a, b) in runs(n, is_shift, 2) {
        if b + 1 - a >= min_len {
            let (s, e) = span(a, b);
            let v = median(
                &(a..=b)
                    .map(shift)
                    .filter(|v| v.is_finite())
                    .collect::<Vec<_>>(),
            );
            segs.push(RawSegment {
                kind: "gross_shift",
                start_s: s,
                end_s: e,
                value: v,
            });
            c.shift_segments += 1;
        }
    }
    // Output jumps: > 250 c within 5 ms while the input moved < 80 c.
    let is_jump = |k: usize| {
        let j = k + 5;
        j < n
            && inp.ann_v[k]
            && [out_c[k], out_c[j], inp.in_c[k], inp.in_c[j]]
                .iter()
                .all(|v| v.is_finite())
            && (out_c[j] - out_c[k]).abs() > 250.0
            && (inp.in_c[j] - inp.in_c[k]).abs() < 80.0
            && in_ok(k)
            && in_ok(j)
            && ((out_c[k] - inp.ann_c[k]).abs() > 150.0 || (out_c[j] - inp.ann_c[j]).abs() > 150.0)
    };
    for (a, b) in runs(n, is_jump, 20) {
        let (s, e) = span(a, b + 5);
        let v = (a..=b)
            .map(|k| out_c[k + 5] - out_c[k])
            .filter(|v| v.is_finite())
            .fold(0.0f64, |m, v| if v.abs() > m.abs() { v } else { m });
        segs.push(RawSegment {
            kind: "out_jump",
            start_s: s,
            end_s: e,
            value: v,
        });
        c.jumps += 1;
    }
    // Output dropouts: ≥ 20 ms unvoiced output inside notes where the input
    // and the annotation are voiced.
    let is_drop = |k: usize| {
        inp.interior[k] && inp.ann_v[k] && inp.in_c[k].is_finite() && !out_c[k].is_finite()
    };
    for (a, b) in runs(n, is_drop, 1) {
        let (s, e) = span(a, b);
        if e - s >= 0.02 {
            segs.push(RawSegment {
                kind: "dropout",
                start_s: s,
                end_s: e,
                value: (e - s) * 1000.0,
            });
            c.dropout_runs += 1;
        }
    }
    // Target flicker: A → B → A with B shorter than 40 ms, contiguous.
    let mut tr_runs: Vec<(f64, f64, f64)> = Vec::new(); // (start, end, target)
    for i in 0..targets.len() {
        if !targets.voiced[i] {
            tr_runs.push((targets.t[i], targets.t[i], f64::NAN));
            continue;
        }
        let v = targets.cents[i];
        match tr_runs.last_mut() {
            Some(last) if (last.2 - v).abs() < 1.0 => last.1 = targets.t[i],
            _ => tr_runs.push((targets.t[i], targets.t[i], v)),
        }
    }
    let mut episode: Option<RawSegment> = None;
    for w in tr_runs.windows(3) {
        let (p, m, q) = (w[0], w[1], w[2]);
        if p.2.is_finite()
            && m.2.is_finite()
            && q.2.is_finite()
            && (p.2 - q.2).abs() < 1.0
            && (m.2 - p.2).abs() >= 1.0
            && m.1 - m.0 < 0.04
        {
            c.flicker_flips += 1;
            let end = m.1 + targets.max_gap_s / 0.6;
            match episode.as_mut() {
                Some(e) if m.0 - e.end_s <= 0.06 => {
                    e.end_s = end;
                    e.value += 1.0;
                }
                _ => {
                    if let Some(e) = episode.take() {
                        segs.push(e);
                        c.flickers += 1;
                    }
                    episode = Some(RawSegment {
                        kind: "flicker",
                        start_s: m.0,
                        end_s: end,
                        value: 1.0,
                    });
                }
            }
        }
    }
    if let Some(e) = episode.take() {
        segs.push(e);
        c.flickers += 1;
    }
    (c, segs)
}

/// Merge click frames (time, excess dB) closer than `gap_s` into spans.
pub fn click_segments(frames: &[(f64, f64)], gap_s: f64) -> Vec<RawSegment> {
    let mut out: Vec<RawSegment> = Vec::new();
    for &(t, ex) in frames {
        match out.last_mut() {
            Some(s) if t - s.end_s <= gap_s => {
                s.end_s = t + 0.001;
                s.value = s.value.max(ex);
            }
            _ => out.push(RawSegment {
                kind: "click",
                start_s: t - 0.001,
                end_s: t + 0.001,
                value: ex,
            }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hist_percentiles_and_within() {
        let mut h = Hist::default();
        for i in 0..1000 {
            h.add(i as f64 * 0.05); // 0 .. 49.95
        }
        h.add(500.0);
        assert_eq!(h.len(), 1001);
        assert!((h.percentile(50.0) - 25.0).abs() < 0.11);
        assert!((h.within(10.0) - 100.0 * 200.0 / 1001.0).abs() < 0.2);
        assert_eq!(h.percentile(100.0), 500.0);
        let mut g = Hist::default();
        g.merge(&h);
        assert_eq!(g, h);
    }

    #[test]
    fn nearest_legal_notes() {
        let chrom = [true; 12];
        assert_eq!(nearest_legal(6940.0, &chrom), 6900.0);
        assert_eq!(nearest_legal(6960.0, &chrom), 7000.0);
        // C major: between E (64) and F (65) both legal; between C and D the
        // black key is illegal.
        let mut cmaj = [false; 12];
        for pc in [0, 2, 4, 5, 7, 9, 11] {
            cmaj[pc] = true;
        }
        assert_eq!(nearest_legal(6090.0, &cmaj), 6000.0); // C# − 10 c → C
        assert_eq!(nearest_legal(6110.0, &cmaj), 6200.0); // C# + 10 c → D
        assert_eq!(nearest_legal(6460.0, &cmaj), 6500.0); // E/F midpoint + 10 c
        assert!(nearest_legal(f64::NAN, &cmaj).is_nan());
        assert!(nearest_legal(6000.0, &[false; 12]).is_nan());
    }

    #[test]
    fn smoothing_needs_enough_valid_frames() {
        let c = vec![
            1.0,
            2.0,
            3.0,
            f64::NAN,
            5.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            9.0,
        ];
        let s = smooth(&c, 1);
        assert!((s[1] - 2.0).abs() < 1e-12);
        assert!((s[2] - 2.5).abs() < 1e-12);
        assert!(s[3].is_nan());
        assert!(s[8].is_nan()); // only itself valid
    }

    #[test]
    fn runs_merge_small_holes() {
        let v = [true, true, false, true, false, false, true];
        assert_eq!(runs(v.len(), |i| v[i], 1), vec![(0, 3), (6, 6)]);
        assert_eq!(runs(v.len(), |i| v[i], 0), vec![(0, 1), (3, 3), (6, 6)]);
    }

    fn grid(n: usize, step: f64, f: impl Fn(usize) -> f64) -> TimedTrack {
        let f0: Vec<(f64, f64)> = (0..n).map(|i| (i as f64 * step, f(i))).collect();
        TimedTrack::from_annotation(&f0)
    }

    #[test]
    fn detector_counts_and_octave_errors() {
        // Annotation: 100 frames at 220 Hz then 50 unvoiced.
        let ann = grid(150, 0.0058, |i| if i < 100 { 220.0 } else { 0.0 });
        // Estimate on a finer grid: 10 c sharp, an octave up for frames 40-49,
        // unvoiced for 60-69, falsely voiced 120-129 (annotation frames).
        let est = grid(1500, 0.00058, |j| {
            let i = j / 10;
            if (60..70).contains(&i) || (100..150).contains(&i) && !(120..130).contains(&i) {
                0.0
            } else if (40..50).contains(&i) {
                440.0 * (10.0f64 / 1200.0).exp2()
            } else {
                220.0 * (10.0f64 / 1200.0).exp2()
            }
        });
        let cmp = compare_frames(&ann, &est, 0.0);
        let (onsets, offsets) = ann.transitions();
        assert_eq!(onsets.len(), 1);
        assert_eq!(offsets.len(), 1);
        let c = det_counts(&cmp, &offsets, 0.02);
        assert_eq!(c.ref_voiced, 100);
        assert_eq!(c.both, 90);
        assert_eq!(c.octave_up, 10);
        assert_eq!(c.hit50, 80);
        assert_eq!(c.hit25, 80);
        assert_eq!(c.chroma50, 90);
        assert_eq!((c.tp, c.fn_, c.fp), (90, 10, 10));
        assert!((c.rpa50() - 0.8).abs() < 1e-12);
        assert!((c.fine.percentile(50.0) - 10.0).abs() < 0.11);
        let segs = detector_segments(&cmp);
        let kinds: Vec<&str> = segs.iter().map(|s| s.kind).collect();
        assert!(kinds.contains(&"octave_err"), "{kinds:?}");
        assert!(kinds.contains(&"voicing_miss"));
        assert!(kinds.contains(&"false_voicing"));
        let oct = segs.iter().find(|s| s.kind == "octave_err").unwrap();
        assert!((oct.start_s - 40.0 * 0.0058).abs() < 1e-9);
        assert!((oct.value - 1210.0).abs() < 1e-6, "{}", oct.value);
    }

    #[test]
    fn lag_scan_finds_a_shifted_estimate() {
        // A 6 Hz ±80 c vibrato; the estimate's timestamps run 7 ms late.
        let f = |t: f64| {
            220.0 * ((80.0 * (2.0 * std::f64::consts::PI * 6.0 * t).sin()) / 1200.0).exp2()
        };
        let ann = grid(400, 0.0058, |i| f(i as f64 * 0.0058));
        let est = grid(4000, 0.00058, |j| f(j as f64 * 0.00058 - 0.007));
        let (lag, score, s0) = best_lag(&ann, &est, 20);
        assert_eq!(lag, 7.0);
        assert!(score < 0.5 && s0 > 5.0, "{score} {s0}");
    }

    #[test]
    fn onset_delay_is_measured_from_the_annotation() {
        let ann = grid(200, 0.0058, |i| {
            if (50..150).contains(&i) { 200.0 } else { 0.0 }
        });
        let est = grid(2000, 0.00058, |j| {
            if (620..1500).contains(&j) { 200.0 } else { 0.0 }
        });
        let (onsets, _) = ann.transitions();
        let (d, missed, bridged) = onset_delays(&onsets, &est, 0.1);
        assert_eq!((missed, bridged), (0, 0));
        assert_eq!(d.len(), 1);
        assert!((d[0] - 120.0 * 0.58).abs() < 0.6, "{d:?}");
    }

    #[test]
    fn clicks_merge_into_events() {
        let f = [(0.100, 12.0), (0.101, 15.0), (0.104, 11.0), (0.300, 20.0)];
        let s = click_segments(&f, 0.01);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].value, 15.0);
        assert!((s[0].end_s - 0.105).abs() < 1e-12);
    }
}
