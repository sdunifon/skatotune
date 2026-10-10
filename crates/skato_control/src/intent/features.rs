//! Features of the learned switch veto: a ring of compact per-hop frames and,
//! for the frame being decided, a fixed feature vector (window means,
//! medians, extremes and spreads of the pitch, slopes, older context, segment
//! age, level cues, voicing and vibrato state; in Studio also statistics of
//! the visible lookahead), then one vector per candidate note.
//!
//! Every pitch feature is relative to the decided frame's pitch r, so the
//! features never see the note grid (a global detune changes nothing).
//! Windows are defined in milliseconds and converted with the hop length, so
//! every sample rate gives the same features. The arithmetic follows the
//! prototype the models were trained and evaluated with operation for
//! operation (bit-identical features); keep it that way unless the models
//! are retrained.
//!
//! Realtime: the ring and the median scratch are allocated in `new`; feature
//! computation is a bounded number of passes over at most `HIST_CAP` frames.

use crate::targets::TargetSet;

/// History capacity, hops (≈ 557 ms at 44.1 kHz with 64-sample hops).
pub(crate) const HIST_CAP: usize = 384;
/// Past-context features.
pub(crate) const NF_PAST: usize = 45;
/// Lookahead features (Studio).
pub(crate) const NF_FUT: usize = 13;
pub(crate) const NF_MAX: usize = NF_PAST + NF_FUT;
/// Candidate features, Live: the 45 shared ones, the candidate offset and its
/// size, 21 pitch summaries relative to the candidate, the size of the first
/// 12 of them, and the gaps to the candidate's legal neighbours.
pub(crate) const NFC_LIVE: usize = NF_PAST + 2 + 21 + 12 + 2;
/// Candidate features, Studio: as Live over all 58 shared features, plus 9
/// lookahead summaries relative to the candidate.
pub(crate) const NFC_STUDIO: usize = NF_MAX + 2 + 21 + 12 + 2 + 9;

/// Indices of the pitch summaries that get a candidate-relative version
/// (m3 m6 m12 m25 m50 m100 m200 md12 md25 md50 md100 md200 ps pc ctr o50_100
/// o100_200 mn25 mx25 mn100 mx100).
const REL_IDX: [usize; 21] = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 28, 29, 30, 22, 23, 16, 17, 18, 19];
/// Lookahead summaries that get a candidate-relative version (f3 f6 f12 f25
/// fall fmd fmn fmx flast).
const REL_FUT_IDX: [usize; 9] = [45, 46, 47, 48, 49, 50, 51, 52, 54];

/// Feature names, in order (training-data dumps; the training script uses
/// them to build the candidate features).
pub const FEATURE_NAMES: [&str; NF_MAX] = [
    "m3", "m6", "m12", "m25", "m50", "m100", "m200", "m400", // window means
    "n25", "n100", "n400", // voiced fraction of the window inside the segment
    "md12", "md25", "md50", "md100", "md200", // window medians
    "mn25", "mx25", "mn100", "mx100", // window min/max
    "sd25", "sd100", // window std
    "o50_100", "o100_200", "o200_400", // older-context means
    "slope12", "slope40", "slope_old", // LS slopes (hundreds of cents per 10 ms)
    "ps", "pc", "ctr", // pre-smoothed, analysis centre, vibrato-free centre
    "seg", "segs", "unv", // segment age (log, linear capped), time since unvoiced
    "lv", "lv_dmax", "lv_d25", "lv_dmin", // level cues
    "conf", "w", "ext", "rate", // detector confidence, vibrato state
    "vf50", "jmp", "tjmp", // voiced fraction (50 ms), largest hop step (25 ms), time since a > 40 c step
    // lookahead (Studio)
    "f3", "f6", "f12", "f25", "fall", "fmd", "fmn", "fmx", "fvf", "flast", "fslope", "flv", "fn",
];

/// Past windows, ms.
const W_MEAN_MS: [f32; 8] = [3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0, 400.0];
const W_MED_MS: [f32; 5] = [12.0, 25.0, 50.0, 100.0, 200.0];

/// One history frame (from the front end's analysis of a hop).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct HFrame {
    pub voiced: bool,
    pub onset: bool,
    /// The note was forgotten at this frame (no segment until the next onset).
    pub forget: bool,
    /// Source pitch p, pre-smoothed p_s, analysis centre p_c and the
    /// vibrato-free centre, cents.
    pub p: f32,
    pub p_s: f32,
    pub p_c: f32,
    pub centre: f32,
    /// Input level, dBFS.
    pub level: f32,
    pub conf: f32,
    /// Vibrato weight, extent (c) and rate (Hz).
    pub w: f32,
    pub ext: f32,
    pub rate: f32,
}

/// A stored frame with its segment clock.
#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    f: HFrame,
    /// Hops since the segment's onset frame (`u32::MAX`: no segment).
    seg: u32,
    /// Hops since the last unvoiced frame (0 when unvoiced).
    since_unv: u32,
}

/// Window lengths in hops (from the hop length).
#[derive(Clone, Copy, Debug)]
struct Windows {
    hop_ms: f32,
    mean: [usize; 8],
    med: [usize; 5],
    h3: usize,
    h6: usize,
    h12: usize,
    h25: usize,
    h40: usize,
    h50: usize,
    h60: usize,
    h100: usize,
}

#[inline]
fn wh(ms: f32, hop_ms: f32) -> usize {
    ((ms / hop_ms).round() as usize).max(1)
}

impl Windows {
    fn new(hop_ms: f32) -> Self {
        Self {
            hop_ms,
            mean: W_MEAN_MS.map(|ms| wh(ms, hop_ms).min(HIST_CAP)),
            med: W_MED_MS.map(|ms| wh(ms, hop_ms).min(HIST_CAP)),
            h3: wh(3.0, hop_ms),
            h6: wh(6.0, hop_ms),
            h12: wh(12.0, hop_ms),
            h25: wh(25.0, hop_ms),
            h40: wh(40.0, hop_ms),
            h50: wh(50.0, hop_ms),
            h60: wh(60.0, hop_ms),
            h100: wh(100.0, hop_ms),
        }
    }
}

/// Least-squares accumulator: (n, Σt, Σt², Σd, Σt·d).
type Ls = [f32; 5];

#[inline]
fn ls_add(a: &mut Ls, t: f32, d: f32) {
    a[0] += 1.0;
    a[1] += t;
    a[2] += t * t;
    a[3] += d;
    a[4] += t * d;
}

/// Slope of an accumulator in hundreds of cents per 10 ms (0 below 3 points).
#[inline]
fn ls_slope(a: &Ls) -> f32 {
    let den = a[0] * a[2] - a[1] * a[1];
    if a[0] >= 3.0 && den.abs() > 1e-9 { 10.0 * (a[0] * a[4] - a[1] * a[3]) / den } else { 0.0 }
}

/// Median of `v` (upper median for an even count), reordering it.
fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mid = v.len() / 2;
    v.select_nth_unstable_by(mid, f32::total_cmp);
    v.get(mid).copied().unwrap_or(0.0)
}

/// The frame history and the feature computation (see the module docs).
pub(crate) struct Features {
    win: Windows,
    hist: Vec<Slot>,
    head: usize,
    len: usize,
    scratch: Vec<f32>,
    /// Features of the last [`Features::compute`].
    pub feats: [f32; NF_MAX],
    /// Decided pitch r of the last `compute`, cents.
    pub r: f32,
    /// Lookahead hops the last `compute` used.
    pub fut: usize,
}

impl Features {
    pub fn new(hop_ms: f32) -> Self {
        Self {
            win: Windows::new(hop_ms),
            hist: vec![Slot::default(); HIST_CAP + 1],
            head: 0,
            len: 0,
            scratch: vec![0.0; HIST_CAP + 1],
            feats: [0.0; NF_MAX],
            r: 0.0,
            fut: 0,
        }
    }

    pub fn reset(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Append the newest front-end frame.
    pub fn push(&mut self, f: HFrame) {
        let cap = self.hist.len();
        let prev = if self.len > 0 { self.slot(0) } else { Slot { seg: u32::MAX, ..Slot::default() } };
        let seg = if f.onset {
            0
        } else if prev.seg != u32::MAX && !f.forget {
            prev.seg.saturating_add(1)
        } else {
            u32::MAX
        };
        let since_unv = if f.voiced { prev.since_unv.saturating_add(1) } else { 0 };
        self.head = (self.head + 1) % cap;
        if let Some(s) = self.hist.get_mut(self.head) {
            *s = Slot { f, seg, since_unv };
        }
        self.len = (self.len + 1).min(cap);
    }

    /// The slot `age` hops older than the newest.
    #[inline]
    fn slot(&self, age: usize) -> Slot {
        let cap = self.hist.len();
        self.hist.get((self.head + cap - (age % cap)) % cap).copied().unwrap_or_default()
    }

    #[inline]
    fn at(&self, age: usize) -> HFrame {
        self.slot(age).f
    }

    /// Hops since the segment onset of the frame `age` hops old (`u32::MAX`:
    /// no segment, or not in the history).
    pub fn segment_age(&self, age: usize) -> u32 {
        if age >= self.len { u32::MAX } else { self.slot(age).seg }
    }

    /// Features of the frame `age` hops back, using the `age` newer frames as
    /// lookahead when `use_future`. False when that frame is unvoiced or not
    /// in the history.
    pub fn compute(&mut self, age: usize, use_future: bool) -> bool {
        if age >= self.len {
            return false;
        }
        let s0 = self.slot(age);
        let f0 = s0.f;
        if !f0.voiced || !f0.p.is_finite() {
            return false;
        }
        let w = self.win;
        let r = f0.p;
        self.r = r;
        let mut x = [0.0f32; NF_MAX];
        let seg_len = if s0.seg == u32::MAX { 1 } else { (s0.seg as usize).saturating_add(1) };
        let avail = (self.len - age).min(seg_len).min(HIST_CAP);
        // One pass over the past (newest first): window sums, extremes, LS sums.
        let wmax = w.mean[7].min(avail);
        let mut sums = [0.0f32; 8];
        let mut sq = [0.0f32; 8];
        let mut cnt = [0usize; 8];
        let (mut mn25, mut mx25, mut mn100, mut mx100) = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        let mut ls12: Ls = [0.0; 5];
        let mut ls40: Ls = [0.0; 5];
        let mut lso: Ls = [0.0; 5];
        let mut jmp = 0.0f32;
        let mut tj: Option<usize> = None;
        let mut prev_p: Option<f32> = None;
        for i in 0..wmax {
            let fr = self.at(age + i);
            if !fr.voiced {
                prev_p = None;
                continue;
            }
            let d = (fr.p - r) * 0.01;
            for (k, &wk) in w.mean.iter().enumerate() {
                if i < wk {
                    sums[k] += d;
                    sq[k] += d * d;
                    cnt[k] += 1;
                }
            }
            if tj.is_none() && prev_p.is_some_and(|pp| (pp - fr.p).abs() > 40.0) {
                tj = Some(i);
            }
            if i < w.h25 {
                mn25 = mn25.min(d);
                mx25 = mx25.max(d);
                if let Some(pp) = prev_p {
                    jmp = jmp.max((pp - fr.p).abs() * 0.01);
                }
            }
            if i < w.h100 {
                mn100 = mn100.min(d);
                mx100 = mx100.max(d);
            }
            let t = -(i as f32) * w.hop_ms; // ms, ≤ 0
            if i < w.h12 {
                ls_add(&mut ls12, t, d);
            }
            if i < w.h40 {
                ls_add(&mut ls40, t, d);
            }
            if i >= w.h12 && i < w.h40 {
                ls_add(&mut lso, t, d);
            }
            prev_p = Some(fr.p);
        }
        let mean = |k: usize| if cnt[k] > 0 { sums[k] / cnt[k] as f32 } else { 0.0 };
        for (k, v) in x.iter_mut().take(8).enumerate() {
            *v = mean(k);
        }
        let frac = |k: usize| cnt[k] as f32 / w.mean[k] as f32;
        x[8] = frac(3);
        x[9] = frac(5);
        x[10] = frac(7);
        // Medians.
        for (j, &wj) in w.med.iter().enumerate() {
            let n = wj.min(avail);
            let mut k = 0;
            for i in 0..n {
                let fr = self.at(age + i);
                if fr.voiced
                    && let Some(slot) = self.scratch.get_mut(k)
                {
                    *slot = (fr.p - r) * 0.01;
                    k += 1;
                }
            }
            x[11 + j] = if k > 0 { self.scratch.get_mut(..k).map_or(0.0, median) } else { 0.0 };
        }
        x[16] = if mn25.is_finite() { mn25 } else { 0.0 };
        x[17] = if mx25.is_finite() { mx25 } else { 0.0 };
        x[18] = if mn100.is_finite() { mn100 } else { 0.0 };
        x[19] = if mx100.is_finite() { mx100 } else { 0.0 };
        let sd = |k: usize| {
            if cnt[k] > 1 {
                let m = sums[k] / cnt[k] as f32;
                (sq[k] / cnt[k] as f32 - m * m).max(0.0).sqrt()
            } else {
                0.0
            }
        };
        x[20] = sd(3);
        x[21] = sd(5);
        let older = |a: usize, b: usize| {
            let n = cnt[b] - cnt[a];
            if n > 0 { (sums[b] - sums[a]) / n as f32 } else { 0.0 }
        };
        x[22] = older(4, 5);
        x[23] = older(5, 6);
        x[24] = older(6, 7);
        x[25] = ls_slope(&ls12);
        x[26] = ls_slope(&ls40);
        x[27] = ls_slope(&lso);
        x[28] = (f0.p_s - r) * 0.01;
        x[29] = (f0.p_c - r) * 0.01;
        x[30] = (f0.centre - r) * 0.01;
        let seg_ms = if s0.seg == u32::MAX { 0.0 } else { s0.seg as f32 * w.hop_ms };
        x[31] = (1.0 + seg_ms).ln() / 7.0;
        x[32] = seg_ms.min(500.0) / 500.0;
        x[33] = (1.0 + s0.since_unv as f32 * w.hop_ms).ln() / 7.0;
        // Level cues over all frames (voiced or not), within what exists.
        let lv_avail = self.len - age;
        let lv = if f0.level.is_finite() { f0.level.max(-120.0) } else { -120.0 };
        let mut lmax = lv;
        let mut lmin = lv;
        for i in 0..w.h100.min(lv_avail) {
            let l = self.at(age + i).level;
            if l.is_finite() {
                lmax = lmax.max(l);
                if i < w.h60 {
                    lmin = lmin.min(l);
                }
            }
        }
        let l25 = if w.h25 < lv_avail { self.at(age + w.h25).level } else { lv };
        let l25 = if l25.is_finite() { l25.max(-120.0) } else { -120.0 };
        x[34] = ((lv + 40.0) / 20.0).clamp(-4.0, 4.0);
        x[35] = ((lv - lmax) / 10.0).clamp(-6.0, 0.0);
        x[36] = ((lv - l25) / 10.0).clamp(-6.0, 6.0);
        x[37] = ((lv - lmin) / 10.0).clamp(0.0, 6.0);
        x[38] = f0.conf;
        x[39] = f0.w;
        x[40] = f0.ext * 0.01;
        x[41] = f0.rate * 0.1;
        let n50 = w.h50.min(lv_avail);
        let v50 = (0..n50).filter(|&i| self.at(age + i).voiced).count();
        x[42] = v50 as f32 / w.h50 as f32;
        x[43] = jmp;
        x[44] = (1.0 + tj.unwrap_or(wmax) as f32 * w.hop_ms).ln() / 7.0;
        self.fut = 0;
        if use_future && age > 0 {
            self.fut = self.lookahead(age, r, lv, &mut x);
        }
        for v in x.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        self.feats = x;
        true
    }

    /// The 13 lookahead features of the frame `age` hops back: statistics of
    /// the newer frames up to the next onset. Returns the hops used.
    fn lookahead(&mut self, age: usize, r: f32, lv: f32, x: &mut [f32; NF_MAX]) -> usize {
        let w = self.win;
        let fw = [w.h3, w.h6, w.h12, w.h25];
        let mut fs = [0.0f32; 4];
        let mut fc = [0usize; 4];
        let mut all_s = 0.0;
        let mut all_c = 0;
        let (mut fmn, mut fmx) = (f32::INFINITY, f32::NEG_INFINITY);
        let mut flast = 0.0;
        let mut ls: Ls = [0.0; 5];
        let mut k = 0;
        let mut n = 0;
        for j in 1..=age {
            let fr = self.at(age - j);
            if fr.onset {
                break;
            }
            n = j;
            if !fr.voiced {
                continue;
            }
            let d = (fr.p - r) * 0.01;
            for q in 0..4 {
                if j <= fw[q] {
                    fs[q] += d;
                    fc[q] += 1;
                }
            }
            all_s += d;
            all_c += 1;
            fmn = fmn.min(d);
            fmx = fmx.max(d);
            flast = d;
            ls_add(&mut ls, j as f32 * w.hop_ms, d);
            if let Some(slot) = self.scratch.get_mut(k) {
                *slot = d;
                k += 1;
            }
        }
        for q in 0..4 {
            x[NF_PAST + q] = if fc[q] > 0 { fs[q] / fc[q] as f32 } else { 0.0 };
        }
        x[NF_PAST + 4] = if all_c > 0 { all_s / all_c as f32 } else { 0.0 };
        x[NF_PAST + 5] = if k > 0 { self.scratch.get_mut(..k).map_or(0.0, median) } else { 0.0 };
        x[NF_PAST + 6] = if fmn.is_finite() { fmn } else { 0.0 };
        x[NF_PAST + 7] = if fmx.is_finite() { fmx } else { 0.0 };
        x[NF_PAST + 8] = all_c as f32 / age as f32;
        x[NF_PAST + 9] = flast;
        x[NF_PAST + 10] = ls_slope(&ls);
        let ln = self.at(age - n.max(1).min(age)).level;
        x[NF_PAST + 11] = if ln.is_finite() { ((ln.max(-120.0) - lv) / 10.0).clamp(-6.0, 6.0) } else { 0.0 };
        x[NF_PAST + 12] = n as f32 * w.hop_ms / 50.0;
        n
    }
}

/// One candidate's feature vector (see [`NFC_LIVE`] / [`NFC_STUDIO`]): the
/// shared features, then the decided pitch relative to the candidate `c`
/// (`r_rel` is r relative to the note grid, where note n sits at 100·n).
pub(crate) fn candidate(feats: &[f32; NF_MAX], studio: bool, r_rel: f32, c: usize, targets: &TargetSet, phi: &mut [f32; NFC_STUDIO]) {
    let nsh = if studio { NF_MAX } else { NF_PAST };
    phi[..nsh].copy_from_slice(&feats[..nsh]);
    let dc = (r_rel - 100.0 * c as f32) * 0.01;
    phi[nsh] = dc;
    phi[nsh + 1] = dc.abs();
    for (q, &fi) in REL_IDX.iter().enumerate() {
        let v = feats[fi] + dc;
        phi[nsh + 2 + q] = v;
        if q < 12 {
            phi[nsh + 2 + 21 + q] = v.abs();
        }
    }
    phi[nsh + 35] = targets.gap(c, -1) * 0.01;
    phi[nsh + 36] = targets.gap(c, 1) * 0.01;
    if studio {
        for (q, &fi) in REL_FUT_IDX.iter().enumerate() {
            phi[nsh + 37 + q] = feats[fi] + dc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOP_MS: f32 = 64.0 / 44.1;

    fn voiced(p: f32, level: f32, onset: bool) -> HFrame {
        HFrame { voiced: true, onset, p, p_s: p, p_c: p, centre: p, level, conf: 0.9, ..HFrame::default() }
    }

    #[test]
    fn unvoiced_or_missing_frames_have_no_features() {
        let mut f = Features::new(HOP_MS);
        assert!(!f.compute(0, false));
        f.push(HFrame::default());
        assert!(!f.compute(0, false));
        f.push(voiced(6000.0, -20.0, true));
        assert!(f.compute(0, false));
        assert!(!f.compute(5, false));
    }

    /// A steady note: every pitch feature is 0 relative to r, the segment
    /// clock counts, and the features ignore where the note sits on the grid.
    #[test]
    fn features_are_relative_to_the_decided_pitch() {
        let mut a = Features::new(HOP_MS);
        let mut b = Features::new(HOP_MS);
        for i in 0..200 {
            a.push(voiced(6000.0, -20.0, i == 0));
            b.push(voiced(6037.0, -20.0, i == 0));
        }
        assert!(a.compute(0, false) && b.compute(0, false));
        assert_eq!(a.feats, b.feats);
        assert!(a.feats[..8].iter().all(|v| *v == 0.0));
        assert!((a.feats[32] - (199.0 * HOP_MS).min(500.0) / 500.0).abs() < 1e-6);
        assert!(a.feats.iter().all(|v| v.is_finite()));
    }

    /// A rising line: positive slopes, past means below r; with lookahead,
    /// future means above r.
    #[test]
    fn slopes_and_lookahead_see_the_movement() {
        let mut f = Features::new(HOP_MS);
        for i in 0..120 {
            f.push(voiced(6000.0 + 2.0 * i as f32, -20.0, i == 0));
        }
        assert!(f.compute(10, true));
        assert!(f.feats[25] > 0.0 && f.feats[26] > 0.0);
        assert!(f.feats[3] < 0.0);
        assert!(f.feats[NF_PAST + 4] > 0.0 && f.feats[NF_PAST + 10] > 0.0);
        assert!((f.feats[NF_PAST + 12] - 10.0 * HOP_MS / 50.0).abs() < 1e-6);
        // Without lookahead the future features stay 0.
        assert!(f.compute(10, false));
        assert!(f.feats[NF_PAST..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn non_finite_input_gives_finite_features() {
        let mut f = Features::new(HOP_MS);
        for i in 0..50 {
            let mut h = voiced(6000.0 + i as f32, if i % 3 == 0 { f32::NAN } else { f32::NEG_INFINITY }, i == 0);
            h.conf = f32::NAN;
            h.ext = f32::INFINITY;
            f.push(h);
        }
        assert!(f.compute(0, true));
        assert!(f.feats.iter().all(|v| v.is_finite()));
    }
}
