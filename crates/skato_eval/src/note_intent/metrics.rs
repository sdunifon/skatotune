//! Note-intent metrics of one replayed run on one vocadito file, poolable
//! across files (counts and value lists).
//!
//! Time base: control hop k is scored at the input time of the content it is
//! applied to ([`Run::t_mid`]), i.e. what the listener hears. Ground truth:
//! the nearest legal note to an annotated note's pitch on the A440 grid (the
//! vocadito evaluation's rule); "tuned" ground truth first removes the file's
//! singer offset.
//!
//! | metric | definition |
//! |---|---|
//! | `acc_agree` | frame accuracy: frames where annotators 1 and 2 cover and agree, whose target is that note |
//! | `note_dom` | per-note accuracy: A1 notes whose most frequent target is the A1 note |
//! | `warble_min` | target flips A → B → A with B < 40 ms (flips within 60 ms joined), per minute |
//! | `p95` | stable-frame p95 of the proxy output's distance to the nearest legal note (the vocadito `correction_pitch` on input + shift) |
//! | `pub_agree` | note agreement of the proxy output (median pitch per note) |
//! | `glitch_tgt` | onset-glitch proxy: the first 100 ms of a settle onset has ≥ 5 ms of voiced control whose target is > 150 c from both the input and the annotation |
//! | `lat_leg_med`, `lat_leg_p90` | legato decision latency: annotated onset (gap < 100 ms) → target on the A1 note for 20 ms, ms |
//! | `lat_missed` | notes the target never reaches within 500 ms, % |

use super::data::{FileCtx, legal_note};
use super::replay::Run;
use crate::vocadito::measure::{self, CorrCounts, TimedTrack};

/// Linear-interpolated percentile of finite values (NaN if none).
pub fn percentile(vals: &[f64], p: f64) -> f64 {
    let mut v: Vec<f64> = vals.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let rank = (p.clamp(0.0, 100.0) / 100.0) * (v.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    let f = rank - lo as f64;
    v[lo] * (1.0 - f) + v[hi] * f
}

pub fn median(vals: &[f64]) -> f64 {
    percentile(vals, 50.0)
}

pub fn pct(num: u64, den: u64) -> f64 {
    if den == 0 { f64::NAN } else { 100.0 * num as f64 / den as f64 }
}

#[derive(Clone, Debug, Default)]
pub struct NoteMetrics {
    pub files: u64,
    pub dur_s: f64,
    // Frame accuracy.
    pub f_a1: u64,
    pub ok_a1: u64,
    pub none_a1: u64,
    pub f_a1_int: u64,
    pub ok_a1_int: u64,
    pub f_a2: u64,
    pub ok_a2: u64,
    pub f_agree: u64,
    pub ok_agree: u64,
    pub ok_tuned: u64,
    // Per note (A1).
    pub notes: u64,
    pub notes_dom_ok: u64,
    pub notes_pure: u64,
    // Inter-annotator ceiling.
    pub f_a1_cov2: u64,
    pub f_a1_same2: u64,
    pub notes_matched2: u64,
    pub notes_same2: u64,
    // Warble and switching.
    pub warble_episodes: u64,
    pub warble_flips: u64,
    pub warble_in_notes: u64,
    pub switches: u64,
    pub switches_in_notes: u64,
    /// Consecutive A1 notes (gap < 300 ms) whose ground-truth notes differ:
    /// the switches a perfect corrector would make.
    pub ann_changes: u64,
    // Decision latency at A1 onsets (notes ≥ 100 ms), ms.
    pub lat_gap: Vec<f64>,
    pub lat_legato: Vec<f64>,
    pub lat_missed_gap: u64,
    pub lat_missed_legato: u64,
    /// The vocadito correction measures on the proxy output.
    pub corr: CorrCounts,
    // Onset-glitch proxy on the target.
    pub og_checked: u64,
    pub og_glitch: u64,
    // Learned note hold diagnostics.
    pub hops: u64,
    pub hold_proposals: u64,
    pub hold_vetoed: u64,
    pub hold_frozen: u64,
}

impl NoteMetrics {
    pub fn merge(&mut self, o: &NoteMetrics) {
        self.files += o.files;
        self.dur_s += o.dur_s;
        self.f_a1 += o.f_a1;
        self.ok_a1 += o.ok_a1;
        self.none_a1 += o.none_a1;
        self.f_a1_int += o.f_a1_int;
        self.ok_a1_int += o.ok_a1_int;
        self.f_a2 += o.f_a2;
        self.ok_a2 += o.ok_a2;
        self.f_agree += o.f_agree;
        self.ok_agree += o.ok_agree;
        self.ok_tuned += o.ok_tuned;
        self.notes += o.notes;
        self.notes_dom_ok += o.notes_dom_ok;
        self.notes_pure += o.notes_pure;
        self.f_a1_cov2 += o.f_a1_cov2;
        self.f_a1_same2 += o.f_a1_same2;
        self.notes_matched2 += o.notes_matched2;
        self.notes_same2 += o.notes_same2;
        self.warble_episodes += o.warble_episodes;
        self.warble_flips += o.warble_flips;
        self.warble_in_notes += o.warble_in_notes;
        self.switches += o.switches;
        self.switches_in_notes += o.switches_in_notes;
        self.ann_changes += o.ann_changes;
        self.lat_gap.extend_from_slice(&o.lat_gap);
        self.lat_legato.extend_from_slice(&o.lat_legato);
        self.lat_missed_gap += o.lat_missed_gap;
        self.lat_missed_legato += o.lat_missed_legato;
        self.corr.merge(&o.corr);
        self.og_checked += o.og_checked;
        self.og_glitch += o.og_glitch;
        self.hops += o.hops;
        self.hold_proposals += o.hold_proposals;
        self.hold_vetoed += o.hold_vetoed;
        self.hold_frozen += o.hold_frozen;
    }

    pub fn minutes(&self) -> f64 {
        self.dur_s / 60.0
    }

    /// Named scalar metrics (tables, CSVs, the frontier).
    pub fn scalars(&self) -> Vec<(&'static str, f64)> {
        let m = self.minutes().max(1e-9);
        let lat_all: Vec<f64> = self.lat_gap.iter().chain(self.lat_legato.iter()).copied().collect();
        let reached = (self.lat_gap.len() + self.lat_legato.len()) as u64;
        let missed = self.lat_missed_gap + self.lat_missed_legato;
        vec![
            ("acc_a1", pct(self.ok_a1, self.f_a1)),
            ("acc_a1_int", pct(self.ok_a1_int, self.f_a1_int)),
            ("acc_a2", pct(self.ok_a2, self.f_a2)),
            ("acc_agree", pct(self.ok_agree, self.f_agree)),
            ("acc_tuned", pct(self.ok_tuned, self.f_a1)),
            ("none_a1", pct(self.none_a1, self.f_a1)),
            ("note_dom", pct(self.notes_dom_ok, self.notes)),
            ("note_pure", pct(self.notes_pure, self.notes)),
            ("ceil_frames", pct(self.f_a1_same2, self.f_a1_cov2)),
            ("ceil_notes", pct(self.notes_same2, self.notes_matched2)),
            ("warble_min", self.warble_episodes as f64 / m),
            ("flips_min", self.warble_flips as f64 / m),
            ("warble_innote_min", self.warble_in_notes as f64 / m),
            ("switch_min", self.switches as f64 / m),
            ("ann_change_min", self.ann_changes as f64 / m),
            ("excess_switch_min", (self.switches as f64 - self.ann_changes as f64) / m),
            ("switch_ratio", self.switches as f64 / (self.ann_changes as f64).max(1.0)),
            ("lat_med", median(&lat_all)),
            ("lat_p90", percentile(&lat_all, 90.0)),
            ("lat_gap_med", median(&self.lat_gap)),
            ("lat_leg_med", median(&self.lat_legato)),
            ("lat_leg_p90", percentile(&self.lat_legato, 90.0)),
            ("lat_missed", pct(missed, reached + missed)),
            ("p50", self.corr.out.percentile(50.0)),
            ("p95", self.corr.out.percentile(95.0)),
            ("pub_agree", self.corr.agree_out_pct()),
            ("settle_med", self.corr.settle_median()),
            ("settle_p90", self.corr.settle_p90()),
            ("glitch_pub", self.corr.onset_glitch_pct()),
            ("glitch_tgt", pct(self.og_glitch, self.og_checked)),
            ("warble_pub", self.corr.flickers as f64),
            ("warble_pub_min", self.corr.flickers as f64 / m),
            ("minutes", self.minutes()),
            ("hold_proposals_pct", pct(self.hold_proposals, self.hops)),
            ("hold_vetoed_pct", pct(self.hold_vetoed, self.hold_proposals)),
            ("hold_frozen", self.hold_frozen as f64),
        ]
    }

    /// One scalar by name (NaN for an unknown name).
    pub fn get(&self, name: &str) -> f64 {
        self.scalars().into_iter().find(|(n, _)| *n == name).map_or(f64::NAN, |x| x.1)
    }
}

/// Score one run of one file.
pub fn score(file: &FileCtx, run: &Run) -> NoteMetrics {
    let legal = &run.legal;
    let mut m = NoteMetrics {
        files: 1,
        dur_s: file.dur_s,
        hops: run.recs.len() as u64,
        hold_proposals: run.hold.proposals,
        hold_vetoed: run.hold.vetoed,
        hold_frozen: run.hold.frozen,
        ..NoteMetrics::default()
    };
    let n = run.recs.len();
    let gt1: Vec<Option<i32>> = file.a1.iter().map(|x| legal_note(x.cents(), legal)).collect();
    let gt2: Vec<Option<i32>> = file.a2.iter().map(|x| legal_note(x.cents(), legal)).collect();
    let gtt: Vec<Option<i32>> = file.a1.iter().map(|x| legal_note(x.cents() - file.offset, legal)).collect();
    let mut per_note: Vec<Vec<Option<i32>>> = vec![Vec::new(); file.a1.len()];
    let in_range = |t: f64| t >= 0.0 && t < file.dur_s;
    // Frames.
    for k in 0..n {
        let t = run.t_mid(k);
        if !in_range(t) {
            continue;
        }
        let note = run.recs[k].note;
        let Some(i1) = file.a1_at(t) else { continue };
        let Some(g1) = gt1[i1] else { continue };
        per_note[i1].push(note);
        m.f_a1 += 1;
        m.ok_a1 += u64::from(note == Some(g1));
        m.none_a1 += u64::from(note.is_none());
        m.ok_tuned += u64::from(note.is_some() && note == gtt[i1]);
        let nt = &file.a1[i1];
        if t >= nt.start_s + 0.03 && t < nt.end_s() - 0.03 {
            m.f_a1_int += 1;
            m.ok_a1_int += u64::from(note == Some(g1));
        }
        if let Some(i2) = file.a2_at(t)
            && let Some(g2) = gt2[i2]
        {
            m.f_a1_cov2 += 1;
            m.f_a1_same2 += u64::from(g1 == g2);
            if g1 == g2 {
                m.f_agree += 1;
                m.ok_agree += u64::from(note == Some(g1));
            }
        }
    }
    for k in 0..n {
        let t = run.t_mid(k);
        if !in_range(t) {
            continue;
        }
        if let Some(i2) = file.a2_at(t)
            && let Some(g2) = gt2[i2]
        {
            m.f_a2 += 1;
            m.ok_a2 += u64::from(run.recs[k].note == Some(g2));
        }
    }
    // Per note.
    for (i, v) in per_note.iter().enumerate() {
        if v.is_empty() {
            continue;
        }
        let Some(g) = gt1[i] else { continue };
        m.notes += 1;
        let mut counts: Vec<(Option<i32>, usize)> = Vec::new();
        for x in v {
            match counts.iter_mut().find(|(n, _)| n == x) {
                Some(c) => c.1 += 1,
                None => counts.push((*x, 1)),
            }
        }
        let dom = counts.iter().max_by_key(|c| c.1).map(|c| c.0).unwrap_or(None);
        m.notes_dom_ok += u64::from(dom == Some(g));
        let ok = v.iter().filter(|x| **x == Some(g)).count();
        m.notes_pure += u64::from(ok * 10 >= v.len() * 9);
        // Note-level inter-annotator agreement: the best-overlapping A2 note.
        let a = &file.a1[i];
        let best = file
            .a2
            .iter()
            .enumerate()
            .map(|(j, b)| (j, a.end_s().min(b.end_s()) - a.start_s.max(b.start_s)))
            .filter(|(_, ov)| *ov > 0.5 * a.dur_s.min(0.1))
            .max_by(|x, y| x.1.total_cmp(&y.1));
        if let Some((j, _)) = best {
            m.notes_matched2 += 1;
            m.notes_same2 += u64::from(gt2[j] == Some(g));
        }
    }
    // Warble and switches on the content-time target series.
    let eps = warble_episodes(run);
    m.warble_episodes = eps.len() as u64;
    m.warble_flips = eps.iter().map(|e| e.flips).sum();
    m.warble_in_notes = eps.iter().filter(|e| file.a1_at(e.t0).is_some()).count() as u64;
    let mut prev: Option<i32> = None;
    for k in 0..n {
        let t = run.t_mid(k);
        if !in_range(t) {
            continue;
        }
        let cur = if run.target_voiced(k) { run.recs[k].note } else { None };
        if let (Some(a), Some(b)) = (prev, cur)
            && a != b
        {
            m.switches += 1;
            m.switches_in_notes += u64::from(file.a1_at(t).is_some());
        }
        if cur.is_some() {
            prev = cur;
        }
    }
    for i in 1..file.a1.len() {
        if file.a1[i].start_s - file.a1[i - 1].end_s() < 0.3 && gt1[i] != gt1[i - 1] {
            m.ann_changes += 1;
        }
    }
    // Decision latency.
    let hold = 0.02;
    for (i, nt) in file.a1.iter().enumerate() {
        let Some(g) = gt1[i] else { continue };
        if nt.dur_s < 0.1 {
            continue;
        }
        let gap = if i == 0 { f64::INFINITY } else { nt.start_s - file.a1[i - 1].end_s() };
        let is_gap = gap >= 0.1;
        let t_end = nt.end_s().min(nt.start_s + 0.5);
        let Some(k0) = run.k_at(nt.start_s) else { continue };
        let mut found = None;
        let mut k = k0;
        while k < n && run.t_mid(k) < t_end - hold {
            if run.recs[k].note == Some(g) {
                let mut j = k;
                let mut held = true;
                while j < n && run.t_mid(j) <= run.t_mid(k) + hold {
                    if run.recs[j].note != Some(g) {
                        held = false;
                        break;
                    }
                    j += 1;
                }
                if held {
                    found = Some(k);
                    break;
                }
                k = j.max(k + 1);
                continue;
            }
            k += 1;
        }
        match (found, is_gap) {
            (Some(k), true) => m.lat_gap.push(((run.t_mid(k) - nt.start_s) * 1000.0).max(0.0)),
            (Some(k), false) => m.lat_legato.push(((run.t_mid(k) - nt.start_s) * 1000.0).max(0.0)),
            (None, true) => m.lat_missed_gap += 1,
            (None, false) => m.lat_missed_legato += 1,
        }
    }
    // The vocadito correction measures on the proxy track.
    let proxy = run.proxy_track(&file.in_track);
    let tel = run.telemetry();
    let targets = TimedTrack::targets_from_telemetry(&tel, run.h.latency, f64::from(run.h.sample_rate));
    let (mut corr, _) = measure::correction_pitch(&file.input, &proxy, &targets, legal);
    corr.dur_s = file.dur_s;
    m.corr = corr;
    // Onset-glitch proxy on the target.
    let min_bad = 0.005;
    for &(t0, _) in &file.input.settle_onsets {
        m.og_checked += 1;
        let mut bad = 0.0;
        let Some(k0) = run.k_at(t0) else { continue };
        let mut k = k0;
        while k < n && run.t_mid(k) < t0 + 0.1 {
            let t = run.t_mid(k);
            let kin = file.input.frame_at(t);
            let in_c = file.input.in_c.get(kin).copied().unwrap_or(f64::NAN);
            let ann_v = file.input.ann_v.get(kin).copied().unwrap_or(false);
            // A voiced control whose target is > 150 c from the input (an
            // unvoiced control, or a voiced one without a target, leaves the
            // input uncorrected: no glitch).
            if ann_v && in_c.is_finite() && run.recs[k].voiced {
                let tgt = run.recs[k].note.filter(|_| run.target_voiced(k));
                if tgt.is_some_and(|nn| {
                    let ann_c = file.input.ann_c.get(kin).copied().unwrap_or(f64::NAN);
                    (in_c - ann_c).abs() <= 150.0
                        && (100.0 * f64::from(nn) - in_c).abs() > 150.0
                        && (100.0 * f64::from(nn) - ann_c).abs() > 150.0
                }) {
                    bad += run.hop_s();
                }
            }
            k += 1;
        }
        m.og_glitch += u64::from(bad >= min_bad);
    }
    m
}

/// A warble episode: target flips A → B → A with B shorter than 40 ms,
/// flips within 60 ms of each other joined (the vocadito evaluation's rule).
#[derive(Clone, Copy, Debug)]
pub struct Episode {
    pub t0: f64,
    pub t1: f64,
    pub flips: u64,
}

pub fn warble_episodes(run: &Run) -> Vec<Episode> {
    // Runs (start, end, note or None).
    let mut runs: Vec<(f64, f64, Option<i32>)> = Vec::new();
    for k in 0..run.recs.len() {
        let t = run.t_mid(k);
        let v = if run.target_voiced(k) { run.recs[k].note } else { None };
        match runs.last_mut() {
            Some(last) if v.is_some() && last.2 == v => last.1 = t,
            _ => runs.push((t, t, v)),
        }
    }
    let step = run.hop_s();
    let mut out: Vec<Episode> = Vec::new();
    let mut cur: Option<Episode> = None;
    for w in runs.windows(3) {
        let (p, mm, q) = (w[0], w[1], w[2]);
        if let (Some(pa), Some(mb), Some(qa)) = (p.2, mm.2, q.2)
            && pa == qa
            && mb != pa
            && mm.1 - mm.0 < 0.04
        {
            let end = mm.1 + step;
            match cur.as_mut() {
                Some(e) if mm.0 - e.t1 <= 0.06 => {
                    e.t1 = end;
                    e.flips += 1;
                }
                _ => {
                    if let Some(e) = cur.take() {
                        out.push(e);
                    }
                    cur = Some(Episode { t0: mm.0, t1: end, flips: 1 });
                }
            }
        }
    }
    if let Some(e) = cur.take() {
        out.push(e);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_interpolate_and_skip_nan() {
        assert!(percentile(&[], 50.0).is_nan());
        assert_eq!(percentile(&[1.0, f64::NAN, 3.0], 50.0), 2.0);
        assert_eq!(percentile(&[4.0, 1.0, 2.0, 3.0], 100.0), 4.0);
        assert!((percentile(&[0.0, 10.0], 90.0) - 9.0).abs() < 1e-12);
        assert!(pct(1, 0).is_nan());
    }
}
