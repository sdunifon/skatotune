//! Pitch HMM (pYIN's MonoPitch model, R§8.6), causal forward filter with
//! optional fixed-lag smoothing (Studio).
//!
//! States: `n` pitch bins of 20/10/5 cents (Eco/HQ/Max) over the voice range,
//! each in a voiced and an unvoiced layer (the unvoiced layer keeps the pitch
//! memory through short gaps, as in pYIN).
//!
//! Transitions:
//! * pitch: triangular within ±D bins, normalised per source bin, plus a
//!   uniform jump floor ε so real leaps stay reachable. R§8.6 rescales pYIN's
//!   maximum rate (35.92 oct/s) to the hop, `D = ceil(35.92·1200·hop_s/bin_c)`,
//!   but at a 1.33 ms hop that is ±57 c, and because pYIN observations are
//!   *zero* between candidates a 2-semitone note change could then only pass
//!   through ε (measured: ≈ 13 hops, 17 ms of extra lag). We therefore keep
//!   pYIN's own per-hop reach: `D·bin_c = max(35.92·1200·max(hop_s, 23.2 ms))
//!   = 1000 c`. Note changes up to a minor seventh are free (the triangle
//!   still prefers small moves); octave relatives (1200 c) and distant
//!   formant candidates still need the jump floor, which is what suppresses
//!   octave errors;
//! * voicing: switch probability `p = 1 − (1 − p_ref)^(hop/23.2 ms)`, i.e.
//!   librosa's 0.01 per 512/22050 s hop rescaled to ours (R§8.6), with
//!   `p_ref` following the Tracking knob. ε is rescaled the same way.
//!
//! Observations: candidate probabilities spread over ±1 bin (voiced layer);
//! the unvoiced layer gets `(1 − vp)/n` per bin, exactly pYIN's scheme.
//!
//! The triangular convolution is done with two running box sums (f64) over
//! zero-padded copies, so a step costs O(n + D) whatever D is, with no
//! branches in the loops; the voiced and unvoiced layers are convolved in the
//! same pass (two independent accumulation chains, which halves the
//! latency-bound cost). Per-source normalisation `1/Z_i` handles the
//! truncated triangles at the range edges exactly.
//!
//! Fixed lag: the last L+1 forward posteriors and observations are kept; each
//! hop a backward pass over them yields the smoothed posterior L frames back
//! (R§8.6 "Studio (fixed-lag)").

/// Most observation entries per frame (candidates × 3 bins).
pub(crate) const MAX_OBS: usize = 48;
/// librosa's pYIN hop (512 / 22050 s) to which its probabilities refer.
const PYIN_HOP_S: f32 = 512.0 / 22_050.0;
/// pYIN maximum pitch transition rate, octaves per second.
const MAX_RATE_OCT_PER_S: f32 = 35.92;
/// Jump floor per pYIN hop (R§8.6 "about 1e-4").
const JUMP_FLOOR_REF: f32 = 1e-4;
/// Probabilities below this are flushed to 0 (keeps subnormals out of the
/// recursion on x86).
const FLUSH: f32 = 1e-30;

/// One sparse observation entry for the voiced layer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ObsEntry {
    pub bin: u32,
    pub p: f32,
}

/// The HMM. All buffers are sized for `max_bins` at construction.
pub(crate) struct PitchHmm {
    max_bins: usize,
    n: usize,
    bin_cents: f32,
    c0: f32,
    d: usize,
    jump: f32,
    p_switch: f32,
    inv_z: Box<[f32]>,
    av: Box<[f32]>,
    au: Box<[f32]>,
    tv: Box<[f32]>,
    tu: Box<[f32]>,
    tri: TriScratch,
    bv: Box<[f32]>,
    // Fixed-lag history (ring of `lag + 1` frames).
    lag: usize,
    hist_av: Box<[f32]>,
    hist_au: Box<[f32]>,
    hist_obs: Box<[[ObsEntry; MAX_OBS]]>,
    hist_nobs: Box<[usize]>,
    hist_vp: Box<[f32]>,
    hist_head: usize,
    hist_count: usize,
    bu_v: Box<[f32]>,
    bu_u: Box<[f32]>,
    gamma_v: Box<[f32]>,
    /// Whether [`PitchHmm::posterior`] returns the smoothed posterior.
    smoothed: bool,
}

impl PitchHmm {
    /// HMM able to hold `max_bins` bins and a fixed lag of `lag` frames.
    pub(crate) fn new(max_bins: usize, lag: usize) -> Self {
        let max_bins = max_bins.max(4);
        let z = |len: usize| vec![0.0f32; len].into_boxed_slice();
        let frames = lag + 1;
        let mut s = Self {
            max_bins,
            n: max_bins,
            bin_cents: 10.0,
            c0: 0.0,
            d: 1,
            jump: 1e-5,
            p_switch: 1e-3,
            inv_z: z(max_bins),
            av: z(max_bins),
            au: z(max_bins),
            tv: z(max_bins),
            tu: z(max_bins),
            tri: TriScratch::new(max_bins),
            bv: z(max_bins),
            lag,
            hist_av: z(if lag > 0 { frames * max_bins } else { 0 }),
            hist_au: z(if lag > 0 { frames * max_bins } else { 0 }),
            hist_obs: vec![[ObsEntry::default(); MAX_OBS]; frames].into_boxed_slice(),
            hist_nobs: vec![0usize; frames].into_boxed_slice(),
            hist_vp: z(frames),
            hist_head: 0,
            hist_count: 0,
            bu_v: z(max_bins),
            bu_u: z(max_bins),
            gamma_v: z(max_bins),
            smoothed: false,
        };
        s.layout(0.0, 10.0 * (max_bins - 1) as f32, 10.0, 0.00133);
        s
    }

    /// Lays out bins of `bin_cents` from `lo_cents` to `hi_cents` (capped at
    /// the allocated count) for a hop of `hop_s` seconds and resets the
    /// posterior. RT-safe (no allocation).
    pub(crate) fn layout(&mut self, lo_cents: f32, hi_cents: f32, bin_cents: f32, hop_s: f32) {
        let bin_cents = if bin_cents.is_finite() && bin_cents > 0.0 { bin_cents } else { 10.0 };
        let span =
            if hi_cents.is_finite() && lo_cents.is_finite() { (hi_cents - lo_cents).max(bin_cents) } else { 1.0 };
        self.n = ((span / bin_cents).ceil() as usize + 1).clamp(4, self.max_bins);
        self.bin_cents = bin_cents;
        self.c0 = if lo_cents.is_finite() { lo_cents } else { 0.0 };
        let hop_s = if hop_s.is_finite() && hop_s > 0.0 { hop_s } else { 0.00133 };
        let reach_cents = MAX_RATE_OCT_PER_S * 1200.0 * hop_s.max(PYIN_HOP_S);
        let d = (reach_cents / bin_cents).ceil() as usize;
        self.d = d.clamp(1, self.n - 1);
        self.jump = 1.0 - (1.0 - JUMP_FLOOR_REF).powf(hop_s / PYIN_HOP_S);
        // Per-source normaliser Z_i = Σ_j tri(j − i) over the valid j.
        let (n, d) = (self.n, self.d as i64);
        for (i, z) in self.inv_z.iter_mut().enumerate().take(n) {
            let lo = (i as i64 - d).max(0);
            let hi = (i as i64 + d).min(n as i64 - 1);
            let sum: i64 = (lo..=hi).map(|j| d + 1 - (j - i as i64).abs()).sum();
            *z = 1.0 / sum.max(1) as f32;
        }
        self.reset();
    }

    /// Sets the per-hop voicing switch probability from the reference value
    /// at pYIN's hop.
    pub(crate) fn set_switch(&mut self, p_ref: f32, hop_s: f32) {
        let p_ref = if p_ref.is_finite() { p_ref.clamp(1e-5, 0.5) } else { 0.01 };
        self.p_switch = 1.0 - (1.0 - p_ref).powf(hop_s.max(1e-5) / PYIN_HOP_S);
    }

    /// Forgets the posterior (uniform over all states) and the history.
    pub(crate) fn reset(&mut self) {
        let u = 0.5 / self.n as f32;
        self.av[..self.n].fill(u);
        self.au[..self.n].fill(u);
        self.hist_head = 0;
        self.hist_count = 0;
        self.smoothed = false;
    }

    #[cfg(test)]
    pub(crate) fn n_bins(&self) -> usize {
        self.n
    }

    /// Bin index of a pitch in cents (clamped to the range).
    pub(crate) fn bin_of(&self, cents: f32) -> u32 {
        let b = ((cents - self.c0) / self.bin_cents).round();
        if b.is_finite() { b.clamp(0.0, (self.n - 1) as f32) as u32 } else { 0 }
    }

    /// Builds the sparse voiced observation for candidates `(cents, p)`:
    /// each spread over ±1 bin with weights ¼, ½, ¼ (R§8.6). Returns the
    /// number of entries written.
    pub(crate) fn observation(&self, cands: impl Iterator<Item = (f32, f32)>, out: &mut [ObsEntry; MAX_OBS]) -> usize {
        let mut k = 0;
        for (cents, p) in cands {
            if p.is_nan() || p <= 0.0 {
                continue;
            }
            let b = self.bin_of(cents) as i64;
            for (off, w) in [(-1i64, 0.25f32), (0, 0.5), (1, 0.25)] {
                let j = b + off;
                if j < 0 || j >= self.n as i64 || k >= MAX_OBS {
                    continue;
                }
                out[k] = ObsEntry { bin: j as u32, p: p * w };
                k += 1;
            }
        }
        k
    }

    /// One forward step with the sparse voiced observation `obs` and voiced
    /// probability `vp` (unvoiced layer observation `(1 − vp)/n`). Returns the
    /// voiced posterior mass.
    pub(crate) fn forward(&mut self, obs: &[ObsEntry], vp: f32) -> f32 {
        let n = self.n;
        let vp = if vp.is_finite() { vp.clamp(0.0, 1.0) } else { 0.0 };
        // Predict both layers through the pitch transition:
        // y = (1 − ε)·tri ⊛ (x / Z) + ε·Σx / n.
        let (sv, su) = self.tri.conv2(
            &self.av[..n],
            &self.au[..n],
            Some(&self.inv_z[..n]),
            self.d,
            &mut self.tv[..n],
            &mut self.tu[..n],
        );
        let eps = self.jump;
        let (fv, fu) = ((eps as f64 * sv / n as f64) as f32, (eps as f64 * su / n as f64) as f32);
        for (tv, tu) in self.tv[..n].iter_mut().zip(self.tu[..n].iter_mut()) {
            *tv = (1.0 - eps) * *tv + fv;
            *tu = (1.0 - eps) * *tu + fu;
        }
        scatter(obs, &mut self.bv[..n]);
        let bu = (1.0 - vp) / n as f32;
        let ps = self.p_switch;
        let mut total = 0.0f64;
        for ((((av, au), &tv), &tu), &bv) in self.av[..n]
            .iter_mut()
            .zip(self.au[..n].iter_mut())
            .zip(self.tv[..n].iter())
            .zip(self.tu[..n].iter())
            .zip(self.bv[..n].iter())
        {
            *av = bv * ((1.0 - ps) * tv + ps * tu);
            *au = bu * ((1.0 - ps) * tu + ps * tv);
            total += (*av + *au) as f64;
        }
        let voiced = normalise(&mut self.av[..n], &mut self.au[..n], total);
        if self.lag > 0 {
            self.push_history(obs, vp);
        }
        voiced
    }

    /// Voiced-layer forward posterior of the newest frame.
    #[cfg(test)]
    pub(crate) fn voiced_posterior(&self) -> &[f32] {
        &self.av[..self.n]
    }

    /// Voiced-layer posterior of the frame chosen by the last
    /// [`PitchHmm::smooth`] (the forward posterior when there is no lag).
    pub(crate) fn posterior(&self) -> &[f32] {
        if self.smoothed { &self.gamma_v[..self.n] } else { &self.av[..self.n] }
    }

    /// Posterior mass of voiced bins within ±`radius` of `bin` in `post`.
    pub(crate) fn mass_near(post: &[f32], bin: u32, radius: u32) -> f32 {
        let lo = bin.saturating_sub(radius) as usize;
        let hi = (bin as usize + radius as usize + 1).min(post.len());
        post.get(lo..hi).map_or(0.0, |s| s.iter().sum())
    }

    fn push_history(&mut self, obs: &[ObsEntry], vp: f32) {
        let frames = self.lag + 1;
        let n = self.n;
        let base = self.hist_head * self.max_bins;
        if let (Some(dv), Some(du)) = (self.hist_av.get_mut(base..base + n), self.hist_au.get_mut(base..base + n)) {
            dv.copy_from_slice(&self.av[..n]);
            du.copy_from_slice(&self.au[..n]);
        }
        if let (Some(o), Some(c), Some(v)) = (
            self.hist_obs.get_mut(self.hist_head),
            self.hist_nobs.get_mut(self.hist_head),
            self.hist_vp.get_mut(self.hist_head),
        ) {
            let m = obs.len().min(MAX_OBS);
            o[..m].copy_from_slice(&obs[..m]);
            *c = m;
            *v = vp;
        }
        self.hist_head = (self.hist_head + 1) % frames;
        self.hist_count = (self.hist_count + 1).min(frames);
    }

    /// Fixed-lag smoothing: computes the voiced-layer posterior of the frame
    /// `lag` hops before the newest (fewer during start-up) given every
    /// observation up to the newest (read it with [`PitchHmm::posterior`]) and
    /// returns that frame's age in hops. Without a lag the posterior is the
    /// forward one (age 0). Cost: `age` backward steps, O(n) each.
    pub(crate) fn smooth(&mut self) -> usize {
        let n = self.n;
        self.smoothed = false;
        if self.lag == 0 || self.hist_count == 0 {
            return 0;
        }
        let frames = self.lag + 1;
        let age = self.hist_count - 1;
        // β at the newest frame = 1.
        self.bu_v[..n].fill(1.0);
        self.bu_u[..n].fill(1.0);
        let newest = (self.hist_head + frames - 1) % frames;
        for step in 0..age {
            let f = (newest + frames - step) % frames;
            // g = b_f ⊙ β_f in both layers.
            scatter(
                self.hist_obs.get(f).map_or(&[][..], |o| &o[..self.hist_nobs.get(f).copied().unwrap_or(0)]),
                &mut self.bv[..n],
            );
            let bu = (1.0 - self.hist_vp.get(f).copied().unwrap_or(0.0)) / n as f32;
            let ps = self.p_switch;
            for (((tv, tu), &bv), (&betv, &betu)) in self.tv[..n]
                .iter_mut()
                .zip(self.tu[..n].iter_mut())
                .zip(self.bv[..n].iter())
                .zip(self.bu_v[..n].iter().zip(self.bu_u[..n].iter()))
            {
                let gv = bv * betv;
                let gu = bu * betu;
                *tv = (1.0 - ps) * gv + ps * gu;
                *tu = (1.0 - ps) * gu + ps * gv;
            }
            // β_{f−1}(i) = (1 − ε)·(tri ⊛ h)(i) / Z_i + ε·Σh / n.
            let (sv, su) =
                self.tri.conv2(&self.tv[..n], &self.tu[..n], None, self.d, &mut self.bu_v[..n], &mut self.bu_u[..n]);
            let eps = self.jump;
            let (fv, fu) = ((eps as f64 * sv / n as f64) as f32, (eps as f64 * su / n as f64) as f32);
            let mut total = 0.0f64;
            for ((bv, bu), &z) in self.bu_v[..n].iter_mut().zip(self.bu_u[..n].iter_mut()).zip(self.inv_z[..n].iter()) {
                *bv = (1.0 - eps) * *bv * z + fv;
                *bu = (1.0 - eps) * *bu * z + fu;
                total += (*bv + *bu) as f64;
            }
            normalise(&mut self.bu_v[..n], &mut self.bu_u[..n], total);
        }
        let target = (newest + frames - age) % frames;
        let base = target * self.max_bins;
        let (Some(fv), Some(fu)) = (self.hist_av.get(base..base + n), self.hist_au.get(base..base + n)) else {
            return 0;
        };
        let mut total = 0.0f64;
        for (((g, &a), &b), (&au, &bu)) in self.gamma_v[..n]
            .iter_mut()
            .zip(fv.iter())
            .zip(self.bu_v[..n].iter())
            .zip(fu.iter().zip(self.bu_u[..n].iter()))
        {
            *g = a * b;
            total += (*g + au * bu) as f64;
        }
        let inv = if total > 0.0 && total.is_finite() { (1.0 / total) as f32 } else { 0.0 };
        for g in self.gamma_v[..n].iter_mut() {
            *g *= inv;
        }
        self.smoothed = true;
        age
    }
}

/// Dense voiced observation from sparse entries.
fn scatter(obs: &[ObsEntry], dense: &mut [f32]) {
    dense.fill(0.0);
    for o in obs {
        if let Some(v) = dense.get_mut(o.bin as usize) {
            *v += o.p;
        }
    }
}

/// Normalises both layers to total mass 1 (uniform restart if degenerate);
/// returns the voiced mass.
fn normalise(v: &mut [f32], u: &mut [f32], total: f64) -> f32 {
    if total.is_nan() || total <= 1e-35 || !total.is_finite() {
        let x = 0.5 / v.len().max(1) as f32;
        v.fill(x);
        u.fill(x);
        return 0.5;
    }
    let inv = (1.0 / total) as f32;
    let mut voiced = 0.0f32;
    for x in v.iter_mut() {
        *x *= inv;
        if *x < FLUSH {
            *x = 0.0;
        }
        voiced += *x;
    }
    for x in u.iter_mut() {
        *x *= inv;
        if *x < FLUSH {
            *x = 0.0;
        }
    }
    voiced.clamp(0.0, 1.0)
}

/// Scratch for [`TriScratch::conv2`]: zero-padded inputs and box sums.
struct TriScratch {
    pad_a: Box<[f64]>,
    pad_b: Box<[f64]>,
    box_a: Box<[f64]>,
    box_b: Box<[f64]>,
}

impl TriScratch {
    /// Scratch for up to `max_bins` bins (and D < `max_bins`).
    fn new(max_bins: usize) -> Self {
        let z = |len: usize| vec![0.0f64; len].into_boxed_slice();
        Self {
            pad_a: z(3 * max_bins + 2),
            pad_b: z(3 * max_bins + 2),
            box_a: z(2 * max_bins + 1),
            box_b: z(2 * max_bins + 1),
        }
    }

    /// Unnormalised triangular convolutions `y[j] = Σ_{|m|≤d} (d + 1 − |m|)·x[j + m]`
    /// (zero outside) of two inputs at once, optionally pre-scaling each input
    /// sample by `pre[j]`. Two box sums of width d + 1 compose to the
    /// triangle. Returns the sums of the raw inputs. Lengths beyond the
    /// scratch capacity are truncated (never panics).
    fn conv2(
        &mut self,
        xa: &[f32],
        xb: &[f32],
        pre: Option<&[f32]>,
        d: usize,
        ya: &mut [f32],
        yb: &mut [f32],
    ) -> (f64, f64) {
        let m = d + 1;
        let n = xa.len().min(xb.len()).min(ya.len()).min(yb.len());
        let n = n.min(self.box_a.len().saturating_sub(m)).min(self.pad_a.len().saturating_sub(2 * m));
        let (pa, pb) = (&mut self.pad_a[..n + 2 * m], &mut self.pad_b[..n + 2 * m]);
        // Layout: [m zeros][x (scaled)][m zeros].
        pa[..m].fill(0.0);
        pb[..m].fill(0.0);
        pa[m + n..].fill(0.0);
        pb[m + n..].fill(0.0);
        let (mut sa, mut sb) = (0.0f64, 0.0f64);
        for (j, ((da, db), (&a, &b))) in
            pa[m..m + n].iter_mut().zip(pb[m..m + n].iter_mut()).zip(xa.iter().zip(xb.iter())).enumerate()
        {
            sa += a as f64;
            sb += b as f64;
            let k = pre.and_then(|p| p.get(j)).copied().unwrap_or(1.0) as f64;
            *da = a as f64 * k;
            *db = b as f64 * k;
        }
        // Causal box: B[j] = Σ_{k=0..d} x[j − k] = B[j − 1] + P[m + j] − P[j], j ∈ [0, n + d].
        let (ba, bb) = (&mut self.box_a[..n + m], &mut self.box_b[..n + m]);
        let (mut acc_a, mut acc_b) = (0.0f64, 0.0f64);
        for (((oa, ob), (&ina, &inb)), (&outa, &outb)) in
            ba.iter_mut().zip(bb.iter_mut()).zip(pa[m..].iter().zip(pb[m..].iter())).zip(pa.iter().zip(pb.iter()))
        {
            acc_a += ina - outa;
            acc_b += inb - outb;
            *oa = acc_a;
            *ob = acc_b;
        }
        // Anti-causal box over B: y[j] = Σ_{k=0..d} B[j + k].
        let mut acc_a: f64 = ba[..m].iter().sum();
        let mut acc_b: f64 = bb[..m].iter().sum();
        for (((oa, ob), (&ina, &inb)), (&outa, &outb)) in ya[..n]
            .iter_mut()
            .zip(yb[..n].iter_mut())
            .zip(ba[m..].iter().zip(bb[m..].iter()))
            .zip(ba.iter().zip(bb.iter()))
        {
            *oa = acc_a.max(0.0) as f32;
            *ob = acc_b.max(0.0) as f32;
            acc_a += ina - outa;
            acc_b += inb - outb;
        }
        (sa, sb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_convolution_matches_direct() {
        let x: Vec<f32> = (0..37).map(|i| ((i * 7919) % 13) as f32 / 13.0).collect();
        let x2: Vec<f32> = x.iter().rev().copied().collect();
        let pre: Vec<f32> = (0..37).map(|i| 1.0 + 0.1 * i as f32).collect();
        let mut s = TriScratch::new(40);
        for d in [1usize, 3, 6, 12, 36] {
            let mut ya = vec![0.0f32; x.len()];
            let mut yb = vec![0.0f32; x.len()];
            let (sa, sb) = s.conv2(&x, &x2, Some(&pre), d, &mut ya, &mut yb);
            assert!((sa - x.iter().map(|&v| v as f64).sum::<f64>()).abs() < 1e-9);
            assert!((sb - sa).abs() < 1e-5);
            for j in 0..x.len() {
                let (mut wa, mut wb) = (0.0f64, 0.0f64);
                for m in -(d as i64)..=(d as i64) {
                    let k = j as i64 + m;
                    if k >= 0 && (k as usize) < x.len() {
                        let w = (d as i64 + 1 - m.abs()) as f64 * pre[k as usize] as f64;
                        wa += w * x[k as usize] as f64;
                        wb += w * x2[k as usize] as f64;
                    }
                }
                assert!((ya[j] as f64 - wa).abs() < 1e-3 * wa.max(1.0), "d={d} j={j}");
                assert!((yb[j] as f64 - wb).abs() < 1e-3 * wb.max(1.0), "d={d} j={j} (b)");
            }
        }
    }

    #[test]
    fn transitions_conserve_mass() {
        let mut h = PitchHmm::new(600, 0);
        h.layout(3000.0, 8990.0, 10.0, 64.0 / 48_000.0);
        let n = h.n_bins();
        assert_eq!(n, 600);
        // pYIN's per-hop reach: ±1000 c.
        assert!((100..=101).contains(&h.d), "D = {}", h.d);
        // Voiced layer: point masses; unvoiced layer empty; no switching.
        h.av[..n].fill(0.0);
        h.au[..n].fill(0.0);
        h.av[0] = 0.7;
        h.av[300] = 0.3;
        h.p_switch = 0.0;
        // A flat voiced observation keeps the prediction's shape.
        let obs: Vec<ObsEntry> = (0..n as u32).map(|bin| ObsEntry { bin, p: 1.0 / n as f32 }).collect();
        let mut tv = vec![0.0f32; n];
        let mut tu = vec![0.0f32; n];
        let (sv, _) = h.tri.conv2(&h.av[..n], &h.au[..n], Some(&h.inv_z[..n]), h.d, &mut tv, &mut tu);
        assert!((sv - 1.0).abs() < 1e-6);
        let s: f32 = tv.iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "mass {s}");
        // Spread is limited to ±D and peaks at the source.
        assert_eq!(tv[300 + h.d + 2], 0.0);
        assert!(tv[300] > tv[300 + h.d / 2] && tv[300 + h.d / 2] > tv[300 + h.d]);
        // Full forward step with the flat observation: mass conserved, floor added.
        let v = h.forward(&obs[..MAX_OBS.min(obs.len())], 1.0);
        assert!(v > 0.99, "{v}");
    }

    #[test]
    fn forward_prefers_continuous_path_over_octave_jump() {
        let mut h = PitchHmm::new(600, 0);
        h.layout(3000.0, 8990.0, 10.0, 64.0 / 48_000.0);
        h.set_switch(0.01, 64.0 / 48_000.0);
        let mut obs = [ObsEntry::default(); MAX_OBS];
        // Settle on 5000 c.
        for _ in 0..50 {
            let k = h.observation([(5000.0, 0.9)].into_iter(), &mut obs);
            h.forward(&obs[..k], 0.9);
        }
        // An octave-up candidate appears with *more* evidence for a few hops:
        // the HMM must stay on the continuous path.
        for _ in 0..4 {
            let k = h.observation([(5000.0, 0.35), (6200.0, 0.6)].into_iter(), &mut obs);
            let v = h.forward(&obs[..k], 0.95);
            assert!(v > 0.9);
            let post = h.voiced_posterior();
            let a = PitchHmm::mass_near(post, h.bin_of(5000.0), 2);
            let b = PitchHmm::mass_near(post, h.bin_of(6200.0), 2);
            assert!(a > b, "{a} vs {b}");
        }
        // Silence: voiced mass collapses.
        let v = h.forward(&[], 0.0);
        assert!(v < 1e-6, "{v}");
    }

    #[test]
    fn fixed_lag_smoothing_uses_future_evidence() {
        // Onset ambiguity: frame 0 slightly favours 6200 c, every later frame
        // clearly says 5000 c. The forward filter's decision on frame 0 is
        // wrong; the smoothed posterior for frame 0 must be right.
        let hop = 64.0 / 48_000.0;
        let mut h = PitchHmm::new(600, 6);
        h.layout(3000.0, 8990.0, 10.0, hop);
        h.set_switch(0.01, hop);
        let mut obs = [ObsEntry::default(); MAX_OBS];
        let k = h.observation([(6200.0, 0.5), (5000.0, 0.4)].into_iter(), &mut obs);
        h.forward(&obs[..k], 0.9);
        let f = h.voiced_posterior();
        assert!(PitchHmm::mass_near(f, h.bin_of(6200.0), 2) > PitchHmm::mass_near(f, h.bin_of(5000.0), 2));
        for _ in 0..6 {
            let k = h.observation([(5000.0, 0.9)].into_iter(), &mut obs);
            h.forward(&obs[..k], 0.9);
        }
        let (bin_a, bin_b) = (h.bin_of(5000.0), h.bin_of(6200.0));
        assert_eq!(h.smooth(), 6);
        let post = h.posterior();
        assert!(
            PitchHmm::mass_near(post, bin_a, 2) > PitchHmm::mass_near(post, bin_b, 2),
            "smoothing did not fix the onset"
        );
        // Further frames keep the window at the configured lag.
        let k = h.observation([(5000.0, 0.9)].into_iter(), &mut obs);
        h.forward(&obs[..k], 0.9);
        assert_eq!(h.smooth(), 6);
        h.reset();
        assert_eq!(h.smooth(), 0);
    }

    #[test]
    fn degenerate_input_restarts_uniform() {
        let mut h = PitchHmm::new(100, 0);
        h.layout(f32::NAN, f32::INFINITY, -3.0, f32::NAN);
        assert!(h.n_bins() >= 4);
        let v = h.forward(&[ObsEntry { bin: 5, p: f32::NAN }], f32::NAN);
        assert!(v.is_finite());
        assert!(h.voiced_posterior().iter().all(|p| p.is_finite()));
        assert_eq!(h.bin_of(f32::NAN), 0);
    }
}
