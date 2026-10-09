//! Offline, non-causal, high-accuracy reference f0 tracker (f64).
//!
//! This is the yardstick for every rendered output, so it trades speed and
//! latency for accuracy and robustness. Three stages:
//!
//! 1. **Coarse, robust.** Every hop, a long centred window (2/f_min) gives an
//!    FFT cross-correlation NSDF with fixed-length segments (both of length W,
//!    so the shrinking-window bias of plain MPM is avoided, R§8.4). Key maxima
//!    are candidates; a Viterbi pass over each voiced run picks the path with
//!    the best clarity, a small octave cost favouring higher f (Praat-style,
//!    R§8.5) and a continuity cost in cents.
//! 2. **Fine, local.** Around the chosen period, a short pitch-adaptive window
//!    (2·T, Hann-weighted) is correlated at integer lags near T, then the
//!    normalised correlation is **sinc-interpolated between lags** (Boersma
//!    1993; R§8.7: parabolic interpolation is not enough at high f0) and
//!    maximised by golden-section search. For an exactly periodic signal the
//!    peak is exactly T (Cauchy-Schwarz).
//! 3. **Kernel deconvolution.** To first order, the normalised-correlation peak
//!    is the average of the local periods P_j weighted by `c_j = w_j·x'(t_j)²`
//!    (derivative energy: expand R(τ)/√E_B(τ) around τ = P_j and set the
//!    derivative to zero). For a voice those weights cluster at the glottal
//!    closures, so the *time* a short-window estimate describes moves by up to
//!    ±0.1·T with the pulse positions, and a curved trajectory is biased by
//!    v·p''/2, where v is the variance of the effective kernel (the c-weighted
//!    spread plus the T²/12 of each period's own span). Each frame therefore
//!    reports its own centroid t_k and variance v_k, and the final value at a
//!    grid point t0 is the intercept `a` of a local least-squares fit
//!    `p_k ≈ a + b·(t_k − t0) + c·((t_k − t0)² + v_k)`: an exact second-order
//!    model of what each window measured. At 70 Hz with a 6 Hz ±50 c vibrato
//!    this removes a ~3 c timing wobble and a ~1.5 c curvature bias.
//!
//! Voicing: a frame is voiced when the coarse clarity and level pass and the
//! fine clarity is at least 0.5. Fast leaps smear the long coarse window, so
//! unvoiced gaps of ≤ 150 ms between two voiced runs are re-examined with a
//! log-interpolated period guess and a wide (±150 c) fine search, accepting
//! frames with fine clarity ≥ 0.8.
//!
//! Validated against the synthesiser's exact `f0_truth` in this module's tests
//! (median < 0.5 c, p99 < 3 c over clean vowels 70-1100 Hz, steady and with
//! vibrato, at 44.1/48/96 kHz).

use crate::dsp::{Correlator64, KaiserSinc, next_pow2};
use crate::synth::hz_to_cents64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceConfig {
    /// Lowest f0 searched, Hz.
    pub f_min: f64,
    /// Highest f0 searched, Hz.
    pub f_max: f64,
    /// Frame spacing, s.
    pub hop_s: f64,
    /// Coarse NSDF clarity needed to call a frame voiced.
    pub voicing_threshold: f64,
    /// Absolute silence floor (mean-square level of the analysis window), dBFS.
    pub silence_dbfs: f64,
    /// Frames this far below the loudest frame are silence, dB.
    pub relative_silence_db: f64,
    /// Shortest voiced run kept, s (shorter runs are dropped as spurious).
    pub min_voiced_s: f64,
    /// Model the window's curvature bias in stage 3 (otherwise only the
    /// centroid timing is corrected, with a local linear fit).
    pub curvature_correction: bool,
}

impl Default for ReferenceConfig {
    fn default() -> Self {
        Self {
            f_min: 60.0,
            f_max: 1500.0,
            hop_s: 0.001,
            voicing_threshold: 0.6,
            silence_dbfs: -75.0,
            relative_silence_db: -55.0,
            min_voiced_s: 0.02,
            curvature_correction: true,
        }
    }
}

/// One analysis frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefFrame {
    /// Sample index the frame describes.
    pub pos: usize,
    pub voiced: bool,
    /// Hz, 0 when unvoiced.
    pub f0_hz: f64,
    /// Absolute cents at `pos` after stage 3, NaN when unvoiced.
    pub cents: f64,
    /// The raw fine-stage estimate (describing time `pos` ± the centroid
    /// offset), NaN when unvoiced.
    pub cents_raw: f64,
    /// Peak normalised correlation of the fine stage (coarse stage when unvoiced).
    pub clarity: f64,
    /// Mean-square level of the coarse window, dBFS.
    pub level_dbfs: f64,
}

/// A pitch track at a fixed hop. Frame `k` sits at sample `k·hop`.
#[derive(Clone, Debug, PartialEq)]
pub struct PitchTrack {
    pub sample_rate: f64,
    pub hop: usize,
    pub frames: Vec<RefFrame>,
}

impl PitchTrack {
    /// Index of the frame nearest to sample `pos`.
    pub fn frame_index(&self, pos: usize) -> usize {
        let k = (pos + self.hop / 2) / self.hop.max(1);
        k.min(self.frames.len().saturating_sub(1))
    }

    /// Cents of the frame nearest to `pos`, if voiced.
    pub fn cents_at(&self, pos: usize) -> Option<f64> {
        self.frames
            .get(self.frame_index(pos))
            .filter(|f| f.voiced)
            .map(|f| f.cents)
    }

    pub fn voiced_count(&self) -> usize {
        self.frames.iter().filter(|f| f.voiced).count()
    }
}

/// Coarse-stage candidate: (fractional lag, NSDF value).
type Cand = (f64, f64);

/// Fine-stage measurement of one frame.
#[derive(Clone, Copy, Debug)]
struct Fine {
    /// Period in samples.
    tau: f64,
    clarity: f64,
    /// Centroid of the effective kernel, absolute samples.
    t_eff: f64,
    /// Variance of the effective kernel, samples².
    var: f64,
}

/// At most this many key maxima per coarse frame enter the Viterbi.
const MAX_CANDS: usize = 12;
/// Viterbi weights (see module docs): clarity deficit, octave cost per octave
/// of lag, continuity cost per 100 c between frames.
const W_CLARITY: f64 = 10.0;
const W_OCTAVE: f64 = 0.2;
const W_JUMP: f64 = 0.5;
/// Fine-stage search half-range around the coarse period, cents.
const FINE_SEARCH_CENTS: f64 = 60.0;
/// Longest unvoiced gap between two voiced runs that the glide rescue bridges.
const RESCUE_GAP_S: f64 = 0.15;
/// Fine search half-range for rescued frames (the interpolated guess can be
/// ~80 c off in the middle of a smooth leap).
const RESCUE_SEARCH_CENTS: f64 = 150.0;
/// Fine clarity a rescued frame needs to count as voiced.
const RESCUE_CLARITY: f64 = 0.8;
/// Fine clarity any voiced frame needs: the long coarse window can call a
/// frame voiced while the short fine window around it still sees a plosive
/// burst or a fricative (no usable period there).
const FINE_MIN_CLARITY: f64 = 0.5;

pub struct ReferenceTracker {
    cfg: ReferenceConfig,
    sr: f64,
    hop: usize,
    tau_min: usize,
    /// Longest lag accepted as a period.
    tau_search: usize,
    /// Longest lag computed: 30 % past `tau_search` so the NSDF lobe of the
    /// longest period can close (low voices have very wide lobes).
    tau_max: usize,
    coarse_w: usize,
    corr: Correlator64,
    sinc: KaiserSinc,
}

impl ReferenceTracker {
    pub fn new(sample_rate: f32, cfg: ReferenceConfig) -> Self {
        let sr = sample_rate as f64;
        let f_min = cfg.f_min.max(20.0);
        let f_max = cfg.f_max.clamp(f_min * 2.0, 0.25 * sr);
        let tau_min = ((sr / f_max).floor() as usize).max(2);
        let tau_search = (sr / f_min).ceil() as usize + 2;
        let tau_max = (1.3 * sr / f_min).ceil() as usize + 2;
        let coarse_w = (2.0 * sr / f_min).round() as usize;
        let hop = ((cfg.hop_s * sr).round() as usize).max(1);
        Self {
            cfg: ReferenceConfig {
                f_min,
                f_max,
                ..cfg
            },
            sr,
            hop,
            tau_min,
            tau_search,
            tau_max,
            coarse_w,
            corr: Correlator64::new(next_pow2(coarse_w + tau_max + 2)),
            sinc: KaiserSinc::new(12, 8.6),
        }
    }

    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Track the whole signal.
    pub fn track(&mut self, x: &[f32]) -> PitchTrack {
        let x: Vec<f64> = x
            .iter()
            .map(|v| if v.is_finite() { *v as f64 } else { 0.0 })
            .collect();
        let n_frames = if x.is_empty() {
            0
        } else {
            (x.len() - 1) / self.hop + 1
        };

        // Stage 1: coarse candidates + voicing evidence.
        let mut cands: Vec<Vec<Cand>> = Vec::with_capacity(n_frames);
        let mut best: Vec<f64> = Vec::with_capacity(n_frames);
        let mut level: Vec<f64> = Vec::with_capacity(n_frames);
        let mut prefix = vec![0.0; x.len() + 1];
        for (i, v) in x.iter().enumerate() {
            prefix[i + 1] = prefix[i] + v * v;
        }
        let span = self.coarse_w + self.tau_max;
        let mut seg = vec![0.0; span];
        let mut seg_prefix = vec![0.0; span + 1];
        let mut r = vec![0.0; self.tau_max + 1];
        let mut nsdf = vec![0.0; self.tau_max + 1];
        // Centre the evidence of a typical lag on the frame position.
        let lead = (self.coarse_w + self.tau_search / 2) / 2;
        for k in 0..n_frames {
            let s = (k * self.hop) as i64 - lead as i64;
            let lvl = window_level_db(&prefix, s, self.coarse_w + self.tau_search / 2);
            level.push(lvl);
            if lvl < self.cfg.silence_dbfs {
                cands.push(Vec::new());
                best.push(0.0);
                continue;
            }
            fill_segment(&x, s, &mut seg);
            for (j, v) in seg.iter().enumerate() {
                seg_prefix[j + 1] = seg_prefix[j] + v * v;
            }
            let (a, _) = seg.split_at(self.coarse_w);
            self.corr.cross_correlate(a, &seg, &mut r);
            let e_a = seg_prefix[self.coarse_w];
            for (tau, (nv, rv)) in nsdf.iter_mut().zip(r.iter()).enumerate() {
                let e_b = seg_prefix[tau + self.coarse_w] - seg_prefix[tau];
                let den = e_a + e_b;
                *nv = if den > 1e-30 { 2.0 * rv / den } else { 0.0 };
            }
            let cs = key_maxima(&nsdf, self.tau_min, self.tau_search);
            best.push(cs.iter().fold(0.0, |m, c| m.max(c.1)));
            cands.push(cs);
        }

        // Voicing: clarity + level, then fill one-frame holes and remove
        // spurious short runs.
        let loudest = level.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut voiced: Vec<bool> = (0..n_frames)
            .map(|k| {
                best[k] >= self.cfg.voicing_threshold
                    && level[k] >= self.cfg.silence_dbfs
                    && level[k] >= loudest + self.cfg.relative_silence_db
            })
            .collect();
        fill_holes(&mut voiced, 1);
        // A filled hole may have no candidate at all; it cannot carry a period.
        for (v, c) in voiced.iter_mut().zip(cands.iter()) {
            *v &= !c.is_empty();
        }
        let min_run = ((self.cfg.min_voiced_s * self.sr) / self.hop as f64).ceil() as usize;
        drop_short_runs(&mut voiced, min_run.max(1));

        // Viterbi per voiced run over the coarse candidates.
        let mut coarse_tau = vec![0.0f64; n_frames];
        for (start, end) in runs(&voiced) {
            self.viterbi(
                &cands[start..end],
                &best[start..end],
                &mut coarse_tau[start..end],
            );
        }

        // Stage 2: fine measurement per voiced frame.
        let mut fine: Vec<Option<Fine>> = vec![None; n_frames];
        for k in 0..n_frames {
            if voiced[k] && coarse_tau[k] > 0.0 {
                fine[k] = self
                    .refine(&x, k * self.hop, coarse_tau[k], FINE_SEARCH_CENTS)
                    .filter(|f| f.clarity >= FINE_MIN_CLARITY);
            }
        }

        // Rescue fast glides: the long coarse window smears a leap of several
        // semitones (low clarity), but the short fine window still sees clean
        // periods. Between two voiced runs, close in time and with no silence
        // in between, guess the period by log-linear interpolation and accept
        // frames the fine stage finds clearly periodic.
        let max_gap = ((RESCUE_GAP_S * self.sr) / self.hop as f64).ceil() as usize;
        let rs = runs(&voiced);
        for pair in rs.windows(2) {
            let ((_, a_end), (b_start, _)) = (pair[0], pair[1]);
            let gap = b_start - a_end;
            let loud = level[a_end..b_start].iter().all(|l| {
                *l >= self.cfg.silence_dbfs && *l >= loudest + self.cfg.relative_silence_db
            });
            let (ta, tb) = (coarse_tau[a_end - 1], coarse_tau[b_start]);
            if gap == 0 || gap > max_gap || !loud || ta <= 0.0 || tb <= 0.0 {
                continue;
            }
            for (k, slot) in fine.iter_mut().enumerate().take(b_start).skip(a_end) {
                let frac = (k + 1 - a_end) as f64 / (gap + 1) as f64;
                let guess = (ta.ln() + (tb.ln() - ta.ln()) * frac).exp();
                if let Some(f) = self.refine(&x, k * self.hop, guess, RESCUE_SEARCH_CENTS)
                    && f.clarity >= RESCUE_CLARITY
                {
                    *slot = Some(f);
                }
            }
        }

        // Stage 3: per grid point, deconvolve the window kernels.
        let mut frames: Vec<RefFrame> = (0..n_frames)
            .map(|k| RefFrame {
                pos: k * self.hop,
                voiced: false,
                f0_hz: 0.0,
                cents: f64::NAN,
                cents_raw: f64::NAN,
                clarity: best[k],
                level_dbfs: level[k],
            })
            .collect();
        let raw: Vec<f64> = fine
            .iter()
            .map(|f| f.map_or(f64::NAN, |f| hz_to_cents64(self.sr / f.tau)))
            .collect();
        let voiced_fine: Vec<bool> = fine.iter().map(|f| f.is_some()).collect();
        for (start, end) in runs(&voiced_fine) {
            for k in start..end {
                let Some(fk) = fine[k] else { continue };
                let c = self
                    .deconvolve(k, &fine[start..end], &raw[start..end], start)
                    .unwrap_or(raw[k]);
                frames[k] = RefFrame {
                    voiced: true,
                    f0_hz: 440.0 * ((c - 6900.0) / 1200.0).exp2(),
                    cents: c,
                    cents_raw: raw[k],
                    clarity: fk.clarity,
                    ..frames[k]
                };
            }
        }
        PitchTrack {
            sample_rate: self.sr,
            hop: self.hop,
            frames,
        }
    }

    /// Min-cost path through the candidates of one voiced run.
    fn viterbi(&self, cands: &[Vec<Cand>], best: &[f64], out: &mut [f64]) {
        let local = |c: &Cand, m: f64| {
            W_CLARITY * (m - c.1) + W_OCTAVE * (c.0 / self.tau_min as f64).log2()
        };
        let mut cost: Vec<Vec<f64>> = Vec::with_capacity(cands.len());
        let mut back: Vec<Vec<usize>> = Vec::with_capacity(cands.len());
        for (f, cs) in cands.iter().enumerate() {
            let m = best[f];
            let mut row = Vec::with_capacity(cs.len());
            let mut brow = Vec::with_capacity(cs.len());
            for c in cs {
                let l = local(c, m);
                let prev = if f > 0 {
                    cands.get(f - 1).zip(cost.get(f - 1))
                } else {
                    None
                };
                match prev {
                    Some((pcs, pcost)) if !pcs.is_empty() => {
                        let (mut bc, mut bi) = (f64::INFINITY, usize::MAX);
                        for (j, (pc, pv)) in pcs.iter().zip(pcost.iter()).enumerate() {
                            let jump = 12.0 * (c.0 / pc.0).log2().abs(); // units of 100 c
                            let v = pv + W_JUMP * jump;
                            if v < bc {
                                bc = v;
                                bi = j;
                            }
                        }
                        row.push(bc + l);
                        brow.push(bi);
                    }
                    _ => {
                        row.push(l);
                        brow.push(usize::MAX);
                    }
                }
            }
            cost.push(row);
            back.push(brow);
        }
        let Some(last) = cost.last() else { return };
        let mut idx = last
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i);
        for f in (0..cands.len()).rev() {
            let Some(i) = idx else { break };
            if let (Some(o), Some(c)) = (out.get_mut(f), cands[f].get(i)) {
                *o = c.0;
            }
            idx = back[f].get(i).copied().filter(|b| *b != usize::MAX);
        }
    }

    /// Fine measurement around `tau_c` (searching ±`search_cents`) for the
    /// frame at `pos`. If the best integer lag sits on the edge of the band
    /// (a poor coarse guess), the search is repeated once, centred there.
    fn refine(&self, x: &[f64], pos: usize, tau_c: f64, search_cents: f64) -> Option<Fine> {
        let first = self.refine_once(x, pos, tau_c, search_cents)?;
        if !first.1 {
            return Some(first.0);
        }
        let second = self.refine_once(x, pos, first.0.tau, search_cents)?;
        Some(if second.0.clarity >= first.0.clarity {
            second.0
        } else {
            first.0
        })
    }

    /// One fine search; the flag reports a peak on the band edge.
    fn refine_once(
        &self,
        x: &[f64],
        pos: usize,
        tau_c: f64,
        search_cents: f64,
    ) -> Option<(Fine, bool)> {
        let w = ((2.0 * tau_c).round() as usize).max(64);
        let ratio = (search_cents / 1200.0).exp2();
        let lo = ((tau_c / ratio).floor() as usize)
            .max(self.tau_min.saturating_sub(1))
            .max(2);
        let hi = ((tau_c * ratio).ceil() as usize).max(lo + 1);
        let pad = self.sinc.half_width() + 2;
        let lag0 = lo.saturating_sub(pad).max(1);
        let lag1 = hi + pad;
        let s = pos as i64 - ((w as f64 + tau_c) / 2.0).round() as i64;
        let get = |i: i64| -> f64 {
            if i < 0 {
                0.0
            } else {
                x.get(i as usize).copied().unwrap_or(0.0)
            }
        };

        let weights: Vec<f64> = (0..w)
            .map(|j| {
                (std::f64::consts::PI * (j as f64 + 0.5) / w as f64)
                    .sin()
                    .powi(2)
            })
            .collect();
        let a: Vec<f64> = (0..w).map(|j| get(s + j as i64)).collect();
        let e_a: f64 = a.iter().zip(&weights).map(|(v, wt)| wt * v * v).sum();
        if e_a <= 1e-30 {
            return None;
        }
        let b: Vec<f64> = (0..w + lag1 + 1).map(|j| get(s + j as i64)).collect();
        let nl = lag1 - lag0 + 1;
        let mut rr = vec![0.0; nl];
        let mut eb = vec![0.0; nl];
        for (i, (rv, ev)) in rr.iter_mut().zip(eb.iter_mut()).enumerate() {
            let tau = lag0 + i;
            let (mut acc_r, mut acc_e) = (0.0, 0.0);
            for ((av, wt), bv) in a.iter().zip(&weights).zip(&b[tau..tau + w]) {
                acc_r += wt * av * bv;
                acc_e += wt * bv * bv;
            }
            *rv = acc_r;
            *ev = acc_e;
        }
        let nsdf_at = |i: usize| -> f64 {
            let e = eb[i];
            if e > 1e-30 {
                rr[i] / (e_a * e).sqrt()
            } else {
                0.0
            }
        };
        let (mut ti, mut tv) = (lo - lag0, f64::NEG_INFINITY);
        for i in (lo - lag0)..=(hi - lag0) {
            let v = nsdf_at(i);
            if v > tv {
                tv = v;
                ti = i;
            }
        }
        // Continuous maximisation of the sinc-interpolated NSDF.
        let f = |t: f64| -> f64 {
            let e = self.sinc.interp(&eb, t);
            if e > 1e-30 {
                self.sinc.interp(&rr, t) / (e_a * e).sqrt()
            } else {
                0.0
            }
        };
        let on_edge = ti == lo - lag0 || ti == hi - lag0;
        let (t_best, v_best) = golden_max(&f, ti as f64 - 1.0, ti as f64 + 1.0, 64);
        let (tau, clarity) = if v_best >= tv {
            (lag0 as f64 + t_best, v_best)
        } else {
            ((lag0 + ti) as f64, tv)
        };

        // Effective kernel: weights c_j = w_j·x'(t_j)² (central difference).
        let (mut c_sum, mut c_t, mut c_tt) = (0.0, 0.0, 0.0);
        for (j, wt) in weights.iter().enumerate() {
            let t = s + j as i64;
            let d = 0.5 * (get(t + 1) - get(t - 1));
            let c = wt * d * d;
            c_sum += c;
            c_t += c * j as f64;
            c_tt += c * (j * j) as f64;
        }
        let (centre, spread) = if c_sum > 1e-30 {
            let m = c_t / c_sum;
            (m, (c_tt / c_sum - m * m).max(0.0))
        } else {
            (0.5 * (w as f64 - 1.0), (w * w) as f64 / 12.0)
        };
        Some((
            Fine {
                tau,
                clarity,
                t_eff: s as f64 + centre + 0.5 * tau,
                var: spread + tau * tau / 12.0,
            },
            on_edge,
        ))
    }

    /// Stage 3 for frame `k` (index into the whole track) using the
    /// measurements of its voiced run (`run_*` slices start at frame `start`).
    fn deconvolve(
        &self,
        k: usize,
        run_fine: &[Option<Fine>],
        run_raw: &[f64],
        start: usize,
    ) -> Option<f64> {
        let fk = run_fine.get(k - start).copied().flatten()?;
        let t0 = (k * self.hop) as f64;
        // Neighbourhood: at least ±4 ms, wide enough to span the kernel.
        let half = (1.5 * fk.var.sqrt())
            .max(0.004 * self.sr)
            .min(0.02 * self.sr);
        let reach = (half / self.hop as f64).ceil() as usize + 1;
        let lo = (k - start).saturating_sub(reach);
        let hi = (k - start + reach).min(run_fine.len() - 1);
        // Normal equations for p ≈ a + b·u + c·(u² + v), u = t_k − t0.
        let quad = self.cfg.curvature_correction;
        let dim = if quad { 3 } else { 2 };
        let mut ata = [[0.0f64; 3]; 3];
        let mut atb = [0.0f64; 3];
        let mut used = 0;
        let scale = 1.0 / half; // keep the system well conditioned
        for i in lo..=hi {
            let (Some(f), Some(p)) = (run_fine.get(i).copied().flatten(), run_raw.get(i)) else {
                continue;
            };
            let u = (f.t_eff - t0) * scale;
            if u.abs() > 1.0 {
                continue;
            }
            let row = [1.0, u, u * u + f.var * scale * scale];
            for r in 0..dim {
                for c in 0..dim {
                    ata[r][c] += row[r] * row[c];
                }
                atb[r] += row[r] * p;
            }
            used += 1;
        }
        if used < dim + 3 {
            return None;
        }
        solve(&ata, &atb, dim).map(|sol| sol[0])
    }
}

/// Solve the `dim`×`dim` symmetric system by Gaussian elimination with partial
/// pivoting; None when (near) singular.
fn solve(a: &[[f64; 3]; 3], b: &[f64; 3], dim: usize) -> Option<[f64; 3]> {
    let mut m = *a;
    let mut v = *b;
    for col in 0..dim {
        let piv = (col..dim).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        v.swap(col, piv);
        let pivot_row = m[col];
        for row in col + 1..dim {
            let f = m[row][col] / pivot_row[col];
            for (dst, src) in m[row].iter_mut().zip(pivot_row.iter()).take(dim).skip(col) {
                *dst -= f * src;
            }
            v[row] -= f * v[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..dim).rev() {
        let mut acc = v[row];
        for c in row + 1..dim {
            acc -= m[row][c] * x[c];
        }
        x[row] = acc / m[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Track `x` with the default configuration.
pub fn track(x: &[f32], sample_rate: f32) -> PitchTrack {
    ReferenceTracker::new(sample_rate, ReferenceConfig::default()).track(x)
}

/// `[start, end)` ranges of consecutive `true`s.
fn runs(v: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut k = 0;
    while k < v.len() {
        if !v[k] {
            k += 1;
            continue;
        }
        let s = k;
        while k < v.len() && v[k] {
            k += 1;
        }
        out.push((s, k));
    }
    out
}

/// Copy `x[s .. s + seg.len()]` into `seg` (zeros outside `x`).
fn fill_segment(x: &[f64], s: i64, seg: &mut [f64]) {
    for (j, v) in seg.iter_mut().enumerate() {
        let i = s + j as i64;
        *v = if i < 0 {
            0.0
        } else {
            x.get(i as usize).copied().unwrap_or(0.0)
        };
    }
}

/// Mean-square level of `x[s .. s + w]` in dB, from prefix sums of x².
fn window_level_db(prefix: &[f64], s: i64, w: usize) -> f64 {
    let n = prefix.len() as i64 - 1;
    let a = s.clamp(0, n) as usize;
    let b = (s + w as i64).clamp(0, n) as usize;
    let e = prefix[b] - prefix[a];
    10.0 * (e / w.max(1) as f64).max(1e-30).log10()
}

/// Positive-lobe maxima of `nsdf` with lag in [tau_min, tau_search], skipping
/// the zero-lag lobe, parabolically refined (R§8.4). A lobe still open at the
/// end of `nsdf` counts if its maximum is interior.
fn key_maxima(nsdf: &[f64], tau_min: usize, tau_search: usize) -> Vec<Cand> {
    let n = nsdf.len();
    let mut out = Vec::new();
    let mut t = 1;
    while t < n && nsdf[t] >= 0.0 {
        t += 1;
    }
    while t < n {
        while t < n && nsdf[t] < 0.0 {
            t += 1;
        }
        if t >= n {
            break;
        }
        let mut b = t;
        while t < n && nsdf[t] >= 0.0 {
            if nsdf[t] > nsdf[b] {
                b = t;
            }
            t += 1;
        }
        if b < tau_min || b > tau_search || b == 0 || b + 1 >= n {
            continue;
        }
        let (ym1, y0, yp1) = (nsdf[b - 1], nsdf[b], nsdf[b + 1]);
        let den = ym1 - 2.0 * y0 + yp1;
        let (off, val) = if den.abs() > 1e-15 {
            let d = (0.5 * (ym1 - yp1) / den).clamp(-1.0, 1.0);
            (d, y0 - 0.25 * (ym1 - yp1) * d)
        } else {
            (0.0, y0)
        };
        out.push((b as f64 + off, val));
    }
    let m = out.iter().fold(0.0f64, |m, c| m.max(c.1));
    out.retain(|c| c.1 >= 0.3 * m && c.1 > 0.0);
    if out.len() > MAX_CANDS {
        // A high voice has many near-equal subharmonic lobes (T, 2T, 3T, …):
        // keep the near-maximal ones shortest-lag first so T always survives,
        // then fill up with the strongest of the rest.
        let (mut keep, mut rest): (Vec<Cand>, Vec<Cand>) =
            out.into_iter().partition(|c| c.1 >= 0.9 * m);
        keep.truncate(MAX_CANDS);
        rest.sort_by(|a, b| b.1.total_cmp(&a.1));
        let room = MAX_CANDS - keep.len();
        keep.extend(rest.into_iter().take(room));
        keep.sort_by(|a, b| a.0.total_cmp(&b.0));
        out = keep;
    }
    out
}

/// Golden-section maximisation of `f` on [a, b].
fn golden_max(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64, iters: usize) -> (f64, f64) {
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = b - g * (b - a);
    let mut d = a + g * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..iters {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    let t = 0.5 * (a + b);
    (t, f(t))
}

/// Set runs of `false` no longer than `max_len` between two `true`s to `true`.
fn fill_holes(v: &mut [bool], max_len: usize) {
    let n = v.len();
    let mut i = 0;
    while i < n {
        if v[i] {
            i += 1;
            continue;
        }
        let s = i;
        while i < n && !v[i] {
            i += 1;
        }
        if s > 0 && i < n && i - s <= max_len {
            v[s..i].fill(true);
        }
    }
}

/// Clear runs of `true` shorter than `min_len`.
fn drop_short_runs(v: &mut [bool], min_len: usize) {
    for (s, e) in runs(v) {
        if e - s < min_len {
            v[s..e].fill(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::percentile;
    use crate::synth::{
        Consonant, Formants, Melody, Note, Signal, SignalBuilder, Vibrato, Voice, VoiceType, Vowel,
    };

    /// |error| in cents of every voiced reference frame at least `guard_s`
    /// inside a truth note, vs the exact synthesised f0 at the frame centre.
    fn frame_errors(sig: &Signal, track: &PitchTrack, guard_s: f64) -> (Vec<f64>, usize) {
        let guard = (guard_s * sig.sample_rate as f64) as usize;
        let mut errs = Vec::new();
        let mut missed = 0;
        for f in &track.frames {
            let Some(note) = sig
                .notes
                .iter()
                .find(|n| f.pos >= n.start + guard && f.pos + guard < n.end)
            else {
                continue;
            };
            let _ = note;
            if !f.voiced {
                missed += 1;
                continue;
            }
            let truth = hz_to_cents64(sig.f0_truth[f.pos] as f64);
            errs.push((f.cents - truth).abs());
        }
        (errs, missed)
    }

    fn voice_for(f0: f64) -> Voice {
        if f0 < 300.0 {
            Voice::male().clean()
        } else {
            Voice::female().clean()
        }
    }

    const F0S: [f64; 11] = [
        70.0, 82.4, 110.0, 146.8, 196.0, 261.6, 349.2, 466.2, 622.3, 830.6, 1100.0,
    ];

    fn validate(sr: f32, f0s: &[f64]) -> (f64, f64, f64, usize) {
        let mut all = Vec::new();
        let mut missed = 0;
        let mut tracker = ReferenceTracker::new(sr, ReferenceConfig::default());
        for (i, &f0) in f0s.iter().enumerate() {
            let vowel = Vowel::ALL[i % 5];
            for vib in [
                None,
                Some(Vibrato::immediate(6.0, 50.0)),
                Some(Vibrato::immediate(5.5, 60.0)),
            ] {
                let sig = Signal::sustained(sr, voice_for(f0), vowel, f0, 0.6, vib);
                let track = tracker.track(&sig.audio);
                let (e, m) = frame_errors(&sig, &track, 0.04);
                missed += m;
                let p99 = percentile(&e, 99.0);
                assert!(p99 < 3.0, "sr {sr} f0 {f0} vib {vib:?}: p99 {p99}");
                all.extend(e);
            }
        }
        (
            percentile(&all, 50.0),
            percentile(&all, 99.0),
            percentile(&all, 100.0),
            missed,
        )
    }

    #[test]
    fn accuracy_on_clean_vowels_44k_48k() {
        for sr in [44_100.0f32, 48_000.0] {
            let (med, p99, max, missed) = validate(sr, &F0S);
            eprintln!(
                "reference @ {sr}: median {med:.4} c, p99 {p99:.4} c, max {max:.4} c, missed {missed}"
            );
            assert!(med < 0.5, "median {med}");
            assert!(p99 < 3.0, "p99 {p99}");
            assert_eq!(missed, 0);
        }
    }

    #[test]
    fn accuracy_on_clean_vowels_96k() {
        let (med, p99, max, missed) = validate(96_000.0, &[70.0, 196.0, 622.3, 1100.0]);
        eprintln!("reference @ 96k: median {med:.4} c, p99 {p99:.4} c, max {max:.4} c");
        assert!(med < 0.5 && p99 < 3.0 && missed == 0);
    }

    #[test]
    fn curvature_correction_matters_at_low_f0() {
        let sr = 48_000.0;
        let sig = Signal::sustained(
            sr,
            Voice::male().clean(),
            Vowel::O,
            70.0,
            0.8,
            Some(Vibrato::immediate(6.0, 50.0)),
        );
        let on = ReferenceTracker::new(sr, ReferenceConfig::default()).track(&sig.audio);
        let off = ReferenceTracker::new(
            sr,
            ReferenceConfig {
                curvature_correction: false,
                ..Default::default()
            },
        )
        .track(&sig.audio);
        let (e_on, _) = frame_errors(&sig, &on, 0.05);
        let (e_off, _) = frame_errors(&sig, &off, 0.05);
        let (m_on, m_off) = (percentile(&e_on, 50.0), percentile(&e_off, 50.0));
        eprintln!("70 Hz vibrato median: corrected {m_on:.3} c, raw {m_off:.3} c");
        assert!(
            m_on < m_off * 0.5,
            "correction did not help: {m_on} vs {m_off}"
        );
    }

    #[test]
    fn no_octave_errors_with_f1_at_twice_f0() {
        // F1 tuned to 2·f0 with a narrow bandwidth boosts the 2nd harmonic far
        // above the fundamental (the classic octave-up trap).
        let sr = 48_000.0;
        for f0 in [220.0, 300.0, 400.0] {
            let mut voice = Voice::female().clean();
            voice.rd = 0.8;
            let fm = Vowel::A.formants(VoiceType::Female).with_f1(2.0 * f0, 50.0);
            let sig = synth_with_formants(sr, voice, f0, fm);
            let track = track(&sig.audio, sr);
            let (e, missed) = frame_errors(&sig, &track, 0.04);
            assert_eq!(missed, 0);
            let worst = percentile(&e, 100.0);
            assert!(
                worst < 3.0,
                "f0 {f0}: worst error {worst} c (octave error?)"
            );
        }
    }

    /// A sustained note whose tract uses explicit formants.
    fn synth_with_formants(sr: f32, voice: Voice, f0: f64, fm: Formants) -> Signal {
        // Render a normal note, then re-filter the source by building the
        // signal through a custom vowel: easiest is to synthesise each vowel
        // and pick the F1-modified formants via the public FormantFilter.
        use crate::synth::{FormantFilter, GlottalTable, LfShape};
        let srf = sr as f64;
        let n = (0.6 * srf) as usize;
        let table = GlottalTable::new(LfShape::from_rd(voice.rd), 400);
        let mut filt = FormantFilter::new(srf, &fm);
        let mut phase = 0.0;
        let mut audio = vec![0.0f32; n];
        let mut f0_truth = vec![0.0f32; n];
        let lead = (0.05 * srf) as usize;
        for i in lead..n - lead {
            let s = table.sample(phase, f0, 0.8 * 0.45 * srf, 0.45 * srf);
            phase = (phase + f0 / srf).fract();
            audio[i] = (0.1 * filt.process(s)) as f32;
            f0_truth[i] = f0 as f32;
        }
        let mut sig = Signal::from_audio(audio, sr);
        sig.f0_truth = f0_truth;
        sig.notes.push(crate::synth::NoteSpan {
            start: lead,
            end: n - lead,
            target_cents: hz_to_cents64(f0),
            detune_cents: 0.0,
            vibrato: None,
            legato_in: false,
            kind: crate::synth::NoteKind::Note,
        });
        sig
    }

    #[test]
    fn silence_and_noise_are_unvoiced() {
        let sr = 48_000.0;
        let sig = SignalBuilder::new(sr)
            .silence(0.2)
            .noise_burst(0.3, -20.0)
            .consonant(Consonant::S, 0.2, -30.0)
            .consonant(Consonant::Breath, 0.3, -35.0)
            .silence(0.2)
            .build();
        let t = track(&sig.audio, sr);
        let v = t.voiced_count();
        assert!(
            v * 100 < t.frames.len(),
            "{v} of {} frames voiced",
            t.frames.len()
        );
    }

    /// On a melody with leaps, scoops, overshoot and gaps. Inside fast leaps
    /// (> 3 c/ms) the radiated pitch is itself ambiguous by ~±15 c: harmonics
    /// sweep through the formants and are delayed by different group delays,
    /// so a single delay cannot describe them (the bare source tracks to ±4 c
    /// there). Accuracy is asserted where the truth moves slower; inside leaps
    /// only the absence of gross (octave) errors is.
    #[test]
    fn follows_melody_with_glides_and_gaps() {
        let sr = 48_000.0;
        let sig = SignalBuilder::new(sr)
            .silence(0.1)
            .melody(
                Melody::new(Voice::male().clean())
                    .note(Note::midi(48.0, 0.4).detune(0.0))
                    .note(Note::midi(55.0, 0.4).detune(0.0).overshoot(30.0))
                    .note(Note::midi(52.0, 0.4).gap(0.15).scoop(-120.0).detune(0.0))
                    .note(Note::midi(60.0, 0.4).detune(0.0).glide(0.08)),
            )
            .silence(0.1)
            .build();
        let t = track(&sig.audio, sr);
        let guard = (0.03 * sr as f64) as usize;
        let slope = |p: usize| -> f64 {
            let d = 96; // ±2 ms
            let c = |i: usize| hz_to_cents64(sig.f0_truth[i] as f64);
            (c(p + d) - c(p - d)).abs() / 4.0 // c/ms
        };
        let (mut slow, mut missed, mut worst_fast) = (Vec::new(), 0, 0.0f64);
        for f in &t.frames {
            if !sig
                .notes
                .iter()
                .any(|n| f.pos >= n.start + guard && f.pos + guard < n.end)
            {
                continue;
            }
            let truth = hz_to_cents64(sig.f0_truth[f.pos] as f64);
            let fast = slope(f.pos) > 3.0;
            match (f.voiced, fast) {
                (true, false) => slow.push((f.cents - truth).abs()),
                (false, false) => missed += 1,
                (true, true) => worst_fast = worst_fast.max((f.cents - truth).abs()),
                (false, true) => {}
            }
        }
        let (med, p99) = (percentile(&slow, 50.0), percentile(&slow, 99.0));
        eprintln!(
            "melody: median {med:.3} p99 {p99:.3} missed {missed} worst-in-leap {worst_fast:.1}"
        );
        assert_eq!(missed, 0);
        assert!(med < 0.5 && p99 < 5.0, "median {med} p99 {p99}");
        assert!(worst_fast < 60.0, "gross error inside a leap: {worst_fast}");
    }

    #[test]
    fn helpers() {
        let mut v = vec![true, false, true, false, false, true, true];
        fill_holes(&mut v, 1);
        assert_eq!(v, vec![true, true, true, false, false, true, true]);
        drop_short_runs(&mut v, 3);
        assert_eq!(v, vec![true, true, true, false, false, false, false]);
        let (t, y) = golden_max(&|x: f64| -(x - 0.3).powi(2), -1.0, 1.0, 80);
        assert!((t - 0.3).abs() < 1e-9 && y.abs() < 1e-12);
    }
}
