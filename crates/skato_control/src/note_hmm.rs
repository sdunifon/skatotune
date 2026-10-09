//! Note decision HMM for HQ/Max, R§8.12 (MonoNote-style; reimplemented, not
//! derived from the GPL pYIN code).
//!
//! **States.** For every candidate note n (MIDI 0-127 masked by the target
//! set, so `Remove` notes never get mass) three states, as in pYIN's
//! MonoNote: attack `A_n`, stable `S_n` and silent `Z_n`.
//!
//! **Transitions** per control hop, rescaled from pYIN's 256/44100 s frame
//! as `p' = p^(Δ/Δ_pyin)` (R§8.12):
//! - `A_n → A_n` 0.9, `A_n → S_n` the rest (mean attack ≈ 55 ms);
//! - `S_n → S_n` 0.99; the remainder `q` is the note-change hazard;
//! - any pitched state of n → `A_m` (m ≠ n) with `q·J(m|n)`. MonoNote routes
//!   note changes only through silence; sung legato needs a direct path, so
//!   this is added (design choice);
//! - unvoiced hops move all mass of note n into `Z_n` (the voicing FSM gives
//!   a hard voicing observation), which keeps the note identity through short
//!   gaps; the first voiced hop re-attacks from `Z_n` with `J0(m|n)`, which
//!   favours the same note. Forgetting after 100 ms is the FSM's job (reset).
//! - `J` is pYIN's Gaussian jump prior (σ = sigma2Note = 0.7 st, maxJump 13;
//!   24 in Max) mixed with a uniform part and a small floor, so leaps of a
//!   fifth or more than maxJump are decoded within a few ms instead of being
//!   blocked by the narrow Gaussian (design choice).
//!
//! **Observations** (on the vibrato-free decision pitch). Each note has a
//! *capture region* reaching to the midpoint with its neighbours plus the
//! Tolerance cents. Inside it the stable likelihood is flat, so a pitch that
//! hovers within the tolerance band around a midpoint yields no evidence and
//! the transition prior keeps the current note: that is the HMM form of the
//! Waves cents tolerance, and it is what prevents flip-flop. Outside it the
//! log-likelihood falls quadratically (`K_TAIL` per ms per c²), capped at
//! `E_CAP` per ms so a one-hop outlier cannot flip the note, plus a linear
//! near-band slope (`K_NEAR` per ms per c) once the corrector reports that
//! the pitch has *persisted* past the current note's band ([`HmmObs::near`]):
//! a pitch held just past the band then switches within tens of ms instead of
//! 0.57 s (CL-2), while brief excursions past it (vibrato peaks, wobble at
//! note ends) keep the quadratic tail's hysteresis and do not warble.
//! Tolerance *time* adds a small linear slope so the evidence for the nearer
//! note grows with how long and how far past the midpoint the pitch sits
//! (R§8.12: the tolerance knobs map onto the evidence, calibrated). The
//! corrector gates every switch with Waves' OR rule (R§7.6.2 F), so neither
//! term can switch before x > T_c or x > 0 has held for T_t. Attack states
//! use the same capture region at half the evidence rate (R§8.12: one
//! observation model around the note pitch). A wider attack region (tried:
//! +25 c, to tolerate scoops) makes the note just entered sticky: after a
//! scoop that commits to the lower neighbour, or in a legato glide through an
//! intermediate note, its attack state keeps full likelihood while the pitch
//! sits 20-30 c past midpoint + tolerance, and the switch came 40-140 ms late
//! in the engine harness.
//! All evidence is expressed per millisecond and multiplied by the hop length
//! and by the detector confidence, so behaviour is hop-length invariant and
//! unclear frames weigh less ("σ scaled by detector confidence").
//!
//! **Decoding.** Live: forward posterior; switch when a new note's mass
//! exceeds [`P_SWITCH`]. Studio: fixed-lag Viterbi over the lookahead, the
//! decision for the frame `lag` hops back being read off the backtrace.

use crate::targets::{N_NOTES, TargetSet};

/// pYIN MonoNote frame: 256 samples at 44.1 kHz, ms.
const PYIN_HOP_MS: f32 = 256.0 * 1000.0 / 44_100.0;
const P_ATTACK_SELF: f32 = 0.9;
const P_STABLE_SELF: f32 = 0.99;
/// pYIN sigma2Note, semitones.
const SIGMA_JUMP: f32 = 0.7;
/// Uniform share of the jump prior within maxJump.
const UNIFORM_JUMP: f32 = 0.03;
/// Floor of the jump prior for every allowed note (also beyond maxJump).
const FLOOR_JUMP: f32 = 1e-4;
pub(crate) const MAX_JUMP_HQ: usize = 13;
pub(crate) const MAX_JUMP_MAX: usize = 24;
/// Linear evidence beyond the capture region once the pitch has persisted
/// there ([`HmmObs::near`]), nats per ms per c: a pitch held 1 c past the
/// band then switches about 20 ms after the persistence test passes, 4 c past
/// in about 5 ms (CL-2; with only the quadratic tail 1 c past took 0.57 s).
const K_NEAR: f32 = 0.4;
/// Quadratic evidence beyond the capture region, nats per ms per c².
const K_TAIL: f32 = 0.006;
/// Evidence cap, nats per ms.
const E_CAP: f32 = 2.5;
/// Past-midpoint distance (c) that switches in about `tolerance_ms`.
const D_REF_TIME: f32 = 5.0;
/// Attack evidence weight.
const ATTACK_SCALE: f32 = 0.5;
/// Capture half-width used at the ends of the candidate range, c.
const OPEN_EDGE: f32 = 10_000.0;
/// Posterior mass a new note needs before the decision switches to it.
pub(crate) const P_SWITCH: f32 = 0.6;
/// Longest fixed lag the Viterbi keeps backpointers for, hops.
pub(crate) const MAX_LAG: usize = 128;
/// "Impossible" log score (finite so arithmetic never produces NaN).
const NEG: f32 = -1.0e30;

/// One observation for the HMM.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct HmmObs {
    /// Decision pitch relative to the note grid (note n sits at 100·n).
    pub rel: f32,
    /// Detector confidence 0-1.
    pub confidence: f32,
    /// FSM-confirmed voicing.
    pub voiced: bool,
    /// The pitch has stayed past the current note's cents band long enough
    /// to count as held there: enables the near-band slope `K_NEAR`.
    pub near: bool,
}

/// Fixed-lag Viterbi state (Studio only).
struct Viterbi {
    la: [f32; N_NOTES],
    ls: [f32; N_NOTES],
    lz: [f32; N_NOTES],
    /// Backpointers: `cap` frames × 3·N states, ring indexed by `head`.
    bp: Vec<u16>,
    cap: usize,
    head: usize,
    frames: usize,
    silent: bool,
    started: bool,
}

const N_STATES: usize = 3 * N_NOTES;

#[inline]
fn state(n: usize, k: usize) -> u16 {
    (3 * n + k) as u16
}

/// MonoNote-style note HMM (see module docs).
pub(crate) struct NoteHmm {
    a: [f32; N_NOTES],
    s: [f32; N_NOTES],
    z: [f32; N_NOTES],
    silent: bool,
    started: bool,
    // Scratch, reused every hop.
    w: [f32; N_NOTES],
    lik_a: [f32; N_NOTES],
    lik_s: [f32; N_NOTES],
    // Geometry from the target set.
    allowed: [bool; N_NOTES],
    count: usize,
    r_lo: [f32; N_NOTES],
    r_hi: [f32; N_NOTES],
    norm_leg: [f32; N_NOTES],
    norm_on: [f32; N_NOTES],
    kernel: [f32; MAX_JUMP_MAX + 1],
    max_jump: usize,
    // Per-hop constants.
    hop_ms: f32,
    q: f32,
    a_self: f32,
    k_t: f32,
    tol_c: f32,
    vit: Option<Viterbi>,
}

impl NoteHmm {
    /// `lag_cap` > 0 allocates the fixed-lag Viterbi for that many hops.
    pub fn new(hop_ms: f32, lag_cap: usize) -> Self {
        let r = hop_ms / PYIN_HOP_MS;
        let vit = (lag_cap > 0).then(|| {
            let cap = lag_cap.min(MAX_LAG) + 1;
            Viterbi {
                la: [NEG; N_NOTES],
                ls: [NEG; N_NOTES],
                lz: [NEG; N_NOTES],
                bp: vec![0; cap * N_STATES],
                cap,
                head: 0,
                frames: 0,
                silent: false,
                started: false,
            }
        });
        let mut h = Self {
            a: [0.0; N_NOTES],
            s: [0.0; N_NOTES],
            z: [0.0; N_NOTES],
            silent: false,
            started: false,
            w: [0.0; N_NOTES],
            lik_a: [1.0; N_NOTES],
            lik_s: [1.0; N_NOTES],
            allowed: [true; N_NOTES],
            count: N_NOTES,
            r_lo: [50.0; N_NOTES],
            r_hi: [50.0; N_NOTES],
            norm_leg: [1.0; N_NOTES],
            norm_on: [1.0; N_NOTES],
            kernel: [0.0; MAX_JUMP_MAX + 1],
            max_jump: MAX_JUMP_HQ,
            hop_ms,
            q: 1.0 - P_STABLE_SELF.powf(r),
            a_self: P_ATTACK_SELF.powf(r),
            k_t: 0.0,
            tol_c: 0.0,
            vit,
        };
        h.build_kernel();
        h.set_targets(&TargetSet::default());
        h
    }

    /// Forget the note (FSM forget, or reset).
    pub fn reset(&mut self) {
        self.a = [0.0; N_NOTES];
        self.s = [0.0; N_NOTES];
        self.z = [0.0; N_NOTES];
        self.silent = false;
        self.started = false;
        if let Some(v) = self.vit.as_mut() {
            v.la = [NEG; N_NOTES];
            v.ls = [NEG; N_NOTES];
            v.lz = [NEG; N_NOTES];
            v.head = 0;
            v.frames = 0;
            v.silent = false;
            v.started = false;
        }
    }

    /// Start from certainty on note `n` (used when switching tier mid-note).
    pub fn seed(&mut self, n: usize) {
        self.reset();
        if n < N_NOTES {
            self.s[n] = 1.0;
            self.started = true;
        }
    }

    /// Tolerance knobs and tier. Cheap; call on every settings change.
    pub fn set_params(&mut self, tolerance_cents: f32, tolerance_ms: f32, max_jump: usize) {
        let mj = max_jump.clamp(1, MAX_JUMP_MAX);
        let tol = tolerance_cents.clamp(0.0, 45.0);
        let geometry_changed = mj != self.max_jump || tol != self.tol_c;
        self.max_jump = mj;
        self.tol_c = tol;
        // Switching cost ≈ −ln q plus the jump prior; a pitch d_ref past the
        // midpoint accumulates 2·d_ref·k_t nats per ms more for the nearer
        // note, so choose k_t to pay that cost in about tolerance_ms.
        self.k_t = if tolerance_ms > 0.0 {
            let cost = -(self.q.max(1e-9)).ln() + 1.0;
            cost / (2.0 * D_REF_TIME * tolerance_ms)
        } else {
            0.0
        };
        if geometry_changed {
            self.build_kernel();
            self.rebuild_geometry();
        }
    }

    fn build_kernel(&mut self) {
        for (d, k) in self.kernel.iter_mut().enumerate() {
            let x = d as f32 / SIGMA_JUMP;
            *k = if d <= self.max_jump { (-0.5 * x * x).exp() + UNIFORM_JUMP } else { 0.0 };
        }
    }

    /// Candidate notes changed (scale, MIDI, learn).
    pub fn set_targets(&mut self, t: &TargetSet) {
        self.allowed = *t.allowed_mask();
        self.rebuild_geometry();
    }

    fn rebuild_geometry(&mut self) {
        self.count = self.allowed.iter().filter(|a| **a).count();
        let mut last: Option<usize> = None;
        for n in 0..N_NOTES {
            if self.allowed[n] {
                self.r_lo[n] = match last {
                    Some(m) => 50.0 * (n - m) as f32 + self.tol_c,
                    None => OPEN_EDGE,
                };
                if let Some(m) = last {
                    self.r_hi[m] = 50.0 * (n - m) as f32 + self.tol_c;
                }
                self.r_hi[n] = OPEN_EDGE;
                last = Some(n);
            }
        }
        // Per-source normalisation of the legato and re-onset priors.
        let floor_total = FLOOR_JUMP * self.count as f32;
        for n in 0..N_NOTES {
            let mut leg = floor_total;
            let mut on = floor_total;
            if self.allowed[n] {
                on += 1.0 + UNIFORM_JUMP;
            }
            for d in 1..=self.max_jump {
                let k = self.kernel[d];
                for m in [n.checked_sub(d), Some(n + d)].into_iter().flatten() {
                    if self.allowed.get(m).copied().unwrap_or(false) {
                        leg += k;
                        on += k;
                    }
                }
            }
            self.norm_leg[n] = if leg > 0.0 { 1.0 / leg } else { 0.0 };
            self.norm_on[n] = if on > 0.0 { 1.0 / on } else { 0.0 };
        }
    }

    /// Evidence energies (nats per ms) of the stable and attack states.
    #[inline]
    fn energies(&self, n: usize, rel: f32, near: bool) -> (f32, f32) {
        let dx = rel - 100.0 * n as f32;
        let out = if dx >= 0.0 { dx - self.r_hi[n] } else { -dx - self.r_lo[n] }.max(0.0);
        let k_near = if near { K_NEAR } else { 0.0 };
        let es = (self.k_t * dx.abs() + (k_near + K_TAIL * out) * out).min(E_CAP);
        let ea = ATTACK_SCALE * es;
        (es, ea)
    }

    fn weight(&self, confidence: f32) -> f32 {
        self.hop_ms * if confidence.is_finite() { confidence.clamp(0.1, 1.0) } else { 0.1 }
    }

    fn likelihoods(&mut self, obs: &HmmObs) {
        let w = self.weight(obs.confidence);
        let cap_s = (-w * E_CAP).exp();
        let cap_a = (-w * ATTACK_SCALE * E_CAP).exp();
        for n in 0..N_NOTES {
            if !self.allowed[n] {
                self.lik_a[n] = 0.0;
                self.lik_s[n] = 0.0;
                continue;
            }
            let (es, ea) = self.energies(n, obs.rel, obs.near);
            self.lik_s[n] = if es >= E_CAP { cap_s } else { (-w * es).exp() };
            self.lik_a[n] = if ea >= ATTACK_SCALE * E_CAP { cap_a } else { (-w * ea).exp() };
        }
    }

    /// Banded jump-prior convolution of `self.w` into target `m`, plus the
    /// floor share of `total`. `with_self` includes `d = 0` (re-onset).
    #[inline]
    fn inflow(&self, m: usize, total: f32, with_self: bool) -> f32 {
        let mut acc = FLOOR_JUMP * total;
        if with_self {
            acc += (1.0 + UNIFORM_JUMP) * self.w[m];
        }
        for d in 1..=self.max_jump {
            let k = self.kernel[d];
            if let Some(v) = m.checked_sub(d).and_then(|i| self.w.get(i)) {
                acc += k * v;
            }
            if let Some(v) = self.w.get(m + d) {
                acc += k * v;
            }
        }
        acc
    }

    /// One hop of the forward filter (and of the Viterbi when allocated).
    pub fn step(&mut self, obs: &HmmObs) {
        if obs.voiced && obs.rel.is_finite() {
            self.likelihoods(obs);
            self.forward_voiced();
        } else if !self.silent {
            for n in 0..N_NOTES {
                self.z[n] = self.a[n] + self.s[n];
                self.a[n] = 0.0;
                self.s[n] = 0.0;
            }
            self.silent = true;
        }
        if self.vit.is_some() {
            self.viterbi_step(obs.voiced && obs.rel.is_finite());
        }
    }

    fn forward_voiced(&mut self) {
        let q = self.q;
        if !self.started {
            // Onset with no memory: attack states weighted by the *stable*
            // (narrow) likelihood, so the onset commits to the note whose
            // capture region holds the pitch. The wider attack region then
            // only serves to tolerate scoops after that commitment; weighting
            // by it here would leave neighbours tied and the first vibrato
            // excursion past the midpoint would flip the note.
            for n in 0..N_NOTES {
                self.a[n] = if self.allowed[n] { self.lik_s[n] } else { 0.0 };
                self.s[n] = 0.0;
                self.z[n] = 0.0;
            }
            self.started = true;
            self.silent = false;
        } else if self.silent {
            // Re-onset after a gap: Z_n → A_m with the same-note-favouring prior.
            let mut total = 0.0;
            for n in 0..N_NOTES {
                self.w[n] = self.z[n] * self.norm_on[n];
                total += self.w[n];
            }
            for m in 0..N_NOTES {
                self.a[m] = if self.allowed[m] { self.inflow(m, total, true) * self.lik_a[m] } else { 0.0 };
                self.s[m] = 0.0;
            }
            self.z = [0.0; N_NOTES];
            self.silent = false;
        } else {
            let mut total = 0.0;
            for n in 0..N_NOTES {
                self.w[n] = (self.a[n] + self.s[n]) * q * self.norm_leg[n];
                total += self.w[n];
            }
            let keep_a = (1.0 - q) * self.a_self;
            let a_to_s = (1.0 - q) * (1.0 - self.a_self);
            for m in 0..N_NOTES {
                if !self.allowed[m] {
                    self.a[m] = 0.0;
                    self.s[m] = 0.0;
                    continue;
                }
                let inflow = self.inflow(m, total, false);
                let a_new = (keep_a * self.a[m] + inflow) * self.lik_a[m];
                let s_new = ((1.0 - q) * self.s[m] + a_to_s * self.a[m]) * self.lik_s[m];
                self.a[m] = a_new;
                self.s[m] = s_new;
            }
        }
        self.normalize();
    }

    fn normalize(&mut self) {
        let sum: f32 = self.a.iter().chain(self.s.iter()).chain(self.z.iter()).sum();
        if sum > 1e-30 && sum.is_finite() {
            let inv = 1.0 / sum;
            for n in 0..N_NOTES {
                self.a[n] *= inv;
                self.s[n] *= inv;
                self.z[n] *= inv;
            }
        } else {
            // Every allowed state was ruled out (or underflow): restart the
            // note from the current evidence.
            let mut acc = 0.0;
            for n in 0..N_NOTES {
                self.a[n] = if self.allowed[n] { self.lik_s[n].max(1e-12) } else { 0.0 };
                self.s[n] = 0.0;
                self.z[n] = 0.0;
                acc += self.a[n];
            }
            if acc > 0.0 {
                for v in self.a.iter_mut() {
                    *v /= acc;
                }
            }
        }
    }

    /// Posterior mass of note `n`.
    #[inline]
    pub fn prob(&self, n: usize) -> f32 {
        match (self.a.get(n), self.s.get(n), self.z.get(n)) {
            (Some(a), Some(s), Some(z)) => a + s + z,
            _ => 0.0,
        }
    }

    /// Most probable note; ties go to the note nearest `rel`.
    pub fn argmax(&self, rel: f32) -> Option<(usize, f32)> {
        if !self.started {
            return None;
        }
        let mut best: Option<(usize, f32)> = None;
        for n in 0..N_NOTES {
            if !self.allowed[n] {
                continue;
            }
            let p = self.prob(n);
            let better = match best {
                None => true,
                Some((b, pb)) => {
                    p > pb * (1.0 + 1e-4) + 1e-12
                        || ((p - pb).abs() <= pb * 1e-4 + 1e-12
                            && (rel - 100.0 * n as f32).abs() < (rel - 100.0 * b as f32).abs())
                }
            };
            if better {
                best = Some((n, p));
            }
        }
        best
    }

    // ----- Studio: fixed-lag Viterbi --------------------------------------

    // The per-note tables (scores, likelihoods, priors, backpointers) are
    // parallel arrays read with neighbour offsets; index loops are clearer.
    #[allow(clippy::needless_range_loop)]
    fn viterbi_step(&mut self, voiced: bool) {
        let q = self.q;
        let ln_keep_s = (1.0 - q).ln();
        let ln_keep_a = ((1.0 - q) * self.a_self).ln();
        let ln_a_to_s = ((1.0 - q) * (1.0 - self.a_self)).ln();
        let ln_floor = FLOOR_JUMP.ln();
        let mut ln_kernel = [NEG; MAX_JUMP_MAX + 1];
        for d in 0..=self.max_jump {
            ln_kernel[d] = self.kernel[d].max(1e-30).ln();
        }
        let ln_same = (1.0 + UNIFORM_JUMP).ln();
        // Moved out for the update so `self`'s tables can be read alongside.
        let Some(mut vit) = self.vit.take() else { return };
        let v = &mut vit;
        let frame = v.head;
        let base = frame * N_STATES;
        let mut na = [NEG; N_NOTES];
        let mut ns = [NEG; N_NOTES];
        let mut nz = v.lz;
        let lnl = |x: f32| if x > 0.0 { x.ln() } else { NEG };
        if voiced {
            if !v.started || (!v.silent && v.la.iter().chain(v.ls.iter()).all(|x| *x <= NEG)) {
                for m in 0..N_NOTES {
                    if self.allowed[m] {
                        na[m] = lnl(self.lik_s[m]);
                    }
                    v.bp[base + 3 * m] = state(m, 0);
                    v.bp[base + 3 * m + 1] = state(m, 1);
                    v.bp[base + 3 * m + 2] = state(m, 2);
                }
                v.started = true;
                nz = [NEG; N_NOTES];
            } else if v.silent {
                // Re-onset from the silent states.
                let mut src = [NEG; N_NOTES];
                let mut gbest = (NEG, 0usize);
                for n in 0..N_NOTES {
                    src[n] = v.lz[n] + lnl(self.norm_on[n]);
                    if src[n] > gbest.0 {
                        gbest = (src[n], n);
                    }
                }
                for m in 0..N_NOTES {
                    if !self.allowed[m] {
                        continue;
                    }
                    let mut best = (gbest.0 + ln_floor, state(gbest.1, 2));
                    let lo = m.saturating_sub(self.max_jump);
                    let hi = (m + self.max_jump).min(N_NOTES - 1);
                    for n in lo..=hi {
                        let d = m.abs_diff(n);
                        let lk = if d == 0 { ln_same } else { ln_kernel[d] };
                        let c = src[n] + lk;
                        if c > best.0 {
                            best = (c, state(n, 2));
                        }
                    }
                    na[m] = best.0 + lnl(self.lik_a[m]);
                    v.bp[base + 3 * m] = best.1;
                }
                nz = [NEG; N_NOTES];
                v.silent = false;
            } else {
                let mut src = [NEG; N_NOTES];
                let mut src_state = [0u16; N_NOTES];
                let mut gbest = (NEG, 0u16);
                let ln_q = q.max(1e-30).ln();
                for n in 0..N_NOTES {
                    let (sc, st) = if v.la[n] >= v.ls[n] { (v.la[n], state(n, 0)) } else { (v.ls[n], state(n, 1)) };
                    src[n] = sc + ln_q + lnl(self.norm_leg[n]);
                    src_state[n] = st;
                    if src[n] > gbest.0 {
                        gbest = (src[n], st);
                    }
                }
                for m in 0..N_NOTES {
                    if !self.allowed[m] {
                        continue;
                    }
                    let mut best = (v.la[m] + ln_keep_a, state(m, 0));
                    let fl = gbest.0 + ln_floor;
                    if fl > best.0 {
                        best = (fl, gbest.1);
                    }
                    let lo = m.saturating_sub(self.max_jump);
                    let hi = (m + self.max_jump).min(N_NOTES - 1);
                    for n in lo..=hi {
                        if n == m {
                            continue;
                        }
                        let c = src[n] + ln_kernel[m.abs_diff(n)];
                        if c > best.0 {
                            best = (c, src_state[n]);
                        }
                    }
                    na[m] = best.0 + lnl(self.lik_a[m]);
                    v.bp[base + 3 * m] = best.1;
                    let stay = v.ls[m] + ln_keep_s;
                    let from_a = v.la[m] + ln_a_to_s;
                    let (sc, st) = if stay >= from_a { (stay, state(m, 1)) } else { (from_a, state(m, 0)) };
                    ns[m] = sc + lnl(self.lik_s[m]);
                    v.bp[base + 3 * m + 1] = st;
                }
            }
        } else if v.started {
            for n in 0..N_NOTES {
                if v.silent {
                    v.bp[base + 3 * n + 2] = state(n, 2);
                } else {
                    let (sc, st) = if v.la[n] >= v.ls[n] { (v.la[n], state(n, 0)) } else { (v.ls[n], state(n, 1)) };
                    nz[n] = sc;
                    v.bp[base + 3 * n + 2] = st;
                }
            }
            v.silent = true;
        }
        // Normalise (subtract the max) and clamp the impossible floor.
        let mx = na.iter().chain(ns.iter()).chain(nz.iter()).fold(NEG, |m, x| m.max(*x));
        if mx > NEG {
            for x in na.iter_mut().chain(ns.iter_mut()).chain(nz.iter_mut()) {
                *x = if *x > NEG { (*x - mx).max(NEG) } else { NEG };
            }
        }
        v.la = na;
        v.ls = ns;
        v.lz = nz;
        v.head = (v.head + 1) % v.cap;
        v.frames = (v.frames + 1).min(v.cap);
        self.vit = Some(vit);
    }

    /// Fixed-lag decision: the note on the best path `lag` hops before the
    /// newest frame. `None` without Viterbi, before any voiced frame, or when
    /// that frame was silent.
    pub fn viterbi_note(&self, lag: usize) -> Option<usize> {
        let v = self.vit.as_ref()?;
        if !v.started || v.frames == 0 {
            return None;
        }
        let lag = lag.min(v.frames - 1);
        // Best state at the newest frame.
        let mut best = (NEG, 0usize);
        for n in 0..N_NOTES {
            for (k, sc) in [v.la[n], v.ls[n], v.lz[n]].into_iter().enumerate() {
                if sc > best.0 {
                    best = (sc, 3 * n + k);
                }
            }
        }
        if best.0 <= NEG {
            return None;
        }
        let mut st = best.1;
        let mut frame = (v.head + v.cap - 1) % v.cap;
        for _ in 0..lag {
            st = usize::from(*v.bp.get(frame * N_STATES + st)?);
            frame = (frame + v.cap - 1) % v.cap;
        }
        (st % 3 != 2).then_some(st / 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOP_MS: f32 = 64.0 / 48.0;

    fn voiced(rel: f32) -> HmmObs {
        HmmObs { rel, confidence: 0.9, voiced: true, near: true }
    }

    fn feed(h: &mut NoteHmm, rel: f32, hops: usize) {
        for _ in 0..hops {
            h.step(&voiced(rel));
        }
    }

    fn top(h: &NoteHmm, rel: f32) -> usize {
        h.argmax(rel).map(|(n, _)| n).unwrap_or(usize::MAX)
    }

    #[test]
    fn onset_picks_nearest() {
        let mut h = NoteHmm::new(HOP_MS, 0);
        h.set_params(8.0, 0.0, MAX_JUMP_HQ);
        // Attack regions are wide (scoops), so neighbours tie at first; the
        // tie goes to the nearest note.
        feed(&mut h, 6020.0, 3);
        assert_eq!(top(&h, 6020.0), 60);
        // As mass moves into the stable states the posterior sharpens.
        feed(&mut h, 6020.0, 60);
        assert_eq!(top(&h, 6020.0), 60);
        assert!(h.prob(60) > 0.8, "p60 {}", h.prob(60));
    }

    #[test]
    fn switches_promptly_and_holds_inside_tolerance() {
        let mut h = NoteHmm::new(HOP_MS, 0);
        h.set_params(8.0, 0.0, MAX_JUMP_HQ);
        feed(&mut h, 6000.0, 200);
        // 5 c past the midpoint is inside the 8 c tolerance: no switch, ever.
        feed(&mut h, 6055.0, 1500);
        assert!(h.prob(61) < P_SWITCH, "p61 {}", h.prob(61));
        // Clearly on the next note: switch within ~10 ms.
        let mut hops = 0;
        while h.prob(61) < P_SWITCH && hops < 100 {
            h.step(&voiced(6100.0));
            hops += 1;
        }
        assert!((hops as f32) * HOP_MS < 10.0, "took {hops} hops");
    }

    #[test]
    fn outlier_does_not_flip() {
        let mut h = NoteHmm::new(HOP_MS, 0);
        h.set_params(8.0, 0.0, MAX_JUMP_HQ);
        feed(&mut h, 6000.0, 200);
        feed(&mut h, 6700.0, 1);
        feed(&mut h, 6000.0, 5);
        assert_eq!(top(&h, 6000.0), 60);
        assert!(h.prob(60) > 0.9);
    }

    #[test]
    fn big_leap_is_decoded_fast() {
        let mut h = NoteHmm::new(HOP_MS, 0);
        h.set_params(8.0, 0.0, MAX_JUMP_HQ);
        feed(&mut h, 6000.0, 200);
        let mut hops = 0;
        while h.prob(67) < P_SWITCH && hops < 100 {
            h.step(&voiced(6700.0));
            hops += 1;
        }
        assert!((hops as f32) * HOP_MS < 8.0, "fifth took {hops} hops");
        // A leap beyond maxJump still gets through via the floor.
        feed(&mut h, 6700.0, 100);
        let mut hops = 0;
        while h.prob(87) < P_SWITCH && hops < 200 {
            h.step(&voiced(8700.0));
            hops += 1;
        }
        assert!((hops as f32) * HOP_MS < 15.0, "20 st leap took {hops} hops");
    }

    /// Hops until note 61's posterior exceeds `P_SWITCH` for a pitch held
    /// `past` cents beyond C4's capture band (midpoint + tolerance).
    fn switch_hops(tol: f32, past: f32) -> usize {
        let mut h = NoteHmm::new(HOP_MS, 0);
        h.set_params(tol, 0.0, MAX_JUMP_HQ);
        feed(&mut h, 6000.0, 200);
        let mut hops = 0;
        while h.prob(61) < P_SWITCH && hops < 2000 {
            h.step(&voiced(6050.0 + tol + past));
            hops += 1;
        }
        hops
    }

    /// CL-2: the cents part of the OR rule. Just past the band the switch
    /// comes within about 20 ms (it took 0.57 s with only the quadratic
    /// tail), and sooner the further past, for every preset tolerance.
    #[test]
    fn pitch_just_past_the_cents_band_switches_promptly() {
        for tol in [0.0f32, 5.0, 8.0, 15.0, 40.0] {
            let one = switch_hops(tol, 1.0) as f32 * HOP_MS;
            let four = switch_hops(tol, 4.0) as f32 * HOP_MS;
            let far = switch_hops(tol, 40.0) as f32 * HOP_MS;
            assert!(one < 25.0, "tol {tol}: 1 c past took {one} ms");
            assert!(four < 10.0, "tol {tol}: 4 c past took {four} ms");
            assert!(far < 6.0, "tol {tol}: 40 c past took {far} ms");
            assert!(far <= four && four <= one, "tol {tol}: {one} / {four} / {far} ms");
        }
    }

    #[test]
    fn silence_keeps_note_and_masks_apply() {
        let mut h = NoteHmm::new(HOP_MS, 0);
        h.set_params(8.0, 0.0, MAX_JUMP_HQ);
        feed(&mut h, 6400.0, 100);
        for _ in 0..30 {
            h.step(&HmmObs { rel: 0.0, confidence: 0.0, voiced: false, near: false });
        }
        assert_eq!(top(&h, 0.0), 64);
        // Re-voicing near the same note stays there.
        feed(&mut h, 6420.0, 3);
        assert_eq!(top(&h, 6420.0), 64);
        // Removing the note moves the mass to legal neighbours.
        let mut t = TargetSet::default();
        let mut states = [skato_core::NoteState::On; 12];
        states[4] = skato_core::NoteState::Remove;
        t.set_scale(&states);
        h.set_targets(&t);
        feed(&mut h, 6420.0, 3);
        assert_eq!(top(&h, 6420.0), 65);
        assert_eq!(h.prob(64), 0.0);
    }

    #[test]
    fn viterbi_fixed_lag_decides_ahead() {
        let lag = 20;
        let mut h = NoteHmm::new(HOP_MS, lag);
        h.set_params(8.0, 0.0, MAX_JUMP_HQ);
        feed(&mut h, 6000.0, 100);
        // Step to the next note; with the lookahead the path for the frame
        // `lag` hops back switches as soon as the newest frames are decisive,
        // i.e. the decision lands on (or just before) the true change frame.
        feed(&mut h, 6100.0, lag);
        assert_eq!(h.viterbi_note(lag), Some(60));
        assert_eq!(h.viterbi_note(lag - 3), Some(61));
        assert_eq!(h.viterbi_note(0), Some(61));
    }
}
