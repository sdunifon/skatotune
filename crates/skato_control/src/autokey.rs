//! Auto-Key: the key of the song from the sung pitch (#6,
//! docs/research/auto-key.md).
//!
//! **Evidence.** Every voiced hop adds the vibrato-free pitch centre to a
//! 12-bin pitch-class histogram, weighted by the detector's confidence, by
//! how steady the centre is (glides and scoops count little) and by how
//! close it sits to a semitone (a note 40 c off the grid counts about a
//! third). The histogram decays with a time constant of *sung* time, so
//! silence keeps it and a modulation takes over within a few phrases. The
//! singer's overall tuning offset can be tracked and removed first, and the
//! last steady note of each phrase is counted separately, because phrases
//! tend to end on the tonic.
//!
//! **Model.** The candidates are pitch-class sets: the 12 diatonic sets, the
//! 12 harmonic-minor and 12 melodic-minor sets, and optionally the 12
//! pentatonic sets. A sung note falls in the set with probability 1 − ε,
//! spread evenly over its notes, or outside it with probability ε. Each set
//! is scored by its likelihood of the histogram, which trades covering the
//! sung notes against being specific, plus a prior per family and a small
//! bonus for how well its best-named key matches a key profile (which breaks
//! ties between sets that explain the notes equally well). The confidence is
//! the posterior of the sets that agree with the best one on every pitch
//! class that has actually been sung: a tune that never sings the seventh
//! can't tell A minor from A harmonic minor, but that ambiguity doesn't make
//! the sung notes uncertain. Correction only needs the set (C major and A
//! minor correct identically). The tonic used to name the set comes from key
//! profiles and the phrase-final notes.
//!
//! **Following.** The proposal is the best set. The confirmed key, which the
//! Detect key source follows, changes only after a new set has been the
//! confident proposal for several seconds, and only between phrases.
//!
//! Realtime safe: fixed-size state, no allocation, a few hundred operations
//! per update (every ~20 ms).

use skato_core::ScaleKind;

/// A named key: tonic plus built-in scale, with its pitch-class set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KeyName {
    /// Tonic, 0 = C .. 11 = B.
    pub root: u8,
    pub kind: ScaleKind,
    /// The pitch-class set, bit 0 = C (always `kind.mask(root)`).
    pub mask: u16,
}

impl KeyName {
    pub fn new(root: u8, kind: ScaleKind) -> Self {
        let root = root % 12;
        Self { root, kind, mask: kind.mask(root) }
    }
}

/// What Auto-Key believes at the moment.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct KeyReading {
    /// Best key so far (None before there is any steady singing).
    pub proposal: Option<KeyName>,
    /// Posterior probability of the proposal's pitch-class set, 0-1.
    pub confidence: f32,
    /// Steady singing behind the estimate, seconds (decayed).
    pub evidence_s: f32,
    /// The key the Detect key source follows (None until confirmed).
    pub confirmed: Option<KeyName>,
    /// `confirmed` was restored from a saved project and the singing has not
    /// confirmed it since.
    pub saved: bool,
}

/// Key profiles for naming the tonic (major, minor; index 0 = tonic).
/// Values as tabulated by the Humdrum `keycor` tool and music21.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// Krumhansl & Kessler (1982), probe-tone ratings.
    KrumhanslKessler,
    /// Temperley (2007), Kostka-Payne corpus.
    Temperley,
    /// Aarden (2003), Essen folksong melodies.
    Aarden,
    /// Bellman (2005) / Budge (1943), chord-frequency based.
    BellmanBudge,
}

impl Profile {
    pub const ALL: [Profile; 4] = [Profile::KrumhanslKessler, Profile::Temperley, Profile::Aarden, Profile::BellmanBudge];

    fn weights(self) -> (&'static [f32; 12], &'static [f32; 12]) {
        match self {
            Profile::KrumhanslKessler => (
                &[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                &[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
            ),
            Profile::Temperley => (
                &[0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400],
                &[0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330],
            ),
            Profile::Aarden => (
                &[17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624, 8.15494, 0.232998, 4.95122],
                &[18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362, 0.702494, 18.6161, 4.56621, 1.93186, 7.37619, 1.75623],
            ),
            Profile::BellmanBudge => (
                &[16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57],
                &[18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21],
            ),
        }
    }
}

/// Families of candidate pitch-class sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// The 7-note major-scale sets (and their modes).
    Diatonic,
    HarmonicMinor,
    MelodicMinor,
    /// The 5-note major/minor pentatonic sets.
    Pentatonic,
}

impl Family {
    pub const ALL: [Family; 4] = [Family::Diatonic, Family::HarmonicMinor, Family::MelodicMinor, Family::Pentatonic];

    /// The scale that generates the family's set at a family root.
    fn generator(self) -> ScaleKind {
        match self {
            Family::Diatonic => ScaleKind::Major,
            Family::HarmonicMinor => ScaleKind::HarmonicMinor,
            Family::MelodicMinor => ScaleKind::MelodicMinor,
            Family::Pentatonic => ScaleKind::MajorPentatonic,
        }
    }

    /// Names a set of this family can take: (semitones above the family
    /// root, scale), each scored with the major or minor profile.
    fn names(self) -> &'static [(u8, ScaleKind, bool)] {
        match self {
            Family::Diatonic => &[(0, ScaleKind::Major, true), (9, ScaleKind::NaturalMinor, false)],
            Family::HarmonicMinor => &[(0, ScaleKind::HarmonicMinor, false)],
            Family::MelodicMinor => &[(0, ScaleKind::MelodicMinor, false)],
            Family::Pentatonic => &[(0, ScaleKind::MajorPentatonic, true), (9, ScaleKind::MinorPentatonic, false)],
        }
    }
}

/// Tuning of the detector. [`KeyConfig::default`] is the shipped setting;
/// the evaluation harness sweeps the rest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyConfig {
    /// Evidence time constant, seconds of steady singing.
    pub tau_s: f32,
    /// A vibrato-free centre moving this fast (cents per ms, over 20 ms)
    /// counts half: glides and scoops count little, real held notes (which
    /// drift and wobble) mostly count.
    pub slope_half: f32,
    /// Probability that a sung note lies outside the key.
    pub eps: f32,
    /// Evidence counts per second of steady singing (sets how fast the
    /// confidence rises; calibrated so confidence tracks accuracy).
    pub rate: f32,
    /// Prior weight per [`Family`] (in `Family::ALL` order; 0 excludes it).
    pub prior: [f32; 4],
    /// Remove the singer's overall tuning offset before counting notes.
    pub offset_track: bool,
    /// Profile used to name the tonic.
    pub profile: Profile,
    /// Weight of the phrase-final notes when naming the tonic.
    pub final_weight: f32,
    /// Bonus (log units per unit of profile correlation) for a set whose
    /// best-named key matches the profile; breaks ties between equally
    /// likely sets.
    pub tonal_bias: f32,
    /// A pitch class counts as sung once it holds this share of the
    /// evidence (sets that differ only on unsung notes agree).
    pub sung_share: f32,
    /// A proposal at least this confident can become the confirmed key...
    pub confirm_conf: f32,
    /// ...after leading this long when nothing is confirmed yet...
    pub confirm_first_s: f32,
    /// ...or this long to replace a confirmed key...
    pub confirm_change_s: f32,
    /// ...and only during a pause in the singing at least this long.
    pub gap_ms: f32,
}

impl Default for KeyConfig {
    /// Chosen on the DEV split by `skato_eval autokey` (synthetic melodies
    /// in 36 keys and vocadito DEV; docs/research/auto-key.md §4).
    fn default() -> Self {
        Self {
            tau_s: 120.0,
            slope_half: 4.0,
            eps: 0.1,
            rate: 4.0,
            prior: [1.0, 0.3, 0.1, 0.0],
            offset_track: true,
            profile: Profile::Temperley,
            final_weight: 0.5,
            tonal_bias: 3.0,
            sung_share: 0.0,
            confirm_conf: 0.8,
            confirm_first_s: 3.0,
            confirm_change_s: 8.0,
            gap_ms: 150.0,
        }
    }
}

/// Steadiness is judged over this span (hop-to-hop detector jitter of a
/// few cents would otherwise read as a fast glide).
const SLOPE_SPAN_MS: f32 = 20.0;
/// Capacity of the recent-centre ring (≥ the span in hops at 22.05 kHz).
const RECENT: usize = 64;
/// A run of one steady pitch class at least this long can end a phrase.
const FINAL_MIN_S: f32 = 0.15;
/// A pause this long ends a phrase (its last steady note is counted).
const PHRASE_GAP_MS: f32 = 300.0;
/// The estimate is refreshed this often.
const UPDATE_MS: f32 = 20.0;
/// The tracked singer offset is used once its resultant length passes this.
const OFFSET_MIN_R: f32 = 0.3;
/// Number of candidate sets (12 roots per family).
const N_CAND: usize = 12 * Family::ALL.len();

/// The Auto-Key detector. One per corrector; it survives engine resets (the
/// key of a song does not change when the transport stops), and its whole
/// state can be copied into a new engine.
#[derive(Clone, Debug)]
pub struct KeyDetector {
    cfg: KeyConfig,
    hop_s: f32,
    /// Per-hop decay of the evidence while singing.
    decay: f32,
    update_hops: u32,
    hist: [f32; 12],
    finals: [f32; 12],
    /// Decayed weighted sums of cos/sin of the grid deviation (singer offset).
    off_cos: f32,
    off_sin: f32,
    off_w: f32,
    /// Recent voiced centres relative to the grid, cents (a ring), how many
    /// are valid since the last break, and the span in hops.
    recent: [f32; RECENT],
    recent_head: usize,
    recent_len: usize,
    span_hops: usize,
    /// Current run of one steady pitch class, and the last steady note.
    run_pc: Option<u8>,
    run_s: f32,
    last_steady: Option<u8>,
    /// Consecutive unvoiced hops.
    gap_hops: u32,
    since_update: u32,
    /// Log-likelihood per sung note, in and out of a set of size k.
    lq_in: [f32; 13],
    lq_out: [f32; 13],
    /// Candidate sets: mask, family, family root.
    cand: [(u16, Family, u8); N_CAND],
    n_cand: usize,
    reading: KeyReading,
    /// Diagnostics: voiced hops, and the sums of their confidence,
    /// steadiness and in-tune weights.
    diag: [f32; 4],
    /// The confident proposal being timed for confirmation.
    pending: u16,
    pending_s: f32,
}

impl KeyDetector {
    pub fn new(hop_s: f32) -> Self {
        Self::with_config(hop_s, KeyConfig::default())
    }

    pub fn with_config(hop_s: f32, cfg: KeyConfig) -> Self {
        let hop_s = if hop_s.is_finite() && hop_s > 0.0 { hop_s } else { 64.0 / 48_000.0 };
        let mut d = Self {
            cfg,
            hop_s,
            decay: 1.0,
            update_hops: ((UPDATE_MS * 1e-3 / hop_s).round() as u32).max(1),
            hist: [0.0; 12],
            finals: [0.0; 12],
            off_cos: 0.0,
            off_sin: 0.0,
            off_w: 0.0,
            recent: [0.0; RECENT],
            recent_head: 0,
            recent_len: 0,
            span_hops: ((SLOPE_SPAN_MS * 1e-3 / hop_s).round() as usize).clamp(1, RECENT - 1),
            run_pc: None,
            run_s: 0.0,
            last_steady: None,
            gap_hops: 0,
            since_update: 0,
            lq_in: [0.0; 13],
            lq_out: [0.0; 13],
            cand: [(0, Family::Diatonic, 0); N_CAND],
            n_cand: 0,
            reading: KeyReading::default(),
            diag: [0.0; 4],
            pending: 0,
            pending_s: 0.0,
        };
        d.set_config(cfg);
        d
    }

    /// Change the tuning (keeps the evidence and the confirmed key).
    pub fn set_config(&mut self, cfg: KeyConfig) {
        self.cfg = cfg;
        self.decay = if cfg.tau_s.is_finite() && cfg.tau_s > 0.0 { (-self.hop_s / cfg.tau_s).exp() } else { 1.0 };
        let eps = cfg.eps.clamp(1e-4, 0.5);
        for k in 1..12 {
            self.lq_in[k] = ((1.0 - eps) / k as f32).ln();
            self.lq_out[k] = (eps / (12 - k) as f32).ln();
        }
        self.n_cand = 0;
        for (fi, fam) in Family::ALL.into_iter().enumerate() {
            if cfg.prior[fi] > 0.0 {
                for r in 0..12u8 {
                    self.cand[self.n_cand] = (fam.generator().mask(r), fam, r);
                    self.n_cand += 1;
                }
            }
        }
    }

    pub fn config(&self) -> KeyConfig {
        self.cfg
    }

    /// Forget the evidence and the confirmed key (the editor's re-listen).
    pub fn clear(&mut self) {
        let cfg = self.cfg;
        *self = Self::with_config(self.hop_s, cfg);
    }

    /// Restore a confirmed key saved with the project. It stays in force
    /// (marked saved) until the singing confirms it or replaces it.
    pub fn seed_confirmed(&mut self, key: KeyName) {
        self.reading.confirmed = Some(key);
        self.reading.saved = true;
    }

    pub fn reading(&self) -> KeyReading {
        self.reading
    }

    /// Diagnostics: mean confidence, steadiness and in-tune weight of the
    /// voiced hops so far, and their number.
    #[doc(hidden)]
    pub fn weight_means(&self) -> (f32, f32, f32, f32) {
        let n = self.diag[0].max(1.0);
        (self.diag[1] / n, self.diag[2] / n, self.diag[3] / n, self.diag[0])
    }

    /// Evaluation: the proposal and confidence this detector's tuning gives
    /// for some evidence (as `evidence()` returns it).
    #[doc(hidden)]
    pub fn estimate(&self, hist: &[f32; 12], finals: &[f32; 12]) -> Option<(KeyName, f32)> {
        score_sets(hist, finals, &self.cand[..self.n_cand], &self.lq_in, &self.lq_out, &self.cfg)
    }

    /// The evidence: decayed seconds of steady singing per pitch class, and
    /// phrase-final note counts (evaluation and diagnostics).
    pub fn evidence(&self) -> ([f32; 12], [f32; 12]) {
        (self.hist, self.finals)
    }

    /// The singer's tuning offset from the grid, cents (0 until it is clear).
    pub fn singer_offset(&self) -> f32 {
        if !self.cfg.offset_track || self.off_w <= 0.0 {
            return 0.0;
        }
        let r = (self.off_cos * self.off_cos + self.off_sin * self.off_sin).sqrt() / self.off_w;
        if r < OFFSET_MIN_R {
            return 0.0;
        }
        self.off_sin.atan2(self.off_cos) / std::f32::consts::TAU * 100.0
    }

    /// One voiced hop: `rel_cents` is the vibrato-free centre relative to the
    /// note grid (note n at 100·n), `confidence` the detector's, `guard` an
    /// octave-guarded frame (no new pitch information).
    pub fn voiced(&mut self, rel_cents: f32, confidence: f32, guard: bool) {
        self.gap_hops = 0;
        if guard || !rel_cents.is_finite() {
            self.recent_len = 0;
            self.tick();
            return;
        }
        // Slope over the span (a fresh run counts as unsteady until it has
        // a span of history: onset scoops).
        let slope = if self.recent_len >= self.span_hops {
            let old = self.recent[(self.recent_head + RECENT - self.span_hops) % RECENT];
            (rel_cents - old).abs() / (self.span_hops as f32 * self.hop_s * 1e3)
        } else {
            f32::INFINITY
        };
        self.recent[self.recent_head] = rel_cents;
        self.recent_head = (self.recent_head + 1) % RECENT;
        self.recent_len = (self.recent_len + 1).min(RECENT);
        let steady = 1.0 / (1.0 + (slope / self.cfg.slope_half.max(1e-3)).powi(2));
        let conf = if confidence.is_finite() { confidence.clamp(0.0, 1.0) } else { 0.0 };

        // Singer offset: circular mean of the deviation from the grid.
        let raw = rel_cents / 100.0;
        let dev = raw - raw.round();
        let ow = conf * steady * self.hop_s;
        let angle = dev * std::f32::consts::TAU;
        self.off_cos = self.off_cos * self.decay + ow * angle.cos();
        self.off_sin = self.off_sin * self.decay + ow * angle.sin();
        self.off_w = self.off_w * self.decay + ow;

        let x = (rel_cents - self.singer_offset()) / 100.0;
        let n = x.round();
        let d = x - n;
        let in_tune = (1.0 - 4.0 * d * d).max(0.0);
        let pc = (n as i64).rem_euclid(12) as usize;
        for h in self.hist.iter_mut().chain(self.finals.iter_mut()) {
            *h *= self.decay;
        }
        self.hist[pc] += conf * steady * in_tune * self.hop_s;
        self.diag[0] += 1.0;
        self.diag[1] += conf;
        self.diag[2] += steady;
        self.diag[3] += in_tune;

        // Phrase-final notes: the last steady run of one pitch class.
        if steady * in_tune > 0.5 {
            if self.run_pc == Some(pc as u8) {
                self.run_s += self.hop_s;
            } else {
                self.run_pc = Some(pc as u8);
                self.run_s = self.hop_s;
            }
            if self.run_s >= FINAL_MIN_S {
                self.last_steady = Some(pc as u8);
            }
        }
        self.tick();
    }

    /// One unvoiced hop.
    pub fn unvoiced(&mut self) {
        self.recent_len = 0;
        self.gap_hops = self.gap_hops.saturating_add(1);
        let phrase_gap = ((PHRASE_GAP_MS * 1e-3 / self.hop_s).round() as u32).max(1);
        if self.gap_hops == phrase_gap {
            if let Some(pc) = self.last_steady.take() {
                self.finals[pc as usize] += 1.0;
            }
            self.run_pc = None;
            self.run_s = 0.0;
        }
        self.tick();
    }

    fn tick(&mut self) {
        self.since_update += 1;
        if self.since_update >= self.update_hops {
            let dt = self.since_update as f32 * self.hop_s;
            self.since_update = 0;
            self.update(dt);
        }
    }

    /// Refresh the proposal and the confirmed key; `dt` = time since the last
    /// update.
    fn update(&mut self, dt: f32) {
        let total: f32 = self.hist.iter().sum();
        self.reading.evidence_s = total;
        let best = score_sets(&self.hist, &self.finals, &self.cand[..self.n_cand], &self.lq_in, &self.lq_out, &self.cfg);
        match best {
            Some((name, conf)) => {
                self.reading.proposal = Some(name);
                self.reading.confidence = conf;
            }
            None => {
                self.reading.proposal = None;
                self.reading.confidence = 0.0;
            }
        }
        self.follow(dt);
    }

    fn follow(&mut self, dt: f32) {
        let confident = self.reading.proposal.filter(|_| self.reading.confidence >= self.cfg.confirm_conf);
        match confident {
            Some(p) if p.mask == self.pending => self.pending_s += dt,
            Some(p) => {
                self.pending = p.mask;
                self.pending_s = dt;
            }
            None => {
                self.pending = 0;
                self.pending_s = 0.0;
            }
        }
        let Some(p) = confident else { return };
        let confirmed = self.reading.confirmed.map(|c| c.mask);
        if confirmed == Some(p.mask) {
            if self.reading.saved && self.pending_s >= self.cfg.confirm_first_s {
                self.reading.saved = false;
            }
            return;
        }
        let need = if confirmed.is_none() { self.cfg.confirm_first_s } else { self.cfg.confirm_change_s };
        let between_phrases = self.gap_hops as f32 * self.hop_s * 1e3 >= self.cfg.gap_ms;
        if self.pending_s >= need && between_phrases {
            self.reading.confirmed = Some(p);
            self.reading.saved = false;
        }
    }
}

/// Best candidate key for the evidence and its confidence (the posterior of
/// the sets that agree with it on every sung pitch class), or None without
/// evidence. `cand` = (mask, family, family root).
pub fn score_sets(
    hist: &[f32; 12],
    finals: &[f32; 12],
    cand: &[(u16, Family, u8)],
    lq_in: &[f32; 13],
    lq_out: &[f32; 13],
    cfg: &KeyConfig,
) -> Option<(KeyName, f32)> {
    let total: f32 = hist.iter().sum();
    // Also rejects a NaN total.
    if total.is_nan() || total <= 1e-3 || cand.is_empty() {
        return None;
    }
    let mut best = (0, f32::NEG_INFINITY, KeyName::new(0, ScaleKind::Major));
    let mut lls = [0.0f32; N_CAND];
    for (i, &(mask, fam, root)) in cand.iter().enumerate() {
        let k = mask.count_ones() as usize;
        let h_in: f32 = (0..12).filter(|pc| mask & (1 << pc) != 0).map(|pc| hist[pc]).sum();
        let fi = Family::ALL.iter().position(|f| *f == fam).unwrap_or(0);
        let (name, fit) = name_set(hist, finals, mask, fam, root, cfg);
        let ll = cfg.rate * (h_in * (lq_in[k] - lq_out[k]) + total * lq_out[k])
            + cfg.prior[fi].max(1e-9).ln()
            + cfg.tonal_bias * fit;
        lls[i] = ll;
        if ll > best.1 {
            best = (i, ll, name);
        }
    }
    let sung = (0..12).filter(|&pc| hist[pc] >= cfg.sung_share * total).fold(0u16, |m, pc| m | 1 << pc);
    let best_mask = cand[best.0].0;
    let (mut agree, mut all) = (0.0f32, 0.0f32);
    for (i, &(mask, _, _)) in cand.iter().enumerate() {
        let p = (lls[i] - best.1).exp();
        all += p;
        if (mask ^ best_mask) & sung == 0 {
            agree += p;
        }
    }
    Some((best.2, (agree / all.max(1e-30)).clamp(0.0, 1.0)))
}

/// Name a candidate set: the tonic and scale among the family's names whose
/// profile best matches the histogram, with the phrase-final notes as a
/// bonus on the tonic. Returns the name and its profile correlation.
pub fn name_set(hist: &[f32; 12], finals: &[f32; 12], mask: u16, fam: Family, root: u8, cfg: &KeyConfig) -> (KeyName, f32) {
    let (major, minor) = cfg.profile.weights();
    let final_total: f32 = finals.iter().sum();
    let mut best: Option<(f32, f32, KeyName)> = None;
    for &(off, kind, is_major) in fam.names() {
        let tonic = (root + off) % 12;
        debug_assert_eq!(kind.mask(tonic), mask);
        let profile = if is_major { major } else { minor };
        let fit = correlation(hist, profile, tonic);
        let mut score = fit;
        if final_total > 0.0 {
            score += cfg.final_weight * finals[tonic as usize] / final_total;
        }
        if best.is_none_or(|(s, _, _)| score > s) {
            best = Some((score, fit, KeyName::new(tonic, kind)));
        }
    }
    best.map_or((KeyName::new(root, fam.generator()), 0.0), |(_, fit, k)| (k, fit))
}

/// Pearson correlation between the histogram and the profile with its tonic
/// on pitch class `tonic`.
pub fn correlation(hist: &[f32; 12], profile: &[f32; 12], tonic: u8) -> f32 {
    let mh = hist.iter().sum::<f32>() / 12.0;
    let mp = profile.iter().sum::<f32>() / 12.0;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (i, &p) in profile.iter().enumerate() {
        let x = hist[(tonic as usize + i) % 12] - mh;
        let y = p - mp;
        sxy += x * y;
        sxx += x * x;
        syy += y * y;
    }
    if sxx > 0.0 && syy > 0.0 { sxy / (sxx * syy).sqrt() } else { 0.0 }
}

/// Classic Krumhansl-Schmuckler key finding, for comparison: the best of the
/// 24 major/minor keys by profile correlation, and the margin to the second.
pub fn profile_key(hist: &[f32; 12], profile: Profile) -> (KeyName, f32) {
    let (major, minor) = profile.weights();
    let mut scores = [(f32::NEG_INFINITY, KeyName::new(0, ScaleKind::Major)); 2];
    for tonic in 0..12u8 {
        for (w, kind) in [(major, ScaleKind::Major), (minor, ScaleKind::NaturalMinor)] {
            let s = correlation(hist, w, tonic);
            if s > scores[0].0 {
                scores[1] = scores[0];
                scores[0] = (s, KeyName::new(tonic, kind));
            } else if s > scores[1].0 {
                scores[1] = (s, KeyName::new(tonic, kind));
            }
        }
    }
    (scores[0].1, scores[0].0 - scores[1].0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOP_S: f32 = 64.0 / 48_000.0;

    /// Sing `notes` (MIDI numbers) for `dur` seconds each with a short breath
    /// between them, `detune` cents off the grid.
    fn sing(d: &mut KeyDetector, notes: &[i32], dur: f32, detune: f32) {
        for &n in notes {
            for _ in 0..(dur / HOP_S) as usize {
                d.voiced(100.0 * n as f32 + detune, 0.9, false);
            }
            for _ in 0..(0.08 / HOP_S) as usize {
                d.unvoiced();
            }
        }
    }

    fn pause(d: &mut KeyDetector, s: f32) {
        for _ in 0..(s / HOP_S) as usize {
            d.unvoiced();
        }
    }

    /// A tune in A minor (natural): A B C D E F G, ending phrases on A.
    const A_MINOR: [i32; 16] = [69, 71, 72, 74, 76, 74, 72, 71, 69, 72, 76, 77, 79, 76, 71, 69];
    /// The same tune moved to D minor.
    const D_MINOR: [i32; 16] = [62, 64, 65, 67, 69, 67, 65, 64, 62, 65, 69, 70, 72, 69, 64, 62];
    /// An A minor tune that never sings the seventh (G or G#).
    const NO_SEVENTH: [i32; 16] = [69, 71, 72, 74, 76, 74, 72, 71, 69, 72, 76, 77, 76, 74, 71, 69];

    #[test]
    fn a_minor_tune_is_named_a_minor_and_confirmed_between_phrases() {
        let mut d = KeyDetector::new(HOP_S);
        assert_eq!(d.reading().proposal, None);
        for _ in 0..4 {
            sing(&mut d, &A_MINOR, 0.4, 0.0);
            pause(&mut d, 0.5);
        }
        let r = d.reading();
        let want = KeyName::new(9, ScaleKind::NaturalMinor);
        assert_eq!(r.proposal, Some(want), "{r:?}");
        assert!(r.confidence > 0.9, "{r:?}");
        assert_eq!(r.confirmed, Some(want), "{r:?}");
        assert!(!r.saved);
    }

    #[test]
    fn a_tune_that_never_sings_the_seventh_is_proposed_but_not_confirmed() {
        // A natural minor and A harmonic minor explain it equally well: the
        // proposal is the more common diatonic set, but not confidently.
        let mut d = KeyDetector::new(HOP_S);
        for _ in 0..6 {
            sing(&mut d, &NO_SEVENTH, 0.4, 0.0);
            pause(&mut d, 0.5);
        }
        let r = d.reading();
        assert_eq!(r.proposal.map(|k| k.mask), Some(ScaleKind::NaturalMinor.mask(9)), "{r:?}");
        assert!(r.confidence < 0.8, "{r:?}");
        assert_eq!(r.confirmed, None);
        // Counting sets that differ only on unsung notes as agreeing makes
        // it confident about what was sung.
        let mut d = KeyDetector::with_config(HOP_S, KeyConfig { sung_share: 0.03, ..KeyConfig::default() });
        for _ in 0..6 {
            sing(&mut d, &NO_SEVENTH, 0.4, 0.0);
            pause(&mut d, 0.5);
        }
        assert!(d.reading().confidence > 0.9, "{:?}", d.reading());
    }

    #[test]
    fn a_detuned_singer_is_still_counted_on_the_right_notes() {
        let mut d = KeyDetector::new(HOP_S);
        for _ in 0..4 {
            sing(&mut d, &A_MINOR, 0.4, 42.0);
            pause(&mut d, 0.5);
        }
        assert!((d.singer_offset() - 42.0).abs() < 3.0, "{}", d.singer_offset());
        assert_eq!(d.reading().proposal.map(|k| k.mask), Some(ScaleKind::NaturalMinor.mask(9)));
    }

    #[test]
    fn the_confirmed_key_changes_only_after_the_new_key_leads_and_between_phrases() {
        let mut d = KeyDetector::new(HOP_S);
        for _ in 0..4 {
            sing(&mut d, &A_MINOR, 0.4, 0.0);
            pause(&mut d, 0.5);
        }
        let a_minor = d.reading().confirmed.expect("confirmed");
        // Modulate. The confirmed key only ever changes in the pauses, never
        // while a phrase is sung (its notes' 80 ms breaths are too short).
        let mut changed_after = None;
        for phrase in 0..30 {
            let before = d.reading().confirmed;
            sing(&mut d, &D_MINOR, 0.4, 0.0);
            assert_eq!(d.reading().confirmed, before, "changed while singing phrase {phrase}");
            pause(&mut d, 0.5);
            if changed_after.is_none() && d.reading().confirmed != Some(a_minor) {
                changed_after = Some(phrase);
            }
        }
        let r = d.reading();
        assert_eq!(r.confirmed.map(|k| k.mask), Some(ScaleKind::NaturalMinor.mask(2)), "{r:?}");
        // ...and only after the new key has led for 8 s (a phrase is ~7.7 s).
        assert!(changed_after.is_some_and(|p| p >= 1), "{changed_after:?}");
    }

    #[test]
    fn harmonic_minor_needs_the_raised_seventh() {
        let mut d = KeyDetector::new(HOP_S);
        // A harmonic minor: G# instead of G, used often.
        let tune = [69, 71, 72, 74, 76, 77, 80, 81, 80, 76, 74, 72, 71, 80, 69, 69];
        for _ in 0..4 {
            sing(&mut d, &tune, 0.4, 0.0);
            pause(&mut d, 0.5);
        }
        assert_eq!(d.reading().proposal, Some(KeyName::new(9, ScaleKind::HarmonicMinor)));
    }

    #[test]
    fn a_saved_key_holds_until_the_singing_replaces_it() {
        let mut d = KeyDetector::new(HOP_S);
        let saved = KeyName::new(0, ScaleKind::Major);
        d.seed_confirmed(saved);
        assert_eq!(d.reading().confirmed, Some(saved));
        assert!(d.reading().saved);
        sing(&mut d, &D_MINOR[..4], 0.3, 0.0);
        assert_eq!(d.reading().confirmed, Some(saved), "a few notes don't replace it");
        for _ in 0..12 {
            sing(&mut d, &D_MINOR, 0.4, 0.0);
            pause(&mut d, 0.5);
        }
        assert_eq!(d.reading().confirmed.map(|k| k.mask), Some(ScaleKind::NaturalMinor.mask(2)));
        assert!(!d.reading().saved);
    }

    #[test]
    fn glides_count_little_and_silence_keeps_the_evidence() {
        let mut d = KeyDetector::new(HOP_S);
        // Fast transitions (300 c sweeps at 8 c/ms, like scoops and
        // portamento) add little evidence over 2 s.
        let per_hop = 8.0 * HOP_S * 1e3;
        for i in 0..(2.0 / HOP_S) as usize {
            d.voiced(6000.0 + (i as f32 * per_hop) % 300.0, 0.9, false);
        }
        let glide = d.evidence().0.iter().sum::<f32>();
        assert!(glide < 0.3, "{glide}");
        sing(&mut d, &A_MINOR, 0.4, 0.0);
        let before = d.evidence().0.iter().sum::<f32>();
        pause(&mut d, 30.0);
        let after = d.evidence().0.iter().sum::<f32>();
        assert_eq!(before, after);
    }

    /// Cost per hop on the audio thread (run with `--release --ignored
    /// --nocapture`): singing with vibrato, updates every 20 ms included.
    #[test]
    #[ignore = "timing; run in release"]
    fn cost_per_hop() {
        let mut d = KeyDetector::new(HOP_S);
        let n = 2_000_000;
        let t0 = std::time::Instant::now();
        for i in 0..n {
            let t = i as f32 * HOP_S;
            let note = [69.0, 71.0, 72.0, 74.0][(t / 0.4) as usize % 4];
            if (t / 0.4).fract() < 0.9 {
                d.voiced(100.0 * note + 30.0 * (std::f32::consts::TAU * 5.5 * t).sin(), 0.9, false);
            } else {
                d.unvoiced();
            }
        }
        let per_hop = t0.elapsed().as_secs_f64() / n as f64;
        let r = std::hint::black_box(d.reading());
        println!("Auto-Key: {:.1} ns per hop ({:.4} % of real time at 48 kHz / 64-sample hops); {r:?}", per_hop * 1e9, 100.0 * per_hop / f64::from(HOP_S));
    }

    #[test]
    fn classic_profile_key_finding_agrees_on_a_clear_tune() {
        let mut d = KeyDetector::new(HOP_S);
        for _ in 0..3 {
            sing(&mut d, &A_MINOR, 0.4, 0.0);
            pause(&mut d, 0.5);
        }
        let (key, margin) = profile_key(&d.evidence().0, Profile::Temperley);
        assert_eq!(key.mask, ScaleKind::NaturalMinor.mask(9));
        assert!(margin > 0.0);
    }
}
