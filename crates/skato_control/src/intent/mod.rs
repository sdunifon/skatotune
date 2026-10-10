//! Learned switch veto, the model half of the Labs "learned note hold"
//! (`docs/research/note-intent.md` §5; the other half is
//! [`crate::release`]).
//!
//! **Rule proposes, model vetoes.** The note decision runs unchanged. When it
//! proposes a switch from the held note `cur` to `m` (HQ/Max: the note HMM
//! through the OR tolerance gate), a candidate-note model scores three
//! candidates: the legal note nearest the decided pitch r and its legal
//! neighbours. Each candidate gets a gradient-boosted tree score (300 trees,
//! at most 31 leaves each) over features relative to r ([`features`]); the
//! sigmoids are normalised over the three. The switch is kept when
//! `P(m) − P(cur) ≥ margin` or `P(cur) < unsure`, otherwise the current note
//! is held for this hop and the note decision proposes again on the next.
//! The veto never adds a switch, never touches an onset, and sees only the
//! detector stream. The corrector applies it only to scale targets (MIDI
//! targets bypass it).
//!
//! | mode | model | features | margin / unsure |
//! |---|---|---|---|
//! | Live, Live+, Live-Min | `models/live_hq.skgb` | 82 (causal) | 0.25 / 0.3 |
//! | Studio | `models/studio_hq.skgb` | 104 (13 over the lookahead) | 0 / 0 |
//!
//! Both models were trained on the 20 DEV singers of vocadito plus synthetic
//! singing and chosen on DEV only (`crates/skato_control/models/README.md`).
//! They are tied to the current detector and front end: a change there
//! needs a retrain.
//!
//! Realtime: the models are parsed once per process, on the first
//! `Corrector::new` (outside the audio thread), into shared read-only arrays;
//! the history is allocated in [`NoteIntent::new`]. A veto computes the
//! features and walks 3 × 300 trees, only on hops where a switch is proposed,
//! with no allocation and a fixed bound on the work.

pub(crate) mod features;
pub mod gbt;

pub use features::FEATURE_NAMES;

/// Features per frame of the learned veto (45 causal + 13 lookahead).
pub const FEATURES: usize = features::NF_MAX;

use std::sync::OnceLock;

use skato_core::LatencyMode;

use crate::targets::TargetSet;
use features::{Features, HFrame, NFC_LIVE, NFC_STUDIO};
use gbt::GbtModel;

/// The embedded models (see the module docs and the README beside them).
const LIVE_HQ: &[u8] = include_bytes!("../../models/live_hq.skgb");
const STUDIO_HQ: &[u8] = include_bytes!("../../models/studio_hq.skgb");

/// Live policy (chosen on DEV): keep a switch when the new note leads by
/// 0.25, or when the model gives the current note less than 0.3.
const LIVE_MARGIN: f32 = 0.25;
const LIVE_UNSURE: f32 = 0.3;
/// Studio policy: keep a switch when the new note is at least as likely.
const STUDIO_MARGIN: f32 = 0.0;
const STUDIO_UNSURE: f32 = 0.0;

/// The model for a latency mode, parsed on first use and shared. `None` if
/// the embedded file does not parse (the veto is then off; a test keeps the
/// shipped files valid).
fn model_for(mode: LatencyMode) -> Option<&'static GbtModel> {
    static LIVE: OnceLock<Option<GbtModel>> = OnceLock::new();
    static STUDIO: OnceLock<Option<GbtModel>> = OnceLock::new();
    let (cell, bytes, nf) = match mode {
        LatencyMode::Studio => (&STUDIO, STUDIO_HQ, NFC_STUDIO),
        _ => (&LIVE, LIVE_HQ, NFC_LIVE),
    };
    cell.get_or_init(|| GbtModel::parse(bytes).ok().filter(|m| m.features() == nf)).as_ref()
}

/// The model the veto of a latency mode uses (diagnostics and tests).
pub fn embedded_model(mode: LatencyMode) -> Option<&'static GbtModel> {
    model_for(mode)
}

/// Counters of the learned note hold since the last reset (diagnostics).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoteHoldStats {
    /// Switches the note decision proposed while the model was consulted.
    pub proposals: u64,
    /// Of those, switches the model held back.
    pub vetoed: u64,
    /// Downward switches the release freeze held back.
    pub frozen: u64,
}

/// The learned switch veto: frame history, features and the model.
pub(crate) struct NoteIntent {
    feat: Features,
    model: Option<&'static GbtModel>,
    /// The model reads the lookahead features (Studio).
    studio: bool,
    margin: f32,
    unsure: f32,
    pub stats: NoteHoldStats,
}

impl NoteIntent {
    pub fn new(hop_ms: f32, mode: LatencyMode) -> Self {
        let studio = mode == LatencyMode::Studio;
        let (margin, unsure) = if studio { (STUDIO_MARGIN, STUDIO_UNSURE) } else { (LIVE_MARGIN, LIVE_UNSURE) };
        Self { feat: Features::new(hop_ms), model: model_for(mode), studio, margin, unsure, stats: NoteHoldStats::default() }
    }

    /// Tests: a veto without a model (as if the embedded file were corrupt).
    #[cfg(test)]
    pub fn without_model(hop_ms: f32) -> Self {
        Self { model: None, ..Self::new(hop_ms, LatencyMode::Live) }
    }

    pub fn reset(&mut self) {
        self.feat.reset();
        self.stats = NoteHoldStats::default();
    }

    /// Append the newest front-end frame (every hop, voiced or not).
    #[inline]
    pub fn push(&mut self, f: HFrame) {
        self.feat.push(f);
    }

    /// The note decision proposes `cur → m` for the frame `age` hops old
    /// (the lookahead, if any, is the `age` newer frames). `offset` places
    /// the note grid (note n at `100·n + offset` cents). True when the switch
    /// stands: the model agrees, is unsure, or cannot judge (no model, or the
    /// frame is not in the history).
    pub fn keeps_switch(&mut self, age: usize, cur: usize, m: usize, offset: f32, targets: &TargetSet) -> bool {
        let Some(model) = self.model else { return true };
        if !self.feat.compute(age, age > 0) {
            return true;
        }
        self.stats.proposals += 1;
        let r_rel = self.feat.r - offset;
        let (pm, pc) = self.posterior(model, r_rel, cur, m, targets);
        let keep = pm - pc >= self.margin || pc < self.unsure;
        if !keep {
            self.stats.vetoed += 1;
        }
        keep
    }

    /// P(m) and P(cur) among the three candidates around `r_rel` (0 for a
    /// note that is not a candidate).
    fn posterior(&self, model: &GbtModel, r_rel: f32, cur: usize, m: usize, targets: &TargetSet) -> (f32, f32) {
        let Some(n) = targets.nearest(r_rel) else { return (0.0, 0.0) };
        let cands = [targets.neighbour(n, -1), Some(n), targets.neighbour(n, 1)];
        let mut phi = [0.0f32; NFC_STUDIO];
        let mut p = [0.0f32; 3];
        for (pj, c) in p.iter_mut().zip(cands) {
            let Some(c) = c else { continue };
            features::candidate(&self.feat.feats, self.studio, r_rel, c, targets, &mut phi);
            let sc = model.raw(&phi);
            *pj = if sc.is_finite() { 1.0 / (1.0 + (-sc).exp()) } else { 0.0 };
        }
        let s: f32 = p.iter().sum();
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for v in p.iter_mut() {
            *v *= inv;
        }
        let prob = |x: usize| cands.iter().zip(p).find(|(c, _)| **c == Some(x)).map_or(0.0, |(_, q)| q);
        (prob(m), prob(cur))
    }

    /// Features of the frame `age` hops old, whether or not a switch is
    /// proposed (training-data dumps): (r, features, lookahead hops used,
    /// segment age in hops or `u32::MAX`). None when that frame is unvoiced.
    pub fn features_at(&mut self, age: usize) -> Option<(f32, [f32; features::NF_MAX], usize, u32)> {
        self.feat.compute(age, age > 0).then(|| (self.feat.r, self.feat.feats, self.feat.fut, self.feat.segment_age(age)))
    }

    /// Features of the last proposal (tests).
    #[cfg(test)]
    pub fn last_features(&self) -> &[f32; features::NF_MAX] {
        &self.feat.feats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped models parse, have the documented shape, and are the
    /// files the README describes (checksums).
    #[test]
    fn embedded_models_parse_with_the_documented_shape() {
        let live = embedded_model(LatencyMode::Live).expect("live model");
        let studio = embedded_model(LatencyMode::Studio).expect("studio model");
        assert_eq!((live.features(), live.trees(), live.nodes()), (NFC_LIVE, 300, 18_300));
        assert_eq!((studio.features(), studio.trees(), studio.nodes()), (NFC_STUDIO, 300, 18_300));
        assert_eq!(live.checksum(), LIVE_CHECKSUM, "live_hq.skgb changed: update models/README.md");
        assert_eq!(studio.checksum(), STUDIO_CHECKSUM, "studio_hq.skgb changed: update models/README.md");
        for mode in [LatencyMode::Live, LatencyMode::LivePlus, LatencyMode::LiveMin] {
            assert!(std::ptr::eq(embedded_model(mode).unwrap(), live));
        }
    }

    /// Checksums of the embedded models (also in `models/README.md`).
    const LIVE_CHECKSUM: u64 = 0x384c_1fa2_4660_be2c;
    const STUDIO_CHECKSUM: u64 = 0x73e2_c583_a6cd_1f89;

    fn voiced(p: f32, onset: bool) -> HFrame {
        HFrame { voiced: true, onset, p, p_s: p, p_c: p, centre: p, level: -20.0, conf: 0.9, ..HFrame::default() }
    }

    /// A singer holding the top of a note who drifts just past the midpoint
    /// for a few hops: the model wants to stay (P(cur) ≈ 0.8). A line that
    /// has really moved to the next note is let through (P(m) ≈ 0.9).
    #[test]
    fn a_drift_past_the_midpoint_is_vetoed_and_a_real_move_is_not() {
        let targets = TargetSet::default();
        let mut v = NoteIntent::new(64.0 / 44.1, LatencyMode::Live);
        for i in 0..300 {
            v.push(voiced(6040.0, i == 0));
        }
        for _ in 0..10 {
            v.push(voiced(6058.0, false));
        }
        assert!(!v.keeps_switch(0, 60, 61, 0.0, &targets), "drift");
        assert!(v.last_features().iter().all(|x| x.is_finite()));
        // A detune of the whole grid changes nothing (features are relative).
        let mut d = NoteIntent::new(64.0 / 44.1, LatencyMode::Live);
        for i in 0..300 {
            d.push(voiced(6040.0 + 23.0, i == 0));
        }
        for _ in 0..10 {
            d.push(voiced(6058.0 + 23.0, false));
        }
        assert!(!d.keeps_switch(0, 60, 61, 23.0, &targets));
        assert_eq!(d.last_features(), v.last_features());
        let mut w = NoteIntent::new(64.0 / 44.1, LatencyMode::Live);
        for i in 0..150 {
            w.push(voiced(6010.0, i == 0));
        }
        for _ in 0..150 {
            w.push(voiced(6105.0, false));
        }
        assert!(w.keeps_switch(0, 60, 61, 0.0, &targets), "real move");
        assert_eq!(v.stats, NoteHoldStats { proposals: 1, vetoed: 1, frozen: 0 });
        assert_eq!(w.stats, NoteHoldStats { proposals: 1, vetoed: 0, frozen: 0 });
    }

    /// No history, or an unvoiced decided frame: the switch stands.
    #[test]
    fn without_features_the_switch_stands() {
        let targets = TargetSet::default();
        let mut v = NoteIntent::new(64.0 / 48.0, LatencyMode::Studio);
        assert!(v.keeps_switch(0, 60, 61, 0.0, &targets));
        v.push(HFrame::default());
        assert!(v.keeps_switch(0, 60, 61, 0.0, &targets));
        assert_eq!(v.stats.proposals, 0);
    }
}
