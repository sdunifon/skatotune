//! Contract tests for the control law (DESIGN §3.3): synthetic
//! `PitchEstimate` sequences in, `ControlOutput`s checked against R§7.6.

mod key_change;
mod midi;
mod note_intent;
mod notes;
mod onsets;
mod retune;
mod robustness;
mod studio;
mod vibrato;
mod voicing;

use skato_core::{LatencyMode, PitchEstimate, Settings, cents_to_hz};

use crate::{ControlOutput, Corrector, CorrectorConfig};

pub(crate) const SR: f32 = 48_000.0;
pub(crate) const HOP: usize = 64;
pub(crate) const HOP_S: f32 = HOP as f32 / SR;
pub(crate) const HOP_MS: f32 = HOP_S * 1000.0;

/// Hops in `ms` milliseconds (rounded).
pub(crate) fn hops(ms: f32) -> usize {
    (ms / HOP_MS).round() as usize
}

pub(crate) fn config(mode: LatencyMode, lookahead_hops: usize) -> CorrectorConfig {
    CorrectorConfig { sample_rate: SR, hop: HOP, latency_mode: mode, lookahead_hops }
}

/// Live corrector with `edit` applied to the default settings.
pub(crate) fn live(edit: impl FnOnce(&mut Settings)) -> Corrector {
    with_config(config(LatencyMode::Live, 0), edit)
}

pub(crate) fn with_config(cfg: CorrectorConfig, edit: impl FnOnce(&mut Settings)) -> Corrector {
    let mut c = Corrector::new(cfg);
    let mut s = Settings::default();
    edit(&mut s);
    c.set_settings(&s);
    c
}

/// A clean voiced estimate at `cents`.
pub(crate) fn est(cents: f32) -> PitchEstimate {
    let f0 = cents_to_hz(cents);
    PitchEstimate {
        sample_pos: 0,
        info_lag: 0.0,
        f0_hz: f0,
        cents,
        period: SR / f0,
        confidence: 0.9,
        voiced_prob: 0.95,
        voiced: true,
        sibilant: false,
        level_db: -20.0,
    }
}

pub(crate) fn unvoiced() -> PitchEstimate {
    PitchEstimate { level_db: -80.0, ..PitchEstimate::default() }
}

/// Run `n` hops with the estimate for hop `i` given by `f(i)`.
pub(crate) fn run(c: &mut Corrector, n: usize, mut f: impl FnMut(usize) -> PitchEstimate) -> Vec<ControlOutput> {
    (0..n).map(|i| c.process_hop(&f(i), None)).collect()
}

/// Run `n` hops at a constant pitch.
pub(crate) fn hold(c: &mut Corrector, cents: f32, n: usize) -> Vec<ControlOutput> {
    run(c, n, |_| est(cents))
}

/// Half of the peak-to-peak range.
pub(crate) fn half_range(xs: impl IntoIterator<Item = f32>) -> f32 {
    let (lo, hi) = xs.into_iter().fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)));
    0.5 * (hi - lo)
}

pub(crate) fn mean(xs: impl IntoIterator<Item = f32>) -> f32 {
    let (s, n) = xs.into_iter().fold((0.0f64, 0usize), |(s, n), x| (s + f64::from(x), n + 1));
    (s / n.max(1) as f64) as f32
}

/// Number of times the target note changes between consecutive voiced frames.
pub(crate) fn target_changes(outs: &[ControlOutput]) -> usize {
    outs.windows(2)
        .filter(|w| w[0].frame.target != 0.0 && w[1].frame.target != 0.0 && (w[0].frame.target - w[1].frame.target).abs() > 1.0)
        .count()
}
