//! Voicing evidence and the voicing state machine (R§8.1, R§8.8).
//!
//! Features are measured on the full-rate conditioned signal over a 10 ms
//! window:
//!
//! * RMS level (dBFS) against the gate (default −55 dBFS) with 6 dB of
//!   hysteresis: the gate opens at `gate_db` and closes below `gate_db − 6`;
//! * sibilance: `E(>4 kHz)/E_total > 0.5` or a zero-crossing rate above
//!   12 000 crossings/s (0.25 per sample at 48 kHz, R§8.1), only while the
//!   gate is open (a silent noise floor is not a sibilant).
//!
//! The state machine turns the per-hop raw decision into the output flag:
//! onset after 2 consecutive voiced hops (≈ 3 ms; the detector can ask for
//! one more after silence and restart the count when the pitch moves), and a
//! short hold (20 ms) across *energetic, non-sibilant* dips in periodicity so
//! a breathy cycle or a voiced consonant does not drop the note (R§8.8
//! "release after a 20-40 ms hold"). Only a segment that has been voiced for
//! a while may hold: a 1-3 hop segment is more often a breath or consonant
//! resonance than a note, and holding it published its pitch for the whole
//! hold. Silence and sibilants release immediately: holding through them
//! would only delay the dry crossfade downstream.

use crate::hist::Mirror;

/// Gate hysteresis, dB (R§7.6.4 "6 dB hysteresis").
pub(crate) const GATE_HYST_DB: f32 = 6.0;
/// High-band energy share above which a frame is sibilant (R§8.1).
const SIBILANT_HF_RATIO: f32 = 0.5;
/// Zero crossings per second above which a frame is sibilant (R§8.1).
const SIBILANT_ZCR_HZ: f32 = 12_000.0;

/// Features of one 10 ms window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Features {
    pub level_db: f32,
    pub hf_ratio: f32,
    pub zcr_hz: f32,
}

impl Features {
    /// Sibilance rule of R§8.1 (gate state is applied by the caller).
    pub(crate) fn sibilant(&self) -> bool {
        self.hf_ratio > SIBILANT_HF_RATIO || self.zcr_hz > SIBILANT_ZCR_HZ
    }
}

/// Measures the window of `len` samples ending at `end` (exclusive) of the
/// conditioned signal `full` and its 4 kHz high-pass `hp`.
pub(crate) fn measure(full: &Mirror, hp: &Mirror, end: i64, len: usize, fs: f32) -> Features {
    let (Some(x), Some(h)) = (full.window(end, len), hp.window(end, len)) else {
        return Features { level_db: -160.0, hf_ratio: 0.0, zcr_hz: 0.0 };
    };
    let e = skato_core::simd::sum_sq(x) as f64;
    let ms = e / len.max(1) as f64;
    let level_db = skato_core::math::lin_to_db(ms.sqrt() as f32);
    let hf_ratio = if e > 1e-20 { (skato_core::simd::sum_sq(h) as f64 / e).min(1.0) as f32 } else { 0.0 };
    let crossings = x.windows(2).filter(|p| (p[0] >= 0.0) != (p[1] >= 0.0)).count();
    let zcr_hz = crossings as f32 * fs / len.max(2) as f32;
    Features { level_db, hf_ratio, zcr_hz }
}

/// Level gate with hysteresis.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Gate {
    open: bool,
}

impl Gate {
    pub(crate) fn update(&mut self, level_db: f32, gate_db: f32) -> bool {
        self.open = if self.open { level_db >= gate_db - GATE_HYST_DB } else { level_db >= gate_db };
        self.open
    }

    pub(crate) fn reset(&mut self) {
        self.open = false;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum State {
    #[default]
    Unvoiced,
    Voiced,
    Hold,
}

/// Onset / hold state machine (see module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct VoicingFsm {
    state: State,
    count: u32,
    onset_hops: u32,
    hold_hops: u32,
    /// Consecutive unvoiced output hops (saturating).
    unvoiced: u32,
    /// Raw-voiced hops of the current voiced segment (saturating).
    age: u32,
    /// A segment may hold only once it is this old (raw-voiced hops).
    min_hold_age: u32,
    /// Extra raw hops an onset after silence needs ([`VoicingFsm::set_onset_extra`]).
    extra: u32,
}

impl VoicingFsm {
    /// Onset after `onset_hops` raw-voiced hops; hold for at most `hold_hops`,
    /// and only in a segment voiced for at least `min_hold_age` raw hops.
    pub(crate) fn new(onset_hops: u32, hold_hops: u32, min_hold_age: u32) -> Self {
        Self {
            state: State::Unvoiced,
            count: 0,
            onset_hops: onset_hops.max(1),
            hold_hops,
            unvoiced: u32::MAX,
            age: 0,
            min_hold_age,
            extra: 0,
        }
    }

    /// Extra raw-voiced hops the next onset needs when it comes after
    /// silence (set per hop by the caller from the evidence).
    pub(crate) fn set_onset_extra(&mut self, extra: u32) {
        self.extra = extra;
    }

    /// Advances one hop. `raw`: this hop is voiced by every criterion;
    /// `holdable`: not voiced, but energetic, non-sibilant and still somewhat
    /// periodic. Returns the voiced flag. `onset` is set on the first voiced
    /// hop after an unvoiced stretch.
    pub(crate) fn update(&mut self, raw: bool, holdable: bool) -> (bool, bool) {
        let was = self.state;
        self.state = match self.state {
            State::Unvoiced => {
                self.count = if raw { self.count + 1 } else { 0 };
                let need = self.onset_hops + if self.after_silence() { self.extra } else { 0 };
                if self.count >= need {
                    self.age = self.count;
                    State::Voiced
                } else {
                    State::Unvoiced
                }
            }
            State::Voiced | State::Hold if raw => {
                self.age = self.age.saturating_add(1);
                State::Voiced
            }
            State::Voiced if holdable && self.hold_hops > 0 && self.age >= self.min_hold_age => {
                self.count = 1;
                State::Hold
            }
            State::Hold if holdable && self.count < self.hold_hops => {
                self.count += 1;
                State::Hold
            }
            _ => State::Unvoiced,
        };
        if self.state == State::Unvoiced && was != State::Unvoiced {
            self.count = 0;
        }
        let voiced = self.state != State::Unvoiced;
        self.unvoiced = if voiced { 0 } else { self.unvoiced.saturating_add(1) };
        (voiced, voiced && was == State::Unvoiced)
    }

    /// True while an onset is being confirmed (unvoiced output, raw-voiced
    /// hops counting).
    pub(crate) fn confirming(&self) -> bool {
        self.state == State::Unvoiced && self.count > 0
    }

    /// Restarts the onset confirmation (the raw evidence moved, e.g. to
    /// another pitch): the next raw hop counts as the first.
    pub(crate) fn restart_onset(&mut self) {
        if self.state == State::Unvoiced {
            self.count = 0;
        }
    }

    /// True when the output has been unvoiced for at least the hold time: a
    /// new note rather than a dip inside one.
    pub(crate) fn after_silence(&self) -> bool {
        self.unvoiced >= self.hold_hops.max(1)
    }

    /// True while holding through a dip (output voiced, evidence weak).
    #[cfg(test)]
    pub(crate) fn holding(&self) -> bool {
        self.state == State::Hold
    }

    pub(crate) fn reset(&mut self) {
        self.state = State::Unvoiced;
        self.count = 0;
        self.unvoiced = u32::MAX;
        self.age = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_hysteresis() {
        let mut g = Gate::default();
        assert!(!g.update(-56.0, -55.0));
        assert!(g.update(-55.0, -55.0));
        assert!(g.update(-60.0, -55.0), "inside hysteresis stays open");
        assert!(!g.update(-61.5, -55.0));
        assert!(!g.update(-58.0, -55.0), "closed until the open threshold");
        g.reset();
        assert!(!g.update(-56.0, -55.0));
    }

    #[test]
    fn fsm_onset_hold_release() {
        let mut f = VoicingFsm::new(2, 3, 0);
        assert_eq!(f.update(true, false), (false, false));
        assert_eq!(f.update(true, false), (true, true));
        assert_eq!(f.update(true, false), (true, false));
        // Weak but holdable: held for 3 hops, then released.
        assert_eq!(f.update(false, true), (true, false));
        assert!(f.holding());
        assert_eq!(f.update(false, true), (true, false));
        assert_eq!(f.update(false, true), (true, false));
        assert_eq!(f.update(false, true), (false, false));
        // Re-onset needs 2 hops again.
        assert!(!f.update(true, false).0);
        assert!(f.update(true, false).0);
        // Not holdable (silence / sibilant): immediate release.
        assert!(!f.update(false, false).0);
        // One unvoiced hop is a dip, not yet silence (hold time 3 hops).
        assert!(!f.after_silence());
        // A single raw-voiced hop between unvoiced ones never triggers.
        assert!(!f.update(true, false).0);
        assert!(!f.update(false, false).0);
        assert!(!f.update(true, false).0);
        assert!(!f.update(false, false).0 && !f.update(false, false).0);
        assert!(f.after_silence());
        f.reset();
        assert!(!f.holding() && f.after_silence());
    }

    #[test]
    fn young_segments_do_not_hold_and_onsets_can_restart() {
        // Hold only once the segment has been raw-voiced for 4 hops.
        let mut f = VoicingFsm::new(2, 3, 4);
        assert!(!f.update(true, false).0 && f.confirming());
        assert!(f.update(true, false).0 && !f.confirming());
        // Age 2 < 4: a weak hop releases at once instead of holding.
        assert_eq!(f.update(false, true), (false, false));
        assert!(!f.holding());
        for _ in 0..4 {
            f.update(true, false);
        }
        // Age 4: held.
        assert_eq!(f.update(false, true), (true, false));
        assert!(f.holding());
        // A restarted confirmation needs onset_hops raw hops again.
        let mut f = VoicingFsm::new(2, 3, 0);
        assert!(!f.update(true, false).0);
        f.restart_onset();
        assert!(!f.update(true, false).0, "restart: this hop is the first again");
        assert!(f.update(true, false).0);
        // No effect once voiced.
        f.restart_onset();
        assert!(f.update(true, false).0);
        // An extra confirming hop applies to onsets after silence only.
        let mut f = VoicingFsm::new(2, 3, 0);
        f.set_onset_extra(1);
        assert!(!f.update(true, false).0 && !f.update(true, false).0, "after silence: 3 hops");
        assert!(f.update(true, false).0);
        assert!(!f.update(false, false).0, "a 1-hop dip, not silence (hold time 3)");
        assert!(!f.update(true, false).0);
        assert!(f.update(true, false).0, "inside a phrase: 2 hops");
    }

    #[test]
    fn features_of_tone_noise_and_silence() {
        let fs = 48_000.0;
        let mut full = Mirror::new(4096);
        let mut hp = Mirror::new(4096);
        let mut hpf = skato_core::filter::Biquad::highpass(fs, 4000.0, 0.707);
        for i in 0..2000 {
            let x = 0.1 * (std::f32::consts::TAU * 220.0 * i as f32 / fs).sin();
            full.push(x);
            hp.push(hpf.process(x));
        }
        let f = measure(&full, &hp, full.len(), 480, fs);
        // 0.1-amplitude sine: −23 dB RMS (±partial-cycle error of a 10 ms window).
        assert!((f.level_db - (-23.0)).abs() < 0.5, "{f:?}");
        assert!(f.hf_ratio < 0.01 && !f.sibilant(), "{f:?}");
        assert!((f.zcr_hz - 440.0).abs() < 120.0, "{f:?}");
        // White noise: ZCR ≈ fs/2 → sibilant.
        let mut s = 99u32;
        for _ in 0..2000 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let x = (s as f32 / u32::MAX as f32) - 0.5;
            full.push(x);
            hp.push(hpf.process(x));
        }
        let f = measure(&full, &hp, full.len(), 480, fs);
        assert!(f.sibilant(), "{f:?}");
        // Unavailable window → silence.
        assert_eq!(measure(&full, &hp, full.len() + 10, 480, fs).level_db, -160.0);
    }
}
