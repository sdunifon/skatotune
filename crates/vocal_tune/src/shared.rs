//! State shared between the audio thread and the editor.
//!
//! Two kinds of data cross the thread boundary, and they use different
//! mechanisms so the audio thread can never block:
//!
//! - **Scalar meters** (CPU load, latency, learned scale, restart flag) are
//!   atomics. `process()` stores, the GUI loads.
//! - **Telemetry frames** travel through the engine's lock-free `rtrb` ring.
//!   The consumer half is created with the engine in `activate()` and handed
//!   to the GUI through [`SharedState::install_telemetry`] /
//!   [`SharedState::take_telemetry`], a small `Mutex<Option<_>>` mailbox that
//!   only `activate()` and the GUI thread ever touch. `process()` never sees
//!   the mutex (DESIGN §3.6).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use atomic_float::AtomicF32;
use vt_core::{NoteState, PitchFrame};

/// Everything the editor reads about the running engine.
pub struct SharedState {
    /// Smoothed processing load as a fraction of the buffer period (0.05 = 5 %).
    pub cpu_load: AtomicF32,
    /// Recent peak load (fast attack, slow release), same units.
    pub cpu_peak: AtomicF32,
    /// Latency of the running engine, samples.
    pub latency_samples: AtomicU32,
    /// Latency the host has been told about while a restart is pending.
    pub pending_latency_samples: AtomicU32,
    /// Sample rate of the running engine (0 until the first activation).
    pub sample_rate: AtomicF32,
    /// A structural parameter changed; the engine is rebuilt on the next
    /// `activate()`.
    pub restart_pending: AtomicBool,
    /// Engine scale after MIDI Learn, packed by [`pack_states`].
    pub learned_states: AtomicU32,
    /// Telemetry frames the engine dropped because the GUI fell behind.
    pub telemetry_dropped: AtomicU64,
    /// Incremented every time a new engine (and telemetry ring) is installed.
    pub engine_generation: AtomicU32,
    /// GUI-side mailbox for the telemetry consumer (see module docs).
    telemetry: Mutex<Option<rtrb::Consumer<PitchFrame>>>,
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            cpu_load: AtomicF32::new(0.0),
            cpu_peak: AtomicF32::new(0.0),
            latency_samples: AtomicU32::new(0),
            pending_latency_samples: AtomicU32::new(0),
            sample_rate: AtomicF32::new(0.0),
            restart_pending: AtomicBool::new(false),
            learned_states: AtomicU32::new(0),
            telemetry_dropped: AtomicU64::new(0),
            engine_generation: AtomicU32::new(0),
            telemetry: Mutex::new(None),
        }
    }
}

impl SharedState {
    /// Hand a fresh telemetry consumer to the GUI. Called from `activate()`
    /// only (never from `process()`). Replaces (and drops, off the audio
    /// thread) any consumer the GUI has not picked up yet.
    pub fn install_telemetry(&self, consumer: rtrb::Consumer<PitchFrame>) {
        let mut slot = self
            .telemetry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *slot = Some(consumer);
        drop(slot);
        self.engine_generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Take the consumer if a new one is waiting. GUI thread only; never
    /// blocks (a concurrent `activate()` just means "try again next frame").
    pub fn take_telemetry(&self) -> Option<rtrb::Consumer<PitchFrame>> {
        self.telemetry
            .try_lock()
            .ok()
            .and_then(|mut slot| slot.take())
    }

    /// The learned scale, if MIDI Learn has produced one.
    pub fn learned(&self) -> Option<[NoteState; 12]> {
        unpack_states(self.learned_states.load(Ordering::Relaxed))
    }
}

/// Bit 31 marks a valid scale; bits `2·pc .. 2·pc+2` hold each pitch class's
/// state (0 = On, 1 = Bypass, 2 = Remove).
const LEARNED_VALID: u32 = 1 << 31;

/// Pack an optional 12-note scale into one atomic word.
pub fn pack_states(states: Option<[NoteState; 12]>) -> u32 {
    let Some(states) = states else { return 0 };
    states
        .iter()
        .enumerate()
        .fold(LEARNED_VALID, |acc, (pc, s)| {
            let code = match s {
                NoteState::On => 0,
                NoteState::Bypass => 1,
                NoteState::Remove => 2,
            };
            acc | (code << (2 * pc))
        })
}

/// Inverse of [`pack_states`].
pub fn unpack_states(word: u32) -> Option<[NoteState; 12]> {
    if word & LEARNED_VALID == 0 {
        return None;
    }
    Some(std::array::from_fn(|pc| match (word >> (2 * pc)) & 0b11 {
        1 => NoteState::Bypass,
        2 => NoteState::Remove,
        _ => NoteState::On,
    }))
}

/// Per-block CPU meter: processing time divided by the block's real-time
/// duration, as an exponential average plus a peak hold (R§9.6).
#[derive(Clone, Copy, Debug)]
pub struct CpuMeter {
    sample_rate: f32,
    ema: f32,
    peak: f32,
}

/// Averaging time constant of the load meter, seconds.
const CPU_EMA_TAU_S: f32 = 0.5;
/// Release time constant of the peak hold, seconds.
const CPU_PEAK_RELEASE_S: f32 = 2.0;

impl CpuMeter {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate: sample_rate.max(1.0),
            ema: 0.0,
            peak: 0.0,
        }
    }

    /// Account one block of `samples` that took `elapsed_s` seconds to
    /// process. Returns `(average, peak)` load.
    pub fn update(&mut self, elapsed_s: f32, samples: usize) -> (f32, f32) {
        if samples == 0 || !elapsed_s.is_finite() {
            return (self.ema, self.peak);
        }
        let period = samples as f32 / self.sample_rate;
        let load = (elapsed_s / period).max(0.0);
        // Time-based coefficients so the meter behaves the same at any block size.
        let a = 1.0 - (-period / CPU_EMA_TAU_S).exp();
        self.ema += a * (load - self.ema);
        self.peak = if load > self.peak {
            load
        } else {
            self.peak * (-period / CPU_PEAK_RELEASE_S).exp()
        };
        (self.ema, self.peak)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_states_round_trip() {
        assert_eq!(unpack_states(pack_states(None)), None);
        let mut s = [NoteState::Remove; 12];
        s[0] = NoteState::On;
        s[4] = NoteState::Bypass;
        s[11] = NoteState::On;
        assert_eq!(unpack_states(pack_states(Some(s))), Some(s));
        let all_on = [NoteState::On; 12];
        assert_eq!(unpack_states(pack_states(Some(all_on))), Some(all_on));
    }

    #[test]
    fn cpu_meter_converges_to_the_load() {
        let mut m = CpuMeter::new(48_000.0);
        // 128-sample blocks (2.67 ms) that take 0.267 ms: 10 % load.
        let mut out = (0.0, 0.0);
        for _ in 0..2000 {
            out = m.update(128.0 / 48_000.0 * 0.1, 128);
        }
        assert!((out.0 - 0.1).abs() < 1e-3, "ema {}", out.0);
        assert!((out.1 - 0.1).abs() < 1e-3, "peak {}", out.1);
        // One slow block shows up in the peak immediately, barely in the average.
        let (avg, peak) = m.update(128.0 / 48_000.0 * 0.9, 128);
        assert!((peak - 0.9).abs() < 1e-6);
        assert!(avg < 0.12);
        // ...and the peak releases afterwards.
        for _ in 0..4000 {
            out = m.update(128.0 / 48_000.0 * 0.1, 128);
        }
        assert!(out.1 < 0.2);
    }

    #[test]
    fn cpu_meter_ignores_degenerate_blocks() {
        let mut m = CpuMeter::new(48_000.0);
        assert_eq!(m.update(0.001, 0), (0.0, 0.0));
        assert_eq!(m.update(f32::NAN, 64), (0.0, 0.0));
    }

    #[test]
    fn telemetry_mailbox_hands_over_once() {
        let shared = SharedState::default();
        assert!(shared.take_telemetry().is_none());
        let (_tx, rx) = rtrb::RingBuffer::<PitchFrame>::new(4);
        shared.install_telemetry(rx);
        assert_eq!(shared.engine_generation.load(Ordering::Acquire), 1);
        assert!(shared.take_telemetry().is_some());
        assert!(shared.take_telemetry().is_none());
    }
}
