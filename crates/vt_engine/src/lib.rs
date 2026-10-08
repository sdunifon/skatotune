//! Realtime engine: wires detection, control and shifting together behind one
//! block-size-agnostic `process()` call (contract: `docs/DESIGN.md` §3.5).
//!
//! STUB: this placeholder implements the public API as a plain delay line so the
//! plugin and GUI can be built in parallel with the DSP crates. The integration
//! stage replaces the internals without changing the API.

use vt_core::{EngineConfig, LatencyMode, MidiEvent, NoteState, PitchFrame, Settings, VoiceRange};

/// Telemetry ring capacity in frames (~10 s at 48 kHz with a 64-sample hop is
/// 7500 frames; the GUI drains every frame so a few thousand is plenty).
const TELEMETRY_CAPACITY: usize = 8192;

/// Reported latency for a config. STUB formula; the real one lives in
/// `vt_shift::latency_samples`.
pub fn latency_samples(cfg: &EngineConfig) -> u32 {
    let ms = match cfg.latency_mode {
        LatencyMode::LiveMin => 1.0,
        LatencyMode::Live => match cfg.voice_range {
            VoiceRange::Soprano => 3.0,
            VoiceRange::AltoTenor => 4.0,
            VoiceRange::LowMale | VoiceRange::Wide => 5.5,
        },
        LatencyMode::LivePlus => cfg.live_plus_ms,
        LatencyMode::Studio => cfg.studio_ms,
    };
    (ms * 0.001 * cfg.sample_rate).round() as u32
}

/// A MIDI event with its sample offset inside the current host block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedMidi {
    pub timing: u32,
    pub event: MidiEvent,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EngineStats {
    pub hops: u64,
    pub voiced_hops: u64,
    pub telemetry_dropped: u64,
}

pub struct Engine {
    cfg: EngineConfig,
    latency: u32,
    /// Per-channel delay lines of exactly `latency` samples.
    delay: Vec<Vec<f32>>,
    delay_pos: usize,
    telemetry_tx: rtrb::Producer<PitchFrame>,
    telemetry_rx: Option<rtrb::Consumer<PitchFrame>>,
    telemetry_enabled: bool,
    hop: usize,
    hop_phase: usize,
    out_pos: u64,
    stats: EngineStats,
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> Self {
        let latency = latency_samples(&cfg);
        let (telemetry_tx, telemetry_rx) = rtrb::RingBuffer::new(TELEMETRY_CAPACITY);
        Self {
            cfg,
            latency,
            delay: vec![vec![0.0; latency.max(1) as usize]; cfg.channels.max(1)],
            delay_pos: 0,
            telemetry_tx,
            telemetry_rx: Some(telemetry_rx),
            telemetry_enabled: false,
            hop: vt_hop(cfg.sample_rate),
            hop_phase: 0,
            out_pos: 0,
            stats: EngineStats::default(),
        }
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn latency_samples(&self) -> u32 {
        self.latency
    }

    pub fn reset(&mut self) {
        for line in &mut self.delay {
            line.fill(0.0);
        }
        self.delay_pos = 0;
        self.hop_phase = 0;
        self.out_pos = 0;
        self.stats = EngineStats::default();
    }

    pub fn set_settings(&mut self, _s: &Settings) {}

    pub fn process(&mut self, io: &mut [&mut [f32]], _midi: &[TimedMidi]) {
        let n = io.first().map_or(0, |c| c.len());
        let len = self.delay.first().map_or(1, |d| d.len());
        for i in 0..n {
            for (ch, line) in io.iter_mut().zip(self.delay.iter_mut()) {
                let Some(x) = ch.get_mut(i) else { continue };
                let input = if x.is_finite() { *x } else { 0.0 };
                if self.latency == 0 {
                    *x = input;
                } else {
                    *x = line[self.delay_pos];
                    line[self.delay_pos] = input;
                }
            }
            self.delay_pos = (self.delay_pos + 1) % len;
            self.out_pos += 1;
            self.hop_phase += 1;
            if self.hop_phase == self.hop {
                self.hop_phase = 0;
                self.stats.hops += 1;
                if self.telemetry_enabled {
                    let frame = PitchFrame { sample_pos: self.out_pos, ..PitchFrame::default() };
                    if self.telemetry_tx.push(frame).is_err() {
                        self.stats.telemetry_dropped += 1;
                    }
                }
            }
        }
    }

    pub fn take_telemetry(&mut self) -> Option<rtrb::Consumer<PitchFrame>> {
        self.telemetry_rx.take()
    }

    pub fn set_telemetry_enabled(&mut self, on: bool) {
        self.telemetry_enabled = on;
    }

    pub fn stats(&self) -> EngineStats {
        self.stats
    }

    pub fn learned_states(&self) -> Option<[NoteState; 12]> {
        None
    }
}

/// Control hop for a sample rate (mirrors `vt_core::hop_for_rate`, which the
/// core stage adds).
fn vt_hop(sample_rate: f32) -> usize {
    if sample_rate <= 50_000.0 {
        64
    } else if sample_rate <= 100_000.0 {
        128
    } else {
        256
    }
}
