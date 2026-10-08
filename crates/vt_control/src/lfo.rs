//! Synthetic vibrato, R§7.6.2 K: `s_lfo = A(t)·sin(2π·f·t)` added after the
//! correction.
//!
//! The phase restarts on every note; the amplitude is 0 for the onset delay,
//! then fades in over the ramp time (raised-cosine, so the fade has no kink)
//! to the depth. Each note gets a deterministic ±10 % variation of rate and
//! depth (R§7.6.4 "variation 10 %") from a fixed-seed generator, so renders
//! are repeatable. Only the sine shape is exposed by `SynthVibrato` in v1.

use vt_core::SynthVibrato;

/// Per-note rate/depth variation (fraction).
const VARIATION: f32 = 0.1;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SynthLfo {
    /// Phase in cycles, [0, 1).
    phase: f32,
    /// Seconds since the note started.
    t: f32,
    rate_mul: f32,
    depth_mul: f32,
    rng: u32,
}

impl Default for SynthLfo {
    fn default() -> Self {
        Self { phase: 0.0, t: 0.0, rate_mul: 1.0, depth_mul: 1.0, rng: 0x9E37_79B9 }
    }
}

impl SynthLfo {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    fn next_unit(&mut self) -> f32 {
        // xorshift32 → uniform in [-1, 1).
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }

    /// A new note starts: restart the phase and draw this note's variation.
    pub fn note_on(&mut self) {
        self.phase = 0.0;
        self.t = 0.0;
        self.rate_mul = 1.0 + VARIATION * self.next_unit();
        self.depth_mul = 1.0 + VARIATION * self.next_unit();
    }

    /// Advance one hop; returns the LFO offset in cents. `depth` is the
    /// (smoothed) depth in cents; the other fields come from `sv`.
    pub fn step(&mut self, sv: &SynthVibrato, depth: f32, hop_s: f32) -> f32 {
        if !sv.enabled {
            return 0.0;
        }
        self.t += hop_s;
        let rate = sv.rate_hz.clamp(0.1, 10.0) * self.rate_mul;
        self.phase = (self.phase + rate * hop_s).fract();
        let delay = sv.delay_ms.max(0.0) * 1e-3;
        let ramp = sv.ramp_ms.max(0.0) * 1e-3;
        let env = if self.t < delay {
            0.0
        } else if ramp > 0.0 && self.t < delay + ramp {
            let x = (self.t - delay) / ramp;
            0.5 - 0.5 * (core::f32::consts::PI * x).cos()
        } else {
            1.0
        };
        env * depth.clamp(0.0, 100.0) * self.depth_mul * (2.0 * core::f32::consts::PI * self.phase).sin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_ramp_and_depth() {
        let sv = SynthVibrato { enabled: true, rate_hz: 5.0, depth_cents: 30.0, delay_ms: 100.0, ramp_ms: 100.0 };
        let mut lfo = SynthLfo::default();
        lfo.note_on();
        let hop_s = 64.0 / 48_000.0;
        let out: Vec<f32> = (0..(1.0 / hop_s) as usize).map(|_| lfo.step(&sv, 30.0, hop_s)).collect();
        let delay_hops = (0.1 / hop_s) as usize;
        assert!(out[..delay_hops].iter().all(|v| *v == 0.0));
        let peak = out[out.len() / 2..].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak > 30.0 * 0.88 && peak < 30.0 * 1.12, "peak {peak}");
        // Disabled → silent.
        let off = SynthVibrato { enabled: false, ..sv };
        assert_eq!(lfo.step(&off, 30.0, hop_s), 0.0);
    }

    #[test]
    fn variation_is_bounded_and_deterministic() {
        let mut a = SynthLfo::default();
        let mut b = SynthLfo::default();
        for _ in 0..100 {
            a.note_on();
            b.note_on();
            assert_eq!(a.rate_mul, b.rate_mul);
            assert!((a.rate_mul - 1.0).abs() <= VARIATION + 1e-6);
            assert!((a.depth_mul - 1.0).abs() <= VARIATION + 1e-6);
        }
    }
}
