//! Foundation crate for Vocal Tune: the shared data contract ([`types`]) plus the
//! realtime-safe building blocks every DSP crate uses.
//!
//! | module | what |
//! |---|---|
//! | [`types`] | plain `Copy` data exchanged between crates (re-exported at the root) |
//! | [`math`] | fast `exp2`/`log2`/`tanh`, cents/ratio/dB conversions, smoothing coefficients, parabolic peaks |
//! | [`ring`] | power-of-two circular buffer addressed by absolute `u64` sample index |
//! | [`interp`] | fractional reads: cubic Hermite and Kaiser-windowed polyphase sinc with anti-alias cutoffs |
//! | [`filter`] | DC blocker, RBJ biquads, one-pole smoother, minimum-phase decimators |
//! | [`window`] | Hann, Kaiser, raised-cosine / equal-power fades |
//! | [`simd`] | dot / sum-of-squares / axpy / windowed accumulate / type-I difference, scalar + NEON + AVX2 |
//! | [`fft`] | FFT correlation and spectra (realfft or Apple vDSP) |
//!
//! Realtime rules (DESIGN §2.1) hold for every per-sample / per-hop function in
//! this crate: no allocation after construction, no locks or syscalls, no
//! panics on any input (out-of-range reads return zeros, NaN is mapped to a
//! safe value), bounded loops.

pub mod fft;
pub mod filter;
pub mod interp;
pub mod math;
pub mod ring;
pub mod simd;
pub mod types;
pub mod window;

pub use types::*;

/// Control hop in samples for a sample rate (DESIGN §2.2): 64 at 44.1/48 kHz,
/// 128 at 88.2/96 kHz, 256 at 176.4/192 kHz, i.e. ≈ 1.3-1.45 ms whatever the
/// rate.
///
/// The hop doubles each time the rate crosses 1.5× a 44.1 kHz-family octave
/// (66.15 kHz, 132.3 kHz, 264.6 kHz, …), so intermediate rates land on the
/// nearer standard family; capped at 1024. Non-finite or non-positive rates
/// give 64.
pub fn hop_for_rate(sample_rate: f32) -> usize {
    let mut hop = 64;
    let mut edge = 66_150.0f32;
    // `>=` is false for NaN, so NaN stays at 64. At most 4 iterations.
    while sample_rate >= edge && hop < 1024 {
        hop *= 2;
        edge *= 2.0;
    }
    hop
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Everything a detector/shifter owns must be movable into the plugin
    /// (nice-plug requires `Plugin: Send`).
    #[test]
    fn realtime_objects_are_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<ring::Ring>();
        check::<interp::Interpolator>();
        check::<interp::SincBank>();
        check::<filter::Decimator>();
        check::<filter::Biquad>();
        check::<filter::DcBlocker>();
        check::<filter::OnePoleLp>();
        check::<fft::Correlator>();
    }

    #[test]
    fn hop_sizes() {
        assert_eq!(hop_for_rate(44_100.0), 64);
        assert_eq!(hop_for_rate(48_000.0), 64);
        assert_eq!(hop_for_rate(88_200.0), 128);
        assert_eq!(hop_for_rate(96_000.0), 128);
        assert_eq!(hop_for_rate(176_400.0), 256);
        assert_eq!(hop_for_rate(192_000.0), 256);
        assert_eq!(hop_for_rate(384_000.0), 512);
        assert_eq!(hop_for_rate(22_050.0), 64);
        assert_eq!(hop_for_rate(f32::NAN), 64);
        assert_eq!(hop_for_rate(-1.0), 64);
        assert_eq!(hop_for_rate(f32::INFINITY), 1024);
    }
}
