//! Offline evaluation harness for Skatotune (contract: `docs/DESIGN.md` §3.7).
//!
//! Everything a quality gate needs, independent of the DSP crates it measures:
//!
//! - [`synth`]: synthetic singing with exact per-sample ground truth (band-limited
//!   LF glottal source, formant cascade, melodies with detune/scoops/drift/
//!   vibrato, consonants and calibration bursts).
//! - [`reference`]: an offline, non-causal f64 pitch tracker accurate to
//!   ~0.1 c (median) on synthetic vowels, used to measure rendered outputs.
//! - [`render`]: drives `skato_engine::Engine` exactly as a host would (block
//!   patterns, MIDI timing, telemetry draining, latency-compensated output).
//! - [`metrics`]: tuning, settling, vibrato, added jitter, voicing F1,
//!   measured latency, clicks, formant LSD, aligned segmental SNR.
//! - [`scenarios`] / [`suite`]: named test signals and the full evaluation
//!   matrix with the DESIGN §4 gates; [`rtbench`]: callback tail latencies.
//! - [`vocadito`]: detector and correction evaluation on real singing (the
//!   vocadito dataset) against its F0 and note annotations.
//! - [`note_intent`]: the note-intent harness: replays cached detector
//!   streams of vocadito through the control law with the Labs options off
//!   and on, on a fixed DEV/TEST split, and scores the note decisions.
//!
//! The `skato_eval` binary (`src/main.rs`) exposes all of this on the command line.

pub mod dsp;
pub mod metrics;
pub mod note_intent;
pub mod reference;
pub mod render;
pub mod rng;
pub mod rtbench;
pub mod scenarios;
pub mod suite;
pub mod synth;
pub mod vocadito;
pub mod wav;
