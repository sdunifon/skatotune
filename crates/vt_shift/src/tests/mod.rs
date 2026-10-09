//! Quality and contract tests for the shifter (DESIGN §3.4). Listening-like
//! objective checks: pitch accuracy with an offline f64 NSDF tracker,
//! LPC-envelope log-spectral distance for formant preservation, a click
//! detector, transparency at ratio 1, exact dry delay when unvoiced,
//! causality counters, block-size and stereo invariance.

mod support;

mod contract;
mod quality;
