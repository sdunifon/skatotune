//! Vocal Tune: realtime vocal pitch correction as a VST3 / CLAP plugin
//! (contract: `docs/DESIGN.md` §3.6).
//!
//! This crate is the thin, host-facing shell around [`vt_engine::Engine`]:
//!
//! - [`params`]: every host parameter, the stable ids, and the pure mapping
//!   from parameter values to the engine's `Settings` and structural config.
//! - [`midi`]: host note events → the control law's MIDI vocabulary.
//! - [`shared`]: atomics and the telemetry mailbox shared with the editor.
//! - [`editor`]: the egui editor with the scrolling pitch graph.
//!
//! # Threads and lifecycle
//!
//! `activate()` (never the audio thread in a DAW) builds the engine for the
//! host's sample rate, block size and channel count plus the structural
//! parameters (latency mode, voice range, Live+/Studio latency), reports the
//! latency, and hands the engine's telemetry consumer to the editor.
//!
//! `process()` copies the parameters into `Settings` (atomic loads, no
//! allocation), converts MIDI into a fixed array, runs the engine, applies the
//! output gain and updates the CPU meter. When the user changes a structural
//! parameter, `process()` tells the host exactly once and keeps running the
//! old engine until the host re-activates the plugin: VST3 gets the new
//! latency (nice-plug turns that into `restartComponent(kLatencyChanged)`),
//! CLAP gets a restart request and learns the latency in `activate()` (the
//! only place CLAP allows it to change). The standalone wrapper cannot
//! restart plugins, so there the editor reloads the plugin state, which
//! re-runs `activate()`.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use nice_plug::prelude::*;
use nice_plug_egui::{EguiEditor, EguiEditorState, RepaintNotifier};
use vt_engine::Engine;

pub mod editor;
pub mod midi;
pub mod params;
pub mod shared;

use midi::MidiBuffer;
use params::{ParamValues, Structural, VocalTuneParams};
use shared::{CpuMeter, SharedState, pack_states};

/// The plugin instance. `pub` so the standalone binary can export it.
pub struct VocalTune {
    params: Arc<VocalTuneParams>,
    shared: Arc<SharedState>,
    editor_state: Arc<EguiEditorState>,
    /// Lets `process()` wake the editor (an atomic store).
    repaint: RepaintNotifier,
    /// The editor app until the wrapper asks for it (queried once).
    editor_app: Option<editor::VocalTuneEditor>,

    engine: Option<Engine>,
    /// What the running engine was built for.
    active: Option<Active>,
    /// Structural configuration already reported to the host and waiting for
    /// a restart, so the request is made exactly once per change.
    requested: Option<Structural>,
    /// Latency last reported to the host.
    reported_latency: u32,
    midi: MidiBuffer,
    cpu: CpuMeter,
}

/// The configuration the running engine was built with.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Active {
    structural: Structural,
    sample_rate: f32,
    max_block: usize,
    engine_channels: usize,
    latency: u32,
}

impl Default for VocalTune {
    fn default() -> Self {
        let params = Arc::new(VocalTuneParams::default());
        let shared = Arc::new(SharedState::default());
        let editor_app = editor::VocalTuneEditor::new(params.clone(), shared.clone());
        Self {
            params,
            shared,
            editor_state: editor::default_state(),
            repaint: RepaintNotifier::new(),
            editor_app: Some(editor_app),
            engine: None,
            active: None,
            requested: None,
            reported_latency: 0,
            midi: MidiBuffer::default(),
            cpu: CpuMeter::new(48_000.0),
        }
    }
}

/// What the block logic needs from the host. [`process()`](Plugin::process)
/// adapts nice-plug's `ProcessContext` to it; tests drive
/// [`VocalTune::process_block`] with a fake.
pub trait BlockHost {
    /// Next note event of this block, in timing order.
    fn next_event(&mut self) -> Option<NoteEvent<()>>;
    /// Report a new latency (nice-plug turns a change into a host restart).
    fn set_latency_samples(&mut self, samples: u32);
    /// Ask the host to deactivate and re-activate the plugin.
    fn request_restart(&mut self);
    /// Which wrapper is hosting us (the restart protocol differs per format).
    fn api(&self) -> PluginApi;
}

struct ContextHost<'a, C>(&'a mut C);

impl<C: ProcessContext<VocalTune>> BlockHost for ContextHost<'_, C> {
    fn next_event(&mut self) -> Option<NoteEvent<()>> {
        self.0.next_event()
    }

    fn set_latency_samples(&mut self, samples: u32) {
        self.0.set_latency_samples(samples);
    }

    fn request_restart(&mut self) {
        self.0.request_restart();
    }

    fn api(&self) -> PluginApi {
        self.0.plugin_api()
    }
}

impl VocalTune {
    /// Build (or keep) the engine for a host configuration and the
    /// structural parameters in `values`, and return its latency in samples.
    /// Allocates; call from `activate()` only.
    pub fn activate_with(
        &mut self,
        values: &ParamValues,
        sample_rate: f32,
        max_block: usize,
        input_channels: usize,
    ) -> u32 {
        let structural = values.structural();
        let engine_channels = input_channels.clamp(1, 2);
        let cfg = structural.engine_config(sample_rate, max_block, engine_channels);

        // Hosts may activate several times in a row (state restore, ...);
        // only rebuild when the configuration actually changed.
        let reuse = self.engine.as_ref().is_some_and(|e| *e.config() == cfg);
        if !reuse {
            let mut engine = Engine::new(cfg);
            if let Some(consumer) = engine.take_telemetry() {
                self.shared.install_telemetry(consumer);
            }
            // The previous engine (if any) is dropped here, off the audio thread.
            self.engine = Some(engine);
        }
        let latency = self.engine.as_ref().map_or(0, Engine::latency_samples);

        self.active = Some(Active {
            structural,
            sample_rate,
            max_block: cfg.max_block,
            engine_channels,
            latency,
        });
        self.requested = None;
        self.reported_latency = latency;
        self.cpu = CpuMeter::new(sample_rate);
        self.shared
            .sample_rate
            .store(sample_rate, Ordering::Relaxed);
        self.shared
            .latency_samples
            .store(latency, Ordering::Relaxed);
        self.shared
            .pending_latency_samples
            .store(latency, Ordering::Relaxed);
        self.shared.restart_pending.store(false, Ordering::Release);
        latency
    }

    /// Latency of the running engine, if activated.
    pub fn latency(&self) -> Option<u32> {
        self.active.map(|a| a.latency)
    }

    /// Whether a structural change is waiting for the host to restart us.
    pub fn restart_pending(&self) -> bool {
        self.requested.is_some()
    }

    /// One host block: `values` is this block's parameter snapshot, `io` the
    /// in-place channel buffers (all the same length). Realtime-safe: no
    /// allocation, no locks, no panics.
    pub fn process_block(
        &mut self,
        values: &ParamValues,
        io: &mut [&mut [f32]],
        host: &mut impl BlockHost,
    ) -> ProcessStatus {
        let started = Instant::now();
        let samples = io.first().map_or(0, |c| c.len());

        self.midi.clear();
        let block_len = u32::try_from(samples).unwrap_or(u32::MAX);
        while let Some(ev) = host.next_event() {
            if let Some(m) = midi::convert_event(&ev, block_len) {
                self.midi.push(m);
            }
        }

        let Some(active) = self.active else {
            // Not activated: leave the audio untouched.
            return ProcessStatus::Normal;
        };
        self.check_structural(values, &active, host);

        let Some(engine) = self.engine.as_mut() else {
            return ProcessStatus::Normal;
        };
        let editor_open = self.editor_state.is_open();
        engine.set_settings(&values.to_settings());
        engine.set_telemetry_enabled(editor_open);

        let channels = active.engine_channels.min(io.len());
        if let Some(engine_io) = io.get_mut(..channels) {
            engine.process(engine_io, self.midi.as_slice());
        }
        // Mono engine feeding a stereo output (the mono → stereo layout).
        if channels == 1
            && let Some((first, rest)) = io.split_first_mut()
        {
            for ch in rest {
                for (dst, src) in ch.iter_mut().zip(first.iter()) {
                    *dst = *src;
                }
            }
        }

        // Output gain, smoothed in dB. Skipped entirely at a steady 0 dB.
        let gain = &self.params.output_gain_db;
        if gain.smoothed.is_smoothing() || gain.value().abs() > 1e-4 {
            for i in 0..samples {
                let g = util::db_to_gain_fast(gain.smoothed.next());
                for ch in io.iter_mut() {
                    if let Some(x) = ch.get_mut(i) {
                        *x *= g;
                    }
                }
            }
        }

        self.shared
            .learned_states
            .store(pack_states(engine.learned_states()), Ordering::Relaxed);
        self.shared
            .telemetry_dropped
            .store(engine.stats().telemetry_dropped, Ordering::Relaxed);
        if editor_open {
            self.repaint.request_repaint();
        }
        let (avg, peak) = self.cpu.update(started.elapsed().as_secs_f32(), samples);
        self.shared.cpu_load.store(avg, Ordering::Relaxed);
        self.shared.cpu_peak.store(peak, Ordering::Relaxed);

        // The engine delays the signal by its latency, so audio keeps coming
        // out for that long after the input goes silent.
        if active.latency > 0 {
            ProcessStatus::Tail(active.latency)
        } else {
            ProcessStatus::Normal
        }
    }

    /// Detect structural parameter changes and tell the host exactly once.
    ///
    /// The protocol differs per format (R§9.5):
    /// - VST3: report the new latency; nice-plug turns a change into
    ///   `restartComponent(kLatencyChanged)`. Unchanged latency → plain restart.
    /// - CLAP: latency may only change inside `activate()`, so only a restart
    ///   is requested here (`host.request_restart()` is thread-safe) and the
    ///   new latency is reported by `activate()`. Reporting it from here would
    ///   make nice-plug call `latency.changed()` from the main thread, possibly
    ///   while deactivated (clap-validator rejects that).
    /// - Standalone: no restart exists; the editor reloads the plugin state,
    ///   which re-runs `activate()`.
    fn check_structural(
        &mut self,
        values: &ParamValues,
        active: &Active,
        host: &mut impl BlockHost,
    ) {
        let wanted = values.structural();
        let api = host.api();
        if wanted.same_engine(&active.structural) {
            if self.requested.take().is_some() {
                // Reverted before the host restarted us: re-report the latency
                // of the engine that is actually running.
                self.shared.restart_pending.store(false, Ordering::Release);
                self.shared
                    .pending_latency_samples
                    .store(active.latency, Ordering::Relaxed);
                if api == PluginApi::Vst3 && self.reported_latency != active.latency {
                    host.set_latency_samples(active.latency);
                    self.reported_latency = active.latency;
                }
            }
            return;
        }
        if self.requested.is_some_and(|r| r.same_engine(&wanted)) {
            return; // already reported, waiting for the host
        }
        self.requested = Some(wanted);
        let cfg =
            wanted.engine_config(active.sample_rate, active.max_block, active.engine_channels);
        let latency = vt_engine::latency_samples(&cfg);
        self.shared
            .pending_latency_samples
            .store(latency, Ordering::Relaxed);
        self.shared.restart_pending.store(true, Ordering::Release);
        match api {
            PluginApi::Standalone => {}
            PluginApi::Vst3 if latency != self.reported_latency => {
                host.set_latency_samples(latency);
                self.reported_latency = latency;
            }
            PluginApi::Vst3 | PluginApi::Clap => host.request_restart(),
        }
    }
}

impl Plugin for VocalTune {
    const NAME: &'static str = "Vocal Tune";
    const VENDOR: &'static str = "Vocal Tune";
    const URL: &'static str = "";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo first (the default; Live instantiates stereo), then mono, then
    // mono in → stereo out (the processed mono voice is copied to both sides).
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            names: PortNames {
                layout: Some("Stereo"),
                ..PortNames::const_default()
            },
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            names: PortNames {
                layout: Some("Mono"),
                ..PortNames::const_default()
            },
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(2),
            names: PortNames {
                layout: Some("Mono to Stereo"),
                ..PortNames::const_default()
            },
            ..AudioIOLayout::const_default()
        },
    ];

    // `MidiCCs` rather than the contract's `Basic`: pitch bend, sustain
    // (CC64) and all-notes-off (CC120/123) are part of the MIDI control law
    // (R§7.6.3) and nice-plug only delivers them at this level.
    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;
    // Block splitting on automation would only add overhead: the engine
    // applies settings at hop boundaries anyway.
    const SAMPLE_ACCURATE_AUTOMATION: bool = false;

    type Editor = EguiEditor<editor::VocalTuneEditor>;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Self::Editor> {
        editor::create(
            self.editor_state.clone(),
            self.repaint.clone(),
            self.editor_app.take()?,
        )
    }

    fn activate(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl ActivateContext<Self>,
    ) -> bool {
        let inputs = audio_io_layout
            .main_input_channels
            .map_or(1, NonZeroU32::get) as usize;
        let values = self.params.values();
        let latency = self.activate_with(
            &values,
            buffer_config.sample_rate,
            buffer_config.max_buffer_size as usize,
            inputs,
        );
        // The standalone wrapper has no latency reporting (it would only log
        // a debug assertion).
        if context.plugin_api() != PluginApi::Standalone {
            context.set_latency_samples(latency);
        }
        true
    }

    fn reset(&mut self) {
        if let Some(engine) = self.engine.as_mut() {
            engine.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let values = self.params.values();
        self.process_block(&values, buffer.as_slice(), &mut ContextHost(context))
    }
}

impl ClapPlugin for VocalTune {
    const CLAP_ID: &'static str = "com.vocaltune.vocal-tune";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Realtime vocal pitch correction");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::PitchCorrection,
        ClapFeature::Stereo,
        ClapFeature::Mono,
    ];

    /// Bitwig remote-control page with the main correction knobs.
    fn remote_controls(&self, context: &mut impl RemoteControlsContext) {
        let p = &self.params;
        context.add_section("Vocal Tune", |section| {
            section.add_page("Correction", |page| {
                page.add_param(&p.retune_ms);
                page.add_param(&p.transition_ms);
                page.add_param(&p.humanize);
                page.add_param(&p.flex_tune);
                page.add_param(&p.amount);
                page.add_param(&p.vibrato_depth);
                page.add_param(&p.tolerance_cents);
                page.add_param(&p.detune_cents);
            });
            section.add_page("Voice", |page| {
                page.add_param(&p.transpose_st);
                page.add_param(&p.formant_st);
                page.add_param(&p.tracking);
                page.add_param(&p.gate_db);
                page.add_param(&p.lfo_rate);
                page.add_param(&p.lfo_depth);
                page.add_param(&p.bend_range);
                page.add_param(&p.output_gain_db);
            });
        });
    }
}

impl Vst3Plugin for VocalTune {
    // Exactly 16 bytes; fixed forever.
    const VST3_CLASS_ID: [u8; 16] = *b"VocalTuneVT1pcor";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::PitchShift];
}

nice_export_clap!(VocalTune);
nice_export_vst3!(VocalTune);

#[cfg(test)]
// Tests read best as "defaults, then change what matters".
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use nice_plug::midi::{Channel, Key, VoiceID};
    use params::{LatencyModeChoice, VoiceRangeChoice};
    use vt_core::{EngineConfig, LatencyMode};

    /// Records what the plugin asks of the host.
    #[derive(Default)]
    struct FakeHost {
        events: Vec<NoteEvent<()>>,
        latency_calls: Vec<u32>,
        restarts: usize,
        api: Option<PluginApi>,
    }

    impl BlockHost for FakeHost {
        fn next_event(&mut self) -> Option<NoteEvent<()>> {
            if self.events.is_empty() {
                None
            } else {
                Some(self.events.remove(0))
            }
        }
        fn set_latency_samples(&mut self, samples: u32) {
            self.latency_calls.push(samples);
        }
        fn request_restart(&mut self) {
            self.restarts += 1;
        }
        fn api(&self) -> PluginApi {
            self.api.unwrap_or(PluginApi::Vst3)
        }
    }

    fn run(
        p: &mut VocalTune,
        values: &ParamValues,
        host: &mut FakeHost,
        n: usize,
    ) -> Vec<Vec<f32>> {
        let mut l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut r = l.clone();
        {
            let mut io: [&mut [f32]; 2] = [&mut l, &mut r];
            p.process_block(values, &mut io, host);
        }
        vec![l, r]
    }

    #[test]
    fn unactivated_plugin_passes_audio_through() {
        let mut p = VocalTune::default();
        let mut host = FakeHost::default();
        let mut x = vec![0.5f32; 64];
        let before = x.clone();
        let mut io: [&mut [f32]; 1] = [&mut x];
        assert_eq!(
            p.process_block(&ParamValues::default(), &mut io, &mut host),
            ProcessStatus::Normal
        );
        assert_eq!(x, before);
    }

    #[test]
    fn activation_reports_engine_latency() {
        let mut p = VocalTune::default();
        let latency = p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let cfg = EngineConfig {
            max_block: 512,
            ..EngineConfig::default()
        };
        assert_eq!(latency, vt_engine::latency_samples(&cfg));
        assert_eq!(p.shared.latency_samples.load(Ordering::Relaxed), latency);
        assert_eq!(p.shared.engine_generation.load(Ordering::Relaxed), 1);
        // Re-activating with the same configuration keeps the engine.
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        assert_eq!(p.shared.engine_generation.load(Ordering::Relaxed), 1);
        // A new sample rate builds a new one.
        p.activate_with(&ParamValues::default(), 96_000.0, 512, 2);
        assert_eq!(p.shared.engine_generation.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn output_is_input_delayed_by_the_reported_latency() {
        // The stub engine is a pure delay; the plugin must not add anything.
        let mut p = VocalTune::default();
        let d = p.activate_with(&ParamValues::default(), 48_000.0, 4096, 2) as usize;
        let mut host = FakeHost::default();
        let n = 2048;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = run(&mut p, &ParamValues::default(), &mut host, n);
        for i in d..n {
            assert!((out[0][i] - input[i - d]).abs() < 1e-6);
            assert!((out[1][i] - input[i - d]).abs() < 1e-6);
        }
    }

    #[test]
    fn structural_change_requests_one_restart() {
        let mut p = VocalTune::default();
        let live = p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        let mut v = ParamValues::default();
        run(&mut p, &v, &mut host, 64);
        assert!(host.latency_calls.is_empty() && host.restarts == 0);

        v.latency_mode = LatencyModeChoice::Studio;
        for _ in 0..5 {
            run(&mut p, &v, &mut host, 64);
        }
        let studio_cfg = EngineConfig {
            latency_mode: LatencyMode::Studio,
            max_block: 512,
            ..EngineConfig::default()
        };
        let studio = vt_engine::latency_samples(&studio_cfg);
        assert_ne!(studio, live);
        assert_eq!(
            host.latency_calls,
            vec![studio],
            "latency reported exactly once"
        );
        assert!(p.restart_pending());
        assert!(p.shared.restart_pending.load(Ordering::Acquire));

        // The host restarts us: the engine is rebuilt with the new latency.
        assert_eq!(p.activate_with(&v, 48_000.0, 512, 2), studio);
        assert!(!p.restart_pending());
        run(&mut p, &v, &mut host, 64);
        assert_eq!(host.latency_calls.len(), 1);
    }

    #[test]
    fn clap_structural_change_only_requests_a_restart() {
        // CLAP: latency may only change inside activate(), so process() must
        // not report it; activate() does after the restart.
        let mut p = VocalTune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost {
            api: Some(PluginApi::Clap),
            ..FakeHost::default()
        };
        let mut v = ParamValues::default();
        v.latency_mode = LatencyModeChoice::Studio;
        for _ in 0..4 {
            run(&mut p, &v, &mut host, 64);
        }
        assert!(host.latency_calls.is_empty());
        assert_eq!(host.restarts, 1);
        // Reverting needs no host call either.
        v.latency_mode = LatencyModeChoice::Live;
        run(&mut p, &v, &mut host, 64);
        assert!(host.latency_calls.is_empty());
        assert_eq!(host.restarts, 1);
        assert!(!p.restart_pending());
    }

    #[test]
    fn same_latency_structural_change_uses_a_plain_restart() {
        let mut p = VocalTune::default();
        let mut v = ParamValues::default();
        v.latency_mode = LatencyModeChoice::Studio;
        p.activate_with(&v, 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        // Studio latency does not depend on the voice range in the current
        // engine, but the engine still has to be rebuilt.
        v.voice_range = VoiceRangeChoice::Soprano;
        run(&mut p, &v, &mut host, 64);
        run(&mut p, &v, &mut host, 64);
        let cfg = v.structural().engine_config(48_000.0, 512, 2);
        if vt_engine::latency_samples(&cfg) == p.latency().unwrap_or(0) {
            assert_eq!(host.restarts, 1);
            assert!(host.latency_calls.is_empty());
        } else {
            assert_eq!(host.latency_calls.len(), 1);
        }
    }

    #[test]
    fn reverting_a_structural_change_restores_the_latency() {
        let mut p = VocalTune::default();
        let live = p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        let mut v = ParamValues::default();
        v.latency_mode = LatencyModeChoice::Studio;
        run(&mut p, &v, &mut host, 64);
        v.latency_mode = LatencyModeChoice::Live;
        run(&mut p, &v, &mut host, 64);
        assert_eq!(host.latency_calls.last(), Some(&live));
        assert!(!p.restart_pending());
        assert!(!p.shared.restart_pending.load(Ordering::Acquire));
    }

    #[test]
    fn irrelevant_latency_edit_does_not_restart() {
        let mut p = VocalTune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        let mut v = ParamValues::default();
        v.studio_ms = 25.0; // Live mode is active
        run(&mut p, &v, &mut host, 64);
        assert!(host.latency_calls.is_empty());
        assert_eq!(host.restarts, 0);
    }

    #[test]
    fn standalone_never_calls_host_latency() {
        let mut p = VocalTune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost {
            api: Some(PluginApi::Standalone),
            ..FakeHost::default()
        };
        let mut v = ParamValues::default();
        v.latency_mode = LatencyModeChoice::Studio;
        run(&mut p, &v, &mut host, 64);
        assert!(host.latency_calls.is_empty());
        assert_eq!(host.restarts, 0);
        assert!(
            p.shared.restart_pending.load(Ordering::Acquire),
            "editor must see the pending restart"
        );
    }

    #[test]
    fn mono_to_stereo_copies_the_processed_channel() {
        let mut p = VocalTune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 1);
        let mut host = FakeHost::default();
        let n = 1024;
        let mut l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut r = vec![0.0f32; n];
        {
            let mut io: [&mut [f32]; 2] = [&mut l, &mut r];
            p.process_block(&ParamValues::default(), &mut io, &mut host);
        }
        assert_eq!(l, r);
        assert!(l.iter().any(|x| x.abs() > 0.1));
    }

    #[test]
    fn midi_events_are_drained_every_block() {
        let mut p = VocalTune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        host.events.push(NoteEvent::NoteOn {
            timing: 3,
            voice_id: VoiceID::Wildcard,
            channel: Channel::Number(0),
            key: Key::Number(64),
            velocity: 1.0,
        });
        host.events.push(NoteEvent::MidiCC {
            timing: 900,
            channel: 0,
            cc: 64,
            value: 1.0,
        });
        run(&mut p, &ParamValues::default(), &mut host, 128);
        assert!(host.events.is_empty());
        assert_eq!(p.midi.len(), 2);
        assert_eq!(
            p.midi.as_slice()[1].timing,
            127,
            "timing clamped into the block"
        );
    }

    #[test]
    fn nan_input_is_sanitized_by_the_chain() {
        let mut p = VocalTune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        let mut l = vec![f32::NAN; 4096];
        let mut r = vec![f32::INFINITY; 4096];
        {
            let mut io: [&mut [f32]; 2] = [&mut l, &mut r];
            p.process_block(&ParamValues::default(), &mut io, &mut host);
        }
        assert!(l.iter().chain(r.iter()).all(|x| x.is_finite()));
    }

    #[test]
    fn cpu_meter_is_published() {
        let mut p = VocalTune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        for _ in 0..50 {
            run(&mut p, &ParamValues::default(), &mut host, 256);
        }
        let load = p.shared.cpu_load.load(Ordering::Relaxed);
        assert!(load.is_finite() && load >= 0.0);
        assert!(p.shared.cpu_peak.load(Ordering::Relaxed) >= load);
    }
}
