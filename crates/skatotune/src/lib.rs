//! Skatotune: realtime vocal pitch correction as a VST3 / CLAP plugin
//! (contract: `docs/DESIGN.md` §3.6).
//!
//! This crate is the thin, host-facing shell around [`skato_engine::Engine`]:
//!
//! - [`params`]: every host parameter, the stable ids, and the pure mapping
//!   from parameter values to the engine's `Settings` and structural config.
//! - [`midi`]: host note events → the control law's MIDI vocabulary.
//! - [`learn`]: MIDI Learn Scale, kept as saved user state.
//! - [`shared`]: atomics and the telemetry mailbox shared with the editor.
//! - [`swap`]: structural changes by swapping in an engine built off the
//!   audio thread.
//! - [`editor`]: the egui editor with the scrolling pitch graph.
//! - [`labs`]: opt-in editor experiments (the header's LABS menu).
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
//! output gain and updates the CPU meter.
//!
//! When the user changes a structural parameter and leaves it alone for
//! [`STRUCTURAL_SETTLE_S`] (so a drag or a burst of clicks costs one rebuild,
//! not one per step), the new engine is built on nice-plug's background
//! thread and `process()` swaps it in at a block boundary behind a 3 ms fade
//! ([`swap`]); nothing is built or freed on the audio thread. A VST3 host
//! hears about a new latency only after that swap (nice-plug turns it into
//! `restartComponent(kLatencyChanged)`), so the latency it reads is always
//! that of the running engine, whether it re-activates us promptly, late or
//! not at all (HI-1); the request is repeated at 1, 3 and 7 s until it does
//! re-activate, in case it dropped the first one (R§9.5). A change that keeps
//! the latency needs no host involvement (HI-3): a voice-range change is
//! applied by the running engine itself (`Engine::set_voice_range`, at once,
//! behind its own fade), any other one is swapped. CLAP only allows a latency
//! change inside `activate()`, so a CLAP change that moves the latency asks
//! the host for a restart and the engine is rebuilt there; the editor offers
//! the request again as a click if the host takes its time. The standalone
//! has no host restart and always swaps.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use nice_plug::prelude::*;
use nice_plug_egui::{EguiEditor, EguiEditorState, RepaintNotifier};
use skato_core::VoiceRange;
use skato_engine::Engine;

pub mod editor;
pub mod labs;
pub mod keysource;
pub mod learn;
pub mod midi;
pub mod params;
pub mod shared;
pub mod swap;

use keysource::MidiScaleTracker;
use learn::LearnTracker;
use midi::MidiBuffer;
use params::{ParamValues, Structural, SkatotuneParams};
use shared::{CpuMeter, SharedState, pack_states};
use swap::{EngineBuilder, EngineSwapAudio, EngineTask, SwapFade};

/// A structural change must stay unchanged this long (seconds of audio)
/// before the host is asked to restart. Dragging the Live+/Studio value or
/// clicking through latency modes would otherwise restart once per step
/// (R§9.5 warns about rapid latency toggling).
pub const STRUCTURAL_SETTLE_S: f32 = 0.25;

/// Input at or below this magnitude counts as silence for the CLAP sleep
/// decision (−120 dBFS).
const QUIET_LEVEL: f32 = 1e-6;

/// VST3: seconds after a latency change at which `kLatencyChanged` is sent
/// again while the host has not re-activated the plugin. A host may drop the
/// first request (no component handler yet, e.g. a state restored during
/// project load, R§9.5); a few spaced retries, never a stream.
pub const LATENCY_RETRY_AT_S: [f32; 3] = [1.0, 3.0, 7.0];

/// The plugin instance. `pub` so the standalone binary can export it.
pub struct Skatotune {
    params: Arc<SkatotuneParams>,
    shared: Arc<SharedState>,
    editor_state: Arc<EguiEditorState>,
    /// Lets `process()` wake the editor (an atomic store).
    repaint: RepaintNotifier,
    /// The editor app until the wrapper asks for it (queried once).
    editor_app: Option<editor::SkatotuneEditor>,

    /// Boxed so the standalone can swap engines by moving a pointer.
    engine: Option<Box<Engine>>,
    /// Input samples the engine has processed since it was built or reset
    /// (mirrors its sample clock, so MIDI can be grouped by control hop).
    engine_pos: u64,
    /// What the running engine was built for.
    active: Option<Active>,
    /// Structural configuration requested (engine build or host restart) and
    /// not running yet, so the request is made exactly once per change.
    requested: Option<Structural>,
    /// A structural change waiting to settle, and how long (samples) it has
    /// been unchanged.
    settling: Option<(Structural, u64)>,
    /// Latency last reported to the host.
    reported_latency: u32,
    midi: MidiBuffer,
    learn: LearnTracker,
    /// MIDI Scale chord latch (audio thread).
    midi_scale: MidiScaleTracker,
    /// Key Source of the previous block (a change re-arms the chord latch).
    last_key_source: Option<params::KeySourceChoice>,
    /// Consecutive quiet input samples (CLAP sleep decision).
    quiet_run: u64,
    /// Audio-thread half of the engine handoff.
    swap: EngineSwapAudio,
    /// Builds engines on the background thread (the task executor's state).
    builder: Arc<EngineBuilder>,
    /// Output fades around an engine swap.
    fade: SwapFade,
    /// VST3: a latency change the host has not re-activated us for yet.
    latency_retry: Option<LatencyRetry>,
    cpu: CpuMeter,
}

/// Progress of the VST3 latency re-requests ([`LATENCY_RETRY_AT_S`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct LatencyRetry {
    /// Samples processed since the latency was reported.
    waited: u64,
    /// Re-requests sent so far.
    sent: usize,
}

/// The configuration the running engine was built with.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Active {
    structural: Structural,
    sample_rate: f32,
    max_block: usize,
    engine_channels: usize,
    latency: u32,
    /// Quiet input needed before the output is certainly silent: D plus the
    /// worst content wander and the unvoiced fade.
    quiet_tail: u64,
}

/// Samples of silent input after which the engine's output is silent too:
/// the reported latency, plus two of the range's longest periods (PSOLA and
/// Classic content wander stays within about one period, R§2.1), plus 10 ms
/// for the voiced → dry crossfade.
fn quiet_tail(latency: u32, sample_rate: f32, range: VoiceRange) -> u64 {
    let sr = if sample_rate.is_finite() && sample_rate > 0.0 {
        sample_rate
    } else {
        48_000.0
    };
    let t_max = sr / range.f_min().max(1.0);
    u64::from(latency) + (2.0 * t_max + 0.01 * sr).ceil() as u64
}

impl Default for Skatotune {
    fn default() -> Self {
        let params = Arc::new(SkatotuneParams::default());
        let shared = Arc::new(SharedState::default());
        let (swap_audio, swap_builder) = swap::engine_swap_channel();
        let editor_app = editor::SkatotuneEditor::new(params.clone(), shared.clone());
        Self {
            params,
            shared,
            editor_state: editor::default_state(),
            repaint: RepaintNotifier::new(),
            editor_app: Some(editor_app),
            engine: None,
            engine_pos: 0,
            active: None,
            requested: None,
            settling: None,
            reported_latency: 0,
            midi: MidiBuffer::default(),
            learn: LearnTracker::default(),
            midi_scale: MidiScaleTracker::default(),
            last_key_source: None,
            quiet_run: 0,
            swap: swap_audio,
            builder: Arc::new(EngineBuilder::new(swap_builder)),
            fade: SwapFade::new(48_000.0),
            latency_retry: None,
            cpu: CpuMeter::new(48_000.0),
        }
    }
}

/// What the block logic needs from the host. [`process()`](Plugin::process)
/// adapts nice-plug's `ProcessContext` to it; tests drive
/// [`Skatotune::process_block`] with a fake.
pub trait BlockHost {
    /// Next note event of this block, in timing order.
    fn next_event(&mut self) -> Option<NoteEvent<()>>;
    /// Report a new latency (nice-plug turns a change into a host restart).
    fn set_latency_samples(&mut self, samples: u32);
    /// Ask the host to deactivate and re-activate the plugin.
    fn request_restart(&mut self);
    /// Run `task` on nice-plug's background thread (the plugin's task
    /// executor, [`Skatotune::run_task`]).
    fn execute_background(&mut self, task: EngineTask);
    /// Which wrapper is hosting us (the restart protocol differs per format).
    fn api(&self) -> PluginApi;
}

struct ContextHost<'a, C>(&'a mut C);

impl<C: ProcessContext<Skatotune>> BlockHost for ContextHost<'_, C> {
    fn next_event(&mut self) -> Option<NoteEvent<()>> {
        self.0.next_event()
    }

    fn set_latency_samples(&mut self, samples: u32) {
        self.0.set_latency_samples(samples);
    }

    fn request_restart(&mut self) {
        self.0.request_restart();
    }

    fn execute_background(&mut self, task: EngineTask) {
        self.0.execute_background(task);
    }

    fn api(&self) -> PluginApi {
        self.0.plugin_api()
    }
}

impl Skatotune {
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
            // A fresh engine starts silent; no swap fade applies to it.
            self.fade = SwapFade::new(sample_rate);
            let mut engine = Box::new(Engine::new(cfg));
            // Auto-Key keeps what it has heard across a rebuild at the same
            // rate; otherwise it starts from the key saved with the project.
            match self.engine.as_ref().filter(|old| old.config().sample_rate == cfg.sample_rate) {
                Some(old) => engine.set_key_detector(old.key_detector()),
                None => {
                    if let Some(k) = keysource::unpack_key(self.params.detected_key.load(Ordering::Relaxed)) {
                        engine.seed_key(k);
                    }
                }
            }
            if let Some(consumer) = engine.take_telemetry() {
                self.shared.install_telemetry(consumer);
            }
            // The previous engine (if any) is dropped here. In a DAW this is
            // never the audio thread; the standalone never re-activates for a
            // structural change (it swaps engines built on the GUI thread).
            self.engine = Some(engine);
            self.engine_pos = 0;
        }
        let latency = self.engine.as_ref().map_or(0, |e| e.latency_samples());

        self.active = Some(Active {
            structural,
            sample_rate,
            max_block: cfg.max_block,
            engine_channels,
            latency,
            quiet_tail: quiet_tail(latency, sample_rate, structural.voice_range),
        });
        self.requested = None;
        self.settling = None;
        self.quiet_run = 0;
        self.reported_latency = latency;
        // The host re-activated us: any latency change has reached it.
        self.latency_retry = None;
        self.cpu = CpuMeter::new(sample_rate);
        self.shared
            .sample_rate
            .store(sample_rate, Ordering::Relaxed);
        self.shared.max_block.store(
            u32::try_from(cfg.max_block).unwrap_or(u32::MAX),
            Ordering::Relaxed,
        );
        self.shared
            .engine_channels
            .store(engine_channels as u32, Ordering::Relaxed);
        self.shared
            .latency_samples
            .store(latency, Ordering::Relaxed);
        self.shared
            .pending_latency_samples
            .store(latency, Ordering::Relaxed);
        self.shared.restart_pending.store(false, Ordering::Release);
        self.shared.awaiting_host.store(false, Ordering::Relaxed);
        latency
    }

    /// Latency of the running engine, if activated.
    pub fn latency(&self) -> Option<u32> {
        self.active.map(|a| a.latency)
    }

    /// Whether a structural change is requested but not running yet.
    pub fn restart_pending(&self) -> bool {
        self.requested.is_some()
    }

    /// The background-task executor's work: build an engine, or free
    /// returned ones and pass telemetry to the editor ([`swap`]). Runs on
    /// nice-plug's background thread; allocates. `pub` so tests and tools
    /// can play the background thread's part.
    pub fn run_task(&self, task: EngineTask) {
        self.builder.run(task, &self.shared);
    }

    /// Test seam: replace the engine handoff with a fresh one and return its
    /// builder half, so a test can hand engines over itself. The plugin's own
    /// builder stops reaching the audio thread.
    pub fn take_engine_swap(&mut self) -> swap::EngineSwapBuilder {
        let (audio, gui) = swap::engine_swap_channel();
        self.swap = audio;
        gui
    }

    /// The shared state the editor reads (meters, latency, restart flag).
    pub fn shared(&self) -> &Arc<SharedState> {
        &self.shared
    }

    /// The plugin's parameters (including the saved learned scale).
    pub fn params_ref(&self) -> &Arc<SkatotuneParams> {
        &self.params
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
        let api = host.api();
        self.shared.block_samples.store(
            u32::try_from(samples).unwrap_or(u32::MAX),
            Ordering::Relaxed,
        );

        // Structural changes first, so a swapped-in engine already runs (and
        // its clock groups the MIDI) in this block.
        if let Some(active) = self.active {
            self.check_structural(values, &active, samples, host);
            self.adopt_handed_engine(values, host);
        }

        // MIDI: group bends by the engine's control hop, learn scales.
        let (hop, phase) = match self.active {
            Some(a) => {
                let hop = skato_core::hop_for_rate(a.sample_rate).max(1);
                (hop as u32, (self.engine_pos % hop as u64) as u32)
            }
            None => (1, 0),
        };
        self.midi.begin_block(hop, phase);
        self.learn.begin_block(values.midi_mode);
        let following_midi = values.key_source == params::KeySourceChoice::MidiScale;
        if self.last_key_source != Some(values.key_source) {
            self.midi_scale.reset();
            self.last_key_source = Some(values.key_source);
        }
        let sr = self.active.map_or(48_000.0, |a| a.sample_rate);
        let block_len = u32::try_from(samples).unwrap_or(u32::MAX);
        while let Some(ev) = host.next_event() {
            if let Some(m) = midi::convert_event(&ev, block_len) {
                if matches!(m.event, skato_core::MidiEvent::NoteOn { velocity, .. } if velocity > 0.0) {
                    self.shared.midi_note_ons.fetch_add(1, Ordering::Relaxed);
                }
                if following_midi {
                    let pos = self.engine_pos.wrapping_add(u64::from(m.timing));
                    if self.midi_scale.event(&m.event, pos, sr, &self.params.midi_scale) {
                        self.shared.midi_latched.store(true, Ordering::Relaxed);
                    }
                } else {
                    self.learn.event(&m.event, &self.params.learned_scale);
                }
                self.midi.push(m);
            }
        }

        let Some(active) = self.active else {
            // Not activated: leave the audio untouched.
            return ProcessStatus::Normal;
        };

        let Some(engine) = self.engine.as_mut() else {
            return ProcessStatus::Normal;
        };
        let editor_open = self.editor_state.is_open();
        let learned_word = self.params.learned_scale.load(Ordering::Relaxed);
        if self.shared.key_clear.swap(false, Ordering::AcqRel) {
            engine.clear_key();
            self.params.detected_key.store(0, Ordering::Relaxed);
        }
        let reading = engine.key_reading();
        let mut settings = values.to_settings();
        let in_effect = keysource::apply(
            &mut settings,
            values,
            learned_word,
            self.params.midi_scale.load(Ordering::Relaxed),
            reading.confirmed,
        );
        // Labs experiments that change the sound, from their lock-free
        // mirror (applies with the editor closed and after a reload).
        labs::apply_sound(&mut settings, self.params.labs_sound_bits());
        engine.set_settings(&settings);
        engine.set_telemetry_enabled(editor_open);

        // CLAP may put the plugin to sleep once its output is quiet; track
        // how long the input has been silent so we only allow that when the
        // delay line has emptied too.
        if api == PluginApi::Clap {
            let quiet = io
                .iter()
                .all(|ch| ch.iter().all(|x| x.abs() <= QUIET_LEVEL));
            self.quiet_run = if quiet {
                self.quiet_run.saturating_add(samples as u64)
            } else {
                0
            };
        }

        let channels = active.engine_channels.min(io.len());
        if let Some(engine_io) = io.get_mut(..channels) {
            let processed = engine_io.iter().map(|c| c.len()).min().unwrap_or(0);
            engine.process(engine_io, self.midi.as_slice());
            self.engine_pos = self.engine_pos.wrapping_add(processed as u64);
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

        // Fades around an engine swap (idle otherwise).
        self.fade.apply(io, samples);

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

        // A learned scale is in effect in every MIDI mode until cleared.
        self.shared
            .learned_states
            .store(pack_states(learn::unpack(learned_word)), Ordering::Relaxed);
        publish_key(engine, &self.shared, &self.params, in_effect);
        self.shared
            .telemetry_dropped
            .store(engine.stats().telemetry_dropped, Ordering::Relaxed);
        if editor_open {
            self.repaint.request_repaint();
        }
        if api == PluginApi::Vst3 {
            self.retry_latency_report(&active, samples, host);
        }
        let (avg, peak) = self.cpu.update(started.elapsed().as_secs_f32(), samples);
        self.shared.cpu_load.store(avg, Ordering::Relaxed);
        self.shared.cpu_peak.store(peak, Ordering::Relaxed);

        // The engine delays the signal by its latency, so audio keeps coming
        // out for that long after the input goes silent. nice-plug maps Tail
        // to VST3 getTailSamples = D, but to CLAP_PROCESS_CONTINUE ("never
        // sleep"); under CLAP, Normal (CONTINUE_IF_NOT_QUIET) is returned once
        // the input has been silent long enough for the output to be silent.
        if api == PluginApi::Clap && self.quiet_run >= active.quiet_tail {
            ProcessStatus::Normal
        } else if active.latency > 0 {
            ProcessStatus::Tail(active.latency)
        } else {
            ProcessStatus::Normal
        }
    }

    /// Swap in an engine built off the audio thread for the pending
    /// structural change ([`swap`]): fade the running engine out, swap at a
    /// block boundary once it is silent, and hand the old engine back to be
    /// freed. An engine that no longer matches what is wanted goes straight
    /// back. Only then does a VST3 host hear about a new latency. Nothing is
    /// allocated or freed here.
    fn adopt_handed_engine(&mut self, values: &ParamValues, host: &mut impl BlockHost) {
        let Some(active) = self.active else { return };
        let Ok((id, offered)) = self.swap.incoming.peek().map(|(id, e)| (*id, *e.config())) else {
            // Nothing waiting: a called-off fade comes back (no-op otherwise).
            self.fade.fade_in();
            return;
        };
        // Room to send an engine back is required before taking one.
        if self.swap.retired.slots() == 0 {
            return;
        }
        let api = host.api();
        let wanted = values.structural();
        let cfg =
            wanted.engine_config(active.sample_rate, active.max_block, active.engine_channels);
        let latency = skato_engine::latency_samples(&offered);
        let useful = self.requested.is_some_and(|r| r.same_engine(&wanted))
            && offered == cfg
            // CLAP latency may only change inside activate().
            && (api != PluginApi::Clap || latency == active.latency);
        if !useful {
            if let Ok((_, stale)) = self.swap.incoming.pop() {
                self.retire(stale);
            }
            self.fade.fade_in();
            host.execute_background(EngineTask::Collect);
            return;
        }
        if !self.fade.silent() {
            self.fade.fade_out(); // swap once the running engine is silent
            return;
        }
        // Publish the id before taking the engine: the builder pairs the
        // telemetry ring with it once it sees the slot free.
        self.shared.adopted_engine.store(id, Ordering::Release);
        let Ok((_, engine)) = self.swap.incoming.pop() else {
            return;
        };
        let latency = engine.latency_samples();
        let old = self.engine.replace(engine);
        // Same activation, same rate: Auto-Key continues (fixed-size copy).
        if let (Some(new), Some(old)) = (self.engine.as_mut(), old.as_ref()) {
            new.set_key_detector(old.key_detector());
        }
        self.engine_pos = 0;
        self.active = Some(Active {
            structural: wanted,
            latency,
            quiet_tail: quiet_tail(latency, active.sample_rate, wanted.voice_range),
            ..active
        });
        self.requested = None;
        self.settling = None;
        self.fade.swapped(latency);
        self.shared
            .latency_samples
            .store(latency, Ordering::Relaxed);
        self.shared
            .pending_latency_samples
            .store(latency, Ordering::Relaxed);
        self.shared.restart_pending.store(false, Ordering::Release);
        self.shared.awaiting_host.store(false, Ordering::Relaxed);
        if let Some(old) = old {
            self.retire(old);
        }
        host.execute_background(EngineTask::Collect);
        if api == PluginApi::Vst3 && latency != self.reported_latency {
            host.set_latency_samples(latency);
            self.reported_latency = latency;
            self.latency_retry = Some(LatencyRetry::default());
        }
    }

    /// Send an engine back to the background thread to be freed.
    fn retire(&mut self, engine: Box<Engine>) {
        if let Err(rtrb::PushError::Full(engine)) = self.swap.retired.push(engine) {
            // Unreachable (callers check for a free slot). Leaking beats
            // freeing a whole engine on the audio thread.
            std::mem::forget(engine);
        }
    }

    /// VST3: repeat `kLatencyChanged` at [`LATENCY_RETRY_AT_S`] while the
    /// host has not re-activated us since a latency change.
    fn retry_latency_report(&mut self, active: &Active, samples: usize, host: &mut impl BlockHost) {
        let Some(retry) = self.latency_retry.as_mut() else {
            return;
        };
        retry.waited = retry.waited.saturating_add(samples as u64);
        match LATENCY_RETRY_AT_S.get(retry.sent) {
            Some(at) if retry.waited >= (at * active.sample_rate.max(1.0)) as u64 => {
                retry.sent += 1;
                host.request_restart();
            }
            Some(_) => {}
            None => self.latency_retry = None,
        }
    }

    /// Detect structural parameter changes and request the new engine exactly
    /// once, after the new value has been left alone for
    /// [`STRUCTURAL_SETTLE_S`].
    ///
    /// - VST3, standalone, and CLAP when the latency stays the same: build it
    ///   on the background thread and swap it in (`adopt_handed_engine`); a
    ///   VST3 host is told the new latency after the swap.
    /// - CLAP with a new latency: latency may only change inside
    ///   `activate()`, so a restart is requested (`host.request_restart()` is
    ///   thread-safe) and `activate()` builds the engine and reports it.
    ///   Reporting it from here would make nice-plug call `latency.changed()`
    ///   from the main thread, possibly while deactivated (clap-validator
    ///   rejects that).
    fn check_structural(
        &mut self,
        values: &ParamValues,
        active: &Active,
        samples: usize,
        host: &mut impl BlockHost,
    ) {
        let wanted = values.structural();
        let api = host.api();
        if wanted.same_engine(&active.structural) {
            self.settling = None;
            if self.requested.take().is_some() {
                // Reverted before the new engine arrived: nothing to apply.
                // (A VST3 host was never told another latency; re-reporting
                // the running one is a safety net.)
                self.shared.restart_pending.store(false, Ordering::Release);
                self.shared.awaiting_host.store(false, Ordering::Relaxed);
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
            self.settling = None;
            return; // already requested: building, or waiting for the host
        }

        // HI-3: a voice-range change that keeps the latency is applied by
        // the running engine itself (`Engine::set_voice_range`: RT-safe, a
        // 5 ms fade, the new range's detector and shifter at the next hop,
        // back in after D samples; notes, MIDI and telemetry carry on). No
        // build, no swap, nothing for the host to do, and no settle delay.
        let in_place = Structural {
            voice_range: wanted.voice_range,
            ..active.structural
        };
        if in_place.same_engine(&wanted)
            && self
                .engine
                .as_mut()
                .is_some_and(|e| e.set_voice_range(wanted.voice_range))
        {
            self.voice_range_switched(active, in_place, host);
            return;
        }

        // Debounce: count how long this exact configuration has been wanted.
        let settle_samples = (STRUCTURAL_SETTLE_S * active.sample_rate.max(1.0)) as u64;
        let waited = match self.settling {
            Some((candidate, waited)) if candidate.same_engine(&wanted) => {
                waited.saturating_add(samples as u64)
            }
            _ => samples as u64,
        };
        if waited < settle_samples {
            self.settling = Some((wanted, waited));
            return;
        }
        self.settling = None;

        self.requested = Some(wanted);
        let cfg =
            wanted.engine_config(active.sample_rate, active.max_block, active.engine_channels);
        let latency = skato_engine::latency_samples(&cfg);
        self.shared
            .pending_latency_samples
            .store(latency, Ordering::Relaxed);
        self.shared.restart_pending.store(true, Ordering::Release);
        let host_restart = api == PluginApi::Clap && latency != active.latency;
        self.shared
            .awaiting_host
            .store(host_restart, Ordering::Relaxed);
        if host_restart {
            host.request_restart();
        } else {
            host.execute_background(EngineTask::Build(cfg));
        }
    }

    /// Bookkeeping after the running engine accepted a voice-range change
    /// (`check_structural`): it now runs `structural` with the same latency.
    /// A build or restart requested for an earlier change is void (a handed
    /// engine for it is sent back by `adopt_handed_engine`; a CLAP host that
    /// restarts anyway re-activates with a matching engine).
    fn voice_range_switched(
        &mut self,
        active: &Active,
        structural: Structural,
        host: &mut impl BlockHost,
    ) {
        self.active = Some(Active {
            structural,
            quiet_tail: quiet_tail(active.latency, active.sample_rate, structural.voice_range),
            ..*active
        });
        self.requested = None;
        self.settling = None;
        self.shared.restart_pending.store(false, Ordering::Release);
        self.shared.awaiting_host.store(false, Ordering::Relaxed);
        self.shared
            .pending_latency_samples
            .store(active.latency, Ordering::Relaxed);
        // A VST3 host is only ever told the running engine's latency; this
        // re-reports it if an earlier request had moved it (safety net, as
        // for a reverted change).
        if host.api() == PluginApi::Vst3 && self.reported_latency != active.latency {
            host.set_latency_samples(active.latency);
            self.reported_latency = active.latency;
        }
    }
}

impl Plugin for Skatotune {
    const NAME: &'static str = "Skatotune";
    const VENDOR: &'static str = "Skatotune";
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

    type Editor = EguiEditor<editor::SkatotuneEditor>;
    type SysExMessage = ();
    type BackgroundTask = EngineTask;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let builder = self.builder.clone();
        let shared = self.shared.clone();
        Box::new(move |task| builder.run(task, &shared))
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
        // Signal state only: the learned scale is user state and lives in the
        // parameters (`learn`), so it survives this.
        if let Some(engine) = self.engine.as_mut() {
            engine.reset();
        }
        self.engine_pos = 0;
        self.quiet_run = 0;
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

impl ClapPlugin for Skatotune {
    const CLAP_ID: &'static str = "com.skatotune.skatotune";
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
        context.add_section("Skatotune", |section| {
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

impl Vst3Plugin for Skatotune {
    // Exactly 16 bytes; fixed forever.
    const VST3_CLASS_ID: [u8; 16] = *b"SkatotuneVT1pcor";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::PitchShift];
}

nice_export_clap!(Skatotune);
nice_export_vst3!(Skatotune);

/// Auto-Key and the Key Source for the editor, and a newly confirmed key
/// into the saved state (audio thread; atomic loads and stores only).
fn publish_key(engine: &Engine, shared: &SharedState, params: &SkatotuneParams, in_effect: keysource::InEffect) {
    let r = engine.key_reading();
    shared.key_proposal.store(r.proposal.map_or(0, keysource::pack_key), Ordering::Relaxed);
    shared.key_confidence.store(r.confidence, Ordering::Relaxed);
    shared.key_evidence.store(r.evidence_s, Ordering::Relaxed);
    let saved = if r.saved { keysource::KEY_SAVED } else { 0 };
    shared.key_confirmed.store(r.confirmed.map_or(0, |k| keysource::pack_key(k) | saved), Ordering::Relaxed);
    shared.key_in_effect.store(keysource::pack_in_effect(in_effect), Ordering::Relaxed);
    if let Some(k) = r.confirmed.filter(|_| !r.saved) {
        let w = keysource::pack_key(k);
        if params.detected_key.load(Ordering::Relaxed) != w {
            params.detected_key.store(w, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
// Tests read best as "defaults, then change what matters".
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use nice_plug::midi::{Channel, Key, VoiceID};
    use labs::LabsSettings;
    use params::{LatencyModeChoice, VoiceRangeChoice};
    use skato_core::EngineConfig;

    /// Records what the plugin asks of the host.
    #[derive(Default)]
    struct FakeHost {
        events: Vec<NoteEvent<()>>,
        latency_calls: Vec<u32>,
        restarts: usize,
        /// Background tasks posted and not run yet ([`run_background`]).
        tasks: Vec<EngineTask>,
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
        fn execute_background(&mut self, task: EngineTask) {
            self.tasks.push(task);
        }
        fn api(&self) -> PluginApi {
            self.api.unwrap_or(PluginApi::Vst3)
        }
    }

    fn run(
        p: &mut Skatotune,
        values: &ParamValues,
        host: &mut FakeHost,
        n: usize,
    ) -> Vec<Vec<f32>> {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        run_signal(p, values, host, &input)
    }

    /// One stereo block of `input` (both channels).
    fn run_signal(
        p: &mut Skatotune,
        values: &ParamValues,
        host: &mut FakeHost,
        input: &[f32],
    ) -> Vec<Vec<f32>> {
        let mut l = input.to_vec();
        let mut r = l.clone();
        {
            let mut io: [&mut [f32]; 2] = [&mut l, &mut r];
            p.process_block(values, &mut io, host);
        }
        vec![l, r]
    }

    /// Run 64-sample blocks until a structural change has had time to
    /// settle (just over `STRUCTURAL_SETTLE_S` at 48 kHz).
    fn settle(p: &mut Skatotune, values: &ParamValues, host: &mut FakeHost) {
        let blocks = (STRUCTURAL_SETTLE_S * 48_000.0 / 64.0) as usize + 2;
        for _ in 0..blocks {
            run(p, values, host, 64);
        }
    }

    #[test]
    fn unactivated_plugin_passes_audio_through() {
        let mut p = Skatotune::default();
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
        let mut p = Skatotune::default();
        let latency = p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let cfg = EngineConfig {
            max_block: 512,
            ..EngineConfig::default()
        };
        assert_eq!(latency, skato_engine::latency_samples(&cfg));
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
        // A 76 Hz sine is below the Alto/Tenor range, so the engine treats it
        // as unvoiced and passes it as an exact delay; the plugin must not
        // add anything.
        let mut p = Skatotune::default();
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
    fn structural_change_is_built_off_thread_then_reported_once() {
        let mut p = Skatotune::default();
        let live = p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        let mut v = ParamValues::default();
        run(&mut p, &v, &mut host, 64);
        assert!(host.latency_calls.is_empty() && host.restarts == 0);
        assert!(host.tasks.is_empty());

        v.latency_mode = LatencyModeChoice::Studio;
        run(&mut p, &v, &mut host, 64);
        assert!(host.tasks.is_empty(), "nothing before it settles");
        settle(&mut p, &v, &mut host);
        let studio_cfg = v.structural().engine_config(48_000.0, 512, 2);
        let studio = skato_engine::latency_samples(&studio_cfg);
        assert_ne!(studio, live);
        assert_eq!(host.tasks, vec![EngineTask::Build(studio_cfg)]);
        assert!(
            host.latency_calls.is_empty(),
            "no new latency before its engine runs"
        );
        assert!(p.restart_pending());
        assert!(p.shared.restart_pending.load(Ordering::Acquire));
        assert!(!p.shared.awaiting_host.load(Ordering::Relaxed));

        until_swapped(&mut p, &v, &mut host);
        assert_eq!(p.latency(), Some(studio));
        assert_eq!(host.latency_calls, vec![studio], "reported exactly once");
        assert_eq!(
            host.restarts, 0,
            "nice-plug's kLatencyChanged is the request"
        );

        // The host restarts us: the running engine is kept.
        let generation = p.shared.engine_generation.load(Ordering::Relaxed);
        assert_eq!(p.activate_with(&v, 48_000.0, 512, 2), studio);
        assert_eq!(
            p.shared.engine_generation.load(Ordering::Relaxed),
            generation
        );
        assert!(!p.restart_pending());
        run(&mut p, &v, &mut host, 64);
        assert_eq!(host.latency_calls.len(), 1);
    }

    #[test]
    fn clap_latency_change_only_requests_a_restart() {
        // CLAP: latency may only change inside activate(), so process() must
        // neither report it nor swap in an engine with another latency;
        // activate() does both after the restart.
        let mut p = Skatotune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost {
            api: Some(PluginApi::Clap),
            ..FakeHost::default()
        };
        let mut v = ParamValues::default();
        v.latency_mode = LatencyModeChoice::Studio;
        settle(&mut p, &v, &mut host);
        assert!(host.latency_calls.is_empty());
        assert!(host.tasks.is_empty(), "no engine is built for a swap");
        assert_eq!(host.restarts, 1);
        assert!(p.shared.awaiting_host.load(Ordering::Relaxed));
        // Reverting needs no host call either.
        v.latency_mode = LatencyModeChoice::Live;
        run(&mut p, &v, &mut host, 64);
        assert!(host.latency_calls.is_empty());
        assert_eq!(host.restarts, 1);
        assert!(!p.restart_pending());
        assert!(!p.shared.awaiting_host.load(Ordering::Relaxed));
    }

    #[test]
    fn reverting_before_the_swap_keeps_the_running_engine() {
        let mut p = Skatotune::default();
        let live = p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let live_cfg = p.engine.as_ref().map(|e| *e.config());
        let mut host = FakeHost::default();
        let mut v = ParamValues::default();
        v.latency_mode = LatencyModeChoice::Studio;
        settle(&mut p, &v, &mut host);
        run_background(&mut p, &mut host); // the Studio engine is waiting
        v.latency_mode = LatencyModeChoice::Live;
        for _ in 0..20 {
            run(&mut p, &v, &mut host, 64);
            run_background(&mut p, &mut host);
        }
        assert!(
            host.latency_calls.is_empty(),
            "the host never heard of Studio"
        );
        assert_eq!(p.latency(), Some(live));
        assert_eq!(p.engine.as_ref().map(|e| *e.config()), live_cfg);
        assert!(p.swap.incoming.peek().is_err(), "stale engine sent back");
        assert_eq!(p.swap.retired.slots(), 4, "and freed off the audio thread");
        assert!(!p.restart_pending());
        assert!(!p.shared.restart_pending.load(Ordering::Acquire));
    }

    #[test]
    fn irrelevant_latency_edit_does_not_restart() {
        let mut p = Skatotune::default();
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
        let mut p = Skatotune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost {
            api: Some(PluginApi::Standalone),
            ..FakeHost::default()
        };
        let mut v = ParamValues::default();
        v.latency_mode = LatencyModeChoice::Studio;
        settle(&mut p, &v, &mut host);
        assert!(p.shared.restart_pending.load(Ordering::Acquire));
        until_swapped(&mut p, &v, &mut host);
        assert!(host.latency_calls.is_empty());
        assert_eq!(host.restarts, 0);
        let cfg = v.structural().engine_config(48_000.0, 512, 2);
        assert_eq!(p.latency(), Some(skato_engine::latency_samples(&cfg)));
    }

    #[test]
    fn mono_to_stereo_copies_the_processed_channel() {
        let mut p = Skatotune::default();
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
        let mut p = Skatotune::default();
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
        let mut p = Skatotune::default();
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
        let mut p = Skatotune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        for _ in 0..50 {
            run(&mut p, &ParamValues::default(), &mut host, 256);
        }
        let load = p.shared.cpu_load.load(Ordering::Relaxed);
        assert!(load.is_finite() && load >= 0.0);
        assert!(p.shared.cpu_peak.load(Ordering::Relaxed) >= load);
    }

    // ---------------------------------------------------------------------
    // Review regression tests
    // ---------------------------------------------------------------------

    /// Harmonic test voice (1/k partials) at `hz`, starting at phase `t0`.
    fn voice(n: usize, hz: f32, sr: f32, t0: usize) -> Vec<f32> {
        (t0..t0 + n)
            .map(|i| {
                let ph = std::f32::consts::TAU * hz * i as f32 / sr;
                (1..=8)
                    .map(|k| (ph * k as f32).sin() / k as f32)
                    .sum::<f32>()
                    * 0.3
            })
            .collect()
    }

    /// Fundamental of `x` by the YIN difference function, searched in
    /// `lo..hi` Hz (accurate to well under 5 c on these signals).
    fn pitch_hz(x: &[f32], sr: f32, lo: f32, hi: f32) -> f32 {
        let tau_min = (sr / hi) as usize;
        let tau_max = (sr / lo) as usize + 1;
        let w = x.len() - tau_max - 1;
        let d: Vec<f32> = (0..=tau_max)
            .map(|tau| (0..w).map(|i| (x[i] - x[i + tau]).powi(2)).sum())
            .collect();
        let mut cum = 0.0;
        let mut cmnd = vec![1.0f32; tau_max + 1];
        for tau in 1..=tau_max {
            cum += d[tau];
            cmnd[tau] = d[tau] * tau as f32 / cum.max(1e-12);
        }
        let mut best = tau_min;
        for tau in tau_min..tau_max {
            if cmnd[tau] < 0.15 {
                best = tau;
                while best + 1 < tau_max && cmnd[best + 1] < cmnd[best] {
                    best += 1;
                }
                break;
            }
            if cmnd[tau] < cmnd[best] {
                best = tau;
            }
        }
        let (a, b, c) = (cmnd[best - 1], cmnd[best], cmnd[best + 1]);
        let den = a - 2.0 * b + c;
        let shift = if den.abs() > 1e-12 {
            0.5 * (a - c) / den
        } else {
            0.0
        };
        sr / (best as f32 + shift)
    }

    fn cents(a: f32, b: f32) -> f32 {
        1200.0 * (a / b).log2()
    }

    /// Feed `seconds` of the voice through the plugin in `block`-sample
    /// blocks (events go with the first block) and return the mono output.
    #[allow(clippy::too_many_arguments)]
    fn render(
        p: &mut Skatotune,
        v: &ParamValues,
        host: &mut FakeHost,
        hz: f32,
        seconds: f32,
        block: usize,
        t0: &mut usize,
    ) -> Vec<f32> {
        let n = (seconds * 48_000.0) as usize;
        let mut out = Vec::with_capacity(n);
        let mut done = 0;
        while done < n {
            let len = block.min(n - done);
            let mut l = voice(len, hz, 48_000.0, *t0);
            let mut r = l.clone();
            {
                let mut io: [&mut [f32]; 2] = [&mut l, &mut r];
                p.process_block(v, &mut io, host);
            }
            out.extend_from_slice(&l);
            *t0 += len;
            done += len;
        }
        out
    }

    fn note_on(timing: u32, key: u8) -> NoteEvent<()> {
        NoteEvent::NoteOn {
            timing,
            voice_id: VoiceID::Wildcard,
            channel: Channel::Number(0),
            key: Key::Number(key),
            velocity: 1.0,
        }
    }

    fn tail_pitch(y: &[f32]) -> f32 {
        pitch_hz(&y[y.len() - 4096..], 48_000.0, 120.0, 900.0)
    }

    /// RT-3: a pitch-bend flood before a note-off must not leave the note
    /// held (MIDI Target would keep pulling the voice to it).
    #[test]
    fn bend_flood_does_not_leave_a_target_note_stuck() {
        let mut p = Skatotune::default();
        let mut v = ParamValues::default();
        v.midi_mode = params::MidiModeChoice::Target;
        p.activate_with(&v, 48_000.0, 2048, 2);
        let mut host = FakeHost::default();
        let a3_35 = 220.0 * 2f32.powf(35.0 / 1200.0);
        let mut t0 = 0;
        host.events.push(note_on(0, 60));
        let held = render(&mut p, &v, &mut host, a3_35, 0.5, 2048, &mut t0);
        let c4 = 261.626;
        assert!(
            cents(tail_pitch(&held), c4).abs() < 15.0,
            "held C4 pulls the voice: {} Hz",
            tail_pitch(&held)
        );
        // One block: 1000 bends (ending centred), then the note-off.
        for i in 0..1000u32 {
            let value = if i == 999 {
                0.5
            } else {
                0.5 + 0.3 * ((i as f32) * 0.1).sin()
            };
            host.events.push(NoteEvent::MidiPitchBend {
                timing: i * 2,
                channel: 0,
                value,
            });
        }
        host.events.push(NoteEvent::NoteOff {
            timing: 2040,
            voice_id: VoiceID::Wildcard,
            channel: Channel::Number(0),
            key: Key::Number(60),
            velocity: 0.0,
        });
        let released = render(&mut p, &v, &mut host, a3_35, 0.6, 2048, &mut t0);
        let got = tail_pitch(&released);
        assert!(
            cents(got, a3_35).abs() < 15.0,
            "released: the voice must pass through at A3+35 c, got {got} Hz ({:+.0} c from C4)",
            cents(got, c4)
        );
    }

    /// HI-4 / CL-4: the learned scale survives a host reset and an engine
    /// rebuild while MIDI Mode stays Learn.
    #[test]
    fn learned_scale_survives_reset_and_rebuild() {
        let mut p = Skatotune::default();
        let mut v = ParamValues::default();
        p.activate_with(&v, 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        let mut t0 = 0;
        // Enter Learn (from Off), then play C, E, G.
        render(&mut p, &v, &mut host, 450.0, 0.05, 512, &mut t0);
        v.midi_mode = params::MidiModeChoice::Learn;
        for k in [60, 64, 67] {
            host.events.push(note_on(0, k));
        }
        let learned = render(&mut p, &v, &mut host, 450.0, 0.5, 512, &mut t0);
        let g4 = 391.995;
        assert!(
            cents(tail_pitch(&learned), g4).abs() < 15.0,
            "450 Hz snaps to G4 of the learned C-E-G: {} Hz",
            tail_pitch(&learned)
        );

        <Skatotune as Plugin>::reset(&mut p);
        let after_reset = render(&mut p, &v, &mut host, 450.0, 0.5, 512, &mut t0);
        assert!(
            cents(tail_pitch(&after_reset), g4).abs() < 15.0,
            "after reset: {} Hz (A4 = 440 means the scale was lost)",
            tail_pitch(&after_reset)
        );

        // A different host block size rebuilds the engine.
        let generation = p.shared.engine_generation.load(Ordering::Relaxed);
        p.activate_with(&v, 48_000.0, 1024, 2);
        assert_ne!(
            p.shared.engine_generation.load(Ordering::Relaxed),
            generation
        );
        let rebuilt = render(&mut p, &v, &mut host, 450.0, 0.5, 512, &mut t0);
        assert!(cents(tail_pitch(&rebuilt), g4).abs() < 15.0);
        assert_eq!(
            p.shared.learned(),
            learn::unpack(learn::pack(&[0, 4, 7])),
            "the editor sees the learned scale"
        );
    }

    /// HI-4 / CL-4: Learn edits the scale (R§7.6.3), so the learned scale
    /// stays in effect after leaving Learn (Off, and Auto with no notes held)
    /// until the user clears it; the editor keeps showing it meanwhile.
    #[test]
    fn learned_scale_survives_leaving_learn_until_cleared() {
        let mut p = Skatotune::default();
        let mut v = ParamValues::default();
        p.activate_with(&v, 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        let mut t0 = 0;
        render(&mut p, &v, &mut host, 450.0, 0.05, 512, &mut t0);
        v.midi_mode = params::MidiModeChoice::Learn;
        for k in [60, 64, 67] {
            host.events.push(note_on(0, k));
        }
        render(&mut p, &v, &mut host, 450.0, 0.3, 512, &mut t0);
        // Keys released: Auto then has no held notes and uses the scale.
        for k in [60, 64, 67] {
            host.events.push(NoteEvent::NoteOff {
                timing: 0,
                voice_id: VoiceID::Wildcard,
                channel: Channel::Number(0),
                key: Key::Number(k),
                velocity: 0.0,
            });
        }
        render(&mut p, &v, &mut host, 450.0, 0.05, 512, &mut t0);
        let g4 = 391.995;
        let learned = learn::unpack(learn::pack(&[0, 4, 7]));

        for mode in [params::MidiModeChoice::Off, params::MidiModeChoice::Auto] {
            v.midi_mode = mode;
            let y = render(&mut p, &v, &mut host, 450.0, 0.5, 512, &mut t0);
            assert!(
                cents(tail_pitch(&y), g4).abs() < 15.0,
                "{mode:?}: 450 Hz must still snap to G4 of the learned C-E-G, got {} Hz",
                tail_pitch(&y)
            );
            assert_eq!(p.shared.learned(), learned, "{mode:?}: editor sees it");
        }

        // Clearing it returns to the Key/Scale parameters (Chromatic: A4 for
        // the next note; G is still legal, so the running note may be kept).
        p.params.clear_learned_scale();
        for _ in 0..20 {
            let mut l = vec![0.0f32; 480];
            let mut r = l.clone();
            let mut io: [&mut [f32]; 2] = [&mut l, &mut r];
            p.process_block(&v, &mut io, &mut host);
        }
        let y = render(&mut p, &v, &mut host, 450.0, 0.5, 512, &mut t0);
        assert!(
            cents(tail_pitch(&y), 440.0).abs() < 15.0,
            "cleared: chromatic A4, got {} Hz",
            tail_pitch(&y)
        );
        assert_eq!(p.shared.learned(), None);
    }

    /// HI-4: loading a project (or preset) saved without a learned scale
    /// clears the one in memory: the field is always part of the state.
    #[test]
    fn loading_a_state_without_a_learned_scale_clears_it() {
        let p = Skatotune::default();
        p.params
            .learned_scale
            .store(learn::pack(&[2, 9]), Ordering::Relaxed);
        let plain = Skatotune::default().params.serialize_fields();
        assert!(plain.contains_key("learned-scale"), "{plain:?}");
        p.params.deserialize_fields(&plain);
        assert_eq!(p.params.learned_states(), None);
    }

    /// Sing `notes` (Hz) for 0.4 s each, with 60 ms of silence between them
    /// when `gaps`, and return the last note's output and its frequency.
    fn sing_notes(p: &mut Skatotune, v: &ParamValues, notes: &[f32], gaps: bool) -> (Vec<f32>, f32) {
        let mut host = FakeHost::default();
        let mut t0 = 0;
        let mut last = Vec::new();
        for &hz in notes {
            last = render(p, v, &mut host, hz, 0.4, 512, &mut t0);
            if gaps {
                for _ in 0..6 {
                    run_signal(p, v, &mut host, &[0.0; 480]);
                }
            }
        }
        (last, notes.last().copied().unwrap_or(0.0))
    }

    /// The Labs experiments that change the sound reach the engine through
    /// their lock-free mirror, with the editor closed, and come back with a
    /// saved project: a singer 30 c sharp is corrected down to A440 by
    /// default and left on their own tuning with Follow singer's tuning.
    #[test]
    fn labs_sound_experiments_apply_with_the_editor_closed_and_after_a_reload() {
        let sharp = 2f32.powf(30.0 / 1200.0);
        let line: Vec<f32> = [220.0f32, 246.94, 261.63, 293.66, 329.63, 293.66, 261.63, 246.94]
            .iter()
            .cycle()
            .take(24)
            .map(|hz| hz * sharp)
            .collect();
        let v = ParamValues::default();
        let offset = |p: &mut Skatotune| {
            p.activate_with(&v, 48_000.0, 512, 2);
            assert!(!p.editor_state.is_open());
            let (y, hz) = sing_notes(p, &v, &line, true);
            cents(tail_pitch(&y), hz / sharp)
        };
        let mut plain = Skatotune::default();
        let a440 = offset(&mut plain);
        assert!(a440.abs() < 5.0, "A440: corrected to the grid, {a440:+.1} c");

        let mut follow = Skatotune::default();
        follow.params.set_labs(LabsSettings { follow_singer: true, ..LabsSettings::DEFAULT });
        let own = offset(&mut follow);
        assert!((own - 30.0).abs() < 5.0, "follow: left on the singer's tuning, {own:+.1} c");

        // A project saved with it loads with it, before any editor opens.
        let saved = follow.params.serialize_fields();
        let mut reloaded = Skatotune::default();
        reloaded.params.deserialize_fields(&saved);
        assert!(reloaded.params.labs().follow_singer);
        let again = offset(&mut reloaded);
        assert!((again - 30.0).abs() < 5.0, "after a reload: {again:+.1} c");

        // Learned note hold reaches the engine too: legato notes make the note
        // decision propose switches, and the model judges them.
        let mut hold = Skatotune::default();
        hold.params.set_labs(LabsSettings { learned_note_hold: true, ..LabsSettings::DEFAULT });
        hold.activate_with(&v, 48_000.0, 512, 2);
        sing_notes(&mut hold, &v, &line, false);
        let judged = hold.engine.as_ref().map_or(0, |e| e.note_hold_stats().proposals);
        assert!(judged > 0, "the learned veto ran");
        assert_eq!(plain.engine.as_ref().map(|e| e.note_hold_stats().proposals), Some(0));
    }

    fn note_off(timing: u32, key: u8) -> NoteEvent<()> {
        NoteEvent::NoteOff {
            timing,
            voice_id: VoiceID::Wildcard,
            channel: Channel::Number(0),
            key: Key::Number(key),
            velocity: 0.0,
        }
    }

    /// F4 + 30 c: corrected to F# in D major (or to a D-F#-A chord).
    const F4_SHARPISH: f32 = 349.228 * 1.017_478_8;
    const FS4: f32 = 369.994;

    /// #5 MIDI Scale: a chord on the MIDI input sets the scale, stays latched
    /// after its release, overrides MIDI Target (its notes are not targets),
    /// and is saved with the project.
    #[test]
    fn midi_scale_latches_a_chord_tunes_to_it_and_saves_it() {
        let mut v = ParamValues::default();
        v.key_source = params::KeySourceChoice::MidiScale;
        v.midi_mode = params::MidiModeChoice::Target;
        v.retune_ms = 0.0;
        let mut p = Skatotune::default();
        p.activate_with(&v, 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        for k in [62u8, 66, 69] {
            host.events.push(note_on(0, k)); // D F# A
        }
        run_signal(&mut p, &v, &mut host, &[0.0; 512]);
        for (i, k) in [69u8, 62, 66].into_iter().enumerate() {
            host.events.push(note_off(100 * i as u32, k));
        }
        run_signal(&mut p, &v, &mut host, &[0.0; 512]);
        let d_chord = (1u16 << 2) | (1 << 6) | (1 << 9);
        assert_eq!(keysource::unpack_midi(p.params.midi_scale.load(Ordering::Relaxed)), Some((d_chord, 2)));
        let mut t0 = 0;
        let y = render(&mut p, &v, &mut host, F4_SHARPISH, 0.6, 512, &mut t0);
        let c = cents(tail_pitch(&y), FS4);
        assert!(c.abs() < 10.0, "tuned to the latched chord: {c:+.1} c from F#");
        assert_eq!(
            keysource::unpack_in_effect(p.shared.key_in_effect.load(Ordering::Relaxed)),
            keysource::InEffect::Midi { mask: d_chord, bass: 2 }
        );
        // A reloaded project is in key before any MIDI arrives.
        let saved = p.params.serialize_fields();
        let mut q = Skatotune::default();
        q.params.deserialize_fields(&saved);
        q.activate_with(&v, 48_000.0, 512, 2);
        let mut t0 = 0;
        let y = render(&mut q, &v, &mut FakeHost::default(), F4_SHARPISH, 0.6, 512, &mut t0);
        assert!(cents(tail_pitch(&y), FS4).abs() < 10.0, "after a reload");
    }

    /// #6 Detect: a confirmed key saved with the project is restored into the
    /// engine at activation and tunes at once (marked as saved until the
    /// singing confirms it).
    #[test]
    fn a_detected_key_saved_with_the_project_is_in_force_after_a_reload() {
        let mut v = ParamValues::default();
        v.key_source = params::KeySourceChoice::Detect;
        v.retune_ms = 0.0;
        let d_major = skato_engine::KeyName::new(2, skato_core::ScaleKind::Major);
        let first = Skatotune::default();
        first.params.detected_key.store(keysource::pack_key(d_major), Ordering::Relaxed);
        let saved = first.params.serialize_fields();
        let mut p = Skatotune::default();
        p.params.deserialize_fields(&saved);
        p.activate_with(&v, 48_000.0, 512, 2);
        let mut t0 = 0;
        let y = render(&mut p, &v, &mut FakeHost::default(), F4_SHARPISH, 0.6, 512, &mut t0);
        assert!(cents(tail_pitch(&y), FS4).abs() < 10.0, "D major: F is corrected to F#");
        let w = p.shared.key_confirmed.load(Ordering::Relaxed);
        assert_eq!(keysource::unpack_key(w & !keysource::KEY_SAVED), Some(d_major));
        assert_ne!(w & keysource::KEY_SAVED, 0, "shown as saved");
        assert_eq!(
            keysource::unpack_in_effect(p.shared.key_in_effect.load(Ordering::Relaxed)),
            keysource::InEffect::Detect(d_major)
        );
    }

    /// Auto-Key keeps what it heard through an engine reset (CLAP sleep/wake,
    /// transport) and through a rebuild at the same sample rate.
    #[test]
    fn auto_key_survives_engine_resets_and_rebuilds() {
        let v = ParamValues::default();
        let mut p = Skatotune::default();
        p.activate_with(&v, 48_000.0, 512, 2);
        let mut t0 = 0;
        for hz in [220.0, 246.94, 261.63, 293.66] {
            render(&mut p, &v, &mut FakeHost::default(), hz, 0.5, 512, &mut t0);
        }
        let heard = p.engine.as_ref().map(|e| e.key_reading().evidence_s).unwrap_or(0.0);
        assert!(heard > 0.5, "{heard}");
        if let Some(e) = p.engine.as_mut() {
            e.reset();
        }
        assert_eq!(p.engine.as_ref().map(|e| e.key_reading().evidence_s), Some(heard));
        // A structural rebuild in activate() at the same rate continues it.
        let mut studio = v;
        studio.latency_mode = params::LatencyModeChoice::Studio;
        p.activate_with(&studio, 48_000.0, 512, 2);
        let after = p.engine.as_ref().map(|e| e.key_reading().evidence_s);
        assert_eq!(after, Some(heard), "carried into the new engine");
    }

    /// HI-4: the learned scale is part of the saved plugin state.
    #[test]
    fn learned_scale_is_saved_with_the_project() {
        let mut p = Skatotune::default();
        let mut v = ParamValues::default();
        p.activate_with(&v, 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        run(&mut p, &v, &mut host, 64);
        v.midi_mode = params::MidiModeChoice::Learn;
        host.events.push(note_on(0, 62));
        host.events.push(note_on(1, 69));
        run(&mut p, &v, &mut host, 64);

        let saved = p.params.serialize_fields();
        assert!(saved.contains_key("learned-scale"), "{saved:?}");
        let restored = Skatotune::default();
        restored.params.deserialize_fields(&saved);
        assert_eq!(
            restored.params.learned_states(),
            learn::unpack(learn::pack(&[2, 9]))
        );
    }

    #[test]
    fn dragging_a_structural_value_restarts_once() {
        for api in [PluginApi::Vst3, PluginApi::Clap] {
            let mut p = Skatotune::default();
            let mut v = ParamValues::default();
            v.latency_mode = LatencyModeChoice::Studio;
            p.activate_with(&v, 48_000.0, 512, 2);
            let mut host = FakeHost {
                api: Some(api),
                ..FakeHost::default()
            };
            for step in 0..20 {
                v.studio_ms = 39.0 - step as f32;
                // A 60 Hz GUI frame of 64-sample blocks.
                for _ in 0..13 {
                    run(&mut p, &v, &mut host, 64);
                }
            }
            settle(&mut p, &v, &mut host);
            let cfg = v.structural().engine_config(48_000.0, 512, 2);
            match api {
                PluginApi::Vst3 => {
                    assert_eq!(host.tasks, vec![EngineTask::Build(cfg)], "one build");
                    until_swapped(&mut p, &v, &mut host);
                    assert_eq!(host.latency_calls, vec![skato_engine::latency_samples(&cfg)]);
                }
                _ => {
                    assert!(host.latency_calls.is_empty());
                    assert_eq!(host.restarts, 1, "{api:?}");
                }
            }
        }
    }

    /// HI-7: under CLAP the plugin lets the host sleep once the input has
    /// been silent long enough for the delayed output to be silent too;
    /// VST3 keeps reporting its tail.
    #[test]
    fn clap_can_sleep_after_silence() {
        for api in [PluginApi::Clap, PluginApi::Vst3] {
            let mut p = Skatotune::default();
            let d = p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
            let mut host = FakeHost {
                api: Some(api),
                ..FakeHost::default()
            };
            let mut status = |p: &mut Skatotune, loud: bool| {
                let mut l = if loud {
                    voice(256, 220.0, 48_000.0, 0)
                } else {
                    vec![0.0; 256]
                };
                let mut r = l.clone();
                let mut io: [&mut [f32]; 2] = [&mut l, &mut r];
                p.process_block(&ParamValues::default(), &mut io, &mut host)
            };
            assert_eq!(status(&mut p, true), ProcessStatus::Tail(d));
            // Just after the input stops, audio is still in the delay line.
            assert_eq!(status(&mut p, false), ProcessStatus::Tail(d));
            let tail = p.active.map_or(0, |a| a.quiet_tail);
            assert!(tail >= u64::from(d));
            let mut last = ProcessStatus::Tail(d);
            for _ in 0..tail / 256 + 2 {
                last = status(&mut p, false);
            }
            let expect = if api == PluginApi::Clap {
                ProcessStatus::Normal
            } else {
                ProcessStatus::Tail(d)
            };
            assert_eq!(last, expect, "{api:?}");
            // Sound again: back to keeping the plugin awake.
            assert_eq!(status(&mut p, true), ProcessStatus::Tail(d));
        }
    }

    /// Run the background tasks the plugin posted (nice-plug's background
    /// thread in a real host).
    fn run_background(p: &mut Skatotune, host: &mut FakeHost) {
        for task in std::mem::take(&mut host.tasks) {
            p.run_task(task);
        }
    }

    /// 64-sample blocks with the background thread keeping up, until the
    /// pending structural change has been swapped in (bounded).
    fn until_swapped(p: &mut Skatotune, v: &ParamValues, host: &mut FakeHost) {
        for _ in 0..200 {
            run(p, v, host, 64);
            run_background(p, host);
            if !p.restart_pending() {
                return;
            }
        }
        panic!("the new engine was never swapped in");
    }

    /// A 76.4 Hz sine (below every range but Low Male/Wide, so the engine
    /// passes it through as an exact delay), continuing from sample `t0`.
    fn sine_block(t0: &mut usize, n: usize) -> Vec<f32> {
        let x = (*t0..*t0 + n).map(|i| (i as f32 * 0.01).sin()).collect();
        *t0 += n;
        x
    }

    /// HI-1: a VST3 host never reads a latency that differs from the running
    /// engine's, whatever it does with the restart request (here: nothing,
    /// it never re-activates). The new engine is built off the audio thread
    /// and swapped in first; only then is the new latency reported. Checked
    /// on the audio itself afterwards.
    #[test]
    fn vst3_reported_latency_always_matches_the_running_engine() {
        let mut p = Skatotune::default();
        let live = p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        // Classic at amount 0 passes the voice through at the reported delay
        // (to within a fraction of a sample), so the audio shows which latency
        // is running.
        let mut v = ParamValues {
            amount_pct: 0.0,
            engine: params::EngineChoice::Classic,
            ..ParamValues::default()
        };
        v.latency_mode = LatencyModeChoice::Studio;
        let studio = skato_engine::latency_samples(&v.structural().engine_config(48_000.0, 512, 2));
        let mut reported = live;
        let mut t0 = 0;
        for block in 0..750 {
            let mut l = sine_block(&mut t0, 64);
            let mut r = l.clone();
            let mut io: [&mut [f32]; 2] = [&mut l, &mut r];
            p.process_block(&v, &mut io, &mut host);
            run_background(&mut p, &mut host);
            if let Some(&l) = host.latency_calls.last() {
                reported = l;
            }
            assert_eq!(
                Some(reported),
                p.latency(),
                "block {block}: host told {reported}, the engine runs {:?}",
                p.latency()
            );
        }
        assert_eq!(p.latency(), Some(studio), "the switch happened");
        assert_eq!(host.latency_calls, vec![studio], "reported once");
        assert!(!p.restart_pending());

        // The audio is the input delayed by the new latency.
        let d = studio as usize;
        let input: Vec<f32> = (0..t0 + 2048).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut l = sine_block(&mut t0, 2048);
        let mut r = l.clone();
        let mut io: [&mut [f32]; 2] = [&mut l, &mut r];
        p.process_block(&v, &mut io, &mut host);
        let start = t0 - 2048;
        for (k, y) in l.iter().enumerate() {
            let x = input[start + k - d];
            // 3e-3 is 0.3 samples of this sine's steepest slope.
            assert!((y - x).abs() < 3e-3, "sample {k}: {y} vs {x}");
        }

        // A host that re-activates later finds the engine already built.
        let generation = p.shared.engine_generation.load(Ordering::Relaxed);
        assert_eq!(p.activate_with(&v, 48_000.0, 512, 2), studio);
        assert_eq!(
            p.shared.engine_generation.load(Ordering::Relaxed),
            generation
        );
    }

    /// HI-3: a voice-range change that keeps the latency (any range in
    /// Studio, Live+ or Live-Min; Low Male and Wide in Live) is applied by the
    /// running engine itself (`Engine::set_voice_range`), at once: no host
    /// restart, no latency report, no engine build or swap, in VST3, CLAP and
    /// the standalone. A change that moves the latency still takes the swap
    /// (VST3) or restart (CLAP) path.
    #[test]
    fn same_latency_voice_range_change_is_applied_in_place() {
        for api in [PluginApi::Vst3, PluginApi::Clap, PluginApi::Standalone] {
            for (mode, from, to) in [
                (
                    LatencyModeChoice::Studio,
                    VoiceRangeChoice::AltoTenor,
                    VoiceRangeChoice::LowMale,
                ),
                (
                    LatencyModeChoice::Live,
                    VoiceRangeChoice::LowMale,
                    VoiceRangeChoice::Wide,
                ),
            ] {
                let mut p = Skatotune::default();
                let mut v = ParamValues::default();
                v.latency_mode = mode;
                v.voice_range = from;
                let d = p.activate_with(&v, 48_000.0, 512, 2);
                let mut host = FakeHost {
                    api: Some(api),
                    ..FakeHost::default()
                };
                run(&mut p, &v, &mut host, 64);
                let before = p.engine.as_deref().map(std::ptr::from_ref);
                v.voice_range = to;
                let cfg = v.structural().engine_config(48_000.0, 512, 2);
                assert_eq!(skato_engine::latency_samples(&cfg), d, "same latency");
                // Requested from the very next block (no settle delay); the
                // engine switches after its 5 ms fade-out, at a hop boundary.
                run(&mut p, &v, &mut host, 64);
                assert_eq!(
                    p.active.map(|a| a.structural.voice_range),
                    Some(cfg.voice_range)
                );
                for _ in 0..5 {
                    run(&mut p, &v, &mut host, 64);
                }
                assert_eq!(
                    p.engine.as_ref().map(|e| e.config().voice_range),
                    Some(cfg.voice_range),
                    "{api:?} {mode:?}: the running engine took the new range"
                );
                for _ in 0..300 {
                    run(&mut p, &v, &mut host, 64);
                }
                assert_eq!(host.restarts, 0, "{api:?} {mode:?}");
                assert!(host.latency_calls.is_empty(), "{api:?} {mode:?}");
                assert!(host.tasks.is_empty(), "{api:?} {mode:?}: nothing built");
                assert_eq!(
                    p.engine.as_deref().map(std::ptr::from_ref),
                    before,
                    "{api:?} {mode:?}: the same engine instance"
                );
                assert!(!p.restart_pending());
                assert!(!p.shared.restart_pending.load(Ordering::Acquire));
            }
        }
        // Live Alto/Tenor → Soprano moves the latency: VST3 builds a new
        // engine off-thread, CLAP asks for a restart.
        for api in [PluginApi::Vst3, PluginApi::Clap] {
            let mut p = Skatotune::default();
            let mut v = ParamValues::default();
            v.latency_mode = LatencyModeChoice::Live;
            let d = p.activate_with(&v, 48_000.0, 512, 2);
            let mut host = FakeHost {
                api: Some(api),
                ..FakeHost::default()
            };
            v.voice_range = VoiceRangeChoice::Soprano;
            let cfg = v.structural().engine_config(48_000.0, 512, 2);
            assert_ne!(skato_engine::latency_samples(&cfg), d);
            for _ in 0..300 {
                run(&mut p, &v, &mut host, 64);
            }
            match api {
                PluginApi::Clap => assert_eq!((host.restarts, host.tasks.len()), (1, 0)),
                _ => assert_eq!(host.tasks, vec![EngineTask::Build(cfg)]),
            }
        }
    }

    /// RT-3 support: the plugin's copy of the engine's sample clock (used to
    /// group bends by hop) stays exact across odd block sizes and resets.
    #[test]
    fn engine_clock_mirror_tracks_the_hops() {
        let mut p = Skatotune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 4096, 2);
        let mut host = FakeHost::default();
        let hop = skato_core::hop_for_rate(48_000.0) as u64;
        for (i, n) in [1usize, 7, 63, 64, 65, 480, 1000, 3]
            .iter()
            .cycle()
            .take(40)
            .enumerate()
        {
            run(&mut p, &ParamValues::default(), &mut host, *n);
            let hops = p.engine.as_ref().map_or(0, |e| e.stats().hops);
            assert_eq!(hops, p.engine_pos / hop, "block {i}");
            if i == 20 {
                <Skatotune as Plugin>::reset(&mut p);
                assert_eq!(p.engine_pos, 0);
            }
        }
    }

    #[test]
    fn standalone_swaps_in_a_handed_engine() {
        let mut p = Skatotune::default();
        let live = p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut gui = p.take_engine_swap();
        let mut host = FakeHost {
            api: Some(PluginApi::Standalone),
            ..FakeHost::default()
        };
        let mut v = ParamValues::default();
        v.latency_mode = LatencyModeChoice::Studio;
        settle(&mut p, &v, &mut host);
        assert!(p.shared.restart_pending.load(Ordering::Acquire));

        // A stale engine (wrong config) is rejected and sent back at once.
        let stale = ParamValues::default()
            .structural()
            .engine_config(48_000.0, 512, 2);
        assert!(gui.outgoing.push((1, Box::new(Engine::new(stale)))).is_ok());
        run(&mut p, &v, &mut host, 64);
        assert!(gui.retired.pop().is_ok(), "stale engine returned");
        assert_eq!(p.latency(), Some(live));

        let cfg = v.structural().engine_config(48_000.0, 512, 2);
        let studio = skato_engine::latency_samples(&cfg);
        assert!(gui.outgoing.push((2, Box::new(Engine::new(cfg)))).is_ok());
        // 3 ms fade-out of the running engine first, then the swap.
        run(&mut p, &v, &mut host, 64);
        assert_eq!(p.latency(), Some(live), "still fading out");
        for _ in 0..3 {
            run(&mut p, &v, &mut host, 64);
        }
        assert_eq!(p.latency(), Some(studio));
        assert!(!p.restart_pending());
        assert!(!p.shared.restart_pending.load(Ordering::Acquire));
        assert_eq!(p.shared.latency_samples.load(Ordering::Relaxed), studio);
        assert_eq!(p.shared.adopted_engine.load(Ordering::Acquire), 2);
        let old = gui.retired.pop().expect("old engine handed back");
        assert_eq!(old.latency_samples(), live);
        assert!(host.latency_calls.is_empty() && host.restarts == 0);
        // The new engine really runs: once its delay line has filled and the
        // 3 ms fade-in is over, the output is the input delayed by D. (A
        // 38 Hz tone, below every voice range, so nothing is pitch-shifted;
        // the 76 Hz tone of `run` is inside Alto/Tenor's search margin and
        // gets corrected once the detector has voiced it.)
        let n = 4096;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.005).sin()).collect();
        let out = run_signal(&mut p, &v, &mut host, &input);
        let d = studio as usize;
        // The swap came at most three 64-sample blocks ago.
        assert!(
            out[0][..d - 192].iter().all(|&y| y == 0.0),
            "muted while D fills"
        );
        let settled = d + 200;
        assert!((settled..n).all(|i| (out[0][i] - input[i - d]).abs() < 1e-6));
    }

    /// HI-1: if a VST3 host has not re-activated the plugin after a latency
    /// change (it may have dropped the request), `kLatencyChanged` is sent
    /// again at 1, 3 and 7 s from the audio thread, editor open or not, and
    /// a re-activation stops it.
    #[test]
    fn vst3_latency_change_is_re_requested_until_the_host_reactivates() {
        let mut p = Skatotune::default();
        p.activate_with(&ParamValues::default(), 48_000.0, 512, 2);
        let mut host = FakeHost::default();
        let mut v = ParamValues::default();
        v.latency_mode = LatencyModeChoice::Studio;
        settle(&mut p, &v, &mut host);
        until_swapped(&mut p, &v, &mut host);
        assert_eq!(host.latency_calls.len(), 1);
        let mut at = Vec::new();
        for block in 0..(10 * 48_000 / 480) {
            run(&mut p, &v, &mut host, 480);
            if host.restarts > at.len() {
                at.push(block as f32 * 0.01);
            }
        }
        assert_eq!(at.len(), 3, "{at:?}");
        assert!((at[0] - 1.0).abs() < 0.02 && (at[1] - 3.0).abs() < 0.02);
        assert!((at[2] - 7.0).abs() < 0.02, "{at:?}");

        // A host that does re-activate stops the retries at once.
        v.latency_mode = LatencyModeChoice::Live;
        settle(&mut p, &v, &mut host);
        until_swapped(&mut p, &v, &mut host);
        let restarts = host.restarts;
        p.activate_with(&v, 48_000.0, 512, 2);
        for _ in 0..(10 * 48_000 / 480) {
            run(&mut p, &v, &mut host, 480);
        }
        assert_eq!(host.restarts, restarts);
    }
}
