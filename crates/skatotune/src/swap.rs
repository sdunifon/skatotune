//! Structural changes without rebuilding on the audio thread (HI-1, HI-3,
//! RT-4).
//!
//! A structural parameter (latency mode, voice range, Live+/Studio latency)
//! needs a new [`Engine`]. Building one allocates, so it never happens in
//! `process()`. Instead:
//!
//! 1. `process()` asks for the engine with [`EngineTask::Build`], which
//!    nice-plug runs on its background thread (`execute_background`, one
//!    lock-free queue push per change).
//! 2. [`EngineBuilder::run`] builds it there and hands it over through a
//!    single-slot `rtrb` ring.
//! 3. `process()` fades the old engine out ([`SwapFade`], 3 ms), swaps the
//!    new one in at a block boundary and sends the old one back through a
//!    second ring, then asks for [`EngineTask::Collect`], which frees it on
//!    the background thread and passes the new engine's telemetry ring to
//!    the editor. The new engine stays muted until its delay line has
//!    filled (D samples) and then fades in.
//! 4. Only then does a VST3 host hear about a new latency
//!    (`set_latency_samples`, which nice-plug turns into
//!    `restartComponent(kLatencyChanged)`). So the latency the host reads is
//!    always the latency of the engine that is running, whether it
//!    re-activates the plugin promptly, late, or only re-reads the latency.
//!    When it re-activates, `activate()` finds the engine already matching and
//!    keeps it.
//!
//! CLAP only allows a latency change inside `activate()`, so a CLAP change
//! that moves the latency still goes through a host restart; a CLAP change
//! that keeps it is swapped like above. The standalone has no host restart
//! at all and always swaps. A voice-range change that keeps the latency (any
//! range in Live-Min, Live+ and Studio; Low Male and Wide in Live) skips all
//! of this: the running engine switches its own detector and shifter
//! (`Engine::set_voice_range`, prebuilt in `Engine::new`).
//!
//! Nothing here allocates, frees, locks or blocks on the audio thread: the
//! audio half only peeks, pops and pushes `rtrb` rings; the builder's mutex
//! is only ever taken by the background thread.

use std::sync::Mutex;
use std::sync::atomic::Ordering;

use skato_core::{EngineConfig, PitchFrame};
use skato_engine::Engine;

use crate::shared::SharedState;

/// Work the audio thread hands to nice-plug's background thread (the
/// plugin's `BackgroundTask`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EngineTask {
    /// Build an engine for this configuration and hand it to the audio
    /// thread (newest request wins).
    Build(EngineConfig),
    /// Free engines the audio thread handed back, and give the editor the
    /// telemetry ring of an engine it swapped in.
    Collect,
}

/// A built engine on its way to the audio thread, tagged with an id so the
/// builder knows which telemetry ring belongs to the engine that got swapped
/// in.
pub type HandedEngine = (u32, Box<Engine>);

/// Audio-thread half of the engine handoff.
pub struct EngineSwapAudio {
    /// New engines from the builder (capacity 1).
    pub incoming: rtrb::Consumer<HandedEngine>,
    /// Replaced or rejected engines going back to be freed off the audio thread.
    pub retired: rtrb::Producer<Box<Engine>>,
}

/// Builder half of the engine handoff.
pub struct EngineSwapBuilder {
    pub outgoing: rtrb::Producer<HandedEngine>,
    pub retired: rtrb::Consumer<Box<Engine>>,
}

/// Create the two lock-free rings of the engine handoff. The retire ring has
/// room for a replaced engine plus rejected ones, so the audio thread never
/// has to drop an engine itself.
pub fn engine_swap_channel() -> (EngineSwapAudio, EngineSwapBuilder) {
    let (outgoing, incoming) = rtrb::RingBuffer::new(1);
    let (retired_tx, retired_rx) = rtrb::RingBuffer::new(4);
    (
        EngineSwapAudio {
            incoming,
            retired: retired_tx,
        },
        EngineSwapBuilder {
            outgoing,
            retired: retired_rx,
        },
    )
}

/// Builds engines for the audio thread and frees the ones it returns. Shared
/// by the plugin and its background-task executor; only ever used off the
/// audio thread.
pub struct EngineBuilder {
    inner: Mutex<BuilderInner>,
}

struct BuilderInner {
    rings: EngineSwapBuilder,
    /// Configuration asked for but not handed over yet (the slot was busy).
    want: Option<EngineConfig>,
    next_id: u32,
    /// Telemetry rings of handed-over engines that may still be swapped in.
    handed: Vec<(u32, rtrb::Consumer<PitchFrame>)>,
}

impl EngineBuilder {
    pub fn new(rings: EngineSwapBuilder) -> Self {
        Self {
            inner: Mutex::new(BuilderInner {
                rings,
                want: None,
                next_id: 1,
                handed: Vec::new(),
            }),
        }
    }

    /// Run one task (background thread; allocates and frees).
    pub fn run(&self, task: EngineTask, shared: &SharedState) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let inner = &mut *guard;
        if let EngineTask::Build(cfg) = task {
            inner.want = Some(cfg);
        }
        // Engines the audio thread replaced or rejected are freed here.
        while inner.rings.retired.pop().is_ok() {}

        // A free slot means the audio thread has taken the last engine we
        // handed over, and it publishes the id of an engine it swaps in
        // before taking it: the matching telemetry ring goes to the editor,
        // every other one belongs to an engine that was rejected.
        if inner.rings.outgoing.slots() == 0 {
            return; // the last engine is still waiting to be swapped in
        }
        let adopted = shared.adopted_engine.load(Ordering::Acquire);
        for (id, rx) in inner.handed.drain(..) {
            if id == adopted {
                shared.install_telemetry(rx);
            }
        }
        let Some(cfg) = inner.want.take() else {
            return;
        };
        let id = inner.next_id;
        inner.next_id = inner.next_id.wrapping_add(1).max(1);
        let mut engine = Box::new(Engine::new(cfg));
        let telemetry = engine.take_telemetry();
        if inner.rings.outgoing.push((id, engine)).is_ok() {
            if let Some(rx) = telemetry {
                inner.handed.push((id, rx));
            }
        } else {
            inner.want = Some(cfg); // unreachable: a free slot was checked
        }
    }
}

/// Length of the fades around an engine swap, seconds.
pub const SWAP_FADE_S: f32 = 0.003;

/// Output gain around an engine swap: fade the old engine out, swap at a
/// block boundary once silent, keep the new engine muted while its delay
/// line fills (it outputs nothing for its first D samples, then the voice
/// would start abruptly) and fade it in. A per-sample multiply only while
/// active; idle costs one comparison per block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwapFade {
    gain: f32,
    target: f32,
    step: f32,
    /// Samples still muted before the fade-in starts.
    hold: u64,
}

impl SwapFade {
    pub fn new(sample_rate: f32) -> Self {
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        Self {
            gain: 1.0,
            target: 1.0,
            step: 1.0 / (SWAP_FADE_S * sr).max(1.0),
            hold: 0,
        }
    }

    /// Nothing to apply.
    pub fn idle(&self) -> bool {
        self.gain >= 1.0 && self.target >= 1.0 && self.hold == 0
    }

    /// The old engine has been faded out completely: swap now.
    pub fn silent(&self) -> bool {
        self.gain <= 0.0 && self.target <= 0.0
    }

    /// Start (or continue) fading the running engine out.
    pub fn fade_out(&mut self) {
        self.target = 0.0;
    }

    /// Fade back in (the swap was called off).
    pub fn fade_in(&mut self) {
        self.target = 1.0;
    }

    /// A new engine with latency `latency` samples was just swapped in.
    pub fn swapped(&mut self, latency: u32) {
        self.gain = 0.0;
        self.target = 1.0;
        self.hold = u64::from(latency);
    }

    /// Apply the envelope to `samples` samples of every channel.
    pub fn apply(&mut self, io: &mut [&mut [f32]], samples: usize) {
        if self.idle() {
            return;
        }
        for i in 0..samples {
            let g = if self.hold > 0 {
                self.hold -= 1;
                0.0
            } else {
                self.gain = if self.target > self.gain {
                    (self.gain + self.step).min(self.target)
                } else {
                    (self.gain - self.step).max(self.target)
                };
                self.gain
            };
            for ch in io.iter_mut() {
                if let Some(x) = ch.get_mut(i) {
                    *x *= g;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(mode: skato_core::LatencyMode) -> EngineConfig {
        EngineConfig {
            max_block: 256,
            latency_mode: mode,
            ..EngineConfig::default()
        }
    }

    #[test]
    fn builder_hands_over_and_frees_off_the_audio_thread() {
        let shared = SharedState::default();
        let (mut audio, gui) = engine_swap_channel();
        let builder = EngineBuilder::new(gui);
        let studio = cfg(skato_core::LatencyMode::Studio);
        builder.run(EngineTask::Build(studio), &shared);

        // The audio thread swaps it in: id published before the pop, the
        // replaced engine goes back, Collect follows.
        let id = audio.incoming.peek().map(|(id, _)| *id).expect("built");
        shared.adopted_engine.store(id, Ordering::Release);
        let (_, engine) = audio.incoming.pop().expect("built");
        assert_eq!(*engine.config(), studio);
        let generation = shared.engine_generation.load(Ordering::Relaxed);
        assert!(
            audio
                .retired
                .push(Box::new(Engine::new(cfg(skato_core::LatencyMode::Live))))
                .is_ok()
        );
        builder.run(EngineTask::Collect, &shared);
        assert_eq!(audio.retired.slots(), 4, "replaced engine freed");
        assert_eq!(
            shared.engine_generation.load(Ordering::Relaxed),
            generation + 1,
            "the swapped-in engine's telemetry reaches the editor"
        );
        assert!(shared.take_telemetry().is_some());

        // Newest request wins while the slot is busy; a rejected engine's
        // telemetry is never installed.
        let live = cfg(skato_core::LatencyMode::Live);
        let live_plus = cfg(skato_core::LatencyMode::LivePlus);
        builder.run(EngineTask::Build(live), &shared);
        builder.run(EngineTask::Build(live_plus), &shared);
        let (_, stale) = audio.incoming.pop().expect("live waiting");
        assert_eq!(*stale.config(), live);
        assert!(audio.retired.push(stale).is_ok());
        builder.run(EngineTask::Collect, &shared);
        assert_eq!(audio.retired.slots(), 4, "rejected engine freed");
        let (_, newest) = audio.incoming.pop().expect("live+ follows");
        assert_eq!(*newest.config(), live_plus);
        assert_eq!(
            shared.engine_generation.load(Ordering::Relaxed),
            generation + 1
        );
        drop((engine, newest));
    }

    #[test]
    fn fade_mutes_across_the_swap_and_comes_back() {
        let mut f = SwapFade::new(48_000.0);
        assert!(f.idle());
        let block = |f: &mut SwapFade, n: usize| {
            let mut x = vec![1.0f32; n];
            f.apply(&mut [&mut x], n);
            x
        };
        assert!(block(&mut f, 8).iter().all(|&g| g == 1.0));
        f.fade_out();
        let out = block(&mut f, 200);
        assert!(out.windows(2).all(|w| w[1] <= w[0]), "monotonic fade");
        assert!(out[100] > 0.0 && out[150] == 0.0, "3 ms = 144 samples");
        assert!(f.silent());
        f.swapped(100);
        let out = block(&mut f, 400);
        assert!(out[..100].iter().all(|&g| g == 0.0), "muted while D fills");
        assert!(out[101] > 0.0 && out[101] < 0.02);
        assert_eq!(out[399], 1.0);
        assert!(f.idle());
        // A called-off swap fades straight back.
        f.fade_out();
        block(&mut f, 50);
        f.fade_in();
        block(&mut f, 200);
        assert!(f.idle());
    }
}
