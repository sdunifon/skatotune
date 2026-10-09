//! The plugin's per-block path must not allocate (DESIGN §2.1): parameter
//! snapshot, Settings resolution, MIDI conversion, engine call, upmix, gain
//! and the CPU meter. A counting global allocator watches only the test's
//! own thread while `process_block` runs. Separate binary because it
//! replaces the global allocator; skipped with `assert_process_allocs`, whose
//! own global allocator already enforces the same rule (R§9.4).

#![cfg(not(feature = "assert_process_allocs"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use nice_plug::midi::{Channel, Key, VoiceID};
use nice_plug::prelude::*;
use skatotune::params::{
    KeyChoice, LatencyModeChoice, MidiModeChoice, ParamValues, ScaleChoice, VoiceRangeChoice,
};
use skatotune::swap::EngineTask;
use skatotune::{BlockHost, Skatotune};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNTING.with(Cell::get) {
            FREES.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations made by `f` on this thread.
fn count<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let (r, allocs, _) = count_both(f);
    (r, allocs)
}

/// Allocations and frees made by `f` on this thread.
fn count_both<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    let before = (
        ALLOCS.load(Ordering::Relaxed),
        FREES.load(Ordering::Relaxed),
    );
    COUNTING.with(|c| c.set(true));
    let r = f();
    COUNTING.with(|c| c.set(false));
    (
        r,
        ALLOCS.load(Ordering::Relaxed) - before.0,
        FREES.load(Ordering::Relaxed) - before.1,
    )
}

/// A host with a fixed event list (no allocation while handing events out).
struct Host {
    events: [NoteEvent<()>; 4],
    next: usize,
    latency_calls: usize,
    /// Background tasks posted (the newest one is kept, nothing allocates).
    tasks: usize,
    last_task: Option<EngineTask>,
    api: PluginApi,
}

impl BlockHost for Host {
    fn next_event(&mut self) -> Option<NoteEvent<()>> {
        let ev = self.events.get(self.next).copied();
        self.next += 1;
        ev
    }
    fn set_latency_samples(&mut self, _samples: u32) {
        self.latency_calls += 1;
    }
    fn request_restart(&mut self) {}
    fn execute_background(&mut self, task: EngineTask) {
        self.tasks += 1;
        self.last_task = Some(task);
    }
    fn api(&self) -> PluginApi {
        self.api
    }
}

fn host() -> Host {
    let note = |timing, key, velocity| NoteEvent::NoteOn {
        timing,
        voice_id: VoiceID::Wildcard,
        channel: Channel::Number(0),
        key: Key::Number(key),
        velocity,
    };
    Host {
        events: [
            note(0, 60, 0.9),
            NoteEvent::MidiPitchBend {
                timing: 10,
                channel: 0,
                value: 0.7,
            },
            NoteEvent::MidiCC {
                timing: 20,
                channel: 0,
                cc: 64,
                value: 1.0,
            },
            note(30, 64, 0.0),
        ],
        next: 0,
        latency_calls: 0,
        tasks: 0,
        last_task: None,
        api: PluginApi::Vst3,
    }
}

#[test]
fn process_block_does_not_allocate() {
    let mut plugin = Skatotune::default();
    let mut values = ParamValues {
        key: KeyChoice::E,
        scale: ScaleChoice::Major,
        midi_mode: MidiModeChoice::Auto,
        output_gain_db: -3.0,
        ..ParamValues::default()
    };
    plugin.activate_with(&values, 48_000.0, 512, 2);

    let mut left = vec![0.0f32; 512];
    let mut right = vec![0.0f32; 512];
    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
        *l = (i as f32 * 0.03).sin() * 0.5;
        *r = *l;
    }

    // Warm up once, then measure many blocks, including a structural change
    // (an engine build posted to the background thread) and odd block sizes.
    {
        let mut io: [&mut [f32]; 2] = [&mut left, &mut right];
        plugin.process_block(&values, &mut io, &mut host());
    }
    let mut latency_calls = 0;
    let mut tasks = Vec::with_capacity(8); // the test's own pushes must not allocate
    let mut block = |plugin: &mut Skatotune, values: &ParamValues, n: usize| {
        let mut h = host();
        let (l, r) = (&mut left[..n], &mut right[..n]);
        let mut io: [&mut [f32]; 2] = [l, r];
        plugin.process_block(values, &mut io, &mut h);
        (h.latency_calls, h.tasks, h.last_task)
    };
    let (_, allocs) = count(|| {
        for i in 0..200 {
            if i == 100 {
                values.latency_mode = LatencyModeChoice::Studio;
            }
            let (calls, n, task) = block(&mut plugin, &values, [512, 1, 7, 64, 480][i % 5]);
            latency_calls += calls;
            if n > 0 {
                tasks.push(task);
            }
        }
    });
    assert_eq!(allocs, 0, "process_block allocated {allocs} times");
    let cfg = values.structural().engine_config(48_000.0, 512, 2);
    assert_eq!(
        tasks,
        vec![Some(EngineTask::Build(cfg))],
        "one build posted"
    );
    assert_eq!(latency_calls, 0, "nothing reported before the swap");

    // Background thread's part (not counted): build the engine.
    plugin.run_task(EngineTask::Build(cfg));
    // Fade-out, swap, hand-back and the latency report: no allocation and
    // no free on the audio thread.
    let ((), allocs, frees) = count_both(|| {
        for i in 0..20 {
            latency_calls += block(&mut plugin, &values, [64, 7, 480][i % 3]).0;
        }
    });
    assert_eq!((allocs, frees), (0, 0), "swap allocated or freed");
    assert_eq!(plugin.latency(), Some(skato_engine::latency_samples(&cfg)));
    assert_eq!(latency_calls, 1, "reported once, after the swap");
    plugin.run_task(EngineTask::Collect); // frees the old engine (not counted)
}

/// RT-4: the standalone's structural change swaps in an engine built off the
/// audio thread. The blocks that fade and swap it in neither allocate nor
/// free (the old engine travels back to be freed off the audio thread).
#[test]
fn standalone_engine_swap_does_not_allocate_or_free() {
    let mut plugin = Skatotune::default();
    let mut values = ParamValues::default();
    plugin.activate_with(&values, 48_000.0, 512, 2);
    let mut gui = plugin.take_engine_swap();
    let mut left = vec![0.1f32; 512];
    let mut right = vec![0.1f32; 512];
    let mut block = |plugin: &mut Skatotune, values: &ParamValues| {
        let mut h = host();
        h.api = PluginApi::Standalone;
        let mut io: [&mut [f32]; 2] = [&mut left, &mut right];
        plugin.process_block(values, &mut io, &mut h);
    };
    block(&mut plugin, &values);
    values.latency_mode = LatencyModeChoice::Studio;
    let ((), allocs, frees) = count_both(|| {
        for _ in 0..30 {
            block(&mut plugin, &values);
        }
    });
    assert_eq!((allocs, frees), (0, 0));
    assert!(plugin.restart_pending(), "the change has settled");

    // Builder's part (not counted): build and hand over.
    let cfg = values.structural().engine_config(48_000.0, 512, 2);
    let engine = Box::new(skato_engine::Engine::new(cfg));
    assert!(gui.outgoing.push((5, engine)).is_ok());
    let ((), allocs, frees) = count_both(|| {
        for _ in 0..3 {
            block(&mut plugin, &values);
        }
    });
    assert_eq!(allocs, 0, "swap allocated");
    assert_eq!(frees, 0, "swap freed memory on the audio thread");
    assert_eq!(plugin.latency(), Some(skato_engine::latency_samples(&cfg)));
    assert!(
        gui.retired.pop().is_ok(),
        "old engine handed back to be freed"
    );
}

/// HI-3: a voice-range change that keeps the latency is applied by the
/// running engine (`Engine::set_voice_range`): the blocks that request it,
/// fade, switch the detector and shifter and fade back in neither allocate
/// nor free, and nothing is posted to the background thread.
#[test]
fn in_place_voice_range_change_does_not_allocate_or_free() {
    let mut plugin = Skatotune::default();
    let mut values = ParamValues {
        latency_mode: LatencyModeChoice::Studio,
        ..ParamValues::default()
    };
    plugin.activate_with(&values, 48_000.0, 512, 2);
    let mut left: Vec<f32> = (0..512).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();
    let mut right = left.clone();
    let mut tasks = 0;
    let mut block = |plugin: &mut Skatotune, values: &ParamValues, n: usize| {
        let mut h = host();
        let mut io: [&mut [f32]; 2] = [&mut left[..n], &mut right[..n]];
        plugin.process_block(values, &mut io, &mut h);
        h.tasks
    };
    block(&mut plugin, &values, 512);
    let ((), allocs, frees) = count_both(|| {
        for (i, range) in [
            VoiceRangeChoice::LowMale,
            VoiceRangeChoice::Soprano,
            VoiceRangeChoice::Wide,
            VoiceRangeChoice::AltoTenor,
        ]
        .into_iter()
        .cycle()
        .take(8)
        .enumerate()
        {
            values.voice_range = range;
            for j in 0..12 {
                tasks += block(&mut plugin, &values, [64, 7, 480][(i + j) % 3]);
            }
        }
    });
    assert_eq!((allocs, frees), (0, 0), "range switch allocated or freed");
    assert_eq!(tasks, 0, "nothing built");
    assert!(!plugin.restart_pending());
}

#[test]
fn counter_sees_allocations() {
    // Negative control: the counter must notice a real allocation.
    let (v, allocs) = count(|| vec![1u8; 64]);
    assert_eq!(v.len(), 64);
    assert!(allocs >= 1);
}
