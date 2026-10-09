//! Labs experiments must not make the editor allocate per pixel column: the
//! correction ribbon, the heat line and the legend key are vertex-coloured
//! meshes whose buffers are reused frame to frame, so however often the
//! correction (and with it the colour) changes along the trace, they cost
//! no allocation. A counting global allocator watches the test's own thread
//! while the editor draws, with the experiments on and off over the same
//! scrolling telemetry: notes with a steady correction each, and the worst
//! case for a colour-per-run design, hard tune flattening a wide vibrato
//! (the correction swings 0-50 c at 5.5 Hz). Separate binary because it
//! replaces the global allocator; skipped with `assert_process_allocs`,
//! which installs its own.

#![cfg(not(feature = "assert_process_allocs"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use egui::{Pos2, RawInput, Rect, vec2};
use nice_plug::context::gui::GuiContext;
use nice_plug::prelude::*;
use vocal_tune::editor::{HEIGHT, VocalTuneEditor, WIDTH};
use vocal_tune::labs::LabsSettings;
use vocal_tune::params::VocalTuneParams;
use vocal_tune::shared::SharedState;
use vt_core::{PitchFrame, frame_flags};

struct Counting;

// Per thread: the tests run in parallel, each counts only its own.
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn note_alloc() {
    if COUNTING.with(Cell::get) {
        ALLOCS.with(|n| n.set(n.get() + 1));
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_alloc();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_alloc();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations made by `f` on this thread.
fn count<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOCS.with(Cell::get);
    COUNTING.with(|c| c.set(true));
    let r = f();
    COUNTING.with(|c| c.set(false));
    (r, ALLOCS.with(Cell::get) - before)
}

/// A GUI context that ignores parameter gestures.
struct Host;

impl GuiContextInner for Host {
    fn plugin_api(&self) -> PluginApi {
        PluginApi::Clap
    }
    unsafe fn raw_begin_set_parameter(&self, _param: ParamPtr) {}
    unsafe fn raw_set_parameter_normalized(&self, _param: ParamPtr, _normalized: f32) {}
    unsafe fn raw_end_set_parameter(&self, _param: ParamPtr) {}
    fn get_state(&self) -> PluginState {
        PluginState {
            version: String::new(),
            params: Default::default(),
            fields: Default::default(),
        }
    }
    fn set_state(&self, _state: PluginState) {}
    fn request_restart(&self) {}
}

const HOP: u64 = 64;

/// Telemetry frame `k`: notes of 1.2 s sung 15-120 c off and corrected
/// fully, a short breath after each.
fn notes(k: u64) -> PitchFrame {
    const NOTES: [(f32, f32); 4] = [
        (6000.0, 40.0),
        (6200.0, -15.0),
        (6400.0, 120.0),
        (6500.0, 25.0),
    ];
    let per_note = 1000;
    let i = (k / per_note) as usize % NOTES.len();
    let (target, detune) = NOTES[i];
    let pos = k * HOP;
    if k % per_note >= 900 {
        return PitchFrame {
            sample_pos: pos,
            ..PitchFrame::default()
        };
    }
    PitchFrame {
        sample_pos: pos,
        detected: target + detune,
        centre: target + detune,
        output: target,
        target,
        confidence: 0.9,
        flags: frame_flags::VOICED,
        ..PitchFrame::default()
    }
}

/// Telemetry frame `k`: hard tune on a sung vibrato of ±50 c at 5.5 Hz.
/// The output sits on the target, so the correction swings between 0 and
/// 50 c twice per cycle: a new heat-line step every few pixel columns.
fn flattened_vibrato(k: u64) -> PitchFrame {
    let per_note = 1500;
    let target = [6200.0, 6400.0, 6700.0][(k / per_note) as usize % 3];
    let pos = k * HOP;
    if k % per_note >= 1400 {
        return PitchFrame {
            sample_pos: pos,
            ..PitchFrame::default()
        };
    }
    let t = (k % per_note) as f32 * HOP as f32 / 48_000.0;
    let sung = target + 50.0 * (std::f32::consts::TAU * 5.5 * t).sin();
    PitchFrame {
        sample_pos: pos,
        detected: sung,
        centre: target,
        output: target,
        target,
        confidence: 0.9,
        flags: frame_flags::VOICED,
        ..PitchFrame::default()
    }
}

/// Median allocations per editor frame for each Labs setting, over the
/// same scrolling `telemetry`.
fn allocations_per_frame(
    telemetry: fn(u64) -> PitchFrame,
    settings: &[LabsSettings],
) -> Vec<(LabsSettings, usize, Vec<usize>)> {
    let params = Arc::new(VocalTuneParams::default());
    let shared = Arc::new(SharedState::default());
    shared.sample_rate.store(48_000.0, Ordering::Relaxed);
    let (mut tx, rx) = rtrb::RingBuffer::new(16_384);
    shared.install_telemetry(rx);
    let mut app = VocalTuneEditor::new(params.clone(), shared);
    let ctx = egui::Context::default();
    app.setup(&ctx, GuiContext::new(Arc::new(Host)));

    // 5 s of history, then 1/60 s more per GUI frame so the graph scrolls.
    let mut k = 0u64;
    let mut feed = |frames: u64| {
        for _ in 0..frames {
            let _ = tx.push(telemetry(k));
            k += 1;
        }
    };
    feed(5 * 750);
    let mut draw = |labs: LabsSettings| {
        params.set_labs(labs);
        feed(750 / 60);
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(WIDTH, HEIGHT))),
            ..Default::default()
        };
        let (mut out, allocs) = count(|| ctx.run_ui(input, |ui| app.draw(ui)));
        out.textures_delta.clear();
        // Like a real integration: tessellate, then drop the frame.
        let _ = ctx.tessellate(out.shapes, out.pixels_per_point);
        allocs
    };
    // A block of frames per setting, measured after egui's caches have
    // settled (its text-layout cache drops what a frame did not use, so
    // flipping the setting every frame would re-lay out the legend key).
    settings
        .iter()
        .map(|&labs| {
            for _ in 0..30 {
                draw(labs);
            }
            let block: Vec<usize> = (0..60).map(|_| draw(labs)).collect();
            (labs, median(&block), block)
        })
        .collect()
}

fn median(v: &[usize]) -> usize {
    let mut s = v.to_vec();
    s.sort_unstable();
    s.get(s.len() / 2).copied().unwrap_or(0)
}

const OFF: LabsSettings = LabsSettings {
    correction_ribbon: false,
    heat_line: false,
};
const RIBBON: LabsSettings = LabsSettings {
    correction_ribbon: true,
    heat_line: false,
};
const HEAT: LabsSettings = LabsSettings {
    correction_ribbon: false,
    heat_line: true,
};
const BOTH: LabsSettings = LabsSettings {
    correction_ribbon: true,
    heat_line: true,
};

/// Each setting against the experiments-off baseline measured in the same
/// run (before and after, the larger): at most `extra` more allocations.
fn assert_bounded(telemetry: fn(u64) -> PitchFrame, name: &str, extra: usize) {
    let runs = allocations_per_frame(telemetry, &[OFF, RIBBON, HEAT, BOTH, OFF]);
    let base = runs
        .iter()
        .filter(|(l, _, _)| *l == OFF)
        .map(|(_, m, _)| *m)
        .max()
        .unwrap_or(0);
    for (labs, m, block) in &runs {
        println!("{name}: {labs:?}: {m} allocations per frame (median)");
        assert!(
            *m <= base + extra,
            "{name} {labs:?}: {m} vs off {base}\n{block:?}"
        );
        // Steady state: nothing accumulates frame to frame.
        let (early, late) = block.split_at(block.len() / 2);
        assert!(
            median(late) <= median(early) + 4,
            "{name} {labs:?}: {block:?}"
        );
    }
}

/// The meshes reuse their buffers (zero); the legend key's labels are the
/// rest. Hundreds of columns in view would cost hundreds.
#[test]
fn labs_experiments_add_a_small_bounded_number_of_allocations() {
    assert_bounded(notes, "steady notes", 24);
}

/// The heat line changes colour every few columns here; with one mesh that
/// is still free (a polyline per colour run cost ~900 here).
#[test]
fn flattened_vibrato_does_not_multiply_allocations() {
    assert_bounded(flattened_vibrato, "flattened vibrato", 24);
}
