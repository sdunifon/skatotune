//! Headless editor renderer: draws the real editor (with demo telemetry) for a
//! few seconds of wall-clock time, then rasterizes the final egui frame in
//! software and writes a 24-bit BMP. For visual checks and regression
//! screenshots on machines (or sandboxes) without a display or
//! screen-recording permission.
//!
//! ```text
//! cargo run -p skatotune --features demo-telemetry --example render_gui -- out.bmp [seconds] [scale] [flags]
//!   flags: --advanced (open the drawer), --studio (Studio mode), --hard (Hard Tune preset),
//!          --pending (show a pending latency restart), --learn (MIDI Learn with a
//!          learned C-E-G scale), --learned (the learned C-E-G scale after leaving
//!          Learn), --learn-empty (MIDI Learn before any note)
//!          --transpose=ST, --vibdepth=PCT, --retune=MS (set those knobs, e.g. to
//!          check the correction colouring under Transpose or flattened vibrato)
//!   Labs:  --no-ribbon (correction ribbon off; it is on by default), --heat
//!          (heat-tinted output line on), --labs (the LABS menu open),
//!          --theme=<teal|green-red|viridis|inferno|blue-orange> (the
//!          Correction colors theme; teal is the default)
//! sips -s format png out.bmp --out out.png   # macOS, optional
//! ```
//!
//! The rasterizer is deliberately simple: egui already anti-aliases by
//! feathering edges into thin alpha-graded triangles, so per-pixel-centre
//! sampling, bilinear texture lookup and premultiplied "over" blending in
//! gamma space (what egui's own painters do on a non-sRGB framebuffer) give
//! a faithful picture.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use egui::epaint::{ClippedPrimitive, ImageData, Primitive, TextureId, Vertex};
use egui::{Color32, Pos2, RawInput, Rect, pos2, vec2};
use nice_plug::context::gui::GuiContext;
use nice_plug::prelude::*;
use skatotune::editor::{HEIGHT, SkatotuneEditor, WIDTH};
use skatotune::labs::CorrectionTheme;
use skatotune::params::{
    KeyChoice, LatencyModeChoice, MidiModeChoice, NoteOverride, ParamValues, Preset, ScaleChoice,
    SkatotuneParams,
};
use skatotune::shared::SharedState;

/// A GUI context that accepts parameter gestures and ignores them.
struct HeadlessHost;

impl GuiContextInner for HeadlessHost {
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

struct Texture {
    size: [usize; 2],
    pixels: Vec<Color32>,
}

fn apply_textures(textures: &mut HashMap<TextureId, Texture>, delta: &egui::TexturesDelta) {
    for (id, deltas) in &delta.set {
        for d in deltas.iter() {
            apply_delta(textures, *id, d);
        }
    }
    for id in &delta.free {
        textures.remove(id);
    }
}

fn apply_delta(
    textures: &mut HashMap<TextureId, Texture>,
    id: TextureId,
    d: &egui::epaint::ImageDelta,
) {
    {
        let id = &id;
        let ImageData::Color(img) = &d.image;
        match d.pos {
            None => {
                textures.insert(
                    *id,
                    Texture {
                        size: img.size,
                        pixels: img.pixels.clone(),
                    },
                );
            }
            Some([x0, y0]) => {
                if let Some(t) = textures.get_mut(id) {
                    for y in 0..img.size[1] {
                        for x in 0..img.size[0] {
                            let (dx, dy) = (x0 + x, y0 + y);
                            if dx < t.size[0] && dy < t.size[1] {
                                t.pixels[dy * t.size[0] + dx] = img.pixels[y * img.size[0] + x];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Premultiplied RGBA in 0..1 (gamma space).
type Px = [f32; 4];

fn to_px(c: Color32) -> Px {
    [
        c.r() as f32 / 255.0,
        c.g() as f32 / 255.0,
        c.b() as f32 / 255.0,
        c.a() as f32 / 255.0,
    ]
}

fn sample(t: &Texture, u: f32, v: f32) -> Px {
    let (w, h) = (t.size[0] as f32, t.size[1] as f32);
    let x = (u * w - 0.5).clamp(0.0, w - 1.0);
    let y = (v * h - 0.5).clamp(0.0, h - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(t.size[0] - 1), (y0 + 1).min(t.size[1] - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |xx: usize, yy: usize| to_px(t.pixels[yy * t.size[0] + xx]);
    let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    let mut out = [0.0; 4];
    for k in 0..4 {
        let top = a[k] + (b[k] - a[k]) * fx;
        let bot = c[k] + (d[k] - c[k]) * fx;
        out[k] = top + (bot - top) * fy;
    }
    out
}

fn raster_triangle(
    img: &mut [Px],
    w: usize,
    h: usize,
    clip: Rect,
    tex: &Texture,
    v: [&Vertex; 3],
    scale: f32,
) {
    let p: [Pos2; 3] = [v[0].pos, v[1].pos, v[2].pos].map(|q| pos2(q.x * scale, q.y * scale));
    let area = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y);
    if area.abs() < 1e-8 {
        return;
    }
    let min_x = p
        .iter()
        .map(|q| q.x)
        .fold(f32::INFINITY, f32::min)
        .max(clip.left())
        .max(0.0)
        .floor() as usize;
    let max_x = p
        .iter()
        .map(|q| q.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .min(clip.right())
        .min(w as f32)
        .ceil() as usize;
    let min_y = p
        .iter()
        .map(|q| q.y)
        .fold(f32::INFINITY, f32::min)
        .max(clip.top())
        .max(0.0)
        .floor() as usize;
    let max_y = p
        .iter()
        .map(|q| q.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .min(clip.bottom())
        .min(h as f32)
        .ceil() as usize;
    let cols = v.map(|q| to_px(q.color));
    for y in min_y..max_y {
        for x in min_x..max_x {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            if px < clip.left() || px > clip.right() || py < clip.top() || py > clip.bottom() {
                continue;
            }
            let w0 = ((p[1].x - px) * (p[2].y - py) - (p[2].x - px) * (p[1].y - py)) / area;
            let w1 = ((p[2].x - px) * (p[0].y - py) - (p[0].x - px) * (p[2].y - py)) / area;
            let w2 = 1.0 - w0 - w1;
            if w0 < -1e-5 || w1 < -1e-5 || w2 < -1e-5 {
                continue;
            }
            let u = w0 * v[0].uv.x + w1 * v[1].uv.x + w2 * v[2].uv.x;
            let vv = w0 * v[0].uv.y + w1 * v[1].uv.y + w2 * v[2].uv.y;
            let texel = sample(tex, u, vv);
            let mut src = [0.0f32; 4];
            for k in 0..4 {
                src[k] = (w0 * cols[0][k] + w1 * cols[1][k] + w2 * cols[2][k]) * texel[k];
            }
            let dst = &mut img[y * w + x];
            for k in 0..4 {
                dst[k] = src[k] + dst[k] * (1.0 - src[3]);
            }
        }
    }
}

fn write_bmp(path: &str, img: &[Px], w: usize, h: usize) -> std::io::Result<()> {
    let row = (w * 3).div_ceil(4) * 4;
    let size = 54 + row * h;
    let mut out = Vec::with_capacity(size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0u8; 24]);
    for y in (0..h).rev() {
        let start = out.len();
        for x in 0..w {
            let p = img[y * w + x];
            let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            out.extend_from_slice(&[c(p[2]), c(p[1]), c(p[0])]);
        }
        out.resize(start + row, 0);
    }
    std::fs::File::create(path)?.write_all(&out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "skatotune_gui.bmp".to_string());
    let seconds: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7.0);
    let scale: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let flag = |f: &str| args.iter().any(|a| a == f);
    let advanced = flag("--advanced");

    // A musically typical setup so the lanes show scale states.
    let mut values = ParamValues {
        key: KeyChoice::C,
        scale: ScaleChoice::Major,
        ..ParamValues::default()
    };
    values.note_overrides[5] = NoteOverride::Bypass; // F sung freely
    if flag("--studio") {
        values.latency_mode = LatencyModeChoice::Studio;
    }
    if flag("--hard") {
        values = Preset::HardTune.apply(&values);
    }
    if flag("--learn") || flag("--learn-empty") {
        values.midi_mode = MidiModeChoice::Learn;
    }
    let number = |name: &str| {
        args.iter()
            .find_map(|a| a.strip_prefix(name))
            .map(|v| v.parse::<f32>().unwrap_or_else(|_| panic!("bad {name}{v}")))
    };
    if let Some(st) = number("--transpose=") {
        values.transpose_st = st;
    }
    if let Some(pct) = number("--vibdepth=") {
        values.vibrato_depth_pct = pct;
    }
    if let Some(ms) = number("--retune=") {
        values.retune_ms = ms;
    }
    let params = Arc::new(SkatotuneParams::with_values(&values));
    if flag("--learn") || flag("--learned") {
        params
            .learned_scale
            .store(skatotune::learn::pack(&[0, 4, 7]), Ordering::Relaxed);
    }
    let shared = Arc::new(SharedState::default());
    shared.sample_rate.store(48_000.0, Ordering::Relaxed);
    let d = if flag("--studio") { 1920 } else { 192 };
    shared.latency_samples.store(d, Ordering::Relaxed);
    shared.cpu_load.store(0.014, Ordering::Relaxed);
    shared.cpu_peak.store(0.031, Ordering::Relaxed);
    if flag("--pending") {
        // A structural change waiting for the host restart.
        shared.restart_pending.store(true, Ordering::Relaxed);
        shared.awaiting_host.store(true, Ordering::Relaxed);
        shared
            .pending_latency_samples
            .store(1920, Ordering::Relaxed);
    }

    let mut labs = params.labs();
    if flag("--no-ribbon") {
        labs.correction_ribbon = false;
    }
    if flag("--heat") {
        labs.heat_line = true;
    }
    if let Some(id) = args.iter().find_map(|a| a.strip_prefix("--theme=")) {
        labs.correction_theme = CorrectionTheme::from_id(id).unwrap_or_else(|| {
            let ids: Vec<&str> = CorrectionTheme::ALL.iter().map(|t| t.id()).collect();
            panic!("unknown --theme={id}; use one of: {}", ids.join(", "));
        });
    }
    params.set_labs(labs);

    let mut app = SkatotuneEditor::new(params, shared);
    app.set_advanced_open(advanced);
    app.set_labs_open(flag("--labs"));
    let ctx = egui::Context::default();
    ctx.set_pixels_per_point(scale);
    app.setup(&ctx, GuiContext::new(Arc::new(HeadlessHost)));

    let mut textures = HashMap::new();
    let start = Instant::now();
    let frames = (seconds * 30.0).max(1.0) as usize;
    let mut last = None;
    for _ in 0..frames {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(WIDTH, HEIGHT))),
            time: Some(start.elapsed().as_secs_f64()),
            ..Default::default()
        };
        let mut out = ctx.run_ui(input, |ui| app.draw(ui));
        apply_textures(&mut textures, &out.textures_delta);
        // Mark the deltas as handled (egui asserts on unapplied deltas).
        out.textures_delta.clear();
        last = Some(out);
        std::thread::sleep(Duration::from_millis(33));
    }
    let Some(out) = last else { return };
    let prims: Vec<ClippedPrimitive> = ctx.tessellate(out.shapes, out.pixels_per_point);

    let (w, h) = ((WIDTH * scale) as usize, (HEIGHT * scale) as usize);
    let bg = to_px(Color32::from_rgb(12, 14, 18));
    let mut img = vec![bg; w * h];
    let mut triangles = 0usize;
    for cp in &prims {
        let Primitive::Mesh(mesh) = &cp.primitive else {
            continue;
        };
        let Some(tex) = textures.get(&mesh.texture_id) else {
            continue;
        };
        let clip = Rect::from_min_max(
            pos2(cp.clip_rect.min.x * scale, cp.clip_rect.min.y * scale),
            pos2(cp.clip_rect.max.x * scale, cp.clip_rect.max.y * scale),
        );
        for [a, b, c] in mesh.indices.as_chunks::<3>().0 {
            let vs = [
                &mesh.vertices[*a as usize],
                &mesh.vertices[*b as usize],
                &mesh.vertices[*c as usize],
            ];
            raster_triangle(&mut img, w, h, clip, tex, vs, scale);
            triangles += 1;
        }
    }
    match write_bmp(&path, &img, w, h) {
        Ok(()) => println!("wrote {path} ({w}x{h}, {triangles} triangles, {frames} frames)"),
        Err(e) => {
            eprintln!("failed to write {path}: {e}");
            std::process::exit(1);
        }
    }
}
