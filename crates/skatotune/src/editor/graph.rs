//! The scrolling piano-roll pitch graph (R§8.23).
//!
//! Drawing order, back to front: piano-roll lanes (scale notes shaded by
//! state, labelled in the left gutter), time grid, target-note bars, the
//! vibrato band, the correction ribbon (Labs), the detected input (thin,
//! alpha = confidence, broken where unvoiced), the vibrato-free centre, and
//! the corrected output (thick, accent colour, or tinted by correction
//! strength with the Labs heat line), then the legend on a translucent
//! backing. Time runs right to left from the display clock's "now", in
//! output samples, so the graph lines up with what is heard.
//!
//! The history holds ~5-6 frames per pixel column, so the frames in view are
//! first folded into one [`Column`] aggregate per pixel (min/max/first of the
//! output, mean of the input and centre, max confidence, mean correction
//! strength, ...) in a reusable, preallocated buffer; every trace is then
//! drawn from the columns. egui shapes own their point lists, so each visible
//! polyline run of the plain traces costs one small `Vec` per frame. The
//! vertex-coloured meshes (ribbon, heat line, legend key) are kept in `Arc`s
//! that egui hands back once the frame is tessellated, so their buffers are
//! reused frame to frame and cost no allocation however often the colour
//! changes along the trace.
//!
//! All drawing goes through [`draw_pitch_roll`], the thin seam R§9.7 asks
//! for, so a future GUI backend change stays contained.

use std::sync::Arc;

use egui::epaint::PathStroke;
use egui::epaint::tessellator::Path;
use egui::{
    Align2, Color32, CornerRadius, Id, Mesh, Pos2, Rect, Response, Sense, Shape, Stroke,
    StrokeKind, Ui, pos2, vec2,
};
use skato_core::{NoteState, PitchFrame, frame_flags};

use super::correction::{
    self, HEAT_STEPS, correction_cents, heat_color, heat_step, heat_step_start, ramp_pos,
    ribbon_color,
};
use super::history::{GraphView, PitchHistory};
use super::theme::{self, ACCENT, BYPASS, CENTRE, DETECTED, MIDI, TEXT, TEXT_DIM, TEXT_FAINT};
use crate::labs::LabsSettings;
use crate::params::NOTE_NAMES;

/// Seconds of history shown.
pub const WINDOW_S: f64 = 6.0;
/// Width of the note-name gutter on the left.
const GUTTER: f32 = 40.0;

/// Inputs for one graph frame besides the history.
pub struct RollContext {
    /// Effective state per pitch class (lane shading).
    pub note_states: [NoteState; 12],
    pub sample_rate: f64,
    /// Display "now" in output samples (right edge).
    pub now_pos: f64,
    /// Control hop in samples (gap detection).
    pub hop: f64,
    /// MIDI note of the current target, highlighted in the gutter.
    pub highlight_note: Option<i32>,
    /// Labs switches that change the graph.
    pub labs: LabsSettings,
}

/// Per-pixel-column aggregate of the frames that fall into it.
#[derive(Clone, Copy, Debug, Default)]
struct Column {
    x: f32,
    /// A voicing gap precedes this column: polylines restart here.
    break_before: bool,
    n: u32,
    det_sum: f32,
    conf_max: f32,
    centre_sum: f32,
    out_min: f32,
    out_max: f32,
    /// Which extreme came first, so a jump inside a column keeps its direction.
    min_first: bool,
    target: f32,
    flags: u8,
    vib_sum: f32,
    out_sum: f32,
    /// Sum and count of `|correction|` over the frames that have one, and
    /// the signed sum (the ribbon's height).
    corr_sum: f32,
    corr_signed_sum: f32,
    corr_n: u32,
    /// Ribbon colour: ramp position of the strength averaged over a few
    /// neighbouring columns ([`smooth_strengths`]).
    ribbon_t: Option<f32>,
    /// Heat-line step of the strongest correction nearby.
    heat: u8,
}

impl Column {
    /// `corr`: the frame's correction in cents ([`correction_cents`]).
    fn new(x: f32, break_before: bool, f: &PitchFrame, corr: Option<f32>) -> Self {
        Self {
            x,
            break_before,
            n: 1,
            det_sum: f.detected,
            conf_max: f.confidence,
            centre_sum: f.centre,
            out_min: f.output,
            out_max: f.output,
            min_first: true,
            target: f.target,
            flags: f.flags,
            vib_sum: f.vib_extent,
            out_sum: f.output,
            corr_sum: corr.map_or(0.0, f32::abs),
            corr_signed_sum: corr.unwrap_or(0.0),
            corr_n: u32::from(corr.is_some()),
            ribbon_t: None,
            heat: 0,
        }
    }

    fn add(&mut self, f: &PitchFrame, corr: Option<f32>) {
        self.n += 1;
        self.out_sum += f.output;
        if let Some(c) = corr {
            self.corr_sum += c.abs();
            self.corr_signed_sum += c;
            self.corr_n += 1;
        }
        self.det_sum += f.detected;
        self.conf_max = self.conf_max.max(f.confidence);
        self.centre_sum += f.centre;
        if f.output < self.out_min {
            // A new minimum after the maximum was set means max came first.
            if self.out_max > self.out_min {
                self.min_first = false;
            }
            self.out_min = f.output;
        }
        if f.output > self.out_max {
            self.out_max = f.output;
        }
        self.target = f.target;
        self.flags |= f.flags;
        self.vib_sum += f.vib_extent;
    }

    fn detected(&self) -> f32 {
        self.det_sum / self.n as f32
    }

    fn centre(&self) -> f32 {
        self.centre_sum / self.n as f32
    }

    fn vib_extent(&self) -> f32 {
        self.vib_sum / self.n as f32
    }

    fn output(&self) -> f32 {
        self.out_sum / self.n as f32
    }

    /// Mean `|correction|` of the column's frames, cents (`None` when no
    /// frame had both pitches).
    fn strength(&self) -> Option<f32> {
        (self.corr_n > 0).then(|| self.corr_sum / self.corr_n as f32)
    }

    /// Mean signed correction, cents: how far, and which way, the plugin
    /// moved the pitch (`None` as for [`Self::strength`]).
    fn moved(&self) -> Option<f32> {
        (self.corr_n > 0).then(|| self.corr_signed_sum / self.corr_n as f32)
    }
}

/// Columns kept per frame at most (wider than any plot).
const MAX_COLUMNS: usize = 4096;

/// Reusable buffers for the graph (allocated once).
pub struct RollScratch {
    columns: Vec<Column>,
    points: Vec<Pos2>,
    /// Correction ribbon (Labs). Shared with egui for one frame at a time.
    ribbon: Arc<Mesh>,
    /// Heat-tinted output line (Labs), tessellated here and recoloured.
    heat: Arc<Mesh>,
    /// egui's path scratch for tessellating the heat line.
    path: Path,
    /// Gradients of the legend (correction key, heat-line swatch).
    key: Arc<Mesh>,
}

impl Default for RollScratch {
    fn default() -> Self {
        Self {
            columns: Vec::with_capacity(MAX_COLUMNS),
            points: Vec::with_capacity(MAX_COLUMNS),
            ribbon: Arc::new(Mesh::default()),
            heat: Arc::new(Mesh::default()),
            path: Path::default(),
            key: Arc::new(Mesh::default()),
        }
    }
}

/// Exclusive, emptied access to a mesh buffer that was handed to egui last
/// frame. egui drops its handle once the frame is tessellated, so this
/// normally reuses the same allocation; if something still holds the old
/// mesh (a renderer keeping the previous frame), a new buffer of the same
/// capacity replaces it instead.
fn reusable_mesh(slot: &mut Arc<Mesh>) -> &mut Mesh {
    if Arc::get_mut(slot).is_none() {
        let mut fresh = Mesh::default();
        fresh.vertices.reserve(slot.vertices.capacity());
        fresh.indices.reserve(slot.indices.capacity());
        *slot = Arc::new(fresh);
    }
    // Unique now, so this never clones.
    let mesh = Arc::make_mut(slot);
    // Not `Mesh::clear`, which frees the vertex buffer.
    mesh.vertices.clear();
    mesh.indices.clear();
    mesh
}

/// Whether a frame carries a usable input pitch.
pub fn is_voiced(f: &PitchFrame) -> bool {
    f.flags & frame_flags::VOICED != 0 && f.detected > 0.0
}

/// `"A4"` for MIDI note 69.
pub fn note_label(midi: i32) -> String {
    let name = NOTE_NAMES
        .get(midi.rem_euclid(12) as usize)
        .copied()
        .unwrap_or("?");
    format!("{name}{}", midi.div_euclid(12) - 1)
}

/// Pitch class 1, 3, 6, 8, 10 are the black keys.
pub fn is_black_key(pc: usize) -> bool {
    matches!(pc, 1 | 3 | 6 | 8 | 10)
}

/// Linear map between absolute cents and screen y inside `plot`.
#[derive(Clone, Copy)]
struct YMap {
    lo: f32,
    scale: f32,
    bottom: f32,
}

impl YMap {
    fn new(plot: Rect, view: &GraphView) -> Self {
        Self {
            lo: view.lo(),
            scale: plot.height() / view.span.max(1.0),
            bottom: plot.bottom(),
        }
    }

    fn y(&self, cents: f32) -> f32 {
        self.bottom - (cents - self.lo) * self.scale
    }
}

/// Draw the graph into `rect` and handle its mouse interaction:
/// drag or scroll pans (and leaves auto-follow), Cmd/Ctrl + scroll or pinch
/// zooms, double-click returns to auto-follow.
pub fn draw_pitch_roll(
    ui: &mut Ui,
    rect: Rect,
    history: &PitchHistory,
    view: &mut GraphView,
    ctx: &RollContext,
    scratch: &mut RollScratch,
) -> Response {
    let plot = Rect::from_min_max(pos2(rect.left() + GUTTER, rect.top()), rect.max);
    let resp = ui.interact(plot, Id::new("skato-pitch-graph"), Sense::click_and_drag());
    handle_input(ui, &resp, plot, view);

    let painter = ui.painter_at(rect);
    let ymap = YMap::new(plot, view);
    let lane_h = 100.0 * ymap.scale;

    draw_lanes(&painter, rect, plot, ymap, lane_h, view, ctx);
    let window = WINDOW_S * ctx.sample_rate.max(1.0);
    let t1 = ctx.now_pos;
    let t0 = t1 - window;
    let px_per_sample = f64::from(plot.width()) / window;
    let x_of = |pos: f64| plot.right() - ((t1 - pos) * px_per_sample) as f32;
    draw_time_grid(&painter, plot, ctx.sample_rate, t0, t1, &x_of);

    // Fold the visible frames into per-pixel columns.
    let first = history.lower_bound(t0.max(0.0) as u64).saturating_sub(1);
    let gap_limit = (3.0 * ctx.hop.max(1.0)) as u64;
    scratch.columns.clear();
    let mut broke = true;
    let mut prev_pos: Option<u64> = None;
    for (f, transpose_st) in history.iter_range_stamped(first, history.len()) {
        if !is_voiced(f) {
            broke = true;
            continue;
        }
        if prev_pos.is_some_and(|p| f.sample_pos.saturating_sub(p) > gap_limit) {
            broke = true;
        }
        prev_pos = Some(f.sample_pos);
        let x = x_of(f.sample_pos as f64).floor();
        // Each frame against the Transpose it was produced under.
        let corr = correction_cents(f, transpose_st);
        let room = scratch.columns.len() < scratch.columns.capacity().min(MAX_COLUMNS);
        match scratch.columns.last_mut() {
            Some(c) if !broke && c.x == x => c.add(f, corr),
            // Never grow past the preallocated capacity.
            _ if room => scratch.columns.push(Column::new(x, broke, f, corr)),
            _ => {}
        }
        broke = false;
    }

    let labs = ctx.labs;
    if labs.shows_correction_colour() {
        // About one vibrato cycle, in pixel columns.
        let heat_half = 0.5 * (correction::HEAT_WINDOW_S * ctx.sample_rate * px_per_sample) as f32;
        smooth_strengths(&mut scratch.columns, heat_half);
    }
    let cols = &scratch.columns;
    // One physical pixel: egui's own anti-aliasing width.
    let feather = 1.0 / painter.pixels_per_point().max(0.1);
    draw_target_bars(&painter, cols, ymap, lane_h, BarStyle::for_labs(labs));
    draw_vibrato_band(&painter, cols, ymap);
    if labs.correction_ribbon {
        let mesh = reusable_mesh(&mut scratch.ribbon);
        build_ribbon(mesh, cols, ymap, feather);
        if !mesh.indices.is_empty() {
            painter.add(Shape::Mesh(Arc::clone(&scratch.ribbon)));
        }
    }
    draw_detected(&painter, cols, ymap);
    let points = &mut scratch.points;
    // Centre: only where the vibrato split is engaged (elsewhere it equals
    // the input and would just double the line).
    draw_polyline(
        &painter,
        cols,
        points,
        Stroke::new(1.3, theme::alpha(CENTRE, 0.8)),
        |c| (c.flags & frame_flags::VIBRATO != 0).then(|| c.centre()),
        ymap,
    );
    // Output: a solid line, accent or heat-tinted. Without the ribbon a soft
    // glow sits under it; with the ribbon the glow would cover exactly the
    // small corrections the ribbon is there to show (±3 px ≈ ±11 c).
    let glow = !labs.correction_ribbon;
    if labs.heat_line {
        let mesh = reusable_mesh(&mut scratch.heat);
        let path = &mut scratch.path;
        if glow {
            build_heat_line(mesh, path, points, cols, ymap, OUTPUT_GLOW, feather);
        }
        build_heat_line(mesh, path, points, cols, ymap, OUTPUT_LINE, feather);
        if !mesh.indices.is_empty() {
            painter.add(Shape::Mesh(Arc::clone(&scratch.heat)));
        }
    } else {
        if glow {
            draw_output(&painter, cols, points, ymap, OUTPUT_GLOW.stroke(ACCENT));
        }
        draw_output(&painter, cols, points, ymap, OUTPUT_LINE.stroke(ACCENT));
    }

    draw_now_marker(&painter, plot, history, ctx, cols, ymap, &x_of);
    // Only when nothing voiced is stored at all: a pause between phrases (or a
    // plugin sleep/wake) must not replace the previous phrase with this hint.
    if cols.is_empty() && !history.any_voiced() {
        painter.text(
            plot.center(),
            Align2::CENTER_CENTER,
            "No pitch detected: sing into the input to see the trace",
            theme::font(13.0),
            TEXT_FAINT,
        );
    }
    draw_legend(&painter, plot, labs, &mut scratch.key);
    painter.rect_stroke(
        rect,
        CornerRadius::same(theme::RADIUS),
        Stroke::new(1.0, theme::BORDER),
        StrokeKind::Inside,
    );
    resp
}

fn handle_input(ui: &Ui, resp: &Response, plot: Rect, view: &mut GraphView) {
    let cents_per_px = view.span / plot.height().max(1.0);
    if resp.dragged() {
        view.pan(resp.drag_delta().y * cents_per_px);
    }
    if resp.double_clicked() {
        view.reset();
    }
    if resp.hovered() {
        let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        if (zoom - 1.0).abs() > 1e-4 {
            view.zoom(zoom);
        } else if scroll.abs() > 0.1 {
            view.pan(scroll * cents_per_px);
        }
    }
}

fn draw_lanes(
    painter: &egui::Painter,
    rect: Rect,
    plot: Rect,
    ymap: YMap,
    lane_h: f32,
    view: &GraphView,
    ctx: &RollContext,
) {
    painter.rect_filled(rect, CornerRadius::same(theme::RADIUS), theme::LANE_OFF);
    let gutter = Rect::from_min_max(rect.min, pos2(plot.left(), rect.bottom()));
    painter.rect_filled(
        gutter,
        CornerRadius {
            nw: theme::RADIUS,
            sw: theme::RADIUS,
            ne: 0,
            se: 0,
        },
        theme::PANEL,
    );

    let n_lo = ((view.lo() - 50.0) / 100.0).floor() as i32;
    let n_hi = ((view.hi() + 50.0) / 100.0).ceil() as i32;
    let label_size = (lane_h * 0.62).clamp(8.5, 11.5);
    for n in n_lo..=n_hi {
        let pc = n.rem_euclid(12) as usize;
        let state = ctx.note_states.get(pc).copied().unwrap_or(NoteState::On);
        let y_top = ymap.y(n as f32 * 100.0 + 50.0);
        let y_bot = ymap.y(n as f32 * 100.0 - 50.0);
        let lane = Rect::from_x_y_ranges(plot.x_range(), y_top..=y_bot).intersect(plot);
        if lane.height() <= 0.0 {
            continue;
        }
        let black = is_black_key(pc);
        let fill = match state {
            NoteState::On if black => theme::LANE_ON_BLACK,
            NoteState::On => theme::LANE_ON,
            NoteState::Bypass => theme::LANE_BYPASS,
            NoteState::Remove => theme::LANE_OFF,
        };
        painter.rect_filled(lane, CornerRadius::ZERO, fill);
        let highlighted = ctx.highlight_note == Some(n);
        if highlighted {
            painter.rect_filled(lane, CornerRadius::ZERO, theme::alpha(ACCENT, 0.06));
        }
        // Octave boundaries (below each C) a little stronger.
        let line = if pc == 0 {
            theme::GRID
        } else {
            theme::LANE_LINE
        };
        if y_bot > plot.top() && y_bot < plot.bottom() {
            painter.hline(plot.x_range(), y_bot, Stroke::new(1.0, line));
        }

        // Gutter: key colour strip and the note name.
        let strip = Rect::from_x_y_ranges(plot.left() - 6.0..=plot.left() - 1.0, y_top..=y_bot)
            .intersect(gutter)
            .shrink2(vec2(0.0, 0.5));
        if strip.height() > 0.0 {
            let key_col = if black {
                Color32::from_rgb(8, 9, 12)
            } else {
                Color32::from_rgb(170, 176, 188)
            };
            painter.rect_filled(strip, CornerRadius::same(1), key_col);
        }
        let centre_y = 0.5 * (y_top + y_bot);
        let show_label = (lane_h >= 10.0 || pc == 0)
            && centre_y > rect.top() + 4.0
            && centre_y < rect.bottom() - 4.0;
        if show_label {
            let col = if highlighted {
                ACCENT
            } else {
                match state {
                    NoteState::On => {
                        if pc == 0 {
                            TEXT
                        } else {
                            TEXT_DIM
                        }
                    }
                    NoteState::Bypass => BYPASS,
                    NoteState::Remove => TEXT_FAINT,
                }
            };
            painter.text(
                pos2(plot.left() - 9.0, centre_y),
                Align2::RIGHT_CENTER,
                note_label(n),
                theme::font(label_size),
                col,
            );
        }
    }
}

fn draw_time_grid(
    painter: &egui::Painter,
    plot: Rect,
    sr: f64,
    t0: f64,
    t1: f64,
    x_of: &impl Fn(f64) -> f32,
) {
    let half = 0.5 * sr.max(1.0);
    let k0 = (t0.max(0.0) / half).ceil() as i64;
    let k1 = (t1 / half).floor() as i64;
    // At most 2 lines per second of the window: bounded loop.
    for k in k0..=k1.min(k0 + 4 * WINDOW_S as i64) {
        let x = x_of(k as f64 * half);
        let col = if k % 2 == 0 {
            theme::GRID
        } else {
            theme::alpha(theme::GRID, 0.45)
        };
        painter.vline(x, plot.y_range(), Stroke::new(1.0, col));
    }
}

/// How the target bars are drawn.
#[derive(Clone, Copy)]
struct BarStyle {
    fill: f32,
    outline: f32,
}

impl BarStyle {
    const NORMAL: Self = Self {
        fill: 0.13,
        outline: 0.42,
    };
    /// With a Labs correction colouring on, the bars step back to (almost)
    /// an outline: their accent fill would otherwise read like a faint
    /// ribbon, and the heat line's darkest step must stay clear of the
    /// outline (2.3:1).
    const LABS: Self = Self {
        fill: 0.05,
        outline: 0.34,
    };

    fn for_labs(labs: LabsSettings) -> Self {
        if labs.shows_correction_colour() {
            Self::LABS
        } else {
            Self::NORMAL
        }
    }
}

fn draw_target_bars(
    painter: &egui::Painter,
    cols: &[Column],
    ymap: YMap,
    lane_h: f32,
    style: BarStyle,
) {
    let h = (lane_h * 0.6).clamp(4.0, 14.0);
    let mut i = 0;
    while i < cols.len() {
        let Some(c) = cols.get(i) else { break };
        if c.target <= 0.0 {
            i += 1;
            continue;
        }
        let target = c.target;
        let flags = c.flags;
        let x0 = c.x;
        let mut x1 = c.x + 1.0;
        let mut j = i + 1;
        while let Some(d) = cols.get(j) {
            if d.break_before || (d.target - target).abs() > 0.5 {
                break;
            }
            x1 = d.x + 1.0;
            j += 1;
        }
        let y = ymap.y(target);
        let bar = Rect::from_min_max(pos2(x0, y - 0.5 * h), pos2(x1.max(x0 + 2.0), y + 0.5 * h));
        let base = if flags & frame_flags::MIDI != 0 {
            MIDI
        } else if flags & frame_flags::BYPASSED != 0 {
            BYPASS
        } else {
            ACCENT
        };
        let r = CornerRadius::same(3);
        painter.rect_filled(bar, r, theme::alpha(base, style.fill));
        painter.rect_stroke(
            bar,
            r,
            Stroke::new(1.0, theme::alpha(base, style.outline)),
            StrokeKind::Inside,
        );
        i = j;
    }
}

fn draw_vibrato_band(painter: &egui::Painter, cols: &[Column], ymap: YMap) {
    let mut mesh = Mesh::default();
    let fill = theme::alpha(CENTRE, 0.13);
    let mut prev: Option<u32> = None;
    for c in cols {
        let ext = c.vib_extent();
        if c.flags & frame_flags::VIBRATO == 0 || ext <= 0.5 {
            prev = None;
            continue;
        }
        if c.break_before {
            prev = None;
        }
        let centre = c.centre();
        let idx = mesh.vertices.len() as u32;
        mesh.colored_vertex(pos2(c.x, ymap.y(centre + ext)), fill);
        mesh.colored_vertex(pos2(c.x, ymap.y(centre - ext)), fill);
        if let Some(p) = prev {
            mesh.add_triangle(p, p + 1, idx);
            mesh.add_triangle(p + 1, idx + 1, idx);
        }
        prev = Some(idx);
    }
    if !mesh.is_empty() {
        painter.add(Shape::mesh(mesh));
    }
}

fn draw_detected(painter: &egui::Painter, cols: &[Column], ymap: YMap) {
    for pair in cols.windows(2) {
        let [a, b] = pair else { continue };
        if b.break_before {
            continue;
        }
        let conf = a.conf_max.min(b.conf_max);
        let col = theme::alpha(DETECTED, 0.15 + 0.6 * conf);
        painter.line_segment(
            [
                pos2(a.x + 0.5, ymap.y(a.detected())),
                pos2(b.x + 0.5, ymap.y(b.detected())),
            ],
            Stroke::new(1.4, col),
        );
    }
}

/// Flush the accumulated run as one polyline (or a dot for a single point).
fn flush_run(painter: &egui::Painter, points: &mut Vec<Pos2>, stroke: Stroke) {
    match points.len() {
        0 => {}
        1 => {
            if let Some(&p) = points.first() {
                painter.circle_filled(p, stroke.width * 0.5, stroke.color);
            }
        }
        _ => {
            painter.add(Shape::line(points.clone(), stroke));
        }
    }
    points.clear();
}

/// Draw `value(column)` as polylines, restarting at voicing gaps and where
/// `value` returns `None`.
fn draw_polyline(
    painter: &egui::Painter,
    cols: &[Column],
    points: &mut Vec<Pos2>,
    stroke: Stroke,
    value: impl Fn(&Column) -> Option<f32>,
    ymap: YMap,
) {
    points.clear();
    for c in cols {
        if c.break_before {
            flush_run(painter, points, stroke);
        }
        match value(c) {
            Some(v) => points.push(pos2(c.x + 0.5, ymap.y(v))),
            None => flush_run(painter, points, stroke),
        }
    }
    flush_run(painter, points, stroke);
}

/// Append a column's output points: its min and max (in time order) so
/// hard-tune steps stay vertical and sharp instead of being averaged away.
fn push_output_points(points: &mut Vec<Pos2>, c: &Column, ymap: YMap) {
    let (y_min, y_max) = (ymap.y(c.out_min), ymap.y(c.out_max));
    if (y_min - y_max).abs() < 1.0 {
        points.push(pos2(c.x + 0.5, 0.5 * (y_min + y_max)));
    } else if c.min_first {
        points.push(pos2(c.x + 0.25, y_min));
        points.push(pos2(c.x + 0.75, y_max));
    } else {
        points.push(pos2(c.x + 0.25, y_max));
        points.push(pos2(c.x + 0.75, y_min));
    }
}

/// Width and opacity of one output-trace layer.
#[derive(Clone, Copy)]
struct OutputLayer {
    width: f32,
    opacity: f32,
}

impl OutputLayer {
    fn stroke(self, color: Color32) -> Stroke {
        Stroke::new(self.width, theme::alpha(color, self.opacity))
    }
}

/// The output trace itself.
const OUTPUT_LINE: OutputLayer = OutputLayer {
    width: 2.4,
    opacity: 1.0,
};
/// The soft glow under it (only without the ribbon).
const OUTPUT_GLOW: OutputLayer = OutputLayer {
    width: 6.0,
    opacity: 0.12,
};

/// The output trace in one colour.
fn draw_output(
    painter: &egui::Painter,
    cols: &[Column],
    points: &mut Vec<Pos2>,
    ymap: YMap,
    stroke: Stroke,
) {
    points.clear();
    for c in cols {
        if c.break_before {
            flush_run(painter, points, stroke);
        }
        push_output_points(points, c, ymap);
    }
    flush_run(painter, points, stroke);
}

/// Voiced runs of `cols`: index ranges between voicing breaks.
fn runs(cols: &[Column]) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
    let mut start = 0;
    std::iter::from_fn(move || {
        if start >= cols.len() {
            return None;
        }
        let len = cols
            .get(start + 1..)
            .and_then(|rest| rest.iter().position(|c| c.break_before))
            .map_or(cols.len() - start, |p| p + 1);
        let run = start..start + len;
        start += len;
        Some(run)
    })
}

/// Fill the smoothed correction values of every column (Labs), each within
/// its own voiced run: the ribbon colour from the mean strength over
/// ±[`correction::RIBBON_SMOOTH_COLS`] px (a tall ribbon then does not flicker
/// column to column), and the heat-line step from the strongest correction
/// within ±`heat_half` px (flattened vibrato reads as one strongly corrected
/// stretch, not as a dashed line). Bounded: at most one window of
/// neighbours per column.
fn smooth_strengths(cols: &mut [Column], heat_half: f32) {
    let ribbon_half = correction::RIBBON_SMOOTH_COLS;
    let reach = heat_half.max(ribbon_half);
    let mut start = 0;
    loop {
        let next = runs(cols.get(start..).unwrap_or_default()).next();
        let Some(run) = next else { break };
        let range = start + run.start..start + run.end;
        start = range.end;
        let Some(run) = cols.get_mut(range) else {
            break;
        };
        for i in 0..run.len() {
            let x = run[i].x;
            let lo = (0..i).rev().take_while(|&j| x - run[j].x <= reach).last();
            let hi = (i..run.len()).take_while(|&j| run[j].x - x <= reach).last();
            let (mut sum, mut n, mut peak) = (0.0f32, 0u32, 0.0f32);
            for c in &run[lo.unwrap_or(i)..=hi.unwrap_or(i)] {
                let Some(s) = c.strength() else { continue };
                let d = (c.x - x).abs();
                if d <= ribbon_half {
                    sum += s;
                    n += 1;
                }
                if d <= heat_half {
                    peak = peak.max(s);
                }
            }
            run[i].ribbon_t = (n > 0).then(|| sum / n as f32).and_then(ramp_pos);
            run[i].heat = heat_step(peak);
        }
    }
}

/// Tessellate the output trace into `mesh` (Labs: heat line) with egui's own
/// path stroker, one stroke per voiced run, then recolour every new vertex
/// by the heat step of the column under it. One mesh for the whole trace, so
/// no seams where the colour changes and no allocation per colour run; the
/// colour blends over the one segment where a step changes.
fn build_heat_line(
    mesh: &mut Mesh,
    path: &mut Path,
    points: &mut Vec<Pos2>,
    cols: &[Column],
    ymap: YMap,
    layer: OutputLayer,
    feather: f32,
) {
    // Opaque white: after tessellation a vertex's alpha is its coverage.
    let stroke = PathStroke::new(layer.width, Color32::WHITE);
    for run in runs(cols) {
        let Some(run) = cols.get(run) else { continue };
        points.clear();
        for c in run {
            push_output_points(points, c, ymap);
        }
        if let [p] = points.as_slice() {
            // A lone column: a short dash, so it still shows.
            let p = *p;
            points.clear();
            points.extend([p - vec2(0.5, 0.0), p + vec2(0.5, 0.0)]);
        }
        if points.len() < 2 {
            continue; // (egui's path asserts at least two points)
        }
        let first = mesh.vertices.len();
        path.clear();
        path.add_open_points(points);
        path.stroke_open(feather, &stroke, mesh);
        for v in mesh.vertices.get_mut(first..).unwrap_or(&mut []) {
            // Nearest column by x (the run is sorted by x).
            let i = run.partition_point(|c| c.x + 0.5 < v.pos.x);
            let near = match (i.checked_sub(1).and_then(|j| run.get(j)), run.get(i)) {
                (Some(a), Some(b)) if v.pos.x - (a.x + 0.5) < (b.x + 0.5) - v.pos.x => a,
                (_, Some(b)) => b,
                (Some(a), None) => a,
                (None, None) => continue,
            };
            let coverage = f32::from(v.color.a()) / 255.0;
            v.color = theme::alpha(heat_color(near.heat), layer.opacity * coverage);
        }
    }
}

/// Fill the correction ribbon into `mesh`. It hangs off the output line:
/// per column one vertex on the output and one at `output − c` (the input
/// plus the Transpose shift, so its height is the correction and matches
/// its colour; at Transpose 0 that is exactly the input trace), plus a
/// transparent one a pixel further out that anti-aliases the free edge.
/// Neighbouring columns are joined by triangles; the strip breaks where the
/// traces break (voicing gaps) and where a column has no strength.
fn build_ribbon(mesh: &mut Mesh, cols: &[Column], ymap: YMap, feather: f32) {
    // Previous column's first vertex and the side its free edge faces.
    let mut prev: Option<(u32, bool)> = None;
    for c in cols {
        if c.break_before {
            prev = None;
        }
        let out = c.output();
        let free = c.moved().map(|m| out - m);
        let (Some(t), Some(free)) = (
            c.ribbon_t,
            free.filter(|f| f.is_finite() && out.is_finite()),
        ) else {
            prev = None;
            continue;
        };
        let Ok(idx) = u32::try_from(mesh.vertices.len()) else {
            break;
        };
        let col = ribbon_color(t);
        let x = c.x + 0.5;
        let (y_out, y_free) = (ymap.y(out), ymap.y(free));
        let below = y_free >= y_out;
        let fringe = if below { feather } else { -feather };
        mesh.colored_vertex(pos2(x, y_out), col);
        mesh.colored_vertex(pos2(x, y_free), col);
        mesh.colored_vertex(pos2(x, y_free + fringe), Color32::TRANSPARENT);
        if let Some((p, p_below)) = prev {
            mesh.add_triangle(p, p + 1, idx);
            mesh.add_triangle(p + 1, idx + 1, idx);
            // Where the correction changes sign the edge crosses the output
            // line; no fringe across that column pair.
            if p_below == below {
                mesh.add_triangle(p + 1, p + 2, idx + 1);
                mesh.add_triangle(p + 2, idx + 2, idx + 1);
            }
        }
        prev = Some((idx, below));
    }
}

fn draw_now_marker(
    painter: &egui::Painter,
    plot: Rect,
    history: &PitchHistory,
    ctx: &RollContext,
    cols: &[Column],
    ymap: YMap,
    x_of: &impl Fn(f64) -> f32,
) {
    painter.vline(
        plot.right() - 1.0,
        plot.y_range(),
        Stroke::new(1.0, theme::alpha(ACCENT, 0.25)),
    );
    // The newest voiced frame, if it is recent, gets a dot on the output.
    let recent = 0.15 * ctx.sample_rate;
    let newest = (0..history.len().min(64))
        .filter_map(|k| history.get(history.len() - 1 - k))
        .find(|f| is_voiced(f));
    if let Some(f) = newest.filter(|f| ctx.now_pos - f.sample_pos as f64 <= recent) {
        let p = pos2(
            x_of(f.sample_pos as f64).min(plot.right() - 3.0),
            ymap.y(f.output),
        );
        // The dot continues the output line (the newest column), so it
        // takes the line's colour.
        let col = match cols.last() {
            Some(c) if ctx.labs.heat_line => heat_color(c.heat),
            _ => ACCENT,
        };
        if plot.contains(p) {
            painter.circle_filled(p, 6.0, theme::alpha(col, 0.2));
            painter.circle_filled(p, 3.5, col);
        }
    }
}

/// Legend text size.
const LEGEND_FONT: f32 = 10.5;
/// Swatch length, and the gaps around it.
const SWATCH_W: f32 = 14.0;
const SWATCH_GAP: f32 = 6.0;
const ITEM_GAP: f32 = 12.0;
/// Width of the correction key's gradient bar.
const KEY_W: f32 = 176.0;
const KEY_H: f32 = 8.0;
/// Ticks of the correction key: cents and label ("" = an unlabelled tick
/// that shows the scale is not linear).
const KEY_TICKS: [(f32, &str); 5] = [
    (0.0, "0"),
    (10.0, ""),
    (25.0, "25"),
    (50.0, "50"),
    (100.0, "100+ c"),
];

/// Series legend in the top-right corner, on a translucent backing so a
/// bright trace or ribbon underneath never washes it out; with a Labs
/// correction experiment on, the correction colour key below it.
fn draw_legend(painter: &egui::Painter, plot: Rect, labs: LabsSettings, key: &mut Arc<Mesh>) {
    let right = plot.right() - 10.0;
    let y = plot.top() + 12.0;
    let font = theme::font(LEGEND_FONT);
    let labels = ["Output", "Input", "Centre", "Target"]
        .map(|l| painter.layout_no_wrap(l.to_owned(), font.clone(), TEXT_DIM));
    // Right to left: label, then its swatch in front of it.
    let mut x = right;
    let mut label_x = [0.0f32; 4];
    for (i, g) in labels.iter().enumerate().rev() {
        let left = x - g.size().x;
        if let Some(slot) = label_x.get_mut(i) {
            *slot = left;
        }
        x = left - SWATCH_GAP - SWATCH_W - ITEM_GAP;
    }
    let row_left = x + ITEM_GAP;

    let with_key = labs.shows_correction_colour();
    let bar = Rect::from_min_size(pos2(right - KEY_W, y + 13.0), vec2(KEY_W, KEY_H));
    let key_label =
        with_key.then(|| painter.layout_no_wrap("Correction".to_owned(), font.clone(), TEXT_DIM));
    let key_left = key_label
        .as_ref()
        .map_or(f32::INFINITY, |g| bar.left() - 8.0 - g.size().x);
    let bottom = if with_key {
        bar.bottom() + 16.0
    } else {
        y + 9.0
    };
    let backing = Rect::from_min_max(
        pos2(row_left.min(key_left) - 8.0, plot.top() + 3.0),
        pos2(plot.right() - 3.0, bottom),
    );
    painter.rect_filled(
        backing,
        CornerRadius::same(theme::RADIUS),
        theme::alpha(theme::BG, 0.85),
    );

    let mesh = reusable_mesh(key);
    let style = BarStyle::for_labs(labs);
    for (i, g) in labels.into_iter().enumerate() {
        let lx = label_x.get(i).copied().unwrap_or(right);
        let s1 = lx - SWATCH_GAP;
        let swatch = Rect::from_min_max(pos2(s1 - SWATCH_W, y - 3.0), pos2(s1, y + 3.0));
        match i {
            // Heat line: the swatch shows the line's own range.
            0 if labs.heat_line => push_gradient(
                mesh,
                Rect::from_center_size(swatch.center(), vec2(SWATCH_W, OUTPUT_LINE.width)),
                heat_color(0),
                heat_color(HEAT_STEPS - 1),
            ),
            0 => line_swatch(painter, swatch, OUTPUT_LINE.width, ACCENT),
            1 => line_swatch(painter, swatch, 1.4, theme::alpha(DETECTED, 0.75)),
            2 => line_swatch(painter, swatch, 1.3, CENTRE),
            // Target: a small bar drawn like the bars on the graph (its
            // outline a touch stronger: it is much smaller than a bar).
            _ => {
                let r = CornerRadius::same(2);
                let bar_swatch = swatch.expand2(vec2(0.0, 0.5));
                painter.rect_filled(bar_swatch, r, theme::alpha(ACCENT, style.fill));
                painter.rect_stroke(
                    bar_swatch,
                    r,
                    Stroke::new(1.0, theme::alpha(ACCENT, style.outline + 0.1)),
                    StrokeKind::Inside,
                );
            }
        }
        painter.galley(pos2(lx, y - 0.5 * g.size().y), g, TEXT_DIM);
    }
    if let Some(g) = key_label {
        painter.galley(
            pos2(key_left, bar.center().y - 0.5 * g.size().y),
            g,
            TEXT_DIM,
        );
        draw_correction_key(painter, bar, labs, mesh);
    }
    if !mesh.indices.is_empty() {
        painter.add(Shape::Mesh(Arc::clone(key)));
    }
    if with_key {
        draw_key_frame_and_ticks(painter, bar);
    }
}

fn line_swatch(painter: &egui::Painter, swatch: Rect, width: f32, color: Color32) {
    painter.line_segment(
        [swatch.left_center(), swatch.right_center()],
        Stroke::new(width, color),
    );
}

/// A left-to-right gradient quad.
fn push_gradient(mesh: &mut Mesh, rect: Rect, from: Color32, to: Color32) {
    let Ok(i) = u32::try_from(mesh.vertices.len()) else {
        return;
    };
    mesh.colored_vertex(rect.left_top(), from);
    mesh.colored_vertex(rect.left_bottom(), from);
    mesh.colored_vertex(rect.right_top(), to);
    mesh.colored_vertex(rect.right_bottom(), to);
    mesh.add_triangle(i, i + 1, i + 2);
    mesh.add_triangle(i + 1, i + 3, i + 2);
}

/// The correction key's colours in `bar` (a lane-coloured strip, like the
/// graph): the ribbon's shading as a smooth gradient, and the heat line's
/// four steps as blocks, through the middle when the ribbon is on as well,
/// filling the bar when the line is the only colouring.
fn draw_correction_key(painter: &egui::Painter, bar: Rect, labs: LabsSettings, mesh: &mut Mesh) {
    painter.rect_filled(bar, CornerRadius::ZERO, theme::LANE_ON);
    let x_at = |t: f32| bar.left() + t * bar.width();
    if labs.correction_ribbon {
        // Slices narrow enough that the mesh's own (linear) vertex-colour
        // interpolation matches the ramp.
        const SLICES: usize = 30;
        for i in 0..SLICES {
            let (t0, t1) = (i as f32 / SLICES as f32, (i + 1) as f32 / SLICES as f32);
            push_gradient(
                mesh,
                Rect::from_x_y_ranges(x_at(t0)..=x_at(t1), bar.y_range()),
                ribbon_color(t0),
                ribbon_color(t1),
            );
        }
    }
    if labs.heat_line {
        let band = if labs.correction_ribbon {
            Rect::from_center_size(bar.center(), vec2(bar.width(), 3.0))
        } else {
            bar
        };
        for step in 0..HEAT_STEPS {
            let t0 = heat_step_start(step);
            let t1 = if step + 1 < HEAT_STEPS {
                heat_step_start(step + 1)
            } else {
                1.0
            };
            let c = heat_color(step);
            push_gradient(
                mesh,
                Rect::from_x_y_ranges(x_at(t0)..=x_at(t1), band.y_range()),
                c,
                c,
            );
        }
    }
}

/// The key's hairline frame (the in-tune end deliberately fades into the
/// lane) and its ticks, labels in the theme's text colours. "100+ c" ends
/// flush with the bar.
fn draw_key_frame_and_ticks(painter: &egui::Painter, bar: Rect) {
    painter.rect_stroke(
        bar.expand(0.5),
        CornerRadius::ZERO,
        Stroke::new(1.0, theme::BORDER_HI),
        StrokeKind::Outside,
    );
    let label_y = bar.bottom() + 9.0;
    for (cents, label) in KEY_TICKS {
        let x = (bar.left() + ramp_pos(cents).unwrap_or(0.0) * bar.width()).round() - 0.5;
        let len = if label.is_empty() { 2.5 } else { 4.0 };
        painter.vline(
            x.clamp(bar.left(), bar.right() - 0.5),
            bar.bottom() + 1.0..=bar.bottom() + 1.0 + len,
            Stroke::new(1.0, TEXT_FAINT),
        );
        if label.is_empty() {
            continue;
        }
        let (pos, align) = if cents >= 100.0 {
            (pos2(bar.right() + 0.5, label_y), Align2::RIGHT_CENTER)
        } else if cents <= 0.0 {
            (pos2(bar.left() - 0.5, label_y), Align2::LEFT_CENTER)
        } else {
            (pos2(x, label_y), Align2::CENTER_CENTER)
        };
        painter.text(pos, align, label, theme::font(9.5), TEXT_DIM);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_labels() {
        assert_eq!(note_label(69), "A4");
        assert_eq!(note_label(60), "C4");
        assert_eq!(note_label(61), "C#4");
        assert_eq!(note_label(59), "B3");
        assert_eq!(note_label(0), "C-1");
    }

    #[test]
    fn black_keys() {
        let black: Vec<usize> = (0..12).filter(|&pc| is_black_key(pc)).collect();
        assert_eq!(black, vec![1, 3, 6, 8, 10]);
    }

    #[test]
    fn column_tracks_extremes_in_order() {
        let f = |output| PitchFrame {
            output,
            flags: frame_flags::VOICED,
            detected: 6000.0,
            ..PitchFrame::default()
        };
        let mut c = Column::new(0.0, false, &f(6000.0), None);
        c.add(&f(6100.0), None);
        c.add(&f(5900.0), None);
        assert_eq!((c.out_min, c.out_max), (5900.0, 6100.0));
        assert!(!c.min_first, "max came before the new min");
        let mut up = Column::new(0.0, false, &f(5900.0), None);
        up.add(&f(6100.0), None);
        assert!(up.min_first);
        assert_eq!(c.n, 3);
        assert!((c.detected() - 6000.0).abs() < 1e-3);
    }

    #[test]
    fn y_map_is_linear_and_inverted() {
        let plot = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 400.0));
        let view = GraphView::default();
        let m = YMap::new(plot, &view);
        assert!((m.y(view.lo()) - 400.0).abs() < 1e-3);
        assert!((m.y(view.hi()) - 0.0).abs() < 1e-3);
        assert!(m.y(view.centre + 100.0) < m.y(view.centre));
    }

    // --- Labs: correction ribbon and heat line ---------------------------

    const SR: f64 = 48_000.0;
    const HOP: u64 = 64;

    /// About 5.9 s of telemetry: four notes sung `detune` cents off and
    /// fully corrected onto the target (plus `transpose_st`), with unvoiced
    /// breaths between them. Returns the history and the newest position.
    fn sung(transpose_st: f32, notes: &[(f32, f32)]) -> (PitchHistory, f64) {
        sung_with(transpose_st, notes, |_| 0.0)
    }

    /// Like [`sung`], with `wobble(seconds into the note)` cents added to
    /// the sung pitch only (vibrato the plugin flattens).
    fn sung_with(
        transpose_st: f32,
        notes: &[(f32, f32)],
        wobble: impl Fn(f32) -> f32,
    ) -> (PitchHistory, f64) {
        let mut h = PitchHistory::default();
        h.ensure_allocated();
        h.set_transpose(transpose_st);
        let mut pos = 0u64;
        for &(target, detune) in notes {
            for k in 0..1000 {
                let sung = target + detune + wobble(k as f32 * HOP as f32 / SR as f32);
                h.push(PitchFrame {
                    sample_pos: pos,
                    detected: sung,
                    centre: target + detune,
                    output: target + 100.0 * transpose_st,
                    target,
                    confidence: 0.9,
                    flags: frame_flags::VOICED,
                    ..PitchFrame::default()
                });
                pos += HOP;
            }
            for _ in 0..100 {
                h.push(PitchFrame {
                    sample_pos: pos,
                    ..PitchFrame::default()
                });
                pos += HOP;
            }
        }
        (h, pos as f64)
    }

    const NOTES: [(f32, f32); 4] = [
        (6000.0, 40.0),
        (6200.0, -8.0),
        (6400.0, 120.0),
        (6500.0, 0.0),
    ];

    fn labs(correction_ribbon: bool, heat_line: bool) -> LabsSettings {
        LabsSettings {
            correction_ribbon,
            heat_line,
        }
    }

    /// One graph frame through real egui; returns egui's output.
    fn frame(
        ctx: &egui::Context,
        history: &PitchHistory,
        now: f64,
        labs: LabsSettings,
        scratch: &mut RollScratch,
    ) -> egui::FullOutput {
        let mut view = GraphView::default();
        view.pan(6250.0 - view.centre);
        let roll = RollContext {
            note_states: [NoteState::On; 12],
            sample_rate: SR,
            now_pos: now,
            hop: HOP as f64,
            highlight_note: None,
            labs,
        };
        let size = vec2(900.0, 400.0);
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, size)),
            ..Default::default()
        };
        let mut out = ctx.run_ui(input, |ui| {
            let rect = Rect::from_min_size(Pos2::ZERO, size);
            draw_pitch_roll(ui, rect, history, &mut view, &roll, scratch);
        });
        out.textures_delta.clear();
        out
    }

    /// The ribbon's columns: (output vertex, free-edge vertex, fringe).
    fn ribbon_columns(scratch: &RollScratch) -> Vec<[egui::epaint::Vertex; 3]> {
        scratch.ribbon.vertices.as_chunks::<3>().0.to_vec()
    }

    #[test]
    fn ribbon_is_one_strip_per_voiced_run_coloured_by_strength() {
        let (h, now) = sung(0.0, &NOTES);
        let ctx = egui::Context::default();
        let mut scratch = RollScratch::default();
        let _ = frame(&ctx, &h, now, labs(true, false), &mut scratch);
        let mesh = &scratch.ribbon;
        assert!(mesh.is_valid());
        let columns = ribbon_columns(&scratch);
        assert_eq!(columns.len() * 3, mesh.vertices.len());
        assert!(
            columns.len() > 400,
            "about one per pixel: {}",
            columns.len()
        );
        // Four notes = four strips: neighbouring columns are joined by two
        // triangles plus two for the anti-aliasing fringe (the correction
        // keeps its sign within each note).
        assert_eq!(mesh.indices.len(), 12 * (columns.len() - NOTES.len()));
        // The 120 c note is the most opaque (clamped at 100+ c), the in-tune
        // one the least; the strip is ordered like the notes.
        let (lo, hi) = (ribbon_color(0.0).a(), ribbon_color(1.0).a());
        let at = |c: f32| ribbon_color(ramp_pos(c).unwrap()).a();
        let mut seen: Vec<u8> = columns.iter().map(|[a, _, _]| a.color.a()).collect();
        seen.dedup();
        assert_eq!(seen, vec![at(40.0), at(8.0), hi, lo], "one colour per note");
        for [out, free, fringe] in &columns {
            assert_eq!(out.color, free.color);
            assert_eq!(fringe.color, Color32::TRANSPARENT);
            assert!((fringe.pos.y - free.pos.y).abs() <= 1.0 + 1e-3);
            // Spans output to input: zero height only in tune.
            assert!((out.pos.y - free.pos.y).abs() > 0.5 || out.color.a() == lo);
        }
    }

    /// Transpose is not correction: an octave-up output with the same
    /// tuning error gives exactly the same ribbon, hung off the (moved)
    /// output line: same colours, same shape, just 1200 c higher. It never
    /// floods the space between the input and the transposed output.
    #[test]
    fn ribbon_ignores_transpose_in_colour_and_shape() {
        let ctx = egui::Context::default();
        let mut plain = RollScratch::default();
        let (h, now) = sung(0.0, &NOTES);
        let _ = frame(&ctx, &h, now, labs(true, false), &mut plain);
        let mut up = RollScratch::default();
        let (h12, now12) = sung(12.0, &NOTES);
        let _ = frame(&ctx, &h12, now12, labs(true, false), &mut up);
        let (a, b) = (ribbon_columns(&plain), ribbon_columns(&up));
        assert_eq!(a.len(), b.len());
        // 400 px tall plot showing 2400 c (the default span).
        let octave_px = 1200.0 * 400.0 / GraphView::default().span;
        for (pa, pb) in a.iter().zip(&b) {
            for (va, vb) in pa.iter().zip(pb) {
                assert_eq!(va.color, vb.color);
                assert!((va.pos.x - vb.pos.x).abs() < 1e-3);
                let dy = va.pos.y - vb.pos.y;
                assert!((dy - octave_px).abs() < 0.05, "{dy} vs {octave_px}");
            }
            // Relative to its own output vertex, the ribbon is identical.
            let rel = |p: &[egui::epaint::Vertex; 3]| p[1].pos.y - p[0].pos.y;
            assert!((rel(pa) - rel(pb)).abs() < 0.05);
        }
    }

    /// Each frame is judged against the Transpose it arrived under: turning
    /// Transpose up mid-phrase neither re-colours nor stretches the ribbon
    /// of the notes already on screen.
    #[test]
    fn a_transpose_change_does_not_recolour_the_history() {
        let mut h = PitchHistory::default();
        h.ensure_allocated();
        let mut pos = 0u64;
        for (target, transpose_st) in [(6200.0, 0.0), (6300.0, 2.0)] {
            h.set_transpose(transpose_st);
            for k in 0..1100 {
                let voiced = k < 1000;
                h.push(PitchFrame {
                    sample_pos: pos,
                    detected: if voiced { target + 40.0 } else { 0.0 },
                    output: if voiced {
                        target + 100.0 * transpose_st
                    } else {
                        0.0
                    },
                    target,
                    flags: if voiced { frame_flags::VOICED } else { 0 },
                    ..PitchFrame::default()
                });
                pos += HOP;
            }
        }
        let ctx = egui::Context::default();
        let mut scratch = RollScratch::default();
        let _ = frame(&ctx, &h, pos as f64, labs(true, false), &mut scratch);
        let columns = ribbon_columns(&scratch);
        assert!(columns.len() > 200);
        let at40 = ribbon_color(ramp_pos(40.0).unwrap());
        let px40 = 40.0 * 400.0 / GraphView::default().span;
        for [out, free, _] in &columns {
            assert_eq!(out.color, at40);
            assert!(((free.pos.y - out.pos.y) + px40).abs() < 0.05);
        }
    }

    #[test]
    fn experiments_off_draw_no_ribbon() {
        let (h, now) = sung(0.0, &NOTES);
        let ctx = egui::Context::default();
        let mut scratch = RollScratch::default();
        let out = frame(&ctx, &h, now, labs(false, false), &mut scratch);
        assert!(scratch.ribbon.vertices.is_empty());
        assert!(scratch.heat.vertices.is_empty());
        let meshes = out
            .shapes
            .iter()
            .filter(|s| matches!(s.shape, Shape::Mesh(_)))
            .count();
        assert_eq!(meshes, 0, "no ribbon, no heat line, no legend key");
    }

    /// Output polylines (width 2.4 or the 6 px glow) and their colours.
    fn output_paths(out: &egui::FullOutput) -> Vec<(f32, Color32)> {
        out.shapes
            .iter()
            .filter_map(|s| match &s.shape {
                Shape::Path(p) if p.stroke.width >= 2.0 => match p.stroke.color {
                    egui::epaint::ColorMode::Solid(c) => Some((p.stroke.width, c)),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    /// Opaque heat-line vertex colours, left to right, deduplicated.
    fn heat_colours(scratch: &RollScratch) -> Vec<Color32> {
        let mut v: Vec<(f32, Color32)> = scratch
            .heat
            .vertices
            .iter()
            .filter(|v| v.color.a() == 255)
            .map(|v| (v.pos.x, v.color))
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut c: Vec<Color32> = v.into_iter().map(|(_, c)| c).collect();
        c.dedup();
        c
    }

    /// With the ribbon on, the output line has no glow (it would hide the
    /// small corrections); without it, the glow is back.
    #[test]
    fn output_glow_only_without_the_ribbon() {
        let (h, now) = sung(0.0, &NOTES);
        let ctx = egui::Context::default();
        let mut scratch = RollScratch::default();
        let with = output_paths(&frame(&ctx, &h, now, labs(true, false), &mut scratch));
        assert_eq!(with, vec![(2.4, ACCENT); 4]);
        let without = output_paths(&frame(&ctx, &h, now, labs(false, false), &mut scratch));
        let glow = OUTPUT_GLOW.stroke(ACCENT);
        assert_eq!(without.len(), 8);
        assert_eq!(
            without
                .iter()
                .filter(|p| **p == (glow.width, glow.color))
                .count(),
            4
        );
    }

    /// The heat line is one recoloured mesh: no polylines at all, each
    /// note in its own step colour, the colour blending over one segment.
    #[test]
    fn heat_line_is_one_mesh_coloured_by_step() {
        let (h, now) = sung(0.0, &NOTES);
        let ctx = egui::Context::default();
        let mut scratch = RollScratch::default();
        let out = frame(&ctx, &h, now, labs(true, true), &mut scratch);
        assert!(output_paths(&out).is_empty(), "no output polylines");
        assert!(scratch.heat.is_valid() && !scratch.heat.indices.is_empty());
        let step = |c: f32| heat_color(heat_step(c));
        assert_eq!(
            heat_colours(&scratch),
            vec![step(40.0), step(8.0), step(120.0), step(0.0)]
        );
        // Without the ribbon the glow is tessellated into the same mesh,
        // under the line.
        let glowless = scratch.heat.indices.len();
        let _ = frame(&ctx, &h, now, labs(false, true), &mut scratch);
        assert!(scratch.heat.indices.len() > glowless);

        // A correction that grows through one note: every step in order,
        // lightness rising left to right.
        let mut ramp = PitchHistory::default();
        ramp.ensure_allocated();
        let n = 4000u64;
        for k in 0..n {
            let c = 100.0 * k as f32 / n as f32;
            ramp.push(PitchFrame {
                sample_pos: k * HOP,
                detected: 6200.0 - c,
                output: 6200.0,
                target: 6200.0,
                flags: frame_flags::VOICED,
                ..PitchFrame::default()
            });
        }
        let _ = frame(
            &ctx,
            &ramp,
            (n * HOP) as f64,
            labs(false, true),
            &mut scratch,
        );
        // Opaque vertices carry exact step colours or a blend of two
        // neighbouring ones; the exact ones appear in order.
        let exact: Vec<Color32> = heat_colours(&scratch)
            .into_iter()
            .filter(|c| correction::HEAT_COLORS.contains(c))
            .collect();
        assert_eq!(exact, correction::HEAT_COLORS.to_vec());
        let lights: Vec<f32> = heat_colours(&scratch)
            .iter()
            .map(|c| correction::tests::oklab(*c)[0])
            .collect();
        assert!(lights.windows(2).all(|p| p[1] >= p[0]), "{lights:?}");
    }

    /// Flattened vibrato (hard tune): the correction swings between 0 and
    /// 60 c at 5.5 Hz. The heat line follows the strongest correction of
    /// each vibrato cycle, so the note reads as one strongly corrected
    /// stretch, not as a dashed line.
    #[test]
    fn heat_line_holds_through_flattened_vibrato() {
        let vib = |t: f32| 60.0 * (std::f32::consts::TAU * 5.5 * t).sin();
        let (h, now) = sung_with(0.0, &[(6200.0, 0.0), (6400.0, 0.0)], vib);
        let ctx = egui::Context::default();
        let mut scratch = RollScratch::default();
        let _ = frame(&ctx, &h, now, labs(true, true), &mut scratch);
        assert_eq!(heat_colours(&scratch), vec![heat_color(HEAT_STEPS - 1)]);
        // The ribbon's colour is smoothed only lightly: it still follows
        // the swing (strong at the peaks, faint at the crossings).
        let alphas: Vec<u8> = ribbon_columns(&scratch)
            .iter()
            .map(|[a, _, _]| a.color.a())
            .collect();
        let (min, max) = (alphas.iter().min(), alphas.iter().max());
        assert!(max > min);
    }

    /// Smoothing stays inside a voiced run and within its window.
    #[test]
    fn smoothing_is_local_to_each_run() {
        let f = PitchFrame {
            output: 6000.0,
            detected: 6000.0,
            flags: frame_flags::VOICED,
            ..PitchFrame::default()
        };
        let col = |x: f32, brk: bool, c: f32| Column::new(x, brk, &f, Some(c));
        // Run 1: a 60 c spike at x = 10 among 0 c columns; run 2 starts at 13.
        let mut cols: Vec<Column> = (0..13)
            .map(|x| col(x as f32, x == 0, if x == 10 { 60.0 } else { 0.0 }))
            .chain((13..20).map(|x| col(x as f32, x == 13, 0.0)))
            .collect();
        smooth_strengths(&mut cols, 4.0);
        let heat: Vec<u8> = cols.iter().map(|c| c.heat).collect();
        // ±4 px around the spike, but not across the break at 13.
        assert_eq!(
            heat,
            [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0]
        );
        // The ribbon averages ±2 px: 60 c over five columns = 12 c.
        assert_eq!(cols[9].ribbon_t, ramp_pos(12.0));
        assert_eq!(
            cols[12].ribbon_t,
            ramp_pos(20.0),
            "3 columns at the run end"
        );
        assert_eq!(cols[13].ribbon_t, ramp_pos(0.0));
        assert_eq!(cols[5].ribbon_t, ramp_pos(0.0));
        // A column without any strength in reach gets no ribbon colour.
        let mut lone = vec![Column::new(0.0, true, &f, None)];
        smooth_strengths(&mut lone, 4.0);
        assert_eq!((lone[0].ribbon_t, lone[0].heat), (None, 0));
    }

    /// The ribbon, heat-line and legend-key meshes are handed to egui as
    /// `Arc`s and refilled in place every frame: same buffer, no growth.
    #[test]
    fn drawing_reuses_the_mesh_buffers() {
        let (h, now) = sung(0.0, &NOTES);
        let ctx = egui::Context::default();
        let mut scratch = RollScratch::default();
        let both = labs(true, true);
        let tessellate = |out: egui::FullOutput| {
            let _ = ctx.tessellate(out.shapes, out.pixels_per_point);
        };
        tessellate(frame(&ctx, &h, now, both, &mut scratch));
        let ptrs = |s: &RollScratch| {
            (
                Arc::as_ptr(&s.ribbon),
                Arc::as_ptr(&s.heat),
                Arc::as_ptr(&s.key),
            )
        };
        let first_ptrs = ptrs(&scratch);
        let caps = |s: &RollScratch| {
            [
                s.ribbon.vertices.capacity(),
                s.ribbon.indices.capacity(),
                s.heat.vertices.capacity(),
                s.heat.indices.capacity(),
                s.key.vertices.capacity(),
                s.key.indices.capacity(),
                s.columns.capacity(),
                s.points.capacity(),
            ]
        };
        let first = caps(&scratch);
        assert!(first[0] > 0 && first[2] > 0 && first[4] > 0);
        assert_eq!(first[6], MAX_COLUMNS);
        // Let every buffer reach its high-water mark, then scroll on: the
        // graph shows a slightly different set of columns each frame.
        for k in 1..=60 {
            let t = now + k as f64 * SR / 60.0;
            tessellate(frame(&ctx, &h, t, both, &mut scratch));
            assert_eq!(ptrs(&scratch), first_ptrs, "frame {k}");
            assert!(Arc::strong_count(&scratch.ribbon) == 1);
            assert!(Arc::strong_count(&scratch.heat) == 1);
            let c = caps(&scratch);
            for (i, (now_cap, first_cap)) in c.iter().zip(first).enumerate() {
                assert!(*now_cap <= first_cap.max(64) * 2, "buffer {i}, frame {k}");
            }
            assert_eq!(c[4..], first[4..], "frame {k}");
        }

        // A renderer that keeps the previous frame alive: the mesh is
        // replaced by a buffer of the same capacity, never shared mutably.
        let kept = frame(&ctx, &h, now, both, &mut scratch);
        let _next = frame(&ctx, &h, now, both, &mut scratch);
        assert_ne!(Arc::as_ptr(&scratch.ribbon), first_ptrs.0);
        assert_ne!(Arc::as_ptr(&scratch.heat), first_ptrs.1);
        drop(kept);
    }

    #[test]
    fn column_strength_is_the_mean_magnitude() {
        let f = PitchFrame {
            output: 6000.0,
            detected: 6000.0,
            flags: frame_flags::VOICED,
            ..PitchFrame::default()
        };
        let mut c = Column::new(0.0, false, &f, Some(10.0));
        c.add(&f, Some(-30.0));
        c.add(&f, None);
        assert_eq!(c.strength(), Some(20.0));
        assert_eq!(Column::new(0.0, false, &f, None).strength(), None);
    }

    #[test]
    fn runs_split_at_voicing_breaks() {
        let f = PitchFrame::default();
        let cols: Vec<Column> = [true, false, false, true, true, false]
            .iter()
            .enumerate()
            .map(|(i, &b)| Column::new(i as f32, b, &f, None))
            .collect();
        let r: Vec<_> = runs(&cols).collect();
        assert_eq!(r, vec![0..3, 3..4, 4..6]);
        assert_eq!(runs(&[]).count(), 0);
    }
}
