//! The scrolling piano-roll pitch graph (R§8.23).
//!
//! Drawing order, back to front: piano-roll lanes (scale notes shaded by
//! state, labelled in the left gutter), time grid, target-note bars, the
//! vibrato band, the detected input (thin, alpha = confidence, broken where
//! unvoiced), the vibrato-free centre, and the corrected output (thick,
//! accent colour). Time runs right to left from the display clock's "now", in
//! output samples, so the graph lines up with what is heard.
//!
//! The history holds ~5-6 frames per pixel column, so the frames in view are
//! first folded into one [`Column`] aggregate per pixel (min/max/first of the
//! output, mean of the input and centre, max confidence, ...) in a reusable,
//! preallocated buffer; every trace is then drawn from the columns. egui
//! shapes own their point lists, so each visible polyline run costs one small
//! `Vec` per frame; nothing else is allocated per frame.
//!
//! All drawing goes through [`draw_pitch_roll`], the thin seam R§9.7 asks
//! for, so a future GUI backend change stays contained.

use egui::{
    Align2, Color32, CornerRadius, Id, Mesh, Pos2, Rect, Response, Sense, Shape, Stroke,
    StrokeKind, Ui, pos2, vec2,
};
use vt_core::{NoteState, PitchFrame, frame_flags};

use super::history::{GraphView, PitchHistory};
use super::theme::{self, ACCENT, BYPASS, CENTRE, DETECTED, MIDI, TEXT, TEXT_DIM, TEXT_FAINT};
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
}

impl Column {
    fn new(x: f32, break_before: bool, f: &PitchFrame) -> Self {
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
        }
    }

    fn add(&mut self, f: &PitchFrame) {
        self.n += 1;
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
}

/// Reusable buffers for the graph (allocated once).
pub struct RollScratch {
    columns: Vec<Column>,
    points: Vec<Pos2>,
}

impl Default for RollScratch {
    fn default() -> Self {
        Self {
            columns: Vec::with_capacity(4096),
            points: Vec::with_capacity(4096),
        }
    }
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
    let resp = ui.interact(plot, Id::new("vt-pitch-graph"), Sense::click_and_drag());
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
    for f in history.iter_range(first, history.len()) {
        if !is_voiced(f) {
            broke = true;
            continue;
        }
        if prev_pos.is_some_and(|p| f.sample_pos.saturating_sub(p) > gap_limit) {
            broke = true;
        }
        prev_pos = Some(f.sample_pos);
        let x = x_of(f.sample_pos as f64).floor();
        let room = scratch.columns.len() < scratch.columns.capacity();
        match scratch.columns.last_mut() {
            Some(c) if !broke && c.x == x => c.add(f),
            // Never grow past the preallocated capacity.
            _ if room => scratch.columns.push(Column::new(x, broke, f)),
            _ => {}
        }
        broke = false;
    }

    let cols = &scratch.columns;
    draw_target_bars(&painter, cols, ymap, lane_h);
    draw_vibrato_band(&painter, cols, ymap);
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
    // Output: a soft glow under a solid accent line.
    draw_output(
        &painter,
        cols,
        points,
        ymap,
        Stroke::new(6.0, theme::alpha(ACCENT, 0.12)),
    );
    draw_output(&painter, cols, points, ymap, Stroke::new(2.4, ACCENT));

    draw_now_marker(&painter, plot, history, ctx, ymap, &x_of);
    if cols.is_empty() {
        painter.text(
            plot.center(),
            Align2::CENTER_CENTER,
            "No pitch detected: sing into the input to see the trace",
            theme::font(13.0),
            TEXT_FAINT,
        );
    }
    draw_legend(&painter, plot);
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

fn draw_target_bars(painter: &egui::Painter, cols: &[Column], ymap: YMap, lane_h: f32) {
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
        painter.rect_filled(bar, r, theme::alpha(base, 0.13));
        painter.rect_stroke(
            bar,
            r,
            Stroke::new(1.0, theme::alpha(base, 0.42)),
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

/// The output trace keeps each column's min and max (in time order) so
/// hard-tune steps stay vertical and sharp instead of being averaged away.
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
    flush_run(painter, points, stroke);
}

fn draw_now_marker(
    painter: &egui::Painter,
    plot: Rect,
    history: &PitchHistory,
    ctx: &RollContext,
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
        if plot.contains(p) {
            painter.circle_filled(p, 6.0, theme::alpha(ACCENT, 0.2));
            painter.circle_filled(p, 3.5, ACCENT);
        }
    }
}

fn draw_legend(painter: &egui::Painter, plot: Rect) {
    let items: [(&str, Color32, f32); 4] = [
        ("Output", ACCENT, 2.4),
        ("Input", theme::alpha(DETECTED, 0.75), 1.4),
        ("Centre", CENTRE, 1.3),
        ("Target", theme::alpha(ACCENT, 0.45), 6.0),
    ];
    let mut x = plot.right() - 10.0;
    let y = plot.top() + 12.0;
    for (label, color, width) in items.iter().rev() {
        let r = painter.text(
            pos2(x, y),
            Align2::RIGHT_CENTER,
            *label,
            theme::font(10.5),
            TEXT_DIM,
        );
        let x_line = r.left() - 6.0;
        painter.line_segment(
            [pos2(x_line - 14.0, y), pos2(x_line, y)],
            Stroke::new(*width, *color),
        );
        x = x_line - 26.0;
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
        let mut c = Column::new(0.0, false, &f(6000.0));
        c.add(&f(6100.0));
        c.add(&f(5900.0));
        assert_eq!((c.out_min, c.out_max), (5900.0, 6100.0));
        assert!(!c.min_first, "max came before the new min");
        let mut up = Column::new(0.0, false, &f(5900.0));
        up.add(&f(6100.0));
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
}
