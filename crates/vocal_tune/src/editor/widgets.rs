//! Hand-painted controls bound to nice-plug parameters.
//!
//! Every control talks to the host through the `ParamSetter` gesture protocol
//! (begin → set → end) so automation recording and undo work. Parameter
//! values set from the GUI only change once the wrapper flushes them, so drags
//! keep their own start value and accumulated distance in egui's temporary
//! memory instead of re-reading the (possibly not yet updated) parameter
//! every frame.
//!
//! Controls are placed at explicit rectangles: the editor has a fixed size
//! and a designed layout, so there is no need for egui's auto-layout here.

use std::f32::consts::PI;

use egui::{
    Align2, Color32, CornerRadius, Id, Pos2, Rect, Response, Sense, Shape, Stroke, StrokeKind, Ui,
    pos2, vec2,
};
use nice_plug::prelude::*;

use super::theme::{self, ACCENT, BORDER, PANEL_HI, PANEL_HOVER, TEXT, TEXT_DIM, TEXT_FAINT};

/// Normalized change per pixel of vertical drag (≈ 200 px for the full range).
const DRAG_SPEED: f32 = 0.005;
/// Shift held: fine adjustment.
const FINE_DRAG_SPEED: f32 = 0.0006;

/// Drag state kept in egui memory while a gesture is in progress:
/// (normalized value at drag start, accumulated normalized change).
#[derive(Clone, Copy, Default)]
struct DragState {
    start: f32,
    accumulated: f32,
}

/// Shared drag/reset/scroll handling for continuous parameters. `pixels` is
/// the drag distance this frame along the control's axis (positive = up/right).
fn drive_float(ui: &Ui, resp: &Response, setter: &ParamSetter, param: &FloatParam, pixels: f32) {
    let id = resp.id.with("drag");
    if resp.drag_started() {
        setter.begin_set_parameter(param);
        let start = param.unmodulated_normalized_value();
        ui.data_mut(|d| {
            d.insert_temp(
                id,
                DragState {
                    start,
                    accumulated: 0.0,
                },
            )
        });
    }
    if resp.dragged() {
        let fine = ui.input(|i| i.modifiers.shift);
        let speed = if fine { FINE_DRAG_SPEED } else { DRAG_SPEED };
        let mut st: DragState = ui.data(|d| d.get_temp(id)).unwrap_or_default();
        // Keep start + accumulated inside [0, 1] so reversing direction at an
        // end stop responds immediately.
        st.accumulated = (st.accumulated + pixels * speed).clamp(-st.start, 1.0 - st.start);
        ui.data_mut(|d| d.insert_temp(id, st));
        setter.set_parameter_normalized(param, st.start + st.accumulated);
    }
    if resp.drag_stopped() {
        setter.end_set_parameter(param);
        ui.data_mut(|d| d.remove::<DragState>(id));
    }
    if resp.double_clicked() {
        setter.begin_set_parameter(param);
        setter.set_parameter_normalized(param, param.default_normalized_value());
        setter.end_set_parameter(param);
    }
    if resp.hovered() && !resp.dragged() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            let norm = (param.unmodulated_normalized_value() + scroll * 0.0015).clamp(0.0, 1.0);
            setter.begin_set_parameter(param);
            setter.set_parameter_normalized(param, norm);
            setter.end_set_parameter(param);
        }
    }
}

/// The value a control should display: the drag target while dragging (the
/// parameter itself may lag by one host block), otherwise the parameter.
fn shown_normalized(ui: &Ui, resp: &Response, param: &FloatParam) -> f32 {
    let id = resp.id.with("drag");
    match ui.data(|d| d.get_temp::<DragState>(id)) {
        Some(st) if resp.dragged() => st.start + st.accumulated,
        _ => param.unmodulated_normalized_value(),
    }
}

/// Appearance options for [`knob`].
pub struct KnobSpec<'a> {
    pub label: &'a str,
    /// Normalized position the value arc starts from (0 for unipolar, the
    /// neutral value for bipolar controls such as Detune).
    pub origin: f32,
    /// Drawn dimmed (e.g. Transition while linked to Retune).
    pub dimmed: bool,
    /// Replaces the parameter's own value text.
    pub value_override: Option<&'a str>,
    pub tooltip: &'a str,
}

/// A 270° rotary knob with label above and value below, filling `rect`.
/// Drag vertically (Shift = fine), scroll, double-click to reset.
pub fn knob(
    ui: &mut Ui,
    rect: Rect,
    setter: &ParamSetter,
    param: &FloatParam,
    spec: &KnobSpec,
) -> Response {
    let diameter = 44.0;
    let centre = pos2(rect.center().x, rect.top() + 22.0 + diameter * 0.5);
    let hit = Rect::from_center_size(centre, vec2(diameter + 10.0, diameter + 10.0));
    let resp = ui.interact(
        hit,
        Id::new(("vt-knob", param.name())),
        Sense::click_and_drag(),
    );
    drive_float(ui, &resp, setter, param, -resp.drag_delta().y);
    let norm = shown_normalized(ui, &resp, param);

    let painter = ui.painter();
    let label_col = if spec.dimmed { TEXT_FAINT } else { TEXT_DIM };
    painter.text(
        pos2(rect.center().x, rect.top() + 8.0),
        Align2::CENTER_CENTER,
        spec.label,
        theme::font(11.0),
        label_col,
    );

    let radius = diameter * 0.5;
    let start = PI * 0.75;
    let sweep = PI * 1.5;
    let accent = if spec.dimmed {
        theme::alpha(ACCENT, 0.35)
    } else {
        ACCENT
    };
    let hovered = resp.hovered() || resp.dragged();

    // Body with a subtle rim.
    painter.circle_filled(
        centre,
        radius - 5.0,
        if hovered { PANEL_HOVER } else { PANEL_HI },
    );
    painter.circle_stroke(centre, radius - 5.0, Stroke::new(1.0, BORDER));

    // Track, then the value arc from the origin to the value.
    let arc = |from: f32, to: f32, color: Color32, width: f32| {
        let (a, b) = if from <= to { (from, to) } else { (to, from) };
        let steps = ((b - a) * 48.0).ceil().max(1.0) as usize;
        let pts: Vec<Pos2> = (0..=steps)
            .map(|i| {
                let t = a + (b - a) * i as f32 / steps as f32;
                let ang = start + t * sweep;
                centre + vec2(ang.cos(), ang.sin()) * (radius - 1.0)
            })
            .collect();
        painter.add(Shape::line(pts, Stroke::new(width, color)));
    };
    arc(0.0, 1.0, theme::BORDER_HI, 3.0);
    if (norm - spec.origin).abs() > 1e-3 {
        arc(spec.origin, norm, accent, 3.0);
    }

    // Pointer.
    let ang = start + norm * sweep;
    let dir = vec2(ang.cos(), ang.sin());
    painter.line_segment(
        [
            centre + dir * (radius * 0.18),
            centre + dir * (radius - 7.0),
        ],
        Stroke::new(2.0, if spec.dimmed { TEXT_DIM } else { TEXT }),
    );

    let text = match spec.value_override {
        Some(t) => t.to_string(),
        None => param.normalized_value_to_string(norm, true),
    };
    painter.text(
        pos2(rect.center().x, centre.y + radius + 9.0),
        Align2::CENTER_CENTER,
        text,
        theme::mono(11.5),
        if spec.dimmed {
            TEXT_DIM
        } else if hovered {
            ACCENT
        } else {
            TEXT
        },
    );
    if spec.tooltip.is_empty() {
        resp
    } else {
        resp.on_hover_text(spec.tooltip)
    }
}

/// A compact horizontal slider: label on the left, bar, value on the right.
pub fn hslider(
    ui: &mut Ui,
    rect: Rect,
    setter: &ParamSetter,
    param: &FloatParam,
    label: &str,
) -> Response {
    let label_w = 62.0;
    let value_w = 58.0;
    let bar = Rect::from_min_max(
        pos2(rect.left() + label_w, rect.center().y - 3.0),
        pos2(rect.right() - value_w, rect.center().y + 3.0),
    );
    let hit = Rect::from_min_max(
        pos2(bar.left() - 4.0, rect.top()),
        pos2(bar.right() + 4.0, rect.bottom()),
    );
    let resp = ui.interact(
        hit,
        Id::new(("vt-slider", param.name())),
        Sense::click_and_drag(),
    );
    // Horizontal drags map one bar width to the full range for direct feel.
    let px = resp.drag_delta().x / (bar.width().max(1.0) * DRAG_SPEED);
    drive_float(ui, &resp, setter, param, px);
    let norm = shown_normalized(ui, &resp, param);
    let hovered = resp.hovered() || resp.dragged();

    let painter = ui.painter();
    painter.text(
        pos2(rect.left(), rect.center().y),
        Align2::LEFT_CENTER,
        label,
        theme::font(11.0),
        TEXT_DIM,
    );
    painter.rect_filled(bar, CornerRadius::same(3), PANEL_HI);
    let fill = Rect::from_min_max(bar.min, pos2(bar.left() + bar.width() * norm, bar.bottom()));
    painter.rect_filled(
        fill,
        CornerRadius::same(3),
        if hovered {
            ACCENT
        } else {
            theme::alpha(ACCENT, 0.75)
        },
    );
    painter.circle_filled(
        pos2(fill.right(), bar.center().y),
        if hovered { 5.0 } else { 4.0 },
        TEXT,
    );
    painter.text(
        pos2(rect.right(), rect.center().y),
        Align2::RIGHT_CENTER,
        param.normalized_value_to_string(norm, true),
        theme::mono(11.0),
        if hovered { ACCENT } else { TEXT },
    );
    resp
}

/// A pill toggle with an LED dot.
pub fn toggle(
    ui: &mut Ui,
    rect: Rect,
    setter: &ParamSetter,
    param: &BoolParam,
    label: &str,
    tooltip: &str,
) -> Response {
    let resp = ui.interact(rect, Id::new(("vt-toggle", param.name())), Sense::click());
    let on = param.value();
    if resp.clicked() {
        setter.begin_set_parameter(param);
        setter.set_parameter(param, !on);
        setter.end_set_parameter(param);
    }
    let hovered = resp.hovered();
    let painter = ui.painter();
    let fill = if on {
        theme::alpha(ACCENT, 0.14)
    } else if hovered {
        PANEL_HOVER
    } else {
        PANEL_HI
    };
    let stroke = if on {
        theme::alpha(ACCENT, 0.7)
    } else {
        BORDER
    };
    painter.rect_filled(rect, CornerRadius::same(255), fill);
    painter.rect_stroke(
        rect,
        CornerRadius::same(255),
        Stroke::new(1.0, stroke),
        StrokeKind::Inside,
    );
    let dot = pos2(rect.left() + 10.0, rect.center().y);
    painter.circle_filled(dot, 3.0, if on { ACCENT } else { TEXT_FAINT });
    painter.text(
        pos2(rect.left() + 19.0, rect.center().y),
        Align2::LEFT_CENTER,
        label,
        theme::font(11.0),
        if on { TEXT } else { TEXT_DIM },
    );
    if tooltip.is_empty() {
        resp
    } else {
        resp.on_hover_text(tooltip)
    }
}

/// A segmented selector for an enum parameter; one click selects.
/// `labels` overrides the variant names (must have the same length).
pub fn segmented<T: Enum + PartialEq + Copy + 'static>(
    ui: &mut Ui,
    rect: Rect,
    setter: &ParamSetter,
    param: &EnumParam<T>,
    labels: Option<&[&str]>,
    tooltip: &str,
) -> Response {
    let names = labels.unwrap_or(T::variants());
    let n = names.len().max(1);
    let current = param.value().to_index();
    let seg_w = rect.width() / n as f32;
    let id = Id::new(("vt-seg", param.name()));
    let mut union = ui.interact(rect, id, Sense::hover());
    let painter = ui.painter().clone();
    painter.rect_filled(rect, CornerRadius::same(theme::RADIUS), PANEL_HI);
    for (i, name) in names.iter().enumerate() {
        let seg = Rect::from_min_size(
            pos2(rect.left() + seg_w * i as f32, rect.top()),
            vec2(seg_w, rect.height()),
        );
        let resp = ui.interact(seg.shrink(1.0), id.with(i), Sense::click());
        let selected = i == current;
        if resp.clicked() && !selected {
            setter.begin_set_parameter(param);
            setter.set_parameter(param, T::from_index(i));
            setter.end_set_parameter(param);
        }
        if selected {
            painter.rect_filled(
                seg.shrink(2.0),
                CornerRadius::same(theme::RADIUS - 1),
                theme::alpha(ACCENT, 0.18),
            );
            painter.rect_stroke(
                seg.shrink(2.0),
                CornerRadius::same(theme::RADIUS - 1),
                Stroke::new(1.0, theme::alpha(ACCENT, 0.8)),
                StrokeKind::Inside,
            );
        } else if resp.hovered() {
            painter.rect_filled(
                seg.shrink(2.0),
                CornerRadius::same(theme::RADIUS - 1),
                PANEL_HOVER,
            );
        }
        painter.text(
            seg.center(),
            Align2::CENTER_CENTER,
            *name,
            theme::font(11.5),
            if selected { TEXT } else { TEXT_DIM },
        );
        union = union.union(resp);
    }
    painter.rect_stroke(
        rect,
        CornerRadius::same(theme::RADIUS),
        Stroke::new(1.0, BORDER),
        StrokeKind::Inside,
    );
    if tooltip.is_empty() {
        union
    } else {
        union.on_hover_text(tooltip)
    }
}

/// A drop-down for enum parameters with many variants (key, scale, range).
pub fn combo<T: Enum + PartialEq + Copy + 'static>(
    ui: &mut Ui,
    rect: Rect,
    setter: &ParamSetter,
    param: &EnumParam<T>,
) {
    let current = param.value().to_index();
    let names = T::variants();
    let selected_text = names.get(current).copied().unwrap_or("");
    ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
        egui::ComboBox::from_id_salt(("vt-combo", param.name()))
            .width(rect.width())
            .height(320.0)
            .selected_text(selected_text)
            .show_ui(ui, |ui| {
                for (i, name) in names.iter().enumerate() {
                    if ui.selectable_label(i == current, *name).clicked() && i != current {
                        setter.begin_set_parameter(param);
                        setter.set_parameter(param, T::from_index(i));
                        setter.end_set_parameter(param);
                    }
                }
            });
    });
}

/// Small upper-case caption above a control.
pub fn caption(ui: &Ui, pos: Pos2, text: &str) {
    ui.painter().text(
        pos,
        Align2::LEFT_BOTTOM,
        text,
        theme::font(10.0),
        TEXT_FAINT,
    );
}

/// Section title with a hairline rule extending to `right`.
pub fn section_title(ui: &Ui, pos: Pos2, right: f32, text: &str) {
    let painter = ui.painter();
    let r = painter.text(pos, Align2::LEFT_CENTER, text, theme::font(10.5), TEXT_DIM);
    if right > r.right() + 8.0 {
        painter.hline(
            r.right() + 6.0..=right,
            pos.y,
            Stroke::new(1.0, theme::BORDER),
        );
    }
}

/// A rounded flat button. Returns the click response.
pub fn button(
    ui: &mut Ui,
    rect: Rect,
    id: Id,
    text: &str,
    active: bool,
    tooltip: &str,
) -> Response {
    let resp = ui.interact(rect, id, Sense::click());
    let painter = ui.painter();
    let (fill, stroke, col) = if active {
        (theme::alpha(ACCENT, 0.16), theme::alpha(ACCENT, 0.8), TEXT)
    } else if resp.hovered() {
        (PANEL_HOVER, theme::BORDER_HI, TEXT)
    } else {
        (PANEL_HI, BORDER, TEXT_DIM)
    };
    painter.rect_filled(rect, CornerRadius::same(theme::RADIUS), fill);
    painter.rect_stroke(
        rect,
        CornerRadius::same(theme::RADIUS),
        Stroke::new(1.0, stroke),
        StrokeKind::Inside,
    );
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        text,
        theme::font(11.5),
        col,
    );
    if tooltip.is_empty() {
        resp
    } else {
        resp.on_hover_text(tooltip)
    }
}

/// A compact draggable value box (drag up/right to increase, double-click
/// resets), with a thin position bar along its bottom edge.
pub fn value_box(ui: &mut Ui, rect: Rect, setter: &ParamSetter, param: &FloatParam) -> Response {
    let resp = ui.interact(
        rect,
        Id::new(("vt-valuebox", param.name())),
        Sense::click_and_drag(),
    );
    let d = resp.drag_delta();
    drive_float(ui, &resp, setter, param, d.x - d.y);
    let norm = shown_normalized(ui, &resp, param);
    let hovered = resp.hovered() || resp.dragged();
    let painter = ui.painter();
    painter.rect_filled(
        rect,
        CornerRadius::same(theme::RADIUS),
        if hovered { PANEL_HOVER } else { PANEL_HI },
    );
    painter.rect_stroke(
        rect,
        CornerRadius::same(theme::RADIUS),
        Stroke::new(1.0, BORDER),
        StrokeKind::Inside,
    );
    let bar = Rect::from_min_max(
        pos2(rect.left() + 4.0, rect.bottom() - 3.0),
        pos2(
            rect.left() + 4.0 + (rect.width() - 8.0) * norm,
            rect.bottom() - 2.0,
        ),
    );
    painter.rect_filled(bar, CornerRadius::ZERO, ACCENT);
    painter.text(
        rect.center() - vec2(0.0, 1.0),
        Align2::CENTER_CENTER,
        param.normalized_value_to_string(norm, true),
        theme::mono(11.5),
        if hovered { ACCENT } else { TEXT },
    );
    resp.on_hover_text("Drag to change, double-click to reset. Changing it restarts the plugin.")
}
