//! The header's LABS button and its menu of experiments ([`crate::labs`]).
//!
//! The menu is a small foreground panel that drops from the header's lower
//! edge, right-aligned to the button: one row per experiment (switch, name,
//! one-line description) and a footer that says what Labs is. A click on a
//! row flips that experiment; Escape or a click anywhere else closes the
//! menu. Nothing here touches host parameters.

use egui::epaint::Shadow;
use egui::{
    Align2, Area, Color32, CornerRadius, Id, Key, Order, Rect, Response, Sense, Shape, Stroke,
    StrokeKind, Ui, pos2, vec2,
};

use super::theme::{
    self, ACCENT, BORDER, BORDER_HI, PANEL, PANEL_HI, PANEL_HOVER, TEXT, TEXT_DIM, TEXT_FAINT,
};
use crate::labs::{EXPERIMENTS, LABS_FOOTER};
use crate::params::VocalTuneParams;

/// Menu width.
pub const MENU_W: f32 = 336.0;
/// Gap between the header's lower edge and the menu.
pub const MENU_GAP: f32 = 4.0;
const PAD: f32 = 12.0;
/// Title row and experiment row heights.
pub const TITLE_H: f32 = 34.0;
pub const ROW_H: f32 = 44.0;
const FOOTER_LINE_H: f32 = 14.0;

/// A small downward caret (painted: no glyph dependency), centred on `c`.
fn caret(p: &egui::Painter, c: egui::Pos2, color: Color32) {
    p.add(Shape::convex_polygon(
        vec![
            pos2(c.x - 3.5, c.y - 1.75),
            pos2(c.x + 3.5, c.y - 1.75),
            pos2(c.x, c.y + 2.25),
        ],
        color,
        Stroke::NONE,
    ));
}

/// The LABS button, a small menu button: "LABS" and a caret. Clicking it
/// toggles the menu (the caller keeps the open state).
pub fn button(ui: &mut Ui, rect: Rect, open: bool) -> Response {
    let resp = ui.interact(rect, Id::new("vt-labs-button"), Sense::click());
    let (fill, stroke, text) = if open {
        (theme::alpha(ACCENT, 0.16), theme::alpha(ACCENT, 0.8), TEXT)
    } else if resp.hovered() {
        (PANEL_HOVER, BORDER_HI, TEXT)
    } else {
        (PANEL_HI, BORDER_HI, TEXT_DIM)
    };
    let p = ui.painter();
    let r = CornerRadius::same(theme::RADIUS);
    p.rect_filled(rect, r, fill);
    p.rect_stroke(rect, r, Stroke::new(1.0, stroke), StrokeKind::Inside);
    let label = p.text(
        pos2(rect.left() + 8.0, rect.center().y),
        Align2::LEFT_CENTER,
        "LABS",
        theme::font(9.5),
        text,
    );
    caret(p, pos2(label.right() + 7.5, rect.center().y), text);
    resp.on_hover_text("Labs: experimental features to try before they join the main UI.")
}

/// A small on/off switch.
fn switch(p: &egui::Painter, rect: Rect, on: bool, hovered: bool) {
    let r = CornerRadius::same(255);
    let (fill, stroke) = if on {
        (theme::alpha(ACCENT, 0.3), theme::alpha(ACCENT, 0.85))
    } else if hovered {
        (PANEL_HOVER, BORDER_HI)
    } else {
        (PANEL_HI, BORDER_HI)
    };
    p.rect_filled(rect, r, fill);
    p.rect_stroke(rect, r, Stroke::new(1.0, stroke), StrokeKind::Inside);
    let knob_r = rect.height() * 0.5 - 3.0;
    let x = if on {
        rect.right() - knob_r - 3.0
    } else {
        rect.left() + knob_r + 3.0
    };
    p.circle_filled(
        pos2(x, rect.center().y),
        knob_r,
        if on { ACCENT } else { TEXT_FAINT },
    );
}

/// Top-left corner of the menu for the LABS button `anchor` in a header
/// whose lower edge is at `header_bottom`: just below the header (never
/// across it), right edge flush with the button's.
pub fn menu_origin(anchor: Rect, header_bottom: f32) -> egui::Pos2 {
    pos2(anchor.right() - MENU_W, header_bottom + MENU_GAP)
}

/// Draw the menu for the LABS button `anchor` (see [`menu_origin`]) and
/// apply clicks to the experiments in `params`. Returns whether the menu
/// stays open.
pub fn menu(ui: &Ui, anchor: Rect, header_bottom: f32, params: &VocalTuneParams) -> bool {
    let ctx = ui.ctx().clone();
    let labs = params.labs();
    let rows_h = ROW_H * EXPERIMENTS.len() as f32;
    let origin = menu_origin(anchor, header_bottom);
    let shown = Area::new(Id::new("vt-labs-menu"))
        .order(Order::Foreground)
        .movable(false)
        .fixed_pos(origin)
        .show(&ctx, |ui| {
            let p = ui.painter().clone();
            let footer_h = FOOTER_LINE_H * LABS_FOOTER.len() as f32;
            let height = TITLE_H + rows_h + 10.0 + footer_h + PAD - 2.0;
            let rect = egui::Rect::from_min_size(origin, vec2(MENU_W, height));
            ui.allocate_rect(rect, Sense::hover());

            let corner = CornerRadius::same(6);
            p.add(
                Shadow {
                    offset: [0, 6],
                    blur: 18,
                    spread: 0,
                    color: Color32::from_black_alpha(140),
                }
                .as_shape(rect, corner),
            );
            p.rect_filled(rect, corner, PANEL);
            p.rect_stroke(
                rect,
                corner,
                Stroke::new(1.0, BORDER_HI),
                StrokeKind::Inside,
            );

            // Title.
            let title_y = rect.top() + TITLE_H * 0.5 + 1.0;
            let t = p.text(
                pos2(rect.left() + PAD, title_y),
                Align2::LEFT_CENTER,
                "LABS",
                theme::font(11.0),
                TEXT,
            );
            p.text(
                pos2(t.right() + 8.0, title_y),
                Align2::LEFT_CENTER,
                "Experimental features",
                theme::font(10.5),
                TEXT_DIM,
            );
            p.hline(
                rect.x_range().shrink(1.0),
                rect.top() + TITLE_H,
                Stroke::new(1.0, BORDER),
            );

            // One row per experiment. The name always reads as a label; the
            // switch alone shows whether it is on.
            for (i, exp) in EXPERIMENTS.iter().enumerate() {
                let top = rect.top() + TITLE_H + ROW_H * i as f32;
                let row = Rect::from_min_size(pos2(rect.left(), top), vec2(MENU_W, ROW_H))
                    .shrink2(vec2(4.0, 2.0));
                let resp = ui.interact(row, Id::new(("vt-labs-row", i)), Sense::click());
                let on = (exp.get)(&labs);
                if resp.clicked() {
                    let mut next = params.labs();
                    (exp.set)(&mut next, !on);
                    params.set_labs(next);
                }
                let on = (exp.get)(&params.labs());
                if resp.hovered() {
                    p.rect_filled(row, CornerRadius::same(theme::RADIUS), PANEL_HI);
                }
                let name_y = top + 15.0;
                switch(
                    &p,
                    Rect::from_center_size(
                        pos2(rect.left() + PAD + 13.0, name_y),
                        vec2(26.0, 14.0),
                    ),
                    on,
                    resp.hovered(),
                );
                let text_x = rect.left() + PAD + 36.0;
                p.text(
                    pos2(text_x, name_y),
                    Align2::LEFT_CENTER,
                    exp.name,
                    theme::font(12.0),
                    TEXT,
                );
                p.text(
                    pos2(text_x, top + 31.0),
                    Align2::LEFT_CENTER,
                    exp.description,
                    theme::font(10.0),
                    TEXT_DIM,
                );
            }

            // Footer: one sentence per line.
            let foot_top = rect.top() + TITLE_H + rows_h + 4.0;
            p.hline(
                rect.x_range().shrink(1.0),
                foot_top,
                Stroke::new(1.0, BORDER),
            );
            for (i, line) in LABS_FOOTER.iter().enumerate() {
                p.text(
                    pos2(
                        rect.left() + PAD,
                        foot_top + 8.0 + FOOTER_LINE_H * (i as f32 + 0.5),
                    ),
                    Align2::LEFT_CENTER,
                    *line,
                    theme::font(9.5),
                    TEXT_DIM,
                );
            }
            rect
        });

    let menu_rect = shown.inner;
    let close = ctx.input(|i| {
        let outside_press = i.pointer.any_pressed()
            && i.pointer
                .interact_pos()
                .is_some_and(|pos| !menu_rect.contains(pos) && !anchor.contains(pos));
        i.key_pressed(Key::Escape) || outside_press
    });
    !close
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name, description and footer line fits the menu on one line.
    #[test]
    fn every_line_fits_the_menu() {
        let ctx = egui::Context::default();
        theme::apply(&ctx);
        let mut out = ctx.run_ui(egui::RawInput::default(), |_| {});
        out.textures_delta.clear();
        let width = |text: &str, size: f32| {
            ctx.fonts_mut(|f| {
                f.layout_no_wrap(text.to_owned(), theme::font(size), TEXT)
                    .size()
                    .x
            })
        };
        let row_text = MENU_W - PAD - 36.0 - PAD;
        for exp in EXPERIMENTS {
            assert!(width(exp.name, 12.0) <= row_text, "{}", exp.name);
            let w = width(exp.description, 10.0);
            assert!(w <= row_text, "{}: {w} > {row_text}", exp.description);
        }
        for line in LABS_FOOTER {
            assert!(width(line, 9.5) <= MENU_W - 2.0 * PAD, "{line}");
        }
    }
}
