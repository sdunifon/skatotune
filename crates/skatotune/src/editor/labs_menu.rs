//! The header's LABS button and its menu of experiments ([`crate::labs`]).
//!
//! The menu is a small foreground panel that drops from the header's lower
//! edge, right-aligned to the button: one row per experiment (switch, name,
//! one-line description), the Correction colors selector (one button per
//! theme, each with a small preview of its ramp, and a line on the chosen
//! one) and a footer that says what Labs is. A click on a row flips that
//! experiment, a click on a theme chooses it; Escape or a click anywhere else
//! closes the menu. Nothing here touches host parameters.

use egui::epaint::Shadow;
use egui::{
    Align2, Area, Color32, CornerRadius, Id, Key, Mesh, Order, Painter, Pos2, Rect, Response,
    Sense, Shape, Stroke, StrokeKind, Ui, pos2, vec2,
};

use super::correction::palette;
use super::theme::{
    self, ACCENT, BORDER, BORDER_HI, PANEL, PANEL_HI, PANEL_HOVER, TEXT, TEXT_DIM, TEXT_FAINT,
};
use crate::labs::{CorrectionTheme, EXPERIMENTS, LABS_FOOTER};
use crate::params::SkatotuneParams;

/// Menu width.
pub const MENU_W: f32 = 336.0;
/// Gap between the header's lower edge and the menu.
pub const MENU_GAP: f32 = 4.0;
const PAD: f32 = 12.0;
/// Title row and experiment row heights.
pub const TITLE_H: f32 = 34.0;
pub const ROW_H: f32 = 44.0;
const FOOTER_LINE_H: f32 = 14.0;

/// Height of the Correction colors block under the switch rows.
const THEME_H: f32 = 86.0;
/// Heading of the block.
const THEME_HEADING: &str = "Correction colors";
/// What the line under the selector says while no experiment is using the
/// theme.
const THEME_IDLE_NOTE: &str = "Switch on the ribbon or the heat line to see it.";
/// Theme buttons: top (from the block's top), height and gap.
const BTN_TOP: f32 = 30.0;
const BTN_H: f32 = 34.0;
const BTN_GAP: f32 = 4.0;
/// The ramp preview inside a button: inset from its sides, from its top,
/// height, and the opacity it is faded to while no experiment uses the theme.
const PREVIEW_INSET: f32 = 6.0;
const PREVIEW_TOP: f32 = 6.0;
const PREVIEW_H: f32 = 7.0;
const PREVIEW_IDLE_OPACITY: f32 = 0.45;
/// Text sizes of the block: a button's label, and the line under the buttons.
const BTN_FONT: f32 = 9.5;
const NOTE_FONT: f32 = 10.0;

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
    let resp = ui.interact(rect, Id::new("skato-labs-button"), Sense::click());
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

/// Top of the Correction colors block, from the menu's top.
fn theme_block_top() -> f32 {
    TITLE_H + ROW_H * EXPERIMENTS.len() as f32
}

/// Height of the whole menu.
pub fn menu_height() -> f32 {
    let footer_h = FOOTER_LINE_H * LABS_FOOTER.len() as f32;
    theme_block_top() + THEME_H + 10.0 + footer_h + PAD - 2.0
}

/// The button of theme `i` (an index into [`CorrectionTheme::ALL`]) in the
/// menu whose top-left corner is `origin`.
pub fn theme_button(origin: Pos2, i: usize) -> Rect {
    let n = CorrectionTheme::ALL.len() as f32;
    let w = (MENU_W - 2.0 * PAD - BTN_GAP * (n - 1.0)) / n;
    Rect::from_min_size(
        pos2(
            origin.x + PAD + (w + BTN_GAP) * i as f32,
            origin.y + theme_block_top() + BTN_TOP,
        ),
        vec2(w, BTN_H),
    )
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
pub fn menu(ui: &Ui, anchor: Rect, header_bottom: f32, params: &SkatotuneParams) -> bool {
    let ctx = ui.ctx().clone();
    let labs = params.labs();
    let origin = menu_origin(anchor, header_bottom);
    let shown = Area::new(Id::new("skato-labs-menu"))
        .order(Order::Foreground)
        .movable(false)
        .fixed_pos(origin)
        .show(&ctx, |ui| {
            let p = ui.painter().clone();
            let rect = egui::Rect::from_min_size(origin, vec2(MENU_W, menu_height()));
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
                let resp = ui.interact(row, Id::new(("skato-labs-row", i)), Sense::click());
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

            theme_selector(ui, &p, origin, params);

            // Footer: one sentence per line.
            let foot_top = rect.top() + theme_block_top() + THEME_H + 4.0;
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

/// A theme's ramp as a small gradient across `strip`: the palette's five
/// stops, which is exactly the ramp (the gradient between two stops is the
/// ramp's own sRGB blend). `opacity` fades it.
fn ramp_preview(p: &Painter, strip: Rect, theme: CorrectionTheme, opacity: f32) {
    let stops = palette(theme).stops;
    let mut mesh = Mesh::default();
    for (k, stop) in stops.iter().enumerate() {
        let x = strip.left() + strip.width() * k as f32 / (stops.len() - 1) as f32;
        let color = theme::alpha(*stop, opacity);
        let idx = mesh.vertices.len() as u32;
        mesh.colored_vertex(pos2(x, strip.top()), color);
        mesh.colored_vertex(pos2(x, strip.bottom()), color);
        if k > 0 {
            mesh.add_triangle(idx - 2, idx - 1, idx);
            mesh.add_triangle(idx - 1, idx + 1, idx);
        }
    }
    p.add(Shape::mesh(mesh));
}

/// The Correction colors block: heading, one button per theme and a line on
/// the chosen one. A click on a button chooses that theme (saved with the
/// project); the line switches to a hint while neither the ribbon nor the heat
/// line is on, since the theme then has nothing to colour.
fn theme_selector(ui: &Ui, p: &Painter, origin: Pos2, params: &SkatotuneParams) {
    let left = origin.x + PAD;
    let top = origin.y + theme_block_top();
    p.hline(
        origin.x + 1.0..=origin.x + MENU_W - 1.0,
        top + 3.0,
        Stroke::new(1.0, BORDER),
    );
    p.text(
        pos2(left, top + 18.0),
        Align2::LEFT_CENTER,
        THEME_HEADING,
        theme::font(12.0),
        TEXT,
    );

    let in_use = params.labs().shows_correction_colour();
    let preview_opacity = if in_use { 1.0 } else { PREVIEW_IDLE_OPACITY };
    for (i, theme) in CorrectionTheme::ALL.into_iter().enumerate() {
        let btn = theme_button(origin, i);
        let resp = ui.interact(btn, Id::new(("skato-labs-theme", i)), Sense::click());
        if resp.clicked() {
            let mut next = params.labs();
            next.correction_theme = theme;
            params.set_labs(next);
        }
        let chosen = params.labs().correction_theme == theme;
        let (fill, stroke, text) = if chosen {
            (theme::alpha(ACCENT, 0.16), theme::alpha(ACCENT, 0.8), TEXT)
        } else if resp.hovered() {
            (PANEL_HOVER, BORDER_HI, TEXT)
        } else {
            (PANEL_HI, BORDER_HI, TEXT_DIM)
        };
        let r = CornerRadius::same(theme::RADIUS);
        p.rect_filled(btn, r, fill);
        p.rect_stroke(btn, r, Stroke::new(1.0, stroke), StrokeKind::Inside);
        let strip = Rect::from_min_size(
            btn.min + vec2(PREVIEW_INSET, PREVIEW_TOP),
            vec2(btn.width() - 2.0 * PREVIEW_INSET, PREVIEW_H),
        );
        ramp_preview(p, strip, theme, preview_opacity);
        p.text(
            pos2(btn.center().x, btn.bottom() - 10.0),
            Align2::CENTER_CENTER,
            theme.label(),
            theme::font(BTN_FONT),
            text,
        );
        resp.on_hover_ui(|ui| {
            ui.label(format!("{}: {}", theme.name(), theme.note()));
        });
    }

    let (note, color) = if in_use {
        (params.labs().correction_theme.note(), TEXT_DIM)
    } else {
        (THEME_IDLE_NOTE, TEXT_FAINT)
    };
    p.text(
        pos2(left, top + BTN_TOP + BTN_H + 12.0),
        Align2::LEFT_CENTER,
        note,
        theme::font(NOTE_FONT),
        color,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labs::LabsSettings;

    /// A context with the editor's fonts and style (and no fade-in: a new
    /// area otherwise draws at a fraction of its opacity at first).
    fn context() -> egui::Context {
        let ctx = egui::Context::default();
        theme::apply(&ctx);
        ctx.style_mut_of(egui::Theme::Dark, |style| style.animation_time = 0.0);
        let mut out = ctx.run_ui(egui::RawInput::default(), |_| {});
        out.textures_delta.clear();
        ctx
    }

    /// Every name, description and footer line fits the menu on one line,
    /// and so does everything in the Correction colors block: the heading,
    /// each theme's button label (inside its button) and its note (the line
    /// under the buttons), and the hint that replaces the note.
    #[test]
    fn every_line_fits_the_menu() {
        let ctx = context();
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

        let inner = MENU_W - 2.0 * PAD;
        assert!(width(THEME_HEADING, 12.0) <= inner);
        assert!(width(THEME_IDLE_NOTE, NOTE_FONT) <= inner);
        let button_w = theme_button(Pos2::ZERO, 0).width();
        for theme in CorrectionTheme::ALL {
            let label = width(theme.label(), BTN_FONT);
            assert!(
                label <= button_w - 6.0,
                "{}: {label} in a {button_w} px button",
                theme.label()
            );
            let note = width(theme.note(), NOTE_FONT);
            assert!(note <= inner, "{}: {note} > {inner}", theme.note());
        }
    }

    /// The selector is a row of equal buttons inside the menu, under the
    /// switch rows and clear of the footer.
    #[test]
    fn theme_buttons_sit_in_a_row_inside_the_menu() {
        let origin = pos2(100.0, 50.0);
        let menu = Rect::from_min_size(origin, vec2(MENU_W, menu_height()));
        let buttons: Vec<Rect> = (0..CorrectionTheme::ALL.len())
            .map(|i| theme_button(origin, i))
            .collect();
        let rows_end = origin.y + TITLE_H + ROW_H * EXPERIMENTS.len() as f32;
        let footer_top = origin.y + theme_block_top() + THEME_H;
        for (i, b) in buttons.iter().enumerate() {
            assert!(menu.contains_rect(*b), "button {i}: {b:?} in {menu:?}");
            assert!(b.top() > rows_end, "under the switch rows");
            assert!(
                b.bottom() + 12.0 + NOTE_FONT <= footer_top,
                "room for the note before the footer"
            );
            assert_eq!(b.top(), buttons[0].top());
            assert!((b.width() - buttons[0].width()).abs() < 1e-3);
            assert!(
                b.width() > 50.0 && b.height() >= 28.0,
                "a comfortable target"
            );
        }
        for pair in buttons.windows(2) {
            let gap = pair[1].left() - pair[0].right();
            assert!((gap - BTN_GAP).abs() < 1e-3, "{gap}");
        }
        // Flush with the content of the rows above it.
        assert_eq!(buttons[0].left(), origin.x + PAD);
        assert!((buttons[4].right() - (menu.right() - PAD)).abs() < 1e-3);
        // The whole menu stays small enough for the editor window.
        assert!(super::super::HEADER_H + MENU_GAP + menu_height() < super::super::HEIGHT);
    }

    /// The LABS button of a header at the editor's size.
    fn anchor() -> Rect {
        Rect::from_min_size(pos2(1034.0, 5.0), vec2(50.0, 16.0))
    }

    /// One frame of the open menu at `time` seconds, with the pointer
    /// `events`.
    fn menu_frame(
        ctx: &egui::Context,
        params: &SkatotuneParams,
        time: f64,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(
                Pos2::ZERO,
                vec2(super::super::WIDTH, super::super::HEIGHT),
            )),
            time: Some(time),
            events,
            ..Default::default()
        };
        let mut out = ctx.run_ui(input, |ui| {
            menu(ui, anchor(), super::super::HEADER_H, params);
        });
        out.textures_delta.clear();
        out
    }

    /// The texts and meshes of a frame.
    fn texts_and_meshes(out: egui::FullOutput) -> (Vec<String>, Vec<Mesh>) {
        let mut texts = Vec::new();
        let mut meshes = Vec::new();
        for clipped in out.shapes {
            match clipped.shape {
                Shape::Text(t) => texts.push(t.galley.text().to_owned()),
                Shape::Mesh(m) => meshes.push((*m).clone()),
                _ => {}
            }
        }
        (texts, meshes)
    }

    /// The open menu over the editor's header, as drawn once egui has sized
    /// it (an area is hidden on its first frame): the texts it draws and
    /// the meshes (the ramp previews).
    fn draw_menu(ctx: &egui::Context, params: &SkatotuneParams) -> (Vec<String>, Vec<Mesh>) {
        let _ = menu_frame(ctx, params, 0.0, vec![]);
        texts_and_meshes(menu_frame(ctx, params, 0.0, vec![]))
    }

    /// The heading, one label per theme, and the line on the chosen theme.
    #[test]
    fn the_selector_names_every_theme_and_describes_the_chosen_one() {
        let ctx = context();
        let params = SkatotuneParams::default();
        let (texts, _) = draw_menu(&ctx, &params);
        assert!(texts.iter().any(|t| t == THEME_HEADING), "{texts:?}");
        for theme in CorrectionTheme::ALL {
            assert!(texts.iter().any(|t| t == theme.label()), "{texts:?}");
        }
        assert!(texts.iter().any(|t| t == CorrectionTheme::Teal.note()));
        assert!(!texts.iter().any(|t| t == THEME_IDLE_NOTE));
        // Only the chosen theme is described.
        let described = |texts: &[String]| {
            CorrectionTheme::ALL
                .into_iter()
                .filter(|t| texts.iter().any(|x| x == t.note()))
                .collect::<Vec<_>>()
        };
        assert_eq!(described(&texts), vec![CorrectionTheme::Teal]);

        for theme in [CorrectionTheme::GreenRed, CorrectionTheme::BlueOrange] {
            let mut labs = params.labs();
            labs.correction_theme = theme;
            params.set_labs(labs);
            let (texts, _) = draw_menu(&ctx, &params);
            assert_eq!(described(&texts), vec![theme]);
        }
        // The red-green caveat is on screen with that theme chosen.
        let mut labs = params.labs();
        labs.correction_theme = CorrectionTheme::GreenRed;
        params.set_labs(labs);
        let (texts, _) = draw_menu(&ctx, &params);
        assert!(
            texts
                .iter()
                .any(|t| t.contains("Red-green colour-blind users may not tell the ends apart")),
            "{texts:?}"
        );
    }

    /// With neither the ribbon nor the heat line on, the theme has nothing to
    /// colour: the line under the selector says how to see it, and goes back
    /// to describing the theme as soon as one is on.
    #[test]
    fn the_selector_hints_while_nothing_uses_the_theme() {
        let ctx = context();
        let params = SkatotuneParams::default();
        params.set_labs(LabsSettings {
            correction_ribbon: false,
            heat_line: false,
            correction_theme: CorrectionTheme::Inferno,
            ..LabsSettings::DEFAULT
        });
        let (texts, _) = draw_menu(&ctx, &params);
        assert!(texts.iter().any(|t| t == THEME_IDLE_NOTE), "{texts:?}");
        assert!(!texts.iter().any(|t| t == CorrectionTheme::Inferno.note()));
        // The choice itself is still shown (the buttons are all there).
        for theme in CorrectionTheme::ALL {
            assert!(texts.iter().any(|t| t == theme.label()));
        }
        params.set_labs(LabsSettings {
            heat_line: true,
            ..params.labs()
        });
        let (texts, _) = draw_menu(&ctx, &params);
        assert!(texts.iter().any(|t| t == CorrectionTheme::Inferno.note()));
        assert!(!texts.iter().any(|t| t == THEME_IDLE_NOTE));
    }

    /// Each button's preview is the theme's own ramp: its five stops as a
    /// gradient across the strip (faded while nothing uses the theme).
    #[test]
    fn each_preview_is_the_themes_ramp() {
        let ctx = context();
        let params = SkatotuneParams::default();
        let ramp_meshes = |meshes: &[Mesh]| -> Vec<Vec<Color32>> {
            meshes
                .iter()
                .filter(|m| m.vertices.len() == 10)
                .map(|m| m.vertices.iter().step_by(2).map(|v| v.color).collect())
                .collect()
        };
        let (_, meshes) = draw_menu(&ctx, &params);
        let drawn = ramp_meshes(&meshes);
        let expected: Vec<Vec<Color32>> = CorrectionTheme::ALL
            .into_iter()
            .map(|t| palette(t).stops.to_vec())
            .collect();
        assert_eq!(drawn, expected);

        params.set_labs(LabsSettings {
            correction_ribbon: false,
            heat_line: false,
            ..params.labs()
        });
        let (_, meshes) = draw_menu(&ctx, &params);
        let faded: Vec<Vec<Color32>> = expected
            .iter()
            .map(|stops| {
                stops
                    .iter()
                    .map(|c| theme::alpha(*c, PREVIEW_IDLE_OPACITY))
                    .collect()
            })
            .collect();
        assert_eq!(ramp_meshes(&meshes), faded);
    }

    /// Resting the pointer on a theme's button shows a tooltip with its full
    /// name and note (for the traffic light, the red-green colour-blindness
    /// caveat); moving off hides it.
    #[test]
    fn hovering_a_theme_shows_its_name_and_note() {
        let ctx = context();
        let params = SkatotuneParams::default();
        let origin = menu_origin(anchor(), super::super::HEADER_H);
        let _ = menu_frame(&ctx, &params, 0.0, vec![]);
        let mut time = 0.0;
        for theme in [CorrectionTheme::GreenRed, CorrectionTheme::BlueOrange] {
            let i = CorrectionTheme::ALL.iter().position(|t| *t == theme);
            let at = theme_button(origin, i.unwrap()).center();
            let tip = format!("{}: {}", theme.name(), theme.note());
            let mut shown_at = None;
            for k in 0..20 {
                time += 0.1;
                let events = if k == 0 {
                    vec![egui::Event::PointerMoved(at)]
                } else {
                    vec![]
                };
                let (texts, _) = texts_and_meshes(menu_frame(&ctx, &params, time, events));
                let seen = texts.contains(&tip);
                if seen && shown_at.is_none() {
                    shown_at = Some(k);
                }
                assert!(shown_at.is_none() || seen, "{theme:?}: tooltip flickers");
            }
            assert!(shown_at.is_some(), "{theme:?}: no tooltip for {tip}");
            // Off the buttons (a blank spot of the menu): no tooltip.
            time += 0.1;
            let away = vec![egui::Event::PointerMoved(
                origin + vec2(MENU_W - 20.0, 12.0),
            )];
            let (texts, _) = texts_and_meshes(menu_frame(&ctx, &params, time, away));
            assert!(!texts.contains(&tip), "{theme:?}");
        }
        // The caveat is part of the traffic light's tooltip.
        assert!(
            CorrectionTheme::GreenRed
                .note()
                .contains("Red-green colour-blind")
        );
    }
}
