//! Colours, fonts and the egui style for the editor.
//!
//! A dark, low-contrast background lets the traces carry the information:
//! the corrected output is the only saturated colour (mint accent), the
//! detected input is near-white and fades with confidence, the vibrato-free
//! centre is a soft blue. State colours (Bypass orange, Remove grey, MIDI
//! violet) are reused everywhere a note state appears.

use egui::{Color32, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Visuals};

pub const BG: Color32 = Color32::from_rgb(12, 14, 18);
pub const PANEL: Color32 = Color32::from_rgb(20, 23, 29);
pub const PANEL_HI: Color32 = Color32::from_rgb(28, 32, 40);
pub const PANEL_HOVER: Color32 = Color32::from_rgb(36, 41, 51);
pub const BORDER: Color32 = Color32::from_rgb(38, 43, 53);
pub const BORDER_HI: Color32 = Color32::from_rgb(58, 65, 79);

pub const TEXT: Color32 = Color32::from_rgb(221, 226, 235);
pub const TEXT_DIM: Color32 = Color32::from_rgb(132, 141, 156);
pub const TEXT_FAINT: Color32 = Color32::from_rgb(82, 90, 104);

/// Corrected output trace and every "active" control.
pub const ACCENT: Color32 = Color32::from_rgb(46, 230, 184);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(24, 110, 92);
/// Detected input pitch.
pub const DETECTED: Color32 = Color32::from_rgb(236, 240, 246);
/// Vibrato-free centre of the input.
pub const CENTRE: Color32 = Color32::from_rgb(110, 168, 255);

pub const BYPASS: Color32 = Color32::from_rgb(255, 159, 67);
pub const REMOVE: Color32 = Color32::from_rgb(86, 92, 105);
pub const MIDI: Color32 = Color32::from_rgb(192, 132, 252);
pub const WARN: Color32 = Color32::from_rgb(255, 196, 77);
pub const BAD: Color32 = Color32::from_rgb(255, 99, 99);

// Piano-roll lanes.
pub const LANE_ON: Color32 = Color32::from_rgb(25, 30, 39);
pub const LANE_ON_BLACK: Color32 = Color32::from_rgb(22, 26, 34);
pub const LANE_OFF: Color32 = Color32::from_rgb(15, 17, 22);
pub const LANE_BYPASS: Color32 = Color32::from_rgb(38, 30, 22);
pub const LANE_LINE: Color32 = Color32::from_rgb(31, 36, 45);
pub const GRID: Color32 = Color32::from_rgb(36, 41, 51);

pub const RADIUS: u8 = 4;

/// Proportional font at `size` points.
pub fn font(size: f32) -> FontId {
    FontId::new(size, FontFamily::Proportional)
}

/// Monospaced font for numeric readouts (digits don't jitter as values change).
pub fn mono(size: f32) -> FontId {
    FontId::new(size, FontFamily::Monospace)
}

/// `color` at `alpha` (0-1) opacity.
pub fn alpha(color: Color32, alpha: f32) -> Color32 {
    color.gamma_multiply(alpha.clamp(0.0, 1.0))
}

/// Install the dark theme on an egui context.
pub fn apply(ctx: &egui::Context) {
    // The proportional default font (Ubuntu Light) lacks some symbols the
    // UI uses (→ in tooltips); fall back to the bundled Hack font for them.
    let mut fonts = egui::FontDefinitions::default();
    if let Some(family) = fonts.families.get_mut(&FontFamily::Proportional) {
        family.push("Hack".to_owned());
    }
    ctx.set_fonts(fonts);

    let mut v = Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = PANEL;
    v.window_stroke = Stroke::new(1.0, BORDER_HI);
    v.extreme_bg_color = Color32::from_rgb(9, 10, 13);
    v.faint_bg_color = PANEL;
    v.selection.bg_fill = ACCENT_DIM;
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    v.window_corner_radius = CornerRadius::same(6);
    v.menu_corner_radius = CornerRadius::same(6);

    let r = CornerRadius::same(RADIUS);
    v.widgets.noninteractive.bg_fill = PANEL;
    v.widgets.noninteractive.weak_bg_fill = PANEL;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_DIM);
    v.widgets.noninteractive.corner_radius = r;

    v.widgets.inactive.bg_fill = PANEL_HI;
    v.widgets.inactive.weak_bg_fill = PANEL_HI;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.inactive.corner_radius = r;

    v.widgets.hovered.bg_fill = PANEL_HOVER;
    v.widgets.hovered.weak_bg_fill = PANEL_HOVER;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, BORDER_HI);
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.hovered.corner_radius = r;

    v.widgets.active.bg_fill = ACCENT_DIM;
    v.widgets.active.weak_bg_fill = ACCENT_DIM;
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.active.corner_radius = r;

    v.widgets.open = v.widgets.active;
    v.widgets.open.bg_fill = PANEL_HOVER;
    v.widgets.open.weak_bg_fill = PANEL_HOVER;

    ctx.set_visuals_of(egui::Theme::Dark, v);
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.spacing.item_spacing = egui::vec2(6.0, 4.0);
        style.spacing.button_padding = egui::vec2(8.0, 3.0);
        style.spacing.interact_size.y = 20.0;
        style.text_styles.insert(TextStyle::Body, font(12.5));
        style.text_styles.insert(TextStyle::Button, font(12.5));
        style.text_styles.insert(TextStyle::Small, font(10.5));
        style.text_styles.insert(TextStyle::Monospace, mono(12.0));
        style.text_styles.insert(TextStyle::Heading, font(18.0));
        // A plugin window is not a document: no text selection on labels.
        style.interaction.selectable_labels = false;
    });
}
