//! Vertical 12-note keyboard: the scale editor.
//!
//! One row per pitch class, B at the top down to C, showing the *effective*
//! state the engine corrects to:
//!
//! - Normally: scale membership resolved with the per-note override.
//!   Clicking a row cycles its override Scale → On → Bypass → Remove;
//!   right-click returns it to the scale. An "M" marks a manual override.
//! - While a MIDI-learned scale is in use (it replaces Key/Scale and the
//!   overrides until cleared, see [`crate::learn`]): learned notes are violet
//!   "ON" chips, the rest plain "OFF", and a click adds or removes that note
//!   in the learned scale. The overrides do not apply, so no "M" markers.

use egui::{Align2, Color32, CornerRadius, Id, Rect, Sense, Stroke, StrokeKind, Ui, pos2, vec2};
use nice_plug::prelude::*;
use vt_core::NoteState;

use super::graph::is_black_key;
use super::theme::{
    self, ACCENT, BORDER, BYPASS, MIDI, PANEL_HOVER, REMOVE, TEXT, TEXT_DIM, TEXT_FAINT,
};
use crate::params::{NOTE_NAMES, NoteOverride, VocalTuneParams};

/// What the keyboard shows besides the parameters.
pub struct KeyboardState {
    /// Effective state per pitch class.
    pub states: [NoteState; 12],
    /// `states` is a MIDI-learned scale (learned notes in violet; clicks
    /// edit the learned scale instead of the overrides).
    pub learned: bool,
    /// Pitch class of the current target note, if any.
    pub target_pc: Option<usize>,
}

fn state_label(s: NoteState, learned: bool) -> &'static str {
    match (s, learned) {
        (NoteState::On, _) => "ON",
        (NoteState::Bypass, _) => "BYPASS",
        // A learned scale has only learned notes and the rest.
        (NoteState::Remove, true) => "OFF",
        (NoteState::Remove, false) => "REMOVE",
    }
}

fn state_colour(s: NoteState, learned: bool) -> Color32 {
    match s {
        NoteState::On if learned => MIDI,
        NoteState::On => ACCENT,
        NoteState::Bypass => BYPASS,
        NoteState::Remove => REMOVE,
    }
}

/// Hover text of one keyboard row.
fn row_tooltip(name: &str, state: NoteState, learned: bool, manual: bool) -> String {
    if learned {
        let what = if state == NoteState::On {
            "in the learned scale"
        } else {
            "not in the learned scale"
        };
        format!(
            "{name}: {what} (learned from MIDI).\nClick: add or remove it in the learned scale.\nCLEAR returns to Key/Scale."
        )
    } else {
        format!(
            "{name}: {} ({}).\nClick: cycle Scale → On → Bypass → Remove.\nRight-click: follow the scale again.",
            state_label(state, false).to_lowercase(),
            if manual {
                "manual override"
            } else {
                "from the scale"
            },
        )
    }
}

/// Draw the keyboard into `rect` and apply clicks to the note overrides.
pub fn keyboard(
    ui: &mut Ui,
    rect: Rect,
    setter: &ParamSetter,
    params: &VocalTuneParams,
    ks: &KeyboardState,
) {
    let row_h = rect.height() / 12.0;
    for row in 0..12usize {
        let pc = 11 - row;
        let r = Rect::from_min_size(
            pos2(rect.left(), rect.top() + row as f32 * row_h),
            vec2(rect.width(), row_h),
        );
        let param = params.note_override(pc);
        let ovr = param.value();
        let state = ks.states.get(pc).copied().unwrap_or(NoteState::On);
        let resp = ui.interact(r, Id::new(("vt-key", pc)), Sense::click());
        if ks.learned {
            if resp.clicked() {
                crate::learn::toggle(&params.learned_scale, pc);
            }
        } else if resp.clicked() {
            setter.begin_set_parameter(param);
            setter.set_parameter(param, ovr.next());
            setter.end_set_parameter(param);
        }
        if !ks.learned && resp.secondary_clicked() && ovr != NoteOverride::Scale {
            setter.begin_set_parameter(param);
            setter.set_parameter(param, NoteOverride::Scale);
            setter.end_set_parameter(param);
        }
        let hovered = resp.hovered();
        let name = NOTE_NAMES.get(pc).copied().unwrap_or("?");
        let painter = ui.painter();

        // Row background and the piano key.
        let inner = r.shrink2(vec2(0.0, 1.0));
        painter.rect_filled(
            inner,
            CornerRadius::same(3),
            if hovered { PANEL_HOVER } else { theme::PANEL },
        );
        let black = is_black_key(pc);
        let key_w = if black { 26.0 } else { 38.0 };
        let key = Rect::from_min_size(inner.min, vec2(key_w, inner.height()));
        let (key_fill, key_text) = if black {
            (Color32::from_rgb(10, 11, 14), TEXT_DIM)
        } else {
            (
                Color32::from_rgb(196, 201, 211),
                Color32::from_rgb(30, 33, 40),
            )
        };
        painter.rect_filled(key, CornerRadius::same(3), key_fill);
        painter.rect_stroke(
            key,
            CornerRadius::same(3),
            Stroke::new(1.0, BORDER),
            StrokeKind::Inside,
        );
        if ks.target_pc == Some(pc) {
            // The note being sung right now: an accent glow on the key.
            painter.rect_stroke(
                key,
                CornerRadius::same(3),
                Stroke::new(1.5, ACCENT),
                StrokeKind::Inside,
            );
            painter.circle_filled(pos2(key.right() + 8.0, inner.center().y), 3.0, ACCENT);
        }
        painter.text(
            pos2(key.left() + 6.0, inner.center().y),
            Align2::LEFT_CENTER,
            name,
            theme::font(11.0),
            key_text,
        );

        // Effective state chip on the right.
        let chip_w = 62.0;
        let chip = Rect::from_min_max(
            pos2(inner.right() - chip_w - 4.0, inner.top() + 2.0),
            pos2(inner.right() - 4.0, inner.bottom() - 2.0),
        );
        let col = state_colour(state, ks.learned);
        // The overrides only matter while no learned scale replaces them.
        let manual = !ks.learned && ovr != NoteOverride::Scale;
        let strong = manual || (ks.learned && state == NoteState::On);
        painter.rect_filled(
            chip,
            CornerRadius::same(255),
            theme::alpha(col, if strong { 0.24 } else { 0.12 }),
        );
        let border = if strong { col } else { theme::alpha(col, 0.35) };
        painter.rect_stroke(
            chip,
            CornerRadius::same(255),
            Stroke::new(1.0, border),
            StrokeKind::Inside,
        );
        painter.text(
            chip.center(),
            Align2::CENTER_CENTER,
            state_label(state, ks.learned),
            theme::font(9.5),
            if state == NoteState::Remove {
                TEXT_DIM
            } else {
                TEXT
            },
        );
        if manual {
            // Override marker: this note no longer follows the scale.
            painter.text(
                pos2(chip.left() - 8.0, inner.center().y),
                Align2::CENTER_CENTER,
                "M",
                theme::font(9.0),
                TEXT_FAINT,
            );
        }
        resp.on_hover_ui(|ui| {
            ui.label(row_tooltip(name, state, ks.learned, manual));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_rows_read_on_off_and_say_how_to_edit_them() {
        assert_eq!(state_label(NoteState::Remove, true), "OFF");
        assert_eq!(state_label(NoteState::Remove, false), "REMOVE");
        assert_eq!(state_colour(NoteState::On, true), MIDI);
        assert_eq!(state_colour(NoteState::On, false), ACCENT);
        let t = row_tooltip("E", NoteState::On, true, false);
        assert!(
            t.contains("in the learned scale") && t.contains("CLEAR"),
            "{t}"
        );
        let t = row_tooltip("F", NoteState::Bypass, false, true);
        assert!(
            t.contains("manual override") && t.contains("Right-click"),
            "{t}"
        );
    }
}
