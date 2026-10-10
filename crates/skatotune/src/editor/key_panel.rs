//! The Key Source rows of the sidebar (#5, #6): a MANUAL / MIDI / DETECT
//! switch, the key selectors (read-only, showing the followed key, while
//! following), and a status line with Auto-Key's proposal and the APPLY /
//! COPY actions (docs/research/host-key-scale.md §3.7).

use std::sync::atomic::Ordering;

use egui::{Align2, Color32, CornerRadius, Id, Rect, Sense, Stroke, StrokeKind, Ui, pos2, vec2};
use nice_plug::prelude::*;
use skato_core::ScaleKind;
use skato_engine::KeyName;

use super::theme::{self, ACCENT, BORDER, MIDI, PANEL_HI, TEXT, TEXT_DIM, TEXT_FAINT, WARN};
use super::widgets;
use crate::keysource::{self, InEffect};
use crate::params::{KeyChoice, KeySourceChoice, NoteOverride, ParamValues, ScaleChoice, SkatotuneParams};
use crate::shared::SharedState;

/// A proposal needs this much steady singing behind it before it is shown.
const SHOW_EVIDENCE_S: f32 = 1.0;

/// What the editor knows about the Key Source this frame.
#[derive(Clone, Copy, Debug)]
pub struct KeyView {
    pub in_effect: InEffect,
    pub proposal: Option<KeyName>,
    pub confidence: f32,
    pub evidence_s: f32,
    /// The followed key came from the saved project (not yet confirmed).
    pub saved: bool,
    /// MIDI Scale latched a chord in this session (not just the saved one).
    pub midi_live: bool,
    /// MIDI notes are arriving while MIDI does nothing (Manual, MIDI Mode Off).
    pub midi_unused: bool,
}

impl KeyView {
    /// Read the audio thread's view. While no audio runs (or before the
    /// next block after a source change) the saved chord or key stands in.
    pub fn read(shared: &SharedState, params: &SkatotuneParams, v: &ParamValues, midi_unused: bool) -> Self {
        let confirmed_word = shared.key_confirmed.load(Ordering::Relaxed);
        let reported = keysource::unpack_in_effect(shared.key_in_effect.load(Ordering::Relaxed));
        let in_effect = match (v.key_source, reported) {
            (KeySourceChoice::Manual, _) => InEffect::Manual,
            (KeySourceChoice::MidiScale, e @ (InEffect::Midi { .. } | InEffect::MidiWaiting)) => e,
            (KeySourceChoice::MidiScale, _) => keysource::unpack_midi(params.midi_scale.load(Ordering::Relaxed))
                .map_or(InEffect::MidiWaiting, |(mask, bass)| InEffect::Midi { mask, bass }),
            (KeySourceChoice::Detect, e @ (InEffect::Detect(_) | InEffect::DetectWaiting)) => e,
            (KeySourceChoice::Detect, _) => {
                keysource::unpack_key(params.detected_key.load(Ordering::Relaxed)).map_or(InEffect::DetectWaiting, InEffect::Detect)
            }
        };
        Self {
            in_effect,
            proposal: keysource::unpack_key(shared.key_proposal.load(Ordering::Relaxed)),
            confidence: shared.key_confidence.load(Ordering::Relaxed),
            evidence_s: shared.key_evidence.load(Ordering::Relaxed),
            saved: confirmed_word & keysource::KEY_SAVED != 0,
            midi_live: shared.midi_latched.load(Ordering::Relaxed),
            midi_unused,
        }
    }

    /// The pitch-class set the followed source puts in effect, if any.
    pub fn followed_mask(&self) -> Option<u16> {
        match self.in_effect {
            InEffect::Midi { mask, .. } => Some(mask),
            InEffect::Detect(k) => Some(k.mask),
            _ => None,
        }
    }
}

pub const NOTE: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/// Compact key name for the status line ("A#m harm.", "D Dorian").
pub fn short_name(k: KeyName) -> String {
    let root = NOTE[k.root as usize % 12];
    match k.kind {
        ScaleKind::Major => root.to_string(),
        ScaleKind::NaturalMinor => format!("{root}m"),
        ScaleKind::HarmonicMinor => format!("{root}m harm."),
        ScaleKind::MelodicMinor => format!("{root}m mel."),
        kind => format!("{root} {}", kind.name()),
    }
}

fn long_name(k: KeyName) -> String {
    format!("{} {}", NOTE[k.root as usize % 12], k.kind.name())
}

/// The manual key as a name (for "using C Major").
fn manual_name(v: &ParamValues) -> String {
    format!("{} {}", NOTE[v.key.pitch_class() as usize], ScaleKind::from(v.scale).name())
}

/// A read-only box styled like a disabled selector.
fn readonly_box(ui: &Ui, rect: Rect, text: &str, color: Color32) {
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(theme::RADIUS), PANEL_HI);
    p.rect_stroke(rect, CornerRadius::same(theme::RADIUS), Stroke::new(1.0, BORDER), StrokeKind::Inside);
    p.text(pos2(rect.left() + 8.0, rect.center().y), Align2::LEFT_CENTER, text, theme::font(12.5), color);
}

fn set_param<T: Enum + PartialEq + Copy + 'static>(setter: &ParamSetter, param: &EnumParam<T>, value: T) {
    setter.begin_set_parameter(param);
    setter.set_parameter(param, value);
    setter.end_set_parameter(param);
}

/// Write a key into Key/Scale (one undoable gesture per parameter).
fn set_manual_key(setter: &ParamSetter, p: &SkatotuneParams, k: KeyName) {
    set_param(setter, &p.key, KeyChoice::from_index(k.root as usize % 12));
    set_param(setter, &p.scale, ScaleChoice::from_kind(k.kind));
}

/// Copy a followed set into the manual controls and switch to Manual: a
/// named set becomes Key/Scale; any other set becomes Chromatic on its bass
/// note with the missing notes set to Remove.
fn copy_to_manual(setter: &ParamSetter, p: &SkatotuneParams, mask: u16, root: u8) {
    match ScaleKind::identify(mask, Some(root)) {
        Some((r, k)) => set_manual_key(setter, p, KeyName::new(r, k)),
        None => {
            set_param(setter, &p.key, KeyChoice::from_index(root as usize % 12));
            set_param(setter, &p.scale, ScaleChoice::Chromatic);
            for pc in 0..12 {
                let param = p.note_override(pc);
                if mask & (1 << pc) == 0 && param.value() == NoteOverride::Scale {
                    set_param(setter, param, NoteOverride::Remove);
                }
            }
        }
    }
    set_param(setter, &p.key_source, KeySourceChoice::Manual);
}

/// Draw the rows from `y` (just below the section title) and return the y
/// after them. `learned`: a MIDI-learned scale replaces Key/Scale (Manual).
#[allow(clippy::too_many_arguments)]
pub fn rows(
    ui: &mut Ui,
    x0: f32,
    x1: f32,
    mut y: f32,
    setter: &ParamSetter,
    params: &SkatotuneParams,
    v: &ParamValues,
    view: &KeyView,
    learned: bool,
) -> f32 {
    let w = x1 - x0;

    // 1. Source switch, with hints: MIDI arriving unused, or a confident
    //    detection that differs from the manual key.
    let seg_w = (w - 8.0) / 3.0;
    let manual_key = KeyName::new(v.key.pitch_class(), v.scale.into());
    let detect_hint = v.key_source == KeySourceChoice::Manual
        && view.confidence >= 0.8
        && view.evidence_s >= SHOW_EVIDENCE_S
        && view.proposal.is_some_and(|k| k.mask != manual_key.mask || learned);
    let segs = [
        (KeySourceChoice::Manual, "MANUAL", "Key and Scale (or a MIDI-learned scale).", false, ACCENT),
        (
            KeySourceChoice::MidiScale,
            "MIDI",
            "MIDI Scale: a chord on the MIDI input sets the scale and stays latched after you let go (MIDI notes are not targets here).\nFollows a DAW's key when its scale arrives as MIDI, e.g. Bitwig: hold all 12 notes through Key Filter+ (Use Global Key) into Skatotune's MIDI input.",
            v.key_source == KeySourceChoice::Manual && view.midi_unused,
            MIDI,
        ),
        (
            KeySourceChoice::Detect,
            "DETECT",
            "Follow Auto-Key, the key detected from the singing. It changes only after a new key has led for 8 s, and only between phrases; until it is sure, Key/Scale applies.",
            detect_hint,
            ACCENT,
        ),
    ];
    for (i, (src, label, tip, hint, hint_col)) in segs.into_iter().enumerate() {
        let r = Rect::from_min_size(pos2(x0 + i as f32 * (seg_w + 4.0), y), vec2(seg_w, 20.0));
        if widgets::button(ui, r, Id::new(("skato-key-source", i)), label, v.key_source == src, tip).clicked() && v.key_source != src {
            set_param(setter, &params.key_source, src);
        }
        if hint {
            ui.painter().circle_filled(pos2(r.right() - 5.0, r.top() + 5.0), 2.5, hint_col);
        }
    }
    y += 28.0;

    // 2. Selectors: editable in Manual (unless a learned scale replaces
    //    them), else the followed key, read-only.
    let key_r = Rect::from_min_size(pos2(x0, y), vec2(54.0, 24.0));
    let scale_r = Rect::from_min_size(pos2(x0 + 62.0, y), vec2(w - 62.0, 24.0));
    let followed = match view.in_effect {
        InEffect::Midi { mask, bass } => Some(keysource::name_midi(mask, bass).map_or_else(
            || (NOTE[bass as usize % 12].to_string(), format!("Custom ({} notes)", mask.count_ones())),
            |k| (NOTE[k.root as usize].to_string(), k.kind.name().to_string()),
        )),
        InEffect::Detect(k) => Some((NOTE[k.root as usize].to_string(), k.kind.name().to_string())),
        _ => None,
    };
    match followed {
        Some((key, scale)) if v.key_source != KeySourceChoice::Manual => {
            let col = if view.saved || matches!(view.in_effect, InEffect::Midi { .. }) && !view.midi_live { TEXT_DIM } else { TEXT };
            readonly_box(ui, key_r, &key, col);
            readonly_box(ui, scale_r, &scale, col);
            ui.interact(key_r.union(scale_r), Id::new("skato-key-followed"), Sense::hover())
                .on_hover_text(format!("Followed key. Key/Scale ({}) apply when the source has nothing.", manual_name(v)));
        }
        _ => {
            ui.add_enabled_ui(!(learned && v.key_source == KeySourceChoice::Manual), |ui| {
                widgets::combo(ui, key_r, setter, &params.key);
                widgets::combo(ui, scale_r, setter, &params.scale);
            });
        }
    }
    y += 30.0;

    // 3. Status line and its action.
    let line_y = y + 8.0;
    let btn = Rect::from_min_size(pos2(x1 - 44.0, line_y - 8.0), vec2(44.0, 16.0));
    let font = theme::font(10.0);
    let confident_name =
        view.proposal.filter(|_| view.evidence_s >= SHOW_EVIDENCE_S).map(|k| format!("{} {:.0}%", short_name(k), 100.0 * view.confidence));
    let heard = format!(
        "Auto-Key: {:.0}% sure, from {:.0} s of steady singing.",
        100.0 * view.confidence,
        view.evidence_s
    );
    let (text, color, tip, action): (String, Color32, String, Option<&str>) = match (v.key_source, view.in_effect) {
        (KeySourceChoice::Manual, _) => match &confident_name {
            Some(name) if view.proposal.is_some_and(|k| k.mask == manual_key.mask) && !learned => (
                format!("KEY MATCHES · {name}"),
                if view.confidence >= 0.8 { ACCENT } else { TEXT_DIM },
                format!("{heard}\nIt uses the same notes as Key/Scale, so correction is already right."),
                None,
            ),
            Some(name) => (
                format!("DETECTED {name}"),
                if view.confidence >= 0.8 { TEXT } else { TEXT_DIM },
                format!("{heard}\nNothing changes until you click APPLY (it writes Key and Scale). Choose DETECT to follow it automatically."),
                Some("APPLY"),
            ),
            None => (
                "LISTENING FOR THE KEY".into(),
                TEXT_FAINT,
                "Auto-Key listens to the singing and proposes a key here after a few phrases. Nothing changes until you click APPLY.".into(),
                None,
            ),
        },
        (_, InEffect::Midi { mask, bass }) => {
            let name = keysource::name_midi(mask, bass).map_or_else(|| format!("Custom ({} notes)", mask.count_ones()), short_name);
            if view.midi_live {
                (format!("MIDI · {name}"), MIDI, "Latched from the MIDI input. Play another chord to change it.".into(), Some("COPY"))
            } else {
                (format!("{name} (saved)"), TEXT_DIM, "The chord saved with the project. Play a chord to change it.".into(), Some("COPY"))
            }
        }
        (_, InEffect::MidiWaiting) => (
            format!("WAITING FOR A CHORD · {}", short_name(manual_key)),
            WARN,
            format!("No chord on the MIDI input yet, so Key/Scale ({}) applies.", manual_name(v)),
            None,
        ),
        (_, InEffect::Detect(k)) => {
            if view.saved {
                (format!("{} (saved)", short_name(k)), TEXT_DIM, format!("The key saved with the project, until the singing confirms or replaces it.\n{heard}"), Some("COPY"))
            } else {
                (format!("DETECT · {}", short_name(k)), ACCENT, format!("Following the detected key ({}).\n{heard}", long_name(k)), Some("COPY"))
            }
        }
        (_, _) => (
            match &confident_name {
                Some(name) => format!("LISTENING · {name}"),
                None => format!("LISTENING · {}", short_name(manual_key)),
            },
            WARN,
            format!("Auto-Key isn't sure yet, so Key/Scale ({}) applies.\n{heard}", manual_name(v)),
            None,
        ),
    };
    let text_right = if action.is_some() { btn.left() - 6.0 } else { x1 };
    let painted = ui.painter().with_clip_rect(Rect::from_min_max(pos2(x0, line_y - 9.0), pos2(text_right, line_y + 9.0))).text(
        pos2(x0, line_y),
        Align2::LEFT_CENTER,
        text,
        font,
        color,
    );
    ui.interact(painted.expand(2.0), Id::new("skato-key-status"), Sense::hover()).on_hover_text(tip);
    if let Some(label) = action {
        let tip = if label == "APPLY" {
            "Write the detected key into Key and Scale (undoable)."
        } else {
            "Copy the followed key into Key and Scale and switch to MANUAL, so you can edit it."
        };
        if widgets::button(ui, btn, Id::new("skato-key-action"), label, false, tip).clicked() {
            match (label, view.in_effect) {
                ("APPLY", _) => {
                    if let Some(k) = view.proposal {
                        set_manual_key(setter, params, k);
                        if learned {
                            params.clear_learned_scale();
                        }
                    }
                }
                (_, InEffect::Midi { mask, bass }) => copy_to_manual(setter, params, mask, bass),
                (_, InEffect::Detect(k)) => copy_to_manual(setter, params, k.mask, k.root),
                _ => {}
            }
        }
    }
    y + 26.0
}
