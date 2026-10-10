//! Labs: opt-in experiments, switched in the header's LABS menu.
//!
//! An experiment is an idea we want to try on real sessions before it earns
//! a permanent place in the main UI. The switches are saved with the project
//! (the `labs` persistent field of
//! [`SkatotuneParams`](crate::params::SkatotuneParams)) and never automated.
//!
//! Most experiments only change the editor and are never read on the audio
//! thread. Two change the sound: [`LabsSettings::learned_note_hold`] and
//! [`LabsSettings::follow_singer`] (`docs/research/note-intent.md`). Their
//! bits ([`LabsSettings::sound_bits`]) are mirrored into a lock-free
//! persisted value (`labs-sound` in the saved state, like the learned scale),
//! which the audio thread reads every block ([`apply_sound`]); so they apply
//! with the editor closed and right after a project is loaded, and the mirror
//! is what the menu shows for them.
//!
//! # Adding an experiment
//!
//! 1. Add a `bool` field to [`LabsSettings`] (and its default in
//!    [`LabsSettings::DEFAULT`]).
//! 2. Add a row to [`EXPERIMENTS`]: name, one-line description, and the
//!    accessor pair for the new field.
//! 3. Read the flag where the editor draws the feature. A flag the audio
//!    thread needs gets a bit in [`LabsSettings::sound_bits`] and is applied
//!    in [`apply_sound`] instead (never lock on the audio thread).
//!
//! # Adding a choice
//!
//! Some settings pick one of a few options instead of switching something on:
//! the [`CorrectionTheme`] of the correction colouring is the example.
//!
//! 1. Add an enum (with a `#[default]` variant and `#[serde(rename_all =
//!    "kebab-case")]`, its stored names) and a field of that type to
//!    [`LabsSettings`].
//! 2. Read the stored value *leniently*, with a `deserialize_with` function
//!    like `lenient_theme`: a name this version does not know, say an option
//!    a newer version added, must load as the default. A plain derive would
//!    reject the whole struct and every Labs switch would fall back to its
//!    default with it.
//! 3. Draw the selector in `editor::labs_menu`, below the switch rows.
//!
//! # Promoting or retiring one
//!
//! A useful experiment gets promoted: it moves into the main UI (on by
//! default, or behind a regular control) and its field and row are deleted
//! here. One that did not work out is simply deleted. Either way, projects
//! saved while it existed still load: unknown fields are ignored and missing
//! ones take their default (`#[serde(default)]`), so the stored JSON never has
//! to be migrated.
//!
//! **Never reuse a field name**, not even a retired one, for a field of a
//! different type (say a `bool` that becomes an enum): old projects still
//! carry the old value under that name, serde would reject the whole struct,
//! and every Labs switch would silently fall back to its default. Give the
//! new field a new name instead.

use serde::{Deserialize, Deserializer, Serialize};
use skato_core::{ReferenceMode, Settings};

/// Colours of the correction-strength displays: the ribbon, the heat line and
/// the key in the graph legend ([`LabsSettings::correction_theme`]). Which
/// colours each theme stands for lives with the drawing code
/// (`editor::correction`); this is only the choice, as it is saved.
///
/// The stored name is the kebab-case variant name ([`Self::id`]), which the
/// `render_gui` example also takes as `--theme=`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CorrectionTheme {
    /// The accent's own hue, lighter = more. The original look.
    #[default]
    Teal,
    /// Traffic light: green for small corrections through yellow and orange
    /// to red.
    GreenRed,
    /// Blue through teal and green to yellow; perceptually even.
    Viridis,
    /// Dark red through orange to yellow; perceptually even.
    Inferno,
    /// Cool to warm.
    BlueOrange,
}

impl CorrectionTheme {
    /// Every theme, in menu order.
    pub const ALL: [Self; 5] = [
        Self::Teal,
        Self::GreenRed,
        Self::Viridis,
        Self::Inferno,
        Self::BlueOrange,
    ];

    /// The stored name (see the type docs).
    pub const fn id(self) -> &'static str {
        match self {
            Self::Teal => "teal",
            Self::GreenRed => "green-red",
            Self::Viridis => "viridis",
            Self::Inferno => "inferno",
            Self::BlueOrange => "blue-orange",
        }
    }

    /// The theme stored under `id`, if there is one.
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.id() == id)
    }

    /// Full name, for tooltips.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Teal => "Teal",
            Self::GreenRed => "Green \u{2192} Red",
            Self::Viridis => "Viridis",
            Self::Inferno => "Inferno / Heat",
            Self::BlueOrange => "Blue \u{2192} Orange",
        }
    }

    /// Short name, for the selector's buttons.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Teal => "Teal",
            Self::GreenRed => "Green-red",
            Self::Viridis => "Viridis",
            Self::Inferno => "Inferno",
            Self::BlueOrange => "Blue-orange",
        }
    }

    /// One line on what the theme is for, shown under the selector.
    pub const fn note(self) -> &'static str {
        match self {
            Self::Teal => "The accent's own hue; brighter = more.",
            Self::GreenRed => {
                "Traffic light. Red-green colour-blind users may not tell the ends apart."
            }
            Self::Viridis => "Blue to yellow via teal and green; colour-blind safe.",
            Self::Inferno => "Dark red to yellow; colour-blind safe.",
            Self::BlueOrange => "Cool to warm; colour-blind safe.",
        }
    }
}

/// Read a theme name, taking anything this version does not know (an option
/// a newer version added, or a value of another type) as the default. A theme
/// the user cannot see is a small loss; a rejected struct would reset every
/// Labs switch (see the module docs, "Adding a choice").
fn lenient_theme<'de, D: Deserializer<'de>>(d: D) -> Result<CorrectionTheme, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Stored {
        Known(CorrectionTheme),
        Unknown(serde::de::IgnoredAny),
    }
    Ok(match Stored::deserialize(d)? {
        Stored::Known(theme) => theme,
        Stored::Unknown(_) => CorrectionTheme::default(),
    })
}

/// The experiment switches and choices. Copy, and small enough to read every
/// frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LabsSettings {
    /// Shade the area between the input and output traces by how far the
    /// plugin moved the pitch.
    pub correction_ribbon: bool,
    /// Colour the output trace by how far the plugin moved the pitch.
    pub heat_line: bool,
    /// Colours of the two experiments above and of the legend's key.
    #[serde(deserialize_with = "lenient_theme")]
    pub correction_theme: CorrectionTheme,
    /// Sound: learned note hold ([`skato_core::Settings::note_hold`]). In the
    /// HQ and Max tiers, holds the current note when a switch looks like
    /// wobble or a phrase-end droop (fewer note flips, at the cost of some
    /// short notes); Eco and MIDI targets are unaffected.
    pub learned_note_hold: bool,
    /// Sound: follow singer's tuning ([`skato_core::ReferenceMode::FollowSinger`]).
    /// Corrects to the singer's own reference instead of A440, for
    /// consistently detuned material; the output is off A440 by design.
    pub follow_singer: bool,
}

impl LabsSettings {
    /// The defaults, as a constant (what `Default` returns), so a test or a
    /// `const` can start from them: `Self { heat_line: true, ..Self::DEFAULT }`.
    pub const DEFAULT: Self = Self {
        correction_ribbon: true,
        heat_line: false,
        correction_theme: CorrectionTheme::Teal,
        learned_note_hold: false,
        follow_singer: false,
    };
}

impl Default for LabsSettings {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Bits of [`LabsSettings::sound_bits`]. Never reuse a retired bit.
const SOUND_NOTE_HOLD: u32 = 1 << 0;
const SOUND_FOLLOW_SINGER: u32 = 1 << 1;

impl LabsSettings {
    /// Any experiment that colours the graph by correction strength is on
    /// (the graph legend then shows the colour key).
    pub fn shows_correction_colour(&self) -> bool {
        self.correction_ribbon || self.heat_line
    }

    /// The experiments that change the sound, as the bits the audio thread
    /// reads (see the module docs).
    pub const fn sound_bits(&self) -> u32 {
        (if self.learned_note_hold { SOUND_NOTE_HOLD } else { 0 })
            | (if self.follow_singer { SOUND_FOLLOW_SINGER } else { 0 })
    }

    /// These settings with the sound experiments taken from `bits`.
    pub const fn with_sound_bits(self, bits: u32) -> Self {
        Self {
            learned_note_hold: bits & SOUND_NOTE_HOLD != 0,
            follow_singer: bits & SOUND_FOLLOW_SINGER != 0,
            ..self
        }
    }
}

/// Apply the sound experiments of `bits` ([`LabsSettings::sound_bits`]) to
/// the engine settings. Audio thread: pure, allocation-free.
pub fn apply_sound(settings: &mut Settings, bits: u32) {
    settings.note_hold = bits & SOUND_NOTE_HOLD != 0;
    settings.reference = if bits & SOUND_FOLLOW_SINGER != 0 {
        ReferenceMode::FollowSinger
    } else {
        ReferenceMode::A440
    };
}

/// One row of the LABS menu.
#[derive(Clone, Copy)]
pub struct Experiment {
    pub name: &'static str,
    /// One line, shown under the name.
    pub description: &'static str,
    pub get: fn(&LabsSettings) -> bool,
    pub set: fn(&mut LabsSettings, bool),
}

/// Every experiment, in menu order.
pub const EXPERIMENTS: [Experiment; 4] = [
    Experiment {
        name: "Correction ribbon",
        description: "Shades the correction under the output; brighter = more.",
        get: |l| l.correction_ribbon,
        set: |l, on| l.correction_ribbon = on,
    },
    Experiment {
        name: "Heat-tinted output line",
        description: "Colours the output trace by correction strength.",
        get: |l| l.heat_line,
        set: |l, on| l.heat_line = on,
    },
    Experiment {
        name: "Learned note hold",
        description: "Holds notes through wobble and phrase-end droops.",
        get: |l| l.learned_note_hold,
        set: |l, on| l.learned_note_hold = on,
    },
    Experiment {
        name: "Follow singer's tuning",
        description: "Corrects to the singer's own tuning, not A440.",
        get: |l| l.follow_singer,
        set: |l, on| l.follow_singer = on,
    },
];

/// Footer of the LABS menu, one sentence per line.
pub const LABS_FOOTER: [&str; 2] = [
    "Experiments may change or be removed.",
    "Useful ones get promoted to the main UI.",
];

#[cfg(test)]
mod tests {
    use super::*;
    use nice_plug::params::persist::{deserialize_field, serialize_field};

    const ALL_OFF: LabsSettings = LabsSettings {
        correction_ribbon: false,
        heat_line: false,
        ..LabsSettings::DEFAULT
    };

    #[test]
    fn defaults_ribbon_on_heat_line_off_teal() {
        let d = LabsSettings::default();
        assert!(d.correction_ribbon);
        assert!(!d.heat_line);
        assert_eq!(d.correction_theme, CorrectionTheme::Teal);
        assert_eq!(d, LabsSettings::DEFAULT);
        assert!(d.shows_correction_colour());
        assert!(!ALL_OFF.shows_correction_colour());
        // The theme alone colours nothing.
        assert!(
            !LabsSettings {
                correction_theme: CorrectionTheme::Viridis,
                ..ALL_OFF
            }
            .shows_correction_colour()
        );
    }

    /// Saved with the project through nice-plug's own (de)serializer.
    #[test]
    fn round_trips_through_the_persist_serializer() {
        let s = LabsSettings {
            correction_ribbon: false,
            heat_line: true,
            ..LabsSettings::DEFAULT
        };
        let json = serialize_field(&s).unwrap();
        let back: LabsSettings = deserialize_field(&json).unwrap();
        assert_eq!(back, s);
    }

    /// Every theme survives a save and a load, under the name the
    /// `--theme=` flag and the docs use.
    #[test]
    fn every_theme_round_trips_under_its_id() {
        for theme in CorrectionTheme::ALL {
            let s = LabsSettings {
                correction_ribbon: false,
                heat_line: true,
                correction_theme: theme,
                ..LabsSettings::DEFAULT
            };
            let json = serialize_field(&s).unwrap();
            assert!(
                json.contains(&format!(r#""correction_theme":"{}""#, theme.id())),
                "{json}"
            );
            let back: LabsSettings = deserialize_field(&json).unwrap();
            assert_eq!(back, s, "{theme:?}");
            assert_eq!(
                serialize_field(&theme).unwrap(),
                format!("\"{}\"", theme.id())
            );
        }
    }

    /// Forward and backward compatible: an older project (fields missing)
    /// gets the defaults, a newer one (fields we no longer know) still loads.
    #[test]
    fn missing_and_unknown_fields_load() {
        let old: LabsSettings = deserialize_field("{}").unwrap();
        assert_eq!(old, LabsSettings::default());
        let partial: LabsSettings = deserialize_field(r#"{"heat_line":true}"#).unwrap();
        assert!(partial.heat_line && partial.correction_ribbon);
        assert_eq!(partial.correction_theme, CorrectionTheme::Teal);
        let newer: LabsSettings = deserialize_field(
            r#"{"correction_ribbon":false,"heat_line":false,"retired_idea":true,"x":{"y":[1]}}"#,
        )
        .unwrap();
        assert!(!newer.correction_ribbon && !newer.heat_line);
    }

    /// A project saved before the theme existed has no `correction_theme`:
    /// Teal, with the switches it did save.
    #[test]
    fn a_project_without_a_theme_loads_as_teal() {
        let before_themes: LabsSettings =
            deserialize_field(r#"{"correction_ribbon":false,"heat_line":true}"#).unwrap();
        assert_eq!(
            before_themes,
            LabsSettings {
                correction_ribbon: false,
                heat_line: true,
                correction_theme: CorrectionTheme::Teal,
                ..LabsSettings::DEFAULT
            }
        );
    }

    /// A theme this version does not know (or a value of the wrong type)
    /// is Teal, and does not take the switches down with it.
    #[test]
    fn an_unknown_theme_loads_as_teal_and_keeps_the_switches() {
        for stored in [
            r#""plasma""#,
            r#""""#,
            r#""Teal""#,
            r#""GREEN-RED""#,
            "5",
            "-1.5",
            "true",
            "null",
            r#"["viridis"]"#,
            r#"{"viridis":1}"#,
        ] {
            let json = format!(
                r#"{{"correction_ribbon":false,"heat_line":true,"correction_theme":{stored}}}"#
            );
            let s: LabsSettings = deserialize_field(&json).unwrap();
            assert_eq!(
                s,
                LabsSettings {
                    correction_ribbon: false,
                    heat_line: true,
                    correction_theme: CorrectionTheme::Teal,
                    ..LabsSettings::DEFAULT
                },
                "{stored}"
            );
        }
        // And a known one right after an unknown field order still reads.
        let s: LabsSettings =
            deserialize_field(r#"{"correction_theme":"inferno","heat_line":true}"#).unwrap();
        assert_eq!(s.correction_theme, CorrectionTheme::Inferno);
    }

    #[test]
    fn theme_ids_are_unique_and_found_again() {
        let mut ids: Vec<&str> = CorrectionTheme::ALL.iter().map(|t| t.id()).collect();
        for theme in CorrectionTheme::ALL {
            assert_eq!(CorrectionTheme::from_id(theme.id()), Some(theme));
            assert!(
                theme
                    .id()
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c == '-'),
                "{}",
                theme.id()
            );
            assert!(!theme.name().is_empty() && !theme.label().is_empty());
            assert!(!theme.note().is_empty());
        }
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), CorrectionTheme::ALL.len());
        assert_eq!(CorrectionTheme::from_id("Teal"), None);
        assert_eq!(CorrectionTheme::from_id(""), None);
        assert_eq!(CorrectionTheme::default(), CorrectionTheme::ALL[0]);
    }

    /// The theme with the red-green caveat says so, in the line the menu
    /// shows under the selector.
    #[test]
    fn green_red_warns_about_red_green_colour_blindness() {
        let note = CorrectionTheme::GreenRed.note().to_lowercase();
        assert!(note.contains("red-green colour-blind"), "{note}");
        assert!(note.contains("ends apart"), "{note}");
    }

    /// The sound experiments are off by default, round-trip through their
    /// bits, and set exactly their engine setting.
    #[test]
    fn sound_experiments_map_to_their_engine_settings() {
        let d = LabsSettings::default();
        assert!(!d.learned_note_hold && !d.follow_singer);
        assert_eq!(d.sound_bits(), 0);
        let mut s = Settings::default();
        apply_sound(&mut s, d.sound_bits());
        assert_eq!(s, Settings::default(), "off is the engine default");
        for (hold, follow) in [(true, false), (false, true), (true, true)] {
            let l = LabsSettings {
                learned_note_hold: hold,
                follow_singer: follow,
                ..LabsSettings::DEFAULT
            };
            assert_eq!(LabsSettings::DEFAULT.with_sound_bits(l.sound_bits()), l);
            let mut s = Settings::default();
            apply_sound(&mut s, l.sound_bits());
            assert_eq!(s.note_hold, hold);
            assert_eq!(s.reference == ReferenceMode::FollowSinger, follow);
            assert_eq!(
                Settings {
                    note_hold: false,
                    reference: ReferenceMode::A440,
                    ..s
                },
                Settings::default(),
                "nothing else changes"
            );
        }
        // The editor experiments have no sound bits.
        let gui = LabsSettings {
            correction_ribbon: true,
            heat_line: true,
            correction_theme: CorrectionTheme::Inferno,
            ..LabsSettings::DEFAULT
        };
        assert_eq!(gui.sound_bits(), 0);
        // Unknown bits (a newer version's) are ignored.
        assert_eq!(LabsSettings::DEFAULT.with_sound_bits(!0).sound_bits(), 3);
    }

    /// A project saved before the sound experiments existed loads with them
    /// off; one saved with them keeps them.
    #[test]
    fn sound_experiments_round_trip_and_default_off() {
        let old: LabsSettings =
            deserialize_field(r#"{"correction_ribbon":false,"heat_line":true}"#).unwrap();
        assert!(!old.learned_note_hold && !old.follow_singer);
        let s = LabsSettings {
            learned_note_hold: true,
            follow_singer: true,
            ..LabsSettings::DEFAULT
        };
        let back: LabsSettings = deserialize_field(&serialize_field(&s).unwrap()).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn footer_says_what_labs_is() {
        assert_eq!(
            LABS_FOOTER.join(" "),
            "Experiments may change or be removed. Useful ones get promoted to the main UI."
        );
    }

    /// Each row's accessors touch exactly their own field (catches a
    /// copy-pasted row).
    #[test]
    fn every_row_toggles_only_its_own_field() {
        for (i, row) in EXPERIMENTS.iter().enumerate() {
            let mut l = ALL_OFF;
            (row.set)(&mut l, true);
            assert!((row.get)(&l), "{}", row.name);
            for (j, other) in EXPERIMENTS.iter().enumerate() {
                if i != j {
                    assert!(!(other.get)(&l), "{} also set {}", row.name, other.name);
                }
            }
            assert_eq!(
                l.correction_theme, ALL_OFF.correction_theme,
                "{} touched the theme",
                row.name
            );
            (row.set)(&mut l, false);
            assert_eq!(l, ALL_OFF, "{}", row.name);
            assert!(!row.description.is_empty());
        }
    }
}
