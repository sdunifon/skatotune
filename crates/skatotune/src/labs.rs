//! Labs: opt-in editor experiments, switched in the header's LABS menu.
//!
//! An experiment is a GUI-only idea we want to try on real sessions before it
//! earns a permanent place in the main UI. The switches are saved with the
//! project (the `labs` persistent field of
//! [`SkatotuneParams`](crate::params::SkatotuneParams)), never automated, and
//! never read on the audio thread.
//!
//! # Adding an experiment
//!
//! 1. Add a `bool` field to [`LabsSettings`] (and its default in the
//!    `Default` impl).
//! 2. Add a row to [`EXPERIMENTS`]: name, one-line description, and the
//!    accessor pair for the new field.
//! 3. Read the flag where the editor draws the feature.
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

use serde::{Deserialize, Serialize};

/// The experiment switches. Copy, and small enough to read every frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LabsSettings {
    /// Shade the area between the input and output traces by how far the
    /// plugin moved the pitch.
    pub correction_ribbon: bool,
    /// Colour the output trace by how far the plugin moved the pitch.
    pub heat_line: bool,
}

impl Default for LabsSettings {
    fn default() -> Self {
        Self {
            correction_ribbon: true,
            heat_line: false,
        }
    }
}

impl LabsSettings {
    /// Any experiment that colours the graph by correction strength is on
    /// (the graph legend then shows the colour key).
    pub fn shows_correction_colour(&self) -> bool {
        self.correction_ribbon || self.heat_line
    }
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
pub const EXPERIMENTS: [Experiment; 2] = [
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

    #[test]
    fn defaults_ribbon_on_heat_line_off() {
        let d = LabsSettings::default();
        assert!(d.correction_ribbon);
        assert!(!d.heat_line);
        assert!(d.shows_correction_colour());
        let off = LabsSettings {
            correction_ribbon: false,
            heat_line: false,
        };
        assert!(!off.shows_correction_colour());
    }

    /// Saved with the project through nice-plug's own (de)serializer.
    #[test]
    fn round_trips_through_the_persist_serializer() {
        let s = LabsSettings {
            correction_ribbon: false,
            heat_line: true,
        };
        let json = serialize_field(&s).unwrap();
        let back: LabsSettings = deserialize_field(&json).unwrap();
        assert_eq!(back, s);
    }

    /// Forward and backward compatible: an older project (fields missing)
    /// gets the defaults, a newer one (fields we no longer know) still loads.
    #[test]
    fn missing_and_unknown_fields_load() {
        let old: LabsSettings = deserialize_field("{}").unwrap();
        assert_eq!(old, LabsSettings::default());
        let partial: LabsSettings = deserialize_field(r#"{"heat_line":true}"#).unwrap();
        assert!(partial.heat_line && partial.correction_ribbon);
        let newer: LabsSettings = deserialize_field(
            r#"{"correction_ribbon":false,"heat_line":false,"retired_idea":true,"x":{"y":[1]}}"#,
        )
        .unwrap();
        assert!(!newer.correction_ribbon && !newer.heat_line);
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
        let all_off = LabsSettings {
            correction_ribbon: false,
            heat_line: false,
        };
        for (i, row) in EXPERIMENTS.iter().enumerate() {
            let mut l = all_off;
            (row.set)(&mut l, true);
            assert!((row.get)(&l), "{}", row.name);
            for (j, other) in EXPERIMENTS.iter().enumerate() {
                if i != j {
                    assert!(!(other.get)(&l), "{} also set {}", row.name, other.name);
                }
            }
            (row.set)(&mut l, false);
            assert_eq!(l, all_off, "{}", row.name);
            assert!(!row.description.is_empty());
        }
    }
}
