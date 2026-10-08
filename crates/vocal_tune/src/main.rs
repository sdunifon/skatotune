//! Standalone entry point with an interactive audio device picker.
//!
//! nice-plug's standalone host takes the audio devices from `--input-device` /
//! `--output-device` flags. To make that friendlier, a plain `cargo standalone`
//! (no device/backend flags, run from a terminal) enumerates the available
//! devices with cpal (the same library and names nice-plug matches against),
//! lets you choose an input and an output, then hands the choice to
//! nice-plug.
//!
//! A pitch corrector needs a microphone, so unlike an effect demo the input
//! defaults to the system's default input. Use headphones: the corrected
//! voice is monitored with only a few milliseconds of latency, and speakers
//! will feed back.
//!
//! The picker lives here at launch, not in the plugin window, by necessity:
//! the editor is host-agnostic (it also runs inside a DAW) and cannot drive
//! the standalone's audio backend. Picking the soundcard is the host's job,
//! and here the host is this launcher.
//!
//! Only compiled with the `standalone` feature. `--backend dummy` runs the GUI
//! without audio (combine with the `demo-telemetry` feature to see the graph).

use std::io::{self, IsTerminal, Write};

use cpal::traits::{DeviceTrait, HostTrait};
use nice_plug::prelude::*;

fn main() {
    let args = resolve_args();
    if !nice_export_standalone_with_args::<vocal_tune::VocalTune, _>(args) {
        std::process::exit(1);
    }
}

/// Start from the real CLI args and, for a plain interactive launch, inject the
/// `--input-device` / `--output-device` flags chosen through the picker.
fn resolve_args() -> Vec<String> {
    let mut args: Vec<String> = std::env::args().collect();

    // Respect explicit device/backend flags and help/version requests, and
    // don't prompt when stdin isn't a terminal (piped input, CI, scripts).
    let explicit = args.iter().skip(1).any(|a| {
        matches!(
            a.as_str(),
            "--input-device"
                | "--output-device"
                | "--backend"
                | "-b"
                | "--help"
                | "-h"
                | "--version"
                | "-V"
        ) || a.starts_with("--input-device=")
            || a.starts_with("--output-device=")
            || a.starts_with("--backend=")
    });
    if explicit || !io::stdin().is_terminal() {
        return args;
    }

    let host = cpal::default_host();

    let inputs = device_names(host.input_devices());
    let default_input = host.default_input_device().and_then(|d| device_name(&d));
    if let Some(name) = pick(
        "input (your microphone)",
        &inputs,
        default_input.as_deref(),
        true,
    ) {
        args.push("--input-device".to_string());
        args.push(name);
        eprintln!("Tip: monitor with headphones; speakers will feed back into the microphone.");
    }

    let outputs = device_names(host.output_devices());
    let default_output = host.default_output_device().and_then(|d| device_name(&d));
    if let Some(name) = pick("output", &outputs, default_output.as_deref(), false) {
        args.push("--output-device".to_string());
        args.push(name);
    }

    args
}

/// The name nice-plug matches `--input-device` / `--output-device` against.
fn device_name(device: &cpal::Device) -> Option<String> {
    device.description().ok().map(|d| d.name().to_string())
}

/// Collect device names, skipping any whose name can't be read.
fn device_names<I>(devices: Result<I, cpal::Error>) -> Vec<String>
where
    I: Iterator<Item = cpal::Device>,
{
    match devices {
        Ok(iter) => iter.filter_map(|d| device_name(&d)).collect(),
        Err(_) => Vec::new(),
    }
}

/// Print a numbered menu and return the chosen device name (`None` = leave this
/// direction unconnected). Pressing Enter accepts the recommended option, the
/// system default device.
fn pick(
    label: &str,
    devices: &[String],
    default: Option<&str>,
    allow_none: bool,
) -> Option<String> {
    if devices.is_empty() {
        eprintln!("No {label} devices available.");
        return None;
    }

    // The 1-based number Enter selects: the system default, else the first.
    let recommended = devices
        .iter()
        .position(|n| default == Some(n.as_str()))
        .map_or(1, |p| p + 1);

    eprintln!("\nSelect {label} device:");
    if allow_none {
        eprintln!("   0) none");
    }
    for (i, name) in devices.iter().enumerate() {
        let marker = if default == Some(name.as_str()) {
            "  (system default)"
        } else {
            ""
        };
        eprintln!("  {:2}) {name}{marker}", i + 1);
    }

    loop {
        eprint!("> [{recommended}] ");
        io::stderr().flush().ok();

        let mut line = String::new();
        // EOF (0 bytes) behaves as if the recommended option was chosen.
        if io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
            eprintln!();
            line = recommended.to_string();
        }

        match interpret_choice(&line, devices.len(), allow_none, recommended) {
            Choice::Device(i) => return devices.get(i).cloned(),
            Choice::None => return None,
            Choice::Invalid => {
                eprintln!(
                    "Enter a number between {} and {}.",
                    if allow_none { 0 } else { 1 },
                    devices.len()
                )
            }
        }
    }
}

/// The outcome of interpreting one line typed at the device menu.
#[derive(Debug, PartialEq)]
enum Choice {
    /// A device was chosen, by 0-based index into the list.
    Device(usize),
    /// The user chose "none" (only offered for the input direction).
    None,
    /// Unparseable or out-of-range input; the menu should re-prompt.
    Invalid,
}

/// Pure selection logic (no I/O) so it can be unit-tested. Empty input selects
/// `recommended`; `0` maps to `None` when `allow_none` is set.
fn interpret_choice(
    input: &str,
    num_devices: usize,
    allow_none: bool,
    recommended: usize,
) -> Choice {
    let trimmed = input.trim();
    let n = if trimmed.is_empty() {
        recommended
    } else {
        match trimmed.parse::<usize>() {
            Ok(n) => n,
            Err(_) => return Choice::Invalid,
        }
    };

    if n == 0 {
        if allow_none {
            Choice::None
        } else {
            Choice::Invalid
        }
    } else if n <= num_devices {
        Choice::Device(n - 1)
    } else {
        Choice::Invalid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_selects_recommended() {
        assert_eq!(interpret_choice("", 3, false, 2), Choice::Device(1));
        assert_eq!(interpret_choice("  \n", 3, true, 1), Choice::Device(0));
    }

    #[test]
    fn number_selects_that_device() {
        assert_eq!(interpret_choice("1", 3, true, 1), Choice::Device(0));
        assert_eq!(interpret_choice("3", 3, true, 1), Choice::Device(2));
        assert_eq!(interpret_choice(" 2 ", 3, false, 1), Choice::Device(1));
    }

    #[test]
    fn zero_is_none_only_when_allowed() {
        assert_eq!(interpret_choice("0", 3, true, 1), Choice::None);
        assert_eq!(interpret_choice("0", 3, false, 1), Choice::Invalid);
    }

    #[test]
    fn out_of_range_or_garbage_is_invalid() {
        assert_eq!(interpret_choice("4", 3, true, 1), Choice::Invalid);
        assert_eq!(interpret_choice("abc", 3, true, 1), Choice::Invalid);
        assert_eq!(interpret_choice("-1", 3, true, 1), Choice::Invalid);
    }
}
