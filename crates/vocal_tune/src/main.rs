//! Standalone entry point with an interactive audio device picker.
//!
//! nice-plug's standalone host takes the audio devices from `--input-device` /
//! `--output-device` flags. To make that friendlier, a plain `cargo standalone`
//! (no device/backend flags, run from a terminal) enumerates the available
//! devices with cpal (the same library and names nice-plug matches against),
//! lets you choose an input and an output, then hands the choice to
//! nice-plug.
//!
//! The launcher also fills in what nice-plug's defaults get wrong for a
//! vocal plugin, unless you set those flags yourself:
//!
//! - **Channel layout.** nice-plug only opens an input whose channel count
//!   matches the layout exactly, and the default layout is stereo, which a
//!   mono microphone does not offer. A 1-channel input gets the Mono to
//!   Stereo layout, a 2-channel one Stereo. Inputs with neither (4- or
//!   8-channel interfaces) are refused with an explanation.
//! - **Monitoring latency.** nice-plug's defaults (512-sample periods and a
//!   150 ms input-to-output buffer) add about 175 ms. The launcher asks for
//!   128-sample periods and a 15 ms input buffer when the devices support it.
//!   What you hear is then about input buffer + 2 periods + the plugin
//!   latency: roughly 15 + 5 + 4.4 ≈ 25 ms at 48 kHz in Live mode. Pass
//!   `--period-size 512 --input-latency 0.15` to go back to the safe defaults
//!   if you hear crackles.
//! - **Backend.** After an interactive pick on macOS the CoreAudio backend is
//!   requested explicitly, so a device that cannot be opened fails loudly
//!   instead of silently falling back to the dummy backend (no audio).
//!
//! A pitch corrector needs a microphone, so unlike an effect demo the input
//! defaults to the system's default input. Use headphones: speakers will feed
//! back into the microphone.
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

/// nice-plug's default sample rate (`--sample-rate`).
const DEFAULT_RATE: u32 = 48_000;
/// Period sizes tried for low-latency monitoring, best first.
const PERIOD_CHOICES: [u32; 4] = [128, 256, 64, 512];
/// Input-to-output buffer, seconds: at least this, and at least 4 periods
/// (input and output may be separate devices on independent clocks).
const MIN_INPUT_LATENCY_S: f32 = 0.015;
/// Capacity of that buffer, seconds (nice-plug wants at least 2× the latency).
const INPUT_CAPACITY_S: f32 = 0.1;
/// 1-based `--audio-layout` indices of `VocalTune::AUDIO_IO_LAYOUTS`.
const LAYOUT_STEREO: &str = "1";
const LAYOUT_MONO_TO_STEREO: &str = "3";

/// One supported stream configuration of a device (cpal's
/// `SupportedStreamConfigRange`, reduced to what the planner needs).
#[derive(Clone, Copy, Debug, PartialEq)]
struct ConfigRange {
    channels: u16,
    min_rate: u32,
    max_rate: u32,
    /// Supported period sizes; `None` when the device does not say (nice-plug
    /// rejects those configurations).
    buffer: Option<(u32, u32)>,
}

impl ConfigRange {
    fn rate_ok(&self, rate: u32) -> bool {
        (self.min_rate..=self.max_rate).contains(&rate)
    }

    fn period_ok(&self, period: u32) -> bool {
        self.buffer
            .is_some_and(|(lo, hi)| (lo..=hi).contains(&period))
    }
}

/// Start from the real CLI args, run the interactive picker for a plain
/// launch, then add the layout / latency / backend flags described in the
/// module docs.
fn resolve_args() -> Vec<String> {
    let mut args: Vec<String> = std::env::args().collect();
    if has_flag(&args, &["--help", "-h", "--version", "-V"]) {
        return args;
    }
    let rate = flag_value(&args, &["--sample-rate", "-r"])
        .and_then(|v| v.parse::<f32>().ok())
        .map_or(DEFAULT_RATE, |r| r as u32);
    let host = cpal::default_host();

    // Respect explicit device/backend flags, and don't prompt when stdin
    // isn't a terminal (piped input, CI, scripts).
    let explicit = has_flag(
        &args,
        &["--input-device", "--output-device", "--backend", "-b"],
    );
    let mut picked = false;
    if !explicit && io::stdin().is_terminal() {
        picked = true;
        if let Some(name) = pick_input(&host, rate) {
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
    }

    let backend = flag_value(&args, &["--backend", "-b"]);
    let device_backend = !matches!(backend.as_deref(), Some("dummy" | "jack"));
    if device_backend {
        let input = flag_value(&args, &["--input-device"]);
        let output = flag_value(&args, &["--output-device"]);
        let in_cfgs = input
            .as_deref()
            .map(|n| input_configs(&host, n))
            .unwrap_or_default();
        let out_cfgs = output_configs(&host, output.as_deref());
        let extra = plan_args(&args, rate, input.is_some(), &in_cfgs, &out_cfgs);
        args.extend(extra);
    }
    if picked && cfg!(target_os = "macos") && !has_flag(&args, &["--backend", "-b"]) {
        args.push("--backend".to_string());
        args.push("core-audio".to_string());
    }
    if has_flag(&args, &["--input-device"]) {
        let latency = flag_value(&args, &["--input-latency"])
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.15);
        let period = flag_value(&args, &["--period-size", "-p"])
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(512.0);
        eprintln!(
            "Monitoring delay ≈ {:.0} ms input buffer + {:.0} ms device periods + the plugin latency shown in its window.",
            latency * 1000.0,
            2.0 * period / rate as f32 * 1000.0
        );
    }
    args
}

/// The flags to add for this launch (pure; the tests drive it with fake
/// device capabilities). Flags the user already gave are never touched.
fn plan_args(
    args: &[String],
    rate: u32,
    has_input: bool,
    in_cfgs: &[ConfigRange],
    out_cfgs: &[ConfigRange],
) -> Vec<String> {
    let mut extra = Vec::new();
    let mut in_channels = None;
    if has_input {
        in_channels = usable_input_channels(in_cfgs, rate);
        if !has_flag(args, &["--audio-layout", "-l"]) {
            let layout = match in_channels {
                Some(2) => Some(LAYOUT_STEREO),
                Some(_) => Some(LAYOUT_MONO_TO_STEREO),
                None => None,
            };
            if let Some(l) = layout {
                extra.push("--audio-layout".to_string());
                extra.push(l.to_string());
            }
        }
    }
    if !has_flag(args, &["--period-size", "-p"]) {
        let input_ok = |p: u32| match in_channels {
            Some(ch) => in_cfgs
                .iter()
                .any(|c| c.channels == ch && c.rate_ok(rate) && c.period_ok(p)),
            None => !has_input,
        };
        let output_ok = |p: u32| out_cfgs.iter().any(|c| c.rate_ok(rate) && c.period_ok(p));
        if let Some(period) = PERIOD_CHOICES
            .iter()
            .copied()
            .find(|&p| input_ok(p) && output_ok(p))
        {
            extra.push("--period-size".to_string());
            extra.push(period.to_string());
            if has_input {
                let latency = MIN_INPUT_LATENCY_S.max(4.0 * period as f32 / rate.max(1) as f32);
                if !has_flag(args, &["--input-latency"]) {
                    extra.push("--input-latency".to_string());
                    extra.push(format!("{latency:.3}"));
                }
                if !has_flag(args, &["--input-capacity"]) {
                    extra.push("--input-capacity".to_string());
                    extra.push(format!("{:.3}", INPUT_CAPACITY_S.max(4.0 * latency)));
                }
            }
        }
    }
    extra
}

/// Channel count nice-plug can open the input with at `rate`: 2 (Stereo)
/// if offered, else 1 (Mono to Stereo). `None` for devices with neither.
fn usable_input_channels(cfgs: &[ConfigRange], rate: u32) -> Option<u16> {
    let offers = |ch: u16| {
        cfgs.iter()
            .any(|c| c.channels == ch && c.rate_ok(rate) && c.buffer.is_some())
    };
    [2, 1].into_iter().find(|&ch| offers(ch))
}

/// Whether any of `names` appears as a flag (`--flag value` or `--flag=value`).
fn has_flag(args: &[String], names: &[&str]) -> bool {
    args.iter().skip(1).any(|a| {
        names
            .iter()
            .any(|n| a == n || a.strip_prefix(n).is_some_and(|rest| rest.starts_with('=')))
    })
}

/// The value of the last occurrence of any of `names`.
fn flag_value(args: &[String], names: &[&str]) -> Option<String> {
    let mut found = None;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        for n in names {
            if a == n {
                found = it.clone().next().cloned();
            } else if let Some(v) = a.strip_prefix(n).and_then(|r| r.strip_prefix('=')) {
                found = Some(v.to_string());
            }
        }
    }
    found
}

/// Pick the input device, re-prompting while the choice has no channel count
/// nice-plug can open.
fn pick_input(host: &cpal::Host, rate: u32) -> Option<String> {
    let inputs = device_names(host.input_devices());
    let default_input = host.default_input_device().and_then(|d| device_name(&d));
    loop {
        let name = pick(
            "input (your microphone)",
            &inputs,
            default_input.as_deref(),
            true,
        )?;
        let cfgs = input_configs(host, &name);
        if usable_input_channels(&cfgs, rate).is_some() {
            return Some(name);
        }
        let mut counts: Vec<u16> = cfgs.iter().map(|c| c.channels).collect();
        counts.sort_unstable();
        counts.dedup();
        eprintln!(
            "'{name}' offers {counts:?}-channel input at {rate} Hz, but the standalone can only open \
             a 1- or 2-channel input (exact match). Pick another input, 0 for none, or build an \
             aggregate device in Audio MIDI Setup."
        );
    }
}

/// Supported input configurations of the named device (empty if unknown).
fn input_configs(host: &cpal::Host, name: &str) -> Vec<ConfigRange> {
    let Ok(mut devices) = host.input_devices() else {
        return Vec::new();
    };
    devices
        .find(|d| device_name(d).as_deref() == Some(name))
        .and_then(|d| d.supported_input_configs().ok())
        .map(|it| it.map(|c| config_range(&c)).collect())
        .unwrap_or_default()
}

/// Supported output configurations of the named (or default) output device.
fn output_configs(host: &cpal::Host, name: Option<&str>) -> Vec<ConfigRange> {
    let device = match name {
        Some(n) => host
            .output_devices()
            .ok()
            .and_then(|mut it| it.find(|d| device_name(d).as_deref() == Some(n))),
        None => host.default_output_device(),
    };
    device
        .and_then(|d| d.supported_output_configs().ok())
        .map(|it| it.map(|c| config_range(&c)).collect())
        .unwrap_or_default()
}

fn config_range(c: &cpal::SupportedStreamConfigRange) -> ConfigRange {
    ConfigRange {
        channels: c.channels(),
        min_rate: c.min_sample_rate(),
        max_rate: c.max_sample_rate(),
        buffer: match c.buffer_size() {
            cpal::SupportedBufferSize::Range { min, max } => Some((*min, *max)),
            cpal::SupportedBufferSize::Unknown => None,
        },
    }
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

    fn args(extra: &[&str]) -> Vec<String> {
        std::iter::once("vocal_tune_standalone")
            .chain(extra.iter().copied())
            .map(String::from)
            .collect()
    }

    /// What this machine's devices report (see HI-5): a mono built-in mic, a
    /// 4-channel loopback device and a stereo output, all 44.1-96 kHz with
    /// 15-4096-sample periods.
    fn dev(channels: u16) -> ConfigRange {
        ConfigRange {
            channels,
            min_rate: 44_100,
            max_rate: 96_000,
            buffer: Some((15, 4096)),
        }
    }

    fn value_of(extra: &[String], flag: &str) -> Option<String> {
        extra
            .iter()
            .position(|a| a == flag)
            .and_then(|i| extra.get(i + 1).cloned())
    }

    #[test]
    fn mono_microphone_gets_the_mono_to_stereo_layout() {
        let extra = plan_args(&args(&[]), 48_000, true, &[dev(1)], &[dev(2)]);
        assert_eq!(value_of(&extra, "--audio-layout").as_deref(), Some("3"));
        let stereo = plan_args(&args(&[]), 48_000, true, &[dev(1), dev(2)], &[dev(2)]);
        assert_eq!(value_of(&stereo, "--audio-layout").as_deref(), Some("1"));
        // The user's own layout wins.
        let own = plan_args(&args(&["-l", "2"]), 48_000, true, &[dev(1)], &[dev(2)]);
        assert_eq!(value_of(&own, "--audio-layout"), None);
    }

    #[test]
    fn multichannel_only_inputs_are_not_usable() {
        assert_eq!(usable_input_channels(&[dev(4)], 48_000), None);
        assert_eq!(usable_input_channels(&[dev(8), dev(1)], 48_000), Some(1));
        // A rate the device cannot run does not count.
        assert_eq!(usable_input_channels(&[dev(1)], 192_000), None);
        let extra = plan_args(&args(&[]), 48_000, true, &[dev(4)], &[dev(2)]);
        assert_eq!(value_of(&extra, "--audio-layout"), None);
    }

    #[test]
    fn monitoring_latency_defaults_are_low() {
        let extra = plan_args(&args(&[]), 48_000, true, &[dev(1)], &[dev(2)]);
        assert_eq!(value_of(&extra, "--period-size").as_deref(), Some("128"));
        let latency: f32 = value_of(&extra, "--input-latency")
            .unwrap()
            .parse()
            .unwrap();
        let capacity: f32 = value_of(&extra, "--input-capacity")
            .unwrap()
            .parse()
            .unwrap();
        assert!((0.010..=0.02).contains(&latency), "{latency}");
        assert!(capacity >= 2.0 * latency);
        // Total monitoring delay well under nice-plug's ~175 ms default.
        let total_ms = latency * 1000.0 + 2.0 * 128.0 / 48.0 + 4.4;
        assert!(total_ms < 30.0, "{total_ms}");
    }

    #[test]
    fn user_latency_flags_and_unsupported_periods_are_respected() {
        let own = plan_args(
            &args(&["--period-size=256", "--input-latency", "0.05"]),
            48_000,
            true,
            &[dev(1)],
            &[dev(2)],
        );
        assert_eq!(value_of(&own, "--period-size"), None);
        assert_eq!(value_of(&own, "--input-latency"), None);
        // A device that only does 512+ gets 512 (nice-plug's own default).
        let big = ConfigRange {
            buffer: Some((512, 4096)),
            ..dev(1)
        };
        let extra = plan_args(&args(&[]), 48_000, true, &[big], &[dev(2)]);
        assert_eq!(value_of(&extra, "--period-size").as_deref(), Some("512"));
        // Nothing fits: leave nice-plug's defaults alone.
        let odd = ConfigRange {
            buffer: Some((1000, 1000)),
            ..dev(1)
        };
        let none = plan_args(&args(&[]), 48_000, true, &[odd], &[dev(2)]);
        assert_eq!(value_of(&none, "--period-size"), None);
        assert_eq!(value_of(&none, "--input-latency"), None);
    }

    #[test]
    fn flag_parsing() {
        let a = args(&[
            "-b",
            "dummy",
            "--input-device=Mic",
            "--sample-rate",
            "44100",
        ]);
        assert!(has_flag(&a, &["--backend", "-b"]));
        assert!(has_flag(&a, &["--input-device"]));
        assert!(!has_flag(&a, &["--input-latency"]));
        assert_eq!(
            flag_value(&a, &["--backend", "-b"]).as_deref(),
            Some("dummy")
        );
        assert_eq!(flag_value(&a, &["--input-device"]).as_deref(), Some("Mic"));
        assert_eq!(
            flag_value(&a, &["--sample-rate", "-r"]).as_deref(),
            Some("44100")
        );
        // `--input-devices` is not `--input-device`.
        assert!(!has_flag(&args(&["--input-devicex"]), &["--input-device"]));
    }
}
