# Skatotune

**Open-source real-time vocal pitch correction, written in Rust.** Skatotune is a VST3 and CLAP plugin (plus a standalone app) that tunes a singing voice live. It has a formant-preserving natural mode, a classic hard-tune mode, and a scrolling pitch graph that shows exactly what it is doing to your voice.

<p align="center">
  <img src="docs/images/hero.png" width="880" alt="Skatotune's editor: the scrolling pitch graph with the sung pitch (white), the vibrato-free centre (blue), the corrected output (green) over target-note bars, with the correction shaded underneath; on the left the Key Source switch (Manual, MIDI, Detect), key and scale, Auto-Key's status (the key it hears matches the one set) and the scale editor; knobs along the bottom.">
  <br><sub>The editor with the Hard Tune preset, rendered from the plugin's GUI using its built-in demo data.</sub>
</p>

> **Status:** v1, pre-release. It builds, passes its plugin validators and quality gates, and runs in Bitwig; checks in more hosts are ongoing ([#19](https://github.com/sdunifon/skatotune/issues/19)). Expect rough edges; see [Known limitations](#known-limitations).

---

## Contents

- [Features](#features)
- [Compatibility](#compatibility)
- [Install and build](#install-and-build)
- [How it works](#how-it-works)
- [How well it works](#how-well-it-works)
- [Roadmap](#roadmap)
- [Contributing, license, credits](#contributing-license-credits)

---

## Features

**Correction**
- **Two engines.** *Natural* shifts pitch one voice cycle at a time (PSOLA), so formants stay put and the voice keeps its character. *Classic* is the classic resample-and-splice method: the hard-tune sound, at about 1 ms.
- **Key and scale** (24 scales, including Bitwig's full scale set) with **per-note states**: *On* (snap to it), *Bypass* (leave pitch near it untouched), *Remove* (pull to a neighbour).
- **The familiar controls:** Retune speed, Note transition, Humanize, Flex-Tune, Amount, Tolerance, Snap min/max, Detune, Transpose, Formant shift (± 12 st, cleanest within ± 3), Tracking and input gate.
- **Vibrato-aware correction.** Skatotune separates each note's centre from its vibrato, tunes the centre and puts the singer's vibrato back, so hard tuning no longer flattens vibrato unless you want it to (Vibrato depth −100 … 200 %). Optional synthetic vibrato.
- **MIDI target notes.** Play the notes you want (*Target*), teach the plugin a scale from MIDI (*Learn*, saved with the project), or fall back to the scale when no key is held (*Auto*). Pitch bend and sustain work.
- **Key Source: follow a key instead of setting it.** *MIDI Scale* latches a chord from the MIDI input, which is how a DAW's key reaches the plugin (in Bitwig, route all 12 notes through Key Filter+ with Use Global Key). *Detect* follows **Auto-Key**, which listens to the singing, proposes a key with its confidence ("Detected Am 86 % · Apply") and, when you choose Detect, switches only after a new key has clearly taken over, and only between phrases ([how it works and how well](docs/research/auto-key.md)). A note you're holding keeps its target through a key change until the next note.
- **Presets:** Hard Tune, Natural, Live Safe, Classic.

**Latency and quality you choose separately**
- **Latency modes:** *Live-Min* (~1 ms, Classic), *Live* (3.1-5.9 ms by voice range), *Live+* (5-10 ms) and *Studio* (20-50 ms lookahead, compensated by the DAW). The latency the plugin reports is exactly the latency you get, verified for every mode, voice range and sample rate.
- **Quality tiers** Eco / HQ / Max inside every mode: more CPU buys better analysis, never more latency.
- **Voice ranges** (Low Male, Alto/Tenor, Soprano, Wide) tune the analysis to the singer.

**See what it's doing**
- **Scrolling pitch graph:** sung pitch, vibrato-free centre, corrected output and target notes on a piano roll, with live cents meters for "input vs target" and "correction".
- **Correction coloring** (in the Labs menu): the gap between your voice and the output is shaded by how hard the plugin is pulling, from faint to bright.
- **Latency readout:** the reported latency plus the average delay you actually hear for the note you're singing, and a CPU meter.

<p align="center">
  <img src="docs/images/labs.png" width="880" alt="The Labs menu open over the editor, with switches for the correction ribbon, the heat-tinted output line, learned note hold and follow singer's tuning, and a picker for five correction color themes.">
</p>

The **Labs** menu holds experimental features before they're promoted to the main UI. Correction coloring comes in five themes: Teal, Green → Red, Viridis, Inferno and Blue → Orange (three of them are color-blind safe). Two Labs switches change the sound, both off by default:
- **Learned note hold** (HQ/Max): a small learned model may hold the current note when a switch looks like wobble rather than a new note, and a note doesn't drop while the voice fades at a phrase end. On real singing it halves hard-tune warble, at the cost of a few short notes ([research](docs/research/note-intent.md)).
- **Follow singer's tuning:** corrects to the singer's own reference instead of A440, for consistently detuned material.

<p align="center">
  <img src="docs/images/themes.png" width="880" alt="The same passage of the pitch graph in each of the five correction color themes.">
</p>

**Built for live use**
- Nothing on the audio thread allocates, locks or can panic; this is enforced by tests.
- Output is bit-identical regardless of the host's buffer size; NaN or extreme input can't break it.
- Stereo, mono and mono-to-stereo; 44.1 to 192 kHz.

## Compatibility

| | Status |
|---|---|
| **Formats** | VST3, CLAP, standalone app. Audio Unit (Logic, GarageBand) planned ([#21](https://github.com/sdunifon/skatotune/issues/21)) |
| **DAWs** | Target hosts: Bitwig Studio 6 (CLAP and VST3) and Ableton Live 12 (VST3). Other VST3/CLAP hosts should work but are untested; a full in-DAW checklist is tracked in [#19](https://github.com/sdunifon/skatotune/issues/19) |
| **macOS** | Apple Silicon (primary; developed on an M4 Max). Intel Macs build but are not yet tested natively |
| **Windows / Linux** | Builds are planned with CI ([#22](https://github.com/sdunifon/skatotune/issues/22), [#31](https://github.com/sdunifon/skatotune/issues/31)); not yet tested |
| **Validators** | clap-validator (37 passed, 0 failed), Steinberg VST3 validator (47/47), pluginval at strictness 10 (pass) |
| **Audio** | 44.1-192 kHz; mono, stereo, mono → stereo; MIDI input for target notes |

## Install and build

Pre-built releases will come from CI ([#31](https://github.com/sdunifon/skatotune/issues/31)). Until then, build from source. You need Rust; the exact toolchain is pinned in `rust-toolchain.toml` and installs automatically.

```sh
git clone https://github.com/sdunifon/skatotune.git
cd skatotune
cargo xtask bundle skatotune --release
```

Then copy the bundles into your plugin folders and rescan in your DAW:

```sh
cp -R target/bundled/Skatotune.clap ~/Library/Audio/Plug-Ins/CLAP/
cp -R target/bundled/Skatotune.vst3 ~/Library/Audio/Plug-Ins/VST3/
```

Run it without a DAW (pick your audio devices in the terminal; use headphones):

```sh
cargo standalone
```

Tests, lint and the evaluation suite:

```sh
cargo test --workspace --release
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p skato_eval --release --test quality_gates
```

## How it works

```mermaid
flowchart LR
  IN["Audio in"] --> DET["Pitch detector<br/>pitch-synchronous candidates<br/>+ FFT verifier, octave checks,<br/>pitch HMM, voicing"]
  MIDI["MIDI in"] --> CTL
  DET --> CTL["Control law<br/>note decision, retune,<br/>humanize, flex-tune,<br/>vibrato split, transpose"]
  CTL -- "pitch ratio, formant" --> SH["Shifter<br/>Classic splice /<br/>causal PSOLA /<br/>lookahead PSOLA"]
  IN --> SH
  SH --> OUT["Audio out"]
  CTL -. "telemetry" .-> GUI["Pitch graph<br/>and meters"]
```

The analysis never delays the audio: only the shifter needs lookahead, and that lookahead *is* the reported latency.

**1. Pitch detection.** A pitch-synchronous candidate generator (a recursive autocorrelation that looks at only one or two cycles of the note being sung) proposes periods; an FFT-based verifier (normalized square difference plus a probabilistic YIN-style score) checks them; prime-harmonic and time-domain octave checks reject octave errors; a causal pitch HMM tracks the result, with fixed-lag smoothing in Studio mode; a voicing state machine separates singing from breath, consonants and silence. On 40 real singers it matches an offline reference tracker (94.7 % of frames within 50 cents) with a quarter of its octave errors.

**2. Deciding the note and the correction.** The control law decides which note you mean (rules in Eco, a note HMM in HQ/Max, decisions over the lookahead in Studio), then smooths the *correction* with a one-pole filter whose time constant is the Retune speed. Humanize slows correction on long sustained notes, Flex-Tune only corrects pitch that is already close, and a causal vibrato tracker splits the pitch into centre and vibrato so the centre can be tuned while the vibrato is kept or scaled.

**3. Shifting.** *Classic* resamples at the commanded ratio and repeats or drops whole voice cycles. *Natural* uses pitch-synchronous overlap-add (PSOLA): it finds each glottal cycle (pitch marks) and re-spaces short, windowed grains, which changes pitch without moving formants. *Studio* uses the same idea with real lookahead and picks the nearest grain. The formant knob reads each grain at a different rate.

The design rests on a large literature review, adversarial fact-checking and measurements, collected in the [research library](docs/research/README.md):
- [Architecture research](docs/research.md): every detector, shifter and control-law option considered, with evidence and sources.
- [Rust performance research](docs/rust-performance.md) and the measured [performance evidence](docs/PERFORMANCE.md).
- Topic reports: [following the DAW's key](docs/research/host-key-scale.md), [Auto-Key](docs/research/auto-key.md), [note intent](docs/research/note-intent.md), [graph/editing mode](docs/research/graph-mode.md), and the [technique registry](docs/research/TECHNIQUES.md).

## How well it works

Everything below is measured by the repository's own evaluation harness ([`crates/skato_eval`](crates/skato_eval)), on synthetic singing with exact ground truth and on **vocadito**, 40 real a-cappella takes from 29 singers with hand-checked annotations.

<p align="center">
  <img src="docs/images/latency.svg" width="760" alt="Bar chart of plugin latency by mode: Live-Min 1.0 ms; Live 3.1, 4.4 and 5.9 ms reported (4.6, 6.6 and 8.7 heard) for soprano, alto/tenor and low male; Live+ 10 ms.">
</p>

**Latency.** *Reported* is what the DAW compensates and what you hear when monitoring. *Heard* is the average delay of the corrected voice: PSOLA re-uses or skips whole voice cycles, so while it is correcting, the voice trails the reported latency by about half a cycle (more for low voices).

**Real singing** (vocadito, Live HQ hard tune unless noted):

| Measure | Result |
|---|---|
| Detector vs offline reference: frames within 50 cents | 94.7 % (reference 94.7 %) |
| Octave errors | 0.31 % (reference 1.30 %) |
| Note starts with a glitch | 6.3 % |
| Time for a note to settle on pitch, median / p90 | 27 / 66 ms (Studio: 17 / 41 ms) |
| Held-note error, in key, p95 | 14.7 cents (Studio: 8.3) |

**Synthetic quality gates** (run as tests):

| Gate | Target | Measured |
|---|---|---|
| Tuning, Live HQ (median / p95) | < 5 / < 15 cents | 1.6 / 4.9 cents |
| Settling after a note starts (Live HQ, retune 0) | every onset < 30 ms | all 18 onsets, worst 27 ms |
| Vibrato kept with preserve-vibrato on | extent within ± 15 % | 5 % |
| Latency measured = reported | every mode × range × rate | 48 / 48 |
| Transparency with correction off | exact (Classic) / ≥ 25 dB (PSOLA) | exact / 99 dB |
| CPU, Live HQ worst case (p99.9) | < 5 % of the buffer at 128 samples | 2.4 % |

## Known limitations

- Real singing is harder than the synthetic tests: held-note error is higher (p95 23-30 cents chromatic) and note starts still glitch about 6 % of the time.
- In Live mode, low voices hear about half a voice cycle more than the reported latency (8.7 ms average for low male voices).
- A CLAP latency change (switching latency mode) needs the host to restart the plugin.
- Builds are ad-hoc signed; macOS may ask you to allow them until releases are notarized ([#20](https://github.com/sdunifon/skatotune/issues/20)).

The full list, with numbers, is in [docs/STATUS.md](docs/STATUS.md).

## Roadmap

Work is tracked in [GitHub issues](https://github.com/sdunifon/skatotune/issues). Highlights:
- Better note decisions on real singing: onset cues, a musical note model ([#7](https://github.com/sdunifon/skatotune/issues/7))
- Lower natural-mode latency ([#8](https://github.com/sdunifon/skatotune/issues/8))
- Follow the DAW's key, and Auto-Key detection ([#5](https://github.com/sdunifon/skatotune/issues/5), [#6](https://github.com/sdunifon/skatotune/issues/6))
- Graph / editing mode ([#10](https://github.com/sdunifon/skatotune/issues/10)), throat model ([#9](https://github.com/sdunifon/skatotune/issues/9))
- CI with releases for macOS, Windows and Linux ([#31](https://github.com/sdunifon/skatotune/issues/31)); Audio Unit ([#21](https://github.com/sdunifon/skatotune/issues/21))

## Contributing, license, credits

- **Contributing:** see [CONTRIBUTING.md](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).
- **License:** [ISC](LICENSE). Third-party licenses are in [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) and inside every bundle.
- **Built on** [nice-plug](https://codeberg.org/RustAudio/nice-plug) (the community continuation of NIH-plug), [egui](https://github.com/emilk/egui), [RustFFT](https://github.com/ejmahler/RustFFT)/[realfft](https://github.com/HEnquist/realfft), and the CLAP and VST3 SDKs.
- **Evaluation and training data:** vocadito (Bittner et al., 2021, [CC BY 4.0](https://zenodo.org/records/5578807)) is downloaded for evaluation and not redistributed here. The Labs learned note hold's models were trained on part of it, so their weights ship in the plugin with the attribution in [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) ([model card](crates/skato_control/models/README.md)).
- **Prior art:** the methods build on published research and expired patents (Hildebrand's pitch-correction patent US 5,973,252; IVL's formant patent US 5,567,901), credited in [docs/research.md](docs/research.md).
- **Trademarks:** VST, Ableton, Bitwig and other names belong to their owners; Skatotune is not affiliated with them. See [TRADEMARKS.md](TRADEMARKS.md).
