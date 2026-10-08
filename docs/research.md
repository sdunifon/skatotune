# Vocal Tune: Research and Architecture

**Project:** real-time monophonic vocal pitch correction ("Auto-Tune style") in Rust, shipped as VST3 and CLAP. Tested in Ableton Live 12 (VST3) and Bitwig Studio (CLAP) on macOS Apple Silicon, and kept cross-platform capable.

**Date:** 2026-10-08
**Status:** architecture baseline v1. It is synthesised from six research reports (DSP pitch detection, neural pitch detection, pitch shifting and formants, correction behaviour, Rust plugin stack, state of the art) and an adversarial verification of each, plus spot checks made while writing.

**How claims are treated**

- When a verifier refuted or corrected a claim, this document uses the corrected version. Section 11.2 lists those corrections.
- Measured numbers marked "(measured, M4 Max)" come from the research sessions' local benchmarks on this machine (Apple M4 Max, release build, f32). The verifiers re-ran them. Plan budgets against a base M1, not the M4 Max.
- Marks used in the text:
  - "(estimate)": an engineering estimate, not a published or measured figure.
  - "(design choice)": our decision, not documented behaviour of any product.
  - "(vendor)": a figure published by the vendor and not independently measured.
- Every key claim links to its primary source. Appendix A collects the main sources by area.

---

## Contents

1. Executive summary
2. Latency vocabulary and the physics behind it
3. State of the art: products and prior art, 2023-2026
4. Pitch detection: candidates, evidence, decision
5. Pitch shifting and formants: candidates, evidence, decision
6. Correction behaviour: prior art
7. The chosen design (signal flow, latency × quality grid, control law, defaults)
8. Implementation notes and pitfalls for each block
9. Rust stack
10. Evaluation plan
11. Risks, open questions, and where the sources disagreed
12. IP and licensing summary
- Appendix A: key sources

---

## 1. Executive summary

### 1.1 Recommendation in one paragraph

The core is **pure DSP**, organised on two independent axes.

- **Latency axis.** The plugin reports a fixed latency per *latency mode*: Live-Min, Live, Live+ and Studio. The latency mode is non-automatable, and so is the voice-range preset, which sets the lowest f0.
- **Quality/CPU axis.** Within each latency mode there are three tiers, Eco, HQ and Max. A tier spends more CPU on detection, note decisions, mark tracking and interpolation. **It never changes the reported latency.**

The building blocks:

- **Pitch detection.** An in-house, allocation-free Rust detector. A pitch-synchronous Hildebrand E/H tracker generates candidates. It is equivalent to an incremental NSDF over 2L samples, and its lag is 1-2 periods of the note being sung. A per-hop FFT NSDF/CMNDF verifier with pYIN-style probabilities checks the candidates. Further stages: harmonic and octave checks, a causal HMM (fixed-lag in Studio), full-rate sub-sample refinement, and the instantaneous period read from the shifter's own pitch marks.
- **Shifting.** Time-domain and pitch-synchronous. "Classic" uses the Auto-Tune patent method (resample plus whole-cycle repeat or drop; the patent has expired). Live uses causal TD-PSOLA/Lent. Studio uses lookahead PSOLA with glottal-closure marks.
- **Formants.** PSOLA preserves them by construction. A formant/throat knob works by resampling inside each grain. Shifts beyond about ±3 semitones use per-grain spectral-envelope re-colouring.
- **Correction law.** A one-pole filter is applied to the correction, never to the target. A causal vibrato/centre separator hard-tunes the centre and adds the vibrato back at a chosen depth (the 2026 "DVA" capability).
- **Neural networks** are not in the shifter. A tiny causal neural *arbiter* (octave and voicing only) may enter the Max tier after it passes an evaluation gate. No public checkpoint can be shipped as is.

### 1.2 Key decisions

| Area | Decision | Main reason |
|---|---|---|
| Pitch detection (all tiers) | In-house DSP: pitch-synchronous E/H candidates, FFT NSDF/type-I-CMNDF verifier, pYIN-style causal HMM, full-rate refinement | No 2020-2026 method replaced the SDF/autocorrelation family for low-latency singing. Pitch-synchronous analysis has the lowest lag. The core costs <2% of a core (measured) |
| Neural F0 | Optional Max-tier octave/voicing arbiter only (PitchDNN-style primary candidate), behind a gate; DSP always sets the exact period | Every leading neural tracker is non-causal (SwiftF0 needs 176 ms of lookahead, FCPE ±0.9 s receptive field, RMVPE uses a BiGRU). Streaming PESTO's VQT still integrates 64-110 ms at vocal f0, and the g7 model costs about 28.5 MMAC per frame |
| Shifter, Live | Causal PSOLA/Lent with short-rise grains, normalised cross-correlation (NCC) marks re-anchored on the LP residual; symmetric 2·T grains whenever the budget allows | The only formant-preserving method within 5-10 ms. Near-transparent at ratio 1. Best or near-best in recent singing tests |
| Shifter, Live-Min | Classic: resample + whole-cycle repeat/drop, 1-1.5 ms crossfade | About 1 ms, the authentic hard-tune character (AutoTune 2026 Classic is reported at 0.8 ms). This is the reference engine and the fallback |
| Shifter, Studio | Lookahead PSOLA with glottal-closure-instant (GCI) marks (SEDREAMS/ZFF), a dynamic-programming mark path and symmetric grains. Optional S2 spectral engine (PGHI phase vocoder / shape-invariant SHIP) for alto and soprano ranges if it wins MUSHRA | The evidence conflicts on higher voices (Röbel Test 2, Drugman & Dutoit) |
| Formant | Inherent preservation; ±3 st shift by grain resampling; per-grain true-envelope/CheapTrick re-colouring beyond that | IVL patent: most realistic within ±18%. Envelope methods are needed for larger shifts |
| Correction law | Correction-domain one-pole (Retune Speed τ), separate Note Transition τ_tr, centre/vibrato split, Waves-style tolerance or a MonoNote-style HMM note decoder, Humanize, Flex-Tune, Snap Min/Max, On/Bypass/Remove, MIDI targets | Matches the semantics of Antares, Waves and zita/fat1. The DVA-like behaviour is the 2026 differentiator |
| Latency reporting | Constant per latency mode. Tiers and engines inside a mode share it | DAW plugin delay compensation (PDC) needs a constant latency. A latency change restarts the plugin |
| Framework | nice-plug 0.4.2 + nice-plug-egui 0.5.1, Rust 1.99.0 pinned, VST3 through MIT `vst3` 0.3.0 bindings, CLAP through clap-sys 0.5.0 | Upstream NIH-plug is in maintenance mode and its VST3 bindings are GPLv3. Ableton Live has no CLAP |
| Evaluation | Synthetic-detune round-trip harness, loopback latency, CPU tail percentiles on a base M1, clap-validator / pluginval / VST3 validator / RealtimeSanitizer, MUSHRA | Public benchmarks are offline, mostly speech, and partly biased |

### 1.3 Latency × quality grid at a glance

All numbers are at 48 kHz. Section 7.2 gives the full per-cell detail.

| Latency mode (reported D) | Eco (lowest CPU) | HQ (default) | Max (when CPU allows) |
|---|---|---|---|
| **Live-Min**, ≈1 ms | Classic splice; decimated E/H detector; rule-based notes | + 1-1.5 ms raised-cosine splice crossfade, sinc read, HMM pitch track | + MonoNote note HMM, resonator vibrato model |
| **Live**, ≈3-5 ms, depends on voice range (estimate); mean content delay D + T/2 | Causal PSOLA, asymmetric grains, cubic read; E/H + FFT verifier at hop 128 (2.67 ms), 20-cent HMM; rule-based notes | Hop 64 (1.33 ms), 10-cent HMM, two-resolution NSDF, 8-16-tap sinc, multi-hypothesis marks, multi-candidate grains, MonoNote HMM | + harmonic-model re-scoring (fastF0ArMl) and, after the gate, a neural octave/voicing arbiter; detector-lag compensation; envelope re-colouring for formant shifts beyond ±3 st |
| **Live+**, ≤10 ms | As Live; symmetric 2·T grains when T + mark delay ≤ D (f0 above about 100-150 Hz) | As Live HQ | As Live Max |
| **Studio**, 20-50 ms (default 40), DAW-compensated | Lookahead PSOLA, centred NSDF windows, fixed-lag HMM | + GCI marks with DP path, RTS centre/vibrato smoother, fixed-lag Viterbi notes, envelope warping beyond ±3 st | + SWIPE′ second opinion for low notes (D ≥ 50 ms), fastF0ArMl, optional S2 spectral engine for high voice ranges if it wins A/B tests |

Estimated CPU for the whole Live-HQ chain is a few percent of one Apple-Silicon core: the detector costs <2% (measured core, estimated HMM), PSOLA <1%, and the control law <0.1%. The release budgets are Live <10% and Max <25% of one base-M1 core at the p99 block time (Section 10.7).

### 1.4 What changed in 2024-2026, and what it means here

- **Vibrato-preserving correction is now table stakes.** Antares AutoTune Advanced (24-25 Sept 2026) introduced a patent-pending "DVA" engine that detects vibrato and passes it through even at the fastest retune speeds. It also restored linked Transpose+Formant across ±12 st, and its Live mode is quoted at 0.84 ms at 44.1 kHz (vendor). We implement our own causal centre/vibrato separation and must get a freedom-to-operate review once the DVA claims are published.
- **Commercial live tuners are still DSP.** None of the 2024-2026 Antares launches mentions a neural network in the tuning engine. Neural real-time products do voice conversion, not tuning, at 35-166 ms.
- **Academic neural pitch correction is better in quality but offline.** BERT-APC (TASLP 2026) scores MOS 4.32 against 3.22 for Auto-Tune and 3.08 for Melodyne, which shows there is headroom in the *note decision*. We spend effort there (an HMM decoder and key priors) rather than on neural synthesis.
- **Neural analysis is the only plausible neural win.** Candidates are PitchDNN (ICASSP 2024, in libopus 1.5, BSD), streaming PESTO (TISMIR 2025), low-rank FrCN (Sept 2026) and DSP-feature fusion (ICASSP 2026). None is validated causally on singing, so we gate them on our own benchmark.
- **The Rust tooling changed.** The Steinberg VST 3 SDK has been MIT-licensed since 3.8. NIH-plug moved to maintenance mode (2026-05-10), and the community fork is nice-plug. egui 0.36 needs rustc 1.95, while the machine's default Rust is 1.89.
- **New evaluation tools.** RT-SWIPE (CMMR 2025) introduced a delay-tolerant RPA. We use it only as a detector diagnostic. For a tuner, lag is itself the defect.

### 1.5 What we deliberately do not do

- No neural vocoder, DDSP resynthesis or voice-conversion model in the shifter path.
- No phase vocoder or SWIPE in Live: harmonic resolution needs about 4 periods (PV) or about 8 periods (SWIPE) of the lowest f0.
- No variable reported latency, and no TC-Helicon/Cor-Tek-style adaptive delay or histogram-adaptive note boundaries (US9123353 is active to about 2033-09-30).
- No GPL or AGPL code in the plugin: upstream nih-plug's VST3 bindings, zita-at1/fat1, Autotalent, Rubber Band, tdpsola, aubio, irapt, pYIN Vamp, Praat. We study algorithms and reimplement from papers.
- No silent tier switching mid-performance. A governor may only *demote*, with crossfades, on CPU overload.

---

## 2. Latency vocabulary and the physics behind it

### 2.1 Four different "latencies"

| Term | Definition | Who pays it | How we control it |
|---|---|---|---|
| **Reported latency D** | Samples the plugin reports for PDC (`set_latency_samples`) | The singer, in live monitoring, because live input ignores PDC. Recorded tracks are compensated | Fixed per latency mode and voice range |
| **Content-delay wander e(t)** | PSOLA and splicing repeat or skip cycles, so the delay of the content wanders within about one input period T around D | Live: mean extra T/2, worst case T (about 12 ms at 80 Hz). Studio with nearest-mark selection: ±T/2 around D | Unavoidable for pitch-synchronous methods. Documented, and measured in the harness |
| **Detection (information) lag** | Age of the audio that the current f0 estimate describes, about W/2 for a fixed window W | Not audio latency, but it mis-corrects any moving pitch by roughly *slope × lag*. Example: a 6 Hz, ±50 c vibrato with 13.75 ms lag gives about 26 c of error; ±100 c gives about 52 c | Pitch-synchronous analysis (1-2 periods of the sung note), the mark-spacing period, slope extrapolation, Studio centred analysis |
| **Correction lag / settling** | Time from a note change until the output sits within tolerance | The listener (late correction at onsets) | Retune Speed, the note decision, onset handling, Studio lookahead |

Monitoring round trip, which the singer actually hears:

```
L_monitor = L_in_buffer + L_out_buffer + L_converters (~1-2 ms) + D + e(t)
            e.g. 48 kHz, 64-sample host buffer: 1.33 ms each direction
```

Live's "Reduced Latency When Monitoring" only toggles delay compensation for input-monitored tracks, so the plugin's own D is always heard. Bone conduction sums with the monitor signal, so a few milliseconds of delay already causes comb filtering. That is why Live targets the low end. The perceptual reference (Lester & Boley, AES 2007) could not be fetched, so a singer-in-the-loop test sets the final budget (Section 10.9).

### 2.2 Periods and the minimum analysis span per method

| f0 | 55 Hz | 65 Hz | 80 Hz | 110 Hz | 220 Hz | 440 Hz | 1.2 kHz | 2.1 kHz |
|---|---|---|---|---|---|---|---|---|
| T (ms) | 18.2 | 15.4 | 12.5 | 9.1 | 4.5 | 2.3 | 0.83 | 0.48 |

| Method | Span needed | At 80 Hz |
|---|---|---|
| SDF/NSDF/MPM (fixed window) | about 2·T_max of the preset | 25 ms (lag about W/2) |
| YIN with early termination | T_max + T | depends on the note |
| Pitch-synchronous E/H (Hildebrand), incremental SNAC | 2·T of the *sung* note | 25 ms only when singing 80 Hz; 9 ms at 220 Hz |
| SWIPE kernels | about 8·T (N = 8·fs/f) | about 100 ms; RT-SWIPE right-aligned delays are about 80/40/20/10 ms at 50/100/200/400 Hz |
| PSOLA centred grain (Hann 2·T) | T of future input | 12.5 ms of lookahead |
| PSOLA short-rise asymmetric grain | rise (1-1.5 ms) + mark-detection delay (about 0.2·T) | about 3.5-4 ms |
| Phase vocoder resolving harmonics | about 4·T_min | 40-50 ms for 80-100 Hz voices |
| Shape-invariant PV (SHIP) | 6-8·T_min | 75-100 ms at 80 Hz; 30-53 ms at 150-200 Hz |

A rectangular window of length W attenuates the *displayed* vibrato depth by sinc(f_v·W). At 6 Hz that is 0.948 for W = 30 ms and 0.879 for W = 46 ms. Short or pitch-synchronous windows keep vibrato intact for display and for the vibrato controls.

### 2.3 Commercial latency reference points

Most of these are host-reported or vendor figures, not loopback measurements.

| Product / mode | Latency | Basis |
|---|---|---|
| AutoTune 2026 Classic / Modern low latency / HQ | 0.8 / 2.3 / about 58 ms | "Cubase reported", Sound On Sound Jan 2026 |
| AutoTune Advanced Live | 0.84 ms at 44.1 kHz; <1 ms up to 192 kHz | vendor |
| Antares blog, AutoTune 2026 low latency | "2.5 ms (175 samples at 44.1 kHz)" | vendor, and self-contradictory: 175 samples is 3.97 ms |
| UA Auto-Tune Realtime (Apollo DSP) Classic / Modern | 1-2 ms (50-100 samples) / 4-7 ms (175-300 samples) at 44.1 kHz, dynamic, not compensated | UA help article |
| Waves Tune Real-Time | reports 0; internal 0-4 ms depending on pitch | Waves, reviews |
| Eventide Eclipse / formant-corrected | 11.1 ms / ≥ 50 ms | JSLHR 2019, Eventide helpdesk |
| Slate MetaTune | about 3 ms nominal; Note Stabilizer adds about 280 ms | SOS |
| iZotope RT-SVC (research plugin) | <15 ms detection + Lent shift, about 60 ms total | DAFx 2023 |
| Rubber Band LiveShifter / Bungee / Signalsmith default / élastique Pitch | ≥ 50 ms / about 40 ms / about 120 ms / 150 ms | library docs, author posts |
| Dreamtonics Vocoflex (neural VC) | 36 / 166 ms (Bonedo measured) | review |

Our Live (3-5 ms reported) sits between Antares Modern low latency and the user's 5-10 ms target. Live-Min (about 1 ms) matches Classic. **Section 10.6 requires loopback measurements of all of these before final targets are set.**

---

## 3. State of the art: products and prior art, 2023-2026

| Product / system | Year | Technique (known or inferred) | Relevance to us |
|---|---|---|---|
| Antares AutoTune 2026 ([SOS review](https://www.soundonsound.com/reviews/antares-autotune-2026), [news](https://www.soundonsound.com/news/antares-release-autotune-2026)) | Oct 2025 | Same algorithm as Pro 11. Low-latency and HQ modes; Classic = the Auto-Tune 5 sound; up to 35% better CPU efficiency at 48 kHz (vendor). Lacks Transpose and Throat/Formant compared with Artist | Defines the control semantics and latency tiers to match |
| Antares AutoTune Advanced ([MBW](https://www.musicbusinessworldwide.com/antares-launches-autotune-advanced-a-pitch-correction-plugin-built-to-leave-a-singers-vibrato-alone/), [Production Expert](https://www.production-expert.com/production-expert-1/antares-autotune-advanced-review-what-makes-it-different), [Mixonline](https://www.mixonline.com/technology/news-products/autotune-advanced-pitch-correction-plug-in-debuts)) | Sept 2026 | "DVA" (patent-pending) vibrato detection and preservation; vibrato visualizer; Vibrato FX; Transpose+Formant linked ±12 st; Formant Designer; Live / Low Latency / HQ. Sources disagree on whether LL/HQ sit inside Live. No neural network mentioned | The capability to match: tuned centre with vibrato kept. Patent watch |
| Auto-Tune patent US 5,973,252 ([Google Patents](https://patents.google.com/patent/US5973252A/en)) | filed 1998, expired 2018-10-14 | Recursive E/H autocorrelation, 8:1 decimation, lags 2-110, an 8-lag full-rate track, quadratic fit; resampling at a smoothed ratio with one-cycle repeat or drop; Decay 0-1 | Free to implement. Basis of the Classic engine and of the pitch-synchronous detector |
| Waves Tune Real-Time ([user guide](https://www.fullcompass.com/common/files/28004-WavesTuneRealTimeUserGuide.pdf)) | 2016 | Speed (0.1-800 ms, default 15) vs Note Transition (default 120); Tolerance in cents/time; Vibrato Depth 0-200%; 5 note states; MIDI target | Best published model of in-note vs transition speed and of explicit hysteresis |
| zita-at1 / x42 fat1 ([retuner.cc](https://github.com/x42/fat1.lv2/blob/master/src/retuner.cc)) | 2009-2026 (v0.8.11 Sept 2026) | FFT autocorrelation; note bias hysteresis (Bias/13 octave); correction one-pole as a time constant; period jumps with a 128-sample crossfade | GPL: reference for ideas only |
| iZotope RT-SVC ([DAFx23](https://www.dafx.de/paper-archive/2023/DAFx23_paper_21.pdf)) | 2023 | Lent shifter + neural linear time-varying (LTV) envelope filter; "vocoder-free" | The pattern where a network never generates pitch. Its PV beat Lent in MOS, but in voice conversion with 11 in-house listeners |
| Melodyne 5 ([Celemony help](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_ToolModulationDrift_2)) | 2020-2026 | Offline note objects; centre / drift / modulation (−100..+100%) | Conceptual model for our centre/drift/vibrato controls. US 8,022,286 is active to June 2029 (polyphonic DNA), so do not copy note-object claims |
| Graillon 3 ([MusicTech](https://musictech.com/news/gear/auburn-sounds-graillon-3/)) | Oct 2024 | G2/G3/I1 engines (I1 about 7x slower); Snap Min dead zone; Snap Max (default 8 st); Inertia | UX ideas: dead zone, maximum snap distance |
| Slate MetaTune ([SOS](https://www.soundonsound.com/reviews/slate-digital-metatune)) | 2020s | Negative speed, Note Stabilizer (40/80/200 ms) that adds about 280 ms, heat map | Counter-example: lookahead vibrato stabilisation is expensive |
| Ableton Auto Shift ([Live 12 manual](https://www.ableton.com/en/live-manual/12/live-audio-effect-reference/)) | Live 12.1+ | Built-in real-time mono corrector with formant, MIDI and a Live Mode; its pitch range changes latency | A/B baseline in the same host. Precedent for making voice range part of the latency mode |
| zplane INTONE ([licensing](https://licensing.zplane.de/technology)) | current | Formant-preserving, low-latency monophonic correction SDK (commercial) | Closest commercial SDK; measure it if licensable |
| Neural VC products (Vocoflex, Shift by Antinode Audio, SoundID VoiceAI) | 2024-2026 | Neural voice conversion | Not tuners. 35-166 ms. Rejected |
| BERT-APC ([arXiv 2511.20006](https://arxiv.org/abs/2511.20006)), MIDIBack ([arXiv 2609.28008](https://arxiv.org/abs/2609.28008)), Deep Autotuner, KaraTuner, Diff-Pitcher | 2019-2026 | Offline, context-aware note-level correction and resynthesis | Shows the value of the intended-note decision and of harmonic context. Not causal |

**Takeaway.** The market leader ships time-domain modes of about 1-2 ms for live use and a mode of about 60 ms for highest quality, and has now made vibrato preservation the headline. We do not try to win with a neural engine. We aim for:

1. vibrato-aware centre correction;
2. a latency × CPU grid that no vendor exposes;
3. better note-decision logic (onsets, hysteresis, confidence, an HMM);
4. a Studio mode that uses true lookahead;
5. a rigorous automated evaluation harness.

## 4. Pitch detection

### 4.1 What the detector must do

The detector has five jobs:

1. Find the f0 of a monophonic sung voice over 55-1400 Hz. A whistle extension to about 2.1 kHz is optional.
2. Add no audio latency. Detection runs on past samples in Live mode, and on a window centred inside the lookahead in Studio mode.
3. Keep information lag low, ideally 1-2 periods of the note being sung.
4. Give sub-cent precision on stable notes.
5. Make robust decisions on octave and voiced/unvoiced (V/UV), because octave errors and voicing flicker are what make correction glitch audibly.

A pure-DSP fallback must always exist.

### 4.2 DSP candidates

| Candidate | Latency / analysis span | CPU | Quality evidence | License | Verdict |
|---|---|---|---|---|---|
| **Hildebrand recursive E/H, pitch-synchronous** ([US5973252](https://patents.google.com/patent/US5973252)). (E−2H)/E is exactly 1 − NSDF over a 2L window | Per-note span 2·T of the *sung* note; detection "within a few cycles" | 4 MAC per lag per sample. All lags at 48 kHz ≈ 2% of a core; 8:1 decimated ≈ 0.034% (measured, M4 Max) | No published benchmark, but it has been the industry reference for 25 years. Its eps threshold is the mechanism behind Antares "Tracking" (inference) | Patent expired | **Adopt as the default candidate generator** (verifier: the cheapest path has the lowest lag) |
| **FFT SDF core: MPM NSDF + YIN CMNDF + pYIN probabilities** ([YIN](https://cagnazzo.wp.imt.fr/files/2013/05/2002_JASA_YIN.pdf), [MPM](https://www.cs.otago.ac.nz/graphics/Geoff/tartini/papers/A_Smarter_Way_to_Find_Pitch.pdf), [librosa pyin](https://raw.githubusercontent.com/librosa/librosa/main/librosa/core/pitch.py)) | W = 2.0-2.5 × T_max: Low Male 37-40 ms, Alto/Tenor 26-28 ms, Soprano 15-16 ms; lag about W/2 | 16.4 µs per frame at 48 kHz (N=4096; 0.62% at a 2.67 ms hop); 3.5 µs at a 12-16 kHz analysis rate (measured, M4 Max) | YIN: 1.03% average gross error on 4 laryngograph *speech* databases, against 3.1-16.8% for others. MPM is stable with 2 periods | Papers; reimplement | **Adopt as the per-hop verifier.** The CMNDF must be type-I normalised (§8.4) |
| **WSNAC / incremental SNAC** (McLeod PhD 2008, [thesis](https://www.cs.otago.ac.nz/graphics/Geoff/tartini/papers/Philip_McLeod_PhD.pdf)) | Per sample | Low | Windowed NSDF is more consistent on frequency ramps (vibrato, glides). The thesis also has a real-time vibrato estimator | Paper | **Adopt** the WSNAC option for the HQ two-resolution verifier |
| pYIN, causal (forward filter) / fixed-lag | Forward: 0. Fixed lag: L·hop | HMM ≈ 28k MAC per frame, banded (estimate) | MIR-1K 95.4% RPA with a 145 ms window. Best noise robustness among classical methods. The Vamp fixed-lag mode is hard-wired to 100 frames, and Essentia's "streaming" pYIN is offline, so there is no published evidence for 20-50 ms lags | Vamp GPL, librosa ISC; reimplement | **Adopt the HMM** (causal), with hop-rescaled transitions |
| Praat filtered autocorrelation ([manual](https://www.fon.hum.uva.nl/praat/manual/how_to_choose_a_pitch_analysis_method.html)) | 3 periods of the floor (6 if very accurate) | Very fast | More than 3x fewer gross octave errors than raw autocorrelation on AtakeDB (speech). Gaussian low-pass with attenuation 0.03 at the pitch top | GPL-3; reimplement ideas | **Borrow**: pre-low-pass (A/B against SIFT/LPC), octave cost 0.01/0.055 per octave, octave-jump cost 0.35, V/UV cost 0.14 |
| SIFT / low-order LPC inverse filter front end (Markel 1972) | ≈0 | A few dozen MAC per sample | Classic fix for formant-induced octave errors (F1 ≈ 2-3·F0) | Public | **Consider**: A/B against Praat's Gaussian low-pass |
| Harmonic-model ML: fastF0Nls / **fastF0ArMl** / Shi 2019 causal Bayesian tracker ([fastF0ArMl](https://github.com/jkjaer/fastF0ArMl), [Shi](https://arxiv.org/abs/1905.08557)) | Segments down to 12.5 ms (speech > 80 Hz); causal filter | About 1-5% of a core per voice (estimate; benchmark it) | Shi: the 15%/20% gains on Keele are at 0 dB *white* noise; at 10 dB babble/factory noise it ties fast NLS. fastF0Nls makes octave errors in low-frequency-heavy noise, so the authors recommend fastF0ArMl | BSD-3, notify on commercial use | **Consider for Max tier**: re-score the top-K SDF candidates. Unproven at the high SNR of a close mic |
| Harmonic chirp ML ([Jensen et al. TSP 2017](https://doi.org/10.1109/TSP.2017.2723342)) | Window | Moderate | Models linear FM inside the window, which reduces vibrato and glide bias; gives slope for lag compensation | Paper | **Consider for Max** on segments flagged as vibrato or glide |
| SHS / SHR (subharmonic summation) | Several periods standalone | Cheap given an FFT | Weak standalone (8.7% gross error in the YIN comparison), but SHS scored 0.657 F1 in pitch-benchmark (speech-heavy) | Public | **Use as an octave-disambiguation score** (prime-harmonic sum) only |
| SWIPE′ / RT-SWIPE ([CMMR 2025](https://audiolabs-erlangen.com/content/05_fau/assistant/00_balke/01_publications/2025_MeierSSMB_RealTimeSWIPE_CMMR.pdf), [ISMIR 2025](https://arxiv.org/html/2507.11233v1)) | About 8·T; right-aligned delay 80/40/20/10 ms at 50/100/200/400 Hz | 20 channels in real time on an M2 | MIR-1K 96.2% RPA with a 327 ms window. RT-SWIPE RPA25 0.931 vs 0.960 offline, on wind/brass, not voice | Paper; swipe-rs Apache-2.0 | **Reject for Live** (8-period kernels). **Studio Max option** as a low-note second opinion when D ≥ 50 ms; offline labeller |
| RAPT / YAAPT / REAPER | Offline DP as published | Low | Babacan singing: RAPT lowest GPE (0.65%), but its ground truth is RAPT-on-EGG (biased) | various; irapt is AGPL | **Borrow ideas** (two-pass decimate/refine, DP costs) |
| Cycfi BACF / "Hz" ([Q docs](https://cycfi.github.io/q/q/v1.5/reference/pitch.html)) | 1-2 cycles | About 50 ns per sample | Tuned for guitar over 2 octaves. No vocal evaluation. The "Hz" detector is unpublished | MIT | **Consider later** only as an Eco experiment after a vocal benchmark |
| Cepstrum; WORLD DIO/Harvest; ESPRIT / ECKF / IRAPT | ≥ 3 periods / offline / heavy | — | Cepstrum: 6.0% gross error vs 1.03% for YIN. DIO RPA 0.696 in penn. ECKF lags at note changes | — | **Reject** |
| Rust crates: pitch-detection, pitch-estimate, swipe-rs, pyin, pitch-core, phonk, yin… | — | — | No crate is RT-safe, mature, causal with voicing, and permissively licensed all at once | mixed (irapt AGPL, aubio-rs GPL) | **Reject as dependencies**. Use pitch-detection, pitch-estimate, swipe-rs and pyin as test oracles only |

### 4.3 Neural candidates

| Candidate | Causality / lag | Cost per frame | Quality evidence | License | Verdict |
|---|---|---|---|---|---|
| **PitchDNN** (DSP-assisted; Subramani, Valin et al., ICASSP 2024; libopus 1.5 [`dnn/pitchdnn.c`](https://github.com/xiph/opus/blob/v1.5/dnn/pitchdnn.c)) | Fully causal, 10 ms algorithmic delay | 68,769 params; 0.059 GFLOPS at 100 fps (≈0.6 MFLOP per frame); about 10 µs (estimate) | Clean PTDB raw-cent accuracy 92.02 vs CREPE 92.31; more robust in noise; *speech only*. Lag range 32-256 samples at 16 kHz = 62.5-500 Hz, which misses soprano | BSD-3 (code). Weights trained on speech | **Primary neural candidate** for the Max-tier arbiter: retrain on singing with lags of about 15-400 samples at 16 kHz |
| PESTO streaming ([TISMIR 2025](https://arxiv.org/abs/2508.01488)) | Frame-independent VQT; buffer refill m = 0.5 gives *reporting* lag h/(2·fs) + τ. But fundamental-bin VQT kernels are about 109/89/64/42 ms at 110/220/440/880 Hz, so *information* lag stays long | g7 (130k params): about 28.5 MMAC, ≈0.6-1.0 ms hand-coded. v1 (28.9k): about 7.4 MMAC, ≈0.15-0.25 ms. ONNX: 0.7 ms per 1024-sample call | MIR-1K 97.7 (97.4 at m = 0.5), but trained in-domain. pitch-benchmark: F1 0.680, only 38.3% of frames within 10 c, worst octave-up rate (3.39%), **best voicing F1 (0.850)** | Code LGPL-3.0; weight license unstated | **Challenger only** (v1 size, higher γ). Value lies in voicing. Do not run g7 on the audio thread at a 5 ms hop |
| Low-rank FrCN ([arXiv 2609.23340](https://arxiv.org/html/2609.23340)) | Frame-independent; VQT window not stated | 11,397 params, 2.67 MMAC; 58 µs in a register-blocked C kernel (x86 4.9 GHz); rank M = 6/9/12 gives 1.80/2.67/3.55 MMAC | MIR-1K in-domain 95.25; the range of training noise mattered about 100x more than the architecture | Paper only, no code | **Recipe** for kernels, augmentation and CPU scaling by rank |
| Strahl & Müller mid-level fusion ([ICASSP 2026](https://audiolabs-erlangen.de/resources/MIR/2026-ICASSP-MidLevelFusionF0)) | Not reported | 6.5k params + DSP features | "Competitive"; no numbers available | No code | **Design reference** (fuse our NSDF/CMNDF maps) |
| SwiftF0 v0.3 ([repo](https://github.com/lars76/swift-f0)) | Non-causal: 176 ms of lookahead to stream | 14,386 params; 180x real time on one core | Best pitch-benchmark F1 (0.781), best octave rates (0.34% up / 0.42% down), 67.7% within 10 c | MIT code; weights trained partly on CC BY-NC MDB-stem-synth | **Offline analysis / teacher**; architecture for a Studio student with asymmetric padding |
| RMVPE, FCPE, CREPE, PENN/FCNF0++, SPICE, HarmoF0, DeepF0 | Non-causal or ≥ 32-64 ms centred windows | Up to 10.6M (FCPE) or about 90M (RMVPE) params | RMVPE tied best (0.768). FCPE ±0.9 s receptive field. CREPE 32 ms delay. PENN 64 ms | Mostly MIT/Apache | **Reject for real time**; RMVPE/SwiftF0 as offline teachers |
| LACOPE technique ([Interspeech 2021](https://www.isca-archive.org/interspeech_2021/schroter21_interspeech.html)) | Target alignment down to 0 ms | Small RNN | On par with pYIN/CREPE on speech | Paper | **Adopt the technique**: train with labels aligned to the newest sample |
| Voting ensemble (Koguchi & Koriyama, [ICASSP 2026](https://arxiv.org/abs/2602.01727)) | Slowest member | Sum of members | Beats single estimators (no numbers given) | Paper | **Consider for Max**: delay- and bias-aligned fusion of DSP and neural |
| Real-time VC F0 stacks (RVC, w-okada, DDSP-SVC); StreamVC | Hundreds of ms; StreamVC 70.8 ms and uses YIN | GPU | — | — | **Reject**; they confirm that DSP is the low-latency backbone |

### 4.4 What the benchmark evidence actually says (after verification)

- **pitch-benchmark v2.1** ([BENCHMARK.md](https://github.com/lars76/pitch-benchmark/blob/main/BENCHMARK.md), written by the SwiftF0 author).
  - Overall F1: SwiftF0 0.781, RMVPE 0.768, FCPE 0.728, CREPE 0.689, PESTO 0.680, SHS 0.657, Praat 0.651, RAPT 0.640, SWIPE 0.610, PENN 0.560, pYIN 0.506.
  - **Seven of its ten corpora are speech, and only Vocadito is singing.** On Vocadito: RMVPE 0.862, SwiftF0 0.847, CREPE 0.781, PESTO 0.759, RAPT 0.713, SHS 0.706, Praat 0.696.
  - Share of frames within 10 cents: RMVPE 68.5, SwiftF0 67.7, FCPE 66.0, CREPE 63.9, SHS 58.7, Praat 57.1, RAPT 56.4, pYIN 52.2, PESTO 38.3. The best neural models beat every DSP method, and PESTO is worst.
  - Best neural vs best DSP: 0.949 vs 0.925 on clean audio, and 0.754 vs 0.606 (SHS) with a background scene.
  - **None of these models is causal at our latency.**
- **The YIN family is the weakest DSP class in that benchmark** (pYIN 0.506), while SHS and RAPT are much stronger and nearly free. The choice of DSP backbone must therefore be made by measurement on singing (Section 10), comparing YIN/NSDF, a causal SHS and a causal NCCF.
- **MIR-1K (ISMIR 2025)**: SWIPE 96.2% (327 ms window), pYIN 95.4% (145 ms), PESTO 96.1% (trained in-domain; 93.5% when trained on MDB). SWIPE and pYIN match neural trackers *only with long offline windows*. **The claimed collapse to 85% at 46 ms belongs to SWIPE-tiny, a neural model, not to classical SWIPE.** Classical SWIPE is rejected for Live on physical grounds (8-period kernels), not on that number.
- **Babacan singing (EGG)**: the ground truth is RAPT-on-EGG, and YIN was scored with RAPT's V/UV decisions, so the ranking is not clean. YIN degrades most under reverb (GPE about 7% at T60 = 100 ms). Recommend dry, close-miked input.
- **SwiftF0 paper**: Vocadito labels are pYIN-derived, and CREPE scoring 70.87% suggests configuration problems. Weak evidence for ranking DSP methods.
- **Local measurements (M4 Max)**:
  - Parabolic interpolation median error on stationary synthetic tones: 0.04-0.86 cents over 70-1200 Hz at 48 kHz analysis, but 0.15-3.5 cents at 12 kHz. Always refine high notes at full rate.
  - Direct time-domain YIN at 48 kHz costs 272 µs per frame. At 12 kHz it costs only 11 µs, unvectorised, so "FFT is mandatory" is an overstatement; it is still the sensible default.

### 4.5 Decision: detector

1. **Default for every tier and mode** (as revised by the DSP verifier):
   - Pitch-synchronous E/H candidate generation over the whole preset lag range at a decimated rate, updated every 4-8 samples.
   - Per-hop FFT verifier: NSDF/WSNAC clarity and key maxima; type-I-normalised CMNDF giving pYIN probabilities; prime-harmonic sum.
   - Full-rate 3-lag parabolic or sinc refinement.
   - Causal pitch HMM; a mean-pitch Kalman filter with innovation gating; voicing logic.
   - The Live shifter's mark spacing gives an almost zero-lag instantaneous period.
2. **The Max tier adds**:
   - fastF0ArMl harmonic-model re-scoring of the top-K candidates, with harmonic-order tracking;
   - optional harmonic-chirp scoring on vibrato and glide segments;
   - and, only after the neural gate (§10.10), one causal neural octave/voicing arbiter. Primary candidate: a PitchDNN-style network retrained on singing, ≤ 0.25 ms per 5 ms hop on a base-M1 P-core. Challengers: a FrCN-size or PESTO-v1-size VQT model. They are fused by delay-aligned voting. DSP still searches ±50 cents around the arbiter's estimate for the exact period.
3. **Studio adds**: windows centred on the output sample (W ≤ 2D), fixed-lag HMM smoothing over L = D/hop frames, and an RTS smoother. SWIPE′ is used as a low-note second opinion when D ≥ 50 ms.
4. **Never in Live**: SWIPE, cepstrum, DIO/Harvest, ESPRIT, any centred neural model.

---

## 5. Pitch shifting and formants

### 5.1 Candidates

| Candidate | Latency | CPU | Quality evidence | License | Verdict |
|---|---|---|---|---|---|
| **Causal pitch-synchronous OLA (Lent / TD-PSOLA)**, asymmetric short-rise grains or symmetric 2·T Hann | Reported ≈ rise (1-1.5 ms) + mark delay (≈0.2·T_max) + interpolation half-kernel + margin ≈ 3-5 ms (estimate). Content wander adds a mean of T/2, worst case T. Symmetric grains need T + mark delay | Very low: 2-4 grains × 4-16 taps; ≈2-5 MFLOPS + 2-10 MFLOPS for marks; well under 1% of an M1 core | Singing ([Liu & Akama 2026](https://arxiv.org/html/2601.10345)): best classical FAD (19.25 vs WORLD 31.40) and best LSD (0.643). Speech ±600 c ([Promonet](https://arxiv.org/html/2407.05471v1)): 61.1 vs WORLD 45.0. **All of this is offline PSOLA with full-file marks; causal short-rise PSOLA on singing is untested** | Public algorithm | **Adopt (Live default)**. The asymmetric variant counts as experimental until MUSHRA shows it beats L0 and comes close to S1 on low male singing |
| **Lookahead PSOLA with GCI marks** (SEDREAMS / ZFF, DP mark path; [GCI review](https://arxiv.org/pdf/2001.00473)) | ≈0.9·T0 SEDREAMS lookahead + tracker lookahead + T; 25-35 ms covers 70-80 Hz voices | Low (5-20 MFLOPS) | SEDREAMS: > 98% identification on *speech* EGG datasets; ZFR/SEDREAMS most robust to noise and reverb. Not tested on singing at high f0 | Public; COVAREP (reference) | **Adopt (Studio default)**; validate marks on singing |
| **Resample + whole-cycle repeat/drop** (Auto-Tune patent; zita-at1) | ≈1 ms reported (AutoTune 2026 Classic: 0.8 ms host-reported); content wanders within one period | Negligible | Two decades of hard-tune use. Formants move with pitch (+1 st ≈ +5.9%) | Patent expired; zita/fat1 GPL (reference only) | **Adopt (Classic / Live-Min, and fallback)** |
| **Formant shift by grain resampling** ([IVL US5567901](https://patents.google.com/patent/US5567901A/en)) | None extra, if γ·h stays inside the lookahead | Negligible | Most realistic within ±18% (≈±3 st); the patent's user range is ±33%. The window is a Hann of 2× the resampled input period | Patent expired 2015 | **Adopt** for the formant/throat knob within about ±3 st |
| Spectral-envelope estimators: true envelope ([Röbel-Rodet DAFx05](https://dafx.de/paper-archive/2005/P_030.pdf)), CheapTrick ([WORLD](https://github.com/mmorise/World/blob/master/src/cheaptrick.cpp)), LPC | TE: one PV window. CheapTrick: 1.5·T0 lookahead (3·T0 window) | TE: 2-6 FFT iterations; the fast version costs < 40% more than LPC | TE does not track individual harmonics when Pc ≤ fs/(2·F0) | Public; WORLD BSD-3 | **Adopt** for formant shifts beyond ±3 st and for any spectral engine |
| LP-PSOLA / residual-domain PSOLA | PSOLA + 0-2 ms | ≈5-10 MFLOPS | **Negative evidence** ([ESOLA paper](https://arxiv.org/pdf/1801.06492)): LP-PSOLA pitch-scaling MOS 1.45-1.85 against TD-PSOLA 1.40-2.40, lower at 4 of 5 factors | Public | **Not a default**. Only a gated experiment for large downward shifts |
| WSOLA / SOLAFS / ESOLA + resampling | 20-40 ms | Low-moderate | In the ESOLA test, pitch-scaling MOS was SOLAFS 3.95-4.30 and ESOLA 4.25-4.60, against TD-PSOLA 1.40-2.40. But resampling shifts formants, so they are not formant-preserving | Public | **Reject** as the shifter (formants). The result is a reminder that PSOLA quality depends on the implementation |
| Phase vocoder with phase locking ([US6549884](https://patents.google.com/patent/US6549884B1/en), expired 2019) | About 4·T_min (40-50 ms at 80-100 Hz; 20-27 ms for f0_min 150-200 Hz) | 0.1-0.2 GFLOPS including TE | iZotope: their PV scored above Lent (4.52 vs 3.99) in SVC with 11 in-house raters. Phasiness, transient smearing | Public | **Consider (Studio S2)** for high voice ranges only |
| PGHI / "Phase Vocoder Done Right" ([arXiv 2202.07382](https://arxiv.org/pdf/2202.07382)) | Window + 1 hop | FFT + O(M log M) heap | Time-stretching CCR test (6 subjects analysed): beats classic PV and is close to IRCAM | Algorithm public; LTFAT GPL | **Consider**: the phase method for S2 |
| Shape-invariant PV (Röbel SHIP, [DAFx10](https://dafx.de/paper-archive/2010/DAFx10/Roebel_DAFx10_P84.pdf)) | 6-8·T_min: too long for bass, 30-53 ms for f0_min 150-200 Hz | 10-20% of a 2010 CPU | Test 1: PSOLA 2.8 up / 3.8 down vs SHIP 3.1 / 1.9. **Test 2 (improved SHIP, 29 listeners): SHIP beat PSOLA on average, up 2.9 vs 2.6, down 2.8 vs 2.6; female up 3.2 vs 2.5, down 2.8 vs 2.0**; PSOLA kept a small edge on male voices | IRCAM proprietary; algorithm published | **Consider (Studio S2)** for alto/soprano ranges |
| Signalsmith Stretch ([repo](https://github.com/Signalsmith-Audio/signalsmith-stretch)) | presetDefault: 0.12 s block / 0.03 s interval | Heavy, bursty | Informal reports only | MIT; crates signalsmith-stretch 0.1.3, ssstretch 0.1.0 (1 day old, Linux-only docs build) | **Prototype tool** for S2 only |
| stftPitchShift ([repo](https://github.com/jurihock/stftPitchShift)) | STFT | Moderate | Cepstral-lifter formant shift | MIT | **Reference** for the envelope whiten/shift/recolour path |
| Rubber Band R3 / LiveShifter; Bungee; élastique; zplane INTONE | ≥ 50 ms / about 40 ms / 150 ms / unknown | — | — | GPL or commercial / MPL-2.0 / commercial / commercial | **Reject in product**; listening-test references (INTONE as a commercial benchmark) |
| WORLD resynthesis; sinusoidal / HNM | ≥ 1.5·T0 + offline F0; 20-40 ms | Moderate | WORLD on singing: FAD 31.4, LSD 3.57; buzzy and not transparent. HNM worst in Röbel | WORLD BSD | **Reject** as shifter (CheapTrick kept as an envelope estimator) |
| Neural vocoders / DDSP / streaming VC (FARGAN, LLVC, StreamVC, SiFiGAN, CLPCNet, BRAVE, GOLF, PC-NSF-HiFiGAN) | FARGAN < 20 ms (16 kHz, f0 ≤ 500 Hz); LLVC < 20 ms (16 kHz); StreamVC 70.8 ms; **BRAVE about 10 ms at 44.1 kHz** (RTF 0.29 on an M1 Pro, RTNeural, 128-sample blocks) | FARGAN 600 MFLOPS; BRAVE 4.9M params (≈30% of a core) | On singing: SiFiGAN FAD 49.0, CLPCNet 77.7 vs PSOLA 19.2. BRAVE is timbre transfer (pitch accuracy 0.68 at a 0.5-semitone tolerance). Not transparent at ratio 1; identity changes | Mixed; PC-NSF-HiFiGAN weights AGPL | **Reject** in the shifter path, on quality and transparency (latency is *not* the blocker on Apple Silicon) |
| Neural restoration post-filter on DSP output (Liu & Akama 2026 shallow diffusion; iZotope LTV filter) | Offline (diffusion); the LTV filter about 45 ms of windows | GPU-class / small LSTM | Liu & Akama: FAD 7.886 vs PSOLA 19.248, MFCC 23.6 vs 39.1, F0 error 2.9 vs 41.8 c; but worse LSD (0.78 vs 0.64) | Research | **Research track** for Studio offline render, and possibly a causal Live-Max post-filter (≤ 3-5 ms, ≤ 25-30% of an M1 core), shipped only if MUSHRA shows a gain |

### 5.2 How the conflicting evidence was resolved

1. **PSOLA vs phase vocoder vs SHIP.**
   - Evidence for PSOLA: Liu & Akama 2026 (offline, mostly Chinese opera, round-trip shifts), Promonet (speech), and Röbel Test 1 (speech, downward shifts).
   - Evidence against: iZotope, where the PV beat Lent but in a voice-conversion setting; Röbel Test 2, where SHIP beat PSOLA on average and clearly on female voices; and Drugman & Dutoit, where DSM and STRAIGHT beat TD-PSOLA on female speakers.
   - Röbel also states that PSOLA quality degrades with increasing pitch.
   - **Resolution**: PSOLA is the default everywhere because it is the only option within the Live budget. For Studio, the **voice-range preset selects the engine**: S1 PSOLA for bass/tenor; S2 (envelope-preserving PGHI-PV or SHIP-style) as a candidate for alto/soprano. S2 ships only if MUSHRA shows a gain.
2. **PSOLA pitch accuracy.**
   - In Liu & Akama's Table 2, TD-PSOLA had the *worst* F0 error (41.78 c, using the offline maxrmorrison/psola package).
   - In the CLPCNet study it was the *most* accurate (8.4-28.9 c RMS against 21.6-119 c for CLPCNet).
   - **Resolution**: PSOLA's accuracy depends on its pitch marks and on the implementation. Our PSOLA must pass the harness's cents-error gates (§10.3); it cannot be assumed good.
3. **LP-PSOLA as the "Live quality lever"** was refuted by the ESOLA listening test. It is demoted to an experiment.
4. **"Neural is too slow"** is refuted by BRAVE (about 10 ms). Neural is rejected in the shifter on quality and transparency evidence, and a gated post-filter research track remains.

### 5.3 Decision: shifter and formant

| Engine | Where | Notes |
|---|---|---|
| **L0 Classic** (resample + whole-cycle splice, 1-1.5 ms raised-cosine crossfade) | Live-Min (only engine), selectable in Live/Studio (padded to the mode's D) | Ship first as the reference. No formant correction. Hard-tune / AutoTune Classic character |
| **L1 Live PSOLA** | Live, Live+ | Symmetric 2·T_s Hann grains whenever T + mark delay ≤ D (most female and pop voices above about 120-150 Hz at D = 10 ms). Asymmetric short-rise grains otherwise (labelled experimental). NCC marks + LP-residual anchor |
| **L2 Live HQ / Max options** | Live HQ, Max | Windowed-sinc 8-16 taps or 2x oversampling; causal multi-hypothesis marks (Viterbi over the last 3-6 periods within 1-2 ms of lookahead); multi-candidate grain selection; per-grain envelope correction beyond ±3 st (Max); band-split aperiodic handling (experimental) |
| **S1 Studio PSOLA** | Studio | GCI marks (SEDREAMS/ZFF + LP-residual refinement) with a DP path, symmetric grains, onset-aware decisions, envelope warping beyond ±3 st |
| **S2 Studio spectral** (optional) | Studio Max, alto/soprano ranges | Envelope-preserving PV with an f0-adaptive window (N ≈ 4/f0_min), PGHI or identity phase locking, true-envelope whitening/recolouring with Röbel's low-band no-warp rule; a SHIP variant is to be tested too. Ship only if MUSHRA shows a gain |
| **Formant** | All PSOLA engines | Preserved by construction (γ = 1). Knob γ = 2^(st/12) by grain resampling, clean within ±3 st (±18%). Throat length: φ = L_ref/L_throat. Beyond ±3 st: per-grain true envelope (order Pc ≈ fs/(2·f0), 2-3 iterations) or CheapTrick (3·T0 window, q1 = −0.15), with a low-band no-warp sigmoid |

---

## 6. Correction behaviour: prior art

| System | What it contributes | Verdict |
|---|---|---|
| Hildebrand US5973252 | Target = nearest scale note, or MIDI note + bend. The raw resample ratio is smoothed by a user Decay 0-1 (0 = instant), which implies a one-pole on the *correction*. Vibrato LFO with depth, rate and onset delay. **No hysteresis** | **Adopt as baseline** |
| Antares semantics ([Artist manual](https://www.barryrudolph.com/recall/manuals/auto-tune_artist.pdf), [Pro X guide](https://antares-web-frontend.sfo3.cdn.digitaloceanspaces.com/documentation/pdfs/Auto-Tune_Pro_X_User_Guide_10.0.pdf), [UA Realtime Advanced manual](https://media.uaudio.com/support/manuals/dd/Auto-Tune%20Realtime%20Advanced%20Manual.pdf)) | Retune Speed in ms (0 = instant and suppresses vibrato; 10-50 natural; tutorials span 0-400). Humanize = slower retune only on the sustained part of long notes. Flex-Tune: corrects only near the target, and higher values shrink the region; notes are "instantly tuned as the sweep enters the correction range". Natural Vibrato ±12, works even when every note is bypassed. Targeting Ignores Vibrato. On/Bypass/Remove (scale-editor edits apply to all octaves; keyboard edits to one octave). MIDI Target Notes (passes audio through when no note is held), Learn Scale, All Octaves. Detune ±100 c, Transpose ±12 st, Tracking default 50. Classic mode disables Formant/Throat/Transpose/Flex (the list differs by product) | **Adopt as the UX spec**. Exact curves are undocumented: calibrate by measurement |
| Antares DVA (Advanced, Sept 2026) | Vibrato detected and passed through at any retune speed | **Match the capability** with our own method; freedom-to-operate review needed |
| Waves Tune Real-Time | Speed (in-note, 0.1-800 ms, default 15) vs Note Transition (default 120; Link option). Tolerance: Cents (Off-40) added to the 50 c halfway point, *or* Time (Off-300 ms) past the halfway point; **"once either threshold is crossed"** the transition begins (OR logic). Vibrato Depth 0-200%. States Legal/Illegal/Illegal-Up/Illegal-Down/Bypass. Range bypass. MIDI target overrides Bypass | **Adopt** two time constants, OR-tolerance, Depth %, directional states |
| zita-at1 / fat1 | `notebias = Bias/13` octave on the current note (default Bias 0.5 ⇒ switch about 23 c past the midpoint); error filtered by a one-pole when the note is unchanged, jump when it changes; corrfilt = 4·frsize/(Filter·fs) (Euler, ≈ hop/τ); Correction 0-1, Offset ±2 st, bend range 2 (max 7); MIDI refcount per pitch class; Auto mode (MIDI if held, else manual scale) | **Adopt mechanisms** (not code; GPL). Replace the jump-on-change with τ_tr |
| Autotalent | Memoryless soft quantiser (sine-segment transition width "Smooth"); note states 1/0/−1; all-removed falls back to chromatic; voiced threshold 0.7 | **Consider** as an optional "Snap Curve" |
| TalentedHack | Constant-velocity slew limiter | **Reject** (mechanical) |
| GSnap | Threshold (snapping region), separate Attack/Release, nearest held MIDI note, bend up to 200 c | **Adopt** attack/release for voicing and MIDI release |
| Graillon 3 | Snap Min dead zone, Snap Max, Inertia, Amount, Smooth | **Adopt** dead zone and Snap Max |
| Melodyne | Centre / drift (slow) / modulation (fast); −100..+100% scaling | **Adopt** the decomposition as the conceptual model |
| Harman/Cor-Tek US9123353 | Histogram-adaptive boundaries (from vocal *and accompaniment*), an error-difference hysteresis of about 0.1 st (≈5 c past midpoint for adjacent semitones), adaptive delay | **Avoid**: Active, anticipated expiry 2033-09-30 |
| pYIN MonoNote HMM ([c4dm/pyin](https://github.com/c4dm/pyin)) | Per-pitch attack/stable/silent states; fixed-lag decoding supported. Defaults (from `MonoNoteParameters.cpp`, checked): nS 69, minPitch 35, nPPS 3, nSPP 3, pAttackSelftrans 0.9, pStableSelftrans 0.99, pStable2Silent 0.01, pSilentSelftrans 0.9999, sigma2Note 0.7, maxJump 13, priorPitchedProb 0.7, priorWeight 0.5, minSemitoneDistance 0.5, sigmaYinPitchAttack 5, sigmaYinPitchStable 0.8, sigmaYinPitchInter 0.1, yinTrust 0.1 | **Adopt** (reimplement; GPL) as the HQ/Max/Studio note decoder |
| Causal extrema vibrato tracker (design proposal from the correction report) | Turning points with 5 c hysteresis; rate 3.5-9 Hz; half-extent 8-150 c; centre anchor (p1 + 2p2 + p3)/4, which cancels vibrato and linear drift | **Adopt** for engagement and GUI readout |
| Christensen 2012 Kalman | AR(1) "fast" state, 0.99 per sample (≈2.3 ms time constant), which absorbs bends and slides; evaluated qualitatively on two guitar takes | **Not a vibrato model.** Use for mean pitch; replace the fast state with an AR(2) resonator at f_v |
| FFT-of-contour vibrato (SoundHound US8494842, expired for fees; Ventura et al.) | 320 ms windows at a 20 ms hop | **Studio GUI statistics only** |
| Saitou F0 dynamics | Overshoot ω = 0.0348 rad/ms, ζ = 0.5422; preparation ω = 0.0292, ζ = 0.6681; vibrato ω = 0.0345, ζ = 0 | **Optional** "human glide" shape in hard-tune/MIDI modes |
| Neural APC: Deep Autotuner, KaraTuner, Diff-Pitcher, BERT-APC, MIDIBack | Context- and accompaniment-aware intended-note inference (BERT-APC MOS 4.32; MIDIBack RPA drops from 81.5% to 35.8% without accompaniment) | **Reject for real time**. Lesson: use causal key and chord priors |
| Harmony / key assists | Antares AutoKey 2 integration; CNN key classifier ([Korzeniowski & Widmer 2018](https://arxiv.org/abs/1808.05340)) | **Adopt off the audio thread**: key profile on a histogram of stable notes; optional sidechain chroma chord prior |

## 7. The chosen design

### 7.1 Signal flow

```
 MIDI in (note on/off, pitch bend, CC64 sustain, CC1 mod, CC120/123 panic) ───────────────────┐
                                                                                              │
 audio in (mono or stereo) ──┬──────────────────────────────────────────────────────────────┐ │
                             │                                                              │ │
                             ▼  analysis tap (selected channel or (L+R)/2)                  │ │
 ┌───────────────────────────── ANALYSIS PATH (adds no audio latency) ──────────────────┐  │ │
 │ DC blocker / HPF 0.6·f_min ─► [opt. SIFT-LPC or Praat Gaussian pre-LPF]              │  │ │
 │   ├─► sibilance / energy features (full rate, >4 kHz ratio, ZCR, RMS)                │  │ │
 │   └─► minimum-phase IIR decimators                                                   │  │ │
 │         ├─ 8-12 kHz ─► E/H pitch-synchronous candidate generator (all preset lags,   │  │ │
 │         │              update every 4-8 samples, (E-2H)/E <= eps(Tracking))          │  │ │
 │         └─ 16/24 kHz ─► per-hop FFT verifier: NSDF/WSNAC clarity + key maxima,       │  │ │
 │                         type-I CMNDF -> pYIN-style probabilities, prime-harmonic sum │  │ │
 │   ─► candidate set (<= 5 per hop) ─► octave checks ─► [Max: fastF0ArMl re-score,     │  │ │
 │      neural arbiter (gated)] ─► pitch HMM (Live: forward | Studio: fixed-lag)        │  │ │
 │   ─► full-rate sub-sample refinement (3-lag parabolic / sinc) ─► voicing FSM         │  │ │
 │   ─► mean-pitch Kalman (innovation gating)                                           │  │ │
 └──────────────────────┬───────────────────────────────────────────────────────────────┘  │ │
         f0, conf, voiced, sibilant, candidates                                            │ │
                        ▼                    ┌──── f_in from mark spacing (Live, ~0 lag) ◄─┐ │ │
 ┌────────────────────── CONTROL LAW (per hop Δ = 64 or 128 samples) ─────────────────┐  │ │ │
 │ source select (marks vs detector) ─► voicing hold/forget ─► octave guard           │  │ │ │
 │ ─► pre-smooth (4 ms) ─► centre/vibrato split (extrema tracker [+ AR(2) Kalman;     │  │ │ │
 │    Studio: fixed-lag RTS]) ─► note decision on centre (Eco: rules + OR-tolerance;  │◄─┼─┼─┘
 │    HQ/Max: MonoNote HMM forward; Studio: fixed-lag Viterbi) with scale / MIDI      │  │ │
 │    candidates, On/Bypass/Remove/Up/Down, key prior                                 │  │ │
 │ ─► error e = flex·snap·amount·(target + detune + bend - ref)                       │  │ │
 │ ─► retune one-pole on the correction (τ | τ_tr, + Humanize) ─► + (g_v-1)·w_v·v     │  │ │
 │ ─► + synthetic vibrato ─► + 100·Transpose ─► S [cents] ─► r = 2^(S/1200)           │  │ │
 └──────────────────────┬──────────────────────────────────────────┬─────────────────┘  │ │
      S(t) interpolated per sample                    telemetry ─► rtrb SPSC ─► GUI piano roll
                        ▼                                                               │ │
 ┌──────────────────── AUDIO PATH (fixed reported latency D per latency mode) ────────┐  │ │
 │ input ring (2^16+ samples) ◄────────────────────────────────────────────────────────┼──┘ │
 │ ─► pitch-mark tracker (Live: NCC chaining + LP-residual anchor;                     │─────┘
 │                        Studio: SEDREAMS/ZFF GCI + DP path)                          │
 │ ─► engine: L0 Classic splice | L1/L2 Live PSOLA | S1 Studio PSOLA | [S2 spectral]   │
 │      grains read at fractional positions with formant factor γ (grain resampling)   │
 │      [per-grain envelope re-colouring for |formant| > 3 st or S2]                   │
 │ ─► voiced/unvoiced equal-gain crossfade with matched-delay dry x[t-D]               │
 │ ─► output: identical r(t) and grain timing on both channels                         │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

Design invariants:

- **The analysis never delays audio.** Its information lag affects only the correction values.
- **The control law runs at a hop rate.** Its output S(t) is linearly interpolated per sample, which avoids zipper noise.
- **One analysis drives both channels.** Stereo input is analysed once, and the same ratio and grain timing are applied to both channels.
- **Fixed reported latency.** It depends only on the latency mode and the voice-range preset. Both are non-automatable.

### 7.2 Latency modes and quality tiers in detail

#### 7.2.1 Latency modes

| Mode | Reported D at 48 kHz | Engines allowed | Content-delay wander | Intended use |
|---|---|---|---|---|
| **Live-Min** | About 1 ms (≈48 samples: interpolation kernel + splice crossfade) | L0 Classic only | 0 to T (resample + splice) | Hard-tune live; lowest possible monitoring latency (Antares Classic class) |
| **Live** (default) | D_live(range) = L_rise + d_mark + h_interp + margin ≈ 3-5 ms (estimate; see below) | L1/L2 PSOLA; L0 Classic, padded to D | Causal grain choice: mean T/2, worst T (about 15-17 ms worst case for bass voices) | Live monitoring, natural correction |
| **Live+** | Up to 10 ms (user-selectable 5/7.5/10 ms) | Same as Live; symmetric 2·T grains when T + d_mark ≤ D | As Live | Singers who accept slightly more latency for better grains |
| **Studio** | 20-50 ms, default 40 ms; minimum depends on voice range | S1 PSOLA; S2 optional; L0 Classic padded | Nearest-mark selection: ±T/2 around D (mean ≈ 0) | Tracking with PDC, mixing, offline render |

**Live latency formula.** The rise is L_rise = 1-1.5 ms. The mark-detection delay is d_mark ≈ 0.1·T + W_f ≈ 0.2·T_max for the preset. The interpolation half-kernel is h_interp: 2 samples for cubic, 4-8 for 8-16-tap sinc. For the presets this gives, as estimates to be fixed by measurement:

- Soprano (T_max 7.1 ms): ≈ 1.0-1.5 + 1.4 + 0.1 ≈ 2.5-3 ms + margin
- Alto/Tenor (T_max 12.5 ms): ≈ 1.0-1.5 + 2.5 + 0.1 ≈ 3.6-4.1 ms + margin
- Low Male (T_max 18.2 ms): ≈ 1.0-1.5 + 3.6 + 0.1 ≈ 4.7-5.2 ms + margin

With content wander, the mean total delay comes to about 5-9 ms for 80-250 Hz voices. That is inside the user's 5-10 ms target.

**Voice range belongs to the latency mode.** The voice-range preset (Input Type) sets f_min, and with it the detector windows, d_mark, the Studio minimum D and the S2 window. It is therefore part of the latency configuration and is non-automatable. Ableton's Auto Shift sets the precedent, since its pitch range also changes latency. Presets:

| Preset | Range |
|---|---|
| Low Male | 55-600 Hz |
| Alto/Tenor | 80-1000 Hz |
| Soprano | 140-1400 Hz (optional whistle extension to about 2.1 kHz) |
| Wide | 55-1400 Hz |

#### 7.2.2 Quality tiers (per latency mode)

Tiers never change D. Switching tiers crossfades over one hop (outputs are aligned because D is identical). A CPU governor may only **demote** a tier on overload (p99 block time above budget), with a crossfade, and never silently promotes.

| Block | Eco | HQ (default) | Max |
|---|---|---|---|
| Analysis rate / hop | Search at 8-12 kHz; verifier at 16 kHz; hop 128 (2.67 ms) | Verifier at 24 kHz; hop 64 (1.33 ms) | As HQ. Neural arbiter at a 5-10 ms hop (more CPU buys model width or rank, not a shorter hop) |
| Candidates | E/H + single-window NSDF + CMNDF probabilities | + two-resolution WSNAC (long 2.2/f_min, short about 2.2/(2·f_min)); prime-harmonic octave check every hop | + fastF0ArMl top-K re-scoring with harmonic-order tracking; optional harmonic-chirp scoring on vibrato/glide segments; neural arbiter (gated) fused by delay-aligned voting |
| Pitch HMM | 20-cent bins | 10-cent bins | 5-cent bins |
| Refinement | Full-rate 3-lag parabolic on notes above about 400 Hz | Full-rate 3-lag parabolic every hop | Sinc / NLS gradient refinement |
| Vibrato / centre | Extrema tracker | + AR(2) resonator Kalman, gated by the extrema tracker | + detector-lag compensation (clip ±25 c) |
| Note decision | Rules: fat1-style bias + Waves OR-tolerance | MonoNote-style HMM, forward posterior | + wider candidate range, key prior, optional sidechain chord prior |
| Shifter interpolation | 4-point cubic Hermite | 8-16-tap Kaiser-windowed sinc (or 2x oversampling) | As HQ |
| Marks | NCC chaining + LP-residual anchor | + causal multi-hypothesis Viterbi over the last 3-6 periods; multi-candidate grain selection (nearest 2-3 grains by overlap correlation) | As HQ |
| Formant | Grain resampling (±3 st clean) | As Eco | + per-grain envelope re-colouring beyond ±3 st; experimental band-split for aperiodic energy |
| Studio additions | Centred windows, fixed-lag HMM, lookahead onset decisions | + GCI marks with DP path, RTS smoother, fixed-lag Viterbi notes, envelope warping beyond ±3 st | + SWIPE′ low-note second opinion (D ≥ 50 ms), S2 spectral engine option |
| CPU (estimate) | Detector about 0.2% of a core; PSOLA < 1% | Detector about 1-2%; PSOLA with sinc < 1-2% | Detector about 3-8% (benchmark first) + neural ≤ 5% (≤ 0.25 ms per 5 ms hop on an M1 P-core) |

**Offline render.** The host signals offline processing through the process mode in the BufferConfig (the NIH-plug API lineage has `ProcessMode::Offline`; confirm the nice-plug spelling). An option, *Max quality on render* (default Off, design choice), raises the tier while rendering. It is off by default because a bounce then differs from what was monitored (Section 11.3).

### 7.3 Pitch detector per cell (summary)

- **All cells.** DC blocker → minimum-phase IIR decimation. The E/H pitch-synchronous candidate generator runs over the preset lag range. A per-hop FFT verifier (NSDF + type-I CMNDF) and an octave check follow. Then the causal pitch HMM, full-rate refinement, the voicing state machine (FSM) and the mean Kalman.
- **Live / Live+ / Live-Min.** Windows are right-aligned. The HMM forward filter runs with no backtrace. In Live and Live+, the PSOLA mark spacing gives the instantaneous period used by the control law, whenever it agrees with the detector to within 50 c.
- **Live-Min.** Hildebrand's exact layout: 8:1 decimation, lags 2-110 at the decimated rate (about 50-2756 Hz at 44.1 kHz), harmonic check at L and 2L at full rate, and an 8-lag full-rate track with a quadratic fit.
- **Studio.** Windows are centred on the output sample (W ≤ 2D). Fixed-lag HMM smoothing over L = D/hop frames. An RTS smoother on the centre/vibrato state. SWIPE′ as a second opinion at D ≥ 50 ms in Max.

### 7.4 Shifter per cell (summary)

| | Eco | HQ | Max |
|---|---|---|---|
| Live-Min | L0 splice, cubic read, no crossfade (patent form) | L0 with a 1-1.5 ms raised-cosine crossfade, sinc read | As HQ |
| Live | L1 asymmetric short-rise grains, cubic | L2: sinc, multi-hypothesis marks, multi-candidate grains | + envelope re-colouring beyond ±3 st; research: causal neural post-filter (gated) |
| Live+ | L1, symmetric 2·T grains when they fit, otherwise asymmetric | L2 symmetric | As above |
| Studio | S1 lookahead PSOLA, NCC marks, symmetric grains | S1 with GCI/DP marks, envelope warping | S1 or S2 (voice-range dependent, if A/B approved) |

### 7.5 Formant preservation and shift

1. **Preservation is inherent.** PSOLA grains are not resampled when γ = 1, so the short-time envelope is kept. With β = 1 the input is reconstructed almost exactly.
2. **Formant knob.** γ = 2^(formant_st/12), applied by reading the grain source at rate γ:
   `y(t) += w_k(t − s_k) · interp(x, m_j + γ·(t − s_k))`.
   When γ > 1, low-pass the source by γ (a sinc kernel scaled by 1/γ). The clean range is about ±3 st (IVL: ±18%).
3. **Throat length.** φ = L_ref/L_throat, so formants scale with 1/(vocal tract length). It multiplies γ.
4. **"Formant follows transpose"** (option). γ_total = γ · (follow ? 2^(transpose/12) : 1). This reproduces AutoTune Advanced's linked Transpose+Formant.
5. **Beyond ±3 st (Studio HQ, Live Max).** Per-grain envelope re-colouring:
   `G_out(k) = G(k) · E_in(k·w(k)) / E_in(k)`, where `w(k) = D(k) + (1 − D(k))/γ` and `D(k) = 1/(1 + exp((k − k0)/T_k))`.
   This is Röbel's sigmoid. It leaves the region around f0 unwarped so the level of the fundamental is preserved. E_in comes from a true envelope (cepstral order Pc ≈ fs/(2·f0), 2-3 iterations) or from CheapTrick (3·T0 window, q1 = −0.15).
6. **Classic engine.** Formants move with pitch, which is intended. Any LPC formant-correction post-filter is a later experiment.
7. **Unvoiced input** passes through unshifted. IVL also does not shift sibilants.

### 7.6 Correction control law

#### 7.6.1 Conventions

- Pitch is expressed in absolute cents: `p = 6900 + 1200·log2(f/440)` (i.e. MIDI × 100).
- **Notes.** For a 12-TET note n: `pitch(n) = 100·n + D_t + o(n)`, where D_t is Detune (±100 c) and o(n) is an optional per-note microtonal offset. A non-12-TET scale is a sorted list `(cents_from_root_i, state_i)` repeated every 1200 c.
- **Note states:** On, Bypass, Remove, Up, Down.
  - Scale-editor edits apply to every octave. Keyboard edits are per-octave overrides (128 entries).
- **Control hop and smoothing.**
  - Control hop Δ: 64 samples in Live HQ/Max, 128 in Eco and Studio.
  - One-pole coefficient: `α(τ) = 1 − exp(−Δ/τ)`, with `α = 1` when τ = 0.
  - Recompute α every hop from smoothed parameter values.
  - Do not parameter-smooth knobs that are themselves time constants.

#### 7.6.2 Per-hop pipeline (equations)

**A. Pitch source (Live L1/L2).**
- `p = p_mark` (from mark spacing, `f_in = fs/(m_{j+1} − m_j)`) when PSOLA is locked and `|p_mark − p_det| < 50 c`.
- Otherwise `p = p_det`.

**B. Voicing finite-state machine (FSM).**
- Raw voicing: `voiced_raw = conf ≥ θ(Tracking) ∧ rms ≥ gate ∧ ¬sibilant`.
- Onset needs 2 consecutive voiced hops.
- During unvoiced time t_unv:
  - if `t_unv > T_hold` (30 ms): `c ← c + α(τ_rel = 40 ms)·(0 − c)`, and the vibrato component goes to 0;
  - if `t_unv > T_forget` (100 ms): forget the note, reset the vibrato tracker, set `g_s = 0`.
- **Octave guard:** accept `|Δp| > 700 c` only after 20 ms of consistent estimates. Until then, hold p and c.

**C. Pre-smoothing.** `p_s = onepole(p, 4 ms)`. This removes fine fluctuation above 10 Hz. It is optional at τ = 0.

**D. Centre/vibrato split** (§8.11).
- `(centre, w_v, f_v, A_v, φ_v) = vib.update(p_s)`
- `p_c = w_v·centre + (1 − w_v)·p_s`
- `v = p_s − p_c`

**E. Decision pitch and candidates.**
- `p_dec = IgnoreVibrato ? p_c : p_s`.
- Candidate set:
  - MIDI Target active with notes held: the held notes (absolute), or every octave of the held pitch classes when All Octaves is on.
  - Otherwise: the scale notes in state On, Bypass, Up or Down. Remove notes are excluded.
- If the candidate set is empty, fall back to chromatic.
- Notes outside the Range limit act as Bypass.

**F. Note decision.**
- **Eco (rules).**
  - Let k be the nearest candidate to p_dec, and n* the current note.
  - Midpoint: `mid = (pitch(n*) + pitch(k))/2`.
  - Distance past the midpoint: `x = (p_dec − mid)·sign(pitch(k) − pitch(n*))`.
  - Switch to k if `(Cents on ∧ x > T_c) ∨ (Time on ∧ x > 0 continuously for ≥ T_t) ∨ (both off ∧ x > 0)`. This is Waves' OR semantics: whichever enabled threshold is crossed first triggers the transition.
  - Redirection: an Up note redirects to the next legal note above, a Down note to the next legal note below.
- **HQ / Max (HMM).** A MonoNote-style forward filter (§8.12). Switch when the posterior mass of the new note's states exceeds p_sw (start at 0.5 and tune).
- **Studio.** Fixed-lag Viterbi with lag `L = D/Δ`, so hysteresis costs no dwell delay. Accept candidate b if the median of p_dec over `[t, t + min(L, 30 ms)]` lies past the midpoint by T_c.
- **Onsets.**
  - Live: onset grace `T_grace = (τ ≤ 5 ms) ? 0 : 20 ms`, then decide the note. If `τ ≤ 10 ms` or the engine is Classic, snap the onset: `c = e`.
  - Studio: the onset note is the one nearest the median p_dec over the first `min(L, 40 ms)`, applied from sample 0.

**G. Error.**
- Correction reference: `ref = PreserveVibrato ? p_c : p_s`. In Max Live, use the lag-compensated ref (§8.10).
- `e0 = (state(n*) = Bypass ∧ ¬MIDI) ? 0 : pitch(n*) + bend_c − ref`. MIDI overrides Bypass.
- `e1 = Amount · e0`, with Amount in [0, 1].
- Snap Min (soft dead zone): `e2 = e1 · x² / (1 + x²)`, with `x = |e1|/D0`. A hard variant is `sign(e1)·max(0, |e1| − D0)`.
- Snap Max: `if |p_dec − pitch(n*)| > SnapMax then e2 = 0`.
- Flex-Tune weight:
  - `d = |p_dec − pitch(n*)|`.
  - G = gap to the adjacent legal note on the side of the deviation (100 c when chromatic).
  - `R = 0.5·G·(1 − F/100)`.
  - `w(d) = 1` for `d ≤ (1 − k_t)·R`.
  - `w(d) = 0.5·(1 + cos(π·(d − (1 − k_t)·R)/(k_t·R)))` for `(1 − k_t)·R < d < R`.
  - `w(d) = 0` for `d ≥ R`.
  - F = 0 gives w ≡ 1. The default taper fraction is k_t = 0.4 (design choice). Antares describes a near-hard edge, so calibrate k_t downward against the A2-A3-A2 sweep.
- `e = w · e2`.
- Clamp automatic correction to |e| ≤ 600 c. With a MIDI target, Snap Max defaults to 1200 c.

**H. Retune filter on the correction.** The filter acts on the correction, never on the target.
- Transition window: after a target change, set `trans = true` and `τ_eff = τ_tr`. End the window (`trans = false`, `τ_eff = τ`) once `t_note ≥ 3·τ_tr` or `|e − c| < 5 c`.
- `τ_eff ← τ_eff + g_s · τ_h(H)` (Humanize, additive, so it also works at τ = 0).
- Update: `c ← c + α(τ_eff)·(e − c)`.
- Optional "Smooth" shape: two cascaded one-poles of τ/2 each (critically damped, no overshoot). Optional "Human glide" (hard-tune and MIDI modes only): a Saitou 2-pole with ω = 34.8 rad/s and ζ = 0.54.
- Classic option: τ_tr = 0, so c jumps to e on a note change (fat1 behaviour).

The meaning of Retune Speed follows from this. With `ref = p_s`, the output's deviation from the target is a *high-pass* of the input deviation:

```
H(f) = j2πfτ / (1 + j2πfτ),   cutoff fc = 1/(2πτ)
residual vibrato at 5.5 Hz:  τ =   0 ms → 0      τ =  30 ms → 0.72
                             τ =   5 ms → 0.17   τ =  50 ms → 0.87
                             τ =  10 ms → 0.33   τ = 100 ms → 0.96
                             τ =  20 ms → 0.57   τ = 200 ms → 0.99
```

This reproduces the Antares wording: 0 suppresses vibrato, 10-50 ms sounds natural. **Retune Speed is defined as the 63% time constant τ** (design choice). The state-of-the-art report instead proposed τ = RS/3 (RS = time to 95%). Antares documents neither, so an "Antares-compatible" mapping is to be fitted from a 50-cent step measured through Auto-Tune (Section 11.3).

**I. Humanize** (sustain-gated slow-down).
- `stable = |p_c − pitch(n*)| < 40 c ∧ |dp_c/dt| < 200 c/s` (slope over 20 ms).
- `g_s ← clamp(g_s + Δ/T_r, 0, 1)` if `stable ∧ t_note > T_s`, otherwise `g_s ← clamp(g_s − Δ/20 ms, 0, 1)`. Here T_s = 150 ms and T_r = 150 ms.
- `τ_h(H) = 300 ms · (H/100)²`. This gives +19, +75, +169 and +300 ms at H = 25, 50, 75 and 100.
- Reset g_s on a note change or an unvoiced gap. Short notes keep the fast τ, which matches Antares' workflow.

**J. Vibrato scaling.**
- `s_vib = (g_v − 1)·w_v·v`, where g_v is Vibrato Depth in −100..200%. Negative values invert the vibrato (Melodyne-style).
- It applies even when the note is bypassed (Antares behaviour).
- With PreserveVibrato on, τ = 0 and g_v = 1: `p_out = p_in + (pitch(n*) − p_c) = pitch(n*) + v`. The centre is hard-tuned and the vibrato is kept, which is the DVA-like behaviour.
- With g_v = 0: `p_out = pitch(n*)`, which is classic hard-tune.

**K. Synthetic vibrato.**
- `s_lfo = A(t)·shape(2π·f_lfo·t + φ0)`.
- Shapes: sine, triangle, square, saw.
- Defaults: rate 5.5 Hz (0.1-10 Hz), depth 30 c, onset delay 250 ms, onset ramp 300 ms, per-note variation 10%. The phase resets per note.
- CC1 maps to depth (0-100 c).

**L. Output.**
- Total shift: `S = c + s_vib + s_lfo + 100·Transpose`, clamped to `|S| ≤ 2400 c`.
- Ratio: `r = 2^(S/1200)`, with S linearly interpolated per sample across the hop.
- Unvoiced: `S → 100·Transpose`, and the ratio ramps over 5 ms. The shifter crossfades to the matched-delay dry signal. Transpose stays applied, so voicing boundaries do not jump; an option bypasses unvoiced segments.
- Output pitch: `p_out = p_in + S`.

#### 7.6.3 MIDI targets

- **Modes**
  - **Off.**
  - **Target:** held notes form the candidate set. The nearest held note wins, with the same tolerance.
  - **Learn Scale:** clears the scale, then each note-on sets that pitch class (or note) to On.
  - **Auto:** behaves as Target when any note is held, otherwise uses the scale (fat1 mode 0).
- **All Octaves.** Pitch-class refcounting (`note % 12`). Default is Off in Target mode and On in Learn mode.
- **Optional Mono-Last priority.** A fixed 16-slot note stack. On legato note-off, return to the previous held note using τ_tr.
- **Note-off of all notes.** Pass through untuned, as Antares does: `c → 0` with `τ_rel_midi = 50 ms`. The *Latch while voiced* option holds the last note until the next unvoiced gap.
- **Note-on during an unvoiced gap.** Arm it, then apply it with onset snap at the voiced onset.
- **Pitch bend.** `bend_c = bend·range`, with a default range of ±2 st (0-12 st), smoothed by a 5 ms one-pole.
- **Other controllers.** CC64 holds notes. CC120/123 is panic. The channel filter defaults to Omni.
- **Timing.**
  - Events arrive sample-accurate and are applied at the next hop (≤ 1.33 ms later).
  - In Studio, events are visible L ahead, so a glide can start at `t_event − τ_tr/2` and be centred on the note-on.

#### 7.6.4 Recommended defaults

| Parameter | Default | Range | Notes |
|---|---|---|---|
| Latency mode | Live | Live-Min / Live / Live+ / Studio | non-automatable |
| Voice range | Alto/Tenor | Low Male / Alto-Tenor / Soprano / Wide | non-automatable (sets D) |
| Quality tier | HQ | Eco / HQ / Max | automation discouraged (crossfades) |
| Engine | Natural (PSOLA) | Classic / Natural / (Studio: Smooth = S2 if shipped) | |
| Key / Scale | C / Chromatic | 12 keys × scales + custom | design choice |
| Retune Speed τ | 20 ms | 0-400 ms, log | 63% time constant |
| Note Transition τ_tr | Linked to τ | 0.1-800 ms (120 when unlinked) | |
| Humanize | 0 | 0-100 | |
| Flex-Tune | 0 | 0-100 | k_t = 0.4 (calibrate) |
| Ignore vibrato for targeting | On | | |
| Preserve vibrato (centre correction) | On | | Off = Auto-Tune 5 behaviour |
| Vibrato Depth | 100% | −100..200% | Antares-style ±12 only after measurement |
| Tolerance | 8 c / Off | 0-40 c / 0-300 ms | OR semantics |
| Amount | 100% | 0-100% | |
| Snap Min / Snap Max | 0 / Off | 0-50 c / 1-12 st | |
| Detune | 0 | ±100 c | |
| Transpose | 0 | ±12 st | applied after correction |
| Formant / Throat | 0 st / 100% | ±12 st (clean ±3) / 50-200% (design choice) | |
| Tracking | 50 | 0-100 | maps to eps, clarity threshold, pYIN prior and HMM switch probability |
| Gate | −55 dBFS, 6 dB hysteresis | | |
| Hold / Release / Forget / Onset grace | 30 / 40 / 100 / 20 ms | | |
| MIDI mode / bend range / MIDI release | Off / ±2 st / 50 ms | | |
| Synthetic vibrato | Off | rate 5.5 Hz, 30 c, delay 250 ms, ramp 300 ms, variation 10% | |

**Presets** (from the correction-behaviour research):

| Preset | Settings |
|---|---|
| **Hard Tune** | τ = 0, τ_tr = 0, Flex 0, Humanize 0, Ignore Vibrato On, Depth 0%, Tolerance 5 c / 0 ms, onset snap On |
| **Natural** | τ = 25 ms (linked), Humanize 30, Flex 20, Depth 100%, Tolerance 15 c / 30 ms |
| **Live Safe** | τ = 15 ms, Tolerance 10 c / 0 ms, Depth 100% |
| **Classic** | Classic engine, τ = 0-5 ms, instant jump on note change, onset snap, formant off, Preserve vibrato Off |

## 8. Implementation notes and pitfalls for each block

Notation:

| Symbol | Meaning |
|---|---|
| fs | Host sample rate |
| fs_a | Analysis rate |
| x | Input ring buffer |
| T_a | Local input period, in samples (fractional) |
| β = f_target/f_in | Pitch factor |
| γ | Formant factor |
| D | Reported latency |
| Δ | Control hop |

Absolute sample positions are **f64**, or are wrapped relative to the ring base. Never use f32 for them.

### 8.0 Processing frame (applies to every block)

- **Block-size agnostic.** Accept any host block size, from 1 sample upward. An internal hop accumulator runs analysis whenever a hop completes. Test with block sizes {1, 7, 32, 64, 128, 480, 4096, random}.
- **Work spreading.** At a 32-sample host buffer, a callback lasts 0.67 ms at 48 kHz. Never put the per-hop FFT, the HMM update and a neural inference into one callback. Stagger them across callbacks inside the hop. Keep the worst-case block time bounded, not just the mean.
- **Allocation.** Allocate everything in `initialize()`/`activate()` from the maximum block size and sample rate: rings of 2^16 samples or more, FFT plans and scratch, HMM arrays, mark rings, grain pool (≤ 4 active), MIDI stack (16), telemetry ring. Nothing is allocated in `process()`.
- **Hop-dependent constants.** Recompute all of them on any sample-rate or hop change: HMM transition width, switch probability, Kalman a^hop, MonoNote self-transitions, one-pole α, and window lengths.
- **Sample rates.** Support 44.1-192 kHz. Do all heavy analysis after decimation. At 96 and 192 kHz the full-rate refinement and PSOLA run at fs, and the costs scale accordingly (an NSDF frame at 96 kHz is 34 µs, measured on the M4 Max).

### 8.1 Input conditioner

- **DC blocker:** `y[n] = x[n] − x[n−1] + 0.995·y[n−1]`. Alternatively, a 2nd-order HPF at 0.6·f_min of the preset.
- **Stereo.** Analyse the selected channel. (L+R)/2 is an option, with a warning about phase cancellation on stereo-miked sources. Apply identical r(t) and grain timing to both channels.
- **Sibilance (full rate, before decimation).** Flag a frame as unvoiced if `E_HP(>4 kHz)/E_total > ~0.5`, or if ZCR > ~0.25 per sample at 48 kHz. Breath is aperiodic: low clarity, broadband.
- **Optional pre-filter (A/B test it).**
  - Praat-style Gaussian low-pass, attenuation 0.03 at the preset's top frequency. Praat reports more than 3x fewer gross errors with it on speech, but it can miss periodicity when only high harmonics are present.
  - Low-order LPC inverse filter (SIFT). It flattens formants while keeping high-harmonic periodicity.
  - Keep any cutoff above f_max.

### 8.2 Decimators

- Phase does not matter for detection, so **use minimum-phase IIR decimators** (for example a 4th-order elliptic) rather than linear-phase half-band cascades, which add milliseconds of group delay. **Count the decimator group delay in the detection-lag budget.**
- Search rate is 8-12 kHz. Use ≥ 12 kHz for Soprano and whistle. Coarse lags may be small (Hildebrand searched lags down to 2 at 5.5 kHz) because refinement happens at full rate.
- Verifier rate: 16 kHz (Eco) or 24 kHz (HQ/Max). Keep `τ_min = fs_a/f_max ≥ 8` samples (at 1.4 kHz: 11.4 samples at 16 kHz, 17 at 24 kHz).
- Divide 44.1/48 kHz by 2-3 for the verifier and by 4-6 for the search. 88.2/96 kHz: by 4 and 8. 176.4/192 kHz: by 8 and 16.

### 8.3 Pitch-synchronous E/H candidate generator (default, every tier)

For each candidate lag L in [L_min, L_max] (from the preset, at the decimated rate), maintain:

```
E_i(L) = E_{i-1}(L) + x_i^2       − x_{i-2L}^2            (energy over the last 2L samples)
H_i(L) = H_{i-1}(L) + x_i·x_{i-L} − x_{i-L}·x_{i-2L}      (lag product over the last L pairs)
E − 2H = Σ_{k=i-L+1..i} (x_k − x_{k-L})^2,   so (E − 2H)/E = 1 − NSDF over a 2L window
periodic(L)  ⇔  E − 2H ≤ eps·E,   eps = map(Tracking): 0..100 → 0.05..0.40
```

One of the reports printed a different recursion, with terms x[n−3L] and mixed window lengths. That version is **wrong** and breaks the identity above. Use the patent form given here.

- **Candidates.** Take the first local minimum of (E−2H)/E below eps (the smallest qualifying lag), plus its octave relatives at 2L and L/2. Also check the second minimum against full-rate E−2H, to handle missing-fundamental and octave cases (Hildebrand).
- **Update rate.** Update E and H every decimated sample. Evaluate the minima every 4-8 samples. Measured cost on the M4 Max: about 0.03-0.1% of a core.
- **Numerical drift.** Use f64 accumulators, or re-initialise E and H from scratch every hop. Gate silence first: at silence E → 0, so add an epsilon.
- **Full-rate local tracking (HQ/Max).** After locking a period P, keep E and H for 8 lags around P at full rate, updated per sample. Every 4-5 samples, fit a 3-point quadratic through the minimum of (E−2H)/E to get a sub-sample period.
  - If the minimum hits the edge of the band, shift the band.
  - If (E−2H)/E > eps, drop the lock and defer to the frame tracker.
  - This tracks sustained notes and vibrato with about 1-2 periods of lag instead of W/2.
- **Mapping to the user's Tracking knob.** eps is the mechanism behind Antares "Tracking" (an inference from the patent and manual). One knob, default 50, maps onto eps, the NSDF clarity threshold, the pYIN threshold prior and the HMM switch probability. Relax it for breathy voices and make it choosier if clicks appear. The exact mapping is tuned by listening.

### 8.4 Per-hop FFT verifier (NSDF/WSNAC + type-I CMNDF + pYIN probabilities)

Frame x[0..W) is right-aligned in Live and centred in Studio. W = 2.2/f_min: Low Male about 40 ms, Alto/Tenor 27.5 ms, Soprano 15.7 ms. Keep τ_max ≤ W/2 to 0.6·W, because MPM's type-II window shrinks with τ.

```
N = next_pow2(W + τ_max);  X = rfft(x zero-padded to N);  r = irfft(|X|^2)/N
m[0] = 2 r[0];  m[τ] = m[τ−1] − x[τ−1]^2 − x[W−τ]^2
nsdf[τ] = 2 r[τ] / max(m[τ], ε)                         (MPM, in [−1, 1])
key maxima: highest nsdf between each +→− zero-crossing pair, τ ∈ [τ_min, τ_max]
MPM pick: first key maximum ≥ k·max, k = 0.9 (paper range 0.8-1.0)
parabolic: δ = (y[i−1] − y[i+1]) / (2(y[i−1] − 2y[i] + y[i+1])),  peak = y[i] − 0.25(y[i−1] − y[i+1])δ
           (interpolate nsdf or raw d, never cmndf: YIN step 5 notes it is biased)
```

- **CMNDF must be type-I.** pYIN's Beta(2,18) threshold prior was calibrated on YIN's type-I difference, which uses a fixed W. The type-II d = m − 2r shrinks with lag. On white noise it gave a median minimum CMNDF of 0.61 over the long-lag half, against 0.89 for type-I, so it reports false periodicity at long lags. **Fix:** keep a W + τ_max buffer and compute the type-I difference:
  - `d(τ) = Σ_{j<W} x_j^2 + Σ_{j<W} x_{j+τ}^2 − 2 Σ_{j<W} x_j·x_{j+τ}`.
  - The cross term is one FFT correlation of the W-sample segment against the W + τ_max buffer. The energy terms come from prefix sums.
  - Alternatively, normalise the type-II d by (W − τ), or use 1 − NSDF.
  - Then `cmndf[0] = 1` and `cmndf[τ] = d[τ]·τ / Σ_{j=1..τ} d[j]`.
  - Recalibrate the Beta prior on vocal data.
- **pYIN probabilities** (librosa defaults):
  - 100 thresholds s = 0.01..1.00 with Beta(2,18) weights.
  - For each s, take the first CMNDF trough below s. If none, assign the global-minimum trough with no_trough_prob = 0.01.
  - Boltzmann parameter 2 over trough ranks.
  - Accumulate P(τ). Unvoiced probability = 1 − ΣP.
- **WSNAC (HQ/Max).** McLeod's windowed NSDF is more consistent on frequency ramps, so use it for the short resolution of the two-resolution verifier. The DSP report's "do not window before the NSDF" holds only for the plain MPM path.
- **Output:** at most 5 tuples (τ, clarity, prob) per hop.
- **Cost (measured, M4 Max, rustfft complex FFT).** 16.4 µs per frame at 48 kHz (N = 4096), 3.5 µs at 12-16 kHz (N = 1024). realfft should roughly halve this. Budget for M1 and older x86 too.

### 8.5 Octave and harmonic checks

For the best candidate τ and its relatives τ/2 and 2τ (when in range):

- `S_nsdf` = NSDF value.
- `S_hs = Σ_{k ∈ {1} ∪ primes ≤ K} k^{−1/2}·sqrt(P(k·f))`. This is a SWIPE′-like prime-harmonic sum read from the existing power spectrum.
- Hildebrand rule: prefer the larger lag only if its full-rate (E−2H)/E ≤ that of the smaller lag + margin.
- Praat-like octave cost of 0.01-0.055 per octave, favouring higher f, plus an octave-jump penalty in the HMM (Praat uses 0.35; treat it as a starting log-penalty).
- Optional *octave-class* prior from the MIDI target or the previous stable note. **Never** apply a fine-pitch prior: it would bias detection toward the scale.

Failure modes to expect:

- *Too high*: F1 near 2·F0 (high soprano notes, open vowels).
- *Too low*: vocal fry, subharmonics, diplophonia.

Vocal fry handling (track f or f/2, or add a fry state?) is an open question.

### 8.6 Pitch HMM (pYIN-derived, causal)

- **States.** Bins of 20/10/5 cents (Eco/HQ/Max) over the preset range, doubled for V/UV. Wide (55-1400 Hz) is about 5600 c, i.e. 560 bins at 10 c.
- **Transitions.**
  - Triangular within ±D bins, with `D = ceil(35.92 oct/s · 1200 · hop_s / bin_c)`. At a 2.67 ms hop that is about 115 c, so D = 12 at 10 c.
  - Add a uniform jump floor (about 1e-4) so that real leaps stay reachable, plus the octave-jump penalty.
- **Voicing switch probability per hop:** `p = 1 − (1 − 0.01)^(hop/23.2 ms)`. This rescales librosa's 0.01, which was set for a 512/22050 hop.
- **Observations.** Map candidate probabilities to bins, spread ±1 bin. Unvoiced states take the unvoiced probability.
- **Live (forward only).**
  - `α_t = normalize(band_matmul(A, α_{t−1}) ⊙ b_t)`.
  - Output the argmax with a local expected value over ±2 bins. Penn showed this helps only a little (12.72 → 12.45 c).
  - The continuous f0 is the refined candidate closest to the chosen bin.
- **Studio (fixed-lag).** Keep the last L = D/hop alphas and observations. Run the backward pass every hop and output the smoothed posterior at t − L.
- **Evidence gap.** No published result covers fixed lags of 20-50 ms. The pYIN Vamp fixed-lag mode is hard-wired to 100 frames. Our harness must establish the benefit.
- **Cost.** About 28k MAC per frame with banded transitions (estimate), under 0.5% of a core.

### 8.7 Full-rate refinement

- Evaluate the full-rate NSDF (or E−2H) at 3 lags around fs/fs_a × the decimated lag, then refine parabolically. This is essential above about 400 Hz when decimating: local tests gave 0.15-3.5 c of error at 12 kHz against 0.04-0.86 c at 48 kHz.
- Max tier: Boersma sinc interpolation of the lag peak, or NLS gradient refinement.
- Real-voice error is dominated by jitter and non-stationarity, not by interpolation.

### 8.8 Voicing, breath, sibilance

- **Energy gate.** 10 ms RMS below the gate (default −55 dBFS, 6 dB hysteresis) means unvoiced. Praat's relative silence thresholds are 0.03 (raw autocorrelation) and 0.09 (filtered); for live use, apply them relative to a running peak.
- **Periodicity.** Unvoiced if NSDF clarity < 0.5 (relaxed) to 0.7 (choosy), or if the HMM voiced posterior < 0.5. For reference, Praat's voicing threshold is 0.45-0.5, and YIN's d′ at the dip is about 0.1-0.35.
- **Hysteresis and hold.** Voiced onset after 2 consecutive voiced hops (about 3-5 ms). Release after a 20-40 ms hold, so short consonants do not reset the tracker.
- **Outputs to the control law:** voiced probability, confidence, and a sibilance flag.

### 8.9 Max tier: harmonic-model re-scoring and the neural arbiter

- **Harmonic-model re-scoring.**
  - Re-score the top-K SDF candidates with **fastF0ArMl** (ML in autoregressive coloured noise, BSD-3, notify on commercial use) rather than fastF0Nls, which makes octave errors when the noise is concentrated at low frequencies (rumble, HVAC, bleed).
  - Track harmonic order with Shi et al.'s causal predict/update form.
  - On segments flagged as vibrato or glide, optionally use harmonic-chirp ML, which also yields a slope estimate.
  - Benchmark the CPU cost before enabling: about 1-5% of a core per voice is an estimate.
- **Neural arbiter (only after the gate in §10.10).**
  - The arbiter gives f0_n and confidence c.
  - DSP then searches NSDF/CMNDF only over `τ ∈ [fs/(f0_n·2^(50/1200)), fs/(f0_n·2^(−50/1200))]` on a window of about 2-3 periods.
  - Accept the DSP result f0_fine if its clarity is ≥ θ_d and `|1200·log2(f0_fine/f0_n)| < 50`. Otherwise use f0_n, interpolated between bins.
  - Voicing = (c > θ_v) ∧ energy gate, with hysteresis (on above 0.5, off below 0.35).
- **Fusion.** Fuse the DSP result with one or two neural members by delay- and bias-aligned voting (ICASSP 2026). Each estimator carries its measured information lag as a timestamp. **Measure step response; do not trust reported lag.** PESTO's buffer-refill lag is reporting lag, not information lag.

### 8.10 Mean-pitch Kalman and detector-lag compensation

- **State.** Pitch in cents (mean, velocity). Measurement noise depends on clarity: `R = R0/max(clarity, 0.1)^2`.
- **Innovation gating.** If |innovation| > 100-150 c and the HMM bin has moved for 2 or more hops, treat it as a note change and reset the mean to the measurement.
- **Do not use Christensen's AR(1) "fast" state as a vibrato model.** At 0.99 per sample its time constant is about 2.3 ms (0.28 per 2.67 ms hop), so it soaks up bends and slides as well. His per-sample values (a_m = 1 − 1e−6, a_v = 0.99, C = diag(25e−10, 4e−8) rad², σ_w² = 1e−6 at 44.1 kHz) are starting points only. Convert them per hop with `a_hop = a^hop_samples`.
- **Lag compensation (Live Max, optional).** The detector reports pitch at t_now − D_det.
  - Predict `p(t_now) = p_c + A_v·cos(φ_v + 2π·f_v·D_det)` when vibrato is active, else `p + ṗ·D_det`.
  - Clip to ±25 c.
  - Validate by listening at retune 0: it must reduce warble without overshoot. A 10 ms lag on a 6 Hz ±50 c vibrato can cause up to about 19 c of error.

### 8.11 Vibrato / centre tracker

**Extrema tracker** (Live default, causal):

- **Input.** p_s, pre-smoothed by a one-pole of about 4 ms or a 20 Hz 2-pole.
- **Turning points.** Detect maxima and minima with 5 c hysteresis.
- **Per extremum.** `hp = t_e − t_{e−1}` and `A = |p_e − p_{e−1}|/2`. Valid when 1/(2·hp) ∈ [3.5, 9] Hz and A ∈ [8, 150] c.
- **Engage** after 3 valid, consistent extrema (hp_max/hp_min ≤ 1.6 and A_max/A_min ≤ 2.0). Then ramp w_v from 0 to 1 over one half-period.
- **Centre anchor.** `(p_{e−2} + 2·p_{e−1} + p_e)/4`, placed at the middle extremum time. This cancels both vibrato amplitude and linear drift. Between anchors, extrapolate along the slope of the last two anchors, clamped to ±300 c/s.
- **Outputs.**
  - `p_c = w_v·centre + (1 − w_v)·p_s`
  - `v = p_s − p_c`
  - `f_v = 1/(2·mean hp)`
  - `A_v = mean A` (half of peak-to-peak)
- **Disengage** when no extremum arrives within 1.6·mean hp. A transition, `|p_s − centre| > 1.8·A_v + 25 c`, resets centre = p_s and ramps w_v to 0 over 10 ms.
- **Classification.** A half-extent above 150 c is a trill or a note, not vibrato.
- **Engagement delay.** About one vibrato cycle (170-250 ms) after vibrato onset. **In Live at retune 0 the first vibrato cycle is flattened.** This is unavoidable causally: document it and measure it. MetaTune's Note Stabilizer shows that the lookahead alternative costs about 280 ms.

**HQ/Max refinement.** A 3-state Kalman filter with state [c, v, v_prev]:

- c is a random walk, `q_c ≈ (2 c)²` per hop.
- v is an AR(2) resonator: `v_t = 2ρ·cos(ω_v·Δ)·v_{t−1} − ρ²·v_{t−2}`, with ρ ≈ 0.98 and ω_v = 2π·f_v taken from the extrema tracker.
- Measurement noise `R ≈ (3 c)²`.
- It is gated by w_v. It gives a smooth centre between anchors and a continuous vibrato phase for lag compensation. It acts as a Kalman adaptive notch with near-zero group delay once locked.

**Studio.** A fixed-lag RTS smoother over the same model, using all L ms of lookahead. It engages vibrato earlier and removes the half-period anchor lag. Run the extrema tracker on the advanced contour as well. An FFT-of-contour estimator (320 ms windows at a 20 ms hop) supplies stable rate and extent readouts for the GUI only.

### 8.12 Note decision HMM (MonoNote-style; HQ/Max/Studio)

- **States.** For each candidate note n (On, Bypass, Up and Down notes; never Remove), three states: attack, stable, silent. These follow pYIN's `MonoNoteParameters.cpp` (nSPP = 3).
- **pYIN defaults to start from:**

| Parameter | Value |
|---|---|
| pAttackSelftrans | 0.9 |
| pStableSelftrans | 0.99 |
| pStable2Silent | 0.01 |
| pSilentSelftrans | 0.9999 |
| sigma2Note (as named in the source) | 0.7 |
| maxJump | 13 semitones |
| priorPitchedProb | 0.7 |
| priorWeight | 0.5 |
| minSemitoneDistance | 0.5 |
| sigmaYinPitchAttack | 5 |
| sigmaYinPitchStable | 0.8 |
| sigmaYinPitchInter | 0.1 |
| yinTrust | 0.1 |

- **Hop rescaling.** These defaults were calibrated at pYIN's own hop. Rescale self-transitions as `p' = p^(Δ/Δ_pyin)`.
- **Observations.** A Gaussian of p_dec (the vibrato-free centre) around the note pitch, with σ scaled by detector confidence. The voiced probability comes from §8.8.
- **Transition prior.** Key-aware: diatonic steps are more likely (design choice). An optional chord-tone weight comes from the sidechain (§9.6).
- **User knobs.** The Tolerance knobs map onto σ and the stable self-transition probability (design choice, calibrate it).
- **Decoding.** Live: forward posterior with a switching threshold. Studio: fixed-lag Viterbi with lag L/Δ. Up/Down redirection is applied *after* decoding.
- **Cost.** Sparse transitions over a few hundred states: microseconds per hop (estimate).
- **Why.** It replaces about six interacting heuristics (tolerance, dwell, onset grace, octave guard, voicing hold, forget timer) with a few interpretable parameters, and it scales with CPU.
- **Risk.** Its benefit over rules is unproven. Compare flip-flop rate and onset-note accuracy in the harness, and keep the rules as Eco.

### 8.13 Pitch-mark tracking

**Live (causal):**

- **Seed.** Take T from the detector, but use the mark spacing itself as the instantaneous period: `f_in = fs/(m_{j+1} − m_j)`.
- **Search.** Predict `p = m_j + T` and search τ ∈ [0.9T, 1.1T]. Vibrato of ±1 st at 6 Hz changes the period by only about 2-3% per period. Widen the search to ±20% at onsets and after low-confidence periods.
- **Score.** Normalised cross-correlation (NCC) of `x[m_j − W_b, m_j + W_f]` against `x[m_j + τ − W_b, m_j + τ + W_f]`, with W_b = 0.5-0.75·T and W_f = 0.1·T. Run it coarse on 2-4x decimated data, then fine at full rate, then parabolic to get τ*. Then `m_{j+1} = m_j + τ*`.
- **Anchor.** Each period, find the argmax of |e[n]| within ±0.3·T, where e is the LP residual (order 2 + fs/1000, autocorrelation method, recomputed per period). If the chained marks drift more than about 0.15·T from the anchor, re-anchor gradually, by at most about 0.02·T per period, so the marks never jump.
- **Unvoiced** if NCC_max < 0.5, or detector aperiodicity > 0.3, or energy is below the gate.
- **Detection delay** ≈ 0.1·T + W_f ≈ 0.2·T. That is 2.5 ms at 80 Hz.
- **HQ.** Causal multi-hypothesis tracking: a Viterbi over (NCC, period continuity) across the last 3-6 periods, with 1-2 ms of lookahead.
- **Octave guard.** Confirm T against 2·T. At 2·T spacing each grain contains two pulses whose spacing does not shift, which produces doubling.

**Studio (GCI):**

- **SEDREAMS.** Compute the mean-based signal with a Blackman window of about 1.75·T0_mean (it needs the detector's mean pitch). Its lookahead is about 0.875·T0. Take the glottal closure inside each bracketed interval as the maximum of the LP residual.
- **ZFF** (zero-frequency resonator plus trend removal) is the alternative.
- **Path.** Choose marks by dynamic programming (DP) / Viterbi with cost `w1·|log((m_{j+1} − m_j)/T0)| + w2·(1 − NCC)`.
- **Validation.** The GCI benchmarks cover speech EGG only. Validate on singing, especially high f0.

### 8.14 Synthesis marks and grain scheduling

- **Synthesis marks.** `s_{k+1} = s_k + T_s(k)`, with `T_s(k) = T_a(s_k)/β(s_k)`. Update β once per synthesis period, so the control rate is pitch-synchronous and continuous ratio changes need no extra smoothing.
- **Grain source.**
  - Live: the latest detected m_j with `m_j ≤ s_k + e_max`, where e_max is bounded by D.
  - Studio: the nearest m_j.
  - The content offset `e_k = m_j − s_k` wanders within about one input period. Upward shifts reuse grains and downward shifts skip them. Bound the offset to [0, lookahead].
- **Mix control.** A wet/dry mix on the pitch path causes time-varying comb filtering. Either offer no mix control or crossfade only near β = 1.
- **Telemetry.** Send the GUI f_in from the mark spacing and f_target per synthesis mark, timestamped in output time s_k, so the graph matches what is heard.

### 8.15 Grain windows

- **Symmetric** (Studio, and Live/Live+ when T + d_mark ≤ D). Grain k spans [s_k − h_r(k), s_k + h_f(k)]:
  - Rising half: `w = 0.5(1 + cos(π·τ/h_r))` for τ ∈ [−h_r, 0].
  - Falling half: `w = 0.5(1 + cos(π·τ/h_f))` for τ ∈ [0, h_f].
  - `h_f(k) = h_r(k+1) = min(T_s(k), ρ·T_a(j)/γ)`, with ρ ≈ 0.9.
- **Shifting up** (T_s < T_a): h = T_s, so adjacent halves are complementary and the window sum is exactly 1. This is the Lent/IVL rule of a window about 2× the target period. No normalisation is needed, and at β = 1 the output is near-transparent.
- **Shifting down** (T_s > ρ·T_a): h = ρ·T_a < T_s, which leaves a zero gap. That is plausible because formant ringing has decayed by then. **Never extend a grain past the neighbouring pulse**, or you get doubling or echo.
- **Asymmetric low-latency** (Live, low voices):
  - Rise over [s_k − L, s_k], with L ≈ 1-1.5 ms and L ≤ 0.25·T_s.
  - Flat at 1 until s_{k+1} − L, then fall complementarily to the next grain's rise.
  - If `γ·T_s > ρ·T_a`, start the fall early at `s_k + ρ·T_a/γ − L` and leave a gap.
  - Normalise the overlap-add by the summed window wherever the window does not sum to 1.
- **Status.** The asymmetric variant is untested on singing (it is a design proposal). Label it experimental until MUSHRA approves it.

### 8.16 Fractional reads, interpolation and formant read

- `y(t) += w_k(t − s_k) · interp(x, m_j + γ·(t − s_k))`.
- **Fractional positions are mandatory.** Example: at f0 = 500 Hz, fs = 48 kHz and T = 96 samples, ±0.5 sample of rounding is about ±9 c of random period jitter, which is audible roughness.
- **Interpolator.** 4-point cubic Hermite (Eco) or 8-16-tap Kaiser-windowed sinc (HQ/Max/Studio), or 2x oversampling.
- **When γ > 1**, low-pass the source by γ (a sinc kernel scaled by 1/γ) to avoid aliasing.
- **Lookahead.** The grain's source span γ·h must stay within the available lookahead. This limits how far γ can go above 1 in Live.

### 8.17 Classic engine (L0)

- **Read pointer.** Advance it by r per output sample, interpolating at about 5 samples behind the pointer (patent) or one kernel half-width behind.
- **Cycle repeat/drop.**
  - When the read pointer would overtake the write pointer, jump back exactly one detected period (a cycle is repeated).
  - When it lags beyond the allowed delay, jump forward one period (a cycle is dropped).
- **Splice.**
  - Eco (patent form): no crossfade.
  - HQ: a raised-cosine crossfade of 1-1.5 ms. zita-at1 uses one 128-sample fragment at 2x internal rate, about 1.3 ms. Jump by a whole number of periods at least as long as the crossfade.
- **Period source.** The Hildebrand full-rate 8-lag track.
- **Ratio smoothing.** The control law supplies smoothing, playing the role of the patent's Decay.
- **Latency.** In Live and Studio modes, Classic is padded with delay to that mode's D.
- **Formants.** They follow the pitch: +1 st gives about +5.9%. That is the intended character.

### 8.18 Unvoiced, breath, transients, onsets

- **Unvoiced** (or confidence below threshold): output = x[t − D], a pure delay with ratio 1.
- **Voiced → unvoiced.** If |e_k| < 0.1·T, switch at the next mark. Otherwise run a 3-5 ms equal-gain crossfade from the PSOLA output to the delayed dry signal. Equal gain is right because the two signals are correlated.
- **Unvoiced → voiced.** Start with s_0 = m_0 (e = 0) and β = 1, then let the correction curve move β. This avoids glitches while the pitch locks (about 2 periods).
- **Plosive bursts** are unvoiced and therefore bypassed.
- **Breathy or voiced-fricative noise** gets periodised when grains repeat (buzz). This is mild within ±1 st. For larger shifts, the Max-tier experiments are:
  - a band split (complementary crossover at about 4-6 kHz, or at the estimated maximum voiced frequency), with the high band delayed by D and not shifted. Caveat: its fixed delay misaligns with PSOLA's wandering content delay.
  - a pitch-synchronous comb harmonic/noise split that keeps the aperiodic part unshifted.
- **Validation.** These are design inferences. Validate them by listening.

### 8.19 Very low and very high voices

- **Low** (E2 at 82 Hz has T = 12.2 ms; fry is below 70 Hz). Use asymmetric grains in Live, because symmetric grains would need 13-15 ms. Guard against subharmonic and octave errors.
- **High** (above about 600 Hz, T < 1.7 ms). Grains are tiny and formants are sparsely sampled, so PSOLA quality falls with pitch (Röbel). Use sinc interpolation and fractional marks, and accept the information limit. In Studio, the S2 engine is the candidate fix.

### 8.20 Per-grain envelope re-colouring (Studio HQ, Live Max)

1. Take the FFT of the windowed grain. Use size next_pow2(2·T·max(1, γ)), roughly 2048 points at 55 Hz.
2. Estimate E_in:
   - True envelope: iterative cepstral smoothing, taking max(log spectrum, current envelope) each pass. Order Pc ≈ fs/(2·f0) for a rectangular lifter, slightly higher with Hamming. Use 2-3 iterations; Lenarczyk ran 2 in real time, though with fixed ratios.
   - Or CheapTrick.
3. Output: `G(k) · E_in(k·w(k)) / E_in(k)` with Röbel's low-band sigmoid (§7.5). IFFT, re-window, overlap-add.
4. Known problems:
   - Dullness when shifting high voices down.
   - Amplified unvoiced high-frequency parts when shifting down.

### 8.21 Studio S2 spectral engine (prototype before committing)

- **Window.** f0-adaptive: N ≈ 4/f0_min, i.e. 20-27 ms for f0_min = 150-200 Hz. Hop N/8 to N/4.
- **Phase.** PGHI phase reconstruction (phase-gradient heap integration, no peak picking), or identity phase locking.
- **Envelope.** True-envelope whitening before the shift and re-colouring after it, with the low-band no-warp rule.
- **Transients.** Flag them.
- **Noise.** Optionally do a harmonic/noise split, so breath is not periodised.
- **Latency.** Analysis + synthesis + any split-computation interval must fit within D.
- **Prototype.** Use signalsmith-stretch 0.1.3 (MIT, cc build of C++) to prototype in days. ssstretch 0.1.0 was published the day before this review and has only a Linux docs build; do not depend on it.
- **Also test** a SHIP-style shape-invariant variant (6-8·T_min, 30-53 ms for these ranges) and the online DNN phase-difference method (Masuyama et al.) as a way to avoid the PGHI lookahead frame.
- **Ship S2 only if it wins MUSHRA** for alto and soprano ranges.

### 8.22 Neural kernels (only if a model passes the gate)

- **Implementation.** Hand-written, allocation-free kernels in the RTNeural/nam-rs pattern:
  - Fixed topology.
  - Weights as safetensors 0.8.0, embedded with `include_bytes!` and `bytemuck` casts, or loaded off-thread and swapped in through basedrop.
  - Scratch preallocated.
  - NEON through `wide` on aarch64 (`vfmaq_f32`-style 4-output × 16-bin blocking), AVX2 on x86.
  - Fast tanh/sigmoid by rational approximation (error < 1e-4).
- **GRU** (PyTorch convention):

```
z = σ(W_z x + b_z + U_z h + c_z);  r = σ(W_r x + b_r + U_r h + c_r)
n = tanh(W_n x + b_n + r ⊙ (U_n h + c_n));  h' = (1 − z) ⊙ n + z ⊙ h
```

- **Measured (M4 Max).** A hand-written GRU(128) + dense step (162k params) runs at a median of 10.8 µs (p99 20.6 µs, max 38.1 µs) with 0 allocations. tract 0.23.8 runs the same step at 8.4 µs but makes 44 allocations per run. rten 0.24 runs it at 19.8 µs single-threaded; its default multithreading is worse (72 µs median, 915 µs max). **Small models are not CPU-bound. Allocation and tail latency are the constraints.**
- **Parity CI.** Export from PyTorch to ONNX and run reference outputs through tract-onnx 0.23.8 in `cargo test`. Require max abs error < 1e-4 and identical argmax on more than 99.9% of a held-out set. FrCN reached 100% on 271,893 frames.
- **Training plan.**
  - Distil the offline teachers (SwiftF0 v0.3 + RMVPE) into a causal student on licensed or self-recorded vocals, plus synthetic exact-F0 singing (WORLD/DDSP resynthesis).
  - Align labels to the newest sample (LACOPE-style).
  - Add PESTO-style self-supervised losses.
  - Augment with noise down to −20 dB SNR (FrCN: the range of training noise mattered about 100x more than the architecture), background music, room IRs and random EQ.
  - **Do not ship any public checkpoint.**
    - PESTO code is LGPL-3.0, and its weights are trained on MIR-1K only.
    - SwiftF0's weights are about 35% CC BY-NC MDB-stem-synth.
    - FrCN has no code.
    - PitchDNN's weights are trained on speech.

### 8.23 GUI telemetry and the piano roll

- **Record.** `#[derive(Clone, Copy)] struct PitchFrame { sample_pos: u64, detected: f32, centre: f32, target: f32, output: f32, confidence: f32, vib_rate: f32, vib_extent: f32, flags: u8 }` (flags: voiced, note state, transition, MIDI-held).
- **Audio side.** Push one frame per analysis hop (about 375/s at 48 kHz with hop 128) into an rtrb SPSC ring, **only while the editor is open**. If the ring is full, the frame is dropped. Then request a repaint (RepaintNotifier is an atomic store).
- **GUI side.**
  - Drain the ring into a GUI-owned history of 5-10 s.
  - Compute `now_pos = last.sample_pos + elapsed·sr` (clamped) so scrolling stays smooth even though block delivery is bursty.
  - Draw, in order:
    - piano-roll lanes, with scale notes highlighted (from a triple_buffer snapshot) and state colours (On, Bypass orange, Remove grey);
    - target note bars, including detune and bend;
    - the Flex zone;
    - the vibrato band;
    - the detected trace, thin, with alpha = confidence, broken at unvoiced frames;
    - the centre line;
    - the output trace, thick;
    - a cents meter (Antares convention: negative = pitch being lowered);
    - a vibrato rate/extent readout.
  - Decimate to about 2 points per pixel column (min/max) and reuse `Vec<Pos2>` buffers.
- **Timestamps.** Use output time, so the graph lines up with what is heard.
- **Optional.** A before/after heat map (MetaTune's idea).

## 9. Rust stack

### 9.1 Decisions

| Concern | Choice | Why / notes |
|---|---|---|
| Framework | **nice-plug =0.4.2** ([Codeberg](https://codeberg.org/RustAudio/nice-plug), ISC, MSRV 1.88, released 2026-09-14) | This is the community continuation of NIH-plug, and upstream NIH-plug points users to it (upstream has been in maintenance mode since 2026-05-10). It exports VST3, CLAP and standalone from one codebase. Its VST3 wrapper has used the MIT/Apache `vst3` 0.3.0 bindings since June 2026 (PR #29), so it no longer depends on the GPLv3 `vst3-sys`. Releases are frequent, the API breaks every minor version, and macOS testing is "limited". |
| VST3 SDK | MIT since 3.8 (Oct 2025; 3.8.1 tagged Aug 2026) ([Steinberg](https://steinbergmedia.github.io/vst3_dev_portal/pages/VST+3+Licensing/Index.html)) | Closed or permissive licensing of our VST3 is now possible. Using the VST name or logo still requires following Steinberg's trademark rules. |
| Formats | VST3 for Ableton Live 12 (Live 12.4.2 has no CLAP, per the official release notes). CLAP (and VST3) for Bitwig 6.1.3 | Ableton testing has to go through VST3. CLAP note ports can only be tested in Bitwig. |
| GUI | **nice-plug-egui =0.5.1** (egui 0.36.2, egui-baseview 0.7.2, baseview 0.3.7) | Immediate-mode GUI with a Painter API, enough for a 60 fps piano roll. Repaints on demand through RepaintNotifier. The OpenGL backend is the default; Metal via `wgpu` (wgpu 30.0.1) is a feature flag. Both builds were verified. |
| Toolchain | **Rust 1.99.0** pinned (stable, dist manifest dated 2026-10-01); workspace `rust-version = "1.95"` | The machine's default is 1.89, and cargo refuses to build egui 0.36 on it ("egui@0.36.2 requires rustc 1.95"). tract 0.23 needs 1.91, rten 0.25+ needs 1.94, and burn 0.22 and clap-validator 0.4.1 need 1.95. |
| Plan B | clack 0.2.0 (CLAP, MIT/Apache) + clap-wrapper-rs (VST3/AUv2). crates.io 0.3.1 bundles clap-wrapper 0.14; 0.16 is git-only | Fallback if nice-plug stalls. Also the route to AUv2 for Logic, via `export_auv2!()` on nice-plug's `clap_entry` |
| Rejected | Upstream nih-plug (GPLv3 VST3, maintenance mode), coupler ("not production-ready"), truce 6.3.0 (one maintainer, quiet since 2026-07-18, open state-recall bug #232), beamer (no CLAP), AURA (GPL-3.0+), vizia-plug (pinned to nice-plug-core 0.1.4) | |

**Existing repo note.** `xtask/Cargo.toml` currently pulls `nih_plug_xtask` from upstream nih-plug (git rev `f36931f7…`). Switch it to `nice-plug-xtask = "=0.1.1"`, or keep every nice-plug crate on one git rev so that there is only one nice-plug-core. The xtask hard-codes `CFBundleIdentifier = com.nice-plug.<package>`, which with the package name in `bundler.toml` gives `com.nice-plug.vocal_tune`. Patch the plist and re-sign if a different identifier is wanted.

### 9.2 Crates

| Crate | Version | License | Role | Notes |
|---|---|---|---|---|
| nice-plug | =0.4.2 | ISC | framework | features `assert_process_allocs` (defaults: editor, vst3) |
| nice-plug-egui | =0.5.1 | ISC | editor | `opengl` by default; try `wgpu` (Metal) in week 1 |
| egui | 0.36.2 | MIT/Apache | GUI | MSRV 1.95 |
| nice-plug-xtask | =0.1.1 | ISC | bundling | ad-hoc codesigns bundles |
| vst3 / clap-sys | 0.3.0 / 0.5.0 | MIT/Apache | transitive bindings | |
| rtrb | 0.4.0 | MIT/Apache | wait-free SPSC: audio→GUI frames, GUI→audio commands, object return queue | |
| triple_buffer | 9.0.0 | MPL-2.0 | latest-state snapshots (scale mask, CPU stats) | file-level copyleft; fine unmodified |
| basedrop | 0.1.3 | MIT/Apache | deferred-free smart pointers for large swaps | |
| atomic_float | 1.1.0 | Apache/MIT/Unlicense | meters, scalars | |
| crossbeam-utils | 0.8.23 | MIT/Apache | `CachePadded` | **not** crossbeam-channel on the audio thread |
| realfft / rustfft | 3.5.0 / 6.4.1 | MIT / MIT-Apache | FFTs (NEON) | measured on M4 Max: 2048-point forward+inverse in 3.06 µs; plan in `activate()`, call `process_with_scratch` |
| wide | 1.7.1 | Zlib/Apache/MIT | explicit SIMD (f32x4/x8) | MSRV 1.89; `std::simd` is still nightly-only |
| multiversion / fearless_simd | 0.9.0 / 1.1.0 | MIT-Apache / Apache-MIT | x86 AVX2/FMA dispatch | only when shipping x86 |
| rubato + audioadapter-buffers | 5.0.1 + 5.2.0 | MIT/Apache | resampling, only if needed on the audio path | `process_into_buffer` does not allocate; add `output_delay()` to D; keep its `log` feature off |
| biquad | 0.6.0 | MIT/Apache | DC blocker, simple filters | or a hand-written TPT SVF (Simper) |
| no_denormals | 0.3.0 | MIT | FTZ/DAZ on x86 only | not needed on aarch64; changing the FP environment is technically UB in Rust |
| safetensors / bytemuck | 0.8.0 / 1.25.2 | Apache / MIT-Apache | NN weights | |
| hound, approx, criterion, proptest | 3.5.1, 0.5.1, 0.8.2, 1.11.0 | permissive | tests and benches (dev) | |
| clack-host | 0.2.0 | MIT/Apache | offline integration-test host (dev) | |
| tract-onnx | 0.23.8 | MIT/Apache | NN parity oracle (dev only) | allocates on every run |
| rtsan-standalone | 0.3.0 (2025-09-27) | Apache-2.0 | RealtimeSanitizer in the test harness (dev) | catches locks and syscalls, not only allocations |
| tracy-client | 0.19 | permissive | profiling builds | per-block histograms and spikes |
| *Rejected on the audio thread* | ort 2.0.0-rc.13, candle 0.11, burn 0.22, rten 0.27 | | | allocate per call and/or use thread pools; ANIRA found real-time violations in ONNX Runtime, LibTorch and TFLite |

### 9.3 Workspace layout

```
vocal_tune/
  rust-toolchain.toml        channel = "1.99.0", components = ["rustfmt","clippy"],
                             targets = ["aarch64-apple-darwin","x86_64-apple-darwin"]
  Cargo.toml                 workspace, resolver = "3", edition 2024, rust-version "1.95"
  .cargo/config.toml         [alias] xtask = "run --package xtask --release --" (exists)
  bundler.toml               [vocal_tune] name = "Vocal Tune" (exists)
  crates/vt_dsp/             pure DSP: detector, vibrato, note logic, control law, engines,
                             formant. No plugin deps; offline-testable; no_std-friendly core
  crates/vt_nn/              (optional) hand-written kernels + safetensors loading
  crates/vocal_tune/         nice-plug wrapper + egui editor; crate-type ["cdylib","lib"];
                             `standalone` behind a NON-default feature
  crates/vt_eval/            CLI that renders WAV through the exact process path
                             (block-size patterns), emits telemetry CSV; driven by Python
  xtask/                     fn main() -> nice_plug_xtask::Result<()> { nice_plug_xtask::main() }
```

Workspace dependency pins. The research session resolved and compiled this set with cargo 1.99.0 on arm64 macOS:

```toml
[workspace.dependencies]
nice-plug      = { version = "=0.4.2", features = ["assert_process_allocs"] }
nice-plug-egui = { version = "=0.5.1" }        # default "opengl"; evaluate features=["wgpu"]
egui           = { version = "0.36.2", features = ["default_fonts"] }
rtrb = "0.4.0"
triple_buffer = "9.0.0"
basedrop = "0.1.3"
atomic_float = "1.1.0"
crossbeam-utils = "0.8.23"
realfft = "3.5.0"
rustfft = "6.4.1"
wide = "1.7.1"
rubato = "5.0.1"
audioadapter-buffers = "5.2.0"
biquad = "0.6.0"
no_denormals = "0.3.0"     # cfg(target_arch = "x86_64") only
multiversion = "0.9.0"     # x86 dispatch only
safetensors = "0.8.0"
bytemuck = "1.25.2"
# dev
hound = "3.5.1"
approx = "0.5.1"
criterion = "0.8.2"
proptest = "1.11.0"
clack-host = "0.2.0"
tract-onnx = "0.23.8"
rtsan-standalone = "0.3.0"

[profile.release]
codegen-units = 1
lto = "thin"
strip = "symbols"

[profile.profiling]
inherits = "release"
debug = true
strip = "none"
```

Commit `Cargo.lock`. After every update, check that `cargo tree -d | grep baseview` still shows **one** version (0.3.7 today). nice-plug 0.4.2 uses the baseview 0.3 line, while egui-baseview 0.8 uses baseview 0.4. Expect a breaking nice-plug release when it moves to 0.4.

### 9.4 Real-time safety rules

1. **Allocate only in `initialize`/`activate`.** In `process()`, never allocate, lock, log, `format!`, drop the last `Arc`, or call into tract/rten/ort.
2. **Avoid the framework's hidden locks.** **Never call `ProcessContext::execute_background()` or `execute_gui()` in steady state.** nice-plug implements them with a bounded crossbeam channel, which takes a `Mutex` and unparks the worker on every send. The call is wrapped in `permit_alloc`, so `assert_process_allocs` cannot see it. On macOS, GUI tasks scheduled from the audio thread also call `CFRunLoopWakeUp`, a mach syscall. They are acceptable only for rare events such as a latency-mode change.
3. **Swapping large objects** (NN weights, tables, new scale maps): build them on the GUI or a background thread, send them over rtrb (or as `basedrop::Shared`), and return the old object over a second rtrb so it is freed off the audio thread.
4. **No panics in `process()`.** nice-plug has no `catch_unwind`, so a panic aborts the host, and Ableton runs plugins in-process. Use `get()`, saturating arithmetic and `debug_assert!`.
5. **Denormals.** FTZ is only needed on x86, under `cfg`. ARM is not affected.
6. **Enforcement:**
   - `assert_process_allocs` in debug builds.
   - **rtsan-standalone** around the process entry in a clack-host harness (run with `RTSAN_ENABLE=1` on the macOS CI runner). Optionally nightly `-Zsanitizer=realtime`.
   - pluginval `--rtcheck` (develop build).
   - A counting `#[global_allocator]` test binary for zero allocations after warm-up. It has to be a separate binary, because `assert_process_allocs` already occupies the crate's global allocator.
7. **Bounded worst case.** Fixed work per hop, spread across callbacks. No data-dependent loops without a cap (search widths, Viterbi beams, heap sizes all have fixed capacities).

### 9.5 Host latency reporting and parameters

- **Latency in `activate()`.** Allocate, then call `ctx.set_latency_samples(mode_latency(sr, voice_range, mode))`.
- **Latency mode change in `process()`.** Call `set_latency_samples(n)` **once**; nice-plug deduplicates repeated calls. VST3 then sends `restartComponent(kLatencyChanged)` on the GUI thread. CLAP sends `request_restart()`, and `latency.changed()` follows in the next `activate`, because the CLAP spec only allows latency to change during activate.
  - Expect a brief dropout while the host reactivates.
  - Never toggle latency rapidly. nice-plug #71 (a CLAP panic/deadlock in Bitwig) was closed in Aug 2026 without the reporter confirming the fix, and only tested on Linux. **Re-test it on macOS Bitwig.**
- **Ableton handler timing.** Ableton supplies the VST3 component handler late (iPlug2 #772), so a restart requested very early may be dropped while `getLatencySamples()` still returns the current value. Verify the reported latency after project reload and after toggling the device off and on.
- **What D includes:**
  - engine lookahead (rise + mark delay, or the Studio lookahead)
  - interpolation half-kernel
  - any resampler `output_delay` and any linear-phase filter delay on the **audio path**
  - S2 analysis + synthesis latency (and split-computation intervals)

  Analysis-path delays are *not* in D; they show up as detection lag.
- **Parameters.**
  - Stable `#[id]`s, `CLAP_ID` and `VST3_CLASS_ID` from day one.
  - These are `EnumParam`s and `.non_automatable()`: latency mode, voice range, engine, quality tier.
  - Per-pitch-class note state: 12 `EnumParam`s {On, Bypass, Remove, Up, Down}. Per-octave keyboard overrides are kept in plugin state, not in params.
  - Key and Scale: `EnumParam`.
  - `MIDI_INPUT = MidiConfig::Basic`.
  - Leave `SAMPLE_ACCURATE_AUTOMATION = false` at first, because it splits blocks.
  - `AUDIO_IO_LAYOUTS`: mono→mono, mono→stereo, stereo→stereo. Live usually instantiates stereo.
- **Smoothers.** Use the built-in smoothers for mix-like parameters (formant shift, transpose glide). Do not smooth time-constant parameters.
- **Ableton MIDI routing.** Sending a MIDI track into a VST3 *audio effect* (needed for MIDI target notes) is untested in Live 12. Make it an early acceptance test (§10.8).

### 9.6 Threads, CPU scaling and assists

- **No real-time worker threads of our own.** VST3 and CLAP give no handle to the host audio workgroup on Apple Silicon, and helper threads have been reported landing on E-cores. CLAP's thread-pool extension (Bitwig only) is not implemented in nice-plug. Everything on the audio path runs inline, with a known worst case.
- **Off-audio-thread assists** run on an ordinary background thread that we own, fed by rtrb (not `execute_background`). They never touch the audio path:
  - key detection: a key profile over a decaying histogram of stable notes;
  - an optional sidechain chroma/chord prior (VST3/CLAP sidechain bus, 100-300 ms latency is fine).

  Results reach the audio thread through `triple_buffer` and only bias the note-HMM prior.
- **CPU meter.** Per-block `Instant::now()` at the start and end of `process()`, kept as an EMA plus p99 in an `AtomicF32`, and shown next to the tier selector together with a measured per-machine tier recommendation. During development, use `tracy-client` for histograms.

### 9.7 GUI

- nice-plug-egui with the egui Painter: piano roll as described in §8.23. Put the drawing behind a thin trait (`draw_pitch_roll(painter, frames, scale)`) so that the coming baseview 0.4 / egui-baseview 0.8 migration stays contained.
- **Week 1:** test the OpenGL and wgpu/Metal backends in Live and Bitwig, including about 50 editor open/close cycles (watch baseview #124, closed 2026-10-04). Prefer Metal if it is stable, because OpenGL is deprecated on macOS.
- Known issues:
  - an extra click is needed for macOS focus (egui-baseview #21);
  - host-intercepted keys (`Editor::on_virtual_key_from_host`);
  - PR #96 (VST3 resize during editor init) is still pending;
  - the first frame can be blank (#95).
- Fallbacks: nice-plug-iced 0.4.1 (open macOS crash, iced_baseview #48), or raw baseview + wgpu once nice-plug moves to baseview 0.4.

### 9.8 Dev loop, signing, hosts

- **Build and install.**
  - `cargo xtask bundle vocal_tune --release` produces `target/bundled/*.vst3|.clap`, ad-hoc signed (`Signature=adhoc` was verified).
  - Copy the bundles (do not symlink) into `~/Library/Audio/Plug-Ins/{VST3,CLAP}`.
  - Both installed DAWs carry `disable-library-validation`, so ad-hoc signatures load.
- **Debug in Bitwig.** Both its audio engine and its plug-in host have `get-task-allow`, so `lldb -p` works in "Within Bitwig" mode and in the sandboxed modes. Use the "Individually" hosting mode for crash isolation.
- **Acceptance-test in Live.** Live runs plugins in-process without `get-task-allow`. Relaunch Live after every rebuild.
- **Re-sign** after any post-processing step (plist edits, `lipo`, `strip`): `codesign --force --deep -s - <bundle>`. Clear quarantine on downloaded CI artifacts with `xattr -dr com.apple.quarantine`.
- **Distribution** needs a Developer ID certificate, the hardened runtime, notarytool and stapling.
- **AU for Logic** (later): add it through clap-wrapper-rs on top of nice-plug's CLAP entry.

---

## 10. Evaluation plan

### 10.1 Principles

- **No public benchmark measures what we ship**, which is causal correction of singing at 1-50 ms. So we build our own harness and make it a CI gate from week one.
- The DSP core (`vt_dsp`) is host-independent. `vt_eval` renders WAV through the *exact* `process()` path, including block-size patterns, and writes telemetry. Python drives the metrics: pyworld for detuning, ViSQOL, VERSA and the F0 ensemble.
- **Never gate a release on delay-tolerant metrics alone.** For a tuner, lag *is* the defect.

### 10.2 Synthetic-detune round trip (primary protocol)

1. Start from in-tune singing with reliable F0.
2. Detune it with a pitch-accurate offline shifter. WORLD had the most stable pitch accuracy in Liu & Akama. The detuning generator applies:
   - per-note offset U(−80, +80) c;
   - drift as an Ornstein-Uhlenbeck process (σ 15 c, τ 1 s);
   - onset scoops of −50 to −150 c decaying with τ = 60 ms;
   - +30 c overshoot;
   - **original vibrato preserved**.
3. Run the plugin and compare the output with the **in-tune original**, which is a valid full reference.
4. Add an *oracle* condition: an ideal offline inverse shift through the same shifter, as an upper bound.
5. Reference F0 for the output: an offline ensemble (pYIN + CREPE-full or RMVPE, median), excluding frames where the members disagree by more than 50 c.

### 10.3 Metrics and proposed targets

The targets are design targets, not literature values. Perceptual mistuning thresholds in prior work range from about 10 to 40 c (Larrouy-Maestri et al. 2019, whose own test norms are 43-65 c), so tuning targets sit well inside 10 c.

| Category | Metric | Proposed target |
|---|---|---|
| Tuning (voiced stable frames, ±30 ms around boundaries excluded; vibrato notes measured on the centre) | \|cents error\| median / p95; % within 5/10/25 c | Live median < 5 c, p95 < 15 c; Studio median < 3 c |
| Settling | Time until the output stays within 10 c for 20 ms after an onset | Live < 30 ms at RS = 0; about RS for RS > 0 |
| Vibrato | Rate error; extent ratio vs the Depth setting | < 0.2 Hz; ±10% |
| Added jitter | RMS of high-passed (> 12 Hz) cents(output) − cents(target) | < 3 c |
| Voicing | V/UV F1 against the reference | ≥ 0.97 |
| Note decision | Flip-flop rate; onset-note accuracy; scoop-quantisation rate | Rules vs HMM comparison (no fixed target yet) |
| Detector | RPA/RCA/OA at 50 and 25 c; fine accuracy within 10/25 c; octave-up/down rates; voicing P/R/F1; **step-response lag** at 110/220/440 Hz; delay-tolerant RPA (RT-SWIPE; τ = 0/10/23 ms) as a diagnostic | Beat the YIN baseline at equal lag; compare against causal SHS and NCCF |
| Timbre / quality | ViSQOL v3 audio mode (aggregate only), LSD, MCD/MFCC (40 coefficients), F1-F3 deviation (LPC tracks), click/buzz detector (energy above 8 kHz at grain rate), FAD with a fixed embedding and sample-size extrapolation (Gui et al. 2024), non-intrusive Audiobox Aesthetics and a SingMOS-trained predictor (SingMOS-Pro is a dataset, so train the predictor ourselves) | Relative ranking between engines and tiers |
| Latency | Reported vs loopback-measured (cross-correlation on unvoiced noise bursts and pulse trains); content-wander distribution | Measured = reported; wander ≤ T |
| CPU | p50 / p99 / p99.9 / max block time | §10.7 |

Caveats:

- ViSQOL correlates well with MUSHRA on codecs but showed low sensitivity to hum and glitches in a 2025 music-production study.
- FAD is biased by sample size and embedding.
- Never trust a single score.

### 10.4 Datasets (roles and licences)

| Dataset | Content / ground truth | License | Role |
|---|---|---|---|
| MDB-stem-synth ([Zenodo](https://zenodo.org/records/1481172)) | 230 resynthesised stems, perfect F0 at a 128/44100 s hop | CC BY-NC 4.0 | Exact-F0 detector evaluation (pick the vocal stems via MedleyDB metadata) |
| PTDB-TUG ([TU Graz](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html)) | 20 speakers, 4720 recordings, laryngograph F0, 48 kHz | ODbL | Detector ground truth (speech) |
| Dagstuhl ChoirSet ([AudioLabs](https://audiolabs-erlangen.de/resources/MIR/2020-DagstuhlChoirSet)) | Larynx-mic channels with F0 | check | Singing with near-exact F0 |
| MIR-1K | 1000 clips, 19 amateurs; semitone labels at 10 ms | research terms | Amateur intonation |
| vocadito ([arXiv](https://arxiv.org/abs/2110.05580)) | 40 excerpts, 7 languages; F0 + 2 note annotations (low agreement) | CC BY 4.0 | Singing F0 and notes |
| VocalSet ([Zenodo](https://zenodo.org/records/1442513)) | 20 pro singers, 10.1 h; techniques (vibrato, straight, breathy, fry, trill); no F0 labels | CC BY 4.0 | Robustness, vibrato; detune round trip |
| M4Singer | 29.77 h, 20 singers, 700 songs, scores | CC BY-NC-SA 4.0 | Score-aligned note targets |
| GTSinger ([arXiv](https://arxiv.org/abs/2409.13832)) | 80.59 h, 20 singers, 9 languages, 6 techniques; F0 from RMVPE | check on HF (reported CC BY-NC-SA 4.0) | Techniques, scores |
| NUS-48E; Intonation (Smule/DAMP, on request) | Phoneme labels / 4702 mostly in-tune amateur performances | various | Note logic, real intonation |
| Own recordings | Breath, sibilants, fry, belting, whistle, melisma, bleed, reverb, crowd noise; close-mic and IEM | own | The decisive set |

- Use NC sets **for internal evaluation only, pending legal review**.
- Use only CC BY or own data for anything trained and shipped.
- Resample every set to one time grid: MIR-1K labels are 10 ms and MDB-stem-synth is about 2.9 ms.

### 10.5 Synthetic and calibration signals

- **LF-glottal-pulse vowels** through formant filters, with vibrato (5.5-7 Hz, ±60 c) and jitter. These give exact F0 and formant ground truth.
- **Step tests.** ±40 c off-grid steps every 500 ms for correction lag. A 50 c step to fit `A(1 − e^(−t/τ))` for Retune-Speed calibration against Auto-Tune.
- **Sweeps.** An A2-A3-A2 sweep to calibrate the Flex-Tune radius. A frame-alignment chirp probe in the style of pitch-benchmark.
- **Latency signals.** Noise bursts and pulse trains for reported-vs-measured latency.
- **Edge cases.** Silence, denormal tails, DC, and a sample-rate sweep (44.1-192 kHz).

### 10.6 Latency measurement

- **Loopback through the DAW** (Live and Bitwig) at 32/64/128-sample buffers, for every latency mode and voice range. Record the reported D, the measured input-to-output delay, and the content-wander distribution.
- **Measure the references the same way:** AutoTune 2026 Classic and Modern, AutoTune Advanced Live (trial, within the EULA), Ableton Auto Shift (normal and Live Mode), and Waves Tune Real-Time if available. Most published competitor figures are host-reported or vendor numbers.

### 10.7 CPU benchmarks

- **Kernels.** criterion 0.8.2 per kernel, reporting p99.9 and max, not only means.
- **Whole plugin.** Per-block timing in the clack-host harness at buffers of 32/64/128/256/512/1024 and 44.1/48/96 kHz, on a **base M1** (the budget machine), the M4 Max, and one x86 machine.
- **Budgets (proposed):**
  - Live Eco/HQ ≤ 10% of a base-M1 core at p99, and Max ≤ 25%.
  - The worst-case callback at a 32-sample buffer must stay well below its 0.67 ms period.
  - Neural arbiter ≤ 0.25 ms per 5 ms hop.
- **Reference points measured on the M4 Max:**
  - NSDF frame: 16.4 µs (48 kHz) / 3.5 µs (12-16 kHz)
  - E/H over all lags: about 2% at 48 kHz, 0.034% decimated
  - FFT round trip: 3.06 µs (2048 points)
  - hand-written GRU step: 10.8 µs
  - tract CNN (404k params): 98 µs

### 10.8 Plugin validation

| Tool | Use |
|---|---|
| clap-validator 0.4.1 | `validate` on every CLAP build (the research skeleton passed: 44 tests, 34 passed, 0 failed, 10 skipped); `fuzz -j4 -d10m` nightly |
| Steinberg VST3 SDK validator (3.8.1, MIT) | Every VST3 build: conformance, resize, latency restart |
| pluginval | Strictness 10, `--skip-gui-tests`; `--rtcheck enabled` requires a develop build (the latest tag, v1.0.4, lacks it) |
| rtsan-standalone 0.3.0 | Process entry in the clack-host harness, run with `RTSAN_ENABLE=1` |
| clack-host 0.2.0 harness | Block sizes {1, 7, 32, 64, 128, 480, 4096, random} give identical output within tolerance; measured delay = reported latency; mode-switch restart |
| Manual matrix | Live 12.4.2 (VST3): latency after project reload and device off/on, MIDI track → VST3 audio effect routing, editor open/close × 50, automation of non-latency params. Bitwig 6.1.3 (CLAP + VST3): latency-mode switch (re-test #71 on macOS), each sandbox mode, CLAP note port |

CI: GitHub Actions on macos-15 (arm64), ubuntu-22.04 and windows-latest. Steps: fmt, clippy `-D warnings`, test, xtask bundle, validators. Upload unsigned artifacts.

### 10.9 Listening tests

- **MUSHRA** (ITU-R BS.1534-3):
  - hidden reference = the in-tune original;
  - anchors = a 3.5 kHz low-pass and a naive phase vocoder without formant preservation;
  - at least 12-15 trained listeners.
- **Conditions.** Shifts of ±0.5, ±2, ±5 and ±12 st, in single-shift and round-trip variants. Material: male, female, breathy, belting, fry, bass.
- **Systems.** L0, L1 (symmetric and asymmetric), L2, S1, S2, plus the references (Signalsmith, Rubber Band offline, captured AutoTune output, Auto Shift).
- **Singer-in-the-loop test.** On in-ear monitors at 32/64/128-sample buffers, to set the Live budget empirically.
- **A/B against Auto-Tune behaviour** for Retune, Humanize, Flex and Natural Vibrato after calibration.

### 10.10 Gates

- **Neural gate** (detector or arbiter). Admit a model only if, on the causal singing benchmark at the same reported latency, it:
  - improves the octave-error rate *and* voicing F1;
  - does **not** worsen cents-error p95 or step-response lag;
  - fits ≤ 0.25 ms per 5 ms hop on a base-M1 P-core, runs allocation-free, and passes parity CI;
  - was trained on commercially usable data.
- **Neural post-filter gate** (research track). A significant MUSHRA gain over the DSP-only output, with ≤ 3-5 ms of added lookahead (Live) and ≤ 25-30% of an M1 core. Offline-render variants are exempt from the latency limit.
- **Engine gates.**
  - L1 asymmetric must beat L0 and come close to S1 on low-male MUSHRA before losing its "experimental" label.
  - S2 must beat S1 for its voice range before it ships.
  - A note HMM replaces the rules in HQ only if it lowers the flip-flop rate and the onset-note error.
- **Release gate.** All validators pass, measured latency equals reported, the tuning and settling targets are met, and there are zero RT violations (assert, rtsan, rtcheck).

## 11. Risks, open questions, and where the sources disagreed

### 11.1 Top risks

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | **Causal Live PSOLA is our own design and has not been tested on singing.** Every PSOLA win in the literature used offline PSOLA with full-file marks. The asymmetric short-rise grain variant has no published listening test | Live may sound rough, especially on low voices | Ship L0 Classic first as the reference. Use symmetric grains whenever the latency budget allows (Live+). Gate the asymmetric variant with MUSHRA. Spend Live HQ CPU on marks and interpolation |
| 2 | **PSOLA is weak on high and female voices** (Röbel Test 2; Drugman & Dutoit; "quality degrades with pitch"). That is the core pop and Auto-Tune use case | Studio quality lags commercial HQ modes for those voices | Let the voice range select the S2 spectral engine in Studio. L2 options (sinc, multi-candidate grains). MUSHRA on alto/soprano material |
| 3 | **Detection lag mis-corrects vibrato and glides.** The error is slope × lag, about 26-52 c at a 13.75 ms lag for ±50/±100 c vibrato | Audible warble at fast retune speeds and with vibrato preservation | Pitch-synchronous E/H as the default path. Instantaneous period from mark spacing. Lag compensation (Max). Centred analysis in Studio |
| 4 | **Octave and voicing errors** (F1 ≈ 2·F0, fry, breath, sibilants, bleed, reverb; YIN-family trackers degrade most under reverb) | Glitches, ±12 st jumps | Octave checks, HMM, a 20 ms octave guard, a ±600 c auto-correction clamp, voicing hysteresis and hold, the gated neural arbiter. Recommend close-miked, dry input |
| 5 | **The first vibrato cycle is flattened** in Live at retune 0, because a causal tracker needs about 170-250 ms to engage | Slightly unnatural vibrato onsets | Accept and document it. In Studio, an RTS smoother engages earlier. Measure it in the harness |
| 6 | **Framework immaturity.** nice-plug is young and only lightly tested on macOS. A breaking move to baseview 0.4 is coming. The #71 fix is unconfirmed on macOS. PR #96 (VST3 resize) is still open | Host-specific bugs, forced upgrades | Pin exact versions and commit the lockfile. Put the editor behind a trait. Keep the clack + clap-wrapper-rs plan B. Run the manual host matrix every release |
| 7 | **Hidden RT-unsafe paths in the framework.** `execute_background`/`execute_gui` take a mutex and unpark a thread, wrapped in `permit_alloc`. macOS adds `CFRunLoopWakeUp` | Priority inversion, dropouts | Never call them in steady state. Run rtsan-standalone and pluginval `--rtcheck` |
| 8 | **CPU tail latency at small buffers on base M1 or x86** (every measurement so far is on an M4 Max) | Dropouts at 32-sample buffers | Spread work across callbacks. Benchmark p99.9 and max. A governor that can only demote. A per-machine tier recommendation |
| 9 | **Host latency-change behaviour.** Ableton delivers the component handler late, and changing latency forces a restart | Wrong PDC or dropouts on mode switch | Latency mode is non-automatable. Set latency in `activate()`. Manual tests after project reload |
| 10 | **IP.** Antares DVA is patent-pending with unknown claims. US9123353 (adaptive boundaries and delay) is active to about 2033-09-30. Melodyne US8022286 runs to June 2029 | Legal exposure for the vibrato engine | Freedom-to-operate review before release. Avoid adaptive boundaries and adaptive delay. Keep our vibrato method traceable to published, pre-existing techniques |
| 11 | **Licensing.** GPL references (fat1, Autotalent, pYIN Vamp, Rubber Band). NC datasets. Public checkpoints (PESTO code under LGPL, SwiftF0 data under CC BY-NC) | Contamination | Clean-room reimplementation from papers. Legal review of NC datasets used for evaluation. Train on own or CC BY data only |
| 12 | **Evaluation validity.** Public benchmarks are mostly speech, offline and sometimes label-biased. ViSQOL is insensitive to hum and glitches. FAD is biased | Wrong engine decisions | Own causal singing harness, several metrics at once, MUSHRA at milestones, a singer-in-the-loop test |
| 13 | **MIDI target routing in Ableton** to a VST3 audio effect is untested | The MIDI Target feature may not work in Live | Early acceptance test. Document the routing |
| 14 | **Toolchain.** The machine's default Rust is 1.89, which cannot build the stack | Build failures | Pin `rust-toolchain.toml` to 1.99.0 |

### 11.2 Where research and verification disagreed

| # | Topic | Original research claim | Verifier's finding | Resolution in this design |
|---|---|---|---|---|
| 1 | SWIPE at short windows | "46 ms windows drop SWIPE's MIR-1K RPA from 96% to 85%" | **Refuted.** Those numbers are for SWIPE-tiny, a neural model with SWIPE kernels. The paper gives no short-window figure for classical SWIPE | Reject SWIPE for Live on physical grounds (8-period kernels; RT-SWIPE delays of 80/40/20/10 ms). Keep it as a Studio low-note second opinion |
| 2 | YIN delay measurement | Derrien measured about 30 ms | **Refuted.** von dem Knesebeck & Zölzer (DAFx-10) measured 27.4 ms. Derrien only gives the 12.5 ms minimum | Use the DAFx-10 latency methodology as a template |
| 3 | Long analysis windows | "Cost only onset responsiveness" | **Partially wrong.** A stale f0 mis-corrects moving pitch by slope × lag | The pitch-synchronous path is the default, not a premium tier |
| 4 | CMNDF in the hybrid core | Type-II difference with the pYIN Beta prior | **Miscalibrated.** On noise the long-lag minimum CMNDF had a median of 0.61 (type-II) vs 0.89 (type-I) | Type-I difference via a W + τ_max buffer; recalibrate the prior |
| 5 | Windowing before the NSDF | "Do not window" | McLeod's WSNAC is more consistent on frequency ramps | Plain MPM stays unwindowed. WSNAC is used for HQ's short resolution |
| 6 | Christensen Kalman as a vibrato split | Adopt for vibrato display and control | **Not a vibrato model.** Its AR(1) state has a 2.3 ms time constant, and it was tested only on two guitar recordings | Kalman for the mean only. Vibrato from the extrema tracker plus an AR(2) resonator; RTS smoother in Studio |
| 7 | Online pYIN evidence | Vamp fixed-lag mode, Essentia streaming | Vamp's fixed lag is hard-wired to 100 frames, and Essentia's version is offline | Fixed-lag benefit at 20-50 ms is **unproven**; the harness decides |
| 8 | Bayesian harmonic tracker gains | 15-20% fewer errors on Keele | Measured at 0 dB *white* noise. At 10 dB babble it ties fast NLS | Max tier only. Prefer fastF0ArMl. Benchmark first |
| 9 | "FFT path mandatory" | Direct YIN at 48 kHz takes 272 µs | Overstated: direct YIN takes 11 µs at 12 kHz | FFT stays the default; this is a minor point |
| 10 | PESTO compute and lag | A few MMAC, 0.05-0.3 ms per frame; "under 10 ms" latency | **Refuted.** g7 needs about 28.5 MMAC per frame (FCPE's paper gives 2.82 G/s independently). The report mixed up the checkpoints. Fundamental-bin kernels are 64-110 ms, so the low latency is reporting lag only | PESTO is demoted to a challenger. PitchDNN-style is the primary candidate |
| 11 | Neural fine precision | "Often worse than DSP" | **Partially wrong.** RMVPE, SwiftF0, FCPE and CREPE beat every DSP method within 10 c; only PESTO, TorchCREPE and PENN are worse | DSP still sets the period, because of latency and label limits, not because neural is imprecise |
| 12 | Neural vs DSP gap under noise | 0.754 vs 0.557 | The best DSP method under noise is SHS at 0.606 | Compare against the *best* causal DSP (YIN/NSDF, SHS, NCCF) |
| 13 | Scaling neural cost with CPU | Shrink the hop to 2.5 ms | Frames already overlap by more than 90% | Spend CPU on model width, rank or ensemble size; keep a 5-10 ms neural hop |
| 14 | Antares latency claims | 2.5 ms core latency | The page says "2.5 ms (175 samples)", but 175 samples is 3.97 ms. The SOS figures are what Cubase reported | Set targets from our own loopback measurements |
| 15 | Liu & Akama 2026 | PSOLA is the best classical method on singing | It also has the **worst F0 error** (41.78 c, offline maxrmorrison/psola), and a neural restorer beats it on FAD, MFCC and F0 | PSOLA must pass our cents-error gates. A neural restoration post-filter becomes a research track |
| 16 | Röbel DAFx10 | PSOLA beats SHIP on downward shifts (Test 1) | In Test 2, the improved SHIP beat PSOLA on average and clearly on female voices | S2 engine is chosen by voice range in Studio |
| 17 | ESOLA as support for epoch PSOLA | "Epoch-aligned OLA beats PSOLA" | ESOLA is time-scaling plus resampling. In its own test TD-PSOLA rated lowest of the OLA methods | Cited only for implementation sensitivity. WSOLA and ESOLA still shift formants |
| 18 | LP-PSOLA | "No listening test found"; the main Live quality lever | **Refuted.** The ESOLA paper rates LP-PSOLA 1.45-1.85 MOS, below TD-PSOLA | Demoted to an experiment |
| 19 | Neural latency | "All miss < 10 ms" | **Refuted.** BRAVE reaches about 10 ms at 44.1 kHz on an M1 Pro | Reject neural in the shifter on *quality and transparency*. Keep a gated post-filter track |
| 20 | iZotope phase vocoder vs Lent | PV beats Lent by about 0.5 MOS | That result comes from voice conversion with 11 in-house raters and cross-singer shifts | Weak evidence either way; the A/B test decides S2 |
| 21 | Retune Speed units | τ = RS as the 63% constant (correction report) **vs** τ = RS/3, RS = time to 95% (state-of-the-art report) | Antares documents neither | τ = RS (63%). Fit an Antares-compatible mapping from a 50 c step measured through Auto-Tune |
| 22 | Waves Tolerance | AND of cents and time | The manual says "once either threshold is crossed", i.e. OR | Use OR semantics |
| 23 | Harman/Cor-Tek US9123353 | "Check status", 0.1 st hysteresis | Active to 2033-09-30. It also uses accompaniment histograms. The 0.1 st figure is an error-difference rule (about 5 c past the midpoint) | Avoid. Use fixed hysteresis and a constant D |
| 24 | Natural Vibrato units | ±12 mapped to dB (×10^(NV/20)) | Unverifiable; Antares never defines the units | Ship Depth % now. Add an Antares-style ±12 only after measuring |
| 25 | Flex-Tune edge shape | Raised-cosine taper | Antares: notes are "instantly tuned as the sweep enters", which implies a hard edge | Taper fraction k_t is a design choice; calibrate it downward |
| 26 | Value of CPU in the control layer | "More CPU does not improve this layer" | **Too strong.** An HMM note decoder scales with CPU and removes heuristics | MonoNote HMM in HQ, Max and Studio; rules stay in Eco |
| 27 | Framework (state-of-the-art report) | nih-plug is suitable | **Refuted.** Its VST3 export is GPLv3 and the project is in maintenance mode | Use nice-plug with the MIT `vst3` bindings |
| 28 | nice-plug #71 | "Fixed" | Closed without confirmation from the reporter, and only on Linux | Re-test in Bitwig on macOS |
| 29 | Ableton CLAP | Citation to a "12.4 review" | It was actually a 12.0 beta review. The official release notes through 12.4.2 still support the conclusion | No CLAP in Live |
| 30 | Crate availability | clap-wrapper 0.16, baseview 0.4.1 | 0.16 is git-only (crates.io 0.3.1 ships 0.14). baseview 0.4.1 exists only as a GitHub release | Pin to crates.io versions |
| 31 | SwiftF0 description | 95,842 params, 42x faster than CREPE (v1 paper) | Current v0.3 has 14,386 params and needs 176 ms of lookahead to stream. Spacings are 2/3/5/7 | Use offline only, or as a teacher |
| 32 | Live latency target | 5 ms default (state-of-the-art report) | Verifier: 2-3 ms for parity with Antares' low-latency mode. User: 5-10 ms | Live reports about 3-5 ms depending on range, giving 5-9 ms mean total; Live-Min about 1 ms; Live+ up to 10 ms |
| 33 | Studio reported latency vs mean content delay | Report D | With causal grains the mean delay is D + T/2, so PDC lands T/2 late | In Studio choose the nearest mark, so wander is ±T/2 around D |
| 34 | Babacan singing ranking | RAPT best GPE | Its ground truth was RAPT run on the EGG signal, so it is biased | Not used to rank methods |

### 11.3 Still unresolved

1. **Studio-HQ engine.** PSOLA-HQ vs PGHI phase vocoder vs SHIP, especially for alto and soprano voices. Only our own MUSHRA can settle it.
2. **Does any tiny causal neural tracker beat the best causal DSP** (NSDF/E-H, SHS, NCCF) on singing at the same latency? Real step-response lag of VQT models at 80-150 Hz is unknown.
3. **Is short-rise asymmetric PSOLA acceptable** on low male voices compared with L0 and S1?
4. **Retune Speed, Flex radius, Humanize timing and Natural Vibrato units:** none are documented by Antares. They need step and sweep measurements.
5. **Rules vs HMM note decoding:** which gives fewer flip-flops and better onset notes?
6. **How to treat vocal fry and subharmonic phonation:** track f or f/2, and whether to add an explicit fry state.
7. **Whistle register (above 1.4-2 kHz):** whether to support it, and the minimum analysis rate it would need.
8. **How a single Tracking knob maps** onto eps, the clarity threshold, the pYIN prior and the HMM switch probability.
9. **Aperiodic handling** for large shifts: band split, comb harmonic/noise split, or nothing.
10. **Whether lag compensation reduces vibrato warble** without overshoot.
11. **MIDI Target with no held note:** pass through (Antares) or latch while voiced? Needs feedback from live users.
12. **Offline render at Max tier** (quality) vs bounce = monitor (consistency). Default Off for now.
13. **Ableton and Bitwig behaviour on a latency-mode switch** mid-session: dropout length, PDC update, and whether to require transport stop.
14. **The DVA patent claims**, once they are published.
15. **Patent landscape beyond the expired US5973252** (Celemony, Waves, Antares' newer patents) has not been researched.
16. **Whether a streamable neural restoration post-filter** can fit within 3-5 ms and 25-30% of a core in the near future.
17. **Coverage gaps from the research sessions.** Their web-search budgets ran out partway through. 2025-2026 work on causal F0, vibrato detection and note tracking may be missing, as may vendor details (Little AlterBoy, MAutoPitch, Inner Pitch latency).

---

## 12. IP and licensing summary

| Item | Status | Our use |
|---|---|---|
| US 5,973,252 (Hildebrand / Antares) | Expired (lifetime), anticipated 2018-10-14 | Free to implement: E/H detector, Classic engine, Decay smoothing |
| US 5,567,901 (IVL, grain resampling for formants) | Expired (2015) | Formant knob |
| US 6,549,884 (Laroche-Dolson PV pitch shift) | Expired (2019) | S2 option |
| US 9,123,353 / US 9,747,918 (Harman → Cor-Tek: adaptive boundaries, dynamic delay) | Active; anticipated expiry 2033-09-30 | **Avoid** |
| Antares DVA | Patent pending, claims unpublished | Freedom-to-operate review before release |
| US 8,022,286 (Celemony polyphonic DNA) | Active to June 2029 | Do not copy note-object resynthesis |
| US 8,494,842 (SoundHound vibrato FFT) | Expired (fee related) | GUI vibrato statistics allowed |
| VST 3 SDK | MIT since 3.8 | Via `vst3` 0.3.0 (MIT/Apache) |
| nice-plug, nice-plug-egui, nice-plug-xtask | ISC | Framework |
| triple_buffer | MPL-2.0 | Use unmodified |
| GPL/AGPL code: upstream nih-plug VST3 bindings (`vst3-sys`), zita-at1/fat1, Autotalent, TalentedHack, QPitch, pYIN Vamp, Praat, Rubber Band, tdpsola, aubio, irapt, LTFAT, PC-NSF-HiFiGAN weights | Copyleft | **Study only; never link or copy** |
| fastF0Nls / fastF0ArMl | BSD-3, notify on commercial use | Max-tier port |
| libopus PitchDNN | BSD-3 | Architecture reference and port; retrain on singing |
| PESTO | Code LGPL-3.0; weights license unstated | Method only; retrain |
| SwiftF0 | MIT code; weights partly trained on CC BY-NC data | Offline teacher only |
| Datasets | Several NC (MDB-stem-synth, M4Singer, GTSinger) | Internal evaluation pending legal review; CC BY or own data for training |

---

## Appendix A: key sources by area

**Pitch detection (DSP)**

- [YIN (JASA 2002)](https://cagnazzo.wp.imt.fr/files/2013/05/2002_JASA_YIN.pdf)
- [MPM (McLeod 2005)](https://www.cs.otago.ac.nz/graphics/Geoff/tartini/papers/A_Smarter_Way_to_Find_Pitch.pdf)
- [McLeod PhD (WSNAC, vibrato)](https://www.cs.otago.ac.nz/graphics/Geoff/tartini/papers/Philip_McLeod_PhD.pdf)
- [librosa pyin](https://raw.githubusercontent.com/librosa/librosa/main/librosa/core/pitch.py)
- [pYIN Vamp](https://github.com/c4dm/pyin/blob/master/PYinVamp.cpp)
- [US5973252](https://patents.google.com/patent/US5973252A/en)
- [Praat filtered AC](https://www.fon.hum.uva.nl/praat/manual/pitch_analysis_by_filtered_autocorrelation.html)
- [RT-SWIPE (CMMR 2025)](https://audiolabs-erlangen.com/content/05_fau/assistant/00_balke/01_publications/2025_MeierSSMB_RealTimeSWIPE_CMMR.pdf)
- [SWIPE re-evaluation (ISMIR 2025)](https://arxiv.org/html/2507.11233v1)
- [dYIN/dSWIPE](https://www.audiolabs-erlangen.com/resources/MIR/2025-TASLPRO-dYIN-dSWIPE/)
- [Shi et al. Bayesian tracker](https://arxiv.org/abs/1905.08557)
- [fastF0Nls](https://github.com/jkjaer/fastF0Nls) and [fastF0ArMl](https://github.com/jkjaer/fastF0ArMl)
- [Christensen 2012](https://vbn.aau.dk/ws/files/71778668/paper.pdf)
- [Harmonic chirp ML](https://doi.org/10.1109/TSP.2017.2723342)
- [Babacan singing study](https://arxiv.org/abs/1912.12609)
- [von dem Knesebeck & Zölzer DAFx-10](https://www.dafx.de/paper-archive/2010/DAFx10/VonDemKnesebeckZoelzer_DAFx10_P102.pdf)
- [Cycfi Q pitch](https://cycfi.github.io/q/q/v1.5/reference/pitch.html)

**Pitch detection (neural)**

- [PESTO TISMIR 2025](https://arxiv.org/abs/2508.01488) and [repo](https://github.com/SonyCSLParis/pesto)
- [PitchDNN paper](https://arxiv.org/html/2309.14507v2) and [libopus code](https://github.com/xiph/opus/blob/v1.5/dnn/pitchdnn.c)
- [FrCN low-rank](https://arxiv.org/html/2609.23340)
- [Strahl & Müller ICASSP 2026](https://audiolabs-erlangen.de/resources/MIR/2026-ICASSP-MidLevelFusionF0)
- [SwiftF0](https://github.com/lars76/swift-f0)
- [pitch-benchmark v2.1](https://github.com/lars76/pitch-benchmark/blob/main/BENCHMARK.md)
- [RMVPE](https://arxiv.org/html/2306.15412v2)
- [FCPE](https://arxiv.org/html/2509.15140v1)
- [CREPE](https://arxiv.org/abs/1802.06182)
- [PENN](https://arxiv.org/html/2301.12258v3)
- [LACOPE](https://www.isca-archive.org/interspeech_2021/schroter21_interspeech.html)
- [Voting ensemble](https://arxiv.org/abs/2602.01727)
- [StreamVC](https://arxiv.org/html/2401.03078)
- [ANIRA](https://arxiv.org/abs/2506.12665)

**Pitch shifting and formants**

- [Liu & Akama 2026](https://arxiv.org/html/2601.10345)
- [Promonet](https://arxiv.org/html/2407.05471v1)
- [Röbel SHIP DAFx10](https://dafx.de/paper-archive/2010/DAFx10/Roebel_DAFx10_P84.pdf)
- [iZotope RT-SVC DAFx23](https://www.dafx.de/paper-archive/2023/DAFx23_paper_21.pdf)
- [ESOLA](https://arxiv.org/pdf/1801.06492)
- [IVL US5567901](https://patents.google.com/patent/US5567901A/en)
- [GCI review](https://arxiv.org/pdf/2001.00473)
- [True envelope DAFx05](https://dafx.de/paper-archive/2005/P_030.pdf)
- [CheapTrick](https://github.com/mmorise/World/blob/master/src/cheaptrick.cpp)
- [PVDR / PGHI](https://arxiv.org/pdf/2202.07382)
- [Laroche-Dolson patent](https://patents.google.com/patent/US6549884B1/en)
- [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch)
- [Rubber Band LiveShifter](https://breakfastquay.com/rubberband/code-doc/classRubberBand_1_1RubberBandLiveShifter.html)
- [Bungee](https://github.com/bungee-audio-stretch/bungee)
- [CLPCNet](https://interactiveaudiolab.github.io/assets/papers/morrison2022neural.pdf)
- [BRAVE](https://arxiv.org/html/2503.11562)
- [FARGAN](https://arxiv.org/abs/2405.21069)
- [Drugman & Dutoit DSM](https://arxiv.org/abs/2001.00579)
- [zita/fat1 retuner](https://github.com/x42/fat1.lv2/blob/master/src/retuner.cc)

**Correction behaviour**

- [Auto-Tune Artist manual](https://www.barryrudolph.com/recall/manuals/auto-tune_artist.pdf)
- [Pro X guide](https://antares-web-frontend.sfo3.cdn.digitaloceanspaces.com/documentation/pdfs/Auto-Tune_Pro_X_User_Guide_10.0.pdf)
- [UA Realtime Advanced manual](https://media.uaudio.com/support/manuals/dd/Auto-Tune%20Realtime%20Advanced%20Manual.pdf)
- [Waves Tune RT guide](https://www.fullcompass.com/common/files/28004-WavesTuneRealTimeUserGuide.pdf)
- [fat1 retuner.h](https://github.com/x42/fat1.lv2/blob/master/src/retuner.h)
- [Autotalent](https://github.com/SizzlingCalamari/autotalent/blob/master/autotalent.c)
- [Graillon 3 guide](https://www.auburnsounds.com/downloads/Graillon%203%20User's%20Guide.pdf)
- [GSnap manual](https://sites.units.it/ramponi/teaching/DSP/materials/S03.4a/GSnapManual.pdf)
- [Melodyne drift/modulation](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_ToolModulationDrift_2)
- [US9123353](https://patents.google.com/patent/US9123353B2/en)
- [pYIN MonoNote](https://github.com/c4dm/pyin)
- [Saitou WASPAA 2007](https://staff.aist.go.jp/m.goto/PAPER/WASPAA2007saitou.pdf)
- [Ventura et al. vibrato](https://fe.up.pt/voicestudies/artts/doc/publications/086%20-%20ACCURATE%20ANALYSIS%20AND%20VISUAL%20FEEDBACK%20OF%20VIBRATO%20IN%20SINGING.pdf)
- [BERT-APC](https://arxiv.org/abs/2511.20006)
- [MIDIBack](https://arxiv.org/abs/2609.28008)
- [Deep Autotuner](https://arxiv.org/abs/2002.05511)

**Products**

- [SOS AutoTune 2026 review](https://www.soundonsound.com/reviews/antares-autotune-2026)
- [AutoTune Advanced (MBW)](https://www.musicbusinessworldwide.com/antares-launches-autotune-advanced-a-pitch-correction-plugin-built-to-leave-a-singers-vibrato-alone/)
- [Production Expert review](https://www.production-expert.com/production-expert-1/antares-autotune-advanced-review-what-makes-it-different)
- [Antares low-latency blog](https://www.antarestech.com/blog/real-time-pitch-correction-for-live-performance-with-autotune-2026)
- [UA latency article](https://help.uaudio.com/hc/en-us/articles/43460658366612)
- [Waves latency table](https://waves.com/support/tech-specs/plugin-latency)
- [Eventide latency (JSLHR 2019)](https://sites.bu.edu/stepplab/files/2019/07/HellerMurrayetal.pdf)
- [MetaTune SOS](https://www.soundonsound.com/reviews/slate-digital-metatune)
- [Vocoflex Bonedo](https://www.bonedo.de/?p=2114800)
- [zplane technology](https://licensing.zplane.de/technology)

**Rust stack**

- [nice-plug](https://codeberg.org/RustAudio/nice-plug) and its [Cargo.toml](https://codeberg.org/RustAudio/nice-plug/src/branch/main/crates/nice-plug/Cargo.toml), [PR #29](https://codeberg.org/RustAudio/nice-plug/pulls/29), [issue #71](https://codeberg.org/RustAudio/nice-plug/issues/71)
- [nih-plug maintenance note](https://github.com/robbert-vdh/nih-plug/commit/60e3f5274127)
- [vst3-rs](https://github.com/coupler-rs/vst3-rs)
- [VST3 MIT licensing](https://steinbergmedia.github.io/vst3_dev_portal/pages/VST+3+Licensing/Index.html)
- [CLAP latency.h](https://github.com/free-audio/clap/blob/main/include/clap/ext/latency.h)
- [iPlug2 #772](https://github.com/iPlug2/iPlug2/issues/772)
- [clack 0.2](https://github.com/prokopyl/clack/releases/tag/v0.2)
- [clap-wrapper-rs](https://github.com/blepfx/clap-wrapper-rs)
- [baseview 0.4.0](https://github.com/RustAudio/baseview/releases/tag/v0.4.0)
- [egui](https://crates.io/crates/egui)
- [rtrb](https://github.com/mgeier/rtrb)
- [basedrop](https://github.com/micahrj/basedrop)
- [rubato docs](https://docs.rs/rubato/5.0.1/rubato/)
- [tract](https://github.com/sonos/tract)
- [rten](https://docs.rs/rten/latest/rten/)
- [nam-rs](https://github.com/OpenSauce/nam-rs)
- [RTNeural](https://github.com/jatinchowdhury18/RTNeural)
- [rtsan-standalone](https://github.com/realtime-sanitizer/rtsan-standalone-rs)
- [clap-validator](https://github.com/free-audio/clap-validator)
- [pluginval options](https://github.com/Tracktion/pluginval/blob/develop/docs/Command%20line%20options.md)
- [Rust stable manifest](https://static.rust-lang.org/dist/channel-rust-stable.toml)
- [Apple workgroups](https://developer.apple.com/documentation/audiotoolbox/workgroup-management)
- [Bitwig plug-in hosting](https://www.bitwig.com/userguide/latest/vst_plug-in_handling_and_options)

**Evaluation**

- [ViSQOL v3](https://arxiv.org/abs/2004.09584)
- [VERSA](https://arxiv.org/abs/2412.17667)
- [SingMOS-Pro](https://arxiv.org/abs/2510.01812)
- [Audiobox Aesthetics](https://arxiv.org/abs/2502.05139)
- [FAD best practice](https://arxiv.org/abs/2311.01616)
- [vocadito](https://arxiv.org/abs/2110.05580)
- [MDB-stem-synth](https://zenodo.org/records/1481172)
- [PTDB-TUG](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html)
- [VocalSet](https://zenodo.org/records/1442513)
- [GTSinger](https://arxiv.org/abs/2409.13832)
- [M4Singer license](https://github.com/M4Singer/M4Singer/blob/master/dataset_license.md)
- [Dagstuhl ChoirSet](https://audiolabs-erlangen.de/resources/MIR/2020-DagstuhlChoirSet)
- [Mistuning perception](https://pmc.ncbi.nlm.nih.gov/articles/PMC6478636/)
