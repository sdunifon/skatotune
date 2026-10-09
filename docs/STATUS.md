# Skatotune: v1 status

State of the tree on 2026-10-09, after the hardening pass (onsets, sustained notes, CPU tail, plugin protocol) and its regression and integration check. Rust 1.99.0, nice-plug 0.4.2, Apple M4 Max. The binding contract is `docs/DESIGN.md`; the algorithms and their evidence are in `docs/research.md` (R§) and `docs/rust-performance.md` (P§); the performance evidence is in `docs/PERFORMANCE.md`.

**In one paragraph.** v1 builds, bundles and passes all three plugin validators. 552 tests pass and clippy is clean. Every DESIGN §4 gate now passes on the synthetic suite, including Live settling: all 18 onsets settle in under 30 ms (worst 27.0 ms), where three missed at v1 close. Loading in Ableton and Bitwig, part of the plugin gate, was not re-checked in this pass. The two held-out scenarios still miss their tuning target (p95 34 c) and 3 of 11 onsets miss settling. On real singing (vocadito, 13.6 min), hard tune in Live HQ now settles a median 27 ms after a note starts (39 ms before this pass; Studio HQ 17 ms, was 35 ms), and onset glitches fell from 13.9 % to 6.3 % of note starts. The detector matches the offline reference tracker's RPA (94.7 %) with a quarter of its octave errors. Averaged over the vocadito singers, Live mode reports 4.0 ms of latency, and the voice you hear while correcting arrives 5.8 ms behind the input.

## 1. What v1 does

- **Plugin:** VST3 (Ableton Live 12) and CLAP (Bitwig), plus a standalone app with a device picker (`cargo standalone`). Stereo, mono and mono→stereo layouts; MIDI input.
- **Latency modes** report a constant latency per mode and voice range:
  - Live-Min: 1 ms, Classic engine.
  - Live: 3.1-5.9 ms, causal PSOLA.
  - Live+: 5-10 ms, user-set.
  - Studio: 20-50 ms, lookahead PSOLA.

  Eco, HQ and Max quality tiers run inside every mode and never change the latency.
- **Detector** (`skato_detect`):
  - Candidates: decimated pitch-synchronous E/H candidates; an FFT NSDF with a type-I CMNDF verifier and pYIN-style probabilities.
  - Octave checks: prime-harmonic score, rough-phonation veto, third-harmonic check and exact-lag re-measurement; in Eco (no harmonic score) a time-domain rough-voice rule.
  - Tracking: causal pitch HMM (fixed-lag smoothing in Studio), full-rate refinement, voicing/sibilance FSM and a mean-pitch Kalman filter.
  - Voicing at note starts: onset hops must agree within 50 c, onsets after silence start ≤ 20 c above the range and (above 150 Hz) take 3 hops, segments younger than 12 ms do not hold, release tails more than 18 dB under the segment's decaying peak are unvoiced, and steady-level segments hold through low-clarity legato leaps.
  - The per-hop analysis can run in four stages over later callbacks (`begin_analysis` / `step_analysis`), bit-identical to one `analyze()`.
- **Shifter** (`skato_shift`):
  - Classic splicer (crossfaded in every tier).
  - Causal NCC mark tracker. HQ resolves competing peaks by period continuity.
  - PSOLA, Live and Studio: in HQ, grains are chosen by overlap correlation; Studio's nearest-mark choice is centred on D. A voiced onset may start on the newest known mark (no wait for one ahead), period decisions use the newest commanded ratio, and a late mark is replaced by the predicted one instead of falling a period behind.
  - Formant shift by grain resampling.
- **Control law** (`skato_control`), all of R§7.6:
  - Note decision: rules in Eco; MonoNote-style HMM in HQ/Max, gated by Waves' OR tolerance rule. A phrase restart (a > 300 c jump early in a phrase or after a gap) is decided as a fresh note.
  - The correction follows the detector pitch (the PSOLA mark pitch is no longer mixed in).
  - Retune one-pole on the correction, transition τ, Humanize, Flex-Tune, Snap Min/Max.
  - Centre/vibrato split with Vibrato Depth, synthetic vibrato, Transpose/Detune.
  - MIDI Target, Learn and Auto, plus pitch bend.
  - Studio decides with up to 40 ms of lookahead, takes the pitch that sets the shift from a slightly newer frame (matching the shifter's ratio timing), and ignores note crossings that return within the lookahead.
- **Engine** (`skato_engine`):
  - Processing: block-size-invariant, allocation-free, NaN-safe.
  - Studio pipeline: each hop's analysis begins at the boundary and runs in stages at fixed points of the next hop, so host blocks smaller than a hop share the work out.
  - Voice-range changes that keep the latency are applied at runtime (`Engine::set_voice_range`).
  - Telemetry for the GUI.
- **GUI** (egui): scrolling piano-roll pitch graph, cents meter, scale keyboard (with MIDI Learn, CLEAR and KEEP), knobs, mode/tier selectors, CPU readout and a latency readout ("X ms reported" plus "heard avg").
- **Evaluation** (`skato_eval`):
  - Synthetic singing with exact ground truth, two held-out scenarios, and real singing (vocadito).
  - Gates as tests (`tests/quality_gates.rs`).
  - CLI: `suite`, `vocadito`, `latency`, `rt-bench`, `render`, `synth`, `track`.
- **Distribution:** ISC `LICENSE` at the repo root; every bundle carries `THIRD-PARTY-NOTICES.txt` and `LICENSE.txt` (`cargo xtask notices`).

## 2. Latency: what we average

There are three different "latencies":

1. **Reported plugin latency D.** The DAW compensates for it, and it is fixed per mode and voice range.
2. **The delay of the corrected voice itself.** The PSOLA grains trail D by up to a period.
3. **How long the correction takes to act.** This is the time from the start of a note until the output is on pitch.

CPU time per block is a separate quantity again.

| Mode | D at 48 kHz (Low Male / Alto-Tenor / Soprano) | D, vocadito average | Heard on voiced audio while correcting (vocadito, measured) |
|---|---|---|---|
| Live-Min | 1.00 ms | 1.02 ms | ≈ D (Classic engine; not covered by the PSOLA measurement) |
| **Live (default)** | 5.85 / **4.44** / 3.10 ms | **3.96 ms** | **5.80 ms** (Low Male 8.73, Alto/Tenor 6.59, Soprano 4.55) |
| Live+ (10 ms) | 10.0 ms | 10.0 ms | 10.25 ms (was 10.44) |
| Studio (40 ms) | 40.0 ms | 40.0 ms | 40.02 ms |

- **Measured equals reported.** All 48 mode × range × rate rows of `skato_eval latency` match, and so do the 12 suite runs. On vocadito the amount-0 output delay equals D on 31/40 files in Live HQ and on all 40 in Studio HQ (38/40 before; the worst Studio file was 56 samples off, now 1). The Live misses are off by ≤ 10 samples, where the cross-correlation locks onto a neighbouring pitch cycle.
- **How the heard column was measured.** The Natural engine ran with an ideal hard-chromatic correction computed from the vocadito F0 annotation (every voiced hop shifted to the nearest semitone, per-file voice range, 44.1 kHz). The column is the shifter's mean content delay over all PSOLA grains (`ShifterDiagnostics::mean_content_delay`), weighted by voiced time.
- **Why Live is heard later than D.** Live grains can only use pitch marks already detected, so below about 250 Hz the voice trails D by about 0.45 periods (R§2.1, DSP-9, kept as a trade for formant quality). Studio is centred on D. The plugin header's "heard avg" readout estimates the same quantity live from the singer's pitch.
- **Time until the correction acts** (vocadito, Live HQ; Studio HQ in parentheses):
  - Pitch is detected a median **12.9 ms** after the annotated note start, p90 19.4 ms (9.3 / 15.1 ms). That is 1.7 ms later than at v1 close: onsets after silence now need consistent hops, and above 150 Hz one more hop, which removed half of the bad first estimates.
  - Hard tune is settled (within 10 c for 20 ms) a median **27.1 ms** after the start, p90 65.5 ms (17.1 / 41.4 ms), against 39.1 / 87.1 (35.1 / 83.1) at v1 close. About 13 ms of the median is the singer's own scoop into the note: an ideal per-frame snap of the sung pitch settles at 14 ms on the same onsets (hardening diagnosis).
- **Interface buffers come on top** when you monitor through a DAW: about 2 × the buffer plus the converters, so 128 samples at 48 kHz add about 5.3 ms. The standalone launcher's defaults give roughly 25 ms in total in Live (15 ms input buffer + 2 periods + D).
- **CPU:** Live HQ costs 9.2 µs per 32-sample block, 1.4 % of real time (PERFORMANCE §11.1).

## 3. Gates (DESIGN §4)

"v1 close" is the state before this pass (`docs/STATUS.md` of the v1 integration). "Now" is this pass: `skato_eval suite`, `skato_eval rt-bench --seconds 5` (median of 3 runs), `cargo test -p skato_eval --release --test quality_gates`.

| Gate | Target | v1 close | Now | Result |
|---|---|---|---|---|
| Tuning, Live HQ (retune 0-20 ms) | median < 5 c, p95 < 15 c | worst median 1.77 c; worst p95 10.78 c (melody_male rt0) | worst median 1.58 c (melody_male rt20); worst p95 4.94 c (low_male rt20) | PASS |
| Tuning, Studio HQ | median < 3 c | 1.86 c | 1.50 c | PASS |
| Settling, Live HQ, retune 0 | every onset < 30 ms (clock from the note-decision point) | **3 of 18 ≥ 30 ms** (low_male #2 35.0, #0 33.0, melody_male #0 31.6), listed as known misses | **0 of 18**; worst 27.0 ms (low_male #2: detected 14.9 ms, convergence 12.1 ms). `KNOWN_SLOW_ONSETS` is empty | **PASS** (was FAIL) |
| Vibrato preservation (τ = 0) | extent ±15 %, centre median < 8 c | vibrato 6.3 % / 2.36 c; from the onset 12.7 % / 1.69 c | vibrato 5.0 % / 2.25 c; from the onset 12.9 % / 1.41 c | PASS |
| Vibrato engagement | ≥ 80 % extent ≤ 2.5 periods; early ratio ≤ 1.15 | slowest 1.83 periods; early ratio 0.37-0.55 | slowest 1.83 periods; early ratio 0.37-0.53 | PASS |
| Added jitter | < 3 c RMS (> 12 Hz) | 0.19 c | 0.00 c | PASS |
| Voicing F1 | ≥ 0.97 | ≥ 0.999 | 1.000 | PASS |
| Latency measured == reported | ±1 sample | 0 of 12 off; 48/48 | 0 of 12 off; 48/48 | PASS |
| Transparency (amount 0) | Classic exact; PSOLA seg. SNR ≥ 25 dB | Classic 0, PSOLA 99.3 dB | Classic 0, PSOLA 99.0 dB | PASS |
| Content delay | \|wander\| ≤ one period (melody_male) | Live max 0.06 periods | Live max 0.07 periods, Live+ / Studio 0.00 | PASS |
| RT safety | 0 allocations; NaN/extreme input safe; block-size invariant | pass | pass: counting-allocator tests in every crate, the plugin (now also the in-place voice-range switch) and the standalone swap; block invariance max diff 0; the Studio split analysis is bit-identical to a one-shot analysis | PASS |
| CPU, Live HQ p99.9 | < 5 % @128, < 10 % @32 | 2.55 % / 5.24 % | **2.38 % / 5.16 %** | PASS |
| CPU, Max tiers p99.9 | < 25 % | 8.84 % (Studio Max @32) | **7.33 %** (Studio Max @32) | PASS |
| Perf proofs | criterion speedups recorded in PERFORMANCE.md | §3, §10 | §3, §10, §11 | PASS (NEON `axpy`/`mul_acc` are the documented exception) |
| Plugin | bundles; clap-validator; loads in Bitwig and Ableton | 37/0/7; VST3 47/47; pluginval SUCCESS | clap-validator 37 pass / 0 fail / 7 skip; VST3 SDK validator 47/47; pluginval strictness 10 SUCCESS; installed in `~/Library/Audio/Plug-Ins` | PASS for bundles and validators; **DAW loading not re-verified** |
| Held-out tuning | same as Tuning, Live HQ | worst p95 39.2 c (heldout_room rt0) | worst p95 34.0 c (heldout_room rt0) | **FAIL** (guard tightened 45 → 40 c) |
| Held-out settling | same as Settling | 5 of 11 ≥ 30 ms, worst 85.0 ms | 3 of 11 ≥ 30 ms (heldout_pentatonic #4 36 ms, heldout_room #1 32 ms, #2 85 ms) | **FAIL** (guard tightened ≤ 6 → ≤ 5 slow, < 120 ms) |
| Held-out voicing F1 | ≥ 0.97 | 0.999 | 0.999 | PASS |
| Real singing, vocadito | detector vs offline reference; hard-tune regression guards | RPA50 94.5 % (ref 94.7), octave 0.32 % (ref 1.30), F1 0.940 (ref 0.942); guards pass | RPA50 94.7 %, octave 0.31 %, other gross 2.09 % (ref 3.22), F1 0.955; guards pass after re-baselining them tighter (§5) | PASS (4/4) |

**Where the former slow onsets now stand** (decision point → detected → target switch → settled, Live HQ, 128-sample blocks):

- **low_male #0 and #2** (E2, 82 Hz, decidable at the onset): 26.0 and 27.0 ms (were 33.0 and 35.0). Detection is unchanged (15.0 / 14.9 ms); convergence after it fell from 18-20 ms to 11-12 ms, because a Live PSOLA onset now starts on the newest known mark instead of waiting up to a period (skato_shift ON-4), period decisions use the newest commanded ratio (ON-3), and a hard-tune shift is applied at once at a PSOLA onset (skato_control ON-9).
- **melody_male #0** (C3 after a slow −75 c scoop): 15.6 ms (was 31.6). The 25 ms near-band persistence is dropped once a phrase's first note has scooped 20 c toward the next note (skato_control ON-5).

The margin is 3-4 ms on the two low-male onsets; Live-Min (Classic engine, not part of this gate) still takes 33-35 ms on them.

## 4. Real singing (vocadito: 40 tracks, 13.6 min, 29 singers, 44.1 kHz)

`cargo run --release -p skato_eval -- vocadito` (≈ 190 s; the full report is `out/vocadito/report.md`, listening WAVs for tracks 1, 2, 9, 21, 26 and 34 in `out/vocadito/listen/`). "v1 close" is the run of 2026-10-09 00:47.

| Measure | v1 close | Now |
|---|---|---|
| Detector RPA50, Live HQ / Studio HQ (offline reference 94.7 %) | 94.5 % / 95.5 % | 94.7 % / 95.1 % |
| Octave errors, Live HQ / Studio HQ (reference 1.30 %) | 0.32 % / 0.33 % | 0.31 % / 0.29 % |
| Octave errors, Live Eco / Studio Eco | 1.01 % / 1.06 % | **0.35 % / 0.29 %** |
| Track 34 (rough voice), Live Eco: RPA50 / octave errors | 63.3 % / 23.9 % | **82.5 % / 5.9 %** (HQ 84.9 % / 5.0 %) |
| Other gross errors, Live HQ (reference 3.22 %) | 3.00 % | 2.09 % |
| Voicing precision / recall / F1, Live HQ | 0.904 / 0.978 / 0.940 | **0.940** / 0.971 / **0.955** |
| Bad estimate at a note start (> 500 c), Live HQ / Studio HQ | 12.6 % / 9.4 % | **6.0 % / 5.6 %** |
| Pitch detected after note start, Live HQ median / p90 | 11.2 / 17.8 ms | 12.9 / 19.4 ms |
| Hard chromatic, Live HQ: output median / p95 | 2.0 / 29.2 c | **0.8 / 26.9 c** |
| Hard chromatic, Live HQ: p95 by range (Low Male / Alto-Tenor / Soprano) | 33.1 / 31.2 / 25.9 c | 30.2 / 29.8 / 23.4 c |
| Hard chromatic, Live HQ: settle median / p90 | 39.1 / 87.1 ms | **27.1 / 65.5 ms** |
| Hard chromatic, Live HQ: onset glitches / output dropout | 13.9 % / 0.31 % | **6.3 % / 0.25 %** |
| Hard chromatic, Live HQ: warble episodes | 909 | **743 (−18 %)** |
| Hard chromatic, Live HQ: notes on the annotated note (raw take 89.6 %) | 87.4 % | 87.1 % |
| Hard in key, Live HQ: p95 / glitches / warbles | 21.1 c / 18.0 % / 638 | **14.7 c / 11.4 % / 525** |
| Natural in key, Live HQ: warbles / glitches | 384 / 1.8 % | 284 / 1.6 % |
| Hard chromatic, Studio HQ: output median / p95 | 2.1 / 22.0 c | **0.7 / 18.9 c** |
| Hard chromatic, Studio HQ: settle median / p90 | 35.1 / 83.1 ms | **17.1 / 41.4 ms** |
| Hard chromatic, Studio HQ: glitches / warbles | 8.2 % / 578 | 5.1 % / 341 |
| Hard in key, Studio HQ: output p95 | 14.1 c | **8.3 c** |
| Live-Min Classic, hard chromatic: settle / glitches / clicks per minute | 41.2 / 95.7 ms, 15.5 %, 0.15 | 33.0 / 74.3 ms, 5.7 %, 0.07 |
| Envelope LSD, hard chromatic, Live HQ / Studio HQ | 0.76 / 0.75 dB | 0.76 / 0.64 dB |
| Amount 0: max PSOLA wander, Live HQ / Studio HQ (track 34) | 2.29 / 3.00 periods | 2.08 / 2.06 periods |
| Pass-through control, Live HQ: jumps / glitches (measurement floor) | 31 / 1.2 % | 24 / 0.8 % |
| NaN / blow-ups (1,400 renders) | 0 | 0 |

**What moved it:** the corrector's phrase-restart rule and the detector's release-tail and young-segment rules (stale notes carried into a new note were the largest glitch cause); the corrector following the detector instead of the noisier mark pitch; the shifter's onset and ratio-timing fixes; and in Studio the matching pitch-frame advance in the control law.

**Not improved, or slightly worse:**

- Detection after a note start is 1.7 ms later (median), the price of the onset-consistency rules.
- Studio HQ RPA50 fell 0.4 points (95.5 → 95.1 %), still above the reference.
- 0.3 points fewer Live HQ hard-chromatic notes land on the annotated note (87.4 → 87.1 %).
- Live HQ amount-0 delay equals D on 31/40 files (33/40 before); the misses are cross-correlation cycle slips of ≤ 10 samples.

## 5. What changed in this pass

**Owner work** (details in each crate's docs and tests; every defect fix has a regression test that fails on the old code):

- **skato_shift:**
  - ON-4: a Live voiced onset may start on the newest known mark (gated: NCC ≥ 0.9, period within 25 c of the control's hint); low_male onsets 33/28/35 → 25/21/27 ms.
  - ON-3: period decisions use the newest commanded ratio, and long symmetric boundaries are committed lazily; Studio shifter ratio lag 8.1 → 2.1 ms at 82 Hz.
  - SN-9: a late next mark is replaced by the predicted one instead of falling back up to 1.25 periods.
  - PERF-5: the Live mark-tracker update after a voiced control runs at the start of the next `process()`; exact early exit in the seed guard; O(1) sliding energies in the mark search. Worst Studio shifter block at 192 kHz 92.6 → 62.7 µs.
  - PERF-6(1): the overlap-add iterates grains by reference (−7 to −14 %).
- **skato_detect:**
  - ON-2a-e: release-tail cut, no hold for segments younger than 12 ms, onset pitch consistency, no onset above the range, one more confirmation hop above 150 Hz.
  - SN-7: steady-level segments hold through low-clarity legato leaps.
  - Eco rough-voice octave rule (track 34: 23.9 → 5.9 % octave errors).
  - PERF-1-4: exact octave confirms from an interpolated FIR bank, newer-half E/H energies on demand, onset NSDF only when needed, split refinement loop.
  - PERF-7: the split analysis API (`begin_analysis`, `step_analysis`, `finish_analysis`, `ANALYSIS_STEPS`).
- **skato_control:**
  - SN-2: the correction follows the detector pitch only.
  - ON-1: phrase restart as a fresh note.
  - ON-9: hard tune at a PSOLA onset applies the full shift at once.
  - ON-5: attack-phase near-band persistence.
  - SN-3: Studio takes the pitch that sets the shift from the frame 0.5·hop + 0.25·T newer.
  - SN-1/SN-5 (Studio part): a switch needs the crossing to persist through the lookahead.
  - Live Max takes the pitch at the analysed sample (`f0_hz`).
- **skato_engine:**
  - PERF-5 (engine side): the Live shifter update lands before the first sample after the boundary.
  - HI-3: `Engine::set_voice_range` with prebuilt detector/shifter kits for every range that keeps D.
- **skatotune (plugin), xtask:**
  - HI-1: a structural change is built off the audio thread, swapped in behind a 3 ms fade, and only then reported to a VST3 host (retries at 1, 3 and 7 s until it re-activates).
  - HI-4/CL-4: the learned scale applies in every MIDI mode until CLEAR, and is saved with the project.
  - LIC-1: repo `LICENSE`, `cargo xtask notices`, notices in every bundle.
  - RT-1 (plugin side): Snap Min below 0.05 c reaches the engine as off.
  - GUI: "reported" / "heard avg" latency readout, the learned-scale keyboard.

**Integration fixes in this pass:**

1. **Studio split analysis adopted** (skato_engine, PERF-7). The Studio pipeline calls `begin_analysis()` at the boundary that closes hop h, runs the four detector stages before samples hop/4, 3·hop/8, hop/2 and 3·hop/4 of hop h+1 and the control law before 7·hop/8, and pushes every sample to the detector as it arrives (the held-sample buffer is gone). The shifter update now lands before the first sample after the boundary in Studio too.
   - **Bit-identical:** output and telemetry hashes of 336 configs (3 rates × 4 scenarios × 7 mode/tier/engine × 4 block patterns, with engine switches and MIDI) are identical to the pre-integration engine. A new unit test, `studio_split_analysis_matches_a_one_shot_analysis`, compares split and one-shot analyses bit for bit at 44.1-192 kHz over block sizes 1-1000 with tier changes landing mid-analysis; it fails if the analysis begins after newer samples were pushed.
   - **One semantic change:** a tier change that lands while an analysis is in flight now lets that analysis finish with the tier it began with (a tier change re-lays out the pitch HMM).
   - **CPU:** see PERFORMANCE §11. At 192 kHz/32 the Studio Max p99.9 block went from 30.1 % to 22.0 % of the deadline (worst block 52.9 → 47.1 µs, set by the mark re-seed); rt-bench Studio Max @32 (48 kHz) 8.84 → 7.33 %. Where the pipeline points sit was measured: the chosen ones keep the re-seed alone in its callback down to a quarter hop per block.
   - New test `studio_analysis_is_spread_over_the_next_hop` replaces the RT-2/PERF-7 placement tests.
2. **`Engine::set_voice_range` adopted in the plugin** (HI-3). A voice-range change that keeps D (any range in Live-Min, Live+ and Studio; Low Male ↔ Wide in Live) is applied by the running engine at once (no 0.25 s settle, no background build, no swap, no host involvement); notes, MIDI and telemetry carry on. Other structural changes keep the swap (VST3) or restart (CLAP) path. Tests: `same_latency_voice_range_change_is_applied_in_place` (VST3, CLAP, standalone; fails on the previous code) and the no-alloc test `in_place_voice_range_change_does_not_allocate_or_free`.
3. **A fragile plugin test fixed.** `standalone_swaps_in_a_handed_engine` checked that the swapped-in Studio engine passes a 76 Hz tone through as a pure delay; with this pass's detector the tone is voiced inside Alto/Tenor's search margin and corrected, so the check now uses a 38 Hz tone below every range.
4. **Settling gate re-checked honestly.** All three `KNOWN_SLOW_ONSETS` entries meet 30 ms, so the list is empty and the gate is strict again; the suite reports Settling PASS.
5. **Guards tightened** (never loosened): held-out tuning 45 → 40 c, held-out slow onsets ≤ 6 → ≤ 5, and the vocadito hard-tune guards to the new values plus the same margins (about 4 c, 20 ms, 6 points): Low Male 37 / 108 / 20 → 34 / 88 / 17, Alto/Tenor 35 / 123 / 22 → 34 / 86 / 11, Soprano 30 / 96 / 18 → 27 / 74 / 13. The opt-in `real_singing_gates` test passes.
6. **Report text.** The vocadito report no longer says telemetry `detected` is the PSOLA-mark pitch.
7. **Docs:** DESIGN §1.2, §3.2, §3.3, §3.5 and §3.6; PERFORMANCE §11; this file.

## 6. Known limitations

- **Real singing is still far from the synthetic targets.** Live HQ hard tune: stable-frame p95 23-30 c per range (synthetic target 15 c), settle p90 65 ms (target 30 ms on every onset).
  - Most of the p95 is note-decision switches by singers who sit off-grid and wander across semitone midpoints: 92 % of frames above the p95 lie within 10 ms of a target switch, and forcing the annotated note gives 10.8 c (hardening diagnosis, SN-1). Causal Live decisions follow the Tolerance trade-off: every hysteresis variant tried lands on the same p95-versus-note-agreement frontier as raising Tolerance. Studio's lookahead persistence helps there (p95 18.9 c).
  - About 13 ms of the settle median is the singers' scoop, which the measurement counts from the annotated onset (ON-7a; the report does not yet measure from the decision point).
- **Note starts on real singing.** Live HQ hard chromatic glitches on 6.3 % of note starts (11.4 % in key); 6.0 % of note starts have an early estimate more than 500 c off.
- **Warble.** Live HQ hard tune has about 55 target-flip episodes per minute on vocadito (743), mostly at releases and fast pitch movement; Studio HQ has 341.
- **Held-out material.** heldout_room (room reverb, breathy onsets): hard tune p95 34 c. 3 of 11 held-out onsets take 32-85 ms; the two room onsets are measurement-limited (a perfectly in-tune resynthesis in the same room also measures 96 / 78 ms, ON-7c).
- **Live-Min low voices** (Classic engine) settle in 33-35 ms on low_male; the settling gate covers Live HQ.
- **Heard latency in Live** is D + about 0.45·T below about 250 Hz: 8.7 ms on average for low male singers, for a reported 5.9 ms. Shortening the asymmetric rise was tried again (DSP-9) and reverted: −1.4 to −4.4 ms heard, but +0.4-1.1 dB envelope error and a failed synthetic onset.
- **PSOLA seed position** (DSP-3) is biased to the newest edge during an attack. Seeding on the |x| peak improved level and onsets on vocadito but doubled the crate's worst formant error (2.8 → 6.0 dB): a joint decision, not made.
- **Formants.** Studio's lazy ratio commit (ON-3) raised the synthetic Studio HQ envelope LSD by 0.15-0.5 dB (vibrato 1.10 → 1.61 dB) while real-singing LSD fell (0.75 → 0.64 dB). The Max tier behaves as HQ in the shifter. Formant LSD is about 4 dB at ±12 st from 220-440 Hz.
- **PSOLA wander at amount 0** reaches 2.1 periods on track 34 (Live and Studio) and 1.76 periods on synthetic `melody_female` in Studio; most real-singing cases are cross-correlation cycle slips (SN-8). The content-delay gate only covers `melody_male`.
- **CPU.**
  - The worst Studio block at 192 kHz/32 is now the shifter's mark re-seed at a voiced onset (≈ 47 µs, 28 % of the block), not the analysis.
  - With two blocks per hop at 192 kHz (128 samples) the split puts the re-seed together with Frames and Checks: worst block 72.6 µs (10.9 %) against 57.6 µs (8.7 %) with the one-chunk analysis.
  - P§5.5's work-spreading ratio (p99.9 at 32 ≤ 0.75 × at 64) is now met in Studio (Max 0.74, HQ 0.68) but not in Live (0.85), whose analysis stays at the boundary by design.
  - `Engine::new` builds a detector and shifter for every voice range that shares D (HI-3): a Studio engine takes about 3.8 MB and 18 ms to build (0.9 MB / 4.6 ms before).
  - rt-bench is unpaced on a normal-priority thread, so its `max` column measures the OS.
- **Hosts:**
  - nice-plug 0.4.2's CLAP `activate()` reports the latency twice, causing one extra restart on first activation and per latency change (HI-8; needs a patched nice-plug).
  - A CLAP latency change (mode, Live+/Studio ms, or a Live range change that moves D) still needs a host restart; CLAP allows a latency change only in `activate()`.
  - Not verified in a real host this pass: PDC right after a VST3 mode switch and after reloading a Studio project in Live 12, the VST3 latency re-request when a host drops the first `kLatencyChanged`, and the learned scale after a Bitwig sleep/wake.
- **Platform.** The x86_64 AVX2 path has only run under Rosetta 2. The detector uses realfft (vDSP measured no faster in situ).

**Contract deviations from DESIGN §3** recorded during v1 (§3 text now covers those changed in this pass):

- **§3.1:** `Correlator` also plans exact f·2^k sizes (`with_sizing`, `FftSizing`, `fast_len`).
- **§3.3:**
  - With Preserve Vibrato off, the correction reference is p + (g_v − 1)·v (CL-1).
  - HQ/Max switches pass the OR gate, with 25 ms near-band persistence (CL-2), dropped in a phrase's first-note scoop (ON-5).
  - Snap Min below 0.05 c is treated as off (RT-1).
  - `process_hop` ignores `mark_cents` (SN-2).
- **§3.5:**
  - Studio runs the detector at `Centered{D − k·hop}` with k from `skato_engine::studio_split`, and `lookahead_hops = ⌈D/hop⌉ − 1` when pipelined; a pipelined Studio engine emits `hops − 1` telemetry frames.
- **§3.6:**
  - The MIDI buffer merges same-hop bends and reserves slots for note and sustain events instead of dropping (RT-3).
  - Structural changes settle for 0.25 s (HI-2) and are swapped in, not rebuilt in `activate()` (HI-1).
  - Learn mode is owned by the plugin and persisted (HI-4).
  - CLAP returns `Normal` after silence (HI-7).

## 7. Build, test, install

```sh
cd skatotune   # the repository root
cargo build --workspace
cargo test --workspace --release            # 552 passed, 0 failed, 2 ignored (opt-in), ≈ 70 s
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p skatotune --release --features standalone,demo-telemetry   # 106 plugin tests incl. standalone

# Gates and evaluation
cargo test -p skato_eval --release --test quality_gates -- --nocapture       # 11 gate tests, ≈ 7 s; prints per-onset breakdowns
cargo test -p skato_eval --release --test vocadito -- --ignored --nocapture  # real-singing gates, ≈ 25 s
cargo run --release -p skato_eval -- suite --out out                          # synthetic suite, 252 renders, ≈ 20 s → out/report.md
cargo run --release -p skato_eval -- vocadito                                 # real singing, ≈ 3.2 min → out/vocadito/report.md
cargo run --release -p skato_eval -- latency                                  # measured vs reported D + content-delay gauge
cargo run --release -p skato_eval -- rt-bench --seconds 5                     # CPU tail latency (run on a quiet machine)

# Plugin bundles → target/bundled/Skatotune.{vst3,clap} (with LICENSE.txt and THIRD-PARTY-NOTICES.txt)
cargo xtask bundle skatotune --release
cargo xtask notices --check                                                # THIRD-PARTY-NOTICES.txt up to date
.tools/clap-validator/bin/clap-validator validate "target/bundled/Skatotune.clap"
.tools/vst3sdk/build/bin/Release/validator "target/bundled/Skatotune.vst3"
.tools/pluginval/pluginval.app/Contents/MacOS/pluginval --strictness-level 10 --skip-gui-tests --validate-in-process --validate "target/bundled/Skatotune.vst3"

# Standalone (pick input/output devices interactively; use headphones)
cargo standalone
```

**Install:** copy `target/bundled/Skatotune.vst3` to `~/Library/Audio/Plug-Ins/VST3/` and `target/bundled/Skatotune.clap` to `~/Library/Audio/Plug-Ins/CLAP/`, then rescan in the DAW. The copies in those folders on this machine are this pass's bundles (2026-10-09 06:16).

## 8. Next steps

1. **Real-DAW pass** with the installed bundles: Live 12 PDC after a mode switch and after reloading a Studio project, the VST3 latency re-request, a voice-range change mid-playback (in place, no restart), Bitwig CLAP sleep/wake with a learned scale, and the standalone with a mono microphone at 128-sample periods.
2. **skato_control: Live note decisions on real singing** (SN-1). The data say causal decisions are bounded by the Tolerance trade-off; options are a note-level centre that does not slow legato onsets, or using Live+'s extra latency as a short lookahead. A/B with `skato_eval vocadito --modes live --tiers hq --conditions hard_chromatic,hard_key,natural_key`.
3. **skato_eval: measure onsets from the decision point** (ON-7a) next to the annotated onset, report correction accuracy against the decided target (SN-4), and add an off-grid synthetic scenario that crosses semitone midpoints, so the synthetic gates exercise what dominates real singing.
4. **skato_shift: the open trade-offs** DSP-3 (seed position) and DSP-9 (a user-selectable short rise for lower Live heard latency), and the octave re-seed disputes on rough voices (SN-10).
5. **skato_detect: Studio onset backdating** (ON-6): forward-aligned observation frames at onsets, worth about 5 ms in Studio.
6. **nice-plug: the CLAP double restart** (HI-8), as a `[patch.crates-io]` fork.
7. **CPU at 192 kHz:** the Studio re-seed is now the worst block; splitting the seed's search over two callbacks, or a paced RT-priority rt-bench mode (P§5.5), are the next steps.
