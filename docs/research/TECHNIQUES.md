# Technique registry

Every technique the research has evaluated, with where it came from, how strong the evidence is, its status, and **every** issue it could help. Check this before designing a feature; add to it after every research task (see [README.md](README.md)).

**Status:** **Adopted** (in the plugin) · **Candidate** (promising, not built) · **Prototyped** (built in a scratch copy and measured) · **Rejected** (with the reason) · **Unknown** (research pending).

Sources: `R§` = [`../research.md`](../research.md), `P§` = [`../rust-performance.md`](../rust-performance.md), `PERF§` = [`../PERFORMANCE.md`](../PERFORMANCE.md), `KEY` = [`host-key-scale.md`](host-key-scale.md), `GRAPH` = [`graph-mode.md`](graph-mode.md), `NI` = [`note-intent.md`](note-intent.md) (audited).

## Pitch detection

| Technique | Source | Evidence | Status | Helps |
|---|---|---|---|---|
| Pitch-synchronous E/H candidates (Hildebrand recursion; info lag 1-2 periods of the sung note) | R§8.3, US5973252 (expired) | 0.03-0.1 % of a core decimated; basis of Auto-Tune's tracker | Adopted | #7 (onset speed), #8, #17 |
| FFT NSDF + type-I CMNDF verifier with pYIN-style probabilities | R§8.4 | vocadito RPA 94.7 %, equal to the offline reference tracker | Adopted | #14, #27 |
| Prime-harmonic octave score, rough-phonation veto, exact-lag re-measure | R§8.5, STATUS | Live HQ octave errors 0.31 % vs 1.30 % for the offline reference | Adopted (HQ/Max) | #14, #16 |
| Cheap time-domain octave check in Eco | STATUS (v1.1) | Track 34 Eco octave errors 23.9 % → 5.9 % | Adopted | #14 |
| Causal pitch HMM with hop-rescaled transitions; fixed-lag smoothing in Studio | R§8.6 | — | Adopted | #7, #10 |
| Voicing FSM with onset acceleration for low voices | R§8.8, STATUS | low_male onsets 33-35 ms → 25-27 ms (with the PSOLA fix) | Adopted | #16, #17 |
| PSOLA mark spacing as the instantaneous pitch source | R§7.6.2 | Noisier than the detector on real voices (±10 c wobble) | Rejected for the control law (v1.1) | #8, #13 |
| fastF0ArMl harmonic-model re-scoring of top candidates | R§8.9 | Unproven at close-mic SNR | Candidate | #13, #14 |
| Neural trackers SwiftF0 / RMVPE / FCPE / CREPE / PENN / streaming PESTO | R§4.3 | More accurate offline, but 32-176 ms of lookahead (PESTO's VQT still integrates 64-110 ms) | Rejected for live use today | #27, #14, #10 (offline analysis) |
| PitchDNN-style causal network as an octave/voicing arbiter | R§4.3, R§8.9 | 69k params, 10 ms causal delay; trained on speech | Candidate | #14, #27, #28 |
| Label alignment to the newest sample (LACOPE) | R§4.3 | On par with pYIN/CREPE on speech at 0 ms alignment | Candidate | #27 |
| Hand-written allocation-free inference (RTNeural-style), low-rank FrCN kernels | R§8.22, R§4.3 | GRU(128) step 10.8 µs on M4 Max; tract allocates per call | Candidate | #14, #27, #28 |

## Pitch shifting and formants

| Technique | Source | Evidence | Status | Helps |
|---|---|---|---|---|
| Causal TD-PSOLA: NCC mark chaining, LP-residual anchor, asymmetric short-rise grains | R§8.13-8.15 | Formant LSD ≤ 2.9 dB up to ±7 st; transparency 99 dB | Adopted (Live) | #8, #9, #15 |
| Early onset on the newest past mark, lazy ratio commit | STATUS (v1.1) | Shifter ratio lag 8 ms → ~2 ms; low-male settling < 30 ms | Adopted | #8, #17 |
| Lookahead PSOLA with nearest-mark grains | R§8.14 | Studio heard delay = D ± ~0.1 T | Adopted (Studio) | #10 |
| Resample + whole-cycle splice (Auto-Tune patent), crossfaded | R§8.17 | ~1 ms; formants move with pitch | Adopted (Live-Min) | #8, #17 |
| Formant shift by grain resampling (IVL patent, expired) | R§7.5 | Clean within about ±3 st | Adopted | #9, #15 |
| Per-grain spectral-envelope re-colouring (true envelope / CheapTrick, low-band no-warp) | R§7.5, R§8.20 | Needed beyond ±3 st | Candidate | #9, #15, #8 |
| Classic splice + low-latency envelope correction ("formant-corrected Classic") | latency deep-dive plan | Could give formant preservation at ~1-1.5 ms | Candidate | #8, #9 |
| Predictive pitch marks (place grains on extrapolated marks) | latency deep-dive plan | Removes most of the mark-detection delay if prediction holds | Candidate | #8, #7 |
| Envelope-preserving phase vocoder (PGHI) / shape-invariant (SHIP) for high voices | R§5 | Beat PSOLA on female voices in Röbel's Test 2 | Candidate (Studio only) | #15 |
| Neural vocoders / DDSP / voice conversion in the shifter | R§5.1 | ~10 ms achievable (BRAVE) but not transparent | Rejected | #15 |
| Phase-interpolated polyphase sinc tables | P§3.4.3 | Read error −106 to −121 dB at no measurable cost | Adopted (core) | #8 |

## Correction behaviour and note decisions

| Technique | Source | Evidence | Status | Helps |
|---|---|---|---|---|
| One-pole on the correction (Retune = 63 % time constant), Humanize, Flex-Tune, Snap | R§7.6 | Residual-vibrato table matches within 0.03 | Adopted | #7 |
| Centre/vibrato split (extrema tracker + AR(2) Kalman), vibrato depth | R§8.11 | Vibrato extent error ≤ 5 % (synthetic) | Adopted | #7, #10 (drift/modulation editing) |
| MonoNote-style note HMM (HQ/Max) | R§8.12 | — | Adopted | #7, #10 |
| Phrase restart on a > 300 c jump near a phrase start | STATUS (v1.1) | Onset glitches 13.9 % → 6.3 % (with the other v1.1 fixes) | Adopted | #7, #16 |
| Studio decisions over the lookahead, pitch advance | STATUS (v1.1) | Studio settle 35/83 ms → 17/41 ms (median/p90) | Adopted | #7, #10 |
| Adaptive singer tuning reference | NI proto A | Hurts real singers as a default; on a detuned-singer benchmark it recovers most of the intent loss | Prototyped; recommended only as an opt-in "Follow singer's tuning" | #7, #6 |
| Musical note model: key/interval/duration priors, glide state | NI proto B | No frontier gain (audit); its b_adapt preset trades note accuracy for warble | Prototyped, not adopted | #7, #6 |
| Release freeze: no downward switch while the level falls > 4 dB in 40 ms | NI proto C, lead | Removes about a third more warble on top of the veto | Prototyped; recommended | #7, #16 |
| Glide-endpoint extrapolation (Saitou-style) | NI proto C | Future pitch change uncorrelated with velocity (R² ≈ 0.05); worse in closed loop | Rejected | #7, #8 |
| Onset cues that aren't pitch (energy dip + rise, voicing gaps) set the note-change hazard | NI literature | Moved the frontier in an offline test (AUC 0.72); level dip/rise features feed the learned veto | Partly adopted via the veto | #7, #16, #10, #17 |
| Learned switch veto ("rule proposes, model vetoes"): gradient-boosted trees, 300 trees, 82 causal features relative to the decided pitch, can only block a proposed switch | NI proto D, lead | Live HQ chromatic TEST: warble 50 → 24 /min, stable p95 25.6 → 20.1 c, frame accuracy +3.3 pts, but per-note accuracy 79.7 → 76.5 %; 10.6 µs per hop, no allocation; audited sound with caveats (Live HQ chromatic only) | Prototyped; recommended for Live HQ pending confirmation on new singers | #7, #14, #27, #16 |
| Studio onset note read from the newest third of the lookahead | NI protos B, C | DEV onset glitches 1.1 → 0.5 %; not confirmed on TEST | Prototyped | #7, #10 |
| Frontier yardstick: compare a change against the Pareto envelope of a Tolerance sweep | NI lead | Separates "better" from "a different setting of the same trade-off" | Prototyped (eval tool) | #26, #12, #7, #8 |
| Uncertainty-scaled transition time | NI literature | Shrinks A→B→A swings without delaying confident switches | Candidate | #7 |
| Key detection: profile matching on a stable-note histogram; optional sidechain chroma | R§6, KEY §5.3 | Vocal-only needs ~20-30 s of singing | Candidate | #6, #5, #7 |

## Host integration

| Technique | Source | Evidence | Status | Helps |
|---|---|---|---|---|
| VST3 `kScaleEvent` / `ProcessContext.chord` via a nice-plug patch | KEY | No host documented as sending scale events | Candidate (cheap) | #5 |
| Latched MIDI-scale key source; Bitwig Key Filter+ → MIDI input | KEY §5 | Works in any DAW that routes MIDI to an effect (Bitwig route untested) | Candidate | #5, #6 |
| Display timeline that survives CLAP sleep/wake resets | commit 704d95d | Graph no longer wipes on pauses | Adopted | #19 |
| In-place voice-range switch when latency is unchanged | STATUS (v1.1) | No host restart needed | Adopted | #19 |
| Capture-based graph mode (Auto-Tune "Track Pitch" model) vs ARA 2 | GRAPH | Bitwig and Live have no ARA | Candidate | #10, #21 |

## Performance and platform

| Technique | Source | Evidence | Status | Helps |
|---|---|---|---|---|
| FFT-based difference function, exact FFT sizes (3·2^k) | PERF§3.2, §3.4 | 3.7-11× vs direct SIMD; exact sizes 25-27 % faster than the next power of two | Adopted | #28, #29 |
| Apple vDSP FFT | PERF§3.3, §12 | Faster in isolation, 4-13 % slower end to end (small FFTs) | Rejected for the detector | #28 |
| Hand-written NEON kernels | PERF§3.1, §12 | 1.05-1.8× per kernel, 2-5 % end to end | Adopted | #28, #29 |
| Fast exp2/log2/tanh approximations | PERF§3.6 | 3.3-4.3×, error ≤ 0.001 cent | Adopted | #14, #27 |
| `target-cpu=native`, PGO, `panic=abort`, algebraic float ops | PERF§7 | ≈ 0 % or worse; PGO not re-measured on LLVM 23 | Rejected (PGO to re-test) | #29 |
| Studio analysis split across callbacks | STATUS (v1.1) | 192 kHz/32 worst block 30 % → 22 % of deadline | Adopted | #29 |
| Real-time worker pool with macOS work intervals, one-hop pipelining | P§4.5 | 0 missed deadlines vs 2-3 % for normal-priority threads under load | Candidate | #13, #14, #27 |
| Neural Engine via Core ML / BNNS Graph | — | Research pending | Unknown | #28, #27, #14 |
| SME2 / AMX through Accelerate | — | Research pending | Unknown | #28, #14 |
| Universal macOS bundle + runtime feature dispatch | — | Standard practice | Candidate | #29, #31 |

## Evaluation

| Technique | Source | Evidence | Status | Helps |
|---|---|---|---|---|
| Synthetic singing with exact ground truth, held-out scenarios | STATUS | Basis of the quality gates | Adopted | #26 |
| vocadito real-singing evaluation | STATUS | 40 takes, two annotators | Adopted | #26, #7 |
| Note-intent replay harness (cached detector, fixed DEV/TEST split, leave-one-singer-out) | NI | Reproduces the in-engine targets; lives in a scratch copy for now | Prototyped (move into `skato_eval`) | #7, #26, #12 |
| Detuned-singer benchmark (real contours, controlled offsets) | NI proto A | Shows benefits invisible on vocadito's labels | Prototyped | #26, #7 |
| Analysis-resynthesis to exact pitch curves; perceptual metrics; listening-test calibration | #26 plan | — | Candidate | #26, #12 |

## Cross-pollination notes

- **Onset cues** (#7) could also fix slow breathy/voiced-consonant onsets (#16), Live-Min low-voice settling (#17), and note segmentation for graph mode (#10).
- **Envelope correction** from the latency work (#8) is the core of a throat model (#9) and of formant quality beyond ±3 st (#15).
- **The worker pool** (#13) is the prerequisite for a neural arbiter (#14) and live neural tracking (#27); the Apple research (#28) decides where that inference runs.
- **Key detection** (#6) feeds the musical note model's key prior (#7) and the DAW-key fallback (#5).
- **The note-intent harness, the detuned-singer benchmark and the frontier yardstick** should become part of the perceptual test suite (#26) and the Auto-Tune A/B (#12).
- **"Rule proposes, model vetoes"** kept the synthetic gates passing where models that decided notes directly failed; the same pattern fits a neural octave/voicing arbiter (#14, #27) and other hysteresis decisions (voicing release, octave guard).
- **Onset glitches are mostly shifter/output-path events** (note-decision proxy 1.1 % of the published 6.3 %), so they belong with the shifter and latency work (#8, #16).
