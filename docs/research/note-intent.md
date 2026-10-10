# Note intent: deciding which note a singer means

**Date:** 2026-10-09 · **Status:** research done, recommendation audited (**sound, with caveats**); implemented as two Labs options, off by default (see "Implementation" below) · **Issue:** [#7](https://github.com/sdunifon/skatotune/issues/7)

**How this was done.** A replay harness re-runs the control law over cached detector output for vocadito's 40 real singing takes, scored against both annotators' notes, with a fixed DEV/TEST split by singer (20 DEV singers, 9 TEST singers) and leave-one-singer-out selection inside DEV. Four prototypes competed (adaptive tuning reference, musical note model, trajectory cues, learned decider); a lead combined them and an independent audit re-ran everything.

## Audit corrections (read these first)

The audit reproduced every number bit for bit and found no tuning on TEST, but it narrowed the claims:

- **Holds:** in **Live HQ** the recommended system ("DC": learned switch veto + release freeze) improves warble, stable-note p95 and frame accuracy for 9/9 TEST singers, and moves the p95-versus-agreement frontier in **chromatic** mode (+4.4 c pooled TEST; +2.4 c at equal per-note accuracy).
- **Real cost:** per-note accuracy falls for 8-9 of 9 TEST singers (Live HQ chromatic 79.7 → 76.5 %), legato decisions are slower at p90 for 8/9, and notes the target never reaches within 500 ms rise from 1.6 to 3.0 %.
- **Not established:** in **key mode** DC is on par with simply raising Tolerance once per-note accuracy is counted; in **Studio** warble, p95 and frame accuracy improve but the overall objective is better for only 7/9 singers (interval crosses zero).
- **Optimistic numbers:** DEV figures for the learned model are optimistic (its design was chosen on the DEV folds); the DSP-only fallback "BC" and preset b_adapt were designed after a TEST run, so their TEST numbers are not held-out.
- **Synthetic gate is weaker evidence than it looks:** HQ passes, but ground-truth target accuracy falls on 17 of 24 HQ melody rows and some held-out scenarios were used for choices. New held-out scenarios and **new singers** are needed before release.
- **Onset glitches are mostly not note decisions:** the published 6.3 % / 11.4 % come mainly from the shifter and output path (decision-side proxy 1.1 %), so they belong to the shifter work (#8, #16).

The full recommendation follows, unchanged except that the location of scratch files is omitted (the experiment code lives outside the repository; implementing the recommendation moves the harness into `skato_eval`).

## Implementation (Labs experiment)

The user chose a Labs experiment first. Both options are in the LABS menu and off by default; with both off the engine is bit-identical to the code before them (golden hashes of 72 renders, `crates/skato_eval/tests/note_intent_off.rs`).

- **Learned note hold** (`Settings::note_hold`): DC exactly as recommended in §5.1-5.2, in HQ/Max for scale targets (Eco and MIDI targets unchanged). The Live model also serves Live+ and Live-Min, and the HQ models serve the Max tier; none of these was evaluated. Models: `crates/skato_control/models` (card in its README). The veto runs only on proposal hops and never allocates.
- **Follow singer's tuning** (`Settings::reference = FollowSinger`): prototype A's fg45 tracker in follow mode, target offset latched per note (§5.3).
- **Harness:** `skato_eval note-intent` (cache, replay, metrics, frontier) with the frozen split in `crates/skato_eval/data/note_intent_splits.txt`; `note-intent synth` for the synthetic scenarios; `dump`, `dump-synth`, `convert-model` and `tools/note_intent/train_veto.py` retrain the models. Every TEST number of §4.1 (Live HQ and Studio HQ, three conditions, the Tolerance sweep) and the frontier gaps of §4.3 reproduce to four decimals, the synthetic proxy rows of §4.6 exactly, and the training dumps and both models byte for byte.
- **Check:** `crates/skato_eval/tests/note_intent.rs` (needs `data/vocadito`) holds the §5.4 Live HQ chromatic TEST numbers.
- **CPU** (`skato_eval rt-bench --note-hold` on a vocadito take, off and on interleaved twice, M4 Max at load 15-23, so tails are noisy): Live HQ at 128 samples keeps its 38 µs median; p99 rises from about 60 to about 80 µs on blocks where the model judges a switch (98 judged in 5 s); p99.9 stays at 4.3-4.9 % of the buffer, inside the 5 % gate. Studio HQ is within noise of off.

---


2026-10-09. Snapshot 4bfe848. Data: vocadito (40 takes, 29 singers). DEV is 28 files from 20 singers; TEST is 12 files from 9 singers (`harness/splits.txt`, unchanged).
All numbers come from the shared harness replay, which is bit-exact with the engine telemetry. They use the proxy output (input + shift, no shifter) unless marked "published".

## 1. Summary

- **What moves the p95-vs-note-agreement frontier: the learned switch veto (prototype D). Nothing else moves it consistently.** Adding C's release freeze keeps D's frontier gain and removes about another third of the warble. This combination is called **DC** below.
- **DC vs the baseline on TEST**, at the plugin's default Tolerance (8 c), Live HQ hard chromatic:
  - frame accuracy (frames where both annotators agree) 70.5 → 73.8 %
  - warble 50.0 → 24.0 per minute
  - stable-frame p95 25.6 → 20.1 c
  - note agreement (agreeP) 87.1 → 86.0 % (95 % CI of the change crosses 0)
  - per-note accuracy 79.7 → 76.5 % (a real cost)
- **DC against the baseline's own Tolerance frontier:** at equal note agreement, p95 is 4.4 c lower on TEST and 1.8 c lower on DEV.
- **Published warble episodes**, all 40 files (DEV with leave-one-singer-out models), Live HQ hard chromatic at 5 c: 743 → 404.
- **Live (causal) recommendation:** keep the snapshot's HMM and OR gate, and add two gates that can only remove switches:
  - a learned switch veto (gradient-boosted trees over 82 causal features);
  - a release freeze: no downward switch while the level falls more than 4 dB in 40 ms.
- **Studio recommendation:** the same, plus reading the onset note from the newest third of the lookahead (Studio's own model uses the lookahead features).
- **User controls:**
  - Tolerance keeps its meaning, and its default stays 8 c.
  - "Follow singer's tuning" becomes an opt-in option (prototype A's gated tracker). It is never a default.
  - The veto strength can later be offered as an advanced "note hold" control.
- **Cost:**
  - CPU: about 10.6 µs per hop on average (0.8 % of a core) in Live HQ and 11.9 µs in Studio HQ, with no allocation. The tail at 32-sample buffers must still be checked against the CPU gate (§6.5).
  - Synthetic gate (HQ): passes. Live Eco: one held-out row fails (inherited from D), so Eco keeps the snapshot law for now.
- **What did not move the frontier:**
  - adaptive reference (A): harmful on real singers unless opt-in;
  - musical priors and the segment model (B): no gain;
  - B's b_adapt glide hold: trades note accuracy for warble, along the frontier;
  - glide extrapolation and every other trajectory cue (C), except the release freeze.
- **DSP-only fallback:** "BC" (B's b_adapt + C) ties DC on the pre-declared objective J. But it does not move the p95-vs-agreement frontier, keeps fewer notes right, and is worse than DC on TEST in Studio (§4.4).

## 2. Step 1: the four prototypes, re-run

I rebuilt every prototype from its own source and re-ran its chosen presets on DEV and TEST. To combine them, I merged the four corrector copies into one crate, `lead/combo/skato_control_combo`, with git history. Each prototype's preset run through that merged crate must still reproduce the prototype's own tables, and so must the all-off setting reproduce the baseline (`lead/tools/validate_combo.sh abcd`).

| prototype / preset | own binary vs own report, DEV + TEST | through the merged crate |
|---|---|---|
| A adaptive reference, fg45 | identical | identical |
| B musical HMM: b, b_aggr, b_adapt | identical | identical |
| C trajectory: A, B, C | identical | identical |
| D learned veto (cgbt_syn, preset 1) | TEST T1 identical; DEV fold models identical | identical (TEST, DEV LOSO) |
| everything off | = baseline | = baseline (DEV + TEST) |

**Re-scoring on one yardstick** (`lead/tools/frontier.py`). Each system's distance to the baseline's Tolerance frontier is measured against the frontier's Pareto envelope over 0-45 c. The prototypes used four different interpolations.
- A's, B's and C's verdicts all hold.
- D's TEST claim for Live HQ chromatic shrinks from +3.6 c to +2.9 c of p95 at equal note agreement. D interpolated the raw sweep; the envelope also credits the baseline with its best Tolerance.
- Studio chromatic (+4.2 c) and Live key (+1.5 c) are unchanged.

## 3. How the combination was built and chosen

The protocol was written down before any combination run (`lead/PROTOCOL.md`, `lead/PREREGISTERED_TEST.md`).

**Objective (D's J, fixed in advance).**
- Hard tune: J = agreeP − 0.5·p95 − 0.1·warble/min − 0.5·glitchT.
- Natural: the same without the p95 term.
- Per configuration, J is the mean over hard chromatic, hard key and Natural key, each at its best Tolerance (hard 5/8/12/16 c, Natural 15 c).

**Honest DEV numbers.**
- Every learned system on DEV uses fold models: each singer is replayed with the model trained without them.
- Knob and Tolerance choices are scored with nested leave-one-singer-out (`exp grid`).
- Differences come with 95 % singer-bootstrap intervals, recomputed from pooled metrics (`exp pair`).

**Stage 1, D plus one component at a time** (DEV, nested-LOSO J, Live HQ):

| added to D | nested J | read-out |
|---|---|---|
| D alone | 80.13 | baseline 78.72 |
| + C release freeze, 4 dB | **80.67** | +0.54 [+0.32, +0.76]; warble −9.1/min, 19/19 singers |
| + C release freeze, 2 dB | 80.28 | |
| + B b_adapt | 80.59 | per-note accuracy −3.0 at 5 c |
| + B preset b (segment veto, key prior) | 80.19 | no gain |
| + A fg45 | 79.24 | worse in every configuration |

- In Studio, the late-window onset reading adds about 0.3 J (glitchT 1.1 → 0.5 %). C's and B's versions of it tie.

**Stage 2, joint grid** (36 points Live, 144 Studio, 6 Eco):
- Release freeze {0, 2, 4} dB × b_adapt {off, on} × Studio late onset × D's veto margin and unsure thresholds.
- The J argmax is D + C + b_adapt ("DBC").
- **DC, DBC and BC tie on J:**
  - DEV, DBC − DC: +0.21 [−0.31, +1.03] Live; about +0.05 Studio (J 86.55 vs 86.51).
  - DEV, BC − DC: +0.14 [−0.49, +0.90] Live; +0.04 [−0.51, +0.40] Studio.
- **They differ in which trade they make:**
  - DC has the largest p95-vs-agreement gain (DEV, 8 c, Live chromatic: +1.8 c, against +1.4 for DBC and +0.5 for BC).
  - DC keeps 1.9-2.6 points more per-note accuracy than BC and DBC (significant on DEV).
- **Decision, made on DEV and recorded before TEST:** DC is the recommendation. This deliberately departs from the literal J argmax. All five systems (baseline, D, DC, BC, DBC) were then run once on TEST.

**Post-TEST observation (DEV data only, no TEST use).** The veto margin behaves like an operating-point knob along the improved frontier.
- At margin 0 ("accept a proposed switch when P(new) ≥ P(current)") plus the freeze, Live HQ chromatic DEV at 8 c gives:
  - per-note accuracy 81.3 % (0.25/0.3 margins: 80.3 %);
  - warble 29.9/min (against 27.2/min);
  - p95 22.9 c (against 21.6 c);
  - the same J.
- This is the basis for the optional "note hold" control in §5.3. That setting has not been tested on held-out data.

## 4. Results

**How to read the tables.**
- Prototypes A-C are their own presets, so their DEV numbers are in-sample.
- D, DC, BC and DBC use fold models on DEV and the full-DEV model on TEST.
- "acc agr" is frame accuracy on frames where annotators 1 and 2 agree. "note %" is the share of annotated notes whose most frequent target equals the annotated note. "agreeP" is the proxy note agreement on the published scale.
- The last row of each table is the inter-annotator ceiling: frames, and notes, where annotator 2 agrees with annotator 1.

### 4.1 Per metric at the harness's hard-tune setting (Tolerance 5 c)

#### Live HQ, hard chromatic

| system | DEV acc agr % | DEV note % | DEV warble /min | DEV p95 c | DEV agreeP % | TEST acc agr % | TEST note % | TEST warble /min | TEST p95 c | TEST agreeP % | glitchT % DEV / TEST | legato latency p90, ms, DEV / TEST |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (snapshot) | 74.1 | 82.9 | 53.7 | 26.8 | 87.4 | 70.1 | 80.8 | 56.6 | 27.6 | 87.1 | 3.3 / 4.2 | 89.0 / 99.3 |
| A adaptive reference (fg45) | 74.0 | 81.9 | 54.3 | 26.1 | 86.3 | 69.8 | 81.3 | 57.2 | 27.6 | 86.9 | 3.3 / 4.2 | 86.4 / 101.4 |
| B musical HMM (b) | 74.9 | 81.5 | 48.1 | 25.4 | 86.6 | 71.6 | 79.5 | 49.7 | 26.2 | 87.1 | 3.3 / 2.1 | 89.0 / 100.7 |
| B musical HMM (b_adapt) | 75.3 | 78.3 | 24.4 | 22.1 | 84.4 | 72.0 | 77.2 | 29.1 | 22.1 | 84.8 | 3.8 / 2.8 | 100.1 / 112.8 |
| C trajectory (release freeze) | 74.8 | 82.4 | 41.2 | 26.4 | 87.6 | 71.2 | 79.0 | 39.2 | 26.8 | 87.4 | 3.3 / 4.2 | 89.1 / 97.8 |
| D learned veto | 76.4 | 80.6 | 38.3 | 22.4 | 85.7 | 73.5 | 79.5 | 40.4 | 21.9 | 86.2 | 3.5 / 4.2 | 94.9 / 105.3 |
| **DC = D + C (recommended)** | 76.9 | 80.2 | 29.7 | 22.1 | 85.8 | 74.1 | 77.4 | 29.4 | 20.9 | 86.2 | 3.5 / 4.9 | 95.2 / 105.6 |
| BC = B b_adapt + C (DSP only) | 75.8 | 78.1 | 18.2 | 21.9 | 84.4 | 72.9 | 76.3 | 21.3 | 21.2 | 84.4 | 3.8 / 2.8 | 100.7 / 111.8 |
| DBC = D + B b_adapt + C | 76.3 | 77.3 | 15.4 | 20.4 | 83.9 | 73.6 | 75.4 | 19.2 | 19.1 | 83.9 | 3.8 / 3.5 | 103.6 / 111.8 |
| *Inter-annotator ceiling, frames / notes* | 90.7 / 84.3 | | | | | 89.3 / 81.0 | | | | | | |

#### Live HQ, hard key

| system | DEV acc agr % | DEV note % | DEV warble /min | DEV p95 c | DEV agreeP % | TEST acc agr % | TEST note % | TEST warble /min | TEST p95 c | TEST agreeP % |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 83.4 | 89.3 | 37.5 | 15.8 | 92.1 | 79.7 | 88.4 | 41.6 | 16.6 | 92.6 |
| A fg45 | 83.0 | 88.6 | 37.1 | 15.6 | 91.0 | 79.4 | 87.7 | 41.9 | 15.8 | 92.4 |
| B b | 83.7 | 88.3 | 33.4 | 14.4 | 91.5 | 80.0 | 87.7 | 36.8 | 14.6 | 92.2 |
| B b_adapt | 83.8 | 86.3 | 20.9 | 12.6 | 90.7 | 79.9 | 85.4 | 25.8 | 10.3 | 90.6 |
| C | 84.0 | 88.4 | 27.4 | 15.3 | 92.2 | 80.6 | 86.5 | 31.2 | 14.8 | 92.4 |
| D | 84.7 | 86.6 | 21.7 | 12.1 | 91.0 | 81.8 | 86.5 | 21.9 | 9.8 | 91.7 |
| **DC** | 84.8 | 85.7 | 16.8 | 11.9 | 91.0 | 82.0 | 84.9 | 19.8 | 9.8 | 91.7 |
| BC | 84.2 | 85.1 | 14.0 | 12.4 | 90.5 | 80.5 | 83.8 | 21.0 | 10.1 | 90.6 |
| DBC | 84.5 | 84.1 | 10.4 | 11.4 | 89.9 | 81.2 | 83.8 | 15.3 | 9.2 | 91.0 |
| *Inter-annotator ceiling, frames / notes* | 93.5 / 88.9 | | | | | 92.4 / 84.9 | | | | |

#### Studio HQ, hard chromatic

| system | DEV acc agr % | DEV note % | DEV warble /min | DEV p95 c | DEV agreeP % | TEST acc agr % | TEST note % | TEST warble /min | TEST p95 c | TEST agreeP % | glitchT % DEV / TEST |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 76.1 | 83.0 | 25.3 | 17.4 | 87.6 | 72.4 | 82.2 | 24.3 | 20.1 | 88.0 | 1.1 / 1.4 |
| A fg45 | 75.9 | 82.4 | 24.5 | 17.1 | 86.7 | 71.9 | 82.4 | 24.9 | 20.4 | 87.8 | 1.1 / 1.4 |
| B b | 76.5 | 82.8 | 23.9 | 16.6 | 87.3 | 73.0 | 81.5 | 22.2 | 18.4 | 87.1 | 0.5 / 0.7 |
| B b_adapt | 76.5 | 82.9 | 23.0 | 16.4 | 87.3 | 73.0 | 81.5 | 21.3 | 18.6 | 86.9 | 0.5 / 0.7 |
| C | 77.0 | 83.3 | 19.6 | 17.4 | 87.8 | 73.9 | 81.5 | 18.6 | 19.9 | 88.3 | 1.1 / 1.4 |
| D | 78.0 | 82.6 | 20.8 | 14.8 | 87.3 | 75.1 | 82.2 | 20.7 | 16.1 | 88.5 | 1.1 / 1.4 |
| **DC (+ late onset)** | 78.6 | 82.8 | 16.2 | 14.6 | 87.3 | 75.7 | 80.8 | 15.6 | 15.7 | 88.0 | 0.5 / 1.4 |
| BC | 77.4 | 83.2 | 17.6 | 16.4 | 87.5 | 74.3 | 80.4 | 15.9 | 18.1 | 87.1 | 0.5 / 0.7 |
| DBC | 78.6 | 82.4 | 14.9 | 13.3 | 87.0 | 75.9 | 79.7 | 13.8 | 14.8 | 87.1 | 0.5 / 1.4 |

#### Studio HQ, hard key (DEV / TEST)

| system | acc agr % | note % | warble /min | p95 c | agreeP % |
|---|---|---|---|---|---|
| Baseline | 85.3 / 81.5 | 90.4 / 90.0 | 16.1 / 18.6 | 4.2 / 5.5 | 93.2 / 93.3 |
| D | 86.6 / 83.3 | 89.4 / 88.8 | 12.7 / 12.6 | 3.6 / 3.0 | 92.3 / 92.9 |
| **DC** | 86.7 / 83.4 | 88.5 / 87.7 | 10.9 / 9.6 | 3.5 / 2.9 | 92.1 / 92.9 |
| BC | 86.1 / 82.5 | 89.6 / 88.6 | 13.2 / 14.4 | 4.2 / 5.2 | 93.1 / 93.1 |
| DBC | 86.7 / 83.4 | 88.7 / 87.7 | 11.2 / 9.3 | 3.5 / 2.9 | 92.2 / 92.4 |

#### Live Eco, hard chromatic, and Live HQ Natural key (DEV / TEST)

- B's and C's gates sit on the HQ path, so in Eco DC = DBC = D and BC = baseline. A (not shown) slightly hurts Eco, as in §4.3.

| system | Eco acc agr % | Eco note % | Eco warble /min | Eco p95 c | Eco agreeP % | HQ Natural acc agr % | HQ Natural note % | HQ Natural warble /min |
|---|---|---|---|---|---|---|---|---|
| Baseline | 73.3 / 70.0 | 84.1 / 82.4 | 94.5 / 93.2 | 30.9 / 30.8 | 88.0 / 89.0 | 81.6 / 78.0 | 84.5 / 83.8 | 20.0 / 23.4 |
| D | 76.8 / 74.1 | 81.1 / 81.3 | 62.9 / 67.1 | 23.4 / 23.2 | 86.0 / 87.8 | 82.6 / 79.4 | 83.1 / 82.2 | 11.5 / 12.3 |
| **DC** | = D | | | | | 82.6 / 79.6 | 82.3 / 81.1 | 9.1 / 10.8 |
| BC | = baseline | | | | | 82.4 / 78.6 | 82.1 / 81.0 | 9.0 / 14.7 |
| DBC | = D | | | | | 82.6 / 79.4 | 81.3 / 81.0 | 6.7 / 10.2 |

**Published warble episodes, all 40 files** (telemetry, exactly as skato_eval counts them; DEV with fold models plus TEST). These are the counts quoted in STATUS.

| | Live HQ chromatic | Live HQ key | Live HQ Natural | Studio HQ chromatic | Studio HQ key | Live Eco chromatic |
|---|---|---|---|---|---|---|
| Baseline | 743 | 525 | 284 | 341 | 228 | 1281 |
| D | 531 | 296 | 158 | 283 | 173 | 870 |
| **DC** | 404 | 239 | 129 | 219 | 144 | 870 |
| BC | 258 | 215 | 142 | 234 | 184 | 1281 |
| DBC | 222 | 159 | 103 | 199 | 146 | 870 |

### 4.2 J on DEV, nested LOSO and TEST

- Each condition is at its best Tolerance.
- **DEV** = all-DEV selection with fold models. **LOSO** = Tolerance chosen without the held-out singer (nested leave-one-singer-out).
- **TEST** = the Tolerances pre-registered from DEV.

| system | Live HQ DEV / LOSO / TEST | Studio HQ DEV / LOSO / TEST | Live Eco DEV / LOSO / TEST |
|---|---|---|---|
| Baseline | 79.11 / 78.72 / 79.14 | 85.63 / 85.49 / 84.86 | 78.88 / 78.88 / 77.22 |
| A fg45 | 78.47 / 78.30 / – | 85.18 / 85.03 / – | 78.52 / 78.52 / – |
| B b | 79.25 / 78.84 / – | 85.84 / 85.70 / – | = baseline |
| B b_adapt | 80.32 / 80.24 / – | 86.11 / 86.11 / – | = baseline |
| C release freeze, 4 dB | 80.12 / 79.90 / – | 86.06 / 85.95 / – | = baseline |
| D | 80.25 / 80.13 / 81.11 | 85.87 / 85.75 / 85.80 | 80.26 / 80.26 / 80.17 |
| **DC** | 80.79 / 80.67 / 81.39 | 86.51 / 86.51 / 86.43 | = D |
| BC | 80.93 / 80.85 / 80.86 | 86.54 / 86.54 / 85.20 | = baseline |
| DBC | 81.01 / 81.01 / 81.66 | 86.55 / 86.42 / 86.35 | = D |

**Nested-LOSO pooled metrics on DEV, Live HQ hard chromatic** (Tolerance picked per fold; mostly 12 c):

| | acc agr % | note % | warble /min | p95 c | agreeP % | glitchT % |
|---|---|---|---|---|---|---|
| Baseline | 74.4 | 81.3 | 44.5 | 23.8 | 86.2 | 3.3 |
| D | 76.1 | 80.1 | 33.4 | 20.8 | 85.2 | 3.5 |
| **DC** | 76.5 | 79.7 | 24.2 | 20.4 | 85.1 | 3.5 |
| BC | 75.8 | 77.8 | 16.8 | 21.2 | 84.7 | 4.1 |
| DBC | 76.3 | 77.1 | 14.4 | 20.1 | 84.1 | 4.1 |

### 4.3 Distance to the baseline's Tolerance frontier (one yardstick)

- Each cell is **dP95@equal agreeP (c) / dAgreeP@equal p95 (points) / dWarble@equal agreeP (per minute) / dNote%@equal p95**. Positive is better than any Tolerance setting of the baseline.
- Prototypes A-C are measured at their own preset Tolerance of 5 c. Rows marked 8 c are the plugin default.

| system | Live HQ chromatic | Live HQ key | Studio HQ chromatic | Studio HQ key | Live Eco chromatic |
|---|---|---|---|---|---|
| A fg45, 5 c, DEV | −2.0 / −0.9 / −9 / −0.5 | −2.6 / −1.1 / −7 / −0.5 | −3.4 / −1.1 / −6 / −0.7 | −0.6 / −1.2 / −3 / −0.9 | −4.9 / −1.3 / −31 / −0.8 |
| A fg45, 5 c, TEST | −2.2 / −0.2 / −8 / +0.5 | −3.3 / −0.2 / −5 / −0.5 | −1.2 / −0.9 / −2 / −0.2 | +0.4 / +0.3 / +2 / +0.1 | −1.1 / −0.5 / −6 / −0.4 |
| B b, 5 c, TEST | −0.7 / 0.0 / 0 / −0.6 | −2.3 / −0.5 / −1 / −0.3 | −1.3 / −0.5 / −2 / −0.5 | +0.1 / +0.1 / +1 / −0.1 | 0 |
| B b_adapt, 5 c, TEST | 0.0 / 0.0 / +9 / +1.2 | −1.0 / −0.7 / +4 / −0.4 | −1.7 / −0.8 / −1 / −0.5 | −0.5 / −0.4 / 0 / 0.0 | 0 |
| C, 5 c, TEST | +1.1 / +0.2 / +18 / −1.4 | −2.3 / −0.2 / +6 / −1.5 | +0.4 / +0.3 / +6 / −0.7 | +0.4 / +0.4 / +3 / −1.3 | 0 |
| D, 8 c, DEV | +1.5 / +1.5 / +7 / +1.8 | +1.1 / +0.9 / +8 / +0.5 | +0.8 / +0.4 / +1 / +1.2 | +0.1 / +0.4 / +1 / +0.7 | +1.4 / +0.5 / +2 / −0.5 |
| D, 8 c, TEST | +3.6 / +2.0 / +12 / +3.2 | +2.3 / +1.3 / +14 / +0.9 | +2.9 / +2.4 / +2 / +1.5 | +0.3 / +0.5 / +4 / −0.1 | +4.8 / +2.5 / +10 / +2.8 |
| **DC, 8 c, DEV** | +1.8 / +1.8 / +16 / +1.6 | +1.1 / +0.9 / +13 / −0.3 | +0.9 / +0.4 / +5 / +1.3 | +0.1 / +0.3 / +2 / −0.1 | = D |
| **DC, 8 c, TEST** | +4.4 / +2.3 / +23 / +1.3 | +2.3 / +1.3 / +16 / −0.7 | +2.7 / +2.1 / +7 / +0.3 | +0.2 / +0.3 / +5 / −1.3 | = D |
| BC, 8 c, DEV | +0.5 / +0.5 / +21 / −0.5 | +0.2 / +0.3 / +15 / −1.7 | +0.2 / +0.1 / +5 / +0.5 | +0.1 / +0.1 / +3 / −0.4 | 0 |
| BC, 8 c, TEST | +0.1 / +0.1 / +17 / +0.7 | −0.6 / −0.5 / +10 / −1.4 | −0.8 / −0.7 / +6 / −2.0 | +0.4 / +0.3 / +3 / −1.7 | 0 |
| DBC, 8 c, DEV | +1.4 / +1.2 / +22 / +0.1 | +0.2 / +0.2 / +14 / −1.5 | +1.0 / +0.6 / +5 / +1.3 | +0.1 / +0.3 / +2 / +0.1 | = D |
| DBC, 8 c, TEST | +1.8 / +0.8 / +17 / +0.7 | +0.8 / +0.6 / +15 / −0.8 | +2.2 / +1.1 / +6 / −0.7 | +0.3 / +0.5 / +5 / −1.2 | = D |

**Mean over the whole 0-45 c sweep (TEST), DC.** Live HQ chromatic +3.1 / +2.1 / +16 / +1.6; Live HQ key +1.2 / +1.1 / +14 / −0.4; Studio HQ chromatic +2.5 / +1.8 / +5 / +0.5.

### 4.4 TEST, paired by singer (9 singers, 95 % bootstrap CI)

DC vs baseline, both at the plugin default Tolerance (8 c):

| | J (mean of 3 conditions) | chromatic: acc agr | warble /min | p95 c | note % | agreeP | key: acc agr | warble /min | p95 c | note % |
|---|---|---|---|---|---|---|---|---|---|---|
| Live HQ | +2.23 [+1.44, +3.30] | +3.2 [+2.1, +4.0], 8/9 singers | −26.1 [−33.2, −22.0], 9/9 | −5.4 [−7.8, −3.1], 9/9 | −3.2 [−5.7, −1.5] | −1.2 [−2.3, +0.8] | +2.2 [+1.4, +2.8] | −18.9 [−26.9, −14.1] | −3.2 [−6.5, −1.5] | −3.0 [−5.4, −1.8] |
| Studio HQ | +1.39 [+0.12, +2.31] | +2.6 [+1.8, +3.3], 9/9 | −7.5 [−12.6, −4.3] | −3.2 [−4.5, −1.8], 9/9 | −2.0 [−4.8, +0.3] | −0.5 [−2.3, +0.7] | +1.8 [+1.0, +2.4] | −7.2 [−10.6, −3.4] | −0.8 [−3.7, −0.3] | −2.7 [−4.1, −1.4] |
| Live Eco (D) | +3.88 [+3.02, +4.80] | +3.7 [+2.7, +4.4] | −23.7 [−29.6, −18.4], 9/9 | −6.7 [−9.4, −4.9], 9/9 | −1.1 [−3.0, +0.3] | −0.7 [−2.2, +0.3] | +2.3 [+1.8, +2.7] | −26.1 [−34.5, −20.6] | −5.8 [−9.6, −3.0] | −1.6 [−3.8, −0.2] |

**Each system at its pre-registered DEV-best Tolerance**, mean J difference:

| comparison | Live HQ | Studio HQ |
|---|---|---|
| DC − baseline | +2.25 [+1.57, +3.28] | +1.57 [+0.53, +2.66] |
| DC − D (what C adds) | +0.28 [+0.05, +0.66] | +0.63 [−0.06, +1.17] |
| DBC − DC | +0.27 [−0.48, +1.35] | −0.07 [−0.38, +0.40] |
| BC − DC | −0.53 [−1.47, +0.73] | −1.22 [−2.02, −0.38] |

- In Live, C's release freeze buys warble −12.6/min (9/9 singers) and costs per-note accuracy −2.5 [−5.7, −0.4] in chromatic.
- DC vs baseline at 8 c, Live HQ: legato decision latency median +0 ms (chromatic) and +5.7 ms (key), p90 +3.7 and +2.5 ms. glitchT +0.7 and +1.4 points (1-2 onsets).

### 4.5 Detuned singers: the "follow singer's tuning" opt-in on top of DC

This is prototype A's detuned-singer benchmark on TEST, Live HQ hard chromatic, scored as intent accuracy %. Every note is moved to its A440 note + δ + N(0, σ), keeping the real micro-contours.

| σ, δ | baseline | DC | DC + fg45 (decision-only) | manual Detune = δ (oracle) | DC + fg45 follow: own-grid p50 |
|---|---|---|---|---|---|
| 0, 0 (in tune) | 74.7 | 77.5 | 77.5 | 74.7 | 1.8 c |
| 0, 20 | 71.2 | 73.6 | 76.5 | 74.7 | 3.6 c (decision-only: 20.1) |
| 0, 35 | 62.3 | 64.5 | 73.3 | 74.7 | 9.5 c (decision-only: 35.0) |
| 15, 35 | 58.2 | 59.1 | 64.8 | 72.5 | 26.1 c |

- Studio is similar: at σ 0, δ 35, intent accuracy goes 65.9 → 75.3.
- On real vocadito singers the tracker lowers J in every configuration (DEV: D + fg45 79.73 vs D 80.25 in Live). That is why it stays opt-in.

### 4.6 Synthetic suite (DESIGN §4 control numbers, proxy) and CPU

**Synthetic suite.** DC uses the full-DEV models, run through `lead/tools/synth_gate.py` with prototype B's no-new-failure rule.
- **HQ: PASS.**
- Worse:
  - melody_male Live rt0: p95 5.53 → 8.02 c.
  - heldout_pentatonic Live rt0: 10.03 → 11.94 c.
  - melody_female Studio rt0: 1.39 → 1.89 c.
  - heldout_room rt20: settle max 116 → 126 ms (Live) and 120 → 124 ms (Studio). The number of onsets settling in ≥ 30 ms is unchanged on every row.
- Better:
  - vibrato: 10.35 → 7.28 c (Live HQ) and 9.89 → 7.33 c (Studio).
  - heldout_room Live rt0: 26.64 → 20.32 c.
- **Live Eco: FAIL.** heldout_pentatonic rt0 goes 14.87 → 19.20 c, past the 15 c target. It comes from D's Eco model.
- BC passes everywhere. DBC behaves like DC.

**CPU and realtime safety** (`exp bench`, single thread, M-series, machine load about 10 of 16 cores):

| | corrector µs/hop, mean | per hop: median / p99 / p99.9 | allocations |
|---|---|---|---|
| Live HQ, snapshot | 1.3 | – | 0 |
| Live HQ, DC | 11.9 (+10.6; 0.8 % of a core at 64-sample hops, 44.1 kHz) | 14.4 / 29 / 45-55 µs | 0 |
| Studio HQ, DC | 16.5 (+11.9) | 20.1 / 38 / 59-66 µs | 0 |
| Live HQ, C alone | +0.0 | | 0 |
| Live HQ, BC | +1.1 | | 0 |

- The veto's work is needed only when the snapshot proposes a switch: 4.4 % of hops in Live HQ and 1.5 % in Studio HQ.
- Evaluating lazily, only on those hops, cuts the mean about 20×.
- It does not lower the worst case. That worst case is the open item for the 32-sample CPU gate (§6.5).

## 5. Recommended design

### 5.1 Live (causal), HQ and Max tiers

1. **Unchanged:** front end, voicing, phrase restart, octave guard, the MonoNote HMM forward filter, and the OR gate with Tolerance.
2. **Release freeze (from C).** A downward switch (m < cur) is blocked while level(t − 40 ms) − level(t) > 4 dB.
   - This is the decay of a phrase end, where singers droop below the midpoint.
   - It needs a 64-hop ring of `level_db` and adds about 0 µs.
3. **Learned switch veto (from D).**
   - When the gated HMM proposes cur → m, a candidate-note GBT scores the nearest legal note to r and its two legal neighbours.
   - The model has 300 trees, ≤ 31 leaves each, and 82 features. The features are relative to the decided pitch r, so they never see the note grid. They come from a 384-hop history: window means, medians and extremes, slopes, segment age, and level dip/rise.
   - The switch is kept if P(m) − P(cur) ≥ 0.25 or P(cur) < 0.3. Otherwise the current note is held for this hop, and the HMM re-proposes on later hops.
   - The veto never adds a switch and never touches onsets.
   - It applies only when the targets come from a scale or key. MIDI target mode bypasses it (untested).
   - Models are trained on the 20 DEV singers plus synthetic melodies from skato_eval's generator (seeds ≥ 50000). They are embedded at build time.
4. **Eco** keeps the snapshot law in the first release. D's Eco veto does move the frontier (TEST +4.8 c at equal agreement), but it fails the synthetic held-out pentatonic row and costs 0.8 % of a core on the tier meant to be cheap. Revisit it once that row is fixed (§6.3).

### 5.2 Studio (lookahead)

- The same as Live, with three differences:
  - Studio's own model has 104 features, including 13 over the D/hop lookahead: means, median, extremes, slope and level change of the visible future.
  - The veto margins are 0/0: a switch is accepted when P(m) ≥ P(cur).
  - The onset note is read from the median of the newest third of the onset window, instead of the whole window. B and C found this independently.
  - It halves Studio onset glitches on DEV (glitchT 1.1 → 0.5 %). On TEST, inside DC, glitchT is unchanged and the proxy published glitches rise by 2 of about 144 onsets, so this part still needs confirmation.
- The release freeze's level comparison uses the newest frame already in the lookahead ring. That is free in Studio.

### 5.3 User controls and defaults

- **Tolerance keeps its meaning.** A switch can only be proposed once the pitch is Tolerance cents past the midpoint (or for Tolerance ms). The veto and the freeze only remove proposals, so they never make the target move earlier or to a note Tolerance would not allow.
- **What changes is the operating point.** The corrector becomes stickier, so at a given Tolerance:
  - warble falls 35-55 % and p95 drops 1-5 c;
  - per-note accuracy drops 2-3.5 points.
- With DC, J is flat from 5 to 16 c. Keep the plugin default of **8 c**.
- Lowering Tolerance does **not** win back per-note accuracy (TEST chromatic: DC at 0 c reaches 78.1 % against the baseline's 79.7 % at 8 c). The control for that trade is the veto margin:
  - Offer it later as an advanced **"Note hold"** control: 0 = snapshot behaviour, bit for bit; low = margin 0; default = 0.25/0.3 as tested.
  - Validate the low setting on new held-out data first (§3).
- **"Follow singer's tuning" becomes an opt-in setting** (Reference: A440 / Follow singer). It uses A's gated and seam-guarded tracker (fg45) in follow mode, with the target offset latched per note.
  - For consistently detuned material (a cappella drift, instruments not at A440) it recovers most of the intent loss (§4.5).
  - The manual Detune knob remains the exact tool when the offset is known.
  - It is never a default: on real singers the tracker hurts, and follow mode moves the output off A440 by design.
- **No other new controls.** B's key and interval priors and C's other cues are not adopted.

### 5.4 Expected outcome

The table is the TEST change against the baseline, both at 8 c. The proxy reproduces the published p95 (per-file r 0.93-1.00); published glitches come mostly from the shifter path.

| | Live HQ chromatic | Live HQ key | Studio HQ chromatic | Studio HQ key |
|---|---|---|---|---|
| stable p95 | 25.6 → 20.1 c | 12.7 → 9.4 c | 17.4 → 14.2 c | 3.8 → 3.0 c |
| warble /min | 50.0 → 24.0 | 37.4 → 18.6 | 20.7 → 13.2 | 16.2 → 9.0 |
| frame accuracy (A1 = A2) | 70.5 → 73.8 % | 79.8 → 81.9 % | 72.8 → 75.4 % | 81.3 → 83.0 % |
| note agreement (agreeP) | 87.1 → 86.0 % | 92.6 → 92.0 % | 87.4 → 86.9 % | 92.0 → 91.3 % |
| per-note accuracy | 79.7 → 76.5 % | 87.5 → 84.5 % | 81.9 → 79.9 % | 89.3 → 86.7 % |

- Settling (proxy, 10 c for 20 ms) is unchanged in Live (TEST median about 23 ms, p90 52-54 ms) and better in Studio (TEST p90 40.9 → 35.4 ms chromatic, 39.3 → 30.0 ms key).
- Notes the target never reaches within 500 ms rise from 1.6 to 3.0 % (TEST Live chromatic, 8 c; DEV 1.1 → 1.4 %). This is the same stickiness as the per-note cost.
- The synthetic tuning target of 15 c is still not reached on real singing in Live HQ chromatic.
- The rest of the gap is decidability, not engineering. The switch-veto oracle (C) reaches p95 8.8 c on DEV, but a causal classifier separates real note changes from wobble only at AUC 0.74.
- The published onset glitches (6.3 % / 11.4 %) are mostly shifter and output-path events (proxy 1.1 %). Note decisions cannot fix them.

## 6. Implementation plan

### 6.1 Order (each step on its own, behind a flag, harness-checked)

1. **Release freeze and Studio late-window onset.**
   - DSP only, about 60 lines, no new state beyond a 64-value level ring.
   - On TEST the freeze alone lowers Live HQ warble by 31 % (C). The late-window onset halves Studio onset glitches on DEV (glitchT 1.1 → 0.5 %).
   - On TEST, inside DC, onset glitches are unchanged (glitchT 1.4 %) and the published-style proxy rises by 2 onsets (0.7 → 2.1 %). Re-check the onset rule on its own before shipping it.
   - Ship first.
2. **Learned switch veto (HQ/Max, Live and Studio).** Port D's veto path only; drop the experimental policies (mu mode, accelerate, proposal model, GRU/MLP).
3. **"Follow singer's tuning" opt-in** (A's tracker, follow mode, latched).
4. **The harness as a permanent skato_eval check** (§6.4). Do this before step 2 lands, so step 2 is gated by it.

### 6.2 skato_control: files and APIs

**New `src/intent/` module**
- `mod.rs`: `NoteIntent` (state, `fn veto(&mut self, cur, m, r_rel, targets) -> bool`), `IntentPolicy { margin, unsure }`, `IntentModel` (per mode/tier).
- `history.rs`: `History` ring of 384 `HFrame`s (p, p_s, p_c, centre, level, conf, vibrato w/extent/rate, voiced, onset, segment clock). It is filled by the front end every hop, voiced or not.
- `features.rs`:
  - `features_live(&History, age) -> [f32; 82]` and `features_studio(&History, lead) -> [f32; 104]`, ported from D's `learned.rs::compute` and the candidate features.
  - Windows are defined in ms and converted with `HopClock`, so 44.1, 48, 96 and 192 kHz give the same features.
  - Medians use a preallocated scratch buffer.
- `gbt.rs`:
  - A flattened tree ensemble: feature `u8`, threshold `f32`, child `u16`, leaf `f32`, about 12 B per node, about 220 kB per model.
  - The format is versioned and hashed. Models are embedded with `include_bytes!` and parsed into `&'static` arrays outside the audio thread.
  - `predict(&[f32]) -> f32`, three candidates, sigmoid, normalisation.
- `models/{live_hq,studio_hq}.vtgb` plus `MODEL_CARD.md` (data, splits, metrics, detector config hash, CC BY 4.0 attribution).

**New `src/release.rs`:** `ReleaseFreeze { ring: [f32; 64], .. }` with `push(level_db)` and `blocks(cur, m, hop_ms) -> bool` (4 dB / 40 ms constants).

**New `src/reftrack.rs`:** A's `RefTracker` (frame mode, τ 4 s, R gate 0.25-0.45, seam guard 15 c). It is active only for scale targets.

**`corrector.rs`**
- New fields: `intent`, `release`, `rt`.
- `set_settings` reads `Settings::note_hold` (future) and `Settings::reference`.
- `reset()` clears history, rings and tracker.

**`corrector/front.rs`**
- Push the history frame and the level ring every hop.
- Feed tracker evidence (A).
- The HMM's `rel` subtracts the tracker's decision offset, which is 0 unless opted in.

**`corrector/back.rs`**
- `decide()` wraps the snapshot decision. On `Change::Switch(m)` from an allowed held note, it computes the features lazily (only on proposal hops) and calls `intent.veto`.
- `release.blocks` is added to the gates in `decide_hmm_live` and `decide_hmm_studio`.
- The Studio onset median uses `lookahead_median_from(newest ⌈n/3⌉)`.

**`skato_core::Settings`**
- `reference: ReferenceMode { A440, FollowSinger }` (default `A440`).
- Optional later: `note_hold: f32` (0-1; 0 = snapshot).
- No change to `tolerance_*`.
- DESIGN §3.3 gets the new rule text: "Note decision: … + learned switch veto + release freeze; Studio onset from the late lookahead."

### 6.3 Tests

**Bit-exactness**
- With the veto, freeze and tracker off, the telemetry of every skato_eval scenario × mode × tier equals the snapshot's (golden hashes). The prototypes all did this; keep it as a test.

**Veto semantics**
- On fuzzed streams, every switch the corrector makes is a switch the snapshot rule proposed in the same state.
- Onsets are never altered.
- MIDI target mode bypasses the veto.
- A missing or corrupt model falls back to the snapshot.

**Model parity**
- 1,000 recorded feature vectors per model; Rust vs the Python export, |Δp| < 1e-5, same argmax.
- Feature parity between sample rates (44.1/48/96/192 kHz) on a resampled synthetic melody.

**Release freeze**
- Blocks only downward switches, only while the level falls > 4 dB in 40 ms.
- A real step down that plateaus is released when the level recovers.

**Realtime safety**
- Counting-allocator test with every component on, all modes × tiers, fuzzed NaN/Inf/extreme input: 0 allocations, finite output (D and C already have these tests).

**Synthetic gates**
- DESIGN §4 rows unchanged or better, with no new failure.
- Before Eco gets the veto, fix heldout_pentatonic Live Eco rt0 (≤ 15 c) by retraining the Eco model with pentatonic-like synthetic material, or with an Eco-specific margin.
- Add STATUS next-step 3's off-grid synthetic scenario that crosses semitone midpoints. It is the synthetic counterpart of what dominates real singing, and the veto should be tested on it.

**CPU**
- `skato_eval rt-bench` with the veto on: Live HQ p99.9 < 5 % at 128 and < 10 % at 32 (currently 2.38 % / 5.16 %), Max < 25 %.

### 6.4 The harness as a permanent skato_eval check

1. **Move the code.**
   - `ni_harness` (cache, replay, metrics, splits, data, eval/grid/pair, frontier) moves into `crates/skato_eval/src/intent/`.
   - CLI: `skato_eval intent {build-cache, eval, grid, pair, frontier, synth, validate, dump}`.
   - Commit `crates/skato_eval/data/intent_splits.txt` (the current `splits.txt`, frozen).
   - The detector-stream cache goes under `out/intent/cache/` (gitignored). It is rebuilt in about 15 s and keyed by a hash of the skato_detect version and engine config, so a detector change invalidates it.
2. **`validate` becomes a test** (`#[ignore]` unless vocadito is present): the replay equals the engine telemetry bit for bit on all 40 files × 3 configs × 3 conditions, and the published warble counts are reproduced.
3. **`skato_eval intent check` is the regression gate** (nightly or pre-merge, about 30 s with the cache). Against `docs/intent_baseline.csv`, which holds this report's TEST numbers at 8 c, it requires:
   - Live HQ and Studio HQ TEST p95, warble, frame accuracy, agreeP and note % within guard bands of the recorded values. Suggested bands: p95 +1 c, warble +3/min, accuracy −1 point, note % −1 point.
   - DEV frontier gain at 8 c ≥ +1.0 c in Live HQ chromatic.
   - The synthetic gate passes.
   - The model hash matches the model card.
   - Changes to skato_detect or the front end must re-run `grid` and `pair` and update the baseline file in the same PR.
4. **Retraining pipeline.**
   - `skato_eval intent dump --split dev`: features plus labels (A1 = A2 frames; fold structure).
   - `tools/intent/train.py`: D's `cand.py` recipe; HistGradientBoosting with 300 iterations, learning rate 0.08, 31 leaves, min leaf 200; synthetic augmentation; fixed seeds.
   - `skato_eval intent export` writes the binary model.
   - TEST is used once per model release, through the pre-registered `pair` run.
5. **STATUS and research docs.** The vocadito section reports the new rows. Add the note-intent results to `docs/research` (TECHNIQUES registry: "learned switch veto", "release freeze").

### 6.5 CPU plan for the veto

- Average cost after lazy evaluation is about 0.5 µs/hop. The worst case is one proposal hop, about 10-30 µs typical and 45-66 µs p99.9 as measured under load.
- If rt-bench at 32-sample buffers exceeds the 10 % gate, cut the worst case:
  - score only the two candidates the veto compares (−33 %);
  - cap the ensemble (e.g. 150 trees of depth ≤ 10), checking that the DEV frontier stays within 0.3 c of the current model;
  - or use D's MLP-64 alternative (+4 µs mean). Its DEV frontier gain was similar, but it was not run through the synthetic guardrail.

## 7. Risks

1. **The per-note accuracy cost is real.**
   - At an equal Tolerance setting, 2-3.5 points of notes keep the previous note's target, mostly short legato notes and falling notes at phrase ends (TEST CIs exclude 0). Even for perfectly in-tune detuned-benchmark singers, DC's per-note intent accuracy is 87.4 % against 89.7 %.
   - At equal p95 the loss mostly disappears in chromatic, not in key (§4.3).
   - Users who value "every note lands" over "no warble" need the Note-hold control.
2. **Effect sizes rest on few singers.** TEST has 9 singers; the gains are larger on TEST than on DEV (frontier +4.4 vs +1.8 c). Intervals are wide, and in Studio key the frontier gain is about 0 at 8 c (+2.0 c at 5 c).
3. **Labels.**
   - vocadito notes are rounded medians from two annotators. The ceiling is 89-93 % of frames and 81-89 % of notes.
   - "Intent" priors (keys, intervals) cannot be credited by these labels, which is one reason B showed nothing.
   - A score-referenced benchmark would be needed to judge them.
4. **Proxy limits.** No shifter is simulated. The published onset glitches and dropouts are outside this work. Results should be confirmed with real `skato_eval vocadito` renders before release.
5. **Generalisation of the model.**
   - It was trained on 20 a-cappella vocadito singers plus synthetic melodies, and is tied to the current detector and front-end features: any change there means retraining.
   - Without the synthetic augmentation, the vocadito-only model wrecked clean legato on the synthetic suite (+218 c of p95 increases). It absorbs dataset priors easily.
   - Belting, fast melisma, harmony stacks, Max tier and Live+ are untested.
6. **CPU tail at small buffers** (§6.5) and model size (≈ 220 kB per model after compaction; about 500 kB as text today).
7. **Eco:** a synthetic held-out regression, so Eco stays on the snapshot law for now.
8. **Selection.**
   - The recommendation departs from the pre-declared J argmax (DBC). The reasons are documented before TEST, and TEST shows DC and DBC tied.
   - The Note-hold low setting comes from DEV only and has not been tested on held-out data.
9. **Licence.** The model weights derive from vocadito (CC BY 4.0), so the plugin's notices need the attribution.
10. **The opt-in tracker** was evaluated on the detuned-singer benchmark. That benchmark shifts the detector stream rather than the audio, and assumes shift equivariance within ±50 c. It does not model phrase-level drift. Follow mode's output is off A440 by design and needs its own synthetic targets before skato_eval can gate it.

## 8. Cross-pollination (for the research index and TECHNIQUES registry)

- **"Rule proposes, model vetoes."** The learned model only removes switches proposed by the existing rule. That kept the HQ synthetic gate passing. Models that decided notes directly all lost to the snapshot. The pattern fits other hysteresis decisions too: voicing FSM release, the octave guard, Studio lookahead persistence.
- **The release freeze is a level cue.** It is the corrector-side twin of skato_detect's release-tail rules, and both could share one level-slope estimate.
- **One frontier yardstick.** The Pareto envelope of a Tolerance sweep (`lead/tools/frontier.py`) turns "is this better or just a different setting?" into a number. Use it for any knob-like trade-off: Retune vs settle, Flex-Tune, the voicing gate.
- **The detuned-singer benchmark** (shifting cached detector streams with known intent) is a cheap way to test any reference or offset feature, MIDI Learn included, without new recordings.
