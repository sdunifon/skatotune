# Learned note hold: embedded models

The Labs option **Learned note hold** (`Settings::note_hold`) holds the current note when the note decision proposes a switch that looks like wobble rather than a new note. These two files are the models behind it: gradient-boosted tree ensembles that score candidate notes. They are embedded with `include_bytes!` (`src/intent/mod.rs`) and parsed once per process. The design, evidence and caveats are in [`docs/research/note-intent.md`](../../../docs/research/note-intent.md).

| file | used in | features per candidate | policy (keep the switch when) | FNV-1a checksum |
|---|---|---|---|---|
| `live_hq.skgb` | Live, Live+, Live-Min at HQ/Max | 82, causal | P(new) − P(current) ≥ 0.25 or P(current) < 0.3 | `0x384c1fa24660be2c` |
| `studio_hq.skgb` | Studio at HQ/Max | 104 (13 over the lookahead) | P(new) ≥ P(current) | `0x73e2c583a6cd1f89` |

Each file is 300 trees with at most 31 leaves (18 300 nodes) in the `SKGB` format described in `src/intent/gbt.rs`. They were converted from the training export (`LDMODEL` text, SHA-256 `4417f082…eac078` for Live and `78b94a1f…cba535` for Studio) with `skato_eval note-intent convert-model`. A test checks the checksums above, so a changed file has to be described here.

## Training data

- **vocadito** (Bittner, Pasalo, Bosch, Meseguer-Brocal and Rubinstein, 2021; [CC BY 4.0](https://zenodo.org/records/5578807)): the 28 takes of the **20 DEV singers** of the fixed split in [`crates/skato_eval/data/note_intent_splits.txt`](../../skato_eval/data/note_intent_splits.txt). The 12 takes of the 9 TEST singers were never used for training or for any choice; they are read once per model release. The attribution is in `THIRD-PARTY-NOTICES.txt` (`cargo xtask notices`).
- **Synthetic singing**: 400 random lines from `skato_eval`'s generator (seeds 50 000-50 399, disjoint from every evaluation scenario), with notes anywhere within ±48 c of the grid, drift, scoops, glides, vibrato and consonant onsets. Without it the vocadito-only model wrecked clean legato on the synthetic suite.
- **Labels**: hops inside an annotator-1 note where both annotators round to the same legal note, in the chromatic grid and in the file's key (DEV); every hop inside a note (synthetic, from the true pitch). Each labelled hop gives three candidate rows (the legal note nearest the decided pitch r and its legal neighbours), labelled 1 for the annotated note. DEV hops are subsampled by 2, synthetic ones by 4.

## Features

Per hop, 45 causal features of the decided frame, all relative to its pitch r so the grid never shows: window means (3-400 ms), voiced fractions, medians, minima/maxima and spreads of the pitch, older-context means, least-squares slopes, the pre-smoothed pitch, analysis centre and vibrato-free centre, segment age, time since unvoiced, level cues (dip, rise, drop over 25 ms), detector confidence, vibrato weight/extent/rate, the voiced fraction over 50 ms, the largest hop step and the time since a step of more than 40 c. Studio adds 13 over its lookahead (means, median, extremes, slope, level change, voiced fraction). Per candidate: the offset of r from it and its size, 21 pitch summaries relative to it and the size of 12 of them, and the gaps to its legal neighbours (Studio: plus 9 lookahead summaries relative to it). The code is `src/intent/features.rs`; the features are tied to the current detector and front end, so a change there needs a retrain.

## Retraining

```sh
cargo build --release -p skato_eval
E=target/release/skato_eval
$E note-intent dump --configs live_hq --split dev --out out/train/live_hq_dev.f32          # needs data/vocadito
$E note-intent dump-synth --configs live_hq --n 400 --seed0 50000 --out out/train/live_hq_synth.f32
python3 tools/note_intent/train_veto.py --dev out/train/live_hq_dev.f32 --synth out/train/live_hq_synth.f32 --out out/train/live_hq.ldm
$E note-intent convert-model out/train/live_hq.ldm crates/skato_control/models/live_hq.skgb
# Studio: the same with studio_hq.
```

The script fits scikit-learn's `HistGradientBoostingClassifier` (300 iterations, learning rate 0.08, 31 leaves, at least 200 samples per leaf, L2 1.0, `random_state` 0). With numpy 2.3.1 and scikit-learn 1.7.1 these steps reproduce both shipped models byte for byte (the dumps match the original training data, and the converted files match the checksums above); another scikit-learn version may give a slightly different ensemble. `--folds` also writes one model per held-out DEV singer, for honest DEV numbers.

Before shipping a new model, choose and check it on DEV only (fold models), then run `skato_eval note-intent --split test` once and record the result in the research report; the vocadito check in `crates/skato_eval/tests/note_intent.rs` holds the current TEST numbers and has to be updated with them.

## What the shipped models do (TEST, 9 singers, Tolerance 8 c)

Live HQ hard chromatic: frame accuracy 70.5 → 73.8 %, warble 50.0 → 24.0 per minute, stable-frame p95 25.6 → 20.2 c, note agreement 87.1 → 86.0 %, per-note accuracy 79.7 → 76.5 % (a real cost). Studio HQ hard chromatic: 72.8 → 75.4 %, 20.7 → 13.2 per minute, 17.4 → 14.2 c, per-note 81.9 → 79.9 %. These numbers include the release freeze and, in Studio, the late onset (`skato_eval note-intent`). Untested: belting, fast melisma, harmonies, the Max tier, Live+ and Live-Min, MIDI targets (bypassed) and new singers.
