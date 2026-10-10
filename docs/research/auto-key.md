# Auto-Key and Key Source: detecting and following the song's key

**Date:** 2026-10-10 · **Status:** implemented: Key Source (Manual / MIDI Scale / Detect) and Auto-Key, measured on synthetic melodies and vocadito · **Issues:** [#6](https://github.com/sdunifon/skatotune/issues/6) Auto-Key, [#5](https://github.com/sdunifon/skatotune/issues/5) Follow the DAW's key

Builds on: [Following the DAW's key](host-key-scale.md) (the Key Source design and its decisions), the [architecture research](../research.md) (R§6 and §9.6: key profile over a decaying histogram of stable notes, off the critical path), the [note-intent study](note-intent.md) (its fixed vocadito DEV/TEST split, detector-stream cache and "tune on DEV, report TEST once" protocol), and the technique registry rows on key detection and the latched MIDI scale.

## Short answer

- **Key Source** decides where the base scale comes from:
  - **Manual:** Key and Scale, as before.
  - **MIDI Scale:** a chord on the MIDI input, latched. This is how a DAW's key reaches the plugin today, since no DAW sends its key to third-party plugins. In Bitwig, hold all 12 notes through Key Filter+ (Use Global Key) into Skatotune's MIDI input.
  - **Detect:** Auto-Key's confirmed key.
- **Auto-Key models keys as pitch-class sets, not just as the classic 24 major/minor profile keys.** That lets it propose harmonic and melodic minor, and gives it a confidence that means something.
  - On synthetic melodies in 36 keys (TEST), the proposed set is right at the end of 99.3 % of 60-second takes. Classic Krumhansl-Schmuckler gets 66.7 % at best, because it can never answer harmonic minor.
  - Proposals reported as confident are right 99.9 % of the time. The median time to a confident, correct proposal is 12 s (p90 25 s).
- **On real singing the evidence is honest about being thin.** vocadito takes average about 12 s of singing, too little to name a key reliably. Auto-Key and classic key finding both land at 25-50 % against the annotation-based reference. Auto-Key confirmed a key on **none** of the 12 TEST takes, so it auto-applied nothing wrong. That is the behaviour we want from "propose, then Apply".
- **Cost:** 75 ns per hop on the audio thread, 0.006 % of real time, with no allocation. The detector survives transport stops, CLAP sleep/wake and engine rebuilds, and its confirmed key is saved with the project.

## 1. What was built

### 1.1 Key Source (#5)

The design, decisions and precedence follow [host-key-scale.md](host-key-scale.md) §3. The user decided on 2026-10-10:

- **Sources:** Manual / MIDI Scale / Detect now. A Host source follows when a DAW sends its key.
- **Overrides:** per-note overrides stay absolute ("Remove F" stays F after a key change).
- **Default:** Manual, with a hint when MIDI or Detect has something.
- **Key changes:** a note being sung keeps its target until the next note starts, at most 200 ms.
- **Scale menu:** Bitwig's extra scales go in now (24 scales).

| Piece | Where | Behaviour |
|---|---|---|
| Key Source parameter | `params.rs` (`key_source`, not automatable) | Manual (default) / MIDI Scale / Detect |
| Resolution per block | `keysource::apply` | **MIDI Scale:** the latched set, overrides on top, MIDI Mode forced Off (its notes set the key, they aren't targets). **Detect:** the confirmed key, overrides on top, MIDI Target/Auto keep precedence, Learn is ignored. **Before anything is latched or confirmed:** the Manual scale. |
| Chord latch | `keysource::MidiScaleTracker` | A note-on sets the scale to every held note (2+ pitch classes). Note-offs never shrink it, except when notes pressed *before* the latest chord are lifted within 50 ms of it (a legato chord change). So staccato chords, staggered releases and legato cluster retriggers all latch correctly. |
| Saved state | `#[persist]` words `midi-scale`, `detected-key` | A project reloads in key before anything is played or sung. A restored detected key is marked "(saved)" until the singing confirms it. |
| Key-change hold | `skato_control` (`KEY_CHANGE_HOLD_MS`) | A scale change that removes the note being sung keeps its target until the next note starts (at most 200 ms), then glides. MIDI target changes still switch at once. |
| Scales | `ScaleKind` (24) | Bitwig's 11 extra scales, appended at the end. VST3 stores the Scale parameter as index / (n − 1), so inserting would move saved automation. |
| GUI | `editor/key_panel.rs` | MANUAL / MIDI / DETECT switch. Read-only key boxes while following. A status line ("DETECTED Am 86 % [APPLY]", "KEY MATCHES", "MIDI · D [COPY]", "DETECT · Am [COPY]", "WAITING FOR A CHORD", "(saved)"). Hint dots when MIDI notes arrive unused or a confident detection differs from Key/Scale. |

### 1.2 Auto-Key (#6)

`skato_control::autokey::KeyDetector` lives in the corrector, which feeds it every hop. It is not cleared by `Corrector::reset()`.

**Evidence.**
- Each voiced hop adds the *uncorrected*, vibrato-free pitch centre (on the user's Detune/bend grid) to a 12-bin pitch-class histogram. Analysing the corrected output would only confirm whatever key is set.
- Each hop is weighted by three factors:
  - the detector's confidence;
  - steadiness: the centre's slope over 20 ms, so fast scoops and portamento count little;
  - closeness to a semitone: 1 − (2d)², where d is the distance in semitones after the singer's offset is removed.
- The singer's overall tuning offset is a decayed circular mean of the grid deviations, used once its resultant length passes 0.3.
- The histogram decays with a time constant of *sung* time (120 s), so silence keeps it.
- The last steady note of each phrase (one that held its pitch class for at least 150 ms before a pause of at least 300 ms) is counted separately.

**Model.**
- **Candidates:** pitch-class sets, namely the 12 diatonic sets, 12 harmonic-minor and 12 melodic-minor sets. The 12 pentatonic sets are available but off by default.
- **Note model:** a sung note falls in the set with probability 1 − ε, spread evenly over the set's notes, or outside with probability ε. This is a set-level version of Temperley's probabilistic key-finding ([Temperley 2007](https://mitpress.mit.edu/9780262515191/music-and-probability/)).
- **Score:** each set gets its log-likelihood of the histogram (in "notes", at 4 per second of steady singing), plus a log prior per family, plus a tonal bias. The bias is 3 × the profile correlation of the set's best-named key, which breaks ties between sets that explain the notes equally well.
- **Confidence:** the posterior of the best set.
- **Naming:** the tonic and scale among the family's names (for example C major vs A minor for one diatonic set). It uses the Temperley/Kostka-Payne profile plus a bonus for phrase-final notes. Correction only needs the set: C major and A minor correct identically.

**Following.** The confirmed key, which the Detect source follows, changes only when two conditions hold:
- a new set has been the confident proposal (≥ 0.8) for 3 s (first key) or 8 s (a change);
- the singer has paused for at least 150 ms.

So it never switches mid-phrase. A key saved with the project seeds the confirmed key and stays until singing confirms or replaces it.

**Cost and safety.** 75.2 ns per hop including the 20 ms updates, 0.0056 % of real time at 48 kHz (`cargo test --release -p skato_control --lib cost_per_hop -- --ignored --nocapture`, M4 Max). Fixed-size state, no allocation. The whole state is copied into a new engine on a rebuild at the same sample rate (`Engine::set_key_detector`).

## 2. Evaluation protocol

`skato_eval autokey` runs material through the real control law (Live HQ) and snapshots the detector every 0.25 s.

- **Synthetic melodies:**
  - 36 keys: every root in major, natural minor and harmonic minor.
  - 4 takes per key per split, 60 s each. DEV seeds and TEST seeds are disjoint.
  - The tunes:
    - stepwise, weighted towards the tonic (3), fifth (2.5) and third (2);
    - phrases of 4-9 notes, mostly ending on the tonic;
    - 3 % chromatic passing notes;
    - scoops on 35 % of notes;
    - vibrato of 5-6.5 Hz and ± 20-60 c on notes ≥ 0.35 s;
    - ± 10 c per-note jitter and a per-take singer detune of up to ± 30 c.
  - The truth is the generating key.
- **vocadito:** real a-cappella singing, using the note-intent DEV/TEST split by singer (28 / 12 takes) and its cached Live HQ detector streams.
  - vocadito has no key labels, so the truth is an **oracle set**: the candidate set that covers the most annotated note time of both annotators (singer offset removed), the smaller set on a tie. It is a proxy; on short takes a single raised seventh can tip it to harmonic minor.
  - The vocadito evaluation's own key (Krumhansl-Kessler on annotator 1's notes) is reported as a second reference ("name").
- **Metrics:**
  - proposal set equals the truth set at 5/10/20/30 s and at the end;
  - tonic and scale named right;
  - precision and share of confident (≥ 0.8) proposals;
  - time until the proposal is right and confident for the rest of the take;
  - confirmed keys;
  - (vocadito) annotated note time inside the final proposal's set.
- **Selection:**
  - Everything was chosen on DEV, in two stages:
    1. the accumulation knobs (decay, steadiness, offset tracking) by replaying;
    2. 3,888 scoring settings (ε, rate, family priors, tonal bias, profile, phrase-final weight, "sung" share) offline on the recorded evidence.
  - The objective was ¼ synthetic accuracy at 10 s + ¼ at 30 s + ¼ confident precision + ¼ vocadito accuracy at the end.
  - TEST was run once, with the chosen setting, which is now `KeyConfig::default()`.

## 3. Results (TEST)

**Synthetic, 144 takes of 60 s:**

| System | @5 s | @10 s | @20 s | @30 s | end | name right | confident proposals right | time to detect (median / p90) |
|---|---|---|---|---|---|---|---|---|
| **Auto-Key** | 62.5 % | 81.2 % | 95.8 % | 97.2 % | **99.3 %** | 99.3 % | **99.9 %** (76.6 % of the time confident) | **12.0 s / 25.1 s** (2 takes never) |
| Krumhansl-Schmuckler, Temperley profile | 46.5 % | 54.9 % | 63.2 % | 66.0 % | 66.7 % | 66.7 % | – | – |
| Krumhansl-Schmuckler, Krumhansl-Kessler | 38.2 % | 50.0 % | 57.6 % | 60.4 % | 65.3 % | 65.3 % | – | – |
| Krumhansl-Schmuckler, Aarden-Essen | 38.2 % | 45.1 % | 59.0 % | 59.7 % | 65.3 % | 63.2 % | – | – |
| Krumhansl-Schmuckler, Bellman-Budge | 31.9 % | 47.2 % | 52.8 % | 56.9 % | 65.3 % | 65.3 % | – | – |

- By key type (end / @10 s): major 97.9 / 95.8 %, natural minor 100 / 81.2 %, harmonic minor 100 / 66.7 %.
- Harmonic minor takes longest because the raised seventh is needed to tell it from natural minor.
- **Detect source:** confirmed a key on 142 of 144 takes, all 142 right.

**vocadito, 12 takes (mean 16.7 s, about 12 s of singing):**

| System | @10 s | @20 s | end | name right (vs KK of A1) | confident proposals right | confirmed |
|---|---|---|---|---|---|---|
| **Auto-Key** | 25.0 % | 41.7 % | 41.7 % | 41.7 % | 68.4 % (confident 4.8 % of the time) | **0 of 12** |
| Krumhansl-Schmuckler, Temperley | 33.3 % | 41.7 % | 41.7 % | 50.0 % | – | – |
| Krumhansl-Schmuckler, Aarden-Essen | 41.7 % | 50.0 % | 50.0 % | 41.7 % | – | – |
| Krumhansl-Schmuckler, Krumhansl-Kessler | 33.3 % | 25.0 % | 25.0 % | 75.0 % | – | – |

- The final proposals' sets cover 80.4 % of the annotated note time; the oracle sets cover 84.8 %.
- The KK "name right" of 75 % is partly circular: the reference was computed with the same profile, from the annotations.
- **Reading it:** 12 short takes can't separate the methods (one take is 8.3 points). Every method gets roughly half the keys. Auto-Key's confidence correctly stays low: the median final confidence is 0.54, and the two takes that reached 0.95 did so only in their last seconds, too late to confirm. So the Detect source auto-applied no key, wrong or right.

**DEV tuning, what mattered:**

| Choice | Finding |
|---|---|
| Set likelihood vs profile correlation | The set model is the large gain: it can answer harmonic and melodic minor and gives a calibrated confidence (synthetic confident precision 99.7-99.9 %). |
| Steadiness gating | Hop-to-hop slopes read detector jitter as glides; over 20 ms, real held notes still drift. The best half-weight slope was the loosest tried (4 c/ms), so gating hardly matters on real singing. It stays loose. |
| Decay | 120 s ≥ 60 s > 30 s (small differences); with 8 s confirmation hysteresis, modulations still take over within a few phrases. |
| Singer-offset tracking | On: synthetic @10 s 83 % vs 79 % off. |
| "Sung" agreement | Counting sets that differ only on unsung notes as agreeing makes a tune that never sings its seventh confident. The DEV objective preferred plain posteriors (higher confident precision), so it is off (`sung_share = 0`): such a tune is proposed (diatonic, by prior) but not confirmed. |
| Tonal bias, profile, phrase-final notes | Bias 3 with the Temperley profile; the phrase-final bonus changes little (top settings differ by < 0.5 points). |
| Pentatonic family | Off in the best settings: a pentatonic tune is proposed as the diatonic set that contains it. |

## 4. Limitations and next steps

- **Real-singing evidence is thin.** vocadito's takes are too short and unlabelled. A proper check needs full songs with labelled keys, vocal stems and modulations. That belongs in the perceptual test suite (#26).
- **Vocal-only is the hard way.** A sidechain from the backing track (chroma) is the reliable route (R§9.6), and it's where Antares puts its Auto-Key ([#34](https://github.com/sdunifon/skatotune/issues/34)). It needs an aux input bus, its own evaluation, and a decision on the bus layout before release.
- **Synthetic material is generated by our own melody model.** It contains only scale notes plus 3 % chromatic passing notes. Blue notes, modal mixture and borrowed chords in real songs will make detection slower and the confidence lower.
- **Host source** ([#33](https://github.com/sdunifon/skatotune/issues/33)): add it when a DAW sends its key (VST3 `kScaleEvent`, or a future CLAP extension). The Key Source resolution and GUI already have the place for it.
- **Cross-pollination:** see §5.

## 5. Cross-pollination

- **#7 note intent / #32 promotion:** the confirmed key can be the key prior of a musical note model (prototype B cut warble 48 % with key/interval priors). The key-change hold applies to every scale change, so it also makes MIDI Learn and manual edits mid-note smoother.
- **#26 perceptual test suite:** add `skato_eval autokey`, plus a real-song key set with modulations, as a suite member. The synthetic melody generator (in-key tunes with scoops, vibrato and detune) is reusable for note-decision tests.
- **#10 graph mode:** offline analysis can run the same detector over a whole take (no confirmation delay) to pre-fill the key.
- **#12 A/B vs Auto-Tune:** Antares Auto-Key works on the mix (a separate plugin), so a fair comparison needs the sidechain path.
- **#28 Apple Silicon:** nothing to offload. At 75 ns per hop, Auto-Key is negligible.

## Techniques evaluated

| Technique | Evidence | Status |
|---|---|---|
| Pitch-class-set likelihood (in/out probability ε, family priors) with posterior confidence | Synthetic TEST 99.3 % set accuracy at the end, confident precision 99.9 %; vocadito inconclusive | Adopted |
| Tonal-bias tie-break (profile correlation of the set's best name) | DEV top settings use bias 3 | Adopted |
| Krumhansl-Schmuckler profile correlation (KK, Temperley, Aarden-Essen, Bellman-Budge) | Synthetic ≤ 66.7 % (no harmonic minor) | Baseline; Temperley profile used for naming |
| Offset-compensated soft pitch-class histogram (circular-mean singer offset, in-tune weight) | DEV synthetic @10 s +4 points | Adopted |
| Steadiness gating of the evidence | No gain on real singing; loosest setting won | Adopted (loose, 4 c/ms) |
| Phrase-final tonic bonus | < 0.5-point effect | Adopted (small weight) |
| "Sung-notes" agreement confidence | More confident on incomplete scales, lower precision on DEV | Available, off by default |
| Confirmation with hysteresis only between phrases | 142/144 synthetic takes confirmed correctly; 0 wrong confirmations on vocadito | Adopted |
| Latched MIDI chord as the key source (older-note lift within 50 ms) | Unit and end-to-end tests: staccato, staggered release, legato retrigger | Adopted |
| Key-change hold (keep the sung note ≤ 200 ms) | Control tests, Live and Studio | Adopted |

## Reproduce

```bash
cargo run --release -p skato_eval -- autokey                  # DEV sweep, then TEST once → out/autokey/report.md
cargo run --release -p skato_eval -- autokey --no-sweep       # TEST with the shipped KeyConfig::default()
cargo test --release -p skato_control --lib autokey           # detector behaviour
```

The profile values are as tabulated by the Humdrum `keycor` tool ([extras.humdrum.org/man/keycor](https://extras.humdrum.org/man/keycor/)) and music21 ([`analysis/discrete.py`](https://github.com/cuthbertLab/music21/blob/master/music21/analysis/discrete.py)). The originals are Krumhansl & Kessler (1982), Temperley (2007), Aarden (2003) and Bellman (2005). music21 notes that the origin of the Aarden-Essen *minor* weights is unknown.
