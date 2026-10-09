# Following the DAW's key in Skatotune: Bitwig 6 and other hosts

## Short answer

- **No, Skatotune can't follow the "C Major" in Bitwig 6's transport bar, and no third-party plugin can today.** The readout is Bitwig's project key. It is made of two transport parameters, Root Note and Scale. You can pick from 23 scales, automate them, and modulate them from the master track ([6.0 notes](https://downloads.bitwig.com/6.0/Release-Notes-6.0.html)). Bitwig only passes the key to its own devices:
  - The 6.0 and 6.1 notes describe no way to send it to plugins ([6.1 notes](https://downloads.bitwig.com/6.1/Release-Notes-6.1.html)).
  - CLAP has no key or scale extension at all, stable or draft, on `main` or `next` ([ext](https://github.com/free-audio/clap/tree/main/include/clap/ext), [draft](https://github.com/free-audio/clap/tree/next/include/clap/ext/draft)).
  - Bitwig's controller-script API can't read the key either. DrivenByMoss's author answered "Sadly, not possible" ([KVR](https://www.kvraudio.com/forum/viewtopic.php?p=9245022)).
- **What works in Bitwig today:** Bitwig's own key-aware note devices (Key Filter+ with "Use Global Key") can feed notes into Skatotune's MIDI input. That carries the full scale and follows key automation. It needs a short test, and one small feature to make it solid (section 5).
- **Almost no DAW sends its key to plugins.** VST3 has a proper scale event in the spec, but no host is documented as sending it. So the plan is:
  - a **Key Source** setting;
  - cheap VST3 support for that scale event;
  - a **latched MIDI scale** mode that works in every DAW;
  - later, **key detection**.

---

## 1. Which hosts expose their key to plugins

"Displays a key" is not the same as "sends it to third-party plugins." The table separates the two.

| Host | Has a project key? | Sends it to third-party plugins? | Mechanism / format | Confidence |
|---|---|---|---|---|
| **Bitwig 6.0 / 6.1** | Yes. Root Note + Scale, 23 scales, automatable ([6.0](https://downloads.bitwig.com/6.0/Release-Notes-6.0.html)) | **No** (CLAP). **Unknown, probably no** (VST3) | CLAP: no extension exists. VST3: nothing documented about ScaleEvent or `ProcessContext.chord`. No ARA. Controller API can't read the key. | CLAP: confirmed. VST3: unconfirmed |
| **Ableton Live 12** | A scale on each clip; the transport shows the scale of the *selected* clip ([FAQ](https://help.ableton.com/hc/articles/11425083250972)) | **No** | Only native "Use Current Scale" devices get it ([release notes](https://www.ableton.com/en/release-notes/live-12/)). Max for Live can read `Song.root_note` / `scale_intervals` ([LOM](https://docs.cycling74.com/apiref/lom/song/)). | Likely |
| **Cubase / Nuendo 15** | Chord track with chord + scale events ([manual](https://archive.steinberg.help/cubase_pro/v12/en/cubase_nuendo/topics/chord_functions/chord_functions_scale_events_c.html)) | **Partly** | Steinberg's own Pitch Correct follows "Chord Track – Scale" through a MIDI track routed to the plugin ([manual](https://www.steinberg.help/r/cubase-pro/cubaseplugref/13.0/en/_shared/topics/plug_ref/pitch_correct_r.html)). It probably uses VST3 ScaleEvent/ChordEvent, but no third-party plugin is confirmed to receive them. | Unconfirmed |
| **Fender Studio Pro 8 / Studio One** | Key signature + chord track ([manual](https://fenderstudiopromanual.fender.com/en/Content/Arranging_Topics/Signature_Track.htm)) | **Only via ARA** | ARA2 content types KeySignatures (42) and SheetChords (45) ([ARAInterface.h](https://github.com/Celemony/ARA_API/blob/main/ARAInterface.h)). The PreSonus VST3 extension has no key field ([header](https://github.com/fenderdigital/presonus-plugin-extensions/blob/main/ipslcontextinfo.h)). | Likely |
| **Logic Pro** | Yes ([Apple](https://support.apple.com/guide/logicpro/set-the-project-key-signature-lgcp42b42603/mac)) | **No** | The AU musical-context block has tempo, meter and beat only (checked in the macOS 27 SDK). ARA chord sharing with Melodyne is absent per [Bonedo 2022](https://www.bonedo.de/artikel/celemony-melodyne-workshop-5-arbeiten-mit-der-akkordspur-und-akkorderkennung-in-melodyne-5/). We don't ship AU anyway. | AU: confirmed |
| **Pro Tools** | Key signature ruler | **API exists** | AAX `GetKeySignature(sampleLocation)`, which can be read at any sample position ([AAX_ITransport.h](https://github.com/juce-framework/JUCE/blob/master/modules/juce_audio_plugin_client/AAX/SDK/Interfaces/AAX_ITransport.h)). Which Pro Tools versions implement it is unknown. We don't ship AAX. | API: confirmed |
| **REAPER 7** | No project key; scale is set per MIDI editor | **No push** | REAPER's own API, which needs the host pointer and runs on the UI thread ([SDK](https://github.com/justinfrankel/reaper-sdk/blob/main/sdk/reaper_vst3_interfaces.h), [ReaScript](https://www.reaper.fm/sdk/reascript/reascripthelp.html)) | Confirmed |
| **FL Studio 2025 / 2026** | The "Project Key" is used by Loop Starter ([What's New](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/basics_new_25.htm)) | Not documented | none | Unconfirmed |
| **Dorico** | Key signatures | Probably no | ProcessContext flags are unreliable by Steinberg's own account ([forum](https://forums.steinberg.net/t/strange-vst3-hosting-issue-dorico-4-and-5/851395/10)) | Likely |
| **Ardour, JUCE-based hosts** | n/a | **No** | Their source code never sets chord or scale ([Ardour](https://github.com/Ardour/ardour/blob/master/libs/ardour/vst3_plugin.cc), [JUCE](https://github.com/juce-framework/JUCE/blob/master/modules/juce_audio_processors_headless/format_types/juce_VST3Common.h)) | Confirmed |

**What each plugin API offers:**

| API | Key/scale carrier | Notes |
|---|---|---|
| CLAP | **None** | `draft/tuning` is microtuning and `draft/octave-number` is note naming. Neither is a key. |
| VST3 | **`kScaleEvent` (type 7)**: root, a 12-bit mask, and a name ([ivstevents.h](https://github.com/steinbergmedia/vst3_pluginterfaces/blob/master/vst/ivstevents.h)) | `ProcessContext.chord` and `kChordEvent` describe the **current chord, not the key** ([ivstprocesscontext.h](https://github.com/steinbergmedia/vst3_pluginterfaces/blob/master/vst/ivstprocesscontext.h)). The plugin must request `kNeedChord`. |
| ARA2 | Key signatures + chords on the timeline | Works on recorded regions (offline). Bitwig and Live don't host ARA. |
| AAX | `GetKeySignature` | The only realtime key feed among the plugin APIs. Not available to us. |
| AU | None | |
| MIDI 2.0 | UMP Flex Data "Set Key Signature" ([spec](https://midi.org/universal-midi-packet-ump-and-midi-2-0-protocol-specification)) | Theoretical: no host is known to send it. |

No shipping third-party pitch corrector is documented as following a host key. Antares links key between its own plugins instead: Auto-Key's "Send to Auto-Tune" ([Antares](https://www.antarestech.com/products/auto-tune/auto-key)). Following the host key where it's possible would be a real point of difference.

---

## 2. What this means for Skatotune

### Bitwig 6
- **CLAP (the better format in Bitwig): impossible** until a CLAP key extension exists. Long term, propose one on free-audio/clap Discussions. Nobody has asked for one yet.
- **VST3: one theoretical channel** (`kScaleEvent`). Bitwig probably doesn't send it. A diagnostic build will tell us in five minutes (section 4). Don't switch Bitwig to VST3 hoping for this.
- **Works now:**
  - **(a)** The Key Filter+ → MIDI route: full scale, follows automation (section 5).
  - **(b)** Root only: one project-level modulator drives both Bitwig's Root Key and our Key. Both base values must sit at C, with identical ranges, because Bitwig wraps the value and ours clamps ([5.0 notes on project modulators](https://downloads.bitwig.com/5.0/Release-Notes-5.0.html)).
  - **(c)** Automate our Key/Scale lanes directly.
- **Gap to fix:** Key and Scale are on none of our Bitwig remote-control pages (`crates/skatotune/src/lib.rs:832`). Add a "Key" page so mapping and modulation are one click.

### Ableton Live 12
- **No route over VST3.** Live's scale follows clip *selection*, not playback, so it can't drive key changes on the timeline anyway.
- **Options:**
  - Key/Scale automation (works today).
  - A MIDI track into our MIDI input. MIDI routing into a VST3 audio effect in Live is still an open acceptance test (`docs/research.md` §9.5).
  - Possibly Live's Scale MIDI effect with "Use Current Scale" filtering a chord cluster, the same idea as the Bitwig trick (untested).
  - An optional Max for Live bridge that writes our Key/Scale. It needs Suite, and the user must add our Key/Scale parameters to Live's configure list because we have more than 64 parameters.

### Others
- **Cubase:** once we decode `kScaleEvent`, routing chord track → MIDI track → Skatotune may just work. That is the routing Steinberg's own Pitch Correct needs; users report it worked in Cubase 12 and broke in 13 ([forum](https://forums.steinberg.net/t/how-do-i-set-the-cubase-pitch-correction-plugin-to-follow-chord-track-scale/729975)). Even today, the chord track plays real chord notes into that MIDI track, so our MIDI Target mode can follow the chords.
- **Studio One:** ARA only. Defer.
- **Logic / Pro Tools:** need AU or AAX, which we don't ship. If we ever add AAX, `GetKeySignature` makes Pro Tools a real target.
- **REAPER / FL:** automation or MIDI.

---

## 3. Recommended design

### 3.1 Model
- The base scale becomes **a tonic plus a 12-bit pitch-class set**, and named scales become presets of that set. **The engine already supports this.** It consumes `Settings.note_states: [NoteState; 12]` (`crates/skato_core/src/types.rs:254`); `Settings.key`/`scale` are informational. Only the plugin layer resolves key and scale today (`resolve_note_states`, `params.rs:485`).
- New parameter **Key Source**, with the options **Manual / Host / MIDI Scale / Detect** (Detect comes later).
  - **Default: Manual.** Projects saved before this change take the default, so they load unchanged. Silently letting a host override the key would confuse people.
  - When Key Source is Manual and a host key arrives, show a non-blocking hint: "Host sends D Dorian. Follow?"
  - Save it with the project but don't make it automatable. It's a mode, not a performance control.

### 3.2 Precedence (top wins)
1. **MIDI Target, or Auto with notes held.** Performance input, exactly as today. It ignores note overrides, as `Targets::set_midi` already does.
2. **Per-note overrides** (On/Bypass/Remove) on top of the base scale.
3. **Base scale from Key Source:**
   - **Manual:** the learned scale if one exists (as today), otherwise the Key + Scale parameters.
   - **Host:** the latched host scale. If nothing has been received, use Manual and show an amber badge.
   - **MIDI Scale:** the latched note set. Before the first chord, use Manual.
   - **Detect:** the confirmed detected key. Before it is confident enough, use Manual.

   The learned scale counts as a manual edit. When the source isn't Manual it's kept but inactive, and greyed in the GUI. Learn mode itself is unchanged (it adds notes); MIDI Scale is the variant where a new chord replaces the set and is latched.

   When Key Source is MIDI Scale, the MIDI input defines the scale, so MIDI Mode is greyed out (Off for targets). Splitting by channel can come later.

**Decision for you: overrides absolute or relative.** Overrides are absolute pitch classes today. "Remove F" stays F after a modulation from C to D. Relative overrides ("remove the 4th degree") would survive key changes but don't match the keyboard UI. I recommend **keeping them absolute** for now and revisiting if host-following gets real use.

### 3.3 Key changes and timing
- **Apply once per host block.** That is at most about 10 ms at 512 samples / 48 kHz, and the corrector already glides between targets with Transition time. No block splitting; `SAMPLE_ACCURATE_AUTOMATION` stays false.
- **Protect the note being sung.** If it is still in the new scale, nothing changes. If it was removed, keep its target until the next note onset, capped at about 200 ms, then glide with Transition. Make this measurable in `skato_eval`.
- **Key changes don't need latency compensation.** They apply on the input timeline, the moment the singer sings, so Studio-mode lookahead doesn't affect them.
- **Chord grouping for MIDI Scale.** A new set starts with a note-on when nothing is held. Notes arriving within about 50 ms join it, and the set stays latched after release. This removes the retrigger gaps that leave audio untuned in Target mode.
- **Several host events in one block:** the last one wins. A mask with fewer than 2 notes counts as invalid and is ignored.

### 3.4 Stopped transport and missing data
- **Latch.** Hosts send scale events only when the scale changes. Nobody documents whether they resend on play or locate. So keep the last host scale across stop/start and locate, and **save it in the project**. Show it as "(saved)" until it is confirmed in this session.
- **Never fall back to Manual just because the transport stopped.** A singer warming up with the transport stopped should stay in key.

### 3.5 Partial host information

| Host provides | Skatotune does |
|---|---|
| Full scale (ScaleEvent) | Use it as the base scale. Name it from our table, or use the host's own name. |
| Root only (modulation, future APIs) | Host root + manual scale type. |
| Chord only (`ProcessContext.chord`, ChordEvent) | **Not a key.** Offer a separate **Chord Tones** option, default Off: *Prefer* biases the note-HMM prior toward chord tones; *Only* makes chord tones the targets, like MIDI Target. |
| Nothing | Manual |

### 3.6 Mapping host masks
- **Decode:** `root % 12`, `mask & 0x0FFF`. The SDK's own major example `0x5ab5` sets bits 12 and 14.
- **Relative or absolute is ambiguous.** The header says "Bit 0 = C". But Steinberg's VST-XML example ("A# Blues 1", key 10, mask `0x6ed`) only makes musical sense if the mask is relative to the root ([spec](https://steinbergmedia.github.io/vst3_dev_portal/pages/Technical+Documentation/Clipboard+VST-XML/Index.html)). So try both readings and pick the one that matches a named scale with the root as tonic. If neither matches, use the relative reading. When the root is C the two are identical. Log raw values in the diagnostics panel.
- **Name table:** our 13 scales plus the Bitwig ones we lack: Harmonic Major, Overtone, Blues Major, Double Harmonic Major/Minor, Whole Tone, Half-diminished, Diminished WH/HW, Major/Minor Triad. Bitwig's "Jazz Minor" is our Melodic Minor. Anything else shows as "Custom (n notes)", and the keyboard displays it.
- **Don't extend the Scale parameter casually.** Custom masks live in saved state, not in the parameter. nice-plug's EnumParam is an integer from 0 to n-1, and VST3 stores automation as `index / (n-1)`. Adding variants after release would shift every existing VST3 Scale automation lane. **Decision for you:** if you want Bitwig's names in the Scale menu, add them once, appended at the end, **before** public release.

### 3.7 GUI
- **Source pill** next to Key/Scale: MANUAL / HOST / MIDI / DETECT.
- **While following another source,** the Key and Scale selectors show the followed value in place (for example "D" and "Dorian"), greyed, with a HOST/MIDI/DETECT badge. The tooltip reads "Manual: C Major, used when the host sends nothing."
- **Badge states:**
  - green: "HOST D Dorian"
  - amber: "HOST, nothing received, using C Major"
  - grey: "(saved)"
- **"Copy to manual"** writes the followed key into Key/Scale through a GUI-thread parameter edit. A custom mask becomes the nearest named scale plus overrides. This is also how you apply a detected key with one click.
- **Graph:** a small marker at each key change, for example "→ D Dorian".
- **LABS → Host diagnostics:** state flags, the last ScaleEvent (root, mask, text), chord fields, and event counts.

### 3.8 Automation, presets and saved state
- **The host key never writes parameters.** That means no recorded automation, no undo spam, and no fight with automation lanes. Parameters can't be set from the audio thread anyway.
- **Key/Scale automation keeps working in Manual.** It is the "key track" that works in every DAW. In Host mode those lanes are ignored and greyed.
- **New saved words**, using the same `#[persist] AtomicU32` pattern as `learned_scale`:
  - the latched host scale (valid bit, encoding, tonic, mask);
  - the latched MIDI scale;
  - the detected key.
- **Presets:** treat Key/Scale/Key Source/overrides as song state. Sound presets should not overwrite them.

### 3.9 Realtime safety
- **The wrapper latches a plain `Copy` value.** No text is handled on the audio thread; the GUI derives the name from the mask.
- **Our block code resolves the base scale** with pure `u16` math, without allocating, and publishes it to the GUI with one atomic store, like `learned_states`.
- **The detector runs on our own background thread,** fed by rtrb, and returns its result through an atomic or `triple_buffer` (`docs/research.md` §9.6).
- **Extend `tests/no_alloc.rs`** to change the host key every block under `assert_process_allocs`.

---

## 4. Implementation plan

### 4.1 Vendor nice-plug
- Copy `nice-plug-0.4.2` and `nice-plug-core-0.4.2` into `vendor/`. The first commit is the untouched copy, so the patch diff stays reviewable. Keep their license files.
- In the root `Cargo.toml`:
  ```toml
  [patch.crates-io]
  nice-plug = { path = "vendor/nice-plug" }
  nice-plug-core = { path = "vendor/nice-plug-core" }
  ```
  The version stays 0.4.2, so `=0.4.2` still resolves, and nice-plug-egui picks up the patched crate automatically.
- Keep the patch commits small and tagged `nice-plug patch:`, then offer them upstream as a PR.

### 4.2 nice-plug-core
- Add raw fields to `Transport` (`src/context/process.rs:127`):
  - `host_scale: Option<HostScale { root: i16, mask: i16, sample_offset: u32 }>`
  - `host_chord: Option<HostChord { key_note: u8, root_note: u8, mask: i16 }>`
- `Transport::new` sets both to `None`, and the CLAP wrapper leaves them `None`.
- I prefer this side channel to a new `NoteEvent` variant. It touches fewer `match` statements in both wrappers, and decoding stays in our crate, where we can test it.

### 4.3 nice-plug VST3 wrapper (`src/wrapper/vst3/wrapper.rs`)
1. `getProcessContextRequirements` (lines 1772–1778): add `kNeedChord`.
2. Event loop, after the `kDataEvent` arm (around line 1329): add arms for `kScaleEvent` (7) and `kChordEvent` (6). Store the raw values in a latch on the wrapper that only the audio thread touches, and bump the diagnostic counters. The loop is gated on `MIDI_INPUT >= Basic` (line 1232); ours is `MidiCCs`, so it runs.
3. Transport build (around lines 1533–1560): when `state & kChordValid`, set `host_chord`; set `host_scale` from the latch.
4. Factor out `decode_musical_event(&Event)` and unit-test it with hand-built `vst3::Event` values. The vst3 0.3.0 bindings already have `ScaleEvent`, `ChordEvent`, `kNeedChord` and `kChordValid`.
5. Optional, later: keep the host context passed to `initialize()` (lines 106–109, currently discarded) for REAPER's API.

### 4.4 skato_core and skato_control
- **skato_core:**
  - Add `PitchClassSet(u16)` with `from_intervals`, `rotate`, `contains`, `len` and `to_note_states`.
  - Add `ScaleKind::mask()`.
  - Add a named-scale table with `identify(tonic, set)`.
- **No engine change needed.**
- **skato_control, optional:**
  - the Chord Tones "Prefer" bias on the note-HMM prior (§9.6 already plans a prior hook);
  - a check of how the note HMM behaves when the allowed set changes mid-note, for the hold-until-onset rule;
  - a later `KeyDetector` as pure code, so `skato_eval` can test it.

### 4.5 skatotune crate
- **`params.rs`:**
  - a `key_source` EnumParam (Manual default, not automatable);
  - the persisted words from 3.8;
  - `resolve_note_states` takes a `PitchClassSet` instead of `(ScaleKind, key)`.
- **New `key_source.rs`,** modelled on `learn.rs`:
  - `decode_vst3_scale(root, mask)` returning the set and which encoding matched;
  - `resolve_base(...)`, which implements the precedence in 3.2;
  - `MidiScaleLatch`, which does the chord grouping.
- **`BlockHost`** gains `fn host_musical(&self)`. `ContextHost` reads it from `Transport`, and test fakes supply it, so every test runs without a DAW.
- **`process_block`** (`lib.rs:341`): after the MIDI loop and before `to_settings()`, resolve the base scale, apply it, and publish it to the GUI.
- **Editor:** source pill, greyed selectors and badge, Copy to manual, the diagnostics panel in `labs_menu.rs`, and a "Key" remote-control page.

### 4.6 Tests
- **Decoding:**
  - `(0, 0x5ab5)` → C major
  - `(2, 0x0AB5)` → D major (relative reading)
  - `(2, 0x0AD6)` → D major (absolute reading)
  - `(10, 0x06ED)` → "A# Blues 1" set
  - `root 127` → G
  - `mask 0` → ignored
  - chord `(60, 0x0048)` → C, E, G
- **Precedence:** a table over source × host valid × MIDI held × overrides × learned scale.
- **Block level with a fake host:**
  - a key change takes effect on the next block;
  - a stopped transport keeps the latch;
  - reloading state restores the latch;
  - a key change during a sustained note holds, then glides.
- **`no_alloc`:** host key toggled every block.
- **`skato_eval`:** add `--key-track "0:C:major,4.0:D:dorian"` and measure switch time and click-free output.
- **Host probe matrix** using the diagnostics panel: Bitwig 6.1 VST3, Live 12, Cubase 15, Fender Studio Pro 8, REAPER 7, FL 2026.
  - In Cubase, use **D major** on the chord track to settle the relative-vs-absolute question, and check whether events are resent on play or locate.
  - Steinberg's HostChecker shows whether a host sets `kChordValid`. Its chord/scale event counters are never displayed ([source](https://github.com/steinbergmedia/vst3_public_sdk/blob/master/samples/vst/hostchecker/source/hostcheckerprocessor.cpp)), so use our own panel for events.

### Phases
| Phase | Work | Size |
|---|---|---|
| P0 | Try the Key Filter+ recipe in Bitwig 6.1; no code | minutes |
| P1 | Vendor nice-plug, VST3 scale/chord patch, diagnostics panel, run the probe matrix | Small |
| P2 | Key Source (Manual/Host/MIDI Scale), latch, precedence, GUI, Key remote page | Medium |
| P3 | Detect (propose, then apply), sidechain chroma, instance key link | Large |
| P4 | CLAP extension proposal; ARA/AAX only if we add those formats | n/a |

---

## 5. Fallbacks for hosts that send nothing

### 5.1 Bitwig today: Key Filter+ into our MIDI input
This route is unconfirmed until tested.
1. On an instrument track, put a looped note clip holding all 12 pitch classes, for example C3–B3 sustained.
2. Add **Key Filter+** with Use Global Key on and Foreign Note Handling = **Filter**. Filter passes only in-key notes ([6.0 notes](https://downloads.bitwig.com/6.0/Release-Notes-6.0.html)).
3. Get those notes to Skatotune, either with the vocal on a Hybrid track or with a Note Receiver in the vocal chain before Skatotune.
4. Set Skatotune to MIDI Mode **Auto** and **MIDI All Octaves** on. Held notes become pitch-class targets (`crates/skato_control/src/targets.rs:71`), and the manual scale covers any gap.

This follows root **and** scale, including all 23 Bitwig scales and automated key changes.

**Caveats:**
- **Don't use Learn mode.** It adds pitch classes on every note-on and never re-arms, so the old and new keys merge (`crates/skatotune/src/learn.rs`).
- **It's unknown whether Key Filter+ re-filters notes that are already held when the key changes.** If it doesn't, retrigger the cluster every beat or bar.
- **Per-note overrides are ignored while MIDI notes are held.**

The MIDI Scale (latched) source fixes the gaps and the retrigger problem.

### 5.2 MIDI Scale (latched)
This works in any DAW that routes MIDI to an effect:
- Cubase's chord track;
- Live's Scale device (untested);
- a singer holding a chord on a keyboard to set the key live.

### 5.3 Key detection (the Detect source)
- **Feasibility:** `docs/research.md` (lines 339 and 1170) already found that key detection is feasible off the audio thread, using a key profile over a decaying histogram of stable notes.
- **Analyse the *detected* (uncorrected) pitch.** Analysing the corrected output would just confirm whatever key is already set.
- **Vocal-only detection is weak.** It needs roughly 20–30 s of singing, and pentatonic lines fit several keys.
  - The relative major/minor ambiguity barely matters for correction: C major and A minor have the same note set.
  - Harmonic or melodic minor vs natural minor does matter.
- **So the default is *propose, then apply*:** "Detected: A minor (82%) [Apply] [Follow]". "Follow" only switches with high confidence and hysteresis (the new key wins for 8 s or more), and never mid-phrase.
- **Optional sidechain from the backing-track bus (chroma)** is much more accurate (§9.6 plans this). That's where Antares puts Auto-Key ([Antares](https://www.antarestech.com/products/auto-tune/auto-key)).

### 5.4 Instance key link
- One leader instance (manual or detecting) broadcasts root, mask and a counter to follower instances. This is the Antares model.
- Within one process, a static atomic is enough. Bitwig can host plugins in separate processes, so this needs shared memory. Untested.

### 5.5 Other options
- **Key/Scale automation:** works in every host now. In Bitwig, also the root-only modulator trick (section 2).
- **Paste or drop a VST-XML scale** from Cubase, Studio One or Dorico. Cheap; for later.
- **Max for Live bridge** for Live: follows the selected clip only.

### 5.6 Other DAW settings worth matching
- **Octave naming.** Our graph labels MIDI 60 as **C4** (`editor/graph.rs:210`). Live labels it **C3**; I believe Bitwig does too, but haven't checked. So the same note reads differently in the DAW and in our graph. Add a "Middle C = C3/C4" display option. CLAP's `draft/octave-number` lets a host tell us ([header](https://github.com/free-audio/clap/blob/next/include/clap/ext/draft/octave-number.h)); which hosts use it is unknown.
- **Reference tuning and microtuning.** Our Detune parameter (±100 cents) covers A=432 and similar. CLAP `draft/tuning` could feed host microtuning ([header](https://github.com/free-audio/clap/blob/next/include/clap/ext/draft/tuning.h)). Bitwig 6.1's Scala browser loads into its native Micro-pitch device, and nothing shows it reaches plugins. Low priority.
- **Remote controls.** Add the "Key" page described in section 2.

---

## Decisions for you
1. Per-note overrides: **absolute** (recommended for now) or relative to the key?
2. Add Bitwig's extra scale names to the Scale menu **before release**, or support them as custom masks only?
3. Default Key Source **Manual** with a "Follow?" hint (recommended), or follow the host automatically?
4. Key-change behaviour: **hold the current note until the next onset**, capped at about 200 ms (recommended), or glide immediately?

Files relevant to this plan:
- `crates/skatotune/src/lib.rs` (BlockHost line 194, process_block line 341, remote_controls line 832, MIDI_INPUT line 746)
- `crates/skatotune/src/params.rs` (resolve_note_states line 485, learned_scale line 973)
- `crates/skatotune/src/learn.rs`
- `crates/skato_core/src/types.rs` (ScaleKind line 98, Settings.note_states line 254)
- `crates/skato_control/src/targets.rs`
- `docs/research.md` (lines 339 and 1170, §9.6)
- `the cargo registry: nice-plug-0.4.2/src/wrapper/vst3/wrapper.rs`
- `the cargo registry: nice-plug-core-0.4.2/src/context/process.rs`