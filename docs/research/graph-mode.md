# Skatotune Graph / Editing Mode: springboard research

**Status:** research only. This report maps the territory; it does not fix a design. Date: 2026-10-09.

**Purpose:** a starting point for a future Graph / Editing mode in Skatotune (formerly Vocal Tune), comparable to Melodyne's note editing or Auto-Tune's Graph Mode. It is not part of v1, and the requirements memory says graph mode was not requested for v1.

**Inputs:**
- Three fact-checked research passes, all dated 2026-10-09: prior art, ARA and host support, and technical architecture.
- `docs/STATUS.md`, `docs/DESIGN.md` and `docs/research.md`.
- A read-only inspection of the plug-in code and of the nice-plug 0.4.2 sources in the cargo registry.

**Evidence labels.** Every claim carries one of these labels:

| Label | Meaning |
|---|---|
| **[F]** | Fact confirmed in the cited primary source, in local code, or by a local measurement |
| **[W]** | Weak evidence: a forum post, search snippet, retailer copy or secondary report, or a source not re-checked in the last pass |
| **[I]** | Inference, estimate or recommendation (ours) |
| **[U]** | Unverified, or the sources conflict |

---

## 1. Executive summary

1. **What graph mode is.** The plug-in learns a take, either by listening to the host play it or by being handed the audio. It turns the take into editable notes and pitch curves. On playback it applies the user's corrections at the right timeline positions instead of deciding notes causally. [I]

2. **Why it matters for Skatotune specifically.**
   - On real singing (vocadito), Live HQ hard tune has a stable-frame p95 of 23-30 c. 92 % of the frames above the p95 lie within 10 ms of a target switch. Forcing the annotated note gives 10.8 c. (`docs/STATUS.md` §6) [F]
   - Our largest real-singing error is therefore *which note* the causal control law picks, not how well the shifter shifts. An edited note map removes that error wherever the user has edited. [I]
   - Every professional competitor has such an editor (Melodyne; Auto-Tune Pro 11 Graph Mode). Our two target hosts have none for vocals:
     - Ableton Live 12 has only the real-time Auto Shift, with meters but no graph ([Live manual](https://www.ableton.com/en/live-manual/12/live-audio-effect-reference/)). [F]
     - Bitwig 6.1's new pitch features live in the Sampler, not on vocal tracks ([MusicTech](https://musictech.com/news/gear/bitwig-studio-6-1/)). [F]

3. **The hard constraint: neither target host supports ARA.**
   - Live 12.4.6 and Bitwig 6.1 have no ARA ([Celemony compatibility](https://services.celemony.com/en/melodyne/editions-and-technical-matters); [Rekkerd Bitwig 6 review](https://rekkerd.org/review-bitwig-studio-6/); [Bitwig 6.1 notes](https://downloads.bitwig.com/6.1/Release-Notes-6.1.html)). [F]
   - In those hosts, Melodyne falls back to "Transfer" and Auto-Tune to "Track Pitch". [F]
   - So in-host graph mode for Live and Bitwig must be **capture-based**. [I]
   - ARA is a later tier for other hosts:
     - Cubase/Nuendo, Fender Studio Pro and REAPER (VST3-ARA).
     - Logic (AU-ARA, and only under Rosetta on Apple silicon).
     - Pro Tools (AAX-ARA).

4. **Recommended MVP [I]: capture pitch only, analyse offline, correct live from a timeline.**
   - The plug-in captures while the host plays, in the Auto-Tune 8 "Track Pitch" model ([AT8 manual](https://www.strumentimusicali.net/manuali/2017/11/06/9b/antares-autotune8-en.pdf) pp. 13, 20, 37).
   - When playback stops, it analyses the take offline and non-causally with our existing detector and note HMM, then discards the audio.
   - The user edits Melodyne-style notes: centre pitch, drift %, modulation %, a Correct Pitch macro, and a separate detection-fix mode.
   - The edits compile into timeline targets that the existing engine follows during playback, while the live detector still supplies the input pitch.
   - State lives in the plug-in chunk, and undo is on screen.
   - The MVP does not store audio and does not edit timing.

5. **Effort [I], in engineer-weeks for one developer who knows the codebase:**

   | Work | Effort |
   |---|---|
   | Host spikes | 1-2 wk |
   | Headless analysis | 2-3 wk |
   | Capture MVP | 6-9 wk |
   | **Usable MVP in Bitwig and Live (total)** | **about 9-14 wk** |
   | Editing depth | +3-4 wk |
   | Audio capture and HQ render | +4-6 wk |
   | Time editing | +4-8 wk |
   | ARA | 8-12+ wk, very uncertain |

   ARA only pays off in hosts we do not target yet.

6. **Cheapest proof first [I].**
   - A headless `skato_eval notes` pipeline can show, before any UI exists, whether the offline note map is good enough and how much it improves the rendered output.
   - The existing standalone app could become a file-in/file-out editor to try the editing model with no host-sync risk.

7. **Top risks:**
   - Captured data goes stale after host edits. Even ARA hosts get this wrong: Celemony reproduced a Cubase + Melodyne undo de-sync in July 2026 ([Steinberg forum](https://forums.steinberg.net/t/cubase-13-melodyne-ara-melodyne-blobs-remain-out-of-sync-after-undo/1038000)). [F]
   - Transport position versus PDC is unknown for Live 12 and Bitwig 6. [U]
   - Note segmentation quality: the state of the art is about 0.6 COnPOff ([QMUL 2021](https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2021/LiDemirelProutskovaDixon-NLP4MusA2021.pdf)). [F]
   - Live swallows keyboard shortcuts. [W]
   - Freedom to operate around note-object editing has not been reviewed. [U]

---

## 2. How other products do it

### 2.1 Matrix A: workflow and host integration

| Product (version, date) | Kind | How the audio gets in | ARA | Usable in Live 12 / Bitwig 6? | Notes |
|---|---|---|---|---|---|
| **Celemony Melodyne 5** (5.4.2; Melodyne 6 unannounced [U]) | Plug-in + standalone | Transfer by playback, or ARA 2 | ARA 2 in Pro Tools 2025.10, Logic 11.2.2 (Rosetta only on Apple silicon), Cubase 15.0.20, Studio Pro 8.0, REAPER 7.52 and others ([Celemony](https://services.celemony.com/en/melodyne/editions-and-technical-matters)) [F] | Yes, by transfer only. In Live, Arrangement View only [F] | During playback, "wherever it has material to play back, Melodyne's signal will replace that of the original track" ([Celemony](https://helpcenter.celemony.com/M5/doc/melodyneEssential5/en/M5tour_TransferARA?env=live)) [F]. In Live the user picks the transfer folder with "Set Path for Transfers" ([Melodyne in Live](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_DAW_Live?env=live)) [F] |
| **Antares Auto-Tune Pro 11, Graph Mode** (11 Apr 2024) | Plug-in | Track Pitch, or Track Pitch + Time, during playback; or ARA 2 | Pro X 10.1 (Jan 2023): Logic and Studio One ([Antares](https://www.antarestech.com/blog/auto-tune-pro-x-now-with-ara2-support-for-logic-pro-and-studio-one)) [F]. Cubase/Nuendo per retailer copy ([Vintage King](https://vintageking.com/antares-auto-tune-pro-11-electronic-delivery)) [W]. No Pro Tools ARA found [F] | Yes, by tracking | After tracking, "the audio you played back will load into Graph Mode" ([Antares blog](https://www.antarestech.com/blog/understanding-graph-mode-in-auto-tune-pro-x)) [F]. Pro 11 added R/V/F drag modifiers and a Zoom Outline bar ([SOS](https://www.soundonsound.com/reviews/antares-auto-tune-pro-11)) [F] |
| **Antares real-time line** (AutoTune 2026, 21 Oct 2025; AutoTune Advanced, article 24 Sep 2026) | Real-time plug-in | n/a | Not mentioned | Yes | No Graph Mode: "Pro 11 … remains the home of Graph Mode". Advanced Live mode is 0.84 ms at 44.1 kHz ([Production Expert](https://www.production-expert.com/production-expert-1/antares-autotune-advanced-new-pitch-correction-plugin-with-vibrato-engine); [SOS](https://www.soundonsound.com/news/antares-release-autotune-2026)) [F] |
| **Waves Tune** | Plug-in | Scans by playback | None found; Waves' ARA guide names only Sync Vx ([Waves](https://www.waves.com/what-is-ara-guide-to-audio-random-access)) [F] | Yes, by scan | Default settings and session-save behaviour [U] |
| **Synchro Arts RePitch 2** (6 Nov 2025) | Plug-in, Elements / Standard | ARA 2, or a real-time Capture version | ARA 2. Pro Tools needs 2024.06 or later. Logic only under Rosetta ([manual](https://www.synchroarts.com/manuals/RePitchV1/Manual/HTML/repitch-plug-in-versions-and-daws.html)) [F] | Capture version [I] | "A 30 second piece of audio takes 30 seconds to capture". "You cannot reposition the DAW's playback cursor from within the VST/AU plugin" [F]. Standard adds SmartAttack and multi-track ([SonicState](https://sonicstate.com/news/2025/11/07/synchro-arts-unveils-repitch-2)) [F] |
| **Synchro Arts Revoice Pro 5** (Dec 2023) | App + Link plug-ins | Link plug-ins capture the audio and return it, in ARA and non-ARA hosts | ARA 2 (Pro Tools 2024.6+; Logic on Intel or Rosetta) ([SonicState](https://sonicstate.com/news/2023/12/12/synchro-arts-release-revoice-pro-5/), [Production Expert](https://www.production-expert.com/production-expert-1/synchro-arts-revoice-pro-5-released)) [F] | Via the Link plug-ins | Separate-app model |
| **Cubase VariAudio 3** (Cubase 15, Nov 2025) | Native to the DAW | Segment analysis inside Cubase | Cubase has hosted ARA 2 since 10.0.30 [F] | No | No VariAudio changes found for 15 [U] |
| **Logic Flex Pitch** (Logic 12.4 current) | Native to the DAW | Native analysis | Logic has hosted ARA 2 since 10.4. Third-party ARA runs **only under Rosetta** on Apple silicon ([Apple 121464](https://support.apple.com/en-us/121464)) [F] | No | Since 12.2, Flex Pitch notes are editable in take folders on first open ([logicstudiotraining](https://logicstudiotraining.com/logic-pro-12-2-update/)) [F] |
| **Fender Studio Pro 8 / 8.1** (Studio One renamed 13 Jan 2026; 8.1 on 9 Jun 2026) | DAW | ARA; **new native "Vocal Tune" plug-in and "Pitch Curves on Audio Events"** in 8.1 ([Production Expert](https://www.production-expert.com/production-expert-1/fender-studio-pro-81-adds-studio-assistant-moises-integration-and-native-vocal-tuning)) [F] | ARA 2 host [F] | No | Includes a Melodyne Essential licence ([Macworld](https://www.macworld.com/article/3037910/fender-studio-pro-daw-review-powerful-updated-rebranded.html), [MusicTech](https://musictech.com/reviews/digital-audio-workstations/fender-studio-pro-8-1-review/)) [F] |
| **Avid Pro Tools** | DAW | Elastic Audio + ARA | ARA 2 since 2022.9; 13 ARA plug-ins by Jul 2025 ([Production Expert](https://www.production-expert.com/production-expert-1/ara-integration-in-pro-tools-full-list-of-supported-plug-ins-and-free-versions)) [F] | No | Melodyne Essential included with paid Pro Tools. Melodyne and Elastic Audio cannot share a track [F] |
| **Cockos REAPER 7** (7.82, 4 Oct 2026) | DAW: ReaTune + ARA host | ReaTune draws its graph during playback [W] | ARA 2 since 5.97 ([whatsnew](https://www.reaper.fm/whatsnew.txt)) [F] | No | CLAP host since 6.71 [F] |
| **Image-Line FL Studio, Newtone** | Native | Loads a sample | No ARA; staff said in Nov 2024 that it is "on the list" ([forum](https://forum.image-line.com/viewtopic.php?p=1979721)) [W] | No | |
| **Ableton Live 12, Auto Shift** (since 12.1; Live 12.4 May 2026) | Native real-time effect | n/a | None [F] | n/a | Input Pitch and Pitch Correction meters, no graph. In Live mode "slight glitches may occur during note onsets" ([manual](https://www.ableton.com/en/live-manual/12/live-audio-effect-reference/); [DJ Mag](https://djmag.com/tech/ableton-live-121-adds-real-time-pitch-correction)) [F] |
| **Bitwig Studio 6 / 6.1** (11 Mar / 26 Aug 2026) | No vocal pitch correction | n/a | None [F] | n/a | 6.1 rebuilt the Sampler with pitch analysis and added a Repitch mode and a Tuner device ([MusicTech](https://musictech.com/news/gear/bitwig-studio-6-1/)) [F] |
| **Tony** (QMUL) | Standalone annotator | File | n/a | n/a | pYIN + HMM note segmentation; alternative pitch tracks and an octave-shift fix ([project](https://code.soundsoftware.ac.uk/projects/tony)) [W] |
| **Praat ManipulationEditor** | Standalone | File | n/a | n/a | Sound + glottal PointProcess + PitchTier + DurationTier; overlap-add or LPC resynthesis ([Praat](https://www.fon.hum.uva.nl/praat/manual/Manipulation.html)) [F]. GPL, so study only [I] |
| **OpenUtau** (singing synthesis) | Standalone | n/a (synthesis) | n/a | n/a | A useful serialisation model, see §2.2 ([schema](https://www.schemastore.org/openutau-ustx.json)) [F] |

### 2.2 Matrix B: editing model

| Product | Edit objects | Per-note controls | Drawing | Starting from automatic | Fixing the analysis | Time editing | Undo and persistence |
|---|---|---|---|---|---|---|---|
| **Melodyne 5** | Notes ("blobs") | Centre pitch; pitch drift and pitch modulation, where 100 % = original, 0 % = straight line, −100 % = inverted ([Celemony](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_ToolModulationDrift_2)) [F]. Formant, amplitude, time tools [U, not re-checked] | No free drawing [U] | **Correct Pitch macro** ([Celemony](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_MacroPitch)) [F]:<br>- centre 0-100 % and drift amounts;<br>- leaves vibrato alone;<br>- at low settings moves only "wildly out of tune" notes;<br>- previews during playback;<br>- skips hand-tuned notes unless asked;<br>- a mixed selection shows min/max.<br>Quantize Time exists [F]; its exact semantics [U] | Note Assignment Mode has no audible effect. Changing the algorithm re-analyses and **loses all edits** [F] | Yes (needs the stored audio) | ARA: archive in the host project. Transfer: files in a folder [F]. Essential edition cannot edit drift or vibrato, and lacks sibilant handling and the Fade tool ([SOS 2020](https://www.soundonsound.com/reviews/celemony-melodyne-5)) [F] |
| **Auto-Tune Pro 11 Graph** | Lines, Curves, Notes (AT8 pp. 38-40) [F] | Per object: Retune Speed, Vibrato, Formant ([Antares](https://www.antarestech.com/blog/understanding-graph-mode-in-auto-tune-pro-x)) [F]. R/V/F drag modifiers [F] | Yes: lines and curves | **Make Curve; Make Notes** (uses Key/Scale); **Import Auto**, "a precise representation of the pitch correction that would result from … Automatic Mode" (AT8 p. 39) [F]. Automatic object creation after tracking is a preference [F] | Re-track. Regenerating notes "will … replace any correction objects that were previously in that range" (AT8 p. 40) [F]. Under ARA, Track Pitch + Time deletes corrections ([Antares help](https://help.antarestech.com/hc/en-us/articles/41108719199636-Why-do-all-Note-and-Time-Corrections-get-deleted-after-clicking-on-Track-Pitch-Time)) [W] | Track Pitch + Time: "must first create a copy of the audio" (AT8) [F] | Up to 20 undo levels, separate for pitch and time (AT8 p. 20) [F]. Track Pitch data is "always stored with the instances … in your session". Pitch + Time audio goes to separate files with a Data File Management dialog ([2018 guide](https://r2.gear4music.com/media/70/709346/download_709346.pdf)) [F]. The detected curve is red and the drawn curve green [F] |
| **RePitch** (v1 manual; v2 Nov 2025) | Notes | Pitch Correction 0-100 %; Level −10..+10 dB; a modulation box (drag down to flatten or invert); Tilt by Alt/Option-dragging an edge [F] | No [U] | Macro [F] | A macro change "will remove all manual editing" [F] | Yes: in Note Edit Mode the top half edits pitch and the bottom half edits time [F] | Double-click centres a note on the scale note (Snap on) or the chromatic note [F] |
| **Cubase VariAudio 3** | Segments | Quantize Pitch, Straighten Curve ([Steinberg](https://steinberg.help/nuendo/v12/en/cubase_nuendo/topics/sample_editor_variaudio/sample_editor_variaudio_section_r.html)) [W] | No | Quantize Pitch | Segment editing [W] | Warp segment start and end [W] | Native host undo and persistence |
| **Logic Flex Pitch** | Notes | Per-note handles ([Apple](https://support.apple.com/en-mz/guide/logicpro/lgcpc53e6bef/mac); handle list unconfirmed) [U] | No | Quantize [U] | n/a | Flex Time | Native |
| **Studio Pro 8.1 Pitch Curves** | Curves drawn on audio events [F] | [U] | Yes [F] | [U] | [U] | [U] | Native; storage format unknown [U] |
| **Tony** | Notes + pitch track | Annotation only | No | Automatic segmentation | Alternative pitch tracks, octave shift [W] | No | File |
| **OpenUtau** | Notes + pitch points | Pitch points: x in ticks from the note start, y in tenths of a semitone relative to the note, shape `l`/`i`/`o`/`io`. Each note has a vibrato object (length %, period ms, depth c, fade in and out ms) [F] | Yes | n/a | n/a | Yes | Project file |

### 2.3 What converges across products

1. **Every note editor uses the same split: centre + slow drift + fast modulation per note.** Melodyne, RePitch, Flex Pitch and VariAudio all do it [F for each product]. This matches `skato_control`'s centre/vibrato split. [I]
2. **There are two editing paradigms, note objects and drawn curves.**
   - Auto-Tune offers both [F]. Studio Pro 8.1 has just added drawn curves natively [F].
   - Reviewers find note editing more intuitive (SOS reviews, 2007 and 2023) [W].
   - Recommendation: notes by default, curves later. [I]
3. **Everyone starts from an automatic result**, then lets the user fix it: Import Auto, Make Notes, the Correct Pitch macro. [F] For us the analogue is the engine's own Studio decisions, run offline over the whole take and turned into notes. [I]
4. **Fixing the analysis is kept apart from editing.** Melodyne's Note Assignment Mode has no audible effect [F]. Where the two are mixed, edits get destroyed: the RePitch macro, the Melodyne algorithm switch and Auto-Tune ARA Track Pitch + Time all wipe them [F].
5. **Time editing forces the product to store audio** (Auto-Tune Pitch + Time, Melodyne Transfer). [F]
6. **Capture costs real time:** "30 seconds … takes 30 seconds" (RePitch). [F]

### 2.4 The market in 2026, and the gap Skatotune fills

- **Live 12 users have no graph editor at all.** Auto Shift is real-time only [F]. ARA requests on the Ableton forum run into Feb 2026 with no staff statement ([forum](https://forum.ableton.com/viewtopic.php?p=1836661)) [W].
- **Bitwig users have nothing for vocal tracks.** The 6.1 pitch analysis is in the Sampler [F]. A user reports that a Bitwig survey asked about interest in ARA ([VI-Control](https://vi-control.net/community/threads/bitwig-6-0-officially-released.170801/)) [W]. That is not a roadmap commitment.
- **Fender Studio Pro 8.1 now ships a native plug-in literally named "Vocal Tune".** [F] This is one more reason for the Skatotune rename. [I]
- **Antares keeps Graph Mode in Pro 11.** Its newer real-time products leave it out. [F]
- The gap for a graph editor is therefore widest in exactly our two hosts. [I]

---

## 3. What users value

### 3.1 What users love

- **Double-click to snap a note to pitch.**
  - MusicTech (2020) called it its favourite Melodyne 5 addition [F, per the prior-art pass].
  - RePitch does the same, centring on the scale or chromatic note [F].
- **Gentle macros that respect the performance.** Melodyne's Correct Pitch [F]:
  - moves only wildly out-of-tune notes at low settings;
  - leaves vibrato alone and skips hand-tuned notes;
  - previews live;
  - is undoable.
- **Sibilants left alone.** Melodyne leaves sibilants acoustically unaltered (Correct Pitch page) [F].
- **Editing during looped playback.** Auto-Tune 8 lets you edit while the loop plays, even under the playhead, and the edit takes effect on mouse release (AT8 pp. 4, 48) [F].
- **Note objects over drawing.** Reviewers find notes more intuitive for most work [W].

### 3.2 What users hate

| Complaint | Evidence |
|---|---|
| **Lost edits** | Steinberg forum, 3 Jun 2026: edits vanished. Steinberg forum, 9 Jun 2026: Delete Unused Media removes rendered "permanent" Melodyne files; users blame Nuendo 15 [F, threads exist]. An Auto-Tune Pro X "lost after reopen" report comes from an aggregator [U] |
| **Sync breaks when clips move** | Without ARA, moving or muting regions meant "the whole transfer process had to be repeated" ([Celemony, Logic](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_LogicARA_SpurenClips?env=logic)) [F]. Even under ARA: Cubase 13.0.51 + Melodyne 5.4.2 blobs go out of sync after undo, and Celemony reproduced it in Cubase 15 [F]. A Live user reports Melodyne playing "from an old position", unanswered ([forum](https://forum.ableton.com/viewtopic.php?p=1799221)) [F] |
| **Data stored outside the project** | Live does not tell plug-ins the project folder, so Melodyne users pick the transfer folder by hand [F]. Auto-Tune Pitch + Time keeps audio "elsewhere on your computer" [F] |
| **Undo split between host and plug-in** | Melodyne 4 tutorial [W]. ARA does not merge the two histories either: host-integrated plug-in undo is postponed beyond SDK 2.3 ([ARA ChangeLog](https://github.com/Celemony/ARA_API/blob/main/ChangeLog.txt)) [F] |
| **Tempo changes break capture** | Melodyne in Live needs tempo set to "Constant", and tempo automation can break sync or end a transfer early [F] |
| **Destructive re-runs** | RePitch macro change, Melodyne algorithm change, Auto-Tune ARA Track Pitch + Time [F] |
| **Session View and clip loops** | Melodyne plays "in the wrong places or not at all" in Session View, and is "not informed about loops hidden within clips" [F] |
| **Artifacts and detection limits** | Weaker on "Th" sounds (MusicTech 2020) [F]. Sibilants cannot be reclassified (SOS 2020) [F] |

### 3.3 Requirements this implies for Skatotune [I]

| # | Requirement |
|---|---|
| R1 | **Edits live in the plug-in state** and are saved with the host project. Every edit marks the state dirty. No external files in pitch-only mode. ARA 2.3 added a "document data changed" notification for exactly this reason. |
| R2 | **Nothing is destructive.** Edits are keyed to timeline spans, not note IDs, so re-analysis and re-segmentation keep them. Re-tracking a region touches only that region, and any discard is confirmed first. |
| R3 | **Stale data is detected and shown**, not silently mis-applied. Offer to re-track the region. |
| R4 | **One visible undo history** with on-screen Undo and Redo. Use the CLAP host-undo extension where a host supports it. |
| R5 | **Start from automatic.** An "Import Auto" note map from our own control law, plus a gentle Correct Pitch macro. |
| R6 | **Fast primitives:** double-click snap, a drag that snaps to semitone or scale, and a fine mode. |
| R7 | **Live preview** during looped playback. Edits apply on mouse release. |
| R8 | **Honest scope in Live:** Arrangement View and constant tempo for the MVP. |
| R9 | **Unvoiced and sibilant frames are never shifted.** The detector's voicing/sibilance FSM already does this. |

---

## 4. Integration options

### 4.1 The routes

| Route | How it works | Hosts | When clips move | Persistence | Time editing | Our effort | Main risk |
|---|---|---|---|---|---|---|---|
| **1. Pitch-only capture** (Auto-Tune 8 "Track Pitch") | An ordinary insert records the take while the host plays, keyed to transport position. It analyses after stop and corrects the host's live audio on playback | Every host that reports valid transport | Data goes stale; must detect and re-track | Plug-in state, about 0.2-1 MB per 3-min take [I] | No | Low-medium | Transport/PDC accuracy per host |
| **2. Audio capture** (Melodyne Transfer, RePitch Capture, Auto-Tune Pitch + Time) | The plug-in keeps a copy of the audio and plays its own render in place of the input wherever it has material | Every host | Data goes stale; re-transfer | External files, about 17-35 MB per 3-min mono take; user picks the folder in Live [F/I] | Yes | Medium-high | Lost or orphaned files; storage |
| **3. ARA 2** | The host gives random access to the audio plus musical context and stores the plug-in's archive in its project | Cubase/Nuendo, Studio Pro, REAPER (VST3-ARA); Logic (AU-ARA, under Rosetta); Pro Tools (AAX-ARA). **Not Live or Bitwig** | Handled by the host (bugs still exist) | Host project | Yes | High | Immature Rust bindings; nice-plug fork; AU and AAX needed |
| **4. Offline import** (standalone editor) | The user drops a WAV into Skatotune Standalone, edits it, renders a corrected WAV and drags it back | Any host, at file level | n/a: a destructive bounce | The editor's own project file | Possible | Low-medium; no host sync; can use full-file marks | Workflow friction; no live link to the session |
| **5. Host MIDI as the editor** | Export the detected notes as a MIDI clip. The user edits it in the DAW piano roll, and the existing MIDI Target mode follows it | Hosts that route MIDI to an audio effect. Bitwig: likely [I]. Live VST3 audio effect: untested (`docs/research.md` risk 13) | MIDI and audio clips move separately | Host project and host undo, for free | No | Very low: MIDI Target, Learn and Auto exist (`docs/STATUS.md` §1) [F] | Notes only, with no drift or modulation control; routing in Live |
| **6. Max for Live companion** | The LOM `Clip` object exposes `file_path` and `warp_markers` for audio clips ([LOM](https://docs.cycling74.com/apiref/lom/clip/)) [F]. A companion device could hand the file to the plug-in for fast offline analysis | Live Suite / Max for Live only | Warp markers could re-map positions [I] | Plug-in state | Possible | Medium | Needs Max; plug-in-to-M4L IPC; Live only |

No file-path access to audio events was found in Bitwig's controller API. [U]

### 4.2 Host support for our target DAWs

| Host (current version) | Skatotune format today | ARA host? (format) | CLAP host? | Native pitch editing | Route 1 (capture) | Route 3 (ARA) cost for us |
|---|---|---|---|---|---|---|
| **Bitwig Studio 6.1** (26 Aug 2026) | CLAP + VST3 [F] | **No** [F] | Yes [F] | None for vocals; Sampler Repitch and Tuner [F] | **Primary target.** nice-plug wiki: Bitwig 6.0.6 gives `pos_samples` over VST3 only and `pos_seconds`/`bar_number` over CLAP only; the getters derive the missing fields ([wiki](https://codeberg.org/RustAudio/nice-plug/wiki/Host-behaviour)) [F] | n/a |
| **Ableton Live 12** (12.4.6, 15 Sep 2026) | VST3 [F] | **No** [F] | **No** (no "CLAP" in the 12.0-12.4.6 notes) [F] | Auto Shift: real-time, meters only [F] | **Primary target.** Live 12.4.1 gives no `pos_seconds`/`bar_number` (same wiki) [F]. Devices that read song position after latency-causing devices can be off-grid ([Ableton FAQ](https://help.ableton.com/hc/en-us/articles/209072409-Delay-Compensation-FAQ), via excerpts) [F]. Arrangement only [I] | n/a |
| **Logic Pro 12** (12.4) | **None**: AU is a backlog item [F] | Yes, AU only. Third-party ARA needs Rosetta on Apple silicon; no 12.x note announces native ARA ([release notes](https://support.apple.com/en-us/109503); [Apple 121464](https://support.apple.com/en-us/121464)) [F] | No [F] | Flex Pitch [F] | Needs an AU wrapper first | AU wrapper + ARA + an x86_64 slice + users running Rosetta. Watch the ARA 3 AUv3/IPC draft [I] |
| **Cubase 15 / Nuendo 14** | VST3 (not tested) | Yes, VST3-ARA since 10.0.30 [F] | No: no announcement found ([SonicState](https://sonicstate.com/news/2025/11/06/steinberg-releases-cubase-15/); [forum](https://forums.steinberg.net/t/cubase-will-be-left-behind-without-clap-auv3/903860?page=2)) [W] | VariAudio 3 [F] | Should work [I] | VST3-ARA through a CLAP-ARA build wrapped by clap-wrapper(-rs), or our own VST3 interfaces [I] |
| **Fender Studio Pro 8.1** | VST3 + CLAP (not tested) | Yes, VST3-ARA (ARA 2 since Studio One 4) [F] | Yes, since Studio One 7 [W] | Melodyne Essential bundled, native Skatotune plug-in, Pitch Curves [F] | Should work [I] | VST3-ARA as for Cubase. A candidate for native CLAP-ARA, unconfirmed [U] |
| **REAPER 7.82** (4 Oct 2026) | VST3 + CLAP (not tested) | Yes, VST3-ARA since 5.97; ARA fixes in 7.78-7.79 [F] | Yes, since 6.71 [F] | ReaTune [W] | Should work [I] | The cheapest ARA test host. A candidate for CLAP-ARA, unconfirmed [U] |
| **Pro Tools 2025.10** | **None**: needs AAX [F] | Yes, AAX-ARA since 2022.9. Avid adds ARA partners in named batches, e.g. six plug-ins in 2024.6 ([Avid PR](https://stg.com/news/avid-pro-tools-delivers-new-ara-support-for-industry-leading-plugins/)) [F] | No [F] | Elastic Audio; Melodyne Essential bundled [F] | Needs AAX first | AAX SDK, signing, and possibly Avid partnership/enablement [U] |

### 4.3 ARA in practice

**What ARA gives a plug-in** (checked against the SDK 2.3 headers and documentation) [F]:
- random access to audio from non-realtime threads;
- a host-owned model of AudioSource, AudioModification and PlaybackRegion;
- musical context from the host: tempo, bars, keys, chords;
- persistence inside the host project;
- a playback controller, so the plug-in can start, stop and locate the transport.

**What ARA does not give** [F unless marked]:
- **No live input.** The playback renderer ignores realtime input and should report no latency. Optional side-chains and realtime input are postponed. ARA mode would therefore be a separate offline renderer next to Live mode. [I]
- **It must be first in the chain**, and only one ARA plug-in fits per chain (SDK PDF §2.8; RePitch manual).
- **No continuous pitch-curve content type.** The plug-in can publish notes, tempo, keys and chords back to the host, but not a curve.
- **No merged plug-in + host undo.** Postponed.
- **No AUv3** in 2.3. It is drafted for **ARA 3.0** on `ARA_API` `develop` together with IPC, which is marked "PRELIMINARY - DO NOT USE FOR SHIPPING PRODUCTS". That branch also drops the CLAP draft compatibility IDs. ARA 3 is the most likely route to native ARA in Logic, and possibly in sandboxed hosts. [I]

**SDK status.**
- ARA SDK 2.3 (released 2025-11-06) makes CLAP and AAX fully supported companion APIs and adds the document-data-changed notification ([ChangeLog](https://github.com/Celemony/ARA_API/blob/main/ChangeLog.txt)). [F]
- The SDK has been Apache-2.0 since 2021. [F]
- No shipping host was found that loads CLAP-ARA directly. [U]

### 4.4 Recommendation [I]

| Priority | Route | Role |
|---|---|---|
| MVP | **Route 1: pitch-only capture** | Works in Bitwig and Live today |
| Dogfood | **Headless analysis, optionally Route 4** | Prove the analysis and the editing model without host-sync risk |
| Optional stopgap | **Route 5: Export notes as MIDI** | Near-free if MIDI Target works in Live (needs a routing test) |
| Later | **Route 2** | For time editing and an offline HQ render |
| Optional accelerator | **Route 6** | Faster capture for Live Suite users |
| Last | **Route 3: ARA** | Only after AU exists, and only if Cubase, Studio Pro or REAPER users become a target |

---

## 5. Technical architecture options and what we can reuse

### 5.1 Playback architectures

| | **A. Pitch-only capture + realtime render** | **B. Audio capture + offline render** | **C. ARA 2** | **D. A + optional render cache** | **E. Standalone offline editor** |
|---|---|---|---|---|---|
| What is stored | Pitch curve, notes, edits | A, plus the audio (and the render) | Edits; the host keeps the audio | A, plus a cache | The whole project in the editor |
| Reported latency D | A fixed Studio-type D | Could be 0 for our own playback [I] | 0 possible [I] | A's D | n/a |
| When edits are heard | At once, during looped playback (the AT8 model) [F] | After a background re-render | At once | At once; HQ follows later | After render |
| Host edits after capture | Data goes stale; detect and re-track | Re-transfer, as Melodyne needed before ARA [F] | The host handles it | As A | n/a |
| Quality ceiling | Studio PSOLA with causal marks | Non-causal, full-file marks: "every PSOLA win in the literature used offline PSOLA with full-file marks" (`docs/research.md` §11.1 risk 1) [F] | As B | B wherever the cache is valid | As B |
| Fits Live 12 / Bitwig 6 | Yes | Yes | **No** | Yes | Yes, at file level |
| Effort | Lowest | Medium-high | Highest | A, then B | Low-medium |

### 5.2 Recommended MVP shape (A, with live re-detection) [I]

```
host audio --> process() --> Engine (fixed Studio D) --> output
                 |                     ^
   pos_samples,  |                     |  TargetSource::Timeline (new): target cents
   playing, loop |                     |  at this timeline position
                 v                     |
       capture ring (rtrb, prealloc)   compiled TargetTimeline snapshot
                 |                     ^  (single-slot rtrb handoff, swap.rs pattern)
                 v                     |
       drain thread --> TakeStore (RAM, audio kept only until analysed)
                              |
                              v
       analysis thread: Studio-HQ detector offline + reference.rs refinement
                        + whole-take note HMM + centre/drift/modulation split
                              |
                              v
               NoteMap + Edits  <---->  GUI graph editor (undo stack)
                              |
                              v
               #[persist = "graph"]: versioned compact binary, base64 String
```

**Key choices:**

- **Analyse offline, store only pitch.** The audio is held in RAM only until the take has been analysed, about 35 MB per 3-min take as f32. Offline analysis can be fully non-causal, so the note map starts out better than the live decisions. [I]
- **Where the input pitch comes from at playback.** There are two options:
  - (i) the stored curve, read at the exact sample;
  - (ii) the live Studio detector, with the stored curve used only for display and alignment checks.

  Auto-Tune 8 appears to do (ii): its Tracking setting applies "during both the Track Pitch function and while correcting pitch" (p. 37) [F]. Start with (ii), because it degrades gracefully when the capture is slightly misaligned under PDC. [I]
- **The targets come from the timeline.** Pre-computed targets remove the need for the note-decision lookahead (`lookahead_hops`, DESIGN §3.5), which is the error that dominates real singing (STATUS §6). [I]
- **Handoff pattern.** Use the same pattern as `crates/skatotune/src/swap.rs`: a single-slot `rtrb::RingBuffer::new(1)` for incoming snapshots and a 4-slot "retired" ring freed off the audio thread [F]. Hook the timeline source in where `Engine::hop_boundary` calls `corrector.process_hop` (`crates/skato_engine/src/lib.rs`, around lines 811 and 886) [F].
- **Threads.**
  - **Audio:** capture push and snapshot lookup only, using `rtrb`.
  - **Drain:** one thread.
  - **Analysis:** its own thread, so it never blocks nice-plug's single background executor, which builds engines.
  - **GUI:** edits the model and publishes snapshots.

  This follows DESIGN §2.1. [I]

### 5.3 Data model sketch [I]

```
Take    { id, sample_rate, hop, start: TimelinePos { samples: i64, beats: f64 }, len,
          analysis_version, fingerprint, curve: [(cents, confidence, voiced)] }
Note    { id: u32 (stable), take, start, end, centre_c, drift_ref, mod_ref,
          vib_rate, vib_extent }
Edit    { span: [start, end) in timeline samples (not bound to a note id),
          kind: PitchOffset | DriftScale | ModScale | Retune | Transition
                | Formant | Curve | Line }
Project { schema_version, takes, notes, edits, scale, granularity }
```

**Design rules for the model:**
- **Edits are keyed to spans, not note IDs**, so re-segmentation keeps them (R2).
- **Record `pos_beats` next to samples.** AT8 needs a re-track after a time-signature change, but only for its bars-and-beats display (p. 36) [F]. Live warps clips when the tempo changes, so the right index (samples or beats) needs a host experiment. [U]
- **Use OpenUtau's representations for curves and vibrato:** pitch points relative to the note, and a per-note vibrato object. [I]
- **Shape it like ARA** (AudioSource, PlaybackRegion, edits in source time), so a later ARA tier maps onto it. [I]

### 5.4 Sizes and persistence

**Sizes for a 3-minute take at 48 kHz** (arithmetic checked) [F]:

| Item | Size |
|---|---|
| Pitch curve at hop 64 (750 frames/s × 180 s × 5 B) | 675 KB |
| Pitch curve at 5 ms frames | 180 KB |
| Captured audio, f32 | 34.56 MB |
| Captured audio, 16-bit | 17.28 MB |

**Persistence:**
- nice-plug state is JSON. Each `#[persist]` field goes through `serde_json::to_string`, so it is a JSON string inside the JSON state (`nice-plug-core-0.4.2/src/params/persist.rs`) [F].
- A raw `Vec<u8>` would serialise as an array of numbers. Use a **base64 `String`** of a versioned binary instead (+33 %). [I]
- Precedent in our code: `#[persist = "learned-scale"]` at `crates/skatotune/src/params.rs:973` [F].
- nice-plug's `zstd` state compression is a feature that our build does not enable. [F]
- **How big a state can hosts take?** The JUCE forum gives anecdotes only: "5-10mb should not cause issues", and about 100 MB took minutes to load in MainStage ([JUCE](https://forum.juce.com/t/theoretical-and-practical-size-limits-for-plugin-states/31349)) [W].
- Budget 1-2 MB per instance and test at 10×. [I]
- How often Live and Bitwig call `getState` (for undo and autosave) is unknown. [U]

### 5.5 Timeline sync and staleness [I unless cited]

- **What nice-plug 0.4.2's `Transport` offers.** Every field is `Option`al by host:
  - `playing`, `recording`, `preroll_active`, `tempo`;
  - `pos_samples`, `pos_seconds`, `pos_beats`, bar start and number;
  - `loop_range_*`.

  It deliberately does not expose VST3's loop-ignoring continuous time (`nice-plug-core-0.4.2/src/context/process.rs`) [F].
- **Our plug-in reads neither the transport nor `ProcessMode` today.** [F]
- **Loops and jumps** have to be inferred from discontinuities in `pos_samples` together with `loop_range_samples`.
- **Alignment and staleness check.** Continuously compare the live detector's pitch with the stored curve over a few hundred milliseconds. If they disagree, mark the region **stale**: show an amber band, keep its edits, and offer to re-track.
- **Re-anchoring.** Pitch-contour cross-correlation can re-anchor a take that has moved by a small offset.
- **PDC calibration per host** is the first spike. Insert a plug-in with known latency upstream and measure where `pos_samples` lands relative to the input in Live 12 (VST3) and Bitwig 6 (CLAP).
  - JUCE threads say REAPER and Tracktion compensate, and call anything else a host bug ([JUCE](https://forum.juce.com/t/ppq-position-and-latency-compensation/57284)) [W].
  - No current evidence exists for Live or Bitwig. [U]
- **Scope for the MVP:** Arrangement View in Live, constant tempo, and a warning if tempo automation is detected. Celemony makes the same restrictions. [F for Celemony]

### 5.6 Latency in graph mode [I]

- **Do not change D on entering graph mode.** A CLAP latency change goes through `request_restart` (the host re-activates the plug-in), and nice-plug 0.4.2 adds a second restart (HI-8, STATUS §6) [F].
- **Recommended: graph mode runs in Studio latency (20-50 ms, fixed).** The DAW compensates during playback, and editing a recorded take is not live monitoring.
- **A lower-D variant is possible.** With stored targets there is no note-decision lookahead, so a dedicated graph latency could sit at the Studio shifter minimum. That minimum is `T_max + d_mark(T_max) + h_interp + margin` (`crates/skato_shift/src/latency.rs`), and it binds only for Low Male and Wide below 23 ms. This is an open question.

### 5.7 What we can reuse

| Need | Existing code | Notes |
|---|---|---|
| Offline robust pitch | `skato_detect` (Studio-HQ path) | Matches the reference tracker's RPA50 (94.7 %) with a quarter of its octave errors (0.31 % against 1.30 %) [F]. Measured at 16-26× real time on one M4 Max core: a 3-min take takes about 7-12 s [F/I] |
| Fine refinement | `skato_eval/src/reference.rs` | Offline, f64, centred NSDF |
| Note segmentation | `skato_control/src/note_hmm.rs` | Fixed-lag Viterbi today; extend it to a whole-take Viterbi with a granularity control [I] |
| Vibrato and drift split | `skato_control/src/vibrato.rs`; the centre/vibrato split | Maps onto Melodyne's drift % and modulation % [I] |
| Realtime render | `skato_shift`, `skato_engine` | Add `TargetSource::Timeline` |
| Latency formula | `skato_shift/src/latency.rs` | |
| Snapshot handoff | `skatotune/src/swap.rs` | rtrb, retired ring |
| Persistence | `params.rs` `#[persist]` | |
| Pitch-roll drawing | `skatotune/src/editor/{graph,history,theme,keyboard}.rs` | `GraphView` already zooms vertically (`history.rs`). The GUI history ring is `HISTORY_CAPACITY = 16_384` frames, about 21.8 s at 48 kHz and keyed to output time, so it **cannot** hold a take; a separate store is needed [F] |
| Evaluation | `skato_eval` vocadito loader with two note annotations (`notes_a1`, `notes_a2`) | Gives segmentation metrics and an inter-annotator ceiling |
| MIDI targets | MIDI Target / Learn / Auto | For Route 5 |
| Bounce quality | DESIGN §3.6 *Max quality on render* (default Off) | Specified but not yet wired, because `ProcessMode` is not read [F] |

---

## 6. Rust and framework feasibility

### 6.1 nice-plug 0.4.2 (read in the local registry)

**Present** [F]:
- `Transport`, as described in §5.5.
- `ProcessMode::Offline`, mapped for both VST3 and CLAP.
- `#[persist]` fields.
- JSON state with optional zstd.
- Track info for CLAP and VST3.
- A background executor.
- `nice-plug-egui` 0.5.1 `ResizableWindow`.

**Missing** [F]:
- **ARA.** A grep finds no ARA code. The CLAP `get_factory` matches only `CLAP_PLUGIN_FACTORY_ID` (`src/wrapper/clap.rs:165`). The CLAP extension chain and the VST3 interface tuple are fixed.
- **Wrapper support for some CLAP extensions:** `undo`, `transport-control`, `resource-directory` and `state-context`.
- **Other formats:** AU and AAX.
- **Loop-ignoring continuous time.**
- **The HI-8 double restart** is a known bug.

**Help that already exists** [F]:
- nice-plug depends on `clap-sys` 0.5.0, which already has bindings for `state_context`, `draft/undo`, `draft/transport_control` and `draft/resource_directory`.
- Exposing them is wrapper work in a `[patch.crates-io]` fork, not new FFI. [I]

### 6.2 egui

- **egui's built-in `Scene` does not fit.** In egui 0.36.2, `Scene` zooms through `TSTransform { scaling: f32, … }`, a single uniform scale. Time and pitch would zoom together, and strokes and text would scale with them [F].
  - A piano roll needs independent time and pitch zoom with constant-width strokes.
  - So extend our own `GraphView` transform, which already zooms pitch only, and the column folding in `graph.rs`. [I]
- **The window is fixed at 1100×720** (`editor/mod.rs:60-61`) [F]. Graph mode wants a resizable window (`ResizableWindow`) or a mode that gives the graph the whole window. [I]
- **egui's `Undoer` is generic** (100 undos, 1 s stable time) [F]. A domain undo stack of edit commands fits better. [I]
- **Keyboard input is unreliable inside hosts.**
  - Hosts intercept keystrokes. REAPER and Ableton have opt-in settings to send all keys to the plug-in, which then blocks host shortcuts ([nih-plug #219](https://github.com/robbert-vdh/nih-plug/issues/219)) [F].
  - Live 10 on Mac took Ctrl-Z, while Bitwig passes it to the plug-in ([VCV forum](https://community.vcvrack.com/t/ctrl-z-issues-in-vst-ableton/15185)) [W].
  - Design mouse-first, with on-screen Undo and Redo. [I]

### 6.3 CLAP extensions that matter (Bitwig)

| Extension | Status | What it would give us |
|---|---|---|
| `clap.state-context/2` | **Stable since CLAP 1.2.0**; contexts are preset, duplicate and project ([header](https://raw.githubusercontent.com/free-audio/clap/main/include/clap/ext/state-context.h)) [F] | Keep captures out of presets; decide what a duplicated instance should copy [I] |
| `clap.undo/4` | Draft. The host keeps one undo history; the plug-in reports changes and can supply deltas ([header](https://raw.githubusercontent.com/free-audio/clap/main/include/clap/ext/draft/undo.h)) [F] | Ctrl-Z in Bitwig does the right thing (R4) [I] |
| transport-control | Draft. The plug-in can ask the host to start, stop or seek ([ChangeLog](https://raw.githubusercontent.com/free-audio/clap/main/ChangeLog.md)) [F] | "Click a note to jump there", which RePitch says VST/AU plug-ins cannot do [I] |
| `clap.resource-directory/1` | Draft. A per-instance directory that is duplicated with the instance; paths must be stored relative ([header](https://raw.githubusercontent.com/free-audio/clap/main/include/clap/ext/draft/resource-directory.h)) [F] | Audio files for Route 2 that live with the project [I] |

Whether Bitwig 6 implements any of the three draft extensions is **unknown**. It could not be checked in this pass because the web-search budget ran out. [U]

### 6.4 ARA from Rust

| Piece | State |
|---|---|
| ARA SDK | Apache-2.0. Use tag `releases/2.3.0`; `develop` is the preliminary ARA 3.0 [F] |
| `ara2-bridge` | Version 0.3.0. The repo was created 2026-05-06. The 0.3.0 publish date conflicts: 2026-07-16 per the crates.io API, 27 Sep 2026 per a docs.rs check [U]. MIT OR Apache-2.0; 3 stars. It has plug-in and host runtimes with CLAP, VST3 and AUv2 companions, and claims interop tests against Celemony's C++ TestHost. It shows **no real-DAW evidence**, no integration with nice-plug or clack, and AAX and AUv3 are out of scope. The repo owner is ambiguous (eas4ai vs entrepeneur4lyf) ([GitHub](https://github.com/eas4ai/ara2-bridge), [docs.rs](https://docs.rs/ara2-bridge)) [F]. **Use it as reference material, not as a dependency yet** [I] |
| clap-wrapper ≥ 0.10.0 (current 0.16.0) | Turns a CLAP-ARA plug-in into **VST3-ARA only**. At HEAD, the AAX, AUv2 and AUv3 wrappers contain no ARA code. It can keep our existing VST3 class ID through `clap.plugin-factory-info-as-vst3` ([PR #255](https://github.com/free-audio/clap-wrapper/pull/255); [CLAP news](https://cleveraudio.org/news/july-2024/)) [F] |
| clap-wrapper-rs | The published crate 0.3.1 vendors clap-wrapper v0.14.0; git `main` vendors v0.16.0, unreleased ([GitHub](https://github.com/blepfx/clap-wrapper-rs)) [F] |
| clack 0.2.0 | Generic factory and extension traits; no ARA [F] |
| JUCE | Has had ARA since JUCE 7, with `ARAPluginDemo`. Useful as reference behaviour [F] |

**Smallest ARA path [I]:**
1. Fork nice-plug's CLAP factory and extension list to expose `org.ara-audio.ara.factory/2` and `org.ara-audio.ara.pluginextension/2`.
2. Generate bindings with bindgen against the 2.3.0 headers, or vet `ara2-bridge`.
3. Implement a DocumentController in Rust.
4. Wrap with clap-wrapper-rs to get VST3-ARA.
5. Validate with Celemony's TestHost (`-clap`, `-vst3`).
6. Test in REAPER, Studio Pro and Cubase.
7. Separately: an AU wrapper plus Rosetta for Logic, and AAX plus Avid enablement for Pro Tools.

### 6.5 What would be built, and what bound or patched

| Item | Build / bind / patch | Size [I] |
|---|---|---|
| Transport plumbing and a per-host quirk layer | Build (plug-in) | S |
| Capture ring, drain thread, TakeStore | Build | S-M |
| `vt_analysis` crate (offline detector, whole-take HMM, decomposition) | Build on existing crates | M |
| `TargetSource::Timeline` in `skato_engine` + compiled snapshot | Build | M |
| Graph editor canvas with a 2-axis transform, tools, inspector | Build (egui) | L |
| Domain undo stack | Build | S |
| Versioned binary state + migration | Build | S |
| Staleness detection and re-anchoring | Build | M |
| CLAP `undo` / `transport-control` / `state-context` | Patch nice-plug wrapper (bindings exist in clap-sys) | S-M each |
| Resizable window | Use `nice-plug-egui` `ResizableWindow` | S |
| Offline HQ render (full-file marks) | Build in `skato_shift` | M-L |
| ARA | Bind the SDK + fork the wrapper + clap-wrapper-rs | XL |
| AU / AAX wrappers | Bind/wrap (clap-wrapper-rs exports AUv2; AAX not covered for ARA) | L each |

**Licences.**
- Skatotune is ISC. The ARA SDK (Apache-2.0) and the VST3 SDK (MIT, [vst3sdk](https://github.com/steinbergmedia/vst3sdk)) are compatible, given NOTICE entries. [F/I]
- Praat is GPL: study it, do not copy it. [I]

**IP.**
- `docs/research.md` records Celemony US 8,022,286 (polyphonic DNA) as active until June 2029, with the advice "Do not copy note-object resynthesis". [F, local doc]
- No search has been made for patents on monophonic note-object editing or on Antares Graph Mode. [U]
- A freedom-to-operate review is needed before release. [I]

---

## 7. UX sketch of an MVP

### 7.1 Principles

These come from §3.3: mouse-first; non-destructive; one visible undo; start from automatic; edit during looped playback; show staleness honestly; Arrangement View and constant tempo in Live. [I]

### 7.2 Wireframe [I]

```
+-- Skatotune --- [ Auto | GRAPH ] --- Studio 40 ms --- HQ --- C major ------ CPU 2.1% --+
| [o TRACK] Captured 0:12.4 - 2:48.0   status: IN SYNC          [Undo] [Redo]   12 edits |
| Tools: [Edit notes] [Split] [Fix detection]   Snap: [Scale v]   Layers: [In][Out][Notes] |
+-----+---------------------------------------------------------------+------------------+
| D5  |                                                               | NOTE  A4         |
| C#5 |            __                                                 | heard  -23 c     |
| C5  |   ======  /  \___======        ~~~~~~~~~~~~                   | Pitch   [ +0 c ] |
| B4  |  (note)  (input curve, thin)   (vibrato kept)                 | Drift   [ 100 %] |
| A4  |                         =========  <- note, dragged +23 c     | Modul.  [ 100 %] |
| ... |   ////// amber band: STALE (audio changed since tracking)     | Retune  [ 20 ms] |
|     |                                                               | Trans.  [ 40 ms] |
+-----+---------------------------------------------------------------+------------------+
| 1.1      2.1      3.1      4.1      (bars | m:s)        > playhead follows the host   |
| Correct Pitch:  Centre [=======   70 %]  Drift [===   30 %]  [ ] include hand-tuned   |
+---------------------------------------------------------------------------------------+
```

**Layers:**
- **Input:** the existing thin trace, with alpha by confidence.
- **Output:** a thick accent trace.
- **Notes:** blobs on the scale lanes, which `graph.rs` and `theme.rs` already draw.
- **Waveform:** optional. A cheap RMS envelope could be stored with the curve. [I]

### 7.3 States and workflow [I]

1. **Empty.** Graph mode is selected but nothing has been captured. Auto correction stays active, or the region passes through (open question).
2. **Tracking.** The user arms **TRACK** and plays the passage in the host. Capture follows the transport and joins up multiple passes.
3. **Analysing.** A progress bar. A 3-min take takes about 7-12 s on one core.
4. **Ready.** Notes appear, pre-filled with an "Import Auto" result: the offline decisions of our control law. Edits are heard on the next pass, or at once on mouse release during looped playback.
5. **Stale.** The live pitch disagrees with the captured curve. The region gets an amber band, its edits are kept, and **Re-track region** is offered.

### 7.4 MVP tools [I]

| Tool | Behaviour | Source pattern |
|---|---|---|
| Select / drag note | Vertical drag snaps to scale or semitone; a modifier gives free cents | Melodyne, RePitch [W/F] |
| Double-click | Snap the note to the nearest scale note | RePitch [F], Melodyne [F] |
| Inspector | Pitch offset; drift % and modulation % (100 original, 0 straight, −100 inverted); retune ms; transition ms | Melodyne [F], Auto-Tune per-object retune [F] |
| Split / merge | Alt-click splits; merge combines the selection | All editors |
| Correct Pitch macro | Centre % and drift %; leaves vibrato alone; skips hand-tuned notes unless ticked; previews live | Melodyne [F] |
| Fix detection | Shift a segment by an octave, mark voiced or unvoiced, move a note boundary. No audible effect on its own; never discards edits | Melodyne Note Assignment [F], Tony [W] |
| Undo / Redo | On-screen buttons; the domain stack | §6.2 |

### 7.5 Deliberately out of the MVP [I]

- Time editing.
- Per-note formant.
- Free-drawn lines and curves.
- Multi-track or multi-instance views.
- ARA.
- Session View clips.
- Tempo-varying projects.
- Click-to-seek (unless Bitwig supports CLAP `transport-control`).
- Polyphony.

---

## 8. Phased roadmap

Effort is in engineer-weeks for one developer who knows the codebase. All estimates are [I], and agent-assisted velocity is not modelled.

| Phase | Scope | Exit gate | Effort | Main risk |
|---|---|---|---|---|
| **0. Host spikes** | (a) `Transport` accuracy in Live 12 (VST3) and Bitwig 6 (CLAP and VST3): loops, pre-roll, PDC with an upstream plug-in of known latency, tempo change, offline bounce or freeze. (b) `getState` frequency and tolerance at 1, 5 and 20 MB. (c) Whether Bitwig implements CLAP `undo`, `transport-control` and `resource-directory`. (d) Whether MIDI reaches a VST3 audio effect in Live (Route 5) | A written host-behaviour table; go / no-go for Route 1 | 1-2 | Hosts misreport position under PDC |
| **1. Headless analysis** | A `vt_analysis` crate. A `skato_eval notes` command that reports COnPOff, COnP and COn against vocadito A1 and A2, plus A1-versus-A2 agreement. **A/B render:** vocadito with annotated (then analysed) notes as timeline targets through the real engine | COnP near 0.75 (QMUL level) and COnPOff at or above 0.52 (Tony level). Rendered p95 approaches the 10.8 c forced-note figure (STATUS §6) | 2-3 | Segmentation quality |
| **1b. (optional) Standalone file editor** | A file-in/file-out editor in the standalone app, sharing the canvas with Phase 2 | Dogfood the editing model | +3-4, mostly shared with Phase 2 | Scope creep |
| **2. Capture MVP in the plug-in** | Transport plumbing; capture ring and drain; `TargetSource::Timeline`; snapshot handoff; 2-axis canvas; note tools, inspector, macro and detection fix; undo; `#[persist = "graph"]`; staleness detection; fake-transport and zero-allocation tests; a real-DAW pass | A 3-min vocal is tracked, edited, saved and reopened in Bitwig and Live, with edits intact and aligned within one hop. Zero allocations on the audio thread. Validators pass | 6-9 | Sync and PDC; keyboard; state size |
| **3. Editing depth** | Curves and lines; Import Auto variants; per-note formant; multiple takes; CLAP undo if Bitwig supports it | User testing | 3-4 | UI complexity |
| **4. Audio capture, HQ render, time editing** | Disk capture to a user-chosen folder (or CLAP resource-directory); an offline full-file-mark PSOLA render cache; then time editing | No lost files across save, reload and move. HQ render beats live Studio on the vocadito listening set | 4-6 (+4-8 for time editing) | Lost or orphaned files; storage |
| **5. ARA** | Only after AU exists, and only if Cubase, Studio Pro or REAPER users matter | VST3-ARA works in REAPER, Studio Pro and Cubase | 8-12+ | Immature bindings; forks of nice-plug and clap-wrapper; Logic needs Rosetta; Pro Tools needs AAX |

**MVP = Phases 0 + 1 + 2, about 9-14 weeks.**

### Risk register [I]

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Captured data goes stale after host edits | High | Wrong notes are applied | Live-versus-stored pitch check, amber stale band, re-track region, pitch-contour re-anchoring |
| Host transport position under PDC is wrong or inconsistent | Medium | Corrections land late or early | Phase 0 calibration; insert first in the chain; per-host offset table; alignment check |
| Segmentation quality (state of the art COnPOff about 0.6) | High | Users fix many notes | Cheap split and merge; a separate detection-fix mode; a granularity control |
| Keyboard capture in Live | High | Shortcuts feel broken | Mouse-first design; on-screen undo; document the host options |
| State too large, or slow on save or undo | Medium | Host hangs | Compact binary; zstd; cap on take length; measure in Phase 0 |
| A CLAP latency change triggers restarts | Known | Glitches when switching modes | Keep D fixed in graph mode; fix HI-8 in a patched nice-plug |
| ARA tooling immaturity | High (if pursued) | Large sunk cost | Defer; prototype against TestHost first |
| IP: Melodyne and Antares patents | Unknown | Legal | Freedom-to-operate review before release; avoid copying note-object resynthesis |
| Two near-identical modes confuse users | Medium | Support burden | Clear Auto / Graph toggle; untouched regions behave exactly like Auto |

---

## 9. Open questions for the user

1. **First surface.** Graph mode inside the plug-in (Bitwig and Live capture), a standalone file editor, or both? The standalone editor avoids host-sync risk and can be built first.
2. **Time editing, ever?** If yes, audio must be stored (Route 2), with external files and the lost-file risks users hate.
3. **Latency.** Should graph mode be locked to Studio latency (20-50 ms), or get its own lower-D "Graph" latency, at most about 23 ms, since it needs no decision lookahead?
4. **Uncaptured or unedited regions.** Fall back to Auto correction, or pass through untouched?
5. **MVP per-note controls.** Is centre, drift %, modulation %, retune and transition the right set? Should formant per note wait?
6. **Live scope.** Is Arrangement View only, at constant tempo, acceptable for the MVP? Celemony imposes the same limits.
7. **State budget.** Is about 1-2 MB of plug-in state per instance acceptable? What should the maximum take length be? Auto-Tune 8 allowed 4 h per instance.
8. **Stopgap.** Do you want an early "Export detected notes as MIDI" feature that uses the existing MIDI Target mode, if routing works in Live?
9. **ARA.** Is it a goal at all? If so, AU (Logic) probably has to come first, with Logic running under Rosetta until Apple ships native ARA.
10. **Freedom to operate.** Should a patent review on note-object editing happen before any public release?
11. **Plan.** Do you approve the phase order, Phase 0 spikes and Phase 1 headless analysis first, and the estimate of about 9-14 weeks to an MVP?

---

## 10. Evidence gaps (check before relying on them)

| Gap | Why it matters |
|---|---|
| Does Bitwig 6 implement the CLAP draft `undo`, `transport-control` or `resource-directory`? Not checked: the search budget ran out in this pass. | Merged undo, click-to-seek and audio files stored with the project |
| PDC and transport behaviour in Live 12 and Bitwig 6. The only Live evidence is a 2011 KVR thread (Live 8/9 era) [W]. | Correct alignment of captured data |
| Does any shipping host load CLAP-ARA directly (REAPER 7.82, Studio Pro 8.x, MultitrackStudio)? Test with Celemony's ARATestPlugIn. | Whether a CLAP-only ARA build is enough |
| Does Auto-Tune Pro 11's non-ARA Graph Mode keep the audio? The Antares blog suggests it does; AT8 is pitch-only. Antares help pages return 403. | Whether pitch-only capture is enough to compete |
| Fender Studio Pro 8.1 "Pitch Curves": detected-pitch editing or automation-style drawing? How are they stored? Only press descriptions were found. | Prior art for drawn curves |
| Melodyne details: the double-click original/none toggle on the drift and modulation tools; Quantize Time semantics; Waves Tune defaults; Revoice APT; the Pro Tools Elastic Audio range; Tony features (site unreachable). All [U]. | Only matters if we copy them |
| Can an offline export or freeze in Live and Bitwig drive the plug-in faster than real time with valid transport positions? | A fast-capture path |

---

## Key sources

**Prior art**
- Celemony help: [Transfer](https://helpcenter.celemony.com/M5/doc/melodyneEssential5/en/M5tour_TransferARA?env=live), [Correct Pitch macro](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_MacroPitch), [Drift/Modulation](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_ToolModulationDrift_2), [Melodyne in Live](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_DAW_Live?env=live), [Logic ARA](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_LogicARA_SpurenClips?env=logic), [Compatibility](https://services.celemony.com/en/melodyne/editions-and-technical-matters)
- Antares: [Auto-Tune 8 manual](https://www.strumentimusicali.net/manuali/2017/11/06/9b/antares-autotune8-en.pdf), [Auto-Tune Pro 2018 guide](https://r2.gear4music.com/media/70/709346/download_709346.pdf), [Graph Mode in Pro X](https://www.antarestech.com/blog/understanding-graph-mode-in-auto-tune-pro-x), [SOS Pro 11 review](https://www.soundonsound.com/reviews/antares-auto-tune-pro-11)
- [RePitch manual: versions and DAWs](https://www.synchroarts.com/manuals/RePitchV1/Manual/HTML/repitch-plug-in-versions-and-daws.html)

**ARA and CLAP**
- ARA: [ChangeLog](https://github.com/Celemony/ARA_API/blob/main/ChangeLog.txt), [ARACLAP.h](https://raw.githubusercontent.com/Celemony/ARA_API/main/ARACLAP.h), [ARA_Examples](https://github.com/Celemony/ARA_Examples)
- CLAP: [undo](https://raw.githubusercontent.com/free-audio/clap/main/include/clap/ext/draft/undo.h), [state-context](https://raw.githubusercontent.com/free-audio/clap/main/include/clap/ext/state-context.h), [resource-directory](https://raw.githubusercontent.com/free-audio/clap/main/include/clap/ext/draft/resource-directory.h), [ChangeLog](https://raw.githubusercontent.com/free-audio/clap/main/ChangeLog.md)

**Hosts**
- [Apple Rosetta article 121464](https://support.apple.com/en-us/121464), [Logic release notes](https://support.apple.com/en-us/109503)
- [Bitwig 6.1 release notes](https://downloads.bitwig.com/6.1/Release-Notes-6.1.html), [Ableton Live 12 manual](https://www.ableton.com/en/live-manual/12/live-audio-effect-reference/), [REAPER whatsnew](https://www.reaper.fm/whatsnew.txt), [Pro Tools ARA list](https://www.production-expert.com/production-expert-1/ara-integration-in-pro-tools-full-list-of-supported-plug-ins-and-free-versions), [Studio Pro 8.1](https://www.production-expert.com/production-expert-1/fender-studio-pro-81-adds-studio-assistant-moises-integration-and-native-vocal-tuning)

**Rust tooling**
- [nice-plug host behaviour wiki](https://codeberg.org/RustAudio/nice-plug/wiki/Host-behaviour), [clap-wrapper-rs](https://github.com/blepfx/clap-wrapper-rs), [ara2-bridge](https://github.com/eas4ai/ara2-bridge)

**Research**
- [QMUL 2021 note transcription](https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2021/LiDemirelProutskovaDixon-NLP4MusA2021.pdf), [vocadito](https://arxiv.org/abs/2110.05580), [OpenUtau schema](https://www.schemastore.org/openutau-ustx.json)

**Local**
- `docs/STATUS.md` §1, §2, §6; `docs/DESIGN.md` §2, §3.5, §3.6; `docs/research.md` §11.1.
- `crates/skatotune/src/{swap.rs,params.rs,editor/history.rs,editor/mod.rs}`; `crates/skato_engine/src/lib.rs`.
- `the cargo registry: nice-plug{,-core}-0.4.2`.
