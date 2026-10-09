//! The egui editor (DESIGN §3.6, R§8.23).
//!
//! Fixed 1100 × 720 logical-pixel layout:
//!
//! ```text
//! ┌ header: name · mode · Live+/Studio ms · range · quality · engine · latency · CPU/LABS ┐
//! ├ sidebar ─────────┬ readouts: target note · input/correction meters · vibrato · view ┤
//! │ key + scale      │ pitch graph (6 s scrolling piano roll)                          │
//! │ presets          │                                                                 │
//! │ 12-note keyboard ├ knob row: Retune … Tracking                                     ┤
//! │ vibrato toggles  ├ ▸ Advanced (tolerance, snap, detection, synth vibrato, MIDI, out) ┤
//! └──────────────────┴──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Data flow: parameters are read directly (atomics) and written through the
//! `ParamSetter` gesture protocol. Telemetry frames arrive through the
//! engine's lock-free ring, whose consumer is picked up from
//! [`SharedState::take_telemetry`] whenever `activate()` installs a new one;
//! the GUI never blocks the audio thread. Everything per frame works on
//! preallocated buffers (history ring, graph columns).
//!
//! The LABS button (top right) opens a menu of opt-in experiments
//! ([`crate::labs`]); they are GUI-only and saved with the project.

mod correction;
mod graph;
mod history;
mod keyboard;
mod labs_menu;
mod theme;
mod widgets;

#[cfg(feature = "demo-telemetry")]
mod demo;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use egui::{Align2, CornerRadius, Id, Rect, Stroke, StrokeKind, Ui, pos2, vec2};
use nice_plug::context::gui::GuiContext;
use nice_plug::editor::dpi::LogicalSize;
use nice_plug::prelude::*;
use nice_plug_egui::{
    EguiEditor, EguiEditorState, EguiNiceSettings, NiceEguiApp, RepaintNotifier, create_egui_editor,
};
use skato_core::{NoteState, PitchFrame, cents_to_hz, frame_flags};

use crate::params::{
    EngineChoice, LatencyModeChoice, MidiModeChoice, NoteOverride, ParamValues, Preset,
    SkatotuneParams,
};
use crate::shared::SharedState;
use graph::{RollContext, RollScratch, draw_pitch_roll, is_voiced, note_label};
use history::{DisplayClock, GraphView, PitchHistory, Timeline};
use theme::{ACCENT, BAD, BORDER, PANEL, TEXT, TEXT_DIM, TEXT_FAINT, WARN};
use widgets::KnobSpec;

/// Logical editor size (DESIGN §3.6).
pub const WIDTH: f32 = 1100.0;
pub const HEIGHT: f32 = 720.0;

const HEADER_H: f32 = 54.0;
const SIDEBAR_W: f32 = 214.0;
const MARGIN: f32 = 10.0;
const TOOLBAR_H: f32 = 46.0;
const KNOBS_H: f32 = 112.0;
const ADV_BAR_H: f32 = 24.0;
const ADV_DRAWER_H: f32 = 140.0;

/// Editor window state (size), shared with the wrapper.
pub fn default_state() -> Arc<EguiEditorState> {
    EguiEditorState::from_size(LogicalSize::new(WIDTH, HEIGHT), 1.0)
}

/// Wrap the app in nice-plug-egui's editor.
pub fn create(
    state: Arc<EguiEditorState>,
    repaint: RepaintNotifier,
    app: SkatotuneEditor,
) -> Option<EguiEditor<SkatotuneEditor>> {
    create_egui_editor(
        state,
        repaint,
        EguiNiceSettings::new().with_tile("Skatotune"),
        app,
    )
}

/// Smoothed values shown in the readout strip.
#[derive(Clone, Copy, Debug, Default)]
struct Readouts {
    /// Vibrato-free input deviation from the target, cents.
    input_dev: f32,
    /// Correction being applied (output − input, without transpose), cents.
    correction: f32,
    target: f32,
    vib_rate: f32,
    vib_extent: f32,
    valid: bool,
}

impl Readouts {
    /// Ease toward the newest voiced frame (`None` = nothing recent).
    fn update(&mut self, frame: Option<&PitchFrame>, transpose_st: f32, dt: f32) {
        let Some(f) = frame else {
            self.valid = false;
            return;
        };
        let a = if self.valid {
            1.0 - (-dt / 0.08).exp()
        } else {
            1.0
        };
        let dev = if f.target > 0.0 {
            f.centre - f.target
        } else {
            0.0
        };
        let corr = f.output - f.detected - 100.0 * transpose_st.round();
        self.input_dev += a * (dev - self.input_dev);
        self.correction += a * (corr - self.correction);
        self.target = f.target;
        self.vib_rate = f.vib_rate;
        self.vib_extent += a * (f.vib_extent - self.vib_extent);
        self.valid = true;
    }
}

/// Mean content wander as a fraction of the period `period` for a reported
/// latency `latency` (both in samples).
///
/// Pitch-synchronous shifting repeats or skips whole periods, so while the
/// pitch is being shifted the voice itself is delayed by `D − e`, with `e`
/// wandering over about one period (R§2.1). How far it trails D depends on
/// how far ahead of the output the engine can see pitch marks: while D minus
/// the mark delay is short compared with T (Live-Min; Live and Live+ for
/// periods longer than the look-ahead) only the latest mark is known and the
/// voice trails D by 0..T, mean ≈ 0.46·T; once D covers about 1.45 periods
/// beyond the mark delay the nearest mark is known and the mean is ≈ 0
/// (Studio; high voices in Live and Live+). Fitted to the shifter's
/// content-offset diagnostic for both engines in every mode (±25/±50 c
/// shifts, 70-600 Hz): within about 0.05·T.
fn wander_fraction(period: f32, latency: f32) -> f32 {
    if !(period > 0.0 && period.is_finite()) {
        return 0.0;
    }
    let lookahead = (latency - 10.0 - 0.2 * period) / period;
    0.46 * ((1.45 - lookahead) / 0.5).clamp(0.0, 1.0)
}

/// Time-averaged extra delay of the voice beyond the reported latency, for
/// the "avg" latency readout.
#[derive(Clone, Copy, Debug, Default)]
struct AvgDelay {
    /// Mean wander over recent voiced time, ms.
    extra_ms: f32,
    /// Mean period of the detected input pitch over the same time, ms.
    period_ms: f32,
    valid: bool,
    /// Reported latency the average was built for (a new engine starts over).
    latency: u32,
}

/// Averaging time of [`AvgDelay`] (voiced time only), seconds.
const AVG_DELAY_TAU_S: f32 = 2.0;

impl AvgDelay {
    /// Fold in the newest voiced frame (`None` = not singing: hold).
    fn update(&mut self, frame: Option<&PitchFrame>, dt: f32, sample_rate: f32, latency: u32) {
        if latency != self.latency {
            *self = Self {
                latency,
                ..Self::default()
            };
        }
        let Some(f) = frame.filter(|f| f.detected > 0.0) else {
            return;
        };
        let hz = cents_to_hz(f.detected);
        if !(hz.is_finite() && hz > 1.0 && sample_rate > 0.0) {
            return;
        }
        let period_ms = 1000.0 / hz;
        // Periods are only repeated or skipped while the pitch is shifted.
        let shifting = (f.output - f.detected).abs() > 1.0;
        let extra = if shifting {
            wander_fraction(sample_rate / hz, latency as f32) * period_ms
        } else {
            0.0
        };
        if self.valid {
            let a = 1.0 - (-dt / AVG_DELAY_TAU_S).exp();
            self.extra_ms += a * (extra - self.extra_ms);
            self.period_ms += a * (period_ms - self.period_ms);
        } else {
            self.extra_ms = extra;
            self.period_ms = period_ms;
            self.valid = true;
        }
    }

    /// Expected delay of the voice: reported latency plus the mean wander.
    fn expected_ms(&self, reported_ms: f32) -> Option<f32> {
        self.valid.then_some(reported_ms + self.extra_ms)
    }
}

/// How long a pending structural change has been waiting, to offer the host
/// restart as a click when the host takes its time (CLAP with a new latency;
/// any change whose engine has not arrived). Automatic re-requests are the
/// audio thread's job (VST3 only, `LATENCY_RETRY_AT_S`): it runs even while
/// the editor is closed, e.g. during project load.
#[derive(Clone, Copy, Debug, Default)]
struct PendingSince {
    since: Option<Instant>,
}

impl PendingSince {
    /// Wait before offering the manual restart.
    const OVERDUE: Duration = Duration::from_secs(1);

    /// Track `pending` (a structural change is waiting).
    fn update(&mut self, pending: bool, now: Instant) {
        self.since = if pending {
            Some(self.since.unwrap_or(now))
        } else {
            None
        };
    }

    /// The change has been waiting long enough to offer a manual restart.
    fn overdue(&self, now: Instant) -> bool {
        self.since
            .is_some_and(|s| now.saturating_duration_since(s) >= Self::OVERDUE)
    }
}

/// The editor app. Lives for the plugin's lifetime and survives editor
/// close/reopen; the egui and GUI contexts are replaced on every open.
pub struct SkatotuneEditor {
    params: Arc<SkatotuneParams>,
    shared: Arc<SharedState>,
    gui: Option<GuiContext>,
    telemetry: Option<rtrb::Consumer<PitchFrame>>,
    history: PitchHistory,
    clock: DisplayClock,
    /// Stitches engine resets (CLAP sleep/wake, rebuilds) into one timeline.
    timeline: Timeline,
    /// Sample rate the stored history was recorded at.
    history_sr: f64,
    view: GraphView,
    scratch: RollScratch,
    readouts: Readouts,
    avg: AvgDelay,
    advanced_open: bool,
    /// The LABS menu is showing.
    labs_open: bool,
    last_ui: Option<Instant>,
    pending: PendingSince,
    #[cfg(feature = "demo-telemetry")]
    demo: demo::DemoSource,
    #[cfg(feature = "demo-telemetry")]
    demo_active: bool,
    #[cfg(feature = "demo-telemetry")]
    last_voiced: Option<Instant>,
}

impl SkatotuneEditor {
    /// Cheap to construct (runs during plugin scans); the large buffers are
    /// allocated when the editor first opens.
    pub fn new(params: Arc<SkatotuneParams>, shared: Arc<SharedState>) -> Self {
        Self {
            params,
            shared,
            gui: None,
            telemetry: None,
            history: PitchHistory::default(),
            clock: DisplayClock::default(),
            timeline: Timeline::default(),
            history_sr: 0.0,
            view: GraphView::default(),
            scratch: RollScratch::default(),
            readouts: Readouts::default(),
            avg: AvgDelay::default(),
            advanced_open: false,
            labs_open: false,
            last_ui: None,
            pending: PendingSince::default(),
            #[cfg(feature = "demo-telemetry")]
            demo: demo::DemoSource::default(),
            #[cfg(feature = "demo-telemetry")]
            demo_active: false,
            #[cfg(feature = "demo-telemetry")]
            last_voiced: None,
        }
    }

    fn sample_rate(&self) -> f64 {
        let sr = self.shared.sample_rate.load(Ordering::Relaxed);
        if sr > 0.0 { f64::from(sr) } else { 48_000.0 }
    }

    /// Drain the telemetry ring into the history (bounded by the ring size).
    /// Returns whether any voiced frame arrived.
    fn pump_telemetry(&mut self, now: Instant) -> bool {
        let sr = self.sample_rate();
        if let Some(consumer) = self.shared.take_telemetry() {
            // A new engine: its positions start again. Keep the history and
            // continue the timeline, unless the sample rate changed (then the
            // stored positions no longer convert to time correctly).
            self.telemetry = Some(consumer);
            self.timeline.restart();
        }
        if sr != self.history_sr {
            self.history_sr = sr;
            self.history.clear();
            self.clock.reset();
            self.timeline.reset();
        }
        let mut newest = None;
        let mut voiced_seen = false;
        if let Some(rx) = self.telemetry.as_mut() {
            while let Ok(f) = rx.pop() {
                voiced_seen |= f.flags & frame_flags::VOICED != 0;
                #[cfg(feature = "demo-telemetry")]
                if self.demo_active {
                    if !voiced_seen {
                        continue; // real silence is ignored while the demo plays
                    }
                    self.demo_active = false;
                    self.history.clear();
                    self.clock.reset();
                    self.timeline.reset();
                }
                // Engine resets restart sample positions; map them onto one
                // monotonic display timeline instead of wiping the graph.
                let f = PitchFrame {
                    sample_pos: self.timeline.map(f.sample_pos, now, sr),
                    ..f
                };
                self.history.push(f);
                newest = Some(f.sample_pos);
            }
        }
        if let Some(pos) = newest {
            self.clock.on_frames(pos, now);
        }
        voiced_seen
    }

    /// Demo mode: after a second without voiced telemetry, synthesize frames
    /// up to `now` (real voiced frames end the demo in `pump_telemetry`).
    #[cfg(feature = "demo-telemetry")]
    fn pump_demo(
        &mut self,
        now: Instant,
        voiced_seen: bool,
        values: &ParamValues,
        states: &[NoteState; 12],
    ) {
        if voiced_seen {
            self.last_voiced = Some(now);
        }
        let quiet = self
            .last_voiced
            .is_none_or(|t| now.saturating_duration_since(t).as_secs_f32() > 1.0);
        if quiet && !self.demo_active {
            self.demo_active = true;
            self.history.clear();
            self.clock.reset();
            self.demo.restart(now);
        }
        if self.demo_active {
            let sr = self.sample_rate() as f32;
            if let Some(pos) = self
                .demo
                .generate(now, sr, values, states, &mut self.history)
            {
                self.clock.on_frames(pos, now);
            }
        }
    }

    /// Newest voiced frame within 150 ms of the display "now".
    fn recent_voiced(&self, now_pos: f64) -> Option<PitchFrame> {
        let recent = 0.15 * self.sample_rate();
        let len = self.history.len();
        (0..len.min(256))
            .filter_map(|k| self.history.get(len - 1 - k))
            .take_while(|f| now_pos - f.sample_pos as f64 <= recent)
            .find(|f| is_voiced(f))
            .copied()
    }
}

impl NiceEguiApp for SkatotuneEditor {
    fn build(
        &mut self,
        egui_ctx: egui::Context,
        nice_gui_ctx: GuiContext,
        frame: &mut nice_plug_egui::Frame,
    ) -> Result<(), nice_plug_egui::baseview::HandlerError> {
        frame.set_clear_color(theme::BG.into());
        self.setup(&egui_ctx, nice_gui_ctx);
        Ok(())
    }

    fn editor_closed(&mut self) {
        self.gui = None;
        // A menu does not outlive its window.
        self.labs_open = false;
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut nice_plug_egui::Frame) {
        self.draw(ui);
    }
}

impl SkatotuneEditor {
    /// Prepare for a newly opened window: theme, buffers, GUI context.
    /// (Also used by the headless renderer in `examples/render_gui.rs`.)
    pub fn setup(&mut self, egui_ctx: &egui::Context, gui: GuiContext) {
        theme::apply(egui_ctx);
        self.history.ensure_allocated();
        self.gui = Some(gui);
    }

    /// Open or close the Advanced drawer.
    pub fn set_advanced_open(&mut self, open: bool) {
        self.advanced_open = open;
    }

    /// Open or close the LABS menu.
    pub fn set_labs_open(&mut self, open: bool) {
        self.labs_open = open;
    }

    /// Draw one editor frame into `ui` (the whole 1100 × 720 window).
    pub fn draw(&mut self, ui: &mut Ui) {
        let Some(gui) = self.gui.clone() else { return };
        let setter = gui.param_setter();
        let now = Instant::now();
        let dt = self
            .last_ui
            .map_or(0.0, |t| now.saturating_duration_since(t).as_secs_f32())
            .min(0.25);
        self.last_ui = Some(now);

        let values = self.params.values();
        let settings = values.to_settings();
        // A learned scale replaces Key/Scale in every MIDI mode until cleared.
        let learned = self.params.learned_states();
        let states = learned.unwrap_or(settings.note_states);
        self.pending
            .update(self.shared.restart_pending.load(Ordering::Acquire), now);
        // Frames arriving now were produced under the current Transpose.
        self.history.set_transpose(settings.transpose_st);
        let voiced_seen = self.pump_telemetry(now);
        #[cfg(feature = "demo-telemetry")]
        self.pump_demo(now, voiced_seen, &values, &states);
        #[cfg(not(feature = "demo-telemetry"))]
        let _ = voiced_seen; // only the demo cares

        let sr = self.sample_rate();
        let now_pos = self.clock.tick(now, sr);
        let recent = self.recent_voiced(now_pos);
        self.readouts
            .update(recent.as_ref(), values.transpose_st, dt);
        self.avg.update(
            recent.as_ref(),
            dt,
            sr as f32,
            self.shared.latency_samples.load(Ordering::Relaxed),
        );
        let follow = recent.map(|f| if f.target > 0.0 { f.target } else { f.output });
        self.view.follow(follow, dt);

        let full = Rect::from_min_size(ui.max_rect().min, vec2(WIDTH, HEIGHT));
        ui.painter()
            .rect_filled(full, CornerRadius::ZERO, theme::BG);

        let header = Rect::from_min_size(full.min, vec2(WIDTH, HEADER_H));
        self.header(ui, header, &setter, &values, now);

        let sidebar = Rect::from_min_max(
            pos2(full.left() + MARGIN, header.bottom() + MARGIN),
            pos2(full.left() + SIDEBAR_W, full.bottom() - MARGIN),
        );
        let target_pc = recent
            .filter(|f| f.target > 0.0)
            .map(|f| ((f.target / 100.0).round() as i32).rem_euclid(12) as usize);
        self.sidebar(
            ui,
            sidebar,
            &setter,
            &values,
            &states,
            learned.is_some(),
            target_pc,
        );

        let main = Rect::from_min_max(
            pos2(sidebar.right() + MARGIN, sidebar.top()),
            pos2(full.right() - MARGIN, sidebar.bottom()),
        );
        let drawer_h = if self.advanced_open {
            ADV_DRAWER_H + 6.0
        } else {
            0.0
        };
        let adv_bar = Rect::from_min_size(
            pos2(main.left(), main.bottom() - drawer_h - ADV_BAR_H),
            vec2(main.width(), ADV_BAR_H),
        );
        let knobs = Rect::from_min_size(
            pos2(main.left(), adv_bar.top() - 6.0 - KNOBS_H),
            vec2(main.width(), KNOBS_H),
        );
        let toolbar = Rect::from_min_size(main.min, vec2(main.width(), TOOLBAR_H));
        let graph_rect = Rect::from_min_max(
            pos2(main.left(), toolbar.bottom() + 6.0),
            pos2(main.right(), knobs.top() - 6.0),
        );

        self.toolbar(ui, toolbar);
        let roll = RollContext {
            note_states: states,
            sample_rate: sr,
            now_pos,
            hop: history::control_hop(sr as f32) as f64,
            highlight_note: recent
                .filter(|f| f.target > 0.0)
                .map(|f| (f.target / 100.0).round() as i32),
            labs: self.params.labs(),
        };
        draw_pitch_roll(
            ui,
            graph_rect,
            &self.history,
            &mut self.view,
            &roll,
            &mut self.scratch,
        );
        self.knob_row(ui, knobs, &setter, &values);
        self.advanced_bar(ui, adv_bar);
        if self.advanced_open {
            let drawer = Rect::from_min_max(pos2(main.left(), adv_bar.bottom() + 6.0), main.max);
            self.advanced_drawer(ui, drawer, &setter);
        }
        // Last, so it floats over everything.
        if self.labs_open {
            self.labs_open =
                labs_menu::menu(ui, labs_button_rect(header), header.bottom(), &self.params);
        }

        // The graph scrolls continuously.
        ui.ctx().request_repaint();
    }
}

/// The LABS button in the header `header`: the top-right corner, on the
/// caption line of the CPU column, right edge flush with the CPU meter.
fn labs_button_rect(header: Rect) -> Rect {
    const W: f32 = 50.0;
    Rect::from_min_size(
        pos2(header.left() + 1084.0 - W, header.top() + 5.0),
        vec2(W, 16.0),
    )
}

/// Rounded panel background.
fn panel(ui: &Ui, rect: Rect) {
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(6), PANEL);
    p.rect_stroke(
        rect,
        CornerRadius::same(6),
        Stroke::new(1.0, BORDER),
        StrokeKind::Inside,
    );
}

/// Horizontal cents meter centred on zero, `range` cents each way.
fn cents_meter(ui: &Ui, rect: Rect, value: f32, range: f32, valid: bool) {
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(3), theme::PANEL_HI);
    let cx = rect.center().x;
    for frac in [-0.5f32, 0.5] {
        let x = cx + frac * rect.width() * 0.5;
        p.vline(x, rect.y_range(), Stroke::new(1.0, BORDER));
    }
    if valid {
        let v = value.clamp(-range, range) / range;
        let x = cx + v * rect.width() * 0.5;
        let mag = value.abs();
        let col = if mag < 10.0 {
            ACCENT
        } else if mag < 25.0 {
            WARN
        } else {
            BAD
        };
        let fill = Rect::from_x_y_ranges(cx.min(x)..=cx.max(x), rect.shrink(2.0).y_range());
        p.rect_filled(fill, CornerRadius::same(2), theme::alpha(col, 0.55));
        p.vline(
            x,
            rect.expand2(vec2(0.0, 3.0)).y_range(),
            Stroke::new(2.0, col),
        );
    }
    p.vline(cx, rect.y_range(), Stroke::new(1.0, TEXT_DIM));
}

impl SkatotuneEditor {
    fn header(
        &mut self,
        ui: &mut Ui,
        rect: Rect,
        setter: &ParamSetter,
        v: &ParamValues,
        now: Instant,
    ) {
        let p = ui.painter().clone();
        p.rect_filled(rect, CornerRadius::ZERO, PANEL);
        p.hline(
            rect.x_range(),
            rect.bottom() - 0.5,
            Stroke::new(1.0, BORDER),
        );

        // Wordmark.
        let mid = rect.center().y;
        let r = p.text(
            pos2(rect.left() + 18.0, mid - 6.0),
            Align2::LEFT_CENTER,
            "SKATO",
            theme::font(19.0),
            TEXT,
        );
        p.text(
            pos2(r.right() + 1.0, mid - 6.0),
            Align2::LEFT_CENTER,
            "TUNE",
            theme::font(19.0),
            ACCENT,
        );
        p.text(
            pos2(rect.left() + 18.0, mid + 12.0),
            Align2::LEFT_CENTER,
            "PITCH CORRECTION",
            theme::font(9.0),
            TEXT_FAINT,
        );

        let row = |x: f32, w: f32| Rect::from_min_size(pos2(x, rect.top() + 22.0), vec2(w, 24.0));
        let params = &self.params;

        widgets::caption(
            ui,
            pos2(176.0 + rect.left(), rect.top() + 19.0),
            "LATENCY MODE",
        );
        widgets::segmented(
            ui,
            row(rect.left() + 176.0, 236.0),
            setter,
            &params.latency_mode,
            None,
            "Live-Min ≈1 ms (Classic engine only) · Live ≈3-5 ms · Live+ 5-10 ms · Studio 20-50 ms with lookahead.\nChanging it restarts the plugin.",
        );
        match v.latency_mode {
            LatencyModeChoice::LivePlus => {
                widgets::caption(ui, pos2(rect.left() + 420.0, rect.top() + 19.0), "LIVE+ MS");
                widgets::value_box(
                    ui,
                    row(rect.left() + 420.0, 70.0),
                    setter,
                    &params.live_plus_ms,
                );
            }
            LatencyModeChoice::Studio => {
                widgets::caption(
                    ui,
                    pos2(rect.left() + 420.0, rect.top() + 19.0),
                    "STUDIO MS",
                );
                widgets::value_box(
                    ui,
                    row(rect.left() + 420.0, 70.0),
                    setter,
                    &params.studio_ms,
                );
            }
            LatencyModeChoice::LiveMin | LatencyModeChoice::Live => {}
        }

        widgets::caption(
            ui,
            pos2(rect.left() + 500.0, rect.top() + 19.0),
            "VOICE RANGE",
        );
        widgets::combo(
            ui,
            row(rect.left() + 500.0, 106.0),
            setter,
            &params.voice_range,
        );

        widgets::caption(ui, pos2(rect.left() + 622.0, rect.top() + 19.0), "QUALITY");
        widgets::segmented(
            ui,
            row(rect.left() + 622.0, 120.0),
            setter,
            &params.tier,
            None,
            "CPU / quality tier. Never changes latency.",
        );

        widgets::caption(ui, pos2(rect.left() + 754.0, rect.top() + 19.0), "ENGINE");
        let engine_tip = if v.latency_mode == LatencyModeChoice::LiveMin {
            "Live-Min always uses the Classic engine."
        } else {
            "Classic: resample + cycle splice, formants move (hard-tune character).\nNatural: PSOLA, formant-preserving."
        };
        widgets::segmented(
            ui,
            row(rect.left() + 754.0, 130.0),
            setter,
            &params.engine,
            None,
            engine_tip,
        );
        if v.latency_mode == LatencyModeChoice::LiveMin && v.engine == EngineChoice::Natural {
            p.text(
                pos2(rect.left() + 819.0, rect.top() + 51.0),
                Align2::CENTER_CENTER,
                "Classic in Live-Min",
                theme::font(9.0),
                WARN,
            );
        }

        // Latency readout: reported D, and the average delay of the voice.
        self.latency_readout(
            ui,
            Rect::from_min_size(
                pos2(rect.left() + 900.0, rect.top()),
                vec2(100.0, rect.height()),
            ),
            now,
        );

        // CPU meter: average bar plus a peak tick, as % of the buffer period.
        let x = rect.left() + 1004.0;
        let load = self.shared.cpu_load.load(Ordering::Relaxed);
        let peak = self.shared.cpu_peak.load(Ordering::Relaxed);
        widgets::caption(ui, pos2(x, rect.top() + 19.0), "CPU");
        // LABS shares the CPU column's caption line, right-aligned; a
        // hairline sets it apart from the CPU caption.
        let labs = labs_button_rect(rect);
        p.vline(
            labs.left() - 7.5,
            labs.y_range().shrink(2.0),
            Stroke::new(1.0, BORDER),
        );
        if labs_menu::button(ui, labs, self.labs_open).clicked() {
            self.labs_open = !self.labs_open;
        }
        let bar = Rect::from_min_size(pos2(x, rect.top() + 28.0), vec2(80.0, 6.0));
        p.rect_filled(bar, CornerRadius::same(3), theme::PANEL_HI);
        let col = |l: f32| {
            if l < 0.25 {
                ACCENT
            } else if l < 0.6 {
                WARN
            } else {
                BAD
            }
        };
        let fill = Rect::from_min_size(
            bar.min,
            vec2(bar.width() * load.clamp(0.0, 1.0), bar.height()),
        );
        p.rect_filled(fill, CornerRadius::same(3), col(load));
        let px = bar.left() + bar.width() * peak.clamp(0.0, 1.0);
        p.vline(
            px,
            bar.expand2(vec2(0.0, 2.0)).y_range(),
            Stroke::new(1.5, col(peak)),
        );
        p.text(
            pos2(x, rect.top() + 45.0),
            Align2::LEFT_CENTER,
            format!("{:.1}%  pk {:.1}%", load * 100.0, peak * 100.0),
            theme::mono(9.5),
            TEXT_DIM,
        );
    }

    /// The LATENCY block of the header (`rect` spans the header height).
    fn latency_readout(&mut self, ui: &mut Ui, rect: Rect, now: Instant) {
        let p = ui.painter().clone();
        let sr = self.sample_rate() as f32;
        let ms = |samples: u32| samples as f32 / sr * 1000.0;
        let latency = self.shared.latency_samples.load(Ordering::Relaxed);
        let x = rect.left();
        widgets::caption(ui, pos2(x, rect.top() + 19.0), "LATENCY");
        // Line 1: the reported latency D (what the host compensates).
        let d = p.text(
            pos2(x, rect.top() + 33.0),
            Align2::LEFT_CENTER,
            format!("{:.1} ms", ms(latency)),
            theme::mono(13.0),
            TEXT,
        );
        p.text(
            pos2(d.right() + 4.0, rect.top() + 34.0),
            Align2::LEFT_CENTER,
            "reported",
            theme::font(8.5),
            TEXT_FAINT,
        );
        let avg = self.avg.expected_ms(ms(latency));
        let line2 = pos2(x, rect.top() + 47.0);
        let api = self.gui.as_ref().map(GuiContext::plugin_api);
        let pending = self.shared.restart_pending.load(Ordering::Acquire);
        if pending {
            let target = ms(self.shared.pending_latency_samples.load(Ordering::Relaxed));
            let host_restarts = matches!(api, Some(PluginApi::Vst3 | PluginApi::Clap));
            let awaiting_host = self.shared.awaiting_host.load(Ordering::Relaxed);
            if host_restarts && self.pending.overdue(now) {
                // The host is taking its time: offer the restart as a click.
                let r = Rect::from_min_size(pos2(x - 2.0, line2.y - 7.0), vec2(100.0, 14.0));
                let resp = ui.interact(r, Id::new("skato-apply-restart"), egui::Sense::click());
                let col = if resp.hovered() { TEXT } else { WARN };
                p.text(
                    line2,
                    Align2::LEFT_CENTER,
                    format!("→ {target:.1} ms · APPLY"),
                    theme::font(9.0),
                    col,
                );
                if resp
                    .on_hover_text(
                        "The new setting is not running yet, so the old latency and voice range are still in use.\nClick to ask the host to restart the plugin.",
                    )
                    .clicked()
                    && let Some(gui) = &self.gui
                {
                    gui.request_restart();
                }
            } else {
                let text = if awaiting_host {
                    format!("→ {target:.1} ms after restart")
                } else {
                    format!("→ {target:.1} ms applying")
                };
                p.text(line2, Align2::LEFT_CENTER, text, theme::font(9.0), WARN);
            }
            return;
        }
        // Line 2: the average delay of the voice you hear (D plus the mean
        // PSOLA wander), once a voice has been detected.
        let (text, col) = match avg {
            Some(a) => (format!("heard avg {a:.1} ms"), TEXT_DIM),
            None => ("heard avg --".to_string(), TEXT_FAINT),
        };
        p.text(line2, Align2::LEFT_CENTER, text, theme::font(9.0), col);

        let block = self.shared.block_samples.load(Ordering::Relaxed);
        let max_block = self.shared.max_block.load(Ordering::Relaxed);
        let period = self.avg.valid.then_some(self.avg.period_ms);
        let hover = Rect::from_min_max(
            pos2(x - 4.0, rect.top() + 8.0),
            pos2(rect.right(), rect.bottom()),
        );
        ui.interact(hover, Id::new("skato-latency-info"), egui::Sense::hover())
            .on_hover_ui(|ui| {
                ui.label(latency_tooltip(LatencyInfo {
                    reported_samples: latency,
                    sample_rate: sr,
                    avg_ms: avg,
                    period_ms: period,
                    extra_ms: self.avg.extra_ms,
                    block,
                    max_block,
                }));
            });
    }

    #[allow(clippy::too_many_arguments)]
    fn sidebar(
        &mut self,
        ui: &mut Ui,
        rect: Rect,
        setter: &ParamSetter,
        v: &ParamValues,
        states: &[NoteState; 12],
        learned: bool,
        target_pc: Option<usize>,
    ) {
        panel(ui, rect);
        let params = self.params.clone();
        let x0 = rect.left() + 12.0;
        let x1 = rect.right() - 12.0;
        let w = x1 - x0;
        let mut y = rect.top() + 14.0;

        // A learned scale replaces Key/Scale: grey the selectors out and say
        // why, rather than letting edits silently do nothing.
        let mut rule_end = x1;
        if learned {
            let note = ui.painter().text(
                pos2(x1, y),
                Align2::RIGHT_CENTER,
                "NOT IN USE",
                theme::font(9.0),
                theme::MIDI,
            );
            rule_end = note.left() - 6.0;
        }
        widgets::section_title(ui, pos2(x0, y), rule_end, "KEY & SCALE");
        y += 12.0;
        let selectors = Rect::from_min_size(pos2(x0, y), vec2(w, 24.0));
        ui.add_enabled_ui(!learned, |ui| {
            widgets::combo(
                ui,
                Rect::from_min_size(pos2(x0, y), vec2(54.0, 24.0)),
                setter,
                &params.key,
            );
            widgets::combo(
                ui,
                Rect::from_min_size(pos2(x0 + 62.0, y), vec2(w - 62.0, 24.0)),
                setter,
                &params.scale,
            );
        });
        if learned {
            ui.interact(
                selectors,
                Id::new("skato-key-scale-learned"),
                egui::Sense::hover(),
            )
            .on_hover_text(LEARNED_REPLACES_SCALE);
        }
        y += 42.0;

        widgets::section_title(ui, pos2(x0, y), x1, "PRESETS");
        y += 12.0;
        let bw = (w - 6.0) * 0.5;
        for (i, preset) in Preset::ALL.iter().enumerate() {
            let col = (i % 2) as f32;
            let row = (i / 2) as f32;
            let r =
                Rect::from_min_size(pos2(x0 + col * (bw + 6.0), y + row * 30.0), vec2(bw, 26.0));
            let active = preset.matches(v);
            if widgets::button(
                ui,
                r,
                Id::new(("skato-preset", i)),
                preset.name(),
                active,
                preset.description(),
            )
            .clicked()
            {
                apply_values(setter, &params, v, &preset.apply(v));
            }
        }
        y += 74.0;

        let learning = v.midi_mode == MidiModeChoice::Learn;
        if learned {
            let keep = Rect::from_min_size(pos2(x1 - 40.0, y - 8.0), vec2(40.0, 16.0));
            let clear = Rect::from_min_size(pos2(keep.left() - 48.0, y - 8.0), vec2(44.0, 16.0));
            let title = ui.painter().text(
                pos2(x0, y),
                Align2::LEFT_CENTER,
                "LEARNED SCALE",
                theme::font(10.5),
                theme::MIDI,
            );
            let title_tip = if learning {
                "Learned from MIDI note-ons. It replaces Key/Scale in every MIDI mode until you clear it, and is saved with the project.\nMIDI Learn is on: each note-on adds its note (the first one after switching to Learn starts a new scale)."
            } else {
                "Learned from MIDI note-ons. It replaces Key/Scale in every MIDI mode until you clear it, and is saved with the project."
            };
            ui.interact(
                title.expand(2.0),
                Id::new("skato-learned-title"),
                egui::Sense::hover(),
            )
            .on_hover_text(title_tip);
            if widgets::button(
                ui,
                clear,
                Id::new("skato-clear-learned"),
                "CLEAR",
                false,
                "Forget the learned scale and go back to Key/Scale and the note settings.",
            )
            .clicked()
            {
                self.params.clear_learned_scale();
            }
            if widgets::button(
                ui,
                keep,
                Id::new("skato-keep-learned"),
                "KEEP",
                false,
                "Turn the learned scale into note settings (saved as parameters, automatable and undoable), then clear the learned scale. The notes stay the same.",
            )
            .clicked()
            {
                keep_learned_scale(setter, &params, states);
                self.params.clear_learned_scale();
            }
        } else if learning {
            let title = ui.painter().text(
                pos2(x0, y),
                Align2::LEFT_CENTER,
                "SCALE NOTES · LEARN: PLAY NOTES",
                theme::font(10.5),
                theme::MIDI,
            );
            ui.interact(
                title.expand(2.0),
                Id::new("skato-learn-title"),
                egui::Sense::hover(),
            )
            .on_hover_text(
                "MIDI Learn: each MIDI note-on adds its note to a learned scale, which then replaces Key/Scale until you clear it.",
            );
        } else {
            widgets::section_title(ui, pos2(x0, y), x1, "SCALE NOTES");
        }
        y += 12.0;
        let kb = Rect::from_min_size(pos2(x0, y), vec2(w, 12.0 * 24.0));
        let ks = keyboard::KeyboardState {
            states: *states,
            learned,
            target_pc,
        };
        keyboard::keyboard(ui, kb, setter, &params, &ks);
        y = kb.bottom() + 18.0;

        widgets::section_title(ui, pos2(x0, y), x1, "VIBRATO");
        y += 12.0;
        widgets::toggle(
            ui,
            Rect::from_min_size(pos2(x0, y), vec2(w, 22.0)),
            setter,
            &params.preserve_vibrato,
            "Preserve vibrato",
            "Correct the vibrato-free centre and keep the sung vibrato.\nOff: classic behaviour, vibrato is corrected too.",
        );
        y += 28.0;
        widgets::toggle(
            ui,
            Rect::from_min_size(pos2(x0, y), vec2(w, 22.0)),
            setter,
            &params.ignore_vibrato,
            "Ignore vibrato for targeting",
            "Choose the target note from the vibrato-free centre, so wide vibrato doesn't flip notes.",
        );

        // Footer: version and host format.
        let api = self.gui.as_ref().map(|g| g.plugin_api());
        let api_text = match api {
            Some(PluginApi::Clap) => "CLAP",
            Some(PluginApi::Vst3) => "VST3",
            Some(PluginApi::Standalone) => "Standalone",
            None => "",
        };
        ui.painter().text(
            pos2(x0, rect.bottom() - 12.0),
            Align2::LEFT_CENTER,
            format!("v{} · {api_text}", env!("CARGO_PKG_VERSION")),
            theme::font(9.5),
            TEXT_FAINT,
        );
        #[cfg(feature = "demo-telemetry")]
        if self.demo_active {
            ui.painter().text(
                pos2(x1, rect.bottom() - 12.0),
                Align2::RIGHT_CENTER,
                "DEMO DATA",
                theme::font(9.5),
                WARN,
            );
        }
    }

    fn toolbar(&mut self, ui: &mut Ui, rect: Rect) {
        panel(ui, rect);
        let p = ui.painter().clone();
        let r = self.readouts;
        let mid = rect.center().y;
        let x = rect.left() + 14.0;

        // Current target note.
        widgets::caption(ui, pos2(x, rect.top() + 14.0), "TARGET");
        if r.valid && r.target > 0.0 {
            let n = (r.target / 100.0).round() as i32;
            let off = r.target - n as f32 * 100.0;
            let t = p.text(
                pos2(x, mid + 7.0),
                Align2::LEFT_CENTER,
                note_label(n),
                theme::font(21.0),
                ACCENT,
            );
            if off.abs() >= 1.0 {
                p.text(
                    pos2(t.right() + 4.0, mid + 9.0),
                    Align2::LEFT_CENTER,
                    format!("{off:+.0}c"),
                    theme::mono(10.0),
                    TEXT_DIM,
                );
            }
        } else {
            p.text(
                pos2(x, mid + 7.0),
                Align2::LEFT_CENTER,
                "--",
                theme::font(21.0),
                TEXT_FAINT,
            );
        }

        // Input deviation and applied correction meters.
        let meter = |x: f32, title: &str, value: f32, range: f32| {
            widgets::caption(ui, pos2(x, rect.top() + 14.0), title);
            let bar = Rect::from_min_size(pos2(x, mid + 1.0), vec2(150.0, 10.0));
            cents_meter(ui, bar, value, range, r.valid);
            let text = if r.valid {
                format!("{value:+5.0} c")
            } else {
                "   -- c".to_string()
            };
            p.text(
                pos2(bar.right() + 8.0, bar.center().y),
                Align2::LEFT_CENTER,
                text,
                theme::mono(11.5),
                TEXT,
            );
        };
        meter(rect.left() + 104.0, "INPUT vs TARGET", r.input_dev, 50.0);
        meter(rect.left() + 330.0, "CORRECTION", r.correction, 100.0);

        // Vibrato readout.
        let vx = rect.left() + 560.0;
        widgets::caption(ui, pos2(vx, rect.top() + 14.0), "VIBRATO");
        let vib = if r.valid && r.vib_rate > 0.0 {
            format!("{:.1} Hz  ±{:.0} c", r.vib_rate, r.vib_extent)
        } else {
            "--".to_string()
        };
        p.text(
            pos2(vx, mid + 6.0),
            Align2::LEFT_CENTER,
            vib,
            theme::mono(11.5),
            TEXT,
        );

        // View controls.
        let bx = rect.right() - 12.0;
        let b = |i: f32, w: f32| Rect::from_min_size(pos2(bx - i, mid - 11.0), vec2(w, 22.0));
        if widgets::button(
            ui,
            b(26.0, 26.0),
            Id::new("skato-zoom-in"),
            "+",
            false,
            "Zoom in (or Cmd/Ctrl + scroll on the graph)",
        )
        .clicked()
        {
            self.view.zoom(1.25);
        }
        if widgets::button(
            ui,
            b(56.0, 26.0),
            Id::new("skato-zoom-out"),
            "-",
            false,
            "Zoom out",
        )
        .clicked()
        {
            self.view.zoom(0.8);
        }
        let auto = self.view.auto;
        if widgets::button(
            ui,
            b(118.0, 56.0),
            Id::new("skato-follow"),
            "FOLLOW",
            auto,
            "Keep the sung pitch in view. Drag or scroll the graph to look around; double-click it to follow again.",
        )
        .clicked()
        {
            if auto {
                self.view.auto = false;
            } else {
                self.view.reset();
            }
        }
    }

    fn knob_row(&mut self, ui: &mut Ui, rect: Rect, setter: &ParamSetter, v: &ParamValues) {
        panel(ui, rect);
        let params = self.params.clone();
        let cells = 10;
        let cw = (rect.width() - 16.0) / cells as f32;
        let cell = |i: usize| {
            Rect::from_min_size(
                pos2(rect.left() + 8.0 + cw * i as f32, rect.top() + 4.0),
                vec2(cw, 82.0),
            )
        };
        let centre_of = |p: &FloatParam, plain: f32| p.preview_normalized(plain);
        let spec = |label, origin, tooltip| KnobSpec {
            label,
            origin,
            dimmed: false,
            value_override: None,
            tooltip,
        };

        widgets::knob(
            ui,
            cell(0),
            setter,
            &params.retune_ms,
            &spec(
                "RETUNE",
                0.0,
                "Retune speed: 63 % time constant of the correction. 0 = instant (hard tune), 10-50 ms natural.",
            ),
        );
        let linked = v.transition_link;
        let tr = KnobSpec {
            label: "TRANSITION",
            origin: 0.0,
            dimmed: linked,
            value_override: if linked { Some("= Retune") } else { None },
            tooltip: "Time constant used right after the target note changes.",
        };
        widgets::knob(ui, cell(1), setter, &params.transition_ms, &tr);
        widgets::knob(
            ui,
            cell(2),
            setter,
            &params.humanize,
            &spec(
                "HUMANIZE",
                0.0,
                "Slows correction on the sustained part of long notes; short notes stay fast.",
            ),
        );
        widgets::knob(
            ui,
            cell(3),
            setter,
            &params.flex_tune,
            &spec(
                "FLEX-TUNE",
                0.0,
                "0 corrects everything. Higher values only correct pitch already close to the target.",
            ),
        );
        widgets::knob(
            ui,
            cell(4),
            setter,
            &params.amount,
            &spec("AMOUNT", 0.0, "Correction amount."),
        );
        widgets::knob(
            ui,
            cell(5),
            setter,
            &params.vibrato_depth,
            &spec(
                "VIBRATO",
                centre_of(&params.vibrato_depth, 100.0),
                "Scale of the sung vibrato: 100 % unchanged, 0 % removed, negative inverts.",
            ),
        );
        widgets::knob(
            ui,
            cell(6),
            setter,
            &params.formant_st,
            &spec(
                "FORMANT",
                centre_of(&params.formant_st, 0.0),
                "Formant shift (throat). Cleanest within ±3 st.",
            ),
        );
        widgets::knob(
            ui,
            cell(7),
            setter,
            &params.transpose_st,
            &spec(
                "TRANSPOSE",
                centre_of(&params.transpose_st, 0.0),
                "Shift after correction, semitones.",
            ),
        );
        widgets::knob(
            ui,
            cell(8),
            setter,
            &params.detune_cents,
            &spec(
                "DETUNE",
                centre_of(&params.detune_cents, 0.0),
                "Move every target note, cents (e.g. A = 442 Hz is +8 c).",
            ),
        );
        widgets::knob(
            ui,
            cell(9),
            setter,
            &params.tracking,
            &spec(
                "TRACKING",
                0.0,
                "0 relaxed (accepts breathy/noisy pitch), 100 choosy.",
            ),
        );

        // Pills under the knobs they modify.
        let pill = |i: usize, w: f32| {
            let c = cell(i);
            Rect::from_center_size(pos2(c.center().x, rect.bottom() - 13.0), vec2(w, 19.0))
        };
        widgets::toggle(
            ui,
            pill(1, 64.0),
            setter,
            &params.transition_link,
            "LINK",
            "Link the note transition to the retune speed.",
        );
        widgets::toggle(
            ui,
            pill(6, 74.0),
            setter,
            &params.formant_follow,
            "FOLLOW",
            "Formants also move with Transpose.",
        );
    }

    fn advanced_bar(&mut self, ui: &mut Ui, rect: Rect) {
        let resp = ui.interact(rect, Id::new("skato-advanced"), egui::Sense::click());
        if resp.clicked() {
            self.advanced_open = !self.advanced_open;
        }
        let p = ui.painter();
        let fill = if resp.hovered() {
            theme::PANEL_HOVER
        } else {
            PANEL
        };
        p.rect_filled(rect, CornerRadius::same(6), fill);
        p.rect_stroke(
            rect,
            CornerRadius::same(6),
            Stroke::new(1.0, BORDER),
            StrokeKind::Inside,
        );
        // Disclosure triangle, painted (no glyph dependency).
        let c = pos2(rect.left() + 16.0, rect.center().y);
        let tri = if self.advanced_open {
            vec![
                pos2(c.x - 4.5, c.y - 2.5),
                pos2(c.x + 4.5, c.y - 2.5),
                pos2(c.x, c.y + 3.5),
            ]
        } else {
            vec![
                pos2(c.x - 2.5, c.y - 4.5),
                pos2(c.x - 2.5, c.y + 4.5),
                pos2(c.x + 3.5, c.y),
            ]
        };
        p.add(egui::Shape::convex_polygon(tri, TEXT_DIM, Stroke::NONE));
        let r = p.text(
            pos2(rect.left() + 28.0, rect.center().y),
            Align2::LEFT_CENTER,
            "ADVANCED",
            theme::font(10.5),
            TEXT,
        );
        p.text(
            pos2(r.right() + 12.0, rect.center().y),
            Align2::LEFT_CENTER,
            "Tolerance · Snap · Detection · Synth vibrato · MIDI · Output",
            theme::font(10.5),
            TEXT_FAINT,
        );
    }

    fn advanced_drawer(&mut self, ui: &mut Ui, rect: Rect, setter: &ParamSetter) {
        panel(ui, rect);
        let params = self.params.clone();
        let cols = 4;
        let cw = (rect.width() - 24.0) / cols as f32;
        let col_x = |i: usize| rect.left() + 12.0 + cw * i as f32;
        let row = |i: usize, r: usize| {
            Rect::from_min_size(
                pos2(col_x(i), rect.top() + 26.0 + 23.0 * r as f32),
                vec2(cw - 18.0, 20.0),
            )
        };
        let title_y = rect.top() + 13.0;

        widgets::section_title(ui, pos2(col_x(0), title_y), col_x(1) - 18.0, "TARGETING");
        widgets::hslider(ui, row(0, 0), setter, &params.tolerance_cents, "Tolerance");
        widgets::hslider(ui, row(0, 1), setter, &params.tolerance_ms, "Tol. time");
        widgets::hslider(ui, row(0, 2), setter, &params.snap_min_cents, "Snap min");
        widgets::hslider(ui, row(0, 3), setter, &params.snap_max_st, "Snap max");

        widgets::section_title(
            ui,
            pos2(col_x(1), title_y),
            col_x(2) - 18.0,
            "DETECTION · OUTPUT",
        );
        widgets::hslider(ui, row(1, 0), setter, &params.gate_db, "Gate");
        let seg = row(1, 1);
        ui.painter().text(
            pos2(seg.left(), seg.center().y),
            Align2::LEFT_CENTER,
            "Analysis",
            theme::font(11.0),
            TEXT_DIM,
        );
        widgets::segmented(
            ui,
            Rect::from_min_max(pos2(seg.left() + 62.0, seg.top()), seg.max),
            setter,
            &params.analysis_channel,
            Some(&["L", "R", "Mix"]),
            "Which input channel feeds pitch detection (stereo inputs).",
        );
        widgets::hslider(ui, row(1, 3), setter, &params.output_gain_db, "Output");

        // The enable switch sits on the title line so the four sliders get full rows.
        let enable = Rect::from_min_size(
            pos2(col_x(3) - 18.0 - 70.0, title_y - 9.0),
            vec2(70.0, 18.0),
        );
        widgets::section_title(
            ui,
            pos2(col_x(2), title_y),
            enable.left() - 8.0,
            "SYNTH VIBRATO",
        );
        widgets::toggle(
            ui,
            enable,
            setter,
            &params.lfo_on,
            "Enable",
            "Add an LFO vibrato after correction.",
        );
        widgets::hslider(ui, row(2, 0), setter, &params.lfo_rate, "Rate");
        widgets::hslider(ui, row(2, 1), setter, &params.lfo_depth, "Depth");
        widgets::hslider(ui, row(2, 2), setter, &params.lfo_delay, "Delay");
        widgets::hslider(ui, row(2, 3), setter, &params.lfo_ramp, "Ramp");

        widgets::section_title(ui, pos2(col_x(3), title_y), rect.right() - 12.0, "MIDI");
        widgets::segmented(
            ui,
            row(3, 0),
            setter,
            &params.midi_mode,
            None,
            "Off · Target: held notes are the targets · Learn: note-ons build the scale · Auto: held notes, else the scale.",
        );
        widgets::toggle(
            ui,
            row(3, 1),
            setter,
            &params.midi_all_octaves,
            "Targets in all octaves",
            "MIDI notes match their pitch class in every octave.",
        );
        widgets::hslider(ui, row(3, 2), setter, &params.bend_range, "Bend");
        let dropped = self.shared.telemetry_dropped.load(Ordering::Relaxed);
        if dropped > 0 {
            let r = row(3, 3);
            ui.painter().text(
                pos2(r.left(), r.center().y),
                Align2::LEFT_CENTER,
                format!("{dropped} graph frames dropped"),
                theme::font(9.5),
                TEXT_FAINT,
            );
        }
    }
}

/// Why Key/Scale is greyed out while a learned scale is in use.
const LEARNED_REPLACES_SCALE: &str = "A MIDI-learned scale is in use and replaces Key and Scale.\nCLEAR it (above the keyboard) to use Key and Scale again.";

/// Make a learned scale permanent: each pitch class's override becomes On
/// (learned) or Remove, through the gesture protocol so it is undoable and
/// saved like any edit. The caller then clears the learned scale.
fn keep_learned_scale(setter: &ParamSetter, p: &SkatotuneParams, learned: &[NoteState; 12]) {
    for (pc, state) in learned.iter().enumerate() {
        let want = match state {
            NoteState::On => NoteOverride::On,
            NoteState::Bypass => NoteOverride::Bypass,
            NoteState::Remove => NoteOverride::Remove,
        };
        let param = p.note_override(pc);
        if param.value() != want {
            setter.begin_set_parameter(param);
            setter.set_parameter(param, want);
            setter.end_set_parameter(param);
        }
    }
}

/// Inputs of the latency tooltip.
struct LatencyInfo {
    reported_samples: u32,
    sample_rate: f32,
    /// Expected delay of the voice, ms (`None` before any voice was detected).
    avg_ms: Option<f32>,
    /// Mean period of the voice, ms.
    period_ms: Option<f32>,
    /// Mean wander beyond the reported latency, ms.
    extra_ms: f32,
    /// Current and maximum host block, samples.
    block: u32,
    max_block: u32,
}

/// Explains reported vs average latency vs the host buffer.
fn latency_tooltip(i: LatencyInfo) -> String {
    let sr = if i.sample_rate > 0.0 {
        i.sample_rate
    } else {
        48_000.0
    };
    let ms = |s: u32| s as f32 / sr * 1000.0;
    let mut t = format!(
        "Reported: {:.1} ms ({} samples at {:.1} kHz). The host compensates this (PDC), and it is the fixed part of what you hear when monitoring through the plugin.\n\n",
        ms(i.reported_samples),
        i.reported_samples,
        sr / 1000.0
    );
    let how = "While it shifts the pitch, the engine repeats or skips whole pitch periods, so the voice wanders up to one period behind the reported latency. With little look-ahead (Live-Min, Live) that averages about half a period; Studio's look-ahead, and Live+ or Live for high voices, keeps it near zero. Unshifted, it is exactly the reported latency.";
    match (i.avg_ms, i.period_ms) {
        (Some(avg), Some(period)) => t.push_str(&format!(
            "Heard avg: {avg:.1} ms, the expected delay of the voice itself. {how} At your average pitch ({:.0} Hz, period {period:.1} ms) that adds {:.1} ms.\n\n",
            1000.0 / period,
            i.extra_ms
        )),
        _ => t.push_str(&format!(
            "Heard avg: shown once a voice is detected; it is the expected delay of the voice itself. {how}\n\n"
        )),
    }
    if i.block > 0 {
        t.push_str(&format!(
            "Host buffer: {} samples ({:.1} ms), up to {}. Monitoring live through a DAW also goes through the audio interface's input and output buffers (about 2 × buffer) plus converter latency. The host does not tell plugins that part.",
            i.block,
            ms(i.block),
            i.max_block
        ));
    } else {
        t.push_str("Host buffer: not running. Monitoring live through a DAW adds the audio interface's input and output buffers (about 2 × buffer) plus converter latency; the host does not tell plugins that part.");
    }
    t
}

/// Write every field that differs between `old` and `new` through the gesture
/// protocol (used by presets). Only correction-behaviour fields are covered,
/// which is everything a [`Preset`] changes.
fn apply_values(setter: &ParamSetter, p: &SkatotuneParams, old: &ParamValues, new: &ParamValues) {
    fn set_f(setter: &ParamSetter, param: &FloatParam, old: f32, new: f32) {
        if (old - new).abs() > 1e-6 {
            setter.begin_set_parameter(param);
            setter.set_parameter(param, new);
            setter.end_set_parameter(param);
        }
    }
    fn set_b(setter: &ParamSetter, param: &BoolParam, old: bool, new: bool) {
        if old != new {
            setter.begin_set_parameter(param);
            setter.set_parameter(param, new);
            setter.end_set_parameter(param);
        }
    }
    if old.engine != new.engine {
        setter.begin_set_parameter(&p.engine);
        setter.set_parameter(&p.engine, new.engine);
        setter.end_set_parameter(&p.engine);
    }
    set_f(setter, &p.retune_ms, old.retune_ms, new.retune_ms);
    set_b(
        setter,
        &p.transition_link,
        old.transition_link,
        new.transition_link,
    );
    set_f(
        setter,
        &p.transition_ms,
        old.transition_ms,
        new.transition_ms,
    );
    set_f(setter, &p.humanize, old.humanize, new.humanize);
    set_f(setter, &p.flex_tune, old.flex_tune, new.flex_tune);
    set_f(setter, &p.amount, old.amount_pct, new.amount_pct);
    set_b(
        setter,
        &p.preserve_vibrato,
        old.preserve_vibrato,
        new.preserve_vibrato,
    );
    set_b(
        setter,
        &p.ignore_vibrato,
        old.ignore_vibrato,
        new.ignore_vibrato,
    );
    set_f(
        setter,
        &p.vibrato_depth,
        old.vibrato_depth_pct,
        new.vibrato_depth_pct,
    );
    set_f(
        setter,
        &p.tolerance_cents,
        old.tolerance_cents,
        new.tolerance_cents,
    );
    set_f(setter, &p.tolerance_ms, old.tolerance_ms, new.tolerance_ms);
    set_f(
        setter,
        &p.snap_min_cents,
        old.snap_min_cents,
        new.snap_min_cents,
    );
    set_f(setter, &p.snap_max_st, old.snap_max_st, new.snap_max_st);
    set_f(setter, &p.formant_st, old.formant_st, new.formant_st);
    set_b(
        setter,
        &p.formant_follow,
        old.formant_follow,
        new.formant_follow,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voiced(output: f32, detected: f32, centre: f32, target: f32) -> PitchFrame {
        PitchFrame {
            output,
            detected,
            centre,
            target,
            flags: frame_flags::VOICED,
            ..PitchFrame::default()
        }
    }

    #[test]
    fn readouts_follow_the_newest_frame() {
        let mut r = Readouts::default();
        r.update(
            Some(&voiced(6900.0, 6880.0, 6885.0, 6900.0)),
            0.0,
            1.0 / 60.0,
        );
        assert!(r.valid);
        // First valid frame is taken as is.
        assert!((r.input_dev + 15.0).abs() < 1e-3);
        assert!((r.correction - 20.0).abs() < 1e-3);
        // Transpose is not "correction".
        let mut t = Readouts::default();
        t.update(Some(&voiced(7100.0, 6880.0, 6880.0, 6900.0)), 2.0, 0.016);
        assert!((t.correction - 20.0).abs() < 1e-3);
        r.update(None, 0.0, 0.016);
        assert!(!r.valid);
    }

    /// A voiced frame at `detected_hz`, shifted by `shift_cents`.
    fn at(detected_hz: f32, shift_cents: f32) -> PitchFrame {
        let detected = skato_core::hz_to_cents(detected_hz);
        PitchFrame {
            detected,
            output: detected + shift_cents,
            flags: frame_flags::VOICED,
            ..PitchFrame::default()
        }
    }

    #[test]
    fn wander_fraction_follows_the_look_ahead() {
        // 48 kHz. Live Alto/Tenor D = 213: half a period for 200 Hz (T = 240),
        // none for 440 Hz (T = 109, the next mark is already known).
        assert!((wander_fraction(240.0, 213.0) - 0.46).abs() < 1e-6);
        assert_eq!(wander_fraction(109.0, 213.0), 0.0);
        // Measured 0.22·T at 330 Hz: the fit is within 0.05·T.
        assert!((wander_fraction(145.5, 213.0) - 0.22).abs() < 0.05);
        // Live-Min (D = 48) always trails; Studio (D = 1920) never.
        assert!((wander_fraction(686.0, 48.0) - 0.46).abs() < 1e-6);
        assert_eq!(wander_fraction(686.0, 1920.0), 0.0);
        // Live+ 10 ms (D = 480): 100 Hz trails, 150 Hz partly (measured 0.18).
        assert!((wander_fraction(480.0, 480.0) - 0.46).abs() < 1e-6);
        assert!((wander_fraction(320.0, 480.0) - 0.18).abs() < 0.05);
        assert_eq!(wander_fraction(0.0, 213.0), 0.0);
        assert_eq!(wander_fraction(f32::NAN, 213.0), 0.0);
    }

    #[test]
    fn avg_latency_adds_the_mean_wander() {
        let sr = 48_000.0;
        let mut a = AvgDelay::default();
        assert_eq!(a.expected_ms(4.44), None, "nothing sung yet");
        // Live Alto/Tenor, 200 Hz being corrected: 4.44 + 0.46 · 5 ms.
        a.update(Some(&at(200.0, 30.0)), 0.016, sr, 213);
        let live = a.expected_ms(4.44).unwrap();
        assert!((live - 6.74).abs() < 0.01, "{live}");
        // Unvoiced frames and silence hold the average.
        a.update(Some(&PitchFrame::default()), 1.0, sr, 213);
        a.update(None, 1.0, sr, 213);
        assert!((a.expected_ms(4.44).unwrap() - 6.74).abs() < 0.01);
        // It follows the singer over a few seconds of voiced time.
        for _ in 0..600 {
            a.update(Some(&at(100.0, -40.0)), 1.0 / 60.0, sr, 213);
        }
        let low = a.expected_ms(4.44).unwrap();
        assert!((low - 9.04).abs() < 0.1, "100 Hz adds 4.6 ms: {low}");
        // A new engine (Studio, D = 1920) starts over: no wander there.
        a.update(Some(&at(100.0, -40.0)), 0.016, sr, 1920);
        assert!((a.expected_ms(40.0).unwrap() - 40.0).abs() < 1e-4);
        // Nothing shifted (amount 0, or already in tune): no wander.
        let mut still = AvgDelay::default();
        still.update(Some(&at(200.0, 0.0)), 0.016, sr, 213);
        assert!((still.expected_ms(4.44).unwrap() - 4.44).abs() < 1e-6);
    }

    #[test]
    fn latency_tooltip_explains_reported_avg_and_buffer() {
        let t = latency_tooltip(LatencyInfo {
            reported_samples: 213,
            sample_rate: 48_000.0,
            avg_ms: Some(6.74),
            period_ms: Some(5.0),
            extra_ms: 2.3,
            block: 128,
            max_block: 512,
        });
        assert!(t.contains("Reported: 4.4 ms (213 samples"), "{t}");
        assert!(t.contains("Heard avg: 6.7 ms"), "{t}");
        assert!(t.contains("200 Hz") && t.contains("adds 2.3 ms"), "{t}");
        assert!(t.contains("Host buffer: 128 samples (2.7 ms)"), "{t}");
    }

    /// HI-1 / HI-3: the manual restart is offered once a change has been
    /// pending for a second, and the clock starts over for a new change.
    #[test]
    fn pending_change_becomes_overdue_after_a_second() {
        let t0 = Instant::now();
        let s = |secs: f32| t0 + Duration::from_secs_f32(secs);
        let mut p = PendingSince::default();
        p.update(true, s(0.0));
        assert!(!p.overdue(s(0.5)));
        p.update(true, s(0.9));
        assert!(p.overdue(s(1.0)), "offer the manual restart after 1 s");
        p.update(false, s(2.0));
        assert!(!p.overdue(s(30.0)));
        p.update(true, s(30.0));
        assert!(!p.overdue(s(30.5)));
        assert!(p.overdue(s(31.05)));
    }

    /// A host GUI context that counts parameter edits.
    #[derive(Default)]
    struct TestGui {
        edits: std::sync::atomic::AtomicUsize,
    }

    impl nice_plug::context::gui::GuiContextInner for TestGui {
        fn plugin_api(&self) -> PluginApi {
            PluginApi::Clap
        }
        unsafe fn raw_begin_set_parameter(&self, _param: ParamPtr) {}
        unsafe fn raw_set_parameter_normalized(&self, _param: ParamPtr, _normalized: f32) {
            self.edits.fetch_add(1, Ordering::Relaxed);
        }
        unsafe fn raw_end_set_parameter(&self, _param: ParamPtr) {}
        fn get_state(&self) -> PluginState {
            PluginState {
                version: String::new(),
                params: Default::default(),
                fields: Default::default(),
            }
        }
        fn set_state(&self, _state: PluginState) {}
        fn request_restart(&self) {}
    }

    /// One left click at `pos` through real egui input (move, press,
    /// release over three frames).
    fn click(ctx: &egui::Context, app: &mut SkatotuneEditor, pos: egui::Pos2) {
        let frame = |events: Vec<egui::Event>| egui::RawInput {
            screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, vec2(WIDTH, HEIGHT))),
            events,
            ..Default::default()
        };
        let button = |pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        for events in [
            vec![egui::Event::PointerMoved(pos)],
            vec![button(true)],
            vec![button(false)],
        ] {
            run_frame(ctx, app, frame(events));
        }
    }

    /// One editor frame; the texture updates are marked handled (nothing
    /// renders here, and epaint asserts on dropped ones in debug builds).
    fn run_frame(ctx: &egui::Context, app: &mut SkatotuneEditor, input: egui::RawInput) {
        let mut out = ctx.run_ui(input, |ui| app.draw(ui));
        out.textures_delta.clear();
    }

    fn key_press(ctx: &egui::Context, app: &mut SkatotuneEditor, key: egui::Key) {
        let event = |pressed| egui::Event::Key {
            key,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        for events in [vec![event(true)], vec![event(false)]] {
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, vec2(WIDTH, HEIGHT))),
                events,
                ..Default::default()
            };
            run_frame(ctx, app, input);
        }
    }

    /// The LABS button opens the experiments menu; a row flips its
    /// experiment in the saved Labs settings (never a host parameter);
    /// Escape, the button again, or a click elsewhere closes it.
    #[test]
    fn labs_menu_toggles_experiments() {
        let params = Arc::new(SkatotuneParams::default());
        let mut app = SkatotuneEditor::new(params.clone(), Arc::new(SharedState::default()));
        let host = Arc::new(TestGui::default());
        let ctx = egui::Context::default();
        app.setup(&ctx, GuiContext::new(host.clone()));
        run_frame(&ctx, &mut app, egui::RawInput::default());

        let header = Rect::from_min_size(egui::Pos2::ZERO, vec2(WIDTH, HEADER_H));
        let button = labs_button_rect(header).center();
        click(&ctx, &mut app, button);
        assert!(app.labs_open, "the button opens the menu");

        // The menu drops from the header's lower edge (never across it),
        // right-aligned to the button; rows sit under its title: the ribbon
        // row first, the heat line second.
        let origin = labs_menu::menu_origin(labs_button_rect(header), header.bottom());
        assert_eq!(origin.y, HEADER_H + labs_menu::MENU_GAP);
        assert_eq!(
            origin.x + labs_menu::MENU_W,
            labs_button_rect(header).right()
        );
        let row = |i: f32| {
            egui::pos2(
                900.0,
                origin.y + labs_menu::TITLE_H + labs_menu::ROW_H * (i + 0.5),
            )
        };
        assert_eq!(params.labs(), crate::labs::LabsSettings::default());
        click(&ctx, &mut app, row(1.0));
        assert!(params.labs().heat_line, "heat line switched on");
        click(&ctx, &mut app, row(0.0));
        assert!(!params.labs().correction_ribbon, "ribbon switched off");
        assert!(app.labs_open, "rows keep the menu open");
        assert_eq!(host.edits.load(Ordering::Relaxed), 0, "no host parameter");

        key_press(&ctx, &mut app, egui::Key::Escape);
        assert!(!app.labs_open, "Escape closes");
        click(&ctx, &mut app, button);
        assert!(app.labs_open);
        click(&ctx, &mut app, button);
        assert!(!app.labs_open, "the button closes it again");
        click(&ctx, &mut app, button);
        click(&ctx, &mut app, egui::pos2(600.0, 400.0));
        assert!(!app.labs_open, "a click elsewhere closes");
        assert_eq!(
            params.labs(),
            crate::labs::LabsSettings {
                correction_ribbon: false,
                heat_line: true
            }
        );
    }

    /// HI-4 GUI: while a learned scale is in use the keyboard edits it (not
    /// the note overrides, which do not apply then), and CLEAR forgets it.
    #[test]
    fn learned_scale_is_edited_and_cleared_from_the_sidebar() {
        let params = Arc::new(SkatotuneParams::default());
        params
            .learned_scale
            .store(crate::learn::pack(&[0, 4, 7]), Ordering::Relaxed);
        let mut app = SkatotuneEditor::new(params.clone(), Arc::new(SharedState::default()));
        let host = Arc::new(TestGui::default());
        let ctx = egui::Context::default();
        app.setup(&ctx, GuiContext::new(host.clone()));
        run_frame(&ctx, &mut app, egui::RawInput::default());

        // Sidebar layout: keyboard rows of 24 px from y = 230, B at the top,
        // so D (pitch class 2) is row 9; the CLEAR button sits at (136, 218).
        let d_row = egui::pos2(100.0, 230.0 + 9.0 * 24.0 + 12.0);
        click(&ctx, &mut app, d_row);
        assert_eq!(
            params.learned_states(),
            crate::learn::unpack(crate::learn::pack(&[0, 2, 4, 7])),
            "D added to the learned scale"
        );
        assert_eq!(
            host.edits.load(Ordering::Relaxed),
            0,
            "no note override was touched"
        );
        click(&ctx, &mut app, egui::pos2(136.0, 218.0));
        assert_eq!(params.learned_states(), None, "CLEAR");

        // Without a learned scale the same click cycles D's override again.
        click(&ctx, &mut app, d_row);
        assert_eq!(host.edits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn editor_app_is_cheap_until_opened() {
        let app = SkatotuneEditor::new(
            Arc::new(SkatotuneParams::default()),
            Arc::new(SharedState::default()),
        );
        assert_eq!(app.history.capacity(), 0);
        assert!(app.gui.is_none());
    }
}
