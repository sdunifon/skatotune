//! Correction strength and its colour scale (Labs: correction ribbon and
//! heat-tinted output line).
//!
//! **Strength** is how far the plugin moved the pitch, in cents:
//! `c = output − detected − 100 · transpose`. Transpose is a deliberate
//! shift, not a correction, so it is taken out. Frames carry no transpose of
//! their own, so the editor stamps each one with the Transpose parameter's
//! value as it drains the telemetry
//! ([`PitchHistory::set_transpose`](super::history::PitchHistory::set_transpose)):
//! turning Transpose does not re-judge the history already on screen.
//! Everything else the plugin does to the pitch counts, on purpose: retuning
//! toward the target, but also vibrato scaled by the Vibrato knob and the
//! synth vibrato, because that too is pitch the singer did not sing.
//!
//! **Colour** encodes the magnitude `|c|`, so it is a *sequential* scale: one
//! hue (the accent's, OKLCH h ≈ 170.5°), ordered by lightness, from OKLCH
//! L 0.60 up to the accent itself (L 0.83), chroma as high as sRGB allows up
//! to the accent's own (0.154). Anchors at 0, 10, 25, 50 and 100 c sit evenly
//! along the ramp, which spends most of its range on the small corrections a
//! natural setting makes while hard tune still saturates.
//!
//! - The **ribbon** shades the ramp colour at a rising opacity, so on a lane
//!   it runs from "nothing" at 0 c to a solid mid-tone teal at 100 c. Its
//!   strongest step stays well below the accent output line (2.5:1), so the
//!   line always reads on top of it.
//! - The **heat line** uses only the upper part of the same ramp (L 0.64 →
//!   0.83) in four ordinal steps, ΔL ≥ 0.06 apart: the line has to stay a
//!   line on every surface it crosses, the target bars included, and its
//!   strongest step is the normal accent output colour.
//!
//! Status colours (warn/bad) and the bypass, MIDI and centre hues keep their
//! meanings and are not used here.

use egui::Color32;
use skato_core::{PitchFrame, frame_flags};

/// `|c|` (cents) of the ramp anchors.
pub const STOP_CENTS: [f32; 5] = [0.0, 10.0, 25.0, 50.0, 100.0];

/// Opaque ramp colour at each anchor (OKLCH L 0.60 → 0.83 in equal steps).
/// Every sRGB channel rises from step to step, so blending neighbours in
/// sRGB (the ramp's own lerp, egui's vertex colours) keeps lightness monotone.
pub const STOP_COLORS: [Color32; 5] = [
    Color32::from_rgb(0x03, 0x96, 0x76), // L 0.60
    Color32::from_rgb(0x05, 0xaa, 0x86), // L 0.66
    Color32::from_rgb(0x08, 0xbe, 0x96), // L 0.71
    Color32::from_rgb(0x0a, 0xd3, 0xa6), // L 0.77
    Color32::from_rgb(0x2e, 0xe6, 0xb8), // L 0.83, the accent
];

/// Ribbon opacity at each anchor: almost nothing in tune, a solid band at
/// 100 c. Shaded areas stay translucent so the lanes read through them.
pub const RIBBON_ALPHA: [f32; 5] = [0.06, 0.32, 0.42, 0.50, 0.56];

/// The heat line's ordinal steps: the upper part of the ramp, OKLCH L 0.643,
/// 0.705, 0.766 and 0.827 (the accent). Step `k` covers `|c|` from
/// `STOP_CENTS[k]` up to the next anchor; the last one everything from 50 c.
pub const HEAT_COLORS: [Color32; 4] = [
    Color32::from_rgb(0x01, 0xa5, 0x82),
    Color32::from_rgb(0x03, 0xbb, 0x94),
    Color32::from_rgb(0x04, 0xd1, 0xa5),
    Color32::from_rgb(0x2e, 0xe6, 0xb8),
];

/// Number of heat-line steps.
pub const HEAT_STEPS: u8 = HEAT_COLORS.len() as u8;

/// The heat line follows the strongest correction within this many seconds
/// (centred): about one vibrato cycle, so flattened vibrato reads as one
/// strongly corrected stretch rather than a dashed line.
pub const HEAT_WINDOW_S: f64 = 0.18;

/// The ribbon's colour is averaged over this many pixel columns on each side,
/// so a tall ribbon does not flicker column to column.
pub const RIBBON_SMOOTH_COLS: f32 = 2.0;

/// Correction of one frame, cents (see the module docs). `None` unless the
/// frame is voiced with both pitches present and finite.
pub fn correction_cents(f: &PitchFrame, transpose_st: f32) -> Option<f32> {
    let voiced = f.flags & frame_flags::VOICED != 0;
    // `> 0.0` is false for NaN, so NaN pitches are rejected here too.
    if !(voiced && f.detected > 0.0 && f.output > 0.0) {
        return None;
    }
    let c = f.output - f.detected - 100.0 * transpose_st;
    c.is_finite().then_some(c)
}

/// Position on the ramp (0 = in tune, 1 = 100 c or more) of a correction of
/// `cents` (either sign). `None` for NaN.
pub fn ramp_pos(cents: f32) -> Option<f32> {
    if cents.is_nan() {
        return None;
    }
    let c = cents.abs();
    let last = STOP_CENTS.len() - 1;
    for (i, pair) in STOP_CENTS.windows(2).enumerate() {
        let [lo, hi] = pair else { continue };
        if c < *hi {
            let frac = (c - lo) / (hi - lo);
            return Some((i as f32 + frac) / last as f32);
        }
    }
    Some(1.0)
}

/// Segment of the stop table and the fraction inside it for ramp position
/// `t` (clamped; NaN counts as 0).
fn segment(t: f32) -> (usize, f32) {
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    let last = STOP_COLORS.len() - 1;
    let x = t * last as f32;
    let i = (x.floor() as usize).min(last - 1);
    (i, x - i as f32)
}

fn lerp_u8(a: u8, b: u8, f: f32) -> u8 {
    (f32::from(a) + (f32::from(b) - f32::from(a)) * f).round() as u8
}

/// Lerp between two opaque colours in sRGB (what egui's own vertex-colour
/// interpolation does between two neighbouring vertices).
pub fn lerp_color(a: Color32, b: Color32, f: f32) -> Color32 {
    Color32::from_rgb(
        lerp_u8(a.r(), b.r(), f),
        lerp_u8(a.g(), b.g(), f),
        lerp_u8(a.b(), b.b(), f),
    )
}

/// Opaque ramp colour at position `t` (0..=1, clamped).
pub fn ramp_color(t: f32) -> Color32 {
    let (i, f) = segment(t);
    match (STOP_COLORS.get(i), STOP_COLORS.get(i + 1)) {
        (Some(a), Some(b)) => lerp_color(*a, *b, f),
        _ => STOP_COLORS[0],
    }
}

/// Ribbon opacity at position `t` (0..=1, clamped).
pub fn ribbon_alpha(t: f32) -> f32 {
    let (i, f) = segment(t);
    match (RIBBON_ALPHA.get(i), RIBBON_ALPHA.get(i + 1)) {
        (Some(a), Some(b)) => a + (b - a) * f,
        _ => RIBBON_ALPHA[0],
    }
}

/// Ribbon fill at position `t`: the ramp colour at the ribbon opacity
/// (premultiplied, ready for a vertex).
pub fn ribbon_color(t: f32) -> Color32 {
    super::theme::alpha(ramp_color(t), ribbon_alpha(t))
}

/// Heat-line step (0..HEAT_STEPS) for a correction of `cents` (either
/// sign): the anchor interval it falls in. NaN counts as in tune.
pub fn heat_step(cents: f32) -> u8 {
    let c = if cents.is_nan() { 0.0 } else { cents.abs() };
    let above = STOP_CENTS[1..usize::from(HEAT_STEPS)]
        .iter()
        .filter(|&&s| c >= s)
        .count();
    above as u8
}

/// Colour of a heat-line step (clamped to the last).
pub fn heat_color(step: u8) -> Color32 {
    HEAT_COLORS[usize::from(step.min(HEAT_STEPS - 1))]
}

/// Ramp position where heat step `step` starts (its lowest `|c|`).
pub fn heat_step_start(step: u8) -> f32 {
    let i = usize::from(step.min(HEAT_STEPS - 1));
    STOP_CENTS.get(i).and_then(|c| ramp_pos(*c)).unwrap_or(0.0)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::theme;
    use super::*;

    /// OKLab of an sRGB colour (Björn Ottosson's matrices).
    pub(crate) fn oklab(c: Color32) -> [f32; 3] {
        let lin = |v: u8| {
            let s = f32::from(v) / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        let (r, g, b) = (lin(c.r()), lin(c.g()), lin(c.b()));
        let l = (0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
        let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
        let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
        [
            0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
            1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
            0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
        ]
    }

    fn hue(c: Color32) -> f32 {
        let [_, a, b] = oklab(c);
        b.atan2(a).to_degrees().rem_euclid(360.0)
    }

    /// WCAG contrast ratio of two opaque colours.
    pub(crate) fn contrast(a: Color32, b: Color32) -> f32 {
        let lum = |c: Color32| {
            let lin = |v: u8| {
                let s = f32::from(v) / 255.0;
                if s <= 0.04045 {
                    s / 12.92
                } else {
                    ((s + 0.055) / 1.055).powf(2.4)
                }
            };
            0.2126 * lin(c.r()) + 0.7152 * lin(c.g()) + 0.0722 * lin(c.b())
        };
        let (x, y) = (lum(a), lum(b));
        (x.max(y) + 0.05) / (x.min(y) + 0.05)
    }

    /// Premultiplied `src` over opaque `dst`, in gamma space (egui's blend).
    pub(crate) fn over(src: Color32, dst: Color32) -> Color32 {
        let k = 1.0 - f32::from(src.a()) / 255.0;
        let ch = |s: u8, d: u8| (f32::from(s) + f32::from(d) * k).round() as u8;
        Color32::from_rgb(
            ch(src.r(), dst.r()),
            ch(src.g(), dst.g()),
            ch(src.b(), dst.b()),
        )
    }

    fn voiced(detected: f32, output: f32) -> PitchFrame {
        PitchFrame {
            detected,
            output,
            flags: frame_flags::VOICED,
            ..PitchFrame::default()
        }
    }

    #[test]
    fn strength_excludes_transpose_only() {
        assert_eq!(correction_cents(&voiced(6880.0, 6900.0), 0.0), Some(20.0));
        // +2 st transpose: the 200 c shift is not correction.
        assert_eq!(correction_cents(&voiced(6880.0, 7100.0), 2.0), Some(20.0));
        assert_eq!(correction_cents(&voiced(6900.0, 6700.0), -2.0), Some(0.0));
        // Pulled down 30 c while transposed up an octave.
        assert_eq!(correction_cents(&voiced(6030.0, 7200.0), 12.0), Some(-30.0));
    }

    #[test]
    fn strength_needs_a_voiced_frame_with_both_pitches() {
        let mut unvoiced = voiced(6900.0, 6950.0);
        unvoiced.flags = 0;
        assert_eq!(correction_cents(&unvoiced, 0.0), None);
        assert_eq!(correction_cents(&voiced(0.0, 6900.0), 0.0), None);
        assert_eq!(correction_cents(&voiced(6900.0, 0.0), 0.0), None);
        assert_eq!(correction_cents(&voiced(f32::NAN, 6900.0), 0.0), None);
        assert_eq!(correction_cents(&voiced(6900.0, f32::NAN), 0.0), None);
        assert_eq!(correction_cents(&voiced(6900.0, f32::INFINITY), 0.0), None);
        assert_eq!(correction_cents(&voiced(6900.0, 6950.0), f32::NAN), None);
    }

    #[test]
    fn ramp_position_hits_the_anchors_and_clamps() {
        for (i, c) in STOP_CENTS.iter().enumerate() {
            let t = ramp_pos(*c).unwrap();
            assert!((t - i as f32 / 4.0).abs() < 1e-6, "{c} c -> {t}");
        }
        assert_eq!(ramp_pos(-25.0), ramp_pos(25.0), "sign does not matter");
        assert_eq!(ramp_pos(400.0), Some(1.0));
        assert_eq!(ramp_pos(f32::INFINITY), Some(1.0));
        assert_eq!(ramp_pos(f32::NAN), None, "NaN gets no colour");
        let mut prev = 0.0;
        for k in 0..=2400 {
            let t = ramp_pos(k as f32 * 0.05).unwrap();
            assert!(t >= prev && (0.0..=1.0).contains(&t));
            prev = t;
        }
    }

    #[test]
    fn ramp_lightness_and_opacity_rise_monotonically() {
        let mut prev_l = f32::NEG_INFINITY;
        let mut prev_a = f32::NEG_INFINITY;
        let mut prev_comp = f32::NEG_INFINITY;
        for k in 0..=1000 {
            let t = k as f32 / 1000.0;
            let l = oklab(ramp_color(t))[0];
            // u8 rounding can repeat a colour between two close samples.
            assert!(l >= prev_l - 1e-6, "lightness falls at t = {t}");
            let a = ribbon_alpha(t);
            assert!(a > prev_a, "opacity falls at t = {t}");
            // What the eye sees: the ribbon over a lane gets lighter too
            // (up to the u8 rounding of a premultiplied colour: one unit
            // of alpha more without one unit of colour more).
            let comp = oklab(over(ribbon_color(t), theme::LANE_ON))[0];
            assert!(comp >= prev_comp - 5e-3, "ribbon on a lane darkens at {t}");
            prev_l = l;
            prev_a = a;
            prev_comp = comp;
        }
        // Equal lightness steps between the anchors, ≈ 0.057 each.
        let ls: Vec<f32> = STOP_COLORS.iter().map(|c| oklab(*c)[0]).collect();
        for pair in ls.windows(2) {
            assert!((pair[1] - pair[0] - 0.057).abs() < 0.006, "{ls:?}");
        }
        assert!(ribbon_alpha(0.0) < 0.1 && ribbon_alpha(1.0) >= 0.5);
        let comp = |t: f32| oklab(over(ribbon_color(t), theme::LANE_ON))[0];
        for k in 0..4 {
            let (a, b) = (k as f32 / 4.0, (k + 1) as f32 / 4.0);
            assert!(comp(b) - comp(a) > 0.05, "anchors {a} -> {b}");
        }
        assert_eq!(STOP_COLORS[4], theme::ACCENT, "the ramp ends at the accent");
    }

    #[test]
    fn ramp_is_one_hue() {
        let accent = hue(theme::ACCENT);
        for c in STOP_COLORS.into_iter().chain(HEAT_COLORS) {
            assert!((hue(c) - accent).abs() < 1.5, "{c:?}");
        }
    }

    #[test]
    fn ramp_clamps_and_ignores_nan() {
        assert_eq!(ramp_color(-1.0), STOP_COLORS[0]);
        assert_eq!(ramp_color(f32::NAN), STOP_COLORS[0]);
        assert_eq!(ramp_color(0.0), STOP_COLORS[0]);
        assert_eq!(ramp_color(1.0), STOP_COLORS[4]);
        assert_eq!(ramp_color(7.0), STOP_COLORS[4]);
        assert_eq!(ramp_color(0.5), STOP_COLORS[2]);
        assert_eq!(ribbon_alpha(2.0), RIBBON_ALPHA[4]);
        assert_eq!(ribbon_alpha(f32::NAN), RIBBON_ALPHA[0]);
        assert_eq!(heat_step(f32::NAN), 0);
        assert_eq!(heat_step(0.0), 0);
        assert_eq!(heat_step(9.9), 0);
        assert_eq!(heat_step(-10.0), 1, "sign does not matter");
        assert_eq!(heat_step(24.0), 1);
        assert_eq!(heat_step(25.0), 2);
        assert_eq!(heat_step(50.0), 3);
        assert_eq!(heat_step(f32::INFINITY), HEAT_STEPS - 1);
        assert_eq!(heat_color(0), HEAT_COLORS[0]);
        assert_eq!(heat_color(200), HEAT_COLORS[3]);
        assert_eq!(heat_step_start(0), 0.0);
        assert_eq!(heat_step_start(3), 0.75);
        assert_eq!(ribbon_color(1.0).a(), 143, "56 % at 100 c");
    }

    /// The target-lane highlight: LANE_ON under a 6 % accent wash.
    fn highlighted_lane() -> Color32 {
        over(theme::alpha(theme::ACCENT, 0.06), theme::LANE_ON)
    }

    /// Every surface the output line crosses: background, lanes, and the
    /// target bars as drawn with a Labs colouring on (fill and outline).
    fn line_surfaces() -> Vec<(&'static str, Color32)> {
        vec![
            ("BG", theme::BG),
            ("LANE_ON", theme::LANE_ON),
            ("LANE_ON_BLACK", theme::LANE_ON_BLACK),
            ("LANE_OFF", theme::LANE_OFF),
            ("LANE_BYPASS", theme::LANE_BYPASS),
            ("highlighted lane", highlighted_lane()),
            (
                "bar fill",
                over(theme::alpha(theme::ACCENT, 0.13), theme::LANE_ON),
            ),
            (
                "Labs bar fill",
                over(theme::alpha(theme::ACCENT, 0.05), theme::LANE_ON),
            ),
            (
                "Labs bar outline",
                over(theme::alpha(theme::ACCENT, 0.34), theme::LANE_ON),
            ),
        ]
    }

    /// The heat line is an ordinal ramp (data-viz rule): one hue, lightness
    /// rising by ≥ 0.06 per step, and even the darkest step stands ≥ 2:1
    /// off every surface it can cross; the brightest is the accent.
    #[test]
    fn heat_steps_are_an_ordinal_ramp_that_reads_everywhere() {
        let ls: Vec<f32> = HEAT_COLORS.iter().map(|c| oklab(*c)[0]).collect();
        assert!(ls.windows(2).all(|p| p[1] - p[0] >= 0.06), "{ls:?}");
        assert_eq!(HEAT_COLORS[3], theme::ACCENT);
        for (name, surface) in line_surfaces() {
            let low = contrast(HEAT_COLORS[0], surface);
            assert!(low >= 2.0, "{name}: {low}");
        }
        let mut prev = 0;
        for k in 0..=1200 {
            let s = heat_step(k as f32 * 0.1);
            assert!(s >= prev && s < HEAT_STEPS);
            prev = s;
        }
    }

    /// The ribbon's 10 c step already stands off a lane; its strongest step
    /// stays clearly below the output line drawn over it.
    #[test]
    fn ribbon_reads_but_stays_under_the_output_line() {
        let on_lane = |c: f32| over(ribbon_color(ramp_pos(c).unwrap()), theme::LANE_ON);
        let weak = contrast(on_lane(10.0), theme::LANE_ON);
        assert!(weak >= 1.6, "10 c: {weak}");
        let strong = contrast(on_lane(100.0), theme::LANE_ON);
        assert!(strong >= 4.0, "100 c: {strong}");
        for c in [50.0, 100.0] {
            let vs_line = contrast(on_lane(c), theme::ACCENT);
            assert!(vs_line >= 2.5, "{c} c vs output: {vs_line}");
        }
        assert!(
            contrast(on_lane(0.0), theme::LANE_ON) < 1.1,
            "in tune recedes"
        );
    }
}
