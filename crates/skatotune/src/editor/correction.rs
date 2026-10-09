//! Correction strength and its colour scale (Labs: correction ribbon and
//! heat-tinted output line, in the colours of the chosen theme).
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
//! **Colour** encodes the magnitude `|c|`, so it is a *sequential* scale:
//! anchors at 0, 10, 25, 50 and 100 c sit evenly along a ramp of five colours,
//! which spends most of its range on the small corrections a natural setting
//! makes while hard tune still saturates. Which five colours is the
//! [`CorrectionTheme`] chosen in the LABS menu: each theme is a [`Palette`]
//! (the ramp and the heat line's four steps), and everything else here is
//! shared by all of them.
//!
//! - The **ribbon** shades the ramp colour at a rising opacity
//!   ([`RIBBON_ALPHA`], the same for every theme), so on a lane it runs from
//!   "nothing" at 0 c to a solid mid-tone band at 100 c. Its strongest step
//!   stays well below the output line, so the line always reads on top of it.
//! - The **heat line** uses four ordinal steps, one per anchor interval
//!   (0-10, 10-25, 25-50 and 50+ c). The line has to stay a line on every
//!   surface it crosses, so no step is below 2:1 against the background or
//!   any lane (the target bars as well, for Teal).
//!
//! # The themes
//!
//! Each is held to the same numbers by the unit tests: neighbouring steps
//! at least ΔE 8 apart in OKLab (single-hue Teal, told apart by lightness
//! alone, keeps the 6 it always had), the ribbon readable on a lane and below
//! the line, the white input and blue centre traces readable over it, and,
//! for the themes that claim to be colour-blind safe, steps still ΔE 6 apart
//! under simulated protan and deutan vision.
//!
//! - **Teal**, the default and the look before themes existed: one hue (the
//!   accent's, OKLCH h ≈ 170.5°) ordered by lightness, from OKLCH L 0.60 up to
//!   the accent itself (L 0.83), chroma as high as sRGB allows up to the
//!   accent's own (0.154). The heat line uses only the upper part (L 0.64 →
//!   0.83, ΔL ≥ 0.06 apart) and its strongest step is the normal accent output
//!   colour.
//! - **Green → red**, a traffic light: green, yellow, orange, red. Not
//!   colour-blind safe: the ends look alike to red-green colour blindness (the
//!   menu says so).
//! - **Viridis** and **Inferno**, entries of matplotlib's published colour
//!   tables (perceptually even, colour-blind safe, lightness rising with the
//!   correction), without their near-black ends, which would vanish on the
//!   lanes.
//! - **Blue → orange**: cool to warm along the blue-yellow axis, which
//!   colour-blind viewers keep.
//!
//! Status colours (warn/bad) and the bypass, MIDI and centre hues keep their
//! meanings in the rest of the editor; the themes borrow warm hues freely,
//! being a Labs experiment.

use egui::Color32;
use skato_core::{PitchFrame, frame_flags};

use crate::labs::CorrectionTheme;

/// `|c|` (cents) of the ramp anchors.
pub const STOP_CENTS: [f32; 5] = [0.0, 10.0, 25.0, 50.0, 100.0];

/// Ribbon opacity at each anchor: almost nothing in tune, a solid band at
/// 100 c. Shaded areas stay translucent so the lanes read through them.
pub const RIBBON_ALPHA: [f32; 5] = [0.06, 0.32, 0.42, 0.50, 0.56];

/// Number of heat-line steps: step `k` covers `|c|` from `STOP_CENTS[k]` up to
/// the next anchor; the last one everything from 50 c.
pub const HEAT_STEPS: u8 = 4;

/// The heat line follows the strongest correction within this many seconds
/// (centred): about one vibrato cycle, so flattened vibrato reads as one
/// strongly corrected stretch rather than a dashed line.
pub const HEAT_WINDOW_S: f64 = 0.18;

/// The ribbon's colour is averaged over this many pixel columns on each side,
/// so a tall ribbon does not flicker column to column.
pub const RIBBON_SMOOTH_COLS: f32 = 2.0;

/// One theme's colours (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Palette {
    /// Opaque ramp colour at each anchor of [`STOP_CENTS`]. Neighbours are
    /// blended in sRGB (the ramp's own lerp, egui's vertex colours), so
    /// every stop is a corner the gradient bends at.
    pub stops: [Color32; STOP_CENTS.len()],
    /// The heat line's colour at each step.
    pub heat: [Color32; HEAT_STEPS as usize],
    /// Whether the legend's Output swatch blends the first heat step straight
    /// into the last. Right for a single hue (Teal, where it is the original
    /// swatch); a ramp that sweeps across hues would turn to mud between its
    /// ends, so the swatch passes through every step instead.
    pub swatch_blends_ends: bool,
}

/// Teal, the default. Ramp: OKLCH L 0.60 → 0.83 in equal steps; every sRGB
/// channel rises from step to step, so blending neighbours keeps lightness
/// monotone. Heat line: the ramp's upper part, OKLCH L 0.643, 0.705, 0.766
/// and 0.827 (the accent). Unchanged since before themes existed (a test
/// holds it to the old values).
pub const TEAL: Palette = Palette {
    stops: [
        Color32::from_rgb(0x03, 0x96, 0x76), // L 0.60
        Color32::from_rgb(0x05, 0xaa, 0x86), // L 0.66
        Color32::from_rgb(0x08, 0xbe, 0x96), // L 0.71
        Color32::from_rgb(0x0a, 0xd3, 0xa6), // L 0.77
        Color32::from_rgb(0x2e, 0xe6, 0xb8), // L 0.83, the accent
    ],
    heat: [
        Color32::from_rgb(0x01, 0xa5, 0x82),
        Color32::from_rgb(0x03, 0xbb, 0x94),
        Color32::from_rgb(0x04, 0xd1, 0xa5),
        Color32::from_rgb(0x2e, 0xe6, 0xb8),
    ],
    swatch_blends_ends: true,
};

/// Green → red, a traffic light along OKLCH hue 148° → 28° (green, lime,
/// yellow, orange, red), lightness peaking at the yellow. The heat line is the
/// four signals: green below 10 c, then yellow, orange and red from 50 c. The
/// heat steps are a touch lighter than the ribbon's, so each stands off the
/// band it hangs from.
pub const GREEN_RED: Palette = Palette {
    stops: [
        Color32::from_rgb(0x4b, 0xd7, 0x6d),
        Color32::from_rgb(0xba, 0xe0, 0x35),
        Color32::from_rgb(0xf6, 0xd7, 0x32),
        Color32::from_rgb(0xf9, 0x7d, 0x14),
        Color32::from_rgb(0xfc, 0x46, 0x3c),
    ],
    heat: [
        Color32::from_rgb(0x37, 0xd1, 0x6c),
        Color32::from_rgb(0xef, 0xd6, 0x2f),
        Color32::from_rgb(0xfe, 0x97, 0x3c),
        Color32::from_rgb(0xfd, 0x57, 0x4b),
    ],
    swatch_blends_ends: false,
};

/// Viridis (matplotlib; van der Walt & Smith, CC0): entries 77, 121, 166, 210
/// and 255 of its 256-colour table, blue-purple → teal → green → yellow. The
/// heat line is the last four (teal, green, lime, yellow). The table's
/// dark-purple start is left out: it would vanish on the lanes.
pub const VIRIDIS: Palette = Palette {
    stops: [
        Color32::from_rgb(0x34, 0x60, 0x8d),
        Color32::from_rgb(0x23, 0x8a, 0x8d),
        Color32::from_rgb(0x2f, 0xb4, 0x7c),
        Color32::from_rgb(0x89, 0xd5, 0x48),
        Color32::from_rgb(0xfd, 0xe7, 0x25),
    ],
    heat: [
        Color32::from_rgb(0x23, 0x8a, 0x8d),
        Color32::from_rgb(0x2f, 0xb4, 0x7c),
        Color32::from_rgb(0x89, 0xd5, 0x48),
        Color32::from_rgb(0xfd, 0xe7, 0x25),
    ],
    swatch_blends_ends: false,
};

/// Inferno (matplotlib; van der Walt & Smith, CC0): the table's entries at
/// 3/8, 1/2, 5/8, 3/4 and 7/8 (indices 96, 128, 159, 191, 223): magenta →
/// crimson → red-orange → orange → gold. The heat line is the last four. The
/// black start and the cream end are left out: one vanishes on the lanes, the
/// other is nearly the white of the input trace.
pub const INFERNO: Palette = Palette {
    stops: [
        Color32::from_rgb(0x8a, 0x22, 0x6a),
        Color32::from_rgb(0xbc, 0x37, 0x54),
        Color32::from_rgb(0xe3, 0x59, 0x33),
        Color32::from_rgb(0xf9, 0x8c, 0x0a),
        Color32::from_rgb(0xf9, 0xc9, 0x32),
    ],
    heat: [
        Color32::from_rgb(0xbc, 0x37, 0x54),
        Color32::from_rgb(0xe3, 0x59, 0x33),
        Color32::from_rgb(0xf9, 0x8c, 0x0a),
        Color32::from_rgb(0xf9, 0xc9, 0x32),
    ],
    swatch_blends_ends: false,
};

/// Blue → orange, cool to warm: indigo-blue → periwinkle → lavender → sand →
/// orange (blue-yellow axis: red-green colour-blind viewers keep it). The blue
/// leans violet so it stays clear of the centre trace's sky blue, and the heat
/// line's orange is lighter than the ribbon's so it stands off the sand-brown
/// band under it.
pub const BLUE_ORANGE: Palette = Palette {
    stops: [
        Color32::from_rgb(0x53, 0x5c, 0xda),
        Color32::from_rgb(0x80, 0x83, 0xe2),
        Color32::from_rgb(0xb8, 0xae, 0xcd),
        Color32::from_rgb(0xe9, 0xb8, 0x60),
        Color32::from_rgb(0xfe, 0x7f, 0x25),
    ],
    heat: [
        Color32::from_rgb(0x6b, 0x79, 0xf9),
        Color32::from_rgb(0xb5, 0xa7, 0xe2),
        Color32::from_rgb(0xe6, 0xcb, 0x89),
        Color32::from_rgb(0xff, 0x90, 0x45),
    ],
    swatch_blends_ends: false,
};

/// The colours of `theme`.
pub fn palette(theme: CorrectionTheme) -> &'static Palette {
    match theme {
        CorrectionTheme::Teal => &TEAL,
        CorrectionTheme::GreenRed => &GREEN_RED,
        CorrectionTheme::Viridis => &VIRIDIS,
        CorrectionTheme::Inferno => &INFERNO,
        CorrectionTheme::BlueOrange => &BLUE_ORANGE,
    }
}

impl Palette {
    /// Opaque ramp colour at position `t` (0..=1, clamped; NaN counts as 0).
    pub fn ramp_color(&self, t: f32) -> Color32 {
        let (i, f) = segment(t);
        match (self.stops.get(i), self.stops.get(i + 1)) {
            (Some(a), Some(b)) => lerp_color(*a, *b, f),
            _ => self.stops[0],
        }
    }

    /// Ribbon fill at position `t`: the ramp colour at the ribbon opacity
    /// (premultiplied, ready for a vertex).
    pub fn ribbon_color(&self, t: f32) -> Color32 {
        super::theme::alpha(self.ramp_color(t), ribbon_alpha(t))
    }

    /// Colour of a heat-line step (clamped to the last).
    pub fn heat_color(&self, step: u8) -> Color32 {
        self.heat[usize::from(step.min(HEAT_STEPS - 1))]
    }
}

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
    let last = STOP_CENTS.len() - 1;
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

/// Ribbon opacity at position `t` (0..=1, clamped).
pub fn ribbon_alpha(t: f32) -> f32 {
    let (i, f) = segment(t);
    match (RIBBON_ALPHA.get(i), RIBBON_ALPHA.get(i + 1)) {
        (Some(a), Some(b)) => a + (b - a) * f,
        _ => RIBBON_ALPHA[0],
    }
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

/// Ramp position where heat step `step` starts (its lowest `|c|`).
pub fn heat_step_start(step: u8) -> f32 {
    let i = usize::from(step.min(HEAT_STEPS - 1));
    STOP_CENTS.get(i).and_then(|c| ramp_pos(*c)).unwrap_or(0.0)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::theme;
    use super::*;

    /// Linear-light value of an sRGB channel.
    fn linear(v: u8) -> f32 {
        let s = f32::from(v) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    }

    /// OKLab of linear-light RGB (Björn Ottosson's matrices).
    fn oklab_linear([r, g, b]: [f32; 3]) -> [f32; 3] {
        let l = (0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
        let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
        let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
        [
            0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
            1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
            0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
        ]
    }

    /// OKLab of an sRGB colour.
    pub(crate) fn oklab(c: Color32) -> [f32; 3] {
        oklab_linear([linear(c.r()), linear(c.g()), linear(c.b())])
    }

    fn hue(c: Color32) -> f32 {
        let [_, a, b] = oklab(c);
        b.atan2(a).to_degrees().rem_euclid(360.0)
    }

    /// Distance of two colours in OKLab, ×100 (the data-viz validator's ΔE),
    /// optionally as seen through a colour-vision deficiency: Machado,
    /// Oliveira & Fernandes (2009) at severity 1.0, in linear light.
    fn delta_e(a: Color32, b: Color32, deficiency: Option<&[[f32; 3]; 3]>) -> f32 {
        let see = |c: Color32| {
            let rgb = [linear(c.r()), linear(c.g()), linear(c.b())];
            let rgb = match deficiency {
                None => rgb,
                Some(m) => m.map(|row| {
                    (row[0] * rgb[0] + row[1] * rgb[1] + row[2] * rgb[2]).clamp(0.0, 1.0)
                }),
            };
            oklab_linear(rgb)
        };
        let (a, b) = (see(a), see(b));
        100.0 * ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    const PROTAN: [[f32; 3]; 3] = [
        [0.152_286, 1.052_583, -0.204_868],
        [0.114_503, 0.786_281, 0.099_216],
        [-0.003_882, -0.048_116, 1.051_998],
    ];
    const DEUTAN: [[f32; 3]; 3] = [
        [0.367_322, 0.860_646, -0.227_968],
        [0.280_085, 0.672_501, 0.047_413],
        [-0.011_820, 0.042_940, 0.968_881],
    ];

    /// WCAG contrast ratio of two opaque colours.
    pub(crate) fn contrast(a: Color32, b: Color32) -> f32 {
        let lum =
            |c: Color32| 0.2126 * linear(c.r()) + 0.7152 * linear(c.g()) + 0.0722 * linear(c.b());
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

    /// Every theme with its colours.
    fn themes() -> impl Iterator<Item = (CorrectionTheme, &'static Palette)> {
        CorrectionTheme::ALL.into_iter().map(|t| (t, palette(t)))
    }

    /// The Teal ramp as it was before themes existed (the old module's
    /// constants and blend, copied here): what the default theme must still
    /// draw, to the last bit.
    mod before_themes {
        use egui::Color32;

        pub const STOP_COLORS: [Color32; 5] = [
            Color32::from_rgb(0x03, 0x96, 0x76),
            Color32::from_rgb(0x05, 0xaa, 0x86),
            Color32::from_rgb(0x08, 0xbe, 0x96),
            Color32::from_rgb(0x0a, 0xd3, 0xa6),
            Color32::from_rgb(0x2e, 0xe6, 0xb8),
        ];
        pub const HEAT_COLORS: [Color32; 4] = [
            Color32::from_rgb(0x01, 0xa5, 0x82),
            Color32::from_rgb(0x03, 0xbb, 0x94),
            Color32::from_rgb(0x04, 0xd1, 0xa5),
            Color32::from_rgb(0x2e, 0xe6, 0xb8),
        ];
        pub const RIBBON_ALPHA: [f32; 5] = [0.06, 0.32, 0.42, 0.50, 0.56];

        pub fn ramp_color(t: f32) -> Color32 {
            let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
            let x = t * 4.0;
            let i = (x.floor() as usize).min(3);
            let f = x - i as f32;
            let (a, b) = (STOP_COLORS[i], STOP_COLORS[i + 1]);
            let ch =
                |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * f).round() as u8;
            Color32::from_rgb(ch(a.r(), b.r()), ch(a.g(), b.g()), ch(a.b(), b.b()))
        }

        pub fn ribbon_alpha(t: f32) -> f32 {
            let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
            let x = t * 4.0;
            let i = (x.floor() as usize).min(3);
            let f = x - i as f32;
            RIBBON_ALPHA[i] + (RIBBON_ALPHA[i + 1] - RIBBON_ALPHA[i]) * f
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
    fn heat_steps_follow_the_anchor_intervals() {
        assert_eq!(heat_step(f32::NAN), 0);
        assert_eq!(heat_step(0.0), 0);
        assert_eq!(heat_step(9.9), 0);
        assert_eq!(heat_step(-10.0), 1, "sign does not matter");
        assert_eq!(heat_step(24.0), 1);
        assert_eq!(heat_step(25.0), 2);
        assert_eq!(heat_step(49.9), 2);
        assert_eq!(heat_step(50.0), 3);
        assert_eq!(heat_step(f32::INFINITY), HEAT_STEPS - 1);
        assert_eq!(heat_step_start(0), 0.0);
        assert_eq!(heat_step_start(3), 0.75);
        assert_eq!(heat_step_start(200), 0.75, "clamped to the last step");
        let mut prev = 0;
        for k in 0..=1200 {
            let s = heat_step(k as f32 * 0.1);
            assert!(s >= prev && s < HEAT_STEPS);
            prev = s;
        }
        assert_eq!(RIBBON_ALPHA.len(), STOP_CENTS.len());
        assert_eq!(usize::from(HEAT_STEPS), STOP_CENTS.len() - 1);
    }

    // --- every theme -----------------------------------------------------

    /// A ramp passes exactly through its declared colours at the anchors
    /// (0, 10, 25, 50, 100 c), and between two anchors every channel lies
    /// between the two colours.
    #[test]
    fn every_ramp_hits_its_anchors_and_blends_between() {
        for (theme, p) in themes() {
            for (i, c) in STOP_CENTS.iter().enumerate() {
                let at = p.ramp_color(ramp_pos(*c).unwrap());
                assert_eq!(at, p.stops[i], "{theme:?} at {c} c");
                // The same through the ribbon: this colour at the ribbon's opacity.
                let t = ramp_pos(*c).unwrap();
                assert_eq!(
                    p.ribbon_color(t),
                    theme::alpha(p.stops[i], RIBBON_ALPHA[i]),
                    "{theme:?} ribbon at {c} c"
                );
            }
            for (i, pair) in p.stops.windows(2).enumerate() {
                let [a, b] = pair else { continue };
                for k in 1..10 {
                    let f = k as f32 / 10.0;
                    let t = (i as f32 + f) / 4.0;
                    let c = p.ramp_color(t);
                    // (`t` rounds, so allow one unit where a blend sits on .5)
                    let expect = lerp_color(*a, *b, f);
                    for (got, want) in [
                        (c.r(), expect.r()),
                        (c.g(), expect.g()),
                        (c.b(), expect.b()),
                    ] {
                        assert!(
                            got.abs_diff(want) <= 1,
                            "{theme:?} segment {i}: {c:?} vs {expect:?}"
                        );
                    }
                    for (ch, lo, hi) in [
                        (c.r(), a.r(), b.r()),
                        (c.g(), a.g(), b.g()),
                        (c.b(), a.b(), b.b()),
                    ] {
                        assert!(ch >= lo.min(hi) && ch <= lo.max(hi), "{theme:?} {c:?}");
                    }
                }
            }
        }
    }

    /// Past the ends the ramp stays on its end colours, either sign of
    /// correction gives one colour, and NaN gets none.
    #[test]
    fn every_ramp_clamps_and_gives_nan_no_colour() {
        for (theme, p) in themes() {
            let (first, last) = (p.stops[0], p.stops[4]);
            assert_eq!(p.ramp_color(-1.0), first, "{theme:?}");
            assert_eq!(p.ramp_color(f32::NEG_INFINITY), first, "{theme:?}");
            assert_eq!(p.ramp_color(f32::NAN), first, "{theme:?}");
            assert_eq!(p.ramp_color(7.0), last, "{theme:?}");
            assert_eq!(p.ramp_color(f32::INFINITY), last, "{theme:?}");
            let at = |cents: f32| ramp_pos(cents).map(|t| p.ramp_color(t));
            assert_eq!(at(0.0), Some(first), "{theme:?}");
            assert_eq!(at(-0.0), Some(first), "{theme:?}");
            assert_eq!(at(100.0), Some(last), "{theme:?}");
            assert_eq!(at(100.5), Some(last), "{theme:?}");
            assert_eq!(at(-4000.0), Some(last), "{theme:?}");
            assert_eq!(at(f32::INFINITY), Some(last), "{theme:?}");
            assert_eq!(at(f32::NAN), None, "{theme:?}: NaN gets no colour");
            assert_eq!(at(-37.0), at(37.0), "{theme:?}: the sign does not matter");
            // Ribbon opacity clamps the same way.
            assert_eq!(p.ribbon_color(2.0).a(), p.ribbon_color(1.0).a());
            assert_eq!(p.ribbon_color(-2.0).a(), p.ribbon_color(0.0).a());
            // Heat steps: the first four anchors' colours, clamped to the last.
            for step in 0..HEAT_STEPS {
                assert_eq!(p.heat_color(step), p.heat[usize::from(step)]);
            }
            assert_eq!(p.heat_color(HEAT_STEPS), p.heat[3], "{theme:?}");
            assert_eq!(p.heat_color(255), p.heat[3], "{theme:?}");
            assert_eq!(p.heat_color(heat_step(f32::NAN)), p.heat[0], "{theme:?}");
            assert_eq!(p.ribbon_color(1.0).a(), 143, "{theme:?}: 56 % at 100 c");
        }
        assert_eq!(ribbon_alpha(2.0), RIBBON_ALPHA[4]);
        assert_eq!(ribbon_alpha(f32::NAN), RIBBON_ALPHA[0]);
    }

    /// The four steps of a heat line are four different colours, clearly
    /// apart (OKLab ΔE ≥ 8, the colour-blind-safety validator's target; the
    /// single-hue Teal, which is told apart by lightness alone, keeps the
    /// 6 it always had), and so are the ramp's own stops (5.5 for Teal);
    /// and no two themes share a ramp.
    #[test]
    fn heat_steps_are_distinct() {
        for (theme, p) in themes() {
            for (i, a) in p.heat.iter().enumerate() {
                for b in &p.heat[i + 1..] {
                    assert_ne!(a, b, "{theme:?}");
                }
            }
            let (heat_floor, stop_floor) = if theme == CorrectionTheme::Teal {
                (6.0, 5.5)
            } else {
                (8.0, 8.0)
            };
            for pair in p.heat.windows(2) {
                let d = delta_e(pair[0], pair[1], None);
                assert!(
                    d >= heat_floor,
                    "{theme:?}: {:?} -> {:?}: {d}",
                    pair[0],
                    pair[1]
                );
            }
            for pair in p.stops.windows(2) {
                let d = delta_e(pair[0], pair[1], None);
                assert!(
                    d >= stop_floor,
                    "{theme:?}: {:?} -> {:?}: {d}",
                    pair[0],
                    pair[1]
                );
            }
        }
        let all: Vec<&Palette> = themes().map(|(_, p)| p).collect();
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.stops, b.stops);
                assert_ne!(a.heat, b.heat);
            }
        }
    }

    /// Every surface a heat line crosses at the editor's size: the
    /// background, every lane, and the highlighted target lane.
    fn lane_surfaces() -> Vec<(&'static str, Color32)> {
        vec![
            ("BG", theme::BG),
            ("LANE_ON", theme::LANE_ON),
            ("LANE_ON_BLACK", theme::LANE_ON_BLACK),
            ("LANE_OFF", theme::LANE_OFF),
            ("LANE_BYPASS", theme::LANE_BYPASS),
            ("highlighted lane", highlighted_lane()),
        ]
    }

    /// The target-lane highlight: LANE_ON under a 6 % accent wash.
    fn highlighted_lane() -> Color32 {
        over(theme::alpha(theme::ACCENT, 0.06), theme::LANE_ON)
    }

    /// Every surface the output line crosses: the lanes above, and the
    /// target bars as drawn with a Labs colouring on (fill and outline).
    fn line_surfaces() -> Vec<(&'static str, Color32)> {
        let mut v = lane_surfaces();
        v.extend([
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
        ]);
        v
    }

    /// Whichever theme is chosen, no step of the heat line sinks into the
    /// background or a lane: ≥ 2:1 against all of them.
    #[test]
    fn every_heat_step_reads_on_every_lane() {
        for (theme, p) in themes() {
            for (step, c) in p.heat.iter().enumerate() {
                for (name, surface) in lane_surfaces() {
                    let ratio = contrast(*c, surface);
                    assert!(ratio >= 2.0, "{theme:?} step {step} on {name}: {ratio}");
                }
            }
        }
    }

    /// The themes that claim to be colour-blind safe keep every neighbouring
    /// pair of ramp stops and heat steps apart (ΔE ≥ 6, the validator's
    /// floor) for protan and deutan vision. The traffic light does not
    /// claim it (its tooltip says so).
    #[test]
    fn colour_blind_safe_themes_keep_their_steps_apart() {
        for theme in [
            CorrectionTheme::Viridis,
            CorrectionTheme::Inferno,
            CorrectionTheme::BlueOrange,
        ] {
            let p = palette(theme);
            for colours in [&p.stops[..], &p.heat[..]] {
                for pair in colours.windows(2) {
                    for (kind, m) in [("protan", &PROTAN), ("deutan", &DEUTAN)] {
                        let d = delta_e(pair[0], pair[1], Some(m));
                        assert!(
                            d >= 6.0,
                            "{theme:?} {kind}: {:?} -> {:?}: {d}",
                            pair[0],
                            pair[1]
                        );
                    }
                }
            }
        }
        // The check can fail: a traffic light's lime and yellow merge.
        let gr = &GREEN_RED;
        let worst = gr
            .stops
            .windows(2)
            .map(|p| delta_e(p[0], p[1], Some(&PROTAN)))
            .fold(f32::INFINITY, f32::min);
        assert!(worst < 6.0, "{worst}");
    }

    /// Each ramp runs the way its name says, step after step: lightness
    /// rises with the correction (Teal, Viridis, Inferno), red-ness does
    /// (Green → red, along OKLab a), and yellow-ness (Blue → orange, along b).
    #[test]
    fn every_ramp_runs_from_its_first_colour_to_its_last() {
        let axis = |theme: CorrectionTheme, c: Color32| {
            let [l, a, b] = oklab(c);
            match theme {
                CorrectionTheme::GreenRed => a,
                CorrectionTheme::BlueOrange => b,
                _ => l,
            }
        };
        for (theme, p) in themes() {
            for colours in [&p.stops[..], &p.heat[..]] {
                let v: Vec<f32> = colours.iter().map(|c| axis(theme, *c)).collect();
                assert!(
                    v.windows(2).all(|w| w[1] > w[0]),
                    "{theme:?}: {v:?} for {colours:?}"
                );
            }
        }
        // And the ends are what the names promise.
        let [_, a, _] = oklab(GREEN_RED.heat[0]);
        let [_, a_end, _] = oklab(GREEN_RED.heat[3]);
        assert!(a < -0.1 && a_end > 0.1, "green to red: {a} .. {a_end}");
        let [_, _, b] = oklab(BLUE_ORANGE.heat[0]);
        let [_, _, b_end] = oklab(BLUE_ORANGE.heat[3]);
        assert!(b < -0.1 && b_end > 0.1, "blue to orange: {b} .. {b_end}");
        // Viridis starts purple-blue and ends yellow; inferno red to yellow.
        assert!((230.0..300.0).contains(&hue(VIRIDIS.stops[0])));
        assert!((90.0..115.0).contains(&hue(VIRIDIS.stops[4])));
        assert!((5.0..40.0).contains(&hue(INFERNO.heat[0])));
        assert!((80.0..115.0).contains(&hue(INFERNO.heat[3])));
    }

    /// The ribbon of every theme: even a 10 c correction stands off the lane,
    /// the strongest one is a clear band, none comes near the weight of the
    /// output line drawn over it (the plain accent line, or the heat line's
    /// step for that correction), and in tune nothing is drawn.
    #[test]
    fn every_ribbon_reads_but_stays_under_the_output_line() {
        for (theme, p) in themes() {
            let on_lane = |c: f32| over(p.ribbon_color(ramp_pos(c).unwrap()), theme::LANE_ON);
            let weak = contrast(on_lane(10.0), theme::LANE_ON);
            assert!(weak >= 1.25, "{theme:?} 10 c: {weak}");
            let strong = contrast(on_lane(100.0), theme::LANE_ON);
            assert!(strong >= 2.0, "{theme:?} 100 c: {strong}");
            for c in [50.0, 100.0] {
                let vs_line = contrast(on_lane(c), theme::ACCENT);
                assert!(
                    vs_line >= 2.0,
                    "{theme:?} {c} c vs the accent line: {vs_line}"
                );
            }
            // The heat line over the ribbon it hangs off: step `k` covers the
            // anchors `k` and `k + 1` (the last one, 50 c and 100 c).
            for (k, step) in p.heat.iter().enumerate() {
                for anchor in [k.min(3), (k + 1).min(4)] {
                    let band = on_lane(STOP_CENTS[anchor]);
                    let vs_band = contrast(*step, band);
                    assert!(
                        vs_band >= 2.0,
                        "{theme:?} step {k} over the {} c ribbon: {vs_band}",
                        STOP_CENTS[anchor]
                    );
                }
            }
            assert!(
                contrast(on_lane(0.0), theme::LANE_ON) < 1.15,
                "{theme:?}: in tune recedes"
            );
            // The white input line (at full confidence) and the blue centre
            // line stay readable across the whole band: the input by
            // lightness, the centre (a blue, so by hue rather than lightness
            // against a yellow or orange band) by OKLab distance.
            for c in STOP_CENTS {
                let band = on_lane(c);
                let input = over(theme::alpha(theme::DETECTED, 0.75), band);
                let centre = over(theme::alpha(theme::CENTRE, 0.8), band);
                let ratio = contrast(input, band);
                assert!(ratio >= 2.0, "{theme:?} {c} c: input {ratio}");
                let apart = delta_e(centre, band, None);
                assert!(apart >= 15.0, "{theme:?} {c} c: centre {apart}");
            }
        }
    }

    #[test]
    fn a_theme_has_one_palette_and_only_teal_blends_its_swatch() {
        assert_eq!(palette(CorrectionTheme::default()), &TEAL);
        for (theme, p) in themes() {
            assert_eq!(p.swatch_blends_ends, theme == CorrectionTheme::Teal);
        }
    }

    // --- Teal: the look before themes ------------------------------------

    /// The default theme draws what the code drew before themes existed:
    /// the same ramp at every position, the same ribbon colours, the same
    /// heat colours (the constants are the old module's, bit for bit).
    #[test]
    fn teal_is_exactly_the_ramp_from_before_themes() {
        assert_eq!(TEAL.stops, before_themes::STOP_COLORS);
        assert_eq!(TEAL.heat, before_themes::HEAT_COLORS);
        assert_eq!(RIBBON_ALPHA, before_themes::RIBBON_ALPHA);
        for k in 0..=1000 {
            let t = k as f32 / 1000.0;
            assert_eq!(TEAL.ramp_color(t), before_themes::ramp_color(t), "t = {t}");
            assert_eq!(ribbon_alpha(t), before_themes::ribbon_alpha(t), "t = {t}");
            assert_eq!(
                TEAL.ribbon_color(t),
                theme::alpha(before_themes::ramp_color(t), before_themes::ribbon_alpha(t)),
                "t = {t}"
            );
        }
        for t in [-3.0, f32::NAN, 1.5, f32::INFINITY] {
            assert_eq!(TEAL.ramp_color(t), before_themes::ramp_color(t));
        }
        for step in 0..=255 {
            assert_eq!(
                TEAL.heat_color(step),
                before_themes::HEAT_COLORS[usize::from(step.min(3))]
            );
        }
    }

    #[test]
    fn teal_lightness_and_opacity_rise_monotonically() {
        let mut prev_l = f32::NEG_INFINITY;
        let mut prev_a = f32::NEG_INFINITY;
        let mut prev_comp = f32::NEG_INFINITY;
        for k in 0..=1000 {
            let t = k as f32 / 1000.0;
            let l = oklab(TEAL.ramp_color(t))[0];
            // u8 rounding can repeat a colour between two close samples.
            assert!(l >= prev_l - 1e-6, "lightness falls at t = {t}");
            let a = ribbon_alpha(t);
            assert!(a > prev_a, "opacity falls at t = {t}");
            // What the eye sees: the ribbon over a lane gets lighter too
            // (up to the u8 rounding of a premultiplied colour: one unit
            // of alpha more without one unit of colour more).
            let comp = oklab(over(TEAL.ribbon_color(t), theme::LANE_ON))[0];
            assert!(comp >= prev_comp - 5e-3, "ribbon on a lane darkens at {t}");
            prev_l = l;
            prev_a = a;
            prev_comp = comp;
        }
        // Equal lightness steps between the anchors, ≈ 0.057 each.
        let ls: Vec<f32> = TEAL.stops.iter().map(|c| oklab(*c)[0]).collect();
        for pair in ls.windows(2) {
            assert!((pair[1] - pair[0] - 0.057).abs() < 0.006, "{ls:?}");
        }
        assert!(ribbon_alpha(0.0) < 0.1 && ribbon_alpha(1.0) >= 0.5);
        let comp = |t: f32| oklab(over(TEAL.ribbon_color(t), theme::LANE_ON))[0];
        for k in 0..4 {
            let (a, b) = (k as f32 / 4.0, (k + 1) as f32 / 4.0);
            assert!(comp(b) - comp(a) > 0.05, "anchors {a} -> {b}");
        }
        assert_eq!(TEAL.stops[4], theme::ACCENT, "the ramp ends at the accent");
    }

    #[test]
    fn teal_is_one_hue() {
        let accent = hue(theme::ACCENT);
        for c in TEAL.stops.into_iter().chain(TEAL.heat) {
            assert!((hue(c) - accent).abs() < 1.5, "{c:?}");
        }
    }

    /// Teal's heat line is an ordinal ramp (data-viz rule): one hue,
    /// lightness rising by ≥ 0.06 per step, and even the darkest step stands
    /// ≥ 2:1 off every surface it can cross, the target bars included; the
    /// brightest is the accent.
    #[test]
    fn teal_heat_steps_are_an_ordinal_ramp_that_reads_everywhere() {
        let ls: Vec<f32> = TEAL.heat.iter().map(|c| oklab(*c)[0]).collect();
        assert!(ls.windows(2).all(|p| p[1] - p[0] >= 0.06), "{ls:?}");
        assert_eq!(TEAL.heat[3], theme::ACCENT);
        for (name, surface) in line_surfaces() {
            let low = contrast(TEAL.heat[0], surface);
            assert!(low >= 2.0, "{name}: {low}");
        }
    }

    /// Teal's ribbon: the 10 c step already stands off a lane; its strongest
    /// step stays clearly below the output line drawn over it.
    #[test]
    fn teal_ribbon_reads_but_stays_under_the_output_line() {
        let on_lane = |c: f32| over(TEAL.ribbon_color(ramp_pos(c).unwrap()), theme::LANE_ON);
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
