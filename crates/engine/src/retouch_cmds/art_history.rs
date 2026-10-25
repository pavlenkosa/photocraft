//! Art History Brush: stylized sensor strokes painted from a history source.
//!
//! Photoshop describes the tool as painting "with stylized sensor strokes that simulate the look
//! of different paint styles", taking their colours from a history state or snapshot, with a
//! Style preset, an Area and a Tolerance on the options bar. Adobe's help defines the last two:
//! Area is "the area covered by the sensor strokes" and "low tolerance restricts strokes to areas
//! that vary little from the source state or snapshot [while] high tolerance lets strokes spread
//! across areas that vary greatly from the source". Clean-room reading implemented here:
//!
//! * Each dab the brush engine places along the stroke (spacing, size and flow dynamics included)
//!   grows a **sensor stroke**: a chain of small soft dabs that starts at the dab centre, follows
//!   the source's contours (the direction across the source's luminance gradient), and may curve
//!   and wander inside the Area disc around the dab. Every sensor dab paints the source colour
//!   under its own centre, so the source's colours are rearranged into painterly strokes.
//! * The Style preset sets the stroke's length, how far it may wander and how much it curves;
//!   `dab` paints one plain dab, `strict` follows the contour exactly.
//! * A sensor dab is skipped where the pre-stroke canvas has drifted further from the source than
//!   Tolerance allows (the mean colour-channel difference), so at 0 % strokes repaint only areas
//!   that still match the source and at 100 % they paint everywhere.
//! * The jitter comes from the counter-based stroke RNG (`seed`, dab index), so a replay of the
//!   same points paints the same strokes, as the Impressionist Pattern Stamp's dabs do.
//!
//! The sensor dabs are round soft dabs of the brush's hardness driven by the dab's flow; shaped
//! tips, texture and dual-brush tips are not applied to them (a partial in `brush-parity.md`).

use photocraft_color::{BlendMode, PixelFormat};
use photocraft_doc::LayerId;
use photocraft_geom::Rect;
use photocraft_paint::retouch::{Region, mix_back, over_native};
use photocraft_paint::{BrushContext, Dab, Stroke, dab_coverage, dabs, rng, tip_reach};
use photocraft_raster::{Surface, from_rgba_into, to_rgba};
use serde_json::{Value, json};

use super::{MAX_STROKE_PIXELS, bad, blend_param, damage_json, has_pixel_layer, history_source, num, parse_brush, reject_huge_stroke, run_stroke};
use crate::{Result, Session};

/// Jitter streams, private to this tool so they stay independent of the brush engine's.
const STREAM_FALLBACK: u64 = 121;
const STREAM_WANDER: u64 = 122;
const STREAM_CURVE: u64 = 123;
const STREAM_DIR: u64 = 124;

pub(super) fn enabled(s: &Session) -> std::result::Result<(), String> {
    has_pixel_layer(s)
}

/// Photoshop's Style presets. `(length, wander, curvature, sub-dab radius)` in units of the dab
/// radius (curvature in degrees): how far the sensor stroke reaches across the dab, how far it
/// may stray sideways, how much it bends along its length, and how thick the sensor dabs are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Style {
    TightShort,
    TightMedium,
    TightLong,
    LooseMedium,
    LooseLong,
    Dab,
    Strict,
    Loose,
}

impl Style {
    fn parse(s: &str, cmd: &str) -> Result<Self> {
        match s {
            "tightShort" => Ok(Self::TightShort),
            "tightMedium" => Ok(Self::TightMedium),
            "tightLong" => Ok(Self::TightLong),
            "looseMedium" => Ok(Self::LooseMedium),
            "looseLong" => Ok(Self::LooseLong),
            "dab" => Ok(Self::Dab),
            "strict" => Ok(Self::Strict),
            "loose" => Ok(Self::Loose),
            other => Err(bad(
                cmd,
                format!("unknown style `{other}` (tightShort|tightMedium|tightLong|looseMedium|looseLong|dab|strict|loose)"),
            )),
        }
    }

    /// `None` for Dab: one plain dab instead of a sensor stroke.
    fn shape(self) -> Option<(f32, f32, f32, f32)> {
        match self {
            Self::TightShort => Some((2.0, 0.15, 10.0, 0.5)),
            Self::TightMedium => Some((4.0, 0.2, 15.0, 0.5)),
            Self::TightLong => Some((6.0, 0.25, 20.0, 0.5)),
            Self::LooseMedium => Some((4.0, 0.7, 30.0, 0.5)),
            Self::LooseLong => Some((8.0, 1.0, 40.0, 0.5)),
            Self::Loose => Some((6.0, 0.85, 35.0, 0.5)),
            Self::Strict => Some((4.0, 0.0, 0.0, 0.5)),
            Self::Dab => None,
        }
    }

    /// The largest sensor dab's radius for a brush tip of `radius`, for the coverage budget.
    fn sub_radius(self, radius: f32) -> f32 {
        radius * self.shape().map_or(1.0, |s| s.3)
    }
}

/// A parsed `paint.artHistoryBrush`.
pub(super) struct Setup {
    stroke: Stroke,
    id: Option<LayerId>,
    mode: BlendMode,
    style: Style,
    area: f32,
    tolerance: f32,
    seed: u64,
    src: Surface,
}

impl Setup {
    pub(super) fn parse(s: &Session, p: &Value) -> Result<Self> {
        const CMD: &str = "paint.artHistoryBrush";
        let (stroke, id) = parse_brush(s, p, CMD)?;
        let mode = blend_param(p, CMD)?;
        let style = Style::parse(p.get("style").and_then(Value::as_str).unwrap_or("tightMedium"), CMD)?;
        let area = num(p, "area", 50.0);
        if !area.is_finite() {
            return Err(bad(CMD, "`area` must be finite"));
        }
        let area = area.clamp(1.0, 1500.0);
        let tolerance = num(p, "tolerance", 0.0).clamp(0.0, 100.0);
        reject_huge_stroke(CMD, &stroke)?;
        // The UI passes the stroke's first point, so a live preview (if one comes later) and the
        // commit agree; without it the whole stroke seeds the jitter, as the other tools do.
        let seed = p.get("seed").and_then(Value::as_u64).unwrap_or_else(|| {
            let bytes = p.get("points").map(std::string::ToString::to_string).unwrap_or_default();
            rng::seed_from_bytes(bytes.as_bytes())
        });
        let src = history_source(s, id, p, CMD)?;
        Ok(Self { stroke, id, mode, style, area, tolerance, seed, src })
    }

    /// The source in the target's format.
    fn source(&self, fmt: PixelFormat) -> Surface {
        if self.src.format() == fmt {
            self.src.clone()
        } else {
            self.src.convert(fmt)
        }
    }

    /// Paint the stroke onto `surf`: the brush engine's dabs, each grown into sensor strokes.
    /// Returns the mixed-back rectangle.
    fn apply(&self, surf: &mut Surface, doc_bounds: Rect, sel: Option<&Surface>, lock: bool, cmd: &str) -> Result<Rect> {
        let ds = dabs(&self.stroke);
        if ds.is_empty() {
            return Ok(Rect::EMPTY);
        }
        let fmt = surf.format();
        let src = self.source(fmt);
        let ctx = BrushContext::new(&self.stroke.brush);
        let bounds = ds.iter().fold(Rect::EMPTY, |r, d| r.union(&ctx.dab_rect(d, false)));
        if bounds.is_empty() {
            return Ok(Rect::EMPTY);
        }
        // Sensor dabs land within the Area disc around their dab and stamp their own tip past
        // that, so the working copy grows by both.
        let sub = self.style.sub_radius(self.stroke.brush.size * 0.5);
        let halo = (f64::from(self.area) * 0.5 + f64::from(tip_reach(sub, self.stroke.brush.hardness))).ceil() as i32 + 1;
        let outer = bounds.inflate(halo);
        let pixels = u64::from(outer.width()).saturating_mul(u64::from(outer.height()));
        if pixels > MAX_STROKE_PIXELS {
            return Err(bad(cmd, format!("stroke too large (at most {MAX_STROKE_PIXELS} px)")));
        }
        let orig = Region::read(surf, outer);
        let mut work = orig.clone();
        {
            // The canvas is still the pre-stroke state while the dabs paint: the Tolerance gate
            // reads it, so strokes skip where the user has edited away from the source.
            let pre = &*surf;
            for d in &ds {
                paint_dab(&mut work, &src, pre, &fmt, doc_bounds, d, self.style, self.area, self.tolerance, self.stroke.brush.hardness, self.seed, self.mode);
            }
        }
        // The sensor strokes reach past the brush footprint, so the whole working area mixes back.
        mix_back(surf, &orig, &work, outer, self.stroke.brush.opacity, sel, lock);
        Ok(outer)
    }
}

pub(super) fn run(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "paint.artHistoryBrush";
    let st = Setup::parse(s, p)?;
    let dmg = run_stroke(s, "Art History Brush", st.id, p, |pre, surf, sel, lock| st.apply(surf, pre.bounds(), sel, lock, CMD))?;
    if let Some(state) = s.active_mut() {
        state.last_damage = Some(dmg);
    }
    Ok(json!({ "damage": damage_json(dmg) }))
}

/// The source's contour direction at `centre`: across the luminance gradient sampled on a ring of
/// `reach` px (the Area's radius: the scale the tool senses at). Flat sources take a seeded random
/// direction.
fn isophote(src: &Surface, fmt: &PixelFormat, centre: (f64, f64), reach: f64, doc_bounds: Rect, seed: u64, index: u64) -> f64 {
    let luma_at = |x: f64, y: f64| -> f64 {
        let (x, y) = (
            x.clamp(f64::from(doc_bounds.x0), f64::from((doc_bounds.x1 - 1).max(doc_bounds.x0))),
            y.clamp(f64::from(doc_bounds.y0), f64::from((doc_bounds.y1 - 1).max(doc_bounds.y0))),
        );
        let mut px = [0.0f32; 8];
        src.read_pixel(x.floor() as i32, y.floor() as i32, &mut px[..fmt.channels()]);
        let rgba = to_rgba(fmt, &px[..fmt.channels()]);
        0.212_6 * f64::from(rgba[0]) + 0.715_2 * f64::from(rgba[1]) + 0.072_2 * f64::from(rgba[2])
    };
    let (cx, cy) = centre;
    let centre_luma = luma_at(cx, cy);
    let (mut gx, mut gy) = (0.0f64, 0.0f64);
    for k in 0..8u32 {
        let a = f64::from(k) * std::f64::consts::FRAC_PI_4;
        let l = luma_at(cx + reach * a.cos(), cy + reach * a.sin()) - centre_luma;
        gx += l * a.cos();
        gy += l * a.sin();
    }
    (gx, gy) = (gx * 0.25, gy * 0.25);
    if !gx.is_finite() || !gy.is_finite() || gx.abs() + gy.abs() < 1.0e-3 {
        return f64::from(rng::rand_signed(seed, index, STREAM_FALLBACK)) * std::f64::consts::PI;
    }
    gy.atan2(gx) + std::f64::consts::FRAC_PI_2
}

/// Whether the canvas at `(x, y)` has drifted further from the source than `tolerance` allows:
/// the mean colour-channel difference (alpha excluded) over 0..1.
fn drifted(pre: &Surface, src: &Surface, fmt: &PixelFormat, x: i32, y: i32, tolerance: f32) -> bool {
    let colors = fmt.mode.color_channels();
    let mut acc = 0.0f32;
    for c in 0..colors {
        acc += (pre.sample_channel(x, y, c) - src.sample_channel(x, y, c)).abs();
    }
    acc / colors as f32 > tolerance / 100.0
}

/// One brush dab's sensor strokes, painted into the working copy.
#[allow(clippy::too_many_arguments)]
fn paint_dab(
    work: &mut Region,
    src: &Surface,
    pre: &Surface,
    fmt: &PixelFormat,
    doc_bounds: Rect,
    dab: &Dab,
    style: Style,
    area: f32,
    tolerance: f32,
    hardness: f32,
    seed: u64,
    mode: BlendMode,
) {
    let (cx, cy) = (dab.center.x, dab.center.y);
    let radius = f64::from(dab.radius.max(0.5));
    let (sub, len, wander, curve_deg) = match style.shape() {
        None => (radius, 0.0, 0.0, 0.0),
        Some(s) => (radius * f64::from(s.3), radius * f64::from(s.0), radius * f64::from(s.1), f64::from(s.2)),
    };
    let reach = (f64::from(area) * 0.5).max(0.0);
    let theta = isophote(src, fmt, (cx, cy), reach.clamp(1.0, 1500.0), doc_bounds, seed, dab.index);
    let step = (sub * 0.75).max(0.75);
    let n = if len <= 0.0 { 0 } else { (len / step).round().max(1.0) as u64 };
    let curve = curve_deg.to_radians() * f64::from(rng::rand_signed(seed, dab.index, STREAM_CURVE));
    // Strict follows the contour exactly: no direction jitter (and it must not
    // depend on the seed, or two seeds would produce different strict strokes).
    let dir = if wander > 0.0 {
        theta + f64::from(rng::rand_signed(seed, dab.index, STREAM_DIR)) * 0.35
    } else {
        theta
    };
    for k in 0..=n {
        let t = if n == 0 { 0.0 } else { k as f64 / n as f64 - 0.5 };
        let s = t * len;
        let ang = dir + curve * t * 2.0;
        let (ca, sa) = (ang.cos(), ang.sin());
        let jitter = if wander > 0.0 { f64::from(rng::rand_signed(seed, dab.index.wrapping_mul(64).wrapping_add(k), STREAM_WANDER)) } else { 0.0 };
        let w = jitter * wander;
        let (mut qx, mut qy) = (cx + s * ca - w * sa, cy + s * sa + w * ca);
        // Sensor strokes stay inside the Area disc around their dab.
        let (dx, dy) = (qx - cx, qy - cy);
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > reach {
            let f = if dist.is_finite() && dist > 0.0 { reach / dist } else { 0.0 };
            qx = cx + dx * f;
            qy = cy + dy * f;
        }
        let (ix, iy) = (qx.floor() as i32, qy.floor() as i32);
        if tolerance < 100.0 && drifted(pre, src, fmt, ix, iy, tolerance) {
            continue;
        }
        stamp(work, fmt, src, qx, qy, sub as f32, hardness, dab.alpha, mode);
    }
}

/// Stamp one sensor dab: the round soft tip's coverage blending the source colour under its
/// centre into the working copy.
#[allow(clippy::too_many_arguments)]
fn stamp(work: &mut Region, fmt: &PixelFormat, src: &Surface, qx: f64, qy: f64, radius: f32, hardness: f32, alpha: f32, mode: BlendMode) {
    let reach = f64::from(tip_reach(radius, hardness)) + 1.0;
    let rect = Rect::new(
        (qx - reach).floor() as i32,
        (qy - reach).floor() as i32,
        (qx + reach).ceil() as i32 + 1,
        (qy + reach).ceil() as i32 + 1,
    )
    .intersect(&work.rect);
    if rect.is_empty() {
        return;
    }
    let ch = work.ch;
    let mut color = [0.0f32; 8];
    src.read_pixel(qx.floor() as i32, qy.floor() as i32, &mut color[..ch]);
    let color = &color[..ch];
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            let dx = f64::from(x) + 0.5 - qx;
            let dy = f64::from(y) + 0.5 - qy;
            let dist = (dx * dx + dy * dy).sqrt() as f32;
            let k = dab_coverage(dist, radius, hardness) * alpha;
            if k <= 0.0 {
                continue;
            }
            let px = work.px_mut(x, y);
            if mode == BlendMode::Normal {
                over_native(fmt, px, color, k, false);
            } else {
                let d = to_rgba(fmt, px);
                let s = to_rgba(fmt, color);
                let o = photocraft_color::blend::composite(mode, d, s, k);
                let mut enc = [0.0f32; 8];
                from_rgba_into(fmt, o, &mut enc);
                px.copy_from_slice(&enc[..ch]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_geom::Point;
    use photocraft_paint::StrokePoint;

    fn surface(w: u32, h: u32) -> Surface {
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(Rect::new(0, 0, w as i32, h as i32), &[0.0, 0.0, 0.0, 1.0]);
        s
    }

    /// A diagonal gradient, so contour-following strokes cross changing colours.
    fn gradient(w: u32, h: u32) -> Surface {
        let mut s = surface(w, h);
        for y in 0..h {
            for x in 0..w {
                s.fill_rect(Rect::new(x as i32, y as i32, x as i32 + 1, y as i32 + 1), &[x as f32 / (w - 1) as f32, y as f32 / (h - 1) as f32, 0.5, 1.0]);
            }
        }
        s
    }

    #[test]
    fn a_sensor_stroke_rearranges_the_source_inside_the_dab() {
        let src = gradient(40, 20);
        let bounds = Rect::new(0, 0, 40, 20);
        let mut work = Region::read(&src, bounds);
        let dab = Dab::round(Point::new(20.0, 10.0), 4.0, 1.0);
        paint_dab(&mut work, &src, &src, &src.format(), bounds, &dab, Style::TightMedium, 50.0, 100.0, 1.0, 7, BlendMode::Normal);
        // The dab's footprint now carries the colours sampled along its sensor stroke, which
        // crossed the diagonal gradient: red and green both vary under it.
        let (mut min_r, mut max_r, mut painted) = (1.0f32, 0.0f32, 0u32);
        for y in 6..15 {
            for x in 14..27 {
                let px = work.px(x, y);
                if px[3] > 0.0 {
                    painted += 1;
                    min_r = min_r.min(px[0]);
                    max_r = max_r.max(px[0]);
                }
            }
        }
        assert!(painted > 20, "the dab painted its footprint ({painted} px)");
        assert!(max_r - min_r > 0.1, "the stroke spread the gradient's colours: {min_r}..{max_r}");
    }

    #[test]
    fn tolerance_skips_where_the_canvas_drifted_from_the_source() {
        let mut src = surface(40, 20);
        src.fill_rect(Rect::new(0, 0, 40, 20), &[1.0, 0.0, 0.0, 1.0]);
        // The canvas: red everywhere except a blue square edited after the source state.
        let mut pre = surface(40, 20);
        pre.fill_rect(Rect::new(0, 0, 40, 20), &[1.0, 0.0, 0.0, 1.0]);
        pre.fill_rect(Rect::new(14, 6, 26, 14), &[0.0, 0.0, 1.0, 1.0]);
        let bounds = Rect::new(0, 0, 40, 20);
        let run = |tolerance: f32| {
            let mut work = Region::read(&pre, bounds);
            let dab = Dab::round(Point::new(20.0, 10.0), 8.0, 1.0);
            paint_dab(&mut work, &src, &pre, &pre.format(), bounds, &dab, Style::Dab, 50.0, tolerance, 1.0, 7, BlendMode::Normal);
            work.px(20, 10)[2] // blue at the edited centre
        };
        assert!(run(0.0) > 0.9, "tolerance 0 leaves the edited pixel alone");
        assert!(run(100.0) < 0.1, "tolerance 100 paints over it");
    }

    #[test]
    fn styles_differ_and_replay_exactly() {
        let src = gradient(60, 30);
        let bounds = Rect::new(0, 0, 60, 30);
        let paint = |style: Style, seed: u64| {
            let mut work = Region::read(&src, bounds);
            let dab = Dab::round(Point::new(30.0, 15.0), 5.0, 1.0);
            paint_dab(&mut work, &src, &src, &src.format(), bounds, &dab, style, 60.0, 100.0, 1.0, seed, BlendMode::Normal);
            work.data.clone()
        };
        assert_eq!(paint(Style::TightMedium, 7), paint(Style::TightMedium, 7), "the same seed replays exactly");
        assert_ne!(paint(Style::TightMedium, 7), paint(Style::TightMedium, 8), "a different seed wanders differently");
        assert_ne!(paint(Style::TightMedium, 7), paint(Style::TightShort, 7), "styles shape the strokes");
        assert_ne!(paint(Style::TightMedium, 7), paint(Style::LooseLong, 7), "loose strokes wander");
        assert_ne!(paint(Style::TightMedium, 7), paint(Style::Dab, 7), "a dab is one plain dab");
        assert_eq!(paint(Style::Strict, 7), paint(Style::Strict, 8), "strict has no jitter to seed");
    }

    #[test]
    fn area_bounds_where_the_sensor_strokes_reach() {
        let src = gradient(80, 40);
        let bounds = Rect::new(0, 0, 80, 40);
        let run = |area: f32| {
            let mut work = Region::read(&src, bounds);
            let dab = Dab::round(Point::new(40.0, 20.0), 4.0, 1.0);
            paint_dab(&mut work, &src, &src, &src.format(), bounds, &dab, Style::TightMedium, area, 100.0, 1.0, 7, BlendMode::Normal);
            work.data.clone()
        };
        let small = run(4.0);
        let large = run(800.0);
        assert_ne!(small, large, "the Area spreads the sampled colours");
        assert_eq!(small.len(), large.len(), "both stay within the working copy");
    }

    #[test]
    fn a_stroke_of_dabs_applies_over_the_working_copy() {
        let src = gradient(60, 30);
        let mut work = Region::read(&src, Rect::new(0, 0, 60, 30));
        let brush = photocraft_paint::BrushSettings { size: 8.0, hardness: 1.0, ..Default::default() };
        let stroke = Stroke { brush, points: vec![StrokePoint::new(10.0, 15.0, 1.0), StrokePoint::new(50.0, 15.0, 1.0)] };
        for d in dabs(&stroke) {
            paint_dab(&mut work, &src, &src, &src.format(), Rect::new(0, 0, 60, 30), &d, Style::TightMedium, 50.0, 100.0, 1.0, 3, BlendMode::Normal);
        }
        // The stroke painted along its path: both ends changed away from the plain gradient.
        let changed = (0..60).filter(|x| work.px(*x, 15)[3] > 0.5 && (work.px(*x, 15)[0] - *x as f32 / 59.0).abs() > 1.0e-4).count();
        assert!(changed > 10, "the stroke left sensor strokes behind ({changed} px changed)");
    }
}
