#!/usr/bin/env python3
"""Generate the README charts as self-contained SVGs (no dependencies).

Every number below is copied from a measured source, cited next to it, so the
charts can be regenerated when the measurements change:

    python3 docs/images/make_charts.py

Colours follow a validated categorical palette, with light and dark variants
chosen by the viewer's colour scheme, so the charts read on GitHub in both
themes. Backgrounds are transparent.
"""
import math
import os

OUT = os.path.dirname(os.path.abspath(__file__))

STYLE = """
<style>
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  .title { font-size: 15px; font-weight: 600; fill: #1f2328; }
  .sub   { font-size: 12px; fill: #59636e; }
  .lab   { font-size: 12.5px; fill: #1f2328; }
  .val   { font-size: 12px; fill: #59636e; }
  .tick  { font-size: 11px; fill: #59636e; }
  .grid  { stroke: #d1d9e0; stroke-width: 1; }
  .axis  { stroke: #818b98; stroke-width: 1; }
  .ref   { stroke: #818b98; stroke-width: 1.5; stroke-dasharray: 4 3; }
  .s1 { fill: #2a78d6; } .s2 { fill: #eb6834; } .s3 { fill: #1baf7a; }
  @media (prefers-color-scheme: dark) {
    .title, .lab { fill: #f0f6fc; }
    .sub, .val, .tick { fill: #9198a1; }
    .grid { stroke: #2f353d; }
    .axis, .ref { stroke: #656c76; }
    .s1 { fill: #3987e5; } .s2 { fill: #d95926; } .s3 { fill: #199e70; }
  }
</style>
"""


def svg(width, height, body, label):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{label}">{STYLE}{body}</svg>\n'
    )


def text(x, y, s, cls, anchor="start"):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def rect(x, y, w, h, cls):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.0):.1f}" height="{h:.1f}" rx="2" class="{cls}"/>'


def fmt1(v):
    """One decimal, rounding halves up (5.85 -> 5.9), as the docs quote them."""
    return f"{math.floor(v * 10 + 0.5) / 10:.1f}"


def legend(x, y, items):
    out = []
    for cls, name in items:
        out.append(rect(x, y - 9, 11, 11, cls))
        out.append(text(x + 16, y, name, "lab"))
        x += 24 + 7.2 * len(name)
    return "".join(out)


def latency_chart():
    # Reported latency D at 48 kHz (docs/PERFORMANCE.md §1) and the heard
    # average while correcting on 40 real singers (vocadito, docs/STATUS.md §2,
    # v1.1 regression pass).
    rows = [
        ("Live-Min (Classic engine)", 1.00, 1.02),
        ("Live · Soprano", 3.10, 4.55),
        ("Live · Alto/Tenor (default)", 4.44, 6.59),
        ("Live · Low Male", 5.85, 8.73),
        ("Live+ (default 10 ms)", 10.00, 10.25),
    ]
    # Auto-Tune 2026, host-reported in Cubase (Sound On Sound, Jan 2026).
    refs = [(0.8, "Auto-Tune Classic 0.8"), (2.3, "Auto-Tune low-latency 2.3")]
    W, left, right, top = 760, 210, 40, 116
    rowh, barh = 44, 14
    xmax = 12.0
    plotw = W - left - right
    H = top + rowh * len(rows) + 46
    sx = lambda v: left + plotw * v / xmax
    b = [text(20, 28, "Plugin latency by mode (ms, lower is better)", "title"),
         text(20, 47, "Reported = what the DAW compensates. Heard = average delay of the corrected voice on 40 real singers.", "sub"),
         legend(20, 72, [("s1", "Reported"), ("s2", "Heard while correcting")])]
    for t in range(0, int(xmax) + 1, 2):
        b.append(f'<line x1="{sx(t):.1f}" y1="{top - 8}" x2="{sx(t):.1f}" y2="{top + rowh * len(rows)}" class="grid"/>')
        b.append(text(sx(t), top + rowh * len(rows) + 16, f"{t}", "tick", "middle"))
    for i, (name, rep, heard) in enumerate(rows):
        y = top + i * rowh
        b.append(text(left - 10, y + barh + 4, name, "lab", "end"))
        b.append(rect(left, y + 2, sx(rep) - left, barh, "s1"))
        b.append(text(sx(rep) + 5, y + barh - 1, fmt1(rep), "val"))
        b.append(rect(left, y + barh + 5, sx(heard) - left, barh, "s2"))
        b.append(text(sx(heard) + 5, y + 2 * barh + 2, fmt1(heard), "val"))
    for v, name in refs:
        b.append(f'<line x1="{sx(v):.1f}" y1="{top - 14}" x2="{sx(v):.1f}" y2="{top + rowh * len(rows)}" class="ref"/>')
    b.append(text(sx(0.8) - 4, top - 18, "Auto-Tune Classic 0.8", "tick", "end"))
    b.append(text(sx(2.3) + 4, top - 18, "Auto-Tune low-latency 2.3 (both host-reported)", "tick"))
    b.append(text(20, H - 10, "Studio mode reports 20-50 ms (default 40), which the DAW compensates on recorded tracks. Values at 48 kHz.", "sub"))
    return svg(W, H, "".join(b), "Plugin latency by mode")


def cpu_chart():
    # Criterion medians of Engine::process at 128-sample blocks, 48 kHz stereo,
    # M4 Max (docs/PERFORMANCE.md §11.1), as a share of the 2.67 ms block.
    period_us = 128 / 48_000 * 1e6
    data = {
        "Live-Min": (19.3, 32.9, 37.7),
        "Live": (20.1, 34.1, 38.8),
        "Live+": (20.2, 33.8, 38.7),
        "Studio": (29.4, 54.1, 73.8),
    }
    W, left, right, top = 760, 110, 60, 92
    groups = list(data.items())
    barh, gap, groupgap = 12, 3, 18
    grouph = 3 * barh + 2 * gap
    xmax = 3.0
    plotw = W - left - right
    H = top + len(groups) * (grouph + groupgap) + 40
    sx = lambda v: left + plotw * v / xmax
    b = [text(20, 28, "CPU per plugin instance (% of one core, lower is better)", "title"),
         text(20, 47, "Apple M4 Max, 48 kHz stereo, 128-sample buffer; mean time per audio block.", "sub"),
         legend(20, 72, [("s1", "Eco"), ("s2", "HQ (default)"), ("s3", "Max")])]
    bottom = top + len(groups) * (grouph + groupgap) - groupgap
    for t in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        b.append(f'<line x1="{sx(t):.1f}" y1="{top - 8}" x2="{sx(t):.1f}" y2="{bottom + 4}" class="grid"/>')
        b.append(text(sx(t), bottom + 20, f"{t:g}%", "tick", "middle"))
    for gi, (mode, us) in enumerate(groups):
        y0 = top + gi * (grouph + groupgap)
        b.append(text(left - 10, y0 + grouph / 2 + 4, mode, "lab", "end"))
        for k, (cls, v) in enumerate(zip(("s1", "s2", "s3"), us)):
            pct = 100 * v / period_us
            y = y0 + k * (barh + gap)
            b.append(rect(left, y, sx(pct) - left, barh, cls))
            b.append(text(sx(pct) + 5, y + barh - 2, f"{pct:.2f}%", "val"))
    return svg(W, H, "".join(b), "CPU per plugin instance")


def analysis_chart():
    # Type-I difference function, window 1024, lags 0-511 (docs/PERFORMANCE.md
    # §3.2): the core of the pitch detector's per-hop verifier.
    rows = [
        ("Naive loop", 304.0, ""),
        ("Direct, auto-vectorised Rust", 25.8, ""),
        ("Direct, hand-written NEON", 16.4, ""),
        ("FFT (realfft, exact size)", 4.47, "shipping"),
        ("FFT (Apple vDSP), in isolation", 3.32, "slower in the full plugin"),
    ]
    W, left, right, top = 760, 230, 170, 70
    rowh, barh = 30, 16
    lo, hi = 1.0, 1000.0
    plotw = W - left - right
    H = top + rowh * len(rows) + 46
    sx = lambda v: left + plotw * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    b = [text(20, 28, "Pitch-analysis kernel: time per window (µs, log scale, lower is better)", "title"),
         text(20, 47, "One 1024-sample analysis window with 512 lags on an M4 Max. Choosing the algorithm mattered far more than SIMD.", "sub")]
    for t in [1, 10, 100, 1000]:
        b.append(f'<line x1="{sx(t):.1f}" y1="{top - 6}" x2="{sx(t):.1f}" y2="{top + rowh * len(rows)}" class="grid"/>')
        b.append(text(sx(t), top + rowh * len(rows) + 16, f"{t:g} µs", "tick", "middle"))
    for i, (name, v, note) in enumerate(rows):
        y = top + i * rowh
        b.append(text(left - 10, y + barh - 3, name, "lab", "end"))
        cls = "s2" if note == "shipping" else "s1"
        b.append(rect(left, y, sx(v) - left, barh, cls))
        label = f"{v:g} µs" + (f"  ({note})" if note else "")
        b.append(text(sx(v) + 6, y + barh - 3, label, "val"))
    return svg(W, H, "".join(b), "Pitch-analysis kernel speed")


for name, fn in [("latency.svg", latency_chart), ("cpu.svg", cpu_chart), ("analysis-speed.svg", analysis_chart)]:
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(fn())
    print("wrote", name)
