#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate the README's charts as SVG files under docs/assets.

The data is inline because bench-results/ is not in the repository. Each
figure names the sweep it was read from; the CSVs are:

  slipstream rows      bench-results/table-20260906-035959/results.csv
                       (commit 03d5a06, 2026-09-06, drift control 0.8 percent)
  TurboFieldfare rows  bench-results/table-20260806-044148/results.csv (2026-08-06)
  llama.cpp rows       bench-results/table-20260806-055113/results.csv (2026-08-06)
  mlx-lm point         playbook/mlx_lm_bench.py, commit 33b863e, 2026-08-05

    uv run playbook/readme_figures.py          # writes docs/assets/*.svg
"""
from __future__ import annotations

import math
import os

FONT = "-apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"
BLUE, VERMILLION, GREEN, PURPLE, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6b6b6b"
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")

PROMPTS = ["1k", "3k", "12k", "24k"]

# name, color, dash, marker, ttft seconds, decode tokens per second
SERIES = [
    ("slipstream, 64 slots, 5.6 GB", BLUE, "", "circle",
     [8.9, 14.3, 53.7, 112.0], [33.1, 31.1, 29.9, 27.2]),
    ("slipstream, 16 slots, 2.5 GB", BLUE, "6 4", "circle",
     [8.8, 14.1, 53.4, 111.6], [28.3, 27.1, 26.2, 24.4]),
    ("TurboFieldfare, 16 slots, 2.0 GB", VERMILLION, "", "square",
     [14.3, 42.5, 185.1, 664.8], [26.1, 23.8, 17.2, 13.0]),
    ("llama.cpp, all in memory, 16.2 GB", GREEN, "", "triangle",
     [1.2, 3.8, 18.5, 44.9], [37.6, 38.6, 38.3, 38.9]),
    ("llama.cpp, CPU experts, 17.2 GB", GREEN, "6 4", "triangle",
     [2.8, 11.0, 51.4, 141.8], [22.3, 20.8, 22.0, 20.0]),
]

# Decode at the 2,940-token prompt against peak footprint, GB.
MEMORY_POINTS = {
    "slipstream": [(2.45, 27.1, "16"), (3.52, 29.1, "32"), (5.64, 31.1, "64"),
                   (7.78, 31.7, "96"), (9.92, 31.9, "128"), (14.18, 25.2, "192")],
    "turbofieldfare": [(1.96, 23.8, "16"), (3.02, 25.3, "32")],
    "llama_resident": (16.22, 38.6),
    "llama_cpu_moe": (16.80, 22.9),
    "llama_cpu_moe_nommap": (17.22, 20.8),
    "mlx": (21.6, 41.2),
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, anchor="start", weight="normal", fill="#1a1a1a", rotate=None):
    t = f' transform="rotate(-90 {x} {y})"' if rotate else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}"{t}>{esc(s)}</text>')


def marker(kind, x, y, color):
    if kind == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
    if kind == "square":
        return f'<rect x="{x-3.5:.1f}" y="{y-3.5:.1f}" width="7" height="7" fill="{color}"/>'
    return (f'<polygon points="{x:.1f},{y-4.5:.1f} {x-4.5:.1f},{y+3.5:.1f} {x+4.5:.1f},{y+3.5:.1f}" '
            f'fill="{color}"/>')


def separate(ys, gap, lo, hi):
    """Push label centers apart to at least `gap`, inside [lo, hi]. Keeps order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = [0.0] * len(ys)
    prev = None
    for i in order:
        y = ys[i] if prev is None else max(ys[i], prev + gap)
        out[i] = y
        prev = y
    overflow = out[order[-1]] - hi
    if overflow > 0:
        for i in reversed(order):
            out[i] -= overflow
            overflow = 0
            # keep the rest in order below the pushed one
        prev = None
        for i in reversed(order):
            y = out[i] if prev is None else min(out[i], prev - gap)
            out[i] = max(y, lo)
            prev = out[i]
    return out


def frame(w, h, title, subtitle_lines):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
             f'font-family="{FONT}" role="img" aria-label="{esc(title)}">',
             f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
             text(20, 26, title, size=16, weight="bold")]
    y = 46
    for line in subtitle_lines:
        parts.append(text(20, y, line, size=12, fill="#444444"))
        y += 18
    return parts


def line_chart(path, title, subtitle, ylabel, values_index, log, ymin, ymax, ticks, fmt, label_merge=None):
    W, H = 760, 470
    left, top, right, bottom = 64, 96, 540, 414
    parts = frame(W, H, title, subtitle)

    def sy(v):
        if log:
            f = (math.log10(v) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin))
        else:
            f = (v - ymin) / (ymax - ymin)
        return bottom - f * (bottom - top)

    def sx(i):
        return left + 28 + i * (right - left - 56) / (len(PROMPTS) - 1)

    # grid and axes
    for t in ticks:
        y = sy(t)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e3e3e3"/>')
        parts.append(text(left - 8, y + 4, fmt(t), size=11, anchor="end", fill="#444444"))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#888888"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#888888"/>')
    for i, p in enumerate(PROMPTS):
        parts.append(text(sx(i), bottom + 18, p, size=11, anchor="middle", fill="#444444"))
    parts.append(text((left + right) / 2, H - 14, "prompt length, tokens", size=12, anchor="middle"))
    parts.append(text(18, (top + bottom) / 2, ylabel, size=12, anchor="middle", rotate=True))

    # series
    ends = []
    for name, color, dash, mk, ttft, dec in SERIES:
        vals = (ttft, dec)[values_index]
        pts = [(sx(i), sy(v)) for i, v in enumerate(vals)]
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{da}/>')
        for x, y in pts:
            parts.append(marker(mk, x, y, color))
        ends.append((name, color, pts[-1][1]))

    # end labels, separated
    names = [e[0] for e in ends]
    if label_merge:
        merged_name, members = label_merge
        keep = [e for e in ends if e[0] not in members]
        member_y = sum(e[2] for e in ends if e[0] in members) / len(members)
        member_color = next(e[1] for e in ends if e[0] in members)
        ends = keep + [(merged_name, member_color, member_y)]
    ys = separate([e[2] for e in ends], 16, top + 6, bottom - 6)
    for (name, color, y0), y in zip(ends, ys):
        parts.append(f'<line x1="{right + 6}" y1="{y0:.1f}" x2="{right + 14}" y2="{y:.1f}" stroke="{color}" stroke-width="1"/>')
        parts.append(text(right + 18, y + 4, name, size=12, fill=color))
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")


def memory_chart(path):
    W, H = 760, 470
    left, top, right, bottom = 64, 96, 730, 414
    xmin, xmax, ymin, ymax = 0, 24, 0, 45
    parts = frame(W, H, "Decode speed against memory at the 2,940-token prompt", [
        "Tokens per second over 512 generated tokens, against the process's peak physical footprint.",
        "The two llama.cpp points that leave mmap on report resident set size instead. Base M5, 24 GB.",
    ])

    def sx(v):
        return left + (v - xmin) / (xmax - xmin) * (right - left)

    def sy(v):
        return bottom - (v - ymin) / (ymax - ymin) * (bottom - top)

    for t in range(0, 50, 10):
        y = sy(t)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e3e3e3"/>')
        parts.append(text(left - 8, y + 4, str(t), size=11, anchor="end", fill="#444444"))
    for t in range(0, 25, 4):
        x = sx(t)
        parts.append(text(x, bottom + 18, str(t), size=11, anchor="middle", fill="#444444"))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#888888"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#888888"/>')
    parts.append(text((left + right) / 2, H - 14, "peak memory footprint, GB, on a machine with 24 GB", size=12, anchor="middle"))
    parts.append(text(18, (top + bottom) / 2, "decode, tokens per second", size=12, anchor="middle", rotate=True))

    # wired limit
    xl = sx(21.3)
    parts.append(f'<line x1="{xl:.1f}" y1="{top + 34}" x2="{xl:.1f}" y2="{bottom}" stroke="{GREY}" stroke-dasharray="4 4"/>')
    parts.append(text(xl - 6, bottom - 8, "GPU wired limit, 21.3 GB", size=11, anchor="end", fill=GREY))

    def curve(points, color, mk, label_dy):
        pts = [(sx(x), sy(y)) for x, y, _ in points]
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for (x, y), (_, _, slots) in zip(pts, points):
            parts.append(marker(mk, x, y, color))
            parts.append(text(x, y + label_dy, slots, size=11, anchor="middle", fill=color))

    curve(MEMORY_POINTS["slipstream"], BLUE, "circle", -9)
    curve(MEMORY_POINTS["turbofieldfare"], VERMILLION, "square", 17)
    parts.append(text(sx(3.0), sy(36.5), "slipstream, 16 to 192 slots", size=12, fill=BLUE))
    parts.append(text(sx(1.3), sy(17.0), "TurboFieldfare, 16 and 32 slots", size=12, fill=VERMILLION))

    lx, ly = MEMORY_POINTS["llama_resident"]
    parts.append(marker("triangle", sx(lx), sy(ly), GREEN))
    parts.append(text(sx(lx) - 10, sy(ly) + 16, "llama.cpp, all weights in memory", size=12, anchor="end", fill=GREEN))
    lx, ly = MEMORY_POINTS["llama_cpu_moe"]
    parts.append(marker("triangle", sx(lx), sy(ly), GREEN))
    parts.append(text(sx(lx) - 10, sy(ly) - 2, "llama.cpp, --n-cpu-moe 32", size=12, anchor="end", fill=GREEN))
    lx, ly = MEMORY_POINTS["llama_cpu_moe_nommap"]
    parts.append(marker("triangle", sx(lx), sy(ly), GREEN))
    parts.append(text(sx(lx) - 10, sy(ly) + 14, "the same with --mmap 0", size=12, anchor="end", fill=GREEN))
    lx, ly = MEMORY_POINTS["mlx"]
    parts.append(f'<polygon points="{sx(lx):.1f},{sy(ly)-5:.1f} {sx(lx)+5:.1f},{sy(ly):.1f} {sx(lx):.1f},{sy(ly)+5:.1f} {sx(lx)-5:.1f},{sy(ly):.1f}" fill="{PURPLE}"/>')
    parts.append(text(sx(lx) - 10, sy(ly) - 6, "mlx-lm, all weights in memory", size=12, anchor="end", fill=PURPLE))
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    line_chart(
        os.path.join(OUT, "ttft-by-prompt-length.svg"),
        "Time to first token by prompt length",
        ["Seconds from request to first token, log scale, one fresh process per point on a base M5 with 24 GB.",
         "slipstream measured 2026-09-06; TurboFieldfare and llama.cpp 2026-08-06, same prompts and harness."],
        "time to first token, seconds, log scale",
        0, True, 1, 1000, [1, 3, 10, 30, 100, 300, 1000], lambda t: f"{t:,}",
        label_merge=("slipstream, 16 or 64 slots", {"slipstream, 64 slots, 5.6 GB", "slipstream, 16 slots, 2.5 GB"}),
    )
    line_chart(
        os.path.join(OUT, "decode-by-prompt-length.svg"),
        "Decode speed by prompt length",
        ["Tokens per second over the 512 tokens generated after each prompt, same runs as the chart above.",
         "Memory is each configuration's peak physical footprint; the resident llama.cpp figure is resident set size."],
        "decode, tokens per second",
        1, False, 0, 40, [0, 10, 20, 30, 40], lambda t: str(t),
    )
    memory_chart(os.path.join(OUT, "decode-by-memory.svg"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
