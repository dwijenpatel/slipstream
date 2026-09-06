#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Summarize an overnight.sh output directory as Markdown.

    overnight_summary.py bench-results/overnight-YYYYMMDD-HHMMSS
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys


def read(path):
    try:
        return open(path, errors="replace").read()
    except FileNotFoundError:
        return ""


def num(pattern, text, default=None, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def cli_row(out, name):
    err = read(os.path.join(out, name + ".err"))
    if not err:
        return None
    tokens = num(r"new=(\d+)tok", err, 0, int)
    decode_s = num(r"decode=([0-9.]+)s", err, 0.0)
    prefill_s = num(r"prefill=\d+tok/([0-9.]+)s", err, 0.0)
    tok_s = num(r"tok/s=([0-9.]+)", err, 0.0)
    io_ms = num(r"expert io await:\s+([0-9.]+) ms", err, 0.0)
    gpu_ms = num(r"gpu cb1 \(attn\+norms\+router\):\s+([0-9.]+) ms", err, 0.0)
    misses = num(r"expert cache:\s+[0-9.]+% hit, (\d+) miss", err, 0, int)
    disk = sum(int(x) for x in re.findall(r"disk_read_bytes=(\d+)", err))
    logical = sum(int(x) for x in re.findall(r"expert_bytes=(\d+)", err))
    recall = re.findall(r"distance (\d+): ([0-9.]+)%", err)
    prefetch = num(r"prefetch:\s+distance (\d+)", err, None, int)
    out_path = os.path.join(out, name + ".out")
    digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()[:12] if os.path.exists(out_path) else ""
    per_tok = max(1, tokens)
    return {
        "arm": name, "tok/s": tok_s, "prefill s": prefill_s,
        "io ms/tok": io_ms / per_tok, "gpu ms/tok": gpu_ms / per_tok,
        "miss/tok": misses / per_tok,
        "phys/logical": (disk / logical) if logical else None,
        "MB/tok": (disk / per_tok / 1e6) if disk else None,
        "prefetch": prefetch, "recall": ", ".join(f"d{d}: {r}%" for d, r in recall),
        "output": digest,
        "residency before": read(os.path.join(out, name + ".residency-before")).strip(),
        "residency after": read(os.path.join(out, name + ".residency-after")).strip(),
    }


def session_row(out, name):
    log = read(os.path.join(out, name + ".server.log"))
    if not log:
        return None
    def total(key):
        return sum(float(x) for x in re.findall(key + r"=([0-9.]+)", log))
    completion = int(total("completion"))
    replay = read(os.path.join(out, name + ".replay.log")).strip().splitlines()
    verdict = replay[-1] if replay else "no replay log"
    requests = len(re.findall(r"request prompt=", log))
    tot = total("total")
    return {
        "arm": name, "requests": requests, "completion tokens": completion,
        "total s": tot, "decode tok/s (approx)": completion / tot if tot else 0,
        "io_await s": total("io_await"), "expert_miss": int(total("expert_miss")),
        "miss/tok": total("expert_miss") / max(1, completion),
        "verdict": verdict,
        "residency before": read(os.path.join(out, name + ".residency-before")).strip(),
        "residency after": read(os.path.join(out, name + ".residency-after")).strip(),
    }


def table(rows, columns):
    if not rows:
        return "_none_\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.2f}")
            elif v is None:
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    out = sys.argv[1]
    names = sorted(f[:-4] for f in os.listdir(out) if f.endswith(".err"))
    sessions = sorted(f[:-11] for f in os.listdir(out) if f.endswith(".server.log"))
    print(f"# Overnight summary: {os.path.basename(out)}\n")
    print(f"commit {read(os.path.join(out, 'commit')).strip()}\n")
    cli_cols = ["arm", "tok/s", "io ms/tok", "gpu ms/tok", "miss/tok", "MB/tok", "phys/logical", "prefetch", "recall", "output"]
    for phase, title in [("p1", "Prefetch A/B, 64 slots, uncached"), ("p6", "Clock hold off against auto, warm")]:
        rows = [r for r in (cli_row(out, n) for n in names if n.startswith(phase + "-")) if r]
        print(f"## {title}\n")
        print(table(rows, cli_cols))
    for phase, title in [("p4", "Prefill only, uncached"), ("p7", "Prefill only, warm")]:
        rows = [r for r in (cli_row(out, n) for n in names if n.startswith(phase + "-")) if r]
        print(f"## {title}\n")
        print(table(rows, ["arm", "prefill s", "phys/logical", "output"]))
    print("## Sessions\n")
    rows = [r for r in (session_row(out, n) for n in sessions) if r]
    print(table(rows, ["arm", "requests", "completion tokens", "total s", "decode tok/s (approx)", "io_await s", "expert_miss", "miss/tok", "verdict"]))
    print("## Residency\n")
    res = []
    for n in names + sessions:
        b = read(os.path.join(out, n + ".residency-before")).strip()
        a = read(os.path.join(out, n + ".residency-after")).strip()
        if b or a:
            res.append({"arm": n, "before": b, "after": a})
    print(table(res, ["arm", "before", "after"]))


if __name__ == "__main__":
    main()
