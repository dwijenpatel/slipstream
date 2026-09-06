#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Replay a recorded session's user messages against a running server and
check that the assistant replies are byte-identical to the recording.

    replay_session.py --recorded state.json --out state-lru.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorded", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8091/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--turns", type=int, default=0, help="replay only the first N turns (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="per-turn cap; below 4096 the identity check is expected to fail")
    args = ap.parse_args()

    recorded = json.load(open(args.recorded))
    messages = []
    turns = []
    mismatches = 0
    for i, m in enumerate(recorded["messages"]):
        if m["role"] in ("system", "user"):
            messages.append(m)
            continue
        if args.turns and len(turns) >= args.turns:
            break
        payload = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens,
                   "temperature": 0, "stream": False}
        req = urllib.request.Request(args.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as resp:
            body = json.load(resp)
        elapsed = time.time() - t0
        reply = body["choices"][0]["message"].get("content") or ""
        same = reply == m["content"]
        mismatches += 0 if same else 1
        messages.append({"role": "assistant", "content": reply})
        usage = body.get("usage", {})
        turns.append({"turn": len(turns) + 1, "seconds": round(elapsed, 1),
                      "identical": same, "completion_tokens": usage.get("completion_tokens")})
        print(f"turn {len(turns)}: {elapsed:.0f}s, {usage.get('completion_tokens')} tokens, "
              f"{'identical' if same else 'DIFFERENT'}", flush=True)
    json.dump({"messages": messages, "turns": turns}, open(args.out, "w"), indent=1)
    print(f"{len(turns)} turns, {mismatches} mismatches")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
