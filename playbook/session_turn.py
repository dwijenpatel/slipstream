#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""One turn of a multi-turn coding session against the slipstream server.

    turn.py --state S.json --system system.txt "first user message"
    turn.py --state S.json "next user message"
    turn.py --state S.json --show          # print the conversation so far

Appends the user message, posts the whole conversation to
POST /v1/chat/completions, appends the assistant reply, saves the state, and
prints the reply plus usage. The server keeps one verified KV prefix, so each
turn prefills only the new tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="?")
    ap.add_argument("--state", required=True)
    ap.add_argument("--system")
    ap.add_argument("--url", default="http://127.0.0.1:8091/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        state = json.load(open(args.state))
    except FileNotFoundError:
        state = {"messages": [], "turns": []}
    if args.system and not state["messages"]:
        state["messages"].append({"role": "system", "content": open(args.system).read()})
    if args.show:
        for m in state["messages"]:
            print(f"--- {m['role']} ---\n{m['content']}\n")
        for t in state["turns"]:
            print(t)
        return
    if not args.message:
        sys.exit("a message is required")
    text = open(args.message[1:]).read() if args.message.startswith("@") else args.message
    state["messages"].append({"role": "user", "content": text})

    payload = {"model": args.model, "messages": state["messages"],
               "max_tokens": args.max_tokens, "temperature": 0, "stream": False}
    req = urllib.request.Request(args.url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        body = json.load(resp)
    elapsed = time.time() - t0
    choice = body["choices"][0]
    reply = choice["message"].get("content") or ""
    state["messages"].append({"role": "assistant", "content": reply})
    usage = body.get("usage", {})
    turn = {"turn": len(state["turns"]) + 1, "seconds": round(elapsed, 1),
            "finish": choice.get("finish_reason"), "usage": usage}
    state["turns"].append(turn)
    json.dump(state, open(args.state, "w"), indent=1)
    print(reply)
    print(f"\n[turn {turn['turn']}: {elapsed:.0f}s, finish={turn['finish']}, usage={usage}]", file=sys.stderr)


if __name__ == "__main__":
    main()
