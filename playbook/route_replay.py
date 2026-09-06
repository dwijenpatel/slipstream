#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Replay a routing trace through expert-cache policies at any slot count.

A trace (`TURBO_FIELDFARE_ROUTE_TRACE=path`) records the router's top-k
expert IDs per token and layer, prefill and decode. A cache policy's behavior
depends only on that sequence, so one trace answers every policy at every
slot count without another model run.

    route_replay.py calibrate TRACE --slots 64
        Decode misses under the production policy, to compare with the run's
        own footer ("expert cache: ... miss of ...").

    route_replay.py compare TRACE [--slots 16,64,128] [--policies lfu,lru,lfu-aging,belady]
        Decode misses per token for each policy and slot count.

    route_replay.py allocate --train T1.bin,T2.bin --test TEST.bin --slots 16,64
        Per-layer slot allocation at a fixed total, trained on the train
        traces, scored on the test trace against the uniform allocation.

Costs: an SSD-served miss measured 0.36 ms on the M5 (2026-09-05, uncached
harness). `--miss-ms` and `--base-ms` (non-I/O time per token) turn misses
into a throughput estimate.
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field

MAGIC = b"RTRC"
VERSION = 1
PREFILL, DECODE = 0, 1
PREFILL_TILE = 8            # PrefillRoutedTileSchedulerConfig.tileExperts
ALLOWED_SLOTS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]


@dataclass
class Trace:
    num_layers: int
    top_k: int
    num_experts: int
    # Units in file order. Prefill: (PREFILL, layer, [experts of each token in the chunk]).
    # Decode: (DECODE, layer, [experts]) for one token.
    units: list = field(default_factory=list)
    decode_tokens: int = 0


def read_trace(path: str) -> Trace:
    data = open(path, "rb").read()
    if data[:4] != MAGIC or data[4] != VERSION:
        sys.exit(f"{path}: not a version-{VERSION} route trace")
    num_layers, top_k = data[5], data[6]
    num_experts = data[7] | (data[8] << 8)
    rec = 6 + top_k
    trace = Trace(num_layers, top_k, num_experts)
    # Prefill records arrive layer by layer for a chunk: all positions of
    # layer 0, then layer 1, ... A run of consecutive prefill records with one
    # layer is one (chunk, layer) unit.
    run_layer, run_phase, run_experts = None, None, []
    decode_positions = set()
    off = 9
    while off + rec <= len(data):
        phase = data[off]
        position = struct.unpack_from("<I", data, off + 1)[0]
        layer = data[off + 5]
        experts = list(data[off + 6 : off + 6 + top_k])
        off += rec
        if phase == DECODE:
            if run_experts:
                trace.units.append((run_phase, run_layer, run_experts))
                run_layer, run_phase, run_experts = None, None, []
            trace.units.append((DECODE, layer, [experts]))
            decode_positions.add(position)
        else:
            if run_layer != layer:
                if run_experts:
                    trace.units.append((run_phase, run_layer, run_experts))
                run_layer, run_phase, run_experts = layer, PREFILL, []
            run_experts.append(experts)
    if run_experts:
        trace.units.append((run_phase, run_layer, run_experts))
    trace.decode_tokens = len(decode_positions)
    return trace


# --- cache policies, one instance per layer -------------------------------


class Cache:
    """Mirrors PreadExpertStreamer.makeExpertCachePlan: hits keep their slot,
    misses take the most evictable free slots, every requested expert counts
    one use, every touched slot records the clock."""

    def __init__(self, slots: int, policy: str, aging: int = 0):
        self.slots = slots
        self.policy = policy
        self.aging = aging
        self.slot_expert = [-1] * slots
        self.slot_last_use = [0] * slots
        self.use_count = defaultdict(int)
        self.where = {}          # expert -> slot
        self.clock = 0
        self.misses = 0
        self.requests = 0

    def evict_key(self, slot: int):
        e = self.slot_expert[slot]
        if e < 0:
            return (0, 0, 0)
        if self.policy == "lru":
            return (1, self.slot_last_use[slot], 0)
        return (1, self.use_count[e], self.slot_last_use[slot])

    def access(self, experts: list[int], next_use=None) -> int:
        """One batch that must be resident together. Returns the miss count."""
        if len(experts) > self.slots:
            raise SystemExit(f"batch of {len(experts)} exceeds {self.slots} slots")
        self.clock += 1
        reserved = set()
        misses = []
        for e in experts:
            s = self.where.get(e)
            if s is not None and s not in reserved:
                reserved.add(s)
                self.slot_last_use[s] = self.clock
            else:
                misses.append(e)
        for e in experts:
            self.use_count[e] += 1
        if misses:
            free = [s for s in range(self.slots) if s not in reserved]
            if self.policy == "belady":
                # Furthest next use first; an expert never used again is furthest.
                free.sort(key=lambda s: (0, 0) if self.slot_expert[s] < 0
                          else (1, -next_use(self.slot_expert[s])))
            else:
                free.sort(key=self.evict_key)
            for e, s in zip(misses, free):
                old = self.slot_expert[s]
                if old >= 0:
                    self.where.pop(old, None)
                self.slot_expert[s] = e
                self.where[e] = s
                self.slot_last_use[s] = self.clock
        self.misses += len(misses)
        self.requests += len(experts)
        if self.aging and self.clock % self.aging == 0:
            for e in list(self.use_count):
                self.use_count[e] //= 2
        return len(misses)


def prefill_batches(chunk_experts: list[list[int]]) -> list[list[int]]:
    """Prefill fetches a chunk's distinct experts in tiles sorted by file
    offset, which is expert order; the same slot cache absorbs them."""
    distinct = sorted({e for token in chunk_experts for e in token})
    return [distinct[i : i + PREFILL_TILE] for i in range(0, len(distinct), PREFILL_TILE)]


def replay(trace: Trace, slots_per_layer, policy: str, aging: int = 0):
    """Returns decode misses per layer and total decode requests."""
    caches = [Cache(slots_per_layer[L], policy, aging) for L in range(trace.num_layers)]
    # Belady needs each layer's future: batch index -> next batch index per expert.
    next_use_fn = [None] * trace.num_layers
    if policy == "belady":
        per_layer_batches = [[] for _ in range(trace.num_layers)]
        for phase, layer, tokens in trace.units:
            batches = prefill_batches(tokens) if phase == PREFILL else tokens
            per_layer_batches[layer].extend(batches)
        for L in range(trace.num_layers):
            batches = per_layer_batches[L]
            nxt = [dict() for _ in batches]
            future = {}
            for i in range(len(batches) - 1, -1, -1):
                nxt[i] = future.copy()
                for e in batches[i]:
                    future[e] = i
            next_use_fn[L] = nxt
    cursor = [0] * trace.num_layers
    decode_misses = [0] * trace.num_layers
    decode_requests = 0
    for phase, layer, tokens in trace.units:
        cache = caches[layer]
        batches = prefill_batches(tokens) if phase == PREFILL else tokens
        for batch in batches:
            if policy == "belady":
                table = next_use_fn[layer][cursor[layer]]
                misses = cache.access(batch, lambda e, t=table: t.get(e, 10**9))
                cursor[layer] += 1
            else:
                misses = cache.access(batch)
            if phase == DECODE:
                decode_misses[layer] += misses
                decode_requests += len(batch)
    return decode_misses, decode_requests


def describe(trace: Trace, decode_misses, decode_requests, args):
    misses = sum(decode_misses)
    tokens = max(1, trace.decode_tokens)
    per_token = misses / tokens
    hit = 1 - misses / max(1, decode_requests)
    io_ms = per_token * args.miss_ms
    tok_s = 1000 / (args.base_ms + io_ms)
    return misses, hit, per_token, io_ms, tok_s


def cmd_calibrate(args):
    trace = read_trace(args.trace)
    dm, dr = replay(trace, [args.slots] * trace.num_layers, "lfu")
    print(f"trace: {trace.decode_tokens} decode tokens, {dr} decode expert requests")
    print(f"lfu @ {args.slots} slots: {sum(dm)} decode misses, hit {100 * (1 - sum(dm) / dr):.1f}%")
    print("compare with the run's footer: 'expert cache: H% hit, M miss of R'")


def cmd_compare(args):
    trace = read_trace(args.trace)
    slots_list = [int(s) for s in args.slots.split(",")]
    policies = args.policies.split(",")
    print(f"trace: {trace.decode_tokens} decode tokens; miss {args.miss_ms} ms, base {args.base_ms} ms/token")
    print(f"{'policy':<10} {'slots':>5} {'hit%':>6} {'miss/tok':>9} {'MB/tok':>7} {'io ms':>7} {'est tok/s':>9}")
    for slots in slots_list:
        for policy in policies:
            aging = 64 if policy == "lfu-aging" else 0
            base = "lfu" if policy == "lfu-aging" else policy
            dm, dr = replay(trace, [slots] * trace.num_layers, base, aging)
            misses, hit, per_token, io_ms, tok_s = describe(trace, dm, dr, args)
            print(f"{policy:<10} {slots:>5} {100 * hit:>6.1f} {per_token:>9.1f} "
                  f"{per_token * args.expert_mb:>7.0f} {io_ms:>7.1f} {tok_s:>9.1f}")


def per_layer_curves(traces: list[Trace], num_layers: int, candidates: list[int], policy: str):
    """misses[L][slots] summed over the traces."""
    curves = [dict() for _ in range(num_layers)]
    for slots in candidates:
        totals = [0] * num_layers
        for trace in traces:
            dm, _ = replay(trace, [slots] * num_layers, policy)
            for L in range(num_layers):
                totals[L] += dm[L]
        for L in range(num_layers):
            curves[L][slots] = totals[L]
    return curves


def allocate(curves, num_layers: int, total: int, candidates: list[int]) -> list[int]:
    """Exact DP: choose slots per layer from candidates, sum == total, min misses."""
    unit = min(candidates)
    budget = total // unit
    INF = float("inf")
    best = [[INF] * (budget + 1) for _ in range(num_layers + 1)]
    choice = [[None] * (budget + 1) for _ in range(num_layers + 1)]
    best[0][0] = 0
    for L in range(num_layers):
        for used in range(budget + 1):
            if best[L][used] == INF:
                continue
            for s in candidates:
                u = used + s // unit
                if u > budget:
                    continue
                cost = best[L][used] + curves[L][s]
                if cost < best[L + 1][u]:
                    best[L + 1][u] = cost
                    choice[L + 1][u] = s
    used = budget
    if best[num_layers][used] == INF:
        sys.exit("no allocation fits the budget")
    alloc = []
    for L in range(num_layers, 0, -1):
        s = choice[L][used]
        alloc.append(s)
        used -= s // unit
    return list(reversed(alloc))


def cmd_allocate(args):
    train = [read_trace(p) for p in args.train.split(",")]
    test = read_trace(args.test)
    num_layers = test.num_layers
    candidates = [s for s in ALLOWED_SLOTS if s <= args.max_slots]
    curves = per_layer_curves(train, num_layers, candidates, "lfu")
    for slots in [int(s) for s in args.slots.split(",")]:
        alloc = allocate(curves, num_layers, slots * num_layers, candidates)
        uniform_dm, dr = replay(test, [slots] * num_layers, "lfu")
        tuned_dm, _ = replay(test, alloc, "lfu")
        print(f"total {slots * num_layers} slots ({slots}/layer uniform)")
        print(f"  uniform: {sum(uniform_dm)} decode misses on test "
              f"({sum(uniform_dm) / max(1, test.decode_tokens):.1f}/token)")
        print(f"  tuned:   {sum(tuned_dm)} decode misses on test "
              f"({sum(tuned_dm) / max(1, test.decode_tokens):.1f}/token), "
              f"{100 * (1 - sum(tuned_dm) / max(1, sum(uniform_dm))):+.1f}% change")
        print("  allocation:", " ".join(str(s) for s in alloc))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--miss-ms", type=float, default=0.36, help="cost of one SSD-served miss")
    parser.add_argument("--base-ms", type=float, default=20.0, help="non-I/O time per token")
    parser.add_argument("--expert-mb", type=float, default=1.769472, help="bytes per expert, MB")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("trace"); c.add_argument("--slots", type=int, required=True)
    p = sub.add_parser("compare"); p.add_argument("trace"); p.add_argument("--slots", default="16,64,128")
    p.add_argument("--policies", default="lfu,lru,lfu-aging,belady")
    a = sub.add_parser("allocate"); a.add_argument("--train", required=True); a.add_argument("--test", required=True)
    a.add_argument("--slots", default="16,64"); a.add_argument("--max-slots", type=int, default=128)
    args = parser.parse_args()
    {"calibrate": cmd_calibrate, "compare": cmd_compare, "allocate": cmd_allocate}[args.command](args)


if __name__ == "__main__":
    main()
