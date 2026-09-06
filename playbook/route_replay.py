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

    route_replay.py calibrate TRACE --slots 64 [--policy lfu-aging:32]
        Decode misses under the run's policy, to compare with the run's own
        footer ("expert cache: ... miss of ..."). Must match exactly.

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
    off = 9
    while off + rec <= len(data):
        phase = data[off]
        layer = data[off + 5]
        experts = list(data[off + 6 : off + 6 + top_k])
        off += rec
        if phase == DECODE:
            if run_experts:
                trace.units.append((run_phase, run_layer, run_experts))
                run_layer, run_phase, run_experts = None, None, []
            trace.units.append((DECODE, layer, [experts]))
            # Positions repeat after request resets; each layer-zero record
            # represents a new decode event, even at a previously seen position.
            if layer == 0:
                trace.decode_tokens += 1
        else:
            if run_layer != layer:
                if run_experts:
                    trace.units.append((run_phase, run_layer, run_experts))
                run_layer, run_phase, run_experts = layer, PREFILL, []
            run_experts.append(experts)
    if run_experts:
        trace.units.append((run_phase, run_layer, run_experts))
    return trace


# --- cache policies, one instance per layer -------------------------------


class Cache:
    """Mirrors PreadExpertStreamer.makeExpertCachePlan: hits keep their slot,
    misses take the most evictable free slots, every requested expert counts
    one use, every touched slot records the clock."""

    def __init__(self, slots: int, policy: str, aging: int = 0, window: int = 0):
        self.slots = slots
        self.policy = policy
        self.aging = aging
        self.window = window
        self.recent = []          # batches inside the window, oldest first
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
        if self.window:
            self.recent.append(list(experts))
            if len(self.recent) > self.window:
                for e in self.recent.pop(0):
                    self.use_count[e] -= 1
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



# --- advanced policies ------------------------------------------------------
# Each takes a batch that must be resident together (the batch is "pinned"
# during the plan) and returns the miss count. Capacity is `slots`.


class LRFU:
    """Lee et al. 2001: score = sum over uses of 2^(-lam * age). lam -> 0 is
    LFU, large lam is LRU."""

    def __init__(self, slots, lam):
        self.slots, self.lam = slots, lam
        self.score, self.last, self.resident = {}, {}, set()
        self.clock = 0

    def decayed(self, e):
        return self.score.get(e, 0.0) * 2 ** (-self.lam * (self.clock - self.last.get(e, self.clock)))

    def access(self, batch, next_use=None):
        self.clock += 1
        misses = [e for e in batch if e not in self.resident]
        for e in batch:
            self.score[e] = self.decayed(e) + 1.0
            self.last[e] = self.clock
        pinned = set(batch)
        for e in misses:
            if len(self.resident) >= self.slots:
                victim = min((x for x in self.resident if x not in pinned), key=self.decayed)
                self.resident.remove(victim)
            self.resident.add(e)
        return len(misses)


class ARC:
    """Megiddo and Modha 2003, with the batch pinned during replacement."""

    def __init__(self, slots):
        from collections import OrderedDict
        self.c = slots
        self.p = 0
        self.T1, self.T2, self.B1, self.B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

    def _lru_unpinned(self, lst, pinned):
        for e in lst:
            if e not in pinned:
                return e
        return None

    def _replace(self, e, pinned):
        if self.T1 and (len(self.T1) > self.p or (e in self.B2 and len(self.T1) == self.p)):
            v = self._lru_unpinned(self.T1, pinned)
            if v is not None:
                del self.T1[v]; self.B1[v] = True
                return
        v = self._lru_unpinned(self.T2, pinned)
        if v is not None:
            del self.T2[v]; self.B2[v] = True
            return
        v = self._lru_unpinned(self.T1, pinned)
        if v is not None:
            del self.T1[v]; self.B1[v] = True

    def access(self, batch, next_use=None):
        pinned = set(batch)
        misses = 0
        for e in batch:
            if e in self.T1:
                del self.T1[e]; self.T2[e] = True
            elif e in self.T2:
                self.T2.move_to_end(e)
            elif e in self.B1:
                misses += 1
                self.p = min(self.c, self.p + max(1, len(self.B2) // max(1, len(self.B1))))
                self._replace(e, pinned); del self.B1[e]; self.T2[e] = True
            elif e in self.B2:
                misses += 1
                self.p = max(0, self.p - max(1, len(self.B1) // max(1, len(self.B2))))
                self._replace(e, pinned); del self.B2[e]; self.T2[e] = True
            else:
                misses += 1
                l1 = len(self.T1) + len(self.B1)
                total = l1 + len(self.T2) + len(self.B2)
                if l1 == self.c:
                    if len(self.T1) < self.c:
                        self.B1.popitem(last=False); self._replace(e, pinned)
                    else:
                        v = self._lru_unpinned(self.T1, pinned)
                        if v is not None: del self.T1[v]
                elif l1 < self.c and total >= self.c:
                    if total == 2 * self.c:
                        self.B2.popitem(last=False)
                    self._replace(e, pinned)
                self.T1[e] = True
            while len(self.T1) + len(self.T2) > self.c:
                v = self._lru_unpinned(self.T1, pinned)
                if v is not None:
                    del self.T1[v]; self.B1[v] = True; continue
                v = self._lru_unpinned(self.T2, pinned)
                if v is None:
                    break
                del self.T2[v]; self.B2[v] = True
            for ghost in (self.B1, self.B2):
                while len(ghost) > self.c:
                    ghost.popitem(last=False)
        return misses


class Segmented:
    """W-TinyLFU shape: a small LRU window in front of an aged-LFU main
    region; a window evictee enters main only if its aged count beats the
    main victim's, else it is dropped."""

    def __init__(self, slots, window, aging=32):
        from collections import OrderedDict
        self.slots, self.w, self.aging = slots, window, aging
        self.window, self.main = OrderedDict(), set()
        self.count = defaultdict(int)
        self.last = {}
        self.clock = 0

    def access(self, batch, next_use=None):
        self.clock += 1
        pinned = set(batch)
        misses = 0
        for e in batch:
            self.count[e] += 1
            self.last[e] = self.clock
            if e in self.main:
                continue
            if e in self.window:
                self.window.move_to_end(e); continue
            misses += 1
            self.window[e] = True
            while len(self.window) > self.w:
                cand = next((x for x in self.window if x not in pinned), None)
                if cand is None:
                    break
                del self.window[cand]
                if len(self.main) < self.slots - self.w:
                    self.main.add(cand); continue
                victim = min((x for x in self.main if x not in pinned),
                             key=lambda x: (self.count[x], self.last[x]), default=None)
                if victim is not None and self.count[cand] > self.count[victim]:
                    self.main.remove(victim); self.main.add(cand)
        if self.aging and self.clock % self.aging == 0:
            for e in list(self.count):
                self.count[e] //= 2
        return misses


class S3FIFO:
    """Yang et al. 2023: small FIFO, main FIFO, ghost; 2-bit frequency."""

    def __init__(self, slots):
        from collections import OrderedDict, deque
        self.slots = slots
        self.small_cap = max(8, slots // 10)
        self.small, self.main = deque(), deque()
        self.ghost = OrderedDict()
        self.freq = {}
        self.where = {}

    def _evict_small(self, pinned):
        for _ in range(len(self.small)):
            e = self.small.popleft()
            if e in pinned:
                self.small.append(e); continue
            if self.freq[e] > 0:
                self.freq[e] = 0; self.main.append(e); self.where[e] = "m"
            else:
                del self.freq[e]; del self.where[e]; self.ghost[e] = True
                while len(self.ghost) > self.slots:
                    self.ghost.popitem(last=False)
            return True
        return False

    def _evict_main(self, pinned):
        for _ in range(len(self.main) * 4):
            e = self.main.popleft()
            if e in pinned:
                self.main.append(e); continue
            if self.freq[e] > 0:
                self.freq[e] -= 1; self.main.append(e); continue
            del self.freq[e]; del self.where[e]
            return True
        return False

    def access(self, batch, next_use=None):
        pinned = set(batch)
        misses = 0
        for e in batch:
            if e in self.where:
                self.freq[e] = min(3, self.freq[e] + 1); continue
            misses += 1
            if e in self.ghost:
                del self.ghost[e]; self.main.append(e); self.where[e] = "m"
            else:
                self.small.append(e); self.where[e] = "s"
            self.freq[e] = 0
            while len(self.where) > self.slots:
                if len(self.small) >= self.small_cap:
                    if not self._evict_small(pinned) and not self._evict_main(pinned): break
                else:
                    if not self._evict_main(pinned) and not self._evict_small(pinned): break
        return misses


class SIEVE:
    """Zhang et al. 2024: one FIFO, a visited bit, a hand that clears bits."""

    def __init__(self, slots):
        self.slots = slots
        self.order = []          # head at the end, tail at index 0
        self.visited = {}
        self.hand = 0

    def access(self, batch, next_use=None):
        pinned = set(batch)
        misses = 0
        for e in batch:
            if e in self.visited:
                self.visited[e] = True; continue
            misses += 1
            while len(self.order) >= self.slots:
                if self.hand >= len(self.order):
                    self.hand = 0
                x = self.order[self.hand]
                if self.visited[x] or x in pinned:
                    self.visited[x] = False if x not in pinned else self.visited[x]
                    self.hand += 1
                    continue
                del self.order[self.hand]; del self.visited[x]
            self.order.append(e); self.visited[e] = False
        return misses


class TransitionAging:
    """Aged LFU plus a MoE-shaped bias: an expert that usually follows the
    current batch, by this layer's own token-to-token transition counts, is
    protected in proportion to that probability."""

    def __init__(self, slots, beta, aging=32):
        self.slots, self.beta, self.aging = slots, beta, aging
        self.count = defaultdict(int)
        self.last = {}
        self.resident = set()
        self.trans = defaultdict(lambda: defaultdict(int))
        self.rowsum = defaultdict(int)
        self.prev = []
        self.clock = 0

    def access(self, batch, next_use=None):
        self.clock += 1
        pinned = set(batch)
        for a in self.prev:
            for e in batch:
                self.trans[a][e] += 1
            self.rowsum[a] += len(batch)
        misses = [e for e in batch if e not in self.resident]
        for e in batch:
            self.count[e] += 1; self.last[e] = self.clock

        def score(x):
            p = sum(self.trans[a][x] / self.rowsum[a] for a in batch if self.rowsum[a])
            return (self.count[x] + self.beta * p, self.last[x])

        for e in misses:
            if len(self.resident) >= self.slots:
                victim = min((x for x in self.resident if x not in pinned), key=score)
                self.resident.remove(victim)
            self.resident.add(e)
        self.prev = list(batch)
        if self.aging and self.clock % self.aging == 0:
            for e in list(self.count):
                self.count[e] //= 2
        return len(misses)


def make_policy(spec: str, slots: int):
    """Policy spec -> per-layer cache object, or None for the base Cache class."""
    name, _, param = spec.partition(":")
    if name == "lrfu":
        return LRFU(slots, float(param or 0.05))
    if name == "arc":
        return ARC(slots)
    if name == "segmented":
        return Segmented(slots, max(8, slots // int(param or 8)))
    if name == "s3fifo":
        return S3FIFO(slots)
    if name == "sieve":
        return SIEVE(slots)
    if name == "transition":
        return TransitionAging(slots, float(param or 64))
    return None


def replay_advanced(trace, slots, spec):
    caches = [make_policy(spec, slots) for _ in range(trace.num_layers)]
    decode_misses = [0] * trace.num_layers
    decode_requests = 0
    for phase, layer, tokens in trace.units:
        batches = prefill_batches(tokens) if phase == PREFILL else tokens
        for batch in batches:
            m = caches[layer].access(batch)
            if phase == DECODE:
                decode_misses[layer] += m
                decode_requests += len(batch)
    return decode_misses, decode_requests


def prefill_batches(chunk_experts: list[list[int]]) -> list[list[int]]:
    """Prefill fetches a chunk's distinct experts in tiles sorted by file
    offset, which is expert order; the same slot cache absorbs them."""
    distinct = sorted({e for token in chunk_experts for e in token})
    return [distinct[i : i + PREFILL_TILE] for i in range(0, len(distinct), PREFILL_TILE)]


def replay(trace: Trace, slots_per_layer, policy: str, aging: int = 0, window: int = 0):
    """Returns decode misses per layer and total decode requests."""
    caches = [Cache(slots_per_layer[L], policy, aging, window) for L in range(trace.num_layers)]
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


def cmd_predict(args):
    """How many of the default policy's misses a no-compute predictor could
    have prefetched: predict token t+1's experts at layer L as the top-K by
    this layer's own transition counts from token t's batch, plus the batch
    itself. Reports recall of misses and wasted prefetches per token."""
    trace = read_trace(args.trace)
    slots = args.slots
    ks = [int(k) for k in args.k.split(",")]
    caches = [Cache(slots, "lfu", 32) for _ in range(trace.num_layers)]
    trans = [defaultdict(lambda: defaultdict(int)) for _ in range(trace.num_layers)]
    prev = [None] * trace.num_layers
    misses_total = 0
    covered = {k: 0 for k in ks}
    wasted = {k: 0 for k in ks}
    predicted_total = {k: 0 for k in ks}
    tokens = 0
    for phase, layer, tokens_or_batches in trace.units:
        batches = prefill_batches(tokens_or_batches) if phase == PREFILL else tokens_or_batches
        for batch in batches:
            cache = caches[layer]
            resident_before = set(cache.where)
            missed = [e for e in batch if e not in resident_before]
            if phase == DECODE and prev[layer] is not None:
                scores = defaultdict(float)
                for a in prev[layer]:
                    row = trans[layer][a]
                    total = sum(row.values()) or 1
                    for e, c in row.items():
                        scores[e] += c / total
                ranked = sorted(scores, key=lambda e: -scores[e])
                for k in ks:
                    pred = set(prev[layer]) | set(ranked[:k])
                    pred_new = pred - resident_before          # what a prefetch would read
                    predicted_total[k] += len(pred_new)
                    covered[k] += sum(1 for e in missed if e in pred)
                    wasted[k] += sum(1 for e in pred_new if e not in batch)
                misses_total += len(missed)
            cache.access(batch)
            if phase == DECODE:
                if prev[layer] is not None:
                    for a in prev[layer]:
                        for e in batch:
                            trans[layer][a][e] += 1
                prev[layer] = list(batch)
                if layer == 0:
                    tokens += 1
    print(f"trace: {tokens} decode tokens, policy lfu-aging:32 at {slots} slots, "
          f"{misses_total / max(1, tokens):.1f} misses per token")
    print(f"{'K':>4} {'misses covered':>15} {'prefetch reads/tok':>19} {'wasted/tok':>11}")
    for k in ks:
        print(f"{k:>4} {100 * covered[k] / max(1, misses_total):>14.1f}% "
              f"{predicted_total[k] / max(1, tokens):>19.1f} {wasted[k] / max(1, tokens):>11.1f}")


def cmd_calibrate(args):
    trace = read_trace(args.trace)
    name, _, param = args.policy.partition(":")
    aging = int(param or 64) if name == "lfu-aging" else 0
    window = int(param or 64) if name == "lfu-window" else 0
    base = "lfu" if name in ("lfu-aging", "lfu-window") else name
    dm, dr = replay(trace, [args.slots] * trace.num_layers, base, aging, window)
    print(f"trace: {trace.decode_tokens} decode tokens, {dr} decode expert requests")
    print(f"{args.policy} @ {args.slots} slots: {sum(dm)} decode misses, hit {100 * (1 - sum(dm) / dr):.1f}%")
    print("compare with the run's footer: 'expert cache: H% hit, M miss of R'")


def cmd_compare(args):
    trace = read_trace(args.trace)
    slots_list = [int(s) for s in args.slots.split(",")]
    policies = args.policies.split(",")
    print(f"trace: {trace.decode_tokens} decode tokens; miss {args.miss_ms} ms, base {args.base_ms} ms/token")
    print(f"{'policy':<10} {'slots':>5} {'hit%':>6} {'miss/tok':>9} {'MB/tok':>7} {'io ms':>7} {'est tok/s':>9}")
    for slots in slots_list:
        for policy in policies:
            name, _, param = policy.partition(":")
            if make_policy(policy, slots) is not None:
                dm, dr = replay_advanced(trace, slots, policy)
            else:
                aging = int(param or 64) if name == "lfu-aging" else 0
                window = int(param or 64) if name == "lfu-window" else 0
                base = "lfu" if name in ("lfu-aging", "lfu-window") else name
                dm, dr = replay(trace, [slots] * trace.num_layers, base, aging, window)
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
    pr = sub.add_parser("predict"); pr.add_argument("trace"); pr.add_argument("--slots", type=int, default=64)
    pr.add_argument("--k", default="0,8,16,32")
    c = sub.add_parser("calibrate"); c.add_argument("trace"); c.add_argument("--slots", type=int, required=True)
    c.add_argument("--policy", default="lfu-aging:32", help="the policy the run used (default: the production default)")
    p = sub.add_parser("compare"); p.add_argument("trace"); p.add_argument("--slots", default="16,64,128")
    p.add_argument("--policies", default="lfu,lru,lfu-aging,belady",
                   help="lfu-aging:N halves counts every N batches; lfu-window:W counts the last W batches; "
                        "also lrfu:LAMBDA, arc, segmented:DIVISOR (window = slots/DIVISOR, min 8), s3fifo, sieve, transition:BETA")
    a = sub.add_parser("allocate"); a.add_argument("--train", required=True); a.add_argument("--test", required=True)
    a.add_argument("--slots", default="16,64"); a.add_argument("--max-slots", type=int, default=128)
    args = parser.parse_args()
    {"calibrate": cmd_calibrate, "compare": cmd_compare, "allocate": cmd_allocate,
     "predict": cmd_predict}[args.command](args)


if __name__ == "__main__":
    main()
