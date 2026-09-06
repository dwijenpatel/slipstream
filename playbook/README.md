# The playbook

The staged pipeline for onboarding a newly released open-weight MoE.
Each stage can veto the ones after it, and each costs roughly ten times
the one before, so they run in this order. Stages 1, 3, and 5 are
scripted today; 2 and 4 are engineering; the goal is a new model in
days.

## Stages

1. **Manifest audit** (minutes, no GPU). Read the checkpoint: expert
   geometry (count, stride, topk), layer types (full attention, SWA,
   linear/GDN), head dims, KV math. This alone predicts cache-slot
   memory, KV growth, and which kernels apply.
2. **Convert and pack** (the expensive stage today). Port the
   architecture to the runtime and pack page-aligned expert blobs.
   Reference: Qwen3.6 support was a full model-family port (upstream
   PR #29). Driving this cost down is the playbook's main engineering
   goal.
3. **Op sweep vs measured ceilings** (an afternoon). Score every op
   against the device's measured (never spec-sheet) roofline. Price
   gaps by share of runtime, not by percent of roofline: the largest
   gap and the largest prize are routinely different ops.
4. **Kernel selection** (days, only where stage 3 says). Reuse the
   kernel library first; extend or evolve only for ops that are both
   inefficient and expensive. Every kernel lands behind a
   byte-identity gate before any timing counts.
5. **Measure the (memory budget x context) matrix** (hours, scripted).
   Slots curve, prefill chunk curve, decode phase split, at each
   supported RAM tier. Sweep hygiene: re-measure the first point last;
   if it does not reproduce, the sweep measured cache state, not the
   parameter.
6. **Emit the profile** (minutes). A `profiles/<model>/` entry:
   recommended settings per RAM tier, expected tok/s and TTFT, and the
   raw evidence that produced them.

## Rules carried from the companion methodology

- Measure the product end to end before optimizing anything; the stock
  path is often embarrassingly close to the ceiling.
- Ceilings are measured per device, never taken from spec sheets.
- A benchmark arm is invalid if any other model process ran during it,
  including test suites.
- Outputs must be byte-identical at fixed seed across any optimization
  that claims to be exact.

## Expert file-cache baseline

```bash
playbook/fill_table.sh --io-baseline --contexts 3k --slots 16,64 --max-new 1024 --passes 2
```

This measurement regime alternates the same release binary with normal expert
reads and `TURBO_FIELDFARE_EXPERT_NOCACHE=1`. The latter sets `F_NOCACHE` on
expert-streaming and expert SHA-verification descriptors; full integrity
verification is retained. The explicit slot budget and model math are unchanged.
It does not prewarm the model, purge caches, or duplicate the installation.

**Cache-disabled is not necessarily uncached.** On macOS, already resident file
pages can still satisfy these reads. `TURBO_FIELDFARE_IO_BASELINE=1` records
128-token decode windows with logical expert bytes, process physical disk-read
bytes (`proc_pid_rusage`), I/O await, elapsed time, and physical footprint. Windows
start at the first emitted token, excluding prefill and the seed token. The last
window is flushed when generation finishes. Physical reads are process-wide and
may include resident-weight page-ins; they are not an exact expert cache-hit
counter. A ratio substantially below one disproves a fully uncached run; a ratio
near one alone does not prove that every expert byte came from storage.

The driver also samples expert-file residency with `mincore` before and after
each run, without touching the mapped data. `--uncached-only` omits normal-cache
arms, so they cannot warm the next arm. It does not clear existing pages: any
exception to the repository's no-purge rule requires separate user approval.
The summary's strict evidence gate requires zero resident expert pages before
and after every arm and physical/logical decode-read ratios of at least 99%.
This gate is separate from drift and byte identity; all three must pass.

All passes, output text, timing footers, commands, binary digest, source diff,
machine specifications, and before/after VM and swap counters are saved beneath
`bench-results/io-baseline-*`. Only the final pass is the reported comparison.
Every arm is rerun as a drift control; either prefill or decode moving more than
5 percent rejects the comparison. Output hashes must agree across all arms and
passes for a given prompt. The original warm table mode remains available.

`playbook/expert_io_probe.c` provides a small read-only check of existing-page
behavior on one Qwen expert file. Build it with `clang`, run with argument `0`
then `1`, and inspect physical bytes on repeated reads. It is a mechanism probe,
not a model performance benchmark.

## Routing trace and cache-policy replay

`TURBO_FIELDFARE_ROUTE_TRACE=<path>` makes the CLI or the server record the
router's top-k expert IDs per token and layer, prefill and decode, to a small
binary file. `route_replay.py` replays that trace through any slot-cache
policy at any slot count in seconds, so one model run answers every policy
question:

```bash
uv run playbook/route_replay.py calibrate TRACE --slots 64      # must match the run's own miss count
uv run playbook/route_replay.py compare TRACE --slots 16,64,128 --policies lfu,lru,lfu-aging,belady
uv run playbook/route_replay.py allocate --train A.bin,B.bin --test C.bin --slots 16,64
```

`session_turn.py` drives one turn of a multi-turn session against the server
and records it; `session_replay.py` replays a recorded session's user
messages against another server configuration and checks the replies are
byte-identical, which they must be for any cache setting. The 2026-09-05
session and its traces are under `bench-results/route-replay-20260905`.

## Overnight plan, 2026-09-06

`overnight.sh` runs the measurements the review left owed, unattended. It
refuses to prompt for a password itself, so authenticate first:
`sudo -v && playbook/overnight.sh`.

1. Uncached, behind the purge: the prefetch A/B at 64 slots (off, one layer
   of lead, two layers, twice, interleaved); the recorded session under
   `lfu`, `lru`, and `lfu-aging`; the session under `lfu-aging` with
   prefetch on; prefill only at 12k and 24k.
2. Warm, after one read of the whole model: the session under `lfu-aging`;
   the clock hold off against auto at 64 slots; prefill only at 12k and 24k;
   then the slot-curve sweep (`fill_table.sh --only slipstream`).

Each arm is a fresh process with residency sampled before and after, and
`overnight_summary.py` writes `SUMMARY.md` in the output directory. With
prefetch on, the expert-cache counters count only misses at plan time, so
compare physical bytes per token, not the hit rate. Prove the plumbing with
`overnight.sh --smoke --skip-purge --skip-sweep`, about ten minutes; the
smoke's session arms cap turns at 64 tokens, so their identity check is
expected to fail.
