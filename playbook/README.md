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
