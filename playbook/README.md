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
