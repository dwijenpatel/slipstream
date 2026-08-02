# slipstream

Fast Mixture-of-Experts inference on Apple Silicon, at any memory budget.

Big MoE models normally demand RAM the size of their weights. slipstream
streams routed experts from SSD through a bounded cache instead, so a
35B-parameter MoE runs in about 1 GB of RAM, and the same runtime scales
smoothly up to fully resident weights when the machine has headroom. Memory
is a dial, not an architecture choice.

## Thesis

Three projects each hold one corner of the problem:

- Resident-weight runtimes (mlx-lm, oMLX) are fast until the model does not
  fit, then stop working entirely.
- TurboFieldfare streams experts from SSD in bounded memory, but leaves
  measured factors of 2 to 5x on the table in decode and prefill.
- Hand-optimized and evolution-searched Metal kernels reach near-roofline
  on exactly the ops these runtimes spend their time in, but have no
  runtime to live in.

slipstream combines them: TurboFieldfare's streaming runtime as the core,
near-roofline kernels where measurement says they pay, idea imports from
oMLX where they are proven (KV block persistence, memory enforcement, NAX
metallib packaging), and one user-facing control that spans the whole
space: a memory budget.

The durable asset is the playbook: a staged, scripted pipeline that turns
a newly released open-weight MoE into a measured, optimized profile in
days. Models age out; the pipeline compounds. See `playbook/`.

## Measured state (M5, 10-core GPU, 24 GB, 2026-08-01)

Qwen3.6-35B-A3B, community long-synthesis case, warm:

| configuration | TTFT (3k prompt) | decode | RSS |
| --- | --- | --- | --- |
| upstream defaults | 63.3 s prefill | ~18 tok/s | 1.13 GB |
| slipstream today | 17.5 s prefill | ~25 tok/s | scales with budget |

Decode phase split at 128 cache slots (per token, 40.7 ms total):
scheduling gaps ~9 ms, expert I/O await 7.8 ms, expert FFN 6.5 ms,
GDN attention 6.1 ms, full attention 6.1 ms, head 2.4 ms, tail 1.3 ms.
The full-attention sdpa runs at ~24 percent of the memory-bandwidth
ceiling on its context-dependent part and is the only term that grows
with context. That measurement, not intuition, sets the roadmap below.

## Roadmap (value order, all priced by measurement)

1. DONE 2026-08-01: GQA split-KV decode attention (full-attn GPU time
   2.49x faster, KV scan ~90 percent of BW ceiling, context term cut
   5.4x, decode 24.9 to 26.8 tok/s at 3k)
2. MEASURED, folded into 3 (2026-08-01): the ~9 ms/token gap is 40
   irreducible ~207 us waitUntilCompleted round trips (router
   readback); a GPU-fence spin bypass is bit-identical but slower.
   The wait cannot shrink; work must overlap into it.
3. BUILT + measured 2026-08-01: predicted routing (layer L+1 router
   on layer L state) = 81.9 percent recall@top-8; env-gated prefetch
   cuts I/O await 55 percent but is NET SLOWER on resident-class
   models (bandwidth-saturated unified memory re-collects the bytes).
   Default off; re-test in the streaming regime (model >> RAM) where
   the GPU is genuinely idle during disk waits
4. SHIPPED v1 2026-08-01: --kv-snapshot exact-prefix resume. Reload
   beats recompute 588x on prefill (17.65 s -> 0.03 s, byte-identical,
   119 MB @3k tokens). Found tradeoff: prefill doubled as the expert
   cache warmer, so post-restore decode ramps from cold; background
   expert sweep after restore is the queued mitigation. Partial-prefix
   block reuse (the oMLX-blueprint hard part) remains open.
5. Memory-budget dial: one flag that sets expert-cache slots, prefill
   chunk, and KV policy together
6. Playbook automation until a new model is days, not weeks

## Lineage and attribution

- Runtime core: forked from
  [TurboFieldfare](https://github.com/drumih/turbo-fieldfare)
  (Apache 2.0), full git history preserved. The bounded-memory expert
  streaming design is theirs.
- Qwen3.6 support started from TurboFieldfare PR #29.
- Design ideas credited to [oMLX](https://github.com/jundot/omlx)
  (Apache 2.0): content-addressed KV block persistence, macOS memory
  enforcement, NAX-metallib packaging. Ideas, not code, so far.
- Kernels and measurement methodology come from a companion research
  repo (measurement discipline, roofline ceilings, evolutionary kernel
  search).

## Building

Unchanged from upstream: macOS 26, Swift 6.2+, Metal 4.

```bash
swift build -c release
Scripts/test.sh
```

Upstream docs preserved at `docs/UPSTREAM_README.md`.
