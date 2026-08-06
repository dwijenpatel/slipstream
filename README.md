# slipstream

Mixture-of-Experts inference on Apple Silicon: experts streamed from SSD,
near-roofline Metal kernels, and a KV cache that survives the process.

Qwen3.6-35B-A3B generates at 21 tokens per second on a base M5 MacBook and
peaks at 1.9 GB of memory. Its weights are 18 GB. slipstream leaves them on
SSD and streams only the eight experts each token routes to, through a
bounded cache you size yourself, so memory becomes a dial rather than an
architecture decision. A 2,940-token prompt takes 18.5 seconds to prefill
the first time and 0.03 seconds on every run after that, because the KV
cache persists to disk and reloads byte for byte. Underneath, the kernels
are scored against this machine's measured ceilings rather than its spec
sheet, which overstates memory bandwidth by 27 percent: decode attention
now reads KV at about 90 percent of the real ceiling, and the output head
at 93.

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

## Measured state (M5, 10-core GPU, 24 GB)

Qwen3.6-35B-A3B, community long-synthesis case, warm, 2,940-token prompt:

| configuration | TTFT | decode | peak memory |
| --- | --- | --- | --- |
| upstream defaults | 63.3 s | ~18 tok/s | 1.13 GB RSS |
| slipstream, out of the box | 18.5 s | 21.4 tok/s | 1.90 GB |
| slipstream, 128-slot expert cache | 17.5 s | ~25 tok/s | grows with the cache |
| slipstream, resuming a saved KV cache | 0.03 s | ramps from cold | plus a 119 MB file |

Upstream's figure is resident set size as they publish it; slipstream's is
peak physical footprint, which counts GPU allocations that resident set
size misses. Measured the same way, slipstream's resident set size is
1.14 GB. The 128-slot and upstream rows were measured 2026-08-01, the rest
2026-08-05. Resuming a KV cache restores the prompt exactly, byte for byte,
but decode then starts against a cold expert cache, because prefill is
what normally warms it. That tradeoff is item 4 in the roadmap below.

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
5. MEASURED 2026-08-04, stage-1 decode polish mostly EXHAUSTED: the
   output head already runs at ~93 percent of the bandwidth ceiling
   (248k int4 vocab, 2.24 ms floor vs 2.4 measured; no kernel prize),
   and command-buffer merging (routed-FFN folded into next cb1, head
   into the final carry; 41 fewer commits + 1 fewer sync per token) is
   bit-identical but only ~0-2 percent, within noise: the 9 ms/token
   gap is wake latency on the one irreducible wait per layer, which
   only multi-token (speculative) decoding can amortize. Merge kept
   (default on; TURBO_FIELDFARE_NO_CB_MERGE=1 restores the old path).
   Speculative decoding is confirmed as the only remaining large
   decode lever: draft model + batched verify + GDN state
   checkpoint/replay via the chunked kernels.
6. Memory-budget dial: one flag that sets expert-cache slots, prefill
   chunk, and KV policy together
7. Playbook automation until a new model is days, not weeks

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
