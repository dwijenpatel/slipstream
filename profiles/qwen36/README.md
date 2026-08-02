# Qwen3.6-35B-A3B profile

First playbook execution. Measured on M5 (10-core GPU), 24 GB, internal
SSD, macOS 26.5.2, 2026-08-01. Community long-synthesis case (2,940-token
prompt), fixed seed, warm steady state.

## Architecture facts (stage 1)

40 layers: 30 GDN linear-attention + 10 full attention (16 Q heads,
2 KV heads, head dim 256, GQA 8:1). 256 experts/layer, topk 8, expert
stride 1.77 MB. Packed experts 17 GB. GDN state is fixed size, so long
context grows only the 10 full layers' KV.

## Recommended settings by RAM tier (stage 5, partial)

| RAM tier | expert-cache slots | prefill chunk | measured |
| --- | --- | --- | --- |
| 8 GB | 16 (default) | auto | not yet measured on-tier |
| 16 GB | 64 | auto | extrapolated, verify |
| 24 GB | 64-128 | auto | 25-27 tok/s decode, 17.5 s prefill @3k |

Notes:
- Slots beyond 64 buy little on a 24 GB machine because the OS page
  cache already absorbs most expert reads; decode is 79 percent
  GPU-bound there. Slots matter most where page cache cannot hold the
  expert file (8-16 GB tiers); those tiers are unmeasured so far.
- `--prefill-chunk auto` is a pure win at every tier measured
  (1/chunks I/O law, verified to the one-sweep floor).

## Decode phase split (24 GB, 128 slots, per token of 40.7 ms)

| bucket | ms | note |
| --- | --- | --- |
| scheduling gaps | ~9.0 | roadmap item 2 |
| expert I/O await | 7.8 | roadmap item 3 |
| expert FFN (cb2) | 6.5 | ~70 percent of ceiling, leave |
| GDN attention (30 layers) | 6.1 | ~75 percent, leave |
| full attention (10 layers) | 2.4 | GQA-grouped split-KV kernel
(2026-08-01): KV scan ~90 percent of BW ceiling; context term cut 5.4x |
| output head | 2.4 | |
| norms + router | 1.3 | |

Raw evidence: companion repo telemetry (slots curve, cb1 split,
context-scaling probe), 2026-08-01 commits.
