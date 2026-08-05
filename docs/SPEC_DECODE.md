# Speculative decoding design

Motivation, from measurement (2026-08-04): single-token decode is fenced
at ~26-27 tok/s by structure, not kernels. Per 512-token decode: the one
irreducible per-layer sync costs ~9 ms/token in wake latency, expert
reads cost ~566 MB/token, and the head reads ~270 MB/token; every
per-token cost divides by the number of tokens a forward pass emits.
Multi-token (speculative) decoding is the only remaining large lever.

## Milestones

M1 (this doc's target): GREEDY spec decode with PROMPT-LOOKUP drafting.
   No draft model: draft tokens are copied from the longest n-gram match
   of the current suffix elsewhere in the context (strong on code and
   agentic text, free, tokenizer-identical by construction). Greedy
   only, so correctness is byte-identity against the greedy baseline.
   Metrics: acceptance rate, emitted tokens per round, tok/s.
M2: draft model (small resident Qwen with the same 248,320-token
    vocabulary) + the same verify machinery.
M3: sampling support via standard rejection sampling (needs
    per-position probabilities, not just argmax).
M4: batched int4 head (one GEMM for k+1 positions instead of k+1
    GEMVs) - the head weights are read once per round instead of once
    per position.

## One round (M1)

Let P = accepted position, k = draft length (adaptive, 0..8).

1. Draft: prompt-lookup proposes d[0..k) continuing the sequence.
   k=0 (no match) falls back to one normal decode step.
2. Checkpoint: GDN recurrent state + conv tails for all 30 linear
   layers copied to preallocated shadow buffers (~62 MB, in-memory).
3. Verify forward: the prefill chunk path runs [t_P, d_0 .. d_{k-1}]
   (k+1 tokens) from position P: full-attn KV written at P..P+k, GDN
   state advanced in place, per-position hidden states left in the
   prefill scratch.
4. Per-position greedy head: the fused norm+GEMV+argmax head runs on
   each of the k+1 hidden rows -> g[0..k]. (M4 batches this.)
5. Accept: a = longest prefix with g[i] == d[i]. Emit
   g[0..a] (a+1 tokens: a accepted drafts + 1 free correction token).
6. State reconciliation:
   - Full acceptance (a == k): state is exactly right; continue.
   - Partial: restore GDN shadow, replay [t_P, d_0..d_{a-1}, g[a]]
     (a+2 tokens) through the same chunk path WITHOUT the head; KV
     position rewinds by bookkeeping alone (slots P+a+1.. are simply
     overwritten next round).
   Replay costs one small forward; its expected cost is bounded by the
   acceptance distribution and measured, not assumed.

## Cost model (to validate against measurement)

Per round with k drafts and a accepted: one (k+1)-token forward
(~ decode cost of 1 token + marginal per extra token, since weights
dominate reads), (k+1) head GEMVs at 2.24 ms until M4, checkpoint
~1 ms, replay on partial acceptance. Break-even needs mean emitted
tokens/round comfortably above the round overhead ratio; prompt-lookup
acceptance on code is the empirical unknown M1 exists to measure.

## Invariants and gates

- Greedy mode output must be BYTE-IDENTICAL to the non-speculative
  greedy baseline on the frozen benchmark prompts. Always-on gate.
- TURBO_FIELDFARE_SPEC=1 enables; default off until measured wins.
- Telemetry: rounds, drafted, accepted, emitted, replay tokens,
  acceptance histogram - printed under TURBO_FIELDFARE_PHASES=1.
- The KV snapshot feature composes: snapshot/restore must remain
  correct with spec decode enabled (positions only ever move forward
  at emission boundaries).
