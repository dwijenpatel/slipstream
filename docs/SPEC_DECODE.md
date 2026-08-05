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

## M1 measured findings (2026-08-05)

The controller works end to end (rounds, accept, replay, mid-queue
repair, telemetry) and taught two things that reshape the plan:

1. BYTE-IDENTITY vs the sequential baseline is unachievable across
   kernel paths: the verify forward (chunked prefill kernels) rounds
   differently from decode kernels, so near-tie argmax positions flip
   (observed at char 1763/2600, both continuations fluent). Revised
   gate: spec output deterministic across repeats; accepted tokens
   consistent with their own verify pass by construction; fluency
   spot-check. Same numerics property as every production spec-decode.
2. The prefill path is the wrong verify vehicle: ~250 ms fixed cost
   per round (40 per-layer router syncs + expert fetch orchestration,
   independent of draft length) vs ~1.9 s total for all 93x9 head
   GEMVs. Long-synthesis greedy: 28.1 -> 12.3 tok/s, acceptance 6.4%
   on prose (prompt-lookup's worst case). No acceptance rate can pay
   a 250 ms round tax; break-even needs a verify forward costing about
   one decode token.

Code-domain probe (2026-08-05, repetitive-structure generation task):
determinism gate PASSES; acceptance 33.9 percent, 3.57 emitted/round;
verify 206 ms/round, replay 71 ms/round (timing counters now in the
telemetry line). Acceptance rose 5x from prose while round cost barely
moved - the tax is prefill-path fixed orchestration, settling the M2'
question quantitatively: at 3.57 emitted/round a ~47 ms batched-decode
verify round yields ~17 ms per emitted token, ~2x over the 27.7 tok/s
sequential baseline, before drafting improvements.

Revised milestone order:
M2': BATCHED DECODE FORWARD - decode-kernel path with tokenCount k+1:
     batched int4 projections, per-position attention (KV grows within
     the batch), CHUNKED GDN step (campaign-4 kernels), one router
     sync per layer per round, batched int4 head + per-row argmax.
     Target: verify round ~ 1.3x one decode token.
M3': adaptive gating (4-gram-only drafts, rolling-acceptance disable)
     + draft model; sampling after that.

## Invariants and gates

- Correctness gate (revised 2026-08-05, see findings): the spec path
  must be deterministic across repeated runs; accepted tokens are
  consistent with their own verify pass by construction; divergence
  from the sequential baseline at near-tie argmax positions is a
  documented numerics property, not a defect - both outputs are valid
  greedy continuations under their kernel's rounding.
- TURBO_FIELDFARE_SPEC=1 enables; default off until measured wins.
- Telemetry: rounds, drafted, accepted, emitted, replay tokens,
  acceptance histogram - printed under TURBO_FIELDFARE_PHASES=1.
- The KV snapshot feature composes: snapshot/restore must remain
  correct with spec decode enabled (positions only ever move forward
  at emission boundaries).
