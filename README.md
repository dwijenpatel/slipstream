# slipstream

Mixture-of-Experts inference on Apple Silicon: experts streamed from SSD,
near-roofline Metal kernels, and a KV cache that survives the process.

Qwen3.6-35B-A3B generates at 21 tokens per second on a base M5 MacBook, in
1.9 GB of memory. Its weights are 18 GB. slipstream leaves them on SSD and
streams only the eight experts each token actually routes to, staying
inside a RAM budget you set, so memory is a dial rather than a number the
model dictates. Prompts do not have to be paid for twice either: the KV
cache persists to disk, so a 2,940-token prompt that costs 18.5 seconds to
read the first time comes back in 0.03 seconds, byte for byte, on every run
after. Doing this without giving up speed is the point of the kernels
underneath, which are hand-written Metal tuned against what this chip
delivers rather than what its spec sheet claims: decode attention and the
output head each run at better than 90 percent of its real memory
bandwidth.

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

Qwen3.6-35B-A3B, community long-synthesis case, warm, 2,940-token prompt.
The first row is TurboFieldfare, the project slipstream forked from, run at
its own defaults:

| configuration | TTFT | decode | peak memory |
| --- | --- | --- | --- |
| TurboFieldfare, its defaults | 63.3 s | ~18 tok/s | 1.13 GB RSS |
| slipstream, out of the box | 18.5 s | 21.4 tok/s | 1.90 GB |
| slipstream, expert cache raised to 128 slots | 17.5 s | ~25 tok/s | up to 9.1 GB |
| slipstream, resuming a saved KV cache | 0.03 s | ramps from cold | plus a 119 MB file |

The expert cache is counted in slots, and one slot holds one expert's
weights for one layer. This model puts 256 experts in each of its 40
layers at 1.7 MB per expert, so the default 16 slots cap the cache at
1.1 GB and 128 slots cap it at 9.1 GB. Slots fill only as experts get
used, so those are ceilings rather than reservations. Eight times the
memory buys under 10 percent more decode speed here, which is the honest
shape of the dial: on this model it decides whether you can run at all far
more than it decides how fast.

TurboFieldfare's memory figure is resident set size, as they publish it;
slipstream's is peak physical footprint, which counts GPU allocations that
resident set size misses. Measured the same way, slipstream's resident set
size is 1.14 GB. The TurboFieldfare and 128-slot rows were measured
2026-08-01, the rest 2026-08-05. Resuming a KV cache restores the prompt
exactly, byte for byte, but decode then starts against a cold expert cache,
because prefill is what normally warms it. The section below returns to
that tradeoff.

Where a token's 40.7 ms went, at 128 cache slots: 9 ms waiting for the GPU
between layers, 7.8 ms awaiting expert reads from SSD, 6.5 ms in the expert
feed-forward, 6.1 ms in linear attention, 6.1 ms in full attention, 2.4 ms
in the output head, 1.3 ms in the tail. The model is a hybrid, so 30 of its
40 layers use a recurrent linear attention that keeps no per-token cache
and 10 use ordinary attention that does. Only the second kind grows with
context, which is why it was the first thing rewritten. Everything below
follows from this split rather than from intuition.

## What is built, and what it measured

Each item below was priced by measurement before and after, and the
negative results are kept because they are the ones that redirected the
work.

**Decode attention, rewritten around grouped-query attention.** This model
gives 16 query heads only 2 key/value heads, so the old kernel re-read the
same keys and values once per query head. The replacement splits the scan
across threadgroups and shares each read. The attention branch runs 2.49x
faster, reads keys and values at about 90 percent of the memory-bandwidth
ceiling, and the part of decode that grows with context shrank by 5.4x.

**The per-layer stall was measured, and it cannot be removed.** Every layer
sends its expert-routing choice back to the CPU, which then fetches those
experts. That round trip costs about 207 microseconds of wake latency, 40
times per token, or roughly 9 ms of every token. Replacing the wait with a
GPU fence and a CPU spin produced bit-identical output and ran 15 percent
slower, because the GPU's writes only become visible to the CPU around the
same boundary anyway. The wait is a floor. Useful work has to move into it
rather than around it.

**Routing can be predicted a layer ahead, but prefetching on it does not
pay here.** Running the next layer's router against the current layer's
state picks 81.9 percent of the experts that layer will actually want.
Prefetching on that prediction cut disk-wait time by 55 percent and still
made decode slower overall: on a machine whose memory bus is already
saturated, the prefetched bytes simply get collected twice. It ships off by
default and is worth revisiting when a model is far larger than RAM and the
GPU is genuinely idle during disk reads.

**The KV cache survives the process.** `--kv-snapshot` writes the cache to
disk and reloads it, turning an 18-second prefill into 0.03 seconds for the
same prompt with byte-identical output, at a cost of 119 MB per 3,000
tokens. The tradeoff found on the way: prefill had been quietly doing a
second job, warming the expert cache, so decode after a resume starts cold.
A background expert sweep after restore is the queued fix. Reusing a cache
across prompts that share only a prefix is still open.

**Single-token decode is close to its floor.** The output head already runs
at about 93 percent of the bandwidth ceiling, so there is no kernel prize
left there. Merging command buffers to cut 41 of the per-token
synchronization boundaries turned out bit-identical and worth under 2
percent, inside the noise. What remains is the 9 ms wake latency above, and
the only way to amortize it is to make one forward pass emit more than one
token.

## What is next

**Speculative decoding**, which is the only large decode lever left. The
machinery works and is measured: it drafts, verifies, and repairs its own
state correctly, and on code it accepts 33.9 percent of drafted tokens for
3.57 emitted tokens per round. It is still slower than plain decode,
because verifying k tokens at once needs matmul kernels that read weights
once for the whole batch, and the ones available here degrade to roughly k
separate passes below 32 rows. Two kernels close that gap. Design and
measurements are in `docs/SPEC_DECODE.md`.

**One memory budget flag** that sets expert-cache slots, prefill chunk size,
and KV policy together, instead of three knobs a user has to reason about
separately.

**Playbook automation**, until onboarding a newly released model is a matter
of days rather than weeks.

## Lineage and attribution

- Runtime core: forked from
  [TurboFieldfare](https://github.com/drumih/turbo-fieldfare)
  (Apache 2.0), full git history preserved. The bounded-memory expert
  streaming design is theirs.
- Qwen3.6 support started from TurboFieldfare PR #29.
- Design ideas credited to [oMLX](https://github.com/jundot/omlx)
  (Apache 2.0): content-addressed KV block persistence, macOS memory
  enforcement, NAX-metallib packaging. Ideas, not code, so far.
- Kernels and measurement methodology come from
  [gpu-kernel](https://github.com/dwijenpatel/gpu-kernel), a companion
  research repo. Its `METHODOLOGY.md` is the reason the numbers above
  carry the caveats they do: it records how each measurement technique
  was found wrong, and what the error cost.

## Building

Unchanged from upstream: macOS 26, Swift 6.2+, Metal 4.

```bash
swift build -c release
Scripts/test.sh
```

Upstream docs preserved at `docs/UPSTREAM_README.md`.
