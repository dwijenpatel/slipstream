# slipstream

Mixture-of-experts inference on Apple Silicon with the weights left on the
SSD. The memory a model uses is a setting, and the setting is measured.

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) has 18 GB of
expert weights.[^1] On a base M5 MacBook Pro with 24 GB of memory, slipstream
runs it in 5.6 GB of peak footprint at 30.8 tokens per second against a
2,940-token prompt, or in 2.5 GB at 26.5 tokens per second.[^2] The weights
stay on the SSD. Each token routes to eight of the 256 experts in each of the
model's 40 layers, and only those eight are read. Everything else on the
machine keeps its memory.

A prompt is paid for once. After a fresh prefill the whole cache, including
the recurrent state of the model's linear-attention layers, can be written to
disk. The same 2,940-token prompt that took 17.4 seconds to read comes back in
0.03 seconds on the next run, with byte-identical output.[^3]

This page states what was measured, on which machine, on which date, and what
was not. The runtime design is inherited from
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare); section 3 says
what this project adds to it.

## 1. What slipstream is

Three projects each hold one part of the problem. Resident-weight runtimes such
as [mlx-lm](https://github.com/ml-explore/mlx-lm) are fast while the model fits
in memory and stop working when it does not. This model runs resident on this
machine only by exceeding the GPU's wired limit, 21.6 GB of peak footprint
against a 21.3 GB limit, and does not fit at all on a 16 GB Mac.
TurboFieldfare streams experts from the SSD inside a bounded memory budget,
and its measured cost is time: 42.5 seconds to prefill a 2,940-token prompt,
and 13.0 tokens per second at a 24k prompt. The companion research repository
[gpu-kernel](https://github.com/dwijenpatel/gpu-kernel) measures this
machine's real ceilings and writes Metal kernels against them, and had no
runtime to put them in.

slipstream is TurboFieldfare's streaming runtime with kernels where measurement
says they pay, a cache that survives the process, and one control the user
sets: how much memory to spend. Its target machine is a laptop the user keeps
working on while a coding model runs, which makes memory the constraint that
binds first, before speed.

## 2. Measured results

Every number in this section comes from one harness,
`playbook/fill_table.sh`, on one machine: a base M5 with a 10-core GPU and
24 GB, macOS 26.5.2, internal SSD, GPU wired limit 21.3 GB. Each run is a
fresh process. The whole arm list runs in two round-robin passes and only the
second is recorded, so every arm is equally warm. The first arm is re-run at
the end as a drift control, and the sweep is discarded if the control moves
more than 5 percent.[^2] The four prompts hold 889, 2,940, 11,738, and 23,827
tokens, and every cell generates 512 tokens.

The expert cache is counted in slots. One slot holds one expert's weights for
one layer, 1.77 MB on this model, so 64 slots across 40 layers cap the cache at
4.5 GB. Slots fill only as experts are used, so the cap is a ceiling rather
than a reservation. Sixty-four slots is the default.

Time to first token, in seconds, by prompt length. The slipstream rows here
predate the prefill attention kernel of section 5, which carries the
2026-09-04 measurement:

| runtime and setting                 | peak memory | 1k   | 3k   | 12k   | 24k   |
| ----------------------------------- | ----------- | ---- | ---- | ----- | ----- |
| TurboFieldfare, default 16 slots    | 2.0 GB      | 14.3 | 42.5 | 185.1 | 664.8 |
| slipstream, 16 slots                | 2.5 GB      | 9.3  | 17.4 | 116.9 | 381.7 |
| TurboFieldfare, 32 slots            | 3.0 GB      | 15.5 | 70.5 | 265.5 | 720.5 |
| slipstream, 32 slots                | 3.5 GB      | 9.4  | 17.3 | 116.9 | 381.4 |
| slipstream, 64 slots (default)      | 5.6 GB      | 9.4  | 17.4 | 117.0 | 381.8 |
| slipstream, 96 slots                | 7.8 GB      | 9.5  | 17.5 | 117.2 | 382.0 |
| slipstream, 128 slots               | 9.9 GB      | 9.5  | 17.5 | 117.4 | 382.8 |
| slipstream, 192 slots               | 14.1 GB     | 9.5  | 17.4 | 145.5 | 475.4 |
| llama.cpp, default                  | 16.2 GB RSS | 1.2  | 3.8  | 18.5  | 44.9  |
| llama.cpp, `--n-cpu-moe 32`         | 16.8 GB RSS | 2.5  | 7.8  | 35.0  | 78.2  |
| llama.cpp, that plus `--mmap 0`     | 17.2 GB     | 2.8  | 11.0 | 51.4  | 141.8 |
| slipstream, resuming a saved cache  | 2.5 GB      |      | 0.03 |       |       |

Sustained decode, in tokens per second, by prompt length:

| runtime and setting                 | peak memory | 1k   | 3k   | 12k  | 24k  |
| ----------------------------------- | ----------- | ---- | ---- | ---- | ---- |
| TurboFieldfare, default 16 slots    | 2.0 GB      | 26.1 | 23.8 | 17.2 | 13.0 |
| slipstream, 16 slots                | 2.5 GB      | 24.8 | 26.5 | 23.7 | 22.0 |
| TurboFieldfare, 32 slots            | 3.0 GB      | 28.1 | 25.3 | 17.5 | 13.1 |
| slipstream, 32 slots                | 3.5 GB      | 26.9 | 28.9 | 24.9 | 22.6 |
| slipstream, 64 slots (default)      | 5.6 GB      | 27.1 | 30.8 | 23.9 | 22.2 |
| slipstream, 96 slots                | 7.8 GB      | 25.6 | 28.7 | 22.6 | 21.3 |
| slipstream, 128 slots               | 9.9 GB      | 26.1 | 29.1 | 22.7 | 22.0 |
| slipstream, 192 slots               | 14.1 GB     | 24.7 | 23.9 | 12.4 | 12.1 |
| llama.cpp, default                  | 16.2 GB RSS | 37.6 | 38.6 | 38.3 | 38.9 |
| llama.cpp, `--n-cpu-moe 32`         | 16.8 GB RSS | 22.9 | 22.9 | 22.9 | 23.3 |
| llama.cpp, that plus `--mmap 0`     | 17.2 GB     | 22.3 | 20.8 | 22.0 | 20.0 |
| mlx-lm, all weights resident        | 21.6 GB     |      | 41.2 |      |      |

Memory is peak physical footprint, which counts GPU allocations that resident
set size misses, except in the two llama.cpp rows that leave `mmap` on. There
the footprint counts almost nothing because the weights are file-backed, so
those rows report resident set size instead. The llama.cpp rows come from
`llama-bench`, which times prefill and generation directly instead of serving
a request, so they are a best case rather than a like-for-like row.[^4] The
mlx-lm row was measured once, at 3k, because its first run also materializes
19 GB of lazily mapped weights and its peak sits above this machine's wired
limit; its time to first token is left out because that run charged the
materialization to prefill.[^5] Ollama, LM Studio's MLX engine, and
[oMLX](https://github.com/jundot/omlx) were not measured.

**The drift control on this sweep read 8.4 percent apart**, so differences
smaller than that within the slipstream rows are noise. A separate three-pass
sweep at the 3k prompt, with a drift control that read 2.2 percent, gives the
slot count its real effect: 25.1 tokens per second at 16 slots, 27.8 at 64, a
gain of 10.5 percent, and 17.9 at 192.[^6] More memory stops helping at 64
slots and starts hurting at 192, because the expert cache and the operating
system's file cache compete for the same RAM. Past about 64 slots the cache
evicts the file pages that were absorbing its own misses, and at 192 slots the
hit rate is 97.7 percent and the arm is the slowest one measured. The optimum
belongs to the host, not the model, and will move on a machine with a
different amount of memory.

The resume row has no decode figure because the number would mislead. Prefill
also fills the expert cache, so decode after a resume starts against an empty
one: 25.3 tokens per second before the snapshot against 13.5 over the first
256 tokens after it, climbing as the slots refill.[^3] The honest version is a
curve, and it has not been measured.

The TurboFieldfare comparison depends on prompt length. Comparing the 16-slot
rows, the closest memory match to the upstream default: at 1k the upstream
runtime is 5 percent faster at decode. At 3k slipstream is 11 percent ahead,
close to the sweep's 8.4 percent drift control. At 12k it is 38 percent ahead
and at 24k 69 percent ahead, because the upstream decode rate falls from 26.1
to 13.0 tokens per second across that range while slipstream holds between
24.8 and 22.0. That is the decode attention kernel of section 3 doing what it
was written for: it cut the part of each token that grows with context.
Prefill moved too, 42.5 to 17.4 seconds at 3k and 664.8 to 381.7 at 24k, from
a larger prefill chunk.

## 3. What is new here, and what is not

Most of slipstream is inherited. The bounded-memory streaming design, the
per-layer least-frequently-used slot cache, the int4 kernels, the repacker,
the Mac app, and the server are TurboFieldfare's. The Qwen3.6 port began from
an upstream pull request.[^7] What this project adds is smaller, and most of
it is evidence rather than mechanism.

**A decode attention kernel that reads each key and value row once per
key-value head.** The model gives 16 query heads only 2 key-value heads.
TurboFieldfare's kernel re-read the same rows once per query head. The
replacement gives one threadgroup to one key-value head and one chunk of the
sequence, holds each query head's vector in one simdgroup's registers, and
shares every row it loads across all eight heads. The full-attention branch of
a decode run fell from 3,106 ms to 1,249 ms, 2.49 times faster, and the
key-value scan runs at about 90 percent of this machine's measured 120.4 GB/s
memory bandwidth.[^8] MLX has a read-once kernel of the same design, but not
for this head dimension: on this head shape MLX's stock kernel measured 57 to
62 percent of bandwidth in August, and a read-once version written in the
companion repository, not yet upstream, measured 94 to 98.[^9] Grouped-query
sharing is not a new idea. Having it in a Swift and Metal streaming runtime,
gated by reference tests at both production shapes, is what is new.

**Whole-prompt prefill chunks, and the measurement that justified them.** The
runtime prefills a prompt in chunks and streams the experts each chunk routes
to. On a streaming runtime that means every chunk re-reads most of the expert
pool. Raising the chunk from 128 tokens to the whole prompt cut the bytes read
during a 2,940-token prefill from 247 GB to 38 GB and the prefill from 63.3 to
18.5 seconds, with byte-identical output.[^10] The rule that fell out is that
prefill I/O on this design scales with the number of chunks, not the number of
tokens, until the chunk covers the prompt.

**A cache snapshot for a hybrid model.** Thirty of the model's 40 layers use a
recurrent linear attention whose state cannot be sliced by token, which is
what makes block-level cache reuse hard on this architecture. slipstream
snapshots the whole state instead: the key-value cache of the ten
full-attention layers, the recurrent state and convolution tails of the
thirty linear layers, and the seed logits, as one 119 MB file for a
2,940-token prompt. Reload is 588 times faster than recompute, and the same
benchmark exposed the cold-cache tradeoff above, which the idea's source never
measured.[^3]

**The memory-versus-speed curve, with its noise stated.** This project has
found no other published measurement of how a streaming MoE runtime on Apple
Silicon responds to its cache budget across prompt lengths with a drift
control on each sweep. The finding that a larger cache can be slower, because
two caches compete for one pool of memory, came from that measurement and not
from a model of it.

**A ported prefill attention kernel for head dimension 256.** The kernel
gpu-kernel's search produced for this shape now runs the model's ten
full-attention layers on the M5's tensor units, with the runtime's strides
and lengths as parameters instead of compiled-in constants. It is a port,
not a new design; section 5 has what it measured.

**Three priced negative results.** Each was built, measured, and left off. They
are in section 4.

**A speculative-decoding scaffold whose measurement located the block.** It
drafts, verifies, and repairs the recurrent state correctly, and it is slower
than plain decode. The measurement says why, and the reason was not the one
the design assumed. Section 4 has the numbers.

None of these is a new algorithm. Grouped-query sharing, whole-prompt prefill,
state snapshots, and speculative decoding all exist elsewhere. The claim this
project makes is narrower: on this class of machine, for this class of model,
these are the measured effects, including the ones that went the wrong way.

## 4. Where a token's time goes

Measured at 128 slots on a 3k prompt, one token costs 40.7 ms.[^11] Of that,
9.0 ms is spent waiting for the GPU between layers, 7.8 ms awaiting expert
reads, 6.5 ms in the routed expert feed-forward, 6.1 ms in the thirty
linear-attention layers, 2.4 ms in the ten full-attention layers, 2.4 ms in
the output head, and 1.3 ms in norms and the router. Those sum to 35.5 ms; the
remaining 5.2 ms is command encoding and time the counters do not attribute.

**The wait between layers is a floor.** Every layer sends its routing choice
back to the CPU, which then fetches the chosen experts, so every layer ends in
a command-buffer round trip. That round trip costs about 207 microseconds on
this machine regardless of the work inside it, and there are 40 of them per
token, about 8.3 ms.[^12] Replacing the wait with a GPU fence and a CPU spin
produced bit-identical output and ran 15 percent slower, because the GPU's
writes become visible to the CPU at about the same boundary anyway.[^13] MLX
avoids the cost by keeping routing indices on the GPU, which a runtime that
fetches from disk on the CPU cannot do.

**Prefetching on predicted routing does not pay, and the obvious objection was
tested.** Running the next layer's router against the current layer's state
picks about 82 percent of the experts that layer will want. Prefetching on
that prediction cut the measured disk wait from 15.95 to 7.01 ms per token at
16 slots and made decode 11.2 percent slower.[^14] The runtime already commits
GPU work before it issues the fetch, so the counter measures an overlapped
wait rather than a stall, and moving the same bytes earlier only crowds a
saturated bus. It ships off by default. The untested case is a model far
larger than memory, where the GPU would stall for real.

**Merging command buffers is worth less than the noise.** Removing 41 of the
per-token synchronization boundaries was bit-identical and under 2 percent.[^15]

**The output head and the attention scan are near their bandwidth ceilings.**
The head reads about 270 MB of weights per token at about 93 percent of the
measured bandwidth, and the attention scan at about 90 percent, so there is no
kernel prize left in either.[^8] The routed expert feed-forward reads about
566 MB per token, which at the measured bandwidth is a 4.7 ms floor against
6.5 ms measured, about 72 percent by arithmetic; the remaining 1.8 ms per
token is the largest kernel-level gap in decode.

**Speculative decoding is the only large lever left, and it is not yet a
win.** The scaffold drafts by prompt lookup, verifies by running the draft
through a batched forward, and repairs the recurrent state on rejection. On
code it accepts 33.9 percent of drafted tokens for 3.57 emitted tokens per
round, and it still decodes slower than the sequential path, 16.9 against
27.7 tokens per second.[^16] The cost is not acceptance. A verify round costs
178 ms against a 47 ms target, because the batched matrix kernels available
here read the weights once per row below 32 rows, so verifying nine tokens
costs about nine GEMVs. Expert traffic also scales with verified tokens, not
emitted ones, although the eight experts each drafted token routes to overlap
55 percent within a round. A multi-row int4 kernel that reads weights once
now serves the projections and the head; on a code continuation at 128 slots
it took speculative decode from 16.9 to 22.3 tokens per second against 28.6
sequential, with byte-identical output.[^17] The routed experts and the
linear-attention projection still run per token, and they are the rest of
the round.

## 5. Where prefill time goes

Until 2026-09-04 the time to first token grew faster than the prompt. Timing
each 4096-token chunk of the 11,738-token prompt gave 26.3, 42.8, and 53.3
seconds for chunks that see 4,096, 8,192, and 11,736 keys.[^18] A fit put the
cost at 4.4 ms per token plus about 1 microsecond per token-key pair. The
second term was attention: 55 percent of the 12k prefill and about 73 percent
of the 24k one. The cause was a kernel-selection gap. Qwen's full-attention
layers have a head dimension of 256, the tensor-unit prefill attention kernel
inherited from upstream accepted only Gemma's 512-wide shape, and Qwen fell
to a scalar fallback that gave one threadgroup to each query token per head,
ran two threadgroup barriers per key, and re-read the keys and values once
for each of the 16 heads, at about one percent of the tensor unit's ceiling.

The kernel that closed the gap is a port of gpu-kernel's prefill-attention
champion: 32-query by 128-key tiles, both matrix products on the tensor units,
a grid ordered so the eight query heads that share a key-value head walk the
keys together, and final tiles that overlap instead of reading past the end.
The same probe, same prompt, same machine, after the port:

| chunk | tokens | keys visible at end | before | after |
| --- | --- | --- | --- | --- |
| 1 | 4096 | 4096 | 26.3 s | 18.3 s |
| 2 | 4096 | 8192 | 42.8 s | 18.1 s |
| 3 | 3544 | 11736 | 53.3 s | 16.1 s |

The per-token cost no longer grows with position. At 24k the six chunks took
between 16 and 19 seconds each, 108.9 seconds in all against 381.7 before.
The harness then measured time to first token at the two cache sizes that
matter, in the same round-robin protocol as section 2:[^19]

| slots | 3k before | 3k after | 12k before | 12k after | 24k before | 24k after |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 17.4 s | 14.5 s | 116.9 s | 53.8 s | 381.7 s | 112.2 s |
| 64 | 17.4 s | 15.2 s | 117.0 s | 54.5 s | 381.8 s | 112.7 s |

Two caveats on the "after" column. The September sweep ran on a busy machine,
with a load average between 5 and 9 and Spotlight indexing, where the August
sweep ran overnight on an idle one. Its decode rates read about 30 percent
below August at every cell, and a same-session interleaved A/B put the
pre-port binary at the same depressed rate, 20.3 and 18.5 tokens per second
against the new binary's 20.0 and 20.0, so the drop is the machine and not
the change; those decode figures are not quoted.[^20] The pre-port binary also
prefilled the 3k prompt in 18.4 and 19.0 seconds that day against 17.4 in
August, so the "after" times carry a few percent of the same load and are
pessimistic. Section 2's tables remain the August measurement until a full
overnight sweep replaces them.

What remains is the linear term, about 4.5 ms per token at every length. Above
4,096 tokens, prefill re-reads the expert pool once per chunk, about 18 GB
each, which a layer-major schedule would cut to one read: about 22 seconds of
the 112 at 24k, and nothing at 3k. The rest is the routed experts, the
linear-attention layers, and the projections, and their shares are not yet
attributed.

## 6. Which engine to run

The field for this model on a Mac is three engine families.
[llama.cpp](https://github.com/ggml-org/llama.cpp) runs GGUF files;
[Ollama](https://ollama.com) and [LM Studio](https://lmstudio.ai) wrap it.
[MLX](https://github.com/ml-explore/mlx) is Apple's framework; mlx-lm is its
reference library, and oMLX is a server built on it with its own kernels,
speculative decoding from a multi-token-prediction head, and a
content-addressed cache that persists to SSD. Expert streaming keeps weights
on the SSD and fetches what each token routes to: TurboFieldfare,
[NVMAI](https://github.com/Pummelchen/NVMAI), and slipstream. Runtimes that
require every weight resident, or have no Apple Silicon build, were ruled out
for this machine.

Take mlx-lm, or LM Studio's MLX engine over it, for speed paid for in memory:
41.2 tokens per second at 21.6 GB, which leaves nothing for a longer context
or a second program. oMLX's own single-stream additions are large on other
machines, 85 to 140 tokens per second on this model on an M3 Ultra by its
authors' measurement, but no number exists for it on an M5, and several of
its Qwen fast paths are disabled there.[^21] Take llama.cpp, or Ollama over
it, for the widest model and quantization choice, and note its two
configurations: resident by default at 16 GB, or expert tensors paged from
the SSD once both `--n-cpu-moe` and `--no-mmap` are set, at which point the
table shows it decoding at 20.8 tokens per second against 38.6 resident, in
the same 17 GB. Take a streaming runtime when memory binds, which on a laptop
that is also doing other work is the common case.

<a id="command-line-interface"></a>

## 7. Using it

The requirements are macOS 26 with Metal 4 and Swift 6.2 or newer; the tree
builds on Swift 6.3. Build the products, then repack the pinned Qwen3.6
checkpoint into the runtime's page-aligned expert layout. The install is about
19.6 GB on disk, and the installer streams byte ranges rather than downloading
a full snapshot.

```bash
swift build -c release
```

```bash
.build/release/slipstream-repack --model qwen36 --output ~/models/qwen36.gturbo
```

Generate from the command line. The default cache is 64 slots and the default
prefill chunk covers the whole prompt. Add `--kv-snapshot <path>` to save the
cache after a fresh prefill and reuse it on the next identical prompt.

```bash
.build/release/slipstream --model ~/models/qwen36.gturbo --prompt "The capital of France is" --max-new 64
```

Serve on the loopback interface. The server speaks the OpenAI chat-completions
API and the Anthropic messages API, serves a chat page at its root, keeps one
conversation's cache prefix in memory, and cancels generation when the client
disconnects. Coding agents run against it: `Scripts/claude-local.sh` points
Claude Code at a local server.

```bash
.build/release/slipstream-server --model ~/models/qwen36.gturbo --port 8091 --max-context 65536
```

The server binds to `127.0.0.1` with no authentication. Do not expose it.
Only one model process should run at a time; a second one contaminates every
measurement and competes for the same memory. `Scripts/test.sh` runs the
serial test suite, 587 tests at the time of writing.

## 8. Lineage and attribution

- Runtime core: forked from
  [TurboFieldfare](https://github.com/drumih/turbo-fieldfare) (Apache 2.0),
  full git history preserved. The bounded-memory expert streaming design is
  theirs, and so are the repacker, the app, and most of the kernels.
- Qwen3.6 support started from TurboFieldfare
  [pull request 29](https://github.com/drumih/turbo-fieldfare/pull/29).
- Ideas credited to [oMLX](https://github.com/jundot/omlx) (Apache 2.0):
  content-addressed cache blocks that persist to the SSD, memory enforcement
  on the process footprint that macOS actually kills on, and shipping
  tensor-unit kernels as a second Metal library gated on the SDK. Ideas, not
  code; the memory guard and the block-aligned cache are not yet built here.
- Kernels and measurement method come from
  [gpu-kernel](https://github.com/dwijenpatel/gpu-kernel), a companion
  research repository. Its methodology document records how each measurement
  technique was found wrong and what the error cost, which is why the numbers
  above carry the caveats they do.

This page describes commit `1c99256` and measurements taken between
2026-08-06 and 2026-09-04.

[^1]: Architecture facts from the
    [model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B): 40 layers, of
    which 30 are gated-delta-net linear attention and 10 are full attention
    with 16 query heads, 2 key-value heads, and head dimension 256; 256 routed
    experts per layer with 8 active per token. The 18 GB figure is the packed
    expert files of the 4-bit checkpoint as installed here, 1,769,472 bytes
    per expert per layer.

[^2]: `playbook/fill_table.sh` at commit `2aff18e`, results in
    `bench-results/table-20260806-022221/results.csv`, 2026-08-06. The
    TurboFieldfare rows are `bench-results/table-20260806-044148` and the
    llama.cpp rows `bench-results/table-20260806-055113`, same day, same
    harness, same prompt files. TurboFieldfare was run from a build of its own
    tree at its defaults. The drift control for the slipstream sweep is the
    `DRIFT-slipstream-slots16` row: 26.909 against 24.822 tokens per second at
    1k, 8.4 percent.

[^3]: Commit `bb06e5d`, 2026-08-01: prefill 17.65 seconds to 0.03 seconds on
    reload, 588 times, outputs byte-identical to a no-snapshot run at a fixed
    seed; decode after restore 25.3 to 13.5 tokens per second over the first
    256 tokens. The snapshot feature is on the command-line interface
    (`--kv-snapshot`); the server keeps one prefix in memory and does not
    persist it.

[^4]: `llama-bench` measures prompt processing and generation as separate
    timed loops; the time to first token in the table is prompt tokens
    divided by its prompt-processing rate. Model file
    `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`.

[^5]: `playbook/mlx_lm_bench.py`, commit `33b863e`, 2026-08-05: 41.2 tokens
    per second at a 21.6 GB peak, `mlx-community/Qwen3.6-35B-A3B-4bit`, same
    3k prompt file, greedy. The first run took about 150 seconds to first
    token because it also materialized the mapped weights; that number is
    not prefill and is not in the table.

[^6]: `bench-results/table-20260806-164925/results.csv`, commit `5693167`,
    three round-robin passes at the 3k prompt, 512 generated tokens: 25.116
    tokens per second at 16 slots, 26.636 at 32, 27.763 at 64, 25.974 at 96,
    25.929 at 128, 17.859 at 192; drift control 24.567 against 25.116, 2.2
    percent. Hit rates 78.7 percent at 64 slots and 97.7 percent at 192 from
    the runtime's own counters. An earlier figure of 27 percent for 16 to 128
    slots was measured without the page-cache leveling and is retracted.

[^7]: [TurboFieldfare pull request 29](https://github.com/drumih/turbo-fieldfare/pull/29),
    open as of 2026-09-04. This project's own upstream contribution, the
    configurable prefill chunk, is
    [pull request 53](https://github.com/drumih/turbo-fieldfare/pull/53), also
    open.

[^8]: Commit `a638cf8`, 2026-08-01, with long-sequence reference tests at
    both production shapes. Timings and bandwidth ratios are from the
    runtime's per-phase GPU counters and the companion repository's
    measurement log; the 120.4 GB/s ceiling is that repository's measured
    sustained read bandwidth on this machine, against a 153 GB/s
    specification. The context length at which the 90 and 93 percent figures
    were taken is not recorded, and figures measured on a working set that
    fits the system cache read high; treat both as approximate.

[^9]: gpu-kernel, `mlx-kernel-a-evidence.md` and `telemetry/kernel_a_ab.csv`,
    2026-08-27: MLX's stock decode kernel at head dimension 256 and a
    query-to-key-value ratio of 8 measured 57 to 62 percent of the 120.4 GB/s
    ceiling at a 32k key-value length; the read-once kernel written there
    measured 94 to 98 percent, 1.71 times faster, in three alternating pairs
    on an idle machine.

[^10]: Commit `c0d3f28`, 2026-08-01, measured with `iostat` and `time -l` on
    the community long-synthesis prompt: bytes read 247 GB to 38 GB, prefill
    63.3 to 18.5 seconds, wall 81.1 to 37.6 seconds, token-identical output.

[^11]: `profiles/qwen36/README.md`, from the runtime's phase counters under
    `TURBO_FIELDFARE_PHASES=1`, 2026-08-01, 128 slots, warm, 3k context. The
    full-attention figure is the post-rewrite one; the pre-rewrite figure was
    6.1 ms.

[^12]: gpu-kernel `METHODOLOGY.md`: a Metal commit-and-wait round trip costs
    about 207 microseconds on this machine regardless of kernel size, measured
    with a standalone 40-line binary; about 15 microseconds when empty and
    about 38 when eight are pipelined.

[^13]: Commit `0c3358f`, 2026-08-01: 26.8 to 22.8 tokens per second with the
    fence-and-spin wait, bit-identical output.

[^14]: Commit `0358b9f`, 2026-08-06, at 16 slots: routing recall 82.5
    percent, disk wait 15.95 to 7.01 ms per token, throughput 40.4 to 45.5 ms
    per token. The first test, at 128 slots, is commit `03e83fb`.

[^15]: Commit `4249aa3`, 2026-08-04.

[^16]: `docs/SPEC_DECODE.md`, 2026-08-05, code-domain probe at 128 slots:
    111 rounds, 33.9 percent acceptance, 3.57 emitted tokens per round,
    verify 178 ms per round. The kernel pricing that follows is recorded in
    the same document and in the companion repository's log of 2026-08-06.

[^17]: `docs/SPEC_DECODE.md`, section "M2' v2, first kernel", 2026-09-04:
    raw code continuation, 512 tokens, greedy, page cache leveled, 128
    slots; acceptance 41.6 percent, 3.91 emitted per round, verify 137 ms
    per round, down from 178. At 64 slots the round's union of experts
    thrashes the cache and the speculative rate is 17.0.

[^18]: Measured 2026-09-04 with a build of commit `01f7d5e` that prints a
    timestamp at each prefill chunk boundary, on the same 11,738-token prompt
    file the tables use, in a fresh process with no other model process
    running. The fit predicts the third chunk at 50.4 seconds against 53.3
    measured. Attention FLOPs at 12k are 11.3 TFLOP, which over the fitted
    67.8 seconds is 0.17 TFLOPS against the tensor unit's measured 15.4.

[^19]: `playbook/fill_table.sh --only 'slipstream, (16|64) of' --contexts
    3k,12k,24k` at commit `1c99256`, 2026-09-04 14:07 to 14:26, results in
    `bench-results/table-20260904-140715/results.csv`. Two round-robin
    passes, the second recorded; drift control 18.508 against 18.152
    tokens per second at 3k, 2.0 percent. The "before" column is the
    2026-08-06 sweep of section 2.

[^20]: Fresh-process runs alternating the binary of commit `1bd6b3e` and the
    binary of commit `1c99256`, 3k prompt, 64 slots, 512 tokens, greedy,
    2026-09-04 14:35: old 19.00 s and 20.263 tokens per second, new 14.70 s
    and 19.989, old 18.40 s and 18.457, new 15.03 s and 19.978. Load
    averages during the runs were 4.9 to 9.4.

[^21]: oMLX commit messages for Lightning MTP and the fused gate and up
    projection on Qwen3.6-35B-A3B, greedy, single stream, M3 Ultra: 85.2 to
    140.4 tokens per second with the multi-token-prediction head, and 104.4
    to 115.6 with the fused projection; both are the authors' own
    measurements. Its Qwen prefill floor stays at 2048 tokens on M5, its
    group-128 native quantized matmul is disabled there, and it carries a
    workaround for an M5 gather kernel, per its source at `origin/main` on
    2026-09-03.
