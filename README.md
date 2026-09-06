# slipstream

Mixture-of-experts inference on Apple Silicon with the weights left on the
SSD. The memory a model uses is a setting, and the setting is measured.

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) has 18 GB of
expert weights.[^1] On a base M5 MacBook Pro with 24 GB of memory, slipstream
runs it in 5.6 GB of peak footprint at 31.1 tokens per second against a
2,940-token prompt, or in 2.5 GB at 27.1 tokens per second.[^2] The weights
stay on the SSD. Each token routes to eight of the 256 experts in each of the
model's 40 layers, and only those eight are read. Everything else on the
machine keeps its memory.

A prompt is paid for once. After a fresh prefill the whole cache, including
the recurrent state of the model's linear-attention layers, can be written to
disk. The same 2,940-token prompt that took 14.3 seconds to read comes back in
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
more than 5 percent. The slipstream rows are the overnight sweep of
2026-09-06, whose drift control read 0.8 percent; the TurboFieldfare and
llama.cpp rows are the sweep of 2026-08-06, same harness, same prompt files,
same machine.[^2] The four prompts hold 889, 2,940, 11,738, and 23,827
tokens, and every cell generates 512 tokens.

The expert cache is counted in slots. One slot holds one expert's weights for
one layer, 1.77 MB on this model, so 64 slots across 40 layers cap the cache at
4.5 GB. Slots fill only as experts are used, so the cap is a ceiling rather
than a reservation. Sixty-four slots is the default.

Time to first token, in seconds, by prompt length:

| runtime and setting                 | peak memory | 1k   | 3k   | 12k   | 24k   |
| ----------------------------------- | ----------- | ---- | ---- | ----- | ----- |
| TurboFieldfare, default 16 slots    | 2.0 GB      | 14.3 | 42.5 | 185.1 | 664.8 |
| slipstream, 16 slots                | 2.5 GB      | 8.8  | 14.1 | 53.4  | 111.6 |
| TurboFieldfare, 32 slots            | 3.0 GB      | 15.5 | 70.5 | 265.5 | 720.5 |
| slipstream, 32 slots                | 3.5 GB      | 8.9  | 14.3 | 53.6  | 112.0 |
| slipstream, 64 slots (default)      | 5.6 GB      | 8.9  | 14.3 | 53.7  | 112.0 |
| slipstream, 96 slots                | 7.8 GB      | 9.0  | 14.4 |       |       |
| slipstream, 128 slots               | 9.9 GB      | 9.1  | 14.4 |       |       |
| slipstream, 192 slots               | 14.2 GB     | 9.3  | 14.7 |       |       |
| llama.cpp, default                  | 16.2 GB RSS | 1.2  | 3.8  | 18.5  | 44.9  |
| llama.cpp, `--n-cpu-moe 32`         | 16.8 GB RSS | 2.5  | 7.8  | 35.0  | 78.2  |
| llama.cpp, that plus `--mmap 0`     | 17.2 GB     | 2.8  | 11.0 | 51.4  | 141.8 |
| slipstream, resuming a saved cache  | 2.5 GB      |      | 0.03 |       |       |

Sustained decode, in tokens per second, by prompt length:

| runtime and setting                 | peak memory | 1k   | 3k   | 12k  | 24k  |
| ----------------------------------- | ----------- | ---- | ---- | ---- | ---- |
| TurboFieldfare, default 16 slots    | 2.0 GB      | 26.1 | 23.8 | 17.2 | 13.0 |
| slipstream, 16 slots                | 2.5 GB      | 28.3 | 27.1 | 26.2 | 24.4 |
| TurboFieldfare, 32 slots            | 3.0 GB      | 28.1 | 25.3 | 17.5 | 13.1 |
| slipstream, 32 slots                | 3.5 GB      | 30.6 | 29.1 | 28.1 | 26.1 |
| slipstream, 64 slots (default)      | 5.6 GB      | 33.1 | 31.1 | 29.9 | 27.2 |
| slipstream, 96 slots                | 7.8 GB      | 34.0 | 31.7 |      |      |
| slipstream, 128 slots               | 9.9 GB      | 34.7 | 31.9 |       |       |
| slipstream, 192 slots               | 14.2 GB     | 31.3 | 25.2 |       |       |
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

**The drift control on the slipstream sweep read 0.8 percent apart**, so
differences of a few percent within those rows are real. The slot count's
effect at 3k: 27.1 tokens per second at 16 slots, 31.1 at 64, 31.9 at 128, and
25.2 at 192. More memory stops helping around 128 slots and hurts at 192,
because the expert cache and the operating system's file cache compete for
the same RAM: past that point the cache evicts the file pages that were
absorbing its own misses, and the arm with the highest hit rate is among the
slowest. An August sweep with a 2.2 percent control had put the knee at 64
slots with the older cache policy.[^6] The optimum belongs to the host, not
the model, and will move on a machine with a different amount of memory.

The resume row has no decode figure because the number would mislead. Prefill
also fills the expert cache, so decode after a resume starts against an empty
one: 25.3 tokens per second before the snapshot against 13.5 over the first
256 tokens after it, climbing as the slots refill.[^3] The honest version is a
curve, and it has not been measured.

The TurboFieldfare comparison depends on prompt length. Comparing the 16-slot
rows, the closest memory match to the upstream default: at 1k slipstream
decodes 8 percent faster, at 3k 14 percent, at 12k 52 percent, and at 24k 88
percent, because the upstream rate falls from 26.1 to 13.0 tokens per second
across that range while slipstream holds between 28.3 and 24.4. Two things
did that: the decode attention kernel of section 3, which cut the part of
each token that grows with context, and the cache policy and clock hold
below, which lifted every cell. Time to first token moved from a larger
prefill chunk and the prefill attention kernel of section 5: 42.5 to 14.1
seconds at 3k and 664.8 to 111.6 at 24k.

### What moved the numbers since August

Three changes account for the difference between this sweep and the one of
2026-08-06, and each carries its own measurement.

**The prefill attention kernel** of section 5, ported on 2026-09-04, took
the 12k time to first token from 116.9 seconds to 53.4 and the 24k one from
381.7 to 111.6. Before it, the model's ten full-attention layers ran on a
scalar fallback at about one percent of the tensor unit's ceiling.

**LFU with aging** replaced the inherited least-frequently-used cache policy
on 2026-09-05. The runtime now records which experts the router chose per
token and layer, and a policy's behavior depends only on that sequence, so
one trace replays any policy at any slot count in seconds, and the replay
reproduces the runtime's own miss counters. On a recorded ten-turn coding
session the old policy kept early favorites resident after the work moved on:
98.8 misses per token at 64 slots, against 71.8 with every expert's use count
halved every 32 tokens. Replayed against LRU, ARC, LRFU, S3-FIFO, and SIEVE,
none did better, and the offline optimum sat at 39. With every expert read
from the SSD, the session ran 15 percent faster under the new policy, and on
a warm machine 7 percent faster.[^7]

**A GPU clock hold**, landed 2026-09-05, keeps one 32-thread threadgroup
looping on a second command queue while decode runs. At 16 slots with every
expert read from the SSD, the same decode kernels had taken 28 ms per token
instead of the 13 ms they take on a warm machine, because the chip lowered
the GPU clock during each layer's read gap; holding the clock took decode
from 9.5 to 12.6 tokens per second with byte-identical output. On a warm
machine the gaps are short and the hold changes nothing measurable.[^8] It
is on by default and off under macOS Low Power Mode.

Five more ideas were built or replayed on the same days, measured, and left
off: prefetch on predicted routing where the GPU stalls on the SSD, a decode
loop that waits on shared events instead of command-buffer round trips, a
zero-copy read of cached expert pages, per-layer slot allocation at a fixed
total, and a layer-major prefill schedule. Sections 4 and 5 have the
numbers.[^9]

## 3. What is new here, and what is not

Most of slipstream is inherited. The bounded-memory streaming design, the
per-layer least-frequently-used slot cache, the int4 kernels, the repacker,
the Mac app, and the server are TurboFieldfare's. The Qwen3.6 port began from
an upstream pull request.[^10] What this project adds is smaller, and most of
it is evidence rather than mechanism.

**A decode attention kernel that reads each key and value row once per
key-value head.** The model gives 16 query heads only 2 key-value heads.
TurboFieldfare's kernel re-read the same rows once per query head. The
replacement gives one threadgroup to one key-value head and one chunk of the
sequence, holds each query head's vector in one simdgroup's registers, and
shares every row it loads across all eight heads. The full-attention branch of
a decode run fell from 3,106 ms to 1,249 ms, 2.49 times faster, and the
key-value scan runs at about 90 percent of this machine's measured 120.4 GB/s
memory bandwidth.[^11] MLX has a read-once kernel of the same design, but not
for this head dimension: on this head shape MLX's stock kernel measured 57 to
62 percent of bandwidth in August, and a read-once version written in the
companion repository, not yet upstream, measured 94 to 98.[^12] Grouped-query
sharing is not a new idea. Having it in a Swift and Metal streaming runtime,
gated by reference tests at both production shapes, is what is new.

**Whole-prompt prefill chunks, and the measurement that justified them.** The
runtime prefills a prompt in chunks and streams the experts each chunk routes
to. On a streaming runtime that means every chunk re-reads most of the expert
pool. Raising the chunk from 128 tokens to the whole prompt cut the bytes read
during a 2,940-token prefill from 247 GB to 38 GB and the prefill from 63.3 to
18.5 seconds, with byte-identical output.[^13] The rule that fell out is that
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

**The GPU clock as a term in the streaming budget.** Section 2 has the
measurement: on a small cache with cold reads, half of what looked like
kernel time was the chip running its kernels at a lowered clock between
reads. No profile of a streaming runtime this project has found reports that
term, and the fix, one idle threadgroup, is not in any of the runtimes
compared here.

**A cache policy chosen by replaying routing traces.** The trace tooling and
the aging policy of section 2. The method is what is new: the runtime's own
counters calibrate the replay exactly, so a policy question is answered in
seconds instead of a night, and the replay showed which policies not to
build.

**Eight priced negative results.** Each was built or replayed, measured, and
left off. They are in sections 2, 4, and 5.

**A speculative-decoding scaffold whose measurement located the block.** It
drafts, verifies, and repairs the recurrent state correctly, and it is slower
than plain decode. The measurement says why, and the reason was not the one
the design assumed. Section 4 has the numbers.

None of these is a new algorithm. Grouped-query sharing, whole-prompt prefill,
state snapshots, and speculative decoding all exist elsewhere. The claim this
project makes is narrower: on this class of machine, for this class of model,
these are the measured effects, including the ones that went the wrong way.

## 4. Where a token's time goes

Measured at 128 slots on a 3k prompt, one token costs 40.7 ms.[^14] Of that,
9.0 ms is spent waiting for the GPU between layers, 7.8 ms awaiting expert
reads, 6.5 ms in the routed expert feed-forward, 6.1 ms in the thirty
linear-attention layers, 2.4 ms in the ten full-attention layers, 2.4 ms in
the output head, and 1.3 ms in norms and the router. Those sum to 35.5 ms; the
remaining 5.2 ms is command encoding and time the counters do not attribute.

**The wait between layers is a floor.** Every layer sends its routing choice
back to the CPU, which then fetches the chosen experts, so every layer ends in
a command-buffer round trip. That round trip costs about 207 microseconds on
this machine regardless of the work inside it, and there are 40 of them per
token, about 8.3 ms.[^15] Replacing the wait with a GPU fence and a CPU spin
produced bit-identical output and ran 15 percent slower, because the GPU's
writes become visible to the CPU at about the same boundary anyway.[^16] A
later probe inverted the dependency with shared events, one command buffer
per token with the GPU waiting for the CPU's signal at each layer: a wait the
CPU has already satisfied costs the GPU nothing, but a parked GPU restarts in
100 to 190 microseconds, as much as the commit it would replace, so only
layers with no expert misses could gain.[^9] MLX avoids the cost by keeping
routing indices on the GPU, which a runtime that fetches from disk on the CPU
cannot do.

**Prefetching on predicted routing does not pay, and the obvious objection was
tested.** Running the next layer's router against the current layer's state
picks about 82 percent of the experts that layer will want. Prefetching on
that prediction cut the measured disk wait from 15.95 to 7.01 ms per token at
16 slots and made decode 11.2 percent slower.[^17] The runtime already commits
GPU work before it issues the fetch, so the counter measures an overlapped
wait rather than a stall, and moving the same bytes earlier only crowds a
saturated bus. That left one untested case, the GPU stalling on the SSD for
real, and it was tested on 2026-09-06 at 64 slots with every expert read
from disk: one layer of lead tied, 22.6 against 22.7 tokens per second, and
two layers of lead lost 10 percent, because the wrong guesses cost 37 percent
more bytes on a device with no room for them.[^18] On the recorded coding
session the same prefetch ran 6 percent slower than none. It ships off by
default, and it is closed in this design.

**Merging command buffers is worth less than the noise.** Removing 41 of the
per-token synchronization boundaries was bit-identical and under 2 percent.[^19]

**The output head and the attention scan are near their bandwidth ceilings.**
The head reads about 270 MB of weights per token at about 93 percent of the
measured bandwidth, and the attention scan at about 90 percent, so there is no
kernel prize left in either.[^11] The routed expert feed-forward reads about
566 MB per token, which at the measured bandwidth is a 4.7 ms floor against
6.5 ms measured, about 72 percent by arithmetic; the remaining 1.8 ms per
token is the largest kernel-level gap in decode.

**Speculative decoding is the only large lever left, and it is not yet a
win.** The scaffold drafts by prompt lookup, verifies by running the draft
through a batched forward, and repairs the recurrent state on rejection. On
code it accepts 33.9 percent of drafted tokens for 3.57 emitted tokens per
round, and it still decodes slower than the sequential path, 16.9 against
27.7 tokens per second.[^20] The cost is not acceptance. A verify round costs
178 ms against a 47 ms target, because the batched matrix kernels available
here read the weights once per row below 32 rows, so verifying nine tokens
costs about nine GEMVs. Expert traffic also scales with verified tokens, not
emitted ones, although the eight experts each drafted token routes to overlap
55 percent within a round. A multi-row int4 kernel that reads weights once
now serves the projections and the head; on a code continuation at 128 slots
it took speculative decode from 16.9 to 22.3 tokens per second against 28.6
sequential, with byte-identical output.[^21] The routed experts and the
linear-attention projection still run per token, and they are the rest of
the round.

## 5. Where prefill time goes

Until 2026-09-04 the time to first token grew faster than the prompt. Timing
each 4096-token chunk of the 11,738-token prompt gave 26.3, 42.8, and 53.3
seconds for chunks that see 4,096, 8,192, and 11,736 keys.[^22] A fit put the
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
matter, in the same round-robin protocol as section 2:[^23]

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
the change; those decode figures are not quoted.[^24] The pre-port binary also
prefilled the 3k prompt in 18.4 and 19.0 seconds that day against 17.4 in
August, so the "after" times carry a few percent of the same load and are
pessimistic. The overnight sweep of 2026-09-06 replaced those tables; section
2 carries it, and its 3k time at 64 slots is 14.3 seconds.

What remains is the linear term, about 4.5 ms per token at every length. The
expert stream is not it. Above 4,096 tokens prefill re-reads the expert pool
once per chunk, about 18 GB each, and reading that from the SSD instead of
from memory changed prefill by under one percent: 54.3 seconds at 12k and
112.6 at 24k with every expert read cold, against 53.9 and 112.4 warm.[^18]
The stream is hidden behind compute, and a layer-major schedule that read the
pool once would save nothing. The term is GPU work and the synchronization
around it: 1,280 expert tiles per 4,096-token chunk, each a command-buffer
round trip, the linear-attention recurrence over the chunk's rows, and the
projections. Their shares are not yet attributed, and timing each dispatch is
the next step.

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
its Qwen fast paths are disabled there.[^25] Take llama.cpp, or Ollama over
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
  code. The footprint guard is built here; the block-aligned cache is not.
- Kernels and measurement method come from
  [gpu-kernel](https://github.com/dwijenpatel/gpu-kernel), a companion
  research repository. Its methodology document records how each measurement
  technique was found wrong and what the error cost, which is why the numbers
  above carry the caveats they do.

This page describes commit `5b5a6d7` and measurements taken between
2026-08-06 and 2026-09-06.

[^1]: Architecture facts from the
    [model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B): 40 layers, of
    which 30 are gated-delta-net linear attention and 10 are full attention
    with 16 query heads, 2 key-value heads, and head dimension 256; 256 routed
    experts per layer with 8 active per token. The 18 GB figure is the packed
    expert files of the 4-bit checkpoint as installed here, 1,769,472 bytes
    per expert per layer.

[^2]: slipstream rows: `playbook/fill_table.sh --only slipstream` at commit
    `03d5a06`, 2026-09-06 04:00 to 04:37, results in
    `bench-results/table-20260906-035959/results.csv`; drift control
    `DRIFT-slipstream-slots16`, 28.487 against 28.259 tokens per second at
    1k, 0.8 percent. The 12k and 24k cells at 96, 128, and 192 slots are
    blank: the overnight sweep refused those prompts through a defect in the
    prefill memory guard, which extrapolated the first chunk's one-time
    growth, the expert slots becoming resident, as a per-chunk rate (fixed in
    commit `5b5a6d7`), and an afternoon rerun on a machine in use,
    `bench-results/table-20260906-140627`, failed its own drift control by
    19.5 percent and is not reported. They wait for the next idle window.
    TurboFieldfare rows:
    `bench-results/table-20260806-044148`; llama.cpp rows:
    `bench-results/table-20260806-055113`; both 2026-08-06, same harness,
    same prompt files, with TurboFieldfare built from its own tree at its
    defaults. The August slipstream sweep those rows were first compared
    against is `bench-results/table-20260806-022221`.

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

[^7]: Traces recorded with `TURBO_FIELDFARE_ROUTE_TRACE` and replayed with
    `playbook/route_replay.py`; the traces, the recorded session, and its
    server logs are in `bench-results/route-replay-20260905`, and the policy
    tables in `docs/REVIEW-2026-09-04.md`. The replay reproduces the
    runtime's own miss counter exactly on a single run, 31,903 at 64 slots
    on the 3k prompt, and within 0.2 percent over the ten-request session.
    The session times are `bench-results/overnight-20260906-022111/SUMMARY.md`:
    with every expert read from the SSD, LFU 1,157.6 s, LRU 1,020.8 s, aging
    984.6 s; warm, aging 788.4 s against LFU's 849.5 s and LRU's 929.7 s on
    earlier nights. All ten replies were byte-identical across policies.

[^8]: `bench-results/gpu-heater-20260905`: five alternated runs at 16 slots
    with the file cache bypassed and physical reads equal to logical, 3k
    prompt, 512 tokens; control 9.71 and 9.47 tokens per second, heater
    12.61, 12.59, and 12.65, kernel time 27.6 to 29.0 ms per token against
    12.7 to 12.8. The built-in hold in the same setup: off 9.59 and 9.32,
    auto 12.72 and 12.82. Warm at 64 slots, 2026-09-06: off 25.9 and 26.2,
    auto 27.5 and one run at 19.9 whose expert-read bucket alone doubled
    (`overnight-20260906-022111`, rows p6).

[^9]: `docs/REVIEW-2026-09-04.md`, addenda of 2026-09-05 and 2026-09-06:
    the shared-event probe (`bench-results/probes-20260905/evprobe*.swift`,
    medians over 300 trials: commit-and-wait 197 µs, a satisfied GPU wait
    1 µs, a parked GPU restarting in 100 to 190 µs), the zero-copy analysis
    (`misscost*.c` and `wrapcost.swift` in the same directory), and the
    per-layer allocation replay, which changed misses by under 0.3 percent
    in both holdout directions.

[^10]: [TurboFieldfare pull request 29](https://github.com/drumih/turbo-fieldfare/pull/29),
    open as of 2026-09-04. This project's own upstream contribution, the
    configurable prefill chunk, is
    [pull request 53](https://github.com/drumih/turbo-fieldfare/pull/53), also
    open.

[^11]: Commit `a638cf8`, 2026-08-01, with long-sequence reference tests at
    both production shapes. Timings and bandwidth ratios are from the
    runtime's per-phase GPU counters and the companion repository's
    measurement log; the 120.4 GB/s ceiling is that repository's measured
    sustained read bandwidth on this machine, against a 153 GB/s
    specification. The context length at which the 90 and 93 percent figures
    were taken is not recorded, and figures measured on a working set that
    fits the system cache read high; treat both as approximate.

[^12]: gpu-kernel, `mlx-kernel-a-evidence.md` and `telemetry/kernel_a_ab.csv`,
    2026-08-27: MLX's stock decode kernel at head dimension 256 and a
    query-to-key-value ratio of 8 measured 57 to 62 percent of the 120.4 GB/s
    ceiling at a 32k key-value length; the read-once kernel written there
    measured 94 to 98 percent, 1.71 times faster, in three alternating pairs
    on an idle machine.

[^13]: Commit `c0d3f28`, 2026-08-01, measured with `iostat` and `time -l` on
    the community long-synthesis prompt: bytes read 247 GB to 38 GB, prefill
    63.3 to 18.5 seconds, wall 81.1 to 37.6 seconds, token-identical output.

[^14]: `profiles/qwen36/README.md`, from the runtime's phase counters under
    `TURBO_FIELDFARE_PHASES=1`, 2026-08-01, 128 slots, warm, 3k context. The
    full-attention figure is the post-rewrite one; the pre-rewrite figure was
    6.1 ms.

[^15]: gpu-kernel `METHODOLOGY.md`: a Metal commit-and-wait round trip costs
    about 207 microseconds on this machine regardless of kernel size, measured
    with a standalone 40-line binary; about 15 microseconds when empty and
    about 38 when eight are pipelined.

[^16]: Commit `0c3358f`, 2026-08-01: 26.8 to 22.8 tokens per second with the
    fence-and-spin wait, bit-identical output.

[^17]: Commit `0358b9f`, 2026-08-06, at 16 slots: routing recall 82.5
    percent, disk wait 15.95 to 7.01 ms per token, throughput 40.4 to 45.5 ms
    per token. The first test, at 128 slots, is commit `03e83fb`.

[^18]: `bench-results/overnight-20260906-022111/SUMMARY.md`, one purge, then
    every arm with the file cache bypassed and 0.1 percent of the expert
    pool resident before and after. Rows p1: prefetch off 22.67 and 22.62
    tokens per second, one layer of lead 22.55 and 21.95, two layers 21.66
    and 19.06; bytes read per token 110, 151, and 162 MB. Row p3: the
    session under prefetch 1,046.3 s against 984.6 s without. Rows p4 and
    p7: prefill only, 12k 54.28 and 54.36 s cold against 53.91 warm, 24k
    112.68 and 112.56 against 112.42.

[^19]: Commit `4249aa3`, 2026-08-04.

[^20]: `docs/SPEC_DECODE.md`, 2026-08-05, code-domain probe at 128 slots:
    111 rounds, 33.9 percent acceptance, 3.57 emitted tokens per round,
    verify 178 ms per round. The kernel pricing that follows is recorded in
    the same document and in the companion repository's log of 2026-08-06.

[^21]: `docs/SPEC_DECODE.md`, section "M2' v2, first kernel", 2026-09-04:
    raw code continuation, 512 tokens, greedy, page cache leveled, 128
    slots; acceptance 41.6 percent, 3.91 emitted per round, verify 137 ms
    per round, down from 178. At 64 slots the round's union of experts
    thrashes the cache and the speculative rate is 17.0.

[^22]: Measured 2026-09-04 with a build of commit `01f7d5e` that prints a
    timestamp at each prefill chunk boundary, on the same 11,738-token prompt
    file the tables use, in a fresh process with no other model process
    running. The fit predicts the third chunk at 50.4 seconds against 53.3
    measured. Attention FLOPs at 12k are 11.3 TFLOP, which over the fitted
    67.8 seconds is 0.17 TFLOPS against the tensor unit's measured 15.4.

[^23]: `playbook/fill_table.sh --only 'slipstream, (16|64) of' --contexts
    3k,12k,24k` at commit `1c99256`, 2026-09-04 14:07 to 14:26, results in
    `bench-results/table-20260904-140715/results.csv`. Two round-robin
    passes, the second recorded; drift control 18.508 against 18.152
    tokens per second at 3k, 2.0 percent. The "before" column is the
    2026-08-06 sweep of section 2.

[^24]: Fresh-process runs alternating the binary of commit `1bd6b3e` and the
    binary of commit `1c99256`, 3k prompt, 64 slots, 512 tokens, greedy,
    2026-09-04 14:35: old 19.00 s and 20.263 tokens per second, new 14.70 s
    and 19.989, old 18.40 s and 18.457, new 15.03 s and 19.978. Load
    averages during the runs were 4.9 to 9.4.

[^25]: oMLX commit messages for Lightning MTP and the fused gate and up
    projection on Qwen3.6-35B-A3B, greedy, single stream, M3 Ultra: 85.2 to
    140.4 tokens per second with the multi-token-prediction head, and 104.4
    to 115.6 with the fused projection; both are the authors' own
    measurements. Its Qwen prefill floor stays at 2048 tokens on M5, its
    group-128 native quantized matmul is disabled there, and it carries a
    workaround for an M5 gather kernel, per its source at `origin/main` on
    2026-09-03.
