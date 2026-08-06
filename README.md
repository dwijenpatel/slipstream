# slipstream

Mixture-of-Experts inference on Apple Silicon: experts streamed from SSD,
near-roofline Metal kernels, and a KV cache that survives the process.

Qwen3.6-35B-A3B generates at 26.5 tokens per second on a base M5 MacBook in
2.5 GB of memory, or 30.8 in 5.6 GB. Its weights are 18 GB.
slipstream leaves them on SSD and streams only the eight experts each
token actually routes to, staying inside a RAM budget you set, so memory is a dial rather than a number the
model dictates. Prompts do not have to be paid for twice either: the KV
cache persists to disk, so a 2,940-token prompt that costs 17.4 seconds to
read the first time comes back in 0.03 seconds, byte for byte, on every run
after. Doing this without giving up speed is the point of the kernels
underneath, which are hand-written Metal tuned against what this chip
delivers rather than what its spec sheet claims: decode attention and the
output head each run at better than 90 percent of its real memory
bandwidth.

## Running a good coding model on a Mac, like Qwen3.6

**The field is three engines and a long tail.** llama.cpp runs GGUF, and
Ollama, LM Studio, Jan, KoboldCpp, and GPT4All all wrap it. MLX is Apple's
framework: mlx-lm is the reference library, oMLX a server built on it, and
LM Studio ships an MLX engine of its own. Expert streaming keeps weights on
SSD and fetches only what each token routes to: TurboFieldfare, SwiftLM,
and slipstream (this project). The tail is MLC-LLM, vLLM, llamafile,
text-generation-webui, transformers on MPS, and the offload projects born
on CUDA: ktransformers, PowerInfer, Fiddler, MoE-Infinity, AirLLM.

**The tail rules itself out.** MLC-LLM compiles ahead of time and shows no
Qwen3.6 support. vLLM's Metal plugin is young and unverified on this
architecture. transformers on MPS demands every weight resident.
llamafile is CPU-first and stagnant. text-generation-webui closed its MLX
pull request and stays GGUF-only. Of the offload projects, ktransformers
has no Apple Silicon build, PowerInfer runs CPU-only on Mac, PowerInfer-2
never shipped code, and Fiddler and MoE-Infinity dodge a PCIe bottleneck
that unified memory does not have. AirLLM does run on macOS and does cut
memory hard, at 50 to 200 times slower than an API call.

**Three contenders, one tradeoff each.** Take **mlx-lm**, or LM Studio's
MLX engine over it, for speed you pay for in memory. Take **llama.cpp**,
or Ollama over it, for the widest model and quantization choice, and note
it has two distinct configurations: resident by default, or expert tensors
paged from SSD once you pass both `--n-cpu-moe` and `--no-mmap`. Take
TurboFieldfare, SwiftLM, or **slipstream** when memory is what binds. That
last case is more common than it sounds: if you want to keep working on the
Mac while a good coding model runs, memory is the constraint you hit first,
not speed.

Same machine, same model, same prompt files, 512 generated tokens per cell.
Column headings are prompt sizes: 889, 2,940, 11,738 and 23,827 tokens.
Blank cells are unmeasured, not unmeasurable.

Time to first token, in seconds:

| option                          | memory  | 1k   | 3k   | 12k   | 24k   |
| ------------------------------- | ------- | ---- | ---- | ----- | ----- |
| TurboFieldfare, defaults        | 2.0 GB  | 14.3 | 42.5 | 185.1 | 664.8 |
| slipstream, 16 of 256 (default) | 2.5 GB  | 9.3  | 17.4 | 116.9 | 381.7 |
| TurboFieldfare, 32 of 256       | 3.0 GB  | 15.5 | 70.5 | 265.5 | 720.5 |
| slipstream, 32 of 256           | 3.5 GB  | 9.4  | 17.3 | 116.9 | 381.4 |
| slipstream, 64 of 256           | 5.6 GB  | 9.4  | 17.4 | 117.0 | 381.8 |
| slipstream, 96 of 256           | 7.8 GB  | 9.5  | 17.5 | 117.2 | 382.0 |
| slipstream, 128 of 256          | 9.9 GB  | 9.5  | 17.5 | 117.4 | 382.8 |
| slipstream, 192 of 256          | 14.1 GB | 9.5  | 17.4 | 145.5 | 475.4 |
| llama.cpp, default              | 16.2 GB | 1.2  | 3.8  | 18.5  | 44.9  |
| llama.cpp, `--n-cpu-moe 32`     | 16.8 GB | 2.5  | 7.8  | 35.0  | 78.2  |
| llama.cpp, that plus `--mmap 0` | 17.2 GB | 2.8  | 11.0 | 51.4  | 141.8 |
| mlx-lm, all weights resident    | 21.6 GB |      |      |       |       |
| Ollama                          |         |      |      |       |       |
| LM Studio, MLX engine           |         |      |      |       |       |
| SwiftLM                         |         |      |      |       |       |
| slipstream, KV resume           |         |      | 0.03 |       |       |

Sustained decode, in tokens per second:

| option                          | memory  | 1k   | 3k   | 12k  | 24k  |
| ------------------------------- | ------- | ---- | ---- | ---- | ---- |
| TurboFieldfare, defaults        | 2.0 GB  | 26.1 | 23.8 | 17.2 | 13.0 |
| slipstream, 16 of 256 (default) | 2.5 GB  | 24.8 | 26.5 | 23.7 | 22.0 |
| TurboFieldfare, 32 of 256       | 3.0 GB  | 28.1 | 25.3 | 17.5 | 13.1 |
| slipstream, 32 of 256           | 3.5 GB  | 26.9 | 28.9 | 24.9 | 22.6 |
| slipstream, 64 of 256           | 5.6 GB  | 27.1 | 30.8 | 23.9 | 22.2 |
| slipstream, 96 of 256           | 7.8 GB  | 25.6 | 28.7 | 22.6 | 21.3 |
| slipstream, 128 of 256          | 9.9 GB  | 26.1 | 29.1 | 22.7 | 22.0 |
| slipstream, 192 of 256          | 14.1 GB | 24.7 | 23.9 | 12.4 | 12.1 |
| llama.cpp, default              | 16.2 GB | 37.6 | 38.6 | 38.3 | 38.9 |
| llama.cpp, `--n-cpu-moe 32`     | 16.8 GB | 22.9 | 22.9 | 22.9 | 23.3 |
| llama.cpp, that plus `--mmap 0` | 17.2 GB | 22.3 | 20.8 | 22.0 | 20.0 |
| mlx-lm, all weights resident    | 21.6 GB |      | 41.2 |      |      |
| Ollama                          |         |      |      |      |      |
| LM Studio, MLX engine           |         |      |      |      |      |
| SwiftLM                         |         |      |      |      |      |
| slipstream, KV resume           |         |      |      |      |      |

The dial saturates early. Going from 16 cached experts to 64 is worth 16
percent of decode speed at a 3k prompt and costs 3 GB; past that it buys
nothing, and at 192 it goes sharply negative as the machine runs short of
memory. On long prompts even the early gain nearly vanishes.

The KV-resume row has no decode figure because it has not been measured.
Restoring a cache skips prefill, and prefill is also what fills the expert
cache, so the first tokens after a resume run against an empty one and the
rate climbs as it fills. A single sustained number would misrepresent
that, and the honest version needs a curve.

Two caveats. The llama.cpp rows come from llama-bench, which measures
prefill and generation directly rather than serving a chat, so its numbers
are a best case rather than a like-for-like request. And mlx-lm was
measured separately, at 3k only, because its first run also materializes
19 GB of lazily mapped weights and its peak sits above this machine's GPU
wired limit; its prefill is left out for the same reason.

Memory is peak physical footprint everywhere except the two llama.cpp rows
that leave mmap on, where that figure counts almost nothing because the
weights are file-backed; those two report resident set size instead.

## Thesis

Three projects each hold one corner of the problem:

- Resident-weight runtimes (mlx-lm, oMLX) are fast until the model does not
  fit, then stop working entirely. This one fits on this machine with
  0.3 GB to spare, as the table below shows, and would not fit at all on a
  16 GB Mac.
- TurboFieldfare streams experts from SSD in bounded memory, and leaves
  its largest measured gaps at length: a 2,940-token prompt takes 42.5
  seconds to prefill against 17.4 here, and its decode falls to 13.0
  tokens per second at 24k where this holds 22.0.
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

Qwen3.6-35B-A3B, community long-synthesis case, warm, 2,940-token prompt,
every row on the same machine and the same prompt file. TurboFieldfare is
the project slipstream forked from, at its default settings:

| configuration                          | TTFT      | decode     | peak memory        |
| -------------------------------------- | --------- | ---------- | ------------------ |
| TurboFieldfare, default settings       | 42.5 s    | 23.8 tok/s | 2.0 GB             |
| slipstream, 16 of 256 cached (default) | 17.4 s    | 26.5 tok/s | 2.5 GB             |
| slipstream, 64 of 256 cached           | 17.4 s    | 30.8 tok/s | 5.6 GB             |
| llama.cpp, default                     | 3.8 s     | 38.6 tok/s | 16.2 GB RSS        |
| mlx-lm, all weights resident           | see below | 41.2 tok/s | 21.6 GB            |
| slipstream, resuming a saved KV cache  | 0.03 s    |            | plus a 119 MB file |

The resident row is the tradeoff stated plainly: keeping all 19 GB of
weights in memory buys about 1.6x the decode speed of the default here,
and costs nine times the memory. It also barely fits. Peak footprint came to 21.6 GB on a
machine whose GPU wired limit is 21.3 GB, so there is no headroom left for
a longer context, a second model, or anything else the machine is doing.
That row is mlx-lm, which is also the inference substrate oMLX builds on,
so a single-stream oMLX number would land near it; oMLX's own additions,
continuous batching and tiered KV caching, pay off across concurrent
requests rather than one. Its TTFT is left out because that run also
materialized the lazily mapped weights on first touch, which the warm
slipstream rows never pay, and quoting it as prefill would flatter
slipstream by about eight times.

The expert cache is counted in slots, and one slot holds one expert's
weights for one layer. This model puts 256 experts in each of its 40
layers at 1.7 MB per expert, so the default 16 slots cap the cache at
1.1 GB and 128 slots cap it at 9.1 GB. Slots fill only as experts get
used, so those are ceilings rather than reservations.

The dial is worth less than it looks, and the measurement says so
plainly. Going from 16 slots to 64 buys 16 percent at a 3k prompt and 1
percent at 24k; going further buys nothing at any length; and 192 slots
costs 45 percent at long prompts, where 14 GB of cache leaves too little
room for the KV cache and the page cache behind an 18 GB model. The drift
control on that sweep came back 8.4 percent apart, so treat anything
smaller than that as noise, which most of this table is.

That shape matches a published result. The MoE-Infinity paper measures
expert locality directly and finds that for models with around 100
experts, fewer than 5 percent are repeatedly activated while decoding a
single request.[^locality] Sixteen slots is 6.25 percent of this model's
256 experts and 64 is 25 percent, which brackets where the curve flattens
here. A small cache already holds what a request keeps returning to, and
the long tail it does not hold has to come off the SSD however much room
you give it.

[^locality]: [MoE-Infinity, arXiv:2401.14361](https://arxiv.org/abs/2401.14361).
    Measured on NVIDIA hardware, not Apple Silicon, so the mechanism
    transfers but the numbers are not this machine's.

Every memory figure here is peak physical footprint, which counts GPU
allocations that resident set size misses. Resident set size is the wrong
number to quote for this project: it is blind to exactly the memory being
managed.

The TurboFieldfare comparison depends entirely on prompt length, and
quoting one number for it would be misleading. At a 1k prompt upstream is
5 percent faster. At 3k they are close. At 12k slipstream is 38 percent
ahead and at 24k it is 69 percent ahead, because upstream decays from 26.1
to 13.0 tokens per second across that range while slipstream holds 24.8 to
22.0. That is the decode-attention rewrite below doing what it was built
to do: it cut the part of decode that grows with context, so it buys
nothing on a short prompt and a great deal on a long one. Prefill moved
too, 42.5 seconds down to 17.4 at 3k and 664.8 down to 381.7 at 24k. Resuming a
cache restores the prompt exactly, byte for byte, but decode then starts
against a cold expert cache, because prefill is what normally warms it.
The section below returns to that tradeoff.

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
pay, and we tested the obvious objection.** Running the next layer's router
against the current layer's state picks about 82 percent of the experts
that layer will actually want. Prefetching on that prediction cuts
disk-wait time by more than half and makes decode slower anyway. The first
measurement ran with a large cache, where little disk wait exists to
reclaim, so it was retested with a small one where 15.95 ms per token looks
like idle GPU: prefetch cut that to 7.01 ms and cost 11.2 percent of
throughput, 40.4 to 45.5 ms per token. The wait was never idle. The runtime
already commits GPU work before issuing the fetch, so the counter measures
time spent in an overlapped wait rather than a stall worth reclaiming, and
moving the same bytes earlier only crowds a saturated bus. It ships off by
default. The untested case is a model far larger than RAM, where the GPU
would genuinely stall.

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
