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

Same machine, same model, same prompt file. Column headings are prompt
sizes in tokens. Blank cells are unmeasured, not unmeasurable.

| option | RAM | <1k prompt | ~3k prompt | ~12k prompt | ~24k prompt |
| --- | --- | --- | --- | --- | --- |
| mlx-lm, resident | 21.6 GB | | 41.2 tok/s | | |
| llama.cpp, default | ~17.4 GB | 0.7 s, 32.6 tok/s | | | |
| llama.cpp, `--n-cpu-moe 32 --no-mmap` | ~5.7 GB | 2.6 s, 19.1 tok/s | | | |
| Ollama | | | | | |
| LM Studio, MLX engine | | | | | |
| SwiftLM | | | | | |
| TurboFieldfare, its defaults | 1.13 GB RSS | | 63.3 s, ~18 tok/s | | |
| slipstream, out of the box | 1.90 GB | | 18.5 s, 21.4 tok/s | | |
| slipstream, 128 slots | up to 9.1 GB | 27.7 tok/s | 17.5 s, ~25 tok/s | 20.2 tok/s | 19.5 tok/s |
| slipstream, KV resume | 1.90 GB | | 0.03 s | | |

Cells give time to first token and sustained decode where both are
comparable, decode alone where prefill is not. Three caveats. The
llama.cpp rows and the under-1k column used prompts of a few dozen tokens,
so their prefill figures mean little. The ~12k and ~24k slipstream cells
generated 96 tokens rather than 512, which understates sustained rate;
read them as a floor. mlx-lm's prefill is absent because its first run also
materializes 19 GB of lazily mapped weights.

## Thesis

Three projects each hold one corner of the problem:

- Resident-weight runtimes (mlx-lm, oMLX) are fast until the model does not
  fit, then stop working entirely. This one fits on this machine with
  0.3 GB to spare, as the table below shows, and would not fit at all on a
  16 GB Mac.
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

Qwen3.6-35B-A3B, community long-synthesis case, warm, 2,940-token prompt,
every row on the same machine and the same prompt file. TurboFieldfare is
the project slipstream forked from, run at its own defaults:

| configuration | TTFT | decode | peak memory |
| --- | --- | --- | --- |
| mlx-lm, all weights resident | see below | 41.2 tok/s | 21.6 GB |
| TurboFieldfare, its defaults | 63.3 s | ~18 tok/s | 1.13 GB RSS |
| slipstream, out of the box | 18.5 s | 21.4 tok/s | 1.90 GB |
| slipstream, expert cache raised to 128 slots | 17.5 s | ~25 tok/s | up to 9.1 GB |
| slipstream, resuming a saved KV cache | 0.03 s | ramps from cold | plus a 119 MB file |

The resident row is the tradeoff stated plainly: keeping all 19 GB of
weights in memory buys roughly twice the decode speed, and costs eleven
times the memory. It also barely fits. Peak footprint came to 21.6 GB on a
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
used, so those are ceilings rather than reservations. Eight times the
memory buys under 10 percent more decode speed here, which is the honest
shape of the dial: on this model it decides whether you can run at all far
more than it decides how fast.

That flatness is not a quirk of this implementation. The MoE-Infinity
paper measures expert locality directly and finds that for models with
around 100 experts, fewer than 5 percent are repeatedly activated while
decoding a single request, with the reuse skew concentrated inside one
sequence and absent across requests.[^locality] The default 16 slots are
6.25 percent of this model's 256 experts, which lands right on that
threshold. Their measurement and this curve agree from opposite
directions: a small cache already holds the experts a request keeps
coming back to, so the rest of the memory buys very little.

[^locality]: [MoE-Infinity, arXiv:2401.14361](https://arxiv.org/abs/2401.14361).
    Measured on NVIDIA hardware, not Apple Silicon, so the mechanism
    transfers but the numbers are not this machine's.

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
