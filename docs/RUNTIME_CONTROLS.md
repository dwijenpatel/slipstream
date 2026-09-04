# Runtime controls

The CLI, the server, and the Mac app share one set of runtime defaults,
written down once in `RuntimeDefaults.swift` and asserted by a drift test.
Generation settings apply to the next request. Load-time settings require a
reload. FP16 is the fixed KV format.

## Generation controls

| Control | CLI flag | CLI default | Server default | Effect |
| --- | --- | --- | --- | --- |
| Maximum response | `--max-new` | 1,024 tokens | 4,096 tokens, or the request's `max_tokens` if lower | Caps the number of generated tokens. |
| Maximum context | `--max-context` | 4K | 16K | Prompt plus response capacity: 4K, 8K, 16K, 32K, or 64K. |
| Temperature | `--temperature` | 0.2 | 0 | `0` is greedy and uses the fused output head; positive values sample. |
| Top-K | `--top-k` | 64 | request | Keeps at most K candidates. CLI `0` turns it off. |
| Top-P | `--top-p` | 0.95 | request | Nucleus truncation before Top-K; effective only while Top-K is enabled. |

The server answers coding agents, which send temperature 0, so its default is
greedy. The Mac app is a chat surface and keeps 0.2. With positive
temperature, a CLI Top-P below `1` requires Top-K between `1` and `256`. To
disable both truncation controls, pass `--top-k 0 --top-p 1`.

## Runtime settings

| Control | Values | CLI flag | Default | Effect |
| --- | --- | --- | --- | --- |
| Expert-cache slots | 8, 16, 24, 32, 48, 64, 96, 128, 192, 256 | `--expert-cache-slots` | 64 | Routed experts retained per layer. Memory cost is slots times layers times the expert stride, about 71 MiB per slot step on Qwen3.6 and 101 MiB on Gemma 4. On the 24 GB M5 host, 64 measured 27.8 tokens per second against 25.1 at 16, and 192 was the slowest arm, because past about 64 the cache evicts the file pages that were absorbing its own misses. Re-measure on other hosts. |
| Prefill chunk tokens | 32 to 4096, or `auto` | `--prefill-chunk` | CLI `auto`; server 4096; app 128 | Tokens processed per prefill chunk. Every chunk re-reads most of the expert pool, so a chunk that covers the prompt reads it once: 247 GB to 38 GB on a 2,940-token prompt. Larger chunks use more prefill scratch. |
| KV snapshot | a file path | `--kv-snapshot` | off | After a fresh prefill, writes the whole cache, including the linear-attention state, to the file. A later run with the identical prompt restores it and skips prefill. CLI only. |
| Prompt cache mode | `off`, `single-prefix` | server `--prompt-cache-mode` | `single-prefix` | The server keeps one conversation's verified KV prefix in memory and reuses it when the next request extends that conversation exactly. |
| Prompt prefill | on, off | none | on | Off disables the chunked prefill path. Diagnostic only. |
| RDADVISE | off, default, bounded, adaptive | `--rdadvise` | off | Read-ahead advice on expert files. Measured neutral or negative on this host; kept for experiments. |

The CLI applies these settings when it loads the model. Changing context
length, expert-cache slots, RDADVISE, or the prefill chunk requires a reload.
Greedy and sampled generation use different output-head paths, and the server
selects the head per request.

## Diagnostics

Environment variables read by the CLI:

| Variable | Effect |
| --- | --- |
| `TURBO_FIELDFARE_PHASES=1` | Prints the decode phase split after the timing footer: command-buffer encode and commit, expert I/O await, GPU waits, and the expert-cache hit and miss counters. |
| `TURBO_FIELDFARE_PREFETCH=1` | Prefetches experts on predicted routing. Measured net negative twice; off by default. |
| `TURBO_FIELDFARE_PRED_ROUTE=1` | Records predicted-routing recall without prefetching. |
| `TURBO_FIELDFARE_NO_CB_MERGE=1` | Disables the merged command-buffer decode path. The merge measured under 2 percent either way. |
| `TURBO_FIELDFARE_SPEC=1` | Enables the speculative-decoding scaffold. Slower than plain decode until its kernels are built; see `docs/SPEC_DECODE.md`. |

## Run an experiment

1. Start from the defaults: 64 slots, prefill on, RDADVISE off, the CLI's
   `auto` prefill chunk.
2. Keep the prompt and generation controls fixed, and use the prompt files in
   `playbook/prompts/`.
3. Run through `playbook/fill_table.sh`, which levels the page cache and runs
   a drift control. A single run in a warm session is not a measurement.
4. Change one runtime control per arm.
5. Compare time to first token, decode rate, peak footprint, and the phase
   split, and report the drift control beside the result.
6. Restore the defaults when the experiment ends.

## Read the results

- **Decode rate** is generated tokens per second after prompt prefill.
- **Time to first token** is prompt prefill plus the wait for the first
  generated token. The server logs it per request as `ttft`.
- **Peak memory** is physical footprint, which counts GPU allocations that
  resident set size misses.
- **Expert I/O await** is time spent waiting for expert reads that were not
  already overlapped with GPU work.
