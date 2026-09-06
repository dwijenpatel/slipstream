# Qwen3.6 expert I/O baseline, 2026-09-05

Status: uncached I/O is now verified. The initial cold sweep passed the
physical-read, residency, and byte-identity gates, but failed timing drift at
64 slots. The three-pass uncached-only repeat also failed timing drift; no second
purge was needed or performed. A quieter measurement window is required
before claiming a stable throughput point. The earlier mixed-cache comparison remains a rejected
performance comparison.

## Implementation and validation

Base commit: `0d88c37fea86d2ebb5765f6c6015beadf3692534`, plus the archived
uncommitted measurement patch. Runtime defaults and model math are unchanged.
`TURBO_FIELDFARE_EXPERT_NOCACHE=1` sets `F_NOCACHE` on both expert streaming
and expert SHA-verification reads, failing if the control is rejected. Full
SHA verification remains enabled. `TURBO_FIELDFARE_IO_BASELINE=1` measures
physical process disk reads, logical expert reads, elapsed time, I/O await,
and footprint in 128-token decode windows after the seed token.

Release build passed. `Scripts/test.sh` passed 641 tests in 126 suites,
including cached/uncached expert-byte and SHA-digest cases. The new Python
harness and shell entry point passed syntax checks; the read-only residency
sampler was checked against a synthetic resident page.

## Preliminary comparison: rejected as an uncached baseline

Machine: base Apple M5, 10 GPU cores, 24 GB, MacBook Pro Mac17,2, internal SSD;
macOS 26.5.2 (25F84), Swift 6.3.3. Prompt: 2,938 computed prefill tokens;
1,024 generated tokens, greedy, seed 20260723, maximum context 32,768.

Exact harness command:

```bash
playbook/fill_table.sh --io-baseline --contexts 3k --slots 16,64 --max-new 1024 --passes 2
```

The same binary alternated normal/cache-disabled reads at 16 and 64 slots,
with two passes and a final control for every arm. Each run was a fresh
process; only one model process ran at a time. No cache purge or explicit
model prewarming occurred. Full SHA verification was retained in both arms.

Final-pass observations, **not accepted performance comparisons**:

| Slots | Expert file I/O | Prefill s | Decode tok/s | Physical/logical decode reads | Decode drift | Prefill drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | Normal | 17.67 | 16.486 | 23.7% | +6.38% | -1.81% |
| 16 | F_NOCACHE | 18.34 | 12.194 | 61.2% | +8.78% | +11.56% |
| 64 | Normal | 17.47 | 17.859 | 39.0% | +34.09% | -16.37% |
| 64 | F_NOCACHE | 19.97 | 15.969 | 88.5% | +34.73% | -10.47% |

All 12 generated outputs are byte-identical (SHA-256
`aa636c2675563acda11ec4f3ff1810960254dfef893a37a97accfc8b95f3c8ac`).
Logical decode expert reads are 302,728,347,648 bytes at 16 slots and
117,296,529,408 bytes at 64 slots over 1,023 post-seed token intervals.
The substantial shortfall in physical reads demonstrates residual file-cache
benefit even with `F_NOCACHE`. A small aligned-read probe independently showed
that setting the flag did not bypass already cached expert pages.

Raw commands, binary digest, source snapshot, full timing footers, output,
per-window observations, VM/swap snapshots, and per-arm drift are in
[`bench-results/io-baseline-20260905-011246`](../bench-results/io-baseline-20260905-011246).
Build and serial test logs are saved there too.

## Strict baseline protocol

The user approved one exception to the repository's no-purge rule. No model
process was running when the cache-clear attempts were made. Unprivileged
`purge` was refused; noninteractive `sudo` requires authentication. The standard
macOS administrator dialog was requested. The first dialog was canceled; at
the user's request it was reopened, authentication succeeded, and the purge
completed before the cold sweep. Its zero-page residency observations confirm
that the expert files were cold.

The initial cold sweep used:

```bash
playbook/fill_table.sh --io-baseline --uncached-only --contexts 3k --slots 16,64 --max-new 1024 --passes 2
```

The driver never purges on its own. It samples all expert files with `mincore`
before and after each run without touching their data. All arms remain
cache-disabled so a normal-cache arm cannot rewarm the next one. Require zero
resident expert pages before and after every arm, physical/logical decode reads
of at least 99%, byte-identical output, and prefill/decode drift within 5%.
Physical reads are process-wide, so interpret the ratio together with residency;
it is not an exact per-expert cache-hit counter. Count resident model weights,
explicit expert slots, KV/recurrent state, and scratch in the memory budget.

The reported prefill time follows the existing CLI timing footer. It is not
process-launch-to-first-visible-content latency. This initial baseline covers
3k context; it does not establish long-context TTFT or all memory tiers.

## Initial verified cold sweep: timing drift failed

Raw campaign: [`io-baseline-20260905-013706`](../bench-results/io-baseline-20260905-013706).
All six runs had zero resident expert pages before and after execution. At
16 slots, physical decode reads exactly matched all 302,728,347,648 requested
expert bytes. At 64 slots, physical reads matched 117,296,529,408 expert bytes
plus only 12–32 KiB of other process reads. All outputs matched the original
SHA-256. System-wide swap-in and swap-out counters did not increase.

| Slots | Pass 1 tok/s | Pass 2 tok/s | Drift tok/s | Decode drift | Prefill drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 8.394 | 8.601 | 8.929 | +3.81% | +2.12% |
| 64 | 19.620 | 17.086 | 19.950 | +16.76% | -0.94% |

This proves the uncached data path, but the failed 64-slot drift disqualifies
the sweep as the final timing comparison. The repeat uses the same binary,
three alternating passes, the same prompts and generation settings, and no
further cache clear.

## Three-pass repeat: uncached verified, timing drift failed again

Raw campaign: [`io-baseline-20260905-155934`](../bench-results/io-baseline-20260905-155934).
Same binary and 1,024-token generation length; three alternating passes and
a final control per arm. All eight runs passed zero expert residency and
physical-read accounting, and generated the same output hash.

| Slots | Pass 1 tok/s | Pass 2 tok/s | Pass 3 tok/s | Control tok/s | Decode drift | Prefill drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 7.788 | 7.476 | 9.273 | 8.802 | -5.08% | +0.99% |
| 64 | 15.492 | 18.341 | 19.022 | 15.533 | -18.34% | +14.79% |

Do not round the 16-slot drift into a pass: its magnitude was 5.079%, above
the 5% gate. The 64-slot control also failed materially. Physical traffic
remained unchanged, so expert file-cache warming cannot explain these changes.
A between-run process check found substantial concurrent desktop and system
CPU activity. The machine was on AC power with low-power mode disabled; macOS
reported no recorded thermal/performance warning, which does not establish
constant clocks. Background contention is a plausible confound, not an isolated
causal finding.

The uncached I/O baseline is established: approximately 296 MB of expert
reads per post-seed token at 16 slots and 115 MB at 64 slots. A stable timing
comparison remains unaccepted. Longer samples should only be attempted in a
quiet window, rather than repeatedly selecting whichever noisy sweep passes.
