# slipstream

Swift and Metal inference for Qwen3.6-35B-A3B on Apple Silicon, with the
experts streamed from the SSD into a bounded cache. Forked from TurboFieldfare;
the Gemma 4 path still builds and runs.

## Read first

- `README.md`: what is measured, on which machine, and what is inherited.
- `docs/REVIEW-2026-09-04.md`: the claims audit, the comparison to upstream,
  oMLX, MLX, and NVMAI, and the ranked TTFT and decode levers.
- `docs/SPEC_DECODE.md`: the speculative-decoding design and its measurements.
- `profiles/qwen36/README.md`: the measured decode budget per token.
- `playbook/README.md`: the measurement rules every number must follow.

## Layout and commands

`Sources/TurboFieldfare/` is the runtime. `Sources/TurboFieldfareRepack/`,
`Sources/TurboFieldfareCLI/`, `Sources/TurboFieldfareServer/`, and
`Sources/TurboFieldfareApp/` are the installer, the CLI, the loopback server,
and the Mac app. The target names keep the upstream prefix; the built
products are `slipstream`, `slipstream-server`, `slipstream-repack`,
`slipstream-mac`, and `slipstream-decode-service`. `Tests/` holds the serial
test suite. `docs/` holds design, benchmark, and experiment notes, including
the upstream Gemma documents. `playbook/` holds the benchmark harness and the
prompt files. `bench-results/` holds raw sweep output.

```bash
swift build -c release
.build/release/slipstream-repack --model qwen36 --output ~/models/qwen36.gturbo
.build/release/slipstream --model ~/models/qwen36.gturbo --prompt "The capital of France is" --max-new 64
.build/release/slipstream-server --model ~/models/qwen36.gturbo --port 8091 --max-context 65536
Scripts/test.sh
```

Runtime tuning defaults live in one place,
`Sources/TurboFieldfare/Runtime/Configuration/RuntimeDefaults.swift`, and a
drift test asserts every surface resolves to them. Change a default there, with
a controlled measurement in the commit message, or nowhere.

## Rules for a model run

- Never start a second model process. Before any run, check
  `pgrep -fl 'slipstream|TurboFieldfare|llama-server|mlx_lm'` and stop if it
  prints anything. A second process contaminates every measurement and
  competes for the same memory.
- Require macOS 26 or later, Swift 6.2 or later, a completed
  `~/models/qwen36.gturbo`, and enough free memory (`memory_pressure -Q`).
- Run the tests through `Scripts/test.sh`. Shared Metal state makes parallel
  tests unreliable.
- Do not download a full checkpoint, duplicate the `.gturbo` install, or purge
  caches to run a test.

## Rules for a performance number

- Use `playbook/fill_table.sh`. It levels the page cache, runs the arm list
  in round-robin passes, records only the last pass, and re-runs the first
  arm as a drift control. A sweep whose control moves more than 5 percent
  measured machine state, not the arms.
- Every measured run is a fresh process. Report the commit, hardware, macOS,
  Swift version, exact command, the timing footer, and the drift control.
- A same-binary A/B with alternating arms is the only interpretable
  comparison. Cross-session tokens-per-second figures are not comparable.
- An optimization that claims to be exact must produce byte-identical output
  at a fixed seed. Across kernel paths that round differently, the gate is the
  reference-tolerance tests at both production shapes.
- Keep negative results. Record what was tried, what it measured, and why it
  is off.

## Diagnostics

`TURBO_FIELDFARE_PHASES=1` prints the decode phase split and the expert-cache
hit counters after the CLI's timing footer. `TURBO_FIELDFARE_PREFETCH=1`,
`TURBO_FIELDFARE_PRED_ROUTE=1`, `TURBO_FIELDFARE_NO_CB_MERGE=1`, and
`TURBO_FIELDFARE_SPEC=1` enable measured-and-rejected or unfinished paths for
experiments only. None is a production setting.

## Server

Keep the server on `127.0.0.1`; it has no authentication or TLS. It serves one
request at a time, keeps one conversation's cache prefix in memory, cancels
generation when the client disconnects, and shuts down on SIGTERM. See
`docs/OPENAI_SERVER.md`.
