#!/bin/bash
# Fill the README comparison tables: every runtime, every context size.
#
# Emits one CSV row per (arm, context) with time to first token, sustained
# decode, and peak physical footprint, so the README tables can be rebuilt
# from data instead of from memory.
#
# HYGIENE, and why each rule is here:
# - Refuses to start if any model process is live. A forgotten server cost
#   us two contaminated sweeps already.
# - Page cache is the dominant confound on a streaming runtime: the SAME
#   config measured 22.15 tok/s early in a session and 25.27 late, a 14%
#   swing from nothing but warmth. So every cell runs twice and only the
#   second run is recorded, and arm 1 is re-run at the end as a drift
#   control. If the control disagrees with its first reading by more than
#   5%, the whole sweep measured machine state, not the arms.
# - Every measured run is a fresh process. In-process sigma understates
#   cross-process spread by a lot.
#
# USAGE
#   playbook/fill_table.sh                 # everything available
#   playbook/fill_table.sh --smoke         # 32 tokens per cell, ~10 min, proves the harness
#   playbook/fill_table.sh --only slipstream   # regex filter on arm name
#   playbook/fill_table.sh --contexts 1k,3k    # subset of context sizes
#   INCLUDE_RESIDENT=1 playbook/fill_table.sh  # add the 21.6 GB mlx-lm arm
#
# The resident mlx-lm arm is OFF by default: it peaked at 21.6 GB against a
# 21.3 GB wired limit and drove the machine deep into swap. Run it alone,
# attended, with browsers closed.
#
# Time estimate at full length: roughly 4-6 hours for the slipstream sweep
# plus llama.cpp and TurboFieldfare. Run it overnight.
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"

MODEL_GTURBO="${MODEL_GTURBO:-$HOME/models/qwen36.gturbo}"
MODEL_GGUF="${MODEL_GGUF:-$HOME/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
MODEL_MLX="${MODEL_MLX:-$HOME/models/qwen3.6-35b-mlx-4bit}"
BASELINE_BIN="${BASELINE_BIN:-/private/tmp/ss-baseline/.build/release/TurboFieldfareCLI}"
BIN="$REPO/.build/release/slipstream"

MAXNEW=512
SMOKE=0
ONLY=".*"
CONTEXTS="1k,3k,12k,24k"
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1; MAXNEW=32; shift;;
    --only) ONLY="$2"; shift 2;;
    --contexts) CONTEXTS="$2"; shift 2;;
    *) echo "unknown flag: $1"; exit 2;;
  esac
done

OUT="$REPO/bench-results/table-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "arm,context,prompt_tokens,ttft_s,decode_tok_s,rss_gb,peak_gb,run" > "$CSV"

# ---- hygiene ---------------------------------------------------------------
if pgrep -fl '\.build/release/slipstream|TurboFieldfareCLI|TurboFieldfareServer|slipstream-server|llama-server|llama-cli|ollama runner|mlx_lm|LM Studio Helper' >/dev/null 2>&1; then
  echo "REFUSING: a model process is already running. Stop it first:"
  pgrep -fl '\.build/release/slipstream|TurboFieldfareCLI|TurboFieldfareServer|slipstream-server|llama-server|llama-cli|ollama runner|mlx_lm|LM Studio Helper'
  exit 1
fi
[ -x "$BIN" ] || { echo "build first: swift build -c release"; exit 1; }
[ -f "$MODEL_GTURBO" ] || [ -d "$MODEL_GTURBO" ] || { echo "missing $MODEL_GTURBO"; exit 1; }

# Level the page cache before ANY arm runs. Per-cell warmup fixes warmth
# within a cell but not across the sweep: the cache keeps filling as the run
# proceeds, so early arms are measured cold and late arms warm. The smoke run
# on 2026-08-06 showed the drift control 31% above its own first reading for
# exactly this reason. Streaming the model once up front starts every arm from
# the same place. Reads the whole file, so it takes a minute on a cold cache.
echo "=== pre-warming the page cache with the whole model ==="
if [ -d "$MODEL_GTURBO" ]; then
  find "$MODEL_GTURBO" -type f -exec cat {} + > /dev/null 2>&1
else
  cat "$MODEL_GTURBO" > /dev/null 2>&1
fi
echo "    done"

echo "=== fill_table $(git rev-parse --short HEAD) $(date) ===" | tee "$OUT/meta.txt"
sw_vers >> "$OUT/meta.txt"
sysctl -n iogpu.wired_limit_mb | sed 's/^/wired_limit_mb: /' >> "$OUT/meta.txt"
echo "max_new=$MAXNEW contexts=$CONTEXTS only=$ONLY" >> "$OUT/meta.txt"

prompt_for() { echo "$REPO/playbook/prompts/ctx-$1.json"; }

record() { # arm context prompt_tokens ttft decode rss peak run
  echo "$1,$2,$3,$4,$5,$6,$7,$8" >> "$CSV"
  printf "    %-42s %-8s ttft %-8s decode %-9s rss %-7s peak %s\n" \
    "$1" "$2" "${4:--}" "${5:--}" "${6:--}" "${7:--}"
}

# Both numbers, because neither alone is honest across runtimes: physical
# footprint misses llama.cpp's mmapped weights (0.24 GB footprint against
# 11.24 GB RSS, measured), and RSS misses GPU allocations that footprint
# catches. Quote whichever fits the runtime and say which.
mem_of() { # timefile -> "rss peak"
  awk '/maximum resident set size/{r=$1} /peak memory footprint/{p=$1}
       END{printf "%.2f %.2f", r/1073741824, p/1073741824}' "$1"
}

# ---- slipstream / TurboFieldfare (same CLI surface) -------------------------
run_gturbo() { # arm binary context extra-args...
  local arm=$1 binary=$2 ctx=$3; shift 3
  local p; p=$(prompt_for "$ctx")
  [ -f "$p" ] || { echo "    missing prompt $p"; return; }
  for run in warm measured; do
    local log="$OUT/$arm.$ctx.$run.log"
    if [ "$run" = "measured" ]; then
      /usr/bin/time -l "$binary" --model "$MODEL_GTURBO" --messages-file "$p" \
        --max-new "$MAXNEW" --max-context 32768 --temperature 0 --seed 20260723 \
        "$@" >/dev/null 2>"$log"
    else
      "$binary" --model "$MODEL_GTURBO" --messages-file "$p" \
        --max-new "$MAXNEW" --max-context 32768 --temperature 0 --seed 20260723 \
        "$@" >/dev/null 2>"$log"
    fi
  done
  local foot ttft dec toks peak
  foot=$(grep -o '\[stop=[^]]*\]' "$OUT/$arm.$ctx.measured.log" | head -1)
  toks=$(echo "$foot" | sed -n 's/.*prefill=\([0-9]*\)tok.*/\1/p')
  ttft=$(echo "$foot" | sed -n 's/.*prefill=[0-9]*tok\/\([0-9.]*\)s.*/\1/p')
  dec=$(echo "$foot"  | sed -n 's/.*tok\/s=\([0-9.]*\).*/\1/p')
  read -r rss peak <<< "$(mem_of "$OUT/$arm.$ctx.measured.log")"
  record "$arm" "$ctx" "${toks:-}" "${ttft:-}" "${dec:-}" "$rss" "$peak" measured
}

# ---- llama.cpp -------------------------------------------------------------
# llama-bench, not llama-cli: llama-cli drops into an interactive prompt and
# hangs a headless run (observed 2026-08-05). llama-bench is purpose-built,
# takes prompt length in tokens directly, and emits CSV.
run_llamacpp() { # arm n_prompt extra-args...
  local arm=$1 nprompt=$2; shift 2
  command -v llama-bench >/dev/null || { echo "    llama-bench not on PATH, skipping"; return; }
  [ -f "$MODEL_GGUF" ] || { echo "    missing $MODEL_GGUF, skipping"; return; }
  local log="$OUT/$arm.$nprompt.measured.log"
  /usr/bin/time -l llama-bench --model "$MODEL_GGUF" -p "$nprompt" -n "$MAXNEW" \
    -ngl 999 -r 2 -o csv "$@" >"$log" 2>"$log.time"
  # llama-bench CSV quotes every field; col 34 n_prompt, 35 n_gen, 40 avg_ts.
  # The prompt-processing row has n_gen=0, the generation row n_prompt=0.
  local parsed pp tg
  parsed=$(python3 - "$log" <<'PYEOF'
import csv, sys
pp = tg = ""
with open(sys.argv[1]) as fh:
    for row in csv.DictReader(fh):
        if row.get("n_gen") == "0":
            pp = row.get("avg_ts", "")
        elif row.get("n_prompt") == "0":
            tg = row.get("avg_ts", "")
print(f"{pp}|{tg}")
PYEOF
)
  pp=${parsed%%|*}; tg=${parsed##*|}
  local ttft=""
  [ -n "$pp" ] && ttft=$(python3 -c "print(f'{$nprompt/float($pp):.2f}')" 2>/dev/null)
  [ -n "$tg" ] && tg=$(python3 -c "print(f'{float($tg):.2f}')" 2>/dev/null)
  local rss peak
  read -r rss peak <<< "$(mem_of "$log.time")"
  record "$arm" "${nprompt}tok" "$nprompt" "${ttft:-}" "${tg:-}" "$rss" "$peak" measured
}

# ---- mlx-lm ----------------------------------------------------------------
run_mlxlm() { # context
  local ctx=$1
  local py="$HOME/repos/gpu-kernel/.venv/bin/python"
  [ -x "$py" ] || { echo "    gpu-kernel venv missing, skipping mlx-lm"; return; }
  [ -d "$MODEL_MLX" ] || { echo "    missing $MODEL_MLX, skipping"; return; }
  local log="$OUT/mlx-lm.$ctx.measured.log"
  /usr/bin/time -l "$py" "$REPO/playbook/mlx_lm_bench.py" \
    "$MODEL_MLX" "$(prompt_for "$ctx")" "$MAXNEW" >"$log" 2>&1
  local ttft dec
  ttft=$(sed -n 's/^TTFT \([0-9.]*\).*/\1/p' "$log")
  dec=$(sed -n 's/.*= \([0-9.]*\) tok\/s.*/\1/p' "$log")
  local rss peak
  read -r rss peak <<< "$(mem_of "$log")"
  record "mlx-lm, resident" "$ctx" "" "${ttft:-}" "${dec:-}" "$rss" "$peak" measured
}

IFS=',' read -ra CTXS <<< "$CONTEXTS"

echo ""
echo "=== global warmup (pages the model in so arm 1 is not penalised) ==="
"$BIN" --model "$MODEL_GTURBO" --messages-file "$(prompt_for 3k)" --max-new 8 \
  --max-context 32768 --temperature 0 --seed 20260723 >/dev/null 2>&1

FIRST_ARM=""
for ctx in "${CTXS[@]}"; do
  echo ""
  echo "=== context $ctx ==="

  # 8 is omitted deliberately: the prefill routed-tile scheduler refuses
  # fewer than 16 slots ("needs 16 slots, has 8"), so it is unrunnable
  # rather than merely slow.
  for slots in 16 32 64 96 128 192; do
    arm="slipstream, $slots of 256 experts cached"
    echo "$arm" | grep -qE "$ONLY" || continue
    [ -z "$FIRST_ARM" ] && FIRST_ARM="$slots|$ctx"
    run_gturbo "slipstream-slots$slots" "$BIN" "$ctx" --expert-cache-slots "$slots"
  done

  if echo "TurboFieldfare" | grep -qE "$ONLY"; then
    if [ -x "$BASELINE_BIN" ]; then
      # Upstream shares the CLI surface, so sweeping its cache at the SAME
      # sizes isolates kernel differences from cache-configuration ones.
      run_gturbo "turbofieldfare-defaults" "$BASELINE_BIN" "$ctx"
      for slots in 32 128; do
        run_gturbo "turbofieldfare-slots$slots" "$BASELINE_BIN" "$ctx" --expert-cache-slots "$slots"
      done
    else
      echo "    baseline binary missing ($BASELINE_BIN), skipping TurboFieldfare"
    fi
  fi

  if echo "llama.cpp" | grep -qE "$ONLY"; then
    case "$ctx" in 1k) np=1000;; 3k) np=3000;; 12k) np=12000;; 24k) np=24000;; *) np=3000;; esac
    run_llamacpp "llamacpp-default" "$np"
    run_llamacpp "llamacpp-ncpumoe32" "$np" --n-cpu-moe 32
  fi

  if [ "${INCLUDE_RESIDENT:-0}" = "1" ] && echo "mlx-lm" | grep -qE "$ONLY"; then
    echo "    WARNING: resident arm peaks near the wired limit; watch memory"
    run_mlxlm "$ctx"
  fi
done

# ---- drift control ---------------------------------------------------------
if [ -n "$FIRST_ARM" ]; then
  slots=${FIRST_ARM%%|*}; ctx=${FIRST_ARM##*|}
  echo ""
  echo "=== drift control: re-running the first arm last ==="
  run_gturbo "DRIFT-slipstream-slots$slots" "$BIN" "$ctx" --expert-cache-slots "$slots"
  echo ""
  echo "Compare DRIFT-slipstream-slots$slots against slipstream-slots$slots at $ctx."
  echo "More than 5% apart means the sweep measured machine state, not the arms."
fi

echo ""
echo "=== results ==="
column -t -s, "$CSV"
echo ""
echo "CSV: $CSV"
echo "Rebuild the README tables with: playbook/table_from_csv.py $CSV"
