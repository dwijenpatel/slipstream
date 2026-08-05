#!/bin/bash
# Full-stack performance matrix for slipstream vs upstream TurboFieldfare.
#
# Measures every shipped feature on the community long-synthesis case:
#   A  upstream baseline (pre-slipstream commit, separate binary)
#   B  slipstream defaults (GQA decode kernel; chunk still 128)
#   C  B + --prefill-chunk auto
#   D  C + --expert-cache-slots 128
#   E  D + phase diagnostics (PHASES + CB1_SPLIT; numbers for the record)
#   F  D + KV snapshot: save arm, then restore arm (TTFT path)
#   G  D + predicted-routing telemetry (recall + its overhead)
#   H  D + prefetch (expected NET SLOWER on resident models; verifies the
#      documented negative result still holds)
#   B' drift control: arm B re-measured last; if it does not reproduce,
#      the sequence measured machine state, not the features.
#
# Hygiene: refuses to run if any model process is live; one warmup run per
# arm in a throwaway process; measured runs are fresh processes; outputs
# byte-checked against arm B's reference where the feature claims
# exactness. ~25 minutes total. Run attended, on power, lid open.
#
# Usage:
#   playbook/bench_all.sh                # full matrix
#   SMOKE=1 playbook/bench_all.sh        # 32-token smoke of the harness
#   BASELINE_BIN=<path> to override the upstream binary location.
set -u
cd "$(dirname "$0")/.."

MODEL="${MODEL:-$HOME/models/qwen36.gturbo}"
PROMPT=docs/benchmark-prompts/real-generation-v1/long-synthesis.json
MAXNEW=$(( ${SMOKE:-0} == 1 ? 32 : 512 ))
BIN=.build/release/slipstream
BASELINE_BIN="${BASELINE_BIN:-/tmp/ss-baseline/.build/release/slipstream}"
OUT="bench-results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

if pgrep -fl '\.build/release/slipstream|TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' >/dev/null; then
  echo "REFUSING: another model process is running (protocol contamination)."
  exit 1
fi
[ -x "$BIN" ] || { echo "build first: swift build -c release"; exit 1; }
[ -x "$BASELINE_BIN" ] || echo "WARN: no baseline binary at $BASELINE_BIN — arm A skipped"

COMMON=(--messages-file "$PROMPT" --max-new "$MAXNEW" --max-context 4096
        --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260723)

run_arm () { # name binary [env=VAL ...] -- [extra cli args ...]
  local name=$1 binary=$2; shift 2
  local envs=() args=()
  local sep=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then sep=1; continue; fi
    if [ $sep -eq 0 ]; then envs+=("$a"); else args+=("$a"); fi
  done
  echo "--- $name"
  env ${envs[@]+"${envs[@]}"} "$binary" --model "$MODEL" "${COMMON[@]}" ${args[@]+"${args[@]}"} \
      > /dev/null 2>&1                                   # warmup
  # A warmup that wrote the snapshot would turn the save arm into a
  # restore arm; clear it so the measured run exercises the save path.
  if [ -n "${RESET_SNAPSHOT:-}" ]; then rm -f "$RESET_SNAPSHOT"; RESET_SNAPSHOT=""; fi
  env ${envs[@]+"${envs[@]}"} /usr/bin/time -l "$binary" --model "$MODEL" "${COMMON[@]}" ${args[@]+"${args[@]}"} \
      > "$OUT/$name.stdout" 2> "$OUT/$name.stderr"
  local foot wall
  foot=$(grep -o '\[stop=[^]]*\]' "$OUT/$name.stderr" | head -1)
  wall=$(awk '/real/{print $1; exit}' "$OUT/$name.stderr")
  echo "$name|$foot|wall=${wall}s" >> "$OUT/summary.raw"
  echo "    $foot wall=${wall}s"
}

echo "=== slipstream bench $(git rev-parse --short HEAD), $(date) ===" | tee "$OUT/meta.txt"
sw_vers >> "$OUT/meta.txt"; git log --oneline -1 >> "$OUT/meta.txt"

[ -x "$BASELINE_BIN" ] && \
run_arm A_upstream      "$BASELINE_BIN" --
run_arm B_defaults      "$BIN" --
run_arm C_chunk_auto    "$BIN" -- --prefill-chunk auto
run_arm D_slots128      "$BIN" -- --prefill-chunk auto --expert-cache-slots 128
run_arm E_diagnostics   "$BIN" TURBO_FIELDFARE_PHASES=1 TURBO_FIELDFARE_CB1_SPLIT=1 \
                        -- --prefill-chunk auto --expert-cache-slots 128
rm -f "$OUT/kv.tfkv"; RESET_SNAPSHOT="$OUT/kv.tfkv"
run_arm F1_snap_save    "$BIN" -- --prefill-chunk auto --expert-cache-slots 128 --kv-snapshot "$OUT/kv.tfkv"
run_arm F2_snap_restore "$BIN" -- --prefill-chunk auto --expert-cache-slots 128 --kv-snapshot "$OUT/kv.tfkv"
run_arm G_pred_route    "$BIN" TURBO_FIELDFARE_PRED_ROUTE=1 TURBO_FIELDFARE_PHASES=1 \
                        -- --prefill-chunk auto --expert-cache-slots 128
run_arm H_prefetch      "$BIN" TURBO_FIELDFARE_PREFETCH=1 \
                        -- --prefill-chunk auto --expert-cache-slots 128
run_arm B2_drift_ctrl   "$BIN" --

echo ""
echo "=== identity gates (vs B_defaults where exactness is claimed) ==="
{
for arm in C_chunk_auto D_slots128 E_diagnostics F1_snap_save F2_snap_restore G_pred_route H_prefetch; do
  [ -f "$OUT/$arm.stdout" ] || continue
  if cmp -s "$OUT/B_defaults.stdout" "$OUT/$arm.stdout"; then
    echo "$arm: IDENTICAL"
  else
    echo "$arm: DIFFERS   <-- investigate before trusting its timing"
  fi
done
} | tee "$OUT/identity.txt"

echo ""
echo "=== summary ==="
column -t -s'|' "$OUT/summary.raw" | tee -a "$OUT/meta.txt"
echo ""
echo "phase detail (arm E):"; grep -E "gpu |await|wait wall|phases over" "$OUT/E_diagnostics.stderr" || true
echo "predicted-route recall (arm G):"; grep "predicted-route" "$OUT/G_pred_route.stderr" || true
echo ""
echo "drift check: compare B_defaults vs B2_drift_ctrl above; >5% divergence"
echo "invalidates ordering-sensitive comparisons (page-cache warming)."
echo "Raw logs: $OUT/"
