#!/bin/bash
# The overnight measurement plan of 2026-09-06. Runs unattended; authenticate
# sudo in your own shell first, because the purge needs it and the script
# never prompts. Every arm is a fresh process, one model process at a time,
# and the machine is held awake for the run.
#
# Phase 1, uncached (the purge, then the cache bypass keeps the pool cold):
#   p1  prefetch A/B at 64 slots: off, distance 1, distance 2, twice, interleaved
#   p2  the recorded ten-turn session under lfu, lru, and lfu-aging
#   p3  the session under lfu-aging with prefetch on
#   p4  prefill only at 12k and 24k, twice each
# Phase 2, warm (the whole model read once to level the cache):
#   p5  the session under lfu-aging, warm
#   p6  clock hold off against auto at 64 slots, twice
#   p7  prefill only at 12k and 24k, once each
#   p8  the slot-curve sweep with the new defaults (fill_table.sh)
# Then overnight_summary.py writes SUMMARY.md.
#
# USAGE
#   sudo -v && playbook/overnight.sh            # the whole plan, about five hours
#   playbook/overnight.sh --smoke --skip-purge --skip-sweep   # ten minutes, proves the plumbing
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
MODEL="${MODEL_GTURBO:-$HOME/models/qwen36.gturbo}"
BIN="$REPO/.build/release/slipstream"
SERVER="$REPO/.build/release/slipstream-server"
RECORDED="$REPO/bench-results/route-replay-20260905/state.json"
PORT=8091

SMOKE=0; SKIP_PURGE=0; SKIP_SWEEP=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1;;
    --skip-purge) SKIP_PURGE=1;;
    --skip-sweep) SKIP_SWEEP=1;;
    *) echo "unknown flag $a"; exit 2;;
  esac
done

OUT="$REPO/bench-results/overnight-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/overnight.log"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

# --- preflight ---------------------------------------------------------------
if pgrep -fl 'slipstream|TurboFieldfare|llama-server|mlx_lm' >/dev/null; then
  echo "a model process is live; refusing to start"; exit 1
fi
[ -x "$BIN" ] && [ -x "$SERVER" ] || { echo "build slipstream and slipstream-server first"; exit 1; }
[ -f "$RECORDED" ] || { echo "recorded session missing: $RECORDED"; exit 1; }
pmset -g batt | rg -q "AC Power" || say "WARNING: not on AC power"
caffeinate -dimsu -w $$ &
say "output: $OUT"
git -C "$REPO" rev-parse HEAD > "$OUT/commit"
git -C "$REPO" status --short > "$OUT/status"

if [ "$SKIP_PURGE" = 0 ]; then
  # Never prompt for the password in here: a prompt inside the script left
  # Ghostty in Secure Input with the keystrokes going nowhere (2026-09-06).
  # Authenticate in your own shell first: sudo -v && playbook/overnight.sh
  if ! sudo -n true 2>/dev/null; then
    echo "sudo is not authenticated; run 'sudo -v' in this shell, then start again"; exit 1
  fi
  say "purging the page cache"
  sudo -n purge || { say "purge failed"; exit 1; }
fi
residency() { uv run "$REPO/playbook/residency.py" "$MODEL/packed_experts"; }
say "residency after purge: $(residency)"

MAXNEW=512; TURNS=0; SESSION_MAX_TOKENS=4096; PREFILL_CTXS="12k 24k"
if [ "$SMOKE" = 1 ]; then MAXNEW=32; TURNS=2; SESSION_MAX_TOKENS=64; PREFILL_CTXS="1k"; fi

UNCACHED="TURBO_FIELDFARE_EXPERT_NOCACHE=1"
COMMON="TURBO_FIELDFARE_IO_BASELINE=1 TURBO_FIELDFARE_PHASES=1"

# cli_arm NAME CTX SLOTS MAXNEW "EXTRA CLI ARGS" ENV...
cli_arm() {
  local name=$1 ctx=$2 slots=$3 maxnew=$4 extra=$5; shift 5
  say "cli $name"
  residency > "$OUT/$name.residency-before"
  # shellcheck disable=SC2086
  env "$@" /usr/bin/time -l "$BIN" --model "$MODEL" \
    --messages-file "$REPO/playbook/prompts/ctx-$ctx.json" \
    --max-new "$maxnew" --max-context 32768 --temperature 0 --seed 20260723 \
    --expert-cache-slots "$slots" $extra > "$OUT/$name.out" 2> "$OUT/$name.err"
  residency > "$OUT/$name.residency-after"
  rg -N "tok/s=|expert io await|expert cache:|gpu cb1|gpu clock hold|prefetch:|predicted-route" "$OUT/$name.err" \
    | sed 's/^/    /' | tee -a "$LOG"
  shasum -a 256 "$OUT/$name.out" | cut -c1-16 | sed 's/^/    output /' | tee -a "$LOG"
}

# session_arm NAME POLICY ENV...
session_arm() {
  local name=$1 policy=$2; shift 2
  say "session $name"
  residency > "$OUT/$name.residency-before"
  env "$@" TURBO_FIELDFARE_ROUTE_TRACE="$OUT/$name.trace.bin" "$SERVER" --model "$MODEL" \
    --port "$PORT" --max-context 65536 --expert-cache-slots 64 --expert-cache-policy "$policy" \
    2> "$OUT/$name.server.log" &
  local pid=$!
  local ready=0
  for _ in $(seq 1 150); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
  done
  if [ "$ready" = 0 ]; then say "server did not become ready"; kill "$pid" 2>/dev/null; return; fi
  uv run "$REPO/playbook/session_replay.py" --recorded "$RECORDED" --out "$OUT/$name.state.json" \
    --turns "$TURNS" --max-tokens "$SESSION_MAX_TOKENS" > "$OUT/$name.replay.log" 2>&1
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  residency > "$OUT/$name.residency-after"
  tail -1 "$OUT/$name.replay.log" | sed 's/^/    /' | tee -a "$LOG"
  rg -o -N "total=[0-9.]+|expert_miss=[0-9]+|io_await=[0-9.]+" "$OUT/$name.server.log" \
    | awk -F= '{s[$1]+=$2} END {printf "    sum total=%.1f s expert_miss=%d io_await=%.1f s\n", s["total"], s["expert_miss"], s["io_await"]}' \
    | tee -a "$LOG"
}

# --- phase 1: uncached -----------------------------------------------------
say "phase 1: uncached"
for round in 1 2; do
  cli_arm "p1-prefetch-off-r$round" 3k 64 "$MAXNEW" "" $UNCACHED $COMMON
  cli_arm "p1-prefetch-d1-r$round"  3k 64 "$MAXNEW" "" $UNCACHED $COMMON TURBO_FIELDFARE_PREFETCH=1
  cli_arm "p1-prefetch-d2-r$round"  3k 64 "$MAXNEW" "" $UNCACHED $COMMON TURBO_FIELDFARE_PREFETCH=1 TURBO_FIELDFARE_PREFETCH_DISTANCE=2
done
for policy in lfu lru lfu-aging; do
  session_arm "p2-session-$policy-uncached" "$policy" $UNCACHED
done
session_arm "p3-session-lfu-aging-prefetch-uncached" lfu-aging $UNCACHED TURBO_FIELDFARE_PREFETCH=1
for ctx in $PREFILL_CTXS; do
  for r in 1 2; do
    cli_arm "p4-prefill-$ctx-uncached-r$r" "$ctx" 64 1 "" $UNCACHED $COMMON
  done
done

# --- phase 2: warm -----------------------------------------------------------
say "phase 2: leveling the page cache (one read of the model)"
fd -t f . "$MODEL" -X cat {} > /dev/null
say "residency after leveling: $(residency)"
session_arm "p5-session-lfu-aging-warm" lfu-aging
for r in 1 2; do
  cli_arm "p6-hold-off-r$r"  3k 64 "$MAXNEW" "--gpu-clock-hold off" $COMMON
  cli_arm "p6-hold-auto-r$r" 3k 64 "$MAXNEW" "" $COMMON
done
for ctx in $PREFILL_CTXS; do
  cli_arm "p7-prefill-$ctx-warm" "$ctx" 64 1 "" $COMMON
done
if [ "$SKIP_SWEEP" = 0 ] && [ "$SMOKE" = 0 ]; then
  say "phase 2: slot-curve sweep (fill_table.sh --only slipstream)"
  "$REPO/playbook/fill_table.sh" --only slipstream 2>&1 | tee -a "$LOG"
fi

# --- summary -----------------------------------------------------------------
uv run "$REPO/playbook/overnight_summary.py" "$OUT" > "$OUT/SUMMARY.md"
say "done; summary at $OUT/SUMMARY.md"
cat "$OUT/SUMMARY.md"
