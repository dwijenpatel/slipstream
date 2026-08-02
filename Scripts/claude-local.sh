#!/bin/bash
# Run Claude Code against the local slipstream server (Anthropic
# /v1/messages endpoint, Qwen3.6-35B-A3B).
#
# Start the server first:
#   .build/release/TurboFieldfareServer --model ~/models/qwen36.gturbo \
#       --port 8091 --max-context 65536
#
# Usage:
#   Scripts/claude-local.sh -p "task prompt"     # headless one-shot
#   Scripts/claude-local.sh                      # interactive session
#
# The trim flags below cut Claude Code's opening request from ~33k tokens
# (tool schemas + CLAUDE.md + memory) to a few thousand, which matters at
# local prefill speeds. Add tools back per task via --tools/--allowedTools.
# Known limitation: one request at a time (single-slot server), and an
# abandoned request currently finishes generating before the next starts —
# do not point two clients at one server yet.
set -u
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8091}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-local-slipstream}"
export ANTHROPIC_MODEL="qwen3.6-35b-a3b"
export ANTHROPIC_SMALL_FAST_MODEL="qwen3.6-35b-a3b"
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
export CLAUDE_CODE_DISABLE_BUNDLED_SKILLS=1

exec claude \
  --model qwen3.6-35b-a3b \
  --bare \
  --tools "Bash,Read,Edit,Write,Glob,Grep" \
  --strict-mcp-config \
  --exclude-dynamic-system-prompt-sections \
  "$@"
