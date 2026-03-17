#!/bin/bash
# Register the current tmux session as the orchestrator session.
# Usage: ./orchestrator/register-orchestrator-session.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$REPO_DIR/orchestrator/run"
STATE_FILE="$RUN_DIR/orchestrator.tmux"

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: not running inside tmux" >&2
    exit 1
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux not found" >&2; exit 1; }

SESSION_NAME="$(tmux display-message -p '#S')"
mkdir -p "$RUN_DIR"
printf '%s\n' "$SESSION_NAME" > "$STATE_FILE"
printf 'orchestrator_tmux=%s\n' "$SESSION_NAME"
