#!/bin/bash
# Send a subagent-finished notification into the registered orchestrator tmux session.
# Usage: ./orchestrator/notify-orchestrator.sh --kernel <name> --session-id <id> --event <event> [--reason <text>]

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_FILE="$REPO_DIR/orchestrator/run/orchestrator.tmux"

KERNEL=""
SESSION_ID=""
EVENT=""
REASON=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel) KERNEL="$2"; shift 2 ;;
        --session-id) SESSION_ID="$2"; shift 2 ;;
        --event) EVENT="$2"; shift 2 ;;
        --reason) REASON="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

if [ -z "$KERNEL" ] || [ -z "$SESSION_ID" ] || [ -z "$EVENT" ]; then
    echo "Usage: ./orchestrator/notify-orchestrator.sh --kernel <name> --session-id <id> --event <event> [--reason <text>]" >&2
    exit 2
fi

if [ ! -f "$STATE_FILE" ]; then
    exit 0
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux not found" >&2; exit 1; }

ORCH_TMUX="$(tr -d '\r\n' < "$STATE_FILE")"
if [ -z "$ORCH_TMUX" ]; then
    exit 0
fi

if ! tmux has-session -t "$ORCH_TMUX" 2>/dev/null; then
    exit 0
fi

MESSAGE="[subagent-finished] kernel=$KERNEL session=$SESSION_ID event=$EVENT"
if [ -n "$REASON" ]; then
    MESSAGE="$MESSAGE reason=$REASON"
fi

ESCAPED_MESSAGE=${MESSAGE//\'/\'\"\'\"\'}
tmux send-keys -t "$ORCH_TMUX" "printf '%s\n' '$ESCAPED_MESSAGE'" C-m
