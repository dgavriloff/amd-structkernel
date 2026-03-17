#!/bin/bash
# Schedule a single future nudge into the registered orchestrator tmux session.
# Usage: ./orchestrator/nudge-status-timer.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SELF="$REPO_DIR/orchestrator/nudge-status-timer.sh"
ORCH_FILE="$REPO_DIR/orchestrator/run/orchestrator.tmux"

if [[ "${1:-}" = "--fire" ]]; then
    sleep 600

    if [ ! -f "$ORCH_FILE" ]; then
        exit 0
    fi

    command -v tmux >/dev/null 2>&1 || exit 0

    ORCH_TMUX="$(tr -d '\r\n' < "$ORCH_FILE")"
    if [ -z "$ORCH_TMUX" ]; then
        exit 0
    fi

    if ! tmux has-session -t "$ORCH_TMUX" 2>/dev/null; then
        exit 0
    fi

    tmux send-keys -t "$ORCH_TMUX" "check status, relaunch any missing workers, start the timer again, then return to idle" Enter
    sleep 1
    tmux send-keys -t "$ORCH_TMUX" Enter
    exit 0
fi

if pgrep -f "$SELF --fire" >/dev/null 2>&1; then
    exit 0
fi

nohup "$SELF" --fire >/dev/null 2>&1 < /dev/null &
