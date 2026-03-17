#!/bin/bash
# Kill the tmux session for a kernel session and record the result.
# Usage: ./orchestrator/end-codex-subagent.sh --kernel-dir <dir> --session-id <id>

set -euo pipefail

KERNEL_DIR=""
SESSION_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel-dir) KERNEL_DIR="$2"; shift 2 ;;
        --session-id) SESSION_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

if [ -z "$KERNEL_DIR" ] || [ -z "$SESSION_ID" ]; then
    echo "Usage: ./orchestrator/end-codex-subagent.sh --kernel-dir <dir> --session-id <id>"
    exit 2
fi

STATE_FILE="$KERNEL_DIR/state/subagents.jsonl"
if [ ! -f "$STATE_FILE" ]; then
    exit 0
fi

tmux_session=$(python3 - "$STATE_FILE" "$SESSION_ID" <<'PY'
import json
import sys

path = sys.argv[1]
session_id = int(sys.argv[2])
tmux_session = ""

with open(path, encoding="utf-8") as fp:
    for line in fp:
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        if event.get("action") == "launch" and event.get("session") == session_id:
            tmux_session = event.get("tmux_session", "")

print(tmux_session)
PY
)

sleep 1

tmux_killed=false
reason="no launch record"

if [ -n "$tmux_session" ]; then
    if tmux has-session -t "$tmux_session" 2>/dev/null; then
        if tmux kill-session -t "$tmux_session" 2>/dev/null; then
            if tmux has-session -t "$tmux_session" 2>/dev/null; then
                reason="kill returned success but tmux session still exists"
            else
                tmux_killed=true
                reason="killed"
            fi
        else
            reason="tmux kill failed"
        fi
    else
        reason="tmux session not found"
    fi
fi

_SESSION_ID="$SESSION_ID" _TMUX_SESSION="$tmux_session" _TMUX_KILLED="$tmux_killed" _REASON="$reason" python3 - <<'PY' >> "$STATE_FILE"
import datetime
import json
import os

print(json.dumps({
    "action": "close",
    "session": int(os.environ["_SESSION_ID"]),
    "tmux_session": os.environ["_TMUX_SESSION"],
    "tmux_killed": os.environ["_TMUX_KILLED"].lower() == "true",
    "reason": os.environ["_REASON"],
    "ts": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
}))
PY
