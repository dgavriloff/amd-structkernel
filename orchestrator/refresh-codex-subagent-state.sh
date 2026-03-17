#!/bin/bash
# Append a return event when the latest launched tmux-backed worker has exited.
# Usage: ./orchestrator/refresh-codex-subagent-state.sh --kernel <name>

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

KERNEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel) KERNEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

if [ -z "$KERNEL" ]; then
    echo "Usage: ./orchestrator/refresh-codex-subagent-state.sh --kernel <name>"
    exit 2
fi

STATE_FILE="$REPO_DIR/kernels/$KERNEL/state/subagents.jsonl"
if [ ! -f "$STATE_FILE" ]; then
    exit 0
fi

latest=$(python3 - "$STATE_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
last = None

with open(path, encoding="utf-8") as fp:
    for line in fp:
        line = line.strip()
        if not line:
            continue
        last = json.loads(line)

if last and last.get("action") == "launch":
    print(f"{last.get('session','')}\t{last.get('tmux_session','')}")
PY
)

if [ -z "$latest" ]; then
    exit 0
fi

IFS=$'\t' read -r session_id tmux_session <<< "$latest"
if [ -z "$tmux_session" ]; then
    exit 0
fi

if tmux has-session -t "$tmux_session" 2>/dev/null; then
    exit 0
fi

_SESSION_ID="$session_id" _TMUX_SESSION="$tmux_session" python3 - <<'PY' >> "$STATE_FILE"
import datetime
import json
import os

print(json.dumps({
    "action": "return",
    "session": int(os.environ["_SESSION_ID"]),
    "tmux_session": os.environ["_TMUX_SESSION"],
    "reason": "tmux session exited",
    "ts": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
}))
PY
