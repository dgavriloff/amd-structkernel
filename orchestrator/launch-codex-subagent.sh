#!/bin/bash
# Launch one tmux-backed Codex worker for a kernel.
# Usage: ./orchestrator/launch-codex-subagent.sh --kernel <name> [--session-id <id>] [--tmux-session <name>] [--model <model>]

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$REPO_DIR/orchestrator/run"

KERNEL=""
SESSION_ID=""
TMUX_SESSION=""
MODEL="gpt-5.4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel) KERNEL="$2"; shift 2 ;;
        --session-id) SESSION_ID="$2"; shift 2 ;;
        --tmux-session) TMUX_SESSION="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

if [ -z "$KERNEL" ]; then
    echo "Usage: ./orchestrator/launch-codex-subagent.sh --kernel <name> [--session-id <id>] [--tmux-session <name>] [--model <model>]"
    exit 2
fi

KERNEL_DIR="$REPO_DIR/kernels/$KERNEL"
STATE_FILE="$KERNEL_DIR/state/subagents.jsonl"

if [ ! -d "$KERNEL_DIR" ] || [ ! -f "$KERNEL_DIR/AGENTS.md" ]; then
    echo "Unknown kernel or missing AGENTS.md: $KERNEL" >&2
    exit 2
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux not found" >&2; exit 1; }
command -v codex >/dev/null 2>&1 || { echo "codex not found" >&2; exit 1; }

if [ -z "$SESSION_ID" ]; then
    SESSION_ID="$("$REPO_DIR/orchestrator/next_id.sh")"
fi

if [ -z "$TMUX_SESSION" ]; then
    TMUX_SESSION="codex-${KERNEL}-${SESSION_ID}"
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "tmux session already exists: $TMUX_SESSION" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
LAUNCHER="$RUN_DIR/${TMUX_SESSION}.sh"
PROMPT="Run: export AGENT_SESSION_ID=$SESSION_ID — then read the AGENTS.md and follow the workflow. You are an AMD kernel optimization agent. IMPORTANT: Do not stop after one iteration. Keep looping through the workflow (hypothesize → propose → implement → test → benchmark) continuously until you hit 5 leaderboard reverts or exhaust your budget. Never pause to summarize and wait for input — just keep going."

cat > "$LAUNCHER" <<EOF
#!/bin/bash
set -euo pipefail
export AGENT_SESSION_ID=$SESSION_ID
exec codex --dangerously-bypass-approvals-and-sandbox --no-alt-screen --model "$MODEL" -C "$KERNEL_DIR" "$PROMPT"
EOF
chmod +x "$LAUNCHER"

tmux new-session -d -s "$TMUX_SESSION" "$LAUNCHER"

mkdir -p "$(dirname "$STATE_FILE")"
_SESSION_ID="$SESSION_ID" _TMUX_SESSION="$TMUX_SESSION" _MODEL="$MODEL" python3 - <<'PY' >> "$STATE_FILE"
import datetime
import json
import os

print(json.dumps({
    "action": "launch",
    "session": int(os.environ["_SESSION_ID"]),
    "tmux_session": os.environ["_TMUX_SESSION"],
    "model": os.environ["_MODEL"],
    "ts": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
}))
PY

printf 'kernel=%s session=%s tmux_session=%s\n' "$KERNEL" "$SESSION_ID" "$TMUX_SESSION"
