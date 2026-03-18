#!/bin/bash
# Async submit — runs popcorn-cli in the background, delivers result to agent.
# Usage: ./tools/submit.sh <test|benchmark|leaderboard>
# Returns immediately. Result is sent to your prompt when ready.

set -euo pipefail
KERNEL_DIR="$(pwd)"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$KERNEL_DIR/state"
SESSIONS_DIR="$STATE_DIR/sessions"

MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "Usage: ./tools/submit.sh <test|benchmark|leaderboard>"
    exit 2
fi

SESSION_ID="${AGENT_SESSION_ID:-}"
if [ -z "$SESSION_ID" ]; then
    echo "ERROR: AGENT_SESSION_ID not set."
    exit 2
fi

SESSION_FILE="$SESSIONS_DIR/${SESSION_ID}.jsonl"

# Check that a proposal exists in the current session
if [ ! -f "$SESSION_FILE" ] || ! grep -q '"action": "propose"' "$SESSION_FILE"; then
    echo "ERROR: No proposal registered. Run ./tools/propose.sh first."
    exit 1
fi

# Get the leaderboard name from submission.py
LEADERBOARD=$(grep '#!POPCORN leaderboard' "$KERNEL_DIR/submission.py" | head -1 | awk '{print $3}')
if [ -z "$LEADERBOARD" ]; then
    echo "ERROR: No #!POPCORN leaderboard directive found in submission.py"
    exit 1
fi

VERSION=$(head -20 "$KERNEL_DIR/submission.py" | grep -oE 'v[0-9]+' | head -1 | tr -d 'v' || echo "0")
KERNEL=$(basename "$KERNEL_DIR")
TMUX_SESSION="codex-${KERNEL}-${SESSION_ID}"

# Snapshot submission.py so the agent can keep editing
SNAPSHOT_DIR="$STATE_DIR/snapshots"
mkdir -p "$SNAPSHOT_DIR"
SNAPSHOT="$SNAPSHOT_DIR/v${VERSION}_${MODE}_$(date +%s).py"
cp "$KERNEL_DIR/submission.py" "$SNAPSHOT"

echo "=== Submitted v$VERSION for $MODE (background) ==="
echo "Results will be delivered to your prompt when ready."
echo "Continue working — do not wait."

# Launch background worker via tmux to ensure it survives Codex shell teardown
BG_SESSION="bg-${KERNEL}-v${VERSION}-${MODE}"
tmux kill-session -t "$BG_SESSION" 2>/dev/null || true
tmux new-session -d -s "$BG_SESSION" \
    "exec $REPO_DIR/tools/_submit_bg.sh $KERNEL_DIR $REPO_DIR $LEADERBOARD $MODE $SNAPSHOT $VERSION $SESSION_FILE $STATE_DIR $SESSION_ID $TMUX_SESSION"
