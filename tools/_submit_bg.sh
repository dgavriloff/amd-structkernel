#!/bin/bash
# Background worker for submit.sh — runs popcorn-cli and delivers result.
# Usage: _submit_bg.sh <kernel_dir> <repo_dir> <leaderboard> <mode> <snapshot> <version> <session_file> <state_dir> <session_id> <tmux_session>
# Not called directly — launched by submit.sh

set -uo pipefail

KERNEL_DIR="$1"
REPO_DIR="$2"
LEADERBOARD="$3"
MODE="$4"
SNAPSHOT="$5"
VERSION="$6"
SESSION_FILE="$7"
STATE_DIR="$8"
SESSION_ID="$9"
TMUX_SESSION="${10}"

# Run popcorn-cli
RESULT=$(cd "$KERNEL_DIR" && popcorn-cli submit --gpu MI355X --leaderboard "$LEADERBOARD" --mode "$MODE" --no-tui "$SNAPSHOT" 2>&1) || true
printf '%s\n' "$RESULT" > "$STATE_DIR/submit_v${VERSION}_${MODE}.log"

# Process result
MESSAGE=$(echo "$RESULT" | python3 "$REPO_DIR/tools/_process_result.py" \
    "$MODE" "$VERSION" "$SESSION_FILE" "$STATE_DIR/best.json" \
    "$STATE_DIR/tried.jsonl" "$KERNEL_DIR" "$SESSION_ID" "$SNAPSHOT" 2>/dev/null) || true

# Deliver to agent
if [ -n "$MESSAGE" ] && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux send-keys -t "$TMUX_SESSION" "$MESSAGE" C-m
    sleep 1
    tmux send-keys -t "$TMUX_SESSION" C-m
fi

# If revert limit reached, run close_branch.sh ourselves and kill the agent
if echo "$MESSAGE" | grep -q "REVERT LIMIT REACHED"; then
    # Summarize what failed from session file
    SUMMARY=$(python3 -c "
import json, sys
props = []
with open('$SESSION_FILE') as f:
    for line in f:
        e = json.loads(line)
        if e.get('action') == 'propose':
            props.append(e.get('what',''))
print('Attempted: ' + '; '.join(props[-5:]) if props else 'no proposals')
" 2>/dev/null || echo "5 leaderboard reverts reached")

    cd "$KERNEL_DIR" && AGENT_SESSION_ID="$SESSION_ID" "$REPO_DIR/tools/close_branch.sh" --what-failed "$SUMMARY" 2>/dev/null || true
fi
