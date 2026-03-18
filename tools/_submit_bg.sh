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

# Process result
MESSAGE=$(echo "$RESULT" | python3 "$REPO_DIR/tools/_process_result.py" \
    "$MODE" "$VERSION" "$SESSION_FILE" "$STATE_DIR/best.json" \
    "$STATE_DIR/tried.jsonl" "$KERNEL_DIR" "$SESSION_ID" "$SNAPSHOT" 2>/dev/null) || true

# Deliver to agent — extra Enter ensures Codex processes queued input
if [ -n "$MESSAGE" ] && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux send-keys -t "$TMUX_SESSION" "$MESSAGE" Enter Enter
fi
