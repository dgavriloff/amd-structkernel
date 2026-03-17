#!/bin/bash
# Returns and reserves the next available session ID.
# Uses a counter file to avoid race conditions.
# Usage: ./orchestrator/next_id.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COUNTER_FILE="$REPO_DIR/orchestrator/.next_id"

# Initialize counter from existing session files if no counter exists
if [ ! -f "$COUNTER_FILE" ]; then
    MAX_ID=0
    for sessions_dir in "$REPO_DIR"/kernels/*/state/sessions; do
        if [ -d "$sessions_dir" ]; then
            for f in "$sessions_dir"/*.jsonl; do
                if [ -f "$f" ]; then
                    id=$(basename "$f" .jsonl)
                    if [[ "$id" =~ ^[0-9]+$ ]] && [ "$id" -gt "$MAX_ID" ]; then
                        MAX_ID=$id
                    fi
                fi
            done
        fi
    done
    echo $((MAX_ID + 1)) > "$COUNTER_FILE"
fi

# Read, increment, write back
ID=$(cat "$COUNTER_FILE")
echo $((ID + 1)) > "$COUNTER_FILE"
echo "$ID"
