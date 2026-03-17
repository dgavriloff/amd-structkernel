#!/bin/bash
# Returns the next available session ID across all kernels.
# Usage: ./orchestrator/next_id.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

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

echo $((MAX_ID + 1))
