#!/bin/bash
# Returns and reserves the next available session ID.
# Uses a counter file to avoid race conditions.
# Usage: ./orchestrator/next_id.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COUNTER_FILE="$REPO_DIR/orchestrator/.next_id"
LOCK_FILE="$REPO_DIR/orchestrator/.next_id.lock"

python3 - "$COUNTER_FILE" "$LOCK_FILE" "$REPO_DIR" <<'PY'
import fcntl
import glob
import os
import sys

counter_file, lock_file, repo_dir = sys.argv[1:4]

os.makedirs(os.path.dirname(counter_file), exist_ok=True)
with open(lock_file, "a+", encoding="utf-8") as lock_fp:
    fcntl.flock(lock_fp, fcntl.LOCK_EX)

    if not os.path.exists(counter_file):
        max_id = 0
        pattern = os.path.join(repo_dir, "kernels", "*", "state", "sessions", "*.jsonl")
        for path in glob.glob(pattern):
            name = os.path.splitext(os.path.basename(path))[0]
            if name.isdigit():
                max_id = max(max_id, int(name))
        with open(counter_file, "w", encoding="utf-8") as counter_fp:
            counter_fp.write(f"{max_id + 1}\n")

    with open(counter_file, "r", encoding="utf-8") as counter_fp:
        raw = counter_fp.read().strip()
    current = int(raw) if raw else 1

    with open(counter_file, "w", encoding="utf-8") as counter_fp:
        counter_fp.write(f"{current + 1}\n")

    print(current)
PY
