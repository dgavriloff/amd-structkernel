#!/bin/bash
# Returns the next version number for this kernel.
# Usage: ./tools/next_version.sh

set -euo pipefail
KERNEL_DIR="$(pwd)"

MAX_V=0

get_version() {
    local v="$1"
    # Strip leading zeros to avoid octal interpretation
    v=$(echo "$v" | sed 's/^0*//')
    v=${v:-0}
    if [ "$v" -gt "$MAX_V" ] 2>/dev/null; then
        MAX_V=$v
    fi
}

# Check submissions/ directory
if [ -d "$KERNEL_DIR/submissions" ]; then
    for f in "$KERNEL_DIR/submissions"/v*.py; do
        [ -f "$f" ] || continue
        v=$(basename "$f" | grep -oE 'v[0-9]+' | head -1 | tr -d 'v' || true)
        [ -n "$v" ] && get_version "$v"
    done
fi

# Check submission.py header
if [ -f "$KERNEL_DIR/submission.py" ]; then
    v=$(head -20 "$KERNEL_DIR/submission.py" | grep -oE 'v[0-9]+' | head -1 | tr -d 'v' || true)
    [ -n "$v" ] && get_version "$v"
fi

echo $((MAX_V + 1))
