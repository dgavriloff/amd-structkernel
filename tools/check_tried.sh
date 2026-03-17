#!/bin/bash
# Check if a technique has been tried before.
# Usage: ./tools/check_tried.sh "keyword1 keyword2 ..."
# Exits 0 if <2 matches, exits 1 if >=2 matches.

set -euo pipefail
KERNEL_DIR="$(pwd)"
TRIED="$KERNEL_DIR/state/tried.jsonl"

if [ -z "${1:-}" ]; then
    echo "Usage: ./tools/check_tried.sh \"keyword1 keyword2 ...\""
    exit 2
fi

if [ ! -f "$TRIED" ]; then
    echo "No prior attempts found."
    exit 0
fi

SEARCH_TERMS="$1"
MATCHES=()

while IFS= read -r line; do
    matched=true
    for term in $SEARCH_TERMS; do
        if ! echo "$line" | grep -qi "$term"; then
            matched=false
            break
        fi
    done
    if $matched; then
        MATCHES+=("$line")
    fi
done < "$TRIED"

COUNT=${#MATCHES[@]}

if [ "$COUNT" -eq 0 ]; then
    echo "No prior attempts matching [$SEARCH_TERMS]."
    exit 0
elif [ "$COUNT" -eq 1 ]; then
    echo "Found $COUNT prior attempt matching [$SEARCH_TERMS]:"
    for m in "${MATCHES[@]}"; do
        echo "$m" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  v{d['v']}: {d['what']} → {'KEPT' if d['kept'] else 'REVERTED'}: {d.get('reason','')}\")" 2>/dev/null || echo "  $m"
    done
    exit 0
else
    echo "Found $COUNT prior attempts matching [$SEARCH_TERMS]:"
    for m in "${MATCHES[@]}"; do
        echo "$m" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  v{d['v']}: {d['what']} → {'KEPT' if d['kept'] else 'REVERTED'}: {d.get('reason','')}\")" 2>/dev/null || echo "  $m"
    done
    echo ""
    echo "This technique has been tried $COUNT times. Use --override in propose.sh to proceed."
    exit 1
fi
