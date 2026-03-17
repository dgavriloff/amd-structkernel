#!/bin/bash
# Close the current session. Called after 5 reverts or when done.
# Usage: ./tools/close_branch.sh --what-failed "..." [--dead-technique "..." --dead-reason "..."]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$KERNEL_DIR/state"
DEAD_FILE="$STATE_DIR/dead.jsonl"
SESSIONS_DIR="$STATE_DIR/sessions"

SESSION_ID="${AGENT_SESSION_ID:-}"
if [ -z "$SESSION_ID" ]; then
    echo "ERROR: AGENT_SESSION_ID not set."
    exit 2
fi

SESSION_FILE="$SESSIONS_DIR/${SESSION_ID}.jsonl"

# Parse args
WHAT_FAILED=""
DEAD_TECHNIQUE=""
DEAD_REASON=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --what-failed) WHAT_FAILED="$2"; shift 2 ;;
        --dead-technique) DEAD_TECHNIQUE="$2"; shift 2 ;;
        --dead-reason) DEAD_REASON="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

if [ -z "$WHAT_FAILED" ]; then
    echo "Usage: ./tools/close_branch.sh --what-failed \"...\" [--dead-technique \"...\" --dead-reason \"...\"]"
    exit 2
fi

# Count reverts and keeps in this session
REVERTS=$(grep -c '"kept": false' "$SESSION_FILE" 2>/dev/null || true)
REVERTS=${REVERTS:-0}
KEEPS=$(grep -c '"kept": true' "$SESSION_FILE" 2>/dev/null || true)
KEEPS=${KEEPS:-0}

# Log close action to session file
_WHAT_FAILED="$WHAT_FAILED" python3 -c "
import json, datetime, os
d = {
    'action': 'close',
    'what_failed': os.environ['_WHAT_FAILED'],
    'reverts': $REVERTS,
    'keeps': $KEEPS,
    'ts': datetime.datetime.now(datetime.UTC).isoformat() + 'Z'
}
print(json.dumps(d))
" >> "$SESSION_FILE"

# If dead technique specified, append to dead.jsonl
if [ -n "$DEAD_TECHNIQUE" ] && [ -n "$DEAD_REASON" ]; then
    if [ -f "$DEAD_FILE" ] && grep -q "$DEAD_TECHNIQUE" "$DEAD_FILE"; then
        echo "Technique already in dead.jsonl, skipping duplicate."
    else
        _DEAD_TECHNIQUE="$DEAD_TECHNIQUE" _DEAD_REASON="$DEAD_REASON" python3 -c "
import json, os
d = {
    'technique': os.environ['_DEAD_TECHNIQUE'],
    'reason': os.environ['_DEAD_REASON'],
    'count': 1,
    'sessions': [$SESSION_ID]
}
print(json.dumps(d))
" >> "$DEAD_FILE"
        echo "Added '$DEAD_TECHNIQUE' to dead.jsonl."
    fi
fi

echo ""
echo "Session $SESSION_ID closed. $REVERTS reverts, $KEEPS keeps."
echo "Summary: $WHAT_FAILED"
