#!/bin/bash
# Register a hypothesis before implementing.
# Usage: ./tools/propose.sh --what "..." --why "..." --keywords "k1,k2,k3" [--override "reason"]
# Checks tried.jsonl for keyword overlap. Exits 1 if >=2 matches and no --override.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TRIED="$KERNEL_DIR/state/tried.jsonl"
SESSIONS_DIR="$KERNEL_DIR/state/sessions"

# Parse args
WHAT=""
WHY=""
KEYWORDS=""
OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --what) WHAT="$2"; shift 2 ;;
        --why) WHY="$2"; shift 2 ;;
        --keywords) KEYWORDS="$2"; shift 2 ;;
        --override) OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

if [ -z "$WHAT" ] || [ -z "$WHY" ] || [ -z "$KEYWORDS" ]; then
    echo "Usage: ./tools/propose.sh --what \"...\" --why \"...\" --keywords \"k1,k2,k3\" [--override \"...\"]"
    exit 2
fi

SESSION_ID="${AGENT_SESSION_ID:-}"
if [ -z "$SESSION_ID" ]; then
    echo "ERROR: AGENT_SESSION_ID not set. The orchestrator must set this."
    exit 2
fi

SESSION_FILE="$SESSIONS_DIR/${SESSION_ID}.jsonl"

# Check tried.jsonl for keyword matches
MATCH_COUNT=0
MATCH_OUTPUT=""

if [ -f "$TRIED" ]; then
    IFS=',' read -ra KEYWORD_ARRAY <<< "$KEYWORDS"
    while IFS= read -r line; do
        all_match=true
        for kw in "${KEYWORD_ARRAY[@]}"; do
            kw_trimmed="$(echo "$kw" | xargs)"
            if ! echo "$line" | grep -qi "$kw_trimmed"; then
                all_match=false
                break
            fi
        done
        if $all_match; then
            MATCH_COUNT=$((MATCH_COUNT + 1))
            formatted=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  v{d['v']}: {d['what']} → {'KEPT' if d['kept'] else 'REVERTED'}: {d.get('reason','')}\")" 2>/dev/null || echo "  $line")
            MATCH_OUTPUT="${MATCH_OUTPUT}${formatted}\n"
        fi
    done < "$TRIED"
fi

if [ "$MATCH_COUNT" -ge 2 ] && [ -z "$OVERRIDE" ]; then
    echo "Found $MATCH_COUNT prior attempts matching [$KEYWORDS]:"
    echo -e "$MATCH_OUTPUT"
    echo "Use --override \"reason this is different\" to proceed."
    exit 1
fi

# Register the proposal
mkdir -p "$SESSIONS_DIR"
_WHAT="$WHAT" _WHY="$WHY" _KEYWORDS="$KEYWORDS" _OVERRIDE="$OVERRIDE" _MATCHES="$MATCH_COUNT" python3 -c "
import json, datetime, os
kws = [k.strip() for k in os.environ['_KEYWORDS'].split(',')]
d = {
    'action': 'propose',
    'what': os.environ['_WHAT'],
    'why': os.environ['_WHY'],
    'keywords': kws,
    'prior_matches': int(os.environ['_MATCHES']),
    'ts': datetime.datetime.now(datetime.UTC).isoformat() + 'Z'
}
override = os.environ.get('_OVERRIDE', '')
if override:
    d['override'] = override
print(json.dumps(d))
" >> "$SESSION_FILE"

if [ "$MATCH_COUNT" -gt 0 ]; then
    echo "Proposal registered with override ($MATCH_COUNT prior matches)."
else
    echo "Proposal registered. No prior matches."
fi
exit 0
