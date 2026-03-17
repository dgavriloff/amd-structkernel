#!/bin/bash
# Submit to popcorn and auto-update state.
# Usage: ./tools/submit.sh <test|leaderboard>
# - Rejects if no proposal registered in current session
# - On leaderboard: compares to best.json, prints KEEP or REVERT, updates state

set -euo pipefail
KERNEL_DIR="$(pwd)"
STATE_DIR="$KERNEL_DIR/state"
BEST_FILE="$STATE_DIR/best.json"
TRIED_FILE="$STATE_DIR/tried.jsonl"
SESSIONS_DIR="$STATE_DIR/sessions"

MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "Usage: ./tools/submit.sh <test|leaderboard>"
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

# Get current version number from submission.py header
VERSION=$(head -20 "$KERNEL_DIR/submission.py" | grep -oE 'v[0-9]+' | head -1 | tr -d 'v' || echo "0")

echo "=== Submitting in $MODE mode (v$VERSION) ==="

# Run popcorn
cd "$KERNEL_DIR"
RESULT=$(popcorn-cli submit --gpu MI355X --leaderboard "$LEADERBOARD" --mode "$MODE" --no-tui submission.py 2>&1)
echo "$RESULT"

# Log to session file
if [ "$MODE" = "test" ]; then
    if echo "$RESULT" | grep -qi "pass"; then
        TEST_RESULT="pass"
    else
        TEST_RESULT="fail"
    fi
    python3 -c "
import json, datetime
d = {'action': 'submit', 'mode': 'test', 'v': $VERSION, 'result': '$TEST_RESULT', 'ts': datetime.datetime.now(datetime.UTC).isoformat() + 'Z'}
print(json.dumps(d))
" >> "$SESSION_FILE"

elif [ "$MODE" = "leaderboard" ]; then
    # Extract ranked geomean from output
    SCORE=$(echo "$RESULT" | grep -i "ranked.*geomean\|geomean.*ranked" | grep -oE '[0-9]+\.[0-9]+' | tail -1 || echo "")
    if [ -z "$SCORE" ]; then
        SCORE=$(echo "$RESULT" | grep -i "geomean" | grep -oE '[0-9]+\.[0-9]+' | tail -1 || echo "")
    fi

    # Fallback: compute geomean from Ranked Benchmark ⏱ lines
    if [ -z "$SCORE" ]; then
        SCORE=$(echo "$RESULT" | python3 -c "
import sys, re, math
text = sys.stdin.read()
# Find the Ranked Benchmark section
idx = text.find('Ranked Benchmark')
if idx == -1:
    sys.exit(1)
ranked_section = text[idx:]
# Extract ⏱ times (mean values)
times = re.findall(r'⏱\s+([0-9]+\.[0-9]+)\s+±', ranked_section)
if not times:
    sys.exit(1)
times = [float(t) for t in times]
geomean = math.exp(sum(math.log(t) for t in times) / len(times))
print(f'{geomean:.2f}')
" 2>/dev/null || echo "")
    fi

    if [ -z "$SCORE" ]; then
        echo "WARNING: Could not extract score from output."
        echo "REVERT — could not parse score."
        python3 -c "
import json, datetime
d = {'action': 'submit', 'mode': 'leaderboard', 'v': $VERSION, 'score': None, 'best': None, 'kept': False, 'reason': 'could not parse score', 'ts': datetime.datetime.now(datetime.UTC).isoformat() + 'Z'}
print(json.dumps(d))
" >> "$SESSION_FILE"
        cp "$KERNEL_DIR/best_submission.py" "$KERNEL_DIR/submission.py"
        exit 0
    fi

    # Compare to best
    BEST_SCORE=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['score'])")
    BEST_VERSION=$(python3 -c "import json; print(json.load(open('$BEST_FILE'))['version'])")

    IMPROVED=$(python3 -c "print('yes' if float('$SCORE') < float('$BEST_SCORE') * 0.999 else 'no')")

    # Get the latest proposal's what/keywords from session file
    PROP_WHAT=$(grep '"action": "propose"' "$SESSION_FILE" | tail -1 | python3 -c "import sys,json; print(json.load(sys.stdin)['what'])")
    PROP_KEYWORDS=$(grep '"action": "propose"' "$SESSION_FILE" | tail -1 | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['keywords']))")

    if [ "$IMPROVED" = "yes" ]; then
        CHANGE=$(python3 -c "print(f'{(float(\"$SCORE\") / float(\"$BEST_SCORE\") - 1) * 100:.1f}%')")
        echo ""
        echo "════════════════════════════════════"
        echo "  KEEP — v$VERSION @ ${SCORE}µs ($CHANGE vs v$BEST_VERSION @ ${BEST_SCORE}µs)"
        echo "════════════════════════════════════"

        # Update best.json
        python3 -c "
import json, datetime
d = {'version': $VERSION, 'score': float('$SCORE'), 'ts': datetime.datetime.now(datetime.UTC).isoformat() + 'Z'}
json.dump(d, open('$BEST_FILE', 'w'))
"
        cp "$KERNEL_DIR/submission.py" "$KERNEL_DIR/best_submission.py"

        # Log to tried.jsonl
        _WHAT="$PROP_WHAT" python3 -c "
import json, datetime, os
d = {'v': $VERSION, 'what': os.environ['_WHAT'], 'keywords': $PROP_KEYWORDS, 'score': float('$SCORE'), 'kept': True, 'reason': 'improved geomean', 'session': $SESSION_ID}
print(json.dumps(d))
" >> "$TRIED_FILE"

        # Auto-commit on KEEP
        cd "$KERNEL_DIR" && git add -A && git commit -m "v${VERSION}: KEEP — ${PROP_WHAT} (${CHANGE})"
    else
        CHANGE=$(python3 -c "print(f'{(float(\"$SCORE\") / float(\"$BEST_SCORE\") - 1) * 100:+.1f}%')")
        echo ""
        echo "════════════════════════════════════"
        echo "  REVERT — v$VERSION @ ${SCORE}µs ($CHANGE vs v$BEST_VERSION @ ${BEST_SCORE}µs)"
        echo "════════════════════════════════════"

        cp "$KERNEL_DIR/best_submission.py" "$KERNEL_DIR/submission.py"

        # Log to tried.jsonl
        _WHAT="$PROP_WHAT" python3 -c "
import json, datetime, os
d = {'v': $VERSION, 'what': os.environ['_WHAT'], 'keywords': $PROP_KEYWORDS, 'score': float('$SCORE'), 'kept': False, 'reason': '$CHANGE vs best', 'session': $SESSION_ID}
print(json.dumps(d))
" >> "$TRIED_FILE"
    fi

    # Log to session file
    python3 -c "
import json, datetime
d = {'action': 'submit', 'mode': 'leaderboard', 'v': $VERSION, 'score': float('$SCORE'), 'best': float('$BEST_SCORE'), 'kept': $( [ "$IMPROVED" = "yes" ] && echo "True" || echo "False"), 'ts': datetime.datetime.now(datetime.UTC).isoformat() + 'Z'}
print(json.dumps(d))
" >> "$SESSION_FILE"
fi
