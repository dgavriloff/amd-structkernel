#!/bin/bash
# Submit to popcorn and auto-update state.
# Usage: ./tools/submit.sh <test|benchmark>
# - Rejects if no proposal registered in current session
# - Agents cannot submit in leaderboard mode (orchestrator handles that)
# - On benchmark: compares to best.json, prints KEEP or REVERT, updates state
# - Rate limited: 10 test/hr, 10 benchmark/hr per kernel

set -euo pipefail
KERNEL_DIR="$(pwd)"
STATE_DIR="$KERNEL_DIR/state"
BEST_FILE="$STATE_DIR/best.json"
TRIED_FILE="$STATE_DIR/tried.jsonl"
SESSIONS_DIR="$STATE_DIR/sessions"
RATE_FILE="$STATE_DIR/rate_limits.jsonl"

MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "Usage: ./tools/submit.sh <test|benchmark>"
    exit 2
fi

# Block agents from leaderboard mode — orchestrator handles LB
if [ "$MODE" = "leaderboard" ]; then
    echo "ERROR: Leaderboard submissions are handled by the orchestrator."
    echo "Use './tools/submit.sh benchmark' to measure performance."
    exit 1
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

# --- Rate limiting ---
MAX_PER_HOUR=10
touch "$RATE_FILE"

(
    flock 200

    while true; do
        NOW=$(date +%s)
        CUTOFF=$((NOW - 3600))

        # Count submissions of this mode in the last hour
        COUNT=$(python3 -c "
import json, sys
count = 0
lines = []
for line in open('$RATE_FILE'):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    if d['ts_epoch'] > $CUTOFF:
        lines.append(line)
        if d['mode'] == '$MODE':
            count += 1
# Prune old entries
with open('$RATE_FILE', 'w') as f:
    for l in lines:
        f.write(l + '\n')
print(count)
")

        if [ "$COUNT" -lt "$MAX_PER_HOUR" ]; then
            break
        fi

        # Find when the oldest entry for this mode expires
        WAIT=$(python3 -c "
import json
now = $(date +%s)
cutoff = now - 3600
oldest = None
for line in open('$RATE_FILE'):
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    if d['mode'] == '$MODE' and d['ts_epoch'] > cutoff:
        if oldest is None or d['ts_epoch'] < oldest:
            oldest = d['ts_epoch']
wait = (oldest + 3600) - now + 5 if oldest else 60
print(max(wait, 5))
")
        echo "Rate limit hit ($COUNT/$MAX_PER_HOUR $MODE/hr). Waiting ${WAIT}s..."
        sleep "$WAIT"
    done

) 200>"$RATE_FILE.lock"

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

# Record rate limit entry
(
    flock 200
    python3 -c "
import json
d = {'mode': '$MODE', 'ts_epoch': $(date +%s)}
print(json.dumps(d))
" >> "$RATE_FILE"
) 200>"$RATE_FILE.lock"

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

elif [ "$MODE" = "benchmark" ]; then
    # Compute geomean from Ranked Benchmark lines (normalizing ms -> us)
    SCORE=$(echo "$RESULT" | python3 -c "
import sys, re, math
text = sys.stdin.read()
# Find the Ranked Benchmark section
idx = text.find('Ranked Benchmark')
if idx == -1:
    idx = text.find('ranked benchmark')
if idx == -1:
    idx = text.find('Ranked benchmark')
if idx == -1:
    print('ERROR: No Ranked Benchmark section found in output', file=sys.stderr)
    sys.exit(1)
ranked_section = text[idx:]
# Extract mean times with units
entries = re.findall(r'⏱\s+([0-9]+(?:\.[0-9]+)?)\s*±[^µm\n]*?(µs|ms)', ranked_section)
if not entries:
    entries = re.findall(r'([0-9]+(?:\.[0-9]+)?)\s*±[^µm\n]*?(µs|ms)', ranked_section)
if not entries:
    print('ERROR: No timing values found in Ranked Benchmark section', file=sys.stderr)
    sys.exit(1)
# Normalize all to us
times_us = []
for val, unit in entries:
    t = float(val)
    if unit == 'ms':
        t *= 1000.0
    times_us.append(t)
geomean = math.exp(sum(math.log(t) for t in times_us) / len(times_us))
print(f'{geomean:.2f}')
" || echo "")

    if [ -z "$SCORE" ]; then
        echo "WARNING: Could not extract score from output."
        echo "DEBUG: Dumping first 100 lines of RESULT:"
        echo "$RESULT" | head -100
        echo "---END DEBUG---"
        echo "REVERT — could not parse score."
        python3 -c "
import json, datetime
d = {'action': 'submit', 'mode': 'benchmark', 'v': $VERSION, 'score': None, 'best': None, 'kept': False, 'reason': 'could not parse score', 'ts': datetime.datetime.now(datetime.UTC).isoformat() + 'Z'}
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
d = {'action': 'submit', 'mode': 'benchmark', 'v': $VERSION, 'score': float('$SCORE'), 'best': float('$BEST_SCORE'), 'kept': $( [ "$IMPROVED" = "yes" ] && echo "True" || echo "False"), 'ts': datetime.datetime.now(datetime.UTC).isoformat() + 'Z'}
print(json.dumps(d))
" >> "$SESSION_FILE"
fi
