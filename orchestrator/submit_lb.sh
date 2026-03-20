#!/bin/bash
# Submit leaderboard for kernels that improved since last LB submission.
# Usage: ./orchestrator/submit_lb.sh
# Designed to run via /loop 60m

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LB_FILE="$REPO_DIR/state/last_lb.json"

# Initialize last_lb.json if missing
if [ ! -f "$LB_FILE" ]; then
    echo "{}" > "$LB_FILE"
fi

echo "=== Leaderboard submission check ($(date -u '+%Y-%m-%dT%H:%M:%SZ')) ==="

for kernel_dir in "$REPO_DIR"/kernels/*/; do
    kernel=$(basename "$kernel_dir")
    best_file="$kernel_dir/state/best.json"

    if [ ! -f "$best_file" ]; then
        echo "[$kernel] No best.json — skipping"
        continue
    fi

    # Read current best BM score
    BM_SCORE=$(python3 -c "import json; print(json.load(open('$best_file'))['score'])")
    BM_VERSION=$(python3 -c "import json; print(json.load(open('$best_file'))['version'])")

    # Read last LB score for this kernel
    LAST_LB_SCORE=$(python3 -c "
import json
lb = json.load(open('$LB_FILE'))
k = lb.get('$kernel', {})
print(k.get('bm_score', k.get('score', 999999)))
")

    # Check if improved (>0.1% better)
    IMPROVED=$(python3 -c "print('yes' if float('$BM_SCORE') < float('$LAST_LB_SCORE') * 0.999 else 'no')")

    if [ "$IMPROVED" != "yes" ]; then
        echo "[$kernel] No improvement (v$BM_VERSION @ ${BM_SCORE}µs vs last LB @ ${LAST_LB_SCORE}µs) — skipping"
        continue
    fi

    echo "[$kernel] Improved! v$BM_VERSION @ ${BM_SCORE}µs (was ${LAST_LB_SCORE}µs). Submitting LB..."

    # Get leaderboard name from best_submission.py
    LEADERBOARD=$(grep '#!POPCORN leaderboard' "$kernel_dir/best_submission.py" | head -1 | awk '{print $3}')
    if [ -z "$LEADERBOARD" ]; then
        echo "[$kernel] ERROR: No #!POPCORN leaderboard directive in best_submission.py — skipping"
        continue
    fi

    # Submit to leaderboard
    cd "$kernel_dir"
    RESULT=$(popcorn-cli submit --gpu MI355X --leaderboard "$LEADERBOARD" --mode leaderboard --no-tui best_submission.py 2>&1) || true
    echo "$RESULT"

    # Check for submission failure
    if echo "$RESULT" | grep -qi "failed\|error\|timeout\|unauthorized"; then
        echo "[$kernel] LB submission FAILED — will retry next cycle"
        continue
    fi

    # Extract LB score if available
    LB_SCORE=$(echo "$RESULT" | python3 -c "
import sys, re, math
text = sys.stdin.read()
idx = text.find('Ranked Benchmark')
if idx == -1: idx = text.find('ranked benchmark')
if idx == -1: idx = text.find('Ranked benchmark')
if idx == -1: idx = text.find('## Benchmarks:')
if idx == -1: idx = text.find('Benchmarks:')
if idx == -1:
    print('')
    sys.exit(0)
ranked_section = text[idx:]
entries = re.findall(r'⏱\s+([0-9]+(?:\.[0-9]+)?)\s*±[^µm\n]*?(µs|ms)', ranked_section)
if not entries:
    entries = re.findall(r'([0-9]+(?:\.[0-9]+)?)\s*±[^µm\n]*?(µs|ms)', ranked_section)
if not entries:
    print('')
    sys.exit(0)
times_us = []
for val, unit in entries:
    t = float(val)
    if unit == 'ms': t *= 1000.0
    times_us.append(t)
geomean = math.exp(sum(math.log(t) for t in times_us) / len(times_us))
print(f'{geomean:.3f}')
" || echo "")

    # Only update last_lb.json if submission succeeded
    python3 -c "
import json, datetime
lb = json.load(open('$LB_FILE'))
lb['$kernel'] = {
    'version': $BM_VERSION,
    'bm_score': float('$BM_SCORE'),
    'lb_score': float('${LB_SCORE:-0}') if '${LB_SCORE:-}' else None,
    'ts': datetime.datetime.now(datetime.UTC).isoformat() + 'Z'
}
json.dump(lb, open('$LB_FILE', 'w'), indent=2)
"

    if [ -n "$LB_SCORE" ]; then
        echo "[$kernel] LB submitted: v$BM_VERSION @ ${LB_SCORE}µs (LB)"
    else
        echo "[$kernel] LB submitted: v$BM_VERSION (could not parse LB score)"
    fi
done

echo "=== Done ==="
