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

    # Save raw output log
    LB_LOGS_DIR="$kernel_dir/state/logs"
    mkdir -p "$LB_LOGS_DIR"
    echo "$RESULT" > "$LB_LOGS_DIR/lb_v${BM_VERSION}_$(date -u '+%Y%m%d_%H%M%S').log"

    # Check for failure
    if echo "$RESULT" | grep -qiE "Application error:|Rate limit exceeded|❌ Benchmarking failed|❌ Testing failed|❌ Leaderboard run failed|Failed to trigger"; then
        echo "[$kernel] LB submission FAILED — will retry next cycle"
        continue
    fi

    # Success — mark this version as submitted (no score parsing, leaderboard.sh reads the website)
    python3 -c "
import json, datetime
lb = json.load(open('$LB_FILE'))
lb['$kernel'] = {
    'version': $BM_VERSION,
    'bm_score': float('$BM_SCORE'),
    'ts': datetime.datetime.now(datetime.UTC).isoformat() + 'Z'
}
json.dump(lb, open('$LB_FILE', 'w'), indent=2)
"
    echo "[$kernel] LB submitted: v$BM_VERSION"
done

echo "=== Done ==="
