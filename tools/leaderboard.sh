#!/bin/bash
# Fetch leaderboard standings for the current kernel.
# Usage: ./tools/leaderboard.sh [--top N]
#   --top N   Show top N entries (default: 5)
#
# Reads the leaderboard name from submission.py header (#!POPCORN leaderboard ...)
# and maps it to the public leaderboard at leaderboard.ooousay.com.
# Compares your best score (from state/best.json) against the leaderboard.
#
# Output: top N scores, your rank estimate, and gap to #1.

set -euo pipefail

# --- Parse args ---
TOP=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --top) TOP="$2"; shift 2 ;;
    *) echo "Usage: ./tools/leaderboard.sh [--top N]"; exit 1 ;;
  esac
done

# --- Resolve kernel dir ---
if [[ ! -f "submission.py" ]]; then
  echo "ERROR: No submission.py in current directory." >&2
  exit 1
fi

# --- Extract leaderboard name from submission.py ---
LEADERBOARD=$(grep -m1 '#!POPCORN leaderboard' submission.py | awk '{print $3}')
if [[ -z "$LEADERBOARD" ]]; then
  echo "ERROR: No '#!POPCORN leaderboard' header in submission.py" >&2
  exit 1
fi

# --- Map popcorn leaderboard name to display name ---
case "$LEADERBOARD" in
  amd-mxfp4-mm)  DISPLAY_NAME="MXFP4 GEMM" ;;
  amd-mixed-mla) DISPLAY_NAME="MLA Decode" ;;
  amd-moe-mxfp4) DISPLAY_NAME="MXFP4 MoE" ;;
  *) echo "ERROR: Unknown leaderboard '$LEADERBOARD'." >&2; exit 1 ;;
esac

# --- Read our best score ---
BEST_SCORE=""
BEST_VERSION=""
if [[ -f "state/best.json" ]]; then
  BEST_SCORE=$(python3 -c "import json; d=json.load(open('state/best.json')); print(d.get('score',''))" 2>/dev/null || true)
  BEST_VERSION=$(python3 -c "import json; d=json.load(open('state/best.json')); print(d.get('version',''))" 2>/dev/null || true)
fi

# --- Fetch and parse ---
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT
curl -s --max-time 15 "https://leaderboard.ooousay.com/" > "$TMPFILE"

if [[ ! -s "$TMPFILE" ]]; then
  echo "ERROR: Failed to fetch leaderboard page." >&2
  exit 1
fi

export DISPLAY_NAME TOP BEST_SCORE BEST_VERSION TMPFILE
python3 << 'PYEOF'
import sys, json, re, os

display_name = os.environ["DISPLAY_NAME"]
top_n = int(os.environ["TOP"])
best_score_str = os.environ.get("BEST_SCORE", "")
best_version = os.environ.get("BEST_VERSION", "")

best_score = float(best_score_str) if best_score_str else None

with open(os.environ["TMPFILE"]) as f:
    html = f.read()

# The page contains escaped JSON like: \"name\":\"MXFP4 GEMM\",...,\"entries\":[...]
# First unescape the backslash-quotes so we can search normally
text = html.replace('\\"', '"')

# Find our leaderboard's entries array
needle = f'"name":"{display_name}"'
idx = text.find(needle)
if idx == -1:
    print(f'ERROR: Could not find "{display_name}" in leaderboard page.', file=sys.stderr)
    sys.exit(1)

# Find "entries":[ after the name
entries_marker = '"entries":['
eidx = text.find(entries_marker, idx)
if eidx == -1:
    print("ERROR: Could not find entries array.", file=sys.stderr)
    sys.exit(1)

# Extract the JSON array by bracket matching
arr_start = eidx + len(entries_marker) - 1  # points to '['
depth = 0
arr_end = arr_start
for i in range(arr_start, min(arr_start + 500000, len(text))):
    if text[i] == '[':
        depth += 1
    elif text[i] == ']':
        depth -= 1
        if depth == 0:
            arr_end = i + 1
            break

entries = json.loads(text[arr_start:arr_end])
# Filter out scores below 5µs (likely cheats or test artifacts)
entries = [e for e in entries if e.get("score", 0) >= 0.000005]
entries.sort(key=lambda x: x.get("score", float("inf")))

# Print table
w_name = 23
print(f"╔{'═' * 58}╗")
pad = 44 - len(display_name)
print(f"║  {display_name} Leaderboard{' ' * pad}║")
print(f"╠{'═' * 4}╦{'═' * 25}╦{'═' * 14}╦{'═' * 11}╣")
print(f"║ #  ║ {'User':<{w_name}} ║ {'Score':>12} ║ {'Subs':>9} ║")
print(f"╠{'═' * 4}╬{'═' * 25}╬{'═' * 14}╬{'═' * 11}╣")

for i, entry in enumerate(entries[:top_n]):
    rank = i + 1
    user = entry.get("user_name", "???")[:w_name]
    score = entry.get("score", 0)
    subs = entry.get("submission_count", "?")
    score_us = score * 1_000_000
    print(f"║ {rank:<2} ║ {user:<{w_name}} ║ {score_us:>9.2f} µs ║ {subs:>9} ║")

print(f"╚{'═' * 4}╩{'═' * 25}╩{'═' * 14}╩{'═' * 11}╝")

# Compare to our best
if best_score is not None:
    leader_score_us = entries[0].get("score", 0) * 1_000_000
    gap = ((best_score - leader_score_us) / leader_score_us) * 100 if leader_score_us > 0 else 0

    our_rank = 1
    best_in_seconds = best_score / 1_000_000
    for entry in entries:
        if entry.get("score", 0) < best_in_seconds:
            our_rank += 1
        else:
            break
    total = len(entries)

    print()
    print(f"  Your best: v{best_version} @ {best_score:.2f} µs")
    leader_name = entries[0].get("user_name", "?")
    print(f"  Leader:    {leader_name} @ {leader_score_us:.2f} µs")
    if gap > 0:
        print(f"  Gap:       +{gap:.1f}% behind #1")
    elif gap < 0:
        print(f"  Gap:       {gap:.1f}% AHEAD of #1!")
    else:
        print(f"  Gap:       TIED with #1!")
    print(f"  Est. rank: #{our_rank} of {total}")
else:
    print()
    print("  No local best.json found — run submit.sh leaderboard first.")
PYEOF
