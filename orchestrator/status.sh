#!/bin/bash
# Print status dashboard for all kernels.
# Usage: ./orchestrator/status.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                              Kernel Optimization Orchestrator Status                                 ║"
echo "╠══════════════╦══════════╦═════════╦═══════════════╦═══════╦══════════════════════════╦══════════╦════════════╣"
echo "║ Kernel       ║ Version  ║ Score   ║ Attempts      ║ Agent ║ Tmux                     ║ Worker   ║ Close      ║"
echo "╠══════════════╬══════════╬═════════╬═══════════════╬═══════╬══════════════════════════╬══════════╬════════════╣"

for kernel_dir in "$REPO_DIR"/kernels/*/; do
    kernel=$(basename "$kernel_dir")
    best_file="$kernel_dir/state/best.json"
    tried_file="$kernel_dir/state/tried.jsonl"
    "$REPO_DIR/orchestrator/refresh-codex-subagent-state.sh" --kernel "$kernel" >/dev/null 2>&1 || true

    if [ -f "$best_file" ]; then
        version=$(python3 -c "import json; print(json.load(open('$best_file'))['version'])")
        score=$(python3 -c "import json; print(json.load(open('$best_file'))['score'])")
    else
        version="—"
        score="—"
    fi

    if [ -f "$tried_file" ]; then
        attempts=$(wc -l < "$tried_file" | xargs)
    else
        attempts="0"
    fi

    subagent_info=$(python3 - "$kernel_dir/state/subagents.jsonl" <<'PY'
import json
import sys

path = sys.argv[1]
launch = None
last_event = None

try:
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            last_event = event
            if event.get("action") == "launch":
                launch = event
except FileNotFoundError:
    pass

agent = "—"
tmux = "—"
last_action = "—"
close_result = "—"

if launch:
    agent = str(launch.get("session", "—"))
    tmux = launch.get("tmux_session", "—")

if last_event:
    last_action = last_event.get("action", "—")
    if last_action == "close":
        if last_event.get("tmux_killed"):
            close_result = "killed"
        else:
            reason = last_event.get("reason", "")
            if reason == "tmux session not found":
                close_result = "not-found"
            elif reason == "no launch record":
                close_result = "no-record"
            else:
                close_result = "failed"

print("\t".join([agent, tmux, last_action, close_result]))
PY
)
    IFS=$'\t' read -r agent tmux last_action close_result <<< "$subagent_info"

    worker="idle"
    if [ "$tmux" != "—" ]; then
        if tmux has-session -t "$tmux" 2>/dev/null; then
            worker="running"
        elif [ "$last_action" = "close" ]; then
            worker="closed"
        elif [ "$last_action" = "return" ]; then
            worker="returned"
        else
            worker="stale"
        fi
    fi

    score_display="—"
    if [ "$score" != "—" ]; then
        score_display="${score}µs"
    fi

    printf "║ %-12s ║ v%-7s ║ %-7s ║ %-13s ║ %-5s ║ %-24s ║ %-8s ║ %-10s ║\n" \
        "$kernel" "$version" "$score_display" "$attempts tried" "$agent" "$tmux" "$worker" "$close_result"
done

echo "╚══════════════╩══════════╩═════════╩═══════════════╩═══════╩══════════════════════════╩══════════╩════════════╝"
