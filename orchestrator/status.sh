#!/bin/bash
# Print status dashboard for all kernels.
# Usage: ./orchestrator/status.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

preview_tail() {
    local tmux_session="$1"
    local captured
    captured=$(tmux capture-pane -pJ -t "$tmux_session" -S -60 2>/dev/null) || true
    python3 -c '
import sys, re

raw = sys.argv[1].splitlines()

chrome_re = re.compile(
    r"^›\s|^\s*(gpt-|claude|sonnet|opus|haiku)|^\s*~/|^Find and fix|^\d+% left|esc to interrupt|^• (Working|Waiting)",
    re.IGNORECASE,
)

lines = [l.rstrip() for l in raw]
while lines and (not lines[-1].strip() or chrome_re.search(lines[-1])):
    lines.pop()

lines = lines[-6:]

if not lines:
    print("\u2014")
else:
    joined = " | ".join(line.strip() for line in lines if line.strip())
    joined = " ".join(joined.split())
    if len(joined) > 80:
        joined = joined[:77] + "..."
    print(joined if joined else "\u2014")
' "$captured"
}

is_agent_idle() {
    local tmux_session="$1"
    local pane
    pane=$(tmux capture-pane -pJ -t "$tmux_session" -S -5 2>/dev/null) || return 1
    if echo "$pane" | grep -qE '^\s*›' && ! echo "$pane" | grep -qE 'Working|Waiting'; then
        return 0
    fi
    return 1
}

count_bg_jobs() {
    local kernel="$1"
    tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -c "^bg-${kernel}-" || echo "0"
}

list_bg_jobs() {
    local kernel="$1"
    tmux list-sessions -F '#{session_name}' 2>/dev/null | grep "^bg-${kernel}-" | sed "s/^bg-${kernel}-//" | tr '\n' ' ' || true
}

echo "╔══════════════╦══════════╦═════════╦═══════════════╦═══════╦══════════════════════════╦══════════╦════════╦═══════════════════════════════════════════════════════════════════╗"
echo "║ Kernel       ║ Version  ║ Score   ║ Attempts      ║ Agent ║ Tmux                     ║ Worker   ║ Queue  ║ Pane Tail                                                         ║"
echo "╠══════════════╬══════════╬═════════╬═══════════════╬═══════╬══════════════════════════╬══════════╬════════╬═══════════════════════════════════════════════════════════════════╣"

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
    preview="—"
    bg_count=$(count_bg_jobs "$kernel")
    if [ "$tmux" != "—" ]; then
        if tmux has-session -t "$tmux" 2>/dev/null; then
            worker="running"
            preview=$(preview_tail "$tmux")
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

    queue_display="${bg_count} jobs"

    printf "║ %-12s ║ v%-7s ║ %-7s ║ %-13s ║ %-5s ║ %-24s ║ %-8s ║ %-6s ║ %-65s ║\n" \
        "$kernel" "$version" "$score_display" "$attempts tried" "$agent" "$tmux" "$worker" "$queue_display" "$preview"
done

echo "╚══════════════╩══════════╩═════════╩═══════════════╩═══════╩══════════════════════════╩══════════╩════════╩═══════════════════════════════════════════════════════════════════╝"
