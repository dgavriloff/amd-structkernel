#!/bin/bash
# Print status dashboard for all kernels.
# Usage: ./orchestrator/status.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "╔══════════════════════════════════════════════════╗"
echo "║           Kernel Optimization Status             ║"
echo "╠══════════════╦══════════╦════════╦═══════════════╣"
echo "║ Kernel       ║ Version  ║ Score  ║ Attempts      ║"
echo "╠══════════════╬══════════╬════════╬═══════════════╣"

for kernel_dir in "$REPO_DIR"/kernels/*/; do
    kernel=$(basename "$kernel_dir")
    best_file="$kernel_dir/state/best.json"
    tried_file="$kernel_dir/state/tried.jsonl"

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

    printf "║ %-12s ║ v%-7s ║ %5sµs ║ %-13s ║\n" "$kernel" "$version" "$score" "$attempts tried"
done

echo "╚══════════════╩══════════╩════════╩═══════════════╝"
