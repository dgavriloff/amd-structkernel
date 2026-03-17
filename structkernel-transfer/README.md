# amd-structkernel Transfer Package

This zip contains everything needed to populate the amd-structkernel repo with real data
from the amd-autokernel optimization campaign. An agent should use this to:

1. Create `problem.md` for each kernel from the CLAUDE.md problem descriptions
2. Populate `state/tried.jsonl` from results.md history
3. Populate `state/dead.jsonl` from exhausted branches
4. Populate `state/best.json` with current best scores
5. Copy submission.py and best_submission.py into each kernel dir

## Contents

### kernel-state/
- `{kernel}/results.md` — full experiment history (the source of truth for migration)
- `{kernel}/CLAUDE.md` — current kernel instructions (extract problem.md from these)
- `{kernel}/submission.py` — current working submission
- `{kernel}/best_submission.py` — current global best

### analysis/
- `audit.md` — full audit of the optimization loop (chat log analysis, agent behavior, issues found)
- `kernelbot-findings.md` — what we learned from the kernelbot eval harness source
- `mla-mxfp4-opportunity.md` — the untapped MXFP4 KV opportunity for mixed-mla
- `leaderboard-gaps.md` — current scores vs #1, what's needed

### reference/
- `eval.py` — the actual eval harness (from kernelbot)
- `run_eval.py` — how the runner loads and times submissions

### source-files/
- The amd-structkernel repo itself (scripts, CLAUDE.md files, directory structure)

## Migration Instructions

For each kernel, the agent should:

1. Extract the problem description from `kernel-state/{kernel}/CLAUDE.md` into `problem.md`
   - What the kernel does
   - Input format
   - Benchmark shapes
   - Reference performance
   - Key optimization opportunities
   - BM vs LB differences
   - Server constraints
   - Reference material pointers

2. Parse `kernel-state/{kernel}/results.md` to populate `state/tried.jsonl`
   - Each version attempt becomes one JSONL entry
   - Extract: version number, what was changed, keywords, score, kept/reverted, reason

3. Extract dead techniques from exhausted branches in results.md → `state/dead.jsonl`

4. Set `state/best.json` from the current best scores:
   - mxfp4-mm: v188 @ 9.01µs
   - mixed-mla: v057 @ 28.8µs (LB) / 26.5µs (BM)
   - moe-mxfp4: v106 @ ~139µs (LB)

5. Copy submission.py and best_submission.py into each kernel dir
