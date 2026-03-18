# AMD Kernel Optimization — Orchestrator

## How to Run
1. Launch one Codex worker per kernel using `./orchestrator/launch-codex-subagent.sh --kernel <name>` — launch all three in parallel.
2. Confirm workers are running via `./orchestrator/status.sh`.
3. Start `/loop 10m` to check status, relaunch dead workers, and nudge idle workers every 10 minutes.

## Kernel Set
- `mixed-mla`
- `moe-mxfp4`
- `mxfp4-mm`

## Launch
Use `./orchestrator/launch-codex-subagent.sh --kernel <name>` for all launches.
The launcher handles session IDs, tmux sessions, env vars, and prompt delivery.

## Status
Run `./orchestrator/status.sh` to see all kernels' current state:
- best version, score, attempts
- current agent session ID and tmux session
- worker status, last close result, pane tail

Treat `running` as healthy.
Treat `returned`, `closed`, `stale`, or missing workers as relaunch conditions.

## Process Model
- Each worker runs in its own tmux session via `launch-codex-subagent.sh`.
- Each kernel tracks lifecycle in `state/subagents.jsonl`.
- Worker tmux session existence is the source of truth for liveness.
- Workers exit their own Codex session after running `./tools/close_branch.sh`.
- Keep exactly one live worker per kernel (3 total).

## Rules
- Do not write or modify code as part of orchestration.
- Never add context to the launch prompt.
- Never summarize previous results to subagents.
- Never stop on your own.
- After relaunching a missing worker and confirming it is running, return to idle.
