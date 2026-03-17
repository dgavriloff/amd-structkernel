# AMD Kernel Optimization — Orchestrator

## How to Run
Launch one tmux-backed Codex subagent in each kernel directory in parallel.
Register the current orchestrator tmux session with `./orchestrator/register-orchestrator-session.sh` before launching workers.
Use `./orchestrator/launch-codex-subagent.sh --kernel <kernel>` for launches.
After each status check, run `./orchestrator/nudge-status-timer.sh` before returning to idle.
Every 10 minutes, check status. If fewer than 3 worker tmux sessions are live, relaunch the missing kernel workers. Never stop. Never ask.

The launcher assigns each agent a session ID via `./orchestrator/next_id.sh`.

## Launch Prompt (exact, do not modify)
cd <KERNEL_DIR> && export AGENT_SESSION_ID=<ID>
You are an AMD kernel optimization agent. Read the AGENTS.md, then follow the workflow.

## Status
Run `./orchestrator/status.sh` to see all kernels' current orchestrator state:
- best version
- best score
- attempts
- current agent session ID
- tmux session
- worker status
- last close result

Treat `running` as healthy.
Treat `returned`, `closed`, `stale`, or missing workers as relaunch conditions.

The status dashboard also refreshes subagent return state from kernel-local `state/subagents.jsonl`.

## Kernel Set
Launch and maintain one worker for each kernel under `./kernels/` that has an `AGENTS.md`:
- `mixed-mla`
- `moe-mxfp4`
- `mxfp4-mm`

## Process Model
The orchestrator runs inside tmux and registers its tmux session via `./orchestrator/register-orchestrator-session.sh`.
Each launched subagent runs inside its own tmux session.
Each kernel records subagent lifecycle in `state/subagents.jsonl`.
Worker tmux session existence is the source of truth for whether a worker is live.
Workers are expected to exit their own Codex session after they finish and run `./tools/close_branch.sh`.
After launching the required workers and confirming they are running, the orchestrator should run `./orchestrator/nudge-status-timer.sh` and return to an idle waiting state at the shell prompt.
It is correct for the orchestrator to do nothing between the 10-minute nudges.

## Rules
- Do not write or modify code as part of orchestration.
- Do not create new scripts, daemons, watchdogs, polling loops, or background supervisors.
- Never add context to the launch prompt.
- Never summarize previous results to subagents.
- Never stop on your own.
- Use the tmux-backed launcher, not API-spawned subagents.
- Keep exactly one live worker per kernel.
- Use the 10-minute status check plus `./orchestrator/nudge-status-timer.sh` as the relaunch mechanism.
- After relaunching a missing worker and confirming it is running, return to idle again.
