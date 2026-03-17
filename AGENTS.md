# AMD Kernel Optimization — Orchestrator

## How to Run
Launch one tmux-backed Codex subagent in each kernel directory in parallel.
Register the current orchestrator tmux session with `./orchestrator/register-orchestrator-session.sh` before launching workers.
Use `./orchestrator/launch-codex-subagent.sh --kernel <kernel>` for launches.
When one returns or closes, immediately relaunch. Never stop. Never ask.

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
The orchestrator itself runs inside tmux and publishes its tmux session name via `./orchestrator/register-orchestrator-session.sh`.
Each launched subagent runs inside its own tmux session.
Each kernel records subagent lifecycle in `state/subagents.jsonl`.
When a kernel agent closes its session, `./tools/close_branch.sh` will trigger tmux cleanup for that agent session.
After cleanup, the helper will send a notification into the orchestrator tmux session that a subagent finished.

## Rules
- Never add context to the launch prompt.
- Never summarize previous results to subagents.
- Never stop on your own.
- Use the tmux-backed launcher, not API-spawned subagents.
- Keep exactly one live worker per kernel.
- Treat tmux notifications about finished subagents as signals to reconcile and relaunch the affected kernel.
