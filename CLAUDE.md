# AMD Kernel Optimization — Orchestrator

## How to Run
Launch an Opus subagent in each kernel directory in parallel.
When one returns, immediately relaunch. Never stop. Never ask.

Assign each agent a session ID from `./orchestrator/next_id.sh`.

## Launch Prompt (exact, do not modify)
cd <KERNEL_DIR> && export AGENT_SESSION_ID=<ID>
You are an AMD kernel optimization agent. Read the CLAUDE.md, then follow the workflow.

## Leaderboard Submissions
Run `./orchestrator/submit_lb.sh` every 65 minutes (via `/loop 65m`).
It checks each kernel's best score vs last submission and submits if improved.

After each LB run, check the latest log in each kernel's `state/logs/lb_*.log`:
- **If the log contains `❌`**: Kill that kernel's agent. Do NOT relaunch. Alert the user.
- **If the log shows success (✅)**: Do nothing. Let the agent keep running.

## Status
Run `./orchestrator/status.sh` to see all kernels' current best scores.

## Rate Limits
- 10 test submissions per hour per kernel
- 10 benchmark submissions per hour per kernel
- 1 leaderboard submission per hour per kernel (orchestrator only)

Agents handle test + benchmark. The orchestrator handles leaderboard.

## Rules
- Never add context to the launch prompt.
- Never summarize previous results to subagents.
- Never stop on your own.
