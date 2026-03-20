# AMD Kernel Optimization — Orchestrator

## How to Run
Launch an Opus subagent in each kernel directory in parallel.
When one returns, immediately relaunch. Never stop. Never ask.

Assign each agent a session ID from `./orchestrator/next_id.sh`.

## Launch Prompt (exact, do not modify)
cd <KERNEL_DIR> && export AGENT_SESSION_ID=<ID>
You are an AMD kernel optimization agent. Read the CLAUDE.md, then follow the workflow.

## Leaderboard Submissions
Run `./orchestrator/submit_lb.sh` hourly (via `/loop 60m`).
It checks each kernel's best BM score vs last LB submission and submits if improved.

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
