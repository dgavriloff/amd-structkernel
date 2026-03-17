# AMD Kernel Optimization — Orchestrator

## How to Run
Launch an Opus subagent in each kernel directory in parallel.
When one returns, immediately relaunch. Never stop. Never ask.

Assign each agent a session ID from `./orchestrator/next_id.sh`.

## Launch Prompt (exact, do not modify)
You are an AMD kernel optimization agent. Your working directory is <KERNEL_DIR>.
Your session ID is <ID>. Run `export AGENT_SESSION_ID=<ID>` first, then read the CLAUDE.md and follow the workflow.

## Status
Run `./orchestrator/status.sh` to see all kernels' current best scores.

## Rules
- Never add context to the launch prompt.
- Never summarize previous results to subagents.
- Never stop on your own.
