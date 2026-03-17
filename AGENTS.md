# AMD Kernel Optimization — Orchestrator

## How to Run
Launch a subagent (the most powerful model you can) in each kernel directory in parallel.
When one returns, immediately relaunch. Never stop. Never ask.

Assign each agent a session ID from `./orchestrator/next_id.sh`.

## Launch Prompt (exact, do not modify)
cd <KERNEL_DIR> && export AGENT_SESSION_ID=<ID>
You are an AMD kernel optimization agent. Read the AGENTS.md, then follow the workflow.

## Status
Run `./orchestrator/status.sh` to see all kernels' current best scores.

## Rules
- Never add context to the launch prompt.
- Never summarize previous results to subagents.
- Never stop on your own.
