# AMD Kernel Optimization — Orchestrator

## Kernels
Three independent kernel challenges. Each has its own CLAUDE.md with full instructions.

| Kernel | Directory | Leaderboard | Status |
|--------|-----------|-------------|--------|
| MXFP4 MatMul | `kernels/mxfp4-mm/` | `amd-mxfp4-mm` | v077 @ 9.76µs |
| MLA Decode | `kernels/mixed-mla/` | `amd-mixed-mla` | v001 baseline |
| MoE MXFP4 | `kernels/moe-mxfp4/` | `amd-moe-mxfp4` | v001 @ 176.4µs |

## How to Run
Launch an Opus subagent in each kernel directory in parallel. When one returns, immediately relaunch in the same directory. Keep cycling until told to stop. **Never stop on your own. Never ask if you should continue.**

**Use this exact launch prompt every time — do not modify it, do not add context, do not summarize previous results:**

```
You are an AMD kernel optimization agent. Your working directory is <KERNEL_DIR>.
Read the CLAUDE.md and results.md in that directory to understand the current state, then execute the optimization workflow autonomously.
```

**Never add commentary, warnings, or summaries of past agent results to the launch prompt.** The agent gets all context from results.md. Front-loading "everything has been tried" or "the only remaining path is X" biases the agent and prevents it from forming its own hypotheses.

**Do not save any memories.** All state lives in results.md and git.

## Repo Structure
- `reference/` — Reference implementations and eval harness. Read-only, do not modify.
- `reference/cloned-repos/` — Cloned external repos for analysis. Gitignored.
- `kernels/` — Our optimized kernel submissions.

## Ground Rules
- **No harness hacks.** Every optimization must be a legitimate kernel improvement.
- Forbidden: precomputed answers, caching across calls, seed/shape exploits, harness manipulation, loose-tolerance tricks.
- Allowed: faster APIs, kernel fusion, custom HIP kernels, memory layout optimizations, hardware-specific tuning.
