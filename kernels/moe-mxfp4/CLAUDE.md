# Kernel Optimization Agent

## Workflow
1. Read `problem.md` and `submission.py`. Run `./tools/leaderboard.sh` to see the competition — know your target score and rank.
2. Form a hypothesis. Use the full extent of your abilities — read source code, clone repos, search the web, analyze ISA references, disassemble binaries, study papers, whatever it takes to find a path forward
3. `./tools/propose.sh --what "..." --why "..." --keywords "k1,k2,k3"`
   - If prior attempts found: prints them, exits 1
   - Use `--override "reason this is different"` to proceed anyway
4. Get next version: `V=$(./tools/next_version.sh)`. Edit submission.py, archive: `cp submission.py submissions/v${V}_description.py`
5. `./tools/submit.sh test`
6. `./tools/submit.sh benchmark`
7. Output says KEEP or REVERT. Follow it.
8. After a KEEP, run `./tools/leaderboard.sh` to check ranking. LB submissions are handled hourly by the orchestrator.
9. Repeat from step 2, or if 5 reverts: `./tools/close_branch.sh`

## BM vs LB — Read This Before Optimizing
Your benchmark (BM) score is NOT your leaderboard (LB) score. See problem.md for the full table, but the critical differences are:
- **LB uses new random data each iteration** (`recheck=True`, seed increments by +13). BM reuses the same input every iteration.
- **LB runs `check_implementation()` every iteration**, included in timing. BM checks once at start.
- **LB has 1ms warmup** (vs 10ms), **100 repeats** (vs 1000).

**What this means**: any optimization that exploits fixed/repeated inputs will show a BM improvement but fail or regress on LB. This includes:
- CUDA graphs (replay fixed execution plan — breaks with changing inputs)
- Input-dependent caching or precomputation that assumes data doesn't change between iterations
- Warmup-dependent tricks (JIT compilation, lazy init) — LB has minimal warmup

LB submissions are **1 per hour** — don't waste them. Only optimize in ways that are correct for arbitrary inputs on every call.

## Rules
- Never edit files in `state/`. The tools do that.
- submit.sh rejects without a registered proposal.
- Read source code of the component you plan to change before implementing.
- Clone repos to `reference/cloned-repos/`. Check what's already cloned first.
