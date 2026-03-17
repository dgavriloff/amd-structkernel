# Kernel Optimization Agent

## Workflow
1. Read `problem.md` and `submission.py`
2. Form a hypothesis. Use the full extent of your abilities — read source code, clone repos, search the web, analyze ISA references, disassemble binaries, study papers, whatever it takes to find a path forward
3. `./tools/propose.sh --what "..." --why "..." --keywords "k1,k2,k3"`
   - If prior attempts found: prints them, exits 1
   - Use `--override "reason this is different"` to proceed anyway
4. Get next version: `V=$(./tools/next_version.sh)`. Edit submission.py, archive: `cp submission.py submissions/v${V}_description.py`
5. `./tools/submit.sh test`
6. `./tools/submit.sh leaderboard`
7. Output says KEEP or REVERT. Follow it.
8. Repeat from step 2, or if 5 reverts: `./tools/close_branch.sh`

## Rules
- Never edit files in `state/`. The tools do that.
- submit.sh rejects without a registered proposal.
- Read source code of the component you plan to change before implementing.
- Clone repos to `reference/cloned-repos/`. Check what's already cloned first.
