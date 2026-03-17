# Kernel Optimization Agent

## Workflow
1. Read `problem.md` and `submission.py`. Run `./tools/leaderboard.sh` to see the competition — know your target score and rank.
2. Form a hypothesis. Use the full extent of your abilities — read source code, clone repos, search the web, analyze ISA references, disassemble binaries, study papers, whatever it takes to find a path forward
3. `./tools/propose.sh --what "..." --why "..." --keywords "k1,k2,k3"`
   - If prior attempts found: prints them, exits 1
   - Use `--override "reason this is different"` to proceed anyway
4. Get next version: `V=$(./tools/next_version.sh)`. Edit submission.py, archive: `cp submission.py submissions/v${V}_description.py`
5. `./tools/submit.sh test`
6. Use benchmark mode as the main ranking loop when quota is available. Use leaderboard mode sparingly as the scarcest confirmation resource.
7. Only spend a leaderboard submission on the strongest current candidate. If leaderboard quota is exhausted, keep researching, benchmarking, and queueing the best next candidate instead of stopping.
8. Treat `test` as the correctness gate, `bm` as the main search loop, and `leaderboard` as confirmation.
9. Adapt to the limits reported by the tools. Do not assume fixed rates. If any quota is exhausted, switch to work that does not spend that quota.
10. After a KEEP, run `./tools/leaderboard.sh` to see if you moved up. Use the gap to #1 to guide your next hypothesis.
11. Repeat from step 2. After 5 leaderboard reverts, run `./tools/close_branch.sh`, then `exit`

## Rules
- Never edit files in `state/`. The tools do that.
- submit.sh rejects without a registered proposal.
- Read source code of the component you plan to change before implementing.
- Clone repos to `reference/cloned-repos/`. Check what's already cloned first.
- Do not let leaderboard rate limits stall the session. Use the cooldown window for research, code changes, and benchmark-driven candidate selection.
- Queue the best candidate for the next scarce leaderboard slot instead of submitting every passing variant.
- When the session is done and you run `./tools/close_branch.sh`, exit the Codex session immediately afterward.
