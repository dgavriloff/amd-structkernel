# Kernel Optimization Agent

## HARNESS CHANGE (2026-03-19)
The organizers updated aiter on the server. ALL old submissions (v1-v89) that use
`mla_decode_fwd` from `aiter.mla` are BROKEN — `mla_decode_stage1_asm_fwd` dropped
the `lse` output parameter, shifting q_scale/kv_scale positions → garbage results.

- DO NOT revert to any old submission from submissions/ — they will all fail.
- The current best_submission.py (v091) uses the new two-stage API directly
  (`mla_decode_stage1_asm_fwd` + `mla_reduce_v1`) and passes all tests.
- page_size>1 is broken with the new aiter. Use page_size=1 for all paths.
- The old tried.jsonl scores (v1-v89) are from the old harness and not comparable.

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

## Rules
- Never edit files in `state/`. The tools do that.
- submit.sh rejects without a registered proposal.
- Read source code of the component you plan to change before implementing.
- Clone repos to `reference/cloned-repos/`. Check what's already cloned first.
