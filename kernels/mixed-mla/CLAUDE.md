# Kernel Optimization Agent

## Workflow
1. Read `problem.md` and `submission.py`. Run `./tools/leaderboard.sh` to see the competition — know your target score and rank.
2. Form a hypothesis. Use the full extent of your abilities — read source code, clone repos, search the web, analyze ISA references, disassemble binaries, study papers, whatever it takes to find a path forward. I want to emphasize that you should be cloning repos and then researching them for functions/libraries you dont completely understand. Especially if theres a chance it can be manually rewritten to be faster for our purposes.
3. `./tools/propose.sh --what "..." --why "..." --keywords "k1,k2,k3"`
   - If prior attempts found: prints them, exits 1
   - Use `--override "reason this is different"` to proceed anyway
4. Get next version: `V=$(./tools/next_version.sh)`. Edit submission.py, archive: `cp submission.py submissions/v${V}_description.py`
5. `./tools/submit.sh test`
6. `./tools/submit.sh benchmark`
7. Output says KEEP or REVERT. Follow it.
9. Repeat from step 2, or if 5 reverts: `./tools/close_branch.sh`

## Pre-Submission Checklist

Answer NO to all of these or your change will fail/regress on LB:

### Will it break?
- [ ] **CUDA graphs?**
- [ ] **Caching tensors the kernel WRITES to?** **When in doubt, allocate fresh.**
- [ ] **Caching anything derived from input data (not shape/config)?**

### Will it regress?
- [ ] **Improvement depends on same data running repeatedly?** — Cache line reuse, TLB warmth, allocator patterns all reset with new data. But this is usually small (1-5%).
- [ ] **Relies on 100ms warmup?** — LB gets 10ms. Triton JIT and autotune need warmup; aiter ASM kernels don't.

### Quick self-test
Imagine calling `custom_kernel()` 100 times with different random data each time. Is every output correct and roughly the same speed?

## Rules
- Never edit files in `state/`. The tools do that.
- submit.sh rejects without a registered proposal.
- Read source code of the component you plan to change before implementing.
- Clone repos to `reference/cloned-repos/`. Check what's already cloned first.
