# Kernel Optimization Agent

## Workflow
1. Read `problem.md` and `submission.py`. Run `./tools/leaderboard.sh` to see the competition — know your target score and rank.
2. Form a hypothesis. Use the full extent of your abilities — read source code, clone repos, search the web, analyze ISA references, disassemble binaries, study papers, whatever it takes to find a path forward
3. `./tools/propose.sh --what "..." --why "..." --keywords "k1,k2,k3"`
   - If prior attempts found: prints them, exits 1
   - Use `--override "reason this is different"` to proceed anyway
4. Get next version: `V=$(./tools/next_version.sh)`. Edit submission.py, archive: `cp submission.py submissions/v${V}_description.py`
5. `./tools/submit.sh test` — this returns immediately. Do NOT wait for results.
6. `./tools/submit.sh benchmark` — also returns immediately.
7. Continue working on your next hypothesis while submissions are in flight.
8. Results arrive as messages at your prompt in the format: `[SUBMIT RESULT] v<N> <mode> <outcome>`. Act on them when they arrive.
9. Use benchmark mode as the main ranking loop. Use leaderboard mode sparingly as the scarcest confirmation resource.
10. Only spend a leaderboard submission on the strongest current candidate. If leaderboard quota is exhausted, keep researching and benchmarking.
11. Treat `test` as the correctness gate, `benchmark` as the main search loop, and `leaderboard` as confirmation.
12. Adapt to the limits reported by the tools. Do not assume fixed rates. If any quota is exhausted, switch to work that does not spend that quota.
13. After a KEEP, run `./tools/leaderboard.sh` to see if you moved up. Use the gap to #1 to guide your next hypothesis.
14. Repeat from step 2. After 5 leaderboard reverts, the session is automatically closed and your tmux session will be terminated.

## Async Submissions
- `./tools/submit.sh` queues your submission and returns instantly. Do NOT block or poll for results.
- Results are delivered to your prompt as `[SUBMIT RESULT]` messages. Process them when they arrive.
- You can have multiple submissions in flight. Keep working between submissions.
- Never call `popcorn-cli` directly — always use `./tools/submit.sh`.
- Do NOT run `tmux ls`, `tmux capture-pane`, or check background tmux sessions. Do NOT read `_submit_bg.sh` or `_process_result.py`. The delivery system is opaque — just wait for `[SUBMIT RESULT]` messages.

## Scoring
- The best score in `best.json` is a leaderboard geomean computed from **random inputs**, so it has variance run-to-run. Small differences (±1-3%) can be noise.
- Benchmark mode uses fixed inputs and is more stable. Use benchmark deltas to judge whether a change is genuinely better — don't reject a candidate just because its BM score is slightly above the stored LB best.
- Only promote to leaderboard when benchmark shows a clear, repeatable improvement.

## Rules
- Never edit files in `state/`. The tools do that.
- Before submitting a benchmark, check `state/benchmarks.jsonl` to see if this version was already benchmarked. Don't waste quota on repeats.
- submit.sh rejects without a registered proposal.
- Read source code of the component you plan to change before implementing.
- Clone repos to `reference/cloned-repos/`. Check what's already cloned first.
- Do not let leaderboard rate limits stall the session. Use the cooldown window for research, code changes, and benchmark-driven candidate selection.
- Queue the best candidate for the next scarce leaderboard slot instead of submitting every passing variant.
- When the session is done and you run `./tools/close_branch.sh`, exit the Codex session immediately afterward.
