# MOE-MXFP4 Kernel — MI355X

## What This Kernel Does
DeepSeek-R1 style MXFP4 Mixture-of-Experts fused kernel on AMD Instinct MI355X.
Flow per token: `bf16 hidden` -> `MXFP4 quant` -> `gate_up GEMM` -> `SwiGLU` -> `down GEMM` -> `weighted reduce across top-k experts` -> `bf16 output`

Two-stage fused pipeline:
1. **Stage 1**: Quant activations + gate_up GEMM (a4w4) + SwiGLU activation
2. **Stage 2**: Down GEMM (a4w4) + weighted expert reduction

## Architecture
- DeepSeek-R1 MoE: 256 routed experts + 1 shared expert = 257 total
- Top-8 routed + 1 shared = 9 experts per token
- d_hidden=7168, d_expert varies (256 for TP=8, 512 for TP=4, 2048 for EP-on)
- Shared expert always selected with weight=1.0
- All GEMMs are MXFP4 (a4w4) with per-1x32 block scaling

## Input Format
Large tuple — see submission.py docstring for full details. Key tensors:
- `hidden_states`: [M, d_hidden] bf16 — input activations
- `gate_up_weight_shuffled`: [E, 2*d_expert_pad, d_hidden_pad//2] fp4x2 — pre-shuffled for CK
- `down_weight_shuffled`: [E, d_hidden_pad, d_expert_pad//2] fp4x2 — pre-shuffled for CK
- `gate_up_weight_scale_shuffled` / `down_weight_scale_shuffled`: e8m0 scales
- `topk_weights`: [M, total_top_k] float32 — routing weights
- `topk_ids`: [M, total_top_k] int32 — expert indices
- `config`: dict with dimensions

Raw (un-shuffled) weights and scales also provided for custom kernel implementations.

Output: [M, d_hidden] bfloat16

## Benchmark Shapes
| bs | E | d_hidden | d_expert | top_k | Config |
|---|---|---|---|---|---|
| 16 | 257 | 7168 | 256 | 9 | TP=8 |
| 128 | 257 | 7168 | 256 | 9 | TP=8 |
| 512 | 257 | 7168 | 256 | 9 | TP=8 |
| 16 | 33 | 7168 | 512 | 9 | TP=4 |
| 128 | 33 | 7168 | 512 | 9 | TP=4 |
| 512 | 33 | 7168 | 512 | 9 | TP=4 |
| 512 | 33 | 7168 | 2048 | 9 | EP-on |

Ranking by geometric mean. Tolerance: rtol=1e-2, atol=1e-2.

## Reference Performance (aiter fused_moe)
| bs | E | d_hidden | d_expert | top_k | time (µs) |
|---|---|---|---|---|---|
| 16 | 257 | 7168 | 256 | 9 | 152.7 |
| 128 | 257 | 7168 | 256 | 9 | 239.0 |
| 512 | 257 | 7168 | 256 | 9 | 336.5 |
| 16 | 33 | 7168 | 512 | 9 | 106.2 |
| 128 | 33 | 7168 | 512 | 9 | 141.1 |
| 512 | 33 | 7168 | 512 | 9 | 225.0 |
| 512 | 33 | 7168 | 2048 | 9 | 380.4 |

## Key Optimization Opportunities
1. **Custom tiling / scheduling** — CK kernel uses fixed tile strategy. Small batch or skewed expert distributions may benefit from custom schedules.
2. **Activation quant fusion** — Fuse dynamic MXFP4 quant into Stage 1 GEMM prologue to save a global memory round-trip.
3. **Inter-stage fusion** — Fuse Stage 1 + Stage 2 into single kernel to eliminate intermediate buffer write/read.
4. **Expert-parallel wave scheduling** — 257 experts but only 9 active per token. Work-stealing or compact-dispatch to minimize wasted waves.
5. **Shared expert fusion** — Shared expert processes ALL tokens unconditionally. Could be a dense GEMM fused with routed reduction.
6. **Split-K for large M** — bs=512 with d_expert=2048 has large enough GEMMs for split-K.

## BM vs LB Differences
BM and LB use different eval parameters (see `reference/eval.py`):

| | Benchmark | Leaderboard |
|---|---|---|
| **recheck** | `False` — same input every iteration | `True` — new seed (+13) per iteration |
| **max_repeats** | 1000 | 100 |
| **time limit** | 50s | 30s |
| **warmup** | 10ms | 1ms |
| **correctness checks** | once at start | **every iteration, included in timing** |

LB times include `check_implementation()` overhead on every iteration. LB uses different random data each iteration (`recheck=True`). LB has 10x fewer samples and 10x shorter warmup.

## Status & Results
See `results.md` for current status, benchmark numbers, and experiment history.

## Tools

### Submitting
```
../../submit.sh kernels/moe-mxfp4 <MODE>
```
Modes: `test` (correctness), `benchmark` (timing), `leaderboard` (ranked submission).

**Always use submit.sh. Never call popcorn-cli directly. Never use run_in_background.**

### Gemini CLI (optional)
Second set of eyes on local files. Claude makes all decisions. Do NOT ask Gemini to search the web or fetch URLs — it loops. Only feed it local content via `cat`.
```
gemini -p "your prompt $(cat local_file.py)"
```
If rate-limited (429), use flash model:
```
gemini --model "gemini-3-flash-preview" -p "your prompt $(cat local_file.py)"
```

For external code research, clone repos to `reference/cloned-repos/` and use Gemini to analyze them.

### Reference Material for Novel Approaches
`reference/cloned-repos/` contains aiter, ck, and popcorn-cli source. `reference/amd-instinct-cdna4-isa.pdf` is the CDNA4 ISA reference for gfx950 instructions.

**HIP kernel references in aiter:**
- `csrc/kernels/quant_kernels.cu` — FP4 group quantization kernel with e8m0 scales, scale shuffling, `fp4x2_t`
- `csrc/include/ck_tile/vec_convert.h` — GFX950 inline assembly: `v_cvt_scalef32_pk_fp4_bf16`
- `csrc/kernels/mla/hk/` — HIP MLA decode kernels (buffer management, softmax)
- `csrc/ck_tile_gemm_moe_2stages/` — CK two-stage MoE implementation

**Triton kernel source in aiter:**
- `aiter/ops/triton/_triton_kernels/attention/mla_decode_rope.py` — MLA decode Triton kernel
- `aiter/ops/triton/_triton_kernels/moe/moe_op_mxfp4_silu_fused.py` — MXFP4 MoE with SiLU fusion
- `aiter/ops/flydsl/kernels/moe_gemm_2stage.py` — FlyDSL two-stage MoE kernel definitions

**tl.inline_asm_elementwise** — Inject custom AMDGCN ISA into Triton kernels. Working example:
- `aiter/ops/triton/_triton_kernels/attention/pod_attention.py` — uses `tl.inline_asm_elementwise` for hardware register reads

**Tuning configs:**
- `aiter/hsa/gfx950/mla/mla_asm.csv` — GFX950 MLA autotuned params
- `aiter/configs/tuned_fmoe.csv` — MoE tuned configs

**CDNA4 ISA** — `reference/amd-instinct-cdna4-isa.pdf` documents all gfx950 instructions including `v_cvt_scalef32_pk_fp4_*` for hardware FP4 conversion.

**Eval harness** — `reference/cloned-repos/kernelbot/` — how submissions are loaded, timed, and constrained.

## Files
- `submission.py` — current working submission
- `best_submission.py` — global best across all branches
- `submissions/` — archive of every version
- `results.md` — experiment log. **Read the optimization tree first** to understand the current architecture and how many failed attempts exist at each level.

## Optimization Workflow
1. **Check results.md** — Read the optimization tree and recent branch logs. Understand what's been tried and what the current per-shape breakdown looks like.
2. **Research** — Read the source code of the component you plan to change before implementing. Understanding the hardware-level mechanism (memory bandwidth, occupancy, instruction mix) makes hypotheses more likely to land. Clone relevant source repos to `reference/cloned-repos/` if needed. Direct code reading for small files.
   - Research is most valuable when the hot path uses opaque or precompiled components — ASM binaries, closed-source library functions, or obscure APIs whose internals aren't obvious. Understanding exactly how these work can reveal optimization opportunities specific to this problem. For example: reading the ASM GEMM kernel source to understand its tile scheduling, or reading the Triton compiler backend to understand launch overhead.
   - When reading source code doesn't reveal new approaches, search for academic papers on the kernel's bottleneck. Focus on papers with low-level GPU algorithms, code, or benchmarks. Implement findings, don't just summarize.
3. **Hypothesize** — Check the "Untried Directions" section of the most recent exhausted branch in results.md. **You must attempt the first item on that list before exploring your own ideas.** If you cannot implement it (blocked by server, missing infrastructure, etc.) or it regresses, document why in results.md, remove it from the Untried Directions list, commit, and end your session. The next agent will attempt the next item. Only after the Untried Directions list is empty may you form your own hypotheses.
   - If the optimization tree shows >10 failed leaves at the current level, parameter tuning is unlikely to help. Research the bottleneck: read kernel source code, disassemble binaries, clone repos, search for papers on the technique. From that research, build something new — a new kernel function, a different algorithm, or a new pipeline structure.
   - If you are tuning a numeric parameter (tile size, split count, wave count, etc.) and 3 consecutive values produce results within ±1% of baseline, that parameter is at its local optimum. Move to a different phase or technique.
4. **Implement** — Edit submission.py, archive to submissions/vNNN.py
5. **Test** — Submit in test mode, must pass
6. **Leaderboard** — Submit in leaderboard mode. The output contains two sections: "Benchmarks" and "Ranked Benchmark". **Use the Ranked Benchmark numbers** — those are the actual leaderboard scores. The regular Benchmark section uses different parameters (recheck=False, more repeats) and is consistently ~5-15% lower than ranked.
7. **Decide** — Keep if any improvement on the **ranked** geomean (even 0.1%). If KEPT, the per-shape ranked numbers ARE your profiling. If reverted, skip benchmark.
   - Do not rely on LB variance for a better score. If a change doesn't improve the ranked geomean on a single submission, treat it as a revert.
   - Log entries as facts only: `**[what was changed]**: [numbers]. ([versions])` — no adjectives, no conclusions, no generalizations.

**Important:** Never use `run_in_background` for popcorn submissions. Run all test/benchmark/leaderboard commands inline. Background tasks leak into the orchestrator's context.

### Branching & Regression
- Explore optimizations as independent branches from baseline
- 5 consecutive reverts on a branch = branch is exhausted
- When exhausted:
  1. Log branch summary in results.md (what was tried, why each failed)
  2. Verify best_submission.py matches the actual global best code
  3. Save snapshot: `cp submission.py submissions/vNNN_description.py`
  4. Commit all changes (results.md, best_submission.py, submissions/)
  5. End your session
- The next session starts fresh from best_submission.py with clean context — results.md is the handoff
- Never declare "all avenues exhausted", "at the floor", "kernel-bound", or any variant. That is not your call to make. There are always more approaches. Your job is to exhaust your current branch and end your session — the next agent will find what you couldn't.
- Combine winning techniques from different branches when applicable
- When writing branch summaries, untried directions are optional. Examples:
  - **Good** (specific mechanism + why you think it works):
    ```
    Untried Directions:
    - Function X accepts parameter Y which controls Z. Setting Y=4 instead of default 8 should reduce overhead because [specific reason from source code reading].
    ```
  - **Empty** (nothing concrete — this is fine, the next agent will form its own hypotheses):
    ```
    (no untried directions)
    ```
  - **Bad** (vague, defeatist, or padding — do not write these):
    ```
    Untried Directions:
    - Newer library versions
    - A fundamentally different algorithm
    - ALL paths exhausted
    ```

### Combination Sweeps
After 3+ branches exhaust with only neutral results, review all previously-neutral changes in results.md (entries marked "NO" with 0% or within-noise results). Pick 2-3 that target different phases (e.g., one dispatch change + one config change + one codegen hint) and combine them in a single version. Small independent effects can compound.

### Architecture Resets
If 20 consecutive branches exhaust without improving the global best, you may start your next branch from any architecture checkpoint in results.md instead of best_submission.py. You must:
1. State which checkpoint you're reverting to and why
2. Explain what architectural direction you're exploring that wasn't possible from the current best
3. Log this as a new branch in results.md: "based on [checkpoint]"

Your goal is still to beat the global best — the checkpoint is just a different starting point.

### Context Continuation
If you're continuing a previous session (context limit hit):
1. Check `git diff` and `git status` to see uncommitted work
2. Read the last entries in results.md to understand where you left off
3. Check submission.py version vs best_submission.py version
4. Resume from where the previous session stopped — don't restart
