# Mixed-MLA Kernel — MI355X

## What This Kernel Does
MLA (Multi-head Latent Attention) decode for DeepSeek R1's forward_absorb path on AMD Instinct MI355X.
Flow: `bf16 Q (absorbed)` + `compressed KV cache` -> `attention scores` -> `softmax` -> `output @ V` -> `bf16 output`

This is a decode-only kernel (q_seq_len=1, kv_seq_len up to 8k) with variable-length batching via indptr.

## Architecture
- 16 query heads, 1 shared KV head (MQA — 16:1 ratio)
- Q/K dim: 576 (512 latent + 64 RoPE)
- V dim: 512 (= kv_lora_rank, first 512 dims of KV buffer)
- sm_scale = 1/sqrt(576)
- KV buffer: full 576 dims as keys, first 512 as values

## Input Format
`(q, kv_data, qo_indptr, kv_indptr, config)` where:
- `q`: (total_q, 16, 576) bfloat16 — absorbed queries
- `kv_data`: dict with THREE KV cache formats provided simultaneously:
  - `"bf16"`: Tensor (total_kv, 1, 576) bfloat16
  - `"fp8"`: (Tensor, Tensor) — fp8 buffer + scalar scale
  - `"mxfp4"`: (Tensor, Tensor) — fp4x2 buffer + fp8_e8m0 block-32 scale
- `qo_indptr`: (batch_size+1,) int32
- `kv_indptr`: (batch_size+1,) int32
- `config`: dict with MLA parameters

Output: (total_q, 16, 512) bfloat16

## Benchmark Shapes
| batch_size | q_seq_len | kv_seq_len |
|---|---|---|
| 4 | 1 | 1024 |
| 4 | 1 | 8192 |
| 32 | 1 | 1024 |
| 32 | 1 | 8192 |
| 64 | 1 | 1024 |
| 64 | 1 | 8192 |
| 256 | 1 | 1024 |
| 256 | 1 | 8192 |

Ranking by geometric mean. Tolerance: rtol=2e-02, atol=8e-03.

## Reference Performance
The reference uses aiter's a8w8 persistent MLA kernel (fp8 Q + fp8 KV).

| Case | a8w8 (µs) | a16w16 (µs) | a8w8 speedup |
|---|---|---|---|
| bs=4, kv=1k | ~118 | ~162 | 1.4x |
| bs=4, kv=8k | ~113 | ~177 | 1.6x |
| bs=64, kv=8k | ~171 | ~353 | 2.1x |
| bs=256, kv=8k | ~349 | ~814 | 2.3x |

## Key Optimization Opportunities
1. **MXFP4 KV cache** — 4x bandwidth savings over bf16, 2x over fp8. Fuse dequant with attention to avoid materializing bf16 intermediate.
2. **Custom split-K / split-batch scheduling** — aiter uses 32-way KV splits. Different splits may be better per batch/seq_len.
3. **MQA broadcast** — 1 KV head shared across 16 query heads. Load KV once, broadcast in LDS.
4. **Split K/V tiling** — 576 dims for keys, 512 for values. Different tile strategies for score vs output stages.

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
../../submit.sh kernels/mixed-mla <MODE>
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

#### Example: results.md branch tracking

A working branch:
```
## Branch: mxfp4-fused-dequant (based on v001_baseline)
| # | Version | Target | Hypothesis | Test | Geomean | vs Best | Keep? |
|---|---------|--------|-----------|------|---------|---------|-------|
| 2 | v002 | bandwidth | Use mxfp4 KV with fused dequant in attention | PASS | 95µs | -20% | YES |
→ Branch active, best: v002 (95µs)
```

A dead branch:
```
## Branch: custom-triton-attention (based on v002)
| # | Version | Target | Hypothesis | Test | Geomean | vs Best | Keep? |
|---|---------|--------|-----------|------|---------|---------|-------|
| 3 | v003 | latency | Custom Triton FlashAttention with MQA broadcast | FAIL | — | — | NO |
| 4 | v004 | latency | Simplified Triton attention without flash | PASS | 150µs | +58% | NO |
→ Branch exhausted. 2 reverts. Reason: Triton attention can't match aiter persistent kernel.
→ Session ended.
```
