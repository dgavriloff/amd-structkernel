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