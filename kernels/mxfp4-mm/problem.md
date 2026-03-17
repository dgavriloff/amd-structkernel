# MXFP4-MM Kernel — MI355X

## What This Kernel Does
MXFP4 quantization + block-scaled matrix multiplication on AMD Instinct MI355X.
Flow: `bf16 A` -> `MXFP4 per-1x32 quant A` -> `gemm_a4w4` -> `bf16 C [m,n]`

## Input Format
`(A, B, B_q, B_shuffle, B_scale_sh)` where:
- `A`: M x K, K-major, bfloat16
- `B`: N x K, K-major, bfloat16
- `B_q`: N x K/2, K-major, MXFP4
- `B_shuffle`: N x K/2, shuffled to (16,16) tile coalesced, MXFP4
- `B_scale_sh`: * x K/32, E8M0 (padded)

B_shuffle and B_scale_sh are pre-computed. Only A needs runtime quantization.

## Benchmark Shapes
All small-M (memory-bound regime). M divisible by 64 not required — actual M values: 4, 16, 32, 64, 256.
N: 2112-7168. K: 512-7168.

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