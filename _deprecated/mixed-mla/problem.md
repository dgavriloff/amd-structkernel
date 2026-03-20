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