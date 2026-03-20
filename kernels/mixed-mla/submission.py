#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v104: Use mla_decode_fwd non-persistent for ALL shapes.

mla_decode_fwd handles splits internally:
- splits==1 + fp8: writes directly to output (no reduce)
- splits>1: calls stage1_asm + Triton stage2 reduce

Pre-allocate output and kv_indices/kv_last_page_len.
Let mla_decode_fwd handle the rest (it allocates logits/lse internally).

Key advantages vs persistent path:
- No get_mla_metadata_v1 overhead
- No metadata caching complexity
- mla_decode_fwd's Triton stage2 may be faster than mla_reduce_v1

WARNING: page_size=1 EVERYWHERE.
"""

import torch
from task import input_t, output_t

from aiter import dtypes as aiter_dtypes
from aiter import per_tensor_quant_hip
from aiter.mla import mla_decode_fwd

FP8_DTYPE = aiter_dtypes.fp8

# MLA constants
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                       # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

_cache = {}


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    q_total = q.shape[0]

    # FP8 path — quantize Q
    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    q_fp8, q_scale = per_tensor_quant_hip(q.view(-1, NUM_HEADS, QK_HEAD_DIM), quant_dtype=FP8_DTYPE)
    q_scale = q_scale.reshape(1)

    kv_buffer_4d = kv_buffer_fp8.view(-1, 1, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])

    # Cache output buffer and kv metadata per shape
    key = (batch_size, kv_seq_len)
    if key not in _cache:
        total_kv = batch_size * kv_seq_len
        kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
        kv_last_page_len = torch.full((batch_size,), kv_seq_len, dtype=torch.int32, device="cuda")
        output = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")
        _cache[key] = (kv_indices, kv_last_page_len, output)

    kv_indices, kv_last_page_len, output = _cache[key]

    mla_decode_fwd(
        q_fp8, kv_buffer_4d, output,
        qo_indptr, kv_indptr,
        kv_indices, kv_last_page_len,
        1,  # max_seqlen_q
        page_size=1, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE,
        q_scale=q_scale, kv_scale=kv_scale,
        intra_batch_mode=False,
    )

    return output
