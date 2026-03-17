#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v134: Non-persistent ASM kernel with fp8 KV for ALL shapes.
Uses aiter's auto-tuned num_kv_splits via mla_decode_fwd non-persistent mode.
FP8 KV (a16w8) for all shapes -- no persistent metadata overhead.
Minimal Python overhead with caching.
"""

import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes

FP8_DTYPE = aiter_dtypes.fp8

# MLA constants
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                       # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

# Cache
_cache = {}


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    q_total = q.shape[0]

    # Use FP8 KV for all shapes
    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    key = (batch_size, kv_seq_len)
    if key not in _cache:
        total_kv_len = batch_size * kv_seq_len
        kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        o = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")
        _cache[key] = (kv_indices, kv_last_page_len, o)

    kv_indices, kv_last_page_len, o = _cache[key]

    kv_buffer_4d = kv_buffer_fp8.view(kv_buffer_fp8.shape[0], 1, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])

    # Non-persistent mode: auto-tuned splits, ASM stage1 kernel
    mla_decode_fwd(
        q.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        1,
        page_size=1,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        kv_scale=kv_scale,
    )

    return o
