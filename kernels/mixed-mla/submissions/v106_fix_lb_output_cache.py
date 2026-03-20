#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v105: Use bf16 Q + bf16 KV (a16w16) to eliminate Q quantization overhead.

The per_tensor_quant_hip call costs ~5-10us. For small batch sizes (bs=4,32)
this is a significant fraction of total time. Using bf16 KV costs 2x bandwidth
but saves a kernel launch + quant compute.

Uses mla_decode_fwd non-persistent mode which handles all split logic internally.
The a16w16 kernel (mla_dec_stage1_bf16_a16w16_subQ16_mqa16) handles bf16+bf16
for qseqlen=1 non-persistent.

WARNING: page_size=1 EVERYWHERE.
"""

import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd

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

    # bf16 path — no quantization needed
    kv_buffer_bf16 = kv_data["bf16"]
    q_bf16 = q.view(-1, NUM_HEADS, QK_HEAD_DIM)

    kv_buffer_4d = kv_buffer_bf16.view(-1, 1, NUM_KV_HEADS, kv_buffer_bf16.shape[-1])

    # Cache kv metadata per shape (constant across calls); allocate output fresh
    key = (batch_size, kv_seq_len)
    if key not in _cache:
        total_kv = batch_size * kv_seq_len
        kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
        kv_last_page_len = torch.full((batch_size,), kv_seq_len, dtype=torch.int32, device="cuda")
        _cache[key] = (kv_indices, kv_last_page_len)

    kv_indices, kv_last_page_len = _cache[key]
    output = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    mla_decode_fwd(
        q_bf16, kv_buffer_4d, output,
        qo_indptr, kv_indptr,
        kv_indices, kv_last_page_len,
        1,  # max_seqlen_q
        page_size=1, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE,
        intra_batch_mode=False,
    )

    return output
