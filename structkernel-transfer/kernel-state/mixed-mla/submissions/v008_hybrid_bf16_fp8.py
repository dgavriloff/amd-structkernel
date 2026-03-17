#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v008: Hybrid bf16/fp8 strategy — use bf16 (a16w16) for small batches, fp8 for large.

Hypothesis: For small batch sizes (bs=4), Q quantization overhead dominates.
Using bf16 Q + bf16 KV avoids quantization entirely. The a16w16 kernel is slower
per-token but may win net when Q quant overhead is eliminated.

Strategy:
- bs <= 4: use bf16 Q + bf16 KV (no quantization)
- bs > 4: use fp8 Q + fp8 KV (current best)
"""

import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8

# MLA constants
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                       # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

PAGE_SIZE = 1
NUM_KV_SPLITS = 32

# Try to use aiter's native scaled_fp8_quant
try:
    from aiter import scaled_fp8_quant as _aiter_fp8_quant
    HAS_AITER_FP8_QUANT = True
except ImportError:
    HAS_AITER_FP8_QUANT = False

# Cache per (batch_size, kv_seq_len, use_bf16)
_cache = {}

# FP8 constants for fallback
_FP8_FINFO = torch.finfo(FP8_DTYPE)


def _quantize_fp8(q):
    """Quantize Q to fp8. Use aiter native if available."""
    if HAS_AITER_FP8_QUANT:
        orig_shape = q.shape
        q_fp8, q_scale = _aiter_fp8_quant(q.reshape(-1, orig_shape[-1]))
        return q_fp8.view(orig_shape), q_scale
    else:
        amax = q.abs().amax().clamp(min=1e-12)
        scale = amax / _FP8_FINFO.max
        q_fp8 = (q / scale).clamp(min=_FP8_FINFO.min, max=_FP8_FINFO.max).to(FP8_DTYPE)
        return q_fp8, scale.to(torch.float32).reshape(1)


def _get_cached(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr, q_dtype, kv_dtype):
    """Get or build all cached buffers for given dtype combination."""
    key = (batch_size, kv_seq_len, q_dtype, kv_dtype)
    if key in _cache:
        return _cache[key]

    total_kv_len = batch_size * kv_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    max_q_len = 1
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, NUM_HEADS, q_dtype, kv_dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=NUM_KV_SPLITS, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        NUM_HEADS // NUM_KV_HEADS,
        NUM_KV_HEADS,
        True,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=False,
        max_split_per_batch=NUM_KV_SPLITS,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )

    meta = {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }

    o = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    cached = (meta, kv_indices, kv_last_page_len, o)
    _cache[key] = cached
    return cached


def custom_kernel(data: input_t) -> output_t:
    """MLA decode with hybrid bf16/fp8 strategy."""
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    # Choose strategy based on batch size
    use_bf16 = (batch_size <= 4)

    if use_bf16:
        # bf16 path: no Q quantization needed
        kv_buffer = kv_data["bf16"]
        q_input = q
        q_scale = None
        kv_scale = None
        q_dtype = torch.bfloat16
        kv_dtype = torch.bfloat16
    else:
        # fp8 path: quantize Q
        kv_buffer_fp8, kv_scale = kv_data["fp8"]
        kv_buffer = kv_buffer_fp8
        q_fp8, q_scale = _quantize_fp8(q)
        q_input = q_fp8
        q_dtype = FP8_DTYPE
        kv_dtype = FP8_DTYPE

    meta, kv_indices, kv_last_page_len, o = _get_cached(
        batch_size, kv_seq_len, q.shape[0], qo_indptr, kv_indptr, q_dtype, kv_dtype
    )

    kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_buffer.shape[-1])

    mla_decode_fwd(
        q_input.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        1,
        page_size=PAGE_SIZE,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=NUM_KV_SPLITS,
        q_scale=q_scale,
        kv_scale=kv_scale if not use_bf16 else None,
        intra_batch_mode=True,
        **meta,
    )

    return o
