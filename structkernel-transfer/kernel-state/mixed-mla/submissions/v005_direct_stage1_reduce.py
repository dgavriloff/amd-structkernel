#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v005: Bypass mla_decode_fwd wrapper — call stage1+reduce directly with cached intermediates.

Changes from v004:
- Call aiter.mla_decode_stage1_asm_fwd + aiter.mla_reduce_v1 directly
- Cache logits and attn_lse intermediate buffers (avoid 2 allocations per call)
- The mla_decode_fwd wrapper allocated these every call
"""

import torch
from task import input_t, output_t

import aiter
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

# Cache per (batch_size, kv_seq_len)
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


def _get_cached(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr):
    """Get or build all cached buffers including intermediates."""
    key = (batch_size, kv_seq_len)
    if key in _cache:
        return _cache[key]

    total_kv_len = batch_size * kv_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    max_q_len = 1
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, NUM_HEADS, FP8_DTYPE, FP8_DTYPE,
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
        dtype_q=FP8_DTYPE,
        dtype_kv=FP8_DTYPE,
    )

    # Output buffer
    o = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    # Intermediate buffers (normally allocated per-call inside mla_decode_fwd)
    n_partials = reduce_partial_map.size(0)
    logits = torch.empty(
        (n_partials * max_q_len, 1, NUM_HEADS, V_HEAD_DIM),
        dtype=torch.float32, device="cuda",
    )
    attn_lse = torch.empty(
        (n_partials * max_q_len, 1, NUM_HEADS, 1),
        dtype=torch.float32, device="cuda",
    )

    cached = {
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
        "o": o,
        "logits": logits,
        "attn_lse": attn_lse,
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }
    _cache[key] = cached
    return cached


def custom_kernel(data: input_t) -> output_t:
    """MLA decode calling stage1+reduce directly with cached intermediates."""
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    kv_buffer_fp8, kv_scale_fp8 = kv_data["fp8"]

    # Quantize Q to fp8
    q_fp8, q_scale = _quantize_fp8(q)

    # Get all cached buffers
    c = _get_cached(batch_size, kv_seq_len, q.shape[0], qo_indptr, kv_indptr)

    # Reshape kv_buffer to 4D
    kv_buffer_4d = kv_buffer_fp8.view(
        kv_buffer_fp8.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_buffer_fp8.shape[-1]
    )

    # Stage 1: attention scores + partial outputs
    aiter.mla_decode_stage1_asm_fwd(
        q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_buffer_4d,
        qo_indptr,
        kv_indptr,
        c["kv_indices"],
        c["kv_last_page_len"],
        None,  # num_kv_splits_indptr (not used in persistent mode)
        c["work_meta_data"],
        c["work_indptr"],
        c["work_info_set"],
        1,  # max_seqlen_q
        PAGE_SIZE,
        NUM_KV_HEADS,
        SM_SCALE,
        c["logits"],
        c["attn_lse"],
        c["o"],
        q_scale,
        kv_scale_fp8,
    )

    # Stage 2: reduce partial outputs
    aiter.mla_reduce_v1(
        c["logits"],
        c["attn_lse"],
        c["reduce_indptr"],
        c["reduce_final_map"],
        c["reduce_partial_map"],
        1,  # max_seqlen_q
        c["o"],
        None,  # final_lse (not needed)
    )

    return c["o"]
