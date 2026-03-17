#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v012: Test fast_mode=True metadata + per-shape tuned max_split_per_batch.

Changes from v009:
- fast_mode=True for metadata generation (smaller work buffers, potentially better scheduling)
- Per-shape max_split_per_batch: 16 for kv=1024 (less split overhead), 64 for kv=8192 (more parallelism)
- bf16 path unchanged: bs <= 4 (all kv), bs <= 32 with kv <= 1024
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

# Try to use aiter's native scaled_fp8_quant
try:
    from aiter import scaled_fp8_quant as _aiter_fp8_quant
    HAS_AITER_FP8_QUANT = True
except ImportError:
    HAS_AITER_FP8_QUANT = False

# Cache per (batch_size, kv_seq_len, q_dtype, kv_dtype)
_cache = {}

# FP8 constants for fallback
_FP8_FINFO = torch.finfo(FP8_DTYPE)


def _should_use_bf16(batch_size, kv_seq_len):
    """Decide whether to use bf16 path (no Q quantization)."""
    if batch_size <= 4:
        return True
    if batch_size <= 32 and kv_seq_len <= 1024:
        return True
    return False


def _get_num_kv_splits(batch_size, kv_seq_len):
    """Per-shape tuned max_split_per_batch."""
    if kv_seq_len <= 1024:
        return 16  # fewer splits for short KV (less overhead)
    else:
        return 64  # more splits for long KV (more parallelism)


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

    num_kv_splits = _get_num_kv_splits(batch_size, kv_seq_len)

    total_kv_len = batch_size * kv_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    max_q_len = 1
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, NUM_HEADS, q_dtype, kv_dtype,
        is_sparse=False, fast_mode=True,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
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
        fast_mode=True,
        max_split_per_batch=num_kv_splits,
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

    cached = (meta, kv_indices, kv_last_page_len, o, num_kv_splits)
    _cache[key] = cached
    return cached


def custom_kernel(data: input_t) -> output_t:
    """MLA decode with hybrid bf16/fp8 strategy."""
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    use_bf16 = _should_use_bf16(batch_size, kv_seq_len)

    if use_bf16:
        kv_buffer = kv_data["bf16"]
        q_input = q
        q_scale = None
        kv_scale = None
        q_dtype = torch.bfloat16
        kv_dtype = torch.bfloat16
    else:
        kv_buffer_fp8, kv_scale = kv_data["fp8"]
        kv_buffer = kv_buffer_fp8
        q_fp8, q_scale = _quantize_fp8(q)
        q_input = q_fp8
        q_dtype = FP8_DTYPE
        kv_dtype = FP8_DTYPE

    meta, kv_indices, kv_last_page_len, o, num_kv_splits = _get_cached(
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
        num_kv_splits=num_kv_splits,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )

    return o
