#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v028: a16w8 persistent for bs>=128 kv<=1024, v016 baseline otherwise.

Hypothesis: For bs=256,kv=1024, the a16w8 persistent kernel (bf16 Q + fp8 KV)
skips Q quantization overhead (~10us) and is faster despite using the qseqlen=4
kernel design. v026 showed -13.4% for this shape (103->89.2us) but geomean
regressed +1.5% due to benchmark noise. Retrying with cleaner implementation
targeting only bs>=128 kv<=1024.

Target: bs=256,kv=1k from 103us -> ~89us. Expected geomean: ~64.5us (-2%).
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

# Cache per (batch_size, kv_seq_len, path_type)
_cache = {}

# FP8 constants for fallback
_FP8_FINFO = torch.finfo(FP8_DTYPE)


def _get_path(batch_size, kv_seq_len):
    """Decide which path to use: 'bf16', 'a16w8', or 'a8w8'."""
    if batch_size <= 4:
        return 'bf16'
    if batch_size <= 64 and kv_seq_len <= 1024:
        return 'bf16'
    if batch_size >= 128 and kv_seq_len <= 1024:
        return 'a16w8'
    return 'a8w8'


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


def _build_persistent_meta(batch_size, q_total, qo_indptr, kv_indptr, dtype_q, dtype_kv):
    """Build persistent mode metadata for given Q/KV dtypes."""
    total_kv_len = int(kv_indptr[-1].item())
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    max_q_len = 1
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, NUM_HEADS, dtype_q, dtype_kv,
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
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
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

    return meta, kv_indices, kv_last_page_len, o


def _get_cached_fp8(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr):
    """Get or build all cached buffers for a8w8 persistent path."""
    key = ('a8w8', batch_size, kv_seq_len)
    if key in _cache:
        return _cache[key]
    cached = _build_persistent_meta(batch_size, q_total, qo_indptr, kv_indptr, FP8_DTYPE, FP8_DTYPE)
    _cache[key] = cached
    return cached


def _get_cached_a16w8(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr):
    """Get or build all cached buffers for a16w8 persistent path (bf16 Q + fp8 KV)."""
    key = ('a16w8', batch_size, kv_seq_len)
    if key in _cache:
        return _cache[key]
    cached = _build_persistent_meta(batch_size, q_total, qo_indptr, kv_indptr, torch.bfloat16, FP8_DTYPE)
    _cache[key] = cached
    return cached


def _get_cached_bf16(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr):
    """Get or build cached buffers for bf16 non-persistent path."""
    key = ('bf16', batch_size, kv_seq_len)
    if key in _cache:
        return _cache[key]

    total_kv_len = batch_size * kv_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    o = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    cached = (kv_indices, kv_last_page_len, o)
    _cache[key] = cached
    return cached


def custom_kernel(data: input_t) -> output_t:
    """MLA decode with hybrid bf16/a16w8/a8w8 strategy."""
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    path = _get_path(batch_size, kv_seq_len)

    if path == 'bf16':
        kv_buffer = kv_data["bf16"]
        kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_buffer.shape[-1])

        kv_indices, kv_last_page_len, o = _get_cached_bf16(
            batch_size, kv_seq_len, q.shape[0], qo_indptr, kv_indptr
        )

        # Non-persistent mode: no metadata, auto-tuned splits
        mla_decode_fwd(
            q.view(-1, NUM_HEADS, QK_HEAD_DIM),
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
        )

    elif path == 'a16w8':
        # bf16 Q + fp8 KV persistent — skip Q quantization
        kv_buffer_fp8, kv_scale = kv_data["fp8"]

        meta, kv_indices, kv_last_page_len, o = _get_cached_a16w8(
            batch_size, kv_seq_len, q.shape[0], qo_indptr, kv_indptr
        )

        kv_buffer_4d = kv_buffer_fp8.view(kv_buffer_fp8.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])

        # Persistent mode — a16w8 kernel dispatched via bf16 Q + fp8 KV
        mla_decode_fwd(
            q.view(-1, NUM_HEADS, QK_HEAD_DIM),
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
            kv_scale=kv_scale,
            intra_batch_mode=True,
            **meta,
        )

    else:
        # a8w8: fp8 Q + fp8 KV persistent
        kv_buffer_fp8, kv_scale = kv_data["fp8"]
        q_fp8, q_scale = _quantize_fp8(q)

        meta, kv_indices, kv_last_page_len, o = _get_cached_fp8(
            batch_size, kv_seq_len, q.shape[0], qo_indptr, kv_indptr
        )

        kv_buffer_4d = kv_buffer_fp8.view(kv_buffer_fp8.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])

        # Persistent mode with 32 splits
        mla_decode_fwd(
            q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
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
            kv_scale=kv_scale,
            intra_batch_mode=True,
            **meta,
        )

    return o
