#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v021: Direct stage1 call with 1 split for bs=256,kv<=1024.

Hypothesis: For bs=256,kv=1024 the auto-tuner selects 1 split. With 1 split,
the stage1 kernel writes output directly and no reduce is needed. Instead of
going through mla_decode_fwd which allocates intermediates each call, we call
mla_decode_stage1_asm_fwd directly with pre-cached intermediates.

This combines the v020 insight (1-split for bs=256,kv=1k) with v017's approach
(direct stage1 call). v017 didn't help for bf16 because PyTorch's allocator
is efficient, but for fp8 with 1 split, the logits can alias the output
directly — saving both allocation AND the reduce step.

Target: bs=256,kv=1024 from 103µs -> ~90-95µs
"""

import torch
import aiter
from task import input_t, output_t

from aiter.mla import mla_decode_fwd, get_meta_param
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


def _should_use_bf16(batch_size, kv_seq_len):
    """Decide whether to use bf16 path (no Q quantization)."""
    if batch_size <= 4:
        return True
    if batch_size <= 64 and kv_seq_len <= 1024:
        return True
    return False


def _should_use_direct_fp8(batch_size, kv_seq_len):
    """Use direct stage1 call with 1 split (skip reduce)."""
    if batch_size >= 128 and kv_seq_len <= 1024:
        return True
    return False


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


def _get_cached_fp8(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr):
    """Get or build all cached buffers for fp8 persistent path."""
    key = (batch_size, kv_seq_len, FP8_DTYPE, FP8_DTYPE)
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


def _get_cached_bf16(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr):
    """Get or build cached buffers for bf16 non-persistent path."""
    key = (batch_size, kv_seq_len, torch.bfloat16, torch.bfloat16)
    if key in _cache:
        return _cache[key]

    total_kv_len = batch_size * kv_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    o = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    cached = (kv_indices, kv_last_page_len, o)
    _cache[key] = cached
    return cached


def _get_cached_direct_fp8(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr):
    """Get or build cached buffers for direct fp8 stage1 call with 1 split."""
    key = (batch_size, kv_seq_len, "direct_fp8_1split")
    if key in _cache:
        return _cache[key]

    total_kv_len = batch_size * kv_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    # Auto-tuned num_kv_splits for this shape
    num_kv_splits, num_kv_splits_indptr = get_meta_param(
        None, batch_size, total_kv_len, NUM_HEADS, 1, FP8_DTYPE
    )

    # Output buffer
    o = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    # For 1-split fp8: logits aliases o (zero-copy)
    if num_kv_splits == 1:
        logits = o.view((q_total, 1, NUM_HEADS, V_HEAD_DIM))
    else:
        logits = torch.empty(
            (q_total, num_kv_splits, NUM_HEADS, V_HEAD_DIM),
            dtype=torch.float32, device="cuda",
        )

    attn_lse = torch.empty(
        (q_total, num_kv_splits, NUM_HEADS, 1),
        dtype=torch.float32, device="cuda",
    )

    cached = (kv_indices, kv_last_page_len, o, num_kv_splits_indptr, logits, attn_lse)
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
    elif _should_use_direct_fp8(batch_size, kv_seq_len):
        # Direct stage1 call with pre-cached intermediates, 1-split (skip reduce)
        kv_buffer_fp8, kv_scale = kv_data["fp8"]
        q_fp8, q_scale = _quantize_fp8(q)

        kv_indices, kv_last_page_len, o, num_kv_splits_indptr, logits, attn_lse = _get_cached_direct_fp8(
            batch_size, kv_seq_len, q.shape[0], qo_indptr, kv_indptr
        )

        kv_buffer_4d = kv_buffer_fp8.view(kv_buffer_fp8.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])

        # Direct stage1 ASM call
        aiter.mla_decode_stage1_asm_fwd(
            q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
            kv_buffer_4d,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_kv_splits_indptr,
            None,   # work_meta_data (non-persistent)
            None,   # work_indptr
            None,   # work_info_set
            1,      # max_seqlen_q
            PAGE_SIZE,
            NUM_KV_HEADS,
            SM_SCALE,
            logits,
            attn_lse,
            o,
            q_scale,
            kv_scale,
        )
        # For 1-split fp8: output is already written directly to o via aliased logits
        # No reduce step needed
    else:
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
