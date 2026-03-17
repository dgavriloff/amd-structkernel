#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v015: Non-persistent bf16 + non-persistent fp8.

Changes from v009:
- Both bf16 and fp8 use non-persistent mode (no metadata, auto-tuned splits)
- This tests if the auto-tuned splits are better than fixed 32 for fp8
- bf16 path: bs <= 4 (all kv), bs <= 32 with kv <= 1024
- fp8 path: everything else
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

PAGE_SIZE = 1

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
    if batch_size <= 32 and kv_seq_len <= 1024:
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


def _get_cached(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr, is_bf16):
    """Get or build cached buffers for non-persistent path."""
    key = (batch_size, kv_seq_len, is_bf16)
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
    """MLA decode with hybrid bf16/fp8 strategy, both non-persistent."""
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    use_bf16 = _should_use_bf16(batch_size, kv_seq_len)

    if use_bf16:
        kv_buffer = kv_data["bf16"]
        q_input = q
        q_scale = None
        kv_scale = None
    else:
        kv_buffer_fp8, kv_scale = kv_data["fp8"]
        kv_buffer = kv_buffer_fp8
        q_fp8, q_scale = _quantize_fp8(q)
        q_input = q_fp8

    kv_indices, kv_last_page_len, o = _get_cached(
        batch_size, kv_seq_len, q.shape[0], qo_indptr, kv_indptr, use_bf16
    )

    kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_buffer.shape[-1])

    # Non-persistent mode: no metadata, auto-tuned splits
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
        q_scale=q_scale,
        kv_scale=kv_scale,
    )

    return o
