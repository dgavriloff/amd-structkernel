#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v023: Custom Triton FlashDecoding kernel for bf16 MLA decode.

Changes from v016:
- Replace non-persistent bf16 (ASM stage1 + Triton reduce) with a single
  custom Triton FlashDecoding kernel for bf16 shapes.
- The custom kernel fuses attention score computation and value accumulation
  in one pass, eliminating the split-K reduce overhead.
- Uses BLOCK_N=64 for KV iteration, handles MQA (16 Q heads, 1 KV head).
- Target: reduce overhead on bf16 shapes (bs<=4 all + bs<=64 kv<=1024)
"""

import torch
import triton
import triton.language as tl
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


@triton.jit
def _mla_flash_decode_kernel(
    Q,          # [total_q, NUM_HEADS, QK_HEAD_DIM] bf16
    KV,         # [total_kv, 1, QK_HEAD_DIM] bf16
    O,          # [total_q, NUM_HEADS, V_HEAD_DIM] bf16
    kv_indptr,  # [batch_size + 1] int32
    sm_scale,
    stride_qb: tl.constexpr,  # stride for q batch (total_q) dim
    stride_qh: tl.constexpr,  # stride for q head dim
    stride_kv_tok,  # stride for kv token dim
    stride_ob: tl.constexpr,  # stride for o batch dim
    stride_oh: tl.constexpr,  # stride for o head dim
    BLOCK_N: tl.constexpr,    # KV tokens per iteration
    BLOCK_DK: tl.constexpr,   # key dim block (>= QK_HEAD_DIM)
    BLOCK_DV: tl.constexpr,   # value dim block (>= V_HEAD_DIM)
    QK_HEAD_DIM_CONST: tl.constexpr,
    V_HEAD_DIM_CONST: tl.constexpr,
):
    """
    FlashDecoding for MLA bf16 decode (qseqlen=1, MQA 16:1).

    Grid: (batch_size, NUM_HEADS)
    Each program computes attention for one (batch, head) pair.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # KV range for this batch
    kv_start = tl.load(kv_indptr + cur_batch)
    kv_end = tl.load(kv_indptr + cur_batch + 1)
    kv_len = kv_end - kv_start

    # Load Q for this head: [QK_HEAD_DIM]
    # q_offset = cur_batch (since total_q = batch_size for qseqlen=1)
    offs_dk = tl.arange(0, BLOCK_DK)
    mask_dk = offs_dk < QK_HEAD_DIM_CONST
    q = tl.load(
        Q + cur_batch * stride_qb + cur_head * stride_qh + offs_dk,
        mask=mask_dk,
        other=0.0,
    ).to(tl.float32)

    # Online softmax + value accumulation
    m_i = -float("inf")  # max score
    l_i = 0.0            # sum of exp(score - m)

    # Accumulated output: [V_HEAD_DIM]
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < V_HEAD_DIM_CONST
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    # Iterate over KV tokens in blocks of BLOCK_N
    for start_n in range(0, kv_len, BLOCK_N):
        # How many valid tokens in this block
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len

        # Load K for this block: [BLOCK_N, QK_HEAD_DIM]
        kv_offs = (kv_start + offs_n)[:, None] * stride_kv_tok + offs_dk[None, :]
        k = tl.load(
            KV + kv_offs,
            mask=mask_n[:, None] & mask_dk[None, :],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_N, QK_HEAD_DIM]

        # Compute attention scores: Q @ K^T -> [BLOCK_N]
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale  # [BLOCK_N]
        scores = tl.where(mask_n, scores, -float("inf"))

        # Online softmax update
        m_ij = tl.max(scores)
        m_new = tl.maximum(m_i, m_ij)

        # Rescale old accumulator
        alpha = tl.exp(m_i - m_new)

        # Compute exp(scores - m_new) for new tokens
        p = tl.exp(scores - m_new)
        l_new = l_i * alpha + tl.sum(p)

        # Load V for this block: [BLOCK_N, V_HEAD_DIM]
        v_offs = (kv_start + offs_n)[:, None] * stride_kv_tok + offs_dv[None, :]
        v = tl.load(
            KV + v_offs,
            mask=mask_n[:, None] & mask_dv[None, :],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_N, V_HEAD_DIM]

        # Update accumulator: rescale old + add new weighted V
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    # Normalize output
    out = (acc / l_i).to(tl.bfloat16)

    # Store output
    tl.store(
        O + cur_batch * stride_ob + cur_head * stride_oh + offs_dv,
        out,
        mask=mask_dv,
    )


def _should_use_bf16(batch_size, kv_seq_len):
    """Decide whether to use bf16 path (no Q quantization)."""
    if batch_size <= 4:
        return True
    if batch_size <= 64 and kv_seq_len <= 1024:
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
    """Get or build cached buffers for bf16 path."""
    key = (batch_size, kv_seq_len, torch.bfloat16, torch.bfloat16)
    if key in _cache:
        return _cache[key]

    o = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    cached = (o,)
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

        (o,) = _get_cached_bf16(
            batch_size, kv_seq_len, q.shape[0], qo_indptr, kv_indptr
        )

        # Custom Triton FlashDecoding kernel
        # KV buffer is (total_kv, 1, 576) — flatten to (total_kv, 576) for simpler indexing
        kv_flat = kv_buffer.view(-1, QK_HEAD_DIM)

        grid = (batch_size, NUM_HEADS)
        _mla_flash_decode_kernel[grid](
            q,
            kv_flat,
            o,
            kv_indptr,
            SM_SCALE,
            q.stride(0),     # stride_qb
            q.stride(1),     # stride_qh
            kv_flat.stride(0),  # stride_kv_tok
            o.stride(0),     # stride_ob
            o.stride(1),     # stride_oh
            BLOCK_N=64,
            BLOCK_DK=1024,   # next_power_of_2(576) = 1024
            BLOCK_DV=512,    # V_HEAD_DIM = 512
            QK_HEAD_DIM_CONST=QK_HEAD_DIM,
            V_HEAD_DIM_CONST=V_HEAD_DIM,
            num_warps=4,
            num_stages=2,
        )
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
