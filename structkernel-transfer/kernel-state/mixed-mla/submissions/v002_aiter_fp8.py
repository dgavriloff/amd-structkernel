#!POPCORN leaderboard mla-py
#!POPCORN gpu MI355X

"""
v002: Use aiter mla_decode_fwd persistent kernel (fp8 Q + fp8 KV).

Instead of naive PyTorch loops with torch._scaled_mm, use aiter's optimized
persistent MLA decode kernel directly. This is the same kernel the reference
uses (a8w8 mode) — fp8 Q quantized on the fly + fp8 KV from kv_data.

This should match or be very close to the reference performance, establishing
a strong baseline from which to optimize further (e.g., tuning NUM_KV_SPLITS,
trying mxfp4 KV, etc.).
"""

import torch
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# FP8 quantization helper (per-tensor, sglang style)
# ---------------------------------------------------------------------------

def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor FP8 quantization. Returns (fp8_tensor, scale)."""
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


# ---------------------------------------------------------------------------
# Persistent mode metadata helpers
# ---------------------------------------------------------------------------

def _make_mla_decode_metadata(
    batch_size: int,
    max_q_len: int,
    nhead: int,
    nhead_kv: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_kv_splits: int = NUM_KV_SPLITS,
):
    """Allocate and populate work buffers for persistent mla_decode_fwd."""
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, nhead, q_dtype, kv_dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        nhead // nhead_kv,   # num_heads_per_head_k
        nhead_kv,            # num_heads_k
        True,                # is_causal
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )

    return {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }


# ---------------------------------------------------------------------------
# Main kernel: aiter mla_decode_fwd with fp8 Q + fp8 KV
# ---------------------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:
    """MLA decode using aiter persistent kernel (fp8 Q + fp8 KV)."""
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]

    # FP8 KV
    kv_buffer_fp8, kv_scale_fp8 = kv_data["fp8"]

    # Quantize Q to fp8
    q_fp8, q_scale = quantize_fp8(q)

    # Build KV indices (flat, since page_size=1)
    total_kv_len = int(kv_indptr[-1].item())
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")

    # Reshape kv_buffer to 4D: (total_kv, page_size, nhead_kv, dim)
    kv_buffer_4d = kv_buffer_fp8.view(kv_buffer_fp8.shape[0], PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])

    max_q_len = q_seq_len
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    # Build persistent-mode metadata
    meta = _make_mla_decode_metadata(
        batch_size, max_q_len, nq, nkv,
        q_fp8.dtype, kv_buffer_fp8.dtype,
        qo_indptr, kv_indptr, kv_last_page_len,
        num_kv_splits=NUM_KV_SPLITS,
    )

    # Allocate output
    o = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")

    # Run aiter persistent MLA decode
    mla_decode_fwd(
        q_fp8.view(-1, nq, dq),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_q_len,
        page_size=PAGE_SIZE,
        nhead_kv=nkv,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=NUM_KV_SPLITS,
        q_scale=q_scale,
        kv_scale=kv_scale_fp8,
        intra_batch_mode=True,
        **meta,
    )

    return o
