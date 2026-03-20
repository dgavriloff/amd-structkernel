#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v101: Call mla_decode_stage1_asm_fwd directly for ALL shapes, pre-allocate
all buffers, skip reduce for splits=1.

Key changes over v099:
1. For splits=1 (non-persistent): call asm kernel directly instead of going
   through mla_decode_fwd wrapper. Pre-allocate attn_lse buffer. The logits
   buffer is output itself (reshaped). No reduce needed.
2. For splits>1 (persistent): same as before with stage1_asm + reduce.
3. All temporary buffers (attn_lse, split_data, split_lse, output) are
   pre-allocated and cached per-shape. Only Q quantization runs per call.
4. NO CUDA GRAPHS — LB uses recheck=True with new data each iteration.

WARNING: page_size=1 EVERYWHERE.
"""

import torch
from task import input_t, output_t

from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter import mla_decode_stage1_asm_fwd, mla_reduce_v1
try:
    from aiter import per_tensor_quant_hip
except ImportError:
    per_tensor_quant_hip = None

FP8_DTYPE = aiter_dtypes.fp8

# MLA constants
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                       # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

MI355X_CU_COUNT = 256

_cache = {}


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize bf16 tensor to fp8 with per-tensor scale."""
    if per_tensor_quant_hip is not None:
        fp8_tensor, scale = per_tensor_quant_hip(tensor, quant_dtype=FP8_DTYPE)
        return fp8_tensor, scale.reshape(1)
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


def _choose_num_kv_splits(batch_size, kv_seq_len):
    """Choose optimal num_kv_splits using aiter's heuristic."""
    avg_kv = kv_seq_len
    overhead = 84.1

    best_score = float("-inf")
    best_splits = 1
    for splits in range(1, 17):
        waves = (batch_size * splits + MI355X_CU_COUNT - 1) // MI355X_CU_COUNT
        score = (
            batch_size * splits
            / (waves * MI355X_CU_COUNT)
            * avg_kv
            / (avg_kv + overhead * splits)
        )
        if score > best_score:
            best_score = score
            best_splits = splits

    # FP8 min_block_n constraint
    min_block_n = 128
    max_splits_for_seqlen = max(1, int(avg_kv + min_block_n - 1) // min_block_n)
    best_splits = min(best_splits, max_splits_for_seqlen)

    return best_splits


def _build_meta_splits1(batch_size, q_seq_len, kv_seq_len, q_total, qo_indptr, kv_indptr):
    """Build metadata for non-persistent mode (splits=1). Asm kernel writes output directly."""
    total_kv_len = batch_size * kv_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    # splits=1 so indptr is just 0,1,2,...,batch_size
    num_kv_splits_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    # Output tensor — for splits=1 fp8, asm writes directly to logits which IS output reshaped
    output = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    # attn_lse is required by asm kernel signature
    attn_lse = torch.empty((q_total, 1, NUM_HEADS, 1), dtype=torch.float32, device="cuda")

    return {
        "kv_indices": kv_indices, "kv_last_page_len": kv_last_page_len,
        "num_kv_splits_indptr": num_kv_splits_indptr,
        "output": output, "attn_lse": attn_lse,
    }


def _build_meta_persistent(batch_size, q_seq_len, kv_seq_len, q_total, qo_indptr, kv_indptr, dtype_q, dtype_kv, num_kv_splits):
    """Build metadata for persistent mode."""
    total_kv_len = batch_size * kv_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_indptr_use = kv_indptr
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    info = get_mla_metadata_info_v1(
        batch_size, q_seq_len, NUM_HEADS, dtype_q, dtype_kv,
        is_sparse=False, fast_mode=True,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr_use, kv_last_page_len,
        NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=1, kv_granularity=16,
        max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
        fast_mode=True, max_split_per_batch=num_kv_splits,
        intra_batch_mode=True, dtype_q=dtype_q, dtype_kv=dtype_kv,
    )

    num_partials = reduce_partial_map.numel()
    split_data = torch.empty((num_partials * q_seq_len, 1, NUM_HEADS, V_HEAD_DIM), dtype=torch.float32, device="cuda")
    split_lse = torch.empty((num_partials * q_seq_len, 1, NUM_HEADS, 1), dtype=torch.float32, device="cuda")
    output = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    return {
        "work_meta_data": work_metadata, "work_indptr": work_indptr,
        "work_info_set": work_info_set, "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map, "reduce_partial_map": reduce_partial_map,
        "split_data": split_data, "split_lse": split_lse, "output": output,
        "kv_indices": kv_indices, "kv_last_page_len": kv_last_page_len,
        "kv_indptr_use": kv_indptr_use,
    }


def _get_cached(key, build_fn, *args, **kwargs):
    if key in _cache:
        return _cache[key]
    cached = build_fn(*args, **kwargs)
    _cache[key] = cached
    return cached


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    q_seq_len = config.get("q_seq_len", 1)
    q_total = q.shape[0]

    # FP8 path for all shapes — fresh quantization every call, no CUDA graphs
    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    q_fp8, q_scale = quantize_fp8(q.view(-1, NUM_HEADS, QK_HEAD_DIM))
    num_splits = _choose_num_kv_splits(batch_size, kv_seq_len)

    kv_buffer_4d = kv_buffer_fp8.view(-1, 1, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])

    if num_splits == 1:
        # Non-persistent: call asm directly, output written to logits buffer
        cached = _get_cached(
            ('s1', batch_size, q_seq_len, kv_seq_len),
            _build_meta_splits1,
            batch_size, q_seq_len, kv_seq_len, q_total,
            qo_indptr, kv_indptr,
        )
        output = cached["output"]
        # For splits=1 + fp8, logits IS the output (reshaped to 4D)
        logits = output.view(q_total, 1, NUM_HEADS, V_HEAD_DIM)

        # 19 args total: Q, KV, qo_indptr, kv_indptr, kv_page_indices,
        #   kv_last_page_lens, num_kv_splits_indptr, work_meta_data, work_indptr,
        #   work_info_set, max_seqlen_q, page_size, nhead_kv, softmax_scale,
        #   splitData, splitLse, output, q_scale, kv_scale
        mla_decode_stage1_asm_fwd(
            q_fp8, kv_buffer_4d, qo_indptr,
            kv_indptr, cached["kv_indices"], cached["kv_last_page_len"],
            cached["num_kv_splits_indptr"],
            None, None, None,  # no work metadata for non-persistent
            q_seq_len, 1, NUM_KV_HEADS, SM_SCALE,
            logits, cached["attn_lse"], output,
            q_scale, kv_scale,
        )
        return output

    else:
        # Persistent mode with multiple splits
        cached = _get_cached(
            ('ps', batch_size, q_seq_len, kv_seq_len, num_splits),
            _build_meta_persistent,
            batch_size, q_seq_len, kv_seq_len, q_total,
            qo_indptr, kv_indptr, FP8_DTYPE, FP8_DTYPE, num_splits,
        )

        mla_decode_stage1_asm_fwd(
            q_fp8, kv_buffer_4d, qo_indptr,
            cached["kv_indptr_use"], cached["kv_indices"], cached["kv_last_page_len"],
            None, cached["work_meta_data"], cached["work_indptr"], cached["work_info_set"],
            q_seq_len, 1, NUM_KV_HEADS, SM_SCALE,
            cached["split_data"], cached["split_lse"], cached["output"],
            q_scale, kv_scale,
        )
        mla_reduce_v1(
            cached["split_data"], cached["split_lse"],
            cached["reduce_indptr"], cached["reduce_final_map"], cached["reduce_partial_map"],
            q_seq_len, cached["output"], None,
        )
        return cached["output"]
