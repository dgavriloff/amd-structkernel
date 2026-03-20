#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v094: Use non-persistent mode with auto split selection via mla_decode_fwd.

Key insight: The non-persistent mode for num_kv_splits=1 with fp8 writes
output directly, skipping the reduce kernel entirely. For bs>=256, splits=1
is optimal (each batch fills a CU). This saves one kernel launch per call.

For small batches, use persistent mode with tuned splits as before.
BF16 path for bs<=4 & kv<=1024 (avoids Q quantization overhead).

WARNING: page_size=1 EVERYWHERE.
"""

import torch
from task import input_t, output_t

from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter import mla_decode_stage1_asm_fwd, mla_reduce_v1
from aiter.mla import mla_decode_fwd
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


def _choose_num_kv_splits(batch_size, kv_seq_len, is_fp8=True):
    """Choose optimal num_kv_splits using aiter's heuristic."""
    total_kv = batch_size * kv_seq_len
    avg_kv = total_kv / batch_size
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

    if is_fp8:
        # fp8 min_block_n constraint: nhead*qseqlen=16 -> min_block_n=128
        min_block_n = 128
        max_splits_for_seqlen = max(1, int(avg_kv + min_block_n - 1) // min_block_n)
        best_splits = min(best_splits, max_splits_for_seqlen)

    return best_splits


def _build_meta_persistent(batch_size, q_seq_len, kv_seq_len, q_total, qo_indptr, kv_indptr, dtype_q, dtype_kv, num_kv_splits):
    """Build metadata for persistent mode (used for multi-split cases)."""
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
        NUM_HEADS // NUM_KV_HEADS,
        NUM_KV_HEADS,
        True,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=q_seq_len,
        uni_seqlen_qo=q_seq_len,
        fast_mode=True,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
    )

    num_partials = reduce_partial_map.numel()

    split_data = torch.empty(
        (num_partials * q_seq_len, 1, NUM_HEADS, V_HEAD_DIM),
        dtype=torch.float32, device="cuda",
    )
    split_lse = torch.empty(
        (num_partials * q_seq_len, 1, NUM_HEADS, 1),
        dtype=torch.float32, device="cuda",
    )
    output = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    return {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "split_data": split_data,
        "split_lse": split_lse,
        "output": output,
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
        "kv_indptr_use": kv_indptr_use,
    }


def _build_meta_nonpersistent(batch_size, q_seq_len, kv_seq_len, q_total, qo_indptr, kv_indptr, num_kv_splits):
    """Build metadata for non-persistent mode (simpler, used for splits=1)."""
    total_kv_len = batch_size * kv_seq_len

    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    num_kv_splits_indptr = torch.arange(
        0, (batch_size + 1) * num_kv_splits, num_kv_splits,
        dtype=torch.int32, device="cuda"
    )
    output = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    return {
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
        "num_kv_splits_indptr": num_kv_splits_indptr,
        "output": output,
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

    # Decide path: bf16 for small shapes (avoid Q quantization),
    # fp8 for everything else
    if batch_size <= 4 and kv_seq_len <= 1024:
        # BF16 path with persistent mode, tuned splits
        kv_buffer = kv_data["bf16"]
        num_splits = _choose_num_kv_splits(batch_size, kv_seq_len, is_fp8=False)
        cached = _get_cached(
            ('bf16_ps', batch_size, q_seq_len, kv_seq_len, num_splits),
            _build_meta_persistent,
            batch_size, q_seq_len, kv_seq_len, q_total,
            qo_indptr, kv_indptr, torch.bfloat16, torch.bfloat16, num_splits,
        )
        kv_buffer_4d = kv_buffer.view(-1, 1, NUM_KV_HEADS, kv_buffer.shape[-1])
        q_shaped = q.view(-1, NUM_HEADS, QK_HEAD_DIM)

        mla_decode_stage1_asm_fwd(
            q_shaped, kv_buffer_4d, qo_indptr,
            cached["kv_indptr_use"], cached["kv_indices"], cached["kv_last_page_len"],
            None, cached["work_meta_data"], cached["work_indptr"], cached["work_info_set"],
            q_seq_len, 1, NUM_KV_HEADS, SM_SCALE,
            cached["split_data"], cached["split_lse"], cached["output"],
            None, None,
        )
        mla_reduce_v1(
            cached["split_data"], cached["split_lse"],
            cached["reduce_indptr"], cached["reduce_final_map"], cached["reduce_partial_map"],
            q_seq_len, cached["output"], None,
        )
        return cached["output"]

    else:
        # FP8 path
        kv_buffer_fp8, kv_scale = kv_data["fp8"]
        q_fp8, q_scale = quantize_fp8(q.view(-1, NUM_HEADS, QK_HEAD_DIM))
        num_splits = _choose_num_kv_splits(batch_size, kv_seq_len, is_fp8=True)

        if num_splits == 1:
            # Non-persistent mode: for splits=1 with fp8, output is written
            # directly by stage1, no reduce needed
            cached = _get_cached(
                ('fp8_np', batch_size, q_seq_len, kv_seq_len),
                _build_meta_nonpersistent,
                batch_size, q_seq_len, kv_seq_len, q_total,
                qo_indptr, kv_indptr, num_splits,
            )
            kv_buffer_4d = kv_buffer_fp8.view(-1, 1, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])
            output = cached["output"]

            # Use mla_decode_fwd which handles the special splits=1 fast path
            mla_decode_fwd(
                q_fp8, kv_buffer_4d, output,
                qo_indptr, kv_indptr,
                cached["kv_indices"], cached["kv_last_page_len"],
                q_seq_len,
                page_size=1, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE,
                num_kv_splits=num_splits,
                num_kv_splits_indptr=cached["num_kv_splits_indptr"],
                q_scale=q_scale, kv_scale=kv_scale,
                intra_batch_mode=False,
            )
            return output

        else:
            # Persistent mode with multiple splits
            cached = _get_cached(
                ('fp8_ps', batch_size, q_seq_len, kv_seq_len, num_splits),
                _build_meta_persistent,
                batch_size, q_seq_len, kv_seq_len, q_total,
                qo_indptr, kv_indptr, FP8_DTYPE, FP8_DTYPE, num_splits,
            )
            kv_buffer_4d = kv_buffer_fp8.view(-1, 1, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])

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
