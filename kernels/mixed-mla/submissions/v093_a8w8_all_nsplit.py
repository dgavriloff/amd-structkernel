#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v093: a8w8 ASM for ALL shapes with adaptive NSPLIT.

Use fp8 Q+KV (a8w8) for everything — no bf16 fallback.
Adaptive NSPLIT based on wave occupancy model:
- Small bs+kv: fewer splits (reduce overhead)
- Large bs+kv: more splits (better parallelism)
Pre-allocate fp8 Q buffer to avoid repeated allocation.

WARNING: Uses NEW aiter two-stage API. Old mla_decode_fwd is BROKEN.
page_size=1 EVERYWHERE.
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
_q_fp8_buf = {}


def _choose_nsplit(batch_size, kv_seq_len):
    """Choose optimal NSPLIT based on wave occupancy model."""
    # The ASM kernel launches batch_size * nsplit * NUM_HEADS work items
    # We want enough parallelism to fill the GPU, but not too much reduce overhead
    best_score = float("-inf")
    best_splits = 1
    reduce_overhead = 50.0  # µs overhead per reduce split

    for splits in [1, 2, 4, 8, 16, 32]:
        total_work = batch_size * splits * NUM_HEADS
        waves = (total_work + MI355X_CU_COUNT - 1) // MI355X_CU_COUNT
        occupancy = total_work / (waves * MI355X_CU_COUNT)

        # Balance occupancy vs reduce overhead
        kv_per_split = kv_seq_len / splits
        # Score: higher is better
        # Occupancy matters, but reduce overhead grows with splits
        score = occupancy * kv_seq_len / (kv_seq_len + reduce_overhead * splits)

        if score > best_score:
            best_score = score
            best_splits = splits
    return best_splits


def _build_meta(batch_size, q_seq_len, kv_seq_len, q_total, qo_indptr, kv_indptr, num_kv_splits):
    total_kv_len = batch_size * kv_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    info = get_mla_metadata_info_v1(
        batch_size, q_seq_len, NUM_HEADS, FP8_DTYPE, FP8_DTYPE,
        is_sparse=False, fast_mode=True,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=1, kv_granularity=16,
        max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
        fast_mode=True, max_split_per_batch=num_kv_splits,
        intra_batch_mode=True, dtype_q=FP8_DTYPE, dtype_kv=FP8_DTYPE,
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
        "work_meta_data": work_metadata, "work_indptr": work_indptr,
        "work_info_set": work_info_set, "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map, "reduce_partial_map": reduce_partial_map,
        "split_data": split_data, "split_lse": split_lse, "output": output,
        "kv_indices": kv_indices, "kv_last_page_len": kv_last_page_len,
        "kv_indptr_use": kv_indptr,
    }


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    q_seq_len = config.get("q_seq_len", 1)
    q_total = q.shape[0]

    # a8w8 for all shapes
    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    # Quantize Q to fp8
    q_view = q.view(-1, NUM_HEADS, QK_HEAD_DIM)
    if per_tensor_quant_hip is not None:
        q_fp8, q_scale = per_tensor_quant_hip(q_view, quant_dtype=FP8_DTYPE)
        q_scale = q_scale.reshape(1)
    else:
        finfo = torch.finfo(FP8_DTYPE)
        amax = q_view.abs().amax().clamp(min=1e-12)
        scale = amax / finfo.max
        q_fp8 = (q_view / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
        q_scale = scale.to(torch.float32).reshape(1)

    num_splits = _choose_nsplit(batch_size, kv_seq_len)

    key = (batch_size, q_seq_len, kv_seq_len, num_splits)
    if key not in _cache:
        _cache[key] = _build_meta(
            batch_size, q_seq_len, kv_seq_len, q_total,
            qo_indptr, kv_indptr, num_splits,
        )
    cached = _cache[key]

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
