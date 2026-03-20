#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v103: Direct ASM kernel calls with fully pre-allocated buffers.

For splits==1 (fp8 Q+KV non-persistent):
- Call mla_decode_stage1_asm_fwd directly with logits=output.view(4D)
- Stage1 writes directly to output, no reduce needed
- Pre-allocate attn_lse (required by stage1 but unused after)

For splits>1 (fp8 persistent):
- Use stage1_asm + mla_reduce_v1 as before
- All buffers pre-allocated

Key savings vs v99:
- No mla_decode_fwd Python wrapper overhead
- No per-call tensor allocations (logits, attn_lse, final_lse)
- No num_kv_splits_indptr computation
- Minimal Python path per call

WARNING: page_size=1 EVERYWHERE.
"""

import torch
from task import input_t, output_t

from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter import mla_decode_stage1_asm_fwd, mla_reduce_v1
from aiter import per_tensor_quant_hip

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


def _build_nonpersistent_splits1(batch_size, q_total):
    """Pre-allocate buffers for non-persistent splits==1 path.
    Stage1 writes directly to output (logits = output reshaped).
    """
    output = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    # logits is just output viewed as 4D - stage1 writes directly here
    logits = output.view(q_total, 1, NUM_HEADS, V_HEAD_DIM)
    # attn_lse is required by stage1 but not used after (no reduce for splits==1)
    attn_lse = torch.empty((q_total, 1, NUM_HEADS, 1), dtype=torch.float32, device="cuda")
    # kv metadata
    total_kv = batch_size * 8192  # max kv_seq_len, oversize is fine
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
    num_kv_splits_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")

    return {
        "output": output, "logits": logits, "attn_lse": attn_lse,
        "kv_indices": kv_indices, "num_kv_splits_indptr": num_kv_splits_indptr,
    }


def _build_persistent(batch_size, q_seq_len, kv_seq_len, q_total, qo_indptr, kv_indptr, num_kv_splits):
    """Build metadata for persistent mode."""
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
    split_data = torch.empty((num_partials * q_seq_len, 1, NUM_HEADS, V_HEAD_DIM), dtype=torch.float32, device="cuda")
    split_lse = torch.empty((num_partials * q_seq_len, 1, NUM_HEADS, 1), dtype=torch.float32, device="cuda")
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

    # FP8 path — quantize Q
    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    q_fp8, q_scale = per_tensor_quant_hip(q.view(-1, NUM_HEADS, QK_HEAD_DIM), quant_dtype=FP8_DTYPE)
    q_scale = q_scale.reshape(1)

    num_splits = _choose_num_kv_splits(batch_size, kv_seq_len)
    kv_buffer_4d = kv_buffer_fp8.view(-1, 1, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])

    if num_splits == 1:
        # Non-persistent: stage1 writes directly to output (no reduce needed)
        key = ('np1', batch_size)
        if key not in _cache:
            _cache[key] = _build_nonpersistent_splits1(batch_size, q_total)
        cached = _cache[key]

        # Compute kv_last_page_len on the fly (cheap)
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

        # Slice kv_indices to actual size
        total_kv = batch_size * kv_seq_len
        kv_indices = cached["kv_indices"][:total_kv]

        mla_decode_stage1_asm_fwd(
            q_fp8, kv_buffer_4d, qo_indptr,
            kv_indptr, kv_indices, kv_last_page_len,
            cached["num_kv_splits_indptr"],  # non-persistent mode
            None, None, None,  # no work metadata
            q_seq_len, 1, NUM_KV_HEADS, SM_SCALE,
            cached["logits"], cached["attn_lse"], cached["output"],
            q_scale, kv_scale,
        )
        return cached["output"]

    else:
        # Persistent mode with multiple splits
        key = ('ps', batch_size, q_seq_len, kv_seq_len, num_splits)
        if key not in _cache:
            _cache[key] = _build_persistent(
                batch_size, q_seq_len, kv_seq_len, q_total,
                qo_indptr, kv_indptr, num_splits,
            )
        cached = _cache[key]

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
