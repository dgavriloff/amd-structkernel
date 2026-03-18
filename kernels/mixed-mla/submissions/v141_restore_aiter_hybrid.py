#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v140: Restore the known-good aiter hybrid path after the slow Triton regression.
- Recover the archived v089 aiter implementation from best_submission.py.
- Use bf16 non-persistent for the smallest shape, bf16 persistent for kv>=4096,
  and a16w8 persistent elsewhere.
- Re-establish a competitive baseline before further tuning.
"""

import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8

NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

NUM_KV_SPLITS = 32

_cache = {}


def _get_path(batch_size, kv_seq_len):
    if batch_size <= 4 and kv_seq_len <= 1024:
        return "bf16"
    if kv_seq_len >= 4096:
        return "bf16_persist"
    return "a16w8"


def _build_persistent_meta(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr, dtype_q, dtype_kv, page_size):
    total_kv_len = batch_size * kv_seq_len

    if page_size > 1:
        num_pages = total_kv_len // page_size
        kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
        kv_indptr_use = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * (kv_seq_len // page_size)
        kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32, device="cuda")
    else:
        kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
        kv_indptr_use = kv_indptr
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    info = get_mla_metadata_info_v1(
        batch_size,
        1,
        NUM_HEADS,
        dtype_q,
        dtype_kv,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=NUM_KV_SPLITS,
        intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (
        work_metadata,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = work

    get_mla_metadata_v1(
        qo_indptr,
        kv_indptr_use,
        kv_last_page_len,
        NUM_HEADS // NUM_KV_HEADS,
        NUM_KV_HEADS,
        True,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=page_size,
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
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
    return meta, kv_indices, kv_last_page_len, kv_indptr_use, page_size, o


def _get_cached_a16w8(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr):
    if kv_seq_len >= 4096:
        page_size = 16
    elif kv_seq_len >= 1024 and batch_size >= 32:
        page_size = 8
    elif kv_seq_len >= 1024:
        page_size = 2
    else:
        page_size = 1

    key = ("a16w8", batch_size, kv_seq_len, page_size)
    if key not in _cache:
        _cache[key] = _build_persistent_meta(
            batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr, torch.bfloat16, FP8_DTYPE, page_size
        )
    return _cache[key]


def _get_cached_bf16(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr):
    key = ("bf16", batch_size, kv_seq_len)
    if key not in _cache:
        total_kv_len = batch_size * kv_seq_len
        kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        o = torch.empty((q_total, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")
        _cache[key] = (kv_indices, kv_last_page_len, o)
    return _cache[key]


def _get_cached_bf16_persist(batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr):
    page_size = 64
    key = ("bf16_persist", batch_size, kv_seq_len, page_size)
    if key not in _cache:
        _cache[key] = _build_persistent_meta(
            batch_size, kv_seq_len, q_total, qo_indptr, kv_indptr, torch.bfloat16, torch.bfloat16, page_size
        )
    return _cache[key]


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    path = _get_path(batch_size, kv_seq_len)

    if path == "bf16":
        kv_buffer = kv_data["bf16"]
        kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], 1, NUM_KV_HEADS, kv_buffer.shape[-1])
        kv_indices, kv_last_page_len, o = _get_cached_bf16(batch_size, kv_seq_len, q.shape[0], qo_indptr, kv_indptr)

        mla_decode_fwd(
            q.view(-1, NUM_HEADS, QK_HEAD_DIM),
            kv_buffer_4d,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            1,
            page_size=1,
            nhead_kv=NUM_KV_HEADS,
            sm_scale=SM_SCALE,
            logit_cap=0.0,
        )
        return o

    if path == "bf16_persist":
        kv_buffer = kv_data["bf16"]
        meta, kv_indices, kv_last_page_len, kv_indptr_use, page_size, o = _get_cached_bf16_persist(
            batch_size, kv_seq_len, q.shape[0], qo_indptr, kv_indptr
        )
        kv_buffer_4d = kv_buffer.view(-1, page_size, NUM_KV_HEADS, kv_buffer.shape[-1])

        mla_decode_fwd(
            q.view(-1, NUM_HEADS, QK_HEAD_DIM),
            kv_buffer_4d,
            o,
            qo_indptr,
            kv_indptr_use,
            kv_indices,
            kv_last_page_len,
            1,
            page_size=page_size,
            nhead_kv=NUM_KV_HEADS,
            sm_scale=SM_SCALE,
            logit_cap=0.0,
            num_kv_splits=NUM_KV_SPLITS,
            intra_batch_mode=True,
            **meta,
        )
        return o

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    meta, kv_indices, kv_last_page_len, kv_indptr_use, page_size, o = _get_cached_a16w8(
        batch_size, kv_seq_len, q.shape[0], qo_indptr, kv_indptr
    )
    kv_buffer_4d = kv_buffer_fp8.view(-1, page_size, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])

    mla_decode_fwd(
        q.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr_use,
        kv_indices,
        kv_last_page_len,
        1,
        page_size=page_size,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=NUM_KV_SPLITS,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )
    return o
