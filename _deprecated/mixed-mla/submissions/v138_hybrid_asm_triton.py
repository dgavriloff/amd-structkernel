#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v138: Hybrid aiter-ASM + Triton MXFP4+FP8.
- Small shapes (bs<=32, kv<=1024): aiter a8w8 ASM (precision-safe).
- Large shapes: Triton MXFP4 K (dot_scaled) + FP8 V (fast path, ~18us).
- Buffer/NSPLIT caching, fp8 scale loaded inside kernel.
"""

import torch
import triton
import triton.language as tl
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

QK_PAD = 1024
PKD_PAD = 512
NSC_PAD = 32

NUM_KV_SPLITS_ASM = 32

_asm_cache = {}
_triton_buf_cache = {}
_nsplit_cache = {}


# ==================== AITER ASM PATH (for small shapes) ====================

def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if per_tensor_quant_hip is not None:
        fp8_tensor, scale = per_tensor_quant_hip(tensor, quant_dtype=FP8_DTYPE)
        return fp8_tensor, scale.reshape(1)
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


def _build_asm_meta(batch_size, q_seq_len, kv_seq_len, q_total, qo_indptr, kv_indptr, dtype_q, dtype_kv, num_kv_splits):
    total_kv_len = batch_size * kv_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
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
        qo_indptr, kv_indptr, kv_last_page_len,
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
        "kv_indptr_use": kv_indptr,
    }


def _run_asm(q, kv_data, qo_indptr, kv_indptr, config):
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    q_seq_len = config.get("q_seq_len", 1)
    q_total = q.shape[0]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    q_fp8, q_scale = quantize_fp8(q.view(-1, NUM_HEADS, QK_HEAD_DIM))

    key = ('a8w8', batch_size, q_seq_len, kv_seq_len)
    if key not in _asm_cache:
        _asm_cache[key] = _build_asm_meta(
            batch_size, q_seq_len, kv_seq_len, q_total, qo_indptr, kv_indptr,
            FP8_DTYPE, FP8_DTYPE, NUM_KV_SPLITS_ASM,
        )
    cached = _asm_cache[key]

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


# ==================== TRITON MXFP4+FP8 PATH (for large shapes) ====================

@triton.jit
def _stage1(
    Q, KV_pk, KV_sc, KV_fp8, kv_indptr, qo_indptr,
    Mid_v, Mid_lse,
    FP8_SCALE_PTR,
    NSPLIT: tl.constexpr,
    BN: tl.constexpr,
    NHEADS: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    FP8_STRIDE: tl.constexpr,
    PKD: tl.constexpr,
    N_SC: tl.constexpr,
    QK_P: tl.constexpr,
    PKD_P: tl.constexpr,
    NSC_P: tl.constexpr,
):
    pid = tl.program_id(0)
    hid = pid % NHEADS
    pid2 = pid // NHEADS
    sid = pid2 % NSPLIT
    bid = pid2 // NSPLIT

    kv_s = tl.load(kv_indptr + bid)
    kv_e = tl.load(kv_indptr + bid + 1)
    kv_len = kv_e - kv_s
    qi = tl.load(qo_indptr + bid)

    per_split = tl.cdiv(kv_len, NSPLIT)
    ss = sid * per_split
    se = tl.minimum(ss + per_split, kv_len)

    lse_off = (qi * NSPLIT + sid) * NHEADS + hid
    v_off = lse_off * V_DIM

    if ss >= kv_len:
        tl.store(Mid_lse + lse_off, float("-inf"))
        ov = tl.arange(0, V_DIM)
        tl.store(Mid_v + v_off + ov, tl.zeros([V_DIM], dtype=tl.bfloat16))
        return

    fp8_scale = tl.load(FP8_SCALE_PTR).to(tl.float32)

    ok = tl.arange(0, QK_P)
    q_vec = tl.load(Q + qi * NHEADS * QK_DIM + hid * QK_DIM + ok,
                     mask=ok < QK_DIM, other=0.0)

    m_i = float("-inf")
    l_i = 0.0
    acc_v = tl.zeros([V_DIM], dtype=tl.float32)
    ov = tl.arange(0, V_DIM)

    for blk_s in range(ss, se, BN):
        bn = tl.minimum(BN, se - blk_s)
        on = tl.arange(0, BN)
        nm = on < bn
        kg = kv_s + blk_s + on

        opk = tl.arange(0, PKD_P)
        kp = tl.load(KV_pk + kg[:, None] * PKD + opk[None, :],
                      mask=nm[:, None] & (opk[None, :] < PKD), other=0)
        osc = tl.arange(0, NSC_P)
        ks = tl.load(KV_sc + kg[:, None] * N_SC + osc[None, :],
                      mask=nm[:, None] & (osc[None, :] < N_SC), other=127)

        kpt = tl.trans(kp)
        scores_2d = tl.dot_scaled(q_vec[None, :].to(tl.bfloat16), None, "bf16",
                                   kpt, ks, "e2m1")
        scores = tl.reshape(scores_2d, [BN])
        scores = scores * 0.041666666666666664
        scores = tl.where(nm, scores, float("-inf"))

        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc_v = acc_v * alpha

        fp8_offsets = (kg[:, None] * FP8_STRIDE + ov[None, :]).to(tl.int64)
        vfp8 = tl.load(KV_fp8 + fp8_offsets, mask=nm[:, None], other=0.0)
        v_f32 = vfp8.to(tl.float32) * fp8_scale

        acc_v += tl.sum(p[:, None] * v_f32, axis=0)
        m_i = m_new

    inv_l = 1.0 / l_i
    tl.store(Mid_lse + lse_off, m_i + tl.log(l_i))
    tl.store(Mid_v + v_off + ov, (acc_v * inv_l).to(tl.bfloat16))


@triton.jit
def _stage2(
    Mid_v, Mid_lse, O, qo_indptr,
    NSPLIT: tl.constexpr,
    NHEADS: tl.constexpr,
    V_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    qi = tl.load(qo_indptr + bid)

    mx = float("-inf")
    for s in tl.static_range(NSPLIT):
        lse = tl.load(Mid_lse + (qi * NSPLIT + s) * NHEADS + hid)
        mx = tl.maximum(mx, lse)

    ov = tl.arange(0, V_DIM)
    acc = tl.zeros([V_DIM], dtype=tl.float32)
    lsum = 0.0
    for s in tl.static_range(NSPLIT):
        lse = tl.load(Mid_lse + (qi * NSPLIT + s) * NHEADS + hid)
        w = tl.exp(lse - mx)
        lsum += w
        base = ((qi * NSPLIT + s) * NHEADS + hid) * V_DIM
        v_bf16 = tl.load(Mid_v + base + ov)
        acc += w * v_bf16.to(tl.float32)

    inv = 1.0 / lsum
    o_base = qi * NHEADS * V_DIM + hid * V_DIM
    tl.store(O + o_base + ov, (acc * inv).to(tl.bfloat16))


def _run_triton(q, kv_data, qo_indptr, kv_indptr, config):
    bs = config["batch_size"]
    tq = q.shape[0]

    kv_pk_raw, kv_sc_raw = kv_data["mxfp4"]
    kv_pk = kv_pk_raw.view(torch.uint8) if kv_pk_raw.dtype != torch.uint8 else kv_pk_raw
    kv_sc = kv_sc_raw.view(torch.uint8) if kv_sc_raw.dtype != torch.uint8 else kv_sc_raw
    if kv_pk.dim() > 2:
        kv_pk = kv_pk.reshape(kv_pk.shape[0], -1)
    if kv_sc.dim() > 2:
        kv_sc = kv_sc.reshape(kv_sc.shape[0], -1)

    kv_fp8_raw, kv_fp8_scale = kv_data["fp8"]
    kv_fp8 = kv_fp8_raw.view(torch.float8_e4m3fnuz).reshape(kv_fp8_raw.shape[0], -1)
    fp8_scale_tensor = kv_fp8_scale.view(-1)

    PKD = kv_pk.shape[-1]
    N_SC = kv_sc.shape[-1]
    FP8_STRIDE = kv_fp8.shape[-1]

    shape_key = (tq, bs)
    if shape_key in _nsplit_cache:
        NSPLIT = _nsplit_cache[shape_key]
    else:
        max_kv = int(kv_indptr[-1].item() - kv_indptr[0].item()) // max(bs, 1)
        NSPLIT = 8 if max_kv <= 2048 else 16
        _nsplit_cache[shape_key] = NSPLIT

    buf_key = (tq, NSPLIT)
    if buf_key not in _triton_buf_cache:
        n_mid_v = tq * NSPLIT * NUM_HEADS * V_HEAD_DIM
        _triton_buf_cache[buf_key] = (
            torch.empty(n_mid_v, dtype=torch.bfloat16, device="cuda"),
            torch.empty(tq * NSPLIT * NUM_HEADS, dtype=torch.float32, device="cuda"),
            torch.empty((tq, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"),
        )
    mid_v, mid_lse, o = _triton_buf_cache[buf_key]

    _stage1[(bs * NSPLIT * NUM_HEADS,)](
        q, kv_pk, kv_sc, kv_fp8,
        kv_indptr, qo_indptr,
        mid_v, mid_lse,
        fp8_scale_tensor,
        NSPLIT=NSPLIT, BN=64, NHEADS=NUM_HEADS,
        QK_DIM=QK_HEAD_DIM, V_DIM=V_HEAD_DIM,
        FP8_STRIDE=FP8_STRIDE,
        PKD=PKD, N_SC=N_SC,
        QK_P=QK_PAD, PKD_P=PKD_PAD, NSC_P=NSC_PAD,
        num_warps=4, num_stages=2,
    )

    _stage2[(bs, NUM_HEADS)](
        mid_v, mid_lse, o, qo_indptr,
        NSPLIT=NSPLIT, NHEADS=NUM_HEADS, V_DIM=V_HEAD_DIM,
        num_warps=4,
    )

    return o


# ==================== DISPATCH ====================

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    # Small shapes: aiter ASM for precision
    # Large shapes: Triton MXFP4+FP8 for speed
    if bs <= 32 and kv_seq_len <= 1024:
        return _run_asm(q, kv_data, qo_indptr, kv_indptr, config)
    else:
        return _run_triton(q, kv_data, qo_indptr, kv_indptr, config)
