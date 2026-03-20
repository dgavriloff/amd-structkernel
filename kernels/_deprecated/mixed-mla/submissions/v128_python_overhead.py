#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v128: Aggressive Python overhead reduction for LB (where recheck=True means
new data each iteration). Cache tensor preprocessing by shape signature to
avoid redundant view/reshape/shape lookups. Pre-cache grid tuples. Minimize
dict accesses and attribute lookups in hot path.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

NUM_HEADS: int = 16
QK_HEAD_DIM: int = 576
V_HEAD_DIM: int = 512
QK_PAD: int = 1024
PKD_PAD: int = 512
NSC_PAD: int = 32

_buf_cache: dict = {}
_nsplit_cache: dict = {}
_shape_cache: dict = {}  # caches (PKD, N_SC, FP8_STRIDE) per kv shape

# Pre-bind frequently used functions/types
_uint8 = torch.uint8
_fp8 = torch.float8_e4m3fnuz
_empty = torch.empty
_bf16 = torch.bfloat16
_f32 = torch.float32


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

    # Load fp8 scale from device memory (avoids .item() CPU sync)
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
        vfp8 = tl.load(KV_fp8 + fp8_offsets,
                        mask=nm[:, None], other=0.0)
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


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    tq = q.shape[0]

    # Index into kv_data dict once, then index tuples
    mxfp4 = kv_data["mxfp4"]
    fp8_pair = kv_data["fp8"]
    kv_pk_raw = mxfp4[0]
    kv_sc_raw = mxfp4[1]
    kv_fp8_raw = fp8_pair[0]

    # Fast path: view/reshape with minimal Python overhead
    kv_pk = kv_pk_raw if kv_pk_raw.dtype == _uint8 else kv_pk_raw.view(_uint8)
    kv_sc = kv_sc_raw if kv_sc_raw.dtype == _uint8 else kv_sc_raw.view(_uint8)
    if kv_pk.dim() > 2:
        kv_pk = kv_pk.reshape(kv_pk.shape[0], -1)
    if kv_sc.dim() > 2:
        kv_sc = kv_sc.reshape(kv_sc.shape[0], -1)

    kv_fp8 = kv_fp8_raw.view(_fp8).reshape(kv_fp8_raw.shape[0], -1)
    fp8_scale_tensor = fp8_pair[1].view(-1)

    # Cache shape constants per kv_pk shape to avoid repeated .shape[-1] lookups
    pk_shape = kv_pk.shape
    sc_shape = kv_sc.shape
    fp8_shape = kv_fp8.shape
    shape_sig = (pk_shape[-1], sc_shape[-1], fp8_shape[-1])

    PKD = shape_sig[0]
    N_SC = shape_sig[1]
    FP8_STRIDE = shape_sig[2]

    # Cache NSPLIT decision
    shape_key = (tq, bs)
    if shape_key in _nsplit_cache:
        NSPLIT = _nsplit_cache[shape_key]
    else:
        max_kv = int(kv_indptr[-1].item() - kv_indptr[0].item()) // max(bs, 1)
        NSPLIT = 8 if max_kv <= 2048 else 16
        _nsplit_cache[shape_key] = NSPLIT

    # Cache intermediate buffers
    buf_key = (tq, NSPLIT)
    if buf_key not in _buf_cache:
        n_mid_v = tq * NSPLIT * NUM_HEADS * V_HEAD_DIM
        _buf_cache[buf_key] = (
            _empty(n_mid_v, dtype=_bf16, device="cuda"),
            _empty(tq * NSPLIT * NUM_HEADS, dtype=_f32, device="cuda"),
            _empty((tq, NUM_HEADS, V_HEAD_DIM), dtype=_bf16, device="cuda"),
        )
    cached = _buf_cache[buf_key]
    mid_v = cached[0]
    mid_lse = cached[1]
    o = cached[2]

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
