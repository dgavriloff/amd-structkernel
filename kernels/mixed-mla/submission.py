#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v119: Micro-optimized v114 - hardcode sm_scale, inline constants.
- Hardcode sm_scale as float literal (0.04166...) to avoid register use.
- Keep MXFP4 K + FP8 V, adaptive NSPLIT (8/16).
- Use int64 only where needed for fp8 offsets.
- Inline V_DIM and reduce intermediate computations.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

NUM_HEADS = 16
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
QK_PAD = 1024
PKD_PAD = 512
NSC_PAD = 32

_cache = {}


@triton.jit
def _stage1(
    Q, KV_pk, KV_sc, KV_fp8, kv_indptr, qo_indptr,
    Mid_v, Mid_lse,
    fp8_scale,
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
        scores = scores * 0.041666666666666664  # 1/24 = 1/sqrt(576)
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

    kv_pk_raw, kv_sc_raw = kv_data["mxfp4"]
    kv_pk = kv_pk_raw.view(torch.uint8) if kv_pk_raw.dtype != torch.uint8 else kv_pk_raw
    kv_sc = kv_sc_raw.view(torch.uint8) if kv_sc_raw.dtype != torch.uint8 else kv_sc_raw
    if kv_pk.dim() > 2:
        kv_pk = kv_pk.reshape(kv_pk.shape[0], -1)
    if kv_sc.dim() > 2:
        kv_sc = kv_sc.reshape(kv_sc.shape[0], -1)

    kv_fp8_raw, kv_fp8_scale = kv_data["fp8"]
    kv_fp8 = kv_fp8_raw.view(torch.float8_e4m3fnuz).reshape(kv_fp8_raw.shape[0], -1)
    fp8_scale_val = kv_fp8_scale.item() if kv_fp8_scale.numel() == 1 else 1.0

    PKD = kv_pk.shape[-1]
    N_SC = kv_sc.shape[-1]
    FP8_STRIDE = kv_fp8.shape[-1]

    max_kv = int(kv_indptr[-1].item() - kv_indptr[0].item()) // max(bs, 1)
    if max_kv <= 2048:
        NSPLIT = 8
    else:
        NSPLIT = 16
    BN = 64

    n_mid_v = tq * NSPLIT * NUM_HEADS * V_HEAD_DIM
    mid_v = torch.empty(n_mid_v, dtype=torch.bfloat16, device="cuda")
    mid_lse = torch.empty(tq * NSPLIT * NUM_HEADS, dtype=torch.float32, device="cuda")
    o = torch.empty((tq, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    _stage1[(bs * NSPLIT * NUM_HEADS,)](
        q.view(-1, NUM_HEADS, QK_HEAD_DIM), kv_pk, kv_sc, kv_fp8,
        kv_indptr, qo_indptr,
        mid_v, mid_lse,
        fp8_scale_val,
        NSPLIT=NSPLIT, BN=BN, NHEADS=NUM_HEADS,
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
