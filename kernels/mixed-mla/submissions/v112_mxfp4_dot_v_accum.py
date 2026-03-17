#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v112: MXFP4 Triton v3: tl.dot for V accum, all 16 heads per program.
Grid: (batch * num_splits). Each program handles all 16 heads.
Uses tl.dot for V accumulation: p[16,BN] @ V[BN,V_PKD] instead of scalar sum.
dot_scaled for Q*K^T, manual dequant for V, tl.dot for V accum.
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
    Q, KV_pk, KV_sc, kv_indptr, qo_indptr,
    Mid_lo, Mid_hi, Mid_lse,
    sm_scale,
    NSPLIT: tl.constexpr,
    BN: tl.constexpr,
    NHEADS: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_PKD: tl.constexpr,
    PKD: tl.constexpr,
    N_SC: tl.constexpr,
    QK_P: tl.constexpr,
    PKD_P: tl.constexpr,
    NSC_P: tl.constexpr,
):
    # Grid: (batch * NSPLIT,)
    pid = tl.program_id(0)
    sid = pid % NSPLIT
    bid = pid // NSPLIT

    kv_s = tl.load(kv_indptr + bid)
    kv_e = tl.load(kv_indptr + bid + 1)
    kv_len = kv_e - kv_s
    qi = tl.load(qo_indptr + bid)

    per_split = tl.cdiv(kv_len, NSPLIT)
    ss = sid * per_split
    se = tl.minimum(ss + per_split, kv_len)

    obase_lse = (qi * NSPLIT + sid) * NHEADS
    obase_v = obase_lse * V_PKD

    oh = tl.arange(0, NHEADS)

    if ss >= kv_len:
        tl.store(Mid_lse + obase_lse + oh, tl.full([NHEADS], float("-inf"), dtype=tl.float32))
        ov = tl.arange(0, V_PKD)
        for h in tl.static_range(NHEADS):
            mask_h = (oh == h)[:, None]
            tl.store(Mid_lo + obase_v + h * V_PKD + ov, tl.zeros([V_PKD], dtype=tl.float32))
            tl.store(Mid_hi + obase_v + h * V_PKD + ov, tl.zeros([V_PKD], dtype=tl.float32))
        return

    # Load Q [NHEADS, QK_P] bf16
    ok = tl.arange(0, QK_P)
    qb = tl.load(Q + qi * NHEADS * QK_DIM + oh[:, None] * QK_DIM + ok[None, :],
                  mask=ok[None, :] < QK_DIM, other=0.0)  # [NHEADS, QK_P]

    m_i = tl.full([NHEADS], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([NHEADS], dtype=tl.float32)
    acc_lo = tl.zeros([NHEADS, V_PKD], dtype=tl.float32)
    acc_hi = tl.zeros([NHEADS, V_PKD], dtype=tl.float32)

    for blk_s in range(ss, se, BN):
        bn = tl.minimum(BN, se - blk_s)
        on = tl.arange(0, BN)
        nm = on < bn
        kg = kv_s + blk_s + on

        # K packed [BN, PKD_P], K scale [BN, NSC_P]
        opk = tl.arange(0, PKD_P)
        kp = tl.load(KV_pk + kg[:, None] * PKD + opk[None, :],
                      mask=nm[:, None] & (opk[None, :] < PKD), other=0)
        osc = tl.arange(0, NSC_P)
        ks = tl.load(KV_sc + kg[:, None] * N_SC + osc[None, :],
                      mask=nm[:, None] & (osc[None, :] < N_SC), other=127)

        # Q@K^T: [NHEADS, QK_P] @ [PKD_P, BN] -> [NHEADS, BN]
        kpt = tl.trans(kp)
        scores = tl.dot_scaled(qb.to(tl.bfloat16), None, "bf16",
                               kpt, ks, "e2m1")
        scores = scores * sm_scale
        scores = tl.where(nm[None, :], scores, float("-inf"))

        # Online softmax
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])  # [NHEADS, BN]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_lo = acc_lo * alpha[:, None]
        acc_hi = acc_hi * alpha[:, None]

        # V dequant [BN, V_PKD]
        ovpk = tl.arange(0, V_PKD)
        vp = tl.load(KV_pk + kg[:, None] * PKD + ovpk[None, :],
                      mask=nm[:, None], other=0)
        sidx = ovpk // 16
        vsc = tl.load(KV_sc + kg[:, None] * N_SC + sidx[None, :],
                       mask=nm[:, None], other=127)
        vsc_f = tl.exp2((vsc.to(tl.int32) - 127).to(tl.float32))

        # fp4 dequant
        lo_raw = (vp & 0xF).to(tl.int32)
        hi_raw = (vp >> 4).to(tl.int32)
        lo_mag = lo_raw & 7
        hi_mag = hi_raw & 7
        lo_e = lo_mag >> 1
        lo_m = lo_mag & 1
        lo_val = tl.where(lo_e == 0,
                          lo_m.to(tl.float32) * 0.5,
                          tl.exp2((lo_e - 1).to(tl.float32)) * (1.0 + lo_m.to(tl.float32) * 0.5))
        lo_val = tl.where(lo_raw >= 8, -lo_val, lo_val) * vsc_f
        hi_e = hi_mag >> 1
        hi_m = hi_mag & 1
        hi_val = tl.where(hi_e == 0,
                          hi_m.to(tl.float32) * 0.5,
                          tl.exp2((hi_e - 1).to(tl.float32)) * (1.0 + hi_m.to(tl.float32) * 0.5))
        hi_val = tl.where(hi_raw >= 8, -hi_val, hi_val) * vsc_f

        # V accum via tl.dot: p[NHEADS, BN] @ V[BN, V_PKD] -> [NHEADS, V_PKD]
        pb = p.to(tl.bfloat16)
        acc_lo += tl.dot(pb, lo_val.to(tl.bfloat16)).to(tl.float32)
        acc_hi += tl.dot(pb, hi_val.to(tl.bfloat16)).to(tl.float32)
        m_i = m_new

    # Normalize
    inv_l = 1.0 / l_i
    acc_lo = acc_lo * inv_l[:, None]
    acc_hi = acc_hi * inv_l[:, None]
    lse_vals = m_i + tl.log(l_i)

    # Store lse
    tl.store(Mid_lse + obase_lse + oh, lse_vals)

    # Store lo/hi per head
    ov = tl.arange(0, V_PKD)
    for h in tl.static_range(NHEADS):
        mask_h = (oh == h)[:, None]
        lo_h = tl.sum(acc_lo * mask_h, axis=0)
        hi_h = tl.sum(acc_hi * mask_h, axis=0)
        tl.store(Mid_lo + obase_v + h * V_PKD + ov, lo_h)
        tl.store(Mid_hi + obase_v + h * V_PKD + ov, hi_h)


@triton.jit
def _stage2(
    Mid_lo, Mid_hi, Mid_lse, O, qo_indptr,
    NSPLIT: tl.constexpr,
    NHEADS: tl.constexpr,
    V_PKD: tl.constexpr,
    V_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    qi = tl.load(qo_indptr + bid)

    mx = float("-inf")
    for s in tl.static_range(NSPLIT):
        lse = tl.load(Mid_lse + (qi * NSPLIT + s) * NHEADS + hid)
        mx = tl.maximum(mx, lse)

    ov = tl.arange(0, V_PKD)
    a_lo = tl.zeros([V_PKD], dtype=tl.float32)
    a_hi = tl.zeros([V_PKD], dtype=tl.float32)
    lsum = 0.0
    for s in tl.static_range(NSPLIT):
        lse = tl.load(Mid_lse + (qi * NSPLIT + s) * NHEADS + hid)
        w = tl.exp(lse - mx)
        lsum += w
        base = ((qi * NSPLIT + s) * NHEADS + hid) * V_PKD
        a_lo += w * tl.load(Mid_lo + base + ov)
        a_hi += w * tl.load(Mid_hi + base + ov)

    inv = 1.0 / lsum
    o_base = qi * NHEADS * V_DIM + hid * V_DIM
    tl.store(O + o_base + ov * 2, (a_lo * inv).to(tl.bfloat16))
    tl.store(O + o_base + ov * 2 + 1, (a_hi * inv).to(tl.bfloat16))


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

    PKD = kv_pk.shape[-1]
    N_SC = kv_sc.shape[-1]
    V_PKD = V_HEAD_DIM // 2

    NSPLIT = 16
    BN = 64

    n_mid = tq * NSPLIT * NUM_HEADS * V_PKD
    mid_lo = torch.empty(n_mid, dtype=torch.float32, device="cuda")
    mid_hi = torch.empty(n_mid, dtype=torch.float32, device="cuda")
    mid_lse = torch.empty(tq * NSPLIT * NUM_HEADS, dtype=torch.float32, device="cuda")
    o = torch.empty((tq, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    _stage1[(bs * NSPLIT,)](
        q.view(-1, NUM_HEADS, QK_HEAD_DIM), kv_pk, kv_sc,
        kv_indptr, qo_indptr,
        mid_lo, mid_hi, mid_lse,
        SM_SCALE,
        NSPLIT=NSPLIT, BN=BN, NHEADS=NUM_HEADS,
        QK_DIM=QK_HEAD_DIM, V_PKD=V_PKD,
        PKD=PKD, N_SC=N_SC,
        QK_P=QK_PAD, PKD_P=PKD_PAD, NSC_P=NSC_PAD,
        num_warps=8, num_stages=2,
    )

    _stage2[(bs, NUM_HEADS)](
        mid_lo, mid_hi, mid_lse, o, qo_indptr,
        NSPLIT=NSPLIT, NHEADS=NUM_HEADS, V_PKD=V_PKD, V_DIM=V_HEAD_DIM,
        num_warps=4,
    )

    return o
