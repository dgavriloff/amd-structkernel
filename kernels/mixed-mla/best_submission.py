#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v113: Adaptive NSPLIT + full V dequant + single mid buffer.
- NSPLIT=8 for kv<=2048, NSPLIT=16 for larger. Reduces stage2 overhead for small kv.
- Full 512-dim V output in stage1 (interleave lo/hi immediately), single mid_v buffer.
- Stage2 loads 512 bf16 values directly, no interleaving needed.
- BN=64, per-head grid for stage1.
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
def _fp4_dequant(packed, scale_f32):
    """Dequant packed uint8 fp4x2. Returns (lo_val, hi_val) fp32."""
    lo_raw = (packed & 0xF).to(tl.int32)
    hi_raw = (packed >> 4).to(tl.int32)
    lo_mag = lo_raw & 7
    hi_mag = hi_raw & 7
    lo_e = lo_mag >> 1
    lo_m = lo_mag & 1
    lo_val = tl.where(lo_e == 0,
                      lo_m.to(tl.float32) * 0.5,
                      tl.exp2((lo_e - 1).to(tl.float32)) * (1.0 + lo_m.to(tl.float32) * 0.5))
    lo_val = tl.where(lo_raw >= 8, -lo_val, lo_val) * scale_f32
    hi_e = hi_mag >> 1
    hi_m = hi_mag & 1
    hi_val = tl.where(hi_e == 0,
                      hi_m.to(tl.float32) * 0.5,
                      tl.exp2((hi_e - 1).to(tl.float32)) * (1.0 + hi_m.to(tl.float32) * 0.5))
    hi_val = tl.where(hi_raw >= 8, -hi_val, hi_val) * scale_f32
    return lo_val, hi_val


@triton.jit
def _stage1(
    Q, KV_pk, KV_sc, kv_indptr, qo_indptr,
    Mid_v, Mid_lse,
    sm_scale,
    NSPLIT: tl.constexpr,
    BN: tl.constexpr,
    NHEADS: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_PKD: tl.constexpr,
    V_DIM: tl.constexpr,
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

    # Load Q for this head: [QK_P] bf16
    ok = tl.arange(0, QK_P)
    q_vec = tl.load(Q + qi * NHEADS * QK_DIM + hid * QK_DIM + ok,
                     mask=ok < QK_DIM, other=0.0)

    m_i = float("-inf")
    l_i = 0.0
    acc_lo = tl.zeros([V_PKD], dtype=tl.float32)
    acc_hi = tl.zeros([V_PKD], dtype=tl.float32)

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

        # Score: [1, QK_P] @ [PKD_P, BN] -> [1, BN]
        q_2d = q_vec[None, :]
        kpt = tl.trans(kp)
        scores_2d = tl.dot_scaled(q_2d.to(tl.bfloat16), None, "bf16",
                                   kpt, ks, "e2m1")
        scores = tl.reshape(scores_2d, [BN])
        scores = scores * sm_scale
        scores = tl.where(nm, scores, float("-inf"))

        # Online softmax
        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc_lo = acc_lo * alpha
        acc_hi = acc_hi * alpha

        # V dequant [BN, V_PKD]
        ovpk = tl.arange(0, V_PKD)
        vp = tl.load(KV_pk + kg[:, None] * PKD + ovpk[None, :],
                      mask=nm[:, None], other=0)
        sidx = ovpk // 16
        vsc = tl.load(KV_sc + kg[:, None] * N_SC + sidx[None, :],
                       mask=nm[:, None], other=127)
        vsc_f = tl.exp2((vsc.to(tl.int32) - 127).to(tl.float32))
        vlo, vhi = _fp4_dequant(vp, vsc_f)

        # V accum: p[BN] @ V[BN, V_PKD] -> [V_PKD]
        acc_lo += tl.sum(p[:, None] * vlo, axis=0)
        acc_hi += tl.sum(p[:, None] * vhi, axis=0)
        m_i = m_new

    inv_l = 1.0 / l_i
    tl.store(Mid_lse + lse_off, m_i + tl.log(l_i))

    # Interleave lo/hi into full V_DIM bf16 and store
    ov = tl.arange(0, V_PKD)
    v_lo_bf16 = (acc_lo * inv_l).to(tl.bfloat16)
    v_hi_bf16 = (acc_hi * inv_l).to(tl.bfloat16)
    tl.store(Mid_v + v_off + ov * 2, v_lo_bf16)
    tl.store(Mid_v + v_off + ov * 2 + 1, v_hi_bf16)


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

    PKD = kv_pk.shape[-1]
    N_SC = kv_sc.shape[-1]
    V_PKD = V_HEAD_DIM // 2

    # Adaptive NSPLIT based on kv_seq_len
    max_kv = int(kv_indptr[-1].item() - kv_indptr[0].item()) // max(bs, 1)
    if max_kv <= 2048:
        NSPLIT = 8
    else:
        NSPLIT = 16
    BN = 64

    # Single mid_v buffer: bf16 (halves bandwidth vs f32)
    n_mid_v = tq * NSPLIT * NUM_HEADS * V_HEAD_DIM
    mid_v = torch.empty(n_mid_v, dtype=torch.bfloat16, device="cuda")
    mid_lse = torch.empty(tq * NSPLIT * NUM_HEADS, dtype=torch.float32, device="cuda")
    o = torch.empty((tq, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    _stage1[(bs * NSPLIT * NUM_HEADS,)](
        q.view(-1, NUM_HEADS, QK_HEAD_DIM), kv_pk, kv_sc,
        kv_indptr, qo_indptr,
        mid_v, mid_lse,
        SM_SCALE,
        NSPLIT=NSPLIT, BN=BN, NHEADS=NUM_HEADS,
        QK_DIM=QK_HEAD_DIM, V_PKD=V_PKD, V_DIM=V_HEAD_DIM,
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
