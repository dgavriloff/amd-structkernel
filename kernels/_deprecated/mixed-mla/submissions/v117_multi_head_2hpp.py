#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v117: Multi-head per program (2 heads) to share KV loads.
- Each program handles 2 Q heads, loading K and V once.
- 2 separate dot_scaled calls share the same K data.
- MXFP4 dot_scaled for Q*K^T, FP8 for V.
- Adaptive NSPLIT (8 for kv<=2048, 16 otherwise).
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
HEADS_PER_PROG = 2

_cache = {}


@triton.jit
def _stage1(
    Q, KV_pk, KV_sc, KV_fp8, kv_indptr, qo_indptr,
    Mid_v, Mid_lse,
    sm_scale, fp8_scale,
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
    N_HEAD_GROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    hgid = pid % N_HEAD_GROUPS
    pid2 = pid // N_HEAD_GROUPS
    sid = pid2 % NSPLIT
    bid = pid2 // NSPLIT

    kv_s = tl.load(kv_indptr + bid)
    kv_e = tl.load(kv_indptr + bid + 1)
    kv_len = kv_e - kv_s
    qi = tl.load(qo_indptr + bid)

    per_split = tl.cdiv(kv_len, NSPLIT)
    ss = sid * per_split
    se = tl.minimum(ss + per_split, kv_len)

    hid0 = hgid * 2
    hid1 = hid0 + 1

    lse_off_0 = (qi * NSPLIT + sid) * NHEADS + hid0
    lse_off_1 = (qi * NSPLIT + sid) * NHEADS + hid1

    if ss >= kv_len:
        tl.store(Mid_lse + lse_off_0, float("-inf"))
        tl.store(Mid_lse + lse_off_1, float("-inf"))
        ov = tl.arange(0, V_DIM)
        tl.store(Mid_v + lse_off_0 * V_DIM + ov, tl.zeros([V_DIM], dtype=tl.bfloat16))
        tl.store(Mid_v + lse_off_1 * V_DIM + ov, tl.zeros([V_DIM], dtype=tl.bfloat16))
        return

    # Load Q for 2 heads: each [1, QK_P] bf16
    ok = tl.arange(0, QK_P)
    q0 = tl.load(Q + qi * NHEADS * QK_DIM + hid0 * QK_DIM + ok,
                  mask=ok < QK_DIM, other=0.0)
    q1 = tl.load(Q + qi * NHEADS * QK_DIM + hid1 * QK_DIM + ok,
                  mask=ok < QK_DIM, other=0.0)

    m_i_0 = float("-inf")
    l_i_0 = 0.0
    acc_v_0 = tl.zeros([V_DIM], dtype=tl.float32)
    m_i_1 = float("-inf")
    l_i_1 = 0.0
    acc_v_1 = tl.zeros([V_DIM], dtype=tl.float32)

    for blk_s in range(ss, se, BN):
        bn = tl.minimum(BN, se - blk_s)
        on = tl.arange(0, BN)
        nm = on < bn
        kg = kv_s + blk_s + on

        # K packed [BN, PKD_P] and scales [BN, NSC_P] -- shared
        opk = tl.arange(0, PKD_P)
        kp = tl.load(KV_pk + kg[:, None] * PKD + opk[None, :],
                      mask=nm[:, None] & (opk[None, :] < PKD), other=0)
        osc = tl.arange(0, NSC_P)
        ks = tl.load(KV_sc + kg[:, None] * N_SC + osc[None, :],
                      mask=nm[:, None] & (osc[None, :] < N_SC), other=127)
        kpt = tl.trans(kp)

        # Score head 0: [1, QK_P] @ [PKD_P, BN] -> [1, BN]
        s0_2d = tl.dot_scaled(q0[None, :].to(tl.bfloat16), None, "bf16",
                               kpt, ks, "e2m1")
        s0 = tl.reshape(s0_2d, [BN]) * sm_scale
        s0 = tl.where(nm, s0, float("-inf"))

        # Score head 1: [1, QK_P] @ [PKD_P, BN] -> [1, BN]
        s1_2d = tl.dot_scaled(q1[None, :].to(tl.bfloat16), None, "bf16",
                               kpt, ks, "e2m1")
        s1 = tl.reshape(s1_2d, [BN]) * sm_scale
        s1 = tl.where(nm, s1, float("-inf"))

        # V from FP8 -- shared
        ov = tl.arange(0, V_DIM)
        fp8_offsets = (kg[:, None] * FP8_STRIDE + ov[None, :]).to(tl.int64)
        vfp8 = tl.load(KV_fp8 + fp8_offsets, mask=nm[:, None], other=0.0)
        v_f32 = vfp8.to(tl.float32) * fp8_scale

        # Head 0 softmax + V accum
        m_ij_0 = tl.max(s0, axis=0)
        m_new_0 = tl.maximum(m_i_0, m_ij_0)
        alpha_0 = tl.exp(m_i_0 - m_new_0)
        p_0 = tl.exp(s0 - m_new_0)
        l_i_0 = l_i_0 * alpha_0 + tl.sum(p_0, axis=0)
        acc_v_0 = acc_v_0 * alpha_0 + tl.sum(p_0[:, None] * v_f32, axis=0)
        m_i_0 = m_new_0

        # Head 1 softmax + V accum
        m_ij_1 = tl.max(s1, axis=0)
        m_new_1 = tl.maximum(m_i_1, m_ij_1)
        alpha_1 = tl.exp(m_i_1 - m_new_1)
        p_1 = tl.exp(s1 - m_new_1)
        l_i_1 = l_i_1 * alpha_1 + tl.sum(p_1, axis=0)
        acc_v_1 = acc_v_1 * alpha_1 + tl.sum(p_1[:, None] * v_f32, axis=0)
        m_i_1 = m_new_1

    ov = tl.arange(0, V_DIM)

    # Store head 0
    inv_l_0 = 1.0 / l_i_0
    tl.store(Mid_lse + lse_off_0, m_i_0 + tl.log(l_i_0))
    tl.store(Mid_v + lse_off_0 * V_DIM + ov, (acc_v_0 * inv_l_0).to(tl.bfloat16))

    # Store head 1
    inv_l_1 = 1.0 / l_i_1
    tl.store(Mid_lse + lse_off_1, m_i_1 + tl.log(l_i_1))
    tl.store(Mid_v + lse_off_1 * V_DIM + ov, (acc_v_1 * inv_l_1).to(tl.bfloat16))


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

    # MXFP4 for K (dot_scaled)
    kv_pk_raw, kv_sc_raw = kv_data["mxfp4"]
    kv_pk = kv_pk_raw.view(torch.uint8) if kv_pk_raw.dtype != torch.uint8 else kv_pk_raw
    kv_sc = kv_sc_raw.view(torch.uint8) if kv_sc_raw.dtype != torch.uint8 else kv_sc_raw
    if kv_pk.dim() > 2:
        kv_pk = kv_pk.reshape(kv_pk.shape[0], -1)
    if kv_sc.dim() > 2:
        kv_sc = kv_sc.reshape(kv_sc.shape[0], -1)

    # FP8 for V
    kv_fp8_raw, kv_fp8_scale = kv_data["fp8"]
    kv_fp8 = kv_fp8_raw.view(torch.float8_e4m3fnuz).reshape(kv_fp8_raw.shape[0], -1)
    fp8_scale_val = kv_fp8_scale.item() if kv_fp8_scale.numel() == 1 else 1.0

    PKD = kv_pk.shape[-1]
    N_SC = kv_sc.shape[-1]
    FP8_STRIDE = kv_fp8.shape[-1]

    # Adaptive NSPLIT
    max_kv = int(kv_indptr[-1].item() - kv_indptr[0].item()) // max(bs, 1)
    if max_kv <= 2048:
        NSPLIT = 8
    else:
        NSPLIT = 16
    BN = 64

    N_HEAD_GROUPS = NUM_HEADS // HEADS_PER_PROG  # 8

    n_mid_v = tq * NSPLIT * NUM_HEADS * V_HEAD_DIM
    mid_v = torch.empty(n_mid_v, dtype=torch.bfloat16, device="cuda")
    mid_lse = torch.empty(tq * NSPLIT * NUM_HEADS, dtype=torch.float32, device="cuda")
    o = torch.empty((tq, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

    _stage1[(bs * NSPLIT * N_HEAD_GROUPS,)](
        q.view(-1, NUM_HEADS, QK_HEAD_DIM), kv_pk, kv_sc, kv_fp8,
        kv_indptr, qo_indptr,
        mid_v, mid_lse,
        SM_SCALE, fp8_scale_val,
        NSPLIT=NSPLIT, BN=BN, NHEADS=NUM_HEADS,
        QK_DIM=QK_HEAD_DIM, V_DIM=V_HEAD_DIM,
        FP8_STRIDE=FP8_STRIDE,
        PKD=PKD, N_SC=N_SC,
        QK_P=QK_PAD, PKD_P=PKD_PAD, NSC_P=NSC_PAD,
        N_HEAD_GROUPS=N_HEAD_GROUPS,
        num_warps=4, num_stages=2,
    )

    _stage2[(bs, NUM_HEADS)](
        mid_v, mid_lse, o, qo_indptr,
        NSPLIT=NSPLIT, NHEADS=NUM_HEADS, V_DIM=V_HEAD_DIM,
        num_warps=4,
    )

    return o
