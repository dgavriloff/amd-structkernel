#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v133: Multi-head (16 heads/program) Triton kernel.
- MXFP4 K via dot_scaled for Q*K^T: [16, QK_P] @ [PKD_P, BN] -> [16, BN]
- FP8 V via tl.dot for P@V: [16, BN] @ [BN, 512] -> [16, 512] (MFMA hw)
- Load KV once per block, reuse across all 16 heads (MQA broadcast).
- BN=32 to manage register pressure with 16-head accumulator.
- Adaptive NSPLIT (8 for kv<=2048, 16 otherwise).
- All caching optimizations from v123.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

NUM_HEADS = 16
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
QK_PAD = 1024
PKD_PAD = 512
NSC_PAD = 32

_buf_cache = {}
_nsplit_cache = {}


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
    # Grid: (batch * NSPLIT,)
    # Each program handles ALL 16 heads for one (batch, split)
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

    # Output layout: Mid_lse[qi, sid, head], Mid_v[qi, sid, head, v_dim]
    lse_base = (qi * NSPLIT + sid) * NHEADS
    v_base = lse_base * V_DIM

    oh = tl.arange(0, NHEADS)

    if ss >= kv_len:
        tl.store(Mid_lse + lse_base + oh, tl.full([NHEADS], float("-inf"), dtype=tl.float32))
        ov = tl.arange(0, V_DIM)
        for h in tl.static_range(NHEADS):
            tl.store(Mid_v + v_base + h * V_DIM + ov, tl.zeros([V_DIM], dtype=tl.bfloat16))
        return

    # Load fp8 scale from device memory
    fp8_scale = tl.load(FP8_SCALE_PTR).to(tl.float32)

    # Load Q for ALL heads: [NHEADS, QK_P]
    ok = tl.arange(0, QK_P)
    q_all = tl.load(Q + qi * NHEADS * QK_DIM + oh[:, None] * QK_DIM + ok[None, :],
                     mask=ok[None, :] < QK_DIM, other=0.0)  # [NHEADS, QK_P] bf16

    # Initialize online softmax state for all heads
    m_i = tl.full([NHEADS], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([NHEADS], dtype=tl.float32)
    acc_v = tl.zeros([NHEADS, V_DIM], dtype=tl.float32)

    ov = tl.arange(0, V_DIM)

    for blk_s in range(ss, se, BN):
        bn = tl.minimum(BN, se - blk_s)
        on = tl.arange(0, BN)
        nm = on < bn
        kg = kv_s + blk_s + on

        # Load K MXFP4: packed [BN, PKD_P] and scale [BN, NSC_P]
        opk = tl.arange(0, PKD_P)
        kp = tl.load(KV_pk + kg[:, None] * PKD + opk[None, :],
                      mask=nm[:, None] & (opk[None, :] < PKD), other=0)
        osc = tl.arange(0, NSC_P)
        ks = tl.load(KV_sc + kg[:, None] * N_SC + osc[None, :],
                      mask=nm[:, None] & (osc[None, :] < N_SC), other=127)

        # Q*K^T via dot_scaled: [NHEADS, QK_P] @ [PKD_P, BN] -> [NHEADS, BN]
        kpt = tl.trans(kp)
        scores = tl.dot_scaled(q_all.to(tl.bfloat16), None, "bf16",
                                kpt, ks, "e2m1")
        scores = scores * 0.041666666666666664  # 1/sqrt(576)
        scores = tl.where(nm[None, :], scores, float("-inf"))

        # Online softmax
        m_ij = tl.max(scores, axis=1)  # [NHEADS]
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])  # [NHEADS, BN]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_v = acc_v * alpha[:, None]

        # Load V from FP8: [BN, V_DIM]
        fp8_offsets = kg[:, None] * FP8_STRIDE + ov[None, :]
        vfp8 = tl.load(KV_fp8 + fp8_offsets,
                        mask=nm[:, None], other=0.0)
        v_bf16 = (vfp8.to(tl.float32) * fp8_scale).to(tl.bfloat16)

        # V accumulation via tl.dot: [NHEADS, BN] @ [BN, V_DIM] -> [NHEADS, V_DIM]
        acc_v += tl.dot(p.to(tl.bfloat16), v_bf16).to(tl.float32)
        m_i = m_new

    # Normalize and store
    inv_l = 1.0 / l_i
    lse_vals = m_i + tl.log(l_i)
    tl.store(Mid_lse + lse_base + oh, lse_vals)

    # Store V results per head
    acc_norm = acc_v * inv_l[:, None]
    for h in tl.static_range(NHEADS):
        mask_h = oh == h  # [NHEADS] bool
        v_h = tl.sum(acc_norm * mask_h[:, None], axis=0)  # [V_DIM]
        tl.store(Mid_v + v_base + h * V_DIM + ov, v_h.to(tl.bfloat16))


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
    fp8_scale_tensor = kv_fp8_scale.view(-1)

    PKD = kv_pk.shape[-1]
    N_SC = kv_sc.shape[-1]
    FP8_STRIDE = kv_fp8.shape[-1]

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
            torch.empty(n_mid_v, dtype=torch.bfloat16, device="cuda"),
            torch.empty(tq * NSPLIT * NUM_HEADS, dtype=torch.float32, device="cuda"),
            torch.empty((tq, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"),
        )
    mid_v, mid_lse, o = _buf_cache[buf_key]

    # Grid: (batch * NSPLIT,) -- each program handles all 16 heads
    _stage1[(bs * NSPLIT,)](
        q, kv_pk, kv_sc, kv_fp8,
        kv_indptr, qo_indptr,
        mid_v, mid_lse,
        fp8_scale_tensor,
        NSPLIT=NSPLIT, BN=32, NHEADS=NUM_HEADS,
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
