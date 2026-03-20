#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
v137: Pure MXFP4 kernel — use MXFP4 for both K (dot_scaled) and V (manual dequant).
- K: dot_scaled with full 576 MXFP4 dims (unchanged from v123)
- V: manual fp4 dequant of first 512 dims from MXFP4 packed data
  - LUT-based fp4 nibble -> f32 dequant (16-entry table, no branching)
  - Two 256-wide accumulators for even/odd V elements, interleaved on store
  - Reuse scale data already fetched for K (from L2 cache)
- Eliminates all FP8 V loads — saves 512B/token, costs 256B/token MXFP4 (net -256B)
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

NUM_HEADS = 16
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
V_HALF = 256  # V_HEAD_DIM // 2
QK_PAD = 1024
PKD_PAD = 512
NSC_PAD = 32

_buf_cache = {}

# E2M1 FP4 lookup table: nibble -> float value
# 0->0, 1->0.5, 2->1.0, 3->1.5, 4->2.0, 5->3.0, 6->4.0, 7->6.0
# 8->-0, 9->-0.5, 10->-1.0, 11->-1.5, 12->-2.0, 13->-3.0, 14->-4.0, 15->-6.0
FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@triton.jit
def _dequant_e2m1_lut(nibbles):
    """Dequant E2M1 fp4 nibbles using arithmetic (no LUT needed in Triton)."""
    sign = (nibbles >> 3) & 1
    exp_bits = (nibbles >> 1) & 3
    mant_bit = nibbles & 1
    is_sub = (exp_bits == 0)
    val = tl.where(
        is_sub,
        mant_bit.to(tl.float32) * 0.5,
        (1.0 + mant_bit.to(tl.float32) * 0.5) * tl.exp2((exp_bits - 1).to(tl.float32))
    )
    return tl.where(sign != 0, -val, val)


@triton.jit
def _stage1(
    Q, KV_pk, KV_sc, kv_indptr, qo_indptr,
    Mid_v, Mid_lse,
    NSPLIT: tl.constexpr,
    BN: tl.constexpr,
    NHEADS: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    PKD: tl.constexpr,
    N_SC: tl.constexpr,
    QK_P: tl.constexpr,
    PKD_P: tl.constexpr,
    NSC_P: tl.constexpr,
    V_HALF_C: tl.constexpr,
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
    acc_lo = tl.zeros([V_HALF_C], dtype=tl.float32)
    acc_hi = tl.zeros([V_HALF_C], dtype=tl.float32)

    # Precompute scale index mapping for V: packed byte j -> scale block j//16
    ovp = tl.arange(0, V_HALF_C)
    scale_idx = ovp // 16  # [256]

    for blk_s in range(ss, se, BN):
        bn = tl.minimum(BN, se - blk_s)
        on = tl.arange(0, BN)
        nm = on < bn
        kg = kv_s + blk_s + on

        # ===== K: MXFP4 dot_scaled for attention scores =====
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
        acc_lo = acc_lo * alpha
        acc_hi = acc_hi * alpha

        # ===== V: MXFP4 manual dequant of first 512 dims =====
        # Load V packed bytes (first 256 bytes of each row)
        # These overlap with kp[:, :256] and will hit L2 cache
        vp = tl.load(KV_pk + kg[:, None] * PKD + ovp[None, :],
                      mask=nm[:, None], other=0)  # [BN, 256] uint8

        # Load V scales (first 16 scales, gathered by scale_idx pattern)
        vs = tl.load(KV_sc + kg[:, None] * N_SC + scale_idx[None, :],
                      mask=nm[:, None], other=127)  # [BN, 256] uint8
        v_scale = tl.exp2(vs.to(tl.float32) - 127.0)  # [BN, 256]

        # Unpack and dequant
        lo_nib = vp & 0xF
        hi_nib = (vp >> 4) & 0xF
        lo_val = _dequant_e2m1_lut(lo_nib) * v_scale  # [BN, 256]
        hi_val = _dequant_e2m1_lut(hi_nib) * v_scale  # [BN, 256]

        # Weighted sum: p[BN] @ v[BN, 256] -> [256]
        acc_lo += tl.sum(p[:, None] * lo_val, axis=0)
        acc_hi += tl.sum(p[:, None] * hi_val, axis=0)

        m_i = m_new

    inv_l = 1.0 / l_i
    tl.store(Mid_lse + lse_off, m_i + tl.log(l_i))

    # Interleave even/odd into V_DIM output
    oj = tl.arange(0, V_HALF_C)
    tl.store(Mid_v + v_off + oj * 2, (acc_lo * inv_l).to(tl.bfloat16))
    tl.store(Mid_v + v_off + oj * 2 + 1, (acc_hi * inv_l).to(tl.bfloat16))


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
    kv_seq_len = config["kv_seq_len"]
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

    NSPLIT = 8 if kv_seq_len <= 2048 else 16

    buf_key = (tq, NSPLIT)
    if buf_key not in _buf_cache:
        n_mid_v = tq * NSPLIT * NUM_HEADS * V_HEAD_DIM
        _buf_cache[buf_key] = (
            torch.empty(n_mid_v, dtype=torch.bfloat16, device="cuda"),
            torch.empty(tq * NSPLIT * NUM_HEADS, dtype=torch.float32, device="cuda"),
            torch.empty((tq, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"),
        )
    mid_v, mid_lse, o = _buf_cache[buf_key]

    _stage1[(bs * NSPLIT * NUM_HEADS,)](
        q, kv_pk, kv_sc,
        kv_indptr, qo_indptr,
        mid_v, mid_lse,
        NSPLIT=NSPLIT, BN=64, NHEADS=NUM_HEADS,
        QK_DIM=QK_HEAD_DIM, V_DIM=V_HEAD_DIM,
        PKD=PKD, N_SC=N_SC,
        QK_P=QK_PAD, PKD_P=PKD_PAD, NSC_P=NSC_PAD,
        V_HALF_C=V_HALF,
        num_warps=4, num_stages=2,
    )

    _stage2[(bs, NUM_HEADS)](
        mid_v, mid_lse, o, qo_indptr,
        NSPLIT=NSPLIT, NHEADS=NUM_HEADS, V_DIM=V_HEAD_DIM,
        num_warps=4,
    )

    return o
