#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v029: Simplified FP4 conversion using direct IEEE 754 field extraction.
- Replace 3-way branch (saturate/denormal/normal) with single formula.
- Same approach as aiter's fused_mxfp4_quant_moe_sort kernel.
- Reduces branch divergence and register pressure in quant op.
- Target: quant phase (46%) — fewer instructions, less register pressure.
"""
import torch
import triton
import triton.language as tl
import aiter
from aiter import dtypes
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm

from task import input_t, output_t

# Pre-allocated buffers keyed by (M, K)
_buffers = {}

# ASM kernel name — 32x128 is optimal for all small-M shapes per tuned CSV analysis
_ASM_KERNEL_32x128 = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0,
    }
)
@triton.jit
def _fused_mxfp4_quant_shuffle_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_ITER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
    SCALE_N_PAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_n = tl.program_id(1) * NUM_ITER
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m_in, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n_in, tl.int64)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n

        if EVEN_M_N:
            x = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
        else:
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(
                tl.float32
            )

        # Inline MXFP4 quant with simplified FP4 conversion (no 3-way branch)
        x_3d = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)

        # Scale computation
        amax = tl.max(tl.abs(x_3d), axis=-1, keep_dims=True)
        amax = amax.to(tl.int32, bitcast=True)
        amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        amax = amax.to(tl.float32, bitcast=True)
        scale_e8m0_unbiased = tl.log2(amax).floor() - 2
        scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
        bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127
        quant_scale = tl.exp2(-scale_e8m0_unbiased)

        # Quantize
        qx = x_3d * quant_scale

        # Simplified FP32-to-FP4 conversion (same as MoE sort kernel)
        qx_uint = qx.to(tl.uint32, bitcast=True)
        s = qx_uint & 0x80000000
        e = (qx_uint >> 23) & 0xFF
        m = qx_uint & 0x7FFFFF

        EXP_BIAS_FP32: tl.constexpr = 127
        EXP_BIAS_FP4: tl.constexpr = 1

        # Handle denormals: shift mantissa for sub-normal FP4 values
        adjusted_exponents = tl.core.sub(EXP_BIAS_FP32, e + 1, sanitize_overflow=False)
        m = tl.where(e < EXP_BIAS_FP32, (0x400000 | (m >> 1)) >> adjusted_exponents, m)

        # Rebias exponent from FP32 to FP4
        e = tl.maximum(e, EXP_BIAS_FP32 - EXP_BIAS_FP4) - (EXP_BIAS_FP32 - EXP_BIAS_FP4)

        # Combine sign, exponent, and mantissa with rounding
        e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((s >> 28) | e2m1_tmp).to(tl.uint8)

        # Pack pairs of FP4 values into uint8
        e2m1_value = tl.reshape(
            e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
        )
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)
        out_tensor = out_tensor.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)
        bs_e8m0 = bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

        # Store fp4 output
        out_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_offs_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
        out_offs = (
            out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )

        if EVEN_M_N:
            tl.store(x_fp4_ptr + out_offs, out_tensor)
        else:
            out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        # Store scales with inline shuffle permutation
        bs_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
        num_bs_cols = (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE

        bs_offs_0 = bs_offs_m[:, None] // 32
        bs_offs_1 = bs_offs_m[:, None] % 32
        bs_offs_2 = bs_offs_1 % 16
        bs_offs_1 = bs_offs_1 // 16
        bs_offs_3 = bs_offs_n[None, :] // 8
        bs_offs_4 = bs_offs_n[None, :] % 8
        bs_offs_5 = bs_offs_4 % 4
        bs_offs_4 = bs_offs_4 // 4
        bs_offs = (
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * SCALE_N_PAD
        )

        bs_mask_valid = (bs_offs_m < M)[:, None] & (bs_offs_n < num_bs_cols)[None, :]
        bs_e8m0 = tl.where(bs_mask_valid, bs_e8m0, 127)

        SCALE_M_PAD = (M + 255) // 256 * 256
        bs_mask = (bs_offs_m < SCALE_M_PAD)[:, None] & (bs_offs_n < SCALE_N_PAD)[
            None, :
        ]
        tl.store(
            bs_ptr + bs_offs,
            bs_e8m0.to(tl.uint8),
            mask=bs_mask,
            cache_modifier=".cg",
        )


def _get_or_create_buffers(M, K, N, device):
    """Get pre-allocated quant + GEMM buffers for given shape."""
    key = (M, K, N)
    if key not in _buffers:
        MXFP4_QUANT_BLOCK_SIZE = 32
        SCALE_N_valid = triton.cdiv(K, MXFP4_QUANT_BLOCK_SIZE)
        SCALE_M = triton.cdiv(M, 256) * 256
        SCALE_N = triton.cdiv(SCALE_N_valid, 8) * 8

        # Block size config for quant kernel
        # BSN=64 NW=2: double work per block with 2 warps for latency hiding.
        NUM_ITER = 1
        BLOCK_SIZE_M = min(32, triton.next_power_of_2(M))
        BLOCK_SIZE_N = 64
        NUM_WARPS = 2
        NUM_STAGES = 1

        BLOCK_SIZE_M = triton.cdiv(BLOCK_SIZE_M, 32) * 32
        BLOCK_SIZE_N = triton.cdiv(BLOCK_SIZE_N, 32) * 32

        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(K, BLOCK_SIZE_N * NUM_ITER),
        )

        # GEMM output must be padded to multiple of 32
        padded_M = (M + 31) // 32 * 32

        _buffers[key] = {
            'x_fp4': torch.empty((M, K // 2), dtype=torch.uint8, device=device),
            'blockscale': torch.empty((SCALE_M, SCALE_N), dtype=torch.uint8, device=device),
            'gemm_out': torch.empty((padded_M, N), dtype=torch.bfloat16, device=device),
            'SCALE_N': SCALE_N,
            'BLOCK_SIZE_M': BLOCK_SIZE_M,
            'BLOCK_SIZE_N': BLOCK_SIZE_N,
            'NUM_ITER': NUM_ITER,
            'NUM_STAGES': NUM_STAGES,
            'NUM_WARPS': NUM_WARPS,
            'grid': grid,
            'M': M,
        }
    return _buffers[key]


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_shuffle.shape[0]

    # Get pre-allocated buffers and cached config
    buf = _get_or_create_buffers(M, K, N, A.device)

    _fused_mxfp4_quant_shuffle_kernel[buf['grid']](
        A,
        buf['x_fp4'],
        buf['blockscale'],
        *A.stride(),
        *buf['x_fp4'].stride(),
        M=M,
        N=K,
        BLOCK_SIZE_M=buf['BLOCK_SIZE_M'],
        BLOCK_SIZE_N=buf['BLOCK_SIZE_N'],
        NUM_ITER=buf['NUM_ITER'],
        NUM_STAGES=buf['NUM_STAGES'],
        MXFP4_QUANT_BLOCK_SIZE=32,
        SCALING_MODE=0,
        SCALE_N_PAD=buf['SCALE_N'],
        num_warps=buf['NUM_WARPS'],
        waves_per_eu=0,
        num_stages=1,
    )

    # Direct ASM call with 32x128 kernel — optimal for all small-M shapes
    gemm_a4w4_asm(
        buf['x_fp4'].view(dtypes.fp4x2),
        B_shuffle,
        buf['blockscale'].view(dtypes.fp8_e8m0),
        B_scale_sh,
        buf['gemm_out'],
        _ASM_KERNEL_32x128,
        None,  # bias
        1.0,   # alpha
        0.0,   # beta
        True,  # bpreshuffle
        log2_k_split=0,
    )

    return buf['gemm_out'][:M]
