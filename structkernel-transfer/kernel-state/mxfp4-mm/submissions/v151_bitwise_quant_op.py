#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v151: Custom inline quant op with bitwise exponent extraction for M=256 two-phase path.

Hypothesis: The library _mxfp4_quant_op uses tl.log2(amax).floor() and tl.exp2(-scale)
which are transcendental functions. Since amax already has mantissa cleared (it's a power
of 2), we can extract the exponent with a simple bitshift: ((amax_bits >> 23) & 0xFF) - 127.
And tl.exp2(-scale_e8m0_unbiased) can be replaced by constructing a float from the exponent.
This eliminates 2 transcendental function calls per quant block in the M=256 quant kernel.
"""
import torch
import triton
import triton.language as tl
from aiter import dtypes
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4 import (
    _gemm_a16wfp4_preshuffle_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_reduce_kernel,
)
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import get_splitk
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

from task import input_t, output_t

# Pre-allocated buffers keyed by (M, K, N)
_buffers = {}

# ASM kernel name — 32x128 is optimal for all small-M shapes per tuned CSV analysis
_ASM_KERNEL_32x128 = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"

# Threshold: use fused for M <= this value
_FUSED_M_THRESHOLD = 64


def _get_fused_config(M, N, K):
    """Get shape-specific config for fused quant+GEMM path.
    All configs use BSK=256 num_stages=2 for Triton software pipelining.
    """
    if K > 4096:
        # Custom split-K=7 BSK=256 for large-K shapes (e.g., 16x2112x7168)
        # BSM=8: 238 blocks (0.93 waves) vs BSM=16: 119 blocks (0.46 waves)
        return {
            "BLOCK_SIZE_M": 8,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 7,
        }
    if M <= 4:
        return {
            "BLOCK_SIZE_M": 4,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 1,
        }
    elif M <= 8:
        return {
            "BLOCK_SIZE_M": 8,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 1,
        }
    elif M <= 32 and K <= 1024:
        return {
            "BLOCK_SIZE_M": 8,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 1,
        }
    elif M <= 32:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": None,
            "NUM_KSPLIT": 1,
        }
    else:
        # M=64 (64x7168x2048): BSM=16 BSN=128 BSK=256 NW=4 NS=2
        # 4*56=224 blocks, 8 K-iters with pipelining
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 1,
        }


@triton.jit
def _fast_mxfp4_quant_op(
    x,
    BLOCK_SIZE_N,
    BLOCK_SIZE_M,
    MXFP4_QUANT_BLOCK_SIZE,
):
    """
    Optimized MXFP4 quant using bitwise exponent extraction instead of tl.log2/tl.exp2.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32
    """
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)

    # Calculate scale - bitwise version
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax_bits = amax.to(tl.int32, bitcast=True)
    # Round up mantissa to next power of 2 (same as library)
    amax_bits = (amax_bits + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000

    # Extract exponent directly via bitshift instead of tl.log2().floor()
    # For a power-of-2 float, exponent = ((bits >> 23) & 0xFF) - 127
    amax_exp = ((amax_bits >> 23) & 0xFF).to(tl.int32)
    scale_e8m0_unbiased = amax_exp - EXP_BIAS_FP32 - 2
    # tl.clamp doesn't support int32, use manual min/max
    scale_e8m0_unbiased = tl.where(scale_e8m0_unbiased < -127, -127, scale_e8m0_unbiased)
    scale_e8m0_unbiased = tl.where(scale_e8m0_unbiased > 127, 127, scale_e8m0_unbiased)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127

    # Construct quant_scale = 2^(-scale_e8m0_unbiased) via bitwise float construction
    # instead of tl.exp2(-scale_e8m0_unbiased)
    # For 2^e, the float32 bits are ((e + 127) << 23)
    neg_scale = -scale_e8m0_unbiased
    quant_scale_bits = ((neg_scale + EXP_BIAS_FP32) << 23).to(tl.uint32)
    quant_scale = quant_scale_bits.to(tl.float32, bitcast=True)

    # Compute quantized x
    qx = x * quant_scale

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign
    s = qx & 0x80000000
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormal numbers
    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    # Normal numbers
    normal_x = qx
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    # Merge results
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


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

        # Use optimized bitwise quant op instead of library _mxfp4_quant_op
        out_tensor, bs_e8m0 = _fast_mxfp4_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

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


def _prepare_splitk_dispatch(M, N, K, config, device):
    """Pre-compute all params for split-K direct dispatch (16x2112x7168)."""
    K_kernel = K // 2
    BSK = config["BLOCK_SIZE_K"]
    NUM_KSPLIT = config["NUM_KSPLIT"]

    SPLITK_BLOCK_SIZE, BSK, NUM_KSPLIT = get_splitk(K_kernel, BSK, NUM_KSPLIT)

    BSN = max(config["BLOCK_SIZE_N"], 32)
    BSM = config["BLOCK_SIZE_M"]

    grid_size = NUM_KSPLIT * triton.cdiv(M, BSM) * triton.cdiv(N, BSN)

    # Pre-allocate y_pp
    y_pp = torch.empty((NUM_KSPLIT, M, N), dtype=torch.float32, device=device)

    # Reduce kernel params — use REDUCE_BSN=16 for fp32 partials (library recommendation)
    REDUCE_BSM = 16
    REDUCE_BSN = 16  # Library says BSN=16 is best for fp32 partials
    ACTUAL_KSPLIT = triton.cdiv(K_kernel, (SPLITK_BLOCK_SIZE // 2))
    reduce_grid = (triton.cdiv(M, REDUCE_BSM), triton.cdiv(N, REDUCE_BSN))

    return {
        'BLOCK_SIZE_M': BSM,
        'BLOCK_SIZE_N': BSN,
        'BLOCK_SIZE_K': BSK,
        'GROUP_SIZE_M': config["GROUP_SIZE_M"],
        'NUM_KSPLIT': NUM_KSPLIT,
        'SPLITK_BLOCK_SIZE': SPLITK_BLOCK_SIZE,
        'num_warps': config["num_warps"],
        'num_stages': config["num_stages"],
        'waves_per_eu': config["waves_per_eu"],
        'matrix_instr_nonkdim': config["matrix_instr_nonkdim"],
        'cache_modifier': config["cache_modifier"],
        'grid_size': grid_size,
        'K_kernel': K_kernel,
        'y_pp': y_pp,
        'reduce_grid': reduce_grid,
        'REDUCE_BSM': REDUCE_BSM,
        'REDUCE_BSN': REDUCE_BSN,
        'ACTUAL_KSPLIT': ACTUAL_KSPLIT,
        'MAX_KSPLIT': triton.next_power_of_2(NUM_KSPLIT),
    }


def _get_or_create_buffers(M, K, N, device):
    """Get pre-allocated buffers for given shape."""
    key = (M, K, N)
    if key not in _buffers:
        if M <= _FUSED_M_THRESHOLD:
            config = _get_fused_config(M, N, K)
            if config["NUM_KSPLIT"] > 1:
                # Split-K path: use direct dispatch with tuned reduce kernel
                splitk_params = _prepare_splitk_dispatch(M, N, K, config, device)
                _buffers[key] = {
                    'mode': 'fused_splitk',
                    'out': torch.empty((M, N), dtype=torch.bfloat16, device=device),
                    'B_w': None,
                    'B_sc': None,
                    'splitk_params': splitk_params,
                }
            else:
                # Non-split-K: use wrapper (no reduce kernel overhead to optimize)
                _buffers[key] = {
                    'mode': 'fused',
                    'out': torch.empty((M, N), dtype=torch.bfloat16, device=device),
                    'B_w': None,
                    'B_sc': None,
                    'config': config,
                }
        else:
            MXFP4_QUANT_BLOCK_SIZE = 32
            SCALE_N_valid = triton.cdiv(K, MXFP4_QUANT_BLOCK_SIZE)
            SCALE_M = triton.cdiv(M, 256) * 256
            SCALE_N = triton.cdiv(SCALE_N_valid, 8) * 8

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

            padded_M = (M + 31) // 32 * 32

            _buffers[key] = {
                'mode': 'two_phase',
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

    buf = _get_or_create_buffers(M, K, N, A.device)

    if buf['mode'] == 'fused_splitk':
        # Split-K path with tuned reduce kernel (REDUCE_BSN=16)
        b_ptr = B_shuffle.data_ptr()
        if buf['B_w'] is None or buf.get('_b_ptr') != b_ptr:
            buf['B_w'] = B_shuffle.view(torch.uint8).reshape(N // 16, (K // 2) * 16)
            bs_shape = B_scale_sh.shape
            buf['B_sc'] = B_scale_sh.view(torch.uint8).reshape(
                bs_shape[0] // 32, bs_shape[1] * 32
            )
            buf['_b_ptr'] = b_ptr

        kp = buf['splitk_params']
        out = buf['out']
        y_pp = kp['y_pp']

        _gemm_a16wfp4_preshuffle_kernel[(kp['grid_size'],)](
            A,
            buf['B_w'],
            y_pp,
            buf['B_sc'],
            M,
            N,
            kp['K_kernel'],
            A.stride(0),
            A.stride(1),
            buf['B_w'].stride(0),
            buf['B_w'].stride(1),
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            buf['B_sc'].stride(0),
            buf['B_sc'].stride(1),
            BLOCK_SIZE_M=kp['BLOCK_SIZE_M'],
            BLOCK_SIZE_N=kp['BLOCK_SIZE_N'],
            BLOCK_SIZE_K=kp['BLOCK_SIZE_K'],
            GROUP_SIZE_M=kp['GROUP_SIZE_M'],
            NUM_KSPLIT=kp['NUM_KSPLIT'],
            SPLITK_BLOCK_SIZE=kp['SPLITK_BLOCK_SIZE'],
            num_warps=kp['num_warps'],
            num_stages=kp['num_stages'],
            waves_per_eu=kp['waves_per_eu'],
            matrix_instr_nonkdim=kp['matrix_instr_nonkdim'],
            PREQUANT=True,
            cache_modifier=kp['cache_modifier'],
        )

        _gemm_afp4wfp4_reduce_kernel[kp['reduce_grid']](
            y_pp,
            out,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            out.stride(0),
            out.stride(1),
            kp['REDUCE_BSM'],
            kp['REDUCE_BSN'],
            kp['ACTUAL_KSPLIT'],
            kp['MAX_KSPLIT'],
        )

        return out

    elif buf['mode'] == 'fused':
        # Non-split-K fused path: use wrapper
        b_ptr = B_shuffle.data_ptr()
        if buf['B_w'] is None or buf.get('_b_ptr') != b_ptr:
            buf['B_w'] = B_shuffle.view(torch.uint8).reshape(N // 16, (K // 2) * 16)
            bs_shape = B_scale_sh.shape
            buf['B_sc'] = B_scale_sh.view(torch.uint8).reshape(
                bs_shape[0] // 32, bs_shape[1] * 32
            )
            buf['_b_ptr'] = b_ptr

        return gemm_a16wfp4_preshuffle(
            A,
            buf['B_w'],
            buf['B_sc'],
            prequant=True,
            dtype=torch.bfloat16,
            y=buf['out'],
            config=buf['config'],
        )
    else:
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

        gemm_a4w4_asm(
            buf['x_fp4'].view(dtypes.fp4x2),
            B_shuffle,
            buf['blockscale'].view(dtypes.fp8_e8m0),
            B_scale_sh,
            buf['gemm_out'],
            _ASM_KERNEL_32x128,
            None,
            1.0,
            0.0,
            True,
            log2_k_split=0,
        )

        return buf['gemm_out'][:M]
