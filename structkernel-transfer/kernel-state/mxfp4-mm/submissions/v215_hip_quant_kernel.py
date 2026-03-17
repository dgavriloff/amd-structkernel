#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v215: Custom HIP quant kernel for M=256 path.

Hypothesis: Replace Triton quant kernel (~6.4us) with compiled HIP kernel
for M=256 two_phase path. Triton JIT has overhead from cache lookup,
specialization hash, and launch runtime. A pre-compiled HIP kernel bypasses
all of this. Uses torch.utils.cpp_extension.load_inline for compilation.
M<=64 paths completely unchanged.
"""
import os
import torch
import triton
import triton.language as tl
from aiter import dtypes
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4 import (
    _gemm_a16wfp4_preshuffle_kernel,
)
from aiter.ops.triton.gluon.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_reduce_kernel as _gluon_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_reduce_kernel,
)
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import get_splitk

from task import input_t, output_t

# Build custom HIP quant kernel at module scope
_hip_quant = None
try:
    from torch.utils.cpp_extension import load_inline
    _HIP_QUANT_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// MXFP4 quant kernel: bf16 input -> fp4x2 output + e8m0 scales (shuffled)
// Matches Triton convention: floor(log2(amax))-2+127 as direct exponent
// Each thread handles one row, processes K elements in groups of 32

__device__ __forceinline__ float bf16_to_float(unsigned short v) {
    unsigned int ui = ((unsigned int)v) << 16;
    return __uint_as_float(ui);
}

__device__ __forceinline__ unsigned char fp32_to_fp4(float val) {
    // Software FP4 conversion matching Triton _mxfp4_quant_op exactly
    unsigned int ui = __float_as_uint(val);
    unsigned int sign = ui & 0x80000000u;
    ui = ui ^ sign; // absolute value
    float absval = __uint_as_float(ui);

    unsigned char result;
    if (absval >= 6.0f) {
        // Saturate to max FP4 = 0x7 (6.0)
        result = 0x7;
    } else if (absval < 1.0f) {
        // Denormal: use magic number addition trick
        // denorm_exp = (127 - 1) + (23 - 1) + 1 = 149
        // denorm_mask_int = 149 << 23 = 0x4A800000
        float denorm = absval + __uint_as_float(0x4A800000u);
        unsigned int du = __float_as_uint(denorm);
        du -= 0x4A800000u;
        result = (unsigned char)(du);
    } else {
        // Normal: re-bias exponent and round
        unsigned int mant_odd = (ui >> 22) & 1u;
        // val_to_add = ((1 - 127) << 23) + (1 << 21) - 1 = 0xC0200000 + 0x001FFFFF
        // (1 - 127) = -126, (-126) << 23 in 2's complement uint32:
        // -126 << 23 as uint32 = 0xC1000000
        // Actually: ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
        // = ((1 - 127) << 23) + (1 << 21) - 1
        // = (-126 << 23) + 2097151
        // -126 in uint32 = 0xFFFFFF82, << 23 = 0xC1000000
        // 0xC1000000 + 0x001FFFFF = 0xC11FFFFF
        ui += 0xC11FFFFFu;
        ui += mant_odd;
        ui = ui >> 22;
        result = (unsigned char)(ui);
    }

    // Add sign (bit 3 of FP4)
    unsigned char sign_bit = (unsigned char)(sign >> 28);
    result = result | sign_bit;
    return result;
}

__global__ void mxfp4_quant_shuffle_kernel(
    const __hip_bfloat16* __restrict__ x_ptr,
    unsigned char* __restrict__ fp4_ptr,
    unsigned char* __restrict__ bs_ptr,
    int M, int K,
    int stride_x_m, int stride_x_k,
    int stride_fp4_m, int stride_fp4_k,
    int scale_n_pad
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    int tid_x = threadIdx.x;  // 0..31
    int num_groups = K / 32;

    // Process groups assigned to this thread within the row
    for (int g = tid_x; g < num_groups; g += blockDim.x) {
        // Load 32 bf16 values and find amax
        float vals[32];
        float amax_val = 0.0f;
        for (int i = 0; i < 32; i++) {
            int col = g * 32 + i;
            vals[i] = bf16_to_float(*(const unsigned short*)&x_ptr[row * stride_x_m + col * stride_x_k]);
            float av = fabsf(vals[i]);
            if (av > amax_val) amax_val = av;
        }

        // Compute scale: match Triton's amax rounding and scale computation
        // amax = (amax_as_int + 0x200000) & 0xFF800000  (round up mantissa)
        unsigned int amax_ui = __float_as_uint(amax_val);
        amax_ui = (amax_ui + 0x200000u) & 0xFF800000u;
        float amax_rounded = __uint_as_float(amax_ui);

        // scale_e8m0_unbiased = floor(log2(amax_rounded)) - 2
        float log2_amax = log2f(amax_rounded);
        float scale_unbiased = floorf(log2_amax) - 2.0f;
        if (scale_unbiased < -127.0f) scale_unbiased = -127.0f;
        if (scale_unbiased > 127.0f) scale_unbiased = 127.0f;

        // e8m0 scale value
        unsigned char bs_val = (unsigned char)((int)scale_unbiased + 127);

        // Quantization scale = 2^(-scale_unbiased)
        float quant_scale = exp2f(-scale_unbiased);

        // Quantize and convert to FP4, pack pairs
        for (int i = 0; i < 16; i++) {
            float qv0 = vals[i * 2] * quant_scale;
            float qv1 = vals[i * 2 + 1] * quant_scale;
            unsigned char fp4_0 = fp32_to_fp4(qv0);
            unsigned char fp4_1 = fp32_to_fp4(qv1);
            unsigned char packed = (fp4_0 & 0xF) | ((fp4_1 & 0xF) << 4);

            int out_col = g * 16 + i;
            fp4_ptr[row * stride_fp4_m + out_col * stride_fp4_k] = packed;
        }

        // Store scale with shuffle permutation
        // bs_offs_0 = row / 32
        // bs_offs_1 = (row % 32) / 16
        // bs_offs_2 = (row % 32) % 16
        // bs_offs_3 = g / 8
        // bs_offs_4 = (g % 8) / 4
        // bs_offs_5 = (g % 8) % 4
        // bs_offs = bs_offs_1 + bs_offs_4*2 + bs_offs_2*4 + bs_offs_5*128 + bs_offs_3*512 + bs_offs_0*32*scale_n_pad
        int r32 = row / 32;
        int r_mod = row % 32;
        int r16 = r_mod / 16;
        int r_lo = r_mod % 16;
        int g8 = g / 8;
        int g_mod = g % 8;
        int g4 = g_mod / 4;
        int g_lo = g_mod % 4;

        int bs_off = r16 + g4 * 2 + r_lo * 4 + g_lo * 128 + g8 * 512 + r32 * 32 * scale_n_pad;

        // Handle valid range: if g < num_groups, store actual scale; pad with 127
        int num_bs_cols = (K + 31) / 32;
        if (g < num_bs_cols) {
            bs_ptr[bs_off] = bs_val;
        }
    }
}

void launch_mxfp4_quant(
    torch::Tensor x,
    torch::Tensor fp4_out,
    torch::Tensor bs_out,
    int M, int K,
    int scale_n_pad
) {
    // Grid: each block handles blockDim.y rows, with blockDim.x threads per row
    // Use 32 threads per row (one per group in parallel), 8 rows per block
    dim3 block(32, 8);
    dim3 grid((M + 7) / 8, 1);

    mxfp4_quant_shuffle_kernel<<<grid, block>>>(
        reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()),
        reinterpret_cast<unsigned char*>(fp4_out.data_ptr()),
        reinterpret_cast<unsigned char*>(bs_out.data_ptr()),
        M, K,
        x.stride(0), x.stride(1),
        fp4_out.stride(0), fp4_out.stride(1),
        scale_n_pad
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_mxfp4_quant", &launch_mxfp4_quant, "MXFP4 quant with shuffle");
}
"""
    _hip_quant = load_inline(
        name="hip_mxfp4_quant",
        cpp_sources="",
        cuda_sources=_HIP_QUANT_SRC,
        functions=["launch_mxfp4_quant"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
    )
except Exception:
    _hip_quant = None

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

        out_tensor, bs_e8m0 = _mxfp4_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        # Store fp4 output
        out_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_offs_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
        out_offs = (
            out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )

        if EVEN_M_N:
            tl.store(x_fp4_ptr + out_offs, out_tensor, cache_modifier=".wt")
        else:
            out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask, cache_modifier=".wt")

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

    # Reduce kernel params — gluon version uses BSN=64 for fp32 partials
    REDUCE_BSM = 16
    REDUCE_BSN = 64  # Gluon default for fp32 partials
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
                splitk_params = _prepare_splitk_dispatch(M, N, K, config, device)
                _buffers[key] = {
                    'mode': 'fused_splitk',
                    'out': torch.empty((M, N), dtype=torch.bfloat16, device=device),
                    'B_w': None,
                    'B_sc': None,
                    'splitk_params': splitk_params,
                }
            else:
                K_kernel = K // 2
                BSK = config["BLOCK_SIZE_K"]
                BSN = max(config["BLOCK_SIZE_N"], 32)
                BSM = config["BLOCK_SIZE_M"]
                SPLITK_BLOCK_SIZE = 2 * K_kernel

                grid_size = triton.cdiv(M, BSM) * triton.cdiv(N, BSN)

                _buffers[key] = {
                    'mode': 'fused_direct',
                    'out': torch.empty((M, N), dtype=torch.bfloat16, device=device),
                    'B_w': None,
                    'B_sc': None,
                    'grid_size': grid_size,
                    'K_kernel': K_kernel,
                    'BLOCK_SIZE_M': BSM,
                    'BLOCK_SIZE_N': BSN,
                    'BLOCK_SIZE_K': BSK,
                    'SPLITK_BLOCK_SIZE': SPLITK_BLOCK_SIZE,
                    'GROUP_SIZE_M': config["GROUP_SIZE_M"],
                    'NUM_KSPLIT': 1,
                    'num_warps': config["num_warps"],
                    'num_stages': config["num_stages"],
                    'waves_per_eu': config["waves_per_eu"],
                    'matrix_instr_nonkdim': config["matrix_instr_nonkdim"],
                    'cache_modifier': config["cache_modifier"],
                }
        else:
            MXFP4_QUANT_BLOCK_SIZE = 32
            SCALE_N_valid = triton.cdiv(K, MXFP4_QUANT_BLOCK_SIZE)
            SCALE_M = triton.cdiv(M, 256) * 256
            SCALE_N = triton.cdiv(SCALE_N_valid, 8) * 8

            padded_M = (M + 31) // 32 * 32

            if _hip_quant is not None:
                # HIP quant path
                _buffers[key] = {
                    'mode': 'two_phase_hip',
                    'x_fp4': torch.empty((M, K // 2), dtype=torch.uint8, device=device),
                    'blockscale': torch.empty((SCALE_M, SCALE_N), dtype=torch.uint8, device=device),
                    'gemm_out': torch.empty((padded_M, N), dtype=torch.bfloat16, device=device),
                    'SCALE_N': SCALE_N,
                    'M': M,
                }
            else:
                # Fallback to Triton quant
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

        _gluon_reduce_kernel[kp['reduce_grid']](
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

    elif buf['mode'] == 'fused_direct':
        b_ptr = B_shuffle.data_ptr()
        if buf['B_w'] is None or buf.get('_b_ptr') != b_ptr:
            buf['B_w'] = B_shuffle.view(torch.uint8).reshape(N // 16, (K // 2) * 16)
            bs_shape = B_scale_sh.shape
            buf['B_sc'] = B_scale_sh.view(torch.uint8).reshape(
                bs_shape[0] // 32, bs_shape[1] * 32
            )
            buf['_b_ptr'] = b_ptr

        out = buf['out']

        _gemm_a16wfp4_preshuffle_kernel[(buf['grid_size'],)](
            A,
            buf['B_w'],
            out,
            buf['B_sc'],
            M,
            N,
            buf['K_kernel'],
            A.stride(0),
            A.stride(1),
            buf['B_w'].stride(0),
            buf['B_w'].stride(1),
            0,
            out.stride(0),
            out.stride(1),
            buf['B_sc'].stride(0),
            buf['B_sc'].stride(1),
            BLOCK_SIZE_M=buf['BLOCK_SIZE_M'],
            BLOCK_SIZE_N=buf['BLOCK_SIZE_N'],
            BLOCK_SIZE_K=buf['BLOCK_SIZE_K'],
            GROUP_SIZE_M=buf['GROUP_SIZE_M'],
            NUM_KSPLIT=buf['NUM_KSPLIT'],
            SPLITK_BLOCK_SIZE=buf['SPLITK_BLOCK_SIZE'],
            num_warps=buf['num_warps'],
            num_stages=buf['num_stages'],
            waves_per_eu=buf['waves_per_eu'],
            matrix_instr_nonkdim=buf['matrix_instr_nonkdim'],
            PREQUANT=True,
            cache_modifier=buf['cache_modifier'],
        )

        return out

    elif buf['mode'] == 'two_phase_hip':
        # HIP quant kernel path for M=256
        _hip_quant.launch_mxfp4_quant(
            A, buf['x_fp4'], buf['blockscale'],
            M, K, buf['SCALE_N']
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
