#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v052: Route M<=16 to two-phase path (leaderboard correctness fix).
Fused preshuffle GEMM fails leaderboard secret tests for M=4 and M=16x2112x7168.
Even library default config (split-K=14) fails for M=16. Route M<=16 to proven-correct
two-phase path (Triton quant + ASM GEMM). Keep fused path only for M=32.
"""
import torch
import triton
import triton.language as tl
from aiter import dtypes
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

from task import input_t, output_t

# Pre-allocated buffers keyed by (M, K, N)
_buffers = {}

# ASM kernel name — 32x128 is optimal for all small-M shapes per tuned CSV analysis
_ASM_KERNEL_32x128 = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"

# Threshold: use fused for M in range (_FUSED_M_MIN, _FUSED_M_MAX]
_FUSED_M_MIN = 16  # M<=16 uses two-phase (fused preshuffle has correctness issues on leaderboard)
_FUSED_M_THRESHOLD = 32


def _get_fused_config(M, N, K):
    """Get shape-specific config for fused quant+GEMM path.
    Only called for M in range (_FUSED_M_MIN, _FUSED_M_THRESHOLD] = (16, 32], i.e. M=32.
    """
    if K <= 1024:
        # For M=32 with short K: BSM=16 creates 2 M-tiles (vs 1 with default BSM=32),
        # doubling parallelism. BSN=128 NW=4 matches tuned config style.
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 1,
        }
    else:
        # Let library pick for M=32 K>1024
        return None


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
    """Get pre-allocated buffers for given shape."""
    key = (M, K, N)
    if key not in _buffers:
        if M > _FUSED_M_MIN and M <= _FUSED_M_THRESHOLD:
            _buffers[key] = {
                'mode': 'fused',
                'out': torch.empty((M, N), dtype=torch.bfloat16, device=device),
                'B_w': None,
                'B_sc': None,
                'config': _get_fused_config(M, N, K),
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

    if buf['mode'] == 'fused':
        if buf['B_w'] is None:
            buf['B_w'] = B_shuffle.view(torch.uint8).reshape(N // 16, (K // 2) * 16)
            bs_shape = B_scale_sh.shape
            buf['B_sc'] = B_scale_sh.view(torch.uint8).reshape(
                bs_shape[0] // 32, bs_shape[1] * 32
            )

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
