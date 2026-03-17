#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v137: Minimize Python dispatch overhead with closure-based per-shape fast paths.

Hypothesis: At 6-13us kernel times, Python overhead (dict lookup, string compare,
kwarg unpacking, data_ptr check, view/reshape) may be measurable. Create a
per-shape callable closure that pre-captures all parameters, eliminating
conditionals and dict access in the hot path.

Target: All shapes — reduce constant overhead floor.
"""
import torch
import triton
import triton.language as tl
from aiter import dtypes
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
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

# Pre-built callable fast paths keyed by (M, K, N)
_fast_paths = {}

# ASM kernel name
_ASM_KERNEL_32x128 = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"

# Threshold: use fused for M <= this value
_FUSED_M_THRESHOLD = 64


def _get_fused_config(M, N, K):
    """Get shape-specific config for fused quant+GEMM path."""
    if K > 4096:
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


def _build_splitk_fast_path(M, N, K, device):
    """Build a closure for split-K path (16x2112x7168)."""
    config = _get_fused_config(M, N, K)
    K_kernel = K // 2
    BSK = config["BLOCK_SIZE_K"]
    NUM_KSPLIT = config["NUM_KSPLIT"]

    SPLITK_BLOCK_SIZE, BSK, NUM_KSPLIT = get_splitk(K_kernel, BSK, NUM_KSPLIT)

    BSN = max(config["BLOCK_SIZE_N"], 32)
    BSM = config["BLOCK_SIZE_M"]

    grid_size = NUM_KSPLIT * triton.cdiv(M, BSM) * triton.cdiv(N, BSN)

    y_pp = torch.empty((NUM_KSPLIT, M, N), dtype=torch.float32, device=device)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    REDUCE_BSM = 16
    REDUCE_BSN = 16
    ACTUAL_KSPLIT = triton.cdiv(K_kernel, (SPLITK_BLOCK_SIZE // 2))
    reduce_grid = (triton.cdiv(M, REDUCE_BSM), triton.cdiv(N, REDUCE_BSN))
    MAX_KSPLIT = triton.next_power_of_2(NUM_KSPLIT)

    # Pre-compute strides for y_pp and out
    ypp_s0 = y_pp.stride(0)
    ypp_s1 = y_pp.stride(1)
    ypp_s2 = y_pp.stride(2)
    out_s0 = out.stride(0)
    out_s1 = out.stride(1)

    # Capture all in closure
    _last_b_ptr = [None]
    _B_w = [None]
    _B_sc = [None]

    def fast_path(A, B_shuffle, B_scale_sh):
        b_ptr = B_shuffle.data_ptr()
        if _last_b_ptr[0] != b_ptr:
            _B_w[0] = B_shuffle.view(torch.uint8).reshape(N // 16, (K // 2) * 16)
            bs_shape = B_scale_sh.shape
            _B_sc[0] = B_scale_sh.view(torch.uint8).reshape(
                bs_shape[0] // 32, bs_shape[1] * 32
            )
            _last_b_ptr[0] = b_ptr

        B_w = _B_w[0]
        B_sc = _B_sc[0]

        _gemm_a16wfp4_preshuffle_kernel[(grid_size,)](
            A,
            B_w,
            y_pp,
            B_sc,
            M,
            N,
            K_kernel,
            A.stride(0),
            A.stride(1),
            B_w.stride(0),
            B_w.stride(1),
            ypp_s0,
            ypp_s1,
            ypp_s2,
            B_sc.stride(0),
            B_sc.stride(1),
            BLOCK_SIZE_M=BSM,
            BLOCK_SIZE_N=BSN,
            BLOCK_SIZE_K=BSK,
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            NUM_KSPLIT=NUM_KSPLIT,
            SPLITK_BLOCK_SIZE=SPLITK_BLOCK_SIZE,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            waves_per_eu=config["waves_per_eu"],
            matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
            PREQUANT=True,
            cache_modifier=config["cache_modifier"],
        )

        _gemm_afp4wfp4_reduce_kernel[reduce_grid](
            y_pp,
            out,
            M,
            N,
            ypp_s0,
            ypp_s1,
            ypp_s2,
            out_s0,
            out_s1,
            REDUCE_BSM,
            REDUCE_BSN,
            ACTUAL_KSPLIT,
            MAX_KSPLIT,
        )

        return out

    return fast_path


def _build_fused_fast_path(M, N, K, device):
    """Build a closure for non-split-K fused path."""
    config = _get_fused_config(M, N, K)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    _last_b_ptr = [None]
    _B_w = [None]
    _B_sc = [None]

    def fast_path(A, B_shuffle, B_scale_sh):
        b_ptr = B_shuffle.data_ptr()
        if _last_b_ptr[0] != b_ptr:
            _B_w[0] = B_shuffle.view(torch.uint8).reshape(N // 16, (K // 2) * 16)
            bs_shape = B_scale_sh.shape
            _B_sc[0] = B_scale_sh.view(torch.uint8).reshape(
                bs_shape[0] // 32, bs_shape[1] * 32
            )
            _last_b_ptr[0] = b_ptr

        return gemm_a16wfp4_preshuffle(
            A,
            _B_w[0],
            _B_sc[0],
            prequant=True,
            dtype=torch.bfloat16,
            y=out,
            config=config,
        )

    return fast_path


def _build_two_phase_fast_path(M, N, K, device):
    """Build a closure for M>64 two-phase path."""
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

    x_fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    blockscale = torch.empty((SCALE_M, SCALE_N), dtype=torch.uint8, device=device)
    gemm_out = torch.empty((padded_M, N), dtype=torch.bfloat16, device=device)

    # Pre-compute strides
    x_fp4_s0 = x_fp4.stride(0)
    x_fp4_s1 = x_fp4.stride(1)

    # Pre-create views
    x_fp4_fp4x2 = x_fp4.view(dtypes.fp4x2)
    blockscale_e8m0 = blockscale.view(dtypes.fp8_e8m0)

    def fast_path(A, B_shuffle, B_scale_sh):
        _fused_mxfp4_quant_shuffle_kernel[grid](
            A,
            x_fp4,
            blockscale,
            A.stride(0),
            A.stride(1),
            x_fp4_s0,
            x_fp4_s1,
            M=M,
            N=K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            NUM_ITER=NUM_ITER,
            NUM_STAGES=NUM_STAGES,
            MXFP4_QUANT_BLOCK_SIZE=32,
            SCALING_MODE=0,
            SCALE_N_PAD=SCALE_N,
            num_warps=NUM_WARPS,
            waves_per_eu=0,
            num_stages=1,
        )

        gemm_a4w4_asm(
            x_fp4_fp4x2,
            B_shuffle,
            blockscale_e8m0,
            B_scale_sh,
            gemm_out,
            _ASM_KERNEL_32x128,
            None,
            1.0,
            0.0,
            True,
            log2_k_split=0,
        )

        return gemm_out[:M]

    return fast_path


def _get_fast_path(M, K, N, device):
    """Get or create a fast-path callable for the given shape."""
    key = (M, K, N)
    if key not in _fast_paths:
        if M <= _FUSED_M_THRESHOLD:
            config = _get_fused_config(M, N, K)
            if config["NUM_KSPLIT"] > 1:
                _fast_paths[key] = _build_splitk_fast_path(M, N, K, device)
            else:
                _fast_paths[key] = _build_fused_fast_path(M, N, K, device)
        else:
            _fast_paths[key] = _build_two_phase_fast_path(M, N, K, device)
    return _fast_paths[key]


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_shuffle.shape[0]
    return _get_fast_path(M, K, N, A.device)(A, B_shuffle, B_scale_sh)
