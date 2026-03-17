#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v012: Unified quant kernel config — fix M<=32 large-K path.
- Previous M<=32 default used BLOCK_SIZE_N=32, 1 warp for ALL K values.
  For M=16, K=7168 this created grid=(1,224) with 1 warp — 224 tiny blocks.
- New: unified config based on K, independent of M. All paths use 4 warps.
  For M=16, K=7168: grid=(1,7) with 4 warps — 32x fewer blocks, 4x more threads.
- Target: quant phase for 16x2112x7168 (6.4µs → lower).
"""
import torch
import triton
import triton.language as tl
import aiter
from aiter import dtypes
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

from task import input_t, output_t

# Pre-allocated buffers keyed by (M, K)
_buffers = {}


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


def _get_or_create_buffers(M, K, device):
    """Get pre-allocated quant buffers for given shape, creating if needed."""
    key = (M, K)
    if key not in _buffers:
        MXFP4_QUANT_BLOCK_SIZE = 32
        SCALE_N_valid = triton.cdiv(K, MXFP4_QUANT_BLOCK_SIZE)
        SCALE_M = triton.cdiv(M, 256) * 256
        SCALE_N = triton.cdiv(SCALE_N_valid, 8) * 8

        # Unified config based on K, always 4 warps, BLOCK_SIZE_M=32
        BLOCK_SIZE_M = 32
        NUM_WARPS = 4
        if K <= 1024:
            # Small K: fit entire K in one or two blocks
            BLOCK_SIZE_N = max(32, min(512, triton.next_power_of_2(K)))
            NUM_ITER = 1
            NUM_STAGES = 1
        else:
            # Large K: use wide blocks with iteration
            BLOCK_SIZE_N = 256
            NUM_ITER = 4
            NUM_STAGES = 2

        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(K, BLOCK_SIZE_N * NUM_ITER),
        )

        _buffers[key] = {
            'x_fp4': torch.empty((M, K // 2), dtype=torch.uint8, device=device),
            'blockscale': torch.empty((SCALE_M, SCALE_N), dtype=torch.uint8, device=device),
            'SCALE_N': SCALE_N,
            'BLOCK_SIZE_M': BLOCK_SIZE_M,
            'BLOCK_SIZE_N': BLOCK_SIZE_N,
            'NUM_ITER': NUM_ITER,
            'NUM_STAGES': NUM_STAGES,
            'NUM_WARPS': NUM_WARPS,
            'grid': grid,
        }
    return _buffers[key]


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data
    M, K = A.shape

    # Get pre-allocated buffers and cached config
    buf = _get_or_create_buffers(M, K, A.device)

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

    return aiter.gemm_a4w4(
        buf['x_fp4'].view(dtypes.fp4x2),
        B_shuffle,
        buf['blockscale'].view(dtypes.fp8_e8m0),
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
