#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v216: Gluon AFP4WFP4 GEMM for M=256 two-phase path.

Hypothesis: Replace ASM GEMM with gluon AFP4WFP4 kernel for M=256. The gluon
kernel uses explicit CDNA4 buffer_load/store, mfma_scaled, and XCD remap which
may produce better codegen than the ASM 32x128 kernel. Requires:
1. Modified quant kernel that stores scales in linear (non-shuffled) format
2. Unshuffle B_scale_sh to get raw B scales at init time
3. Use B_q (non-shuffled FP4) directly instead of B_shuffle
M<=64 paths are completely unchanged.
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
from aiter.ops.triton.gluon.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_reduce_kernel as _gluon_reduce_kernel,
    _gemm_afp4wfp4_kernel as _gluon_gemm_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_reduce_kernel,
)
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import get_splitk

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
        # waves_per_eu=2: tuned JSON uses this for M>=16 shapes
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


@triton.jit
def _mxfp4_quant_linear_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m,
    stride_x_k,
    stride_fp4_m,
    stride_fp4_k,
    stride_bs_m,
    stride_bs_k,
    M,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
):
    """Quant kernel that stores scales in linear (M, K//32) format — no shuffle."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_K // MXFP4_QUANT_BLOCK_SIZE

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Load A in bf16
    x_offs = offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
    x_mask = (offs_m < M)[:, None] & (offs_k < K)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(tl.float32)

    # Quantize
    out_tensor, bs_e8m0 = _mxfp4_quant_op(
        x, BLOCK_SIZE_K, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
    )

    # Store fp4 output — linear (M, K//2) format
    out_offs_m = offs_m
    out_offs_k = pid_k * BLOCK_SIZE_K // 2 + tl.arange(0, BLOCK_SIZE_K // 2)
    out_offs = out_offs_m[:, None] * stride_fp4_m + out_offs_k[None, :] * stride_fp4_k
    out_mask = (offs_m < M)[:, None] & (out_offs_k < (K // 2))[None, :]
    tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask, cache_modifier=".wt")

    # Store scales — linear (M, K//32) format, NO shuffle
    bs_offs_m = offs_m
    bs_offs_k = pid_k * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
    scale_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_k[None, :] * stride_bs_k
    num_scale_cols = (K + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    bs_mask = (offs_m < M)[:, None] & (bs_offs_k < num_scale_cols)[None, :]
    tl.store(
        bs_ptr + scale_offs,
        bs_e8m0.to(tl.uint8),
        mask=bs_mask,
        cache_modifier=".wt",
    )


def _unshuffle_e8m0(scale_sh):
    """Inverse of e8m0_shuffle: convert shuffled scales back to linear (M, K//32) format.
    e8m0_shuffle does: view(sm//32, 2, 16, sn//8, 2, 4).permute(0, 3, 5, 2, 4, 1)
    Inverse: view(sm//32, sn//8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2)
    """
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    s = s.view(sm, sn)
    return s


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
                # Non-split-K: direct dispatch (bypass wrapper overhead)
                K_kernel = K // 2
                BSK = config["BLOCK_SIZE_K"]
                BSN = max(config["BLOCK_SIZE_N"], 32)
                BSM = config["BLOCK_SIZE_M"]
                SPLITK_BLOCK_SIZE = 2 * K_kernel  # No split-K

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
            # M=256: gluon AFP4WFP4 GEMM path with non-shuffled scales
            K_scale = triton.cdiv(K, 32)  # K // 32, non-padded

            # Quant kernel config
            QUANT_BSM = 32
            QUANT_BSK = 64
            quant_grid = (triton.cdiv(M, QUANT_BSM), triton.cdiv(K, QUANT_BSK))

            # Gluon GEMM config for M=256, N=3072, K_packed=768
            # BSM=32 BSN=128 BSK=256: grid = 8*24 = 192 blocks
            K_packed = K // 2  # FP4 packed
            gluon_config = {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 4,
                "NUM_KSPLIT": 1,
            }
            SPLITK_BLOCK_SIZE = 2 * K_packed
            gluon_grid_size = (
                triton.cdiv(M, gluon_config["BLOCK_SIZE_M"])
                * triton.cdiv(N, gluon_config["BLOCK_SIZE_N"])
            )

            _buffers[key] = {
                'mode': 'gluon_fp4',
                'x_fp4': torch.empty((M, K // 2), dtype=torch.uint8, device=device),
                'x_scales': torch.empty((M, K_scale), dtype=torch.uint8, device=device),
                'gemm_out': torch.empty((M, N), dtype=torch.bfloat16, device=device),
                'B_scales_raw': None,  # Unshuffled B scales, cached per B_scale_sh
                'B_q_T': None,  # B_q transposed for gluon kernel
                'K_scale': K_scale,
                'K_packed': K_packed,
                'QUANT_BSM': QUANT_BSM,
                'QUANT_BSK': QUANT_BSK,
                'quant_grid': quant_grid,
                'gluon_config': gluon_config,
                'SPLITK_BLOCK_SIZE': SPLITK_BLOCK_SIZE,
                'gluon_grid_size': gluon_grid_size,
                '_bs_ptr': None,
            }
    return _buffers[key]


def custom_kernel(data: input_t) -> output_t:
    A, _, B_q, B_shuffle, B_scale_sh = data
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
        # Non-split-K fused path: direct kernel dispatch (bypass wrapper)
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
            0,  # stride_ck (no split-K)
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
    else:
        # M=256: gluon AFP4WFP4 path
        # Step 1: Unshuffle B scales if needed (new B data each LB iteration)
        bs_ptr = B_scale_sh.data_ptr()
        if buf['_bs_ptr'] != bs_ptr:
            # Unshuffle B_scale_sh to get raw (N, K//32) scales
            buf['B_scales_raw'] = _unshuffle_e8m0(B_scale_sh)
            # B_q is already (N, K//2) non-shuffled FP4, transpose for gluon kernel
            buf['B_q_T'] = B_q.view(torch.uint8).T.contiguous()
            buf['_bs_ptr'] = bs_ptr

        # Step 2: Quant A to FP4 with linear (non-shuffled) scales
        _mxfp4_quant_linear_kernel[buf['quant_grid']](
            A,
            buf['x_fp4'],
            buf['x_scales'],
            A.stride(0),
            A.stride(1),
            buf['x_fp4'].stride(0),
            buf['x_fp4'].stride(1),
            buf['x_scales'].stride(0),
            buf['x_scales'].stride(1),
            M,
            K,
            BLOCK_SIZE_M=buf['QUANT_BSM'],
            BLOCK_SIZE_K=buf['QUANT_BSK'],
            MXFP4_QUANT_BLOCK_SIZE=32,
            num_warps=2,
        )

        # Step 3: Gluon AFP4WFP4 GEMM
        cfg = buf['gluon_config']
        out = buf['gemm_out']
        B_q_uint8 = B_q.view(torch.uint8)
        B_scales_raw = buf['B_scales_raw']

        # Gluon kernel expects: a (M, K//2), b (K//2, N) [transposed from N,K//2]
        # a_scales (M, K//32), b_scales (N, K//32)
        b_T = buf['B_q_T']  # (K//2, N) contiguous

        _gluon_gemm_kernel[(buf['gluon_grid_size'],)](
            buf['x_fp4'],         # a_ptr: (M, K//2) uint8
            b_T,                  # b_ptr: (K//2, N) uint8
            out,                  # c_ptr: (M, N) bf16
            buf['x_scales'],      # a_scales_ptr: (M, K//32) uint8
            B_scales_raw,         # b_scales_ptr: (N, K//32) uint8
            M,
            N,
            buf['K_packed'],      # K (in packed fp4 elements = K_original // 2)
            buf['x_fp4'].stride(0),    # stride_am
            buf['x_fp4'].stride(1),    # stride_ak
            b_T.stride(0),             # stride_bk
            b_T.stride(1),             # stride_bn
            0,                         # stride_ck (no split-K)
            out.stride(0),             # stride_cm
            out.stride(1),             # stride_cn
            buf['x_scales'].stride(0), # stride_asm
            buf['x_scales'].stride(1), # stride_ask
            B_scales_raw.stride(0),    # stride_bsn
            B_scales_raw.stride(1),    # stride_bsk
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            NUM_KSPLIT=cfg["NUM_KSPLIT"],
            SPLITK_BLOCK_SIZE=buf['SPLITK_BLOCK_SIZE'],
            num_warps=4,
            num_stages=2,
            waves_per_eu=0,
            matrix_instr_nonkdim=16,
            cache_modifier=".cg",
        )

        return out
