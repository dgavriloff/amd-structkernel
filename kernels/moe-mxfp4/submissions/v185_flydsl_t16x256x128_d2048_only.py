#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v185: FlyDSL stage2 t16x256x128_atomic for E=33/bs=512/d=2048 only.
tile_n=256 halves N-tiles for the largest GEMM while keeping t16x128x128
for d=512 shapes where it performed better.
"""
import os
import functools
import torch
import triton
from typing import Dict, Tuple, Optional
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    get_2stage_cfgs, get_padded_M, get_inter_dim,
    ck_moe_stage1, cktile_moe_stage1, cktile_moe_stage2,
    _flydsl_stage2_wrapper,
)
import aiter.fused_moe as _fused_moe_module
import aiter.ops.flydsl.moe_kernels as _flydsl_moe_kernels
from aiter.ops.triton._triton_kernels.quant.fused_mxfp4_quant import (
    _fused_dynamic_mxfp4_quant_moe_sort_kernel,
)
from aiter.utility import fp4_utils

# Register FlyDSL tile_k=128 kernels
_flydsl_moe_kernels._KERNEL_PARAMS["flydsl_moe2_afp4_wfp4_bf16_t32x128x128_atomic"] = {
    "stage": 2, "a_dtype": "fp4", "b_dtype": "fp4", "out_dtype": "bf16",
    "tile_m": 32, "tile_n": 128, "tile_k": 128, "mode": "atomic", "MPerBlock": 32,
}
_flydsl_moe_kernels._KERNEL_PARAMS["flydsl_moe2_afp4_wfp4_bf16_t32x256x128_atomic"] = {
    "stage": 2, "a_dtype": "fp4", "b_dtype": "fp4", "out_dtype": "bf16",
    "tile_m": 32, "tile_n": 256, "tile_k": 128, "mode": "atomic", "MPerBlock": 32,
}
_flydsl_moe_kernels._KERNEL_PARAMS["flydsl_moe2_afp4_wfp4_bf16_t16x256x128_atomic"] = {
    "stage": 2, "a_dtype": "fp4", "b_dtype": "fp4", "out_dtype": "bf16",
    "tile_m": 16, "tile_n": 256, "tile_k": 128, "mode": "atomic", "MPerBlock": 16,
}
_flydsl_moe_kernels._KERNEL_PARAMS["flydsl_moe2_afp4_wfp4_bf16_t16x128x128_atomic"] = {
    "stage": 2, "a_dtype": "fp4", "b_dtype": "fp4", "out_dtype": "bf16",
    "tile_m": 16, "tile_n": 128, "tile_k": 128, "mode": "atomic", "MPerBlock": 16,
}

# Shape configs
_CUSTOM_CONFIGS = {}

def _make_key(token, inter_dim, expert):
    return (
        256, token, 7168, inter_dim, expert, 9,
        "ActivationType.Silu", "torch.bfloat16",
        "torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2",
        "QuantType.per_1x32", True, False,
    )

_4WG_STAGE1_M128 = "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
_4WG_STAGE1_M32 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
_FLYDSL_STAGE2_M16_K128 = "flydsl_moe2_afp4_wfp4_bf16_t16x128x128_atomic"

# E=33 shapes
_CUSTOM_CONFIGS[_make_key(16, 512, 33)] = {
    "block_m": 32, "ksplit": 2, "kernelName1": "", "kernelName2": "",
    "run_1stage": False,
}
_CUSTOM_CONFIGS[_make_key(128, 512, 33)] = {
    "block_m": 64, "ksplit": 0,
    "kernelName1": _4WG_STAGE1_M128, "kernelName2": _FLYDSL_STAGE2_M16_K128,
    "run_1stage": False,
}
_CUSTOM_CONFIGS[_make_key(512, 512, 33)] = {
    "block_m": 64, "ksplit": 0,
    "kernelName1": _4WG_STAGE1_M128, "kernelName2": _FLYDSL_STAGE2_M16_K128,
    "run_1stage": False,
}
_FLYDSL_STAGE2_M16_N256_K128 = "flydsl_moe2_afp4_wfp4_bf16_t16x256x128_atomic"
_CUSTOM_CONFIGS[_make_key(512, 2048, 33)] = {
    "block_m": 64, "ksplit": 0,
    "kernelName1": _4WG_STAGE1_M128, "kernelName2": _FLYDSL_STAGE2_M16_N256_K128,
    "run_1stage": False,
}

# E=257 shapes
_CUSTOM_CONFIGS[_make_key(16, 256, 257)] = {
    "block_m": 16, "ksplit": 2, "kernelName1": "", "kernelName2": "",
    "run_1stage": False,
}
_CUSTOM_CONFIGS[_make_key(128, 256, 257)] = {
    "block_m": 16, "ksplit": 2, "kernelName1": "", "kernelName2": "",
    "run_1stage": False,
}
_CUSTOM_CONFIGS[_make_key(512, 256, 257)] = {
    "block_m": 32, "ksplit": 0,
    "kernelName1": _4WG_STAGE1_M32, "kernelName2": _FLYDSL_STAGE2_M16_K128,
    "run_1stage": False,
    "use_non_temporal_load": True,
}

# Pre-allocated buffer cache
_buffer_cache = {}

def _get_or_alloc_sorting_buffers(M, E, topk, model_dim, block_size_M, device):
    """Pre-allocate moe_sorting output buffers."""
    key = ("sort", M, E, topk, model_dim, block_size_M)
    if key in _buffer_cache:
        return _buffer_cache[key]

    max_num_tokens_padded = int(M * topk + E * block_size_M - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size_M - 1) // block_size_M)

    bufs = {
        "sorted_ids": torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device),
        "sorted_weights": torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device),
        "sorted_expert_ids": torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device),
        "num_valid_ids": torch.empty(2, dtype=dtypes.i32, device=device),
        "moe_buf": torch.empty((M, model_dim), dtype=torch.bfloat16, device=device),
    }
    _buffer_cache[key] = bufs
    return bufs

def _get_or_alloc_a2(M, topk, inter_dim, device):
    """Pre-allocate a2 intermediate buffer."""
    key = ("a2", M, topk, inter_dim)
    if key in _buffer_cache:
        return _buffer_cache[key]
    buf = torch.empty((M, topk, inter_dim), dtype=torch.bfloat16, device=device)
    _buffer_cache[key] = buf
    return buf

def _get_or_alloc_quant_buffers(M, N, sorted_ids_len, topk, device):
    """Pre-allocate quantization output buffers for fused_dynamic_mxfp4_quant_moe_sort."""
    MXFP4_QUANT_BLOCK_SIZE = 32
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8
    BLOCK_SIZE_M_u32, BLOCK_SIZE_N_u32 = 16, 4

    key = ("quant", M, N, sorted_ids_len, topk)
    if key in _buffer_cache:
        return _buffer_cache[key]

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=device)
    scaleN = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
    M_o = sorted_ids_len
    N_o = scaleN

    blockscale_e8m0_sorted = torch.empty(
        (
            triton.cdiv(M_o, BLOCK_SIZE_M),
            triton.cdiv(N_o, BLOCK_SIZE_N),
            BLOCK_SIZE_N_u32,
            BLOCK_SIZE_M_u32,
            4,
        ),
        dtype=torch.uint8,
        device=device,
    )

    bufs = {"x_fp4": x_fp4, "blockscale": blockscale_e8m0_sorted}
    _buffer_cache[key] = bufs
    return bufs

def _quant_prealloc(x, sorted_ids, num_valid_ids, token_num, topk, block_size, device):
    """Inline fused_dynamic_mxfp4_quant_moe_sort with pre-allocated output buffers."""
    M, N = x.shape
    MXFP4_QUANT_BLOCK_SIZE = 32
    BLOCK_SIZE_Mx = 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8

    scaleN = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
    M_i, N_i = M, scaleN
    M_o = sorted_ids.shape[0]

    # Get pre-allocated buffers
    qbufs = _get_or_alloc_quant_buffers(M, N, M_o, topk, device)
    x_fp4 = qbufs["x_fp4"]
    blockscale_e8m0_sorted = qbufs["blockscale"]

    num_pid = triton.cdiv(M, BLOCK_SIZE_Mx) * scaleN + triton.cdiv(
        M_o, BLOCK_SIZE_M
    ) * triton.cdiv(N_i, BLOCK_SIZE_N)

    _fused_dynamic_mxfp4_quant_moe_sort_kernel[(num_pid,)](
        x,
        x_fp4,
        sorted_ids,
        num_valid_ids,
        blockscale_e8m0_sorted,
        M,
        N,
        scaleN,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0_sorted.stride(),
        token_num=token_num,
        M_i=M_i,
        N_i=N_i,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        BLOCK_SIZE_Mx=BLOCK_SIZE_Mx,
        BLOCK_SIZE_M=BLOCK_SIZE_M // 2,
        BLOCK_SIZE_N=BLOCK_SIZE_N // 2,
        TOPK=topk,
    )

    return (
        x_fp4.view(dtypes.fp4x2),
        blockscale_e8m0_sorted.view(dtypes.fp8_e8m0).view(-1, scaleN),
    )


_injected = False

def _inject_configs():
    global _injected
    if _injected:
        return
    _injected = True

    if _fused_moe_module.cfg_2stages is None:
        import pandas as pd
        from aiter.jit.core import AITER_CONFIGS
        tune_file = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
        if os.path.exists(tune_file):
            _INDEX_COLS = [
                "cu_num", "token", "model_dim", "inter_dim", "expert", "topk",
                "act_type", "dtype", "q_dtype_a", "q_dtype_w", "q_type",
                "use_g1u1", "doweight_stage1",
            ]
            df = pd.read_csv(tune_file)
            if "_tag" in df.columns:
                df = df[df["_tag"].fillna("") == ""]
            _fused_moe_module.cfg_2stages = df.set_index(_INDEX_COLS).to_dict("index")
        else:
            _fused_moe_module.cfg_2stages = {}

    _fused_moe_module.cfg_2stages.update(_CUSTOM_CONFIGS)

    # Monkeypatch get_2stage_cfgs to support use_non_temporal_load from config
    _original_get_2stage_cfgs = _fused_moe_module.get_2stage_cfgs

    @functools.lru_cache(maxsize=2048)
    def _patched_get_2stage_cfgs(
        token, model_dim, inter_dim, expert, topk,
        dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
        activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled=True,
    ):
        metadata = _original_get_2stage_cfgs(
            token, model_dim, inter_dim, expert, topk,
            dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
            activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled,
        )
        from aiter.jit.utils.chip_info import get_cu_num
        cu_num = get_cu_num()
        keys = (
            cu_num, token, model_dim, inter_dim, expert, topk,
            str(activation), str(dtype), str(q_dtype_a), str(q_dtype_w),
            str(q_type), use_g1u1, doweight_stage1,
        )
        cfg = _fused_moe_module.cfg_2stages.get(keys)
        if cfg and cfg.get("use_non_temporal_load") is not None:
            nt = cfg["use_non_temporal_load"]
            old_s1 = metadata.stage1
            if hasattr(old_s1, 'func') and old_s1.func is not None:
                if 'use_non_temporal_load' in (old_s1.keywords or {}):
                    new_kw = dict(old_s1.keywords)
                    new_kw['use_non_temporal_load'] = nt
                    metadata = _fused_moe_module.MOEMetadata(
                        functools.partial(old_s1.func, **{k: v for k, v in new_kw.items()}),
                        metadata.stage2,
                        metadata.block_m,
                        metadata.ksplit,
                        metadata.run_1stage,
                        metadata.has_bias,
                        nt,
                    )
                    old_s2 = metadata.stage2
                    if old_s2 and hasattr(old_s2, 'keywords') and 'use_non_temporal_load' in (old_s2.keywords or {}):
                        new_kw2 = dict(old_s2.keywords)
                        new_kw2['use_non_temporal_load'] = nt
                        metadata = _fused_moe_module.MOEMetadata(
                            metadata.stage1,
                            functools.partial(old_s2.func, **{k: v for k, v in new_kw2.items()}),
                            metadata.block_m,
                            metadata.ksplit,
                            metadata.run_1stage,
                            metadata.has_bias,
                            nt,
                        )
        return metadata

    _fused_moe_module.get_2stage_cfgs = _patched_get_2stage_cfgs


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    _inject_configs()

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device
    w1 = gate_up_weight_shuffled
    w2 = down_weight_shuffled
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    padded_M = get_padded_M(M)
    metadata = get_2stage_cfgs(
        padded_M, model_dim, inter_dim, E, topk,
        torch.bfloat16, dtypes.fp4x2, dtypes.fp4x2,
        QuantType.per_1x32, True, ActivationType.Silu,
        False, hidden_pad, intermediate_pad, True,
    )

    block_size_M = int(metadata.block_m)

    # === Pre-allocated moe_sorting ===
    bufs = _get_or_alloc_sorting_buffers(M, E, topk, model_dim, block_size_M, device)
    sorted_ids = bufs["sorted_ids"]
    sorted_weights = bufs["sorted_weights"]
    sorted_expert_ids = bufs["sorted_expert_ids"]
    num_valid_ids = bufs["num_valid_ids"]
    moe_out = bufs["moe_buf"]

    aiter.moe_sorting_fwd(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out,
        E, int(block_size_M), None, None, 0,
    )

    # === Inline 2-stage pipeline ===
    token_num = M

    if metadata.ksplit > 1:
        # cktile_moe path: bf16 activations, no fp4 quant
        a1 = hidden_states.to(torch.bfloat16)
        a1_scale = None
        w1_scale_view = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
        w2_scale_view = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)

        a2 = metadata.stage1(
            a1, w1, w2,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            _get_or_alloc_a2(M, topk, inter_dim, device),  # pre-allocated
            topk,
            block_m=block_size_M,
            a1_scale=a1_scale,
            w1_scale=w1_scale_view,
            sorted_weights=None,
        )

        # cktile_moe stage2: a2 is bf16, no inter-stage requant
        a2_scale = None
        metadata.stage2(
            a2, w1, w2,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            moe_out, topk,
            w2_scale=w2_scale_view,
            a2_scale=a2_scale,
            block_m=block_size_M,
            sorted_weights=sorted_weights,
        )
    else:
        # CK 2-stage path: fp4 activation quant with pre-allocated buffers
        w1_scale_view = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
        w2_scale_view = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)

        # Stage 1: quant activations + gate_up GEMM + SwiGLU
        a1, a1_scale = _quant_prealloc(
            hidden_states, sorted_ids, num_valid_ids,
            token_num, 1, block_size_M, device,
        )

        a2 = _get_or_alloc_a2(M, topk, inter_dim, device)
        a2 = metadata.stage1(
            a1, w1, w2,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            a2, topk,
            block_m=block_size_M,
            a1_scale=a1_scale,
            w1_scale=w1_scale_view,
            sorted_weights=None,
        )

        # Inter-stage requant: bf16 -> fp4 with pre-allocated buffers
        a2_flat = a2.view(-1, inter_dim)
        a2_quant, a2_scale = _quant_prealloc(
            a2_flat, sorted_ids, num_valid_ids,
            token_num, topk, block_size_M, device,
        )
        a2_quant = a2_quant.view(token_num, topk, -1)

        # Stage 2: down GEMM + weighted reduction
        metadata.stage2(
            a2_quant, w1, w2,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            moe_out, topk,
            w2_scale=w2_scale_view,
            a2_scale=a2_scale,
            block_m=block_size_M,
            sorted_weights=sorted_weights,
        )

    return moe_out
