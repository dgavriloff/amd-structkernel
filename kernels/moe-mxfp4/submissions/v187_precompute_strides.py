#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v187: Extend cached execution plan with pre-computed Triton kernel strides
and grid tuples. Remove hidden_states.to(bf16) no-op cast in cktile path.
Pre-compute x_fp4/blockscale stride tuples at plan-build time to avoid
per-call .stride() unpacking overhead.
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
_CUSTOM_CONFIGS[_make_key(512, 2048, 33)] = {
    "block_m": 64, "ksplit": 0,
    "kernelName1": _4WG_STAGE1_M128, "kernelName2": _FLYDSL_STAGE2_M16_K128,
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


# ============================================================
# Cached execution plan: build once per shape, reuse forever
# ============================================================
_plan_cache = {}


def _build_quant_constants(M, N, sorted_ids_len):
    """Pre-compute quant kernel launch constants."""
    MXFP4_QUANT_BLOCK_SIZE = 32
    BLOCK_SIZE_Mx = 128
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 8
    BLOCK_SIZE_M_u32 = 16
    BLOCK_SIZE_N_u32 = 4

    scaleN = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
    M_o = sorted_ids_len
    num_pid = triton.cdiv(M, BLOCK_SIZE_Mx) * scaleN + triton.cdiv(
        M_o, BLOCK_SIZE_M
    ) * triton.cdiv(scaleN, BLOCK_SIZE_N)

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device="cuda")
    blockscale = torch.empty(
        (
            triton.cdiv(M_o, BLOCK_SIZE_M),
            triton.cdiv(scaleN, BLOCK_SIZE_N),
            BLOCK_SIZE_N_u32,
            BLOCK_SIZE_M_u32,
            4,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    # Pre-compute strides and grid to avoid per-call overhead
    x_fp4_strides = x_fp4.stride()
    bs_strides = blockscale.stride()
    grid = (num_pid,)
    return {
        "x_fp4": x_fp4,
        "blockscale": blockscale,
        "scaleN": scaleN,
        "grid": grid,
        "M": M,
        "N": N,
        "M_i": M,
        "N_i": scaleN,
        "x_fp4_stride_0": x_fp4_strides[0],
        "x_fp4_stride_1": x_fp4_strides[1],
        "bs_stride_0": bs_strides[0],
        "bs_stride_1": bs_strides[1],
        "bs_stride_2": bs_strides[2],
        "bs_stride_3": bs_strides[3],
        "bs_stride_4": bs_strides[4],
    }


def _run_quant(qc, x, sorted_ids, num_valid_ids, token_num, topk):
    """Run quant kernel using pre-computed constants, buffers, and strides."""
    x_fp4 = qc["x_fp4"]
    blockscale = qc["blockscale"]

    _fused_dynamic_mxfp4_quant_moe_sort_kernel[qc["grid"]](
        x,
        x_fp4,
        sorted_ids,
        num_valid_ids,
        blockscale,
        qc["M"],
        qc["N"],
        qc["scaleN"],
        *x.stride(),
        qc["x_fp4_stride_0"], qc["x_fp4_stride_1"],
        qc["bs_stride_0"], qc["bs_stride_1"], qc["bs_stride_2"],
        qc["bs_stride_3"], qc["bs_stride_4"],
        token_num=token_num,
        M_i=qc["M_i"],
        N_i=qc["N_i"],
        MXFP4_QUANT_BLOCK_SIZE=32,
        BLOCK_SIZE_Mx=128,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=4,
        TOPK=topk,
    )

    return (
        x_fp4.view(dtypes.fp4x2),
        blockscale.view(dtypes.fp8_e8m0).view(-1, qc["scaleN"]),
    )


def _build_plan(M, E, topk, model_dim, inter_dim, hidden_pad, intermediate_pad, device):
    """Build execution plan with all buffers and metadata pre-resolved."""
    _inject_configs()

    padded_M = get_padded_M(M)
    metadata = get_2stage_cfgs(
        padded_M, model_dim, inter_dim, E, topk,
        torch.bfloat16, dtypes.fp4x2, dtypes.fp4x2,
        QuantType.per_1x32, True, ActivationType.Silu,
        False, hidden_pad, intermediate_pad, True,
    )

    block_size_M = int(metadata.block_m)
    is_ksplit = metadata.ksplit > 1
    stage1 = metadata.stage1
    stage2 = metadata.stage2

    # Pre-allocate sorting buffers
    max_num_tokens_padded = int(M * topk + E * block_size_M - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size_M - 1) // block_size_M)

    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=torch.bfloat16, device=device)

    # Pre-allocate a2 buffer
    a2_buf = torch.empty((M, topk, inter_dim), dtype=torch.bfloat16, device=device)

    plan = {
        "block_size_M": block_size_M,
        "is_ksplit": is_ksplit,
        "stage1": stage1,
        "stage2": stage2,
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "moe_buf": moe_buf,
        "a2_buf": a2_buf,
        "E": E,
        "M": M,
        "topk": topk,
        "inter_dim": inter_dim,
    }

    if not is_ksplit:
        # Pre-compute quant constants for stage1 (input quant)
        plan["qc1"] = _build_quant_constants(M, model_dim, max_num_tokens_padded)
        # Pre-compute quant constants for inter-stage (a2 quant)
        plan["qc2"] = _build_quant_constants(M * topk, inter_dim, max_num_tokens_padded)

    return plan


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    M = hidden_states.shape[0]
    w1 = gate_up_weight_shuffled
    w2 = down_weight_shuffled

    # Fast shape key for plan cache
    plan_key = (M, w1.shape[0], w1.shape[1], w2.shape[2])
    plan = _plan_cache.get(plan_key)
    if plan is None:
        topk = topk_ids.shape[1]
        device = topk_ids.device
        E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
        hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
        intermediate_pad = config["d_expert_pad"] - config["d_expert"]
        plan = _build_plan(M, E, topk, model_dim, inter_dim, hidden_pad, intermediate_pad, device)
        _plan_cache[plan_key] = plan

    # Extract pre-resolved plan components (local vars for speed)
    sorted_ids = plan["sorted_ids"]
    sorted_weights = plan["sorted_weights"]
    sorted_expert_ids = plan["sorted_expert_ids"]
    num_valid_ids = plan["num_valid_ids"]
    moe_out = plan["moe_buf"]
    block_size_M = plan["block_size_M"]
    stage1 = plan["stage1"]
    stage2 = plan["stage2"]
    topk = plan["topk"]
    E = plan["E"]

    aiter.moe_sorting_fwd(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out,
        E, block_size_M, None, None, 0,
    )

    w1_scale_view = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    w2_scale_view = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)

    if plan["is_ksplit"]:
        # cktile_moe path: bf16 activations, no fp4 quant
        a2 = stage1(
            hidden_states, w1, w2,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            plan["a2_buf"], topk,
            block_m=block_size_M,
            a1_scale=None,
            w1_scale=w1_scale_view,
            sorted_weights=None,
        )

        stage2(
            a2, w1, w2,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            moe_out, topk,
            w2_scale=w2_scale_view,
            a2_scale=None,
            block_m=block_size_M,
            sorted_weights=sorted_weights,
        )
    else:
        # CK 2-stage path: fp4 activation quant with pre-allocated buffers
        a1, a1_scale = _run_quant(plan["qc1"], hidden_states, sorted_ids, num_valid_ids, M, 1)

        a2 = stage1(
            a1, w1, w2,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            plan["a2_buf"], topk,
            block_m=block_size_M,
            a1_scale=a1_scale,
            w1_scale=w1_scale_view,
            sorted_weights=None,
        )

        # Inter-stage requant: bf16 -> fp4
        inter_dim = plan["inter_dim"]
        a2_flat = a2.view(-1, inter_dim)
        a2_quant, a2_scale = _run_quant(plan["qc2"], a2_flat, sorted_ids, num_valid_ids, M, topk)
        a2_quant = a2_quant.view(M, topk, -1)

        stage2(
            a2_quant, w1, w2,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            moe_out, topk,
            w2_scale=w2_scale_view,
            a2_scale=a2_scale,
            block_m=block_size_M,
            sorted_weights=sorted_weights,
        )

    return moe_out
