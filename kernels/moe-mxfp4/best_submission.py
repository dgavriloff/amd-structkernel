#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v165: Pre-allocate moe_sorting + intermediate buffers per shape.
Bypass fused_moe wrapper, call low-level aiter ops directly with
pre-allocated tensors to eliminate torch.empty allocation overhead.
"""
import os
import functools
import torch
from typing import Dict, Tuple
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    moe_sorting, get_2stage_cfgs, get_padded_M, get_inter_dim,
    fused_moe_2stages, ck_moe_stage1,
)
import aiter.fused_moe as _fused_moe_module
import aiter.ops.flydsl.moe_kernels as _flydsl_moe_kernels
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.utility import fp4_utils

# Register FlyDSL tile_k=128 kernels that aren't in server's default registration
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

# Inject ksplit=2 configs for shapes that benefit from cktile_moe path
_CUSTOM_CONFIGS = {}

def _make_key(token, inter_dim, expert):
    return (
        256, token, 7168, inter_dim, expert, 9,
        "ActivationType.Silu", "torch.bfloat16",
        "torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2",
        "QuantType.per_1x32", True, False,
    )

# === E=33 shapes ===
_CUSTOM_CONFIGS[_make_key(16, 512, 33)] = {
    "block_m": 32, "ksplit": 2, "kernelName1": "", "kernelName2": "",
    "run_1stage": False,
}

_4WG_STAGE1_M128 = "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
_4WG_STAGE1_M32 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
_FLYDSL_STAGE2_M16_N128_K128 = "flydsl_moe2_afp4_wfp4_bf16_t16x128x128_atomic"

_CUSTOM_CONFIGS[_make_key(128, 512, 33)] = {
    "block_m": 64, "ksplit": 0,
    "kernelName1": _4WG_STAGE1_M128,
    "kernelName2": _FLYDSL_STAGE2_M16_N128_K128,
    "run_1stage": False,
}

# === E=257 shapes ===
_CUSTOM_CONFIGS[_make_key(16, 256, 257)] = {
    "block_m": 16, "ksplit": 2, "kernelName1": "", "kernelName2": "",
    "run_1stage": False,
}
_CUSTOM_CONFIGS[_make_key(128, 256, 257)] = {
    "block_m": 16, "ksplit": 2, "kernelName1": "", "kernelName2": "",
    "run_1stage": False,
}

# === bs=512 shapes ===
_CUSTOM_CONFIGS[_make_key(512, 2048, 33)] = {
    "block_m": 64, "ksplit": 0,
    "kernelName1": _4WG_STAGE1_M128,
    "kernelName2": _FLYDSL_STAGE2_M16_N128_K128,
    "run_1stage": False,
}
_CUSTOM_CONFIGS[_make_key(512, 512, 33)] = {
    "block_m": 64, "ksplit": 0,
    "kernelName1": _4WG_STAGE1_M128,
    "kernelName2": _FLYDSL_STAGE2_M16_N128_K128,
    "run_1stage": False,
}
_CUSTOM_CONFIGS[_make_key(512, 256, 257)] = {
    "block_m": 32, "ksplit": 0,
    "kernelName1": _4WG_STAGE1_M32,
    "kernelName2": _FLYDSL_STAGE2_M16_N128_K128,
    "run_1stage": False,
    "use_non_temporal_load": True,
}

# Pre-allocated buffer cache: shape_key -> dict of pre-allocated tensors
_buffer_cache = {}

def _get_shape_key(M, E, topk, model_dim, inter_dim, block_size_M):
    return (M, E, topk, model_dim, inter_dim, block_size_M)

def _get_or_alloc_buffers(M, E, topk, model_dim, inter_dim, block_size_M, device):
    """Get pre-allocated buffers for the given shape, allocating on first call."""
    key = _get_shape_key(M, E, topk, model_dim, inter_dim, block_size_M)
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
        # a2 intermediate buffer for 2-stage path
        "a2": torch.empty((M, topk, inter_dim), dtype=torch.bfloat16, device=device),
    }
    _buffer_cache[key] = bufs
    return bufs


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


def _fast_moe_sorting(topk_ids, topk_weights, num_experts, model_dim, block_size, device):
    """Call moe_sorting with pre-allocated buffers to avoid per-call allocation."""
    M, topk = topk_ids.shape
    bufs = _get_or_alloc_buffers(M, num_experts, topk, model_dim, model_dim, block_size, device)

    sorted_ids = bufs["sorted_ids"]
    sorted_weights = bufs["sorted_weights"]
    sorted_expert_ids = bufs["sorted_expert_ids"]
    num_valid_ids = bufs["num_valid_ids"]
    moe_buf = bufs["moe_buf"]

    aiter.moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        int(block_size),
        None,  # expert_mask
        None,  # num_local_tokens
        0,     # dispatch_policy
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


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
    E, model_dim, inter_dim = get_inter_dim(
        gate_up_weight_shuffled.shape, down_weight_shuffled.shape
    )

    padded_M = get_padded_M(M)
    metadata = get_2stage_cfgs(
        padded_M, model_dim, inter_dim, E, topk,
        torch.bfloat16, dtypes.fp4x2, dtypes.fp4x2,
        QuantType.per_1x32, True, ActivationType.Silu,
        False, hidden_pad, intermediate_pad, True,
    )

    block_size_M = int(metadata.block_m)

    # Use pre-allocated buffers for moe_sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = \
        _fast_moe_sorting(topk_ids, topk_weights, E, model_dim, block_size_M, topk_ids.device)

    # Now run the 2-stage pipeline using the standard fused_moe_2stages
    # which handles quantization + stage1 + re-quant + stage2
    return fused_moe_2stages(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        True,  # isG1U1
        block_size_M,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        q_dtype_a=dtypes.fp4x2,
        q_dtype_w=dtypes.fp4x2,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )
