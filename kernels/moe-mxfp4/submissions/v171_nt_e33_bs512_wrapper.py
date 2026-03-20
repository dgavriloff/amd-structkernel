#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v171: Enable non-temporal loads for bs=512/E=33/d=512 and d=2048.
Uses wrapper-based approach (fused_moe) to avoid server stream check.
"""
import os
import functools
import torch
from typing import Dict
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as _fused_moe_module
import aiter
import aiter.ops.flydsl.moe_kernels as _flydsl_moe_kernels

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
    "use_non_temporal_load": True,
}
_CUSTOM_CONFIGS[_make_key(512, 2048, 33)] = {
    "block_m": 64, "ksplit": 0,
    "kernelName1": _4WG_STAGE1_M128, "kernelName2": _FLYDSL_STAGE2_M16_K128,
    "run_1stage": False,
    "use_non_temporal_load": True,
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

    output = fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

    return output
