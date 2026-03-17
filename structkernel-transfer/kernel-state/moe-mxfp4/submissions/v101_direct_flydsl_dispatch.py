#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v101: Direct FlyDSL stage2 dispatch - bypass flydsl_moe_stage2 Python overhead.
Instead of going through _flydsl_stage2_wrapper -> flydsl_moe_stage2 -> _get_compiled_stage2,
monkeypatch the stage2 partial to call _get_compiled_stage2's tensor_api directly.
Saves ~5-10us per FlyDSL stage2 call from avoiding parameter parsing, dummy tensor
creation, and persist_m computation on every invocation.
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

# Inject ksplit=2 configs for shapes that benefit from cktile_moe path
_CUSTOM_CONFIGS = {}

def _make_key(token, inter_dim, expert):
    return (
        256, token, 7168, inter_dim, expert, 9,
        "ActivationType.Silu", "torch.bfloat16",
        "torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2",
        "QuantType.per_1x32", True, False,
    )

# === E=33 shapes (from v018, proven) ===
# bs=16/E=33/d=512: cktile_moe gives 59.6us vs 88.7us baseline (-32.8%)
_CUSTOM_CONFIGS[_make_key(16, 512, 33)] = {
    "block_m": 32,
    "ksplit": 2,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": False,
}

# bs=128/E=33/d=512: cktile_moe gives 108us vs 124us baseline (-12.9%)
_CUSTOM_CONFIGS[_make_key(128, 512, 33)] = {
    "block_m": 64,
    "ksplit": 2,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": False,
}

# === E=257 shapes (NEW in v020) ===
# bs=16/E=257/d=256: cktile_moe ksplit=2
_CUSTOM_CONFIGS[_make_key(16, 256, 257)] = {
    "block_m": 16,
    "ksplit": 2,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": False,
}

# bs=128/E=257/d=256: cktile_moe ksplit=2
_CUSTOM_CONFIGS[_make_key(128, 256, 257)] = {
    "block_m": 16,
    "ksplit": 2,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": False,
}

# === bs=512/E=33 shapes: inject 4-WG stage1 kernel + FlyDSL stage2 ===
_4WG_STAGE1 = "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
_FLYDSL_STAGE2 = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic"

_CUSTOM_CONFIGS[_make_key(512, 2048, 33)] = {
    "block_m": 64,
    "ksplit": 0,
    "kernelName1": _4WG_STAGE1,
    "kernelName2": _FLYDSL_STAGE2,
    "run_1stage": False,
}

# === bs=512/E=33/d=512: inject FlyDSL stage2 ===
_CUSTOM_CONFIGS[_make_key(512, 512, 33)] = {
    "block_m": 64,
    "ksplit": 0,
    "kernelName1": "",
    "kernelName2": _FLYDSL_STAGE2,
    "run_1stage": False,
}

# === bs=512/E=257: inject tuned CSV kernels + NT=True ===
_CUSTOM_CONFIGS[_make_key(512, 256, 257)] = {
    "block_m": 32,
    "ksplit": 0,
    "kernelName1": "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    "kernelName2": "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    "run_1stage": False,
    "use_non_temporal_load": True,
}

_injected = False

def _make_direct_flydsl_stage2(kernelName):
    """Create a direct FlyDSL stage2 callable that bypasses flydsl_moe_stage2 Python overhead.

    The standard path: _flydsl_stage2_wrapper -> get_flydsl_kernel_params -> flydsl_moe_stage2
      -> compute persist_m -> _get_compiled_stage2 -> tensor_api(...)
    This direct path: pre-compute all params once, call tensor_api directly.
    """
    parsed = aiter.ops.flydsl.moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        return None  # Fall back to standard wrapper

    tile_m = parsed["tile_m"]
    tile_n = parsed["tile_n"]
    tile_k = parsed["tile_k"]
    a_dtype = parsed["a_dtype"]
    b_dtype = parsed["b_dtype"]
    out_dtype = parsed["out_dtype"]
    mode = parsed.get("mode", "atomic")
    accumulate = mode != "reduce"

    # Cache for compiled tensor_api per (model_dim, inter_dim, E, topk)
    _api_cache = {}

    def direct_stage2(
        inter_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        w2_scale=None,
        a2_scale=None,
        sorted_weights=None,
        **_kwargs,
    ):
        token_num = inter_states.shape[0]
        E = w2.shape[0]
        model_dim = w2.shape[1]
        inter_dim = inter_states.shape[2]
        if a_dtype == "fp4":
            inter_dim_compile = inter_dim * 2
        else:
            inter_dim_compile = inter_dim

        cache_key = (model_dim, inter_dim_compile, E, topk)
        tensor_api = _api_cache.get(cache_key)
        if tensor_api is None:
            from aiter.ops.flydsl.moe_kernels import _get_compiled_stage2
            # Auto-select persistent M
            _persist_m = 4 if int(sorted_expert_ids.numel()) > 256 else 1
            try:
                tensor_api = _get_compiled_stage2(
                    model_dim=model_dim,
                    inter_dim=inter_dim_compile,
                    experts=E,
                    topk=topk,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    doweight=(sorted_weights is not None),
                    a_dtype=a_dtype,
                    b_dtype=b_dtype,
                    out_dtype=out_dtype,
                    accumulate=accumulate,
                    persist_m=_persist_m,
                )
            except TypeError:
                # Server may not have persist_m parameter
                tensor_api = _get_compiled_stage2(
                    model_dim=model_dim,
                    inter_dim=inter_dim_compile,
                    experts=E,
                    topk=topk,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    doweight=(sorted_weights is not None),
                    a_dtype=a_dtype,
                    b_dtype=b_dtype,
                    out_dtype=out_dtype,
                    accumulate=accumulate,
                )
            _api_cache[cache_key] = tensor_api

        dev = inter_states.device
        sw = (
            sorted_weights
            if sorted_weights is not None
            else torch.empty(sorted_token_ids.shape, dtype=torch.float32, device=dev)
        )

        # Call tensor_api directly
        if a_dtype == "fp4" or b_dtype == "fp4":
            empty_bias = torch.empty(0, device=dev, dtype=torch.float32)
            stream = torch.cuda.current_stream().cuda_stream
            tensor_api(
                out,
                inter_states,
                w2,
                a2_scale,
                w2_scale,
                sorted_token_ids,
                sorted_expert_ids,
                sw,
                num_valid_ids,
                empty_bias,
                token_num,
                model_dim,
                inter_dim_compile,
                int(sorted_expert_ids.numel()),
                stream,
            )
        else:
            tensor_api(
                out,
                inter_states,
                w2,
                a2_scale,
                w2_scale,
                sorted_token_ids,
                sorted_expert_ids,
                sw,
                num_valid_ids,
                token_num,
                model_dim,
                inter_dim_compile,
                int(sorted_expert_ids.numel()),
            )

    return direct_stage2


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

    # Create direct FlyDSL stage2 wrapper
    _direct_flydsl = _make_direct_flydsl_stage2(_FLYDSL_STAGE2)

    # Monkeypatch get_2stage_cfgs to support:
    # 1. use_non_temporal_load from config
    # 2. Direct FlyDSL dispatch (bypass Python wrapper overhead)
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

        # Replace FlyDSL stage2 wrapper with direct dispatch
        if _direct_flydsl is not None and cfg is not None:
            kn2 = str(cfg.get("kernelName2", ""))
            if kn2 == _FLYDSL_STAGE2:
                # Replace the stage2 partial with direct dispatch
                metadata = _fused_moe_module.MOEMetadata(
                    metadata.stage1,
                    _direct_flydsl,
                    metadata.block_m,
                    metadata.ksplit,
                    metadata.run_1stage,
                    metadata.has_bias,
                    getattr(metadata, 'use_non_temporal_load', False),
                )

        # Handle NT override
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
