#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v142: FlyDSL stage1 kernel (t32x256x256) for bs=512/E=33 shapes.
Replace CK 4WG M128 stage1 with FlyDSL native fp4 stage1 kernel.
FlyDSL stage1 kernels use tile_m=32, tile_n=256, tile_k=256 for fp4.
Create _flydsl_stage1_wrapper analogous to _flydsl_stage2_wrapper.
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
import aiter.ops.flydsl as _flydsl_ops

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

# Register FlyDSL stage1 kernel params (stage1 fp4 default is t32x256x256)
_flydsl_moe_kernels._KERNEL_PARAMS["flydsl_moe1_afp4_wfp4_bf16_t32x256x256"] = {
    "stage": 1, "a_dtype": "fp4", "b_dtype": "fp4", "out_dtype": "bf16",
    "tile_m": 32, "tile_n": 256, "tile_k": 256, "MPerBlock": 32,
}
_flydsl_moe_kernels._KERNEL_PARAMS["flydsl_moe1_afp4_wfp4_bf16_t16x256x256"] = {
    "stage": 1, "a_dtype": "fp4", "b_dtype": "fp4", "out_dtype": "bf16",
    "tile_m": 16, "tile_n": 256, "tile_k": 256, "MPerBlock": 16,
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

# === E=33 shapes (from v018, proven) ===
_CUSTOM_CONFIGS[_make_key(16, 512, 33)] = {
    "block_m": 32,
    "ksplit": 2,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": False,
}

_CUSTOM_CONFIGS[_make_key(128, 512, 33)] = {
    "block_m": 64,
    "ksplit": 2,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": False,
}

# === E=257 shapes ===
_CUSTOM_CONFIGS[_make_key(16, 256, 257)] = {
    "block_m": 16,
    "ksplit": 2,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": False,
}

_CUSTOM_CONFIGS[_make_key(128, 256, 257)] = {
    "block_m": 16,
    "ksplit": 2,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": False,
}

# === bs=512/E=33 shapes: FlyDSL stage1 + FlyDSL stage2 ===
# v142: Replace CK 4WG M128 stage1 with FlyDSL stage1 (t32x256x256)
_FLYDSL_STAGE1 = "flydsl_moe1_afp4_wfp4_bf16_t32x256x256"
_FLYDSL_STAGE2_M16_N128_K128 = "flydsl_moe2_afp4_wfp4_bf16_t16x128x128_atomic"

_CUSTOM_CONFIGS[_make_key(512, 2048, 33)] = {
    "block_m": 128,
    "ksplit": 0,
    "kernelName1": _FLYDSL_STAGE1,  # v142: FlyDSL stage1 instead of CK 4WG
    "kernelName2": _FLYDSL_STAGE2_M16_N128_K128,
    "run_1stage": False,
}

_CUSTOM_CONFIGS[_make_key(512, 512, 33)] = {
    "block_m": 128,
    "ksplit": 0,
    "kernelName1": _FLYDSL_STAGE1,  # v142: FlyDSL stage1 instead of CK 4WG
    "kernelName2": _FLYDSL_STAGE2_M16_N128_K128,
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


def _flydsl_stage1_wrapper(
    hidden_states, w1, w2,
    sorted_token_ids, sorted_expert_ids, num_valid_ids,
    out, topk,
    block_m=32,
    a1_scale=None, w1_scale=None,
    sorted_weights=None,
    kernelName="",
    **_kwargs,
):
    """Wrapper that adapts fused_moe_2stages stage1 calling convention to FlyDSL stage1."""
    parsed = _flydsl_moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        raise ValueError(f"Invalid FlyDSL stage1 kernel name: {kernelName}")

    # Convert w1_scale to e8m0 view if needed (fp4 weights have e8m0 scales)
    if w1_scale is not None and w1.dtype == torch.float4_e2m1fn_x2:
        w1_scale = w1_scale.view(torch.float8_e8m0fnu)

    _flydsl_ops.flydsl_moe_stage1(
        a=hidden_states,
        w1=w1,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=parsed["tile_m"],
        tile_n=parsed["tile_n"],
        tile_k=parsed["tile_k"],
        a_dtype=parsed["a_dtype"],
        b_dtype=parsed["b_dtype"],
        out_dtype=parsed["out_dtype"],
        w1_scale=w1_scale,
        a1_scale=a1_scale,
        sorted_weights=sorted_weights,
    )
    return out


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

    # Monkeypatch get_2stage_cfgs to:
    # 1. Support use_non_temporal_load from config
    # 2. Support FlyDSL stage1 kernels (kernelName1 starting with "flydsl_")
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
        if cfg is None:
            return metadata

        # Handle FlyDSL stage1 kernel override
        kn1 = str(cfg.get("kernelName1", ""))
        if kn1.startswith("flydsl_"):
            # Replace CK stage1 with FlyDSL stage1 wrapper
            metadata = _fused_moe_module.MOEMetadata(
                functools.partial(_flydsl_stage1_wrapper, kernelName=kn1),
                metadata.stage2,
                metadata.block_m,
                metadata.ksplit,
                metadata.run_1stage,
                metadata.has_bias,
                metadata.use_non_temporal_load,
            )

        # Handle NT override
        if cfg.get("use_non_temporal_load") is not None:
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
