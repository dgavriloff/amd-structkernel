#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v072: FlyDSL stage1 for bs512/E33/d=2048.
Monkeypatch ck_moe_stage1 to redirect to FlyDSL stage1 when kernel name
contains "FLYDSL_REDIRECT". FlyDSL stage1 with tile_m=64 may be faster
than CK 4-WG stage1 for this shape.
Gate-up GEMM: M=~139 tokens/expert, N=2*2048=4096, K=7168.
"""
import torch
import functools
from typing import Dict
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as _fused_moe_module

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
    "block_m": 32, "ksplit": 2, "kernelName1": "", "kernelName2": "", "run_1stage": False,
}
_CUSTOM_CONFIGS[_make_key(128, 512, 33)] = {
    "block_m": 64, "ksplit": 2, "kernelName1": "", "kernelName2": "", "run_1stage": False,
}

# === E=257 shapes ===
_CUSTOM_CONFIGS[_make_key(16, 256, 257)] = {
    "block_m": 16, "ksplit": 2, "kernelName1": "", "kernelName2": "", "run_1stage": False,
}
_CUSTOM_CONFIGS[_make_key(128, 256, 257)] = {
    "block_m": 16, "ksplit": 2, "kernelName1": "", "kernelName2": "", "run_1stage": False,
}

# === bs=512/E=33 shapes ===
_FLYDSL_STAGE2 = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic"

# Encode FlyDSL stage1 info in kernelName1 with "ck2stages" to pass the code path check.
# Format: "ck2stages_FLYDSL_REDIRECT_flydsl_moe1_afp4_wfp4_bf16_t64x256x256"
_FLYDSL_S1_D2048 = "ck2stages_FLYDSL_REDIRECT_flydsl_moe1_afp4_wfp4_bf16_t64x256x256"

_CUSTOM_CONFIGS[_make_key(512, 2048, 33)] = {
    "block_m": 64,
    "ksplit": 0,
    "kernelName1": _FLYDSL_S1_D2048,
    "kernelName2": _FLYDSL_STAGE2,
    "run_1stage": False,
}

_CUSTOM_CONFIGS[_make_key(512, 512, 33)] = {
    "block_m": 64,
    "ksplit": 0,
    "kernelName1": "",
    "kernelName2": _FLYDSL_STAGE2,
    "run_1stage": False,
}

# --- FlyDSL stage1 monkeypatch ---
_orig_ck_moe_stage1 = _fused_moe_module.ck_moe_stage1

def _patched_ck_moe_stage1(
    hidden_states, w1, w2,
    sorted_token_ids, sorted_expert_ids, num_valid_ids,
    out, topk, block_m, a1_scale, w1_scale,
    kernelName="", sorted_weights=None,
    quant_type=QuantType.No, activation=ActivationType.Gelu,
    splitk=1, use_non_temporal_load=False, dtype=None,
):
    if kernelName and "FLYDSL_REDIRECT" in kernelName:
        # Extract real FlyDSL kernel name after the redirect marker
        real_name = kernelName.split("FLYDSL_REDIRECT_")[1]

        import aiter.ops.flydsl.moe_kernels as flydsl_kernels
        parsed = flydsl_kernels.get_flydsl_kernel_params(real_name)
        if parsed is None:
            raise ValueError(f"Invalid FlyDSL stage1 kernel name: {real_name}")

        import aiter.ops.flydsl as flydsl_mod
        flydsl_mod.flydsl_moe_stage1(
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

    return _orig_ck_moe_stage1(
        hidden_states, w1, w2,
        sorted_token_ids, sorted_expert_ids, num_valid_ids,
        out, topk, block_m, a1_scale, w1_scale,
        kernelName, sorted_weights, quant_type, activation,
        splitk, use_non_temporal_load, dtype,
    )


_injected = False

def _inject_configs():
    global _injected
    if _injected:
        return
    _injected = True

    # Monkeypatch ck_moe_stage1 in the module's global namespace
    _fused_moe_module.ck_moe_stage1 = _patched_ck_moe_stage1

    # Also need to clear the get_2stage_cfgs LRU cache so it picks up
    # the patched function when creating new functools.partial objects
    _fused_moe_module.get_2stage_cfgs.cache_clear()

    if _fused_moe_module.cfg_2stages is None:
        import os, pandas as pd
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
