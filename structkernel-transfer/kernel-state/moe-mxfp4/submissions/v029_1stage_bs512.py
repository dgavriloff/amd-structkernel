#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v029: Try run_1stage=True for bs=512 shapes.
The fmoe_g1u1 1-stage kernel exists for per_1x32/fp4x2 on gfx950 (fused_moe_1stage_dict)
but the heuristic explicitly skips it for per_1x32 (only sets run_1stage for other quant types).
The 1-stage kernel fuses gate_up GEMM + SwiGLU + down GEMM into a single kernel launch,
eliminating the inter-stage buffer write/read. For bs=512 with enough tokens per expert,
this should reduce kernel launch overhead and memory traffic.
"""
import torch
from typing import Dict
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as _fused_moe_module

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

# === E=257 shapes (from v020/v021, proven) ===
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

# === NEW: 1-stage for bs=512 shapes ===
# bs=512/E=257/d=256: 1-stage fuses both GEMMs, eliminating inter-stage buffer
_CUSTOM_CONFIGS[_make_key(512, 256, 257)] = {
    "block_m": 32,
    "ksplit": 0,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": True,
}

# bs=512/E=33/d=512: 1-stage for larger expert dim
_CUSTOM_CONFIGS[_make_key(512, 512, 33)] = {
    "block_m": 32,
    "ksplit": 0,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": True,
}

# bs=512/E=33/d=2048: 1-stage for EP-on shape
_CUSTOM_CONFIGS[_make_key(512, 2048, 33)] = {
    "block_m": 32,
    "ksplit": 0,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": True,
}

# NOTE: bs=16/128 shapes use cktile_moe (ksplit=2) which is already optimal

_injected = False

def _inject_configs():
    global _injected
    if _injected:
        return
    _injected = True

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
