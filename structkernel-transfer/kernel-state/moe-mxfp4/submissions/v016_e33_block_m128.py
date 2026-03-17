#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v016: Inject tile configs for E=33 shapes — try 256x32x128x128_1x4 stage1 kernel
for bs=128 (heuristic picks 256x64x128x128_1x4 via block_m=64) to match the tuned
E=257 pattern where block_m=32 wins. Only change bs=128 and bs=512 for d=512.
Keep d=2048 at heuristic default (block_m=128).
"""
import torch
from typing import Dict
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as _fused_moe_module

# Kernel names from the compiled CK instances
_STAGE1_32 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
_STAGE1_128 = "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
# Stage2 kernels for inter_dim > 256
_STAGE2_32_LARGE = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
_STAGE2_64_LARGE = "moe_ck2stages_gemm2_256x64x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
_STAGE2_128_LARGE = "moe_ck2stages_gemm2_256x128x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
# E=257 tuned stage2 (inter_dim=256 <= 256)
_STAGE2_128_SMALL = "moe_ck2stages_gemm2_64x128x128x128_1x1_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

_E33_CONFIGS = {}
_COMMON = {"ksplit": 0, "run_1stage": False}

def _make_key(token, inter_dim, expert=33):
    return (
        256, token, 7168, inter_dim, expert, 9,
        "ActivationType.Silu", "torch.bfloat16",
        "torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2",
        "QuantType.per_1x32", True, False,
    )

# bs=128/E=33/d=512: heuristic picks block_m=64. Try block_m=128 with larger tiles.
# With 128*9/33≈35 tokens/expert, block_m=128 may be too large for per-expert M,
# but the 1x4 wave layout can still be efficient for the N=512 dimension.
_E33_CONFIGS[_make_key(128, 512)] = {
    "block_m": 128,
    "kernelName1": _STAGE1_128,
    "kernelName2": _STAGE2_128_LARGE,
    **_COMMON,
}

# bs=512/E=33/d=512: heuristic picks block_m=64. Try block_m=128 with larger tiles.
# With 512*9/33≈140 tokens/expert, block_m=128 is well-suited.
_E33_CONFIGS[_make_key(512, 512)] = {
    "block_m": 128,
    "kernelName1": _STAGE1_128,
    "kernelName2": _STAGE2_128_LARGE,
    **_COMMON,
}

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

    _fused_moe_module.cfg_2stages.update(_E33_CONFIGS)


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
