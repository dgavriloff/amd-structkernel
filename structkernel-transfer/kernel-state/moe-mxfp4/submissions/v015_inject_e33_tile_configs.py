#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v015: Inject tuned tile configs for E=33 shapes via cfg_2stages monkeypatch.
The tuned CSV only covers E=257. For E=33, the heuristic picks suboptimal block_m
values (64 or 128) for larger batch sizes. Force block_m=32 with the small-tile
64x32x32x128_1x1 kernels that perform best in the E=257 tuned configs.
"""
import torch
from typing import Dict
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as _fused_moe_module

# Inject tuned configs for E=33 shapes into cfg_2stages
# Key format: (cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1)

_STAGE1_KERNEL = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
# For inter_dim > 256, stage2 with block_m=32 should use the large-tile variant
_STAGE2_KERNEL_LARGE = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

_E33_CONFIGS = {}
_COMMON = {"ksplit": 0, "run_1stage": False}

# E=33 shapes with inter_dim=512 (TP=4): bs=16,128,512 → tokens=16,128,512
for token in [16, 128, 512]:
    key = (
        256, token, 7168, 512, 33, 9,
        "ActivationType.Silu", "torch.bfloat16",
        "torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2",
        "QuantType.per_1x32", True, False,
    )
    _E33_CONFIGS[key] = {
        "block_m": 32,
        "kernelName1": _STAGE1_KERNEL,
        "kernelName2": _STAGE2_KERNEL_LARGE,
        **_COMMON,
    }

# E=33 shape with inter_dim=2048 (EP-on): bs=512
key = (
    256, 512, 7168, 2048, 33, 9,
    "ActivationType.Silu", "torch.bfloat16",
    "torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2",
    "QuantType.per_1x32", True, False,
)
_E33_CONFIGS[key] = {
    "block_m": 32,
    "kernelName1": _STAGE1_KERNEL,
    "kernelName2": _STAGE2_KERNEL_LARGE,
    **_COMMON,
}

_injected = False

def _inject_configs():
    """Inject E=33 configs into cfg_2stages after CSV has been loaded."""
    global _injected
    if _injected:
        return
    _injected = True

    # Trigger CSV load if not yet loaded
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

    # Now inject our E=33 configs on top
    _fused_moe_module.cfg_2stages.update(_E33_CONFIGS)


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    _inject_configs()

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    output = fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )

    return output
