#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v078: Force persist_m=4 for FlyDSL stage2 on bs512/E33 shapes.
The auto-selector uses persist_m=1 because sorted_expert_ids < 256 blocks.
With persist_m=4, each workgroup processes 4 M-tiles persistently, reducing
dispatch overhead and improving occupancy for the down-GEMM.
"""
import torch
from typing import Dict
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as _fused_moe_module
import aiter.ops.flydsl.moe_kernels as _flydsl_moe_kernels

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

# === bs=512/E=33 shapes ===
_4WG_STAGE1 = "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
_FLYDSL_STAGE2 = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic"

_CUSTOM_CONFIGS[_make_key(512, 2048, 33)] = {
    "block_m": 64,
    "ksplit": 0,
    "kernelName1": _4WG_STAGE1,
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

    # Monkeypatch flydsl_moe_stage2 to force persist_m=4
    _orig_flydsl_moe_stage2 = _flydsl_moe_kernels.flydsl_moe_stage2

    def _patched_flydsl_moe_stage2(
        inter_states, w2, sorted_token_ids, sorted_expert_ids,
        num_valid_ids, out=None, topk=1, **kwargs
    ):
        # Force persist_m=4 by making sorted_expert_ids appear large
        # The auto-selector in flydsl_moe_stage2 checks:
        #   _persist_m = 4 if int(sorted_expert_ids.numel()) > 256 else 1
        # We want persist_m=4, so we need numel > 256.
        # Instead of modifying the tensor, monkeypatch _get_compiled_stage2
        # to always use persist_m=4.
        token_num = inter_states.shape[0]
        E = w2.shape[0]
        model_dim = w2.shape[1]
        inter_dim = inter_states.shape[2]

        parsed = _flydsl_moe_kernels.get_flydsl_kernel_params(kwargs.get('tile_m', None) and "dummy" or "")
        # Just call original with persist_m override not possible via API
        # Instead, directly call _get_compiled_stage2 with persist_m=4
        return _orig_flydsl_moe_stage2(
            inter_states, w2, sorted_token_ids, sorted_expert_ids,
            num_valid_ids, out, topk, **kwargs
        )

    # Actually, the persist_m is computed INSIDE flydsl_moe_stage2, not exposed.
    # We need to override _get_compiled_stage2 or the persist_m logic.
    # Let's override the function to always use persist_m=4:
    import types
    _orig_stage2 = _flydsl_moe_kernels.flydsl_moe_stage2

    def _force_persist_m4_stage2(
        inter_states, w2, sorted_token_ids, sorted_expert_ids,
        num_valid_ids, out=None, topk=1, **kwargs
    ):
        assert out is not None
        token_num = inter_states.shape[0]
        E = w2.shape[0]
        model_dim = w2.shape[1]
        inter_dim = inter_states.shape[2]

        a_dtype = kwargs.get("a_dtype", "fp8")
        b_dtype = kwargs.get("b_dtype", "fp4")
        out_dtype = kwargs.get("out_dtype", "bf16")
        mode = kwargs.get("mode", "atomic")
        tile_m = kwargs.get("tile_m", 32)
        tile_n = kwargs.get("tile_n", 128)
        tile_k = kwargs.get("tile_k", 256)
        w2_scale = kwargs.get("w2_scale", None)
        a2_scale = kwargs.get("a2_scale", None)
        sorted_weights = kwargs.get("sorted_weights", None)

        if a_dtype == "fp4":
            inter_dim = inter_dim * 2

        accumulate = mode != "reduce"

        dev = inter_states.device
        sw = (
            sorted_weights
            if sorted_weights is not None
            else torch.empty(sorted_token_ids.shape, dtype=torch.float32, device=dev)
        )

        # Force persist_m=4 instead of auto-selection
        _persist_m = 4

        tensor_api = _flydsl_moe_kernels._get_compiled_stage2(
            model_dim=model_dim,
            inter_dim=inter_dim,
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
            int(sorted_expert_ids.numel()),
        )
        return out

    # Replace the module-level function
    import aiter.ops.flydsl
    aiter.ops.flydsl.flydsl_moe_stage2 = _force_persist_m4_stage2
    _flydsl_moe_kernels.flydsl_moe_stage2 = _force_persist_m4_stage2


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
