#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v081: FlyDSL afp8 stage2 for bs512/E33/d=2048.
Use fp8 inter-stage quantization (cheaper than fp4) + FlyDSL afp8_wfp4 stage2.
The fp8 quant is a simple per-token scale cast vs fp4's per-1x32 block scaling.
This saves the expensive fused_dynamic_mxfp4_quant_moe_sort Triton kernel.
"""
import torch
import functools
from typing import Dict, Optional
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe, fused_moe_2stages as _orig_fused_moe_2stages
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

# === E=257 shapes (from v020) ===
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

# d=2048: Try FlyDSL afp8 stage2 (skip expensive fp4 inter-stage quant)
_FLYDSL_STAGE2_AFP8 = "flydsl_moe2_afp8_wfp4_bf16_t32x128x256_atomic"

_CUSTOM_CONFIGS[_make_key(512, 2048, 33)] = {
    "block_m": 64,
    "ksplit": 0,
    "kernelName1": _4WG_STAGE1,
    "kernelName2": _FLYDSL_STAGE2_AFP8,  # Try afp8 stage2
    "run_1stage": False,
}

# d=512: default stage1 + FlyDSL stage2 (proven v056)
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

    # Monkeypatch fused_moe_2stages to handle afp8 stage2 path
    # When the config specifies an afp8 FlyDSL kernel, we need to:
    # 1. Run stage1 normally (produces bf16 a2)
    # 2. Quantize a2 to fp8 (instead of fp4x2) - much cheaper
    # 3. Run FlyDSL afp8 stage2
    _orig_2stages = _fused_moe_module.fused_moe_2stages

    def _patched_fused_moe_2stages(
        hidden_states, w1, w2, topk,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_out, isG1U1, block_size_M,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        doweight_stage1=False,
        q_dtype_a=None, q_dtype_w=None,
        w1_scale=None, w2_scale=None,
        a1_scale=None, a2_scale=None,
        num_local_tokens=None,
        hidden_pad=0, intermediate_pad=0,
        bias1=None, bias2=None,
    ):
        from aiter.fused_moe import get_2stage_cfgs, get_padded_M, get_inter_dim
        from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort
        from aiter import get_hip_quant as get_quant
        from aiter.utility import fp4_utils, dtypes

        token_num, _ = hidden_states.shape
        E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
        dtype = moe_out.dtype
        device = hidden_states.device
        is_shuffled = getattr(w1, "is_shuffled", False)

        metadata = get_2stage_cfgs(
            get_padded_M(token_num),
            model_dim, inter_dim, E, topk, dtype,
            q_dtype_a, q_dtype_w, quant_type, isG1U1,
            activation, doweight_stage1,
            hidden_pad, intermediate_pad, is_shuffled,
        )

        # Check if stage2 uses afp8 FlyDSL
        stage2_keywords = getattr(metadata.stage2, 'keywords', {})
        kernelName2 = stage2_keywords.get('kernelName', '')
        uses_afp8 = 'afp8' in kernelName2

        if not uses_afp8:
            # Standard path
            return _orig_2stages(
                hidden_states, w1, w2, topk,
                sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                moe_out, isG1U1, block_size_M,
                activation=activation, quant_type=quant_type,
                doweight_stage1=doweight_stage1,
                q_dtype_a=q_dtype_a, q_dtype_w=q_dtype_w,
                w1_scale=w1_scale, w2_scale=w2_scale,
                a1_scale=a1_scale, a2_scale=a2_scale,
                num_local_tokens=num_local_tokens,
                hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
                bias1=bias1, bias2=bias2,
            )

        # afp8 path: custom inter-stage quantization
        quant_func = get_quant(quant_type)

        # Stage 1: standard fp4 quantization + CK GEMM
        a1, a1_scale_val = fused_dynamic_mxfp4_quant_moe_sort(
            hidden_states,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            topk=1,
            block_size=block_size_M,
        )

        a2 = torch.empty(
            (token_num, topk, inter_dim),
            dtype=dtype,
            device=device,
        )

        a2 = metadata.stage1(
            a1, w1, w2,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            a2, topk,
            block_m=block_size_M,
            a1_scale=a1_scale_val,
            w1_scale=(w1_scale.view(dtypes.fp8_e8m0) if w1.dtype == dtypes.fp4x2 else w1_scale),
            sorted_weights=sorted_weights if doweight_stage1 else None,
        )

        # Inter-stage: quantize bf16 -> fp8 (per-token scaling)
        # Much cheaper than bf16 -> fp4x2 (per-1x32 block scaling)
        a2_flat = a2.view(-1, inter_dim)
        a2_fp8 = torch.empty_like(a2_flat, dtype=dtypes.fp8)
        a2_fp8_scale = torch.empty(a2_flat.shape[0], dtype=dtypes.fp32, device=device)
        aiter.dynamic_per_token_scaled_quant(a2_fp8, a2_flat, a2_fp8_scale)
        a2_fp8 = a2_fp8.view(token_num, topk, inter_dim)

        # No need for moe_mxfp4_sort for fp8 - the scale is per-token, not per-block
        # But FlyDSL stage2 expects a2_scale in the same sorted format
        # For afp8, the scale needs to be passed as-is

        # Stage 2: FlyDSL afp8 GEMM
        metadata.stage2(
            a2_fp8, w1, w2,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            moe_out, topk,
            w2_scale=(w2_scale.view(dtypes.fp8_e8m0) if w2.dtype == dtypes.fp4x2 else w2_scale),
            a2_scale=a2_fp8_scale,
            block_m=block_size_M,
            sorted_weights=sorted_weights if not doweight_stage1 else None,
        )

        return moe_out

    _fused_moe_module.fused_moe_2stages = _patched_fused_moe_2stages


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
