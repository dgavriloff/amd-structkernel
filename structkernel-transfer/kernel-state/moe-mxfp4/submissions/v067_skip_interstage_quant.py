#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v067: Skip inter-stage re-quantization for bs512/E33 shapes.
Use FlyDSL stage2 with afp16 (bf16 activations) instead of afp4.
This eliminates the fused_dynamic_mxfp4_quant_moe_sort call between stage1 and stage2.
Hypothesis: quant overhead is significant for bs512 with many token-expert pairs.
"""
import torch
import functools
from typing import Dict, Optional
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

# === bs=512/E=33 shapes: 4-WG stage1 + FlyDSL afp16 stage2 (skip inter-stage quant) ===
_4WG_STAGE1 = "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"

# Use afp16 variant to accept bf16 activations directly (no inter-stage re-quant)
_FLYDSL_STAGE2_AFP16 = "flydsl_moe2_afp16_wfp4_bf16_t32x128x256_atomic"
# Keep afp4 variant for fallback
_FLYDSL_STAGE2_AFP4 = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic"

_CUSTOM_CONFIGS[_make_key(512, 2048, 33)] = {
    "block_m": 64,
    "ksplit": 0,
    "kernelName1": _4WG_STAGE1,
    "kernelName2": _FLYDSL_STAGE2_AFP16,
    "run_1stage": False,
    "_skip_interstage_quant": True,
}

_CUSTOM_CONFIGS[_make_key(512, 512, 33)] = {
    "block_m": 64,
    "ksplit": 0,
    "kernelName1": "",
    "kernelName2": _FLYDSL_STAGE2_AFP16,
    "run_1stage": False,
    "_skip_interstage_quant": True,
}

# NOTE: bs=512/E=257 uses tuned CSV (cktile_moe worse for that shape)

_injected = False
_patched = False

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


# Shapes that should skip inter-stage re-quantization
_SKIP_QUANT_SHAPES = set()

def _patch_pipeline():
    """Monkeypatch fused_moe_2stages to skip inter-stage requant for afp16 shapes."""
    global _patched
    if _patched:
        return
    _patched = True

    # Build set of (token, inter_dim, expert) tuples that skip quant
    for key, cfg in _CUSTOM_CONFIGS.items():
        if cfg.get("_skip_interstage_quant", False):
            # key format: (cu_num, token, model_dim, inter_dim, expert, topk, ...)
            token = key[1]
            inter_dim = key[3]
            expert = key[4]
            _SKIP_QUANT_SHAPES.add((token, inter_dim, expert))

    _orig_fused_moe_2stages = _fused_moe_module.fused_moe_2stages

    def _patched_fused_moe_2stages(
        hidden_states, w1, w2, topk, sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids, moe_out, isG1U1, block_size_M,
        activation=ActivationType.Silu, quant_type=QuantType.No,
        doweight_stage1=False, q_dtype_a=None, q_dtype_w=None,
        w1_scale=None, w2_scale=None, a1_scale=None, a2_scale=None,
        num_local_tokens=None, hidden_pad=0, intermediate_pad=0,
        bias1=None, bias2=None,
    ):
        import aiter
        from aiter import dtypes
        from aiter.fused_moe import (
            get_inter_dim, get_padded_M, get_2stage_cfgs,
            fused_dynamic_mxfp4_quant_moe_sort,
        )
        from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
        from aiter import get_hip_quant as get_quant

        token_num, _ = hidden_states.shape
        E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
        dtype = moe_out.dtype
        device = hidden_states.device
        is_shuffled = getattr(w1, "is_shuffled", False)

        # Check if this shape should skip inter-stage quant
        should_skip = (token_num, inter_dim, E) in _SKIP_QUANT_SHAPES

        if not should_skip:
            # Use original function for non-patched shapes
            return _orig_fused_moe_2stages(
                hidden_states, w1, w2, topk, sorted_ids, sorted_weights,
                sorted_expert_ids, num_valid_ids, moe_out, isG1U1, block_size_M,
                activation=activation, quant_type=quant_type,
                doweight_stage1=doweight_stage1, q_dtype_a=q_dtype_a, q_dtype_w=q_dtype_w,
                w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a1_scale, a2_scale=a2_scale,
                num_local_tokens=num_local_tokens, hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
                bias1=bias1, bias2=bias2,
            )

        # Custom pipeline with skipped inter-stage quant
        metadata = get_2stage_cfgs(
            get_padded_M(token_num), model_dim, inter_dim, E, topk,
            dtype, q_dtype_a, q_dtype_w, quant_type, isG1U1, activation,
            doweight_stage1, hidden_pad, intermediate_pad, is_shuffled,
        )

        # Step 1: Quantize input activations (same as original)
        if token_num <= 1024:
            a1, a1_scale_out = fused_dynamic_mxfp4_quant_moe_sort(
                hidden_states,
                sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
                token_num=token_num, topk=1, block_size=block_size_M,
            )
        else:
            quant_func = get_quant(quant_type)
            a1, a1_scale_out = quant_func(
                hidden_states, scale=a1_scale, quant_dtype=q_dtype_a,
                num_rows=num_local_tokens,
            )
            from aiter.utility import fp4_utils
            a1_scale_out = fp4_utils.moe_mxfp4_sort(
                a1_scale_out, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
                token_num=token_num, block_size=block_size_M,
            )

        # Step 2: Stage1 (fp4 x fp4 -> bf16)
        a2 = torch.empty((token_num, topk, inter_dim), dtype=dtype, device=device)
        metadata.stage1(
            a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
            a2, topk, block_m=block_size_M,
            a1_scale=a1_scale_out,
            w1_scale=w1_scale.view(dtypes.fp8_e8m0) if w1.dtype == dtypes.fp4x2 else w1_scale,
            sorted_weights=sorted_weights if doweight_stage1 else None,
        )

        # Step 3: SKIP inter-stage re-quantization! Pass bf16 a2 directly.
        # FlyDSL afp16 stage2 can handle bf16 activations.
        metadata.stage2(
            a2, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
            moe_out, topk,
            w2_scale=w2_scale.view(dtypes.fp8_e8m0) if w2.dtype == dtypes.fp4x2 else w2_scale,
            a2_scale=None,  # No activation scale needed for bf16 input
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
    _patch_pipeline()

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
