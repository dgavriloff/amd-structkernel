#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v100: Bypass fused_moe_ torch.ops dispatch wrapper.
Call fused_moe_ inner function body directly to skip:
- torch.ops.aiter dispatcher overhead (~2-5us per call)
- Redundant get_2stage_cfgs call in fused_moe_2stages
- Enum conversions and assert checks
Pre-compute all static metadata at injection time.
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
# bs=16/E=257/d=256: try cktile_moe ksplit=2 (overrides tuned CSV config)
# With 144 token-expert pairs across 257 experts, most experts get 0-1 tokens.
# Skipping activation quantization + using split-K may help.
_CUSTOM_CONFIGS[_make_key(16, 256, 257)] = {
    "block_m": 16,
    "ksplit": 2,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": False,
}

# bs=128/E=257/d=256: try cktile_moe ksplit=2 (overrides tuned CSV config)
# bs=128 has ~4.5 tokens/expert avg, similar to E=33 where ksplit=2 helped (-12.9%)
_CUSTOM_CONFIGS[_make_key(128, 256, 257)] = {
    "block_m": 16,
    "ksplit": 2,
    "kernelName1": "",
    "kernelName2": "",
    "run_1stage": False,
}

# === bs=512/E=33 shapes: inject 4-WG stage1 kernel ===
# The 256x64x128x128_1x4 kernel uses 4 workgroups per CU for better utilization.
# v037 showed d=2048: -3.2% (349->338µs). Now also try d=512 with same 4-WG kernel.
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
# v050 tried t32x256x256_reduce -> correctness failure. Try t32x128x256_atomic.
# Down-GEMM: K=512, tile_k=256 -> 2 K-iterations. N=7168, tile_n=128 -> 56 N-blocks.
_CUSTOM_CONFIGS[_make_key(512, 512, 33)] = {
    "block_m": 64,
    "ksplit": 0,
    "kernelName1": "",
    "kernelName2": _FLYDSL_STAGE2,
    "run_1stage": False,
}

# === bs=512/E=257: inject tuned CSV kernels + NT=True ===
# Heuristic says NT=True for 17 tokens/expert, but tuned CSV path always sets NT=False.
# Inject the same kernel names as tuned CSV but add use_non_temporal_load flag.
_CUSTOM_CONFIGS[_make_key(512, 256, 257)] = {
    "block_m": 32,
    "ksplit": 0,
    "kernelName1": "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    "kernelName2": "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    "run_1stage": False,
    "use_non_temporal_load": True,  # custom flag, read by monkeypatch
}

_injected = False

# Cache for per-shape metadata to avoid redundant lookups
_metadata_cache = {}

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
        # Get the original metadata
        metadata = _original_get_2stage_cfgs(
            token, model_dim, inter_dim, expert, topk,
            dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
            activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled,
        )

        # Check if this shape has a custom NT setting
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
            # Rebuild stage1 partial with NT override
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
                    # Also patch stage2 if it's a CK kernel (not FlyDSL)
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

    # Monkeypatch fused_moe_ to bypass torch.ops dispatch wrapper.
    # The torch_compile_guard decorator wraps fused_moe_ in torch.ops.aiter.fused_moe_
    # which goes through PyTorch's dispatcher. We replace it with direct function call.
    # Access the original unwrapped function via the closure.
    _original_fused_moe_2stages = _fused_moe_module.fused_moe_2stages

    def _direct_fused_moe_(
        hidden_states, w1, w2, topk_weight, topk_ids,
        expert_mask=None, activation=ActivationType.Silu.value,
        quant_type=QuantType.per_1x32.value, doweight_stage1=False,
        w1_scale=None, w2_scale=None, a1_scale=None, a2_scale=None,
        block_size_M=-1, num_local_tokens=None, moe_sorting_dispatch_policy=0,
        dtype=None, hidden_pad=0, intermediate_pad=0, bias1=None, bias2=None,
    ):
        # Inline the fused_moe_ body, skipping torch.ops dispatch
        activation_enum = ActivationType(activation)
        quant_type_enum = QuantType(quant_type)
        if block_size_M == -1:
            block_size_M = None

        M, topk = topk_ids.shape
        E, model_dim, inter_dim = _fused_moe_module.get_inter_dim(w1.shape, w2.shape)
        isG1U1 = inter_dim != w1.shape[1]
        isShuffled = getattr(w1, "is_shuffled", False)
        global_E = E
        if expert_mask is not None:
            global_E = expert_mask.numel()
        dtype = hidden_states.dtype if dtype is None else dtype
        quant_type_enum = _fused_moe_module.quant_remap.get(quant_type_enum, quant_type_enum)
        q_dtype_w = w1.dtype
        q_dtype_a = _fused_moe_module.dtypes.fp4x2  # For per_1x32 + Silu, always fp4x2

        padded_M = _fused_moe_module.get_padded_M(M)
        metadata = _fused_moe_module.get_2stage_cfgs(
            padded_M, model_dim, inter_dim, E, topk,
            dtype, q_dtype_a, q_dtype_w, quant_type_enum, isG1U1,
            activation_enum, doweight_stage1, hidden_pad, intermediate_pad, isShuffled,
        )

        block_size_M = metadata.block_m if block_size_M is None else block_size_M
        if block_size_M is not None:
            block_size_M = int(block_size_M)

        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _fused_moe_module.moe_sorting(
            topk_ids, topk_weight, global_E, model_dim, dtype,
            block_size_M, expert_mask, num_local_tokens, moe_sorting_dispatch_policy,
        )

        # Inline fused_moe_2stages for the per_1x32 fp4x2 path
        # Skip the redundant get_2stage_cfgs and get_inter_dim calls
        is_shuffled = isShuffled
        token_num = M
        device = hidden_states.device

        if metadata.ksplit > 1 and is_shuffled:
            # cktile_moe path: skip quantization
            a1 = hidden_states.to(dtype)
            a1_scale_local = None
        else:
            # 2-stage a4w4 path: need quantization
            if token_num <= 1024:
                a1, a1_scale_local = _fused_moe_module.fused_dynamic_mxfp4_quant_moe_sort(
                    hidden_states,
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=token_num,
                    topk=1,
                    block_size=block_size_M,
                )
            else:
                quant_func = _fused_moe_module.get_quant(quant_type_enum)
                a1, a1_scale_local = quant_func(
                    hidden_states, scale=a1_scale, quant_dtype=q_dtype_a,
                    num_rows=num_local_tokens,
                )
                a1_scale_local = _fused_moe_module.fp4_utils.moe_mxfp4_sort(
                    a1_scale_local, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
                    token_num=token_num, block_size=block_size_M,
                )

        a2 = torch.empty((token_num, topk, inter_dim), dtype=dtype, device=device)

        a2 = metadata.stage1(
            a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
            a2, topk, block_m=block_size_M,
            a1_scale=a1_scale_local,
            w1_scale=w1_scale.view(_fused_moe_module.dtypes.fp8_e8m0) if w1.dtype == _fused_moe_module.dtypes.fp4x2 else w1_scale,
            sorted_weights=sorted_weights if doweight_stage1 else None,
        )

        if metadata.ksplit > 1 and is_shuffled:
            # cktile_moe path: skip inter-stage quantization
            a2_scale_local = None
        else:
            # 2-stage a4w4 path: inter-stage quantization
            a2 = a2.view(-1, inter_dim)
            if token_num <= 1024:
                a2, a2_scale_local = _fused_moe_module.fused_dynamic_mxfp4_quant_moe_sort(
                    a2,
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=token_num,
                    topk=topk,
                    block_size=block_size_M,
                )
            else:
                quant_func = _fused_moe_module.get_quant(quant_type_enum)
                a2, a2_scale_local = quant_func(
                    a2, scale=a2_scale, quant_dtype=q_dtype_a,
                    num_rows=num_local_tokens, num_rows_factor=topk,
                )
                a2_scale_local = _fused_moe_module.fp4_utils.moe_mxfp4_sort(
                    a2_scale_local[:token_num * topk, :].view(token_num, topk, -1),
                    sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
                    token_num=token_num, block_size=block_size_M,
                )
            a2 = a2.view(token_num, topk, -1)

        metadata.stage2(
            a2, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
            moe_buf, topk,
            w2_scale=w2_scale.view(_fused_moe_module.dtypes.fp8_e8m0) if w2.dtype == _fused_moe_module.dtypes.fp4x2 else w2_scale,
            a2_scale=a2_scale_local,
            block_m=block_size_M,
            sorted_weights=sorted_weights if not doweight_stage1 else None,
        )

        return moe_buf

    # Replace fused_moe_ with our direct version
    _fused_moe_module.fused_moe_ = _direct_fused_moe_


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
