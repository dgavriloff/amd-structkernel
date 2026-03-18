#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v195: Keep the v194 no-cache kernel, but remove the global
AITER_USE_OPUS_MOE_SORTING env var.

Policy:
- E=257, M<=128: CK Tile with BYPASS_TUNE_CONFIG=1, KSPLIT=7, block_size_M=64
- E=257, M>128: CK Tile with BYPASS_TUNE_CONFIG=1 and KSPLIT=1
- E=33, M<=16: CK Tile with KSPLIT=7
- E=33, 16<M<=128: CK Tile with KSPLIT=2 and block_size_M=32
- E=33, M>128: default path with block_size_M=64 and Swiglu for d<=512

This isolates whether the shared sorting path benefits at all from the OPUS
env var once the kernel is using the current mixed dispatch policy.
"""
import gc
import os

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe, get_2stage_cfgs
from task import input_t, output_t

gc.disable()

_current_mode = None


def custom_kernel(data: input_t) -> output_t:
    global _current_mode
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

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    d_expert = config["d_expert"]

    if E > 64 and M <= 128:
        mode = "cktile_e257_k7"
    elif E > 64:
        mode = "cktile_e257_k1"
    elif E <= 64 and M <= 16:
        mode = "cktile_k7"
    elif E <= 64 and M <= 128:
        mode = "cktile_k2"
    else:
        mode = "default"

    if mode != _current_mode:
        if mode == "cktile_e257_k7":
            os.environ["AITER_KSPLIT"] = "7"
            os.environ["AITER_BYPASS_TUNE_CONFIG"] = "1"
        elif mode == "cktile_e257_k1":
            os.environ["AITER_KSPLIT"] = "1"
            os.environ["AITER_BYPASS_TUNE_CONFIG"] = "1"
        elif mode == "cktile_k7":
            os.environ["AITER_KSPLIT"] = "7"
            os.environ.pop("AITER_BYPASS_TUNE_CONFIG", None)
        elif mode == "cktile_k2":
            os.environ["AITER_KSPLIT"] = "2"
            os.environ.pop("AITER_BYPASS_TUNE_CONFIG", None)
        else:
            os.environ.pop("AITER_KSPLIT", None)
            os.environ.pop("AITER_BYPASS_TUNE_CONFIG", None)
        get_2stage_cfgs.cache_clear()
        _current_mode = mode

    if mode == "default" and E <= 64:
        bsm = 64
    elif mode == "cktile_k2":
        bsm = 32
    elif mode == "cktile_e257_k7":
        bsm = 64
    elif mode == "cktile_e257_k1":
        bsm = 32
    else:
        bsm = None

    act = ActivationType.Silu
    if mode == "default" and d_expert <= 512:
        act = ActivationType.Swiglu

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=act,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        block_size_M=bsm,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )
