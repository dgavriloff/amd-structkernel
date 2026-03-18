#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
v202: Cached medium-auto / large-k1 control for the E257 branch family.

Policy:
- E=257, M<=128: CK Tile with BYPASS_TUNE_CONFIG=1 and KSPLIT=7
- E=257, M>128: CK Tile with BYPASS_TUNE_CONFIG=1, KSPLIT=1, block_size_M=32
- E=33, M<=16: CK Tile with KSPLIT=7
- E=33, 16<M<=128: CK Tile with KSPLIT=2 and block_size_M=32
- E=33, M>128: default path with block_size_M=64 and Swiglu for d<=512

This closes the last clean gap in the family by combining the medium-auto E257
path with the large-E257 ksplit=1 branch.
"""
import gc
import os

os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"

import torch
import aiter
import aiter.fused_moe as _fmoe
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe, get_2stage_cfgs
from task import input_t, output_t

gc.disable()

_sorting_cache = {}


def _cached_sorting_impl(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    expert_mask,
    num_local_tokens,
    dispatch_policy,
    use_opus,
):
    M, topk = topk_ids.shape
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    device = topk_ids.device
    key = (M, model_dim, max_num_tokens_padded, max_num_m_blocks)
    if key not in _sorting_cache:
        _sorting_cache[key] = (
            torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device),
            torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device),
            torch.empty(max_num_m_blocks, dtype=torch.int32, device=device),
            torch.empty(2, dtype=torch.int32, device=device),
            torch.empty((M, model_dim), dtype=moebuf_dtype, device=device),
        )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sorting_cache[key]
    fwd_fn = aiter.moe_sorting_opus_fwd if use_opus else aiter.moe_sorting_fwd
    fwd_fn(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        int(block_size),
        expert_mask,
        num_local_tokens,
        dispatch_policy,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


_fmoe._moe_sorting_impl = _cached_sorting_impl

_stage1_cache = {}


def _cached_cktile_stage1(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    block_m,
    a1_scale,
    w1_scale,
    sorted_weights=None,
    n_pad_zeros=0,
    k_pad_zeros=0,
    bias1=None,
    activation=ActivationType.Silu,
    split_k=1,
    dtype=torch.bfloat16,
):
    token_num = hidden_states.shape[0]
    _, n1, k1 = w1.shape
    _, k2, n2 = w2.shape
    D = n2 if k2 == k1 else n2 * 2
    if w1.dtype is torch.uint32:
        D = D * 8
    key = (token_num, topk, D, n1, split_k)
    if key not in _stage1_cache:
        device = hidden_states.device
        _stage1_cache[key] = (
            torch.empty((token_num, topk, D), dtype=dtype, device=device),
            torch.empty((token_num, topk, n1), dtype=hidden_states.dtype, device=device)
            if split_k > 1
            else None,
        )
    out_buf, tmp_buf = _stage1_cache[key]
    if split_k > 1:
        tmp_buf.zero_()
        tmp_out = tmp_buf
    else:
        tmp_out = out_buf
    aiter.moe_cktile2stages_gemm1(
        hidden_states,
        w1,
        tmp_out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        n_pad_zeros,
        k_pad_zeros,
        sorted_weights,
        a1_scale,
        w1_scale,
        bias1,
        activation,
        block_m,
        split_k,
    )
    if split_k > 1:
        if activation == ActivationType.Silu:
            aiter.silu_and_mul(out_buf, tmp_out)
        else:
            aiter.gelu_and_mul(out_buf, tmp_out)
    return out_buf


_fmoe.cktile_moe_stage1 = _cached_cktile_stage1

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
