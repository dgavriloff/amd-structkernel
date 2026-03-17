#!POPCORN leaderboard 3_moe_mxfp4
#!POPCORN gpu MI355X

"""
v002: Use Swiglu activation instead of Silu to leverage cktile path.
This changes the quantization path:
- Silu: activations quantized to fp4x2 (two quant steps: input + inter-stage)
- Swiglu: activations stay bf16 (zero quant steps, uses cktile bf16-activation GEMM)
Both compute the same math: SiLU(gate) * up, but Swiglu path avoids activation quantization overhead.
On gfx950 (MI355X), Swiglu with per_1x32 uses cktile_moe_stage1/stage2 kernels.
"""
import torch
from typing import Dict
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe


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

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    output = fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Swiglu,
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
