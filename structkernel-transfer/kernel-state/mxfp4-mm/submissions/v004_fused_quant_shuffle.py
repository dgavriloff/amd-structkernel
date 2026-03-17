#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v004: Fused quant+shuffle via fp4_utils.dynamic_mxfp4_quant(shuffle=True).
Target: quant phase — eliminate separate e8m0_shuffle call.
Risk: fp4_utils version may be unpatched (#974/#975). Testing correctness.
"""
from aiter import dtypes
from aiter.utility.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle
import aiter

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data

    # Fused quant + shuffle in one kernel call
    A_q, A_scale_sh = dynamic_mxfp4_quant(A, shuffle=True)

    return aiter.gemm_a4w4(
        A_q.view(dtypes.fp4x2),
        B_shuffle,
        A_scale_sh.view(dtypes.fp8_e8m0),
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
