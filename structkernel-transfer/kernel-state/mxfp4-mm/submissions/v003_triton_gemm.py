#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v003: Try Triton GEMM (gemm_afp4wfp4_preshuffle) instead of CK/ASM gemm_a4w4.
Hypothesis: CK/ASM GEMM hits untuned fallback for most shapes.
Triton GEMM may be faster with autotuning.
"""
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data

    # Quant A to MXFP4 with shuffled scales
    A_q, A_scale = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale)

    return gemm_afp4wfp4_preshuffle(
        A_q.view(dtypes.fp4x2),
        B_shuffle,
        A_scale_sh.view(dtypes.fp8_e8m0),
        B_scale_sh,
        dtype=dtypes.bf16,
    )
