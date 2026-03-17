#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v002: Module-level imports, dead code removal.
- Move all imports to module scope (avoid per-call import overhead)
- Remove unused B.contiguous() and B unpacking
- Remove A.contiguous() (A is already contiguous from generate_input/clone)
"""
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data

    # Quant A to MXFP4 with shuffled scales
    A_q, A_scale = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale)

    return aiter.gemm_a4w4(
        A_q.view(dtypes.fp4x2),
        B_shuffle,
        A_scale_sh.view(dtypes.fp8_e8m0),
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
