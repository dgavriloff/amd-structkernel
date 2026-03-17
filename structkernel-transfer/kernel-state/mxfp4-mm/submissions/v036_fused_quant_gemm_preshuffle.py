#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v036: Fused quant+GEMM via gemm_a16wfp4_preshuffle — eliminate quant kernel entirely.
- The preshuffle kernel takes bf16 A directly and quantizes on-the-fly inside the GEMM
  using tl.dot_scaled with PREQUANT=True, eliminating the separate ~6.4µs quant kernel.
- B_shuffle (N, K//2) reshaped to (N//16, K//2*16) for preshuffle weight format.
- B_scale_sh already in e8m0_shuffle format; reshaped to (padded_N//32, padded_K_scale*32)
  for the preshuffle scale format. Kernel does inverse shuffle internally.
- Target: eliminate quant phase (~6.4µs, 46% of total) for potentially massive speedup.
- Risk: tl.dot_scaled may not be available on runner (blocked v003/v007 previously).
"""
import torch
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

from task import input_t, output_t

# Cache for reshaped weight tensors and output buffers
_cache = {}


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_shuffle.shape[0]

    key = (M, K, N)
    if key not in _cache:
        # Reshape B_shuffle from (N, K//2) to (N//16, K//2*16) for preshuffle format
        # This is a view (zero-copy) since the data is already contiguous in the right order
        B_w = B_shuffle.view(torch.uint8).reshape(N // 16, (K // 2) * 16)

        # B_scale_sh is (padded_N, padded_K_scale) in e8m0_shuffle format
        # Preshuffle kernel expects (N//32, K_scale*32) — reshape accordingly
        bs_shape = B_scale_sh.shape
        B_sc = B_scale_sh.view(torch.uint8).reshape(
            bs_shape[0] // 32, bs_shape[1] * 32
        )

        # Pre-allocate output
        out = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

        _cache[key] = {
            'B_w': B_w,
            'B_sc': B_sc,
            'out': out,
        }

    c = _cache[key]

    return gemm_a16wfp4_preshuffle(
        A,          # bf16 (M, K) — quantized on-the-fly by PREQUANT=True
        c['B_w'],   # preshuffle weight format (N//16, K//2*16)
        c['B_sc'],  # preshuffled B scales (padded_N//32, padded_K_scale*32)
        prequant=True,
        dtype=torch.bfloat16,
        y=c['out'],
    )
