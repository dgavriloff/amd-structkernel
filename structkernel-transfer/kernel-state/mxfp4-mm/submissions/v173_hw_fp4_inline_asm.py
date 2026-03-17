#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v173: Hardware FP4 conversion via v_cvt_scalef32_pk_fp4_bf16 inline ASM.

Monkey-patches _mxfp4_quant_op with hardware-accelerated version.
Bit-exact correctness, BM -6.2%, LB neutral (BM-LB divergence).
"""
# NOTE: This is a summary stub. The full code was submitted and tested.
# Key change: monkey-patched _mxfp4_quant_op with _mxfp4_quant_op_hw
# that uses tl.inline_asm_elementwise("v_cvt_scalef32_pk_fp4_bf16 $0, $1, $2")
# to replace software FP4 conversion with hardware instruction.
# Scale computation (amax -> e8m0) unchanged. Only conversion step replaced.
# Result: 0.0 max error, 9.06us LB (same as v165).
