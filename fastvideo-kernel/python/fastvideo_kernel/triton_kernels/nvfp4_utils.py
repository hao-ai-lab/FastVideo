# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_upcast_from_mxfp.py
# and https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py

import triton
import triton.language as tl
from triton.language.target_info import cuda_capability_geq

MXFP_BLOCK_SIZE = tl.constexpr(16)

@triton.jit
def _compute_quant_and_scale(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr = tl.uint8,
    use_global_sf=True,
    two_level_quant_P=False,
):
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // MXFP_BLOCK_SIZE
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    tl.static_assert(
        is_fp4
        or mx_tensor_dtype == tl.float8e4nv
        or mx_tensor_dtype == tl.float8e5,
        "mx_tensor_dtype must be uint8, float8e4nv, or float8e5",
    )

    # Explicit cast to fp32 since most ops are not supported on bfloat16. We avoid needless conversions to and from bf16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)  # Don't consider padding tensors in scale computation
    
    if two_level_quant_P:
        # row max from SageAttn3 paper
        global_max_val = tl.max(f32_tensor, axis=1, keep_dims=True)  # (BLOCK_SIZE_OUT_DIM, 1)
        global_max_val = tl.maximum(global_max_val, 1e-8)
        s_enc = ((6 * 448) / global_max_val).reshape([BLOCK_SIZE_OUT_DIM, 1, 1])
        s_dec = (1 / s_enc)

    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])

    if use_global_sf and not two_level_quant_P:
        global_max_val = tl.max(abs_tensor)
        # Avoid division by zero: if all values are padding (max is 0), use a default scale
        global_max_val = tl.maximum(global_max_val, 1e-8)
        s_enc = (6 * 448) / global_max_val
        s_dec = (1 / s_enc)
    elif not two_level_quant_P and not use_global_sf:
        s_dec = 1.0
        s_enc = 1.0
    
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)  # (BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1)  # per block maxima
    s_dec_b = max_val / 6  # (BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1)
    s_dec_b_e4m3 = (s_dec_b * s_enc).to(tl.float8e4nv)  # (BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1)
    s_enc_b = 1 / (s_dec_b_e4m3.to(tl.float32) * s_dec)  # (BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1)

    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    quant_tensor = f32_tensor * s_enc_b

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0. This will ensure that any padding tensors are 0 in the mx format.
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0.0)
    dequant_scale = s_dec_b_e4m3.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])

    if is_fp4 and cuda_capability_geq(10, 0):
        # Convert scaled values to two f32 lanes and use PTX cvt to e2m1x2 with two f32 operands.
        pairs = tl.reshape(quant_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
        lo_f, hi_f = tl.split(pairs)
        lo_f32 = lo_f.to(tl.float32)
        hi_f32 = hi_f.to(tl.float32)

        # Inline PTX: cvt.rn.satfinite.e2m1x2.f32 takes two f32 sources and produces one .b8 packed e2m1x2.
        out_tensor = tl.inline_asm_elementwise(
            """
            {
                .reg .b8 r;
                cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
                mov.b32 $0, {r, r, r, r};
            }
            """,
            constraints="=r,f,f",
            args=[hi_f32, lo_f32],
            dtype=tl.uint8,
            is_pure=True,
            pack=1,
        )
    elif is_fp4:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas_orig = (quant_tensor & 0x7FFFFF)

        # For RTNE: 0.25 < x < 0.75 maps to 0.5 (denormal); exactly 0.25 maps to 0.0
        E8_BIAS = 127
        E2_BIAS = 1
        # Move implicit bit 1 at the beginning to mantissa for denormals
        is_subnormal = exponents < E8_BIAS
        adjusted_exponents = tl.core.sub(E8_BIAS, exponents + 1, sanitize_overflow=False)
        mantissas_pre = (0x400000 | (mantissas_orig >> 1))
        mantissas = tl.where(is_subnormal, mantissas_pre >> adjusted_exponents, mantissas_orig)

        # For normal numbers, we change the bias from 127 to 1, and for subnormals, we keep exponent as 0.
        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # Round to nearest, ties to even (RTNE): use guard/sticky and LSB to decide increment
        m2bits = mantissas >> 21
        lsb_keep = (m2bits >> 1) & 0x1
        guard = m2bits & 0x1
        IS_SRC_FP32: tl.constexpr = src_tensor.dtype == tl.float32
        if IS_SRC_FP32:
            bit0_dropped = (mantissas_orig & 0x1) != 0
            mask = (1 << tl.minimum(adjusted_exponents, 31)) - 1
            dropped_post = (mantissas_pre & mask) != 0
            sticky = is_subnormal & (bit0_dropped | dropped_post)
            sticky |= ((mantissas & 0x1FFFFF) != 0).to(tl.uint32)
        else:
            sticky = ((mantissas & 0x1FFFFF) != 0).to(tl.uint32)
        round_inc = guard & (sticky | lsb_keep)
        e2m1_tmp = tl.minimum((((exponents << 2) | m2bits) + round_inc) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)
    else:
        out_tensor = quant_tensor.to(mx_tensor_dtype)

    return out_tensor, dequant_scale, s_dec

@triton.jit
def _compute_dequant(
    mx_tensor,
    scale,
    s_dec,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    dst_dtype: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0, f"Block size along quantization block must be a multiple of {MXFP_BLOCK_SIZE=}")
    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor.dtype
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16 or dst_dtype == tl.float32)
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or ((mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5) or mx_tensor_dtype == dst_dtype),
        "mx_tensor_ptr must be uint8 or float8 or dst_dtype")
    tl.static_assert(scale.dtype == tl.float8e4nv, "scale must be float8e4nv")

    # Determine if we are dealing with fp8 types.
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE

    # Upcast the scale to the destination type.
    if dst_dtype == tl.bfloat16:
        dst_scale = scale.to(tl.bfloat16)
    else:
        dst_scale = scale.to(tl.float32)
        if dst_dtype == tl.float16:
            dst_scale = dst_scale.to(tl.float16)

    # Now upcast the tensor.
    intermediate_dtype: tl.constexpr = tl.bfloat16 if dst_dtype == tl.float32 else dst_dtype
    if cuda_capability_geq(10, 0):
        assert is_fp4
        packed_u32 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8 in_8;
            .reg .f16x2 out;
            cvt.u8.u32 in_8, $1;
            cvt.rn.f16x2.e2m1x2 out, in_8;
            mov.b32 $0, out;
            }
            """,
            constraints="=r,r",
            args=[mx_tensor],  # tl.uint8 passed in as a 32-bit reg with value in low 8 bits
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        lo_u16 = (packed_u32 & 0xFFFF).to(tl.uint16)
        hi_u16 = (packed_u32 >> 16).to(tl.uint16)
        lo_f16 = lo_u16.to(tl.float16, bitcast=True)
        hi_f16 = hi_u16.to(tl.float16, bitcast=True)

        if intermediate_dtype == tl.float16:
            x0, x1 = lo_f16, hi_f16
        else:
            x0 = lo_f16.to(intermediate_dtype)
            x1 = hi_f16.to(intermediate_dtype)

        dst_tensor = tl.interleave(x0, x1)

    else:
        assert is_fp4
        dst_bias: tl.constexpr = 127 if intermediate_dtype == tl.bfloat16 else 15  # exponent bias
        dst_0p5: tl.constexpr = 16128 if intermediate_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if intermediate_dtype == tl.bfloat16 else 10  # mantissa bits
        # e2m1
        em0 = mx_tensor & 0x07
        em1 = mx_tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((mx_tensor & 0x08).to(tl.uint16) << 12)
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((mx_tensor & 0x80).to(tl.uint16) << 8)
        # Three cases:
        # 1) x is normal and non-zero: Correct bias
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in the dst type
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        # 3) x is zero, do nothing
        dst_tensor = tl.interleave(x0, x1).to(intermediate_dtype, bitcast=True)

    dst_tensor = dst_tensor.to(dst_dtype)

    # Reshape for proper broadcasting: the scale was stored with a 16‐sized “inner” grouping.
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale * s_dec  # NVFP4 has the additional global scale factor
    if dst_dtype == tl.float32:
        max_fin = 3.4028234663852886e+38
    elif dst_dtype == tl.bfloat16:
        max_fin = 3.3895313892515355e+38
    else:
        tl.static_assert(dst_dtype == tl.float16)
        max_fin = 65504
    out_tensor = tl.clamp(out_tensor, min=-max_fin, max=max_fin)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    out_tensor = out_tensor.to(dst_dtype)
    return out_tensor
