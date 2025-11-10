# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_upcast_from_mxfp.py
# and https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py

import triton
import triton.language as tl
from triton_kernels.target_info import cuda_capability_geq

MXFP_BLOCK_SIZE = tl.constexpr(32)


@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")

@triton.jit
def _get_max_power_of_2_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 4.0
    elif dtype == tl.float8e5:
        return 32768.0
    elif dtype == tl.float8e4nv:
        return 256.0

@triton.jit
def _compute_quant_and_scale(src_tensor, valid_src_mask, mx_tensor_dtype: tl.constexpr,
                             DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0):
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // MXFP_BLOCK_SIZE

    # Explicit cast to fp32 since most ops are not supported on bfloat16. We avoid needless conversions to and from bf16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # DequantScaleRoundingMode.ROUND_UP
        # compute 2 ** ceil(log2(dequant_scale))
        # Adding 0x007FFFFF adds exponent by 1 unless mantissa is all zeros
        # A corner case: exponent is 0xFF that will overflow but that's already
        # NaN so assume we don't care.
        dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
        dequant_scale_exponent = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    else:
        # DequantScaleRoundingMode.ROUND_DOWN
        # compute 2 ** floor(log2(dequant_scale))
        assert DEQUANT_SCALE_ROUNDING_MODE == 1
        dequant_scale = max_val / _get_max_power_of_2_quant_val(mx_tensor_dtype)
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0. This will ensure that any padding tensors are 0 in the mx format.
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])

    # First, we simply extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    # Now we must convert the tensors to the mx format.
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    elif cuda_capability_geq(10, 0):
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
    else:
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

    return out_tensor, dequant_scale_exponent

@triton.jit
def _downcast_to_mxfp(
    mx_tensor_ptr, stride_mxt_outer, stride_mxt_quant: tl.constexpr,
    mx_scale_ptr, stride_mx_scale_outer, stride_mx_scale_quant,
    src_ptr, stride_src_outer, stride_src_quant, outer_dim, quant_dim,
    BLOCK_SIZE_OUT_DIM:tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr,
):

    tl.static_assert(stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1.")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0, f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32")

    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(mx_tensor_dtype == tl.uint8 or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
                     f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.")

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, f"{mx_scale_ptr.dtype.element_ty=} must be uint8")
    tl.static_assert((src_dtype == tl.bfloat16) or (src_dtype == tl.float16) or (src_dtype == tl.float32), f"{src_dtype=} must be bfloat16 or float16 or float32")
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < quant_dim // K_DIVISOR  # requires quant_dim % K_DIVISOR == 0
    full_mask_mxt = mask_mxt_quant & mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < quant_dim // MXFP_BLOCK_SIZE  # requires quant_dim % MXFP_BLOCK_SIZE == 0
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    mx_scale_offsets = offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    mx_tensor_offsets = offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_quant_and_scale(src_tensor, full_mask_src, mx_tensor_dtype,
                                                        DEQUANT_SCALE_ROUNDING_MODE)

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)

@triton.jit
def _upcast_from_mxfp(
    out_desc,
    mx_tensor_desc,
    mx_scale_ptr,
    stride_scale_outer,
    stride_scale_quant,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
):

    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0, f"Block size along quantization block must be a multiple of {MXFP_BLOCK_SIZE=}")
    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_desc.dtype
    dst_dtype: tl.constexpr = out_desc.dtype
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16 or dst_dtype == tl.float32)
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or ((mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5) or mx_tensor_dtype == dst_dtype),
        "mx_tensor_ptr must be uint8 or float8 or dst_dtype")
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")

    # Determine if we are dealing with fp8 types.
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    # Compute starting indices for the quantized (packed) dimension and the outer dimension.
    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    # Load the quantized value tensor.
    tensor = mx_tensor_desc.load([start_out.to(tl.int32), start_mxt_quant.to(tl.int32)])

    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    mask_outer = start_out + offs_outer < outer_dim
    mask_scale = start_mx_scale_quant + offs_scale < tl.cdiv(quant_dim, MXFP_BLOCK_SIZE)
    full_scale_mask = mask_scale & mask_outer
    scale_offsets = offs_scale * stride_scale_quant + offs_outer * stride_scale_outer
    scale_ptr_base = mx_scale_ptr + start_out * stride_scale_outer + start_mx_scale_quant * stride_scale_quant
    scale = tl.load(scale_ptr_base + scale_offsets, mask=full_scale_mask)

    # Upcast the scale to the destination type.
    if dst_dtype == tl.bfloat16:
        dst_scale = (scale.to(tl.uint16) << 7).to(dst_dtype, bitcast=True)
    else:
        dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        if dst_dtype == tl.float16:
            dst_scale = dst_scale.to(tl.float16)

    # Now upcast the tensor.
    intermediate_dtype: tl.constexpr = tl.bfloat16 if dst_dtype == tl.float32 else dst_dtype
    if is_fp8:
        dst_tensor = tensor.to(intermediate_dtype)
        if tensor.dtype == tl.float8e5:
            from_e_bits: tl.constexpr = 5
            from_m_bits: tl.constexpr = 2
            to_e_bits: tl.constexpr = 8 if intermediate_dtype == tl.bfloat16 else 5
            to_m_bits: tl.constexpr = 7 if intermediate_dtype == tl.bfloat16 else 10

            # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
            non_finite_mask_src: tl.constexpr = ((1 << from_e_bits) - 1) << from_m_bits
            non_finite_mask_dst: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
            dst_tensor = tl.where(
                (tensor.to(tl.uint8, bitcast=True) & non_finite_mask_src) == non_finite_mask_src,
                (dst_tensor.to(tl.uint16, bitcast=True) | non_finite_mask_dst).to(intermediate_dtype, bitcast=True),
                dst_tensor,
            )

    elif cuda_capability_geq(10, 0):
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
            args=[tensor],  # tl.uint8 passed in as a 32-bit reg with value in low 8 bits
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
        dst_bias: tl.constexpr = 127 if intermediate_dtype == tl.bfloat16 else 15
        dst_0p5: tl.constexpr = 16128 if intermediate_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if intermediate_dtype == tl.bfloat16 else 10
        # e2m1
        em0 = tensor & 0x07
        em1 = tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((tensor & 0x08).to(tl.uint16) << 12)
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((tensor & 0x80).to(tl.uint16) << 8)
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

    # Reshape for proper broadcasting: the scale was stored with a 32‐sized “inner” grouping.
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    if dst_dtype == tl.float32:
        max_fin = 3.4028234663852886e+38
    elif dst_dtype == tl.bfloat16:
        max_fin = 3.3895313892515355e+38
    else:
        tl.static_assert(dst_dtype == tl.float16)
        max_fin = 65504
    # TODO: handle infinity same as upcast_from_mxfp_torch together with the
    # above FIXME
    out_tensor = tl.clamp(out_tensor, min=-max_fin, max=max_fin)
    # Correct any NaNs encoded via the scale.
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    out_desc.store([start_out.to(tl.int32), start_out_quant.to(tl.int32)], out_tensor)