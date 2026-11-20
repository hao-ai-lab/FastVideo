# SPDX-License-Identifier: Apache-2.0
"""Vectorized BF16 head-128 normalization with the original four-warp sum order.

The scalar-row kernel reduces within each 32-lane warp first, then combines
its four warp sums. A plain vectorized ``tl.sum`` chooses another tree and
changes BF16 rounding at a few inputs. Explicit butterfly permutations keep
that original order while processing multiple rows in registers together.
"""
import triton
import triton.language as tl

from .swiglu_quant import _rcp


@triton.jit
def qknorm_rope_vector_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    codes_ptr=None,
    scales_ptr=None,
    PACK_FP4: tl.constexpr = False,
    START: tl.constexpr = 0,
    input_index_ptr=None,
    INPUT_ROWS: tl.constexpr = 0,
    GATHER_INPUT: tl.constexpr = False,
):
    # int64, like the sibling kernels: with int32 program ids,
    # ``row * head_dim`` wraps once the flattened input reaches 2**31
    # elements (H3's 56 heads x 128 head_dim crosses that at
    # batch*seq >= 299_593 tokens per rank) and the loads/stores below
    # become out-of-bounds. ``seq_index`` inherits int64 from ``row``.
    tl.static_assert(BLOCK_SIZE == 128 and head_dim == 128)
    row = tl.program_id(0).to(tl.int64) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    seq_index = (row // num_heads) % seq_len
    cols = tl.arange(0, BLOCK_SIZE)[None, :]
    head_mask = (cols < head_dim) & (row < ROWS)
    row_offset = row * head_dim
    if GATHER_INPUT:
        source = tl.load(input_index_ptr + seq_index, row < ROWS, other=-1).to(tl.int64)
        input_offset = (((row // (seq_len * num_heads)) * INPUT_ROWS + source) * num_heads
                        + row % num_heads) * head_dim
        input_valid = source >= 0
    else:
        input_offset, input_valid = row_offset, True
    x = tl.load(x_ptr + input_offset + cols, mask=head_mask & input_valid, other=0.0).to(tl.float32)
    # Warp-local offsets 16..1, then the four-warp tree at offsets 64, 32.
    squared = x * x
    for shift in tl.static_range(5):
        other = tl.gather(squared, tl.broadcast_to(cols ^ (16 >> shift), (ROWS_PER_PROGRAM, BLOCK_SIZE)), axis=1)
        squared = squared + other
    for shift in tl.static_range(2):
        other = tl.gather(squared, tl.broadcast_to(cols ^ (64 >> shift), (ROWS_PER_PROGRAM, BLOCK_SIZE)), axis=1)
        squared = squared + other
    variance = tl.gather(squared, tl.full((ROWS_PER_PROGRAM, 1), 0, tl.int32), axis=1) / head_dim
    inv_rms = tl.math.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=head_mask, other=0.0).to(tl.float32)
    normalized = x * inv_rms * weight

    rotary_mask = (cols < rotary_dim) & (row < ROWS)
    first_half = cols < half_rotary_dim
    partner_col = tl.where(first_half, cols + half_rotary_dim, cols - half_rotary_dim)
    # Reuse the identical normalized partner instead of loading and normalizing it again.
    partner_normalized = tl.gather(
        normalized, tl.broadcast_to(partner_col, (ROWS_PER_PROGRAM, BLOCK_SIZE)), axis=1,
    )
    rotated = tl.where(first_half, -partner_normalized, partner_normalized)

    table_offset = seq_index * rotary_dim + cols
    cos = tl.load(cos_ptr + table_offset, mask=rotary_mask, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptr + table_offset, mask=rotary_mask, other=0.0).to(tl.float32)
    rotary_output = normalized * cos + rotated * sin
    output = tl.where(rotary_mask, rotary_output, normalized)
    tl.store(out_ptr + row_offset + cols, output.to(out_ptr.dtype.element_ty), mask=head_mask)
    if PACK_FP4:
        rounded = output.to(tl.bfloat16).to(tl.float32)
        grouped = tl.reshape(rounded, (ROWS_PER_PROGRAM, BLOCK_SIZE // 16, 16))
        maximum = tl.max(tl.abs(grouped), 2)
        sf = (maximum * (1.0 / 6.0)).to(tl.float8e4nv)
        inv = tl.where(sf.to(tl.float32) == 0, 0, _rcp(sf.to(tl.float32)))
        values = tl.reshape(grouped * inv[:, :, None], (ROWS_PER_PROGRAM, BLOCK_SIZE // 2, 2))
        lo, hi = tl.split(values)
        packed = tl.inline_asm_elementwise('''{
            .reg .b8 q;
            cvt.rn.satfinite.e2m1x2.f32 q, $1, $2;
            mov.b32 $0, {q,q,q,q};
        }''', constraints='=r,f,f', args=[hi, lo], dtype=tl.uint8, is_pure=True, pack=1)
        batch = row // (seq_len * num_heads)
        head = row % num_heads
        c4 = tl.arange(0, BLOCK_SIZE // 2)[None, :]
        offset = ((batch * num_heads + head) * (seq_len - START) + seq_index - START) * (head_dim // 2)
        tl.store(codes_ptr + offset + c4, packed, (row < ROWS) & (seq_index >= START) & (c4 < head_dim // 2))
        group = tl.arange(0, BLOCK_SIZE // 16)[None, :]
        qi = seq_index - START
        si = ((qi // 64) * (64 * head_dim // 16) + (group // 4) * 256 + group % 4
              + (qi % 64 // 16) * 4 + (qi % 16) * 16)
        scale_offset = (batch * num_heads + head) * (seq_len - START) * (head_dim // 16) + si
        tl.store(scales_ptr + scale_offset, sf,
                 (row < ROWS) & (seq_index >= START) & (group < head_dim // 16))


