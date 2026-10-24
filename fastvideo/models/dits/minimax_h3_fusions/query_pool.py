# SPDX-License-Identifier: Apache-2.0
"""Prepare exact pooled/FP4 queries while retaining BF16 only for dense prefix attention."""
import torch
import triton
import triton.language as tl

from .swiglu_quant import _rcp
from .tile_pool import _tile_sum


@triton.jit
def _query_pool_kernel(
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
    TILE_START: tl.constexpr = 0,
    codes_ptr=None,
    scales_ptr=None,
    PACK_FP4: tl.constexpr = False,
    START: tl.constexpr = 0,
    input_index_ptr=None,
    INPUT_ROWS: tl.constexpr = 0,
    GATHER_INPUT: tl.constexpr = False,
    pool_ptr=None,
    valid_ptr=None,
    PREFIX: tl.constexpr = 0,
    PREFIX_ONLY: tl.constexpr = False,
    PREFIX_FP8: tl.constexpr = False,
):
    # int64, like the sibling kernels: with int32 program ids,
    # ``row * head_dim`` wraps once the flattened input reaches 2**31
    # elements (H3's 56 heads x 128 head_dim crosses that at
    # batch*seq >= 299_593 tokens per rank) and the loads/stores below
    # become out-of-bounds. ``seq_index`` inherits int64 from ``row``.
    tl.static_assert(BLOCK_SIZE == 128 and head_dim == 128)
    tile = tl.program_id(0).to(tl.int64) + TILE_START
    current_head = tl.program_id(1)
    current_batch = tl.program_id(2)
    seq_index = tile * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    row = (current_batch * seq_len + seq_index) * num_heads + current_head
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
    if PREFIX_ONLY:
        output_offset = ((current_batch * PREFIX + seq_index) * num_heads + current_head) * head_dim
        if PREFIX_FP8:
            if tile * ROWS_PER_PROGRAM < PREFIX:
                prefix_value = output.to(tl.bfloat16).to(tl.float32)
                scale8 = tl.max(tl.abs(prefix_value), 1) / 448.0
                inv8 = tl.where(scale8 == 0, 0, 1.0 / scale8)
                prefix_value = tl.minimum(tl.maximum(prefix_value * inv8[:, None], -448.0), 448.0)
                prefix_value = prefix_value.to(tl.float8e4nv).to(tl.float32) * scale8[:, None]
                tl.store(out_ptr + output_offset + cols, prefix_value, mask=head_mask & (seq_index < PREFIX))
        else:
            tl.store(out_ptr + output_offset + cols, output.to(out_ptr.dtype.element_ty),
                     mask=head_mask & (seq_index < PREFIX))
    else:
        tl.store(out_ptr + row_offset + cols, output.to(out_ptr.dtype.element_ty), mask=head_mask)
    rounded_pool = output.to(tl.bfloat16).to(tl.float32)
    total = _tile_sum(rounded_pool, ROWS_PER_PROGRAM)
    count = tl.load(valid_ptr + tile).to(tl.float32)
    pool_offset = ((current_batch * (seq_len // ROWS_PER_PROGRAM) + tile) * num_heads + current_head) * 128
    tl.store(pool_ptr + pool_offset + tl.arange(0, 128), tl.div_rn(total, count))
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




def prepare_pooled_queries(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float,
    input_index: torch.Tensor | None, valid: torch.Tensor, prefix_rows: int, *, tile_elems: int = 64,
    prefix_fp8: bool = False, split_prefix: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return BF16 prefix queries, FP4 video queries, and exact FP32 tile means.

    The native suffix starts at the preceding 128-token boundary, so its FP4
    data also includes a half-prefix block when the prefix ends on tile 64.
    An absent row map reads padded projection rows directly.
    Omni tile-128 callers may also fuse the existing prefix FP8 round trip.
    Splitting the prefix and suffix launches specializes away unused packing
    and prefix stores while retaining every consumed value and tile sum.
    Pooling and native codes always use the original BF16-normalized queries.
    No BF16 video-query intermediate is needed by either attention branch.
    """
    from .qknorm_rope import _validate_inputs

    if x.ndim != 4 or (input_index is not None and input_index.ndim != 1):
        raise ValueError("Pooled queries require BSHD input and a one-dimensional row map")
    rows = x.shape[1] if input_index is None else input_index.numel()
    shape_view = x[:, :1].expand(-1, rows, -1, -1)
    batch, _, heads, dim, rotary_dim = _validate_inputs(shape_view, weight, cos, sin, eps)
    if (not x.is_cuda or x.dtype != torch.bfloat16 or dim != 128 or rows % 128
            or tile_elems not in (64, 128) or (prefix_fp8 and tile_elems != 128)
            or (input_index is not None and (input_index.device != x.device
                or input_index.dtype not in (torch.int32, torch.int64) or not input_index.is_contiguous()))):
        raise ValueError("Pooled queries require CUDA BF16 head-128 and an aligned integer row map")
    if (valid.shape != (rows // tile_elems,) or valid.device != x.device
            or valid.dtype not in (torch.int32, torch.int64) or not valid.is_contiguous()
            or prefix_rows <= 0 or prefix_rows >= rows or prefix_rows % tile_elems):
        raise ValueError("Pooled queries require tile validity and a nonempty aligned prefix")
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        raise RuntimeError("Pooled query preparation requires eager inference")
    if split_prefix and (tile_elems != 128 or not prefix_fp8):
        raise ValueError("Split query preparation requires tile-128 FP8 prefix queries")
    x, weight, cos, sin = (t.contiguous() for t in (x, weight, cos, sin))
    start = prefix_rows // 128 * 128
    prefix = torch.empty((batch, prefix_rows, heads, dim), device=x.device, dtype=x.dtype)
    codes = torch.empty((batch, heads, rows - start, 64), device=x.device, dtype=torch.uint8)
    scales = torch.empty((batch, heads, rows - start, 8), device=x.device, dtype=torch.float8_e4m3fn)
    pooled = torch.empty((batch, rows // tile_elems, heads, dim), device=x.device, dtype=torch.float32)
    launches = ((0, rows // tile_elems, True, prefix_fp8, prefix_rows),)
    if split_prefix:
        prefix_tiles = prefix_rows // tile_elems
        launches = ((0, prefix_tiles, False, True, prefix_rows),
                    (prefix_tiles, rows // tile_elems - prefix_tiles, True, False, 0))
    for first_tile, tiles, pack, fp8, keep_prefix in launches:
        _query_pool_kernel[(tiles, heads, batch)](
            prefix, x, weight, cos, sin, dim, rotary_dim, rotary_dim // 2, heads, rows, eps,
            BLOCK_SIZE=128, ROWS=batch * rows * heads, ROWS_PER_PROGRAM=tile_elems, TILE_START=first_tile,
            codes_ptr=codes, scales_ptr=scales, PACK_FP4=pack, START=start,
            input_index_ptr=input_index, INPUT_ROWS=x.shape[1], GATHER_INPUT=input_index is not None,
            pool_ptr=pooled, valid_ptr=valid, PREFIX=keep_prefix, PREFIX_ONLY=True, PREFIX_FP8=fp8, num_warps=8,
        )
    return prefix, (codes, scales), pooled.permute(0, 2, 1, 3)
