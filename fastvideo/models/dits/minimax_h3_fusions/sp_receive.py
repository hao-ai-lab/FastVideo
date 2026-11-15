# SPDX-License-Identifier: Apache-2.0
"""Share SP FP8 receive reads across BF16 expansion, tile pooling and Q/V FP4 packing."""
import torch
import triton
import triton.language as tl

from fastvideo.distributed.async_collectives import PeerSlices

from .value_quant import _left_four
from .swiglu_quant import _rcp

@triton.jit
def _receive_all(Payload, Scale, Inverse, Valid, Out, Pool, Codes, Scales, QCodes, QScales, ROWS: tl.constexpr,
                 S: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, START: tl.constexpr,
                 Remote=None, PAIR: tl.constexpr = False, RANK: tl.constexpr = 0,
                 QUERY_ROWS: tl.constexpr = -1, PrefixOut=None, PrefixInverse=None,
                 PREFIX_VALUES: tl.constexpr = False, STORE_KEYS: tl.constexpr = True):
    tile = tl.program_id(1)
    projection = tl.program_id(2)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.arange(0, 16)
    head = col // D
    acc = tl.full((16, BLOCK), 0, tl.float32)
    for i in tl.static_range(4):
        token = tile * 64 + i * 16 + row
        source = tl.load(Inverse + token)
        packed = (source // ROWS * 3 + projection) * ROWS + source % ROWS
        valid = (col[None, :] < H * D) & (source[:, None] >= 0)
        if PAIR:
            local_offset = (projection * ROWS + source % ROWS)[:, None] * H * D + col[None, :]
            pointer = tl.where((source // ROWS)[:, None] == RANK, Payload + local_offset, Remote + local_offset)
        else:
            pointer = Payload + packed[:, None] * H * D + col[None, :]
        value = tl.load(pointer, valid, other=0.0).to(tl.float32)
        scale = tl.load(Scale + packed[:, None] * H + head[None, :], valid, other=0)
        x = (value * scale).to(tl.bfloat16).to(tl.float32)
        store_mask = (col[None, :] < H * D) & (STORE_KEYS | (projection != 1))
        if QUERY_ROWS >= 0:
            store_mask = store_mask & ((projection != 0) | (token[:, None] < QUERY_ROWS))
        if PREFIX_VALUES and projection == 2:
            destination = tl.load(PrefixInverse + token)
            tl.store(PrefixOut + destination[:, None] * H * D + col[None, :], x,
                     (col[None, :] < H * D) & (destination[:, None] >= 0))
        else:
            tl.store(Out + (projection * S + token[:, None]) * H * D + col[None, :], x, store_mask)
        acc = acc + x
        if projection == 2:
            maximum = tl.max(tl.abs(x), 0)
            sf = (maximum * (1.0 / 6.0)).to(tl.float8e4nv)
            inverse = tl.where(sf.to(tl.float32) == 0, 0, _rcp(sf.to(tl.float32)))
            scaled = x * inverse[None, :]
            pair = tl.trans(tl.reshape(scaled, (8, 2, BLOCK)), (0, 2, 1))
            lo, hi = tl.split(pair)
            codes = tl.inline_asm_elementwise('''{
                .reg .b8 q;
                cvt.rn.satfinite.e2m1x2.f32 q, $1, $2;
                mov.b32 $0, {q,q,q,q};
            }''',
                                              constraints='=r,f,f',
                                              args=[hi, lo],
                                              dtype=tl.uint8,
                                              is_pure=True,
                                              pack=1)
            qrow = tl.arange(0, 8)
            offset = (col[None, :] * (S // 2) + tile * 32 + i * 8 + qrow[:, None])
            tl.store(Codes + offset, codes, col[None, :] < H * D)
            group = tile * 4 + i
            dim = col % D
            offset = (head * D * (S // 16) + (dim // 64) * 64 * (S // 16) + (group // 4) * 256 + group % 4 +
                      (dim % 64 // 16) * 4 + (dim % 16) * 16)
            tl.store(Scales + offset, sf, col < H * D)
        elif projection == 0:
            groups = tl.reshape(x, (16, BLOCK // 16, 16))
            maximum = tl.max(tl.abs(groups), 2)
            sf = (maximum * (1.0 / 6.0)).to(tl.float8e4nv)
            inverse = tl.where(sf.to(tl.float32) == 0, 0, _rcp(sf.to(tl.float32)))
            quant = tl.reshape(groups * inverse[:, :, None], (16, BLOCK // 2, 2))
            lo, hi = tl.split(quant)
            packed_q = tl.inline_asm_elementwise('''{
                .reg .b8 q;
                cvt.rn.satfinite.e2m1x2.f32 q, $1, $2;
                mov.b32 $0, {q,q,q,q};
            }''',
                                                 constraints='=r,f,f',
                                                 args=[hi, lo],
                                                 dtype=tl.uint8,
                                                 is_pure=True,
                                                 pack=1)
            qrow = token - START
            qseq: tl.constexpr = S - START
            qcol = tl.program_id(0) * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
            qhead, qdim = qcol // (D // 2), qcol % (D // 2)
            tl.store(QCodes + (qhead[None, :] * qseq + qrow[:, None]) * (D // 2) + qdim[None, :], packed_q,
                     (qrow[:, None] >= 0) & (qcol[None, :] < H * D // 2))
            group = tl.program_id(0) * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
            qhead, qdim = group // (D // 16), group % (D // 16)
            offset = ((qrow[:, None] // 64) * 512 + (qdim[None, :] // 4) * 256 + qdim[None, :] % 4 +
                      (qrow[:, None] % 64 // 16) * 4 + (qrow[:, None] % 16) * 16)
            tl.store(QScales + qhead[None, :] * qseq * (D // 16) + offset, sf,
                     (qrow[:, None] >= 0) & (group[None, :] < H * D // 16))
    even, odd = tl.split(tl.reshape(tl.trans(acc), (BLOCK, 4, 2, 2)))
    y0, y2 = tl.split(even)
    y1, y3 = tl.split(odd)
    y0, y1 = _left_four(y0, BLOCK), _left_four(y1, BLOCK)
    y2, y3 = _left_four(y2, BLOCK), _left_four(y3, BLOCK)
    total = (y0 + y2) + (y1 + y3)
    count = tl.load(Valid + tile).to(tl.float32)
    mean = tl.div_rn(total, count)
    tl.store(Pool + (projection * (S // 64) + tile) * H * D + col, mean, col < H * D)


def unpack_pool_quantize_fp8(
    payload: torch.Tensor | PeerSlices, scale: torch.Tensor, inverse: torch.Tensor,
    valid: torch.Tensor, output: torch.Tensor, query_start: int, *, query_rows: int | None = None,
    prefix_values: tuple[torch.Tensor, torch.Tensor] | None = None, store_keys: bool = True,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Expand FP8 QKV and prepare exact tile means plus native Q/V FP4 inputs.

    By default every tile row is written, including zero padding. When the
    caller consumes native FP4 queries, ``query_rows`` may limit BF16 query
    stores to the dense prefix. The omitted suffix is untouched and must not
    be read; pooling and native query codes still cover every row. K/V stores
    remain complete by default. ``prefix_values=(inverse, output)`` instead
    writes BF16 values directly into dense-prefix order; the entire raw V
    tensor is then untouched and must not be read. Native V codes and pooling
    still cover every row. Pooling keeps ATen's
    tile-64 accumulation order and reads the rounded BF16 expansion. Query
    scales use Sage's native suffix layout; value scales use its transposed
    layout. With ``store_keys=False``, raw K is untouched; the caller must
    prepare keys directly from these packets. Prefix values use the
    caller's inverse map, whose entries are -1 or in-bounds output row indices.
    """
    pair = payload if isinstance(payload, PeerSlices) else None
    payload = pair.template if pair is not None else payload
    start, block = query_start, 256
    _, _, rows, heads, dim = payload.shape
    assert payload.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert payload.shape[1] == 3 and dim == 128 and output.dtype == torch.bfloat16
    assert inverse.numel() % 128 == 0 and 0 <= start < inverse.numel() and start % 128 == 0
    assert output.shape == (3, inverse.numel(), heads, dim) and scale.shape == payload.shape[:-1]
    assert valid.shape == (inverse.numel() // 64,)
    assert all(t.is_contiguous() for t in (payload, scale, inverse, valid, output))
    assert inverse.dtype in (torch.int32, torch.int64) and valid.dtype in (torch.int32, torch.int64)
    assert all(t.device == payload.device for t in (scale, inverse, valid, output))
    assert payload.is_cuda and not torch.is_grad_enabled()
    prefix_inverse, prefix_output = (None, None) if prefix_values is None else prefix_values
    if prefix_values is not None:
        if (prefix_inverse.shape != inverse.shape or prefix_inverse.dtype not in (torch.int32, torch.int64)
                or prefix_inverse.device != payload.device or not prefix_inverse.is_contiguous()
                or prefix_output.ndim != 4 or prefix_output.shape[0] != 1
                or prefix_output.shape[2:] != (heads, dim) or prefix_output.dtype != torch.bfloat16
                or prefix_output.device != payload.device or not prefix_output.is_contiguous()):
            raise ValueError("Prefix values require a tile inverse and contiguous BF16 [1, valid, heads, 128] output")
    seq = inverse.numel()
    if query_rows is not None and not 0 <= query_rows <= seq:
        raise ValueError("BF16 query rows must lie within the tile sequence")
    pool = torch.empty((3, seq // 64, heads, dim), device=payload.device, dtype=torch.float32)
    codes = torch.empty((1, heads, dim, seq // 2), device=payload.device, dtype=torch.uint8)
    scales = torch.empty((1, heads, dim, seq // 16), device=payload.device, dtype=torch.float8_e4m3fn)
    qcodes = torch.empty((1, heads, seq - start, dim // 2), device=payload.device, dtype=torch.uint8)
    qscales = torch.empty((1, heads, seq - start, dim // 16), device=payload.device, dtype=torch.float8_e4m3fn)
    _receive_all[(triton.cdiv(heads * dim, block), seq // 64, 3)](
        pair.local if pair is not None else payload, scale, inverse, valid, output, pool, codes, scales, qcodes, qscales,
        rows, seq, heads, dim, block, start, Remote=pair.remote if pair is not None else payload,
        PAIR=pair is not None, RANK=pair.rank if pair is not None else 0,
        QUERY_ROWS=-1 if query_rows is None else query_rows, PrefixOut=prefix_output, PrefixInverse=prefix_inverse,
        PREFIX_VALUES=prefix_values is not None, STORE_KEYS=store_keys, num_warps=4,
    )
    return pool.permute(0, 2, 1, 3), (codes, scales), (qcodes, qscales)
