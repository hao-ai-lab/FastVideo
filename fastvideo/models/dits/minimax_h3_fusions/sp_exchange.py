# SPDX-License-Identifier: Apache-2.0
"""Dequantize an H3 Ulysses exchange directly into the VSA tile layout."""

import torch
import triton
import triton.language as tl

from fastvideo.distributed.async_collectives import PeerSlices

from .swiglu_quant import _rcp as rcp


@triton.jit
def _unpack_fp8_tiles(Payload, Scale, Inverse, Out, LOCAL_ROWS: tl.constexpr,
                      TILE_ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr,
                      Codes=None, QueryScales=None, PACK_QUERY: tl.constexpr = False, START: tl.constexpr = 0):
    tile_row = tl.program_id(0).to(tl.int64)
    projection = tl.program_id(1)
    column = tl.arange(0, BLOCK)
    packed_row = tl.load(Inverse + tile_row).to(tl.int64)
    source = packed_row // LOCAL_ROWS
    local_row = packed_row % LOCAL_ROWS
    source_row = (source * 3 + projection) * LOCAL_ROWS + local_row
    valid = (packed_row >= 0) & (column < HEADS * DIM)
    value = tl.load(Payload + source_row * HEADS * DIM + column, valid, other=0.0).to(tl.float32)
    scale = tl.load(Scale + source_row * HEADS + column // DIM, valid, other=0)
    decoded = (value * scale).to(tl.bfloat16)
    destination = (projection * TILE_ROWS + tile_row) * HEADS * DIM + column
    tl.store(Out + destination, decoded, column < HEADS * DIM)
    if PACK_QUERY and projection == 0 and tile_row >= START:
        grouped = tl.reshape(decoded.to(tl.float32), (BLOCK // 16, 16))
        maximum = tl.max(tl.abs(grouped), 1)
        sf = (maximum * (1.0 / 6.0)).to(tl.float8e4nv)
        inverse = tl.where(sf.to(tl.float32) == 0, 0, rcp(sf.to(tl.float32)))
        values = tl.reshape(grouped * inverse[:, None], (BLOCK // 2, 2))
        lo, hi = tl.split(values)
        packed = tl.inline_asm_elementwise('''{
            .reg .b8 q;
            cvt.rn.satfinite.e2m1x2.f32 q, $1, $2;
            mov.b32 $0, {q,q,q,q};
        }''', constraints="=r,f,f", args=[hi, lo], dtype=tl.uint8, is_pure=True, pack=1)
        qrow = tile_row - START
        qrows: tl.constexpr = TILE_ROWS - START
        code_col = tl.arange(0, BLOCK // 2)
        head, local_code = code_col // (DIM // 2), code_col % (DIM // 2)
        tl.store(Codes + (head * qrows + qrow) * (DIM // 2) + local_code,
                  packed, code_col < HEADS * DIM // 2)
        group = tl.arange(0, BLOCK // 16)
        head, local_group = group // (DIM // 16), group % (DIM // 16)
        swizzled = ((qrow // 64) * (64 * DIM // 16) + (local_group // 4) * 256 + local_group % 4
                    + (qrow % 64 // 16) * 4 + (qrow % 16) * 16)
        tl.store(QueryScales + head * qrows * (DIM // 16) + swizzled, sf, group < HEADS * DIM // 16)


def unpack_fp8_tiles(payload: torch.Tensor, scale: torch.Tensor, inverse: torch.Tensor,
                     output: torch.Tensor, *, query_start: int | None = None
                     ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Map `[W, 3, local_rows, Hs, D]` FP8 receives to `[3, tile_rows, Hs, D]` BF16.

    `inverse[tile_row]` is the global packed row, or -1 for VSA padding. Every
    output element is written, including padding, so a reused buffer cannot
    retain stale values. The float32 multiply and BF16 rounding match the
    separate unpack and index-copy operations. An optional aligned query suffix
    also receives native Sage FP4 codes and scales after that BF16 rounding.
    """
    _, projections, local_rows, heads, dim = payload.shape
    assert projections == 3 and payload.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32 and output.dtype == torch.bfloat16
    assert payload.is_contiguous() and scale.is_contiguous() and output.is_contiguous()
    assert output.shape == (3, inverse.numel(), heads, dim)
    if query_start is not None:
        if dim != 128 or inverse.numel() % 128 or query_start % 128 or not 0 <= query_start < inverse.numel():
            raise ValueError("Fused query packing requires head-128 and an aligned token suffix")
        query_rows = inverse.numel() - query_start
        codes = torch.empty((1, heads, query_rows, dim // 2), device=payload.device, dtype=torch.uint8)
        query_scales = torch.empty((1, heads, query_rows, dim // 16), device=payload.device, dtype=torch.float8_e4m3fn)
    else:
        codes, query_scales = output, output
    block = triton.next_power_of_2(heads * dim)
    _unpack_fp8_tiles[(inverse.numel(), 3)](
        payload, scale, inverse, output, local_rows, inverse.numel(), heads, dim, block,
        Codes=codes, QueryScales=query_scales, PACK_QUERY=query_start is not None,
        START=query_start if query_start is not None else 0, num_warps=8 if block >= 2048 else 4,
    )
    return output if query_start is None else (output, (codes, query_scales))


@triton.jit
def _unpack_fp8_gate(Payload, Scale, Compressed, Gate, RowTile, Out, Codes, Scales, GlobalScale,
                     ROWS: tl.constexpr, HEADS: tl.constexpr, WORLD: tl.constexpr,
                     TILES: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr, QUANTIZED: tl.constexpr, CHUNK: tl.constexpr, CHUNKS: tl.constexpr,
                     Remote=None, PAIR: tl.constexpr = False, RANK: tl.constexpr = 0):
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    head, dim = column // DIM, column % DIM
    source, local_head = head // HEADS, head % HEADS
    source_head = (source * ROWS + row) * HEADS + local_head
    valid = column < WORLD * HEADS * DIM
    if PAIR:
        local_offset = (row * HEADS + local_head) * DIM + dim
        pointer = tl.where(source == RANK, Payload + local_offset, Remote + local_offset)
    else:
        pointer = Payload + source_head * DIM + dim
    value = tl.load(pointer, valid, other=0.0).to(tl.float32)
    scale = tl.load(Scale + source_head, valid, other=0.0)
    # Retain the materialized unpack's BF16 rounding before applying the gate.
    decoded = (value * scale).to(tl.bfloat16).to(tl.float32)
    tile = tl.load(RowTile + row).to(tl.int64)
    compressed = tl.load(Compressed + (head * TILES + tile) * DIM + dim, valid, other=0.0).to(tl.float32)
    full_head = source * HEADS * CHUNKS + CHUNK * HEADS + local_head
    destination = row * WORLD * HEADS * CHUNKS * DIM + full_head * DIM + dim
    gate = tl.load(Gate + destination, valid, other=0.0).to(tl.float32)
    if QUANTIZED:
        width: tl.constexpr = WORLD * HEADS * CHUNKS * DIM
        rounded = tl.where(valid, tl.fma(compressed, gate, decoded).to(tl.bfloat16).to(tl.float32), 0.0)
        grouped = tl.reshape(rounded, (BLOCK // 16, 16))
        maximum = tl.max(tl.abs(grouped), 1)
        global_scale = tl.load(GlobalScale)
        fp4_scale = (global_scale * (maximum * rcp(6.0))).to(tl.float8e4nv)
        inverse = tl.where(maximum != 0, rcp(fp4_scale.to(tl.float32) * rcp(global_scale)), 0.0)
        quantized = tl.reshape(grouped * inverse[:, None], (BLOCK // 2, 2))
        lo, hi = tl.split(quantized)
        packed = tl.inline_asm_elementwise('''{
            .reg .b8 q;
            cvt.rn.satfinite.e2m1x2.f32 q, $1, $2;
            mov.b32 $0, {q,q,q,q};
        }''', constraints="=r,f,f", args=[hi, lo], dtype=tl.uint8, is_pure=True, pack=1)
        code_column = tl.program_id(1) * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
        code_head = code_column // (DIM // 2)
        full_code_head = (code_head // HEADS) * HEADS * CHUNKS + CHUNK * HEADS + code_head % HEADS
        full_code_column = full_code_head * (DIM // 2) + code_column % (DIM // 2)
        tl.store(Codes + row * (width // 2) + full_code_column, packed, code_column < WORLD * HEADS * DIM // 2)
        group = tl.program_id(1) * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
        group_head = group // (DIM // 16)
        full_group_head = (group_head // HEADS) * HEADS * CHUNKS + CHUNK * HEADS + group_head % HEADS
        full_group = full_group_head * (DIM // 16) + group % (DIM // 16)
        scale_tile = (row // 128) * (width // 64) + full_group // 4
        scale_index = ((scale_tile * 32 + row % 32) * 4 + (row // 32) % 4) * 4 + full_group % 4
        tl.store(Scales + scale_index, fp4_scale.to(tl.uint8, bitcast=True), group < WORLD * HEADS * DIM // 16)
    else:
        tl.store(Out + destination, tl.fma(compressed, gate, decoded), valid)


def unpack_fp8_gate(payload: torch.Tensor | PeerSlices, scale: torch.Tensor, compressed: torch.Tensor,
                     gate: torch.Tensor, row_tile: torch.Tensor, global_scale: torch.Tensor | None = None,
                     *, head_chunk: int = 0, head_chunks: int = 1, output: torch.Tensor | None = None,
                     fp4_output: tuple[torch.Tensor, torch.Tensor] | None = None,
                     ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack head groups and apply the local gate without a BF16 intermediate.

    Receives are `[W, rows, Hs, D]`; the gathered compression branch is
    `[W, Hs, tiles, D]`, and the local gate/result are `[rows, W*Hs, D]`.
    A chunked exchange writes only its corresponding heads in a shared full
    output, preserving rank-major then chunk-major head order without copies.
    """
    pair = payload if isinstance(payload, PeerSlices) else None
    payload = pair.template if pair is not None else payload
    world, rows, heads, dim = payload.shape
    assert payload.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert compressed.dtype == gate.dtype == torch.bfloat16
    assert all(t.is_contiguous() for t in (payload, scale, compressed, gate, row_tile))
    assert head_chunks > 0 and 0 <= head_chunk < head_chunks
    assert gate.shape == (rows, world * heads * head_chunks, dim)
    assert compressed.shape[:2] == (world, heads) and compressed.shape[-1] == dim
    quantized = global_scale is not None
    if quantized:
        if output is not None or (head_chunks > 1 and fp4_output is None):
            raise ValueError("Chunked FP4 preparation requires shared codes/scales and no BF16 output")
        width = world * heads * head_chunks * dim
        if width % 64 or dim % 16:
            raise ValueError("Fused SP gate quantization requires a width divisible by 64")
        if global_scale.numel() != 1 or global_scale.dtype != torch.float32 or global_scale.device != gate.device:
            raise ValueError("Fused SP gate quantization requires a scalar FP32 scale on the input device")
        if fp4_output is None:
            codes = torch.empty((rows, width // 2), device=gate.device, dtype=torch.uint8)
            fp4_scales = torch.zeros((triton.cdiv(rows, 128) * 128, width // 16), device=gate.device, dtype=torch.uint8)
        else:
            codes, fp4_scales = fp4_output
            assert codes.shape == (rows, width // 2)
            assert fp4_scales.shape == (triton.cdiv(rows, 128) * 128, width // 16)
            assert all(t.dtype == torch.uint8 and t.device == gate.device and t.is_contiguous()
                       for t in fp4_output)
        output = gate  # Unused pointer when the kernel writes FP4 codes directly.
    else:
        if output is None:
            output = torch.empty_like(gate)
        assert output.shape == gate.shape and output.dtype == gate.dtype and output.is_contiguous()
        codes, fp4_scales, global_scale = output, output, output
    _unpack_fp8_gate[(rows, triton.cdiv(world * heads * dim, 512))](
        pair.local if pair is not None else payload, scale, compressed, gate, row_tile, output, codes, fp4_scales, global_scale,
        rows, heads, world, compressed.shape[2], dim, 512, quantized, head_chunk, head_chunks,
        Remote=pair.remote if pair is not None else payload, PAIR=pair is not None, RANK=pair.rank if pair is not None else 0,
    )
    return (codes, fp4_scales, global_scale) if quantized else output
