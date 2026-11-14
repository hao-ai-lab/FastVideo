# SPDX-License-Identifier: Apache-2.0
"""Pack SP attention's split prefix/video results without a concatenation buffer."""
import torch
import triton
import triton.language as tl

@triton.jit
def _pack_split(Prefix, Video, Untile, Payload, Scale, LENGTH: tl.constexpr, ROWS: tl.constexpr, H: tl.constexpr,
                D: tl.constexpr, PREFIX: tl.constexpr, VIDEO: tl.constexpr, START: tl.constexpr, BLOCK_H: tl.constexpr,
                BLOCK_D: tl.constexpr, PSH: tl.constexpr, PSS: tl.constexpr, VSH: tl.constexpr, VSS: tl.constexpr):
    row_head = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    column = tl.arange(0, BLOCK_D)
    row, head = row_head // H, row_head % H
    tile_row = tl.load(Untile + row, row < LENGTH, other=-1)
    valid = (row_head[:, None] < ROWS * H) & (tile_row[:, None] >= 0) & (column[None, :] < D)
    is_prefix = tile_row < PREFIX
    prefix = tl.load(Prefix + head[:, None] * PSH + tile_row[:, None] * PSS + column[None, :],
                     valid & is_prefix[:, None],
                     other=0).to(tl.float32)
    video = tl.load(Video + head[:, None] * VSH + (tile_row[:, None] - START) * VSS + column[None, :],
                    valid & ~is_prefix[:, None],
                    other=0).to(tl.float32)
    value = tl.where(is_prefix[:, None], prefix, video)
    scale = tl.maximum(tl.max(tl.abs(value), 1) / 448.0, 1e-12)
    quant = (value / scale[:, None]).to(tl.float8e4nv)
    tl.store(Payload + row_head[:, None] * D + column[None, :], quant,
             (row_head[:, None] < ROWS * H) & (column[None, :] < D))
    tl.store(Scale + row_head, scale, row_head < ROWS * H)


def pack_split_fp8(parts, untile, world, rows, block=4) -> tuple[torch.Tensor, torch.Tensor]:
    prefix, video, start = parts
    _, heads, _, dim = video.shape
    assert prefix.dtype == video.dtype == torch.bfloat16 and prefix.shape[0] == video.shape[0] == 1
    assert prefix.shape[1] == heads and prefix.shape[3] == dim and prefix.stride(-1) == video.stride(-1) == 1
    assert untile.is_contiguous() and untile.dtype in (torch.int32, torch.int64)
    assert prefix.device == video.device == untile.device and untile.numel() <= world * rows
    assert 0 <= start <= prefix.shape[2] and video.is_cuda
    payload = torch.empty((world, rows, heads, dim), device=video.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(payload.shape[:-1], device=video.device, dtype=torch.float32)
    _pack_split[(triton.cdiv(world * rows * heads, block), )](prefix,
                                                              video,
                                                              untile,
                                                              payload,
                                                              scale,
                                                              untile.numel(),
                                                              world * rows,
                                                              heads,
                                                              dim,
                                                              prefix.shape[2],
                                                              video.shape[2],
                                                              start,
                                                              block,
                                                              triton.next_power_of_2(dim),
                                                              prefix.stride(1),
                                                              prefix.stride(2),
                                                              video.stride(1),
                                                              video.stride(2),
                                                              num_warps=4)
    return payload, scale


