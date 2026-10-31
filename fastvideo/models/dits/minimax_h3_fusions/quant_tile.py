# SPDX-License-Identifier: Apache-2.0
"""Reuse one packed-input quantization for tile-order QKV and packed-order gate."""
import torch
import triton
import triton.language as tl


@triton.jit
def _tile_quantized(Q, S, Index, OQ, OS, WIDTH: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    src = tl.load(Index + row).to(tl.int64)
    val = tl.load(Q + src * (WIDTH // 2) + col, (src >= 0) & (col < WIDTH // 2), other=0)
    tl.store(OQ + row * (WIDTH // 2) + col, val, col < WIDTH // 2)
    group = tl.program_id(1) * (BLOCK // 8) + tl.arange(0, BLOCK // 8)
    src_scale = ((((src // 128) * (WIDTH // 64) + group // 4) * 32 + src % 32) * 4 + (src // 32) % 4) * 4 + group % 4
    dst_scale = ((((row // 128) * (WIDTH // 64) + group // 4) * 32 + row % 32) * 4 + (row // 32) % 4) * 4 + group % 4
    scale = tl.load(S + src_scale, (src >= 0) & (group < WIDTH // 16), other=0)
    tl.store(OS + dst_scale, scale, group < WIDTH // 16)


def tile_quantized(q: torch.Tensor, s: torch.Tensor, inverse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather NVFP4 rows, including 128x4-swizzled scales; -1 indices produce zero padding."""
    rows, width = inverse.numel(), q.shape[-1] * 2
    oq = torch.empty((rows, width // 2), device=q.device, dtype=q.dtype)
    os = torch.empty((rows, width // 16), device=s.device, dtype=s.dtype)
    _tile_quantized[(rows, triton.cdiv(width // 2, 512))](q, s, inverse, oq, os, width, rows, 512)
    return oq, os

