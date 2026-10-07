# SPDX-License-Identifier: Apache-2.0
"""Heads-first tile scatter for VSA-H3 (``FASTVIDEO_H3_VSA_HEADS_FIRST_TILE``).

``MiniMaxH3VSAImpl.tile`` scatters packed ``[B, S, H, D]`` rows into a padded
``[B, S_pad, H, D]`` tile buffer, and the 64/128-token kernels then copy that
buffer to ``[B, H, S_pad, D]`` with ``transpose(1, 2).contiguous()`` before
every call, for query, key and value. This kernel scatters the rows straight
into a ``[B, H, S_pad, D]`` buffer instead, and ``tile`` returns its
``transpose(1, 2)`` view: the same logical tensor, so every consumer reads the
same values, and the kernels' ``.contiguous()`` becomes free. It only moves
bytes, so the result is bit-identical.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None
    tl = None

HAVE_TRITON = triton is not None
_BLOCK_S = 64

if HAVE_TRITON:

    @triton.jit
    def _scatter_rows_heads_first_kernel(src, dst, slots, seq, s_sn, s_ss, s_sh, d_sn, d_sh, d_ss, D: tl.constexpr,
                                         BLOCK_S: tl.constexpr):
        rows = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        h = tl.program_id(1).to(tl.int64)
        n = tl.program_id(2).to(tl.int64)
        keep = rows < seq
        slot = tl.load(slots + rows, mask=keep, other=0).to(tl.int64)
        cols = tl.arange(0, D)
        values = tl.load(src + n * s_sn + rows[:, None].to(tl.int64) * s_ss + h * s_sh + cols[None, :],
                         mask=keep[:, None])
        tl.store(dst + n * d_sn + h * d_sh + slot[:, None] * d_ss + cols[None, :], values, mask=keep[:, None])


def supports_heads_first_scatter(x: torch.Tensor, slots: torch.Tensor) -> bool:
    """Whether ``scatter_rows_heads_first`` handles ``x`` ``[B, S, H, D]`` and its row slots."""
    dim = x.shape[-1]
    return (HAVE_TRITON and x.is_cuda and x.dim() == 4 and x.stride(-1) == 1 and dim > 0 and dim & (dim - 1) == 0
            and slots.is_cuda and slots.dim() == 1 and slots.shape[0] == x.shape[1] and slots.is_contiguous())


def scatter_rows_heads_first(x: torch.Tensor, slots: torch.Tensor, buffer: torch.Tensor) -> None:
    """``buffer.transpose(1, 2)[:, slots] = x`` for ``x`` ``[B, S, H, D]`` and ``buffer`` ``[B, H, S_pad, D]``."""
    batch, seq, heads, dim = x.shape
    _scatter_rows_heads_first_kernel[(triton.cdiv(seq, _BLOCK_S), heads, batch)](
        x,
        buffer,
        slots,
        seq,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        buffer.stride(0),
        buffer.stride(1),
        buffer.stride(2),
        D=dim,
        BLOCK_S=_BLOCK_S,
    )


__all__ = ["HAVE_TRITON", "scatter_rows_heads_first", "supports_heads_first_scatter"]
