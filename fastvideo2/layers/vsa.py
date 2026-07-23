"""Video Sparse Attention (VSA) — vendored from fastvideo-main.

Provenance: hao-ai-lab/FastVideo fastvideo/attention/backends/video_sparse_attn.py
@ e3f47dc2de2d1fa0c68c5839a0a41ed25b04a953, sp=1, 64-element-tile path.

The tile/untile index math is pure torch and vendored verbatim (it defines
which tokens share a block — part of the trained FastWan artifact). The
kernel itself comes from the installed ``fastvideo_kernel`` package; when it
is missing, VSA attention FAILS CLOSED (never a silent dense fallback — a
dense forward of a VSA-distilled artifact is a different model).

Metadata is geometry+sparsity-derived and constant across denoise steps
(main rebuilds it per step, but nothing in it depends on the step index).

Import stays torch-free; torch only inside functions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

VSA_TILE_SIZE = (4, 4, 4)  # 64-element tiles -> the TK/Triton kernel path


@dataclass
class VSAMeta:
    """Per-(grid, sparsity) attention metadata. ``buf`` is the reusable tiled
    scratch (pad positions stay zero — written once, never touched)."""
    grid: tuple[int, int, int]
    sparsity: float
    tile_index: Any        # [S] long — token -> tiled position order
    non_pad_index: Any     # [S] long — tiled positions that carry real tokens
    untile_index: Any      # [S] long — fused non_pad[reverse] gather
    block_sizes: Any       # [n_tiles] long — voxels per (possibly partial) tile
    topk: int
    padded_len: int
    buf: Any = None


def build_vsa_meta(grid: tuple[int, int, int], sparsity: float, device: Any) -> "VSAMeta":
    """main's VideoSparseAttentionMetadataBuilder.build, verbatim math.
    ``grid`` is the post-patch token grid (T, H, W)."""
    import torch
    T, H, W = grid
    ts, hs, ws = VSA_TILE_SIZE
    n_t, n_h, n_w = math.ceil(T / ts), math.ceil(H / hs), math.ceil(W / ws)

    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    parts = []
    for t in range(n_t):
        for h in range(n_h):
            for w in range(n_w):
                parts.append(indices[t * ts:min(t * ts + ts, T),
                                     h * hs:min(h * hs + hs, H),
                                     w * ws:min(w * ws + ws, W)].flatten())
    tile_index = torch.cat(parts, dim=0)
    reverse = torch.argsort(tile_index)

    def _sizes(dim_len: int, tile: int, n: int) -> Any:
        sizes = torch.full((n,), tile, dtype=torch.int, device=device)
        remainder = dim_len - (n - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    block_sizes = (_sizes(T, ts, n_t)[:, None, None]
                   * _sizes(H, hs, n_h)[None, :, None]
                   * _sizes(W, ws, n_w)[None, None, :]).flatten()

    max_block = math.prod(VSA_TILE_SIZE)
    n_win = block_sizes.shape[0]
    starts = torch.arange(n_win, device=device) * max_block
    index_pad = starts[:, None] + torch.arange(max_block, device=device)[None, :]
    mask = torch.arange(max_block, device=device)[None, :] < block_sizes[:, None]
    non_pad_index = index_pad[mask]

    topk = max(1, min(math.ceil((1 - sparsity) * n_win), n_win))
    return VSAMeta(grid=grid, sparsity=float(sparsity),
                   tile_index=tile_index, non_pad_index=non_pad_index,
                   untile_index=non_pad_index[reverse],
                   block_sizes=block_sizes, topk=topk,
                   padded_len=n_t * ts * n_h * hs * n_w * ws)


def vsa_attention(qkvg: Any, meta: "VSAMeta") -> Any:
    """main's DistributedAttention_VSA core at sp=1: tile the stacked
    [4B, S, H, D] q/k/v/gate ONCE (shared zero-padded buffer), run the
    block-sparse kernel (BHSD round-trip), untile back to [B, S, H, D].
    RoPE is the CALLER's job (applied to q,k before stacking, like main)."""
    import torch
    try:
        from fastvideo_kernel import video_sparse_attn
    except ImportError as e:  # fail closed — dense output would be a different model
        raise RuntimeError(
            "VSA requires the fastvideo_kernel package (video_sparse_attn); "
            "refusing to serve a VSA-distilled artifact with dense attention"
        ) from e

    target = (qkvg.shape[0], meta.padded_len, qkvg.shape[-2], qkvg.shape[-1])
    buf = meta.buf
    if buf is None or buf.shape != target or buf.dtype != qkvg.dtype or buf.device != qkvg.device:
        buf = torch.zeros(target, device=qkvg.device, dtype=qkvg.dtype)
        meta.buf = buf
    buf[:, meta.non_pad_index] = qkvg[:, meta.tile_index]

    q, k, v, gate = buf.chunk(4, dim=0)
    out = video_sparse_attn(q.transpose(1, 2).contiguous(),
                            k.transpose(1, 2).contiguous(),
                            v.transpose(1, 2).contiguous(),
                            meta.block_sizes, meta.block_sizes, meta.topk,
                            block_size=VSA_TILE_SIZE,
                            compress_attn_weight=gate.transpose(1, 2).contiguous()
                            ).transpose(1, 2)
    return out[:, meta.untile_index]
