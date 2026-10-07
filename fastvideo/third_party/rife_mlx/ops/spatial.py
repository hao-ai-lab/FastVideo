"""NHWC pixel-shuffle — verified (C,r,r) ordering matching torch.nn.PixelShuffle.

Reused from the realesrgan-mlx port (mlx-porting pitfall #7). RIFE's IFBlock
lastconv ends in nn.PixelShuffle(2).
"""

from __future__ import annotations

import mlx.core as mx


def pixel_shuffle_nhwc(x: mx.array, r: int) -> mx.array:
    """[N,H,W,C*r^2] -> [N,H*r,W*r,C], channel split (C,r,r) (torch-matching)."""
    n, h, w, c_in = x.shape
    c = c_in // (r * r)
    x = x.reshape((n, h, w, c, r, r))
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))  # (N,H,r_i,W,r_j,C)
    return x.reshape((n, h * r, w * r, c))
