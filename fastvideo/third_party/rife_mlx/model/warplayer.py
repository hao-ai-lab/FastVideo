"""warp — MLX port of model/warplayer.py (backward warp via grid_sample).

Isomorphic to upstream: builds a normalized identity grid, adds the
per-pixel flow (normalized by (size-1)/2), and bilinear grid_samples the
input with border padding + align_corners=True.

NHWC throughout. flow is [N,H,W,2] with channel 0 = x (horizontal), 1 = y.
"""

from __future__ import annotations

import mlx.core as mx

from ..ops.grid_sample import grid_sample_bilinear

_grid_cache: dict[tuple, mx.array] = {}


def _identity_grid(N: int, H: int, W: int) -> mx.array:
    key = (N, H, W)
    if key not in _grid_cache:
        xs = mx.linspace(-1.0, 1.0, W).reshape(1, 1, W, 1)
        ys = mx.linspace(-1.0, 1.0, H).reshape(1, 1, H, 1)  # placeholder shape
        xs = mx.broadcast_to(xs, (N, H, W, 1))
        ys = mx.broadcast_to(mx.linspace(-1.0, 1.0, H).reshape(1, H, 1, 1), (N, H, W, 1))
        _grid_cache[key] = mx.concatenate([xs, ys], axis=-1)  # [N,H,W,2]
    return _grid_cache[key]


def warp(tenInput: mx.array, tenFlow: mx.array) -> mx.array:
    """tenInput: [N,H,W,C]; tenFlow: [N,H,W,2] (x,y pixel-space flow)."""
    N, H, W, _ = tenFlow.shape
    grid = _identity_grid(N, H, W)
    fx = tenFlow[..., 0:1] / ((W - 1.0) / 2.0)
    fy = tenFlow[..., 1:2] / ((H - 1.0) / 2.0)
    g = grid + mx.concatenate([fx, fy], axis=-1)
    return grid_sample_bilinear(tenInput, g, align_corners=True, padding_mode="border")
