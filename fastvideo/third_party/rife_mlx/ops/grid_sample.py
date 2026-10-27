"""Bilinear grid_sample (NHWC) — MLX equivalent of torch.nn.functional.grid_sample.

RIFE's `warp` calls grid_sample(mode='bilinear', padding_mode='border',
align_corners=True). No native MLX op — hand-rolled here. This is the crux
surface of the port; parity-locked vs torch in P2 (Gate A).

Conventions (match torch):
- grid[..., 0] = x (width, normalized [-1,1]); grid[..., 1] = y (height).
- align_corners=True: -1 -> pixel 0, +1 -> pixel (size-1).
- padding_mode='border': neighbor integer indices are clamped to the valid
  range; bilinear weights come from the UNCLAMPED continuous coordinate.
"""

from __future__ import annotations

import mlx.core as mx


def grid_sample_bilinear(inp: mx.array, grid: mx.array,
                         align_corners: bool = True,
                         padding_mode: str = "border") -> mx.array:
    """inp: [N,H,W,C]; grid: [N,gH,gW,2] in [-1,1]. Returns [N,gH,gW,C]."""
    if padding_mode != "border":
        raise NotImplementedError("only padding_mode='border' is needed for RIFE")
    N, H, W, C = inp.shape
    _, gH, gW, _ = grid.shape

    gx = grid[..., 0]  # [N,gH,gW]
    gy = grid[..., 1]
    if align_corners:
        ix = (gx + 1) * 0.5 * (W - 1)
        iy = (gy + 1) * 0.5 * (H - 1)
    else:
        ix = ((gx + 1) * W - 1) * 0.5
        iy = ((gy + 1) * H - 1) * 0.5

    x0 = mx.floor(ix); y0 = mx.floor(iy)
    x1 = x0 + 1; y1 = y0 + 1
    wx1 = ix - x0; wx0 = 1.0 - wx1
    wy1 = iy - y0; wy0 = 1.0 - wy1

    def clampx(a): return mx.clip(a, 0, W - 1).astype(mx.int32)
    def clampy(a): return mx.clip(a, 0, H - 1).astype(mx.int32)
    x0c, x1c = clampx(x0), clampx(x1)
    y0c, y1c = clampy(y0), clampy(y1)

    inp_flat = inp.reshape(N, H * W, C)

    def gather(yc, xc):  # yc,xc: [N,gH,gW] int -> [N,gH,gW,C]
        idx = (yc * W + xc).reshape(N, gH * gW, 1)
        idx = mx.broadcast_to(idx, (N, gH * gW, C))
        g = mx.take_along_axis(inp_flat, idx, axis=1)
        return g.reshape(N, gH, gW, C)

    v00 = gather(y0c, x0c); v01 = gather(y0c, x1c)
    v10 = gather(y1c, x0c); v11 = gather(y1c, x1c)

    w00 = (wy0 * wx0)[..., None]; w01 = (wy0 * wx1)[..., None]
    w10 = (wy1 * wx0)[..., None]; w11 = (wy1 * wx1)[..., None]
    return v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
