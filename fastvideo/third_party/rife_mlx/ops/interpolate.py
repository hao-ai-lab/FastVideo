"""Bilinear resize (NHWC) — MLX equivalent of F.interpolate(mode='bilinear').

RIFE uses bilinear resize for the coarse-to-fine scale pyramid inside IFNet and
for the `--scale` knob (0.25..4.0). align_corners is pinned from the bundled
IFNet_HDv3.py in P3 (RIFE 4.x typically uses align_corners=False). Parity-locked
in P2 alongside grid_sample.
"""

from __future__ import annotations

import mlx.core as mx


def _sample_coords(out: int, in_: int, align_corners: bool) -> mx.array:
    dst = mx.arange(out, dtype=mx.float32)
    if align_corners:
        scale = (in_ - 1) / (out - 1) if out > 1 else 0.0
        return dst * scale
    scale = in_ / out
    return (dst + 0.5) * scale - 0.5


def _bilinear_1d(x: mx.array, axis: int, out: int, align_corners: bool) -> mx.array:
    """Resample one spatial axis with bilinear weights + border clamp."""
    in_ = x.shape[axis]
    if in_ == out:
        return x
    src = _sample_coords(out, in_, align_corners)       # [out]
    i0 = mx.floor(src); i1 = i0 + 1
    w1 = src - i0; w0 = 1.0 - w1
    i0 = mx.clip(i0, 0, in_ - 1).astype(mx.int32)
    i1 = mx.clip(i1, 0, in_ - 1).astype(mx.int32)
    g0 = mx.take(x, i0, axis=axis)
    g1 = mx.take(x, i1, axis=axis)
    shape = [1] * x.ndim; shape[axis] = out
    w0 = w0.reshape(shape); w1 = w1.reshape(shape)
    return g0 * w0 + g1 * w1


def interpolate_bilinear(x: mx.array, size: tuple[int, int] | None = None,
                         scale_factor: float | None = None,
                         align_corners: bool = False) -> mx.array:
    """x: [N,H,W,C]. Resize H,W to `size` or by `scale_factor` (bilinear)."""
    N, H, W, C = x.shape
    if size is not None:
        oH, oW = size
    else:
        oH, oW = int(round(H * scale_factor)), int(round(W * scale_factor))
    x = _bilinear_1d(x, axis=1, out=oH, align_corners=align_corners)
    x = _bilinear_1d(x, axis=2, out=oW, align_corners=align_corners)
    return x
