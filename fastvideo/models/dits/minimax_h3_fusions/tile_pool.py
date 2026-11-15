# SPDX-License-Identifier: Apache-2.0
"""ATen-compatible tile sums for the fused attention producers."""
import triton
import triton.language as tl

from .value_quant import _left_four


@triton.jit
def _tile_sum(values, R: tl.constexpr):
    """Sum 64 or 128 rows in the same order as the eager FP32 tile reduction."""
    raw = tl.trans(tl.reshape(values, (R // 16, 16, 128)), (1, 2, 0))
    if R == 64:
        even, odd = tl.split(tl.reshape(raw, (16, 128, 2, 2)))
        x0, x2 = tl.split(even)
        x1, x3 = tl.split(odd)
        acc = ((x0 + x1) + x2) + x3
    else:
        even, odd = tl.split(tl.reshape(raw, (16, 128, 4, 2)))
        x04, x26 = tl.split(tl.reshape(even, (16, 128, 2, 2)))
        x15, x37 = tl.split(tl.reshape(odd, (16, 128, 2, 2)))
        x0, x4 = tl.split(x04)
        x1, x5 = tl.split(x15)
        x2, x6 = tl.split(x26)
        x3, x7 = tl.split(x37)
        acc = ((((((x0 + x1) + x2) + x3) + x4) + x5) + x6) + x7
    even, odd = tl.split(tl.reshape(tl.trans(acc), (128, 4, 2, 2)))
    y0, y2 = tl.split(even)
    y1, y3 = tl.split(odd)
    y0, y1 = _left_four(y0, 128), _left_four(y1, 128)
    y2, y3 = _left_four(y2, 128), _left_four(y3, 128)
    return (y0 + y2) + (y1 + y3)
