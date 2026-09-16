# SPDX-License-Identifier: Apache-2.0
"""Fused FP8 (e4m3fn) activation quantization for the generic FP8 linear path.

The eager formulas in ``fp8_config`` (``abs`` -> ``amax`` -> ``div`` -> ``clamp`` ->
``to(float8)``) are four to five memory passes over the activation. At MiniMax-H3's
shapes that costs about as much as the FP8 ``_scaled_mm`` it feeds (1.06 ms next
to a 1.25 ms GEMM for a 65,536 x 5,376 activation on MI355X). These kernels do the
same math in one pass (rowwise) or an ``aminmax`` reduction plus one pass
(tensorwise): 3x (tensorwise) and 4-5x (rowwise) faster at those shapes.
"""
from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _rowwise_quant_kernel(x_ptr, out_ptr, scale_ptr, K, stride_x, stride_out, FP8_MAX: tl.constexpr,
                              MIN_SCALE: tl.constexpr, BLOCK: tl.constexpr):
        """One program per row: amax over the row, then scale/clamp/cast in a second sweep.

        The row is re-read from L2 in the second sweep, so K is not limited by
        the register tile and the kernel stays one launch."""
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_x
        amax = tl.zeros([BLOCK], dtype=tl.float32)
        for start in range(0, K, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            x = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.abs(x))
        scale = tl.maximum(tl.max(amax, axis=0) / FP8_MAX, MIN_SCALE)
        inv = 1.0 / scale
        out_row = out_ptr + row * stride_out
        for start in range(0, K, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            x = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
            y = tl.minimum(tl.maximum(x * inv, -FP8_MAX), FP8_MAX)
            tl.store(out_row + offs, y.to(out_ptr.dtype.element_ty), mask=offs < K)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _scale_cast_kernel(x_ptr, out_ptr, inv_scale_ptr, n_elements, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
        """Tensorwise second pass: one read, one FP8 write."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        inv = tl.load(inv_scale_ptr)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.minimum(tl.maximum(x * inv, -FP8_MAX), FP8_MAX)
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def triton_quant_available(x: torch.Tensor) -> bool:
    return _HAS_TRITON and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)


def quantize_rowwise_fused(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(x_fp8 [M, K], x_scale [M, 1] float32)``; one kernel launch."""
    x_2d = x_2d.contiguous()
    m, k = x_2d.shape
    out = torch.empty((m, k), dtype=FP8_DTYPE, device=x_2d.device)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x_2d.device)
    block = min(4096, triton.next_power_of_2(k))
    _rowwise_quant_kernel[(m, )](x_2d,
                                 out,
                                 scale,
                                 k,
                                 x_2d.stride(0),
                                 out.stride(0),
                                 FP8_MAX=FP8_MAX,
                                 MIN_SCALE=FP8_MIN_SCALE,
                                 BLOCK=block,
                                 num_warps=8)
    return out, scale


def quantize_tensorwise_fused(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(x_fp8 [M, K], x_scale [1] float32)``; an ``aminmax`` reduction + one pass."""
    x_2d = x_2d.contiguous()
    lo, hi = torch.aminmax(x_2d)
    amax = torch.maximum(lo.abs(), hi.abs()).float()
    scale = (amax / FP8_MAX).clamp(min=FP8_MIN_SCALE).view(1)
    inv = 1.0 / scale
    out = torch.empty_like(x_2d, dtype=FP8_DTYPE)
    n = x_2d.numel()
    block = 4096
    _scale_cast_kernel[(triton.cdiv(n, block), )](x_2d, out, inv, n, FP8_MAX=FP8_MAX, BLOCK=block, num_warps=8)
    return out, scale
