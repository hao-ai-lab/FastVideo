# SPDX-License-Identifier: Apache-2.0
"""Eager INT8 VAE epilogue with the reference's separate FP32 operations."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_bias(Acc, XScale, WScale, Bias, Out, M: tl.constexpr, N: tl.constexpr,
                  XS: tl.constexpr, WS: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows, cols = offsets // N, offsets % N
    valid = rows < M
    acc = tl.load(Acc + offsets, valid, 0).to(tl.float32)
    x_scale = tl.load(XScale + rows * XS, valid, 0).to(tl.float32)
    weight_scale = tl.load(WScale + cols * WS).to(tl.float32)
    out = acc * x_scale
    out = out * weight_scale
    if HAS_BIAS:
        out = out + tl.load(Bias + cols).to(tl.float32)
    tl.store(Out + offsets, out, valid)


def fused_int8_dequant_bias(acc: torch.Tensor, x_scale: torch.Tensor,
                            weight_scale: torch.Tensor, bias: torch.Tensor | None,
                            dtype: torch.dtype) -> torch.Tensor:
    """Avoid full-size FP32 scaling intermediates; retain both rounding steps."""
    rows, cols = acc.shape
    if not acc.is_cuda or acc.dtype != torch.int32 or not acc.is_contiguous():
        raise ValueError("INT8 VAE epilogue requires a contiguous CUDA INT32 matrix")
    if x_scale.shape != (rows, 1) or weight_scale.shape != (cols, 1):
        raise ValueError("INT8 VAE epilogue requires per-row and per-output-channel scales")
    if any(t.device != acc.device for t in (x_scale, weight_scale)):
        raise ValueError("INT8 VAE epilogue scales must be on the accumulator device")
    if bias is not None and (bias.device != acc.device or bias.shape != (cols,) or not bias.is_contiguous()):
        raise ValueError("INT8 VAE epilogue bias must be contiguous on the accumulator device")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("INT8 VAE epilogue supports FP32, FP16 and BF16 outputs")
    out = torch.empty((rows, cols), device=acc.device, dtype=dtype)
    _dequant_bias[(triton.cdiv(rows * cols, 1024),)](
        acc, x_scale, weight_scale, bias if bias is not None else acc, out,
        rows, cols, x_scale.stride(0), weight_scale.stride(0), bias is not None,
        BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return out
