# SPDX-License-Identifier: Apache-2.0
"""Triton helpers for per-token x per-channel FP8 linears on GPUs whose rowwise ``_scaled_mm`` is slow.

On sm89 (RTX 4090 / L40S / RTX 6000 Ada) ``torch._scaled_mm`` with rowwise scales runs at
~70 TFLOPS, below bf16 (~160), while the per-tensor kernel reaches 220-305 TFLOPS. The
per-token x per-channel result is recovered exactly by running the per-tensor kernel with unit
scales and applying ``out[i, j] *= sx[i] * sw[j]`` in one pass over the output, which costs
5-10% of the GEMM instead of 2-4x.
"""
from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


@functools.cache
def rowwise_scaled_mm_is_slow() -> bool:
    """Ada (sm89) has no fast rowwise-scaled FP8 GEMM in torch; Hopper and Blackwell do."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9)


@triton.jit
def _quantize_rowwise_kernel(x_ptr, q_ptr, s_ptr, K, stride_x, stride_q, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride_x
    amax = tl.zeros((BLOCK_K, ), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        cols = k + tl.arange(0, BLOCK_K)
        amax = tl.maximum(amax, tl.abs(tl.load(x_row + cols, mask=cols < K, other=0.0).to(tl.float32)))
    scale = tl.maximum(tl.max(amax, axis=0) / 448.0, 1.0 / (448.0 * 512.0))
    tl.store(s_ptr + row, scale)
    inv = 1.0 / scale
    for k in range(0, K, BLOCK_K):
        cols = k + tl.arange(0, BLOCK_K)
        v = tl.load(x_row + cols, mask=cols < K, other=0.0).to(tl.float32) * inv
        v = tl.minimum(tl.maximum(v, -448.0), 448.0)
        tl.store(q_ptr + row * stride_q + cols, v.to(tl.float8e4nv), mask=cols < K)


def quantize_rowwise_fp8(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token FP8 quantization in one launch. Returns ``(x_fp8 [M, K], x_scale [M, 1] float32)``."""
    x_2d = x_2d.contiguous()
    M, K = x_2d.shape
    q = torch.empty((M, K), device=x_2d.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((M, 1), device=x_2d.device, dtype=torch.float32)
    if M:
        _quantize_rowwise_kernel[(M, )](x_2d, q, s, K, x_2d.stride(0), q.stride(0), BLOCK_K=1024, num_warps=4)
    return q, s


@triton.jit
def _scale_rows_cols_kernel(o_ptr, sx_ptr, sw_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    # int64 offsets: a 78k-token fc_in output has 2.2e9 elements, past int32.
    ptrs = o_ptr + rows[:, None] * N + cols[None, :]
    v = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    sx = tl.load(sx_ptr + rows, mask=rows < M, other=0.0)
    sw = tl.load(sw_ptr + cols, mask=cols < N, other=0.0)
    tl.store(ptrs, (v * sx[:, None] * sw[None, :]).to(o_ptr.dtype.element_ty), mask=mask)


def scaled_mm_token_channel(x_fp8: torch.Tensor, x_scale: torch.Tensor, w_fp8_t: torch.Tensor,
                            w_scale: torch.Tensor) -> torch.Tensor:
    """``(x_fp8 * x_scale) @ (w_fp8_t * w_scale)`` in bf16 via the fast per-tensor GEMM plus a scale epilogue.

    The unit-scale GEMM output is at most 448^2 * K, far inside bf16 range, and its relative
    precision is that of any bf16 output, so the epilogue loses nothing against rowwise scaling.
    """
    one = torch.ones((), device=x_fp8.device, dtype=torch.float32)
    out = torch._scaled_mm(x_fp8, w_fp8_t, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    if isinstance(out, tuple):
        out = out[0]
    M, N = out.shape
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
    _scale_rows_cols_kernel[grid](out, x_scale.reshape(-1), w_scale.reshape(-1), M, N, BLOCK_M=64, BLOCK_N=128)
    return out
