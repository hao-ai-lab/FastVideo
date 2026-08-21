# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3's value-first packed SwiGLU fusion."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only in environments without Triton
    triton = None
    tl = None
    HAVE_TRITON = False


def _validate_input(x: torch.Tensor) -> int:
    if x.ndim == 0:
        raise ValueError("MiniMax H3 SwiGLU expects at least one dimension")

    packed_width = x.shape[-1]
    if packed_width == 0 or packed_width % 2 != 0:
        raise ValueError(
            "MiniMax H3 SwiGLU expects a positive even last dimension containing packed (value, gate) halves, "
            f"got {packed_width}"
        )
    if not x.is_floating_point():
        raise TypeError(f"MiniMax H3 SwiGLU expects a floating-point tensor, got {x.dtype}")
    return packed_width // 2


if HAVE_TRITON:

    @triton.jit
    def _minimax_h3_swiglu_kernel(
        out_ptr,
        x_ptr,
        ffn_dim,
        stride_in_row,
        stride_out_row,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < ffn_dim

        value = tl.load(x_ptr + row * stride_in_row + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + row * stride_in_row + ffn_dim + cols, mask=mask, other=0.0).to(tl.float32)
        # Match Sol-Engine: keep the complete SwiGLU expression in FP32 and
        # convert only the final output store.
        out = value * (gate * tl.sigmoid(gate))
        tl.store(out_ptr + row * stride_out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

else:
    _minimax_h3_swiglu_kernel = None


def _num_warps(block_size: int) -> int:
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def minimax_h3_swiglu(x: torch.Tensor) -> torch.Tensor:
    """Run the forward-only Triton fusion over an H3 ``(..., 2 * ffn_dim)`` input.

    This is intentionally a strict kernel wrapper: callers own fallback policy and
    must only invoke it for a supported CUDA inference path.
    """
    ffn_dim = _validate_input(x)
    if not x.is_cuda:
        raise ValueError("MiniMax H3 fused SwiGLU requires a CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"MiniMax H3 fused SwiGLU supports float16, bfloat16, and float32, got {x.dtype}")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("MiniMax H3 fused SwiGLU is forward-only and does not implement autograd")
    if _minimax_h3_swiglu_kernel is None:
        raise RuntimeError("MiniMax H3 fused SwiGLU requires Triton")

    packed_width = x.shape[-1]
    flat = x.reshape(-1, packed_width).contiguous()
    output_shape = (*x.shape[:-1], ffn_dim)
    if flat.shape[0] == 0:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)

    out = torch.empty((flat.shape[0], ffn_dim), dtype=x.dtype, device=x.device)
    block_size = triton.next_power_of_2(ffn_dim)
    _minimax_h3_swiglu_kernel[(flat.shape[0],)](
        out,
        flat,
        ffn_dim,
        flat.stride(0),
        out.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return out.view(output_shape)


__all__ = ["HAVE_TRITON", "minimax_h3_swiglu"]
