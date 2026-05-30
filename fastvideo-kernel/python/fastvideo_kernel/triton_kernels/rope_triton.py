# SPDX-License-Identifier: Apache-2.0
"""Fused rotary positional embedding (RoPE) Triton kernel.

Replaces the reshape/unbind/stack/flatten/cast chain used by the PyTorch
implementation with a single CTA-per-(token, head) kernel that loads x, cos and
sin once and writes the rotated output once.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


_MODE_ROTATE_HALF = 0
_MODE_GPTJ = 1
_MODE_NEOX = 2


@triton.jit
def _rope_kernel(
    X_ptr,
    COS_ptr,
    SIN_ptr,
    OUT_ptr,
    x_stride_t,
    x_stride_h,
    cos_stride_t,
    D: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    d = tl.arange(0, BLOCK_D)
    d_mask = d < D

    x_row = X_ptr + pid_t * x_stride_t + pid_h * x_stride_h
    out_row = OUT_ptr + pid_t * x_stride_t + pid_h * x_stride_h
    cs_row = pid_t * cos_stride_t

    x = tl.load(x_row + d, mask=d_mask, other=0.0).to(tl.float32)

    if MODE == 0:
        # rotate_half: cos/sin have head_size, partner via adjacent swap.
        c = tl.load(COS_ptr + cs_row + d, mask=d_mask, other=0.0).to(tl.float32)
        s = tl.load(SIN_ptr + cs_row + d, mask=d_mask, other=0.0).to(tl.float32)
        partner = tl.load(x_row + (d ^ 1), mask=d_mask, other=0.0).to(tl.float32)
        sign = tl.where((d & 1) == 0, -1.0, 1.0)
        out = x * c + sign * partner * s
    elif MODE == 1:
        # GPT-J: cos/sin have head_size/2, partner via adjacent swap, c_idx = d//2.
        c = tl.load(COS_ptr + cs_row + (d // 2), mask=d_mask, other=0.0).to(tl.float32)
        s = tl.load(SIN_ptr + cs_row + (d // 2), mask=d_mask, other=0.0).to(tl.float32)
        partner = tl.load(x_row + (d ^ 1), mask=d_mask, other=0.0).to(tl.float32)
        sign = tl.where((d & 1) == 0, -1.0, 1.0)
        out = x * c + sign * partner * s
    else:
        # Neox: cos/sin have head_size/2, partner via half-swap.
        half = D // 2
        is_low = d < half
        c_idx = tl.where(is_low, d, d - half)
        c = tl.load(COS_ptr + cs_row + c_idx, mask=d_mask, other=0.0).to(tl.float32)
        s = tl.load(SIN_ptr + cs_row + c_idx, mask=d_mask, other=0.0).to(tl.float32)
        partner_idx = tl.where(is_low, d + half, d - half)
        partner = tl.load(x_row + partner_idx, mask=d_mask, other=0.0).to(tl.float32)
        sign = tl.where(is_low, -1.0, 1.0)
        out = x * c + sign * partner * s

    tl.store(out_row + d, out.to(OUT_ptr.dtype.element_ty), mask=d_mask)


def _resolve_mode(head_size: int, rope_dim: int, is_neox_style: bool) -> int | None:
    """Return the MODE constant matching the cos/sin layout, or None if unsupported."""
    if rope_dim == head_size:
        return _MODE_ROTATE_HALF
    if rope_dim * 2 == head_size:
        return _MODE_NEOX if is_neox_style else _MODE_GPTJ
    return None


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _is_eligible(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> bool:
    """Only accept contiguous CUDA float tensors with matching token counts."""
    if not (x.is_cuda and cos.is_cuda and sin.is_cuda):
        return False
    if x.dtype not in _SUPPORTED_DTYPES or cos.dtype not in _SUPPORTED_DTYPES:
        return False
    if sin.dtype != cos.dtype or sin.shape != cos.shape:
        return False
    if x.dim() != 3 or cos.dim() != 2 or x.shape[0] != cos.shape[0]:
        return False
    if not (x.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()):
        return False
    return True


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor | None:
    """Apply RoPE to ``x`` with a fused Triton kernel.

    Args:
        x: ``[num_tokens, num_heads, head_size]`` float tensor (contiguous).
        cos, sin: ``[num_tokens, head_size]`` (rotate_half) or
            ``[num_tokens, head_size // 2]`` (GPT-J / Neox).
        is_neox_style: True for Neox split-half layout, False for GPT-J interleaved
            (ignored when ``cos`` has the full head dim).

    Returns:
        A new tensor shaped and typed like ``x``, or ``None`` if the inputs are
        ineligible for the fused path. Callers should fall back to PyTorch on ``None``.
    """
    if not _is_eligible(x, cos, sin):
        return None

    num_tokens, num_heads, head_size = x.shape
    if head_size % 2 != 0:
        return None

    mode = _resolve_mode(head_size, cos.shape[-1], is_neox_style)
    if mode is None:
        return None

    out = torch.empty_like(x)
    block_d = triton.next_power_of_2(head_size)
    num_warps = max(1, min(8, block_d // 32))

    _rope_kernel[(num_tokens, num_heads)](
        x,
        cos,
        sin,
        out,
        x.stride(0),
        x.stride(1),
        cos.stride(0),
        D=head_size,
        MODE=mode,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out
