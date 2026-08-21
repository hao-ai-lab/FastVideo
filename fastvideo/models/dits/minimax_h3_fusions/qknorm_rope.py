# SPDX-License-Identifier: Apache-2.0
"""Fused per-head RMSNorm and partial rotary embedding for MiniMax H3."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only in environments without Triton
    triton = None
    tl = None
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _qknorm_partial_rope_kernel(
        out_ptr,
        x_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        head_dim,
        rotary_dim,
        half_rotary_dim,
        num_heads,
        seq_len,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # int64, like the sibling kernels: with int32 program ids,
        # ``row * head_dim`` wraps once the flattened input reaches 2**31
        # elements (H3's 56 heads x 128 head_dim crosses that at
        # batch*seq >= 299_593 tokens per rank) and the loads/stores below
        # become out-of-bounds. ``seq_index`` inherits int64 from ``row``.
        row = tl.program_id(0).to(tl.int64)
        seq_index = (row // num_heads) % seq_len
        cols = tl.arange(0, BLOCK_SIZE)
        head_mask = cols < head_dim
        row_offset = row * head_dim

        x = tl.load(x_ptr + row_offset + cols, mask=head_mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / head_dim
        inv_rms = tl.math.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + cols, mask=head_mask, other=0.0).to(tl.float32)
        normalized = x * inv_rms * weight

        rotary_mask = cols < rotary_dim
        first_half = cols < half_rotary_dim
        partner_col = tl.where(first_half, cols + half_rotary_dim, cols - half_rotary_dim)
        partner_x = tl.load(
            x_ptr + row_offset + partner_col,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        partner_weight = tl.load(weight_ptr + partner_col, mask=rotary_mask, other=0.0).to(tl.float32)
        partner_normalized = partner_x * inv_rms * partner_weight
        rotated = tl.where(first_half, -partner_normalized, partner_normalized)

        table_offset = seq_index * rotary_dim + cols
        cos = tl.load(cos_ptr + table_offset, mask=rotary_mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + table_offset, mask=rotary_mask, other=0.0).to(tl.float32)
        rotary_output = normalized * cos + rotated * sin
        output = tl.where(rotary_mask, rotary_output, normalized)
        tl.store(out_ptr + row_offset + cols, output.to(out_ptr.dtype.element_ty), mask=head_mask)


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[int, int, int, int, int]:
    for name, tensor in (("x", x), ("weight", weight), ("cos", cos), ("sin", sin)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")

    if x.ndim != 4:
        raise ValueError(f"x must have shape (batch, seq, heads, head_dim), got {tuple(x.shape)}")
    batch, seq_len, num_heads, head_dim = x.shape
    if min(batch, seq_len, num_heads, head_dim) <= 0:
        raise ValueError(f"x dimensions must all be positive, got {tuple(x.shape)}")
    if weight.shape != (head_dim, ):
        raise ValueError(f"weight must have shape ({head_dim},), got {tuple(weight.shape)}")
    if cos.ndim != 2:
        raise ValueError(f"cos must have shape (seq, rotary_dim), got {tuple(cos.shape)}")
    if sin.shape != cos.shape:
        raise ValueError(f"sin must match cos shape {tuple(cos.shape)}, got {tuple(sin.shape)}")
    if cos.shape[0] != seq_len:
        raise ValueError(f"cos/sin sequence length must be {seq_len}, got {cos.shape[0]}")

    rotary_dim = cos.shape[1]
    if rotary_dim <= 0:
        raise ValueError(f"rotary_dim must be positive, got {rotary_dim}")
    if rotary_dim > head_dim:
        raise ValueError(f"rotary_dim must not exceed head_dim, got rotary_dim={rotary_dim}, head_dim={head_dim}")
    if rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")

    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x dtype must be float16, bfloat16, or float32, got {x.dtype}")
    for name, tensor in (("weight", weight), ("cos", cos), ("sin", sin)):
        if tensor.dtype != x.dtype:
            raise TypeError(f"{name} dtype must match x dtype {x.dtype}, got {tensor.dtype}")
        if tensor.device != x.device:
            raise ValueError(f"{name} device must match x device {x.device}, got {tensor.device}")

    if not isinstance(eps, (float, int)) or not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError(f"eps must be a positive finite number, got {eps!r}")
    return batch, seq_len, num_heads, head_dim, rotary_dim


def fused_qknorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run per-head RMSNorm and partial RoPE in one Sol-Engine-style kernel.

    RMSNorm reduction and RoPE arithmetic stay in FP32 registers until the
    final store. Triton's reduction order and the absence of eager's BF16
    intermediate materializations can produce small, expected rounding drift.

    Row offsets are computed in int64, so inputs beyond 2**31 total elements
    (about 300k tokens per rank at H3's 56 heads x 128 head_dim) address
    correctly.
    """
    batch, seq_len, num_heads, head_dim, rotary_dim = _validate_inputs(x, weight, cos, sin, eps)
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    if not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("cos and sin must be contiguous (seq, rotary_dim) tables")
    if not x.is_cuda:
        raise RuntimeError("fused_qknorm_rope requires CUDA tensors")
    if not HAVE_TRITON:
        raise RuntimeError("fused_qknorm_rope requires Triton")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (x, weight, cos, sin)):
        raise RuntimeError("fused_qknorm_rope is inference-only and does not implement autograd")

    flat_x = x.reshape(-1, head_dim).contiguous()
    flat_out = torch.empty_like(flat_x)
    block_size = 1 << (head_dim - 1).bit_length()
    _qknorm_partial_rope_kernel[(flat_x.shape[0], )](
        flat_out,
        flat_x,
        weight,
        cos,
        sin,
        head_dim,
        rotary_dim,
        rotary_dim // 2,
        num_heads,
        seq_len,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return flat_out.view(batch, seq_len, num_heads, head_dim)


__all__ = ["HAVE_TRITON", "fused_qknorm_rope"]
