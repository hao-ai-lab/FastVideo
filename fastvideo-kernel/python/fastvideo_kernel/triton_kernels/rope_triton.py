# SPDX-License-Identifier: Apache-2.0
"""Fused rotary positional embedding (RoPE) Triton kernel.

Replaces the reshape/unbind/stack/flatten/cast chain used by the PyTorch
implementation with a single CTA-per-(token, head) kernel that loads x, cos and
sin once and writes the rotated output once.

The kernel addresses every operand through explicit strides, so it accepts both
the 3D ``[num_tokens, num_heads, head_size]`` layout used by ``forward_native``
and the 4D ``[batch, seq, num_heads, head_size]`` layout used by every attention
call site, and it tolerates the strided / non-contiguous ``cos`` / ``sin``
tensors produced by ``chunk`` / ``tensor_split`` / ``cos[:, ::2]`` slicing.

The 4D batch/seq axes are flattened into a single token axis on grid dim 0 (which
allows the large dimension to exceed the 65535 CUDA y/z grid limit), and the cos
row for token ``t`` is ``t % seq`` so cos/sin broadcast over batch and heads.
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
    cos_stride_r,
    cos_stride_d,
    out_stride_t,
    out_stride_h,
    SEQ,
    D: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    d = tl.arange(0, BLOCK_D)
    d_mask = d < D

    # x / out last dim is contiguous (stride 1, guaranteed by the wrapper).
    x_row = X_ptr + pid_t * x_stride_t + pid_h * x_stride_h
    out_row = OUT_ptr + pid_t * out_stride_t + pid_h * out_stride_h
    # cos / sin broadcast over batch and heads: indexed by sequence position only.
    # For 4D the token axis is b*SEQ + s, so the cos row is the token modulo SEQ.
    cs_row = (pid_t % SEQ) * cos_stride_r

    x = tl.load(x_row + d, mask=d_mask, other=0.0).to(tl.float32)

    if MODE == 0:
        # rotate_half: cos/sin have head_size, partner via adjacent swap.
        c = tl.load(COS_ptr + cs_row + d * cos_stride_d, mask=d_mask, other=0.0).to(tl.float32)
        s = tl.load(SIN_ptr + cs_row + d * cos_stride_d, mask=d_mask, other=0.0).to(tl.float32)
        partner = tl.load(x_row + (d ^ 1), mask=d_mask, other=0.0).to(tl.float32)
        sign = tl.where((d & 1) == 0, -1.0, 1.0)
        out = x * c + sign * partner * s
    elif MODE == 1:
        # GPT-J: cos/sin have head_size/2, partner via adjacent swap, c_idx = d//2.
        c = tl.load(COS_ptr + cs_row + (d // 2) * cos_stride_d, mask=d_mask, other=0.0).to(tl.float32)
        s = tl.load(SIN_ptr + cs_row + (d // 2) * cos_stride_d, mask=d_mask, other=0.0).to(tl.float32)
        partner = tl.load(x_row + (d ^ 1), mask=d_mask, other=0.0).to(tl.float32)
        sign = tl.where((d & 1) == 0, -1.0, 1.0)
        out = x * c + sign * partner * s
    else:
        # Neox: cos/sin have head_size/2, partner via half-swap.
        half = D // 2
        is_low = d < half
        c_idx = tl.where(is_low, d, d - half)
        c = tl.load(COS_ptr + cs_row + c_idx * cos_stride_d, mask=d_mask, other=0.0).to(tl.float32)
        s = tl.load(SIN_ptr + cs_row + c_idx * cos_stride_d, mask=d_mask, other=0.0).to(tl.float32)
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


def _normalize_layout(x: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Map ``x`` to ``(num_tokens, seq, head_size, token_stride)``.

    ``cos`` / ``sin`` are indexed by sequence position, so:
    - 3D ``[T, H, D]``: each token is its own sequence position (``seq == T``).
    - 4D ``[B, S, H, D]``: batch and seq fold into one token axis (``num_tokens = B*S``)
      only when the batch stride equals ``S * seq_stride`` (always true for the
      contiguous / last-dim-sliced tensors call sites pass); the cos row is then
      ``token % S``.

    The last dim must be contiguous (stride 1); the kernel reads adjacent RoPE
    partners directly.
    """
    if x.stride(-1) != 1:
        return None
    if x.dim() == 3:
        seq, _heads, head_size = x.shape
        return seq, seq, head_size, x.stride(0)
    if x.dim() == 4:
        batch, seq, _heads, head_size = x.shape
        # Only mergeable when batch advances by exactly one seq block.
        if x.stride(0) != seq * x.stride(1):
            return None
        return batch * seq, seq, head_size, x.stride(1)
    return None


def _is_eligible(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq: int) -> bool:
    """Only accept CUDA float tensors (any leading stride) with matching seq length."""
    # Inference-only: the fused path returns a bare tensor with no autograd graph,
    # so it would silently drop gradients on the q/k projections during training.
    if torch.is_grad_enabled():
        return False
    if not (x.is_cuda and cos.is_cuda and sin.is_cuda):
        return False
    # All operands must live on the same CUDA device (the kernel uses a single grid).
    if not (x.device == cos.device == sin.device):
        return False
    if x.dtype not in _SUPPORTED_DTYPES or cos.dtype not in _SUPPORTED_DTYPES:
        return False
    if sin.dtype != cos.dtype or sin.shape != cos.shape:
        return False
    if cos.dim() != 2 or cos.shape[0] != seq:
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
        x: ``[num_tokens, num_heads, head_size]`` or ``[batch, seq, num_heads, head_size]``
            float tensor whose last dim is contiguous. Leading dims may be strided.
        cos, sin: ``[seq, head_size]`` (rotate_half) or ``[seq, head_size // 2]``
            (GPT-J / Neox), where ``seq`` matches the sequence axis of ``x``. May be
            non-contiguous (e.g. ``chunk`` / ``tensor_split`` / ``cos[:, ::2]`` views).
        is_neox_style: True for Neox split-half layout, False for GPT-J interleaved
            (ignored when ``cos`` has the full head dim).

    Returns:
        A new contiguous tensor shaped and typed like ``x``, or ``None`` if the inputs
        are ineligible for the fused path. Callers should fall back to PyTorch on ``None``.
    """
    layout = _normalize_layout(x)
    if layout is None:
        return None
    num_tokens, seq, head_size, token_stride = layout

    if not _is_eligible(x, cos, sin, seq):
        return None
    if head_size % 2 != 0:
        return None

    mode = _resolve_mode(head_size, cos.shape[-1], is_neox_style)
    if mode is None:
        return None

    num_heads = x.shape[-2]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    block_d = triton.next_power_of_2(head_size)
    num_warps = max(1, min(8, block_d // 32))

    # Output is freshly contiguous: token stride = num_heads * head_size, head stride = head_size.
    out_token_stride = num_heads * head_size
    _rope_kernel[(num_tokens, num_heads)](
        x,
        cos,
        sin,
        out,
        token_stride,
        x.stride(-2),
        cos.stride(0),
        cos.stride(1),
        out_token_stride,
        head_size,
        seq,
        D=head_size,
        MODE=mode,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out
