# SPDX-License-Identifier: Apache-2.0
"""Triton-fused residual + LayerNorm + scale/shift for inference.

Collapses the eager chain

    residual_out = residual + x * gate        (or residual + x)
    normalized   = FP32LayerNorm(residual_out)
    modulated    = normalized * (1 + scale) + shift        (optional)
    ... followed by the caller's .to(orig_dtype) casts ...

into a single kernel with two outputs, both already in the stream dtype, so the
caller-side casts become no-ops. All arithmetic is fp32 regardless of I/O dtype.

Numerics: the eager path's rounding points are replicated exactly. When the
gate is a fp32 tensor, eager type promotion keeps the whole chain in fp32 with
one final round -- the kernel does the same. When the gate is the scalar 1 and
the stream is bf16, eager materializes bf16 intermediates (the residual sum and
the norm output), so the kernel round-trips through bf16 at the same two
points. The only remaining difference from eager is the reduction order inside
mean/variance (last-ulp fp32).

Inference-only: callers must gate on ``torch.is_grad_enabled()``; there is no
backward. Disable globally with FASTVIDEO_DISABLE_FUSED_NORM=1.
"""

from __future__ import annotations

import torch

import fastvideo.envs as envs

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only installs
    _HAS_TRITON = False

_SUPPORTED_STREAM_DTYPES = (torch.bfloat16, torch.float32)
_MAX_HIDDEN = 8192

if _HAS_TRITON:

    @triton.jit
    def _fused_residual_norm_mod_kernel(
        X,
        RES,
        GATE,
        W,
        B,
        SHIFT,
        SCALE,
        OUT,
        RES_OUT,
        S,
        H,
        x_sb,
        x_ss,
        r_sb,
        r_ss,
        g_sb,
        g_ss,
        sh_sb,
        sh_ss,
        sc_sb,
        sc_ss,
        o_sb,
        o_ss,
        ro_sb,
        ro_ss,
        eps,
        HAS_GATE_TENSOR: tl.constexpr,
        INTERMEDIATE_ROUND: tl.constexpr,
        HAS_AFFINE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_MOD: tl.constexpr,
        OUT_BF16: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        b = row // S
        s = row % S
        cols = tl.arange(0, BLOCK_H)
        mask = cols < H

        x = tl.load(X + b * x_sb + s * x_ss + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(RES + b * r_sb + s * r_ss + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_GATE_TENSOR:
            g = tl.load(GATE + b * g_sb + s * g_ss + cols, mask=mask, other=0.0).to(tl.float32)
            acc = r + x * g
        else:
            acc = r + x

        if OUT_BF16:
            res_bf16 = acc.to(tl.bfloat16)
            tl.store(RES_OUT + b * ro_sb + s * ro_ss + cols, res_bf16, mask=mask)
            if INTERMEDIATE_ROUND:
                # eager produced a bf16 residual tensor here; the norm consumed
                # the rounded values.
                acc = res_bf16.to(tl.float32)
        else:
            tl.store(RES_OUT + b * ro_sb + s * ro_ss + cols, acc, mask=mask)

        mean = tl.sum(tl.where(mask, acc, 0.0), axis=0) / H
        diff = tl.where(mask, acc - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / H
        xhat = diff * tl.rsqrt(var + eps)
        if HAS_AFFINE:
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            xhat = xhat * w
            if HAS_BIAS:
                bias = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
                xhat = xhat + bias

        if HAS_MOD:
            if INTERMEDIATE_ROUND and OUT_BF16:
                # eager rounded the norm output to bf16 before modulating.
                xhat = xhat.to(tl.bfloat16).to(tl.float32)
            sc = tl.load(SCALE + b * sc_sb + s * sc_ss + cols, mask=mask, other=0.0).to(tl.float32)
            sh = tl.load(SHIFT + b * sh_sb + s * sh_ss + cols, mask=mask, other=0.0).to(tl.float32)
            y = xhat * (1.0 + sc) + sh
        else:
            y = xhat

        if OUT_BF16:
            tl.store(OUT + b * o_sb + s * o_ss + cols, y.to(tl.bfloat16), mask=mask)
        else:
            tl.store(OUT + b * o_sb + s * o_ss + cols, y, mask=mask)


def _broadcast_strides(t: torch.Tensor, batch: int, seq: int) -> tuple[int, int] | None:
    """(batch_stride, seq_stride) for a [B|1, S|1, H] tensor, or None if unsupported."""
    if t.dim() != 3 or t.stride(-1) != 1 or t.shape[-1] == 0:
        return None
    tb, ts, _ = t.shape
    if tb not in (1, batch) or ts not in (1, seq):
        return None
    return (t.stride(0) if tb == batch else 0, t.stride(1) if ts == seq else 0)


def fused_path_supported(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | int,
    shift: torch.Tensor | None,
    scale: torch.Tensor | None,
    norm: torch.nn.Module,
) -> bool:
    """Cheap eligibility check; any False falls back to the eager path."""
    if not _HAS_TRITON or envs.FASTVIDEO_DISABLE_FUSED_NORM:
        return False
    if torch.is_grad_enabled():
        return False
    # LayerNorm family only (RMS keeps its own path); weight/bias fp32 as
    # FP32LayerNorm guarantees after .float().
    if not isinstance(norm, torch.nn.LayerNorm):
        return False
    if residual.dim() != 3 or residual.shape != x.shape:
        return False
    if residual.dtype != x.dtype or residual.dtype not in _SUPPORTED_STREAM_DTYPES:
        return False
    if not (residual.is_cuda and x.is_cuda):
        return False
    if not (residual.is_contiguous() and x.is_contiguous()):
        return False
    batch, seq, hidden = residual.shape
    if hidden > _MAX_HIDDEN or norm.normalized_shape != (hidden, ):
        return False
    if isinstance(gate, torch.Tensor):
        if gate.dtype not in (torch.float32, residual.dtype) or not gate.is_cuda:
            return False
        if _broadcast_strides(gate, batch, seq) is None:
            return False
    elif gate != 1:
        return False
    if (shift is None) != (scale is None):
        return False
    if shift is not None and scale is not None:
        for t in (shift, scale):
            if t.dtype not in (torch.float32, residual.dtype) or not t.is_cuda:
                return False
            if _broadcast_strides(t, batch, seq) is None:
                return False
    return True


def fused_residual_norm_mod(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | int,
    shift: torch.Tensor | None,
    scale: torch.Tensor | None,
    norm: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused kernel. Caller must have checked fused_path_supported().

    Returns (modulated_or_normalized, residual_out), both in the stream dtype.
    """
    batch, seq, hidden = residual.shape
    stream_dtype = residual.dtype
    out = torch.empty_like(residual)
    res_out = torch.empty_like(residual)

    gate_is_tensor = isinstance(gate, torch.Tensor)
    has_mod = shift is not None
    # Eager promotion keeps the chain fp32 (no intermediate rounding) exactly
    # when the gate is a fp32 tensor; the scalar-1 gate path materializes
    # stream-dtype intermediates.
    intermediate_round = (not gate_is_tensor or gate.dtype == stream_dtype) and stream_dtype == torch.bfloat16

    if gate_is_tensor:
        g_sb, g_ss = _broadcast_strides(gate, batch, seq)  # type: ignore[misc]
        gate_arg = gate
    else:
        g_sb = g_ss = 0
        gate_arg = residual  # unused placeholder pointer
    if has_mod:
        sh_sb, sh_ss = _broadcast_strides(shift, batch, seq)  # type: ignore[arg-type,misc]
        sc_sb, sc_ss = _broadcast_strides(scale, batch, seq)  # type: ignore[arg-type,misc]
        shift_arg, scale_arg = shift, scale
    else:
        sh_sb = sh_ss = sc_sb = sc_ss = 0
        shift_arg = scale_arg = residual  # unused placeholder pointers

    weight = norm.weight
    bias = norm.bias
    has_affine = weight is not None
    has_bias = bias is not None

    block_h = triton.next_power_of_2(hidden)
    num_warps = 4 if block_h <= 2048 else 8

    _fused_residual_norm_mod_kernel[(batch * seq, )](
        x,
        residual,
        gate_arg,
        weight if has_affine else residual,
        bias if has_bias else residual,
        shift_arg,
        scale_arg,
        out,
        res_out,
        seq,
        hidden,
        x.stride(0),
        x.stride(1),
        residual.stride(0),
        residual.stride(1),
        g_sb,
        g_ss,
        sh_sb,
        sh_ss,
        sc_sb,
        sc_ss,
        out.stride(0),
        out.stride(1),
        res_out.stride(0),
        res_out.stride(1),
        norm.eps,
        HAS_GATE_TENSOR=gate_is_tensor,
        INTERMEDIATE_ROUND=intermediate_round,
        HAS_AFFINE=has_affine,
        HAS_BIAS=has_bias,
        HAS_MOD=has_mod,
        OUT_BF16=stream_dtype == torch.bfloat16,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
    return out, res_out
