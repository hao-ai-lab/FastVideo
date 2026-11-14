# SPDX-License-Identifier: Apache-2.0
"""Bit-exact Triton kernels for the elementwise chains around Wan's LayerNorms.

A Wan block (fastvideo/models/wan/transformer.py) wraps each FP32LayerNorm in
eager elementwise chains: the gated residual sum before the norm, and the
``normed * (1 + scale) + shift`` modulation plus the ``.to(stream dtype)``
casts after it. Each chain is three to five full-width launches that
materialize fp32 intermediates. These kernels run each chain in one pass and
leave ``F.layer_norm`` itself eager, so the output equals the eager output bit
for bit. That is the difference from ``fastvideo/layers/triton_fused_norm.py``
(``FASTVIDEO_ENABLE_FUSED_NORM=1``), which fuses the norm as well and so changes
its reduction order.

Two details keep the rounding exact, as in
``fastvideo/models/dits/minimax_h3_fusions/exact.py``:

- every fp32 multiply and add is issued as explicit round-to-nearest PTX
  (``mul.rn``/``add.rn``), so the compiler cannot contract them into one FMA;
- every intermediate that eager materializes in bf16 is rounded to bf16
  (round-to-nearest-even) at the same point, and nowhere else.

Where eager rounds depends on the gate. A fp32 gate tensor promotes the residual
sum to fp32: the norm consumes the unrounded sum and returns fp32. The scalar
gate ``1`` keeps the sum in bf16: the norm consumes the rounded sum and rounds
its own output to bf16 before the modulation. ``gate_residual`` and
``modulate`` expose these as flags; ``residual_norm_modulate`` derives them for
``ScaleResidualLayerNormScaleShift``.

Inference-only (no backward). Every entry point has a ``supports_*`` predicate
that also refuses grad-enabled, compiling, CPU and non-bf16-stream inputs;
callers fall back to the eager expression outside it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastvideo.layers.triton_fused_norm import _broadcast_strides

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None
    tl = None

HAVE_TRITON = triton is not None

_BLOCK_C = 1024

if HAVE_TRITON:

    @triton.jit
    def _mul_rn(a, b):
        return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;",
                                         "=r,r,r", [a, b],
                                         dtype=tl.float32,
                                         is_pure=True,
                                         pack=1)

    @triton.jit
    def _add_rn(a, b):
        return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;",
                                         "=r,r,r", [a, b],
                                         dtype=tl.float32,
                                         is_pure=True,
                                         pack=1)

    @triton.jit
    def _bf16(v):
        return v.to(tl.bfloat16, fp_downcast_rounding="rtne")

    @triton.jit
    def _gate_residual_kernel(X, RES, GATE, OUT16, OUT32, S, cols, g_sb, g_ss, HAS_GATE: tl.constexpr,
                              FP32_COPY: tl.constexpr, BLOCK_C: tl.constexpr):
        # OUT16 = bf16(residual + x * gate), rounded once as eager's final cast
        # does. OUT32 = the fp32 tensor the eager norm consumes: the unrounded
        # sum for a fp32 gate tensor (promotion), the bf16 sum upcast for the
        # scalar gate (eager materializes bf16, FP32LayerNorm calls .float()).
        row = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        m = c < cols
        x = tl.load(X + row * cols + c, mask=m, other=0.0).to(tl.float32)
        r = tl.load(RES + row * cols + c, mask=m, other=0.0).to(tl.float32)
        if HAS_GATE:
            b = row // S
            s = row % S
            g = tl.load(GATE + b * g_sb + s * g_ss + c, mask=m, other=0.0)
            acc = _add_rn(r, _mul_rn(x, g))
        else:
            acc = _add_rn(r, x)
        o16 = _bf16(acc)
        tl.store(OUT16 + row * cols + c, o16, mask=m)
        if FP32_COPY:
            if not HAS_GATE:
                acc = o16.to(tl.float32)
            tl.store(OUT32 + row * cols + c, acc, mask=m)

    @triton.jit
    def _modulate_kernel(N, SCALE, SHIFT, OUT, S, cols, sc_sb, sc_ss, sh_sb, sh_ss, ROUND_INPUT: tl.constexpr,
                         BLOCK_C: tl.constexpr):
        # OUT = bf16(n * (1 + scale) + shift) with n fp32; ROUND_INPUT first
        # rounds n to bf16 as FP32LayerNorm does when its input was bf16.
        row = tl.program_id(0).to(tl.int64)
        b = row // S
        s = row % S
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        m = c < cols
        n = tl.load(N + row * cols + c, mask=m, other=0.0)
        if ROUND_INPUT:
            n = _bf16(n).to(tl.float32)
        sc = tl.load(SCALE + b * sc_sb + s * sc_ss + c, mask=m, other=0.0)
        sh = tl.load(SHIFT + b * sh_sb + s * sh_ss + c, mask=m, other=0.0)
        one_plus = _add_rn(sc, tl.full([BLOCK_C], 1.0, tl.float32))
        y = _add_rn(_mul_rn(n, one_plus), sh)
        tl.store(OUT + row * cols + c, _bf16(y), mask=m)


def _runtime_ok(*tensors: torch.Tensor) -> bool:
    return (HAVE_TRITON and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
            and all(t.is_cuda for t in tensors))


def _modulation_ok(t: torch.Tensor, batch: int, seq: int, hidden: int) -> bool:
    return t.dtype == torch.float32 and t.is_cuda and _broadcast_strides(t, batch, seq, hidden) is not None


def _scalar_gate(gate: torch.Tensor | int | None) -> bool:
    return gate is None or (not isinstance(gate, torch.Tensor) and gate == 1)


def supports_gate_residual(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor | int | None) -> bool:
    """Whether ``gate_residual`` reproduces eager ``residual + x * gate`` on these inputs.

    The stream must be bf16 and the gate a fp32 ``[B|1, S|1, H]`` tensor, ``None`` or
    ``1``. A bf16 gate tensor rounds the product in eager and is left to eager.
    """
    if not _runtime_ok(residual, x):
        return False
    if (residual.dim() != 3 or residual.shape != x.shape or residual.dtype != torch.bfloat16
            or x.dtype != torch.bfloat16 or not residual.is_contiguous() or not x.is_contiguous()):
        return False
    if _scalar_gate(gate):
        return True
    if not isinstance(gate, torch.Tensor):
        return False
    batch, seq, hidden = residual.shape
    return _modulation_ok(gate, batch, seq, hidden)


def supports_modulate(normed: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> bool:
    """Whether ``modulate`` reproduces eager ``(normed * (1 + scale) + shift).to(bf16)``."""
    if not _runtime_ok(normed, scale, shift):
        return False
    if normed.dim() != 3 or normed.dtype != torch.float32 or not normed.is_contiguous():
        return False
    batch, seq, hidden = normed.shape
    return _modulation_ok(scale, batch, seq, hidden) and _modulation_ok(shift, batch, seq, hidden)


def supports_residual_norm(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor | int,
                           shift: torch.Tensor | None, scale: torch.Tensor | None) -> bool:
    """Whether ``residual_norm_modulate`` reproduces ``ScaleResidualLayerNormScaleShift`` with FP32LayerNorm."""
    if not supports_gate_residual(residual, x, gate):
        return False
    if shift is None and scale is None:
        return True
    if shift is None or scale is None:
        return False
    batch, seq, hidden = residual.shape
    return _modulation_ok(scale, batch, seq, hidden) and _modulation_ok(shift, batch, seq, hidden)


def gate_residual(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor | int | None, *,
                  fp32_copy: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``residual + x * gate`` (``residual + x`` for a scalar gate) as eager computes it, in one launch.

    Returns the bf16 stream tensor that the caller's ``.to(stream dtype)`` would
    produce and, with ``fp32_copy``, the fp32 tensor the eager norm consumes.
    """
    batch, seq, hidden = residual.shape
    gate_t = gate if isinstance(gate, torch.Tensor) else None
    out16 = torch.empty_like(residual)
    out32 = torch.empty(residual.shape, dtype=torch.float32, device=residual.device) if fp32_copy else out16
    if gate_t is None:
        g_sb = g_ss = 0
        gate_arg = residual  # unused placeholder pointer
    else:
        g_sb, g_ss = _broadcast_strides(gate_t, batch, seq, hidden)  # type: ignore[misc]
        gate_arg = gate_t
    _gate_residual_kernel[(batch * seq, triton.cdiv(hidden, _BLOCK_C))](x,
                                                                        residual,
                                                                        gate_arg,
                                                                        out16,
                                                                        out32,
                                                                        seq,
                                                                        hidden,
                                                                        g_sb,
                                                                        g_ss,
                                                                        HAS_GATE=gate_t is not None,
                                                                        FP32_COPY=fp32_copy,
                                                                        BLOCK_C=_BLOCK_C)
    return out16, (out32 if fp32_copy else None)


def modulate(normed: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, *, round_input: bool) -> torch.Tensor:
    """``(normed * (1 + scale) + shift).to(bfloat16)`` for a fp32 ``normed``, in one launch.

    ``round_input`` first rounds ``normed`` to bf16, as FP32LayerNorm does to its
    output when its input was bf16.
    """
    batch, seq, hidden = normed.shape
    out = torch.empty(normed.shape, dtype=torch.bfloat16, device=normed.device)
    sc_sb, sc_ss = _broadcast_strides(scale, batch, seq, hidden)  # type: ignore[misc]
    sh_sb, sh_ss = _broadcast_strides(shift, batch, seq, hidden)  # type: ignore[misc]
    _modulate_kernel[(batch * seq, triton.cdiv(hidden, _BLOCK_C))](normed,
                                                                   scale,
                                                                   shift,
                                                                   out,
                                                                   seq,
                                                                   hidden,
                                                                   sc_sb,
                                                                   sc_ss,
                                                                   sh_sb,
                                                                   sh_ss,
                                                                   ROUND_INPUT=round_input,
                                                                   BLOCK_C=_BLOCK_C)
    return out


def modulate_to(normed: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``(normed * (1 + scale) + shift).to(dtype)``; one launch when the exact kernel applies."""
    if dtype == torch.bfloat16 and supports_modulate(normed, scale, shift):
        return modulate(normed, scale, shift, round_input=False)
    return (normed * (1 + scale) + shift).to(dtype)


def residual_norm_modulate(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor | int,
                           shift: torch.Tensor | None, scale: torch.Tensor | None,
                           norm: torch.nn.LayerNorm) -> tuple[torch.Tensor, torch.Tensor]:
    """``ScaleResidualLayerNormScaleShift``'s eager chain with FP32LayerNorm, bit for bit, in three launches.

    Caller must have checked ``supports_residual_norm``. Returns ``(output, residual)``
    with the residual in the stream dtype, a modulated output in the stream dtype,
    and a bare normalized output in the dtype eager returns (fp32 for a fp32 gate,
    bf16 for the scalar gate), so a caller's ``.to(stream dtype)`` is a no-op or
    the single remaining cast.
    """
    sum_is_bf16 = _scalar_gate(gate)
    res16, sum32 = gate_residual(residual, x, gate, fp32_copy=True)
    weight = norm.weight.float() if norm.weight is not None else None
    bias = norm.bias.float() if norm.bias is not None else None
    normed32 = F.layer_norm(sum32, norm.normalized_shape, weight, bias, norm.eps)
    if shift is None or scale is None:
        return (normed32.to(torch.bfloat16) if sum_is_bf16 else normed32), res16
    return modulate(normed32, scale, shift, round_input=sum_is_bf16), res16


__all__ = [
    "HAVE_TRITON",
    "gate_residual",
    "modulate",
    "modulate_to",
    "residual_norm_modulate",
    "supports_gate_residual",
    "supports_modulate",
    "supports_residual_norm",
]
