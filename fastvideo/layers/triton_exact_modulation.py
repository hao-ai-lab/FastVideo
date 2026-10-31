# SPDX-License-Identifier: Apache-2.0
"""Bit-exact Triton kernels for the elementwise chains around Wan's LayerNorms.

A Wan block (fastvideo/models/wan/transformer.py, causal_transformer.py) wraps
each LayerNorm in eager elementwise chains: the gated residual sum before the
norm, and the ``normed * (1 + scale) + shift`` modulation plus the
``.to(stream dtype)`` casts after it. Each chain is three to five full-width
launches that materialize intermediates. These kernels run each chain in one
pass and leave the LayerNorm call itself eager, so the output equals the eager
output bit for bit. That is the difference from
``fastvideo/layers/triton_fused_norm.py`` (``FASTVIDEO_ENABLE_FUSED_NORM=1``),
which fuses the norm as well and so changes its reduction order.

Two details keep the rounding exact, as in
``fastvideo/models/dits/minimax_h3_fusions/exact.py``:

- every fp32 multiply and add is issued as explicit round-to-nearest PTX
  (``mul.rn``/``add.rn``), so the compiler cannot contract them into one FMA;
- every intermediate that eager materializes in bf16 is rounded to bf16
  (round-to-nearest-even) at the same point, and nowhere else.

Where eager rounds depends on the dtypes it promotes to. The dense block
modulates with fp32 tables: a fp32 gate promotes the residual sum to fp32, the
norm consumes the unrounded sum and returns fp32, and the scalar gate ``1``
keeps the sum in bf16, so the norm consumes the rounded sum and rounds its own
output to bf16 before the modulation. The causal block modulates with bf16
tables, so every product and sum rounds to bf16. ``gate_residual`` and
``modulate`` derive the rounding points from the gate/table dtype;
``residual_norm_modulate`` reproduces ``ScaleResidualLayerNormScaleShift``.

Modulation tables are ``[B|1, S|1, H]`` (one row per token or per batch) or the
per-frame ``[B|1, F, 1, H]`` that ``x.unflatten(1, (F, S // F))`` broadcasts in
eager. Inference-only (no backward): every entry point has a ``supports_*``
predicate that also refuses grad-enabled, compiling, CPU and non-bf16-stream
inputs; callers fall back to the eager expression outside it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.layers.triton_fused_norm import modulation_layout

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None
    tl = None

HAVE_TRITON = triton is not None

_BLOCK_C = 1024
_TABLE_DTYPES = (torch.float32, torch.bfloat16)

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
    def _gate_residual_kernel(X, RES, GATE, OUT16, OUT32, S, cols, g_sb, g_ss, g_sd, HAS_GATE: tl.constexpr,
                              GATE_BF16: tl.constexpr, FP32_COPY: tl.constexpr, BLOCK_C: tl.constexpr):
        # OUT16 = bf16(residual + x * gate) as eager materializes it. OUT32 = the
        # fp32 tensor the eager norm consumes: the unrounded sum for a fp32 gate
        # (promotion), the bf16 sum upcast for a bf16 or scalar gate (eager
        # materializes bf16, FP32LayerNorm calls .float()).
        row = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        m = c < cols
        x = tl.load(X + row * cols + c, mask=m, other=0.0).to(tl.float32)
        r = tl.load(RES + row * cols + c, mask=m, other=0.0).to(tl.float32)
        if HAS_GATE:
            b = row // S
            f = (row % S) // g_sd
            g = tl.load(GATE + b * g_sb + f * g_ss + c, mask=m, other=0.0).to(tl.float32)
            prod = _mul_rn(x, g)
            if GATE_BF16:
                # eager: bf16 * bf16 -> bf16 product before the bf16 sum.
                prod = _bf16(prod).to(tl.float32)
            acc = _add_rn(r, prod)
        else:
            acc = _add_rn(r, x)
        o16 = _bf16(acc)
        tl.store(OUT16 + row * cols + c, o16, mask=m)
        if FP32_COPY:
            if GATE_BF16 or not HAS_GATE:
                acc = o16.to(tl.float32)
            tl.store(OUT32 + row * cols + c, acc, mask=m)

    @triton.jit
    def _modulate_kernel(N, SCALE, SHIFT, OUT, S, cols, sc_sb, sc_ss, sc_sd, sh_sb, sh_ss, sh_sd,
                         ROUND_INPUT: tl.constexpr, ROUND_ONE_PLUS: tl.constexpr, ROUND_PRODUCT: tl.constexpr,
                         BLOCK_C: tl.constexpr):
        # OUT = bf16(n * (1 + scale) + shift), rounding where eager's type
        # promotion materializes bf16: ROUND_INPUT rounds a fp32 n first (an
        # FP32LayerNorm whose input was bf16), ROUND_ONE_PLUS rounds 1 + scale
        # (bf16 table), ROUND_PRODUCT rounds n * (1 + scale) (bf16 n and table).
        row = tl.program_id(0).to(tl.int64)
        b = row // S
        s = row % S
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        m = c < cols
        n = tl.load(N + row * cols + c, mask=m, other=0.0).to(tl.float32)
        if ROUND_INPUT:
            n = _bf16(n).to(tl.float32)
        sc = tl.load(SCALE + b * sc_sb + (s // sc_sd) * sc_ss + c, mask=m, other=0.0).to(tl.float32)
        sh = tl.load(SHIFT + b * sh_sb + (s // sh_sd) * sh_ss + c, mask=m, other=0.0).to(tl.float32)
        one_plus = _add_rn(sc, tl.full([BLOCK_C], 1.0, tl.float32))
        if ROUND_ONE_PLUS:
            one_plus = _bf16(one_plus).to(tl.float32)
        y = _mul_rn(n, one_plus)
        if ROUND_PRODUCT:
            y = _bf16(y).to(tl.float32)
        tl.store(OUT + row * cols + c, _bf16(_add_rn(y, sh)), mask=m)


def _runtime_ok(*tensors: torch.Tensor) -> bool:
    return (HAVE_TRITON and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
            and all(t.is_cuda for t in tensors))


def _table_ok(t: torch.Tensor, batch: int, seq: int, hidden: int) -> bool:
    return t.dtype in _TABLE_DTYPES and t.is_cuda and modulation_layout(t, batch, seq, hidden) is not None


def _tables_ok(scale: torch.Tensor, shift: torch.Tensor, batch: int, seq: int, hidden: int) -> bool:
    return (scale.dtype == shift.dtype and _table_ok(scale, batch, seq, hidden)
            and _table_ok(shift, batch, seq, hidden))


def _scalar_gate(gate: torch.Tensor | int | None) -> bool:
    return gate is None or (not isinstance(gate, torch.Tensor) and gate == 1)


def _sum_is_bf16(gate: torch.Tensor | int | None) -> bool:
    """Whether eager materializes ``residual + x * gate`` in bf16 (scalar or bf16 gate) rather than fp32."""
    return _scalar_gate(gate) or gate.dtype == torch.bfloat16  # type: ignore[union-attr]


def supports_gate_residual(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor | int | None) -> bool:
    """Whether ``gate_residual`` reproduces eager ``residual + x * gate`` on these inputs.

    The stream must be bf16 and the gate a fp32 or bf16 table (per token, per
    batch or per frame), ``None`` or ``1``.
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
    return _table_ok(gate, batch, seq, hidden)


def supports_modulate(normed: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> bool:
    """Whether ``modulate`` reproduces eager ``normed * (1 + scale) + shift`` rounded to bf16."""
    if not _runtime_ok(normed, scale, shift):
        return False
    if normed.dim() != 3 or normed.dtype not in _TABLE_DTYPES or not normed.is_contiguous():
        return False
    batch, seq, hidden = normed.shape
    return _tables_ok(scale, shift, batch, seq, hidden)


def supports_residual_norm(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor | int,
                           shift: torch.Tensor | None, scale: torch.Tensor | None, norm: nn.Module) -> bool:
    """Whether ``residual_norm_modulate`` reproduces ``ScaleResidualLayerNormScaleShift`` with a LayerNorm."""
    if not isinstance(norm, nn.LayerNorm) or not supports_gate_residual(residual, x, gate):
        return False
    if shift is None and scale is None:
        return True
    if shift is None or scale is None:
        return False
    batch, seq, hidden = residual.shape
    return _tables_ok(scale, shift, batch, seq, hidden)


def gate_residual(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor | int | None, *,
                  fp32_copy: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``residual + x * gate`` (``residual + x`` for a scalar gate) as eager computes it, in one launch.

    Returns the bf16 stream tensor eager materializes (or that the caller's
    ``.to(stream dtype)`` would produce) and, with ``fp32_copy``, the fp32
    tensor an FP32LayerNorm consumes.
    """
    batch, seq, hidden = residual.shape
    gate_t = gate if isinstance(gate, torch.Tensor) else None
    out16 = torch.empty_like(residual)
    out32 = torch.empty(residual.shape, dtype=torch.float32, device=residual.device) if fp32_copy else out16
    if gate_t is None:
        g_sb = g_ss = 0
        g_sd = 1
        gate_arg = residual  # unused placeholder pointer
    else:
        g_sb, g_ss, g_sd = modulation_layout(gate_t, batch, seq, hidden)  # type: ignore[misc]
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
                                                                        g_sd,
                                                                        HAS_GATE=gate_t is not None,
                                                                        GATE_BF16=gate_t is not None
                                                                        and gate_t.dtype == torch.bfloat16,
                                                                        FP32_COPY=fp32_copy,
                                                                        BLOCK_C=_BLOCK_C)
    return out16, (out32 if fp32_copy else None)


def modulate(normed: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, *, round_input: bool) -> torch.Tensor:
    """``(normed * (1 + scale) + shift)`` rounded to bf16, in one launch.

    ``round_input`` first rounds a fp32 ``normed`` to bf16, as FP32LayerNorm
    does to its output when its input was bf16. The remaining rounding points
    follow eager's type promotion: ``1 + scale`` rounds to bf16 for a bf16
    table, the product rounds when both operands are bf16, and the sum rounds
    at the store (eager's final cast, or its bf16 result).
    """
    batch, seq, hidden = normed.shape
    table_bf16 = scale.dtype == torch.bfloat16
    normed_bf16 = normed.dtype == torch.bfloat16 or round_input
    out = torch.empty(normed.shape, dtype=torch.bfloat16, device=normed.device)
    sc_sb, sc_ss, sc_sd = modulation_layout(scale, batch, seq, hidden)  # type: ignore[misc]
    sh_sb, sh_ss, sh_sd = modulation_layout(shift, batch, seq, hidden)  # type: ignore[misc]
    _modulate_kernel[(batch * seq, triton.cdiv(hidden, _BLOCK_C))](normed,
                                                                   scale,
                                                                   shift,
                                                                   out,
                                                                   seq,
                                                                   hidden,
                                                                   sc_sb,
                                                                   sc_ss,
                                                                   sc_sd,
                                                                   sh_sb,
                                                                   sh_ss,
                                                                   sh_sd,
                                                                   ROUND_INPUT=round_input
                                                                   and normed.dtype == torch.float32,
                                                                   ROUND_ONE_PLUS=table_bf16,
                                                                   ROUND_PRODUCT=table_bf16 and normed_bf16,
                                                                   BLOCK_C=_BLOCK_C)
    return out


def eager_modulate(normed: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """The eager expression, with the per-frame unflatten the Wan blocks write for 4-D tables."""
    if scale.dim() == 4:
        frames = scale.shape[1]
        frame_seqlen = normed.shape[1] // frames
        return (normed.unflatten(dim=1, sizes=(frames, frame_seqlen)) * (1 + scale) + shift).flatten(1, 2)
    return normed * (1 + scale) + shift


def modulate_to(normed: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                dtype: torch.dtype | None) -> torch.Tensor:
    """``eager_modulate(...).to(dtype)`` (no cast when ``dtype`` is None); one launch when the exact kernel applies.

    The kernel writes bf16, so it applies when the caller casts to bf16, or
    casts nothing and eager's result is bf16 already (bf16 normed and tables).
    """
    out_dtype = dtype if dtype is not None else torch.promote_types(normed.dtype, scale.dtype)
    if out_dtype == torch.bfloat16 and supports_modulate(normed, scale, shift):
        return modulate(normed, scale, shift, round_input=False)
    out = eager_modulate(normed, scale, shift)
    return out if dtype is None else out.to(dtype)


def residual_norm_modulate(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor | int,
                           shift: torch.Tensor | None, scale: torch.Tensor | None,
                           norm: nn.LayerNorm) -> tuple[torch.Tensor, torch.Tensor]:
    """``ScaleResidualLayerNormScaleShift``'s eager chain, bit for bit, in three launches.

    Caller must have checked ``supports_residual_norm``. The norm is called the
    way eager calls it: ``FP32LayerNorm`` on the fp32 sum it would upcast to, a
    plain ``nn.LayerNorm`` on the sum in its eager dtype. Returns
    ``(output, residual)`` with the residual in the stream dtype, a modulated
    output in bf16, and a bare normalized output in the dtype eager returns, so
    a caller's ``.to(stream dtype)`` is a no-op or the single remaining cast.
    """
    from fastvideo.layers.layernorm import FP32LayerNorm

    sum_is_bf16 = _sum_is_bf16(gate)
    fp32_norm = isinstance(norm, FP32LayerNorm)
    res16, sum32 = gate_residual(residual, x, gate, fp32_copy=fp32_norm or not sum_is_bf16)
    if fp32_norm:
        weight = norm.weight.float() if norm.weight is not None else None
        bias = norm.bias.float() if norm.bias is not None else None
        normed = F.layer_norm(sum32, norm.normalized_shape, weight, bias, norm.eps)
        if shift is None or scale is None:
            return (normed.to(torch.bfloat16) if sum_is_bf16 else normed), res16
        return modulate(normed, scale, shift, round_input=sum_is_bf16), res16
    normed = norm(res16 if sum_is_bf16 else sum32)
    if shift is None or scale is None:
        return normed, res16
    return modulate(normed, scale, shift, round_input=False), res16


__all__ = [
    "HAVE_TRITON",
    "eager_modulate",
    "gate_residual",
    "modulate",
    "modulate_to",
    "residual_norm_modulate",
    "supports_gate_residual",
    "supports_modulate",
    "supports_residual_norm",
]
