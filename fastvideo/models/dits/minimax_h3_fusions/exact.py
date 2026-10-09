# SPDX-License-Identifier: Apache-2.0
"""Bit-exact Triton replacements for MiniMax-H3's eager RoPE, AdaLN and SwiGLU.

Each kernel performs exactly the BF16-rounded operation sequence the eager
PyTorch expression performs, in the same order, so its output equals the eager
output bit for bit. That is the difference from the Sol-Engine fusions in this
package (``FASTVIDEO_MINIMAX_H3_FUSIONS``), which keep the whole expression in
FP32 and round once at the store: faster still, but not the eager bits.

The kernels gain by fusing the eager expression's separate elementwise
launches (and the ``index_select`` gathers of the AdaLN tables) into one pass
over the activations, not by changing the arithmetic. Two details keep the
rounding exact:

- every FP32 multiply, add and divide is issued as explicit round-to-nearest
  PTX (``mul.rn``/``add.rn``/``div.rn``), so the compiler cannot contract a
  multiply and an add into one FMA, or fold the FP32 result and the BF16
  downcast into one BF16 instruction, either of which rounds once where eager
  rounds twice;
- every intermediate eager materializes as a BF16 tensor is rounded to BF16
  (round-to-nearest-even) at the same point.

Selected with ``FASTVIDEO_MINIMAX_H3_EXACT_KERNELS``. Every entry point has a
``supports_*`` predicate; callers fall back to the eager expression for inputs
outside it.
"""

from __future__ import annotations

import torch

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
    def _div_rn(a, b):
        return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;",
                                         "=r,r,r", [a, b],
                                         dtype=tl.float32,
                                         is_pure=True,
                                         pack=1)

    @triton.jit
    def _bf16(v):
        return v.to(tl.bfloat16, fp_downcast_rounding="rtne")

    @triton.jit
    def _rope_prefix_kernel(x, cos, sin, out, heads, x_ss, x_sh, c_ss, D: tl.constexpr, R: tl.constexpr,
                            RP: tl.constexpr, TAIL: tl.constexpr, BLOCK_H: tl.constexpr):
        # out[..., :R] = x * cos + rotate_half(x) * sin; out[..., R:] = x[..., R:].
        s = tl.program_id(0).to(tl.int64)
        hs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        hmask = hs < heads
        base = s * x_ss + hs[:, None].to(tl.int64) * x_sh
        half: tl.constexpr = R // 2
        d = tl.arange(0, RP)
        rmask = d < R
        partner = tl.where(d < half, d + half, d - half)
        m = hmask[:, None] & rmask[None, :]
        xv = tl.load(x + base + d[None, :], mask=m, other=0.0).to(tl.float32)
        pv = tl.load(x + base + partner[None, :], mask=m, other=0.0).to(tl.float32)
        # torch.cat((-second_half, first_half)): negation is exact.
        pv = _mul_rn(pv, tl.where(d < half, -1.0, 1.0)[None, :])
        cv = tl.load(cos + s * c_ss + d, mask=rmask, other=0.0).to(tl.float32)
        sv = tl.load(sin + s * c_ss + d, mask=rmask, other=0.0).to(tl.float32)
        a = _bf16(_mul_rn(xv, cv[None, :])).to(tl.float32)
        b = _bf16(_mul_rn(pv, sv[None, :])).to(tl.float32)
        tl.store(out + base + d[None, :], _bf16(_add_rn(a, b)), mask=m)
        if D - R > 0:
            tail = R + tl.arange(0, TAIL)
            tmask = hmask[:, None] & (tail < D)[None, :]
            tl.store(out + base + tail[None, :], tl.load(x + base + tail[None, :], mask=tmask), mask=tmask)

    @triton.jit
    def _modulate_kernel(n, scale, shift, idx, out, cols, n_rs, sc_rs, sh_rs, o_rs, BLOCK_C: tl.constexpr):
        # out = n * (1 + scale[idx]) + shift[idx], each op rounded to BF16 as eager does.
        row = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        m = c < cols
        r = tl.load(idx + row).to(tl.int64)
        nv = tl.load(n + row * n_rs + c, mask=m).to(tl.float32)
        sc = tl.load(scale + r * sc_rs + c, mask=m).to(tl.float32)
        sh = tl.load(shift + r * sh_rs + c, mask=m).to(tl.float32)
        one_plus = _bf16(_add_rn(sc, tl.full([BLOCK_C], 1.0, tl.float32))).to(tl.float32)
        y = _bf16(_mul_rn(nv, one_plus)).to(tl.float32)
        tl.store(out + row * o_rs + c, _bf16(_add_rn(y, sh)), mask=m)

    @triton.jit
    def _gate_residual_kernel(h, gate, y, idx, out, cols, h_rs, y_rs, g_rs, o_rs, BLOCK_C: tl.constexpr):
        # out = h + gate[idx] * y, each op rounded to BF16 as eager does.
        row = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        m = c < cols
        r = tl.load(idx + row).to(tl.int64)
        hv = tl.load(h + row * h_rs + c, mask=m).to(tl.float32)
        gv = tl.load(gate + r * g_rs + c, mask=m).to(tl.float32)
        yv = tl.load(y + row * y_rs + c, mask=m).to(tl.float32)
        prod = _bf16(_mul_rn(gv, yv)).to(tl.float32)
        tl.store(out + row * o_rs + c, _bf16(_add_rn(hv, prod)), mask=m)

    @triton.jit
    def _swiglu_kernel(x, out, cols, x_rs, o_rs, BLOCK_C: tl.constexpr):
        # Value-first halves: out = value * silu(gate). Eager F.silu computes
        # g / (1 + exp(-g)) in FP32 and stores BF16; the product rounds again.
        row = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        m = c < cols
        v = tl.load(x + row * x_rs + c, mask=m).to(tl.float32)
        g = tl.load(x + row * x_rs + cols + c, mask=m).to(tl.float32)
        e = tl.extra.cuda.libdevice.exp(-g)
        s = _bf16(_div_rn(g, _add_rn(tl.full([BLOCK_C], 1.0, tl.float32), e))).to(tl.float32)
        tl.store(out + row * o_rs + c, _bf16(_mul_rn(v, s)), mask=m)


def _rows(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def _flattens_to_rows(t: torch.Tensor) -> bool:
    if t.dim() <= 2:
        return True
    try:
        t.view(-1, t.shape[-1])
    except RuntimeError:
        return False
    return True


def supports_rowwise(*tensors: torch.Tensor) -> bool:
    """Whether ``modulate``/``gate_residual``/``swiglu`` reproduce eager on these tensors."""
    return (HAVE_TRITON and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
            and all(t.is_cuda and t.dtype == torch.bfloat16 and t.stride(-1) == 1 and _flattens_to_rows(t)
                    for t in tensors))


def supports_rope(hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None) -> bool:
    """Whether ``rope_prefix`` reproduces eager ``_apply_rotary_emb`` on these inputs."""
    if rotary_emb is None or not HAVE_TRITON or torch.is_grad_enabled() or torch.compiler.is_compiling():
        return False
    cos, sin = rotary_emb
    dim = hidden_states.shape[-1]
    return (hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16 and hidden_states.dim() == 4
            and hidden_states.shape[0] == 1 and hidden_states.is_contiguous() and dim & (dim - 1) == 0
            and cos.dim() == 2 and cos.shape == sin.shape and cos.shape[0] == hidden_states.shape[1]
            and cos.shape[-1] % 2 == 0 and cos.shape[-1] <= dim)


def rope_prefix(hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Eager ``MiniMaxH3Attention._apply_rotary_emb`` for contiguous BF16 ``[1, S, H, D]``."""
    cos, sin = rotary_emb
    cos = cos.to(hidden_states.dtype).contiguous()
    sin = sin.to(hidden_states.dtype).contiguous()
    _, seq, heads, dim = hidden_states.shape
    rotary = cos.shape[-1]
    out = torch.empty_like(hidden_states)
    block_h = 4
    _rope_prefix_kernel[(seq, triton.cdiv(heads, block_h))](
        hidden_states,
        cos,
        sin,
        out,
        heads,
        hidden_states.stride(1),
        hidden_states.stride(2),
        cos.stride(0),
        D=dim,
        R=rotary,
        RP=triton.next_power_of_2(rotary),
        TAIL=triton.next_power_of_2(max(dim - rotary, 1)),
        BLOCK_H=block_h,
    )
    return out


def modulate(normed: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """``normed * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)``."""
    n = _rows(normed)
    indices = indices.contiguous()
    out = torch.empty_like(n)
    cols = n.shape[1]
    _modulate_kernel[(n.shape[0], triton.cdiv(cols, _BLOCK_C))](n,
                                                               scale,
                                                               shift,
                                                               indices,
                                                               out,
                                                               cols,
                                                               n.stride(0),
                                                               scale.stride(0),
                                                               shift.stride(0),
                                                               out.stride(0),
                                                               BLOCK_C=_BLOCK_C)
    return out.view(normed.shape)


def gate_residual(hidden: torch.Tensor, gate: torch.Tensor, update: torch.Tensor,
                  indices: torch.Tensor) -> torch.Tensor:
    """``hidden + gate.index_select(0, indices) * update``."""
    h, y = _rows(hidden), _rows(update)
    indices = indices.contiguous()
    out = torch.empty_like(h)
    cols = h.shape[1]
    _gate_residual_kernel[(h.shape[0], triton.cdiv(cols, _BLOCK_C))](h,
                                                                    gate,
                                                                    y,
                                                                    indices,
                                                                    out,
                                                                    cols,
                                                                    h.stride(0),
                                                                    y.stride(0),
                                                                    gate.stride(0),
                                                                    out.stride(0),
                                                                    BLOCK_C=_BLOCK_C)
    return out.view(hidden.shape)


def swiglu(packed: torch.Tensor) -> torch.Tensor:
    """``value, gate = packed.chunk(2, -1); value * F.silu(gate)``."""
    x = _rows(packed)
    cols = x.shape[1] // 2
    out = torch.empty((x.shape[0], cols), device=x.device, dtype=x.dtype)
    _swiglu_kernel[(x.shape[0], triton.cdiv(cols, _BLOCK_C))](x,
                                                             out,
                                                             cols,
                                                             x.stride(0),
                                                             out.stride(0),
                                                             BLOCK_C=_BLOCK_C)
    return out.view(*packed.shape[:-1], cols)


__all__ = [
    "HAVE_TRITON",
    "gate_residual",
    "modulate",
    "rope_prefix",
    "supports_rope",
    "supports_rowwise",
    "swiglu",
]
