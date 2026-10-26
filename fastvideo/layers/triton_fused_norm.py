# SPDX-License-Identifier: Apache-2.0
"""Triton-fused residual + LayerNorm + scale/shift for inference.

Collapses the eager chain

    residual_out = residual + x * gate        (or residual + x, or no residual)
    normalized   = FP32LayerNorm(residual_out)
    modulated    = normalized * (1 + scale) + shift        (optional)
    ... followed by the caller's .to(orig_dtype) casts ...

into a single kernel with two outputs, both already in the stream dtype, so the
caller-side casts become no-ops. All arithmetic is fp32 regardless of I/O dtype.
``fused_norm_mod`` is the residual-free form for Wan's pre-attention
``norm1 + modulate`` site. Modulation tables are ``[B|1, S|1, H]`` or the
per-frame ``[B|1, F, 1, H]`` that eager broadcasts through ``unflatten``.

Numerics: the elementwise parts replicate the eager rounding points that type
promotion dictates. A fp32 gate keeps the residual sum in fp32; a bf16 gate or
the scalar 1 materializes a bf16 product and sum; the norm output is rounded to
bf16 where eager's LayerNorm call returns bf16 (FP32LayerNorm casting back to a
bf16 input, or a plain LayerNorm outside autocast); bf16 tables round
``1 + scale`` and, when the norm output is bf16, the product. What remains is
the reduction order inside mean/variance (last-ulp fp32), which the bf16 output
rounding turns into at most one bf16 step on a few elements. The dense Wan
models absorb that (SSIM about 0.98 against eager); a self-forcing causal
rollout amplifies any such perturbation chunk by chunk (SSIM about 0.8 on
SFWan2.1-1.3B, the same as a deliberate one-ulp flip on 0.1% of the
modulated elements), so the causal block opts out (``fused_norm=False``) and
keeps the bit-exact kernels regardless of the environment variable.

Inference-only: callers must gate on ``torch.is_grad_enabled()``; there is no
backward. Enable with FASTVIDEO_ENABLE_FUSED_NORM=1; the default runs the
bit-exact kernels in fastvideo/layers/triton_exact_modulation.py around an
eager LayerNorm instead.
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
    def _mul_rn(a, b):
        # Explicit round-to-nearest PTX: the compiler may not contract a plain
        # multiply and add into one FMA, which would skip the rounding points
        # the eager chain has (and any bf16 round trip between them).
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
    def _round_bf16(v):
        # An eager bf16 intermediate, round-to-nearest-even.
        return v.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

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
        g_sd,
        sh_sb,
        sh_ss,
        sh_sd,
        sc_sb,
        sc_ss,
        sc_sd,
        o_sb,
        o_ss,
        ro_sb,
        ro_ss,
        eps,
        HAS_RESIDUAL: tl.constexpr,
        HAS_GATE_TENSOR: tl.constexpr,
        GATE_ROUND: tl.constexpr,
        INTERMEDIATE_ROUND: tl.constexpr,
        LN_OUT_ROUND: tl.constexpr,
        ROUND_ONE_PLUS: tl.constexpr,
        ROUND_PRODUCT: tl.constexpr,
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
        if HAS_RESIDUAL:
            r = tl.load(RES + b * r_sb + s * r_ss + cols, mask=mask, other=0.0).to(tl.float32)
            if HAS_GATE_TENSOR:
                g = tl.load(GATE + b * g_sb + (s // g_sd) * g_ss + cols, mask=mask, other=0.0).to(tl.float32)
                prod = _mul_rn(x, g)
                if GATE_ROUND:
                    # eager: bf16 * bf16 rounds the product before the sum.
                    prod = _round_bf16(prod)
                acc = _add_rn(r, prod)
            else:
                acc = _add_rn(r, x)

            if OUT_BF16:
                res_bf16 = acc.to(tl.bfloat16, fp_downcast_rounding="rtne")
                tl.store(RES_OUT + b * ro_sb + s * ro_ss + cols, res_bf16, mask=mask)
                if INTERMEDIATE_ROUND:
                    # eager produced a bf16 residual tensor here; the norm consumed
                    # the rounded values.
                    acc = res_bf16.to(tl.float32)
            else:
                tl.store(RES_OUT + b * ro_sb + s * ro_ss + cols, acc, mask=mask)
        else:
            # norm1 site: eager normalizes x.float(), no residual output.
            acc = x

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
            if LN_OUT_ROUND:
                # eager rounded the norm output to bf16 before modulating.
                xhat = _round_bf16(xhat)
            sc = tl.load(SCALE + b * sc_sb + (s // sc_sd) * sc_ss + cols, mask=mask, other=0.0).to(tl.float32)
            sh = tl.load(SHIFT + b * sh_sb + (s // sh_sd) * sh_ss + cols, mask=mask, other=0.0).to(tl.float32)
            one_plus = _add_rn(sc, tl.full([BLOCK_H], 1.0, tl.float32))
            if ROUND_ONE_PLUS:
                # eager: int + bf16 table -> bf16.
                one_plus = _round_bf16(one_plus)
            y = _mul_rn(xhat, one_plus)
            if ROUND_PRODUCT:
                # eager: bf16 norm output * bf16 table -> bf16.
                y = _round_bf16(y)
            y = _add_rn(y, sh)
        else:
            y = xhat

        if OUT_BF16:
            tl.store(OUT + b * o_sb + s * o_ss + cols, y.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)
        else:
            tl.store(OUT + b * o_sb + s * o_ss + cols, y, mask=mask)


def modulation_layout(t: torch.Tensor, batch: int, seq: int, hidden: int) -> tuple[int, int, int] | None:
    """(batch_stride, row_stride, tokens_per_row) for a modulation table, or None if unsupported.

    A ``[B|1, S|1, H]`` table has one row per token (or per batch); a
    ``[B|1, F, 1, H]`` table has one row per frame of ``S // F`` tokens, which is
    what eager broadcasts through ``x.unflatten(1, (F, S // F))``. Token ``s``
    reads row ``s // tokens_per_row``. The last dim must equal ``hidden``
    exactly: the kernels index columns ``0..hidden-1`` of every table, so a
    narrower (which eager would broadcast) or wider last dim would silently read
    the wrong elements instead of raising.
    """
    if t.stride(-1) != 1 or t.shape[-1] != hidden:
        return None
    if t.dim() == 3:
        tb, ts, _ = t.shape
        if tb not in (1, batch) or ts not in (1, seq):
            return None
        return (t.stride(0) if tb == batch else 0, t.stride(1) if ts == seq else 0, 1)
    if t.dim() == 4:
        tb, frames, one, _ = t.shape
        if tb not in (1, batch) or one != 1 or frames < 1 or seq % frames:
            return None
        return (t.stride(0) if tb == batch else 0, t.stride(1), seq // frames)
    return None


def fused_path_supported(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | int,
    shift: torch.Tensor | None,
    scale: torch.Tensor | None,
    norm: torch.nn.Module,
) -> bool:
    """Cheap eligibility check; any False falls back to the eager path."""
    if not _HAS_TRITON or not envs.FASTVIDEO_ENABLE_FUSED_NORM.get():
        return False
    if torch.is_grad_enabled():
        return False
    if torch.compiler.is_compiling():
        # Dynamo must not trace into the Triton launcher (see the custom-op
        # boundaries in fastvideo/models/dits/minimax_h3_fusions/modulation.py).
        # Until this kernel is wrapped in torch.library.custom_op + register_fake
        # the way that module does, compiled forwards take the eager path, so
        # enable_torch_compile keeps its pre-fusion behavior instead of hitting
        # an unregistered launch inside the compiled graph.
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
        if modulation_layout(gate, batch, seq, hidden) is None:
            return False
    elif gate != 1:
        return False
    if (shift is None) != (scale is None):
        return False
    if shift is not None and scale is not None:
        return _tables_supported(shift, scale, residual, batch, seq, hidden)
    return True


def _tables_supported(shift: torch.Tensor, scale: torch.Tensor, stream: torch.Tensor, batch: int, seq: int,
                      hidden: int) -> bool:
    for t in (shift, scale):
        if t.dtype not in (torch.float32, stream.dtype) or not t.is_cuda:
            return False
        if modulation_layout(t, batch, seq, hidden) is None:
            return False
    return True


def fused_norm_mod_supported(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, norm: torch.nn.Module) -> bool:
    """Eligibility of ``fused_norm_mod`` (the residual-free norm1 + modulate site)."""
    if not _HAS_TRITON or not envs.FASTVIDEO_ENABLE_FUSED_NORM.get():
        return False
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        return False
    if not isinstance(norm, torch.nn.LayerNorm):
        return False
    if x.dim() != 3 or x.dtype not in _SUPPORTED_STREAM_DTYPES or not x.is_cuda or not x.is_contiguous():
        return False
    batch, seq, hidden = x.shape
    if hidden > _MAX_HIDDEN or norm.normalized_shape != (hidden, ):
        return False
    return _tables_supported(shift, scale, x, batch, seq, hidden)


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
    stream_bf16 = stream_dtype == torch.bfloat16
    sum_is_bf16 = stream_bf16 and (not gate_is_tensor or gate.dtype == torch.bfloat16)
    gate_round = gate_is_tensor and stream_bf16 and gate.dtype == torch.bfloat16
    ln_out_bf16 = sum_is_bf16 and _eager_norm_keeps_bf16(norm)
    table_bf16 = has_mod and shift.dtype == torch.bfloat16  # type: ignore[union-attr]

    if gate_is_tensor:
        g_sb, g_ss, g_sd = modulation_layout(gate, batch, seq, hidden)  # type: ignore[misc]
        gate_arg = gate
    else:
        g_sb = g_ss = 0
        g_sd = 1
        gate_arg = residual  # unused placeholder pointer
    if has_mod:
        sh_sb, sh_ss, sh_sd = modulation_layout(shift, batch, seq, hidden)  # type: ignore[arg-type,misc]
        sc_sb, sc_ss, sc_sd = modulation_layout(scale, batch, seq, hidden)  # type: ignore[arg-type,misc]
        shift_arg, scale_arg = shift, scale
    else:
        sh_sb = sh_ss = sc_sb = sc_ss = 0
        sh_sd = sc_sd = 1
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
        g_sd,
        sh_sb,
        sh_ss,
        sh_sd,
        sc_sb,
        sc_ss,
        sc_sd,
        out.stride(0),
        out.stride(1),
        res_out.stride(0),
        res_out.stride(1),
        norm.eps,
        HAS_RESIDUAL=True,
        HAS_GATE_TENSOR=gate_is_tensor,
        GATE_ROUND=gate_round,
        INTERMEDIATE_ROUND=sum_is_bf16,
        LN_OUT_ROUND=has_mod and ln_out_bf16,
        ROUND_ONE_PLUS=table_bf16,
        ROUND_PRODUCT=table_bf16 and ln_out_bf16,
        HAS_AFFINE=has_affine,
        HAS_BIAS=has_bias,
        HAS_MOD=has_mod,
        OUT_BF16=stream_dtype == torch.bfloat16,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
    return out, res_out


def _eager_norm_keeps_bf16(norm: torch.nn.Module) -> bool:
    """Whether eager's LayerNorm call returns bf16 for a bf16 input.

    FP32LayerNorm casts back to its input dtype; a plain nn.LayerNorm returns
    the input dtype unless autocast upcasts the op to fp32.
    """
    from fastvideo.layers.layernorm import FP32LayerNorm
    return isinstance(norm, FP32LayerNorm) or not torch.is_autocast_enabled("cuda")


def fused_norm_mod(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, norm: torch.nn.Module, *,
                   ln_input_fp32: bool) -> torch.Tensor:
    """``LayerNorm(x.float()) * (1 + scale) + shift`` in the stream dtype, one launch; no residual.

    Caller must have checked ``fused_norm_mod_supported``. ``ln_input_fp32``
    says the caller normalizes ``x.float()`` (the dense block), so the norm
    output is fp32 in eager; otherwise it is bf16 for a bf16 ``x`` unless
    autocast upcasts it. Up to the LayerNorm reduction order the kernel
    reproduces that chain's rounding points.
    """
    batch, seq, hidden = x.shape
    stream_dtype = x.dtype
    ln_out_bf16 = stream_dtype == torch.bfloat16 and not ln_input_fp32 and _eager_norm_keeps_bf16(norm)
    table_bf16 = shift.dtype == torch.bfloat16
    out = torch.empty_like(x)
    sh_sb, sh_ss, sh_sd = modulation_layout(shift, batch, seq, hidden)  # type: ignore[misc]
    sc_sb, sc_ss, sc_sd = modulation_layout(scale, batch, seq, hidden)  # type: ignore[misc]
    weight = norm.weight
    bias = norm.bias
    block_h = triton.next_power_of_2(hidden)
    num_warps = 4 if block_h <= 2048 else 8
    _fused_residual_norm_mod_kernel[(batch * seq, )](
        x,
        x,  # unused residual pointer
        x,  # unused gate pointer
        weight if weight is not None else x,
        bias if bias is not None else x,
        shift,
        scale,
        out,
        out,  # unused residual-out pointer
        seq,
        hidden,
        x.stride(0),
        x.stride(1),
        0,
        0,
        0,
        0,
        1,
        sh_sb,
        sh_ss,
        sh_sd,
        sc_sb,
        sc_ss,
        sc_sd,
        out.stride(0),
        out.stride(1),
        0,
        0,
        norm.eps,
        HAS_RESIDUAL=False,
        HAS_GATE_TENSOR=False,
        GATE_ROUND=False,
        INTERMEDIATE_ROUND=False,
        LN_OUT_ROUND=ln_out_bf16,
        ROUND_ONE_PLUS=table_bf16,
        ROUND_PRODUCT=table_bf16 and ln_out_bf16,
        HAS_AFFINE=weight is not None,
        HAS_BIAS=bias is not None,
        HAS_MOD=True,
        OUT_BF16=stream_dtype == torch.bfloat16,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
    return out
