# SPDX-License-Identifier: Apache-2.0
"""Fused PAN2 block kernels with backward passes, ported from the PAN2 reference trainer.

The PAN2 reference fuses its memory-bound per-token work into Triton kernels with hand-written backwards
(``pan/models/transformers/modules/triton_kernels.py``). Two of them map onto FastVideo's unpacked layout:

- ``qk_rmsnorm_rope``: the per-head query/key RMSNorm and, for the video stream, the interleaved 3-axis RoPE, in one
  pass over the projection output (the reference's ``qkv_permute_rms`` + ``qkv_rstd_rope`` without its
  concat/interleave permute). The eager path runs about twenty kernels with FP32 temporaries for this.
- ``layer_norm_modulate``: the affine-free LayerNorm and its adaptive ``* (1 + scale) + shift`` (the reference's
  ``packed_modulate`` with ``ln_stats``), computed row-wise in one pass instead of from cached statistics.

Each op is a ``torch.library.custom_op`` with a registered autograd formula, so activation checkpointing and
``torch.compile`` see one node. The forwards reproduce the eager path's roundings (the normalized value and the
weighted value round to the input dtype before the next step, as ``RMSNorm.forward_native`` and ``_apply_rotary_emb``
do), so the fused and eager forwards differ only by FP32 summation order. The backwards keep FP32 throughout, where
eager autograd rounds some intermediate gradients to BF16.

On CUDA the ops launch Triton; elsewhere they run the PyTorch reference of the same arithmetic.
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
# Tests set this to drive CPU tensors through the Triton kernels under ``TRITON_INTERPRET=1``.
_FORCE_TRITON = False


def fusions_available(x: torch.Tensor) -> bool:
    """Whether the fused ops run their Triton kernels for ``x``."""
    return HAVE_TRITON and (x.is_cuda or _FORCE_TRITON)


def _use_triton(x: torch.Tensor) -> bool:
    return fusions_available(x)


# --------------------------------------------------------------------------------------------------------------------
# Query/key RMSNorm + RoPE
# --------------------------------------------------------------------------------------------------------------------

_QK_BLOCK_ROWS = 16

if HAVE_TRITON:

    @triton.jit
    def _qk_rmsnorm_rope_fwd_kernel(X, W, COS, SIN, OUT, RSTD, n_rows, seq_len, num_heads, stride_xr, stride_xh,
                                    stride_or, stride_oh, stride_cs, eps, HEAD_DIM: tl.constexpr,
                                    BLOCK_ROWS: tl.constexpr, HAS_ROPE: tl.constexpr):
        HALF: tl.constexpr = HEAD_DIM // 2
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        head = tl.program_id(1)
        pairs = tl.arange(0, HALF)
        keep = (rows < n_rows)[:, None]
        rows64 = rows.to(tl.int64)
        x_base = X + rows64[:, None] * stride_xr + head * stride_xh
        x_even = tl.load(x_base + 2 * pairs[None, :], mask=keep, other=0.0).to(tl.float32)
        x_odd = tl.load(x_base + 2 * pairs[None, :] + 1, mask=keep, other=0.0).to(tl.float32)
        mean_square = (tl.sum(x_even * x_even, axis=1) + tl.sum(x_odd * x_odd, axis=1)) / HEAD_DIM
        rstd = tl.rsqrt(mean_square + eps)
        # The eager RMSNorm rounds the normalized value, then the weighted value, to the input dtype.
        n_even = (x_even * rstd[:, None]).to(OUT.dtype.element_ty).to(tl.float32)
        n_odd = (x_odd * rstd[:, None]).to(OUT.dtype.element_ty).to(tl.float32)
        w_even = tl.load(W + 2 * pairs).to(tl.float32)
        w_odd = tl.load(W + 2 * pairs + 1).to(tl.float32)
        y_even = (n_even * w_even[None, :]).to(OUT.dtype.element_ty).to(tl.float32)
        y_odd = (n_odd * w_odd[None, :]).to(OUT.dtype.element_ty).to(tl.float32)
        if HAS_ROPE:
            positions = (rows % seq_len).to(tl.int64)
            cs_offsets = positions[:, None] * stride_cs + pairs[None, :]
            cos = tl.load(COS + cs_offsets, mask=keep, other=1.0)
            sin = tl.load(SIN + cs_offsets, mask=keep, other=0.0)
            out_even = y_even * cos - y_odd * sin
            out_odd = y_odd * cos + y_even * sin
        else:
            out_even = y_even
            out_odd = y_odd
        out_base = OUT + rows64[:, None] * stride_or + head * stride_oh
        tl.store(out_base + 2 * pairs[None, :], out_even.to(OUT.dtype.element_ty), mask=keep)
        tl.store(out_base + 2 * pairs[None, :] + 1, out_odd.to(OUT.dtype.element_ty), mask=keep)
        tl.store(RSTD + rows64 * num_heads + head, rstd, mask=rows < n_rows)

    @triton.jit
    def _qk_rmsnorm_rope_bwd_kernel(X, W, COS, SIN, GRAD_OUT, RSTD, GRAD_X, GRAD_W_PARTIAL, n_rows, seq_len, num_heads,
                                    stride_xr, stride_xh, stride_gr, stride_gh, stride_cs, HEAD_DIM: tl.constexpr,
                                    BLOCK_ROWS: tl.constexpr, HAS_ROPE: tl.constexpr):
        HALF: tl.constexpr = HEAD_DIM // 2
        block = tl.program_id(0)
        head = tl.program_id(1)
        rows = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        pairs = tl.arange(0, HALF)
        keep = (rows < n_rows)[:, None]
        rows64 = rows.to(tl.int64)
        x_base = X + rows64[:, None] * stride_xr + head * stride_xh
        x_even = tl.load(x_base + 2 * pairs[None, :], mask=keep, other=0.0).to(tl.float32)
        x_odd = tl.load(x_base + 2 * pairs[None, :] + 1, mask=keep, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + rows64 * num_heads + head, mask=rows < n_rows, other=0.0)
        n_even = x_even * rstd[:, None]
        n_odd = x_odd * rstd[:, None]
        g_base = GRAD_OUT + rows64[:, None] * stride_gr + head * stride_gh
        g_even = tl.load(g_base + 2 * pairs[None, :], mask=keep, other=0.0).to(tl.float32)
        g_odd = tl.load(g_base + 2 * pairs[None, :] + 1, mask=keep, other=0.0).to(tl.float32)
        if HAS_ROPE:
            positions = (rows % seq_len).to(tl.int64)
            cs_offsets = positions[:, None] * stride_cs + pairs[None, :]
            cos = tl.load(COS + cs_offsets, mask=keep, other=1.0)
            sin = tl.load(SIN + cs_offsets, mask=keep, other=0.0)
            # The transpose of each pair's rotation.
            gy_even = g_even * cos + g_odd * sin
            gy_odd = g_odd * cos - g_even * sin
        else:
            gy_even = g_even
            gy_odd = g_odd
        # The weight multiplies the rounded normalized value; its gradient sums over this block's rows.
        rounded_even = n_even.to(GRAD_X.dtype.element_ty).to(tl.float32)
        rounded_odd = n_odd.to(GRAD_X.dtype.element_ty).to(tl.float32)
        partial = GRAD_W_PARTIAL + (block * num_heads + head).to(tl.int64) * HEAD_DIM
        tl.store(partial + 2 * pairs, tl.sum(gy_even * rounded_even, axis=0))
        tl.store(partial + 2 * pairs + 1, tl.sum(gy_odd * rounded_odd, axis=0))
        w_even = tl.load(W + 2 * pairs).to(tl.float32)
        w_odd = tl.load(W + 2 * pairs + 1).to(tl.float32)
        gn_even = gy_even * w_even[None, :]
        gn_odd = gy_odd * w_odd[None, :]
        projection = (tl.sum(gn_even * n_even, axis=1) + tl.sum(gn_odd * n_odd, axis=1)) / HEAD_DIM
        gx_even = rstd[:, None] * (gn_even - n_even * projection[:, None])
        gx_odd = rstd[:, None] * (gn_odd - n_odd * projection[:, None])
        gx_base = GRAD_X + rows64[:, None] * (num_heads * HEAD_DIM) + head * HEAD_DIM
        tl.store(gx_base + 2 * pairs[None, :], gx_even.to(GRAD_X.dtype.element_ty), mask=keep)
        tl.store(gx_base + 2 * pairs[None, :] + 1, gx_odd.to(GRAD_X.dtype.element_ty), mask=keep)


def _rows_view(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """``[B, S, H, D]`` (last dim contiguous) as row/head strides over ``B * S`` rows, copying only when it must."""
    if x.stride(-1) != 1 or x.stride(0) != x.shape[1] * x.stride(1):
        x = x.contiguous()
    return x, x.stride(1), x.stride(2)


def _check_rope(x: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None) -> None:
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin go together")
    if cos is not None and (cos.shape != (x.shape[1], x.shape[-1] // 2) or sin.shape != cos.shape):
        raise ValueError(f"RoPE tables must be [seq_len, head_dim / 2] = [{x.shape[1]}, {x.shape[-1] // 2}], got "
                         f"{tuple(cos.shape)} and {tuple(sin.shape)}")


def _qk_forward_triton(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None,
                       eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, num_heads, head_dim = x.shape
    x, stride_r, stride_h = _rows_view(x)
    out = torch.empty((batch, seq_len, num_heads, head_dim), device=x.device, dtype=x.dtype)
    rstd = torch.empty((batch * seq_len, num_heads), device=x.device, dtype=torch.float32)
    n_rows = batch * seq_len
    has_rope = cos is not None
    tables = (cos.float().contiguous(), sin.float().contiguous()) if has_rope else (rstd, rstd)
    grid = (triton.cdiv(n_rows, _QK_BLOCK_ROWS), num_heads)
    _qk_rmsnorm_rope_fwd_kernel[grid](x, weight.contiguous(), tables[0], tables[1], out, rstd, n_rows, seq_len,
                                      num_heads, stride_r, stride_h, out.stride(1), out.stride(2),
                                      tables[0].stride(0), eps, HEAD_DIM=head_dim, BLOCK_ROWS=_QK_BLOCK_ROWS,
                                      HAS_ROPE=has_rope)
    return out, rstd


def _qk_backward_triton(grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor | None,
                        sin: torch.Tensor | None, rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, num_heads, head_dim = x.shape
    x, stride_r, stride_h = _rows_view(x)
    grad_out, stride_gr, stride_gh = _rows_view(grad_out)
    n_rows = batch * seq_len
    num_blocks = triton.cdiv(n_rows, _QK_BLOCK_ROWS)
    grad_x = torch.empty((batch, seq_len, num_heads, head_dim), device=x.device, dtype=x.dtype)
    partial = torch.empty((num_blocks * num_heads, head_dim), device=x.device, dtype=torch.float32)
    has_rope = cos is not None
    tables = (cos.float().contiguous(), sin.float().contiguous()) if has_rope else (rstd, rstd)
    _qk_rmsnorm_rope_bwd_kernel[(num_blocks, num_heads)](x, weight.contiguous(), tables[0], tables[1], grad_out, rstd,
                                                        grad_x, partial, n_rows, seq_len, num_heads, stride_r,
                                                        stride_h, stride_gr, stride_gh, tables[0].stride(0),
                                                        HEAD_DIM=head_dim, BLOCK_ROWS=_QK_BLOCK_ROWS,
                                                        HAS_ROPE=has_rope)
    return grad_x, partial.sum(0).to(weight.dtype)


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """FP32, the eager kernels' arithmetic, except for FP64 inputs (gradient checks)."""
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _qk_forward_reference(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None,
                          eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The eager arithmetic: ``RMSNorm.forward_native``, ``.to(dtype)``, then ``_apply_rotary_emb`` (GPT-J pairs)."""
    compute = _compute_dtype(x)
    x32 = x.to(compute)
    rstd = torch.rsqrt(x32.pow(2).mean(-1) + eps)
    normalized = (x32 * rstd[..., None]).to(x.dtype)
    y = (normalized.to(compute) * weight.to(compute)).to(x.dtype)
    if cos is not None:
        y_even, y_odd = y[..., ::2].to(compute), y[..., 1::2].to(compute)
        cos, sin = cos.to(compute)[None, :, None, :], sin.to(compute)[None, :, None, :]
        out_even = (y_even * cos - y_odd * sin).to(x.dtype)
        out_odd = (y_odd * cos + y_even * sin).to(x.dtype)
        y = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    return y.contiguous(), rstd.reshape(-1, x.shape[2])


def _qk_backward_reference(grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor | None,
                           sin: torch.Tensor | None, rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 formulas of the Triton backward (rounding steps count as identities, as eager autograd treats casts)."""
    compute = _compute_dtype(x)
    x32 = x.to(compute)
    rstd = rstd.reshape(x.shape[:3])[..., None].to(compute)
    normalized = x32 * rstd
    grad = grad_out.to(compute)
    if cos is not None:
        g_even, g_odd = grad[..., ::2], grad[..., 1::2]
        cos, sin = cos.to(compute)[None, :, None, :], sin.to(compute)[None, :, None, :]
        grad = torch.stack((g_even * cos + g_odd * sin, g_odd * cos - g_even * sin), dim=-1).flatten(-2)
    grad_weight = (grad * normalized.to(x.dtype).to(compute)).sum(dim=(0, 1, 2)).to(weight.dtype)
    grad_normalized = grad * weight.to(compute)
    projection = (grad_normalized * normalized).mean(-1, keepdim=True)
    grad_x = rstd * (grad_normalized - normalized * projection)
    return grad_x.to(x.dtype), grad_weight


@torch.library.custom_op("fastvideo::pan2_qk_rmsnorm_rope", mutates_args=())
def _qk_rmsnorm_rope_op(q: torch.Tensor, k: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
                        cos: torch.Tensor | None, sin: torch.Tensor | None,
                        eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    forward = _qk_forward_triton if _use_triton(q) else _qk_forward_reference
    q_out, q_rstd = forward(q, q_weight, cos, sin, eps)
    k_out, k_rstd = forward(k, k_weight, cos, sin, eps)
    return q_out, k_out, q_rstd, k_rstd


@_qk_rmsnorm_rope_op.register_fake
def _qk_rmsnorm_rope_fake(q, k, q_weight, k_weight, cos, sin, eps):
    del q_weight, k_weight, cos, sin, eps
    q_rows = q.shape[0] * q.shape[1]
    k_rows = k.shape[0] * k.shape[1]
    return (q.new_empty(q.shape), k.new_empty(k.shape), q.new_empty((q_rows, q.shape[2]), dtype=_compute_dtype(q)),
            k.new_empty((k_rows, k.shape[2]), dtype=_compute_dtype(k)))


def _qk_setup_context(ctx, inputs, output) -> None:
    q, k, q_weight, k_weight, cos, sin, _eps = inputs
    _q_out, _k_out, q_rstd, k_rstd = output
    ctx.save_for_backward(q, k, q_weight, k_weight, cos, sin, q_rstd, k_rstd)
    ctx.mark_non_differentiable(q_rstd, k_rstd)


@torch.library.custom_op("fastvideo::pan2_qk_rmsnorm_rope_backward", mutates_args=())
def _qk_rmsnorm_rope_backward_op(grad_q_out: torch.Tensor, grad_k_out: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                                 q_weight: torch.Tensor, k_weight: torch.Tensor, cos: torch.Tensor | None,
                                 sin: torch.Tensor | None, q_rstd: torch.Tensor,
                                 k_rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward as its own op: compiled graphs (AOTAutograd traces the autograd formula) hold one opaque node
    instead of tracing into the Triton launch."""
    backward = _qk_backward_triton if _use_triton(q) else _qk_backward_reference
    grad_q, grad_q_weight = backward(grad_q_out, q, q_weight, cos, sin, q_rstd)
    grad_k, grad_k_weight = backward(grad_k_out, k, k_weight, cos, sin, k_rstd)
    return grad_q, grad_k, grad_q_weight, grad_k_weight


@_qk_rmsnorm_rope_backward_op.register_fake
def _qk_rmsnorm_rope_backward_fake(grad_q_out, grad_k_out, q, k, q_weight, k_weight, cos, sin, q_rstd, k_rstd):
    del grad_q_out, grad_k_out, cos, sin, q_rstd, k_rstd
    return q.new_empty(q.shape), k.new_empty(k.shape), q_weight.new_empty(q_weight.shape), k_weight.new_empty(
        k_weight.shape)


def _qk_backward(ctx, grad_q_out, grad_k_out, _grad_q_rstd, _grad_k_rstd):
    q, k, q_weight, k_weight, cos, sin, q_rstd, k_rstd = ctx.saved_tensors
    grad_q, grad_k, grad_q_weight, grad_k_weight = _qk_rmsnorm_rope_backward_op(grad_q_out, grad_k_out, q, k,
                                                                                q_weight, k_weight, cos, sin, q_rstd,
                                                                                k_rstd)
    return grad_q, grad_k, grad_q_weight, grad_k_weight, None, None, None


_qk_rmsnorm_rope_op.register_autograd(_qk_backward, setup_context=_qk_setup_context)


def qk_rmsnorm_rope(q: torch.Tensor,
                    k: torch.Tensor,
                    q_weight: torch.Tensor,
                    k_weight: torch.Tensor,
                    eps: float,
                    cos: torch.Tensor | None = None,
                    sin: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm of ``q`` and ``k`` (``[B, S, H, D]``, e.g. views of a fused projection), then RoPE.

    ``cos``/``sin`` are the half-dimension tables ``[S, D / 2]`` of ``get_1d_rotary_pos_embed(use_real=False)``,
    applied to the interleaved ``(x[2i], x[2i + 1])`` pairs as ``_apply_rotary_emb(is_neox_style=False)`` does; without
    them only the normalization runs. Returns contiguous ``[B, S, H, D]`` tensors in the input dtype.
    """
    if q.dim() != 4 or k.shape[:2] != q.shape[:2] or q.shape[-1] % 2:
        raise ValueError(f"q and k must be [B, S, H, D] with an even D, got {tuple(q.shape)} and {tuple(k.shape)}")
    _check_rope(q, cos, sin)
    q_out, k_out, _, _ = _qk_rmsnorm_rope_op(q, k, q_weight, k_weight, cos, sin, float(eps))
    return q_out, k_out


# --------------------------------------------------------------------------------------------------------------------
# LayerNorm + adaptive modulation
# --------------------------------------------------------------------------------------------------------------------

_LN_BWD_ROWS_PER_PROGRAM = 16

if HAVE_TRITON:

    @triton.jit
    def _layer_norm_modulate_fwd_kernel(X, SCALE, SHIFT, OUT, MEAN, RSTD, seq_len, n_cols, stride_xr, stride_or,
                                        stride_scale_b, stride_shift_b, eps, BLOCK_C: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        batch = row // seq_len
        cols = tl.arange(0, BLOCK_C)
        valid = cols < n_cols
        x = tl.load(X + row * stride_xr + cols, mask=valid, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / n_cols
        centered = tl.where(valid, x - mean, 0.0)
        rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / n_cols + eps)
        scale = tl.load(SCALE + batch * stride_scale_b + cols, mask=valid, other=0.0).to(tl.float32)
        shift = tl.load(SHIFT + batch * stride_shift_b + cols, mask=valid, other=0.0).to(tl.float32)
        out = centered * rstd * scale + shift
        tl.store(OUT + row * stride_or + cols, out.to(OUT.dtype.element_ty), mask=valid)
        tl.store(MEAN + row, mean)
        tl.store(RSTD + row, rstd)

    @triton.jit
    def _layer_norm_modulate_bwd_kernel(X, SCALE, GRAD_OUT, MEAN, RSTD, GRAD_X, GRAD_SCALE_PARTIAL,
                                        GRAD_SHIFT_PARTIAL, seq_len, n_cols, rows_per_program, num_chunks, stride_xr,
                                        stride_gr, stride_scale_b, BLOCK_C: tl.constexpr):
        chunk = tl.program_id(0)
        batch = tl.program_id(1).to(tl.int64)
        cols = tl.arange(0, BLOCK_C)
        valid = cols < n_cols
        scale = tl.load(SCALE + batch * stride_scale_b + cols, mask=valid, other=0.0).to(tl.float32)
        grad_scale = tl.zeros([BLOCK_C], dtype=tl.float32)
        grad_shift = tl.zeros([BLOCK_C], dtype=tl.float32)
        first = chunk * rows_per_program
        last = tl.minimum(first + rows_per_program, seq_len)
        for offset in range(first, last):
            row = batch * seq_len + offset
            x = tl.load(X + row * stride_xr + cols, mask=valid, other=0.0).to(tl.float32)
            grad = tl.load(GRAD_OUT + row * stride_gr + cols, mask=valid, other=0.0).to(tl.float32)
            normalized = tl.where(valid, (x - tl.load(MEAN + row)) * tl.load(RSTD + row), 0.0)
            grad_scale += grad * normalized
            grad_shift += grad
            grad_normalized = grad * scale
            mean_grad = tl.sum(grad_normalized, axis=0) / n_cols
            mean_projection = tl.sum(grad_normalized * normalized, axis=0) / n_cols
            grad_x = tl.load(RSTD + row) * (grad_normalized - mean_grad - normalized * mean_projection)
            tl.store(GRAD_X + row * n_cols + cols, grad_x.to(GRAD_X.dtype.element_ty), mask=valid)
        partial = (batch * num_chunks + chunk) * n_cols
        tl.store(GRAD_SCALE_PARTIAL + partial + cols, grad_scale, mask=valid)
        tl.store(GRAD_SHIFT_PARTIAL + partial + cols, grad_shift, mask=valid)


def _batch_rows(table: torch.Tensor, batch: int, n_cols: int) -> torch.Tensor:
    """A ``[B, C]`` (or broadcastable ``[C]``) table with a contiguous last dimension."""
    table = table.expand(batch, n_cols)
    return table if table.stride(-1) == 1 else table.contiguous()


def _ln_forward_triton(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                       eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, n_cols = x.shape
    rows = x.reshape(-1, n_cols)
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    out = torch.empty((batch, seq_len, n_cols), device=x.device, dtype=x.dtype)
    mean = torch.empty(batch * seq_len, device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    scale, shift = _batch_rows(scale, batch, n_cols), _batch_rows(shift, batch, n_cols)
    block = triton.next_power_of_2(n_cols)
    _layer_norm_modulate_fwd_kernel[(batch * seq_len, )](rows, scale, shift, out, mean, rstd, seq_len, n_cols,
                                                         rows.stride(0), n_cols, scale.stride(0), shift.stride(0), eps,
                                                         BLOCK_C=block, num_warps=8 if block >= 2048 else 4)
    return out, mean, rstd


def _ln_backward_triton(grad_out: torch.Tensor, x: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor,
                        rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, n_cols = x.shape
    rows = x.reshape(-1, n_cols)
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    grad_rows = grad_out.reshape(-1, n_cols)
    if grad_rows.stride(-1) != 1:
        grad_rows = grad_rows.contiguous()
    scale_rows = _batch_rows(scale, batch, n_cols)
    num_chunks = triton.cdiv(seq_len, _LN_BWD_ROWS_PER_PROGRAM)
    grad_x = torch.empty((batch, seq_len, n_cols), device=x.device, dtype=x.dtype)
    grad_scale = torch.empty((batch, num_chunks, n_cols), device=x.device, dtype=torch.float32)
    grad_shift = torch.empty_like(grad_scale)
    block = triton.next_power_of_2(n_cols)
    _layer_norm_modulate_bwd_kernel[(num_chunks, batch)](rows, scale_rows, grad_rows, mean, rstd, grad_x, grad_scale,
                                                         grad_shift, seq_len, n_cols, _LN_BWD_ROWS_PER_PROGRAM,
                                                         num_chunks, rows.stride(0), grad_rows.stride(0),
                                                         scale_rows.stride(0), BLOCK_C=block,
                                                         num_warps=8 if block >= 2048 else 4)
    return grad_x, grad_scale.sum(1), grad_shift.sum(1)


def _ln_forward_reference(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                          eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The autocast arithmetic: FP32 LayerNorm, ``* (1 + scale) + shift`` in FP32, one rounding at the end."""
    compute = _compute_dtype(x)
    x32 = x.to(compute)
    mean = x32.mean(-1)
    rstd = torch.rsqrt(x32.var(-1, unbiased=False) + eps)
    normalized = (x32 - mean[..., None]) * rstd[..., None]
    batch, _, n_cols = x.shape
    out = (normalized * _batch_rows(scale, batch, n_cols).to(compute)[:, None] +
           _batch_rows(shift, batch, n_cols).to(compute)[:, None])
    return out.to(x.dtype), mean.reshape(-1), rstd.reshape(-1)


def _ln_backward_reference(grad_out: torch.Tensor, x: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor,
                           rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compute = _compute_dtype(x)
    batch, seq_len, n_cols = x.shape
    mean, rstd = mean.reshape(batch, seq_len, 1).to(compute), rstd.reshape(batch, seq_len, 1).to(compute)
    normalized = (x.to(compute) - mean) * rstd
    grad = grad_out.to(compute)
    grad_normalized = grad * _batch_rows(scale, batch, n_cols).to(compute)[:, None]
    grad_x = rstd * (grad_normalized - grad_normalized.mean(-1, keepdim=True) -
                     normalized * (grad_normalized * normalized).mean(-1, keepdim=True))
    return grad_x.to(x.dtype), (grad * normalized).sum(1), grad.sum(1)


@torch.library.custom_op("fastvideo::pan2_layer_norm_modulate", mutates_args=())
def _layer_norm_modulate_op(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                            eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward = _ln_forward_triton if _use_triton(x) else _ln_forward_reference
    return forward(x, scale, shift, eps)


@_layer_norm_modulate_op.register_fake
def _layer_norm_modulate_fake(x, scale, shift, eps):
    del scale, shift, eps
    rows = x.shape[0] * x.shape[1]
    stats = _compute_dtype(x)
    return x.new_empty(x.shape), x.new_empty(rows, dtype=stats), x.new_empty(rows, dtype=stats)


def _ln_setup_context(ctx, inputs, output) -> None:
    x, scale, shift, _eps = inputs
    _out, mean, rstd = output
    ctx.save_for_backward(x, scale, mean, rstd)
    ctx.scale_shape, ctx.shift_shape = scale.shape, shift.shape
    ctx.scale_dtype, ctx.shift_dtype = scale.dtype, shift.dtype
    ctx.mark_non_differentiable(mean, rstd)


@torch.library.custom_op("fastvideo::pan2_layer_norm_modulate_backward", mutates_args=())
def _layer_norm_modulate_backward_op(grad_out: torch.Tensor, x: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor,
                                     rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(grad x, grad scale, grad shift), the tables' gradients per sample ``[B, C]``; an op of its own for the same
    reason as ``_qk_rmsnorm_rope_backward_op``."""
    backward = _ln_backward_triton if _use_triton(x) else _ln_backward_reference
    return backward(grad_out, x, scale, mean, rstd)


@_layer_norm_modulate_backward_op.register_fake
def _layer_norm_modulate_backward_fake(grad_out, x, scale, mean, rstd):
    del grad_out, scale, mean, rstd
    tables = (x.shape[0], x.shape[2])
    stats = _compute_dtype(x)
    return x.new_empty(x.shape), x.new_empty(tables, dtype=stats), x.new_empty(tables, dtype=stats)


def _ln_backward(ctx, grad_out, _grad_mean, _grad_rstd):
    x, scale, mean, rstd = ctx.saved_tensors
    grad_x, grad_scale, grad_shift = _layer_norm_modulate_backward_op(grad_out, x, scale, mean, rstd)
    # Tables shared by the batch ([C]) receive the batch's summed gradient.
    grad_scale = grad_scale.sum_to_size(ctx.scale_shape).to(ctx.scale_dtype)
    grad_shift = grad_shift.sum_to_size(ctx.shift_shape).to(ctx.shift_dtype)
    return grad_x, grad_scale, grad_shift, None


_layer_norm_modulate_op.register_autograd(_ln_backward, setup_context=_ln_setup_context)


def layer_norm_modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
    """``LayerNorm(x) * scale + shift`` for ``x`` ``[B, S, C]``, affine-free LayerNorm, in one pass.

    ``scale`` (PAN2's ``1 + scale``) and ``shift`` are ``[B, C]`` per sample or ``[C]`` shared. The arithmetic is the
    autocast path's: an FP32 LayerNorm and FP32 modulation, rounded once to ``x``'s dtype. Under autocast that is
    exactly the value the next linear layer would have cast its FP32 input to.
    """
    if x.dim() != 3 or scale.shape[-1] != x.shape[-1] or shift.shape[-1] != x.shape[-1]:
        raise ValueError(f"x must be [B, S, C] with [B, C] or [C] tables, got {tuple(x.shape)}, "
                         f"{tuple(scale.shape)} and {tuple(shift.shape)}")
    out, _, _ = _layer_norm_modulate_op(x, scale, shift, float(eps))
    return out



__all__ = [
    "HAVE_TRITON",
    "fusions_available",
    "layer_norm_modulate",
    "qk_rmsnorm_rope",
]
