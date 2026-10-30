# SPDX-License-Identifier: Apache-2.0
"""Fused QK RMSNorm + partial RoPE for the MiniMax-H3 video VAE decoder.

One Triton pass per decoder attention layer replaces the eager prologue of
``MiniMaxH3VideoAttention.forward``:

    query = norm_q(query.float()).to(query.dtype)        # fp32 RMSNorm, no affine
    key = norm_k(key.float()).to(key.dtype)
    q = cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass])
    k = cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass])

i.e. two casts, two fp32 RMSNorm kernels, four ``torch.cat`` copies, two negations
and four mul/add kernels per layer per temporal chunk. The fused kernel reads Q and
K once, normalizes in fp32, rounds to the activation dtype exactly as the eager path
does, applies the rotate-half RoPE with the same per-op rounding, and writes the
result back in place.

The arithmetic is the eager arithmetic (fp32 RMSNorm, RoPE on the activation dtype);
only the fp32 RMSNorm reduction order differs, because ``tl.sum`` reduces the 64
per-head values with a different tree than ``at::native`` layer norm.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    num_heads,
    eps,
    HD: tl.constexpr,
    RD: tl.constexpr,
    HB: tl.constexpr,
    NHG: tl.constexpr,
):
    """Normalize and rope ``HB`` heads of one (batch, token) row of Q and K.

    ``q``/``k`` are contiguous ``[B, S, H, HD]``; ``cos``/``sin`` are contiguous
    ``[B, S, 1, RD]``. ``HB`` heads share one program; ``HD`` and ``RD`` must be
    powers of two and ``RD <= HD``.
    """
    pid = tl.program_id(0)
    h0 = (pid % NHG) * HB
    bt = pid // NHG
    tok = bt % seq_len
    batch = bt // seq_len

    NB: tl.constexpr = HB * HD
    off = tl.arange(0, NB)
    hrow = h0 + off // HD
    mask = hrow < num_heads
    base = bt * (num_heads * HD) + h0 * HD

    idx = tl.arange(0, HD)
    # rotate_half: out[i] = -x[i + RD/2] for i < RD/2, +x[i - RD/2] otherwise.
    pidx = tl.where(idx < RD, tl.where(idx < RD // 2, idx + RD // 2, idx - RD // 2), idx)
    gidx = tl.broadcast_to(pidx[None, :], (HB, HD))

    cb = (batch * seq_len + tok) * RD
    rmask = idx < RD
    cos = tl.load(cos_ptr + cb + idx, mask=rmask, other=1.0)
    sin = tl.load(sin_ptr + cb + idx, mask=rmask, other=0.0)
    dt = q_ptr.dtype.element_ty
    # The eager path casts cos/sin to the activation dtype before the multiply.
    cos = tl.where(rmask, cos.to(dt).to(tl.float32), 1.0)
    sin = tl.where(rmask, sin.to(dt).to(tl.float32), 0.0)
    sin = tl.where(idx < RD // 2, -sin, sin)
    cos2 = tl.broadcast_to(cos[None, :], (HB, HD))
    sin2 = tl.broadcast_to(sin[None, :], (HB, HD))

    for which in tl.static_range(2):
        ptr = q_ptr if which == 0 else k_ptr
        x = tl.load(ptr + base + off, mask=mask, other=0.0).to(tl.float32)
        x = tl.reshape(x, (HB, HD))
        inv = tl.math.rsqrt(tl.sum(x * x, 1) / HD + eps)[:, None]
        y = (x * inv).to(dt).to(tl.float32)
        rotated = tl.gather(y, gidx, 1)
        y = (y * cos2).to(dt).to(tl.float32) + (rotated * sin2).to(dt).to(tl.float32)
        tl.store(ptr + base + off, tl.reshape(y, (NB, )).to(dt), mask=mask)


def _heads_per_program(num_heads: int) -> int:
    """Largest power-of-two head group (<= 8) that divides ``num_heads``."""
    for group in (8, 4, 2):
        if num_heads % group == 0:
            return group
    return 1


def qk_norm_rope_(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float = 1e-5,
) -> None:
    """Fuse QK RMSNorm + partial RoPE in place on contiguous ``[B, S, H, D]`` Q/K."""
    batch, seq_len, heads, _ = query.shape
    group = _heads_per_program(heads)
    _qk_norm_rope_kernel[(batch * seq_len * triton.cdiv(heads, group), )](
        query,
        key,
        cos,
        sin,
        seq_len,
        heads,
        eps,
        HD=head_dim,
        RD=rope_dim,
        HB=group,
        NHG=triton.cdiv(heads, group),
        num_warps=1,
    )
