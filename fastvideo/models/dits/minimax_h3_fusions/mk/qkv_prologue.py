# SPDX-License-Identifier: Apache-2.0
"""MotionKernel H3 kernel: Fused H3 QKV prologue: per-head QK RMSNorm + partial RoPE + stacked QKV output.

Found with MotionKernel (agent-driven kernel search). Tuned on GB200 / sm_100 for the
MiniMax-H3 inference shapes; opt-in through ``FASTVIDEO_MINIMAX_H3_FUSIONS=mk``.

Fused QKV prologue: per-head QK RMSNorm + partial RoPE on Q and K plus a straight V copy.

Production runs two Sol-Engine ``_qknorm_partial_rope_kernel`` launches (one Triton program per
128-wide (token, head) row of Q and of K) plus ``torch.cat([q, k, v])``. Here a single launch
produces Q, K and V together, so every byte is read and written exactly once.

Each program owns ``HB`` consecutive head rows of one token, i.e. a *contiguous flat* chunk of
``HB * head_dim`` elements. Because the tile is flat and contiguous, each tensor costs exactly
one wide (128-bit) load and one wide store per thread instead of a pile of per-sub-block
addressing. The grid is ordered so that the head-group index is the *fastest* varying dimension:
consecutive CTAs then cover one token's whole ``H * D`` slab in address order, which turns the
six DRAM streams (q, k, v in; q, k, v out) into pure sequential scans (~7.1 vs ~6.6 TB/s).

All three input tiles are issued as loads *before* any store (q, k and v in flight together,
then the stores), so the six streams overlap instead of being serialised by the store that
follows each load. With ``HB=4`` (512 elements, one warp) this reaches the measured streaming
roofline of this GPU: 7.08 TB/s versus 7.10 TB/s for a pure three-stream copy of the same bytes
(the previous load/store/load/store schedule measured 7.00 TB/s).

The rotate-half partner ``x[i +/- rotary_dim/2]`` is materialised with an in-register
``tl.gather`` along the head dim (warp-shuffle lowering for a single warp), so the whole row is
loaded once: no gather loads from memory, no lane padding. All arithmetic is fp32 with one
rounding at the bf16 store; V is copied verbatim.
"""

import torch
import triton
import triton.language as tl

H3_NORM_EPS = 1e-5


@triton.jit
def _qkv_norm_rope_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qw_ptr,
    kw_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    num_heads,
    seq_len,
    plane,  # elements per output plane = B*S*H*D
    eps,
    HD: tl.constexpr,
    RD: tl.constexpr,
    HB: tl.constexpr,
    NHG: tl.constexpr,
    I64: tl.constexpr,
):
    # 1-D grid: gridDim.y is capped at 65535, below the 73k-token shape.
    pid = tl.program_id(0)
    h0 = (pid % NHG) * HB
    bt = pid // NHG
    tok = bt % seq_len
    batch = bt // seq_len

    NB: tl.constexpr = HB * HD
    off = tl.arange(0, NB)
    hrow = h0 + off // HD
    mask = hrow < num_heads

    if I64:
        base = (batch.to(tl.int64) * seq_len + tok) * num_heads * HD + h0 * HD
    else:
        base = (batch * seq_len + tok) * num_heads * HD + h0 * HD

    idx = tl.arange(0, HD)
    pidx = tl.where(idx < RD, tl.where(idx < RD // 2, idx + RD // 2, idx - RD // 2), idx)
    gidx = tl.broadcast_to(pidx[None, :], (HB, HD))
    c = tl.load(cos_ptr + tok * RD + idx, mask=idx < RD, other=1.0).to(tl.float32)[None, :]
    s = tl.load(sin_ptr + tok * RD + idx, mask=idx < RD, other=0.0).to(tl.float32)
    s = tl.where(idx < RD // 2, -s, s)[None, :]

    wq = tl.load(qw_ptr + idx).to(tl.float32)[None, :]
    wk = tl.load(kw_ptr + idx).to(tl.float32)[None, :]

    xq = tl.load(q_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    xk = tl.load(k_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    xv = tl.load(v_ptr + base + off, mask=mask, other=0.0)

    tl.store(out_ptr + 2 * plane + base + off, xv, mask=mask)

    y = tl.reshape(xq, (HB, HD))
    inv = tl.math.rsqrt(tl.sum(y * y, 1) / HD + eps)[:, None]
    y = y * inv * wq
    p = tl.gather(y, gidx, 1)
    y = y * c + p * s
    tl.store(out_ptr + base + off, tl.reshape(y, (NB, )).to(out_ptr.dtype.element_ty), mask=mask)

    y = tl.reshape(xk, (HB, HD))
    inv = tl.math.rsqrt(tl.sum(y * y, 1) / HD + eps)[:, None]
    y = y * inv * wk
    p = tl.gather(y, gidx, 1)
    y = y * c + p * s
    tl.store(out_ptr + plane + base + off, tl.reshape(y, (NB, )).to(out_ptr.dtype.element_ty), mask=mask)


def _pick_config(num_heads: int):
    """(HB, num_warps) -- consecutive head rows per program.

    One warp per program with 4 rows gives 16 elements (two 128-bit accesses) per thread per
    tensor, the measured streaming optimum for this GPU: every row stays inside one warp so the
    RMSNorm reduction and the rotate-half gather are pure warp shuffles, and 32 such CTAs per SM
    keep enough loads in flight to run at the copy roofline.
    """
    if num_heads >= 4:
        return 4, 1
    hb = max(triton.next_power_of_2(max(num_heads, 1)), 1)
    return hb, 1


def _launch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    b, s, h, d = query.shape
    out = torch.empty((3 * b, s, h, d), device=query.device, dtype=query.dtype)
    hb, warps = _pick_config(h)
    nhg = triton.cdiv(h, hb)
    _qkv_norm_rope_kernel[(b * s * nhg, )](
        query,
        key,
        value,
        q_weight,
        k_weight,
        cos,
        sin,
        out,
        h,
        s,
        b * s * h * d,
        H3_NORM_EPS,
        HD=d,
        RD=cos.shape[1],
        HB=hb,
        NHG=nhg,
        I64=out.numel() >= 2**31,
        num_warps=warps,
        num_stages=1,
    )
    return out


def qkv_prologue(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Returns the [3*B, S, H, D] stack (rotated Q, rotated K, V), as FastVideo's attention path builds it."""
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query/key/value must share a [B, S, H, D] shape")
    if cos.shape != sin.shape or cos.shape[0] != query.shape[1]:
        raise ValueError("cos/sin must be [S, rotary_dim] tables")
    if not query.is_contiguous():
        query = query.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    if not q_weight.is_contiguous():
        q_weight = q_weight.contiguous()
    if not k_weight.is_contiguous():
        k_weight = k_weight.contiguous()
    if cos.dtype != query.dtype:
        cos = cos.to(query.dtype)
    if sin.dtype != query.dtype:
        sin = sin.to(query.dtype)
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()
    return _launch(query, key, value, q_weight, k_weight, cos, sin)
