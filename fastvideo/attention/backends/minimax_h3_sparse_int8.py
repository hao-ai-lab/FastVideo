# SPDX-License-Identifier: Apache-2.0
"""Experimental sm89 tile-64 VSA: INT8 QK and FP8 PV with FP32 accumulation.

Retains each query tile's original key selection and masks partial key tiles.
Unlike a 128-query adapter, it adds no attention blocks. Q/K use per-token
scales; K centering is a softmax-invariant shift. V uses one scale per head
and channel, so its dequantization can be applied once in the epilogue.
Numerical validation and same-seed clip review are required before enabling.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_qk(X, Mean, VBS, Y, Scale, L: tl.constexpr, D: tl.constexpr, CENTER: tl.constexpr, ROWS: tl.constexpr):
    hz = tl.program_id(1)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    x = tl.load(X + (hz * L + rows[:, None]) * D + cols[None, :], rows[:, None] < L, 0).to(tl.float32)
    if CENTER:
        mean = tl.load(Mean + hz * D + cols)
        valid_size = tl.load(VBS + rows // 64, rows < L, 0)
        x = tl.where((rows % 64 < valid_size)[:, None], x - mean[None, :], 0.0)
    scale = tl.maximum(tl.max(tl.abs(x), 1) / 127.0, 1e-8)
    y = tl.floor(x / scale[:, None] + 0.5).to(tl.int8)
    tl.store(Y + (hz * L + rows[:, None]) * D + cols[None, :], y, rows[:, None] < L)
    tl.store(Scale + hz * L + rows, scale, rows < L)


@triton.jit
def _quantize_v(X, Scale, Y, L: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr):
    hz = tl.program_id(1)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    scale = tl.load(Scale + hz * D + cols)
    x = tl.load(X + (hz * L + rows[:, None]) * D + cols[None, :], rows[:, None] < L, 0).to(tl.float32)
    tl.store(Y + (hz * L + rows[:, None]) * D + cols[None, :], (x / scale[None, :]).to(tl.float8e4nv), rows[:, None]
             < L)


@triton.autotune(
    configs=[triton.Config({}, num_warps=w, num_stages=s) for w, s in ((4, 2), (4, 3), (4, 4), (8, 2), (8, 3))],
    key=["L", "D"])
@triton.jit
def _sparse_int8_fp8(Q, K, V, QS, KS, VS, Index, Count, VBS, Out, L: tl.constexpr, D: tl.constexpr):
    tile, hz = tl.program_id(0), tl.program_id(1)
    nt: tl.constexpr = L // 64
    rows = tile * 64 + tl.arange(0, 64)
    cols = tl.arange(0, D)
    q = tl.load(Q + (hz * L + rows[:, None]) * D + cols[None, :])
    qs = tl.load(QS + hz * L + rows)
    nblocks = tl.load(Count + hz * nt + tile)
    m = tl.full((64, ), -float("inf"), tl.float32)
    den = tl.zeros((64, ), tl.float32)
    acc = tl.zeros((64, D), tl.float32)
    for block in range(nblocks):
        kv = tl.load(Index + (hz * nt + tile) * nt + block)
        key_rows = kv * 64 + tl.arange(0, 64)
        k = tl.load(K + (hz * L + key_rows[None, :]) * D + cols[:, None])
        ks = tl.load(KS + hz * L + key_rows)
        valid = tl.load(VBS + kv)
        if valid > 0:
            logits = tl.dot(q, k).to(tl.float32) * qs[:, None] * ks[None, :] * (1.4426950408889634 / D**0.5)
            logits = tl.where((tl.arange(0, 64) < valid)[None, :], logits, -float("inf"))
            new_m = tl.maximum(m, tl.max(logits, 1))
            p = tl.exp2(logits - new_m[:, None])
            alpha = tl.exp2(m - new_m)
            den = den * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(V + (hz * L + key_rows[:, None]) * D + cols[None, :])
            acc += tl.dot((p * 448.0).to(tl.float8e4nv), v, out_dtype=tl.float32)
            m = new_m
    vs = tl.load(VS + hz * D + cols)
    result = acc / den[:, None] * (vs[None, :] / 448.0)
    result = tl.where(den[:, None] > 0, result, 0.0)
    tl.store(Out + (hz * L + rows[:, None]) * D + cols[None, :], result.to(Out.dtype.element_ty))


def sparse_int8_fp8_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
                              vbs: torch.Tensor) -> torch.Tensor:
    """Forward-only ``[B,H,S,128]`` BF16 attention on sm89, with 64-token tiles."""
    if torch.is_grad_enabled():
        raise ValueError("Sparse INT8/FP8 attention is inference-only")
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (8, 9):
        raise ValueError("Sparse INT8/FP8 attention requires sm89 CUDA")
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Sparse INT8/FP8 attention requires matching BF16 Q/K/V with head dimension 128")
    b, h, length, dim = q.shape
    if length != vbs.numel() * 64 or mask.shape != (b, h, length // 64, length // 64):
        raise ValueError("Sparse INT8/FP8 attention requires a tile-64 mask and validity vector")
    from fastvideo_kernel.triton_kernels.index import map_to_index

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    vbs = vbs.to(device=q.device, dtype=torch.int32).contiguous()
    # Tile pads are zero by the VSA contract. Avoid a full FP32 copy for the reduction.
    mean = k.sum(dim=2, dtype=torch.float32) / vbs.sum().clamp_min(1)
    qi, ki = torch.empty_like(q, dtype=torch.int8), torch.empty_like(k, dtype=torch.int8)
    qs = torch.empty((b, h, length), device=q.device, dtype=torch.float32)
    ks = torch.empty_like(qs)
    grid = (triton.cdiv(length, 16), b * h)
    _quantize_qk[grid](q, mean, vbs, qi, qs, length, dim, CENTER=False, ROWS=16, num_warps=4)
    _quantize_qk[grid](k, mean, vbs, ki, ks, length, dim, CENTER=True, ROWS=16, num_warps=4)
    vs = (v.abs().amax(dim=2).float() / 448).clamp_min(1e-8)
    vf = torch.empty_like(v, dtype=torch.float8_e4m3fn)
    _quantize_v[grid](v, vs, vf, length, dim, ROWS=16, num_warps=4)
    index, count = map_to_index(mask.contiguous())
    out = torch.empty_like(q)
    _sparse_int8_fp8[(length // 64, b * h)](qi, ki, vf, qs, ks, vs, index, count, vbs, out, length, dim)
    return out
