# SPDX-License-Identifier: Apache-2.0
"""Experimental sm89 tile-64 VSA with BF16 or INT8 QK and FP32 accumulation.

Retains each query tile's original key selection and masks partial key tiles.
Unlike a 128-query adapter, it adds no attention blocks. Q/K use per-token
scales; K centering is a softmax-invariant shift. V uses one scale per head
and channel, so its dequantization can be applied once in the epilogue.
BF16 PV is the default: FP8 PV had excessive error on real H3 inputs.
Numerical validation and same-seed clip review are required before enabling.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_qk(X, Mean, VBS, Y, Scale, L: tl.constexpr, D: tl.constexpr, H: tl.constexpr, XB: tl.constexpr,
                 XH: tl.constexpr, XS: tl.constexpr, XD: tl.constexpr, CENTER: tl.constexpr, ROWS: tl.constexpr):
    hz = tl.program_id(1)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    offset = (hz // H) * XB + (hz % H) * XH + rows[:, None] * XS + cols[None, :] * XD
    x = tl.load(X + offset, rows[:, None] < L, 0).to(tl.float32)
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


@triton.jit
def _quantize_v_tiles(X, Y, Scale, L: tl.constexpr, D: tl.constexpr):
    tile, hz = tl.program_id(0), tl.program_id(1)
    rows = tile * 64 + tl.arange(0, 64)
    cols = tl.arange(0, D)
    x = tl.load(X + (hz * L + rows[:, None]) * D + cols[None, :]).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0) / 448.0, 1e-8)
    tl.store(Scale + (hz * (L // 64) + tile) * D + cols, scale)
    tl.store(Y + (hz * L + rows[:, None]) * D + cols[None, :], (x / scale[None, :]).to(tl.float8e4nv))


@triton.autotune(
    configs=[triton.Config({}, num_warps=w, num_stages=s) for w, s in ((4, 2), (4, 3), (4, 4), (8, 2), (8, 3))],
    key=["L", "D"])
@triton.jit
def _sparse_int8_fp8(Q, K, V, QS, KS, VS, Index, Count, VBS, Out, L: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                     VB: tl.constexpr, VH: tl.constexpr, VS_ROW: tl.constexpr, VD: tl.constexpr, INT8_QK: tl.constexpr,
                     FP8_PV: tl.constexpr, V_TILE: tl.constexpr, P_DYNAMIC: tl.constexpr):
    tile, hz = tl.program_id(0), tl.program_id(1)
    nt: tl.constexpr = L // 64
    rows = tile * 64 + tl.arange(0, 64)
    cols = tl.arange(0, D)
    q = tl.load(Q + (hz * L + rows[:, None]) * D + cols[None, :])
    if INT8_QK:
        qs = tl.load(QS + hz * L + rows)
    nblocks = tl.load(Count + hz * nt + tile)
    m = tl.full((64, ), -float("inf"), tl.float32)
    den = tl.zeros((64, ), tl.float32)
    acc = tl.zeros((64, D), tl.float32)
    for block in range(nblocks):
        kv = tl.load(Index + (hz * nt + tile) * nt + block)
        key_rows = kv * 64 + tl.arange(0, 64)
        k = tl.load(K + (hz * L + key_rows[None, :]) * D + cols[:, None])
        if INT8_QK:
            ks = tl.load(KS + hz * L + key_rows)
        valid = tl.load(VBS + kv)
        if valid > 0:
            logits = tl.dot(q, k).to(tl.float32)
            if INT8_QK:
                logits = logits * qs[:, None] * ks[None, :]
            logits = logits * (1.4426950408889634 / D**0.5)
            logits = tl.where((tl.arange(0, 64) < valid)[None, :], logits, -float("inf"))
            block_max = tl.max(logits, 1)
            new_m = tl.maximum(m, block_max)
            p = tl.exp2(logits - new_m[:, None])
            alpha = tl.exp2(m - new_m)
            den = den * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(V + (hz // H) * VB + (hz % H) * VH + key_rows[:, None] * VS_ROW + cols[None, :] * VD)
            if FP8_PV:
                if P_DYNAMIC:
                    pscale = tl.maximum(tl.exp2(block_max - new_m) / 448.0, 1e-30)
                    pv = tl.dot((p / pscale[:, None]).to(tl.float8e4nv), v, out_dtype=tl.float32)
                    pv = pv * (pscale[:, None] * 448.0)
                else:
                    pv = tl.dot((p * 448.0).to(tl.float8e4nv), v, out_dtype=tl.float32)
                if V_TILE:
                    scale = tl.load(VS + (hz * nt + kv) * D + cols)
                    acc += pv * (scale[None, :] / 448.0)
                else:
                    acc += pv
            else:
                acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
            m = new_m
    result = acc / den[:, None]
    if FP8_PV and not V_TILE:
        vs = tl.load(VS + hz * D + cols)
        result = result * (vs[None, :] / 448.0)
    result = tl.where(den[:, None] > 0, result, 0.0)
    tl.store(Out + (hz * L + rows[:, None]) * D + cols[None, :], result.to(Out.dtype.element_ty))


def sparse_sm89_attention(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          mask: torch.Tensor,
                          vbs: torch.Tensor,
                          *,
                          int8_qk: bool = True,
                          fp8_pv: bool = False,
                          fp8_v_tiles: bool = False,
                          fp8_dynamic_p: bool = False) -> torch.Tensor:
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

    # The production INT8-QK/BF16-PV route reads BSHD-backed views directly.
    # Quantized Q/K and the output remain contiguous BHSD. Other ablations
    # retain their established layout and arithmetic.
    if not int8_qk or fp8_pv:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    vbs = vbs.to(device=q.device, dtype=torch.int32).contiguous()
    grid = (triton.cdiv(length, 16), b * h)
    qi, ki, vf = q, k, v
    qs, ks, vs = q, k, v  # unused pointers in BF16 ablations
    if int8_qk:
        # Tile pads are zero by contract; avoid a full FP32 copy for the reduction.
        # Preserve the exact reduction used by the old contiguous adapter;
        # its temporary copy dies before Q/K quantization and fine attention.
        mean = k.contiguous().sum(dim=2, dtype=torch.float32) / vbs.sum().clamp_min(1)
        qi = torch.empty(q.shape, device=q.device, dtype=torch.int8)
        ki = torch.empty(k.shape, device=k.device, dtype=torch.int8)
        qs = torch.empty((b, h, length), device=q.device, dtype=torch.float32)
        ks = torch.empty_like(qs)
        _quantize_qk[grid](q, mean, vbs, qi, qs, length, dim, h, *q.stride(), CENTER=False, ROWS=16, num_warps=4)
        _quantize_qk[grid](k, mean, vbs, ki, ks, length, dim, h, *k.stride(), CENTER=True, ROWS=16, num_warps=4)
    if fp8_pv:
        vf = torch.empty_like(v, dtype=torch.float8_e4m3fn)
        if fp8_v_tiles:
            vs = torch.empty((b, h, length // 64, dim), device=q.device, dtype=torch.float32)
            _quantize_v_tiles[(length // 64, b * h)](v, vf, vs, length, dim, num_warps=8)
        else:
            vs = (v.abs().amax(dim=2).float() / 448).clamp_min(1e-8)
            _quantize_v[grid](v, vs, vf, length, dim, ROWS=16, num_warps=4)
    index, count = map_to_index(mask.contiguous())
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _sparse_int8_fp8[(length // 64, b * h)](qi,
                                            ki,
                                            vf,
                                            qs,
                                            ks,
                                            vs,
                                            index,
                                            count,
                                            vbs,
                                            out,
                                            length,
                                            dim,
                                            h,
                                            *vf.stride(),
                                            INT8_QK=int8_qk,
                                            FP8_PV=fp8_pv,
                                            V_TILE=fp8_v_tiles,
                                            P_DYNAMIC=fp8_dynamic_p)
    return out
