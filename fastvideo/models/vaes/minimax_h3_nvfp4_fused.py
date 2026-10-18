# SPDX-License-Identifier: Apache-2.0
"""Bit-exact fused inference kernels for the NVFP4 MiniMax-H3 decoder blocks.

The eager NVFP4 block (``NVFP4DecoderLinear`` inside ``MiniMaxH3VideoTransformerBlock``)
spends most of its time in small elementwise kernels around the FP4 GEMMs: bias adds,
RMSNorms, casts, RoPE, SiLU, residuals and three separate quantizations of the shared
q/k/v input. These Triton kernels fuse them while reproducing every rounding step of
the eager path, so the decoder output is bit-identical:

* ``rms_norm`` follows ATen's ``vectorized_layer_norm_kernel`` reduction order
  (128 threads, 4-wide vectors, sequential FMA per thread, warp shuffle-down tree,
  then a 4-warp tree), so the fp32 statistics match bit for bit;
* NVFP4 quantization follows FlashInfer's ``cvt_warp_fp16_to_fp4`` (approximate
  reciprocals, E4M3 block scales, ``layout_128x4`` swizzle, zeroed padding rows);
* every bf16 rounding of the op-by-op eager path is kept, and products that eager
  rounds before an add use ``mul.rn`` so they are never contracted into an FMA.

``fastvideo/tests/vaes/test_minimax_h3_nvfp4_decoder.py`` checks each kernel and whole
decodes against the eager path with ``torch.equal``.
"""
from __future__ import annotations

import functools
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import fastvideo.envs as envs
from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import NVFP4DecoderLinear, NVFP4FusedState

NVFP4_BLOCK = 16
SF_ROW_TILE = 128
# ATen's vectorized layer/RMS norm: 4 warps x 32 lanes, 4 elements per vector load.
_NORM_THREADS = 128
_NORM_VEC = 4


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;",
                                     "=r,r", [x],
                                     dtype=tl.float32,
                                     is_pure=True,
                                     pack=1)


@triton.jit
def _rsqrt_approx(x):
    return tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _mul_rn(a, b):
    # A rounded product that the compiler may not contract into an FMA with a following add.
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b],
                                     dtype=tl.float32,
                                     is_pure=True,
                                     pack=1)


@triton.jit
def _e4m3_bits(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, $1, $1; cvt.u32.u16 $0, t; }",
        "=r,r", [x],
        dtype=tl.int32,
        is_pure=True,
        pack=1) & 0xFF


@triton.jit
def _e4m3_to_f32(bits):
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
    return tl.where(exponent == 0, mantissa.to(tl.float32) * 0.001953125, normal)


@triton.jit
def _e2m1x2(lo, hi):
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t; }",
        "=r,r,r", [lo, hi],
        dtype=tl.int32,
        is_pure=True,
        pack=1)


@triton.jit
def _bf16(x):
    """Round fp32 to the nearest-even bf16 value (kept in fp32).

    Inline PTX, because the compiler may fold a float ``truncf``/``extf`` round trip into a
    following add and skip the rounding that eager applies.
    """
    return tl.inline_asm_elementwise("{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
                                     "=r,r", [x],
                                     dtype=tl.float32,
                                     is_pure=True,
                                     pack=1)


@triton.jit
def _quantize_tile(x, gsf, R: tl.constexpr, N: tl.constexpr):
    """NVFP4-quantize ``x`` [R, N] (fp32 holding bf16 values) like FlashInfer; return (packed, sf bits)."""
    blocks = tl.reshape(x, (R, N // 16, 16))
    vec_max = tl.max(tl.abs(blocks), axis=2)
    sf_value = gsf * (vec_max * _rcp_approx_ftz(tl.full(vec_max.shape, 6.0, tl.float32)))
    sf_bits = _e4m3_bits(sf_value)
    out_scale = _rcp_approx_ftz(_e4m3_to_f32(sf_bits) * _rcp_approx_ftz(gsf + tl.zeros(vec_max.shape, tl.float32)))
    out_scale = tl.where(vec_max != 0, out_scale, 0.0)
    scaled = tl.reshape(blocks * out_scale[:, :, None], (R, N // 2, 2))
    lo, hi = tl.split(scaled)
    return _e2m1x2(lo, hi).to(tl.uint8), sf_bits.to(tl.uint8)


@triton.jit
def _sf_offsets(rows, sf_cols, NUM_SF_COLS: tl.constexpr):
    """``layout_128x4`` swizzled offsets of block scales (rows [R, 1], sf_cols [1, C])."""
    k_tiles: tl.constexpr = NUM_SF_COLS // 4
    return ((rows // 128) * (k_tiles * 512) + (sf_cols // 4) * 512 + (rows % 32) * 16 + ((rows % 128) // 32) * 4 +
            (sf_cols % 4))


@triton.jit
def _store_quantized(q_ptr, sf_ptr, packed, sf_bits, rows, col0, row_ok, sf_row_ok, R: tl.constexpr,
                     N: tl.constexpr, K: tl.constexpr):
    """Store a quantized [R, N] tile that starts at input column ``col0`` of a K-wide activation."""
    rows64 = rows.to(tl.int64)
    byte_cols = col0 // 2 + tl.arange(0, N // 2)
    tl.store(q_ptr + rows64[:, None] * (K // 2) + byte_cols[None, :], packed, mask=row_ok[:, None])
    sf_cols = col0 // 16 + tl.arange(0, N // 16)
    tl.store(sf_ptr + _sf_offsets(rows64[:, None], sf_cols[None, :], K // 16), sf_bits, mask=sf_row_ok[:, None])


@triton.jit
def _lane_tree_sum16(s, ROWS: tl.constexpr):
    """Sum [ROWS, 16] lanes in warp shuffle-down order: lane t pairs with t + 8, then + 4, + 2, + 1."""
    s = tl.sum(tl.reshape(s, (ROWS, 2, 8)), axis=1)
    s = tl.sum(tl.reshape(s, (ROWS, 2, 4)), axis=1)
    s = tl.sum(tl.reshape(s, (ROWS, 2, 2)), axis=1)
    return tl.sum(s, axis=1)


@triton.jit
def _lane_tree_sum32(s, ROWS: tl.constexpr):
    """Sum [ROWS, 32] lanes in warp shuffle-down order (lane t pairs with t + 16 first)."""
    return _lane_tree_sum16(tl.sum(tl.reshape(s, (ROWS, 2, 16)), axis=1), ROWS)


@triton.jit
def _residual_norm_quant_kernel(h_ptr, o_ptr, bias_ptr, scale_ptr, gamma_ptr, gsf_ptr, q_ptr, sf_ptr, M, M_PAD, eps,
                                D: tl.constexpr, R: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
                                HAS_NORM: tl.constexpr):
    """``h += bf16(o + bias) * scale`` (optional), then NVFP4(bf16(rms_norm(h) * gamma)) (optional)."""
    rows = tl.program_id(0) * R + tl.arange(0, R)
    row_ok = rows < M
    rows64 = rows.to(tl.int64)
    acc = tl.zeros((R, 128), tl.float32)
    for k in tl.static_range(D // 512):
        cols = k * 512 + tl.arange(0, 512)
        offs = rows64[:, None] * D + cols[None, :]
        h = tl.load(h_ptr + offs, mask=row_ok[:, None], other=0.0)
        if HAS_RESIDUAL:
            o = tl.load(o_ptr + offs, mask=row_ok[:, None], other=0.0).to(tl.float32)
            biased = _bf16(o + tl.load(bias_ptr + cols).to(tl.float32)[None, :])
            h = h + _mul_rn(biased, tl.load(scale_ptr + cols)[None, :])
            tl.store(h_ptr + offs, h, mask=row_ok[:, None])
        if HAS_NORM:
            # Column 512k + 4t + i belongs to ATen thread t; each thread sums its 4-vectors in order.
            evens, odds = tl.split(tl.reshape(h, (R, 128, 2, 2)))
            x0, x2 = tl.split(evens)
            x1, x3 = tl.split(odds)
            acc = tl.fma(x0, x0, acc)
            acc = tl.fma(x1, x1, acc)
            acc = tl.fma(x2, x2, acc)
            acc = tl.fma(x3, x3, acc)
    if HAS_NORM:
        warp_sums = _lane_tree_sum32(tl.reshape(acc, (R * 4, 32)), R * 4)
        sigma2 = tl.sum(tl.sum(tl.reshape(warp_sums, (R, 2, 2)), axis=1), axis=1)
        rstd = _rsqrt_approx(sigma2 * (1.0 / D) + eps)
        gsf = tl.load(gsf_ptr)
        if HAS_RESIDUAL:
            tl.debug_barrier()
        sf_row_ok = rows < M_PAD
        for k in tl.static_range(D // 512):
            cols = k * 512 + tl.arange(0, 512)
            h = tl.load(h_ptr + rows64[:, None] * D + cols[None, :], mask=row_ok[:, None], other=0.0)
            gamma = tl.load(gamma_ptr + cols)
            normed = _bf16(gamma[None, :] * (rstd[:, None] * h))
            packed, sf_bits = _quantize_tile(normed, gsf, R, 512)
            _store_quantized(q_ptr, sf_ptr, packed, sf_bits, rows, k * 512, row_ok, sf_row_ok, R, 512, D)


@triton.jit
def _quantize_kernel(x_ptr, gsf_ptr, q_ptr, sf_ptr, M, M_PAD, K: tl.constexpr, R: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    row_ok = rows < M
    x = tl.load(x_ptr + rows.to(tl.int64)[:, None] * K + cols[None, :], mask=row_ok[:, None], other=0.0)
    packed, sf_bits = _quantize_tile(x.to(tl.float32), tl.load(gsf_ptr), R, BN)
    _store_quantized(q_ptr, sf_ptr, packed, sf_bits, rows, tl.program_id(1) * BN, row_ok, rows < M_PAD, R, BN, K)


@triton.jit
def _swiglu_quantize_kernel(o_ptr, bias_ptr, silu_ptr, gsf_ptr, q_ptr, sf_ptr, M, M_PAD, F: tl.constexpr,
                            R: tl.constexpr, BN: tl.constexpr):
    """NVFP4(bf16(bf16(value + b) * silu(bf16(gate + b)))) on a value-first packed projection.

    ``silu_ptr`` maps every bf16 bit pattern to eager ``F.silu`` of it, so SiLU costs one cached load.
    """
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    row_ok = rows < M
    offs = rows.to(tl.int64)[:, None] * (2 * F) + cols[None, :]
    value = _bf16(
        tl.load(o_ptr + offs, mask=row_ok[:, None], other=0.0).to(tl.float32) +
        tl.load(bias_ptr + cols).to(tl.float32)[None, :])
    gate = (tl.load(o_ptr + offs + F, mask=row_ok[:, None], other=0.0).to(tl.float32) +
            tl.load(bias_ptr + F + cols).to(tl.float32)[None, :]).to(tl.bfloat16)
    silu = tl.load(silu_ptr + gate.to(tl.uint16, bitcast=True).to(tl.int32)).to(tl.float32)
    # Padding rows (only their zero block scales are stored) must not pick up the bias.
    hidden = tl.where(row_ok[:, None], _bf16(value * silu), 0.0)
    packed, sf_bits = _quantize_tile(hidden, tl.load(gsf_ptr), R, BN)
    _store_quantized(q_ptr, sf_ptr, packed, sf_bits, rows, tl.program_id(1) * BN, row_ok, rows < M_PAD, R, BN, F)


@triton.jit
def _qkv_epilogue_kernel(q_ptr, k_ptr, v_ptr, bq_ptr, bk_ptr, bv_ptr, cos_ptr, sin_ptr, q_out, k_out, v_out, M, eps,
                         H: tl.constexpr, HD: tl.constexpr, ROT: tl.constexpr, R: tl.constexpr):
    """Bias + per-head RMSNorm + partial RoPE for q and k, bias for v (eager rounding throughout)."""
    D: tl.constexpr = H * HD
    which = tl.program_id(1)
    if which == 2:
        rows = tl.program_id(0) * R + tl.arange(0, R)
        cols = tl.arange(0, D)
        offs = rows.to(tl.int64)[:, None] * D + cols[None, :]
        mask = (rows < M)[:, None]
        v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32) + tl.load(bv_ptr + cols).to(tl.float32)[None, :]
        tl.store(v_out + offs, v.to(tl.bfloat16), mask=mask)
    else:
        src = q_ptr
        bias_ptr = bq_ptr
        dst = q_out
        if which == 1:
            src = k_ptr
            bias_ptr = bk_ptr
            dst = k_out
        RH: tl.constexpr = R * H
        head_rows = tl.arange(0, RH)
        tokens = tl.program_id(0) * R + head_rows // H
        token_ok = (tokens < M)[:, None]
        base = tokens.to(tl.int64)[:, None] * D + ((head_rows % H) * HD)[:, None]
        bias_base = ((head_rows % H) * HD)[:, None]
        d = tl.arange(0, HD)[None, :]
        x = _bf16(
            tl.load(src + base + d, mask=token_ok, other=0.0).to(tl.float32) +
            tl.load(bias_ptr + bias_base + d).to(tl.float32))
        # ATen's per-row RMSNorm over HD = 64: thread t < 16 sums elements 4t..4t+3 in order.
        evens, odds = tl.split(tl.reshape(x, (RH, HD // 4, 2, 2)))
        x0, x2 = tl.split(evens)
        x1, x3 = tl.split(odds)
        acc = tl.fma(x0, x0, tl.zeros((RH, HD // 4), tl.float32))
        acc = tl.fma(x1, x1, acc)
        acc = tl.fma(x2, x2, acc)
        acc = tl.fma(x3, x3, acc)
        sigma2 = _lane_tree_sum16(acc, RH)
        rstd = _rsqrt_approx(sigma2 * (1.0 / HD) + eps)[:, None]
        normed = _bf16(rstd * x)
        half: tl.constexpr = ROT // 2
        rotary = d < ROT
        partner = tl.where(d < half, d + half, d - half)
        partner_x = _bf16(
            tl.load(src + base + partner, mask=token_ok & rotary, other=0.0).to(tl.float32) +
            tl.load(bias_ptr + bias_base + partner, mask=rotary, other=0.0).to(tl.float32))
        partner_normed = _bf16(rstd * partner_x)
        rotated = tl.where(d < half, -partner_normed, partner_normed)
        trig = tokens.to(tl.int64)[:, None] * ROT + d
        cos = tl.load(cos_ptr + trig, mask=token_ok & rotary, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + trig, mask=token_ok & rotary, other=0.0).to(tl.float32)
        roped = _bf16(_bf16(normed * cos) + _bf16(rotated * sin))
        tl.store(dst + base + d, tl.where(rotary, roped, normed).to(tl.bfloat16), mask=token_ok)


def _padded_rows(m: int) -> int:
    return (m + SF_ROW_TILE - 1) // SF_ROW_TILE * SF_ROW_TILE


def new_quantized(m: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Buffers shaped like FlashInfer's ``nvfp4_quantize`` output for an [m, k] activation."""
    packed = torch.empty((m, k // 2), dtype=torch.uint8, device=device)
    scales = torch.empty((_padded_rows(m), k // NVFP4_BLOCK), dtype=torch.uint8, device=device)
    return packed, scales


def residual_norm_quantize(h: torch.Tensor, residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
                           norm: tuple[torch.Tensor, float, torch.Tensor] | None,
                           rows_per_program: int = 4) -> tuple[torch.Tensor, torch.Tensor] | None:
    """In place ``h += bf16(o + bias) * scale`` and/or NVFP4 of ``bf16(rms_norm(h, gamma, eps))``.

    ``h`` is a contiguous fp32 [M, D] tensor; ``residual`` is ``(o, bias_bf16, scale_fp32)``;
    ``norm`` is ``(gamma_fp32, eps, global_sf)``. Returns the quantized activation when ``norm`` is set.
    """
    m, d = h.shape
    if d % 512:
        raise ValueError(f"hidden size must be a multiple of 512, got {d}")
    quantized = new_quantized(m, d, h.device) if norm is not None else (h, h)
    o, bias, scale = residual if residual is not None else (h, h, h)
    gamma, eps, gsf = norm if norm is not None else (h, 0.0, h)
    rows = _padded_rows(m) if norm is not None else m
    grid = (triton.cdiv(rows, rows_per_program), )
    _residual_norm_quant_kernel[grid](h,
                                      o,
                                      bias,
                                      scale,
                                      gamma,
                                      gsf,
                                      quantized[0],
                                      quantized[1],
                                      m,
                                      rows,
                                      eps,
                                      D=d,
                                      R=rows_per_program,
                                      HAS_RESIDUAL=residual is not None,
                                      HAS_NORM=norm is not None,
                                      num_warps=4)
    return quantized if norm is not None else None


def quantize(x: torch.Tensor, gsf: torch.Tensor, rows_per_program: int = 8,
             block_n: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer-exact NVFP4 quantization (``layout_128x4``) of a contiguous bf16 [M, K] tensor."""
    m, k = x.shape
    quantized = new_quantized(m, k, x.device)
    grid = (triton.cdiv(_padded_rows(m), rows_per_program), triton.cdiv(k, block_n))
    _quantize_kernel[grid](x, gsf, quantized[0], quantized[1], m, _padded_rows(m), K=k, R=rows_per_program, BN=block_n)
    return quantized


@functools.cache
def _silu_table(device_index: int) -> torch.Tensor:
    """Eager ``F.silu`` of every bf16 value, indexed by its 16-bit pattern."""
    patterns = torch.arange(1 << 16, dtype=torch.int32, device=torch.device("cuda", device_index))
    return F.silu(patterns.to(torch.int16).view(torch.bfloat16))  # int32 -> int16 wraps to the bit pattern


def swiglu_quantize(o: torch.Tensor, bias: torch.Tensor, gsf: torch.Tensor, rows_per_program: int = 2,
                    block_n: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 of the H3 SwiGLU of a value-first packed projection ``o`` [M, 2F] (bias not yet added)."""
    m, two_f = o.shape
    f = two_f // 2
    quantized = new_quantized(m, f, o.device)
    grid = (triton.cdiv(_padded_rows(m), rows_per_program), triton.cdiv(f, block_n))
    _swiglu_quantize_kernel[grid](o,
                                  bias,
                                  _silu_table(o.device.index),
                                  gsf,
                                  quantized[0],
                                  quantized[1],
                                  m,
                                  _padded_rows(m),
                                  F=f,
                                  R=rows_per_program,
                                  BN=block_n,
                                  num_warps=2)
    return quantized


def qkv_epilogue(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, biases: tuple[torch.Tensor, torch.Tensor,
                                                                               torch.Tensor], cos: torch.Tensor,
                 sin: torch.Tensor, heads: int, eps: float,
                 rows_per_program: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projected q/k/v [M, H*HD] bf16 (bias not yet added) -> roped q, k and biased v, each [M, H, HD].

    ``cos``/``sin`` are bf16 [M, ROT] (the eager path's ``cos.to(query.dtype)``).
    """
    m, d = q.shape
    head_dim = d // heads
    outs = [torch.empty((m, heads, head_dim), dtype=torch.bfloat16, device=q.device) for _ in range(3)]
    grid = (triton.cdiv(m, rows_per_program), 3)
    _qkv_epilogue_kernel[grid](q,
                               k,
                               v,
                               *biases,
                               cos,
                               sin,
                               *outs,
                               m,
                               eps,
                               H=heads,
                               HD=head_dim,
                               ROT=cos.shape[-1],
                               R=rows_per_program,
                               num_warps=4)
    return outs[0], outs[1], outs[2]


class _BlockPlan(NamedTuple):
    qkv: tuple[NVFP4FusedState, NVFP4FusedState, NVFP4FusedState]
    out: NVFP4FusedState
    ff_in: NVFP4FusedState
    ff_out: NVFP4FusedState
    norm1: tuple[torch.Tensor, float]
    norm2: tuple[torch.Tensor, float]
    qk_eps: float
    scale1: torch.Tensor
    scale2: torch.Tensor
    heads: int
    attention: Any


def _rms_norm_args(norm: nn.Module, affine: bool) -> tuple[torch.Tensor | None, float] | None:
    if not isinstance(norm, nn.RMSNorm) or (norm.weight is not None) != affine:
        return None
    if affine and norm.weight.dtype != torch.float32:
        return None
    return norm.weight, norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps


def _clear_block_plan(module: nn.Module) -> None:
    """Drop a block's cached fused plan, whose tensors point at the previous weight storage."""
    module.__dict__.pop("_nvfp4_fused_plan", None)


def _block_plan(block: nn.Module) -> _BlockPlan | None:
    """The fused plan of one ``MiniMaxH3VideoTransformerBlock``, or None if it must run eagerly."""
    attn, ff = block.attn, block.ff
    linears = (attn.to_q, attn.to_k, attn.to_v, attn.to_out[0], ff.net[0].proj, ff.net[2])
    if not all(isinstance(layer, NVFP4DecoderLinear) for layer in linears):
        return None
    # The plan holds views of the parameter storage, so the offload walk has to
    # drop it when the module moves to the host (parameters/buffers alone miss it).
    if "clear_offload_caches" not in block.__dict__:
        block.__dict__["clear_offload_caches"] = lambda module=block: _clear_block_plan(module)
    states = [layer.fused_state() for layer in linears]
    if any(state is None for state in states):
        return None
    key = tuple(id(state) for state in states)
    cached = getattr(block, "_nvfp4_fused_plan", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    norm1, norm2 = _rms_norm_args(block.norm1, True), _rms_norm_args(block.norm2, True)
    norm_q, norm_k = _rms_norm_args(attn.norm_q, False), _rms_norm_args(attn.norm_k, False)
    dim = attn.heads * attn.dim_head
    # The quantization kernels tile the inner dimension by 512 columns without a column mask.
    ff_inner = ff.net[0].proj.out_features
    plan = None
    if (None not in (norm1, norm2, norm_q, norm_k) and norm_q[1] == norm_k[1] and attn.attn_impl is not None
            and attn.dim_head == 64 and attn.heads & (attn.heads - 1) == 0 and dim % 512 == 0
            and ff_inner % 512 == 0
            and block.scale1.dtype == block.scale2.dtype == torch.float32
            and isinstance(ff.net[0].activation, nn.SiLU)
            and all(state.global_sf_x is states[0].global_sf_x or torch.equal(state.global_sf_x, states[0].global_sf_x)
                    for state in states[1:3])):
        plan = _BlockPlan(tuple(states[:3]), states[3], states[4], states[5], norm1, norm2, norm_q[1], block.scale1,
                          block.scale2, attn.heads, attn.attn_impl)
    block._nvfp4_fused_plan = (key, plan)
    return plan


_GEMM_WORKSPACE_BYTES = 40 << 20
_GEMM_TUNING_REPEATS = 3


@functools.cache
def _sm100_cutlass_fp4_module(device_index: int) -> Any | None:
    """FlashInfer's CUTLASS FP4 GEMM extension on sm_100, called without ``mm_fp4``'s per-call checks.

    ``mm_fp4`` costs ~100 us of host time per call (a single 1797-token tile GEMM runs in ~12 us).
    Other architectures keep ``mm_fp4``.
    """
    if torch.cuda.get_device_capability(device_index) != (10, 0):
        return None
    try:
        from flashinfer.gemm.gemm_base import gen_gemm_sm100_module_cutlass_fp4

        return gen_gemm_sm100_module_cutlass_fp4().build_and_load()
    except (ImportError, AttributeError, RuntimeError):
        return None


@functools.cache
def _gemm_workspace(device_index: int) -> torch.Tensor:
    return torch.empty(_GEMM_WORKSPACE_BYTES, dtype=torch.uint8, device=torch.device("cuda", device_index))


_GEMM_TACTICS: dict[tuple[int, int, int, int], int] = {}


def _gemm_tactic(module: Any, args: tuple, out: torch.Tensor) -> int:
    """Fastest CUTLASS tactic for this GEMM shape, measured once.

    Every tactic computes the same K-ordered fp32 accumulation, so the choice changes speed only
    (``test_cutlass_fp4_tactics_are_bit_identical`` checks this).
    """
    key = (out.device.index, out.shape[0], out.shape[1], args[0].shape[1])
    if key not in _GEMM_TACTICS:
        timings = []
        for tactic in range(-1, module.fp4_gemm_tactic_num()):
            try:
                module.fp4_gemm(*args, out, _gemm_workspace(out.device.index), tactic)
            except RuntimeError:
                continue
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(_GEMM_TUNING_REPEATS):
                module.fp4_gemm(*args, out, _gemm_workspace(out.device.index), tactic)
            end.record()
            end.synchronize()
            timings.append((start.elapsed_time(end), tactic))
        _GEMM_TACTICS[key] = min(timings)[1] if timings else -1
    return _GEMM_TACTICS[key]


def _gemm(quantized: tuple[torch.Tensor, torch.Tensor], state: NVFP4FusedState) -> torch.Tensor:
    """``nvfp4_linear_inference``'s GEMM (bias excluded) on an already quantized activation."""
    x_fp4, x_inv_scale = quantized
    out = torch.empty((x_fp4.shape[0], state.out_features), device=x_fp4.device, dtype=torch.bfloat16)
    module = _sm100_cutlass_fp4_module(out.device.index)
    if module is not None:
        # mm_fp4(backend="cutlass") hands the module the same operands (it transposes b and b's scales back).
        args = (x_fp4, state.packed, x_inv_scale, state.inv_scale, state.alpha)
        key = (out.device.index, out.shape[0], out.shape[1], x_fp4.shape[1])
        # Tuning synchronizes, so a CUDA graph capture uses an already tuned (or the default) tactic.
        tactic = (_GEMM_TACTICS.get(key, -1)
                  if torch.cuda.is_current_stream_capturing() else _gemm_tactic(module, args, out))
        module.fp4_gemm(*args, out, _gemm_workspace(out.device.index), tactic)
        return out
    import flashinfer

    from fastvideo.models.vaes import minimax_h3_nvfp4_decoder

    backend = minimax_h3_nvfp4_decoder.mm_fp4_backend(out.device)
    flashinfer.mm_fp4(x_fp4,
                      state.packed.T,
                      x_inv_scale,
                      state.inv_scale.T,
                      state.alpha,
                      torch.bfloat16,
                      out,
                      block_size=NVFP4_BLOCK,
                      use_8x4_sf_layout=False,
                      backend=backend)
    return out


def _run_blocks(plans: list[_BlockPlan], hidden: torch.Tensor, rotary_emb: tuple[torch.Tensor,
                                                                                  torch.Tensor]) -> torch.Tensor:
    batch, seq, dim = hidden.shape
    rows = batch * seq
    cos, sin = (t.to(torch.bfloat16).expand(batch, seq, 1, -1).reshape(rows, -1).contiguous() for t in rotary_emb)
    h = hidden.reshape(rows, dim).clone()
    first = plans[0]
    quantized = residual_norm_quantize(h, None, (*first.norm1, first.qkv[0].global_sf_x))
    for index, plan in enumerate(plans):
        projections = [_gemm(quantized, state) for state in plan.qkv]
        query, key, value = qkv_epilogue(*projections, tuple(state.bias for state in plan.qkv), cos, sin, plan.heads,
                                         plan.qk_eps)
        del projections
        shape = (batch, seq, plan.heads, dim // plan.heads)
        attended = plan.attention.forward(query.view(shape), key.view(shape), value.view(shape), None)
        del query, key, value
        attended = attended.reshape(rows, dim).contiguous()
        out = _gemm(quantize(attended, plan.out.global_sf_x), plan.out)
        del attended
        quantized = residual_norm_quantize(h, (out, plan.out.bias, plan.scale1),
                                           (*plan.norm2, plan.ff_in.global_sf_x))
        out = _gemm(quantized, plan.ff_in)
        out = _gemm(swiglu_quantize(out, plan.ff_in.bias, plan.ff_out.global_sf_x), plan.ff_out)
        following = plans[index + 1] if index + 1 < len(plans) else None
        quantized = residual_norm_quantize(h, (out, plan.ff_out.bias, plan.scale2),
                                           None if following is None else
                                           (*following.norm1, following.qkv[0].global_sf_x))
        del out
    return h.view(batch, seq, dim)


# Calls up to this many tokens are launch-bound (a 1797-token tile runs ~400 kernels in ~5 ms of GPU
# time), so their block stack is captured once per shape and replayed as a CUDA graph. Larger calls
# are GPU-bound and run directly, without holding a graph memory pool.
GRAPH_MAX_ROWS = 4 * 1797


# Replayed geometries kept at once. Each entry holds static tensors and a graph's memory pool, so a
# long-lived generator that walks through tile batching choices must not keep every shape it has seen.
GRAPH_CACHE_LIMIT = 2


class _GraphedStack(NamedTuple):
    plan_ids: tuple[int, ...]
    graph: torch.cuda.CUDAGraph
    hidden: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    out: torch.Tensor


def _weight_pointers(blocks: nn.ModuleList) -> tuple[int, ...]:
    """A signature of where the blocks' weights live.

    A captured graph bakes the weight addresses into its kernels, so a replay is
    only valid while they stay put. ``pinned_offload`` hands out fresh device
    storage on every load, which changes them; including them in the cache key
    re-captures instead of replaying against freed addresses.
    """
    return tuple(t.data_ptr() for t in (*blocks.parameters(), *blocks.buffers()))


def _trim_graphs(graphs: dict) -> None:
    """Keep the most recent ``GRAPH_CACHE_LIMIT`` geometries (the ``warm`` markers are tiny)."""
    geometries = [key for key in graphs if not (isinstance(key, tuple) and key[:1] == ("warm", ))]
    for key in geometries[:-GRAPH_CACHE_LIMIT]:
        graphs.pop(key, None)
        graphs.pop(("warm", key), None)


def _clear_graph_entries(graphs: dict) -> None:
    """Release every captured graph (and its memory pool) in ``graphs``."""
    for entry in graphs.values():
        if isinstance(entry, _GraphedStack):
            entry.graph.reset()
    graphs.clear()


def _graphed_blocks(blocks: nn.ModuleList, plans: list[_BlockPlan], hidden: torch.Tensor,
                    rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """``_run_blocks`` through a CUDA graph captured per input geometry (same kernels, same bits)."""
    key = (hidden.device, tuple(hidden.shape), tuple(rotary_emb[0].shape), rotary_emb[0].dtype,
           _weight_pointers(blocks))
    graphs: dict = blocks.__dict__.setdefault("_nvfp4_cuda_graphs", {})
    if "clear_offload_caches" not in blocks.__dict__:
        # The graph cache lives on this ModuleList, not on a parameter or buffer of
        # it, so the generic offload walk needs a hook to find it. An unload hands
        # out fresh device storage, and a replay against the old addresses is wrong.
        blocks.__dict__["clear_offload_caches"] = lambda: _clear_graph_entries(graphs)
    plan_ids = tuple(id(plan) for plan in plans)
    entry = graphs.get(key)
    if entry is None or entry.plan_ids != plan_ids:
        if graphs.get(("warm", key)) != plan_ids:
            # First call: run eagerly so Triton compiles, GEMM tactics are tuned and lazy tables exist.
            graphs[("warm", key)] = plan_ids
            return _run_blocks(plans, hidden, rotary_emb)
        static_hidden = hidden.clone()
        static_cos, static_sin = (t.clone() for t in rotary_emb)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = _run_blocks(plans, static_hidden, (static_cos, static_sin))
        entry = _GraphedStack(plan_ids, graph, static_hidden, static_cos, static_sin, static_out)
        graphs[key] = entry
        _trim_graphs(graphs)
    entry.hidden.copy_(hidden)
    entry.cos.copy_(rotary_emb[0])
    entry.sin.copy_(rotary_emb[1])
    entry.graph.replay()
    return entry.out.clone()


@torch.compiler.disable
def fused_nvfp4_blocks_forward(blocks: nn.ModuleList, hidden: torch.Tensor,
                               rotary_emb: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor | None:
    """Run every decoder block through the fused kernels; None (caller runs the blocks) when not applicable.

    Applies at inference (no grad) to fp32 residual streams on CUDA when every block's linears are NVFP4
    with a precomputed activation scale. Bit-identical to the eager block loop. ``torch.compile`` treats it
    as a boundary: a compiled decoder compiles the code around the block stack and runs the stack here.
    Calls of at most ``GRAPH_MAX_ROWS`` tokens replay a per-shape CUDA graph
    (``FASTVIDEO_H3_VAE_NVFP4_CUDA_GRAPH``).
    """
    if (torch.compiler.is_compiling() or torch.is_grad_enabled() or not envs.FASTVIDEO_H3_VAE_NVFP4_FUSED.get()
            or rotary_emb is None or len(blocks) == 0 or hidden.dtype != torch.float32 or not hidden.is_cuda):
        return None
    plans = [_block_plan(block) for block in blocks]
    if any(plan is None for plan in plans):
        return None
    if (hidden.shape[0] * hidden.shape[1] <= GRAPH_MAX_ROWS and envs.FASTVIDEO_H3_VAE_NVFP4_CUDA_GRAPH.get()
            and not torch.cuda.is_current_stream_capturing()):
        return _graphed_blocks(blocks, plans, hidden, rotary_emb)
    return _run_blocks(plans, hidden, rotary_emb)
