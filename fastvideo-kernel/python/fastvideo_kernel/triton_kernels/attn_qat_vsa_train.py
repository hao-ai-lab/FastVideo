# SPDX-License-Identifier: Apache-2.0
"""Block-sparse FP4 Attn-QAT for VSA tile-128 training (forward + backward).

The forward reproduces the deployed sm_120/121 sparse FP4 kernel
(``attn_qat_infer.api.sageattn_blackwell_sparse_bshd`` on lists from
``vsa_tile_mask_to_fp4_blocks(mask, 128, vbs)``) on any CUDA GPU with Triton,
so a GB200 can train against Spark/5090 numerics. The numeric contract is
spelled out in ``attn_qat_vsa_reference`` (the slow PyTorch specification the
unit tests compare against):

* Q/K/V are fake-quantized once per call (NVFP4, per-16 E4M3 scales, no global
  scale; V grouped along tokens) and stored in BF16, which holds every
  dequantized value exactly.
* Every query row visits its block's listed 128-key blocks in ascending order;
  the running max is updated per 128-key block before P is formed, P is
  quantized per 16 keys (single- or two-level), the row sum is unquantized.

Backward is a straight-through estimator with one deliberate choice per
operand (the lessons of the GB10 Attn-QAT dV fix):

* dV is forward-consistent: it is built from the forward's quantized
  online-softmax weights, recomputed from the saved per-(row, visited block)
  running maxima, never by re-quantizing the normalized probabilities (whose
  E4M3 group scales underflow on long sequences and silence whole heads).
* dQ/dK are the exact gradients of the high-precision surrogate
  ``softmax(q_hat k_hat^T) v_hat`` on the same sparse mask; delta uses the
  high-precision output.
* The Q/K/V quantizers pass gradients through unchanged.

Rows are independent, so programs cover 64 query rows (the forward keeps whole
128-key blocks, where the running max is defined); the backward splits keys
into 64-column halves, which the per-16 P groups allow.
"""
from __future__ import annotations

import math


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .attn_qat_vsa_reference import BLOCK, LOG2_INV_TWO_LEVEL, LOG2E, invert_block_lists

_ROWS = 64  # query rows per program
_HALF = BLOCK // 2
_QUANT_ROWS = 64  # fake-quant rows per program (a multiple of 16; L is a multiple of 128)
_WARPS = 8  # 8 beats 4 by 15-20% on GB200 at 35k-73k tokens


@triton.jit
def _e2m1_rn(y):
    """Round to the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to even, saturating at 6.

    The grid is 0.5-spaced below 2, 1-spaced in [2, 4) and 2-spaced above, so
    round-half-even on the scaled value is exact (ties land on even codes).
    """
    a = tl.minimum(tl.abs(y), 6.0)
    q = tl.where(a < 2.0,
                 libdevice.rint(a * 2.0) * 0.5,
                 tl.where(a < 4.0, libdevice.rint(a),
                          libdevice.rint(a * 0.5) * 2.0))
    return tl.where(y < 0, -q, q)


@triton.jit
def _e4m3_rn(x):
    """Round non-negative fp32 to E4M3 (RN, saturating at 448) and back."""
    return tl.minimum(x, 448.0).to(tl.float8e4nv).to(tl.float32)


@triton.jit
def _quantized_p(s, m_scaled, first_visit, scale_log2, FLOOR: tl.constexpr, FLOOR_VALUE: tl.constexpr,
                 ROWS: tl.constexpr, COLS: tl.constexpr):
    """``P~`` and its fake-quantized value for raw scores ``s`` (masked = -inf) under row maxima ``m_scaled``."""
    p_tilde = tl.math.exp2(s * scale_log2 - m_scaled[:, None])
    group_max = tl.max(tl.reshape(s, (ROWS, COLS // 16, 16)), axis=2)
    if FLOOR:
        group_max = tl.where(first_visit, tl.maximum(group_max, FLOOR_VALUE), group_max)
    s_p = tl.math.exp2(group_max * scale_log2 - m_scaled[:, None] - 2.584962500721156)
    # Per-group values (fp32 divisor, E4M3 multiplier) broadcast back over each group's 16 keys.
    s_full = tl.reshape(tl.broadcast_to(s_p[:, :, None], (ROWS, COLS // 16, 16)), (ROWS, COLS))
    s_e4m3 = tl.reshape(tl.broadcast_to(_e4m3_rn(s_p)[:, :, None], (ROWS, COLS // 16, 16)), (ROWS, COLS))
    ratio = tl.where(s_full == 0.0, 0.0, p_tilde / s_full)
    return p_tilde, _e2m1_rn(ratio) * s_e4m3


@triton.jit
def _fake_quant_kernel(X, Y, n_rows, D: tl.constexpr, ROWS: tl.constexpr, ALONG_ROWS: tl.constexpr):
    """NVFP4 fake quant of a ``[n_rows, D]`` matrix, groups of 16 along D or (``ALONG_ROWS``) along rows."""
    pid = tl.program_id(0).to(tl.int64)
    offs_r = pid * ROWS + tl.arange(0, ROWS)
    offs_d = tl.arange(0, D)
    x = tl.load(X + offs_r[:, None] * D + offs_d[None, :]).to(tl.float32)
    if ALONG_ROWS:
        x3 = tl.reshape(x, (ROWS // 16, 16, D))
        amax = tl.max(tl.abs(x3), axis=1)
        scale = _e4m3_rn(amax / 6.0)
        inv = tl.where(scale == 0.0, 0.0, 1.0 / scale)
        y3 = _e2m1_rn(x3 * inv[:, None, :]) * scale[:, None, :]
        y = tl.reshape(y3, (ROWS, D))
    else:
        x3 = tl.reshape(x, (ROWS, D // 16, 16))
        amax = tl.max(tl.abs(x3), axis=2)
        scale = _e4m3_rn(amax / 6.0)
        inv = tl.where(scale == 0.0, 0.0, 1.0 / scale)
        y3 = _e2m1_rn(x3 * inv[:, :, None]) * scale[:, :, None]
        y = tl.reshape(y3, (ROWS, D))
    tl.store(Y + offs_r[:, None] * D + offs_d[None, :], y.to(Y.dtype.element_ty))


def fake_quant_qkv_triton(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          quantize_v: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fake_quant_qkv`` (Q/K along D, V along tokens) as one Triton pass each; BF16 ``[B, H, L, D]`` out."""
    outs = []
    for x, along_rows in ((q, False), (k, False), (v, True)):
        x = x.contiguous()
        if along_rows and not quantize_v:
            outs.append(x.to(torch.bfloat16))
            continue
        y = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
        rows = x.numel() // x.shape[-1]
        _fake_quant_kernel[(rows // _QUANT_ROWS, )](x, y, rows, D=x.shape[-1], ROWS=_QUANT_ROWS,
                                                     ALONG_ROWS=along_rows, num_warps=4)
        outs.append(y)
    return outs[0], outs[1], outs[2]


@triton.jit
def _key_ok(KV_VALID, n_block, cols, HAS_VALID: tl.constexpr, HALF: tl.constexpr):
    """Valid key columns (block-local ``cols`` in [0, 128)) of 128-key block ``n_block``."""
    if HAS_VALID:
        valid0 = tl.load(KV_VALID + 2 * n_block)
        valid1 = tl.load(KV_VALID + 2 * n_block + 1)
        limit = tl.where(cols < HALF, valid0, valid1)
        return (cols % HALF) < limit
    return cols >= 0


@triton.jit
def _fwd_kernel(Q, K, V, O, OHP, LSE, MRUN, Q2K_IDX, Q2K_NUM, KV_VALID, BIAS, QGROUP, n_groups, seq_len, n_qblocks,
                list_width, scale_log2, OFFSET: tl.constexpr, QUANT: tl.constexpr, HAS_VALID: tl.constexpr,
                HAS_BIAS: tl.constexpr, FLOOR: tl.constexpr, FLOOR_VALUE: tl.constexpr, HIGH_PREC_O: tl.constexpr,
                SAVE_MRUN: tl.constexpr,
                D: tl.constexpr, ROWS: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    halves: tl.constexpr = BLOCK_N // ROWS
    m_block = pid // halves
    row0 = (pid % halves) * ROWS
    offs_r = tl.arange(0, ROWS)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    base = bh * seq_len * D
    rows = m_block * BLOCK_N + row0 + offs_r
    q = tl.load(Q + base + rows[:, None] * D + offs_d[None, :])
    meta = bh * n_qblocks + m_block
    count = tl.load(Q2K_NUM + meta)
    m_i = tl.full((ROWS, ), float("-inf"), tl.float32)
    l_i = tl.zeros((ROWS, ), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    acc_hp = tl.zeros((ROWS, D), tl.float32)
    if HAS_BIAS:
        bias_row = BIAS + (bh * n_groups + tl.load(QGROUP + m_block)) * seq_len
    for it in range(0, count):
        slot = count - 1 - it
        n_block = tl.load(Q2K_IDX + meta * list_width + slot)
        keys = n_block * BLOCK_N + offs_n
        k = tl.load(K + base + keys[:, None] * D + offs_d[None, :])
        s = tl.dot(q, tl.trans(k))
        if HAS_BIAS:  # per-key score correction (Q smoothing: mu_q . k_j, high precision)
            s = s + tl.load(bias_row + keys)[None, :]
        ok = _key_ok(KV_VALID, n_block, offs_n, HAS_VALID, BLOCK_N // 2)
        s = tl.where(ok[None, :], s, float("-inf"))
        group_max = tl.max(tl.reshape(s, (ROWS, BLOCK_N // 16, 16)), axis=2)
        if FLOOR:
            group_max = tl.where(it == 0, tl.maximum(group_max, FLOOR_VALUE), group_max)
        m_new = tl.maximum(m_i, tl.max(group_max, axis=1))
        m_scaled = m_new * scale_log2 + OFFSET
        alpha = tl.math.exp2((m_i - m_new) * scale_log2)
        if QUANT:
            p_tilde, p_hat = _quantized_p(s, m_scaled, it == 0, scale_log2, FLOOR, FLOOR_VALUE, ROWS, BLOCK_N)
        else:
            p_tilde = tl.math.exp2(s * scale_log2 - m_scaled[:, None])
            p_hat = p_tilde
        l_i = l_i * alpha + tl.sum(p_tilde, axis=1)
        v = tl.load(V + base + keys[:, None] * D + offs_d[None, :])
        acc = acc * alpha[:, None] + tl.dot(p_hat.to(tl.bfloat16), v)
        if HIGH_PREC_O:
            acc_hp = acc_hp * alpha[:, None] + tl.dot(p_tilde.to(tl.bfloat16), v)
        if SAVE_MRUN:
            tl.store(MRUN + (meta * list_width + slot) * BLOCK_N + row0 + offs_r, m_new)
        m_i = m_new
    inv = tl.where((l_i == 0.0) | (l_i != l_i), 0.0, 1.0 / l_i)
    tl.store(O + base + rows[:, None] * D + offs_d[None, :], (acc * inv[:, None]).to(O.dtype.element_ty))
    if HIGH_PREC_O:
        tl.store(OHP + base + rows[:, None] * D + offs_d[None, :], (acc_hp * inv[:, None]).to(OHP.dtype.element_ty))
    tl.store(LSE + bh * seq_len + rows, m_i * scale_log2 + OFFSET + tl.math.log2(l_i))


@triton.jit
def _bwd_dq_kernel(Q, K, V, DO, LSE, DELTA, DQ, Q2K_IDX, Q2K_NUM, KV_VALID, seq_len, n_qblocks, list_width,
                   scale_log2, sm_scale, HAS_VALID: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    halves: tl.constexpr = BLOCK_N // ROWS
    m_block = pid // halves
    offs_r = tl.arange(0, ROWS)
    offs_c = tl.arange(0, ROWS)  # key columns per step: one 64-key half
    offs_d = tl.arange(0, D)
    base = bh * seq_len * D
    rows = m_block * BLOCK_N + (pid % halves) * ROWS + offs_r
    q = tl.load(Q + base + rows[:, None] * D + offs_d[None, :])
    do = tl.load(DO + base + rows[:, None] * D + offs_d[None, :])
    lse = tl.load(LSE + bh * seq_len + rows)
    delta = tl.load(DELTA + bh * seq_len + rows)
    meta = bh * n_qblocks + m_block
    count = tl.load(Q2K_NUM + meta)
    dq = tl.zeros((ROWS, D), tl.float32)
    for it in range(0, count * halves):
        n_block = tl.load(Q2K_IDX + meta * list_width + it // halves)
        cols = (it % halves) * ROWS + offs_c
        keys = n_block * BLOCK_N + cols
        k = tl.load(K + base + keys[:, None] * D + offs_d[None, :])
        v = tl.load(V + base + keys[:, None] * D + offs_d[None, :])
        s = tl.dot(q, tl.trans(k))
        ok = _key_ok(KV_VALID, n_block, cols, HAS_VALID, BLOCK_N // 2)
        p = tl.where(ok[None, :], tl.math.exp2(s * scale_log2 - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.bfloat16), k)
    tl.store(DQ + base + rows[:, None] * D + offs_d[None, :], (dq * sm_scale).to(DQ.dtype.element_ty))


@triton.jit
def _bwd_dkdv_kernel(Q, K, V, DO, LSE, DELTA, MRUN, DK, DV, K2Q_IDX, K2Q_SLOT, K2Q_NUM, Q2K_NUM, KV_VALID, seq_len,
                     n_qblocks, n_kblocks, list_width, scale_log2, sm_scale, OFFSET: tl.constexpr,
                     QUANT: tl.constexpr, HAS_VALID: tl.constexpr, FLOOR: tl.constexpr,
                     FLOOR_VALUE: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    halves: tl.constexpr = BLOCK_N // ROWS
    n_block = pid // halves
    offs_r = tl.arange(0, ROWS)
    offs_d = tl.arange(0, D)
    cols = (pid % halves) * ROWS + tl.arange(0, ROWS)
    base = bh * seq_len * D
    keys = n_block * BLOCK_N + cols
    k = tl.load(K + base + keys[:, None] * D + offs_d[None, :])
    v = tl.load(V + base + keys[:, None] * D + offs_d[None, :])
    ok = _key_ok(KV_VALID, n_block, cols, HAS_VALID, BLOCK_N // 2)
    kmeta = bh * n_kblocks + n_block
    count = tl.load(K2Q_NUM + kmeta)
    dk = tl.zeros((ROWS, D), tl.float32)
    dv = tl.zeros((ROWS, D), tl.float32)
    for it in range(0, count * halves):
        entry = kmeta * n_qblocks + it // halves
        m_block = tl.load(K2Q_IDX + entry)
        slot = tl.load(K2Q_SLOT + entry)
        row0 = (it % halves) * ROWS
        rows = m_block * BLOCK_N + row0 + offs_r
        q = tl.load(Q + base + rows[:, None] * D + offs_d[None, :])
        do = tl.load(DO + base + rows[:, None] * D + offs_d[None, :])
        lse = tl.load(LSE + bh * seq_len + rows)
        delta = tl.load(DELTA + bh * seq_len + rows)
        s = tl.dot(q, tl.trans(k))
        s = tl.where(ok[None, :], s, float("-inf"))
        p = tl.math.exp2(s * scale_log2 - lse[:, None])
        if QUANT:
            qmeta = bh * n_qblocks + m_block
            first_visit = slot == tl.load(Q2K_NUM + qmeta) - 1
            m_run = tl.load(MRUN + (qmeta * list_width + slot) * BLOCK_N + row0 + offs_r)
            m_scaled = m_run * scale_log2 + OFFSET
            _, p_hat = _quantized_p(s, m_scaled, first_visit, scale_log2, FLOOR, FLOOR_VALUE, ROWS, ROWS)
            weight = p_hat * tl.math.exp2(m_scaled - lse)[:, None]
        else:
            weight = p
        dv += tl.dot(tl.trans(weight.to(tl.bfloat16)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q)
    tl.store(DK + base + keys[:, None] * D + offs_d[None, :], (dk * sm_scale).to(DK.dtype.element_ty))
    tl.store(DV + base + keys[:, None] * D + offs_d[None, :], dv.to(DV.dtype.element_ty))


class _FP4VSAAttnQAT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, q2k_idx, q2k_num, kv_valid, quantize, quantize_pv, two_level_p, first_block_max_floor,
                sm_scale, high_prec_o, key_bias=None, q_group=None, need_backward=False):  # type: ignore[override]
        batch, heads, seq_len, dim = q.shape
        if seq_len % BLOCK or dim not in (64, 128):
            raise ValueError(f"need L % {BLOCK} == 0 and D in (64, 128), got L={seq_len}, D={dim}")
        n_blocks = seq_len // BLOCK
        if q2k_idx.shape[:3] != (batch, heads, n_blocks):
            raise ValueError(f"q2k_idx {tuple(q2k_idx.shape)} does not match {(batch, heads, n_blocks)}")
        if quantize:
            q_hat, k_hat, v_hat = fake_quant_qkv_triton(q, k, v, quantize_v=quantize_pv)
        else:
            q_hat, k_hat, v_hat = (t.to(torch.bfloat16).contiguous() for t in (q, k, v))
        q2k_idx = q2k_idx.to(torch.int32).contiguous()
        q2k_num = q2k_num.to(torch.int32).contiguous()
        has_valid = kv_valid is not None
        kv_valid_arg = kv_valid.to(torch.int32).contiguous() if has_valid else q2k_num
        width = q2k_idx.shape[-1]
        out = torch.empty_like(q_hat)
        out_hp = torch.empty_like(q_hat) if high_prec_o else out
        lse = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
        scale_log2 = sm_scale * LOG2E
        offset = LOG2_INV_TWO_LEVEL if two_level_p else 0.0
        floor = first_block_max_floor is not None
        # Only the P quantizer reads the kernel's QUANT path; QK-only mode runs it with exact softmax.
        quantize = bool(quantize and quantize_pv)
        # dV only reads these maxima for quantized P. Inference and QK-only
        # backward need no quadratic buffer; LSE is an unused pointer there.
        save_mrun = bool(need_backward and quantize)
        m_run = (torch.empty((batch, heads, n_blocks, width, BLOCK), device=q.device, dtype=torch.float32)
                 if save_mrun else lse)
        has_bias = key_bias is not None
        if has_bias:
            key_bias = key_bias.float().contiguous()
            q_group = (q_group if q_group is not None else torch.zeros(n_blocks, dtype=torch.int32,
                                                                       device=q.device)).to(torch.int32).contiguous()
        grid = (n_blocks * (BLOCK // _ROWS), batch * heads)
        _fwd_kernel[grid](q_hat, k_hat, v_hat, out, out_hp, lse, m_run, q2k_idx, q2k_num, kv_valid_arg,
                          key_bias if has_bias else lse, q_group if has_bias else q2k_num,
                          key_bias.shape[2] if has_bias else 1, seq_len, n_blocks, width, scale_log2, OFFSET=offset,
                          QUANT=bool(quantize), HAS_VALID=has_valid, HAS_BIAS=has_bias, FLOOR=floor, FLOOR_VALUE=float(first_block_max_floor or 0.0),
                          HIGH_PREC_O=bool(high_prec_o), SAVE_MRUN=save_mrun, D=dim, ROWS=_ROWS, BLOCK_N=BLOCK, num_warps=_WARPS,
                          num_stages=2)
        if need_backward:
            ctx.save_for_backward(q_hat, k_hat, v_hat, out_hp, lse, m_run, q2k_idx, q2k_num, kv_valid_arg)
        ctx.options = (bool(quantize), has_valid, floor, float(first_block_max_floor or 0.0), offset, sm_scale)
        ctx.has_bias = has_bias
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        if ctx.has_bias:
            raise NotImplementedError("the Q-smoothing score bias is an eval-only emulator option (no backward)")
        q_hat, k_hat, v_hat, out_hp, lse, m_run, q2k_idx, q2k_num, kv_valid = ctx.saved_tensors
        quantize, has_valid, floor, floor_value, offset, sm_scale = ctx.options
        batch, heads, seq_len, dim = q_hat.shape
        n_blocks = seq_len // BLOCK
        width = q2k_idx.shape[-1]
        grad_out = grad_out.to(torch.bfloat16).contiguous()
        delta = (grad_out.float() * out_hp.float()).sum(-1).contiguous()
        k2q_idx, k2q_slot, k2q_num = invert_block_lists(q2k_idx, q2k_num, n_blocks)
        dq = torch.empty_like(q_hat)
        dk = torch.empty_like(k_hat)
        dv = torch.empty_like(v_hat)
        scale_log2 = sm_scale * LOG2E
        grid = (n_blocks * (BLOCK // _ROWS), batch * heads)
        _bwd_dq_kernel[grid](q_hat, k_hat, v_hat, grad_out, lse, delta, dq, q2k_idx, q2k_num, kv_valid, seq_len,
                             n_blocks, width, scale_log2, sm_scale, HAS_VALID=has_valid, D=dim, ROWS=_ROWS,
                             BLOCK_N=BLOCK, num_warps=_WARPS, num_stages=2)
        _bwd_dkdv_kernel[grid](q_hat, k_hat, v_hat, grad_out, lse, delta, m_run, dk, dv, k2q_idx, k2q_slot, k2q_num,
                               q2k_num, kv_valid, seq_len, n_blocks, n_blocks, width, scale_log2, sm_scale,
                               OFFSET=offset, QUANT=quantize, HAS_VALID=has_valid, FLOOR=floor,
                               FLOOR_VALUE=floor_value, D=dim, ROWS=_ROWS, BLOCK_N=BLOCK, num_warps=_WARPS,
                               num_stages=2)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


def _hadamard(dim: int, device: torch.device) -> torch.Tensor:
    """Normalized Sylvester Hadamard matrix ``[dim, dim]`` (orthogonal, symmetric), fp32."""
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    if h.shape[0] != dim:
        raise ValueError(f"Hadamard rotation needs a power-of-two head dim, got {dim}")
    return h / math.sqrt(dim)


def fp8_per_token_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """E4M3 fake quant with one fp32 scale per token (row), ``amax / 448``; fp32 out."""
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True) / 448.0
    inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    return (xf * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale


def mxfp8_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """MXFP8 fake quant along the last dim: E4M3 values, one E8M0 (power-of-two) scale per 32; fp32 out.

    The scale is ``2**ceil(log2(amax / 448))``, bit-identical to ``flashinfer.mxfp8_quantize`` (the FA4
    MXFP8 Q/K operands) on sm_100.
    """
    if x.shape[-1] % 32:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of 32")
    xf = x.float().unflatten(-1, (-1, 32))
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)).clamp(-127.0, 127.0))
    return ((xf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale).flatten(-2)


def _query_groups(n_blocks: int, num_prefix_tiles: int | None, segmented: bool, device: torch.device) -> torch.Tensor:
    """Per query tile: 0 for dense-prefix tiles, 1 for video tiles (all 0 unless ``segmented``)."""
    groups = torch.zeros(n_blocks, dtype=torch.int32, device=device)
    if segmented:
        groups[int(num_prefix_tiles or 0):] = 1
    return groups


def _smooth_queries(q: torch.Tensor, k: torch.Tensor, groups: torch.Tensor,
                    n_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``q - mu_g`` per query group and the exact correction ``bias[g, j] = mu_g . k_j`` (``[B, H, G, L]``)."""
    row_group = groups.repeat_interleave(BLOCK).long()
    qf = q.float()
    mus = []
    for g in range(n_groups):
        rows = row_group == g
        mus.append(qf[:, :, rows].mean(dim=-2) if bool(rows.any()) else qf.new_zeros(qf.shape[:2] + qf.shape[-1:]))
    mu = torch.stack(mus, dim=2)  # [B, H, G, D]
    bias = torch.einsum("bhgd,bhld->bhgl", mu, k.float())
    return qf - mu[:, :, row_group], bias


def fp4_vsa_attn_qat(q: torch.Tensor,
                     k: torch.Tensor,
                     v: torch.Tensor,
                     q2k_idx: torch.Tensor,
                     q2k_num: torch.Tensor,
                     kv_valid: torch.Tensor | None = None,
                     *,
                     quantize: bool = True,
                     quantize_pv: bool = True,
                     two_level_p: bool = False,
                     smooth_k: bool = False,
                     first_block_max_floor: float | None = None,
                     sm_scale: float | None = None,
                     high_prec_o: bool = True,
                     hadamard: bool = False,
                     smooth_q: str | None = None,
                     prefix_precision: str | None = None,
                     num_prefix_tiles: int | None = None) -> torch.Tensor:
    """Block-sparse attention with the deployed FP4 numerics in the forward and STE gradients.

    ``q``/``k``/``v`` are ``[B, H, L, D]`` (L a multiple of 128, tile-ordered,
    pad rows zero); the lists are ``vsa_tile_mask_to_fp4_blocks(mask, 128, vbs)``
    outputs (``q2k_quad`` must be None: 128-token tiles map 1:1 onto blocks).
    ``smooth_k`` subtracts each head's token mean from K first (exact for
    softmax, differentiable through autograd). ``quantize_pv=False`` quantizes
    Q/K only and keeps P and V in BF16 (the FA4 NVFP4-QK + BF16-PV kernel).
    Returns ``[B, H, L, D]`` BF16.

    Q/K-only error-reduction options (eval studies; the last three are forward-only):

    * ``hadamard``: rotate Q and K by a normalized 128x128 Hadamard matrix
      (after K smoothing) before NVFP4 quantization. Exact in real arithmetic.
    * ``smooth_q``: ``"global"`` subtracts each head's mean query, ``"segment"``
      one mean for the dense-prefix query tiles and one for the video tiles;
      the exact correction ``mu_q . k_j`` is added to every score as a
      high-precision per-key bias.
    * ``prefix_precision``: ``"bf16"``, ``"fp8"`` (E4M3, per-token scales) or
      ``"mxfp8"`` (E4M3, power-of-two scale per 32 channels, the FA4 kernel's
      MXFP8 operands) Q/K for the first ``num_prefix_tiles`` query tiles (text,
      reference and audio rows), as a second call over the same lists; video
      rows stay FP4.
    """
    qk_only = quantize and not quantize_pv
    if qk_only and (two_level_p or first_block_max_floor is not None):
        raise ValueError("two_level_p and first_block_max_floor shape the P quantizer; QK-only mode has none")
    # ``hadamard`` and ``smooth_q`` rewrite Q/K, which only the QK-only route reads.
    # ``prefix_precision`` also works with FP4 P.V, so the FP4 call below takes the
    # caller's P quantizer settings rather than assuming the QK-only ones.
    if (hadamard or smooth_q) and not qk_only:
        raise ValueError("hadamard / smooth_q are Q/K-only (quantize_pv=False) options")
    if smooth_q not in (None, "global", "segment"):
        raise ValueError(f"smooth_q must be None, 'global' or 'segment', got {smooth_q!r}")
    if prefix_precision not in (None, "bf16", "fp8", "mxfp8"):
        raise ValueError(f"prefix_precision must be None, 'bf16', 'fp8' or 'mxfp8', got {prefix_precision!r}")
    if (smooth_q == "segment" or prefix_precision) and num_prefix_tiles is None:
        raise ValueError("smooth_q='segment' and prefix_precision need num_prefix_tiles")
    if smooth_k:
        k = k - k.mean(dim=-2, keepdim=True)
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    # Autograd.Function.forward runs with grad disabled even during training.
    # Capture the caller's grad mode here, including no_grad on grad-enabled inputs.
    need_backward = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))
    if not (hadamard or smooth_q or prefix_precision):
        return _FP4VSAAttnQAT.apply(q, k, v, q2k_idx, q2k_num, kv_valid, quantize, quantize_pv, two_level_p,
                                    first_block_max_floor, scale, high_prec_o, None, None, need_backward)
    n_blocks = q.shape[2] // BLOCK
    hp_tiles = min(int(num_prefix_tiles or 0), n_blocks) if prefix_precision else 0
    num_fp4 = q2k_num
    if hp_tiles:
        tile = torch.arange(n_blocks, device=q2k_num.device)
        num_fp4 = q2k_num.masked_fill(tile < hp_tiles, 0)
    q_lo, k_lo = q, k
    if hadamard:
        rot = _hadamard(q.shape[-1], q.device)
        q_lo, k_lo = q.float() @ rot, k.float() @ rot
    bias = groups = None
    if smooth_q:
        n_groups = 2 if smooth_q == "segment" else 1
        groups = _query_groups(n_blocks, num_prefix_tiles, smooth_q == "segment", q.device)
        q_lo, bias = _smooth_queries(q_lo, k_lo, groups, n_groups)
    out = _FP4VSAAttnQAT.apply(q_lo, k_lo, v, q2k_idx, num_fp4, kv_valid, quantize, quantize_pv, two_level_p,
                               first_block_max_floor, scale, high_prec_o, bias, groups, need_backward)
    if not hp_tiles:
        return out
    num_hp = q2k_num - num_fp4
    hp_quant = {"bf16": lambda t: t, "fp8": fp8_per_token_fake_quant, "mxfp8": mxfp8_fake_quant}[prefix_precision]
    q_hp, k_hp = hp_quant(q), hp_quant(k)
    out_hp = _FP4VSAAttnQAT.apply(q_hp, k_hp, v, q2k_idx, num_hp, kv_valid, False, True, False, None, scale,
                                  high_prec_o, None, None, need_backward)
    return out + out_hp


__all__ = ["fake_quant_qkv_triton", "fp4_vsa_attn_qat", "fp8_per_token_fake_quant", "mxfp8_fake_quant"]
