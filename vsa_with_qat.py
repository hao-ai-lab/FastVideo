# SPDX-License-Identifier: Apache-2.0
#
# Triton VSA (block-sparse attention) kernels with QAT support.
# Combines:
# - Sparse metadata flow from fastvideo_kernel.block_sparse_attn
# - Sparse forward/backward kernel structure from block_sparse_attn_triton
# - QAT fake-quant behavior from qat_attn

from __future__ import annotations

import math
from typing import Tuple

import torch
import triton
import triton.language as tl

from quant_utils import fake_quantize, fake_quantize_kv, fake_quantize_q


@triton.jit
def _attn_fwd_sparse_qat(
    Q,
    K,
    V,
    sm_scale,
    q2k_index,
    q2k_num,
    max_kv_blks,
    variable_block_sizes,
    M,
    Out,
    HighPrecOut,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_hoz,
    stride_hoh,
    stride_hom,
    stride_hon,
    Z,
    H,
    N_CTX_Q,
    N_CTX_KV,
    Q_TILES,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_QAT: tl.constexpr,
    fake_quant_P: tl.constexpr,
    two_level_quant_P: tl.constexpr,
    use_global_sf_P: tl.constexpr,
):
    q_blk = tl.program_id(0)
    off_hz = tl.program_id(1)
    b = off_hz // H
    h = off_hz % H
    meta_base = ((b * H + h) * Q_TILES + q_blk)

    kv_blocks = tl.load(q2k_num + meta_base)
    kv_ptr = q2k_index + meta_base * max_kv_blks

    q_off = (b.to(tl.int64) * stride_qz + h.to(tl.int64) * stride_qh)
    kv_off = (b.to(tl.int64) * stride_kz + h.to(tl.int64) * stride_kh)

    Q_ptr = tl.make_block_ptr(
        base=Q + q_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(q_blk * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    K_base = tl.make_block_ptr(
        base=K + kv_off,
        shape=(HEAD_DIM, N_CTX_KV),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )

    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_base = tl.make_block_ptr(
        base=V + (b.to(tl.int64) * stride_vz + h.to(tl.int64) * stride_vh),
        shape=(N_CTX_KV, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )

    O_ptr = tl.make_block_ptr(
        base=Out + q_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(q_blk * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    HighPrecO_ptr = tl.make_block_ptr(
        base=HighPrecOut + (b.to(tl.int64) * stride_hoz + h.to(tl.int64) * stride_hoh),
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_hom, stride_hon),
        offsets=(q_blk * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = q_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    q_valid = offs_m < N_CTX_Q
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    if IS_QAT:
        high_prec_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    else:
        high_prec_acc = acc
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_ptr)
    q = tl.where(q_valid[:, None], q, 0.0)

    for i in range(0, kv_blocks):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        block_size = tl.load(variable_block_sizes + kv_idx)
        K_ptr = tl.advance(K_base, (0, kv_idx * BLOCK_N))
        V_ptr = tl.advance(V_base, (kv_idx * BLOCK_N, 0))

        k = tl.load(K_ptr)
        qk = tl.dot(q, k)
        col_valid = tl.arange(0, BLOCK_N) < block_size
        qk = tl.where(col_valid[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])

        p_for_acc = p
        p_for_high_prec = p
        if IS_QAT and fake_quant_P:
            p_for_acc, p_for_lse = fake_quantize(
                src_tensor=p,
                valid_src_mask=col_valid[None, :] & q_valid[:, None],
                BLOCK_SIZE_OUT_DIM=BLOCK_M,
                BLOCK_SIZE_QUANT_DIM=BLOCK_N,
                dst_dtype=p.dtype,
                two_level_quant_P=two_level_quant_P,
                use_global_sf=use_global_sf_P,
            )
            l_ij = tl.sum(p_for_lse, 1)
            p_for_high_prec = p_for_lse
        else:
            l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        if IS_QAT:
            high_prec_acc = high_prec_acc * alpha[:, None]

        v = tl.load(V_ptr)
        acc = tl.dot(p_for_acc.to(tl.bfloat16), v, acc)
        if IS_QAT:
            high_prec_acc = tl.dot(p_for_high_prec.to(tl.bfloat16), v, high_prec_acc)
        m_i = m_ij

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    if IS_QAT:
        high_prec_acc = high_prec_acc / l_i[:, None]
    tl.store(M + off_hz * N_CTX_Q + offs_m, m_i, mask=q_valid)
    tl.store(O_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))
    if IS_QAT:
        tl.store(HighPrecO_ptr, high_prec_acc.to(HighPrecOut.type.element_ty), boundary_check=(0, 1))


@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,
    Delta,
    Z,
    H,
    N_CTX_Q,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    valid = off_m < N_CTX_Q
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX_Q + off_m[:, None] * HEAD_DIM + off_n[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX_Q + off_m[:, None] * HEAD_DIM + off_n[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX_Q + off_m, delta, mask=valid)


@triton.jit
def _attn_bwd_dkdv_sparse_qat(
    dk,
    dv,
    Q,
    k,
    v,
    sm_scale,
    DO,
    M,
    D,
    k2q_index,
    k2q_num,
    max_q_blks,
    variable_block_sizes,
    stride_tok_q,
    stride_tok_kv,
    stride_d_q,
    stride_d_kv,
    H,
    N_CTX_Q,
    KV_TILES,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n,
    start_m,
    num_steps,
    IS_QAT: tl.constexpr,
    fake_quant_P: tl.constexpr,
    two_level_quant_P: tl.constexpr,
    use_global_sf_P: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok_q + offs_k[:, None] * stride_d_q
    do_ptrs = DO + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q

    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    step_m = BLOCK_M1

    kv_blk = tl.program_id(0)
    off_hz = tl.program_id(2)
    b = off_hz // H
    h = off_hz % H
    meta_base = ((b * H + h) * KV_TILES + kv_blk)

    q_blocks = tl.load(k2q_num + meta_base)
    q_ptr = k2q_index + meta_base * max_q_blks
    block_size = tl.load(variable_block_sizes + kv_blk)
    row_valid = tl.arange(0, BLOCK_N1) < block_size

    for blk_idx in range(q_blocks * 2):
        block_sparse_offset = (tl.load(q_ptr + blk_idx // 2).to(tl.int32) * 2 + blk_idx % 2) * step_m

        qT = tl.load(qT_ptrs + block_sparse_offset * stride_tok_q)
        offs_m = start_m + block_sparse_offset + tl.arange(0, BLOCK_M1)
        q_valid = offs_m < N_CTX_Q
        m = tl.load(M + offs_m, mask=q_valid, other=0.0)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        pT = tl.where(row_valid[:, None] & q_valid[None, :], pT, 0.0)

        do = tl.load(do_ptrs + block_sparse_offset * stride_tok_q)

        p_for_dv = pT
        if IS_QAT and fake_quant_P:
            p_for_dv, _ = fake_quantize(
                src_tensor=pT,
                valid_src_mask=row_valid[:, None] & q_valid[None, :],
                BLOCK_SIZE_OUT_DIM=BLOCK_N1,
                BLOCK_SIZE_QUANT_DIM=BLOCK_M1,
                dst_dtype=pT.dtype,
                two_level_quant_P=two_level_quant_P,
                use_global_sf=use_global_sf_P,
            )

        dv += tl.dot(p_for_dv.to(tl.bfloat16), do)

        Di = tl.load(D + offs_m, mask=q_valid, other=0.0)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.bfloat16)
        dk += tl.dot(dsT, tl.trans(qT))
    return dk, dv


@triton.jit
def _attn_bwd_dq_sparse(
    dq,
    q,
    K,
    V,
    do,
    m,
    D,
    q2k_index,
    q2k_num,
    max_kv_blks,
    variable_block_sizes,
    stride_tok_q,
    stride_tok_kv,
    stride_d_q,
    stride_d_kv,
    H,
    N_CTX_Q,
    N_CTX_KV,
    Q_TILES,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_Q_BLOCK_MASK: tl.constexpr,
    start_m,
    start_n,
    num_steps,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok_kv + offs_k[:, None] * stride_d_kv
    vT_ptrs = V + offs_n[None, :] * stride_tok_kv + offs_k[:, None] * stride_d_kv
    Di = tl.load(D + offs_m, mask=offs_m < N_CTX_Q, other=0.0)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    step_n = BLOCK_N2

    q_blk = tl.program_id(0)
    off_hz = tl.program_id(2)
    b = off_hz // H
    h = off_hz % H
    meta_base = ((b * H + h) * Q_TILES + q_blk)

    kv_blocks = tl.load(q2k_num + meta_base)
    kv_ptr = q2k_index + meta_base * max_kv_blks
    block_size_q = tl.load(variable_block_sizes + q_blk)

    for blk_idx in range(kv_blocks * 2):
        kv_idx = tl.load(kv_ptr + blk_idx // 2).to(tl.int32)
        if USE_Q_BLOCK_MASK:
            col_valid = tl.arange(0, BLOCK_N2) < block_size_q.to(tl.int32)
        else:
            block_size = tl.load(variable_block_sizes + kv_idx)
            col_valid = tl.arange(0, BLOCK_N2) < block_size.to(tl.int32)
        block_sparse_offset = (kv_idx * 2 + blk_idx % 2) * step_n * stride_tok_kv
        kT = tl.load(kT_ptrs + block_sparse_offset)
        vT = tl.load(vT_ptrs + block_sparse_offset)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        p = tl.where(col_valid[None, :], p, 0.0)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        dq += tl.dot(ds.to(tl.bfloat16), tl.trans(kT))
    return dq


@triton.jit
def _attn_bwd_sparse_qat_dkdv(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DK,
    DV,
    M,
    D,
    k2q_index,
    k2q_num,
    max_q_blks,
    variable_block_sizes,
    stride_z_q,
    stride_h_q,
    stride_tok_q,
    stride_d_q,
    stride_z_kv,
    stride_h_kv,
    stride_tok_kv,
    stride_d_kv,
    H,
    N_CTX_Q,
    N_CTX_KV,
    KV_TILES,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_QAT: tl.constexpr,
    fake_quant_P: tl.constexpr,
    two_level_quant_P: tl.constexpr,
    use_global_sf_P: tl.constexpr,
):
    bhid = tl.program_id(2)
    pid = tl.program_id(0)
    off_q = (stride_h_q * (bhid % H) + stride_z_q * (bhid // H)).to(tl.int64)
    off_kv = (stride_h_kv * (bhid % H) + stride_z_kv * (bhid // H)).to(tl.int64)
    off_q_ctx = (bhid * N_CTX_Q).to(tl.int64)

    Q += off_q
    K += off_kv
    V += off_kv
    DO += off_q
    DK += off_kv
    DV += off_kv
    M += off_q_ctx
    D += off_q_ctx

    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    kv_valid = offs_n < N_CTX_KV

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    k = tl.load(K + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None], other=0.0)
    v = tl.load(V + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv, mask=kv_valid[:, None], other=0.0)

    dk, dv = _attn_bwd_dkdv_sparse_qat(
        dk,
        dv,
        Q,
        k,
        v,
        sm_scale,
        DO,
        M,
        D,
        k2q_index,
        k2q_num,
        max_q_blks,
        variable_block_sizes,
        stride_tok_q,
        stride_tok_kv,
        stride_d_q,
        stride_d_kv,
        H,
        N_CTX_Q,
        KV_TILES,
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,
        start_n,
        0,
        (N_CTX_Q + BLOCK_M1 - 1) // BLOCK_M1,
        IS_QAT,
        fake_quant_P,
        two_level_quant_P,
        use_global_sf_P,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv
    tl.store(dv_ptrs, dv, mask=kv_valid[:, None])
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok_kv + offs_k[None, :] * stride_d_kv
    tl.store(dk_ptrs, dk, mask=kv_valid[:, None])


@triton.jit
def _attn_bwd_sparse_qat_dq(
    Q,
    K,
    V,
    DO,
    DQ,
    M,
    D,
    q2k_index,
    q2k_num,
    max_kv_blks,
    variable_block_sizes,
    stride_z_q,
    stride_h_q,
    stride_tok_q,
    stride_d_q,
    stride_z_kv,
    stride_h_kv,
    stride_tok_kv,
    stride_d_kv,
    H,
    N_CTX_Q,
    N_CTX_KV,
    Q_TILES,
    IS_SELF_ATTN: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    LN2 = 0.6931471824645996
    bhid = tl.program_id(2)
    pid = tl.program_id(0)
    off_q = (stride_h_q * (bhid % H) + stride_z_q * (bhid // H)).to(tl.int64)
    off_kv = (stride_h_kv * (bhid % H) + stride_z_kv * (bhid // H)).to(tl.int64)
    off_q_ctx = (bhid * N_CTX_Q).to(tl.int64)

    Q += off_q
    K += off_kv
    V += off_kv
    DO += off_q
    DQ += off_q
    M += off_q_ctx
    D += off_q_ctx

    offs_k = tl.arange(0, HEAD_DIM)
    start_m = pid * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    q_valid = offs_m < N_CTX_Q
    q = tl.load(Q + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q, mask=q_valid[:, None], other=0.0)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q, mask=q_valid[:, None], other=0.0)
    m = tl.load(M + offs_m, mask=q_valid, other=0.0)[:, None]

    dq = _attn_bwd_dq_sparse(
        dq,
        q,
        K,
        V,
        do,
        m,
        D,
        q2k_index,
        q2k_num,
        max_kv_blks,
        variable_block_sizes,
        stride_tok_q,
        stride_tok_kv,
        stride_d_q,
        stride_d_kv,
        H,
        N_CTX_Q,
        N_CTX_KV,
        Q_TILES,
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,
        IS_SELF_ATTN,
        start_m,
        0,
        (N_CTX_KV + BLOCK_N2 - 1) // BLOCK_N2,
    )
    dq_ptrs = DQ + offs_m[:, None] * stride_tok_q + offs_k[None, :] * stride_d_q
    dq *= LN2
    tl.store(dq_ptrs, dq, mask=q_valid[:, None])


def _map_to_index_torch(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    if block_map.dim() != 4:
        raise ValueError(f"block_map must be [B,H,Q,KV] (or [H,Q,KV]), got shape={tuple(block_map.shape)}")
    if block_map.dtype != torch.bool:
        block_map = block_map.to(torch.bool)

    B, H, Q, KV = block_map.shape
    index = torch.full((B, H, Q, KV), -1, dtype=torch.int32, device=block_map.device)
    num = torch.zeros((B, H, Q), dtype=torch.int32, device=block_map.device)

    for b in range(B):
        for h in range(H):
            for q in range(Q):
                kv_idx = torch.nonzero(block_map[b, h, q], as_tuple=False).flatten().to(torch.int32)
                n = int(kv_idx.numel())
                if n:
                    index[b, h, q, :n] = kv_idx
                num[b, h, q] = n
    return index, num


def _triton_vsa_qat_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    IS_QAT: bool = True,
    fake_quant_P: bool = True,
    two_level_quant_P: bool = False,
    use_global_sf_P: bool = False,
    use_high_prec_o: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N_CTX_Q, D = q.shape
    N_CTX_KV = k.shape[2]
    sm_scale = 1.0 / math.sqrt(D)
    max_kv_blks = q2k_index.shape[-1]
    assert N_CTX_Q % 64 == 0, f"N_CTX_Q must be a multiple of 64, but got {N_CTX_Q}"
    assert N_CTX_KV % 64 == 0, f"N_CTX_KV must be a multiple of 64, but got {N_CTX_KV}"
    q_tiles = N_CTX_Q // 64
    assert q2k_num.shape[-1] == q_tiles, f"shape mismatch: expected q_tiles={q_tiles}, got {q2k_num.shape[-1]}"

    o = torch.empty_like(q)
    if IS_QAT and use_high_prec_o:
        high_prec_o = torch.empty_like(q)
    else:
        high_prec_o = o
    M = torch.empty((B, H, N_CTX_Q), dtype=torch.float32, device=q.device)

    grid = lambda _: (triton.cdiv(N_CTX_Q, 64), B * H, 1)
    _attn_fwd_sparse_qat[grid](
        q,
        k,
        v,
        sm_scale,
        q2k_index,
        q2k_num,
        max_kv_blks,
        variable_block_sizes,
        M,
        o,
        high_prec_o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        high_prec_o.stride(0),
        high_prec_o.stride(1),
        high_prec_o.stride(2),
        high_prec_o.stride(3),
        B,
        H,
        N_CTX_Q,
        N_CTX_KV,
        q_tiles,
        HEAD_DIM=D,
        BLOCK_M=64,
        BLOCK_N=64,
        IS_QAT=IS_QAT,
        fake_quant_P=fake_quant_P,
        two_level_quant_P=two_level_quant_P,
        use_global_sf_P=use_global_sf_P,
    )
    o_for_bwd = high_prec_o if (IS_QAT and use_high_prec_o) else o
    return o, M, o_for_bwd


def triton_vsa_qat_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    IS_QAT: bool = True,
    fake_quant_P: bool = True,
    two_level_quant_P: bool = False,
    use_global_sf_P: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    o, M, _ = _triton_vsa_qat_forward_impl(
        q,
        k,
        v,
        q2k_index,
        q2k_num,
        variable_block_sizes,
        IS_QAT=IS_QAT,
        fake_quant_P=fake_quant_P,
        two_level_quant_P=two_level_quant_P,
        use_global_sf_P=use_global_sf_P,
        use_high_prec_o=True,
    )
    return o, M


def triton_vsa_qat_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    M: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    k2q_index: torch.Tensor,
    k2q_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    IS_QAT: bool = True,
    fake_quant_P: bool = True,
    two_level_quant_P: bool = False,
    use_global_sf_P: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert do.is_contiguous()

    B, H, N_CTX_Q, D = q.shape
    N_CTX_KV = k.shape[2]
    sm_scale = 1.0 / math.sqrt(D)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    PRE_BLOCK = 64
    assert N_CTX_Q % PRE_BLOCK == 0
    assert N_CTX_KV % 64 == 0
    pre_grid = (N_CTX_Q // PRE_BLOCK, B * H)
    delta = torch.empty_like(M)
    _attn_bwd_preprocess[pre_grid](
        o,
        do,
        delta,
        B,
        H,
        N_CTX_Q,
        BLOCK_M=PRE_BLOCK,
        HEAD_DIM=D,
    )

    max_q_blks = k2q_index.shape[-1]
    max_kv_blks = q2k_index.shape[-1]
    q_tiles = N_CTX_Q // 64
    kv_tiles = N_CTX_KV // 64
    is_self_attn = N_CTX_Q == N_CTX_KV

    arg_k = k * (sm_scale * 1.4426950408889634)
    grid_dkdv = (kv_tiles, 1, B * H)
    _attn_bwd_sparse_qat_dkdv[grid_dkdv](
        q,
        arg_k,
        v,
        sm_scale,
        do,
        dk,
        dv,
        M,
        delta,
        k2q_index,
        k2q_num,
        max_q_blks,
        variable_block_sizes,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        H,
        N_CTX_Q,
        N_CTX_KV,
        kv_tiles,
        BLOCK_M1=32,
        BLOCK_N1=64,
        HEAD_DIM=D,
        IS_QAT=IS_QAT,
        fake_quant_P=fake_quant_P,
        two_level_quant_P=two_level_quant_P,
        use_global_sf_P=use_global_sf_P,
    )

    grid_dq = (q_tiles, 1, B * H)
    _attn_bwd_sparse_qat_dq[grid_dq](
        q,
        arg_k,
        v,
        do,
        dq,
        M,
        delta,
        q2k_index,
        q2k_num,
        max_kv_blks,
        variable_block_sizes,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        H,
        N_CTX_Q,
        N_CTX_KV,
        q_tiles,
        IS_SELF_ATTN=is_self_attn,
        BLOCK_M2=64,
        BLOCK_N2=32,
        HEAD_DIM=D,
    )
    return dq, dk, dv


class _vsa_qat_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        use_qat_qkv: bool = True,
        IS_QAT: bool = True,
        fake_quant_P: bool = True,
        two_level_quant_P: bool = False,
        use_global_sf_P: bool = False,
        use_global_sf_QKV: bool = False,
        use_high_prec_o: bool = True,
    ):
        if k.shape[2] != v.shape[2]:
            raise RuntimeError("VSA Triton path requires k and v to share the same padded sequence length.")

        q2k_index, q2k_num = _map_to_index_torch(block_map.to(torch.bool))
        k2q_index, k2q_num = _map_to_index_torch(block_map.transpose(-1, -2).contiguous().to(torch.bool))

        q_in = q
        k_in = k
        v_in = v
        if IS_QAT:
            B, H, N_CTX_Q, D = q.shape
            _, _, N_CTX_KV, _ = k.shape
            BLOCK_M = 32
            BLOCK_N = 32
            fake_q = torch.empty_like(q)
            fake_k = torch.empty_like(k)
            fake_v = torch.empty_like(v)
            grid_q = (triton.cdiv(N_CTX_Q, BLOCK_M), B * H, 1)
            grid_kv = (triton.cdiv(N_CTX_KV, BLOCK_N), B * H, 1)
            fake_quantize_q[grid_q](
                q,
                fake_q,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                fake_q.stride(0),
                fake_q.stride(1),
                fake_q.stride(2),
                fake_q.stride(3),
                H,
                N_CTX_Q,
                BLOCK_M=BLOCK_M,
                HEAD_DIM=D,
                use_global_sf=use_global_sf_QKV,
            )
            fake_quantize_kv[grid_kv](
                k,
                v,
                fake_k,
                fake_v,
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                fake_k.stride(0),
                fake_k.stride(1),
                fake_k.stride(2),
                fake_k.stride(3),
                H,
                N_CTX_KV,
                BLOCK_N=BLOCK_N,
                HEAD_DIM=D,
                use_global_sf=use_global_sf_QKV,
            )
            q_in, k_in, v_in = fake_q, fake_k, fake_v

        o, M, o_for_bwd = _triton_vsa_qat_forward_impl(
            q_in.contiguous(),
            k_in.contiguous(),
            v_in.contiguous(),
            q2k_index.contiguous(),
            q2k_num.contiguous(),
            variable_block_sizes.contiguous(),
            IS_QAT=IS_QAT,
            fake_quant_P=fake_quant_P,
            two_level_quant_P=two_level_quant_P,
            use_global_sf_P=use_global_sf_P,
            use_high_prec_o=use_high_prec_o,
        )

        if IS_QAT:
            q_in = fake_q
            k_in = fake_k
            v_in = fake_v

        ctx.save_for_backward(
            q_in.contiguous(),
            k_in.contiguous(),
            v_in.contiguous(),
            o_for_bwd.contiguous(),
            M.contiguous(),
            q2k_index.contiguous(),
            q2k_num.contiguous(),
            k2q_index.contiguous(),
            k2q_num.contiguous(),
            variable_block_sizes.contiguous(),
        )
        ctx.IS_QAT = IS_QAT
        ctx.fake_quant_P = fake_quant_P
        ctx.two_level_quant_P = two_level_quant_P
        ctx.use_global_sf_P = use_global_sf_P
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        (
            q,
            k,
            v,
            o,
            M,
            q2k_index,
            q2k_num,
            k2q_index,
            k2q_num,
            variable_block_sizes,
        ) = ctx.saved_tensors

        dq, dk, dv = triton_vsa_qat_backward(
            do.contiguous(),
            q,
            k,
            v,
            o,
            M,
            q2k_index,
            q2k_num,
            k2q_index,
            k2q_num,
            variable_block_sizes,
            IS_QAT=ctx.IS_QAT,
            fake_quant_P=ctx.fake_quant_P,
            two_level_quant_P=ctx.two_level_quant_P,
            use_global_sf_P=ctx.use_global_sf_P,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def vsa_qat_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    use_qat_qkv: bool = True,
    IS_QAT: bool = True,
    fake_quant_P: bool = True,
    two_level_quant_P: bool = False,
    use_global_sf_P: bool = False,
    use_global_sf_QKV: bool = False,
    use_high_prec_o: bool = True,
) -> torch.Tensor:
    return _vsa_qat_attention.apply(
        q,
        k,
        v,
        block_map,
        variable_block_sizes,
        use_qat_qkv,
        IS_QAT,
        fake_quant_P,
        two_level_quant_P,
        use_global_sf_P,
        use_global_sf_QKV,
        use_high_prec_o,
    )


def build_sparse_index_pair(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q2k_index, q2k_num = _map_to_index_torch(block_map)
    k2q_index, k2q_num = _map_to_index_torch(block_map.transpose(-1, -2).contiguous())
    return q2k_index, q2k_num, k2q_index, k2q_num


def triton_vsa_qat_forward_from_block_map(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    IS_QAT: bool = True,
    fake_quant_P: bool = True,
    two_level_quant_P: bool = False,
    use_global_sf_P: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q2k_index, q2k_num, _, _ = build_sparse_index_pair(block_map.to(torch.bool))
    return triton_vsa_qat_forward(
        q,
        k,
        v,
        q2k_index.contiguous(),
        q2k_num.contiguous(),
        variable_block_sizes.contiguous(),
        IS_QAT=IS_QAT,
        fake_quant_P=fake_quant_P,
        two_level_quant_P=two_level_quant_P,
        use_global_sf_P=use_global_sf_P,
    )


def triton_vsa_qat_backward_from_block_map(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    M: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    IS_QAT: bool = True,
    fake_quant_P: bool = True,
    two_level_quant_P: bool = False,
    use_global_sf_P: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q2k_index, q2k_num, k2q_index, k2q_num = build_sparse_index_pair(block_map.to(torch.bool))
    return triton_vsa_qat_backward(
        do,
        q,
        k,
        v,
        o,
        M,
        q2k_index.contiguous(),
        q2k_num.contiguous(),
        k2q_index.contiguous(),
        k2q_num.contiguous(),
        variable_block_sizes.contiguous(),
        IS_QAT=IS_QAT,
        fake_quant_P=fake_quant_P,
        two_level_quant_P=two_level_quant_P,
        use_global_sf_P=use_global_sf_P,
    )


