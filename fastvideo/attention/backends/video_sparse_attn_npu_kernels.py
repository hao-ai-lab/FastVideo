# SPDX-License-Identifier: Apache-2.0
"""Forward adaptation of FastVideo's Apache-2.0 Triton VSA kernels.

Compression/selection: fastvideo-kernel/python/fastvideo_kernel/triton_kernels/
fused_compress_topk.py at 6743f4b4b76634f7854942eddf49a68fe65e66f0.
Sparse forward: block_sparse_attn_triton.py at the same revision, with logical
64-token indexing separated from physical query/key scheduling. Its original
attribution is: Flash Attention v2 algorithm from Tri Dao
(https://tridao.me/publications/flash2/flash2.pdf); credits: OpenAI kernel team.
This module exposes forward only and retains the upstream BF16 exponential cast.
"""
import logging

import torch

import triton

import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fused_block_mean_kernel(
    X_ptr,
    Out_ptr,
    VBS_ptr,
    stride_x_bh,
    stride_x_seq,
    stride_o_bh,
    stride_o_blk,
    num_blocks,
    BLOCK_ELEMENTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Fused block mean: one program computes mean of one block for one (b,h).

    X is viewed as [B*H, num_blocks*BLOCK_ELEMENTS, HEAD_DIM] contiguous.
    Out is [B*H, num_blocks, HEAD_DIM] contiguous.
    2D load + parallel tl.sum reduction, accumulates in fp32.
    """
    block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)

    if block_idx >= num_blocks:
        return

    vbs = tl.load(VBS_ptr + block_idx).to(tl.float32)

    x_base = X_ptr + bh_idx * stride_x_bh + block_idx * BLOCK_ELEMENTS * stride_x_seq

    row_offsets = tl.arange(0, BLOCK_ELEMENTS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    offsets = row_offsets[:, None] * stride_x_seq + dim_offsets[None, :]
    block_data = tl.load(x_base + offsets).to(tl.float32)
    acc = tl.sum(block_data, axis=0) / vbs

    out_base = Out_ptr + bh_idx * stride_o_bh + block_idx * stride_o_blk + dim_offsets
    tl.store(out_base, acc.to(OUTPUT_DTYPE))


_TORCH_TO_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _fused_block_mean_fwd(
    x: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    block_elements: int,
) -> torch.Tensor:
    B, H, seq_len, D = x.shape
    num_blocks = seq_len // block_elements
    assert seq_len % block_elements == 0

    x = x.contiguous()
    out = torch.empty(B, H, num_blocks, D, dtype=x.dtype, device=x.device)

    x_flat = x.view(B * H, seq_len, D)
    out_flat = out.view(B * H, num_blocks, D)

    grid = (num_blocks, B * H)

    _fused_block_mean_kernel[grid](
        x_flat,
        out_flat,
        variable_block_sizes,
        x_flat.stride(0),
        x_flat.stride(1),
        out_flat.stride(0),
        out_flat.stride(1),
        num_blocks,
        BLOCK_ELEMENTS=block_elements,
        HEAD_DIM=D,
        OUTPUT_DTYPE=_TORCH_TO_TRITON_DTYPE[x.dtype],
    )

    return out


@triton.jit
def _fused_topk_mask_kernel(
    Scores_ptr,
    Mask_ptr,
    stride_s_bh,
    stride_s_q,
    stride_s_kv,
    stride_m_bh,
    stride_m_q,
    stride_m_kv,
    kv_blocks: tl.constexpr,
    topk: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    """Build topk boolean mask via randomized pivot selection (quickselect-style).

    For each (b,h,q_block) row: find the k-th largest score using iterative
    pivot-based partitioning, then build mask by comparing against threshold.

    Grid: (num_q_blocks, B * H)
    """
    q_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)

    kv_offsets = tl.arange(0, KV_BLOCK_SIZE)
    score_base = Scores_ptr + bh_idx * stride_s_bh + q_idx * stride_s_q
    mask_base = Mask_ptr + bh_idx * stride_m_bh + q_idx * stride_m_q

    valid_mask = kv_offsets < kv_blocks
    scores = tl.load(score_base + kv_offsets * stride_s_kv, mask=valid_mask, other=-float("inf"))
    scores_f32 = scores.to(tl.float32)

    # Binary search for threshold: find value T such that count(scores > T) <= topk
    # and count(scores >= T) >= topk
    # Exclude -inf from lo so the binary search can converge when masked
    # scores are present (mid = (-inf + hi) * 0.5 = -inf would stall).
    finite_mask = valid_mask & (scores_f32 > float("-inf"))
    lo = tl.min(tl.where(finite_mask, scores_f32, float("inf")), axis=0)
    hi = tl.max(tl.where(valid_mask, scores_f32, float("-inf")), axis=0)
    # If all valid scores are -inf, lo > hi; threshold stays at -inf and
    # the tie-breaking logic below selects the first topk positions.
    lo = tl.minimum(lo, hi)

    # 32 iterations of fp32 bisection give range-relative resolution (hi-lo)/2^32.
    # For VSA's softmax-input scores (q_c@k_c/sqrt(d), O(1) magnitude, range < ~10),
    # this resolves to ~2e-9, well below the bf16 ULP at that magnitude (~8e-3),
    # so the threshold converges exactly to the k-th bf16 score value and the
    # > / == comparisons below are exact.
    for _i in range(32):
        mid = (lo + hi) * 0.5
        count_ge = tl.sum(((scores_f32 >= mid) & valid_mask).to(tl.int32), axis=0)
        lo = tl.where(count_ge >= topk, mid, lo)
        hi = tl.where(count_ge >= topk, hi, mid)

    # lo is our threshold: count(scores >= lo) >= topk
    threshold = lo
    above_threshold = scores_f32 > threshold
    at_threshold = scores_f32 == threshold
    n_above = tl.sum(above_threshold.to(tl.int32), axis=0)
    n_needed_at_thresh = topk - n_above

    at_thresh_cumsum = tl.cumsum(at_threshold.to(tl.int32), axis=0)
    at_thresh_selected = at_threshold & (at_thresh_cumsum <= n_needed_at_thresh)

    final_mask = above_threshold | at_thresh_selected

    tl.store(mask_base + kv_offsets * stride_m_kv, final_mask, mask=valid_mask)


MAX_KV_BLOCK_SIZE = 4096


def _pytorch_topk_mask_fallback(
    scores: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    topk_idx = torch.topk(scores, topk, dim=-1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)


def fused_topk_mask(
    scores: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Build topk boolean mask from scores using fused Triton kernel.

    Falls back to PyTorch when kv_blocks exceeds the Triton block size
    limit (MAX_KV_BLOCK_SIZE) to avoid compilation failures.

    Args:
        scores: [B, H, q_blocks, kv_blocks] block-level attention scores
        topk: number of top blocks to select per q-block

    Returns:
        mask: [B, H, q_blocks, kv_blocks] bool tensor with exactly topk True per row
    """
    B, H, q_blocks, kv_blocks = scores.shape
    topk = min(topk, kv_blocks)

    KV_BLOCK_SIZE = triton.next_power_of_2(kv_blocks)
    if KV_BLOCK_SIZE > MAX_KV_BLOCK_SIZE:
        logger.debug(
            "fused_topk_mask: kv_blocks=%d exceeds Triton limit %d, "
            "falling back to PyTorch topk (slower)",
            kv_blocks,
            MAX_KV_BLOCK_SIZE,
        )
        return _pytorch_topk_mask_fallback(scores, topk)

    mask = torch.zeros(B, H, q_blocks, kv_blocks, dtype=torch.bool, device=scores.device)

    scores_flat = scores.contiguous().view(B * H, q_blocks, kv_blocks)
    mask_flat = mask.view(B * H, q_blocks, kv_blocks)

    grid = (q_blocks, B * H)

    _fused_topk_mask_kernel[grid](
        scores_flat,
        mask_flat,
        scores_flat.stride(0),
        scores_flat.stride(1),
        scores_flat.stride(2),
        mask_flat.stride(0),
        mask_flat.stride(1),
        mask_flat.stride(2),
        kv_blocks=kv_blocks,
        topk=topk,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )

    return mask


# The measured forward uses fixed scheduling and no autotuner or backward.


@triton.jit
def _attn_fwd_sparse(Q, K, V, sm_scale, q2k_index, q2k_num, max_kv_blks, variable_block_sizes, M, Out, stride_qz,
                     stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh,
                     stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX_Q, N_CTX_KV,
                     HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    """
    Forward with a single normalization over all selected valid logical blocks.
    Production dispatch fixes BLOCK_M=BLOCK_N=64. No backward is implemented.
    """
    q_blk = tl.program_id(0)
    off_hz = tl.program_id(1)
    b = off_hz // H
    h = off_hz % H
    q_tiles = N_CTX_Q // 64
    meta_base = (b * H + h) * q_tiles + q_blk * BLOCK_M // 64
    kv_blocks = tl.load(q2k_num + meta_base)
    kv_ptr = q2k_index + meta_base * max_kv_blks
    q_off = b.to(tl.int64) * stride_qz + h.to(tl.int64) * stride_qh
    k_off = b.to(tl.int64) * stride_kz + h.to(tl.int64) * stride_kh
    v_off = b.to(tl.int64) * stride_vz + h.to(tl.int64) * stride_vh
    o_off = b.to(tl.int64) * stride_oz + h.to(tl.int64) * stride_oh
    Q_ptr = tl.make_block_ptr(base=Q + q_off,
                              shape=(N_CTX_Q, HEAD_DIM),
                              strides=(stride_qm, stride_qk),
                              offsets=(q_blk * BLOCK_M, 0),
                              block_shape=(BLOCK_M, HEAD_DIM),
                              order=(1, 0))
    K_base = tl.make_block_ptr(base=K + k_off,
                               shape=(HEAD_DIM, N_CTX_KV),
                               strides=(stride_kk, stride_kn),
                               offsets=(0, 0),
                               block_shape=(HEAD_DIM, BLOCK_N),
                               order=(0, 1))
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_base = tl.make_block_ptr(base=V + v_off,
                               shape=(N_CTX_KV, HEAD_DIM),
                               strides=(stride_vk, stride_vn),
                               offsets=(0, 0),
                               block_shape=(BLOCK_N, HEAD_DIM),
                               order=v_order)
    O_ptr = tl.make_block_ptr(base=Out + o_off,
                              shape=(N_CTX_Q, HEAD_DIM),
                              strides=(stride_om, stride_on),
                              offsets=(q_blk * BLOCK_M, 0),
                              block_shape=(BLOCK_M, HEAD_DIM),
                              order=(1, 0))
    offs_m = q_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.full([BLOCK_M], -float('inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_ptr)
    for i in range(0, kv_blocks):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        block_size = tl.load(variable_block_sizes + kv_idx)
        for sub in range(0, 64 // BLOCK_N):
            K_ptr = tl.advance(K_base, (0, kv_idx * 64 + sub * BLOCK_N))
            V_ptr = tl.advance(V_base, (kv_idx * 64 + sub * BLOCK_N, 0))
            k = tl.load(K_ptr)
            qk = tl.dot(q, k)
            mask = tl.arange(0, BLOCK_N) < block_size - sub * BLOCK_N
            qk = tl.where(mask[None, :], qk, -float('inf'))
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v = tl.load(V_ptr)
            acc = tl.dot(p.to(tl.bfloat16), v, acc)
            m_i = m_ij
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(M + off_hz * N_CTX_Q + offs_m, m_i)
    tl.store(O_ptr, acc.to(Out.type.element_ty))
