"""
Fused Triton kernels for VSA compress (block mean) and topk mask construction.

Replaces the multi-kernel PyTorch pipeline:
  Original compress: .view() -> .float() -> .sum(dim=3) -> / vbs -> .to(bf16)
  Original topk:     torch.topk() -> zeros() -> scatter_()

With single-pass fused kernels:
  fused_block_mean:  read bf16, accumulate fp32, div by vbs, write bf16
  fused_topk_mask:   read scores, find k-th value, write bool mask
"""

import torch
import triton
import triton.language as tl


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
):
    """Fused block mean: one program computes mean of one block for one (b,h).

    X is viewed as [B*H, num_blocks*BLOCK_ELEMENTS, HEAD_DIM] contiguous.
    Out is [B*H, num_blocks, HEAD_DIM] contiguous.
    Accumulates in fp32, outputs in original dtype.
    """
    block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)

    if block_idx >= num_blocks:
        return

    vbs = tl.load(VBS_ptr + block_idx).to(tl.float32)

    x_base = X_ptr + bh_idx * stride_x_bh + block_idx * BLOCK_ELEMENTS * stride_x_seq

    dim_offsets = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for i in range(BLOCK_ELEMENTS):
        row_ptr = x_base + i * stride_x_seq + dim_offsets
        x_val = tl.load(row_ptr).to(tl.float32)
        acc += x_val

    acc = acc / vbs

    out_base = Out_ptr + bh_idx * stride_o_bh + block_idx * stride_o_blk + dim_offsets
    tl.store(out_base, acc.to(tl.bfloat16))


@triton.jit
def _fused_block_mean_bwd_kernel(
    GradOut_ptr,
    GradX_ptr,
    VBS_ptr,
    stride_go_bh,
    stride_go_blk,
    stride_gx_bh,
    stride_gx_seq,
    num_blocks,
    BLOCK_ELEMENTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Backward of block mean: broadcast grad_out / vbs to each token in the block.

    Mirrors the forward kernel: one program per (block, bh).
    GradOut is [B*H, num_blocks, HEAD_DIM].
    GradX  is [B*H, num_blocks*BLOCK_ELEMENTS, HEAD_DIM].
    """
    block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)

    if block_idx >= num_blocks:
        return

    vbs = tl.load(VBS_ptr + block_idx).to(tl.float32)

    go_base = GradOut_ptr + bh_idx * stride_go_bh + block_idx * stride_go_blk
    dim_offsets = tl.arange(0, HEAD_DIM)
    grad_val = tl.load(go_base + dim_offsets).to(tl.float32) / vbs

    gx_base = GradX_ptr + bh_idx * stride_gx_bh + block_idx * BLOCK_ELEMENTS * stride_gx_seq
    grad_out_cast = grad_val.to(tl.bfloat16)
    for i in range(BLOCK_ELEMENTS):
        tl.store(gx_base + i * stride_gx_seq + dim_offsets, grad_out_cast)


def _fused_block_mean_bwd(
    grad_output: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    block_elements: int,
) -> torch.Tensor:
    B, H, num_blocks, D = grad_output.shape
    seq_len = num_blocks * block_elements

    grad_x = torch.empty(B, H, seq_len, D, dtype=grad_output.dtype, device=grad_output.device)

    go_flat = grad_output.contiguous().view(B * H, num_blocks, D)
    gx_flat = grad_x.view(B * H, seq_len, D)

    grid = (num_blocks, B * H)

    _fused_block_mean_bwd_kernel[grid](
        go_flat, gx_flat, variable_block_sizes,
        go_flat.stride(0), go_flat.stride(1),
        gx_flat.stride(0), gx_flat.stride(1),
        num_blocks,
        BLOCK_ELEMENTS=block_elements,
        HEAD_DIM=D,
    )

    return grad_x


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
        x_flat, out_flat, variable_block_sizes,
        x_flat.stride(0), x_flat.stride(1),
        out_flat.stride(0), out_flat.stride(1),
        num_blocks,
        BLOCK_ELEMENTS=block_elements,
        HEAD_DIM=D,
    )

    return out


class _FusedBlockMeanAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, variable_block_sizes, block_elements):
        ctx.save_for_backward(variable_block_sizes)
        ctx.block_elements = block_elements
        return _fused_block_mean_fwd(x, variable_block_sizes, block_elements)

    @staticmethod
    def backward(ctx, grad_output):
        variable_block_sizes, = ctx.saved_tensors
        block_elements = ctx.block_elements
        return _fused_block_mean_bwd(grad_output, variable_block_sizes, block_elements), None, None


def fused_block_mean(
    x: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    block_elements: int,
) -> torch.Tensor:
    """Compute block-wise mean with fp32 accumulation, fused in one kernel.

    Forward: fused Triton kernel (bf16 read → fp32 accumulate → div → bf16 write).
    Backward: broadcasts grad_output / vbs back to each token position.

    Args:
        x: [B, H, seq_len, D] in bf16
        variable_block_sizes: [num_blocks] number of valid tokens per block
        block_elements: tokens per block (e.g. 64)

    Returns:
        [B, H, num_blocks, D] in bf16
    """
    return _FusedBlockMeanAutograd.apply(x, variable_block_sizes, block_elements)


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
    # Use +inf/-inf sentinels so min/max ignore padding positions
    lo = tl.min(tl.where(valid_mask, scores_f32, float("inf")), axis=0)
    hi = tl.max(tl.where(valid_mask, scores_f32, float("-inf")), axis=0)

    for _i in range(32):
        mid = (lo + hi) * 0.5
        count_ge = tl.sum(((scores_f32 >= mid) & valid_mask).to(tl.int32), axis=0)
        # If count >= topk, threshold is at or above mid
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


def fused_topk_mask(
    scores: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Build topk boolean mask from scores using fused Triton kernel.

    Args:
        scores: [B, H, q_blocks, kv_blocks] block-level attention scores
        topk: number of top blocks to select per q-block

    Returns:
        mask: [B, H, q_blocks, kv_blocks] bool tensor with exactly topk True per row
    """
    B, H, q_blocks, kv_blocks = scores.shape
    topk = min(topk, kv_blocks)

    mask = torch.zeros(B, H, q_blocks, kv_blocks, dtype=torch.bool, device=scores.device)

    KV_BLOCK_SIZE = triton.next_power_of_2(kv_blocks)

    scores_flat = scores.contiguous().view(B * H, q_blocks, kv_blocks)
    mask_flat = mask.view(B * H, q_blocks, kv_blocks)

    grid = (q_blocks, B * H)

    _fused_topk_mask_kernel[grid](
        scores_flat, mask_flat,
        scores_flat.stride(0), scores_flat.stride(1), scores_flat.stride(2),
        mask_flat.stride(0), mask_flat.stride(1), mask_flat.stride(2),
        kv_blocks=kv_blocks,
        topk=topk,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )

    return mask
