# SPDX-License-Identifier: Apache-2.0
# Adapted from TurboDiffusion SLA implementation
# Copyright (c) 2025 by SLA team.
#
# Citation:
# @article{zhang2025sla,
#   title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention},
#   author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and
#           Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and
#           Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
#   journal={arXiv preprint arXiv:2509.24006},
#   year={2025}
# }

import torch
import triton
import triton.language as tl


@triton.jit
def compress_kernel(
    X, XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    x = tl.load(X + x_offset + offs_l[:, None] * D + offs_d[None, :], mask=offs_l[:, None] < L)

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x: torch.Tensor, BLK: int) -> torch.Tensor:
    """Mean pool tensor along sequence dimension with block size BLK."""
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def get_block_map(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_ratio: float,
    BLKQ: int = 64,
    BLKK: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute sparse block map for attention based on QK similarity.
    
    Args:
        q: Query tensor of shape (B, H, L, D)
        k: Key tensor of shape (B, H, L, D)
        topk_ratio: Ratio of key blocks to attend to (0-1)
        BLKQ: Query block size
        BLKK: Key block size
        
    Returns:
        sparse_map: Binary mask of shape (B, H, num_q_blocks, num_k_blocks)
        lut: Top-k indices of shape (B, H, num_q_blocks, topk)
        topk: Number of key blocks selected
    """
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)  # smooth-k technique from SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


def get_cuda_arch(device_index: int) -> str:
    """Get CUDA architecture string for the given device."""
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"
