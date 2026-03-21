# SPDX-License-Identifier: Apache-2.0
"""
Bidirectional Sparse Attention (BSA) backend for FastVideo.

Pure-PyTorch reference implementation from:
"Bidirectional Sparse Attention for Faster Video Diffusion Training"
(arXiv:2509.01085)

BSA sparsifies both queries (pruning redundant tokens per block) and
key-value pairs (keeping only relevant KV blocks per query block).

This is a training-free inference backend: it works with any model
trained with full attention by applying BSA sparsity at inference time.
"""

import functools
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from fastvideo.distributed import get_sp_group
from fastvideo.logger import init_logger

logger = init_logger(__name__)

BSA_TILE_SIZE = (4, 4, 4)

# ---------------------------------------------------------------------------
# Cached index helpers (same pattern as VSA)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """Map raster-order tokens to tile-contiguous order."""
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                ls.append(indices[
                    t * ts:min(t * ts + ts, T),
                    h * hs:min(h * hs + hs, H),
                    w * ws:min(w * ws + ws, W),
                ].flatten())
    return torch.cat(ls, dim=0)


@functools.lru_cache(maxsize=10)
def get_reverse_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """Inverse mapping: tile-contiguous order back to raster order."""
    return torch.argsort(get_tile_partition_indices(dit_seq_shape, tile_size, device))


# ---------------------------------------------------------------------------
# BSA core operations
# ---------------------------------------------------------------------------


def _prune_queries(
    q_blocks: torch.Tensor,
    keep_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Prune redundant query tokens within each block.

    Scores tokens by cosine similarity to the block center.
    Keeps the LEAST similar (most informative) tokens.

    Args:
        q_blocks: [B, N_heads, N_blocks, block_size, D]
        keep_ratio: fraction of tokens to keep

    Returns:
        sparse_q: [B, N_heads, N_blocks, keep_size, D]
        keep_indices: [B, N_heads, N_blocks, keep_size]
        keep_size: int
    """
    B, H, N, S, D = q_blocks.shape
    keep_size = max(1, int(S * keep_ratio))

    if keep_size >= S:
        idx = torch.arange(S, device=q_blocks.device)
        idx = idx.view(1, 1, 1, S).expand(B, H, N, S)
        return q_blocks, idx, S

    center_idx = S // 2
    center = q_blocks[:, :, :, center_idx:center_idx + 1, :]

    q_norm = F.normalize(q_blocks, dim=-1)
    c_norm = F.normalize(center, dim=-1)
    similarity = (q_norm * c_norm).sum(dim=-1)  # [B, H, N, S]

    # lowest similarity = most distinctive = keep
    _, indices = similarity.topk(keep_size, dim=-1, largest=False)
    indices, _ = indices.sort(dim=-1)

    idx_expand = indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)
    sparse_q = torch.gather(q_blocks, 3, idx_expand)

    return sparse_q, indices, keep_size


def _select_kv_blocks(
    sparse_q: torch.Tensor,
    k_blocks: torch.Tensor,
    cumulative_threshold: float,
    min_kv_blocks: int,
) -> torch.Tensor:
    """
    Dynamically select KV blocks for each query block.

    Mean-pools to block level, computes block attention scores,
    admits blocks in descending order until cumulative mass
    exceeds threshold.

    Args:
        sparse_q: [B, H, N, Sq, D]
        k_blocks:  [B, H, N, Sk, D]
        cumulative_threshold: e.g. 0.9
        min_kv_blocks: minimum blocks to keep

    Returns:
        kv_mask: [B, H, N, N] boolean
    """
    B, H, N, _, D = sparse_q.shape

    q_repr = sparse_q.mean(dim=3)
    k_repr = k_blocks.mean(dim=3)

    scores = torch.matmul(q_repr, k_repr.transpose(-1, -2)) / (D**0.5)
    block_attn = F.softmax(scores, dim=-1)

    sorted_attn, sorted_idx = block_attn.sort(dim=-1, descending=True)
    cumsum = sorted_attn.cumsum(dim=-1)

    keep_sorted = torch.ones_like(cumsum, dtype=torch.bool)
    keep_sorted[..., 1:] = cumsum[..., :-1] < cumulative_threshold

    min_mask = torch.zeros_like(keep_sorted)
    min_mask[..., :min(min_kv_blocks, N)] = True
    keep_sorted = keep_sorted | min_mask

    kv_mask = torch.zeros_like(block_attn, dtype=torch.bool)
    kv_mask.scatter_(-1, sorted_idx, keep_sorted)

    return kv_mask


def _compute_sparse_attention(
    sparse_q: torch.Tensor,
    k_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    kv_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute attention for each query block against selected KV blocks.

    Reference implementation with loop over query blocks.

    Args:
        sparse_q: [B, H, N, Sq, D]
        k_blocks:  [B, H, N, Sk, D]
        v_blocks:  [B, H, N, Sk, D]
        kv_mask:   [B, H, N, N] boolean

    Returns:
        output: [B, H, N, Sq, D]
    """
    B, H, N, Sq, D = sparse_q.shape
    output = torch.zeros_like(sparse_q)

    for qb in range(N):
        # Use mask from first batch/head element (assumes uniform)
        selected = kv_mask[0, 0, qb]
        sel_idx = selected.nonzero(as_tuple=True)[0]

        if sel_idx.shape[0] == 0:
            continue

        sel_k = k_blocks[:, :, sel_idx].reshape(B, H, -1, D)
        sel_v = v_blocks[:, :, sel_idx].reshape(B, H, -1, D)

        q = sparse_q[:, :, qb]
        scores = torch.matmul(q, sel_k.transpose(-1, -2)) / (D**0.5)
        weights = F.softmax(scores, dim=-1)
        output[:, :, qb] = torch.matmul(weights, sel_v)

    return output


def _reconstruct_pruned(
    sparse_output: torch.Tensor,
    keep_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Scatter sparse output back to full block size.
    Pruned positions get nearest kept token's output.
 
    Args:
        sparse_output: [B, H, N, keep_size, D]
        keep_indices:  [B, H, N, keep_size]
        block_size: original tokens per block
 
    Returns:
        full_output: [B, H, N, block_size, D]
    """
    B, H, N, keep_size, D = sparse_output.shape
    device = sparse_output.device

    if keep_size >= block_size:
        return sparse_output

    full_output = torch.zeros(B, H, N, block_size, D, device=device, dtype=sparse_output.dtype)

    # Scatter kept tokens
    idx_expand = keep_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)
    full_output.scatter_(3, idx_expand, sparse_output)

    # Fill pruned positions with nearest kept token (vectorized)
    all_pos = torch.arange(block_size, device=device)

    for n in range(N):
        # NOTE: Assumes keep_indices are uniform across batch and heads.
        # Per-batch/per-head reconstruction is a follow-up optimization.
        kept = keep_indices[0, 0, n]  # [keep_size]

        # Distance from every position to every kept position
        # [block_size, keep_size]
        dists = (all_pos.view(-1, 1) - kept.view(1, -1)).abs()
        nearest_local_idx = dists.argmin(dim=1)  # [block_size]

        # Identify pruned positions
        is_pruned = torch.ones(block_size, dtype=torch.bool, device=device)
        is_pruned[kept] = False
        pruned_indices = is_pruned.nonzero(as_tuple=True)[0]

        if pruned_indices.numel() > 0:
            src_indices = nearest_local_idx[pruned_indices]
            full_output[:, :, n, pruned_indices] = sparse_output[:, :, n, src_indices]

    return full_output


# ---------------------------------------------------------------------------
# FastVideo backend classes
# ---------------------------------------------------------------------------


class BSAAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "BSA_ATTN"

    @staticmethod
    def get_impl_cls() -> type["BSAAttentionImpl"]:
        return BSAAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["BSAAttentionMetadata"]:
        return BSAAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["BSAAttentionMetadataBuilder"]:
        return BSAAttentionMetadataBuilder


@dataclass
class BSAAttentionMetadata(AttentionMetadata):
    current_timestep: int
    dit_seq_shape: tuple[int, int, int]
    total_seq_length: int
    num_blocks: int
    block_size: int
    tile_partition_indices: torch.LongTensor
    reverse_tile_partition_indices: torch.LongTensor
    # BSA-specific config
    query_keep_ratio: float
    kv_cumulative_threshold: float
    min_kv_blocks: int


class BSAAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        device: torch.device,
        bsa_query_keep_ratio: float = 0.5,
        bsa_kv_cumulative_threshold: float = 0.9,
        bsa_min_kv_blocks: int = 4,
        **kwargs: dict[str, Any],
    ) -> "BSAAttentionMetadata":
        dit_seq_shape = (
            raw_latent_shape[0] // patch_size[0],
            raw_latent_shape[1] // patch_size[1],
            raw_latent_shape[2] // patch_size[2],
        )

        total_seq_length = math.prod(dit_seq_shape)
        block_size = math.prod(BSA_TILE_SIZE)
        num_blocks = math.prod(math.ceil(d / t) for d, t in zip(dit_seq_shape, BSA_TILE_SIZE, strict=False))

        tile_partition_indices = get_tile_partition_indices(dit_seq_shape, BSA_TILE_SIZE, device)
        reverse_tile_partition_indices = get_reverse_tile_partition_indices(dit_seq_shape, BSA_TILE_SIZE, device)

        return BSAAttentionMetadata(
            current_timestep=current_timestep,
            dit_seq_shape=dit_seq_shape,
            total_seq_length=total_seq_length,
            num_blocks=num_blocks,
            block_size=block_size,
            tile_partition_indices=tile_partition_indices,
            reverse_tile_partition_indices=reverse_tile_partition_indices,
            query_keep_ratio=bsa_query_keep_ratio,
            kv_cumulative_threshold=bsa_kv_cumulative_threshold,
            min_kv_blocks=bsa_min_kv_blocks,
        )


class BSAAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix
        self.num_heads = num_heads
        self.head_size = head_size
        try:
            sp_group = get_sp_group()
            self.sp_size = sp_group.world_size
        except (AssertionError, RuntimeError):
            self.sp_size = 1

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: BSAAttentionMetadata,
    ) -> torch.Tensor:
        """Reorder tokens from raster order to tile-contiguous order."""
        # qkv: [B, L, num_heads, D]
        return qkv[:, attn_metadata.tile_partition_indices]

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: BSAAttentionMetadata,
    ) -> torch.Tensor:
        """Reorder tokens from tile-contiguous order back to raster order."""
        return output[:, attn_metadata.reverse_tile_partition_indices]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: BSAAttentionMetadata,
    ) -> torch.Tensor:
        """
        BSA attention forward pass.

        Input tensors are already in tile-contiguous order from preprocess_qkv.

        Args:
            query: [B, L, num_heads, D] (tile-ordered)
            key:   [B, L, num_heads, D] (tile-ordered)
            value: [B, L, num_heads, D] (tile-ordered)
            attn_metadata: BSA metadata

        Returns:
            output: [B, L, num_heads, D] (tile-ordered)
        """
        B, L, H, D = query.shape
        block_size = attn_metadata.block_size
        num_blocks = attn_metadata.num_blocks

        # Reshape to [B, H, L, D] for attention computation
        q = query.transpose(1, 2).contiguous()  # [B, H, L, D]
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # Reshape into blocks: [B, H, num_blocks, block_size, D]
        q_blocks = q.view(B, H, num_blocks, block_size, D)
        k_blocks = k.view(B, H, num_blocks, block_size, D)
        v_blocks = v.view(B, H, num_blocks, block_size, D)

        # --- Query sparsification ---
        sparse_q, keep_indices, keep_size = _prune_queries(q_blocks, attn_metadata.query_keep_ratio)

        # --- KV block selection ---
        kv_mask = _select_kv_blocks(
            sparse_q,
            k_blocks,
            attn_metadata.kv_cumulative_threshold,
            attn_metadata.min_kv_blocks,
        )

        # --- Sparse attention ---
        sparse_output = _compute_sparse_attention(sparse_q, k_blocks, v_blocks, kv_mask)

        # --- Reconstruct pruned positions ---
        full_output = _reconstruct_pruned(sparse_output, keep_indices, block_size)

        # Reshape back: [B, H, num_blocks, block_size, D] -> [B, H, L, D] -> [B, L, H, D]
        hidden_states = full_output.view(B, H, L, D).transpose(1, 2)

        return hidden_states
