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

try:
    from fastvideo.attention.utils.flash_attn_no_pad import (
        flash_attn_varlen_func_impl, )
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    try:
        from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_impl
        FLASH_ATTN_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_AVAILABLE = False

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

    Handles per-batch and per-head KV masks correctly.
    Uses flash_attn_varlen_func when available on GPU.
    Falls back to pure-PyTorch reference on CPU.

    Args:
        sparse_q: [B, H, N, Sq, D]
        k_blocks:  [B, H, N, Sk, D]
        v_blocks:  [B, H, N, Sk, D]
        kv_mask:   [B, H, N, N] boolean (per-batch, per-head)

    Returns:
        output: [B, H, N, Sq, D]
    """
    if FLASH_ATTN_AVAILABLE and sparse_q.is_cuda:
        return _compute_sparse_attention_flash(sparse_q, k_blocks, v_blocks, kv_mask)
    else:
        return _compute_sparse_attention_reference(sparse_q, k_blocks, v_blocks, kv_mask)


def _compute_sparse_attention_reference(
    sparse_q: torch.Tensor,
    k_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    kv_mask: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch fallback with per-batch, per-head mask support."""
    B, H, N, Sq, D = sparse_q.shape
    output = torch.zeros_like(sparse_q)

    for b in range(B):
        for h in range(H):
            for qb in range(N):
                selected = kv_mask[b, h, qb]  # [N] boolean
                sel_idx = selected.nonzero(as_tuple=True)[0]

                if sel_idx.shape[0] == 0:
                    continue

                # [num_sel * Sk, D]
                sel_k = k_blocks[b, h, sel_idx].reshape(-1, D)
                sel_v = v_blocks[b, h, sel_idx].reshape(-1, D)

                q = sparse_q[b, h, qb]  # [Sq, D]
                scores = torch.matmul(q, sel_k.transpose(-1, -2)) / (D**0.5)
                weights = F.softmax(scores, dim=-1)
                output[b, h, qb] = torch.matmul(weights, sel_v)

    return output


def _compute_sparse_attention_flash(
    sparse_q: torch.Tensor,
    k_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    kv_mask: torch.Tensor,
) -> torch.Tensor:
    """
    FlashAttention implementation with per-batch, per-head mask support.

    Strategy: check if all heads share the same mask. If so, use a single
    FlashAttention call per batch (fast path). If not, process each head
    separately (correct path).

    Args:
        sparse_q: [B, H, N, Sq, D]
        k_blocks:  [B, H, N, Sk, D]
        v_blocks:  [B, H, N, Sk, D]
        kv_mask:   [B, H, N, N] boolean

    Returns:
        output: [B, H, N, Sq, D]
    """
    B, H, N, Sq, D = sparse_q.shape
    Sk = k_blocks.shape[3]
    device = sparse_q.device
    output = torch.zeros_like(sparse_q)

    for b in range(B):
        # Check if all heads share the same mask for this batch element
        # Compare each head's mask to head 0's mask
        head0_mask = kv_mask[b, 0]  # [N, N]
        all_heads_same = all(torch.equal(kv_mask[b, h], head0_mask) for h in range(1, H))

        if all_heads_same:
            # Fast path: all heads share the same mask, single FA call
            _flash_attn_single_mask(
                sparse_q[b],
                k_blocks[b],
                v_blocks[b],
                head0_mask,
                output[b],
                H,
                N,
                Sq,
                Sk,
                D,
                device,
            )
        else:
            # Per-head path: process each head individually
            for h in range(H):
                head_mask = kv_mask[b, h]  # [N, N]
                # Process single head: squeeze head dim, run FA, put back
                _flash_attn_single_head(
                    sparse_q[b, h],
                    k_blocks[b, h],
                    v_blocks[b, h],
                    head_mask,
                    output,
                    b,
                    h,
                    N,
                    Sq,
                    Sk,
                    D,
                    device,
                )

    return output


def _flash_attn_single_mask(
    sparse_q_b: torch.Tensor,  # [H, N, Sq, D]
    k_blocks_b: torch.Tensor,  # [H, N, Sk, D]
    v_blocks_b: torch.Tensor,  # [H, N, Sk, D]
    mask: torch.Tensor,  # [N, N] boolean
    output_b: torch.Tensor,  # [H, N, Sq, D] (modified in-place)
    H: int,
    N: int,
    Sq: int,
    Sk: int,
    D: int,
    device: torch.device,
) -> None:
    """Run FlashAttention for all heads sharing the same KV mask."""
    q_list = []
    k_list = []
    v_list = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    active_blocks = []

    for qb in range(N):
        selected = mask[qb]  # [N] boolean
        sel_idx = selected.nonzero(as_tuple=True)[0]

        if sel_idx.shape[0] == 0:
            continue

        active_blocks.append(qb)
        num_kv_tokens = sel_idx.shape[0] * Sk

        # [H, Sq, D] -> [Sq, H, D]
        q_block = sparse_q_b[:, qb].permute(1, 0, 2)
        q_list.append(q_block)

        # [H, num_sel, Sk, D] -> [num_kv_tokens, H, D]
        sel_k = k_blocks_b[:, sel_idx].permute(1, 2, 0, 3).reshape(num_kv_tokens, H, D)
        sel_v = v_blocks_b[:, sel_idx].permute(1, 2, 0, 3).reshape(num_kv_tokens, H, D)
        k_list.append(sel_k)
        v_list.append(sel_v)

        cu_seqlens_q.append(cu_seqlens_q[-1] + Sq)
        cu_seqlens_k.append(cu_seqlens_k[-1] + num_kv_tokens)

    if not q_list:
        return

    flat_q = torch.cat(q_list, dim=0)
    flat_k = torch.cat(k_list, dim=0)
    flat_v = torch.cat(v_list, dim=0)

    # Compute max_seqlen_k from the Python list before moving to GPU to
    # avoid a `.item()` round-trip that would force a host/device sync.
    max_seqlen_q = Sq
    max_seqlen_k = max(b - a for a, b in zip(cu_seqlens_k[:-1], cu_seqlens_k[1:], strict=False))

    cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)

    orig_dtype = flat_q.dtype
    compute_dtype = orig_dtype
    if compute_dtype not in (torch.float16, torch.bfloat16):
        compute_dtype = torch.bfloat16
        flat_q = flat_q.to(compute_dtype)
        flat_k = flat_k.to(compute_dtype)
        flat_v = flat_v.to(compute_dtype)

    flat_out = flash_attn_varlen_func_impl(
        flat_q,
        flat_k,
        flat_v,
        cu_seqlens_q_t,
        cu_seqlens_k_t,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
    )

    if compute_dtype != orig_dtype:
        flat_out = flat_out.to(orig_dtype)

    idx = 0
    for qb in active_blocks:
        block_out = flat_out[idx:idx + Sq]  # [Sq, H, D]
        output_b[:, qb] = block_out.permute(1, 0, 2)  # [H, Sq, D]
        idx += Sq


def _flash_attn_single_head(
    sparse_q_bh: torch.Tensor,  # [N, Sq, D]
    k_blocks_bh: torch.Tensor,  # [N, Sk, D]
    v_blocks_bh: torch.Tensor,  # [N, Sk, D]
    mask: torch.Tensor,  # [N, N] boolean
    output: torch.Tensor,  # [B, H, N, Sq, D] (modified in-place)
    b: int,
    h: int,
    N: int,
    Sq: int,
    Sk: int,
    D: int,
    device: torch.device,
) -> None:
    """Run FlashAttention for a single head with its own KV mask."""
    q_list = []
    k_list = []
    v_list = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    active_blocks = []

    for qb in range(N):
        selected = mask[qb]
        sel_idx = selected.nonzero(as_tuple=True)[0]

        if sel_idx.shape[0] == 0:
            continue

        active_blocks.append(qb)
        num_kv_tokens = sel_idx.shape[0] * Sk

        # [Sq, D] -> [Sq, 1, D] (single head)
        q_block = sparse_q_bh[qb].unsqueeze(1)
        q_list.append(q_block)

        # [num_sel, Sk, D] -> [num_kv_tokens, 1, D]
        sel_k = k_blocks_bh[sel_idx].reshape(num_kv_tokens, 1, D)
        sel_v = v_blocks_bh[sel_idx].reshape(num_kv_tokens, 1, D)
        k_list.append(sel_k)
        v_list.append(sel_v)

        cu_seqlens_q.append(cu_seqlens_q[-1] + Sq)
        cu_seqlens_k.append(cu_seqlens_k[-1] + num_kv_tokens)

    if not q_list:
        return

    flat_q = torch.cat(q_list, dim=0)
    flat_k = torch.cat(k_list, dim=0)
    flat_v = torch.cat(v_list, dim=0)

    # Compute max_seqlen_k from the Python list before moving to GPU to
    # avoid a `.item()` round-trip that would force a host/device sync.
    max_seqlen_q = Sq
    max_seqlen_k = max(b - a for a, b in zip(cu_seqlens_k[:-1], cu_seqlens_k[1:], strict=False))

    cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)

    orig_dtype = flat_q.dtype
    compute_dtype = orig_dtype
    if compute_dtype not in (torch.float16, torch.bfloat16):
        compute_dtype = torch.bfloat16
        flat_q = flat_q.to(compute_dtype)
        flat_k = flat_k.to(compute_dtype)
        flat_v = flat_v.to(compute_dtype)

    flat_out = flash_attn_varlen_func_impl(
        flat_q,
        flat_k,
        flat_v,
        cu_seqlens_q_t,
        cu_seqlens_k_t,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
    )

    if compute_dtype != orig_dtype:
        flat_out = flat_out.to(orig_dtype)

    idx = 0
    for qb in active_blocks:
        block_out = flat_out[idx:idx + Sq]  # [Sq, 1, D]
        output[b, h, qb] = block_out.squeeze(1)  # [Sq, D]
        idx += Sq


def _reconstruct_pruned(
    sparse_output: torch.Tensor,
    keep_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Scatter sparse output back to full block size.
    Pruned positions get nearest kept token's output.

    Handles per-batch, per-head indices correctly.

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

    for b in range(B):
        for h in range(H):
            for n in range(N):
                kept = keep_indices[b, h, n]  # [keep_size]

                # Distance from every position to every kept position
                dists = (all_pos.view(-1, 1) - kept.view(1, -1)).abs()
                nearest_local_idx = dists.argmin(dim=1)  # [block_size]

                # Identify pruned positions
                is_pruned = torch.ones(block_size, dtype=torch.bool, device=device)
                is_pruned[kept] = False
                pruned_indices = is_pruned.nonzero(as_tuple=True)[0]

                if pruned_indices.numel() > 0:
                    src_indices = nearest_local_idx[pruned_indices]
                    full_output[b, h, n, pruned_indices] = sparse_output[b, h, n, src_indices]

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
        # Ensure patching does not drop tokens silently.
        assert all(r % p == 0 for r, p in zip(raw_latent_shape, patch_size, strict=False)), (
            "raw_latent_shape must be divisible by patch_size for BSA", )

        dit_seq_shape = (
            raw_latent_shape[0] // patch_size[0],
            raw_latent_shape[1] // patch_size[1],
            raw_latent_shape[2] // patch_size[2],
        )

        total_seq_length = math.prod(dit_seq_shape)
        block_size = math.prod(BSA_TILE_SIZE)
        # Require exact tiling to avoid reshape failures later.
        assert all(d % t == 0 for d, t in zip(dit_seq_shape, BSA_TILE_SIZE, strict=False)), (
            "dit_seq_shape must be divisible by BSA_TILE_SIZE", )
        num_blocks = total_seq_length // block_size

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
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise ValueError("BSA backend does not support grouped-query attention")
        if causal:
            raise ValueError("BSA backend is bidirectional; causal=True is unsupported")
        if softmax_scale is not None:
            expected_scale = 1.0 / math.sqrt(self.head_size)
            if not math.isclose(softmax_scale, expected_scale, rel_tol=1e-4, abs_tol=1e-5):
                raise ValueError("softmax_scale must be default (1/sqrt(d)) for BSA")
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
        assert num_blocks * block_size == L, "Sequence length must match tiling"

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
