# SPDX-License-Identifier: Apache-2.0
"""
Tests for BSA (Bidirectional Sparse Attention) backend.

Run with: python -m pytest tests/test_bsa.py -v
"""

import torch
import torch.nn.functional as F
import pytest

from fastvideo.attention.backends.bsa_attn import (
    BSAAttentionBackend,
    BSAAttentionMetadata,
    BSAAttentionMetadataBuilder,
    BSAAttentionImpl,
    _prune_queries,
    _select_kv_blocks,
    get_tile_partition_indices,
    get_reverse_tile_partition_indices,
)

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkv(B, H, L, D):
    """Create random Q, K, V tensors in [B, H, L, D] layout."""
    q = torch.randn(B, H, L, D, device=DEVICE)
    k = torch.randn(B, H, L, D, device=DEVICE)
    v = torch.randn(B, H, L, D, device=DEVICE)
    return q, k, v


def full_attention(q, k, v):
    """Reference full attention on [B, H, L, D] tensors."""
    D = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


# ---------------------------------------------------------------------------
# Backend registration tests
# ---------------------------------------------------------------------------

class TestBackendRegistration:
    """Test that BSA backend classes are properly defined."""

    def test_backend_name(self):
        assert BSAAttentionBackend.get_name() == "BSA_ATTN"

    def test_backend_impl_cls(self):
        assert BSAAttentionBackend.get_impl_cls() is BSAAttentionImpl

    def test_backend_metadata_cls(self):
        assert BSAAttentionBackend.get_metadata_cls() is BSAAttentionMetadata

    def test_backend_builder_cls(self):
        assert BSAAttentionBackend.get_builder_cls() is BSAAttentionMetadataBuilder

    def test_metadata_builder(self):
        builder = BSAAttentionMetadataBuilder()
        builder.prepare()
        metadata = builder.build(
            current_timestep=0,
            raw_latent_shape=(32, 64, 64),
            patch_size=(1, 2, 2),
            device=torch.device("cpu"),
        )
        assert isinstance(metadata, BSAAttentionMetadata)
        assert metadata.dit_seq_shape == (32, 32, 32)
        assert metadata.block_size == 64
        assert metadata.query_keep_ratio == 0.5

# ---------------------------------------------------------------------------
# Tile partition index tests
# ---------------------------------------------------------------------------

class TestTileIndices:
    """Test tile partition index generation."""

    def test_roundtrip(self):
        shape = (8, 8, 8)
        tile = (4, 4, 4)
        device = torch.device("cpu")

        fwd = get_tile_partition_indices(shape, tile, device)
        rev = get_reverse_tile_partition_indices(shape, tile, device)

        x = torch.arange(512)
        assert torch.allclose(x[fwd][rev], x)

    def test_index_coverage(self):
        shape = (8, 8, 8)
        tile = (4, 4, 4)
        device = torch.device("cpu")

        fwd = get_tile_partition_indices(shape, tile, device)
        assert fwd.shape[0] == 512
        assert set(fwd.tolist()) == set(range(512))


# ---------------------------------------------------------------------------
# Query pruning tests
# ---------------------------------------------------------------------------

class TestQueryPruning:
    """Test query sparsification logic."""

    def test_keep_all(self):
        B, H, N, S, D = 1, 2, 4, 64, 32
        q = torch.randn(B, H, N, S, D, device=DEVICE)

        sparse_q, indices, keep_size = _prune_queries(q, keep_ratio=1.0)

        assert sparse_q.shape == q.shape
        assert keep_size == S

    def test_keep_half(self):
        B, H, N, S, D = 1, 2, 8, 64, 32
        q = torch.randn(B, H, N, S, D, device=DEVICE)

        sparse_q, indices, keep_size = _prune_queries(q, keep_ratio=0.5)

        assert sparse_q.shape == (B, H, N, 32, D)
        assert indices.shape == (B, H, N, 32)
        assert keep_size == 32

    def test_indices_in_range(self):
        B, H, N, S, D = 1, 2, 4, 64, 32
        q = torch.randn(B, H, N, S, D, device=DEVICE)

        _, indices, _ = _prune_queries(q, keep_ratio=0.5)

        assert indices.min() >= 0
        assert indices.max() < S

    def test_indices_sorted(self):
        B, H, N, S, D = 1, 2, 4, 64, 32
        q = torch.randn(B, H, N, S, D, device=DEVICE)

        _, indices, _ = _prune_queries(q, keep_ratio=0.5)

        for b in range(B):
            for h in range(H):
                for n in range(N):
                    idx = indices[b, h, n]
                    assert torch.all(idx[1:] >= idx[:-1]), "Indices not sorted"

    def test_keeps_distinctive_tokens(self):
        B, H, N, D = 1, 1, 1, 32
        S = 8

        center_vec = torch.ones(D)
        q = center_vec.unsqueeze(0).repeat(S, 1)
        q[0] = -center_vec
        q[7] = torch.randn(D) * 10

        q = q.view(B, H, N, S, D).float()

        sparse_q, indices, _ = _prune_queries(q, keep_ratio=0.25)

        kept = set(indices[0, 0, 0].tolist())
        assert 0 in kept, "Token 0 (most distinctive) should be kept"
        assert 7 in kept, "Token 7 (most distinctive) should be kept"

    def test_minimum_one_token(self):
        B, H, N, S, D = 1, 1, 4, 64, 32
        q = torch.randn(B, H, N, S, D, device=DEVICE)

        sparse_q, _, keep_size = _prune_queries(q, keep_ratio=0.001)

        assert keep_size >= 1


# ---------------------------------------------------------------------------
# KV block selection tests
# ---------------------------------------------------------------------------

class TestKVSelection:
    """Test dynamic KV block selection."""

    def test_mask_shape(self):
        B, H, N, S, D = 1, 2, 8, 32, 32
        sparse_q = torch.randn(B, H, N, S, D, device=DEVICE)
        k_blocks = torch.randn(B, H, N, 64, D, device=DEVICE)

        mask = _select_kv_blocks(sparse_q, k_blocks, 0.9, min_kv_blocks=4)

        assert mask.shape == (B, H, N, N)
        assert mask.dtype == torch.bool

    def test_minimum_blocks_enforced(self):
        B, H, N, S, D = 1, 2, 16, 32, 32
        sparse_q = torch.randn(B, H, N, S, D, device=DEVICE)
        k_blocks = torch.randn(B, H, N, 64, D, device=DEVICE)

        min_kv = 4
        mask = _select_kv_blocks(sparse_q, k_blocks, 0.9, min_kv_blocks=min_kv)

        blocks_per_q = mask.sum(dim=-1)
        assert (blocks_per_q >= min_kv).all()

    def test_high_threshold_keeps_more(self):
        B, H, N, S, D = 1, 2, 16, 32, 32
        sparse_q = torch.randn(B, H, N, S, D, device=DEVICE)
        k_blocks = torch.randn(B, H, N, 64, D, device=DEVICE)

        mask_low = _select_kv_blocks(sparse_q, k_blocks, 0.5, min_kv_blocks=1)
        mask_high = _select_kv_blocks(sparse_q, k_blocks, 0.99, min_kv_blocks=1)

        assert mask_high.sum() >= mask_low.sum()

    def test_threshold_1_keeps_all(self):
        B, H, N, S, D = 1, 1, 8, 16, 16
        sparse_q = torch.randn(B, H, N, S, D, device=DEVICE)
        k_blocks = torch.randn(B, H, N, 16, D, device=DEVICE)

        mask = _select_kv_blocks(sparse_q, k_blocks, 1.0, min_kv_blocks=1)

        assert mask.all()


# ---------------------------------------------------------------------------
# Sparsity statistics
# ---------------------------------------------------------------------------

class TestSparsityStats:
    """Verify sparsity levels match config."""

    def test_query_sparsity_ratio(self):
        B, H, N, S, D = 1, 1, 8, 64, 32
        q = torch.randn(B, H, N, S, D, device=DEVICE)

        keep_ratio = 0.5
        sparse_q, _, _ = _prune_queries(q, keep_ratio)

        actual_ratio = sparse_q.shape[3] / S
        assert abs(actual_ratio - keep_ratio) < 0.02

    def test_kv_sparsity_with_low_threshold(self):
        B, H, N, S, D = 1, 1, 32, 16, 32
        sparse_q = torch.randn(B, H, N, S, D, device=DEVICE)
        k_blocks = torch.randn(B, H, N, 16, D, device=DEVICE)

        mask = _select_kv_blocks(sparse_q, k_blocks, 0.5, min_kv_blocks=1)

        avg_selected = mask.float().sum(dim=-1).mean().item()
        assert avg_selected < N * 0.8


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
