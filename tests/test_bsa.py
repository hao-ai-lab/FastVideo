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
    _compute_sparse_attention,    # NEW
    _reconstruct_pruned,          # NEW
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

    def test_builder_requires_patch_divisible(self):
        builder = BSAAttentionMetadataBuilder()
        builder.prepare()
        with pytest.raises(AssertionError):
            builder.build(
                current_timestep=0,
                raw_latent_shape=(33, 64, 64),
                patch_size=(1, 2, 2),
                device=torch.device("cpu"),
            )

    def test_builder_requires_tile_divisible(self):
        builder = BSAAttentionMetadataBuilder()
        builder.prepare()
        with pytest.raises(AssertionError):
            builder.build(
                current_timestep=0,
                raw_latent_shape=(32, 66, 64),
                patch_size=(1, 1, 1),
                device=torch.device("cpu"),
            )

    def test_num_blocks_matches_sequence(self):
        builder = BSAAttentionMetadataBuilder()
        builder.prepare()
        metadata = builder.build(
            current_timestep=0,
            raw_latent_shape=(8, 8, 8),
            patch_size=(1, 1, 1),
            device=torch.device("cpu"),
        )
        assert metadata.total_seq_length == 512
        assert metadata.block_size == 64
        assert metadata.num_blocks == metadata.total_seq_length // metadata.block_size

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
# Sparse attention computation tests
# ---------------------------------------------------------------------------
 
class TestComputeSparseAttention:
    """Test _compute_sparse_attention in isolation."""
 
    def test_output_shape(self):
        B, H, N, Sq, Sk, D = 1, 2, 8, 32, 64, 32
        sparse_q = torch.randn(B, H, N, Sq, D, device=DEVICE)
        k_blocks = torch.randn(B, H, N, Sk, D, device=DEVICE)
        v_blocks = torch.randn(B, H, N, Sk, D, device=DEVICE)
        kv_mask = torch.ones(B, H, N, N, dtype=torch.bool, device=DEVICE)
 
        output = _compute_sparse_attention(sparse_q, k_blocks, v_blocks, kv_mask)
 
        assert output.shape == (B, H, N, Sq, D)
 
    def test_no_nan_or_inf(self):
        B, H, N, Sq, Sk, D = 2, 4, 4, 16, 64, 32
        sparse_q = torch.randn(B, H, N, Sq, D, device=DEVICE)
        k_blocks = torch.randn(B, H, N, Sk, D, device=DEVICE)
        v_blocks = torch.randn(B, H, N, Sk, D, device=DEVICE)
        kv_mask = torch.ones(B, H, N, N, dtype=torch.bool, device=DEVICE)
 
        output = _compute_sparse_attention(sparse_q, k_blocks, v_blocks, kv_mask)
 
        assert not torch.isnan(output).any(), "NaN in sparse attention output"
        assert not torch.isinf(output).any(), "Inf in sparse attention output"
 
    def test_full_mask_matches_dense(self):
        """With all KV blocks selected, should match dense attention over blocks."""
        B, H, N, S, D = 1, 1, 4, 16, 16
        q_blocks = torch.randn(B, H, N, S, D, device=DEVICE)
        k_blocks = torch.randn(B, H, N, S, D, device=DEVICE)
        v_blocks = torch.randn(B, H, N, S, D, device=DEVICE)
        kv_mask = torch.ones(B, H, N, N, dtype=torch.bool, device=DEVICE)
 
        output = _compute_sparse_attention(q_blocks, k_blocks, v_blocks, kv_mask)
 
        # Compare against manual dense attention over all KV
        all_k = k_blocks.reshape(B, H, N * S, D)
        all_v = v_blocks.reshape(B, H, N * S, D)
        for qb in range(N):
            q = q_blocks[:, :, qb]  # [B, H, S, D]
            scores = torch.matmul(q, all_k.transpose(-1, -2)) / (D ** 0.5)
            weights = F.softmax(scores, dim=-1)
            expected = torch.matmul(weights, all_v)
            assert torch.allclose(output[:, :, qb], expected, atol=1e-5), (
                f"Block {qb} doesn't match dense attention"
            )
 
    def test_sparse_mask_excludes_blocks(self):
        """With some KV blocks masked out, output should differ from full mask."""
        B, H, N, S, D = 1, 1, 8, 16, 16
        sparse_q = torch.randn(B, H, N, S, D, device=DEVICE)
        k_blocks = torch.randn(B, H, N, S, D, device=DEVICE)
        v_blocks = torch.randn(B, H, N, S, D, device=DEVICE)
 
        full_mask = torch.ones(B, H, N, N, dtype=torch.bool, device=DEVICE)
        sparse_mask = torch.zeros(B, H, N, N, dtype=torch.bool, device=DEVICE)
        sparse_mask[:, :, :, :2] = True  # only keep first 2 KV blocks
 
        out_full = _compute_sparse_attention(sparse_q, k_blocks, v_blocks, full_mask)
        out_sparse = _compute_sparse_attention(sparse_q, k_blocks, v_blocks, sparse_mask)
 
        assert not torch.allclose(out_full, out_sparse, atol=1e-3), (
            "Sparse and full mask should produce different outputs"
        )

# ---------------------------------------------------------------------------
# Reconstruction tests
# ---------------------------------------------------------------------------
 
class TestReconstructPruned:
    """Test _reconstruct_pruned in isolation."""
 
    def test_output_shape(self):
        B, H, N, keep_size, D = 1, 2, 4, 32, 64
        block_size = 64
        sparse_output = torch.randn(B, H, N, keep_size, D, device=DEVICE)
        keep_indices = torch.stack([
            torch.randperm(block_size)[:keep_size].sort().values
            for _ in range(B * H * N)
        ]).view(B, H, N, keep_size)
 
        full_output = _reconstruct_pruned(sparse_output, keep_indices, block_size)
 
        assert full_output.shape == (B, H, N, block_size, D)
 
    def test_kept_positions_preserved(self):
        """Kept token outputs should be exactly preserved."""
        B, H, N, keep_size, D = 1, 1, 2, 4, 8
        block_size = 8
        sparse_output = torch.randn(B, H, N, keep_size, D, device=DEVICE)
        keep_indices = torch.tensor([[[[0, 2, 5, 7], [1, 3, 4, 6]]]])
 
        full_output = _reconstruct_pruned(sparse_output, keep_indices, block_size)
 
        # Check that kept positions match exactly
        for n in range(N):
            for i, pos in enumerate(keep_indices[0, 0, n].tolist()):
                assert torch.allclose(
                    full_output[0, 0, n, pos],
                    sparse_output[0, 0, n, i],
                    atol=1e-6,
                ), f"Kept position {pos} not preserved in block {n}"
 
    def test_pruned_positions_filled(self):
        """Pruned positions should not be zero (filled with nearest kept)."""
        B, H, N, keep_size, D = 1, 1, 1, 4, 8
        block_size = 8
        sparse_output = torch.ones(B, H, N, keep_size, D, device=DEVICE)
        keep_indices = torch.tensor([[[[0, 2, 5, 7]]]])
 
        full_output = _reconstruct_pruned(sparse_output, keep_indices, block_size)
 
        # All positions should be non-zero since sparse_output is all ones
        for pos in range(block_size):
            assert full_output[0, 0, 0, pos].abs().sum() > 0, (
                f"Position {pos} is still zero after reconstruction"
            )
 
    def test_no_pruning_identity(self):
        """If keep_size == block_size, output should be unchanged."""
        B, H, N, D = 1, 2, 4, 32
        block_size = 16
        sparse_output = torch.randn(B, H, N, block_size, D, device=DEVICE)
        keep_indices = torch.arange(block_size, device=DEVICE).view(1, 1, 1, block_size).expand(B, H, N, -1)
 
        full_output = _reconstruct_pruned(sparse_output, keep_indices, block_size)
 
        assert torch.allclose(full_output, sparse_output, atol=1e-6)
 
 # ---------------------------------------------------------------------------
# End-to-end BSAAttentionImpl.forward tests
# ---------------------------------------------------------------------------
 
class TestBSAImplForward:
    """Test BSAAttentionImpl.forward end-to-end."""
 
    def _make_impl_and_metadata(self, T=8, Hv=8, Wv=8, num_heads=4, head_dim=32,
                                 keep_ratio=0.5, threshold=0.9):
        impl = BSAAttentionImpl(
            num_heads=num_heads,
            head_size=head_dim,
            causal=False,
            softmax_scale=head_dim ** -0.5,
        )
 
        builder = BSAAttentionMetadataBuilder()
        builder.prepare()
        metadata = builder.build(
            current_timestep=0,
            raw_latent_shape=(T, Hv * 2, Wv * 2),  # pre-patch
            patch_size=(1, 2, 2),
            device=torch.device(DEVICE),
            bsa_query_keep_ratio=keep_ratio,
            bsa_kv_cumulative_threshold=threshold,
        )
 
        return impl, metadata
 
    def test_output_shape(self):
        B, H, D = 1, 4, 32
        T, Hv, Wv = 8, 8, 8
        L = T * Hv * Wv
        impl, metadata = self._make_impl_and_metadata(T, Hv, Wv, H, D)
 
        # Input layout: [B, L, H, D]
        q = torch.randn(B, L, H, D, device=DEVICE)
        k = torch.randn(B, L, H, D, device=DEVICE)
        v = torch.randn(B, L, H, D, device=DEVICE)
 
        output = impl.forward(q, k, v, metadata)
 
        assert output.shape == (B, L, H, D)
 
    def test_no_nan(self):
        B, H, D = 1, 4, 32
        T, Hv, Wv = 8, 8, 8
        L = T * Hv * Wv
        impl, metadata = self._make_impl_and_metadata(T, Hv, Wv, H, D)
 
        q = torch.randn(B, L, H, D, device=DEVICE)
        k = torch.randn(B, L, H, D, device=DEVICE)
        v = torch.randn(B, L, H, D, device=DEVICE)
 
        output = impl.forward(q, k, v, metadata)
 
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
 
    def test_no_sparsity_approximates_full_attention(self):
        """With keep_ratio=1.0 and threshold=1.0, should approximate full attention."""
        B, H, D = 1, 2, 16
        T, Hv, Wv = 4, 4, 4
        L = T * Hv * Wv
        impl, metadata = self._make_impl_and_metadata(
            T, Hv, Wv, H, D, keep_ratio=1.0, threshold=1.0
        )
 
        q = torch.randn(B, L, H, D, device=DEVICE)
        k = torch.randn(B, L, H, D, device=DEVICE)
        v = torch.randn(B, L, H, D, device=DEVICE)
 
        bsa_out = impl.forward(q, k, v, metadata)
 
        # Reference full attention in [B, H, L, D] layout
        q_ref = q.transpose(1, 2)
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / (D ** 0.5)
        weights = F.softmax(scores, dim=-1)
        full_out = torch.matmul(weights, v_ref).transpose(1, 2)
 
        cos_sim = F.cosine_similarity(
            bsa_out.flatten(), full_out.flatten(), dim=0
        )
        assert cos_sim > 0.95, f"BSA with no sparsity diverges: cos_sim={cos_sim:.4f}"
 
    def test_batch_size_2(self):
        B, H, D = 2, 4, 32
        T, Hv, Wv = 8, 8, 8
        L = T * Hv * Wv
        impl, metadata = self._make_impl_and_metadata(T, Hv, Wv, H, D)
 
        q = torch.randn(B, L, H, D, device=DEVICE)
        k = torch.randn(B, L, H, D, device=DEVICE)
        v = torch.randn(B, L, H, D, device=DEVICE)
 
        output = impl.forward(q, k, v, metadata)
 
        assert output.shape == (B, L, H, D)
        assert not torch.isnan(output).any()
 
    def test_high_sparsity(self):
        B, H, D = 1, 2, 32
        T, Hv, Wv = 8, 8, 8
        L = T * Hv * Wv
        impl, metadata = self._make_impl_and_metadata(
            T, Hv, Wv, H, D, keep_ratio=0.25, threshold=0.5
        )
 
        q = torch.randn(B, L, H, D, device=DEVICE)
        k = torch.randn(B, L, H, D, device=DEVICE)
        v = torch.randn(B, L, H, D, device=DEVICE)
 
        output = impl.forward(q, k, v, metadata)
 
        assert output.shape == (B, L, H, D)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_rejects_causal(self):
        with pytest.raises(ValueError):
            BSAAttentionImpl(
                num_heads=2,
                head_size=32,
                causal=True,
                softmax_scale=32 ** -0.5,
            )

    def test_rejects_gqa(self):
        with pytest.raises(ValueError):
            BSAAttentionImpl(
                num_heads=4,
                head_size=32,
                causal=False,
                softmax_scale=32 ** -0.5,
                num_kv_heads=2,
            )

    def test_rejects_custom_softmax_scale(self):
        with pytest.raises(ValueError):
            BSAAttentionImpl(
                num_heads=2,
                head_size=32,
                causal=False,
                softmax_scale=0.1,
            )

    def test_length_mismatch_asserts(self):
        B, H, D = 1, 2, 16
        T, Hv, Wv = 4, 4, 4
        impl, metadata = self._make_impl_and_metadata(T, Hv, Wv, H, D)

        L = metadata.total_seq_length
        q = torch.randn(B, L - 1, H, D, device=DEVICE)
        k = torch.randn(B, L - 1, H, D, device=DEVICE)
        v = torch.randn(B, L - 1, H, D, device=DEVICE)

        with pytest.raises(AssertionError):
            impl.forward(q, k, v, metadata)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
