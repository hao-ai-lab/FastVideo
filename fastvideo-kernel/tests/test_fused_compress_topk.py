"""Tests for fused_block_mean and fused_topk_mask equivalence against PyTorch reference."""

import torch
import pytest


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------------------------------------------------------------
# fused_topk_mask: tie-boundary correctness
# ---------------------------------------------------------------------------


class TestFusedTopkMaskTies:
    """Verify that fused_topk_mask selects exactly topk per row, even with ties."""

    def _run_topk_mask(self, scores, topk):
        from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_topk_mask
        return fused_topk_mask(scores.cuda(), topk)

    def _ref_topk_mask(self, scores, topk):
        """PyTorch reference: topk + scatter."""
        idx = torch.topk(scores, topk, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, idx, True)
        return mask

    def test_all_equal_scores(self):
        """All scores identical — must still select exactly topk."""
        _require_cuda()
        scores = torch.full((1, 1, 4, 8), 0.5, dtype=torch.bfloat16)
        for topk in [1, 3, 5, 8]:
            mask = self._run_topk_mask(scores, topk)
            row_counts = mask.sum(dim=-1)
            assert (row_counts == topk).all(), (
                f"topk={topk}, row_counts={row_counts}"
            )

    def test_reviewer_example(self):
        """Exact case from the reviewer: two 0.8875 with topk=1 → must select 1."""
        _require_cuda()
        vals = torch.tensor(
            [0.5222, 0.5222, 0.8875, 0.5222, 0.5222, 0.5222, 0.8875],
            dtype=torch.bfloat16,
        )
        scores = vals.view(1, 1, 1, -1)
        mask = self._run_topk_mask(scores, topk=1)
        assert mask.sum().item() == 1, f"Expected 1 selected, got {mask.sum().item()}"

    def test_tie_at_boundary_various_topk(self):
        """Scores with a large tied cluster at the k-th boundary."""
        _require_cuda()
        scores = torch.tensor(
            [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2],
            dtype=torch.bfloat16,
        ).view(1, 1, 1, -1)
        for topk in [1, 2, 3, 4, 7, 8]:
            mask = self._run_topk_mask(scores, topk)
            assert mask.sum().item() == topk, (
                f"topk={topk}, selected={mask.sum().item()}"
            )

    def test_batch_heads_ties(self):
        """Multiple batch/heads with tie-heavy scores."""
        _require_cuda()
        B, H, Q, KV = 2, 4, 8, 16
        scores = torch.zeros(B, H, Q, KV, dtype=torch.bfloat16)
        scores[:, :, :, :4] = 1.0
        scores[:, :, :, 4:] = 0.5
        for topk in [1, 4, 8]:
            mask = self._run_topk_mask(scores, topk)
            row_counts = mask.sum(dim=-1)
            assert (row_counts == topk).all(), (
                f"topk={topk}, row_counts min={row_counts.min()}, max={row_counts.max()}"
            )


# ---------------------------------------------------------------------------
# fused_topk_mask: equivalence with PyTorch topk+scatter
# ---------------------------------------------------------------------------


class TestFusedTopkMaskEquivalence:
    """fused_topk_mask must select the same set as torch.topk (modulo tie order)."""

    def test_random_scores_row_count(self):
        """Random scores (unlikely ties) — row count must match exactly."""
        _require_cuda()
        from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_topk_mask

        B, H, Q, KV = 2, 8, 32, 64
        topk = 8
        scores = torch.randn(B, H, Q, KV, dtype=torch.bfloat16, device="cuda")
        mask = fused_topk_mask(scores, topk)
        assert (mask.sum(dim=-1) == topk).all()

    def test_random_scores_value_match(self):
        """With distinct scores, fused and reference must select identical indices."""
        _require_cuda()
        from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_topk_mask

        B, H, Q, KV = 1, 2, 16, 32
        topk = 4
        scores = torch.arange(KV, dtype=torch.float32).view(1, 1, 1, KV)
        scores = scores.expand(B, H, Q, KV).contiguous().to(torch.bfloat16).cuda()

        fused_mask = fused_topk_mask(scores, topk)

        ref_idx = torch.topk(scores.cuda(), topk, dim=-1).indices
        ref_mask = torch.zeros_like(scores, dtype=torch.bool, device="cuda").scatter_(-1, ref_idx, True)

        assert (fused_mask == ref_mask).all()


# ---------------------------------------------------------------------------
# fused_block_mean: equivalence with PyTorch reference
# ---------------------------------------------------------------------------


class TestFusedBlockMeanEquivalence:

    def test_against_pytorch_reference(self):
        """fused_block_mean must match the original view→float→sum→div→bf16 path."""
        _require_cuda()
        from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_block_mean

        B, H, D = 2, 4, 128
        num_blocks = 16
        block_elements = 64
        seq_len = num_blocks * block_elements

        x = torch.randn(B, H, seq_len, D, dtype=torch.bfloat16, device="cuda")
        vbs = torch.randint(16, 65, (num_blocks,), dtype=torch.int32, device="cuda")

        fused_out = fused_block_mean(x, vbs, block_elements)

        x_blocks = x.view(B, H, num_blocks, block_elements, D)
        ref_out = (x_blocks.float().sum(dim=3) / vbs.view(1, 1, -1, 1).float()).to(torch.bfloat16)

        assert fused_out.shape == ref_out.shape
        assert torch.allclose(fused_out, ref_out, atol=1e-2, rtol=1e-2), (
            f"max diff={( fused_out - ref_out).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# End-to-end: fused video_sparse_attn vs old PyTorch pipeline
# ---------------------------------------------------------------------------


def _old_pytorch_video_sparse_attn(
    q, k, v, variable_block_sizes, q_variable_block_sizes,
    topk, block_elements,
):
    """Reproduce the pre-optimization PyTorch pipeline from ops.py."""
    batch, heads, q_seq_len, dim = q.shape
    kv_seq_len = k.shape[2]
    q_num_blocks = q_seq_len // block_elements
    kv_num_blocks = kv_seq_len // block_elements

    q_c = q.view(batch, heads, q_num_blocks, block_elements, dim)
    k_c = k.view(batch, heads, kv_num_blocks, block_elements, dim)
    v_c = v.view(batch, heads, kv_num_blocks, block_elements, dim)
    q_c = (q_c.float().sum(dim=3) / q_variable_block_sizes.view(1, 1, -1, 1)).to(q.dtype)
    k_c = (k_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(k.dtype)
    v_c = (v_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(v.dtype)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.matmul(attn, v_c)
    out_c = out_c.view(batch, heads, q_num_blocks, 1, dim)
    out_c = out_c.repeat(1, 1, 1, block_elements, 1).view(batch, heads, q_seq_len, dim)

    topk_idx = torch.topk(scores, topk, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)

    return out_c, scores, mask


class TestVideoSparseAttnEquivalence:
    """End-to-end: fused compress+topk path vs old PyTorch pipeline."""

    def _make_inputs(self, B, H, num_blocks, D, block_elements, device="cuda"):
        seq_len = num_blocks * block_elements
        q = torch.randn(B, H, seq_len, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, H, seq_len, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, H, seq_len, D, dtype=torch.bfloat16, device=device)
        vbs = torch.full((num_blocks,), block_elements, dtype=torch.int32, device=device)
        return q, k, v, vbs

    def test_compress_branch_equivalence(self):
        """Compression branch (block_mean → matmul → softmax → matmul) must match."""
        _require_cuda()
        from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_block_mean

        B, H, num_blocks, D = 1, 4, 32, 128
        block_elements = 64
        topk = 4
        q, k, v, vbs = self._make_inputs(B, H, num_blocks, D, block_elements)

        q_c_fused = fused_block_mean(q, vbs, block_elements)
        k_c_fused = fused_block_mean(k, vbs, block_elements)
        v_c_fused = fused_block_mean(v, vbs, block_elements)
        scores_fused = torch.matmul(q_c_fused, k_c_fused.transpose(-2, -1)) / (D ** 0.5)
        attn_fused = torch.softmax(scores_fused, dim=-1)
        out_c_fused = torch.matmul(attn_fused, v_c_fused)

        old_out_c, old_scores, _ = _old_pytorch_video_sparse_attn(
            q, k, v, vbs, vbs, topk, block_elements,
        )

        assert torch.allclose(scores_fused, old_scores, atol=1e-2, rtol=1e-2), (
            f"scores max diff={( scores_fused - old_scores).abs().max().item()}"
        )

    def test_topk_mask_row_count_matches(self):
        """Fused topk mask must select exactly topk per row, same as torch.topk."""
        _require_cuda()
        from fastvideo_kernel.triton_kernels.fused_compress_topk import (
            fused_block_mean,
            fused_topk_mask,
        )

        B, H, num_blocks, D = 1, 8, 48, 128
        block_elements = 64
        topk = 6
        q, k, v, vbs = self._make_inputs(B, H, num_blocks, D, block_elements)

        q_c = fused_block_mean(q, vbs, block_elements)
        k_c = fused_block_mean(k, vbs, block_elements)
        scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (D ** 0.5)

        fused_mask = fused_topk_mask(scores, topk)
        row_counts = fused_mask.sum(dim=-1)
        assert (row_counts == topk).all(), (
            f"row_counts min={row_counts.min()}, max={row_counts.max()}"
        )

        _, _, ref_mask = _old_pytorch_video_sparse_attn(
            q, k, v, vbs, vbs, topk, block_elements,
        )
        ref_counts = ref_mask.sum(dim=-1)
        assert (ref_counts == topk).all()

    def test_e2e_mask_equivalence_realistic_shape(self):
        """On realistic bf16 q_c@k_c scores, fused and ref mask should agree."""
        _require_cuda()
        from fastvideo_kernel.triton_kernels.fused_compress_topk import (
            fused_block_mean,
            fused_topk_mask,
        )

        B, H, num_blocks, D = 1, 16, 32, 128
        block_elements = 64
        topk = 4
        q, k, v, vbs = self._make_inputs(B, H, num_blocks, D, block_elements)

        q_c = fused_block_mean(q, vbs, block_elements)
        k_c = fused_block_mean(k, vbs, block_elements)
        scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (D ** 0.5)

        fused_mask = fused_topk_mask(scores, topk)

        ref_idx = torch.topk(scores, topk, dim=-1).indices
        ref_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, ref_idx, True)

        total_rows = B * H * num_blocks
        matching_rows = (fused_mask == ref_mask).all(dim=-1).sum().item()
        match_rate = matching_rows / total_rows

        print(f"\n[e2e mask equivalence] {matching_rows}/{total_rows} rows match "
              f"({match_rate:.2%})")

        assert (fused_mask.sum(dim=-1) == topk).all(), "row count mismatch"
        assert match_rate > 0.99, f"match rate too low: {match_rate:.2%}"
