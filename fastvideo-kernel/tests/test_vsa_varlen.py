"""Correctness tests for variable-length block-sparse attention.

Reference: per-sequence calls to block_sparse_attn_from_indices.
Test: single-launch via block_sparse_attn_varlen.
Tests cover both forward and backward (gradient) correctness.
"""

import torch
import pytest

from .test_vsa import (
    BLOCK_M,
    generate_variable_block_sizes,
    get_non_pad_index,
    vsa_pad,
    generate_tensor,
)
from .utils import generate_block_sparse_mask_for_function
from fastvideo_kernel.block_sparse_attn import (
    block_sparse_attn_from_indices,
    _map_to_index,
)
from fastvideo_kernel.block_sparse_attn_varlen import block_sparse_attn_varlen


def _reference_per_sequence(
    q_list, k_list, v_list,
    block_masks, vbs_list,
    non_pad_q_list, non_pad_kv_list,
    q_nblocks_list, kv_nblocks_list,
):
    """Run per-sequence block_sparse_attn and concat outputs."""
    outs = []
    for i in range(len(q_list)):
        q_pad = vsa_pad(q_list[i], non_pad_q_list[i], q_nblocks_list[i], BLOCK_M)
        k_pad = vsa_pad(k_list[i], non_pad_kv_list[i], kv_nblocks_list[i], BLOCK_M)
        v_pad = vsa_pad(v_list[i], non_pad_kv_list[i], kv_nblocks_list[i], BLOCK_M)

        q2k_idx, q2k_num = _map_to_index(block_masks[i].unsqueeze(0))
        o_pad, _ = block_sparse_attn_from_indices(
            q_pad, k_pad, v_pad, q2k_idx, q2k_num, vbs_list[i],
        )
        o = o_pad[:, :, non_pad_q_list[i], :]
        outs.append(o.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def _run_varlen_test(
    seq_configs: list,
    h: int = 8,
    d: int = 64,
    topk: int = 2,
    atol: float = 0.05,
    rtol: float = 0.02,
):
    """Core test: compare varlen vs per-sequence reference.

    seq_configs: list of (num_q_blocks, num_kv_blocks) per sequence.
    """
    device = "cuda"
    num_seqs = len(seq_configs)

    q_list = []
    k_list = []
    v_list = []
    block_masks = []
    vbs_list = []
    q_vbs_list = []
    non_pad_q_list = []
    non_pad_kv_list = []
    q_nblocks_list = []
    kv_nblocks_list = []
    q2k_idx_list = []
    q2k_num_list = []
    q_vbs_for_varlen = []

    cu_q = [0]
    cu_kv = [0]

    for nq, nkv in seq_configs:
        vbs_kv = generate_variable_block_sizes(nkv, device=device)
        vbs_q = generate_variable_block_sizes(nq, device=device)
        sq = int(vbs_q.sum().item())
        skv = int(vbs_kv.sum().item())

        q = generate_tensor((1, h, sq, d), torch.bfloat16, device)
        k = generate_tensor((1, h, skv, d), torch.bfloat16, device)
        v = generate_tensor((1, h, skv, d), torch.bfloat16, device)

        mask = generate_block_sparse_mask_for_function(h, nq, nkv, topk, device)
        npq = get_non_pad_index(vbs_q, nq, BLOCK_M)
        npkv = get_non_pad_index(vbs_kv, nkv, BLOCK_M)

        q2k_idx, q2k_num = _map_to_index(mask.unsqueeze(0))

        q_list.append(q)
        k_list.append(k)
        v_list.append(v)
        block_masks.append(mask)
        vbs_list.append(vbs_kv)
        q_vbs_list.append(vbs_q)
        non_pad_q_list.append(npq)
        non_pad_kv_list.append(npkv)
        q_nblocks_list.append(nq)
        kv_nblocks_list.append(nkv)
        q2k_idx_list.append(q2k_idx)
        q2k_num_list.append(q2k_num)
        q_vbs_for_varlen.append(vbs_q)

        cu_q.append(cu_q[-1] + sq)
        cu_kv.append(cu_kv[-1] + skv)

    ref_out = _reference_per_sequence(
        q_list, k_list, v_list,
        block_masks, vbs_list,
        non_pad_q_list, non_pad_kv_list,
        q_nblocks_list, kv_nblocks_list,
    )

    q_packed = torch.cat(
        [qi.squeeze(0).transpose(0, 1) for qi in q_list], dim=0,
    )
    k_packed = torch.cat(
        [ki.squeeze(0).transpose(0, 1) for ki in k_list], dim=0,
    )
    v_packed = torch.cat(
        [vi.squeeze(0).transpose(0, 1) for vi in v_list], dim=0,
    )

    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_seqlens_kv = torch.tensor(cu_kv, dtype=torch.int32, device=device)

    varlen_out = block_sparse_attn_varlen(
        q_packed, k_packed, v_packed,
        cu_seqlens_q, cu_seqlens_kv,
        q2k_idx_list, q2k_num_list,
        vbs_list,
        q_variable_block_sizes_list=q_vbs_for_varlen,
    )

    max_abs = (ref_out - varlen_out).abs().max().item()
    mean_abs = ref_out.abs().mean().item()
    max_rel = max_abs / (mean_abs + 1e-8)

    print(f"  seqs={[c for c in seq_configs]}, max_abs={max_abs:.4e}, max_rel={max_rel:.4e}")
    assert max_rel < rtol, f"max relative error {max_rel:.4e} exceeds threshold {rtol}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVSAVarlen:

    def test_equal_length(self):
        """Two sequences with same number of blocks."""
        _run_varlen_test([(4, 4), (4, 4)], h=8, d=64)

    def test_different_lengths(self):
        """Three sequences with different block counts."""
        _run_varlen_test([(2, 3), (5, 4), (3, 6)], h=8, d=64)

    def test_single_sequence(self):
        """Degenerate case: single sequence should match non-varlen path."""
        _run_varlen_test([(8, 8)], h=8, d=64)

    def test_many_heads(self):
        """More heads to stress the packing logic."""
        _run_varlen_test([(3, 4), (5, 3)], h=16, d=128)

    def test_many_sequences(self):
        """Stress test: 8 sequences with varying block counts."""
        configs = [(i + 2, i + 3) for i in range(8)]
        _run_varlen_test(configs, h=8, d=64)

    def test_topk_equals_num_blocks(self):
        """Edge: topk covers all KV blocks (dense attention)."""
        _run_varlen_test([(3, 3), (4, 4)], h=8, d=64, topk=8)

    def test_single_block_per_sequence(self):
        """Minimal: each sequence has exactly 1 Q block and 1 KV block."""
        _run_varlen_test([(1, 1), (1, 1), (1, 1)], h=8, d=64, topk=1)

    def test_asymmetric_q_kv(self):
        """Q and KV have very different block counts."""
        _run_varlen_test([(1, 8), (8, 1)], h=8, d=64, topk=1)

    def test_without_q_vbs(self):
        """Test the default path where q_variable_block_sizes_list is None.

        Uses full block_size=64 for Q blocks so the None path is valid.
        """
        device = "cuda"
        h, d, topk = 4, 64, 2
        nq, nkv = 3, 4

        vbs_kv = generate_variable_block_sizes(nkv, device=device)
        sq = nq * BLOCK_M
        skv = int(vbs_kv.sum().item())

        q = generate_tensor((1, h, sq, d), torch.bfloat16, device)
        k = generate_tensor((1, h, skv, d), torch.bfloat16, device)
        v = generate_tensor((1, h, skv, d), torch.bfloat16, device)

        mask = generate_block_sparse_mask_for_function(h, nq, nkv, topk, device)
        npkv = get_non_pad_index(vbs_kv, nkv, BLOCK_M)
        q2k_idx, q2k_num = _map_to_index(mask.unsqueeze(0))

        k_pad = vsa_pad(k, npkv, nkv, BLOCK_M)
        v_pad = vsa_pad(v, npkv, nkv, BLOCK_M)
        ref_out, _ = block_sparse_attn_from_indices(q, k_pad, v_pad, q2k_idx, q2k_num, vbs_kv)
        ref_flat = ref_out.squeeze(0).transpose(0, 1)

        q_flat = q.squeeze(0).transpose(0, 1)
        k_flat = k.squeeze(0).transpose(0, 1)
        v_flat = v.squeeze(0).transpose(0, 1)
        cu_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
        cu_kv = torch.tensor([0, skv], dtype=torch.int32, device=device)

        varlen_out = block_sparse_attn_varlen(
            q_flat, k_flat, v_flat,
            cu_q, cu_kv,
            [q2k_idx], [q2k_num], [vbs_kv],
        )

        max_abs = (ref_flat - varlen_out).abs().max().item()
        mean_abs = ref_flat.abs().mean().item()
        max_rel = max_abs / (mean_abs + 1e-8)
        print(f"  without_q_vbs: max_abs={max_abs:.4e}, max_rel={max_rel:.4e}")
        assert max_rel < 0.02, f"max relative error {max_rel:.4e} exceeds threshold"


def _run_varlen_backward_test(
    seq_configs: list,
    h: int = 8,
    d: int = 64,
    topk: int = 2,
    grad_rtol: float = 0.05,
):
    """Backward correctness: compare dQ/dK/dV from varlen vs per-sequence reference.

    Both paths use the same underlying block_sparse_attn_from_indices kernel
    (which has registered autograd). The varlen wrapper's scatter/gather must
    correctly propagate gradients through PyTorch's in-place slice assignment.
    """
    device = "cuda"
    num_seqs = len(seq_configs)

    q_list = []
    k_list = []
    v_list = []
    block_masks = []
    vbs_list = []
    q_vbs_list = []
    non_pad_q_list = []
    non_pad_kv_list = []
    q_nblocks_list = []
    kv_nblocks_list = []
    q2k_idx_list = []
    q2k_num_list = []

    cu_q = [0]
    cu_kv = [0]

    for nq, nkv in seq_configs:
        vbs_kv = generate_variable_block_sizes(nkv, device=device)
        vbs_q = generate_variable_block_sizes(nq, device=device)
        sq = int(vbs_q.sum().item())
        skv = int(vbs_kv.sum().item())

        q = generate_tensor((1, h, sq, d), torch.bfloat16, device)
        k = generate_tensor((1, h, skv, d), torch.bfloat16, device)
        v = generate_tensor((1, h, skv, d), torch.bfloat16, device)

        mask = generate_block_sparse_mask_for_function(h, nq, nkv, topk, device)
        npq = get_non_pad_index(vbs_q, nq, BLOCK_M)
        npkv = get_non_pad_index(vbs_kv, nkv, BLOCK_M)

        q2k_idx, q2k_num = _map_to_index(mask.unsqueeze(0))

        q_list.append(q)
        k_list.append(k)
        v_list.append(v)
        block_masks.append(mask)
        vbs_list.append(vbs_kv)
        q_vbs_list.append(vbs_q)
        non_pad_q_list.append(npq)
        non_pad_kv_list.append(npkv)
        q_nblocks_list.append(nq)
        kv_nblocks_list.append(nkv)
        q2k_idx_list.append(q2k_idx)
        q2k_num_list.append(q2k_num)

        cu_q.append(cu_q[-1] + sq)
        cu_kv.append(cu_kv[-1] + skv)

    # --- Reference: per-sequence backward ---
    ref_q_grads = []
    ref_k_grads = []
    ref_v_grads = []
    ref_outs = []
    for i in range(num_seqs):
        qi = q_list[i].detach().requires_grad_(True)
        ki = k_list[i].detach().requires_grad_(True)
        vi = v_list[i].detach().requires_grad_(True)

        q_pad = vsa_pad(qi, non_pad_q_list[i], q_nblocks_list[i], BLOCK_M)
        k_pad = vsa_pad(ki, non_pad_kv_list[i], kv_nblocks_list[i], BLOCK_M)
        v_pad = vsa_pad(vi, non_pad_kv_list[i], kv_nblocks_list[i], BLOCK_M)

        q2k_idx, q2k_num = _map_to_index(block_masks[i].unsqueeze(0))
        o_pad, _ = block_sparse_attn_from_indices(
            q_pad, k_pad, v_pad, q2k_idx, q2k_num, vbs_list[i],
        )
        o = o_pad[:, :, non_pad_q_list[i], :]
        o_flat = o.squeeze(0).transpose(0, 1)
        ref_outs.append(o_flat)

        dO = torch.ones_like(o_flat)
        o_flat.backward(dO)

        ref_q_grads.append(qi.grad.squeeze(0).transpose(0, 1))
        ref_k_grads.append(ki.grad.squeeze(0).transpose(0, 1))
        ref_v_grads.append(vi.grad.squeeze(0).transpose(0, 1))

    ref_dq = torch.cat(ref_q_grads, dim=0)
    ref_dk = torch.cat(ref_k_grads, dim=0)
    ref_dv = torch.cat(ref_v_grads, dim=0)

    # --- Varlen backward ---
    q_packed = torch.cat(
        [qi.squeeze(0).transpose(0, 1) for qi in q_list], dim=0,
    ).detach().requires_grad_(True)
    k_packed = torch.cat(
        [ki.squeeze(0).transpose(0, 1) for ki in k_list], dim=0,
    ).detach().requires_grad_(True)
    v_packed = torch.cat(
        [vi.squeeze(0).transpose(0, 1) for vi in v_list], dim=0,
    ).detach().requires_grad_(True)

    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_seqlens_kv = torch.tensor(cu_kv, dtype=torch.int32, device=device)

    varlen_out = block_sparse_attn_varlen(
        q_packed, k_packed, v_packed,
        cu_seqlens_q, cu_seqlens_kv,
        q2k_idx_list, q2k_num_list,
        vbs_list,
        q_variable_block_sizes_list=q_vbs_list,
    )

    dO = torch.ones_like(varlen_out)
    varlen_out.backward(dO)

    varlen_dq = q_packed.grad
    varlen_dk = k_packed.grad
    varlen_dv = v_packed.grad

    for name, ref, actual in [
        ("dQ", ref_dq, varlen_dq),
        ("dK", ref_dk, varlen_dk),
        ("dV", ref_dv, varlen_dv),
    ]:
        assert actual is not None, f"{name}: gradient is None (autograd chain broken)"
        max_abs = (ref - actual).abs().max().item()
        mean_abs = ref.abs().mean().item()
        max_rel = max_abs / (mean_abs + 1e-8)
        print(f"  {name}: max_abs={max_abs:.4e}, max_rel={max_rel:.4e}")
        assert max_rel < grad_rtol, (
            f"{name}: max relative error {max_rel:.4e} exceeds threshold {grad_rtol}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVSAVarlenBackward:

    def test_backward_equal_length(self):
        """Backward: two sequences with same number of blocks."""
        _run_varlen_backward_test([(4, 4), (4, 4)], h=8, d=64)

    def test_backward_different_lengths(self):
        """Backward: three sequences with different block counts."""
        _run_varlen_backward_test([(2, 3), (5, 4), (3, 6)], h=8, d=64)

    def test_backward_single_sequence(self):
        """Backward: single sequence should match non-varlen gradient path."""
        _run_varlen_backward_test([(8, 8)], h=8, d=64)

    def test_backward_many_heads(self):
        """Backward: more heads to stress gradient routing."""
        _run_varlen_backward_test([(3, 4), (5, 3)], h=16, d=128)

    def test_backward_asymmetric_q_kv(self):
        """Backward: Q and KV have very different block counts."""
        _run_varlen_backward_test([(1, 8), (8, 1)], h=8, d=64, topk=1)

    def test_backward_grad_nonzero(self):
        """Smoke test: gradients are non-zero (autograd chain is connected)."""
        device = "cuda"
        h, d, topk = 4, 64, 2
        nq, nkv = 3, 4

        vbs_kv = generate_variable_block_sizes(nkv, device=device)
        vbs_q = generate_variable_block_sizes(nq, device=device)
        sq = int(vbs_q.sum().item())
        skv = int(vbs_kv.sum().item())

        q = torch.randn(sq, h, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(skv, h, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(skv, h, d, device=device, dtype=torch.bfloat16, requires_grad=True)

        mask = generate_block_sparse_mask_for_function(h, nq, nkv, topk, device)
        q2k_idx, q2k_num = _map_to_index(mask.unsqueeze(0))

        cu_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
        cu_kv = torch.tensor([0, skv], dtype=torch.int32, device=device)

        out = block_sparse_attn_varlen(
            q, k, v,
            cu_q, cu_kv,
            [q2k_idx], [q2k_num], [vbs_kv],
            q_variable_block_sizes_list=[vbs_q],
        )

        loss = out.sum()
        loss.backward()

        assert q.grad is not None, "q.grad is None"
        assert k.grad is not None, "k.grad is None"
        assert v.grad is not None, "v.grad is None"
        assert q.grad.abs().sum().item() > 0, "q.grad is all zeros"
        assert k.grad.abs().sum().item() > 0, "k.grad is all zeros"
        assert v.grad.abs().sum().item() > 0, "v.grad is all zeros"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
