"""
Correctness test for video_sparse_attn_varlen.

Tests that packing multiple sequences of different Q and KV lengths into flat
tensors with cu_seqlens bookmarks produces the same result as running
video_sparse_attn independently on each sequence.
"""

import pytest
import torch

from .utils import generate_block_sparse_mask_for_function, create_full_mask_from_block_mask
from .test_vsa import generate_variable_block_sizes, generate_tensor
from .test_vsa_forward import pytorch_forward


def _build_varlen_inputs(seq_configs, h, d, topk, device):
    """
    Given a list of (num_q_blocks, num_kv_blocks) configs, build:
      - flat q, k, v tensors
      - cu_seqlens_q, cu_seqlens_kv
      - per-seq variable_block_sizes lists
      - per-seq block masks and full element-level masks (for reference)
    """
    q_seqs, k_seqs, v_seqs = [], [], []
    cu_q = [0]
    cu_kv = [0]
    vbs_list = []
    q_vbs_list = []
    ref_outputs = []

    for num_q_blocks, num_kv_blocks in seq_configs:
        q_vbs = generate_variable_block_sizes(num_q_blocks, device=device)
        kv_vbs = generate_variable_block_sizes(num_kv_blocks, device=device)

        S_q = int(q_vbs.sum().item())
        S_kv = int(kv_vbs.sum().item())

        Q = generate_tensor((1, h, S_q, d), torch.bfloat16, device)
        K = generate_tensor((1, h, S_kv, d), torch.bfloat16, device)
        V = generate_tensor((1, h, S_kv, d), torch.bfloat16, device)

        block_mask = generate_block_sparse_mask_for_function(h, num_q_blocks, num_kv_blocks, topk, device)
        full_mask = create_full_mask_from_block_mask(block_mask, q_vbs, kv_vbs, device)

        # PyTorch dense reference for this sequence
        ref_out = pytorch_forward(Q, K, V, full_mask)  # [1, h, S_q, d]
        ref_outputs.append(ref_out.squeeze(0).transpose(0, 1))  # [S_q, h, d]

        # Flat: [S, h, d]
        q_seqs.append(Q.squeeze(0).transpose(0, 1))   # [S_q, h, d]
        k_seqs.append(K.squeeze(0).transpose(0, 1))   # [S_kv, h, d]
        v_seqs.append(V.squeeze(0).transpose(0, 1))   # [S_kv, h, d]

        cu_q.append(cu_q[-1] + S_q)
        cu_kv.append(cu_kv[-1] + S_kv)
        vbs_list.append(kv_vbs)
        q_vbs_list.append(q_vbs)

    q_flat = torch.cat(q_seqs, dim=0)   # [total_q, h, d]
    k_flat = torch.cat(k_seqs, dim=0)   # [total_kv, h, d]
    v_flat = torch.cat(v_seqs, dim=0)   # [total_kv, h, d]

    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_seqlens_kv = torch.tensor(cu_kv, dtype=torch.int32, device=device)

    ref_flat = torch.cat(ref_outputs, dim=0)  # [total_q, h, d]

    return q_flat, k_flat, v_flat, cu_seqlens_q, cu_seqlens_kv, vbs_list, q_vbs_list, ref_flat


def run_varlen_forward(
    seq_configs=None,
    h=16,
    d=128,
    topk=2,
    num_iterations=3,
):
    """
    Compare video_sparse_attn_varlen against per-sequence PyTorch reference.
    Returns (avg_abs_err, max_rel_err).
    """
    assert torch.cuda.is_available(), "VSA kernels require CUDA"
    device = "cuda"

    if seq_configs is None:
        seq_configs = [
            (8, 12),   # Q < KV
            (16, 16),  # Q == KV
            (6, 20),   # Q < KV, different ratio
        ]

    from fastvideo_kernel import video_sparse_attn_varlen

    sum_diff = 0.0
    sum_abs = 0.0
    max_rel_diff = 0.0
    total_elems = 0

    for _ in range(num_iterations):
        q_flat, k_flat, v_flat, cu_seqlens_q, cu_seqlens_kv, vbs_list, q_vbs_list, ref_flat = \
            _build_varlen_inputs(seq_configs, h, d, topk, device)

        out_flat = video_sparse_attn_varlen(
            q_flat, k_flat, v_flat,
            cu_seqlens_q, cu_seqlens_kv,
            vbs_list, q_vbs_list,
            topk=topk,
            block_size=64,
        )

        diff = (ref_flat - out_flat).abs()
        sum_diff += diff.sum().item()
        sum_abs += ref_flat.abs().sum().item()
        rel = diff.max() / (ref_flat.abs().mean() + 1e-6)
        max_rel_diff = max(max_rel_diff, rel.item())
        total_elems += ref_flat.numel()

    avg_abs_err = sum_diff / total_elems
    return avg_abs_err, max_rel_diff


def test_vsa_varlen_forward_mixed_lengths():
    """Batch of 3 sequences with different Q and KV block counts."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    avg_err, max_rel = run_varlen_forward(
        seq_configs=[(8, 12), (16, 16), (6, 20)],
        h=16, d=128, topk=2,
    )
    print(f"Mixed lengths: avg |ΔO| = {avg_err:.6e}, max rel ΔO = {max_rel:.6e}")
    assert avg_err < 1e-2, f"avg abs error too large: {avg_err}"


def test_vsa_varlen_forward_equal_lengths():
    """Batch where all sequences have equal Q and KV lengths (sanity check)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    avg_err, max_rel = run_varlen_forward(
        seq_configs=[(16, 16), (16, 16), (16, 16)],
        h=16, d=128, topk=2,
    )
    print(f"Equal lengths: avg |ΔO| = {avg_err:.6e}, max rel ΔO = {max_rel:.6e}")
    assert avg_err < 1e-2, f"avg abs error too large: {avg_err}"


def test_vsa_varlen_forward_single_sequence():
    """Single sequence — varlen should match standard video_sparse_attn."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    avg_err, max_rel = run_varlen_forward(
        seq_configs=[(12, 16)],
        h=16, d=128, topk=2,
    )
    print(f"Single sequence: avg |ΔO| = {avg_err:.6e}, max rel ΔO = {max_rel:.6e}")
    assert avg_err < 1e-2, f"avg abs error too large: {avg_err}"


if __name__ == "__main__":
    print("=== VSA Varlen Forward Tests ===")

    print("\n[1] Mixed sequence lengths (Q != KV)")
    avg, rel = run_varlen_forward(seq_configs=[(8, 12), (16, 16), (6, 20)])
    print(f"    avg |ΔO| = {avg:.6e},  max rel ΔO = {rel:.6e}")

    print("\n[2] Equal sequence lengths")
    avg, rel = run_varlen_forward(seq_configs=[(16, 16), (16, 16), (16, 16)])
    print(f"    avg |ΔO| = {avg:.6e},  max rel ΔO = {rel:.6e}")

    print("\n[3] Single sequence")
    avg, rel = run_varlen_forward(seq_configs=[(12, 16)])
    print(f"    avg |ΔO| = {avg:.6e},  max rel ΔO = {rel:.6e}")
