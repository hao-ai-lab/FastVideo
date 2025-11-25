import os
import sys
from typing import Tuple

import torch

# Make sure we can import from the project root (`vsa`, `tests.utils`, etc.)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from tests.utils import (
    generate_block_sparse_mask_for_function,
    create_full_mask_from_block_mask,
)
from vsa import block_sparse_attn, BLOCK_M
import test_vsa as ref  # reuse helper functions from backward test


def pytorch_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_sparse_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Dense PyTorch reference forward:
    - Q: [1, h, S_q, d]
    - K,V: [1, h, S_kv, d]
    - block_sparse_mask: [h, S_q, S_kv] bool
    """
    q = Q.clone().float()
    k = K.clone().float()
    v = V.clone().float()

    attn = torch.matmul(q, k.transpose(-2, -1))  # [1, h, S_q, S_kv]
    attn = attn / (q.size(-1) ** 0.5)
    attn = attn.masked_fill(~block_sparse_mask.unsqueeze(0), float("-inf"))
    attn = torch.nn.functional.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)  # [1, h, S_q, d]
    return out.to(torch.bfloat16)


def block_sparse_forward_test(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_sparse_mask: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_non_pad_index: torch.Tensor,
    kv_non_pad_index: torch.Tensor,
    q_num_blocks: int,
    kv_num_blocks: int,
) -> torch.Tensor:
    """
    Forward-only wrapper around `block_sparse_attn`, mirroring `block_sparse_kernel_test`
    but without any backward / grad logic.
    """
    Q = Q.detach()
    K = K.detach()
    V = V.detach()

    q_padded = ref.vsa_pad(Q, q_non_pad_index, q_num_blocks, BLOCK_M)
    k_padded = ref.vsa_pad(K, kv_non_pad_index, kv_num_blocks, BLOCK_M)
    v_padded = ref.vsa_pad(V, kv_non_pad_index, kv_num_blocks, BLOCK_M)

    out_padded, _ = block_sparse_attn(
        q_padded, k_padded, v_padded, block_sparse_mask, variable_block_sizes
    )
    # Remove padding on the query side
    out = out_padded[:, :, q_non_pad_index, :]
    return out


def run_forward_equal_qk(
    h: int = 16,
    d: int = 128,
    num_blocks: int = 16,
    k: int = 2,
    num_iterations: int = 5,
) -> Tuple[float, float]:
    """
    Forward-only correctness test for the case S_q == S_kv.
    Mirrors `check_correctness` but only compares forward outputs.
    """
    assert torch.cuda.is_available(), "VSA kernels require CUDA"
    device = "cuda"

    variable_block_sizes = ref.generate_variable_block_sizes(
        num_blocks, device=device
    )
    S = int(variable_block_sizes.sum().item())
    non_pad_index = ref.get_non_pad_index(
        variable_block_sizes, num_blocks, BLOCK_M
    )

    block_mask = generate_block_sparse_mask_for_function(
        h, num_blocks, num_blocks, k, device
    )
    full_mask = create_full_mask_from_block_mask(
        block_mask, variable_block_sizes, variable_block_sizes, device
    )
    print(f"[qkequal] h: {h}, d: {d}, num_blocks: {num_blocks}, k: {k}")
    print(f"[qkequal] variable_block_sizes: {variable_block_sizes}, non_pad_index: {non_pad_index.shape}, block_mask: {block_mask.shape}, full_mask: {full_mask.shape}")
    sum_diff = 0.0
    sum_abs = 0.0
    max_rel_diff = 0.0

    for i in range(num_iterations):
        Q = ref.generate_tensor((1, h, S, d), torch.bfloat16, device)
        K = ref.generate_tensor((1, h, S, d), torch.bfloat16, device)
        V = ref.generate_tensor((1, h, S, d), torch.bfloat16, device)

        if i == 0: print(f"[qkequal] Q: {Q.shape}, K: {K.shape}, V: {V.shape}, full_mask: {full_mask.shape}")
        if i == 0: print(f"[qkequal] block_mask: {block_mask.shape}")

        pt_o = pytorch_forward(Q, K, V, full_mask)
        bs_o = block_sparse_forward_test(
            Q,
            K,
            V,
            block_mask.unsqueeze(0),
            variable_block_sizes,
            non_pad_index,
            non_pad_index,
            num_blocks,
            num_blocks,
        )

        diff = (pt_o - bs_o).abs()
        sum_diff += diff.sum().item()
        sum_abs += pt_o.abs().sum().item()
        rel_max = diff.max() / (pt_o.abs().mean() + 1e-6)
        max_rel_diff = max(max_rel_diff, rel_max.item())

    total_elems = h * S * d * num_iterations
    avg_abs_err = sum_diff / total_elems
    return avg_abs_err, max_rel_diff


def run_forward_qk_diff(
    h: int = 16,
    d: int = 128,
    num_q_blocks: int = 16,
    num_kv_blocks: int = 32,
    k: int = 2,
    num_iterations: int = 5,
) -> Tuple[float, float]:
    """
    Forward-only correctness test for the case S_q != S_kv.

    NOTE:
      - The Triton backend supports different Q/KV logical lengths via padding.
      - The SM90 (H100) CUDA backend currently assumes the same number of blocks
        for Q and KV, so we skip this test there.
    """
    assert torch.cuda.is_available(), "VSA kernels require CUDA"

    device = "cuda"

    q_variable_block_sizes = ref.generate_variable_block_sizes(
        num_q_blocks, device=device
    )
    kv_variable_block_sizes = ref.generate_variable_block_sizes(
        num_kv_blocks, device=device
    )

    S_q = int(q_variable_block_sizes.sum().item())
    S_kv = int(kv_variable_block_sizes.sum().item())

    q_non_pad_index = ref.get_non_pad_index(
        q_variable_block_sizes, num_q_blocks, BLOCK_M
    )
    kv_non_pad_index = ref.get_non_pad_index(
        kv_variable_block_sizes, num_kv_blocks, BLOCK_M
    )

    block_mask = generate_block_sparse_mask_for_function(
        h, num_q_blocks, num_kv_blocks, k, device
    )
    full_mask = create_full_mask_from_block_mask(
        block_mask, q_variable_block_sizes, kv_variable_block_sizes, device
    )

    sum_diff = 0.0
    sum_abs = 0.0
    max_rel_diff = 0.0

    for _ in range(num_iterations):
        Q = ref.generate_tensor((1, h, S_q, d), torch.bfloat16, device)
        K = ref.generate_tensor((1, h, S_kv, d), torch.bfloat16, device)
        V = ref.generate_tensor((1, h, S_kv, d), torch.bfloat16, device)

        pt_o = pytorch_forward(Q, K, V, full_mask)
        bs_o = block_sparse_forward_test(
            Q,
            K,
            V,
            block_mask.unsqueeze(0),
            kv_variable_block_sizes,
            q_non_pad_index,
            kv_non_pad_index,
            num_q_blocks,
            num_kv_blocks,
        )

        diff = (pt_o - bs_o).abs()
        sum_diff += diff.sum().item()
        sum_abs += pt_o.abs().sum().item()
        rel_max = diff.max() / (pt_o.abs().mean() + 1e-6)
        max_rel_diff = max(max_rel_diff, rel_max.item())

    total_elems = h * S_q * d * num_iterations
    avg_abs_err = sum_diff / total_elems
    return avg_abs_err, max_rel_diff


if __name__ == "__main__":
    h, d = 16, 128
    print("Forward Block Sparse Attention Check (QK Equal)")
    print("=" * 80)
    avg_err_eq, max_rel_eq = run_forward_equal_qk(h, d, num_blocks=32, k=2)
    print(f"QK equal: avg |ΔO| = {avg_err_eq:.6e}, max rel ΔO = {max_rel_eq:.6e}")

    print("\nForward Block Sparse Attention Check (QK Different)")
    print("=" * 80)
    avg_err_diff, max_rel_diff = run_forward_qk_diff(
        h, d, num_q_blocks=32, num_kv_blocks=48, k=2
    )
    print(
        f"QK diff:  avg |ΔO| = {avg_err_diff:.6e}, max rel ΔO = {max_rel_diff:.6e}"
    )

