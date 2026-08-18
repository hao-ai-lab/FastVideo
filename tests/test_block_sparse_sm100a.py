# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the sm_100a CUDA block-sparse VSA forward.

Compared against an explicit PyTorch reference rather than the Triton kernel: Triton's
block-sparse forward is hardcoded to 64-token blocks (BLOCK_M = BLOCK_N = 64) while this
extension also carries a 128-token build, so a direct comparison would be comparing two
different sparsity granularities. The reference below applies exactly the semantics the
kernel is supposed to implement -- selected blocks only, keys past variable_block_sizes
masked. Every case runs at both block sizes.

Run with: python -m pytest tests/test_block_sparse_sm100a.py -v
"""

import pytest
import torch

from fastvideo_kernel import block_sparse_attn_sm100a as vsa

HEAD_DIM = 128
BLOCK_SIZES = [64, 128]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0)
    or not vsa._HAS_VSA_SM100A,
    reason="requires Blackwell (sm_100a) and a built fastvideo_kernel extension",
)


def make_case(block, num_blocks=8, topk=4, heads=4, batch=1, ragged=False, seed=0):
    torch.manual_seed(seed)
    S = num_blocks * block
    shape = (batch, heads, S, HEAD_DIM) if vsa.BHSD else (batch, S, heads, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    idx = torch.empty((batch * heads * num_blocks, topk), dtype=torch.int32, device="cuda")
    for r in range(idx.shape[0]):
        idx[r] = torch.randperm(num_blocks, device="cuda")[:topk].to(torch.int32).sort().values
    num = torch.full((batch * heads * num_blocks, ), topk, dtype=torch.int32, device="cuda")

    if ragged:
        vbs = torch.randint(block // 2, block + 1, (num_blocks, ), dtype=torch.int32,
                            device="cuda")
    else:
        vbs = torch.full((num_blocks, ), block, dtype=torch.int32, device="cuda")
    return q, k, v, idx, num, vbs


def reference(q, k, v, idx, num, vbs, block):
    """Dense attention restricted to the selected blocks, with padded keys masked."""
    if not vsa.BHSD:
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # -> [B, H, S, D]
    B, H, S, D = q.shape
    num_blocks = vbs.numel()
    scale = 1.0 / (D**0.5)

    keep = torch.zeros((B, H, S, S), dtype=torch.bool, device=q.device)
    for b in range(B):
        for h in range(H):
            for qb in range(num_blocks):
                row = (b * H + h) * num_blocks + qb
                for j in range(int(num[row])):
                    kb = int(idx[row, j])
                    valid = int(vbs[kb])
                    keep[b, h, qb * block:(qb + 1) * block,
                         kb * block:kb * block + valid] = True

    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~keep, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out = p @ v.float()
    lse = torch.logsumexp(scores, dim=-1) * 1.4426950408889634
    return out, lse


def run_and_compare(block, ragged, num_blocks=8, topk=4, heads=4, atol=0.02):
    q, k, v, idx, num, vbs = make_case(block, num_blocks=num_blocks, topk=topk, heads=heads,
                                       ragged=ragged)
    assert vsa.is_supported(q, vbs)
    got, got_lse = vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)
    ref, ref_lse = reference(q, k, v, idx, num, vbs, block)

    got_o = got if vsa.BHSD else got.transpose(1, 2)
    diff = (got_o.float() - ref).abs().max().item()
    assert diff < atol, f"out: max |diff| = {diff:.5f}"
    lse_diff = (got_lse.float() - ref_lse).abs().max().item()
    assert lse_diff < 0.05, f"lse: max |diff| = {lse_diff:.5f}"


@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_forward_matches_reference(block):
    run_and_compare(block, ragged=False)


@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_forward_matches_reference_ragged(block):
    """variable_block_sizes is what FastVideo always passes; padded keys must be masked."""
    run_and_compare(block, ragged=True)


@pytest.mark.parametrize("block", BLOCK_SIZES)
@pytest.mark.parametrize("topk", [1, 2, 3, 5, 7])
def test_topk_not_a_multiple_of_the_group(block, topk):
    """The kernel groups selected blocks; a ragged final group must still be correct."""
    run_and_compare(block, ragged=True, num_blocks=8, topk=topk)


@pytest.mark.parametrize("block", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", [4, 8, 16])
def test_sequence_lengths(block, num_blocks):
    run_and_compare(block, ragged=True, num_blocks=num_blocks, topk=3)


@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_lse_is_not_vacuous(block):
    """Guards the lse assertion: a wrong lse must actually fail the comparison."""
    q, k, v, idx, num, vbs = make_case(block, ragged=True)
    _, got_lse = vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)
    _, ref_lse = reference(q, k, v, idx, num, vbs, block)
    assert (got_lse.float() - (ref_lse + 1.0)).abs().max().item() > 0.5


def test_unsupported_is_rejected():
    q, _, _, _, _, vbs = make_case(64)
    assert not vsa.is_supported(q.float(), vbs)                 # wrong dtype
    assert not vsa.is_supported(q[..., :64].contiguous(), vbs)  # wrong head_dim
    odd = torch.full((7, ), 64, dtype=torch.int32, device="cuda")
    assert not vsa.is_supported(q, odd)                         # seqlen/blocks mismatch
    thirty_two = torch.full((16, ), 32, dtype=torch.int32, device="cuda")
    assert not vsa.is_supported(q, thirty_two)                  # block size with no build
