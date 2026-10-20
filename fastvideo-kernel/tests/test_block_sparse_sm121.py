# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the GB10 (DGX Spark, sm_121a) block-sparse VSA forward, 128-token blocks.

Compared against an explicit FP32 PyTorch reference of the semantics the kernel implements: each
valid query row attends to the valid keys of its block's selected key blocks; query rows at or past
their block's variable_block_sizes, and every row of a block that selects nothing, are zero (H3
discards them; the kernel writes them anyway, since its output comes from torch::empty_like). So
that a missed store cannot hide behind a recycled zero buffer, the tests that check the zero rows
first fill the allocator with NaN.

Run with: python -m pytest fastvideo-kernel/tests/test_block_sparse_sm121.py -v
"""

import pytest
import torch

from fastvideo_kernel import block_sparse_attn_sm121 as vsa

BLOCK = 128
HEAD_DIM = 128

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1) or not vsa._HAS_VSA_SM121,
    reason="requires a GB10 (DGX Spark, sm_121) and a fastvideo_kernel build with the sm_121a forward",
)


def make_case(num_blocks=8, topk=4, heads=4, batch=1, ragged=False, counts=None, empty_blocks=(), seed=0):
    """q/k/v [B, H, S, 128] with zero padding rows, map_to_index-shaped q2k_idx/q2k_num, and vbs.

    ``counts`` cycles over the rows instead of ``topk``; ``empty_blocks`` get variable_block_sizes 0.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    seq = num_blocks * BLOCK
    q, k, v = (torch.randn((batch, heads, seq, HEAD_DIM), device="cuda", dtype=torch.bfloat16, generator=gen)
               for _ in range(3))
    rows = batch * heads * num_blocks
    idx = torch.full((rows, num_blocks), -1, dtype=torch.int32, device="cuda")
    num = torch.empty(rows, dtype=torch.int32, device="cuda")
    for r in range(rows):
        n = min(num_blocks, topk if counts is None else int(counts[r % len(counts)]))
        idx[r, :n] = torch.randperm(num_blocks, device="cuda", generator=gen)[:n].sort().values.to(torch.int32)
        num[r] = n
    if ragged:
        vbs = torch.randint(1, BLOCK + 1, (num_blocks, ), dtype=torch.int32, device="cuda", generator=gen)
    else:
        vbs = torch.full((num_blocks, ), BLOCK, dtype=torch.int32, device="cuda")
    for b in empty_blocks:
        vbs[b] = 0
    valid = (torch.arange(BLOCK, device="cuda")[None] < vbs[:, None]).reshape(-1)
    for t in (q, k, v):
        t[:, :, ~valid] = 0
    return q, k, v, idx.view(batch, heads, num_blocks, num_blocks), num.view(batch, heads, num_blocks), vbs


def reference(q, k, v, idx, num, vbs):
    """FP32 attention restricted to the selected blocks' valid keys; rows without any key are zero."""
    B, H, S, D = q.shape
    keep = torch.zeros((B, H, S, S), dtype=torch.bool, device=q.device)
    sizes = vbs.tolist()
    for b in range(B):
        for h in range(H):
            for i, rows in enumerate(sizes):
                for j in idx[b, h, i, :int(num[b, h, i])].tolist():
                    keep[b, h, i * BLOCK:i * BLOCK + rows, j * BLOCK:j * BLOCK + sizes[j]] = True
    scores = (q.float() @ k.float().transpose(-1, -2)) / D**0.5
    has = keep.any(-1, keepdim=True)
    p = torch.softmax(scores.masked_fill(~keep, float("-inf")).masked_fill(~has, 0.0), dim=-1) * has
    return p @ v.float(), has.squeeze(-1)


def run_with_nan_pool(q, k, v, idx, num, vbs):
    """Call the kernel after leaving a NaN block of the output's size in the caching allocator."""
    sentinel = torch.full_like(q, float("nan"))
    del sentinel
    out, lse = vsa.block_sparse_attn_sm121(q, k, v, idx, num, vbs)
    torch.cuda.synchronize()
    assert lse is None
    return out


def check(case, atol=0.02):
    q, k, v, idx, num, vbs = case
    assert vsa.is_supported(q, vbs)
    out = run_with_nan_pool(q, k, v, idx, num, vbs)
    ref, has = reference(q, k, v, idx, num, vbs)
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), "the kernel left elements unwritten"
    assert (out[~has] == 0).all(), "rows without a key (padding rows, empty selections) must be zero"
    torch.testing.assert_close(out.float()[has], ref[has], atol=atol, rtol=0.02)


def test_forward_matches_reference():
    check(make_case())


def test_forward_matches_reference_ragged():
    """variable_block_sizes is what FastVideo always passes: padded keys masked, padded rows zero."""
    check(make_case(ragged=True, seed=1))


@pytest.mark.parametrize("topk", [1, 2, 3, 5, 7, 8])
def test_any_selection_size(topk):
    check(make_case(ragged=True, topk=topk, seed=topk))


@pytest.mark.parametrize("num_blocks", [1, 3, 7, 9])
def test_odd_block_counts(num_blocks):
    """Unlike the ptx_blackwell kernel, this one needs no even block count."""
    check(make_case(num_blocks=num_blocks, topk=max(1, num_blocks // 2), ragged=True, seed=num_blocks))


def test_zero_count_rows_and_non_uniform_counts():
    check(make_case(num_blocks=8, counts=(0, 3, 0, 8, 1, 5), ragged=True, seed=7))


def test_selected_blocks_without_valid_keys():
    """A selected block whose variable_block_sizes is 0 contributes nothing; a row left without keys is zero."""
    check(make_case(num_blocks=8, topk=3, ragged=True, empty_blocks=(2, 5), seed=11))


def test_batch_and_heads():
    check(make_case(num_blocks=6, topk=3, heads=3, batch=2, ragged=True, seed=13))


def test_many_waves():
    """More CTAs than SMs, long selections: the phased walk starts mid-selection and wraps."""
    check(make_case(num_blocks=64, topk=24, heads=2, ragged=True, seed=17))


def test_deterministic():
    case = make_case(num_blocks=16, topk=6, ragged=True, seed=19)
    first = vsa.block_sparse_attn_sm121(*case)[0]
    second = vsa.block_sparse_attn_sm121(*case)[0]
    assert torch.equal(first, second)


def test_is_supported_rejects_what_it_cannot_run():
    q, k, v, idx, num, vbs = make_case(num_blocks=2, topk=1)
    assert vsa.is_supported(q, vbs)
    assert not vsa.is_supported(q.half(), vbs)
    assert not vsa.is_supported(q[..., :64].contiguous(), vbs)
    assert not vsa.is_supported(q.transpose(1, 2), vbs)
    assert not vsa.is_supported(q[:, :, :BLOCK], vbs)
    assert not vsa.is_supported(q, vbs.float())
    assert not vsa.is_supported(q.cpu(), vbs)
