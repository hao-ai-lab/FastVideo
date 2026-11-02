# SPDX-License-Identifier: Apache-2.0
"""ptx_sm107a VSA forward: CLC work-item scheduler under slot reuse.

The persistent forward steals not-yet-launched CTAs with clusterlaunchcontrol.try_cancel and
hands the 16-byte responses through a two-slot ring. The ring is only exercised when the grid
is several waves deep: with every CTA resident no cancel succeeds and no slot is ever reused,
which is the case for the small correctness tests. Here the grid is >= 3 waves for each of the
two schedules (M-split: one query block per CTA item; dual-Q: two), every call's output and LSE
are compared bitwise against a golden call of the same binary, and the buffers the kernel writes
are pre-filled with NaN, so a tile the kernel never wrote shows up as NaN and a tile written from
the wrong work item as a value mismatch.
"""

import pytest
import torch

from fastvideo_kernel import block_sparse_attn_ptx_sm107a as vsa

HEAD_DIM = 128
BLOCK = 128

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7) or not vsa._HAS_VSA_PTX_SM107A,
    reason="requires a Rubin (sm_107a) device and a fastvideo_kernel extension built with 10.7a",
)


def make_case(num_blocks, heads, topk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    S = num_blocks * BLOCK
    q, k, v = (torch.randn((1, heads, S, HEAD_DIM), device="cuda", dtype=torch.bfloat16, generator=g)
               for _ in range(3))
    scores = torch.rand((heads * num_blocks, num_blocks), device="cuda", generator=g)
    idx = scores.topk(topk, dim=1).indices.sort(dim=1).values.to(torch.int32)
    idx = idx.view(1, heads, num_blocks, topk).contiguous()
    num = torch.randint(1, topk + 1, (1, heads, num_blocks), device="cuda", generator=g, dtype=torch.int32)
    vbs = torch.randint(BLOCK // 2, BLOCK + 1, (num_blocks, ), device="cuda", generator=g, dtype=torch.int32)
    return q, k, v, idx, num, vbs


# (num_blocks, heads, query blocks per CTA item): the launcher picks M-split for the first (640 items on 212
# SMs) and dual-Q for the second (1024 items).
@pytest.mark.parametrize("num_blocks,heads,per_item", [(40, 16, 1), (128, 16, 2)], ids=["m_split", "dual_q"])
def test_clc_slot_reuse_never_drops_or_misplaces_a_tile(num_blocks, heads, per_item):
    topk, calls = 4, 300
    q, k, v, idx, num, vbs = make_case(num_blocks, heads, topk)
    assert vsa.is_supported(q, vbs)
    ctas = heads * num_blocks // per_item
    assert ctas >= 3 * torch.cuda.get_device_properties(0).multi_processor_count, ctas

    out_g, lse_g = (t.clone() for t in vsa.block_sparse_attn_ptx_sm107a(q, k, v, idx, num, vbs, need_lse=True))
    assert torch.isfinite(out_g).all() and torch.isfinite(lse_g).all()
    for _ in range(3):
        out, lse = vsa.block_sparse_attn_ptx_sm107a(q, k, v, idx, num, vbs, need_lse=True)
        assert torch.equal(out, out_g) and torch.equal(lse, lse_g), "forward is not bitwise deterministic"
        del out, lse

    S = num_blocks * BLOCK
    tile_shape = (1, heads, num_blocks, BLOCK * HEAD_DIM)
    nan = float("nan")
    failures = []
    for call in range(calls):
        # Pre-fill the blocks the op is about to allocate: same sizes, freed in reverse order.
        poison_lse = torch.full((1, heads, S), nan, device="cuda", dtype=torch.float32)
        poison_out = torch.full_like(q, nan)
        out_ptr, lse_ptr = poison_out.data_ptr(), poison_lse.data_ptr()
        del poison_out, poison_lse
        out, lse = vsa.block_sparse_attn_ptx_sm107a(q, k, v, idx, num, vbs, need_lse=True)
        assert out.data_ptr() == out_ptr and lse.data_ptr() == lse_ptr, "allocator did not reuse the poisoned blocks"
        wrong = (out != out_g).view(tile_shape).any(-1) | (lse != lse_g).view(1, heads, num_blocks, BLOCK).any(-1)
        if bool(wrong.any()):
            never_written = int(out.isnan().view(tile_shape).all(-1).sum())
            failures.append((call, int(wrong.sum()), never_written))
        del out, lse
    assert not failures, (f"{len(failures)} of {calls} calls had wrong tiles; first (call, wrong tiles, "
                          f"tiles never written): {failures[:5]}")
