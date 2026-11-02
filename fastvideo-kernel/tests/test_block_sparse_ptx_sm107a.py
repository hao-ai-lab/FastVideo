# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the Rubin (sm_107a) ptx block-sparse VSA forward, 128-token blocks.

Compared against an explicit PyTorch reference (selected blocks only, keys past variable_block_sizes
masked) rather than Triton, whose block-sparse forward is 64-token only. The kernel picks one of two
schedules per shape -- dual-Q (two adjacent query blocks per CTA item, sharing K / V loads of blocks both
select) and M-split (one query block split across the CTA's two M-tiles) -- so the cases are run on shapes
that select each.

Run with: python -m pytest fastvideo-kernel/tests/test_block_sparse_ptx_sm107a.py -v
"""

import os
import subprocess
import sys

import pytest
import torch

from fastvideo_kernel import block_sparse_attn_ptx_sm107a as vsa

HEAD_DIM = 128
BLOCK = 128
LOG2E = 1.4426950408889634

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7) or not vsa._HAS_VSA_PTX_SM107A,
    reason="requires a Rubin (sm_107a) device and a fastvideo_kernel extension built with 10.7a",
)


def schedule_heads(schedule, num_blocks=8):
    """Head count that makes the launcher pick `schedule` (it compares full rounds of CTA items on the SMs):
    M-split for a handful of items, dual-Q when the dual-Q items exactly fill the SMs."""
    if schedule == "m_split":
        return 4
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return max(1, sms // (num_blocks // 2))


def make_case(num_blocks=8, topk=4, heads=4, batch=1, ragged=False, seed=0, pad_to=None):
    torch.manual_seed(seed)
    S = num_blocks * BLOCK
    q, k, v = (torch.randn((batch, heads, S, HEAD_DIM), device="cuda", dtype=torch.bfloat16) for _ in range(3))
    rows = batch * heads * num_blocks
    width = pad_to or topk
    idx = torch.full((rows, width), -1, dtype=torch.int32, device="cuda")   # map_to_index pads with -1
    for r in range(rows):
        idx[r, :topk] = torch.randperm(num_blocks, device="cuda")[:topk].to(torch.int32).sort().values
    num = torch.full((rows, ), topk, dtype=torch.int32, device="cuda")
    if ragged:
        vbs = torch.randint(BLOCK // 2, BLOCK + 1, (num_blocks, ), dtype=torch.int32, device="cuda")
    else:
        vbs = torch.full((num_blocks, ), BLOCK, dtype=torch.int32, device="cuda")
    return q, k, v, idx, num, vbs


def make_block_map(idx, num, batch, heads, num_blocks):
    """Expand the compact rows to the bool map H3 produces."""
    block_map = torch.zeros((batch, heads, num_blocks, num_blocks), dtype=torch.bool, device=idx.device)
    rows = block_map.view(-1, num_blocks)
    for row in range(rows.shape[0]):
        rows[row, idx[row, :int(num[row])].long()] = True
    return block_map


def reference(q, k, v, idx, num, vbs):
    """Dense attention restricted to the selected blocks, with padded keys masked. [B, H, S, D]."""
    B, H, S, D = q.shape
    num_blocks = vbs.numel()
    keep = torch.zeros((B, H, S, S), dtype=torch.bool, device=q.device)
    for b in range(B):
        for h in range(H):
            for qb in range(num_blocks):
                row = (b * H + h) * num_blocks + qb
                for j in range(int(num[row])):
                    kb = int(idx[row, j])
                    keep[b, h, qb * BLOCK:(qb + 1) * BLOCK, kb * BLOCK:kb * BLOCK + int(vbs[kb])] = True
    scores = (q.float() @ k.float().transpose(-1, -2)) / (D**0.5)
    scores = scores.masked_fill(~keep, float("-inf"))
    out = torch.softmax(scores, dim=-1).nan_to_num(0.0) @ v.float()
    lse = torch.logsumexp(scores, dim=-1) * LOG2E
    return out, lse


def run_and_compare(ragged, num_blocks=8, topk=4, heads=4, atol=0.02, q_prescaled=False, pad_to=None):
    q, k, v, idx, num, vbs = make_case(num_blocks=num_blocks, topk=topk, heads=heads, ragged=ragged, pad_to=pad_to)
    assert vsa.is_supported(q, vbs)
    q_in = (q.float() * (LOG2E / HEAD_DIM**0.5)).to(torch.bfloat16) if q_prescaled else q
    got, got_lse = vsa.block_sparse_attn_ptx_sm107a(q_in, k, v, idx, num, vbs, q_prescaled=q_prescaled)
    # the reference sees the kernel's actual q (pre-scaling adds one bf16 rounding)
    ref_q = (q_in.float() / (LOG2E / HEAD_DIM**0.5)) if q_prescaled else q
    ref, ref_lse = reference(ref_q, k, v, idx, num, vbs)
    diff = (got.float() - ref).abs().max().item()
    assert diff < atol, f"out: max |diff| = {diff:.5f}"
    lse_diff = (got_lse.float() - ref_lse).abs().max().item()
    assert lse_diff < 0.05, f"lse: max |diff| = {lse_diff:.5f}"


@pytest.mark.parametrize("schedule", ["m_split", "dual_q"])
def test_forward_matches_reference(schedule):
    run_and_compare(ragged=False, heads=schedule_heads(schedule))


@pytest.mark.parametrize("schedule", ["m_split", "dual_q"])
def test_forward_matches_reference_ragged(schedule):
    """variable_block_sizes is what FastVideo always passes; padded keys must be masked."""
    run_and_compare(ragged=True, heads=schedule_heads(schedule))


@pytest.mark.parametrize("schedule", ["m_split", "dual_q"])
@pytest.mark.parametrize("topk", [1, 2, 3, 5, 7])
def test_odd_and_small_topk(schedule, topk):
    """M-split splits a list across the two M-tiles; an odd count leaves M-tile 1 a masked last step."""
    run_and_compare(ragged=True, num_blocks=8, topk=topk, heads=schedule_heads(schedule))


@pytest.mark.parametrize("num_blocks", [2, 4, 8, 16])
def test_sequence_lengths(num_blocks):
    run_and_compare(ragged=True, num_blocks=num_blocks, topk=min(3, num_blocks))


@pytest.mark.parametrize("schedule", ["m_split", "dual_q"])
def test_padded_rows(schedule):
    """map_to_index pads rows to num_blocks with -1; the padding is never used as a block id."""
    run_and_compare(ragged=True, num_blocks=8, topk=3, heads=schedule_heads(schedule), pad_to=8)


@pytest.mark.parametrize("schedule", ["m_split", "dual_q"])
def test_q_prescaled(schedule):
    """q multiplied by log2(e)/sqrt(d) by the caller: same output and lse contract."""
    run_and_compare(ragged=True, heads=schedule_heads(schedule), q_prescaled=True)


def test_lse_is_not_vacuous():
    """Guards the lse assertion: a wrong lse must actually fail the comparison."""
    q, k, v, idx, num, vbs = make_case(ragged=True)
    _, got_lse = vsa.block_sparse_attn_ptx_sm107a(q, k, v, idx, num, vbs)
    _, ref_lse = reference(q, k, v, idx, num, vbs)
    assert (got_lse.float() - (ref_lse + 1.0)).abs().max().item() > 0.5


def test_inference_no_lse_matches_eager_under_fullgraph_compile():
    """The production no-LSE route is one opaque, fake-backed graph node."""
    q, k, v, idx, num, vbs = make_case(ragged=True)

    def inference_path(q, k, v, idx, num, vbs):
        out, lse = vsa.block_sparse_attn_ptx_sm107a(q, k, v, idx, num, vbs, need_lse=False)
        assert lse is None
        return out

    with torch.inference_mode():
        expected = inference_path(q, k, v, idx, num, vbs)
        compiled = torch.compile(inference_path, backend="eager", fullgraph=True)
        actual = compiled(q, k, v, idx, num, vbs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_inference_custom_op_fake_matches_real():
    """Pin the custom op schema and fake output metadata used by Dynamo."""
    q, k, v, idx, num, vbs = make_case(ragged=True)
    torch.library.opcheck(
        torch.ops.fastvideo_kernel.block_sparse_attn_ptx_sm107a_inference.default,
        (q, k, v, idx, num, vbs, False),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )


def test_mask_inference_matches_index_route_under_fullgraph_compile():
    """H3's bool mask compaction and the launch stay in one opaque op."""
    batch, heads, num_blocks = 1, 4, 8
    q, k, v, idx, num, vbs = make_case(batch=batch, heads=heads, num_blocks=num_blocks, ragged=True)
    block_map = make_block_map(idx, num, batch, heads, num_blocks)

    def inference_path(q, k, v, block_map, vbs):
        out, lse = vsa.block_sparse_attn_ptx_sm107a_from_mask(q, k, v, block_map, vbs)
        assert lse is None
        return out

    with torch.inference_mode():
        expected, _ = vsa.block_sparse_attn_ptx_sm107a(q, k, v, idx, num, vbs, need_lse=False)
        compiled = torch.compile(inference_path, backend="eager", fullgraph=True)
        actual = compiled(q, k, v, block_map, vbs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_unsupported_is_rejected():
    q, _, _, _, _, vbs = make_case()
    assert not vsa.is_supported(q.float(), vbs)                  # wrong dtype
    assert not vsa.is_supported(q[..., :64].contiguous(), vbs)   # wrong head_dim
    odd = torch.full((7, ), BLOCK, dtype=torch.int32, device="cuda")
    assert not vsa.is_supported(q, odd)                          # seqlen / blocks mismatch
    sixty_four = torch.full((16, ), 64, dtype=torch.int32, device="cuda")
    assert not vsa.is_supported(q, sixty_four)                   # 64-token blocks: no build
    assert not vsa.is_supported(q.transpose(1, 2), vbs)          # not contiguous [B, H, S, D]


# ---------------------------------------------------------------------------
# Per-q-block q2k_num. A dual-Q CTA item owns an ADJACENT PAIR of query blocks; the kernel must honor each
# row's own count (the id clamp, the SHARE_KV windows and the masking) -- the pair runs max(c0, c1, 1) steps.
# ---------------------------------------------------------------------------


def make_two_class_case(num_blocks=16, prefix=3, topk=4, heads=4, seed=0, zero_rows=()):
    """`prefix` DENSE rows (count = num_blocks) then rows at count prefix + topk. With an ODD `prefix` the pair
    (prefix-1, prefix) straddles the classes, so the two rows of one item carry different counts."""
    torch.manual_seed(seed)
    S = num_blocks * BLOCK
    q, k, v = (torch.randn((1, heads, S, HEAD_DIM), device="cuda", dtype=torch.bfloat16) for _ in range(3))
    vbs = torch.randint(BLOCK // 2, BLOCK + 1, (num_blocks, ), dtype=torch.int32, device="cuda")
    rows = heads * num_blocks
    # Pad past each row's count with 0 -- a VALID block id -- so a pair-shared-count regression reads plausible
    # in-bounds data and fails by wrong values.
    idx = torch.zeros((rows, num_blocks), dtype=torch.int32, device="cuda")
    num = torch.zeros((rows, ), dtype=torch.int32, device="cuda")
    g = torch.Generator().manual_seed(seed + 1)
    for h in range(heads):
        for t in range(num_blocks):
            r = h * num_blocks + t
            if t < prefix:
                sel = torch.arange(num_blocks, dtype=torch.int32)
            else:
                vid = torch.randperm(num_blocks - prefix, generator=g)[:topk] + prefix
                sel = torch.cat([torch.arange(prefix), vid.sort().values]).to(torch.int32)
            if t in zero_rows:
                sel = sel[:0]
            idx[r, :sel.numel()] = sel.cuda()
            num[r] = sel.numel()
    return q, k, v, idx, num, vbs


@pytest.mark.parametrize("schedule", ["m_split", "dual_q"])
def test_nonuniform_counts_straddling_a_pair(schedule):
    q, k, v, idx, num, vbs = make_two_class_case(heads=schedule_heads(schedule, num_blocks=16))
    assert vsa.is_supported(q, vbs)
    got, got_lse = vsa.block_sparse_attn_ptx_sm107a(q, k, v, idx, num, vbs)
    ref, ref_lse = reference(q, k, v, idx, num, vbs)
    diff = (got.float() - ref).abs().max().item()
    assert diff < 0.02, f"out: max |diff| = {diff:.5f}"
    lse_diff = (got_lse.float() - ref_lse).abs().max().item()
    assert lse_diff < 0.05, f"lse: max |diff| = {lse_diff:.5f}"


@pytest.mark.parametrize("schedule", ["m_split", "dual_q"])
def test_same_seed_determinism(schedule):
    """Two same-seed runs must be bitwise identical."""
    runs = []
    for _ in range(2):
        q, k, v, idx, num, vbs = make_case(ragged=True, seed=3, heads=schedule_heads(schedule))
        o, m = vsa.block_sparse_attn_ptx_sm107a(q, k, v, idx, num, vbs)
        torch.cuda.synchronize()
        runs.append((o, m))
    assert torch.equal(runs[0][0], runs[1][0]), "out differs across same-seed runs"
    assert torch.equal(runs[0][1], runs[1][1]), "lse differs across same-seed runs"


def _zero_count_case_main(heads):
    """Body of test_zero_count_rows, run in a subprocess (see the test)."""
    zero_rows = (5, 6, 8, 9)   # odd member; even member with nonzero sibling 7; a whole pair
    q, k, v, idx, num, vbs = make_two_class_case(zero_rows=zero_rows, heads=heads)
    assert vsa.is_supported(q, vbs)
    got, got_lse = vsa.block_sparse_attn_ptx_sm107a(q, k, v, idx, num, vbs)
    torch.cuda.synchronize()
    ref, ref_lse = reference(q, k, v, idx, num, vbs)
    got_o = got.float()
    empty = torch.zeros(ref.shape[:-1], dtype=torch.bool, device="cuda")   # [B, H, S]
    for t in zero_rows:
        rows = slice(t * BLOCK, (t + 1) * BLOCK)
        empty[:, :, rows] = True
        zmax = got_o[:, :, rows].abs().max().item()
        assert zmax == 0.0, f"zero-count tile {t}: expected exact zeros, got max |out| = {zmax}"
    assert torch.isfinite(got_lse).all(), "lse must stay finite on zero-count rows"
    diff = (got_o - ref).abs().amax(dim=-1)
    assert diff[~empty].max().item() < 0.02
    lse_diff = (got_lse.float() - ref_lse).abs()
    assert lse_diff[~empty].max().item() < 0.05
    print("ZERO_COUNT_CASE_OK")


@pytest.mark.parametrize("schedule", ["m_split", "dual_q"])
def test_zero_count_rows(schedule):
    """q2k_num == 0 rows produce exactly-zero output (finite lse) and leave every other row intact. In a
    subprocess with a timeout: the failure mode this pins is a HANG (a zero trip count starved the
    correction warps' barrier), which must fail the suite, not wedge it."""
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--zero-count-case",
         str(schedule_heads(schedule, num_blocks=16))],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0 and "ZERO_COUNT_CASE_OK" in proc.stdout, (
        f"rc={proc.returncode}\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--zero-count-case":
        _zero_count_case_main(int(sys.argv[2]))
