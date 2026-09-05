# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the sm_100a CUDA block-sparse VSA backward.

Reference: fp32 dense attention restricted to the selected blocks, keys past
variable_block_sizes masked to -inf, differentiated with torch autograd on fp32 copies of the
bf16 inputs with loss = (out * grad_o).sum(). The kernel is fed the reference's lse in Triton's
M format (logsumexp * log2e) and the reference output rounded to bf16 as ``o`` -- exactly what
the sm_100a forward hands it in FastVideo. The k2q inversion is done here in torch (a stable
sort) so the tests do not need Triton.

Run with: python -m pytest tests/test_block_sparse_bwd_sm100a.py -v
"""

import itertools
import os

import pytest
import torch

from fastvideo_kernel import block_sparse_attn_bwd_sm100a as bwd

HEAD_DIM = 128
BLOCK = 64
LOG2E = 1.4426950408889634

# Per-tensor tolerances on the bf16 outputs against the fp32 reference. Measured over all 15
# cases on GB200 (2026-09-03, VSA_BWD_TEST_VERBOSE=1): max|diff|/max|ref| up to 3.0e-3 (dq) and
# 5.3e-3 (dk, dv); mean|diff| up to 2.8e-4 against mean|ref| of 0.06-0.12. The bounds below leave
# about 2x (rel max) and 3.5x (mean abs) headroom.
REL_MAX_TOL = 1e-2    # max|got - ref| / max|ref|
MEAN_ABS_TOL = 1e-3   # mean|got - ref|

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0)
    or not bwd._HAS_VSA_BWD_SM100A,
    reason="requires a compute-capability (10, 0) GPU (sm_100a) and a fastvideo_kernel "
    "extension built with block_sparse_sm100a_bwd",
)


def make_case(num_blocks=8, topk=4, heads=4, batch=1, ragged=False, seed=0, kv_pool=None):
    """Random bf16 q/k/v/grad_o plus q2k metadata [B, H, Nq, topk] / [B, H, Nq] and vbs.

    ``kv_pool`` restricts the kv blocks a q block may select (default: all), which is how the
    zero-count case leaves some kv blocks unselected by every q block.
    """
    torch.manual_seed(seed)
    S = num_blocks * BLOCK
    shape = (batch, heads, S, HEAD_DIM) if bwd.BHSD else (batch, S, heads, HEAD_DIM)
    q, k, v, grad_o = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4))

    pool = torch.arange(num_blocks) if kv_pool is None else torch.as_tensor(list(kv_pool))
    assert topk <= pool.numel()
    rows = batch * heads * num_blocks
    idx = torch.empty((rows, topk), dtype=torch.int32)
    for r in range(rows):
        idx[r] = pool[torch.randperm(pool.numel())[:topk]].sort().values.to(torch.int32)
    idx = idx.view(batch, heads, num_blocks, topk).cuda()
    num = torch.full((batch, heads, num_blocks), topk, dtype=torch.int32, device="cuda")

    if ragged:
        vbs = torch.randint(BLOCK // 2, BLOCK + 1, (num_blocks, ), dtype=torch.int32,
                            device="cuda")
    else:
        vbs = torch.full((num_blocks, ), BLOCK, dtype=torch.int32, device="cuda")
    return q, k, v, grad_o, idx, num, vbs


def invert_indices_torch(q2k_idx, q2k_num, num_kv_blocks, pad_value=0):
    """k2q from q2k without Triton: a stable sort, so each row lists its q blocks ascending.

    Returns (k2q_idx [B, H, num_kv_blocks, Nq] int32, k2q_num [B, H, num_kv_blocks] int32), the
    layout fastvideo_kernel.triton_kernels.index.invert_indices produces. Entries past a row's
    count hold ``pad_value`` -- a VALID block id, so an over-read fails by wrong values rather
    than by luck.
    """
    B, H, Nq, Mk = q2k_idx.shape
    device = q2k_idx.device
    valid = torch.arange(Mk, device=device).view(1, 1, 1, Mk) < q2k_num.view(B, H, Nq, 1)
    row = (torch.arange(B * H, device=device).view(B, H, 1, 1) * num_kv_blocks
           + q2k_idx.long())[valid]
    qblock = torch.arange(Nq, device=device).view(1, 1, Nq, 1).expand(B, H, Nq, Mk)[valid]
    order = torch.sort(row * Nq + qblock).indices
    row, qblock = row[order], qblock[order]
    counts = torch.bincount(row, minlength=B * H * num_kv_blocks)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(row.numel(), device=device) - starts[row]
    k2q_idx = torch.full((B * H * num_kv_blocks, Nq), pad_value, dtype=torch.int32,
                         device=device)
    k2q_idx[row, slot] = qblock.to(torch.int32)
    return (k2q_idx.view(B, H, num_kv_blocks, Nq),
            counts.to(torch.int32).view(B, H, num_kv_blocks))


def reference(q, k, v, grad_o, idx, num, vbs):
    """fp32 masked-dense autograd reference.

    Returns (o bf16, lse fp32 [B, H, S] in M format, dq, dk, dv fp32) -- o and the grads in
    q's layout.
    """
    if not bwd.BHSD:
        q, k, v, grad_o = (t.transpose(1, 2) for t in (q, k, v, grad_o))   # -> [B, H, S, D]
    B, H, S, D = q.shape
    num_blocks = vbs.numel()
    scale = 1.0 / (D**0.5)

    idx, num, vbs = idx.cpu(), num.cpu(), vbs.cpu()
    keep = torch.zeros((B, H, S, S), dtype=torch.bool, device=q.device)
    for b in range(B):
        for h in range(H):
            for qb in range(num_blocks):
                for j in range(int(num[b, h, qb])):
                    kb = int(idx[b, h, qb, j])
                    valid = int(vbs[kb])
                    keep[b, h, qb * BLOCK:(qb + 1) * BLOCK, kb * BLOCK:kb * BLOCK + valid] = True

    q32, k32, v32 = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    scores = (q32 @ k32.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~keep, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out = p @ v32
    loss = (out * grad_o.float()).sum()
    dq, dk, dv = torch.autograd.grad(loss, (q32, k32, v32))
    lse = (torch.logsumexp(scores, dim=-1) * LOG2E).detach().contiguous()
    o = out.detach().to(torch.bfloat16)

    if not bwd.BHSD:
        o, dq, dk, dv = (t.transpose(1, 2).contiguous() for t in (o, dq, dk, dv))
    return o, lse, dq.detach(), dk.detach(), dv.detach()


def check_close(name, got, ref):
    got, ref = got.float(), ref.float()
    assert got.shape == ref.shape, f"{name}: shape {tuple(got.shape)} vs {tuple(ref.shape)}"
    assert torch.isfinite(got).all(), f"{name}: non-finite values"
    diff = (got - ref).abs()
    rel_max = diff.max().item() / max(ref.abs().max().item(), 1e-6)
    mean_abs = diff.mean().item()
    if os.environ.get("VSA_BWD_TEST_VERBOSE"):
        print(f"{name}: rel_max={rel_max:.3e} mean_abs={mean_abs:.3e} "
              f"mean|ref|={ref.abs().mean().item():.3e}")
    assert rel_max <= REL_MAX_TOL and mean_abs <= MEAN_ABS_TOL, (
        f"{name}: max|diff|/max|ref| = {rel_max:.3e} (tol {REL_MAX_TOL:.1e}), "
        f"mean|diff| = {mean_abs:.3e} (tol {MEAN_ABS_TOL:.1e}), "
        f"max|ref| = {ref.abs().max().item():.3e}, mean|ref| = {ref.abs().mean().item():.3e}")


def run_and_compare(num_blocks=8, topk=4, heads=4, batch=1, ragged=False, seed=0, kv_pool=None):
    q, k, v, grad_o, idx, num, vbs = make_case(num_blocks=num_blocks, topk=topk, heads=heads,
                                               batch=batch, ragged=ragged, seed=seed,
                                               kv_pool=kv_pool)
    assert bwd.is_supported(q, vbs)
    o, lse, ref_dq, ref_dk, ref_dv = reference(q, k, v, grad_o, idx, num, vbs)
    k2q_idx, k2q_num = invert_indices_torch(idx, num, num_blocks)
    dq, dk, dv = bwd.block_sparse_attn_backward_sm100a_from_k2q(grad_o, q, k, v, o, lse,
                                                                k2q_idx, k2q_num, vbs)
    torch.cuda.synchronize()
    for name, got, ref in (("dq", dq, ref_dq), ("dk", dk, ref_dk), ("dv", dv, ref_dv)):
        assert got.dtype == torch.bfloat16, f"{name}: dtype {got.dtype}"
        check_close(name, got, ref)
    return (dq, dk, dv), (ref_dq, ref_dk, ref_dv), vbs


def _to_bhsd(t):
    return t if bwd.BHSD else t.transpose(1, 2)


def test_backward_matches_reference():
    run_and_compare()


def test_batch_two():
    run_and_compare(batch=2)


def test_ragged_block_sizes():
    """variable_block_sizes is what FastVideo always passes; padded keys must be masked."""
    run_and_compare(ragged=True)


def test_ragged_padded_key_rows_are_zero():
    """Keys at or past a block's count get P^T = 0 in-kernel, so their dk/dv rows are exactly 0
    (Triton's backward stores zeros there as well)."""
    (dq, dk, dv), _, vbs = run_and_compare(ragged=True, seed=1)
    dk, dv = _to_bhsd(dk).float(), _to_bhsd(dv).float()
    checked = 0
    for kb in range(vbs.numel()):
        if int(vbs[kb]) == BLOCK:
            continue  # a full block has no padded rows
        rows = slice(kb * BLOCK + int(vbs[kb]), (kb + 1) * BLOCK)
        assert dk[:, :, rows].abs().max().item() == 0.0, f"dk: padded rows of kv block {kb}"
        assert dv[:, :, rows].abs().max().item() == 0.0, f"dv: padded rows of kv block {kb}"
        checked += 1
    assert checked > 0, "the ragged draw produced no padded kv block; change the seed"


@pytest.mark.parametrize("topk", [1, 2, 3, 5, 7])
def test_topk_not_a_multiple_of_the_quad(topk):
    """The kernel walks each kv block's q list in quads of 4; a ragged tail quad must be exact."""
    run_and_compare(ragged=True, num_blocks=8, topk=topk)


@pytest.mark.parametrize("num_blocks", [4, 8, 16])
def test_sequence_lengths(num_blocks):
    run_and_compare(ragged=True, num_blocks=num_blocks, topk=3)


def test_zero_count_kv_blocks():
    """kv blocks no q block selects: the main kernel skips them and the preprocess must write
    their dk/dv rows as exact zeros (the outputs are empty_like), every other block exact."""
    num_blocks = 8
    excluded = (2, 5)
    pool = [kb for kb in range(num_blocks) if kb not in excluded]
    (dq, dk, dv), _, _ = run_and_compare(num_blocks=num_blocks, topk=4, ragged=True,
                                         kv_pool=pool)
    dk, dv = _to_bhsd(dk).float(), _to_bhsd(dv).float()
    for kb in excluded:
        rows = slice(kb * BLOCK, (kb + 1) * BLOCK)
        assert dk[:, :, rows].abs().max().item() == 0.0, f"dk: unselected kv block {kb}"
        assert dv[:, :, rows].abs().max().item() == 0.0, f"dv: unselected kv block {kb}"


def test_invert_indices_torch_matches_q2k():
    """Guards the test's own k2q builder: every (q, kv) pair appears once, counts add up."""
    num_blocks = 8
    _, _, _, _, idx, num, _ = make_case(num_blocks=num_blocks, topk=3, heads=2, batch=2)
    k2q_idx, k2q_num = invert_indices_torch(idx, num, num_blocks)
    B, H, Nq, Mk = idx.shape
    assert k2q_idx.shape == (B, H, num_blocks, Nq) and k2q_num.shape == (B, H, num_blocks)
    assert int(k2q_num.sum()) == B * H * Nq * Mk
    for b, h, qb, j in itertools.product(range(B), range(H), range(Nq), range(Mk)):
        kb = int(idx[b, h, qb, j])
        listed = k2q_idx[b, h, kb, :int(k2q_num[b, h, kb])].tolist()
        assert listed.count(qb) == 1
        assert listed == sorted(listed)


def test_unsupported_is_rejected():
    q, k, v, grad_o, idx, num, vbs = make_case()
    assert not bwd.is_supported(q.float(), vbs)                 # wrong dtype
    assert not bwd.is_supported(q[..., :64].contiguous(), vbs)  # wrong head_dim
    seven = torch.full((7, ), 64, dtype=torch.int32, device="cuda")
    assert not bwd.is_supported(q, seven)                       # seqlen != 64 * num_blocks
    q500 = (q[:, :, :500] if bwd.BHSD else q[:, :500]).contiguous()
    assert not bwd.is_supported(q500, vbs)                      # S not a multiple of 64
    # An odd block count (S % 128 != 0) is refused statically too, so FastVideo falls back to
    # Triton instead of tripping the binding's check.
    q448 = (q[:, :, :448] if bwd.BHSD else q[:, :448]).contiguous()
    assert not bwd.is_supported(q448, seven)

    # The binding itself refuses bad dtypes before touching the GPU.
    k2q_idx, k2q_num = invert_indices_torch(idx, num, vbs.numel())
    o = torch.zeros_like(q)
    heads = q.shape[1] if bwd.BHSD else q.shape[2]
    lse = torch.zeros((q.shape[0], heads, vbs.numel() * BLOCK), dtype=torch.float32,
                      device="cuda")
    with pytest.raises(RuntimeError):
        bwd.block_sparse_attn_backward_sm100a_from_k2q(grad_o.float(), q.float(), k.float(),
                                                       v.float(), o.float(), lse, k2q_idx,
                                                       k2q_num, vbs)
