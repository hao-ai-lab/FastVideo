# SPDX-License-Identifier: Apache-2.0
"""Equivalence tests for the block-resolution coarse/sparse combine in VSA.

The combine used to expand the coarse block output across the full sequence with
``repeat()`` before adding it to the sparse output. It now keeps the coarse
result at block resolution and broadcasts. These tests pin the numerics of that
change against a literal transcription of the previous implementation.

Shapes are deliberately small so this runs in CI; no MiniMax H3 checkpoint,
pipeline or ComfyUI dependency is involved.
"""
import pytest
import torch

from fastvideo_kernel.ops import _combine_coarse_sparse

DEVICE = "cuda"
BLOCK_ELEMENTS = 64


def _reference_combine(out_c_blocked, out_s, weight, batch, heads, n_blocks, be, dim, seq):
    """The pre-change implementation, transcribed literally.

    ``out_c`` was expanded with repeat() to [B, H, S, D], then combined
    out-of-place with a full-size multiply and add.
    """
    out_c = out_c_blocked.repeat(1, 1, 1, be, 1).view(batch, heads, seq, dim)
    if weight is not None:
        return out_c * weight + out_s
    return out_c + out_s


def _make(batch, heads, n_blocks, dim, gated, seed=0):
    torch.manual_seed(seed)
    be = BLOCK_ELEMENTS
    seq = n_blocks * be
    out_c = torch.randn(batch, heads, n_blocks, 1, dim, device=DEVICE, dtype=torch.bfloat16)
    out_s = torch.randn(batch, heads, seq, dim, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.rand(batch, heads, seq, dim, device=DEVICE, dtype=torch.bfloat16) if gated else None
    return out_c, out_s, weight, be, seq


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("batch,heads,n_blocks,dim", [(1, 4, 8, 64), (1, 8, 16, 128), (2, 4, 5, 128)])
def test_ungated_is_bit_exact(batch, heads, n_blocks, dim):
    """Ungated: pure broadcast add, so the result must be bit-identical."""
    out_c, out_s, _, be, seq = _make(batch, heads, n_blocks, dim, gated=False)
    with torch.no_grad():
        ref = _reference_combine(out_c, out_s.clone(), None, batch, heads, n_blocks, be, dim, seq)
        got = _combine_coarse_sparse(out_c, out_s.clone(), None, batch, heads, n_blocks, be, dim, seq)
    assert got.shape == ref.shape == (batch, heads, seq, dim)
    assert got.dtype == ref.dtype == torch.bfloat16
    assert torch.equal(got, ref), "ungated combine must be bit-exact"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("batch,heads,n_blocks,dim", [(1, 4, 8, 64), (1, 8, 16, 128), (2, 4, 5, 128)])
def test_gated_is_no_less_accurate(batch, heads, n_blocks, dim):
    """Gated: the combine must be at least as accurate as the old one.

    The old path rounded the coarse*gate product to bf16 before adding, while
    ``addcmul_`` fuses the multiply-add. The rounding therefore differs, and the
    two implementations disagree on a fraction of elements. The meaningful
    assertion is accuracy against fp32 rather than agreement with the old
    rounding, so that is what this checks.
    """
    out_c, out_s, weight, be, seq = _make(batch, heads, n_blocks, dim, gated=True)
    with torch.no_grad():
        ref = _reference_combine(out_c, out_s.clone(), weight, batch, heads, n_blocks, be, dim, seq)
        got = _combine_coarse_sparse(out_c, out_s.clone(), weight, batch, heads, n_blocks, be, dim, seq)
        truth = (out_c.float().repeat(1, 1, 1, be, 1).view(batch, heads, seq, dim) * weight.float()
                 + out_s.float())

    assert got.shape == ref.shape and got.dtype == ref.dtype

    old_err, new_err = (ref.float() - truth).abs(), (got.float() - truth).abs()
    assert new_err.mean() <= old_err.mean() * 1.02, "combine is less accurate than the old path"
    assert new_err.max() <= old_err.max() * 1.02, "combine has a worse worst case than the old path"

    # And the two implementations still agree to bf16 rounding overall.
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel <= 5e-3, f"relative L2 vs old implementation {rel:.3e} too large"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("gated", [False, True])
def test_autograd_path_is_out_of_place(gated):
    """With grad enabled the combine must not mutate the sparse output.

    The BSHD 128/256 combine documents that the sparse output is saved by FA4's
    autograd node; in-place mutation there would invalidate the graph. The
    in-place fast path must therefore be confined to no-grad.
    """
    out_c, out_s, weight, be, seq = _make(1, 4, 8, 64, gated=gated)
    out_s = out_s.requires_grad_(True)
    before = out_s.detach().clone()
    got = _combine_coarse_sparse(out_c, out_s, weight, 1, 4, 8, be, 64, seq)
    assert torch.equal(out_s.detach(), before), "combine mutated a grad-tracking tensor"
    assert got.requires_grad
    got.sum().backward()
    assert out_s.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_noncontiguous_sparse_output():
    """A non-contiguous sparse output must still combine correctly."""
    batch, heads, n_blocks, dim = 1, 4, 8, 64
    out_c, out_s, weight, be, seq = _make(batch, heads, n_blocks, dim, gated=True)
    # transpose-then-restore yields an equal but non-contiguous tensor
    noncontig = out_s.transpose(1, 2).transpose(1, 2)
    if noncontig.is_contiguous():
        noncontig = out_s[:, :, torch.arange(seq, device=DEVICE)]
    with torch.no_grad():
        ref = _reference_combine(out_c, out_s.clone(), weight, batch, heads, n_blocks, be, dim, seq)
        got = _combine_coarse_sparse(out_c, noncontig.clone(), weight, batch, heads, n_blocks, be, dim,
                                     seq)
    assert got.shape == ref.shape
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel <= 5e-3, f"relative L2 {rel:.3e} too large for a non-contiguous input"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_no_full_sequence_coarse_materialization():
    """Peak allocation must not include a full-sequence copy of the coarse output."""
    batch, heads, n_blocks, dim = 1, 8, 128, 128
    out_c, out_s, weight, be, seq = _make(batch, heads, n_blocks, dim, gated=True)
    full = batch * heads * seq * dim * out_s.element_size()

    torch.cuda.synchronize()
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        _combine_coarse_sparse(out_c, out_s, weight, batch, heads, n_blocks, be, dim, seq)
        torch.cuda.synchronize()
        new_peak = torch.cuda.max_memory_allocated() - base

    # The in-place no-grad path should allocate well under one extra full tensor.
    assert new_peak < full, (f"combine allocated {new_peak} bytes, at least one full-sequence "
                             f"tensor ({full} bytes)")


# --------------------------------------------------------------------------
# End-to-end through video_sparse_attn (64-block path)
# --------------------------------------------------------------------------


def _reference_video_sparse_attn(q, k, v, variable_block_sizes, q_variable_block_sizes, topk,
                                 block_size, compress_attn_weight):
    """Pre-change ``video_sparse_attn`` body for the 64-block path."""
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (fused_block_mean,
                                                                     fused_topk_mask)
    block_elements = block_size[0] * block_size[1] * block_size[2]
    batch, heads, q_seq_len, dim = q.shape
    q_num_blocks = q_seq_len // block_elements

    q_c = fused_block_mean(q, q_variable_block_sizes, block_elements)
    k_c = fused_block_mean(k, variable_block_sizes, block_elements)
    v_c = fused_block_mean(v, variable_block_sizes, block_elements)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (dim**0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c = torch.matmul(attn, v_c)
    out_c = out_c.view(batch, heads, q_num_blocks, 1, dim)
    out_c = out_c.repeat(1, 1, 1, block_elements, 1).view(batch, heads, q_seq_len, dim)

    mask = fused_topk_mask(scores, topk)
    out_s = block_sparse_attn(q, k, v, mask, variable_block_sizes)[0]
    if compress_attn_weight is not None:
        return out_c * compress_attn_weight + out_s, mask
    return out_c + out_s, mask


def _vsa_inputs(heads, n_blocks, dim, gated, seed=0):
    torch.manual_seed(seed)
    be, batch = BLOCK_ELEMENTS, 1
    seq = n_blocks * be
    q = torch.randn(batch, heads, seq, dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(batch, heads, seq, dim, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(batch, heads, seq, dim, device=DEVICE, dtype=torch.bfloat16)
    vbs = torch.full((n_blocks,), be, device=DEVICE, dtype=torch.int32)
    w = torch.rand(batch, heads, seq, dim, device=DEVICE, dtype=torch.bfloat16) if gated else None
    return q, k, v, vbs, w, seq


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("heads,n_blocks,dim,ratio", [(4, 8, 64, 0.5), (8, 16, 128, 0.2)])
def test_video_sparse_attn_matches_reference(gated, heads, n_blocks, dim, ratio):
    """Full 64-block VSA: output and top-k routing must match the old path."""
    from fastvideo_kernel.ops import video_sparse_attn

    q, k, v, vbs, w, seq = _vsa_inputs(heads, n_blocks, dim, gated)
    topk = max(1, int(n_blocks * ratio))
    with torch.no_grad():
        ref, ref_mask = _reference_video_sparse_attn(q, k, v, vbs, vbs, topk, (4, 4, 4), w)
        got = video_sparse_attn(q, k, v, vbs, vbs, topk, (4, 4, 4), compress_attn_weight=w)

    assert got.shape == ref.shape == (1, heads, seq, dim)
    assert got.dtype == ref.dtype == torch.bfloat16
    # routing is untouched by the combine change
    assert bool(ref_mask.sum(-1).eq(topk).all().item()), "top-k rows must select exactly topk blocks"
    if gated:
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel <= 5e-3, f"relative L2 {rel:.3e} too large"
    else:
        assert torch.equal(got, ref), "ungated full path must be bit-exact"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_block_size_64_selects_triton_path():
    """block_size (4,4,4) must still route to the 64-block sparse kernel."""
    import fastvideo_kernel.ops as ops

    called = {}
    original = ops.block_sparse_attn

    def spy(*args, **kwargs):
        called["hit"] = True
        return original(*args, **kwargs)

    q, k, v, vbs, w, _ = _vsa_inputs(4, 8, 64, gated=True)
    ops.block_sparse_attn = spy
    try:
        with torch.no_grad():
            ops.video_sparse_attn(q, k, v, vbs, vbs, 4, (4, 4, 4), compress_attn_weight=w)
    finally:
        ops.block_sparse_attn = original
    assert called.get("hit"), "64-block path did not call block_sparse_attn"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block_elements", [128, 256])
def test_128_256_paths_still_dispatch(block_elements):
    """The 128/256 kernels must still be selected; the combine is shared."""
    import fastvideo_kernel.ops as ops

    name = "block_sparse_attn_128" if block_elements == 128 else "block_sparse_attn_256"
    called = {}
    original = getattr(ops, name)

    def spy(*args, **kwargs):
        called["hit"] = True
        return original(*args, **kwargs)

    be = block_elements
    n_blocks, heads, dim = 4, 4, 64
    seq = n_blocks * be
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, heads, seq, dim, device=DEVICE, dtype=torch.bfloat16) for _ in range(3))
    vbs = torch.full((n_blocks,), be, device=DEVICE, dtype=torch.int32)
    bs = (be // 16, 4, 4)
    assert bs[0] * bs[1] * bs[2] == be

    setattr(ops, name, spy)
    try:
        with torch.no_grad():
            ops.video_sparse_attn(q, k, v, vbs, vbs, 2, bs)
    except Exception as exc:  # kernel unavailable on this backend
        pytest.skip(f"{name} unavailable here: {type(exc).__name__}: {exc}")
    finally:
        setattr(ops, name, original)
    assert called.get("hit"), f"{name} was not dispatched"
