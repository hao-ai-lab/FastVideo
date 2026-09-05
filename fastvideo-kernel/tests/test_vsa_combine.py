# SPDX-License-Identifier: Apache-2.0
"""Tests for the block-resolution coarse/sparse combine in VSA.

The combine used to expand the coarse block output across the full sequence with
``repeat()`` before multiplying by the gate and adding the sparse output, three
full-sequence temporaries in all. It now keeps the coarse result at block
resolution, broadcasts it over a blocked view, and accumulates into the sparse
output in place whenever that tensor is not part of an autograd graph.

These tests pin the numerics against a literal transcription of the previous
implementation and against an fp32 oracle, and pin the allocation behaviour of
both branches. Shapes are deliberately small so this runs in CI.
"""
import pytest
import torch

import fastvideo_kernel.ops as ops
from fastvideo_kernel.ops import _combine_coarse_sparse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

DEVICE = "cuda"
BE = 64
# bf16 keeps 8 significand bits, so a single round-to-nearest of an fp32 value
# is off by at most half an ulp, i.e. at most 2**-8 relative.
BF16_HALF_ULP_REL = 2.0**-8

# How the combine is invoked. "grad_leaf" is the training case (``out_s`` is
# saved by the sparse kernel's autograd node and must not be mutated); the other
# three are the cases in which accumulating into ``out_s`` is safe.
MODES = ["no_grad", "inference_mode", "grad_untracked", "grad_leaf"]


def _reference_combine(out_c, out_s, weight, be):
    """The pre-change BHSD implementation, transcribed literally.

    ``out_c`` [B, H, n_blocks, D] was expanded with repeat() to [B, H, S, D],
    then combined out-of-place with a full-size multiply and add.
    """
    batch, heads, n_blocks, dim = out_c.shape
    out_c = out_c.unsqueeze(3).repeat(1, 1, 1, be, 1).view(batch, heads, n_blocks * be, dim)
    if weight is not None:
        return out_c * weight + out_s
    return out_c + out_s


def _reference_combine_bshd(out_c, out_s, weight, be):
    """The pre-change BSHD implementation (``video_sparse_attn_bshd``), transcribed literally."""
    batch, n_blocks, heads, dim = out_c.shape
    out_view = out_s.view(batch, n_blocks, be, heads, dim)
    if weight is not None:
        out = out_view + out_c.unsqueeze(2) * weight.view(batch, n_blocks, be, heads, dim)
    else:
        out = out_view + out_c.unsqueeze(2)
    return out.view(batch, n_blocks * be, heads, dim)


def _fp32_truth(out_c, out_s, weight, be, seq_dim=2):
    """The exact-product, once-rounded fp32 value of the combine."""
    out_c = out_c.float().unsqueeze(seq_dim + 1)
    blocked = out_s.shape[:seq_dim] + (out_c.shape[seq_dim], be) + out_s.shape[seq_dim + 1:]
    out_s = out_s.float().view(*blocked)
    if weight is not None:
        return (out_c * weight.float().view(*blocked) + out_s).view(weight.shape)
    return (out_c + out_s).view(out_s.shape[:seq_dim] + (-1, ) + out_s.shape[seq_dim + 2:])


def _make(batch, heads, n_blocks, dim, gated, be=BE, seed=0, dtype=torch.bfloat16, seq_dim=2):
    torch.manual_seed(seed)
    seq = n_blocks * be
    if seq_dim == 2:
        out_c = torch.randn(batch, heads, n_blocks, dim, device=DEVICE, dtype=dtype)
        full = (batch, heads, seq, dim)
    else:
        out_c = torch.randn(batch, n_blocks, heads, dim, device=DEVICE, dtype=dtype)
        full = (batch, seq, heads, dim)
    out_s = torch.randn(*full, device=DEVICE, dtype=dtype)
    weight = torch.rand(*full, device=DEVICE, dtype=dtype) if gated else None
    return out_c, out_s, weight


def _run(mode, out_c, out_s, weight, be=BE, seq_dim=2):
    """Run the combine under ``mode`` on clones; returns (result, out_s clone)."""
    out_s = out_s.clone()
    if mode == "no_grad":
        with torch.no_grad():
            return _combine_coarse_sparse(out_c, out_s, weight, be, seq_dim), out_s
    if mode == "inference_mode":
        with torch.inference_mode():
            return _combine_coarse_sparse(out_c, out_s, weight, be, seq_dim), out_s
    if mode == "grad_untracked":
        with torch.enable_grad():
            return _combine_coarse_sparse(out_c, out_s, weight, be, seq_dim), out_s
    assert mode == "grad_leaf"
    out_s.requires_grad_(True)
    with torch.enable_grad():
        return _combine_coarse_sparse(out_c, out_s, weight, be, seq_dim), out_s


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch,heads,n_blocks,dim", [(1, 4, 8, 64), (1, 8, 16, 128), (2, 4, 5, 128)])
def test_ungated_is_bit_exact(mode, batch, heads, n_blocks, dim):
    """Ungated: pure broadcast add, so every branch must be bit-identical to the old path."""
    out_c, out_s, _ = _make(batch, heads, n_blocks, dim, gated=False)
    ref = _reference_combine(out_c, out_s.clone(), None, BE)
    got, _ = _run(mode, out_c, out_s, None)
    assert got.shape == ref.shape == (batch, heads, n_blocks * BE, dim)
    assert got.dtype == ref.dtype == torch.bfloat16
    assert torch.equal(got.detach(), ref), "ungated combine must be bit-exact"


@pytest.mark.parametrize("batch,heads,n_blocks,dim", [(1, 4, 8, 64), (1, 8, 16, 128), (2, 4, 5, 128)])
def test_gated_is_within_one_bf16_rounding_of_fp32(batch, heads, n_blocks, dim):
    """Gated: every element is the once-rounded fp32 value, in every branch.

    ``addcmul`` computes ``out_s + out_c * w`` in fp32 and rounds once. The
    product of two bf16 values is exact in fp32, so the fused result is within
    half a bf16 ulp of the fp32 truth for every element. The old path rounded
    the product to bf16 before adding and violates that bound on a noticeable
    fraction of elements, so this assertion discriminates the two and would
    catch a regression to an unfused implementation.
    """
    out_c, out_s, weight = _make(batch, heads, n_blocks, dim, gated=True)
    truth = _fp32_truth(out_c, out_s, weight, BE)
    ref = _reference_combine(out_c, out_s.clone(), weight, BE)
    bound = BF16_HALF_ULP_REL * truth.abs()

    results = {mode: _run(mode, out_c, out_s, weight)[0].detach() for mode in MODES}
    for mode, got in results.items():
        assert got.shape == ref.shape and got.dtype == ref.dtype
        new_err = (got.float() - truth).abs()
        assert bool((new_err <= bound).all()), f"[{mode}] combine is not within one bf16 rounding of fp32"

    # The in-place and out-of-place branches must agree bit for bit, so results
    # do not depend on whether the caller runs under grad.
    first = results[MODES[0]]
    for mode, got in results.items():
        assert torch.equal(got, first), f"[{mode}] differs from [{MODES[0]}]"

    # The old path is never more accurate.
    old_err = (ref.float() - truth).abs()
    new_err = (first.float() - truth).abs()
    assert new_err.mean() <= old_err.mean()
    assert new_err.max() <= old_err.max()
    # And the two implementations still agree to bf16 rounding overall.
    rel = ((first.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel <= 5e-3, f"relative L2 vs old implementation {rel:.3e} too large"


# --------------------------------------------------------------------------
# Autograd and aliasing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gated", [False, True])
def test_autograd_path_is_out_of_place(gated):
    """With ``out_s`` in a graph the combine must not mutate it, and grads match the old path."""
    out_c, out_s, weight = _make(1, 4, 8, 64, gated=gated)
    out_s = out_s.requires_grad_(True)
    out_c = out_c.requires_grad_(True)
    if gated:
        weight = weight.requires_grad_(True)
    before = out_s.detach().clone()

    got = _combine_coarse_sparse(out_c, out_s, weight, BE, 2)
    assert torch.equal(out_s.detach(), before), "combine mutated a grad-tracking tensor"
    assert got.requires_grad and got.data_ptr() != out_s.data_ptr()

    torch.manual_seed(1)
    grad_out = torch.randn_like(got)
    got.backward(grad_out)
    got_grads = [t.grad.clone() for t in (out_c, out_s, weight) if t is not None]
    for t in (out_c, out_s, weight):
        if t is not None:
            t.grad = None

    ref = _reference_combine(out_c, out_s, weight, BE)
    ref.backward(grad_out)
    ref_grads = [t.grad for t in (out_c, out_s, weight) if t is not None]
    for name, g, r in zip(("out_c", "out_s", "weight"), got_grads, ref_grads):
        torch.testing.assert_close(g, r, rtol=2.0**-6, atol=0.0, msg=lambda m: f"grad of {name}: {m}")


@pytest.mark.parametrize("gated", [False, True])
def test_untracked_sparse_output_accumulates_in_place_under_grad(gated):
    """Grad enabled but ``out_s`` outside any graph: accumulate into it, keep grads for the rest.

    This is the case an eval/teacher forward or a frozen-attention finetune hits
    when the surrounding code has not wrapped the call in ``no_grad``. In-place
    is safe because no autograd node saved ``out_s`` and addcmul's backward only
    needs the other two operands.
    """
    out_c, out_s, weight = _make(1, 4, 8, 64, gated=gated)
    out_c = out_c.requires_grad_(True)
    if gated:
        weight = weight.requires_grad_(True)
    with torch.enable_grad():
        got = _combine_coarse_sparse(out_c, out_s, weight, BE, 2)
    assert got.data_ptr() == out_s.data_ptr(), "untracked sparse output was not reused"
    assert got.requires_grad
    torch.manual_seed(1)
    grad_out = torch.randn_like(got)
    got.backward(grad_out)

    ref_c = out_c.detach().clone().requires_grad_(True)
    ref_w = weight.detach().clone().requires_grad_(True) if gated else None
    _reference_combine(ref_c, torch.zeros_like(out_s), ref_w, BE).backward(grad_out)
    torch.testing.assert_close(out_c.grad, ref_c.grad, rtol=2.0**-6, atol=0.0)
    if gated:
        torch.testing.assert_close(weight.grad, ref_w.grad, rtol=2.0**-6, atol=0.0)


def test_in_place_requires_no_grad_tracking_on_sparse_output_only():
    """``out_c`` requiring grad under no_grad does not stop the in-place path."""
    out_c, out_s, weight = _make(1, 4, 8, 64, gated=True)
    out_c.requires_grad_(True)
    with torch.no_grad():
        got = _combine_coarse_sparse(out_c, out_s, weight, BE, 2)
    assert got.data_ptr() == out_s.data_ptr()
    assert not got.requires_grad


def test_inference_tensor_contract():
    """An inference-mode ``out_s`` combines in place inside inference mode and raises outside it.

    The helper does not guard this (``Tensor.is_inference`` is untraceable by
    ``torch.compile``); ``video_sparse_attn`` always combines in the mode that
    produced ``out_s``, and this pins the documented behaviour for other callers.
    """
    out_c, out_s, weight = _make(1, 4, 8, 64, gated=True)
    expected = _run("no_grad", out_c, out_s, weight)[0]
    with torch.inference_mode():
        out_s_inf = out_s.clone()
        got = _combine_coarse_sparse(out_c, out_s_inf, weight, BE, 2)
    assert got.is_inference() and got.data_ptr() == out_s_inf.data_ptr()
    assert torch.equal(got, expected)

    with torch.inference_mode():
        out_s_inf = out_s.clone()
    with torch.no_grad(), pytest.raises(RuntimeError, match="[Ii]nference"):
        _combine_coarse_sparse(out_c, out_s_inf, weight, BE, 2)


# --------------------------------------------------------------------------
# Layout, shape and dtype contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["no_grad", "grad_leaf"])
def test_noncontiguous_sparse_output_and_gate(mode):
    """Non-contiguous ``out_s`` and a transposed BSHD gate combine without a copy.

    Splitting the sequence axis is always a view, so neither operand needs to
    be contiguous: the production caller can hand over the transposed BSHD gate
    without ``.contiguous()``.
    """
    batch, heads, n_blocks, dim = 1, 4, 8, 64
    seq = n_blocks * BE
    out_c, out_s, weight = _make(batch, heads, n_blocks, dim, gated=True)
    noncontig_s = out_s.transpose(2, 3).contiguous().transpose(2, 3)
    assert not noncontig_s.is_contiguous() and torch.equal(noncontig_s, out_s)
    torch.manual_seed(3)
    gate_bshd = torch.rand(batch, seq, heads, dim, device=DEVICE, dtype=torch.bfloat16)
    gate = gate_bshd.transpose(1, 2)
    assert not gate.is_contiguous()

    contiguous_result, _ = _run(mode, out_c, out_s, gate.contiguous())
    got, arg = _run(mode, out_c, noncontig_s, gate)
    assert not arg.is_contiguous(), "fixture lost its strides"
    assert got.shape == (batch, heads, seq, dim)
    assert torch.equal(got.detach(), contiguous_result.detach()), "strides changed the arithmetic"
    if mode == "no_grad":
        assert got.data_ptr() == arg.data_ptr(), "non-contiguous out_s was copied instead of reused"


def test_shape_mismatch_raises():
    """A same-numel gate or sparse output in the wrong layout must raise, not compute garbage."""
    batch, heads, n_blocks, dim = 1, 4, 8, 64
    seq = n_blocks * BE
    out_c, out_s, weight = _make(batch, heads, n_blocks, dim, gated=True)
    bshd_gate = torch.rand(batch, seq, heads, dim, device=DEVICE, dtype=torch.bfloat16)
    for mode in ("no_grad", "grad_leaf"):
        with pytest.raises(ValueError, match="compress_attn_weight"):
            _run(mode, out_c, out_s, bshd_gate)
        with pytest.raises(ValueError, match="out_s"):
            _run(mode, out_c, out_s.view(batch, seq, heads, dim), weight)
        with pytest.raises(ValueError):
            _run(mode, out_c, out_s, weight[:, :, :, :1])  # broadcasting gates are not supported
        with pytest.raises(ValueError):
            _run(mode, out_c, out_s, weight, be=BE // 2)


@pytest.mark.parametrize("mode", MODES)
def test_dtype_promotion_is_never_downcast(mode):
    """A wider gate or coarse output promotes the result like the old path did."""
    out_c, out_s, weight = _make(1, 4, 8, 64, gated=True)
    ref = _reference_combine(out_c, out_s.clone(), weight.float(), BE)
    got, arg = _run(mode, out_c, out_s, weight.float())
    assert got.dtype == ref.dtype == torch.float32
    assert got.data_ptr() != arg.data_ptr()
    assert torch.equal(arg.detach(), out_s), "in-place path would have downcast into out_s"
    torch.testing.assert_close(got.detach(), ref, rtol=2.0**-7, atol=0.0)

    ref = _reference_combine(out_c.float(), out_s.clone(), None, BE)
    got, arg = _run(mode, out_c.float(), out_s, None)
    assert got.dtype == ref.dtype == torch.float32
    assert torch.equal(got.detach(), ref)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("gated", [False, True])
def test_bshd_layout_matches_old_bshd_combine(mode, gated):
    """seq_dim=1 (the 128/256 BSHD entry): bit-exact ungated, once-rounded gated, in place when allowed."""
    out_c, out_s, weight = _make(2, 4, 6, 64, gated=gated, seq_dim=1)
    ref = _reference_combine_bshd(out_c, out_s.clone(), weight, BE)
    got, arg = _run(mode, out_c, out_s, weight, seq_dim=1)
    assert got.shape == ref.shape == (2, 6 * BE, 4, 64)
    if gated:
        truth = _fp32_truth(out_c, out_s, weight, BE, seq_dim=1)
        assert bool(((got.detach().float() - truth).abs() <= BF16_HALF_ULP_REL * truth.abs()).all())
        assert torch.equal(got.detach(), _run("no_grad", out_c, out_s, weight, seq_dim=1)[0])
    else:
        assert torch.equal(got.detach(), ref)
    assert (got.data_ptr() == arg.data_ptr()) == (mode != "grad_leaf")


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------


def _peak_bytes(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base, out


@pytest.mark.parametrize("seq_dim", [2, 1])
@pytest.mark.parametrize("gated", [False, True])
def test_no_grad_combine_allocates_nothing(gated, seq_dim):
    """The in-place branch must not allocate at all, in either layout."""
    out_c, out_s, weight = _make(1, 8, 128, 128, gated=gated, seq_dim=seq_dim)
    with torch.no_grad():
        peak, out = _peak_bytes(lambda: _combine_coarse_sparse(out_c, out_s, weight, BE, seq_dim))
    assert peak == 0, f"no-grad combine allocated {peak} bytes"
    assert out.data_ptr() == out_s.data_ptr()


@pytest.mark.parametrize("seq_dim", [2, 1])
@pytest.mark.parametrize("gated", [False, True])
def test_grad_combine_allocates_exactly_one_output(gated, seq_dim):
    """The out-of-place branch allocates its output and nothing else.

    The old BHSD path peaked at three full-sequence tensors (the repeated
    coarse output, the product and the sum) and the old BSHD path at two;
    ``addcmul`` fuses the product away.
    """
    out_c, out_s, weight = _make(1, 8, 128, 128, gated=gated, seq_dim=seq_dim)
    out_s.requires_grad_(True)
    full = out_s.numel() * out_s.element_size()
    with torch.enable_grad():
        peak, out = _peak_bytes(lambda: _combine_coarse_sparse(out_c, out_s, weight, BE, seq_dim))
    assert peak == full, f"grad-mode combine allocated {peak} bytes, expected one full tensor ({full})"
    assert out.data_ptr() != out_s.data_ptr()


# --------------------------------------------------------------------------
# torch.compile
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["no_grad", "grad_leaf"])
def test_combine_compiles_fullgraph(mode):
    """The branch predicate must be traceable so a compiled model keeps one graph."""
    torch._dynamo.reset()
    compiled = torch.compile(_combine_coarse_sparse, fullgraph=True, dynamic=False)
    out_c, out_s, weight = _make(1, 4, 8, 64, gated=True)
    eager, _ = _run(mode, out_c, out_s, weight)
    out_s = out_s.clone()
    if mode == "grad_leaf":
        out_s.requires_grad_(True)
        with torch.enable_grad():
            got = compiled(out_c, out_s, weight, BE, 2)
    else:
        with torch.no_grad():
            got = compiled(out_c, out_s, weight, BE, 2)
    assert torch.equal(got.detach(), eager.detach())


# --------------------------------------------------------------------------
# End-to-end through video_sparse_attn
# --------------------------------------------------------------------------


def _reference_video_sparse_attn(q, k, v, variable_block_sizes, q_variable_block_sizes, topk,
                                 block_size, compress_attn_weight):
    """Pre-change ``video_sparse_attn`` body, dispatching the sparse branch like ``ops`` does.

    The kernels are imported from their modules rather than looked up on
    ``ops`` so the spies installed on ``ops`` see only the code under test.
    """
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_128, block_sparse_attn_256
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
    if block_elements in (128, 256):
        attention = block_sparse_attn_128 if block_elements == 128 else block_sparse_attn_256
    else:
        attention = block_sparse_attn
    out_s = attention(q, k, v, mask, variable_block_sizes)[0]
    if compress_attn_weight is not None:
        return out_c * compress_attn_weight + out_s, mask
    return out_c + out_s, mask


def _vsa_inputs(heads, n_blocks, dim, be, seed=0):
    torch.manual_seed(seed)
    batch = 1
    seq = n_blocks * be
    q, k, v = (torch.randn(batch, heads, seq, dim, device=DEVICE, dtype=torch.bfloat16) for _ in range(3))
    vbs = torch.full((n_blocks,), be, device=DEVICE, dtype=torch.int32)
    return q, k, v, vbs, seq


def _gate(layout, batch, heads, seq, dim, seed=1):
    torch.manual_seed(seed)
    if layout == "none":
        return None
    if layout == "bhsd":
        return torch.rand(batch, heads, seq, dim, device=DEVICE, dtype=torch.bfloat16)
    assert layout == "bshd_transposed"  # the model's native gate layout, handed over as a view
    gate = torch.rand(batch, seq, heads, dim, device=DEVICE, dtype=torch.bfloat16).transpose(1, 2)
    assert not gate.is_contiguous()
    return gate


def _kernel_name(block_elements):
    return {64: "block_sparse_attn", 128: "block_sparse_attn_128", 256: "block_sparse_attn_256"}[block_elements]


@pytest.mark.parametrize("gate_layout", ["none", "bhsd", "bshd_transposed"])
@pytest.mark.parametrize("block_elements", [64, 128, 256])
@pytest.mark.parametrize("heads,n_blocks,dim,ratio", [(4, 8, 64, 0.5), (8, 16, 128, 0.2)])
def test_video_sparse_attn_matches_reference(monkeypatch, gate_layout, block_elements, heads, n_blocks, dim,
                                             ratio):
    """Inference VSA at every tile size: same kernel, same top-k routing, same output as the old path."""
    q, k, v, vbs, seq = _vsa_inputs(heads, n_blocks, dim, block_elements)
    gate = _gate(gate_layout, 1, heads, seq, dim)
    topk = max(1, int(n_blocks * ratio))
    block_size = (block_elements // 16, 4, 4)

    # Spy on the kernel dispatch and capture the routing mask of the code under test.
    name = _kernel_name(block_elements)
    original = getattr(ops, name)
    calls = []
    monkeypatch.setattr(ops, name, lambda *a, **kw: calls.append(True) or original(*a, **kw))
    masks = []
    original_topk = ops.fused_topk_mask
    monkeypatch.setattr(ops, "fused_topk_mask", lambda *a, **kw: masks.append(original_topk(*a, **kw)) or masks[-1])

    with torch.no_grad():
        ref, ref_mask = _reference_video_sparse_attn(q, k, v, vbs, vbs, topk, block_size, gate)
        got = ops.video_sparse_attn(q, k, v, vbs, vbs, topk, block_size, compress_attn_weight=gate)

    assert len(calls) == 1, f"{name} was dispatched {len(calls)} times, expected once"
    assert len(masks) == 1 and torch.equal(masks[0], ref_mask), "top-k routing changed"
    assert got.shape == ref.shape == (1, heads, seq, dim)
    assert got.dtype == ref.dtype == torch.bfloat16
    if gate is None:
        assert torch.equal(got, ref), "ungated full path must be bit-exact"
    else:
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel <= 5e-3, f"relative L2 {rel:.3e} too large"


def test_video_sparse_attn_inference_allocates_no_combine_temporaries():
    """Peak memory of a 64-tile inference call must drop by at least two full-sequence tensors.

    Only the 64-tile Triton kernel keeps the combine's temporaries at the top of
    the peak: the 128/256 entries copy q/k/v into the kernel's layout first, so
    their peak is set inside the kernel on both the Triton and CuTe routes. The
    combine's own allocations on those paths are pinned by the combine-level
    tests above.
    """
    heads, n_blocks, dim, block_elements = 8, 16, 128, 64
    q, k, v, vbs, seq = _vsa_inputs(heads, n_blocks, dim, block_elements)
    gate = _gate("bhsd", 1, heads, seq, dim)
    block_size = (4, 4, 4)
    full = q.numel() * q.element_size()
    with torch.no_grad():
        # warm up Triton compilation outside the measurement
        ops.video_sparse_attn(q, k, v, vbs, vbs, 4, block_size, compress_attn_weight=gate)
        _reference_video_sparse_attn(q, k, v, vbs, vbs, 4, block_size, gate)
        old_peak, _ = _peak_bytes(lambda: _reference_video_sparse_attn(q, k, v, vbs, vbs, 4, block_size, gate))
        new_peak, _ = _peak_bytes(lambda: ops.video_sparse_attn(q, k, v, vbs, vbs, 4, block_size,
                                                                compress_attn_weight=gate))
    assert old_peak - new_peak >= 2 * full, (f"expected at least two fewer full tensors ({2 * full} bytes), "
                                             f"got old={old_peak} new={new_peak}")


@pytest.mark.parametrize("gated", [False, True])
def test_video_sparse_attn_training_matches_reference(gated):
    """Grad-enabled 64-tile VSA: output and input gradients match the old path."""
    heads, n_blocks, dim = 4, 8, 64
    q, k, v, vbs, seq = _vsa_inputs(heads, n_blocks, dim, 64)
    gate = _gate("bhsd" if gated else "none", 1, heads, seq, dim)
    leaves_ref = [t.clone().requires_grad_(True) for t in (q, k, v) + ((gate, ) if gated else ())]
    leaves_got = [t.clone().requires_grad_(True) for t in (q, k, v) + ((gate, ) if gated else ())]

    ref, _ = _reference_video_sparse_attn(*leaves_ref[:3], vbs, vbs, 4, (4, 4, 4), leaves_ref[3] if gated else None)
    got = ops.video_sparse_attn(*leaves_got[:3], vbs, vbs, 4, (4, 4, 4),
                                compress_attn_weight=leaves_got[3] if gated else None)
    torch.manual_seed(2)
    grad_out = torch.randn_like(ref)
    ref.backward(grad_out)
    got.backward(grad_out)
    if gated:
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel <= 5e-3
    else:
        assert torch.equal(got, ref)
    for name, a, b in zip(("q", "k", "v", "gate"), leaves_got, leaves_ref):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, rtol=2.0**-5, atol=2.0**-5, msg=lambda m: f"grad of {name}: {m}")


# --------------------------------------------------------------------------
# End-to-end through video_sparse_attn_bshd (128/256-token tiles, [B, S, H, D])
# --------------------------------------------------------------------------


def _reference_video_sparse_attn_bshd(q, k, v, variable_block_sizes, q_variable_block_sizes, topk, block_size,
                                      compress_attn_weight):
    """Pre-change ``video_sparse_attn_bshd`` body."""
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_128_bshd, block_sparse_attn_256_bshd
    from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_topk_mask
    block_elements = block_size[0] * block_size[1] * block_size[2]
    batch, q_seq_len, heads, dim = q.shape
    q_num_blocks = q_seq_len // block_elements
    kv_num_blocks = k.shape[1] // block_elements

    q_c = q.view(batch, q_num_blocks, block_elements, heads, dim)
    k_c = k.view(batch, kv_num_blocks, block_elements, heads, dim)
    v_c = v.view(batch, kv_num_blocks, block_elements, heads, dim)
    q_c = (q_c.float().sum(dim=2) / q_variable_block_sizes.view(1, -1, 1, 1)).to(q.dtype)
    k_c = (k_c.float().sum(dim=2) / variable_block_sizes.view(1, -1, 1, 1)).to(k.dtype)
    v_c = (v_c.float().sum(dim=2) / variable_block_sizes.view(1, -1, 1, 1)).to(v.dtype)
    q_ch, k_ch, v_ch = (t.permute(0, 2, 1, 3).contiguous() for t in (q_c, k_c, v_c))

    scores = torch.matmul(q_ch, k_ch.transpose(-2, -1)) / (dim**0.5)
    attn = torch.softmax(scores, dim=-1)
    out_c_blk = torch.matmul(attn, v_ch).permute(0, 2, 1, 3).contiguous()

    mask = fused_topk_mask(scores, topk)
    attention = block_sparse_attn_128_bshd if block_elements == 128 else block_sparse_attn_256_bshd
    out_s, _ = attention(q, k, v, mask, variable_block_sizes)
    return _reference_combine_bshd(out_c_blk, out_s, compress_attn_weight, block_elements), mask


def _vsa_inputs_bshd(heads, n_blocks, dim, be, seed=0):
    torch.manual_seed(seed)
    seq = n_blocks * be
    q, k, v = (torch.randn(1, seq, heads, dim, device=DEVICE, dtype=torch.bfloat16) for _ in range(3))
    vbs = torch.full((n_blocks,), be, device=DEVICE, dtype=torch.int32)
    return q, k, v, vbs, seq


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("block_elements", [128, 256])
@pytest.mark.parametrize("heads,n_blocks,dim,ratio", [(4, 8, 64, 0.5), (8, 16, 128, 0.2)])
def test_video_sparse_attn_bshd_matches_reference(monkeypatch, gated, block_elements, heads, n_blocks, dim, ratio):
    """Inference BSHD VSA: same routing mask and same output as the old path."""
    q, k, v, vbs, seq = _vsa_inputs_bshd(heads, n_blocks, dim, block_elements)
    torch.manual_seed(1)
    gate = torch.rand(1, seq, heads, dim, device=DEVICE, dtype=torch.bfloat16) if gated else None
    topk = max(1, int(n_blocks * ratio))
    block_size = (block_elements // 16, 4, 4)

    masks = []
    original_topk = ops.fused_topk_mask
    monkeypatch.setattr(ops, "fused_topk_mask", lambda *a, **kw: masks.append(original_topk(*a, **kw)) or masks[-1])
    with torch.no_grad():
        ref, ref_mask = _reference_video_sparse_attn_bshd(q, k, v, vbs, vbs, topk, block_size, gate)
        got = ops.video_sparse_attn_bshd(q, k, v, vbs, vbs, topk, block_size, compress_attn_weight=gate)

    assert len(masks) == 1 and torch.equal(masks[0], ref_mask), "top-k routing changed"
    assert got.shape == ref.shape == (1, seq, heads, dim) and got.dtype == torch.bfloat16
    if gated:
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel <= 5e-3, f"relative L2 {rel:.3e} too large"
    else:
        assert torch.equal(got, ref), "ungated BSHD path must be bit-exact"


@pytest.mark.parametrize("gated", [False, True])
def test_video_sparse_attn_bshd_training_matches_reference(gated):
    """Grad-enabled BSHD VSA (256 tiles): output and input gradients match the old path."""
    heads, n_blocks, dim, be = 4, 8, 64, 256
    q, k, v, vbs, seq = _vsa_inputs_bshd(heads, n_blocks, dim, be)
    torch.manual_seed(1)
    gate = torch.rand(1, seq, heads, dim, device=DEVICE, dtype=torch.bfloat16) if gated else None
    tensors = (q, k, v) + ((gate, ) if gated else ())
    leaves_ref = [t.clone().requires_grad_(True) for t in tensors]
    leaves_got = [t.clone().requires_grad_(True) for t in tensors]

    ref, _ = _reference_video_sparse_attn_bshd(*leaves_ref[:3], vbs, vbs, 4, (16, 4, 4),
                                               leaves_ref[3] if gated else None)
    got = ops.video_sparse_attn_bshd(*leaves_got[:3], vbs, vbs, 4, (16, 4, 4),
                                     compress_attn_weight=leaves_got[3] if gated else None)
    torch.manual_seed(2)
    grad_out = torch.randn_like(ref)
    ref.backward(grad_out)
    got.backward(grad_out)
    if gated:
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel <= 5e-3
    else:
        assert torch.equal(got, ref)
    for name, a, b in zip(("q", "k", "v", "gate"), leaves_got, leaves_ref):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, rtol=2.0**-5, atol=2.0**-5, msg=lambda m: f"grad of {name}: {m}")
