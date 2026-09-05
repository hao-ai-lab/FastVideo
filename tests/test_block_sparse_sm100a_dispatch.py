# SPDX-License-Identifier: Apache-2.0
"""Routing tests for the opt-in sm_100a/sm_103a dispatch in ``block_sparse_attn_from_indices``.

``FASTVIDEO_VSA_SM100A=1`` routes the forward to the sm_100a extension when
``block_sparse_attn_sm100a.is_supported`` passes. Its backward is the sm_100a
CUDA backward where that op is built and ``block_sparse_attn_bwd_sm100a.is_supported``
passes (64-token blocks on an sm_100a device), the Triton backward otherwise; the
sm_100a lse is already in Triton's M format, so either pairing needs no conversion.
Everything else -- env unset, unsupported input, ``FASTVIDEO_VSA_TRITON`` override --
must keep the pre-existing selection, bit-for-bit.

Run with: python -m pytest tests/test_block_sparse_sm100a_dispatch.py -v
"""

import importlib

import pytest
import torch

from fastvideo_kernel import block_sparse_attn_sm100a as vsa
from fastvideo_kernel.block_sparse_attn import block_sparse_attn_from_indices

HEAD_DIM = 128
ENV = "FASTVIDEO_VSA_SM100A"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}
    or not vsa._HAS_VSA_SM100A,
    reason="requires data-center Blackwell (sm_100a/sm_103a) and a built fastvideo_kernel extension",
)


def make_case(block, num_blocks=8, topk=4, heads=4, batch=1, seed=0, requires_grad=False):
    torch.manual_seed(seed)
    S = num_blocks * block
    shape = (batch, heads, S, HEAD_DIM) if vsa.BHSD else (batch, S, heads, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16,
                           requires_grad=requires_grad) for _ in range(3))

    # from_indices takes Triton-shaped metadata: [B, H, Nq, Mk] / [B, H, Nq].
    # The sm_100a extension reads both flat, so the same tensors serve both paths.
    idx = torch.empty((batch * heads * num_blocks, topk), dtype=torch.int32, device="cuda")
    for r in range(idx.shape[0]):
        idx[r] = torch.randperm(num_blocks, device="cuda")[:topk].to(torch.int32).sort().values
    idx = idx.view(batch, heads, num_blocks, topk)
    num = torch.full((batch, heads, num_blocks), topk, dtype=torch.int32, device="cuda")
    vbs = torch.full((num_blocks, ), block, dtype=torch.int32, device="cuda")
    return q, k, v, idx, num, vbs


def triton_reference(q, k, v, idx, num, vbs, monkeypatch):
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON", "1")
    out = block_sparse_attn_from_indices(q, k, v, idx, num, vbs)
    monkeypatch.delenv("FASTVIDEO_VSA_TRITON")
    return out


@pytest.mark.parametrize("block", [64, 128])
def test_env_routes_to_sm100a(block, monkeypatch):
    """With the env set on supported input, from_indices runs the sm_100a op:
    bitwise-equal to the wrapper called directly, allclose to Triton."""
    q, k, v, idx, num, vbs = make_case(block)
    want_o, want_m = vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)

    monkeypatch.setenv(ENV, "1")
    got_o, got_m = block_sparse_attn_from_indices(q, k, v, idx, num, vbs)
    assert torch.equal(got_o, want_o) and torch.equal(got_m, want_m)

    if block == 64:  # Triton's forward is hardcoded to 64-token blocks
        ref_o, ref_m = triton_reference(q, k, v, idx, num, vbs, monkeypatch)
        assert torch.allclose(got_o.float(), ref_o.float(), atol=2e-2, rtol=2e-2)
        assert torch.allclose(got_m, ref_m, atol=1e-3, rtol=1e-3)


def test_env_unset_keeps_default_backend(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    q, k, v, idx, num, vbs = make_case(64)
    got = block_sparse_attn_from_indices(q, k, v, idx, num, vbs)
    ref = triton_reference(q, k, v, idx, num, vbs, monkeypatch)
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_unsupported_input_falls_back_to_triton(monkeypatch):
    """Env set but is_supported False (odd block count): silent Triton fallback."""
    monkeypatch.setenv(ENV, "1")
    q, k, v, idx, num, vbs = make_case(64, num_blocks=7, topk=3)
    assert not vsa.is_supported(q, vbs)
    got = block_sparse_attn_from_indices(q, k, v, idx, num, vbs)
    ref = triton_reference(q, k, v, idx, num, vbs, monkeypatch)
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_unsupported_blk128_raises_not_implemented(monkeypatch):
    """128-token metadata cannot fall back to Triton's 64-token kernels."""
    monkeypatch.setenv(ENV, "1")
    q, k, v, idx, num, vbs = make_case(128, num_blocks=7, topk=3)
    assert not vsa.is_supported(q, vbs)
    with pytest.raises(NotImplementedError, match="128-token"):
        block_sparse_attn_from_indices(q, k, v, idx, num, vbs)


def test_force_triton_overrides_sm100a(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON", "1")
    q, k, v, idx, num, vbs = make_case(64)
    got = block_sparse_attn_from_indices(q, k, v, idx, num, vbs)
    monkeypatch.delenv("FASTVIDEO_VSA_TRITON")
    sm100a_o, _ = vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)
    ref = triton_reference(q, k, v, idx, num, vbs, monkeypatch)
    assert torch.equal(got[0], ref[0])
    assert not torch.equal(got[0], sm100a_o)


def _grads_sm100a_route_vs_triton(monkeypatch, vbs=None):
    """Grads of (out**2).sum() through the sm_100a route vs the all-Triton route.

    On a device where the sm_100a backward is built and supported, the route must not enter
    the Triton backward at all (it is monkeypatched to fail); elsewhere the route pairs the
    sm_100a forward with the Triton backward, as before.
    """
    # The package exports a FUNCTION named block_sparse_attn; the module needs importlib.
    dispatch = importlib.import_module("fastvideo_kernel.block_sparse_attn")
    from fastvideo_kernel import block_sparse_attn_bwd_sm100a as vsa_bwd

    q, k, v, idx, num, default_vbs = make_case(64, requires_grad=True)
    vbs = default_vbs if vbs is None else vbs

    q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    monkeypatch.setenv("FASTVIDEO_VSA_TRITON", "1")
    out2, _ = block_sparse_attn_from_indices(q2, k2, v2, idx, num, vbs)
    out2.float().square().sum().backward()
    ref = [t.grad.float() for t in (q2, k2, v2)]
    monkeypatch.delenv("FASTVIDEO_VSA_TRITON")

    monkeypatch.setenv(ENV, "1")
    sm100a_backward = vsa_bwd.is_supported(q, vbs)
    if sm100a_backward:

        def no_triton_backward(*args, **kwargs):
            raise AssertionError("Triton backward entered on a supported sm_100a input")

        monkeypatch.setattr(dispatch, "block_sparse_attn_backward_triton", no_triton_backward)
    out, _ = block_sparse_attn_from_indices(q, k, v, idx, num, vbs)
    out.float().square().sum().backward()
    got = [t.grad.float() for t in (q, k, v)]
    return got, ref, sm100a_backward


def _assert_grads_close(got, ref):
    # Two bf16 kernels against each other (not an fp32 reference): twice the tolerances the
    # sm_100a backward holds against fp32 in tests/test_block_sparse_bwd_sm100a.py.
    for g, r, name in zip(got, ref, "qkv"):
        diff = (g - r).abs()
        rel_max = diff.max().item() / max(r.abs().max().item(), 1e-6)
        mean_abs = diff.mean().item()
        assert rel_max <= 2e-2 and mean_abs <= 2e-3, \
            f"d{name}: max|diff|/max|ref|={rel_max:.3e} mean|diff|={mean_abs:.3e}"


def test_backward_matches_triton(monkeypatch):
    """sm_100a route (CUDA backward where built, Triton backward otherwise) vs all-Triton."""
    got, ref, _ = _grads_sm100a_route_vs_triton(monkeypatch)
    _assert_grads_close(got, ref)


def test_backward_ragged_block_sizes_matches_triton(monkeypatch):
    """variable_block_sizes below 64 (padded kv rows) through the same two routes."""
    torch.manual_seed(1)
    vbs = torch.randint(32, 65, (8, ), dtype=torch.int32, device="cuda")
    got, ref, _ = _grads_sm100a_route_vs_triton(monkeypatch, vbs=vbs)
    _assert_grads_close(got, ref)


def test_backward_uses_sm100a_kernel_when_built(monkeypatch):
    """Guards against a silent Triton fallback: with the op built on an sm_100a device the
    CUDA backward must be the one that runs."""
    from fastvideo_kernel import block_sparse_attn_bwd_sm100a as vsa_bwd
    if not vsa_bwd._HAS_VSA_BWD_SM100A or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("sm_100a backward not built for this device")
    _, _, sm100a_backward = _grads_sm100a_route_vs_triton(monkeypatch)
    assert sm100a_backward


def test_blk128_backward_raises(monkeypatch):
    """128-token blocks: forward runs, backward refuses (both backwards are 64-block only)."""
    monkeypatch.setenv(ENV, "1")
    q, k, v, idx, num, vbs = make_case(128, requires_grad=True)
    out, _ = block_sparse_attn_from_indices(q, k, v, idx, num, vbs)
    with pytest.raises(RuntimeError, match="64-token"):
        out.float().sum().backward()
