# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the Triton-fused residual+norm+modulate inference path.

The fused path claims to replicate the eager rounding sequence exactly, with
the layer-norm reduction order as the only remaining difference. These tests
pin that claim: outputs must match eager to within a bf16 ulp everywhere, with
the overwhelming majority of elements bit-identical.
"""
import os

import pytest
import torch

from fastvideo.layers.layernorm import ScaleResidualLayerNormScaleShift
import fastvideo.envs as envs

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

B, S, H = 1, 9360, 1536  # the Wan2.1-T2V-1.3B sp2 shapes


@pytest.fixture(autouse=True)
def opt_into_fused_norm(env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_ENABLE_FUSED_NORM.override(True))


def _make_modules(elementwise_affine: bool, seed: int = 0):
    """One eager module and one fused module with identical weights."""
    torch.manual_seed(seed)
    kwargs = dict(
        norm_type="layer",
        eps=1e-6,
        elementwise_affine=elementwise_affine,
        dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    eager = ScaleResidualLayerNormScaleShift(H, **kwargs).cuda()
    fused = ScaleResidualLayerNormScaleShift(H, fuse_inference=True, **kwargs).cuda()
    if elementwise_affine:
        with torch.no_grad():
            w = torch.randn(H) * 0.02 + 1.0
            bias = torch.randn(H) * 0.01
            for m in (eager, fused):
                m.norm.weight.copy_(w)
                m.norm.bias.copy_(bias)
    return eager, fused


def _inputs(stream_dtype: torch.dtype, seed: int = 1):
    torch.manual_seed(seed)
    residual = (torch.randn(B, S, H, device="cuda") * 3).to(stream_dtype)
    x = torch.randn(B, S, H, device="cuda").to(stream_dtype)
    return residual, x


def _assert_close_to_eager(fused_out, eager_out, stream_dtype, shift=None):
    """Fused must match eager to within the provable error bound.

    The fused kernel differs from eager only in reduction order, rsqrt, and FMA
    contraction -- all fp32-ulp effects on the *norm output*. After bf16
    rounding that is at most one bf16 step in xhat (bf16 has a 7-bit
    mantissa, so one step is 2^-7 of the local binade), which modulation then
    amplifies by |1 + scale|. Since xhat * (1 + scale) == y - shift, the bound
    on the final output is

        |f - e| <= 2^-7 * (|e - shift| + |e|) + 1e-6

    (one amplified xhat step plus one final-rounding step; the 1e-6 floor
    covers near-cancellation elements whose magnitude is below bf16
    resolution). Any structural defect -- wrong eps, missing bias, bad mean --
    violates this by orders of magnitude.
    """
    f = fused_out.float()
    e = eager_out.float()
    assert not torch.isnan(f).any()
    if stream_dtype == torch.bfloat16:
        sh = 0.0 if shift is None else shift.float()
        tol = 2**-7 * ((e - sh).abs() + e.abs()) + 1e-6
        bad = (f - e).abs() > tol
        assert not bad.any(), (f"{bad.sum().item()} elements exceed the ulp bound; "
                               f"max abs diff {(f - e).abs().max().item():.3e}")
        bit_equal = (fused_out.view(torch.int16) == eager_out.view(torch.int16)).float().mean().item()
        assert bit_equal > 0.999, f"only {bit_equal:.4%} of elements bit-identical"
    else:
        torch.testing.assert_close(f, e, rtol=1e-5, atol=1e-5)


@cuda_only
@pytest.mark.parametrize("stream_dtype", [torch.bfloat16, torch.float32])
def test_site2_gate_vector_norm_only(stream_dtype):
    """Wan post-self-attn: fp32 tensor gate, affine norm, no modulation."""
    eager, fused = _make_modules(elementwise_affine=True)
    residual, x = _inputs(stream_dtype)
    gate = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        e_mod, e_res = eager(residual, x, gate, None, None)
        f_mod, f_res = fused(residual, x, gate, None, None)

    # fused returns the stream dtype; eager promotes to fp32 -- compare after
    # the caller-side cast eager relies on.
    assert f_mod.dtype == stream_dtype and f_res.dtype == stream_dtype
    _assert_close_to_eager(f_mod, e_mod.to(stream_dtype), stream_dtype)
    _assert_close_to_eager(f_res, e_res.to(stream_dtype), stream_dtype)


@cuda_only
@pytest.mark.parametrize("stream_dtype", [torch.bfloat16, torch.float32])
def test_site3_scalar_gate_with_modulation(stream_dtype):
    """Wan post-cross-attn: gate==1, no-affine norm, scale/shift modulation."""
    eager, fused = _make_modules(elementwise_affine=False)
    residual, x = _inputs(stream_dtype, seed=2)
    shift = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)
    scale = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        e_mod, e_res = eager(residual, x, 1, shift, scale)
        f_mod, f_res = fused(residual, x, 1, shift, scale)

    assert f_mod.dtype == stream_dtype and f_res.dtype == stream_dtype
    _assert_close_to_eager(f_mod, e_mod.to(stream_dtype), stream_dtype, shift=shift)
    _assert_close_to_eager(f_res, e_res.to(stream_dtype), stream_dtype)


@cuda_only
def test_per_token_modulation():
    """ti2v-style per-token scale/shift ([B, S, H]) goes down the fused path."""
    eager, fused = _make_modules(elementwise_affine=False)
    residual, x = _inputs(torch.bfloat16, seed=3)
    shift = torch.randn(B, S, H, device="cuda", dtype=torch.float32)
    scale = torch.randn(B, S, H, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        e_mod, e_res = eager(residual, x, 1, shift, scale)
        f_mod, f_res = fused(residual, x, 1, shift, scale)

    _assert_close_to_eager(f_mod, e_mod.to(torch.bfloat16), torch.bfloat16, shift=shift)
    _assert_close_to_eager(f_res, e_res.to(torch.bfloat16), torch.bfloat16)


@cuda_only
def test_residual_out_feeds_next_site_bit_exactly():
    """Site-3 chaining: the bf16 residual the fused kernel emits must be
    bit-identical to eager's, because it feeds the next block's sum."""
    eager, fused = _make_modules(elementwise_affine=False)
    residual, x = _inputs(torch.bfloat16, seed=4)

    with torch.no_grad():
        _, e_res = eager(residual, x, 1, None, None)
        _, f_res = fused(residual, x, 1, None, None)

    # eager: bf16 + bf16 -> bf16 tensor; fused: fp32 sum rounded to bf16.
    # PyTorch's eager bf16 add also computes in fp32, so these must agree
    # bitwise -- no reduction is involved in the residual sum.
    assert torch.equal(e_res, f_res)


@cuda_only
def test_grad_enabled_falls_back_to_eager():
    """Training path must not touch the fused kernel (it has no backward)."""
    _, fused = _make_modules(elementwise_affine=True)
    residual, x = _inputs(torch.bfloat16, seed=5)
    gate = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)

    out, res = fused(residual, x, gate, None, None)
    # eager promotion produces fp32 outputs; the fused kernel would have
    # produced bf16. dtype is the witness for which path ran.
    assert out.dtype == torch.float32
    assert res.dtype == torch.float32


@cuda_only
def test_default_tier_is_exact(env_overrides):
    """Without FASTVIDEO_ENABLE_FUSED_NORM the module runs the bit-exact kernels
    (fastvideo/layers/triton_exact_modulation.py): eager values after the
    caller-side cast, and the dtype witness fp32 normalized / bf16 residual that
    is neither eager (fp32/fp32) nor the fused-norm kernel (bf16/bf16)."""
    env_overrides.enter_context(envs.FASTVIDEO_ENABLE_FUSED_NORM.override(False))
    eager, fused = _make_modules(elementwise_affine=True)
    residual, x = _inputs(torch.bfloat16, seed=6)
    gate = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        expected, expected_residual = eager(residual, x, gate, None, None)
        actual, actual_residual = fused(residual, x, gate, None, None)
    assert actual.dtype == torch.float32 and actual_residual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
    assert torch.equal(actual_residual, expected_residual.to(torch.bfloat16))

    eager, fused = _make_modules(elementwise_affine=False)
    residual, x = _inputs(torch.bfloat16, seed=7)
    shift = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)
    scale = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        expected, expected_residual = eager(residual, x, 1, shift, scale)
        actual, actual_residual = fused(residual, x, 1, shift, scale)
    assert actual.dtype == torch.bfloat16 and actual_residual.dtype == torch.bfloat16
    assert torch.equal(actual, expected.to(torch.bfloat16))
    assert torch.equal(actual_residual, expected_residual)


@cuda_only
def test_mixed_none_still_raises():
    _, fused = _make_modules(elementwise_affine=True)
    residual, x = _inputs(torch.bfloat16, seed=7)
    with pytest.raises(ValueError):
        with torch.no_grad():
            fused(residual, x, 1, None, torch.randn(1, 1, H, device="cuda"))


@cuda_only
def test_default_module_never_fuses():
    """Existing users (fuse_inference unset) keep eager dtype semantics."""
    eager, _ = _make_modules(elementwise_affine=True)
    residual, x = _inputs(torch.bfloat16, seed=8)
    gate = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        out, res = eager(residual, x, gate, None, None)
    assert out.dtype == torch.float32
    assert res.dtype == torch.float32


# ---------------------------------------------------------------------------
# Stage 2: the residual-free norm1 + modulate site and per-frame tables.
# ---------------------------------------------------------------------------


@cuda_only
@pytest.mark.parametrize("stream_dtype", [torch.bfloat16, torch.float32])
def test_site1_norm_modulate_fused(stream_dtype):
    """Wan pre-self-attn: ``(norm1(x.float()) * (1 + scale) + shift).to(orig)`` in one launch."""
    from fastvideo.layers.layernorm import FP32LayerNorm, norm_modulate
    from fastvideo.layers.triton_fused_norm import fused_norm_mod, fused_norm_mod_supported
    norm1 = FP32LayerNorm(H, 1e-6, elementwise_affine=False).cuda()
    x, _ = _inputs(stream_dtype, seed=20)
    shift = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)
    scale = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        eager = (norm1(x.float()) * (1 + scale) + shift).to(stream_dtype)
        assert fused_norm_mod_supported(x, shift, scale, norm1)
        fused_out = fused_norm_mod(x, shift, scale, norm1, ln_input_fp32=True)
        dispatched = norm_modulate(x, norm1, shift, scale, out_dtype=stream_dtype, norm_input_fp32=True)
    assert fused_out.dtype == stream_dtype
    _assert_close_to_eager(fused_out, eager, stream_dtype, shift=shift)
    assert torch.equal(dispatched, fused_out)


@cuda_only
def test_per_frame_tables_fused():
    """[B, F, 1, H] tables (causal Wan) broadcast per frame inside the fused kernel."""
    eager, fused = _make_modules(elementwise_affine=False)
    frames, frame_seqlen = 3, 1560
    torch.manual_seed(21)
    residual = (torch.randn(1, frames * frame_seqlen, H, device="cuda") * 3).to(torch.bfloat16)
    x = torch.randn(1, frames * frame_seqlen, H, device="cuda").to(torch.bfloat16)
    shift = torch.randn(1, frames, 1, H, device="cuda", dtype=torch.float32)
    scale = torch.randn(1, frames, 1, H, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        e_mod, e_res = eager(residual, x, 1, shift, scale)
        f_mod, f_res = fused(residual, x, 1, shift, scale)
    assert f_mod.dtype == torch.bfloat16
    expanded_shift = shift.expand(1, frames, frame_seqlen, H).flatten(1, 2)
    _assert_close_to_eager(f_mod, e_mod.to(torch.bfloat16), torch.bfloat16, shift=expanded_shift)
    assert torch.equal(f_res, e_res)


@cuda_only
def test_causal_bf16_gate_fused():
    """bf16 per-frame gate (causal Wan): the product and the sum round as eager's bf16 chain does."""
    eager, fused = _make_modules(elementwise_affine=True)
    frames, frame_seqlen = 3, 400
    torch.manual_seed(22)
    residual = (torch.randn(1, frames * frame_seqlen, H, device="cuda") * 3).to(torch.bfloat16)
    x = torch.randn(1, frames * frame_seqlen, H, device="cuda").to(torch.bfloat16)
    gate = (torch.randn(1, frames, 1, H, device="cuda") * 0.5).to(torch.bfloat16)
    with torch.no_grad():
        e_norm, e_res = eager(residual, x, gate, None, None)
        f_norm, f_res = fused(residual, x, gate, None, None)
    assert f_norm.dtype == torch.bfloat16 and f_res.dtype == torch.bfloat16
    assert torch.equal(f_res, e_res)
    _assert_close_to_eager(f_norm, e_norm, torch.bfloat16)


@cuda_only
def test_causal_bf16_tables_modulation_fused():
    """bf16 per-frame shift/scale after a bf16 residual sum: 1 + scale and the product round as in eager."""
    eager, fused = _make_modules(elementwise_affine=False)
    frames, frame_seqlen = 3, 400
    torch.manual_seed(23)
    residual = (torch.randn(1, frames * frame_seqlen, H, device="cuda") * 3).to(torch.bfloat16)
    x = torch.randn(1, frames * frame_seqlen, H, device="cuda").to(torch.bfloat16)
    shift = (torch.randn(1, frames, 1, H, device="cuda") * 0.5).to(torch.bfloat16)
    scale = (torch.randn(1, frames, 1, H, device="cuda") * 0.5).to(torch.bfloat16)
    with torch.no_grad():
        e_mod, e_res = eager(residual, x, 1, shift, scale)
        f_mod, f_res = fused(residual, x, 1, shift, scale)
    assert e_mod.dtype == torch.bfloat16  # the eager chain is bf16 throughout
    assert torch.equal(f_res, e_res)
    expanded_shift = shift.expand(1, frames, frame_seqlen, H).flatten(1, 2)
    _assert_close_to_eager(f_mod, e_mod, torch.bfloat16, shift=expanded_shift)


@cuda_only
@pytest.mark.parametrize("autocast", [False, True], ids=["eager-bf16-ln", "autocast-fp32-ln"])
def test_plain_layernorm_causal_regimes_fused(autocast):
    """Causal Wan's plain nn.LayerNorm in bf16: eager returns bf16 outside autocast and fp32 inside it."""
    torch.manual_seed(24)
    kwargs = dict(norm_type="layer", eps=1e-6, elementwise_affine=True, dtype=torch.float32)
    eager = ScaleResidualLayerNormScaleShift(H, **kwargs).cuda()
    fused = ScaleResidualLayerNormScaleShift(H, fuse_inference=True, **kwargs).cuda()
    with torch.no_grad():
        w = torch.randn(H) * 0.02 + 1.0
        bias = torch.randn(H) * 0.01
        for m in (eager, fused):
            m.norm.weight.copy_(w)
            m.norm.bias.copy_(bias)
    eager, fused = eager.to(torch.bfloat16), fused.to(torch.bfloat16)
    frames, frame_seqlen = 3, 400
    residual = (torch.randn(1, frames * frame_seqlen, H, device="cuda") * 3).to(torch.bfloat16)
    x = torch.randn(1, frames * frame_seqlen, H, device="cuda").to(torch.bfloat16)
    shift = (torch.randn(1, frames, 1, H, device="cuda") * 0.5).to(torch.bfloat16)
    scale = (torch.randn(1, frames, 1, H, device="cuda") * 0.5).to(torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        e_mod, e_res = eager(residual, x, 1, shift, scale)
        f_mod, f_res = fused(residual, x, 1, shift, scale)
    assert e_mod.dtype == (torch.float32 if autocast else torch.bfloat16)
    assert torch.equal(f_res, e_res)
    expanded_shift = shift.expand(1, frames, frame_seqlen, H).flatten(1, 2)
    _assert_close_to_eager(f_mod, e_mod.to(torch.bfloat16), torch.bfloat16, shift=expanded_shift)


@cuda_only
def test_fused_norm_opt_out_keeps_exact_tier():
    """A module built with fused_norm=False (the causal block) ignores the opt-in and stays bit-exact."""
    kwargs = dict(norm_type="layer", eps=1e-6, elementwise_affine=True, dtype=torch.float32,
                  compute_dtype=torch.float32)
    torch.manual_seed(25)
    eager = ScaleResidualLayerNormScaleShift(H, **kwargs).cuda()
    opted_out = ScaleResidualLayerNormScaleShift(H, fuse_inference=True, fused_norm=False, **kwargs).cuda()
    with torch.no_grad():
        for m in (eager, opted_out):
            m.norm.weight.copy_(torch.linspace(0.9, 1.1, H))
            m.norm.bias.copy_(torch.linspace(-0.01, 0.01, H))
    residual, x = _inputs(torch.bfloat16, seed=25)
    gate = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        expected, expected_residual = eager(residual, x, gate, None, None)
        actual, actual_residual = opted_out(residual, x, gate, None, None)
    # The exact tier's dtype witness (fp32 normalized, bf16 residual), not the fused kernel's bf16/bf16.
    assert actual.dtype == torch.float32 and actual_residual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
    assert torch.equal(actual_residual, expected_residual.to(torch.bfloat16))
