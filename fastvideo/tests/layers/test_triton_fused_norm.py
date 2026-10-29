# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the Triton-fused residual+norm+modulate inference path.

The fused path claims to replicate the eager rounding sequence exactly, with
the layer-norm reduction order as the only remaining difference. These tests
pin that claim: outputs must match eager to within a bf16 ulp everywhere, with
the overwhelming majority of elements bit-identical.
"""
import os
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

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
def test_env_kill_switch(env_overrides):
    import fastvideo.envs as envs
    env_overrides.enter_context(envs.FASTVIDEO_DISABLE_FUSED_NORM.override(True))
    _, fused = _make_modules(elementwise_affine=True)
    residual, x = _inputs(torch.bfloat16, seed=6)
    gate = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        out, _ = fused(residual, x, gate, None, None)
    assert out.dtype == torch.float32  # eager promotion, not the fused bf16


@cuda_only
def test_fused_norm_requires_explicit_opt_in(env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_ENABLE_FUSED_NORM.override(False))
    eager, fused = _make_modules(elementwise_affine=True)
    residual, x = _inputs(torch.bfloat16, seed=6)
    gate = torch.randn(1, 1, H, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        expected, expected_residual = eager(residual, x, gate, None, None)
        actual, actual_residual = fused(residual, x, gate, None, None)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
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


class _Tanh(torch.nn.Module):
    """Sub-layer stand-in: the Kandinsky tests only exercise the norm wiring."""

    def forward(self, x, **kwargs):
        return torch.tanh(x)


@cuda_only
@pytest.mark.parametrize("decoder", [False, True], ids=["encoder", "decoder"])
def test_kandinsky5_block_matches_unfused_forward(distributed_setup, decoder):
    """Every post-residual norm of a Kandinsky-5 block takes the fused path, and the
    block output matches main's unfused forward (bf16-rounded residual, then norm)."""
    import fastvideo.layers.triton_fused_norm as tfn
    from fastvideo.models.dits import kandinsky5 as k5
    from fastvideo.platforms import AttentionBackendEnum

    D, T = 512, 128
    torch.manual_seed(0)
    cls = k5.Kandinsky5TransformerDecoderBlock if decoder else k5.Kandinsky5TransformerEncoderBlock
    block = cls(D, T, 4 * D, 64, supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ))
    block.self_attention, block.feed_forward = _Tanh(), _Tanh()
    if decoder:
        block.cross_attention = _Tanh()
    mod = block.visual_modulation if decoder else block.text_modulation
    torch.nn.init.normal_(mod.out_layer.weight, std=0.1)  # zero-initialised by default
    block = block.cuda().to(torch.bfloat16)
    x = torch.randn(2, 256, D, device="cuda", dtype=torch.bfloat16)
    t = torch.randn(2, T, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        params = [torch.chunk(p, 3, dim=-1) for p in torch.chunk(mod(t).unsqueeze(1), 3 if decoder else 2, dim=-1)]
        shift, scale, gate = params[0]
        out = block.self_attention_norm(x.float(), shift=shift, scale=scale, convert_modulation_dtype=True).type_as(x)
        ref = x
        for shift, scale, next_gate in params[1:]:
            ref = (ref.float() + gate.float() * torch.tanh(out).float()).type_as(x)
            out = (F.layer_norm(ref.float(), (D, ), eps=1e-5) * (1 + scale) + shift).type_as(x)
            gate = next_gate
        ref = (ref.float() + gate.float() * torch.tanh(out).float()).type_as(x)

        with patch.object(tfn, "fused_residual_norm_mod", wraps=tfn.fused_residual_norm_mod) as spy:
            actual = block(x, x, t, None, None) if decoder else block(x, t, None)

    assert spy.call_count == len(params) - 1
    torch.testing.assert_close(actual, ref, rtol=2e-2, atol=2e-2)
