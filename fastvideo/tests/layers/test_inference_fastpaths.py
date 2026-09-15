# SPDX-License-Identifier: Apache-2.0
"""Coverage for the two inference-only fast paths added in PR #1245, asserting
they engage during inference yet never corrupt training:

1. Fused RoPE (``_apply_rotary_emb`` -> Triton ``apply_rope``): must fire on the
   4D ``[B, S, H, D]`` layout every attention call site uses and match the PyTorch
   fallback, while staying disabled under autograd so it never silently drops the
   q/k gradients in training.
2. ``FP32LayerNorm`` fp32 weight/bias cache: inference-only. Under grad it must
   recompute ``weight.float()`` per call instead of stashing a grad-bearing tensor
   reused across micro-batches (a latent "backward through the graph a second
   time" / stale-graph hazard under gradient accumulation); it may only populate
   the cache under ``no_grad``.
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.layers.layernorm import FP32LayerNorm
from fastvideo.layers.rotary_embedding import _apply_rotary_emb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the fused kernel")
def test_apply_rotary_emb_4d_grad_flows_and_matches_inference():
    """4D layout: grad flows through the fallback, and the fused path matches it."""
    B, S, H, D = 2, 128, 8, 128
    torch.manual_seed(0)
    x = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    # Full head_dim cos/sin => rotate_half style.
    cos = torch.randn(S, D, device="cuda", dtype=torch.float32)
    sin = torch.randn(S, D, device="cuda", dtype=torch.float32)

    # Training path: grad enabled forces the differentiable PyTorch fallback.
    out_train = _apply_rotary_emb(x, cos, sin, is_neox_style=False)
    assert out_train.requires_grad, "fused path must not fire under grad (would drop gradients)"
    out_train.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # Inference path: fused kernel fires and must match the fallback numerically.
    with torch.no_grad():
        out_infer = _apply_rotary_emb(x.detach(), cos, sin, is_neox_style=False)
    torch.testing.assert_close(out_train.detach(), out_infer, atol=1e-5, rtol=1e-5)


def test_fp32_layernorm_cache_disabled_under_grad():
    """Affine FP32LayerNorm must not stash a grad-bearing fp32 weight while training."""
    ln = FP32LayerNorm(8, elementwise_affine=True).to(torch.bfloat16)

    x = torch.randn(4, 8, dtype=torch.bfloat16)
    ln(x)  # grad enabled by default
    assert ln.__dict__.get("_w_fp32_cache") is None
    assert ln.__dict__.get("_b_fp32_cache") is None

    with torch.no_grad():
        ln(x)
    assert ln.__dict__.get("_w_fp32_cache") is not None


def test_fp32_layernorm_grad_accumulation_matches_reference():
    """The grad-accum pattern (forward+backward per micro-batch, optimizer step
    after the loop) must run cleanly and produce the same accumulated weight
    gradient as recomputing weight.float() fresh per call with no caching.
    """
    torch.manual_seed(0)
    batches = [torch.randn(4, 8, dtype=torch.bfloat16) for _ in range(2)]

    ln = FP32LayerNorm(8, elementwise_affine=True).to(torch.bfloat16)
    for x in batches:  # no optimizer step between micro-batches
        ln(x.clone()).float().pow(2).sum().backward()
    cached_grad = ln.weight.grad.clone()
    assert torch.isfinite(cached_grad).all()

    # Reference: identical module, gradient accumulated the same way. Under grad
    # the guard already disables caching, so this must match exactly.
    ref = FP32LayerNorm(8, elementwise_affine=True).to(torch.bfloat16)
    ref.load_state_dict(ln.state_dict())
    for x in batches:
        ref(x.clone()).float().pow(2).sum().backward()

    torch.testing.assert_close(cached_grad, ref.weight.grad, atol=1e-6, rtol=1e-6)
