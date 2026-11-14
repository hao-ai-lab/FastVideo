# SPDX-License-Identifier: Apache-2.0
"""Wan's default inference path must equal eager over a whole denoising run, bit for bit.

The default tier (fastvideo/layers/triton_exact_modulation.py) replaces the
elementwise chains around every FP32LayerNorm in the Wan block. This test runs
the three-step UniPC trajectory of the Wan denoising golden gate twice in one
process, once with those kernels and once with the eager expressions, and
compares every latent with ``torch.equal``. No stored reference is involved, so
it holds on any CUDA device and attention backend.
"""

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.layers import triton_exact_modulation as exact
from fastvideo.tests.golden_gate._harness import distributed_runtime
from fastvideo.tests.golden_gate._tensor_golden import deterministic_forward
from fastvideo.tests.golden_gate.test_wan_denoising import denoising_outputs

__all__ = ["distributed_runtime"]


def test_wan_denoising_exact_tier_equals_eager(distributed_runtime, env_overrides, monkeypatch):
    if not exact.HAVE_TRITON:
        pytest.skip("the exact tier requires Triton")
    env_overrides.enter_context(envs.FASTVIDEO_CFG_GATE_STEP.override(1.0))
    env_overrides.enter_context(envs.FASTVIDEO_ENABLE_FUSED_NORM.override(False))

    kernel_calls = []
    real_gate_residual = exact.gate_residual

    def counting_gate_residual(*args, **kwargs):
        kernel_calls.append(1)
        return real_gate_residual(*args, **kwargs)

    with deterministic_forward() as device:
        with monkeypatch.context() as spy:
            spy.setattr(exact, "gate_residual", counting_gate_residual)
            actual, _ = denoising_outputs(device)
        with monkeypatch.context() as eager_only:
            eager_only.setattr(exact, "HAVE_TRITON", False)
            expected, _ = denoising_outputs(device)

    assert kernel_calls, "the exact kernels did not run; the comparison would be eager against eager"
    assert actual.keys() == expected.keys()
    for key in sorted(actual):
        a, b = actual[key], expected[key]
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert torch.equal(a, b), f"{key}: max abs diff {(a.float() - b.float()).abs().max().item():.3e}"
