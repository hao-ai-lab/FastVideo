# SPDX-License-Identifier: Apache-2.0
"""MXFP8 GEMMs must accept flashinfer's ``uint8`` E8M0 scale buffers.

``torch.nn.functional.scaled_mm`` requires ``torch.float8_e8m0fnu`` scales and
rejects the ``uint8`` byte buffers that ``flashinfer.mxfp8_quantize`` returns
with a generic "Invalid scaling configuration" ValueError. FastVideo's own
Triton quantizers already return the e8m0 dtype, so the two paths disagreed on
the same logical scale values.
"""

from __future__ import annotations

import pytest
import torch

from fastvideo.layers.mxfp8linear import (
    _as_e8m0_scales,
    mxfp8_scaled_mm,
    quantize_mxfp8_blockwise,
)


def test_uint8_scales_are_presented_as_e8m0() -> None:
    raw = torch.zeros(8, dtype=torch.uint8)
    assert _as_e8m0_scales(raw).dtype == torch.float8_e8m0fnu
    e8m0 = raw.view(torch.float8_e8m0fnu)
    assert _as_e8m0_scales(e8m0) is e8m0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the block-scaled MXFP8 GEMM needs a Blackwell GPU")
def test_mxfp8_scaled_mm_accepts_uint8_and_e8m0_scales() -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    activation_values, activation_scales = quantize_mxfp8_blockwise(x)
    weight_values, weight_scales = quantize_mxfp8_blockwise(w)

    expected = mxfp8_scaled_mm(activation_values, activation_scales, weight_values, weight_scales, None)
    # Same bytes, flashinfer's dtype spelling.
    from_uint8 = mxfp8_scaled_mm(
        activation_values,
        activation_scales.view(torch.uint8),
        weight_values,
        weight_scales.view(torch.uint8),
        None,
    )
    torch.testing.assert_close(from_uint8, expected, atol=0, rtol=0)

    reference = x.float() @ w.float().T
    relative_error = (expected.float() - reference).norm() / reference.norm()
    assert relative_error < 0.05, f"MXFP8 GEMM relative error {relative_error:.4f} is too large"
