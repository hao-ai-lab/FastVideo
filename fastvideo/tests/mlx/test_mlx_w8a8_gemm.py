# SPDX-License-Identifier: Apache-2.0
"""Correctness contracts for the W8A8 fused INT8 Metal GEMM prototype.

Requires MLX + Metal. Skips cleanly when unavailable so Linux CI stays green.
"""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx")
mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - non-Apple CI
    pytest.skip("Metal unavailable", allow_module_level=True)

from fastvideo.mlx_runtime.w8a8_gemm import (  # noqa: E402
    dequant_reference,
    quantize_activations_per_token,
    quantize_per_row,
    quantize_weights_per_out_channel,
    w8a8_linear,
    w8a8_matmul,
)


@pytest.mark.parametrize("kind", ["naive", "tiled"])
@pytest.mark.parametrize("shape", [(32, 48, 64), (17, 33, 65), (128, 128, 128)])
def test_w8a8_matches_dequant_reference(kind: str, shape: tuple[int, int, int]) -> None:
    m, n, k = shape
    rng = np.random.default_rng(0)
    x = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((n, k)).astype(np.float32)
    a = quantize_activations_per_token(x)
    b = quantize_weights_per_out_channel(w)
    ref = dequant_reference(a, b)
    got = np.array(w8a8_matmul(a, b, kind=kind))
    # Kernel must match the integer reference; allow tiny fp32 reduction noise.
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-4)


def test_quantize_per_row_roundtrip_bound() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 64)).astype(np.float32) * 3.0
    q, scale = quantize_per_row(x)
    recon = q.astype(np.float32) * scale[:, None]
    # Symmetric int8 absmax quant: max err ≤ scale/2 per row.
    max_err = np.max(np.abs(recon - x), axis=1)
    np.testing.assert_array_less(max_err, scale * 0.5 + 1e-6)


def test_w8a8_linear_with_bias() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    bias = rng.standard_normal((32,)).astype(np.float32)
    y = np.array(w8a8_linear(x, w, bias=mx.array(bias), kind="naive"))
    a = quantize_activations_per_token(x)
    b = quantize_weights_per_out_channel(w)
    ref = dequant_reference(a, b) + bias
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-4)


def test_k_mismatch_raises() -> None:
    a = quantize_activations_per_token(np.zeros((4, 16), dtype=np.float32))
    b = quantize_weights_per_out_channel(np.zeros((8, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="K mismatch"):
        w8a8_matmul(a, b)
