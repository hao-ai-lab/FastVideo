# SPDX-License-Identifier: Apache-2.0
"""MXFP4 QAT numerics gate for the Apple/MLX deployment path.

The 5B QAD run must train against the same MXFP4 grid used by MLX deploy:
``mx.quantize(w, mode="mxfp4")`` followed by ``mx.dequantize``. This test pins
the torch-only fake-quant twin to the real MLX 0.31.2 Metal behavior before any
GPU training budget is spent.

Unlike affine INT8, MXFP4 dequantizes to bf16 and uses a fixed block-scaled
format: 32 weights share one E8M0 power-of-two scale, and each element stores a
4-bit E2M1 code. For the finite random weights used in QAT, the torch twin can
match MLX's packed codes, scale bytes, and dequantized bf16 values exactly.
The tolerance for the dequantized tensor is therefore zero.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core", reason="MLX is required for QAT numerics parity tests")

from fastvideo.layers.quantization.mlx_mxfp4_qat import (  # noqa: E402
    DEFAULT_GROUP_SIZE,
    fake_quantize_mlx_mxfp4,
    mlx_mxfp4_dequantize_reference,
    mlx_mxfp4_quantize_reference,
)


def _unpack_mxfp4_codes(packed: np.ndarray, *, out_cols: int) -> np.ndarray:
    words = packed.astype(np.uint64)
    codes = np.zeros((*packed.shape[:-1], packed.shape[-1] * 8), dtype=np.int32)
    for k in range(8):
        codes[..., k::8] = ((words >> (k * 4)) & 0xF).astype(np.int32)
    return codes[..., :out_cols]


def _mlx_mxfp4_quantize(w_np: np.ndarray, *, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with mx.stream(device):
        q, scales = mx.quantize(mx.array(w_np), mode="mxfp4")
        deq = mx.dequantize(q, scales, mode="mxfp4")
        mx.eval(q, scales, deq)
    return np.array(q), np.array(scales), np.array(deq.astype(mx.float32))


@pytest.mark.skipif(not mx.metal.is_available(), reason="MXFP4 deploy parity is pinned to MLX Metal")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 64), (7, 96), (2, 128), (3, 320)])
def test_mxfp4_quantizer_bitmatches_mlx_metal(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    """Codes, E8M0 scale bytes, and bf16 dequant values match MLX Metal exactly."""
    torch.manual_seed(2026)
    base = torch.randn(*shape, dtype=torch.float32) * 0.3
    # Add a deterministic ramp so the test covers saturation and E2M1 midpoint
    # neighborhoods instead of only near-zero Gaussian weights.
    ramp = torch.linspace(-2.0, 2.0, shape[-1], dtype=torch.float32).expand(shape[0], -1)
    w = (base + ramp).to(dtype)

    codes, scales = mlx_mxfp4_quantize_reference(w)
    deq = mlx_mxfp4_dequantize_reference(codes, scales, out_shape=w.shape)

    w_np = w.float().numpy() if dtype == torch.float32 else w.numpy()
    q_mlx, scales_mlx, deq_mlx = _mlx_mxfp4_quantize(w_np, device=mx.gpu)
    codes_mlx = _unpack_mxfp4_codes(q_mlx, out_cols=shape[-1])

    np.testing.assert_array_equal(codes.reshape(shape[0], -1).numpy(), codes_mlx)
    np.testing.assert_array_equal(scales.numpy(), scales_mlx)
    np.testing.assert_array_equal(deq.numpy(), deq_mlx)


@pytest.mark.skipif(not mx.metal.is_available(), reason="MXFP4 deploy parity is pinned to MLX Metal")
def test_fake_quantize_matches_mlx_metal_and_passes_gradients() -> None:
    """The QAT forward equals real MLX MXFP4 dequant, and STE gradients pass unchanged."""
    torch.manual_seed(99)
    w = (torch.randn(4, 128, dtype=torch.float32) * 0.4).to(torch.bfloat16).requires_grad_(True)

    fq = fake_quantize_mlx_mxfp4(w, simulate_dtype=torch.float16)

    w_fp16 = w.detach().to(torch.float16)
    _, _, deq_mlx = _mlx_mxfp4_quantize(w_fp16.numpy(), device=mx.gpu)
    np.testing.assert_array_equal(fq.detach().numpy(), deq_mlx)

    fq.sum().backward()
    assert w.grad is not None
    np.testing.assert_array_equal(w.grad.float().numpy(), np.ones_like(w.grad.float().numpy()))


def test_indivisible_group_size_raises() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        mlx_mxfp4_quantize_reference(torch.randn(4, 100))


def test_non_default_group_size_raises() -> None:
    with pytest.raises(ValueError, match="fixed group_size 32"):
        mlx_mxfp4_quantize_reference(torch.randn(4, DEFAULT_GROUP_SIZE), group_size=64)
