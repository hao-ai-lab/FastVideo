# SPDX-License-Identifier: Apache-2.0
"""The fused FP8 activation quantizers must reproduce the eager formulas: identical
scales, and dequantized values within one FP8 rounding step (the fused kernels scale
in FP32 where the eager path scales in the activation dtype). NaN inputs must give
NaN scales and NaN FP8 values, as they do in the eager path."""
import pytest
import torch

from fastvideo.layers.quantization import fp8_config, fp8_quant_kernels
from fastvideo.layers.quantization.fp8_quant_kernels import (
    FP8_DTYPE,
    FP8_MAX,
    quantize_rowwise_fused,
    quantize_tensorwise_fused,
    triton_quant_available,
)

# Triton is optional: without it the eager formulas run on the GPU too.
needs_gpu_and_triton = pytest.mark.skipif(not (torch.cuda.is_available() and fp8_quant_kernels._HAS_TRITON),
                                          reason="needs a GPU with Triton")


def _eager_tensorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x.abs().amax().float() / FP8_MAX).clamp(min=fp8_config.FP8_MIN_SCALE)
    return (x / scale.to(x.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE), scale.view(1)


def _eager_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x.abs().amax(dim=-1, keepdim=True).float() / FP8_MAX).clamp(min=fp8_config.FP8_MIN_SCALE)
    return (x / scale.to(x.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE), scale


def _assert_eager_path(x: torch.Tensor) -> None:
    assert not triton_quant_available(x)
    for quantize, eager in ((fp8_config._quantize_tensorwise, _eager_tensorwise),
                            (fp8_config._quantize_rowwise, _eager_rowwise)):
        q, s = quantize(x)
        q_ref, s_ref = eager(x)
        assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(s, s_ref)


def test_cpu_inputs_keep_the_eager_path() -> None:
    _assert_eager_path(torch.randn(8, 64, dtype=torch.bfloat16))


def test_cpu_empty_rows_keep_the_eager_path() -> None:
    x = torch.empty((0, 96), dtype=torch.bfloat16)
    q, s = fp8_config._quantize_rowwise(x)
    q_ref, s_ref = _eager_rowwise(x)
    assert q.shape == (0, 96) and s.shape == (0, 1)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(s, s_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_without_triton_keeps_the_eager_path(monkeypatch) -> None:
    monkeypatch.setattr(fp8_quant_kernels, "_HAS_TRITON", False)
    # On sm89 rowwise quantization would otherwise take the fp8_kernels path,
    # which scales in FP32 and so is not bitwise equal to the eager formula.
    monkeypatch.setattr(fp8_config, "_rowwise_scaled_mm_is_slow", lambda: False)
    _assert_eager_path(torch.randn(8, 64, device="cuda", dtype=torch.bfloat16))


@needs_gpu_and_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("m", "k", "magnitude"),
    [
        (1000, 5376, 3.0),
        (4096, 14336, 3.0),
        (7, 96, 3.0),
        # Scales near FP8_MIN_SCALE are subnormal in fp16, where the eager path
        # rounds the scale to the activation dtype before dividing.
        (64, 5376, 1e-4),
    ])
def test_fused_matches_eager_on_gpu(m: int, k: int, magnitude: float, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = (torch.randn(m, k, device="cuda") * magnitude).to(dtype)
    x[0, 0] = 200.0 * magnitude / 3.0  # an outlier separates the tensorwise and rowwise scales
    assert triton_quant_available(x)
    for fused, eager in ((quantize_tensorwise_fused, _eager_tensorwise), (quantize_rowwise_fused, _eager_rowwise)):
        q, s = fused(x)
        q_ref, s_ref = eager(x)
        assert q.dtype == FP8_DTYPE and q.shape == x.shape and s.shape == s_ref.shape
        torch.testing.assert_close(s, s_ref, rtol=1e-6, atol=0)
        dequant, dequant_ref = q.float() * s, q_ref.float() * s_ref
        # one FP8 ulp at the row/tensor maximum, scaled back to the activation's range
        ulp = (s * FP8_MAX / 2**3).max()
        assert (dequant - dequant_ref).abs().max() <= ulp


@needs_gpu_and_triton
def test_fused_rowwise_empty_rows_on_gpu() -> None:
    x = torch.empty((0, 96), device="cuda", dtype=torch.bfloat16)
    q, s = quantize_rowwise_fused(x)
    q_ref, s_ref = _eager_rowwise(x)
    assert q.shape == q_ref.shape and s.shape == s_ref.shape
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(s, s_ref)


@needs_gpu_and_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_keeps_nan_like_eager_on_gpu(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 96, device="cuda").to(dtype)
    x[1, 5] = float("nan")
    for fused, eager in ((quantize_tensorwise_fused, _eager_tensorwise), (quantize_rowwise_fused, _eager_rowwise)):
        q, s = fused(x)
        q_ref, s_ref = eager(x)
        assert s_ref.isnan().any()  # the eager behaviour that the fused kernels must keep
        assert torch.equal(s.isnan(), s_ref.isnan())
        assert torch.equal(q.float().isnan(), q_ref.float().isnan())
        finite = ~s_ref.isnan()
        torch.testing.assert_close(s[finite], s_ref[finite], rtol=1e-6, atol=0)
