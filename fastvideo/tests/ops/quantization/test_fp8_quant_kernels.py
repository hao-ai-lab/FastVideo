# SPDX-License-Identifier: Apache-2.0
"""The fused FP8 activation quantizers must reproduce the eager formulas: identical
scales, and dequantized values within one FP8 rounding step (the fused kernels scale
in FP32 where the eager path scales in the activation dtype)."""
import pytest
import torch

from fastvideo.layers.quantization import fp8_config
from fastvideo.layers.quantization.fp8_quant_kernels import (
    FP8_DTYPE,
    FP8_MAX,
    quantize_rowwise_fused,
    quantize_tensorwise_fused,
    triton_quant_available,
)


def _eager_tensorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x.abs().amax().float() / FP8_MAX).clamp(min=fp8_config.FP8_MIN_SCALE)
    return (x / scale.to(x.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE), scale.view(1)


def _eager_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x.abs().amax(dim=-1, keepdim=True).float() / FP8_MAX).clamp(min=fp8_config.FP8_MIN_SCALE)
    return (x / scale.to(x.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE), scale


def test_cpu_inputs_keep_the_eager_path() -> None:
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    assert not triton_quant_available(x)
    q, s = fp8_config._quantize_tensorwise(x)
    q_ref, s_ref = _eager_tensorwise(x)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(s, s_ref)
    q, s = fp8_config._quantize_rowwise(x)
    q_ref, s_ref = _eager_rowwise(x)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(s, s_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU with Triton")
@pytest.mark.parametrize(("m", "k"), [(1000, 5376), (4096, 14336), (7, 96)])
def test_fused_matches_eager_on_gpu(m: int, k: int) -> None:
    torch.manual_seed(0)
    x = (torch.randn(m, k, device="cuda") * 3).to(torch.bfloat16)
    x[0, 0] = 200.0  # an outlier separates the tensorwise and rowwise scales
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
