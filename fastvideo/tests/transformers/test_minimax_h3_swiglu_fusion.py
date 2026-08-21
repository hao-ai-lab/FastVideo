# SPDX-License-Identifier: Apache-2.0
"""Focused tests for MiniMax H3's value-first packed SwiGLU fusion."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastvideo.models.dits.minimax_h3_fusions.swiglu import minimax_h3_swiglu


def _require_bf16_triton_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("MiniMax H3 fused SwiGLU requires CUDA")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("MiniMax H3 fused SwiGLU parity requires BF16 support")
    pytest.importorskip("triton", reason="MiniMax H3 fused SwiGLU requires Triton")


def _assert_bf16_parity(x: torch.Tensor) -> None:
    value, gate = x.chunk(2, dim=-1)
    expected = value * F.silu(gate)
    actual = minimax_h3_swiglu(x)
    assert actual.shape == (*x.shape[:-1], x.shape[-1] // 2)
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    # Sol-Engine keeps the full SwiGLU expression in FP32 until the output
    # store, while eager F.silu materializes a BF16 intermediate.
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("last_dim", [0, 7])
def test_minimax_h3_swiglu_rejects_nonpositive_or_odd_last_dimension(last_dim: int) -> None:
    x = torch.empty((2, last_dim), dtype=torch.float32)

    with pytest.raises(ValueError, match="positive even last dimension"):
        minimax_h3_swiglu(x)


def test_minimax_h3_swiglu_strict_wrapper_rejects_cpu() -> None:
    with pytest.raises(ValueError, match="requires a CUDA tensor"):
        minimax_h3_swiglu(torch.randn(2, 8))


@pytest.mark.gpu
def test_minimax_h3_swiglu_multidimensional_bf16_gpu_parity() -> None:
    _require_bf16_triton_cuda()
    torch.manual_seed(0)
    x = torch.randn((2, 3, 5, 66), device="cuda", dtype=torch.bfloat16)

    _assert_bf16_parity(x)


@pytest.mark.gpu
def test_minimax_h3_swiglu_noncontiguous_bf16_gpu_parity() -> None:
    _require_bf16_triton_cuda()
    torch.manual_seed(1)
    storage = torch.randn((2, 3, 148), device="cuda", dtype=torch.bfloat16)
    x = storage[..., ::2]
    assert not x.is_contiguous()

    _assert_bf16_parity(x)


@pytest.mark.gpu
def test_minimax_h3_swiglu_real_ffn_dim_bf16_gpu() -> None:
    _require_bf16_triton_cuda()
    torch.manual_seed(2)
    ffn_dim = 14336
    x = torch.randn((1, 2 * ffn_dim), device="cuda", dtype=torch.bfloat16)

    _assert_bf16_parity(x)
