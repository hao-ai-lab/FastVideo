# SPDX-License-Identifier: Apache-2.0
"""Blackwell numerical parity tests for FastVideo MXFP8 operations."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from quack.mx_utils import to_blocked, to_mx

from fastvideo.layers.mxfp8linear import (
    quantize_mxfp8_blockwise,
    quantize_mxfp8_weight_blockwise,
    swiglu_quantize_mxfp8_blockwise,
)


def _require_blackwell() -> None:
    """Require hardware that can execute MXFP8 block-scaled GEMMs."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 numerical parity requires an NVIDIA Blackwell GPU")


def _quack_mxfp8(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one matrix with the independent Quack reference operations."""
    values, natural_scales = to_mx(matrix.contiguous(), 32)
    return values, to_blocked(natural_scales)


def _assert_same_mxfp8(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Require identical FP8 values and E8M0 scale bytes."""
    actual_values, actual_scales = actual
    expected_values, expected_scales = expected
    torch.testing.assert_close(actual_values.float(), expected_values.float(), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_scales.view(torch.uint8),
        expected_scales.view(torch.uint8),
        rtol=0,
        atol=0,
    )


def test_mxfp8_weight_quantization_matches_quack() -> None:
    """Compare prequantized weight values and scales with Quack."""
    _require_blackwell()
    generator = torch.Generator().manual_seed(20260905)
    weight = torch.randn(129, 256, generator=generator, dtype=torch.bfloat16).cuda()

    _assert_same_mxfp8(quantize_mxfp8_weight_blockwise(weight), _quack_mxfp8(weight))


def test_mxfp8_activation_quantization_matches_quack() -> None:
    """Compare fused activation quantization and scale swizzling with Quack."""
    _require_blackwell()
    generator = torch.Generator().manual_seed(20260905)
    activation = torch.randn(129, 256, generator=generator, dtype=torch.bfloat16).cuda()

    _assert_same_mxfp8(quantize_mxfp8_blockwise(activation), _quack_mxfp8(activation))


def test_mxfp8_swiglu_quantization_matches_bf16_quack() -> None:
    """Compare fused SwiGLU quantization with BF16 SwiGLU followed by Quack."""
    _require_blackwell()
    generator = torch.Generator().manual_seed(20260905)
    preactivation = torch.randn(129, 512, generator=generator, dtype=torch.bfloat16).cuda()
    values, gates = preactivation.chunk(2, dim=-1)
    bf16_swiglu = (values.float() * gates.float() * torch.sigmoid(gates.float())).to(torch.bfloat16)

    _assert_same_mxfp8(
        swiglu_quantize_mxfp8_blockwise(preactivation),
        _quack_mxfp8(bf16_swiglu),
    )


def test_minimax_h3_mxfp8_feed_forward_matches_bf16() -> None:
    """Compare the complete MXFP8 H3 feed-forward output with BF16."""
    _require_blackwell()
    from fastvideo.layers.quantization.mxfp8_config import MXFP8Config, convert_model_to_mxfp8
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3FeedForward

    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        dense = MiniMaxH3FeedForward(128, 256, prefix="minimax_h3.transformer_blocks.0.ff")
        quantized = MiniMaxH3FeedForward(
            128,
            256,
            quant_config=MXFP8Config(),
            prefix="minimax_h3.transformer_blocks.0.ff",
        )
    finally:
        torch.set_default_dtype(previous_default_dtype)

    generator = torch.Generator().manual_seed(20260905)
    with torch.no_grad():
        for parameter in dense.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.02)
    quantized.load_state_dict(dense.state_dict(), strict=True)
    dense = dense.cuda().eval()
    quantized = quantized.cuda().eval()
    assert convert_model_to_mxfp8(quantized) == 2

    hidden_states = torch.randn(2, 128, 128, generator=generator, dtype=torch.bfloat16).cuda()
    with torch.inference_mode():
        dense_output = dense(hidden_states)
        quantized_output = quantized(hidden_states)

    dense_output_flat = dense_output.float().flatten()
    quantized_output_flat = quantized_output.float().flatten()
    similarity = F.cosine_similarity(dense_output_flat, quantized_output_flat, dim=0)
    relative_l2_error = torch.linalg.vector_norm(dense_output_flat - quantized_output_flat) / torch.linalg.vector_norm(
        dense_output_flat
    )
    assert dense_output.dtype == torch.bfloat16
    assert quantized_output.dtype == torch.bfloat16
    assert similarity > 0.995
    assert relative_l2_error < 0.10
