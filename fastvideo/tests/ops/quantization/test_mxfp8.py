# SPDX-License-Identifier: Apache-2.0
"""Blackwell numerical parity tests for FastVideo MXFP8 operations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
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


class _FeedForwardBlock(nn.Module):
    """One H3 transformer block, so its feed-forward can be offloaded as a unit."""

    def __init__(self, feed_forward: nn.Module) -> None:
        super().__init__()
        self.ff = feed_forward

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ff(hidden_states)


class _TransformerBlocks(nn.Module):
    """ModuleList owner, which is what ``enable_layerwise_offload`` attaches to.

    The trailing ``nn.Identity`` keeps the circular prefetch link off the FFN block: a
    single block would prefetch into its own still-resident state.
    """

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([block, nn.Identity()])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def _relative_l2_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative L2 error between two tensors, flattened to one vector."""
    actual_flat = actual.float().flatten()
    expected_flat = expected.float().flatten()
    return (torch.linalg.vector_norm(actual_flat - expected_flat) / torch.linalg.vector_norm(expected_flat)).item()


def _build_lora_mxfp8_round_trip(
        layerwise_offload: bool
) -> tuple[nn.Module, nn.Module, SimpleNamespace, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an MXFP8 FFN with a merged adapter plus its dense BF16 references.

    Returns the MXFP8 transformer, its feed-forward, a pipeline stand-in, the inputs and
    the dense BF16 references: base-only, adapter merged into the weights, and adapter
    unmerged (applied as a runtime delta, as BaseLayerWithLoRA.forward computes it).
    """
    from fastvideo.hooks.layerwise_offload import enable_layerwise_offload
    from fastvideo.layers.lora.linear import get_lora_layer
    from fastvideo.layers.quantization.mxfp8_config import MXFP8Config
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3FeedForward
    from fastvideo.pipelines.lora_pipeline import (LoRAModelLayers, _convert_quantized_weights_after_lora_change,
                                                   _get_hook_ctx)

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

    block = _FeedForwardBlock(quantized).cuda().eval()
    transformer = _TransformerBlocks(block).cuda().eval()
    if layerwise_offload:
        enable_layerwise_offload(transformer)

    # An adapter delta well above the 0.02 weight scale, so the adapter-active states
    # are distinguishable from the base-only output by more than the MXFP8 error.
    adapter = {
        "fc_in": (
            torch.randn(8, 128, generator=generator, dtype=torch.bfloat16).cuda() * 0.02,
            torch.randn(512, 8, generator=generator, dtype=torch.bfloat16).cuda(),
        ),
        "fc_out": (
            torch.randn(8, 256, generator=generator, dtype=torch.bfloat16).cuda() * 0.02,
            torch.randn(128, 8, generator=generator, dtype=torch.bfloat16).cuda(),
        ),
    }

    lora_layers = LoRAModelLayers([("transformer_blocks.0", block)])
    with _get_hook_ctx(block):
        for name in ("fc_in", "fc_out"):
            wrapped = get_lora_layer(getattr(quantized, name), lora_rank=8, lora_alpha=8)
            assert wrapped is not None
            setattr(quantized, name, wrapped)
            lora_layers.add_lora_layer("transformer_blocks.0", f"ff.{name}", wrapped)
            wrapped.set_lora_weights(*adapter[name], lora_alpha=8)
        # The pipeline repacks inside this scope; outside it the block is a placeholder.
        _convert_quantized_weights_after_lora_change([block])

    hidden_states = torch.randn(2, 128, 128, generator=generator, dtype=torch.bfloat16).cuda()
    with torch.inference_mode():
        dense_base = dense(hidden_states)
    for name in ("fc_in", "fc_out"):
        wrapped = get_lora_layer(getattr(dense, name), lora_rank=8, lora_alpha=8)
        assert wrapped is not None
        setattr(dense, name, wrapped)
        wrapped.set_lora_weights(*adapter[name], lora_alpha=8)
    with torch.inference_mode():
        dense_merged = dense(hidden_states)
    for name in ("fc_in", "fc_out"):
        getattr(dense, name).unmerge_lora_weights()
    with torch.inference_mode():
        dense_unmerged = dense(hidden_states)

    pipeline = SimpleNamespace(
        lora_layers={"transformer": lora_layers},
        trainable_transformer_modules={"transformer": transformer},
    )
    return transformer, quantized, pipeline, hidden_states, dense_base, dense_merged, dense_unmerged


@pytest.mark.parametrize("layerwise_offload", [False, True])
def test_minimax_h3_mxfp8_feed_forward_lora_round_trip_matches_bf16(layerwise_offload: bool) -> None:
    """Merge, unmerge and remerge a real adapter on an MXFP8 FFN and compare with BF16."""
    _require_blackwell()
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    transformer, quantized, pipeline, hidden_states, dense_base, dense_merged, dense_unmerged = (
        _build_lora_mxfp8_round_trip(layerwise_offload))

    with torch.inference_mode():
        merged_output = transformer(hidden_states)
    assert _relative_l2_error(merged_output, dense_merged) < 0.10
    # Repacking outside the offload scope would leave 0-size buffers behind.
    assert quantized.fc_in.base_layer._mxfp8_weight.shape == (512, 128)
    assert quantized.fc_out.base_layer._mxfp8_weight.shape == (128, 256)

    LoRAPipeline.unmerge_lora_weights(pipeline)
    with torch.inference_mode():
        unmerged_output = transformer(hidden_states)
    # Unmerged inference keeps the adapter active through the runtime delta, so it must
    # match the unmerged BF16 LoRA execution and stay closer to it than to base-only.
    assert _relative_l2_error(unmerged_output, dense_unmerged) < 0.10
    assert _relative_l2_error(unmerged_output, dense_unmerged) < _relative_l2_error(unmerged_output, dense_base)

    LoRAPipeline.merge_lora_weights(pipeline)
    with torch.inference_mode():
        remerged_output = transformer(hidden_states)
    assert _relative_l2_error(remerged_output, dense_merged) < 0.10
