# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 NVFP4 targeting and inference-LoRA lifecycle tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import fastvideo.layers.quantization.mxfp8_config as mxfp8
import fastvideo.layers.quantization.nvfp4_config as nvfp4


class _TaggedModel(nn.Module):
    """Minimal model whose child carries one quantization method."""

    def __init__(self, quant_method: object) -> None:
        super().__init__()
        self.linear = nn.Module()
        self.linear.quant_method = quant_method


@pytest.mark.parametrize(
    "prefix",
    [
        "transformer_blocks.0.ff.fc_in",
        "transformer_blocks.49.ff.fc_out",
        "minimax_h3.transformer_blocks.12.ff.fc_in",
    ],
)
def test_is_minimax_h3_nvfp4_linear_prefix_main_ffn_linear(prefix: str) -> None:
    assert nvfp4.is_minimax_h3_nvfp4_linear_prefix(prefix)


@pytest.mark.parametrize(
    "prefix",
    [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.adaln_proj.linear",
        "token_refiner.refiner_blocks.0.ff.fc_in",
        "transformer_blocks.0.ff.extra",
    ],
)
def test_is_minimax_h3_nvfp4_linear_prefix_non_main_ffn_linear(prefix: str) -> None:
    assert not nvfp4.is_minimax_h3_nvfp4_linear_prefix(prefix)


def test_nvfp4config_get_quant_method_minimax_h3_feed_forward() -> None:
    if not torch.cuda.is_available():
        pytest.skip("NVFP4QuantizeMethod construction requires CUDA")

    from fastvideo.models.dits.minimax_h3 import MiniMaxH3FeedForward

    feed_forward = MiniMaxH3FeedForward(
        hidden_size=16,
        ffn_dim=32,
        quant_config=nvfp4.NVFP4Config(),
        prefix="minimax_h3.transformer_blocks.0.ff",
    )

    assert isinstance(feed_forward.fc_in.quant_method, nvfp4.NVFP4QuantizeMethod)
    assert isinstance(feed_forward.fc_out.quant_method, nvfp4.NVFP4QuantizeMethod)


def test_minimax_h3_feed_forward_nvfp4_cuda_similarity() -> None:
    """Run both H3 FFN GEMMs through NVFP4 and compare with BF16."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MiniMax-H3 NVFP4 inference requires an NVIDIA Blackwell GPU")
    pytest.importorskip("flashinfer")

    from fastvideo.models.dits.minimax_h3 import MiniMaxH3FeedForward

    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        dense = MiniMaxH3FeedForward(128, 256, prefix="minimax_h3.transformer_blocks.0.ff")
        quantized = MiniMaxH3FeedForward(
            128,
            256,
            quant_config=nvfp4.NVFP4Config(),
            prefix="minimax_h3.transformer_blocks.0.ff",
        )
    finally:
        torch.set_default_dtype(previous_default_dtype)

    generator = torch.Generator().manual_seed(20260901)
    with torch.no_grad():
        for parameter in dense.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.02)
    quantized.load_state_dict(dense.state_dict(), strict=True)
    dense = dense.cuda().eval()
    quantized = quantized.cuda().eval()
    nvfp4.convert_model_to_nvfp4(quantized)

    inputs = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        dense_output = dense(inputs)
        quantized_output = quantized(inputs)

    similarity = torch.nn.functional.cosine_similarity(
        dense_output.float().flatten(),
        quantized_output.float().flatten(),
        dim=0,
    )
    assert dense_output.dtype == torch.bfloat16
    assert quantized_output.dtype == torch.bfloat16
    assert similarity > 0.97
    assert quantized.fc_in.weight is None
    assert quantized.fc_out.weight is None


def test_maybe_quantize_model_lora_defers_nvfp4(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastvideo.models.loader.fsdp_load import _maybe_quantize_model

    model = _TaggedModel(object.__new__(nvfp4.NVFP4QuantizeMethod))
    converted: list[nn.Module] = []
    monkeypatch.setattr(nvfp4, "convert_model_to_nvfp4", converted.append)

    _maybe_quantize_model(model, defer_weight_conversion_until_lora_merge=True)
    assert converted == []

    _maybe_quantize_model(model)
    assert converted == [model]


def test_convert_quantized_weights_after_lora_merge_dispatches_nvfp4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastvideo.pipelines.lora_pipeline import _convert_quantized_weights_after_lora_merge

    model = _TaggedModel(object.__new__(nvfp4.NVFP4QuantizeMethod))
    nvfp4_converted: list[nn.Module] = []
    mxfp8_converted: list[nn.Module] = []
    monkeypatch.setattr(nvfp4, "convert_model_to_nvfp4", nvfp4_converted.append)
    monkeypatch.setattr(mxfp8, "convert_model_to_mxfp8", mxfp8_converted.append)

    _convert_quantized_weights_after_lora_merge({"transformer": model})

    assert nvfp4_converted == [model]
    assert mxfp8_converted == []


def test_unmerge_lora_weights_rejects_quantized_mxfp8() -> None:
    """Reject LoRA unmerge while the MXFP8 weight still contains the merged adapter."""
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    transformer = _TaggedModel(object.__new__(mxfp8.MXFP8QuantizeMethod))
    transformer.linear.register_buffer("_mxfp8_weight", torch.empty(1))
    pipeline = SimpleNamespace(trainable_transformer_modules={"transformer": transformer})

    with pytest.raises(RuntimeError, match="LoRA unmerge is unsupported after MXFP8 weight quantization"):
        LoRAPipeline.unmerge_lora_weights(pipeline)


@pytest.mark.parametrize("retain_bf16_weight", [False, True])
def test_unmerge_lora_weights_rejects_quantized_nvfp4(retain_bf16_weight: bool) -> None:
    """Reject NVFP4 LoRA unmerge whether the BF16 weight was removed or retained."""
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    transformer = _TaggedModel(object.__new__(nvfp4.NVFP4QuantizeMethod))
    transformer.linear.register_parameter(
        "weight",
        nn.Parameter(torch.empty(1), requires_grad=False) if retain_bf16_weight else None,
    )
    transformer.linear.register_buffer("_nvfp4_weight", torch.empty(1))
    pipeline = SimpleNamespace(trainable_transformer_modules={"transformer": transformer})

    with pytest.raises(RuntimeError, match="LoRA unmerge is unsupported after NVFP4 weight quantization"):
        LoRAPipeline.unmerge_lora_weights(pipeline)


def test_set_lora_adapter_rejects_nvfp4_without_bf16_weight() -> None:
    """Reject runtime adapter changes after NVFP4 removes the BF16 weight."""
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    transformer = _TaggedModel(object.__new__(nvfp4.NVFP4QuantizeMethod))
    transformer.linear.register_parameter("weight", None)
    transformer.linear.register_buffer("_nvfp4_weight", torch.empty(1))
    pipeline = SimpleNamespace(
        _setting_constructor_adapter=False,
        cur_adapter_name="first-adapter",
        cur_adapter_path="/first-adapter",
        cur_adapter_strength=1.0,
        lora_adapter_paths={},
        trainable_transformer_modules={"transformer": transformer},
    )

    with pytest.raises(RuntimeError, match="Runtime LoRA adapter changes are unsupported after NVFP4 quantization"):
        LoRAPipeline.set_lora_adapter(pipeline, "second-adapter", "/second-adapter")


def test_set_lora_adapter_mxfp8_requantizes_after_adapter_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requantize MXFP8 weights after merging a replacement LoRA adapter."""
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    events: list[str] = []

    class _RecordingLoRALayer:

        def set_lora_weights(self, lora_a: torch.Tensor, lora_b: torch.Tensor, **kwargs) -> None:
            del lora_a, lora_b, kwargs
            events.append("adapter merged")

    transformer = _TaggedModel(object.__new__(mxfp8.MXFP8QuantizeMethod))
    transformer.linear.register_buffer("_mxfp8_weight", torch.empty(1))
    transformer_lora_layers = SimpleNamespace(
        lora_layers_by_block=lambda: iter([(None, {"linear": _RecordingLoRALayer()})])
    )
    pipeline = SimpleNamespace(
        _setting_constructor_adapter=False,
        _constructor_dense_lora_path=None,
        cur_adapter_name="first-adapter",
        cur_adapter_path=None,
        cur_adapter_strength=1.0,
        lora_adapter_paths={},
        lora_adapters={
            "second-adapter": {
                "linear.lora_A": torch.empty(1, 1),
                "linear.lora_B": torch.empty(1, 1),
            }
        },
        lora_initialized=True,
        lora_layers={"transformer": transformer_lora_layers},
        fastvideo_args=SimpleNamespace(training_mode=False),
        trainable_transformer_modules={"transformer": transformer},
    )
    monkeypatch.setattr("fastvideo.pipelines.lora_pipeline.dist.get_rank", lambda: 0)
    monkeypatch.setattr(mxfp8, "convert_model_to_mxfp8", lambda model: events.append("weights requantized"))

    LoRAPipeline.set_lora_adapter(pipeline, "second-adapter")

    assert events == ["adapter merged", "weights requantized"]
