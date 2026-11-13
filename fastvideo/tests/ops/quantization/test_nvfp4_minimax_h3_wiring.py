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


class _RecordingLoRALayer(nn.Module):
    """LoRA layer stand-in that records the lifecycle call it receives."""

    def __init__(self, events: list[str], quant_method: object) -> None:
        super().__init__()
        self.events = events
        self.quant_method = quant_method

    def unmerge_lora_weights(self, **kwargs) -> None:
        del kwargs
        self.events.append("adapter unmerged")

    def merge_lora_weights(self, **kwargs) -> None:
        del kwargs
        self.events.append("adapter merged")

    def set_lora_weights(self, lora_a: torch.Tensor, lora_b: torch.Tensor, **kwargs) -> None:
        del lora_a, lora_b, kwargs
        self.events.append("adapter merged")


def _mxfp8_pipeline(events: list[str], **extra: object) -> SimpleNamespace:
    """Build a pipeline stand-in whose only LoRA layer is not inside a block."""
    transformer = _TaggedModel(object.__new__(mxfp8.MXFP8QuantizeMethod))
    transformer.linear.register_buffer("_mxfp8_weight", torch.empty(1))
    layer = _RecordingLoRALayer(events, object.__new__(mxfp8.MXFP8QuantizeMethod))
    transformer_lora_layers = SimpleNamespace(lora_layers_by_block=lambda: iter([(None, {"linear": layer})]))
    return SimpleNamespace(
        lora_layers={"transformer": transformer_lora_layers},
        trainable_transformer_modules={"transformer": transformer},
        **extra,
    )


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


@pytest.mark.parametrize(
    "prefix",
    [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.41.attn.to_out",
        "minimax_h3.transformer_blocks.12.ff.fc_in",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.3.ff.fc_out",
    ],
)
def test_is_minimax_h3_nvfp4_dit_linear_prefix_attn_and_ffn(prefix: str) -> None:
    assert nvfp4.is_minimax_h3_nvfp4_dit_linear_prefix(prefix)


@pytest.mark.parametrize(
    "prefix",
    [
        "transformer_blocks.0.adaln_proj.linear",
        "token_refiner.refiner_blocks.0.ff.fc_in",
        "transformer_blocks.0.attn.to_gate_compress",
        "proj_in",
        "transformer_blocks.0.attn.to_q.weight",
    ],
)
def test_is_minimax_h3_nvfp4_dit_linear_prefix_rejects_non_export_linears(prefix: str) -> None:
    assert not nvfp4.is_minimax_h3_nvfp4_dit_linear_prefix(prefix)


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


def test_convert_quantized_weights_after_lora_change_dispatches_nvfp4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastvideo.pipelines.lora_pipeline import _convert_quantized_weights_after_lora_change

    model = _TaggedModel(object.__new__(nvfp4.NVFP4QuantizeMethod))
    nvfp4_converted: list[nn.Module] = []
    mxfp8_converted: list[nn.Module] = []
    monkeypatch.setattr(nvfp4, "convert_model_to_nvfp4", nvfp4_converted.append)
    monkeypatch.setattr(mxfp8, "convert_model_to_mxfp8", mxfp8_converted.append)

    _convert_quantized_weights_after_lora_change([model])

    assert nvfp4_converted == [model]
    assert mxfp8_converted == []


def test_unmerge_lora_weights_mxfp8_requantizes_after_unmerge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requantize MXFP8 weights after unmerging the LoRA adapter from BF16."""
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    events: list[str] = []
    pipeline = _mxfp8_pipeline(events)
    monkeypatch.setattr(mxfp8, "convert_model_to_mxfp8", lambda model: events.append("weights requantized"))

    LoRAPipeline.unmerge_lora_weights(pipeline)

    assert events == ["adapter unmerged", "weights requantized"]


def test_merge_lora_weights_mxfp8_requantizes_after_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requantize MXFP8 weights after merging the LoRA adapter into BF16."""
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    events: list[str] = []
    pipeline = _mxfp8_pipeline(events)
    monkeypatch.setattr(mxfp8, "convert_model_to_mxfp8", lambda model: events.append("weights requantized"))

    LoRAPipeline.merge_lora_weights(pipeline)

    assert events == ["adapter merged", "weights requantized"]


def test_unmerge_then_merge_lora_weights_mxfp8_requantizes_each_step(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Requantize MXFP8 weights after both unmerge and the following remerge."""
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    events: list[str] = []
    pipeline = _mxfp8_pipeline(events)
    monkeypatch.setattr(mxfp8, "convert_model_to_mxfp8", lambda model: events.append("weights requantized"))

    LoRAPipeline.unmerge_lora_weights(pipeline)
    LoRAPipeline.merge_lora_weights(pipeline)

    assert events == [
        "adapter unmerged",
        "weights requantized",
        "adapter merged",
        "weights requantized",
    ]


@pytest.mark.parametrize("retain_bf16_weight", [False, True])
@pytest.mark.parametrize("operation", ["merge_lora_weights", "unmerge_lora_weights"])
def test_nvfp4_without_lora_layers_is_a_lifecycle_noop(retain_bf16_weight: bool, operation: str,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """Public merge/unmerge only act on wrapped LoRA layers, not an adapter-free packed model."""
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline, _has_nvfp4_weights_without_bf16

    transformer = _TaggedModel(object.__new__(nvfp4.NVFP4QuantizeMethod))
    transformer.linear.register_parameter(
        "weight",
        nn.Parameter(torch.empty(1), requires_grad=False) if retain_bf16_weight else None,
    )
    transformer.linear.register_buffer("_nvfp4_weight", torch.empty(1))
    pipeline = SimpleNamespace(trainable_transformer_modules={"transformer": transformer}, lora_layers={})
    previous_weight = transformer.linear.weight
    previous_packed = transformer.linear._nvfp4_weight

    def unexpected_pack(model):
        pytest.fail("An adapter-free lifecycle operation must not repack the model")

    monkeypatch.setattr(nvfp4, "convert_model_to_nvfp4", unexpected_pack)
    # The conservative helper still scans an unwrapped transformer; public operations have no layers to change.
    assert _has_nvfp4_weights_without_bf16(pipeline, pipeline.trainable_transformer_modules) is (not retain_bf16_weight)
    getattr(LoRAPipeline, operation)(pipeline)
    assert transformer.linear.weight is previous_weight
    assert transformer.linear._nvfp4_weight is previous_packed


def test_set_lora_adapter_mxfp8_requantizes_after_adapter_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requantize MXFP8 weights after merging a replacement LoRA adapter."""
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    events: list[str] = []
    pipeline = _mxfp8_pipeline(
        events,
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
        fastvideo_args=SimpleNamespace(training_mode=False),
    )
    monkeypatch.setattr("fastvideo.pipelines.lora_pipeline.dist.get_rank", lambda: 0)
    monkeypatch.setattr(mxfp8, "convert_model_to_mxfp8", lambda model: events.append("weights requantized"))

    LoRAPipeline.set_lora_adapter(pipeline, "second-adapter")

    assert events == ["adapter merged", "weights requantized"]


class _CpuPackedNVFP4Method(nvfp4.NVFP4QuantizeMethod):
    """CPU stand-in for packed storage and parameter purge, not an NVFP4 kernel emulator."""

    def __init__(self, retain: bool) -> None:
        self.retain = retain
        self._retain_original_weights = retain
        self._is_refine_only_layer = False

    def apply(self, layer, x, bias=None):
        weight = layer._nvfp4_weight.to(x.dtype) * layer._nvfp4_weight_scale.to(x.dtype)
        return torch.nn.functional.linear(x, weight, bias)


def _cpu_packed_state(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    # Deterministic reduced-precision storage makes stale packed values distinguishable from a fresh rebuild.
    return {
        "_nvfp4_weight": torch.round(weight.float() * 32).to(torch.int16),
        "_nvfp4_weight_scale": torch.full((), 1 / 32),
        "_weight_global_sf": weight.float().abs().amax().to(torch.bfloat16),
        "_nvfp4_alpha": weight.float().abs().amax().reciprocal(),
    }


def _cpu_pack_nvfp4(model: nn.Module) -> None:
    for layer in model.modules():
        if isinstance(getattr(layer, "quant_method", None), _CpuPackedNVFP4Method) and layer.weight is not None:
            for name, value in _cpu_packed_state(layer.weight.detach()).items():
                layer.register_buffer(name, value, persistent=False)
            if not layer.quant_method.retain:
                layer.register_buffer("weight_device", torch.empty(0, device=layer.weight.device), persistent=False)
                layer.register_parameter("weight", None)


@pytest.fixture
def cpu_nvfp4_lifecycle(monkeypatch: pytest.MonkeyPatch):
    from fastvideo.layers.lora import linear as lora_linear

    monkeypatch.setattr(lora_linear, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nvfp4, "convert_model_to_nvfp4", _cpu_pack_nvfp4)
    monkeypatch.setattr("fastvideo.pipelines.lora_pipeline.dist.get_rank", lambda: 0)
    with torch._dynamo.config.patch(disable=True):
        yield


def _nvfp4_lifecycle_pipeline(retain: bool):
    from fastvideo.layers.linear import ReplicatedLinear
    from fastvideo.layers.lora.linear import get_lora_layer

    torch.manual_seed(7)
    base = ReplicatedLinear(32, 32, bias=False, params_dtype=torch.bfloat16)
    with torch.no_grad():
        base.weight.copy_(torch.randn_like(base.weight) / 8)
    original = base.weight.detach().clone()
    base.quant_method = _CpuPackedNVFP4Method(retain)
    wrapped = get_lora_layer(base)
    first = (torch.randn(4, 32, dtype=torch.bfloat16) / 8, torch.randn(32, 4, dtype=torch.bfloat16) / 4)
    second = (torch.randn(4, 32, dtype=torch.bfloat16) / 4, torch.randn(32, 4, dtype=torch.bfloat16) / 4)
    wrapped.set_lora_weights(*first, lora_alpha=4)
    _cpu_pack_nvfp4(wrapped)
    transformer = nn.Module()
    transformer.linear = wrapped
    layer_group = SimpleNamespace(lora_layers_by_block=lambda: iter([(None, {"linear": wrapped})]))
    pipeline = SimpleNamespace(
        lora_layers={"transformer": layer_group},
        trainable_transformer_modules={"transformer": transformer},
        _setting_constructor_adapter=False,
        _constructor_dense_lora_path=None,
        _apply_constructor_adapter=lambda: None,
        cur_adapter_name="first",
        cur_adapter_path=None,
        cur_adapter_strength=1.0,
        lora_initialized=True,
        lora_adapter_paths={},
        lora_adapters={
            "first": {"linear.lora_A": first[0], "linear.lora_B": first[1]},
            "second": {"linear.lora_A": second[0], "linear.lora_B": second[1]},
        },
        fastvideo_args=SimpleNamespace(training_mode=False),
    )
    return pipeline, wrapped, original, first, second


def _assert_cpu_packed_weight(wrapped, expected: torch.Tensor, retain: bool) -> None:
    for name, value in _cpu_packed_state(expected).items():
        torch.testing.assert_close(getattr(wrapped.base_layer, name), value, rtol=0, atol=0)
    assert (wrapped.base_layer.weight is not None) is retain


@pytest.mark.parametrize("retain", [False, True])
def test_nvfp4_cpu_backed_unmerge_remerge_rebuilds_every_buffer(cpu_nvfp4_lifecycle, retain: bool) -> None:
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    pipeline, wrapped, original, first, _ = _nvfp4_lifecycle_pipeline(retain)
    merged = original + first[1] @ first[0]
    _assert_cpu_packed_weight(wrapped, merged, retain)
    inputs = torch.randn(3, 32, dtype=torch.bfloat16)
    for _ in range(2):
        LoRAPipeline.unmerge_lora_weights(pipeline)
        _assert_cpu_packed_weight(wrapped, original, retain)
        assert not wrapped.merged
        if retain:
            assert not wrapped.base_layer.weight.requires_grad
        # Unmerge keeps the adapter active beside the freshly packed base.
        packed = _cpu_packed_state(original)
        expected = torch.nn.functional.linear(
            inputs, packed["_nvfp4_weight"].to(inputs) * packed["_nvfp4_weight_scale"].to(inputs))
        expected += inputs @ first[0].T @ first[1].T
        torch.testing.assert_close(wrapped(inputs)[0], expected, rtol=0, atol=0)
        LoRAPipeline.merge_lora_weights(pipeline)
        _assert_cpu_packed_weight(wrapped, merged, retain)
        torch.testing.assert_close(wrapped.cpu_weight, original, rtol=0, atol=0)


def test_nvfp4_restore_recreates_a_frozen_inference_parameter(cpu_nvfp4_lifecycle) -> None:
    _, wrapped, original, _, _ = _nvfp4_lifecycle_pipeline(False)
    wrapped.unmerge_lora_weights()

    assert not wrapped.base_layer.weight.requires_grad
    assert wrapped.base_layer.weight.data_ptr() != wrapped.cpu_weight.data_ptr()
    torch.testing.assert_close(wrapped.base_layer.weight, original, rtol=0, atol=0)


def test_lora_training_keeps_adapter_gradients_without_inference_restore(cpu_nvfp4_lifecycle,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared wrapper trains A/B while freezing the base and never restoring inference weights."""
    from fastvideo.layers.linear import ReplicatedLinear
    from fastvideo.layers.lora.linear import get_lora_layer

    torch.manual_seed(11)
    base = ReplicatedLinear(8, 6, bias=False, params_dtype=torch.float32)
    # Start with a trainable parameter so wrapper initialization must explicitly freeze it.
    base.weight.requires_grad_(True)
    wrapped = get_lora_layer(base, lora_rank=2, lora_alpha=2, training_mode=True)
    assert wrapped.cpu_weight is None

    def unexpected_restore():
        pytest.fail("LoRA training must not restore or merge inference weights")

    monkeypatch.setattr(wrapped, "_restore_base_weight", unexpected_restore)
    wrapped.set_lora_weights(torch.randn(2, 8), torch.randn(6, 2), training_mode=True)
    assert not wrapped.merged and not wrapped.base_layer.weight.requires_grad
    assert wrapped.lora_A.requires_grad and wrapped.lora_B.requires_grad
    original = wrapped.base_layer.weight.detach().clone()
    adapter_before = [wrapped.lora_A.detach().clone(), wrapped.lora_B.detach().clone()]
    optimizer = torch.optim.SGD([wrapped.lora_A, wrapped.lora_B], lr=0.01)
    inputs = torch.randn(3, 8, requires_grad=True)

    wrapped(inputs)[0].square().mean().backward()

    assert wrapped.base_layer.weight.grad is None
    assert inputs.grad is not None and inputs.grad.abs().sum() > 0
    for parameter in (wrapped.lora_A, wrapped.lora_B):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    optimizer.step()
    torch.testing.assert_close(wrapped.base_layer.weight, original, rtol=0, atol=0)
    for parameter, previous in zip((wrapped.lora_A, wrapped.lora_B), adapter_before, strict=True):
        assert not torch.equal(parameter, previous)


@pytest.mark.parametrize("retain", [False, True])
def test_nvfp4_cpu_backed_adapter_and_strength_changes_replace_old_delta(cpu_nvfp4_lifecycle, retain: bool) -> None:
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    pipeline, wrapped, original, first, second = _nvfp4_lifecycle_pipeline(retain)
    for nickname, strength, factors in [("second", 0.5, second), ("second", 1.5, second), ("first", 1.0, first)]:
        LoRAPipeline.set_lora_adapter(pipeline, nickname, strength=strength)
        _assert_cpu_packed_weight(wrapped, original + (factors[1] @ factors[0]) * strength, retain)
        torch.testing.assert_close(wrapped.cpu_weight, original, rtol=0, atol=0)
    LoRAPipeline.clear_lora_adapter(pipeline)
    _assert_cpu_packed_weight(wrapped, original, retain)
    assert wrapped.lora_A is None and wrapped.lora_B is None


def test_nvfp4_retained_bf16_accumulation_requantizes(cpu_nvfp4_lifecycle) -> None:
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    pipeline, wrapped, original, first, second = _nvfp4_lifecycle_pipeline(True)
    LoRAPipeline.set_lora_adapter(pipeline, "second", strength=0.5, accumulate=True)
    _assert_cpu_packed_weight(wrapped, (original + first[1] @ first[0]) + (second[1] @ second[0]) * 0.5, True)


@pytest.mark.parametrize("entrypoint", ["pipeline", "layer", "merge"])
def test_nvfp4_purged_accumulation_reconstructs_before_appending(cpu_nvfp4_lifecycle, entrypoint: str) -> None:
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    pipeline, wrapped, original, first, second = _nvfp4_lifecycle_pipeline(False)
    expected = original + first[1] @ first[0]
    if entrypoint == "pipeline":
        LoRAPipeline.set_lora_adapter(pipeline, "second", accumulate=True)
    elif entrypoint == "layer":
        wrapped.set_lora_weights(*second, accumulate=True)
        _cpu_pack_nvfp4(wrapped)
    else:
        wrapped.merge_lora_weights(accumulate=True)
        _cpu_pack_nvfp4(wrapped)
    added = first if entrypoint == "merge" else second
    _assert_cpu_packed_weight(wrapped, expected + added[1] @ added[0], False)
    assert len(wrapped._merged_loras) == 2
    torch.testing.assert_close(wrapped.cpu_weight, original, rtol=0, atol=0)


@pytest.mark.parametrize("retain", [False, True])
def test_disjoint_nvfp4_accumulation_then_clear(cpu_nvfp4_lifecycle, retain: bool) -> None:
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    pipeline, first_layer, original, first, second = _nvfp4_lifecycle_pipeline(retain)
    _, second_layer, _, _, _ = _nvfp4_lifecycle_pipeline(retain)
    model = nn.Module()
    model.first, model.second = first_layer, second_layer
    layers = {"first": first_layer, "second": second_layer}
    pipeline.trainable_transformer_modules = {"transformer": model}
    pipeline.lora_layers = {"transformer": SimpleNamespace(lora_layers_by_block=lambda: iter([(None, layers)]))}
    pipeline.lora_adapters = {
        "a": {"first.lora_A": first[0], "first.lora_B": first[1]},
        "b": {"second.lora_A": second[0], "second.lora_B": second[1]},
    }
    LoRAPipeline.clear_lora_adapter(pipeline)
    LoRAPipeline.set_lora_adapter(pipeline, "a")
    LoRAPipeline.set_lora_adapter(pipeline, "b", accumulate=True)
    _assert_cpu_packed_weight(first_layer, original + first[1] @ first[0], retain)
    _assert_cpu_packed_weight(second_layer, original + second[1] @ second[0], retain)
    LoRAPipeline.clear_lora_adapter(pipeline)
    for layer in layers.values():
        _assert_cpu_packed_weight(layer, original, retain)
        assert layer.lora_A is None and layer.lora_B is None
        assert layer._merged_loras in (None, [])


@pytest.mark.parametrize("missing", ["snapshot", "weight_device"])
def test_nvfp4_invalid_restore_fails_at_layer_boundary_without_mutation(cpu_nvfp4_lifecycle, missing: str) -> None:
    """Restoration checks its own required state rather than scanning unrelated pipeline modules."""
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline

    pipeline, wrapped, _, _, _ = _nvfp4_lifecycle_pipeline(False)
    if missing == "snapshot":
        wrapped.cpu_weight = None
    else:
        wrapped.base_layer.weight_device = None
    previous_a, previous_b = wrapped.lora_A, wrapped.lora_B
    with pytest.raises(RuntimeError, match="snapshot|weight_device"):
        LoRAPipeline.unmerge_lora_weights(pipeline)
    assert wrapped.base_layer.weight is None and wrapped.merged
    assert wrapped.lora_A is previous_a and wrapped.lora_B is previous_b


def test_nvfp4_restore_uses_placement_marker_without_reading_packed_values(cpu_nvfp4_lifecycle) -> None:
    _, wrapped, original, _, _ = _nvfp4_lifecycle_pipeline(False)
    # Payloads are not needed to recover the original full-precision snapshot.
    wrapped.base_layer._nvfp4_weight = None
    wrapped._restore_base_weight()
    torch.testing.assert_close(wrapped.base_layer.weight, original, rtol=0, atol=0)
    assert wrapped.base_layer.weight.device == wrapped.base_layer.weight_device.device
    assert wrapped.base_layer.weight_device.numel() == 0
    assert "weight_device" not in wrapped.base_layer.state_dict()


@pytest.mark.parametrize("requires_grad", [False, True])
def test_dtensor_lora_restore_preserves_parameter_gradient_flag(tmp_path, requires_grad: bool) -> None:
    """Replacing a real CPU DTensor parameter preserves freezing or trainability despite the no-grad scope."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
    from fastvideo.layers.linear import ReplicatedLinear
    from fastvideo.layers.lora.linear import get_lora_layer

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU DTensor restoration requires the Gloo backend")
    if dist.is_initialized():
        pytest.skip("This singleton CPU DTensor test must own its process group")
    dist.init_process_group("gloo", init_method=(tmp_path / "dtensor-store").as_uri(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1, ))
        original = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        base = ReplicatedLinear(4, 4, bias=False, params_dtype=torch.float32)
        base.weight = nn.Parameter(distribute_tensor(original.clone(), mesh, [Replicate()]),
                                   requires_grad=requires_grad)
        wrapped = get_lora_layer(base)
        with torch.no_grad():
            wrapped.base_layer.weight.add_(5)
        wrapped.merged = True

        wrapped.unmerge_lora_weights()

        restored = wrapped.base_layer.weight
        assert isinstance(restored, DTensor)
        assert restored.requires_grad is requires_grad
        assert not wrapped.merged
        torch.testing.assert_close(restored.to_local(), original, rtol=0, atol=0)
        torch.testing.assert_close(wrapped.cpu_weight.to_local(), original, rtol=0, atol=0)
        if requires_grad:
            restored.to_local().sum().backward()
            assert isinstance(restored.grad, DTensor)
            torch.testing.assert_close(restored.grad.to_local(), torch.ones_like(original), rtol=0, atol=0)
        else:
            assert restored.grad is None
    finally:
        dist.destroy_process_group()


def test_nvfp4_cpu_snapshot_restore_and_repack_stay_in_offload_scope(cpu_nvfp4_lifecycle, monkeypatch) -> None:
    from contextlib import contextmanager
    from fastvideo.pipelines import lora_pipeline

    pipeline, wrapped, _, _, _ = _nvfp4_lifecycle_pipeline(False)
    events = []

    @contextmanager
    def scope(module):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    restore = wrapped._restore_base_weight

    def restore_in_scope():
        assert events[-1] == "enter"
        events.append("restore")
        restore()

    def pack_in_scope(module):
        assert events[-1] == "restore"
        events.append("pack")
        _cpu_pack_nvfp4(module)

    monkeypatch.setattr(lora_pipeline, "_get_hook_ctx", scope)
    monkeypatch.setattr(wrapped, "_restore_base_weight", restore_in_scope)
    monkeypatch.setattr(nvfp4, "convert_model_to_nvfp4", pack_in_scope)
    lora_pipeline.LoRAPipeline.unmerge_lora_weights(pipeline)
    assert events == ["enter", "restore", "pack", "exit"]


class _NVFP4LifecycleTransformer(nn.Module):

    def __init__(self, feed_forward: nn.Module) -> None:
        super().__init__()
        # A trailing block keeps circular prefetch from targeting its own still-resident FFN.
        self.transformer_blocks = nn.ModuleList([nn.Sequential(feed_forward), nn.Identity()])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            inputs = block(inputs)
        return inputs


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("layerwise_offload", [False, True])
def test_nvfp4_lora_round_trip_matches_bf16_on_blackwell(retain: bool, layerwise_offload: bool) -> None:
    """Real NVFP4 FFN: preserve active LoRA through unmerge/remerge, with CPU restoration after purge."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 lifecycle numerical validation requires an NVIDIA Blackwell GPU")
    pytest.importorskip("flashinfer")
    from fastvideo.hooks.layerwise_offload import enable_layerwise_offload
    from fastvideo.layers.lora.linear import get_lora_layer
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3FeedForward
    from fastvideo.pipelines.lora_pipeline import (LoRAModelLayers, LoRAPipeline,
                                                   _convert_quantized_weights_after_lora_change, _get_hook_ctx)

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        dense = MiniMaxH3FeedForward(128, 256, prefix="minimax_h3.transformer_blocks.0.ff")
        quantized = MiniMaxH3FeedForward(
            128, 256,
            quant_config=nvfp4.NVFP4Config(retain_original_weights=retain),
            prefix="minimax_h3.transformer_blocks.0.ff")
    finally:
        torch.set_default_dtype(previous_dtype)
    generator = torch.Generator().manual_seed(20261105)
    with torch.no_grad():
        for parameter in dense.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.02)
    quantized.load_state_dict(dense.state_dict(), strict=True)
    dense = dense.cuda().eval()
    transformer = _NVFP4LifecycleTransformer(quantized).cuda().eval()
    block = transformer.transformer_blocks[0]
    if layerwise_offload:
        enable_layerwise_offload(transformer)
    factors = {
        "fc_in": (torch.randn(8, 128, generator=generator, dtype=torch.bfloat16).cuda() * 0.02,
                  torch.randn(512, 8, generator=generator, dtype=torch.bfloat16).cuda()),
        "fc_out": (torch.randn(8, 256, generator=generator, dtype=torch.bfloat16).cuda() * 0.02,
                   torch.randn(128, 8, generator=generator, dtype=torch.bfloat16).cuda()),
    }
    layers = LoRAModelLayers([("transformer_blocks.0", block)])
    with _get_hook_ctx(block):
        for name in ("fc_in", "fc_out"):
            wrapped = get_lora_layer(getattr(quantized, name))
            setattr(quantized, name, wrapped)
            wrapped.set_lora_weights(*factors[name], lora_alpha=8)
            layers.add_lora_layer("transformer_blocks.0", name, wrapped)
        _convert_quantized_weights_after_lora_change([block])
    inputs = torch.randn(2, 128, 128, generator=generator, dtype=torch.bfloat16).cuda()
    with torch.inference_mode():
        dense_base = dense(inputs)
    for name in ("fc_in", "fc_out"):
        wrapped = get_lora_layer(getattr(dense, name))
        setattr(dense, name, wrapped)
        wrapped.set_lora_weights(*factors[name], lora_alpha=8)
    with torch.inference_mode():
        dense_merged = dense(inputs)
    for name in ("fc_in", "fc_out"):
        getattr(dense, name).unmerge_lora_weights()
    with torch.inference_mode():
        dense_unmerged = dense(inputs)
    pipeline = SimpleNamespace(lora_layers={"transformer": layers},
                               trainable_transformer_modules={"transformer": transformer})

    def relative_error(actual, expected):
        return (actual.float() - expected.float()).norm() / expected.float().norm()

    with torch.inference_mode():
        assert relative_error(transformer(inputs), dense_merged) < 0.20
    LoRAPipeline.unmerge_lora_weights(pipeline)
    with torch.inference_mode():
        unmerged = transformer(inputs)
        assert relative_error(unmerged, dense_unmerged) < 0.20
        assert relative_error(unmerged, dense_unmerged) < relative_error(unmerged, dense_base)
    LoRAPipeline.merge_lora_weights(pipeline)
    with torch.inference_mode():
        assert relative_error(transformer(inputs), dense_merged) < 0.20

    accumulated_factors = {
        "fc_in": (torch.randn(4, 128, generator=generator, dtype=torch.bfloat16).cuda() * 0.02,
                  torch.randn(512, 4, generator=generator, dtype=torch.bfloat16).cuda()),
        "fc_out": (torch.randn(4, 256, generator=generator, dtype=torch.bfloat16).cuda() * 0.02,
                   torch.randn(128, 4, generator=generator, dtype=torch.bfloat16).cuda()),
    }
    for name in ("fc_in", "fc_out"):
        getattr(dense, name).merge_lora_weights()
        getattr(dense, name).set_lora_weights(*accumulated_factors[name], lora_alpha=4, strength=0.5, accumulate=True)
    with _get_hook_ctx(block):
        for name in ("fc_in", "fc_out"):
            getattr(quantized, name).set_lora_weights(*accumulated_factors[name], lora_alpha=4, strength=0.5,
                                                     accumulate=True)
        _convert_quantized_weights_after_lora_change([block])
    with torch.inference_mode():
        assert relative_error(transformer(inputs), dense(inputs)) < 0.20
    if not retain:
        assert len(quantized.fc_in._merged_loras) == 2
        assert len(quantized.fc_out._merged_loras) == 2

    for name in ("fc_in", "fc_out"):
        getattr(dense, name).set_lora_weights(*accumulated_factors[name], lora_alpha=4, strength=0.5)
    with _get_hook_ctx(block):
        for name in ("fc_in", "fc_out"):
            getattr(quantized, name).set_lora_weights(*accumulated_factors[name], lora_alpha=4, strength=0.5)
        _convert_quantized_weights_after_lora_change([block])
    with torch.inference_mode():
        assert relative_error(transformer(inputs), dense(inputs)) < 0.20
    assert (quantized.fc_in.base_layer.weight is not None) is retain
    assert (quantized.fc_out.base_layer.weight is not None) is retain
    if not retain:
        assert len(quantized.fc_in._merged_loras) == 1
        assert len(quantized.fc_out._merged_loras) == 1
        assert quantized.fc_in.base_layer.weight_device.device == inputs.device
        assert quantized.fc_out.base_layer.weight_device.device == inputs.device
