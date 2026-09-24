# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the serialized int8 MiniMax-H3 transformer path.

Small shapes keep the int32 products exact, so the CPU reference and the CUDA
``torch._int_mm`` path compute the same numbers.
"""
from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29516")

from fastvideo.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from fastvideo.layers.quantization.minimax_h3_int8 import (
    DENSE_TRANSFORMER_LINEARS,
    GATE_LINEAR,
    INT8_TENSOR_SUFFIXES,
    TRANSFORMER_LINEARS,
    MiniMaxH3SerializedInt8Config,
    MiniMaxH3SerializedInt8LinearMethod,
    finalize_serialized_int8_model,
    has_serialized_int8_linears,
    int8_linear,
    is_minimax_h3_int8_linear_prefix,
    quantize_rows_int8,
    reject_lora_on_serialized_int8,
    serialized_int8_quantization_config,
    transformer_quantization_config_from_metadata,
    validate_int8_geometry,
)

BLOCK = "minimax_h3.transformer_blocks.3"


def _config() -> MiniMaxH3SerializedInt8Config:
    return MiniMaxH3SerializedInt8Config.from_config(serialized_int8_quantization_config())


def _block_linear(name: str = "attn.to_q", input_size: int = 64, output_size: int = 32) -> ReplicatedLinear:
    return ReplicatedLinear(input_size, output_size, bias=False, quant_config=_config(), prefix=f"{BLOCK}.{name}")


def _fill(layer: ReplicatedLinear, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_int8, weight_scale = quantize_rows_int8(weight)
    layer.weight.data = weight_int8
    layer.weight_scale.data = weight_scale
    layer.quant_method.process_weights_after_loading(layer)
    return weight_int8, weight_scale


def test_metadata_round_trip_and_names():
    config = serialized_int8_quantization_config(producer={"converter": "test"})
    assert config["quant_method"] == "int8"
    assert config["linears"] == list(TRANSFORMER_LINEARS)
    assert config["producer"] == {"converter": "test"}
    parsed = MiniMaxH3SerializedInt8Config.from_config(config)
    assert parsed.get_name() == "int8"
    assert INT8_TENSOR_SUFFIXES == ("weight", "weight_scale")


@pytest.mark.parametrize("missing", ["activation_scheme", "weight_granularity", "activation_granularity", "linears"])
def test_metadata_requires_every_field(missing: str):
    config = serialized_int8_quantization_config()
    config.pop(missing)
    with pytest.raises(ValueError, match="explicit quantization_config fields"):
        MiniMaxH3SerializedInt8Config.from_config(config)


def test_metadata_rejects_other_values():
    config = serialized_int8_quantization_config()
    config["weight_granularity"] = "tensor"
    with pytest.raises(ValueError, match="weight_granularity"):
        MiniMaxH3SerializedInt8Config.from_config(config)
    config = serialized_int8_quantization_config()
    config["linears"] = list(TRANSFORMER_LINEARS[:-1])
    with pytest.raises(ValueError, match="seven block linears"):
        MiniMaxH3SerializedInt8Config.from_config(config)
    with pytest.raises(ValueError, match="quant_method"):
        MiniMaxH3SerializedInt8Config.from_config({"quant_method": "fp8"})


def test_metadata_dispatch():
    assert transformer_quantization_config_from_metadata(None) is None
    assert transformer_quantization_config_from_metadata({}) is None
    parsed = transformer_quantization_config_from_metadata(serialized_int8_quantization_config())
    assert isinstance(parsed, MiniMaxH3SerializedInt8Config)
    with pytest.raises(ValueError, match="Unsupported serialized transformer quantization"):
        transformer_quantization_config_from_metadata({"quant_method": "nvfp4"})
    with pytest.raises(ValueError, match="mapping"):
        transformer_quantization_config_from_metadata("int8")


def test_dense_metadata_leaves_the_gate_unquantized():
    """A dense checkpoint lists six linears, so the gate the VSA backend adds stays a float layer."""
    assert GATE_LINEAR in TRANSFORMER_LINEARS and GATE_LINEAR not in DENSE_TRANSFORMER_LINEARS
    assert len(DENSE_TRANSFORMER_LINEARS) == 6
    metadata = serialized_int8_quantization_config(linears=DENSE_TRANSFORMER_LINEARS)
    assert metadata["linears"] == list(DENSE_TRANSFORMER_LINEARS)
    config = MiniMaxH3SerializedInt8Config.from_config(metadata)
    gate = ReplicatedLinear(64, 32, bias=False, quant_config=config, prefix=f"{BLOCK}.{GATE_LINEAR}")
    assert isinstance(gate.quant_method, UnquantizedLinearMethod)
    assert gate.weight.dtype != torch.int8
    query = ReplicatedLinear(64, 32, bias=False, quant_config=config, prefix=f"{BLOCK}.attn.to_q")
    assert isinstance(query.quant_method, MiniMaxH3SerializedInt8LinearMethod)
    for declared in (["attn.to_q"], list(reversed(TRANSFORMER_LINEARS))):
        with pytest.raises(ValueError, match="seven block linears"):
            serialized_int8_quantization_config(linears=declared)


@pytest.mark.parametrize("prefix, quantized", [
    (f"{BLOCK}.attn.to_q", True),
    (f"{BLOCK}.attn.to_out", True),
    (f"{BLOCK}.attn.to_gate_compress", True),
    (f"{BLOCK}.ff.fc_in", True),
    (f"{BLOCK}.ff.fc_out", True),
    ("transformer_blocks.49.ff.fc_out", True),
    (f"{BLOCK}.adaln_proj.linear", False),
    ("minimax_h3.token_refiner.refiner_blocks.0.attn.to_q", False),
    ("minimax_h3.token_refiner.refiner_blocks.1.ff.fc_in", False),
    ("minimax_h3.context_embedder", False),
    ("minimax_h3.time_embedder.fc_in", False),
    ("minimax_h3.transformer_blocks.3.attn.to_q.extra", False),
])
def test_prefix_routing(prefix: str, quantized: bool):
    assert is_minimax_h3_int8_linear_prefix(prefix) is quantized
    layer = ReplicatedLinear(64, 32, bias=False, quant_config=_config(), prefix=prefix)
    if quantized:
        assert isinstance(layer.quant_method, MiniMaxH3SerializedInt8LinearMethod)
    else:
        assert isinstance(layer.quant_method, UnquantizedLinearMethod)
        assert layer.weight.dtype != torch.int8


def test_create_weights_allocates_int8_weight_and_row_scale():
    layer = _block_linear()
    assert layer.weight.dtype == torch.int8
    assert tuple(layer.weight.shape) == (32, 64)
    assert layer.weight_scale.dtype == torch.float32
    assert tuple(layer.weight_scale.shape) == (32, )
    assert not layer.weight.requires_grad and not layer.weight_scale.requires_grad
    assert layer._int8_finalized is False
    assert not hasattr(layer.weight, "output_dim")


def test_geometry_outside_int8_tiles_is_rejected():
    with pytest.raises(ValueError, match="divisible by 8"):
        validate_int8_geometry(60, 64)
    with pytest.raises(ValueError, match="divisible by 8"):
        _block_linear(input_size=60)


def test_quantize_rows_int8_round_trip():
    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    weight_int8, weight_scale = quantize_rows_int8(weight)
    assert weight_int8.dtype == torch.int8 and weight_scale.dtype == torch.float32
    assert tuple(weight_scale.shape) == (32, )
    assert bool((weight_scale > 0).all())
    assert int(weight_int8.abs().max()) == 127
    dequantized = weight_int8.float() * weight_scale.unsqueeze(1)
    relative = (dequantized - weight.float()).norm() / weight.float().norm()
    assert relative < 0.02
    with pytest.raises(ValueError, match="2-D"):
        quantize_rows_int8(weight.unsqueeze(0))


def test_int8_linear_matches_bf16_product_and_chunking_is_invisible():
    torch.manual_seed(0)
    x = torch.randn(48, 64, dtype=torch.bfloat16)
    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    weight_int8, weight_scale = quantize_rows_int8(weight)
    reference = x.float() @ weight.float().t()
    output = int8_linear(x, weight_int8, weight_scale, out_dtype=torch.float32)
    assert tuple(output.shape) == (48, 32)
    relative = (output - reference).norm() / reference.norm()
    assert relative < 0.03
    chunked = int8_linear(x, weight_int8, weight_scale, out_dtype=torch.float32, row_chunk=16)
    assert torch.equal(output, chunked)
    with pytest.raises(ValueError, match="int8 weight"):
        int8_linear(x, weight, weight_scale)
    with pytest.raises(ValueError, match="width"):
        int8_linear(x[:, :32], weight_int8, weight_scale)


def test_apply_needs_finalization_and_runs_on_3d_input():
    layer = _block_linear()
    x = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not finalized"):
        layer(x)
    with pytest.raises(ValueError, match="finite and positive"):
        layer.quant_method.process_weights_after_loading(layer)
    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    weight_int8, weight_scale = _fill(layer, weight)
    assert layer._int8_finalized is True
    output, bias = layer(x)
    assert bias is None
    assert output.dtype == torch.bfloat16
    assert tuple(output.shape) == (2, 5, 32)
    expected = int8_linear(x.reshape(-1, 64), weight_int8, weight_scale).view(2, 5, 32)
    assert torch.equal(output, expected)
    empty, _ = layer(torch.empty(0, 64, dtype=torch.bfloat16))
    assert tuple(empty.shape) == (0, 32)


def test_finalization_rejects_wrong_dtypes_and_shapes():
    layer = _block_linear()
    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    weight_int8, weight_scale = quantize_rows_int8(weight)
    layer.weight.data = weight_int8
    layer.weight_scale.data = weight_scale.to(torch.bfloat16)
    with pytest.raises(ValueError, match="float32"):
        layer.quant_method.process_weights_after_loading(layer)
    layer.weight_scale.data = weight_scale[:16]
    with pytest.raises(ValueError, match="does not match"):
        layer.quant_method.process_weights_after_loading(layer)
    layer.weight_scale.data = weight_scale
    layer.weight.data = weight
    with pytest.raises(ValueError, match="must be int8"):
        layer.quant_method.process_weights_after_loading(layer)


def test_strict_loader_refuses_other_dtypes():
    layer = _block_linear()
    with pytest.raises(ValueError, match="declared dtype"):
        layer.weight.weight_loader(layer.weight, torch.zeros(32, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="declared dtype"):
        layer.weight_scale.weight_loader(layer.weight_scale, torch.ones(32, dtype=torch.bfloat16))
    layer.weight.weight_loader(layer.weight, torch.full((32, 64), 3, dtype=torch.int8))
    layer.weight_scale.weight_loader(layer.weight_scale, torch.full((32, ), 0.5))
    assert int(layer.weight[0, 0]) == 3 and float(layer.weight_scale[0]) == 0.5


def test_finalize_model_visits_only_int8_linears():
    model = torch.nn.Module()
    model.q = _block_linear("attn.to_q")
    model.k = _block_linear("attn.to_k")
    model.refiner = ReplicatedLinear(64, 32, bias=False, quant_config=_config(),
                                     prefix="minimax_h3.token_refiner.refiner_blocks.0.attn.to_q")
    for layer in (model.q, model.k):
        weight_int8, weight_scale = quantize_rows_int8(torch.randn(32, 64, dtype=torch.bfloat16))
        layer.weight.data = weight_int8
        layer.weight_scale.data = weight_scale
    assert finalize_serialized_int8_model(model) == 2
    assert model.q._int8_finalized and model.k._int8_finalized
    assert not hasattr(model.refiner, "_int8_finalized")


class _TinyTransformer(torch.nn.Module):
    """One quantized block linear next to one refiner linear, with H3's dtype selector."""

    def __init__(self) -> None:
        super().__init__()
        self.q = _block_linear("attn.to_q")
        self.refiner = ReplicatedLinear(64, 32, bias=False, quant_config=_config(),
                                        prefix="minimax_h3.token_refiner.refiner_blocks.0.attn.to_q")

    def _get_parameter_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        if name.endswith(".weight_scale"):
            return torch.float32
        return default_dtype


def _load(model: torch.nn.Module, tensors: dict[str, torch.Tensor], dense_lora_patch=None):
    from fastvideo.models.loader.fsdp_load import load_model_from_full_model_state_dict
    from fastvideo.models.loader.utils import get_param_names_mapping
    return load_model_from_full_model_state_dict(
        model,
        iter(tensors.items()),
        torch.device("cpu"),
        torch.bfloat16,
        strict=True,
        param_names_mapping=get_param_names_mapping({}),
        dense_lora_patch=dense_lora_patch,
    )


def test_production_loader_keeps_int8_and_row_scales():
    """The transformer loader assigns checkpoint tensors straight into the parameters."""
    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    weight_int8, weight_scale = quantize_rows_int8(weight)
    refiner_weight = torch.randn(32, 64, dtype=torch.float32)
    model = _TinyTransformer()
    result = _load(model, {"q.weight": weight_int8, "q.weight_scale": weight_scale, "refiner.weight": refiner_weight})
    assert not result.missing_keys and not result.unexpected_keys
    assert model.q.weight.dtype == torch.int8 and torch.equal(model.q.weight, weight_int8)
    assert not model.q.weight.requires_grad and not model.q.weight_scale.requires_grad
    assert model.q.weight_scale.dtype == torch.float32 and torch.equal(model.q.weight_scale, weight_scale)
    assert model.refiner.weight.dtype == torch.bfloat16
    assert finalize_serialized_int8_model(model) == 1
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    output, _ = model.q(x)
    assert torch.equal(output, int8_linear(x, weight_int8, weight_scale))


def _tensors_without_gate() -> dict[str, torch.Tensor]:
    weight_int8, weight_scale = quantize_rows_int8(torch.randn(32, 64, dtype=torch.bfloat16))
    return {"q.weight": weight_int8, "q.weight_scale": weight_scale,
            "refiner.weight": torch.randn(32, 64, dtype=torch.float32)}


def test_production_loader_zero_fills_the_float_gate_of_a_dense_checkpoint():
    """Under VSA a dense int8 checkpoint gets the same all-zero float gate a bf16 one gets."""
    dense_config = MiniMaxH3SerializedInt8Config.from_config(
        serialized_int8_quantization_config(linears=DENSE_TRANSFORMER_LINEARS))
    model = _TinyTransformer()
    model.to_gate_compress = ReplicatedLinear(64, 32, bias=False, quant_config=dense_config,
                                              prefix=f"{BLOCK}.{GATE_LINEAR}")
    result = _load(model, _tensors_without_gate())
    assert not result.missing_keys and not result.unexpected_keys
    assert model.to_gate_compress.weight.dtype == torch.bfloat16
    assert not model.to_gate_compress.weight.any()
    assert finalize_serialized_int8_model(model) == 1

    # Declaring all seven without shipping the gate builds an int8 gate the loader
    # cannot fill, and finalization says so instead of running with it.
    seven = _TinyTransformer()
    seven.to_gate_compress = ReplicatedLinear(64, 32, bias=False, quant_config=_config(),
                                              prefix=f"{BLOCK}.{GATE_LINEAR}")
    _load(seven, _tensors_without_gate())
    with pytest.raises(ValueError, match="must be int8"):
        finalize_serialized_int8_model(seven)


def test_lora_is_refused_on_serialized_int8_and_allowed_elsewhere():
    """Both merge paths assume a float base weight, so an adapter has to fail loudly."""
    model = _TinyTransformer()
    assert has_serialized_int8_linears(model) is True
    with pytest.raises(NotImplementedError, match="cannot be applied to a serialized int8"):
        reject_lora_on_serialized_int8(model, "/tmp/some-adapter")

    dense = torch.nn.Module()
    dense.refiner = ReplicatedLinear(64, 32, bias=False, quant_config=_config(),
                                     prefix="minimax_h3.token_refiner.refiner_blocks.0.attn.to_q")
    assert has_serialized_int8_linears(dense) is False
    reject_lora_on_serialized_int8(dense, "/tmp/some-adapter")


def test_production_loader_refuses_a_dense_lora_patch_next_to_int8_weights():
    """The loader's own check, for a caller that reaches it without the adapter refusal."""
    from fastvideo.models.loader.lora_patch import DenseLoRAPatch
    with pytest.raises(NotImplementedError, match="serialized int8 weights"):
        _load(_TinyTransformer(), _tensors_without_gate(), dense_lora_patch=DenseLoRAPatch([], {}, {}))


def test_production_loader_refuses_float_shards_for_int8_weights():
    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    _, weight_scale = quantize_rows_int8(weight)
    with pytest.raises(ValueError, match="quantization_config disagree"):
        _load(_TinyTransformer(), {
            "q.weight": weight,
            "q.weight_scale": weight_scale,
            "refiner.weight": torch.randn(32, 64, dtype=torch.float32),
        })
