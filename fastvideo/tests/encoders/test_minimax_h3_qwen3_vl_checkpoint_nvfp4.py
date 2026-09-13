# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29515")

import fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 as h3_nvfp4
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import (
    MiniMaxH3Qwen3VLArchConfig,
    MiniMaxH3Qwen3VLConfig,
)
from fastvideo.layers.linear import ColumnParallelLinear, RowParallelLinear, UnquantizedLinearMethod
from fastvideo.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod, VocabParallelEmbedding
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.encoders.minimax_h3_checkpoint_fp8 import MiniMaxH3SerializedFP8Config
from fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 import (
    LANGUAGE_PROJECTIONS,
    NVFP4_TENSOR_SUFFIXES,
    MiniMaxH3SerializedNVFP4Config,
    MiniMaxH3SerializedNVFP4LinearMethod,
    serialized_nvfp4_quantization_config,
)
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner
from fastvideo.models.loader.text_encoder_quantization import (
    _configure_text_encoder_quantization,
    _process_quantized_text_encoder_weights,
    _read_text_encoder_checkpoint_quantization_config,
)

LAYER_PREFIX = "minimax_h3_qwen3_vl.language_model.layers.0"
E4M3_ONE = 0x38


def _checkpoint_quantization_config(**overrides) -> dict:
    config = serialized_nvfp4_quantization_config()
    config.update(overrides)
    return config


def _language_linear(config: MiniMaxH3SerializedNVFP4Config,
                     input_size: int = 128,
                     output_size: int = 256,
                     proj: str = "self_attn.q_proj",
                     bias: bool = False) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=bias,
        quant_config=config,
        prefix=f"{LAYER_PREFIX}.{proj}",
    )


def _fill_loaded(layer: torch.nn.Module, global_scale: float = 2.0) -> None:
    layer.weight_packed.data.fill_(0x11)
    layer.weight_scale.data.fill_(E4M3_ONE)
    layer.weight_global_scale.data.fill_(global_scale)


def test_h3_accepts_only_the_serialized_nvfp4_contract() -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    assert config.bf16_projections == ()
    assert config.get_name() == "nvfp4"
    assert config.get_supported_act_dtypes() == [torch.bfloat16]

    with pytest.raises(ValueError, match="group_size=16"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(group_size=32))
    with pytest.raises(ValueError, match="scale_layout='128x4'"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(scale_layout="linear"))
    with pytest.raises(ValueError, match="dynamic activation"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(activation_scheme="static"))
    with pytest.raises(ValueError, match="E2M1"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(fmt="e4m3"))
    with pytest.raises(ValueError, match="vision stack"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(modules_to_not_convert=["lm_head"]))
    with pytest.raises(ValueError, match="quant_method 'fp8'"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(quant_method="fp8"))


def test_h3_requires_every_layout_field_to_be_explicit() -> None:
    bare = {"quant_method": "nvfp4", "modules_to_not_convert": ["model.visual", "lm_head"]}
    with pytest.raises(ValueError, match="explicit quantization_config fields"):
        MiniMaxH3SerializedNVFP4Config.from_config(bare)


def test_h3_keeps_bf16_per_projection_kind_only() -> None:
    kept = MiniMaxH3SerializedNVFP4Config.from_config(
        _checkpoint_quantization_config(modules_to_not_convert=["model.visual", "lm_head", "mlp.down_proj"]))
    assert kept.bf16_projections == ("mlp.down_proj", )
    assert kept.is_kept_bf16(f"{LAYER_PREFIX}.mlp.down_proj")
    assert not kept.is_kept_bf16(f"{LAYER_PREFIX}.mlp.up_proj")

    with pytest.raises(ValueError, match="per projection kind only"):
        MiniMaxH3SerializedNVFP4Config.from_config(
            _checkpoint_quantization_config(modules_to_not_convert=["model.visual", "language_model.layers.3"]))
    with pytest.raises(ValueError, match="mlp.dwn_proj"):
        MiniMaxH3SerializedNVFP4Config.from_config(
            _checkpoint_quantization_config(modules_to_not_convert=["model.visual", "mlp.dwn_proj"]))
    with pytest.raises(ValueError, match="non-empty strings"):
        MiniMaxH3SerializedNVFP4Config.from_config(
            _checkpoint_quantization_config(modules_to_not_convert=["model.visual", ""]))


def test_converter_metadata_round_trips_and_tolerates_producer_notes() -> None:
    metadata = serialized_nvfp4_quantization_config(producer={"converter": "x.py", "flashinfer": "0.6.13rc2"})
    assert MiniMaxH3SerializedNVFP4Config.from_config(metadata).bf16_projections == ()
    assert metadata["producer"]["flashinfer"] == "0.6.13rc2"

    mixed = serialized_nvfp4_quantization_config(keep_bf16=("mlp.down_proj", ))
    assert mixed["modules_to_not_convert"] == ["model.visual", "lm_head", "mlp.down_proj"]
    assert MiniMaxH3SerializedNVFP4Config.from_config(mixed).bf16_projections == ("mlp.down_proj", )
    with pytest.raises(ValueError, match="unknown projection kinds"):
        serialized_nvfp4_quantization_config(keep_bf16=("mlp.dwn_proj", ))
    assert set(LANGUAGE_PROJECTIONS) == {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
        "mlp.up_proj", "mlp.down_proj"
    }


def test_serialized_nvfp4_allocates_packed_weight_and_scales_without_a_bf16_weight(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    layer = _language_linear(config)

    assert isinstance(layer.quant_method, MiniMaxH3SerializedNVFP4LinearMethod)
    assert layer.weight is None
    assert layer.weight_packed.dtype == torch.uint8
    assert layer.weight_packed.shape == (256, 64)
    assert layer.weight_scale.dtype == torch.uint8
    assert layer.weight_scale.shape == (256, 8)
    assert layer.weight_global_scale.dtype == torch.float32
    assert layer.weight_global_scale.shape == (1, )
    for name in ("weight_packed", "weight_scale", "weight_global_scale"):
        param = getattr(layer, name)
        assert callable(getattr(param, "weight_loader", None))
        assert not hasattr(param, "output_dim") and not hasattr(param, "input_dim")

    _fill_loaded(layer, global_scale=4.0)
    packed_pointer = layer.weight_packed.data_ptr()
    scale_pointer = layer.weight_scale.data_ptr()
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight_packed.data_ptr() == packed_pointer
    assert layer.weight_scale.data_ptr() == scale_pointer
    assert layer.weight is None
    assert layer._nvfp4_alpha.dtype == torch.float32
    assert layer._nvfp4_alpha.item() == pytest.approx(0.25)
    assert layer._nvfp4_x_global_scale.item() == 1.0


def test_serialized_nvfp4_finalization_rejects_tensors_the_checkpoint_never_filled(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    layer = _language_linear(config)

    with pytest.raises(ValueError, match="weight_global_scale was not loaded"):
        layer.quant_method.process_weights_after_loading(layer)

    layer.weight_global_scale.data.fill_(2.0)
    with pytest.raises(ValueError, match="weight_scale was not loaded"):
        layer.quant_method.process_weights_after_loading(layer)

    layer.weight_scale.data.fill_(E4M3_ONE)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer._nvfp4_alpha.item() == pytest.approx(0.5)


def test_serialized_nvfp4_tensors_arrive_through_the_layer_weight_loader(distributed_setup) -> None:
    """Drive the three tensors the way ``load_weights`` does, through the layer's own loader."""
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    column = _language_linear(config, proj="self_attn.q_proj")
    row = RowParallelLinear(
        input_size=128,
        output_size=256,
        bias=False,
        quant_config=config,
        prefix=f"{LAYER_PREFIX}.self_attn.o_proj",
    )
    for layer in (column, row):
        assert isinstance(layer.quant_method, MiniMaxH3SerializedNVFP4LinearMethod)
        packed = torch.full((256, 64), 0x22, dtype=torch.uint8)
        scale = torch.full((256, 8), E4M3_ONE, dtype=torch.uint8)
        layer.weight_packed.weight_loader(layer.weight_packed, packed)
        layer.weight_scale.weight_loader(layer.weight_scale, scale)
        layer.weight_global_scale.weight_loader(layer.weight_global_scale, torch.tensor(8.0))
        layer.quant_method.process_weights_after_loading(layer)
        assert torch.equal(layer.weight_packed.data, packed)
        assert layer._nvfp4_alpha.item() == pytest.approx(0.125)

    with pytest.raises(AssertionError):
        column.weight_packed.weight_loader(column.weight_packed, torch.zeros(256, 128, dtype=torch.uint8))
    # A float8 scale tensor would be value-cast into uint8 by a plain copy; the loader refuses it.
    with pytest.raises(ValueError, match="declared dtype"):
        column.weight_scale.weight_loader(column.weight_scale, torch.zeros(256, 8, dtype=torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="declared dtype"):
        column.weight_global_scale.weight_loader(column.weight_global_scale, torch.tensor(8.0, dtype=torch.float16))


def test_serialized_nvfp4_quantizes_only_language_linears(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    visual_linear = ColumnParallelLinear(
        input_size=128,
        output_size=128,
        bias=False,
        quant_config=config,
        prefix="minimax_h3_qwen3_vl.visual.blocks.0.attn.proj",
    )
    embedding = VocabParallelEmbedding(
        num_embeddings=128,
        embedding_dim=128,
        org_num_embeddings=128,
        quant_config=config,
        prefix="minimax_h3_qwen3_vl.language_model.embed_tokens",
    )

    assert isinstance(visual_linear.quant_method, UnquantizedLinearMethod)
    assert visual_linear.weight.dtype == torch.get_default_dtype()
    assert isinstance(embedding.quant_method, UnquantizedEmbeddingMethod)
    assert embedding.weight.dtype == torch.get_default_dtype()


def test_kept_bf16_projection_builds_a_plain_linear(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(
        _checkpoint_quantization_config(modules_to_not_convert=["model.visual", "lm_head", "mlp.down_proj"]))
    down = _language_linear(config, input_size=128, output_size=128, proj="mlp.down_proj")
    up = _language_linear(config, input_size=128, output_size=128, proj="mlp.up_proj")

    assert isinstance(down.quant_method, UnquantizedLinearMethod)
    assert down.weight is not None and down.weight.shape == (128, 128)
    assert not hasattr(down, "weight_packed")
    assert isinstance(up.quant_method, MiniMaxH3SerializedNVFP4LinearMethod)
    assert up.weight is None


def test_serialized_nvfp4_rejects_geometry_outside_flashinfer_tiles(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    with pytest.raises(ValueError, match="input_size divisible by 64"):
        _language_linear(config, input_size=96, output_size=256)
    with pytest.raises(ValueError, match="output_size divisible by 128"):
        _language_linear(config, input_size=128, output_size=200)


def test_serialized_nvfp4_cpu_execution_fails_closed(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    layer = _language_linear(config, input_size=128, output_size=128)
    _fill_loaded(layer)
    layer.quant_method.process_weights_after_loading(layer)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        layer(torch.zeros(2, 128, dtype=torch.bfloat16))


def test_runtime_preflight_gates_device_capability_parallelism_and_flashinfer(monkeypatch) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        config.validate_runtime(torch.device("cpu"))

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    with pytest.raises(RuntimeError, match="sm100 or newer"):
        config.validate_runtime(torch.device("cuda"))

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 1))
    monkeypatch.setattr(h3_nvfp4, "get_tp_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="single GPU"):
        config.validate_runtime(torch.device("cuda"))

    monkeypatch.setattr(h3_nvfp4, "get_tp_world_size", lambda: 1)

    def missing_flashinfer():
        raise ImportError("NVFP4 quantization requires flashinfer")

    monkeypatch.setattr(h3_nvfp4, "_require_flashinfer", missing_flashinfer)
    with pytest.raises(ImportError, match="requires flashinfer"):
        config.validate_runtime(torch.device("cuda"))

    class LayoutWithoutSwizzle:
        pass

    monkeypatch.setattr(h3_nvfp4, "_require_flashinfer", lambda: (LayoutWithoutSwizzle, None, None))
    with pytest.raises(RuntimeError, match="layout_128x4"):
        config.validate_runtime(torch.device("cuda"))

    class Layout:
        layout_128x4 = 1

    registered = []
    monkeypatch.setattr(h3_nvfp4, "_require_flashinfer", lambda: (Layout, None, None))
    monkeypatch.setattr(h3_nvfp4, "_register_ops_once", lambda: registered.append(True))
    config.validate_runtime(torch.device("cuda"))
    assert registered == [True]


def test_loader_selects_nvfp4_from_checkpoint_metadata(tmp_path) -> None:
    checkpoint_config = _checkpoint_quantization_config()
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": checkpoint_config}), encoding="utf-8")

    assert _read_text_encoder_checkpoint_quantization_config(str(tmp_path)) == checkpoint_config
    model_config = MiniMaxH3Qwen3VLConfig()
    quant_config = _configure_text_encoder_quantization(model_config, MiniMaxH3Qwen3VLConditioner, str(tmp_path))
    assert isinstance(quant_config, MiniMaxH3SerializedNVFP4Config)
    assert model_config.quant_config is quant_config

    with pytest.raises(ValueError, match="does not support serialized 'nvfp4'"):
        _configure_text_encoder_quantization(MiniMaxH3Qwen3VLConfig(), TextEncoder, str(tmp_path))


def test_conditioner_routes_each_serialized_scheme_to_its_config() -> None:
    fp8_metadata = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["model.visual", "lm_head"],
    }
    assert MiniMaxH3Qwen3VLConditioner.supported_checkpoint_quantization_methods == frozenset({"fp8", "nvfp4"})
    assert isinstance(MiniMaxH3Qwen3VLConditioner.checkpoint_quantization_config_from_metadata(fp8_metadata),
                      MiniMaxH3SerializedFP8Config)
    assert isinstance(
        MiniMaxH3Qwen3VLConditioner.checkpoint_quantization_config_from_metadata(_checkpoint_quantization_config()),
        MiniMaxH3SerializedNVFP4Config)
    with pytest.raises(ValueError, match="no serialized 'int4'"):
        MiniMaxH3Qwen3VLConditioner.checkpoint_quantization_config_from_metadata({"quant_method": "int4"})


def test_post_load_processing_visits_only_serialized_nvfp4_linears(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    quantized = _language_linear(config, input_size=128, output_size=128)
    plain = ColumnParallelLinear(input_size=128, output_size=128, bias=False, prefix="plain")
    _fill_loaded(quantized)
    model = torch.nn.ModuleList([quantized, plain])

    assert _process_quantized_text_encoder_weights(model, torch.device("cpu")) == 1
    assert quantized.weight_packed.device.type == "cpu"
    assert quantized._nvfp4_alpha.device.type == "cpu"
    assert plain.weight.device.type == "cpu"


def test_nvfp4_linear_hands_mm_fp4_transposed_operands_and_a_bf16_output(monkeypatch) -> None:
    x_fp4 = torch.zeros(4, 64, dtype=torch.uint8)
    x_scale = torch.zeros(128, 8, dtype=torch.uint8)
    weight_packed = torch.zeros(256, 64, dtype=torch.uint8)
    weight_scale = torch.zeros(256, 8, dtype=torch.uint8)
    alpha = torch.tensor(0.5, dtype=torch.float32)
    receipt: dict[str, object] = {}

    def fake_mm_fp4(a, b, a_scale, b_scale, alpha_arg, out_dtype, out, **kwargs):
        receipt.update(a=a, b=b, a_scale=a_scale, b_scale=b_scale, alpha=alpha_arg, out_dtype=out_dtype, out=out,
                       kwargs=kwargs)
        return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype)

    monkeypatch.setattr(h3_nvfp4, "_mm_fp4", fake_mm_fp4)
    output = h3_nvfp4._nvfp4_linear(x_fp4, x_scale, weight_packed, weight_scale, alpha)

    assert output.shape == (4, 256)
    assert output.dtype == torch.bfloat16
    assert receipt["a"] is x_fp4
    assert receipt["a_scale"] is x_scale
    assert receipt["b"].shape == (64, 256)
    assert receipt["b"].data_ptr() == weight_packed.data_ptr()
    assert receipt["b_scale"].shape == (8, 256)
    assert receipt["b_scale"].data_ptr() == weight_scale.data_ptr()
    assert receipt["alpha"] is alpha
    assert receipt["out_dtype"] == torch.bfloat16
    assert receipt["out"] is None
    assert receipt["kwargs"] == {"backend": "auto"}


def test_apply_flattens_quantizes_and_restores_the_leading_dims(distributed_setup, monkeypatch) -> None:
    """The finalized forward on a 3-D fp32 input: coerce to bf16, quantize the 2-D view with the
    unit global scale, GEMM with the layer's alpha, add the bias, restore [batch, seq, out]."""
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    layer = _language_linear(config, bias=True)
    _fill_loaded(layer, global_scale=4.0)
    layer.quant_method.process_weights_after_loading(layer)
    layer.bias.data.fill_(3.0)
    receipt: dict[str, object] = {}

    def fake_quantize(x_2d, global_scale):
        receipt.update(x_2d=x_2d, global_scale=global_scale)
        return torch.zeros(x_2d.shape[0], 64, dtype=torch.uint8), torch.zeros(128, 8, dtype=torch.uint8)

    def fake_linear(x_fp4, x_scale, weight_packed, weight_scale, alpha):
        receipt.update(alpha=alpha, weight_packed=weight_packed, weight_scale=weight_scale)
        return torch.zeros(x_fp4.shape[0], weight_packed.shape[0], dtype=torch.bfloat16)

    monkeypatch.setattr(h3_nvfp4, "_quantize_activation_nvfp4", fake_quantize)
    monkeypatch.setattr(h3_nvfp4, "_nvfp4_linear", fake_linear)
    output = MiniMaxH3SerializedNVFP4LinearMethod._apply_finalized(layer, torch.ones(1, 5, 128), layer.bias)

    assert receipt["x_2d"].shape == (5, 128)
    assert receipt["x_2d"].dtype == torch.bfloat16
    assert receipt["global_scale"] is layer._nvfp4_x_global_scale
    assert receipt["alpha"] is layer._nvfp4_alpha
    assert receipt["weight_packed"] is layer.weight_packed
    assert output.shape == (1, 5, 256)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, torch.full((1, 5, 256), 3.0, dtype=torch.bfloat16))


def _tiny_conditioner_config(keep_bf16: tuple[str, ...] = ("mlp.down_proj", )) -> MiniMaxH3Qwen3VLConfig:
    """One language layer whose linears satisfy the FlashInfer tile geometry, and a small vision tower."""
    config = MiniMaxH3Qwen3VLConfig()
    config.arch_config = MiniMaxH3Qwen3VLArchConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        output_hidden_state_index=1,
        num_hidden_layers_override=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        rope_scaling={
            "mrope_interleaved": True,
            "mrope_section": [32, 16, 16],
            "rope_type": "default",
        },
        vision_depth=1,
        vision_hidden_size=64,
        vision_intermediate_size=128,
        vision_num_heads=1,
        vision_deepstack_visual_indexes=(),
        vision_out_hidden_size=128,
    )
    config.quant_config = MiniMaxH3SerializedNVFP4Config.from_config(
        _checkpoint_quantization_config(modules_to_not_convert=["model.visual", "lm_head", *keep_bf16]))
    return config


def test_conditioner_loads_converter_named_tensors_end_to_end(distributed_setup) -> None:
    """The real chain: a conditioner built with the NVFP4 config, checkpoint keys spelled the way the
    converter writes them, ``load_weights``, the strict missing-tensor check, then the post-load hook."""
    model = MiniMaxH3Qwen3VLConditioner(_tiny_conditioner_config())
    layer = model.language_model.layers[0]
    assert isinstance(layer.self_attn.q_proj.quant_method, MiniMaxH3SerializedNVFP4LinearMethod)
    assert isinstance(layer.self_attn.o_proj.quant_method, MiniMaxH3SerializedNVFP4LinearMethod)
    assert isinstance(layer.mlp.down_proj.quant_method, UnquantizedLinearMethod)
    assert layer.self_attn.q_proj.weight is None and layer.mlp.down_proj.weight is not None

    packed_name, scale_name, global_scale_name = NVFP4_TENSOR_SUFFIXES
    checkpoint: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name.endswith("." + packed_name):
            tensor = torch.full_like(param, 0x11)
        elif name.endswith("." + scale_name):
            tensor = torch.full_like(param, E4M3_ONE)
        elif name.endswith("." + global_scale_name):
            tensor = torch.full_like(param, 2.0)
        else:
            tensor = torch.zeros_like(param)
        checkpoint["model." + name] = tensor
    expected = {name for name, _ in model.named_parameters()}
    assert sum(name.endswith("." + packed_name) for name in expected) == 6

    loaded = model.load_weights(iter(checkpoint.items()))
    assert loaded == expected
    assert _process_quantized_text_encoder_weights(model, torch.device("cpu")) == 6
    assert layer.self_attn.q_proj._nvfp4_alpha.item() == pytest.approx(0.5)
    assert layer.mlp.up_proj._nvfp4_alpha.item() == pytest.approx(0.5)

    # A tensor the checkpoint lacks is reported by name through the loaded set, which is what the
    # loader's strict check subtracts, and the hook then refuses the unfilled layer.
    missing = f"model.language_model.layers.0.mlp.up_proj.{scale_name}"
    layer.mlp.up_proj.weight_scale.data.fill_(0xFF)
    partial = {key: value for key, value in checkpoint.items() if key != missing}
    loaded = model.load_weights(iter(partial.items()))
    assert expected - loaded == {missing[len("model."):]}
    with pytest.raises(ValueError, match="weight_scale was not loaded"):
        _process_quantized_text_encoder_weights(model, torch.device("cpu"))

    wrong_dtype = dict(checkpoint)
    wrong_dtype[missing] = wrong_dtype[missing].to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="declared dtype"):
        model.load_weights(iter(wrong_dtype.items()))

    stray = dict(checkpoint)
    stray["model.language_model.layers.0.self_attn.q_proj.weight"] = torch.zeros(128, 128)
    with pytest.raises(ValueError, match="Unexpected"):
        model.load_weights(iter(stray.items()))
