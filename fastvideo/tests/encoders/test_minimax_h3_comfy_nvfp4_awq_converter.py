# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Comfy MiniMax-H3 NVFP4-AWQ converter."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "checkpoint_conversion" / \
    "convert_minimax_h3_comfy_nvfp4_awq.py"


@pytest.fixture(scope="module")
def converter():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("convert_minimax_h3_comfy_nvfp4_awq", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(SCRIPT.parent))


# The documented Comfy artifact's projection geometry as (output_size,
# input_size), written out independently of converter.PROJECTION_SHAPES so the
# suite checks the converter against the artifact and the model instead of
# against itself. q_proj and o_proj are 8192 wide because
# MiniMaxH3Qwen3VLTextAttention builds them with output_size/input_size =
# num_attention_heads * head_dim = 64 * 128 (fastvideo/models/encoders/
# minimax_h3_qwen3_vl.py); the packed uint8 weights are (8192, 2560) and
# (5120, 4096).
PROJECTION_GEOMETRY = {
    "self_attn.q_proj": (8192, 5120),
    "self_attn.k_proj": (1024, 5120),
    "self_attn.v_proj": (1024, 5120),
    "self_attn.o_proj": (5120, 8192),
    "mlp.gate_proj": (25600, 5120),
    "mlp.up_proj": (25600, 5120),
    "mlp.down_proj": (5120, 25600),
}


def test_projection_shapes_match_the_model_contract_and_the_artifact(converter) -> None:
    arch = converter.ARCH
    attention_width = arch.num_attention_heads * arch.head_dim  # q_proj out, o_proj in
    kv_width = arch.num_key_value_heads * arch.head_dim
    assert (attention_width, kv_width, arch.hidden_size, arch.intermediate_size) == (8192, 1024, 5120, 25600)
    assert converter.PROJECTION_SHAPES == {
        "self_attn.q_proj": (attention_width, arch.hidden_size),
        "self_attn.k_proj": (kv_width, arch.hidden_size),
        "self_attn.v_proj": (kv_width, arch.hidden_size),
        "self_attn.o_proj": (arch.hidden_size, attention_width),
        "mlp.gate_proj": (arch.intermediate_size, arch.hidden_size),
        "mlp.up_proj": (arch.intermediate_size, arch.hidden_size),
        "mlp.down_proj": (arch.hidden_size, arch.intermediate_size),
    }
    assert converter.PROJECTION_SHAPES == PROJECTION_GEOMETRY


def test_fastvideo_name_maps_comfy_qwen3_vl_names(converter) -> None:
    assert converter.fastvideo_name("model.layers.3.self_attn.q_proj.weight") == \
        "model.language_model.layers.3.self_attn.q_proj.weight"
    assert converter.fastvideo_name("model.embed_tokens.weight") == \
        "model.language_model.embed_tokens.weight"
    assert converter.fastvideo_name("visual.blocks.0.norm1.weight") == "model.visual.blocks.0.norm1.weight"
    # The final norm must land on the one spelling the loader accepts as an
    # omitted key; 'model.norm.weight' itself is rejected by load_weights.
    assert converter.fastvideo_name("model.norm.weight") == "model.language_model.norm.weight"
    assert converter.fastvideo_name("lm_head.weight") == "lm_head.weight"


def test_decode_marker_requires_json_with_a_format(converter) -> None:
    marker = torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)
    assert converter.decode_marker(marker, "marker") == {"format": "nvfp4"}
    with pytest.raises(ValueError, match="Invalid Comfy quantization marker"):
        converter.decode_marker(torch.tensor([0xff], dtype=torch.uint8), "bad")
    with pytest.raises(ValueError, match="has no string format"):
        converter.decode_marker(torch.tensor(list(json.dumps({"other": 1}).encode()), dtype=torch.uint8), "bad")


def test_quantized_tensor_conversion_preserves_values_and_changes_contract(converter) -> None:
    name, packed = converter.convert_quantized_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        torch.tensor([[0x12, 0xA5]], dtype=torch.uint8),
    )
    assert name == "model.language_model.layers.0.self_attn.q_proj.weight_packed"
    assert torch.equal(packed, torch.tensor([[0x21, 0x5A]], dtype=torch.uint8))

    scale = torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn)
    name, scale_bytes = converter.convert_quantized_tensor(
        "model.layers.0.self_attn.q_proj.weight_scale", scale)
    assert name == "model.language_model.layers.0.self_attn.q_proj.weight_scale"
    assert scale_bytes.dtype == torch.uint8
    assert torch.equal(scale_bytes, scale.view(torch.uint8))

    name, global_scale = converter.convert_quantized_tensor(
        "model.layers.0.self_attn.q_proj.weight_scale_2", torch.tensor(0.25))
    assert name == "model.language_model.layers.0.self_attn.q_proj.weight_global_scale"
    assert global_scale.shape == (1, )
    assert global_scale.item() == 4.0


def test_embedding_dequantization_is_rowwise_bf16(converter) -> None:
    weight = torch.tensor([[1, -2], [3, 4]], dtype=torch.int8)
    scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    output = converter.dequantize_embedding(weight, scale, rows=1)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, torch.tensor([[0.5, -1.0], [0.75, 1.0]], dtype=torch.bfloat16))


def test_invalid_scale_contract_fails_before_writing(converter) -> None:
    with pytest.raises(ValueError, match="finite positive FP32 scalar"):
        converter.convert_quantized_tensor(
            "model.layers.0.self_attn.q_proj.weight_scale_2", torch.tensor(0.0))
    with pytest.raises(ValueError, match="E4M3 block scales"):
        converter.convert_quantized_tensor(
            "model.layers.0.self_attn.q_proj.weight_scale", torch.ones(2, dtype=torch.float32))


def test_source_inspection_requires_the_complete_50_layer_contract(converter) -> None:
    nvfp4 = torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)
    int8 = torch.tensor(list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8)
    class TensorSpec:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    tensors = {
        "model.embed_tokens.comfy_quant": int8,
        "model.embed_tokens.weight": TensorSpec((converter.ARCH.vocab_size, converter.ARCH.hidden_size), torch.int8),
        "model.embed_tokens.weight_scale": TensorSpec((converter.ARCH.vocab_size, 1), torch.float32),
    }
    for layer in range(converter.EXPECTED_LAYERS):
        for projection in converter.LANGUAGE_PROJECTIONS:
            prefix = f"model.layers.{layer}.{projection}"
            tensors[prefix + ".comfy_quant"] = nvfp4
            output_size, input_size = PROJECTION_GEOMETRY[projection]
            tensors[prefix + ".weight"] = TensorSpec((output_size, input_size // 2), torch.uint8)
            tensors[prefix + ".weight_scale"] = TensorSpec((output_size, input_size // 16), torch.float8_e4m3fn)
            tensors[prefix + ".weight_scale_2"] = TensorSpec((), torch.float32)
    tensors["model.layers.0.self_attn.o_proj.pre_quant_scale"] = \
        TensorSpec((PROJECTION_GEOMETRY["self_attn.o_proj"][1], ), torch.bfloat16)

    class Handle:
        def keys(self):
            return tensors.keys()

        def get_tensor(self, name):
            return tensors[name]

    quantized, pre_scaled = converter.inspect_source(Handle())
    assert len(quantized) == 350
    assert pre_scaled == {"model.layers.0.self_attn.o_proj"}

    q_weight = tensors["model.layers.0.self_attn.q_proj.weight"]
    tensors["model.layers.0.self_attn.q_proj.weight"] = TensorSpec((1, 1), torch.uint8)
    with pytest.raises(ValueError, match="Unexpected model.layers.0.self_attn.q_proj.weight shape or dtype"):
        converter.inspect_source(Handle())
    tensors["model.layers.0.self_attn.q_proj.weight"] = q_weight

    del tensors["model.layers.49.mlp.down_proj.comfy_quant"]
    with pytest.raises(ValueError, match="Expected 350 H3 language linears"):
        converter.inspect_source(Handle())


def test_staging_destination_rejects_nonempty_output(converter, tmp_path) -> None:
    output = tmp_path / "text_encoder"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep")
    with pytest.raises(SystemExit, match="not an empty directory"):
        converter.staging_destination(output)
    assert (output / "unrelated.txt").read_text() == "keep"


def test_source_inspection_rejects_keys_without_a_loader_destination(converter) -> None:
    class Handle:

        def keys(self):
            return ["model.layers.0.self_attn.q_proj.bias"]

        def get_tensor(self, name):
            raise AssertionError("inspection must reject the key before reading tensors")

    with pytest.raises(ValueError, match="has no FastVideo loader destination"):
        converter.inspect_source(Handle())


def _comfy_source_name(parameter_name: str) -> str:
    """Comfy spelling of a FastVideo-native parameter name.

    Stated here independently of converter.fastvideo_name so the round trip
    below checks the converter against the loader's contract instead of
    against itself.
    """
    if parameter_name.startswith("language_model.layers."):
        return "model.layers." + parameter_name[len("language_model.layers."):]
    if parameter_name.startswith("language_model.embed_tokens."):
        return "model.embed_tokens." + parameter_name[len("language_model.embed_tokens."):]
    if parameter_name.startswith("visual."):
        return parameter_name
    raise AssertionError(f"unexpected parameter name {parameter_name}")


def test_converted_names_load_through_the_conditioner(converter, distributed_setup) -> None:
    """AGENTS.md step 5: every key the converter writes must reach the loader
    as a parameter, the skipped lm_head.weight, or the omitted final norm."""
    from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import (
        MiniMaxH3Qwen3VLArchConfig,
        MiniMaxH3Qwen3VLConfig,
    )
    from fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 import (
        MiniMaxH3SerializedNVFP4Config,
        serialized_nvfp4_quantization_config,
    )
    from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner

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
        serialized_nvfp4_quantization_config(pre_quant_scale=True))
    model = MiniMaxH3Qwen3VLConditioner(config)

    source_suffix = {
        "weight_packed": "weight",
        "weight_scale": "weight_scale",
        "weight_global_scale": "weight_scale_2",
        "pre_quant_scale": "pre_quant_scale",
    }
    checkpoint: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        stem, _, suffix = name.rpartition(".")
        if any(stem.endswith(projection) for projection in converter.LANGUAGE_PROJECTIONS):
            source = _comfy_source_name(stem) + "." + source_suffix[suffix]
            if suffix == "weight_packed":
                converted, tensor = converter.convert_quantized_tensor(
                    source, torch.zeros(param.shape, dtype=torch.uint8))
            elif suffix == "weight_scale":
                converted, tensor = converter.convert_quantized_tensor(
                    source, torch.zeros(param.shape, dtype=torch.float8_e4m3fn))
            elif suffix == "weight_global_scale":
                converted, tensor = converter.convert_quantized_tensor(
                    source, torch.ones((), dtype=torch.float32))
            else:
                converted, tensor = converter.fastvideo_name(source), torch.zeros_like(param)
        else:
            converted = converter.fastvideo_name(_comfy_source_name(name))
            tensor = torch.zeros_like(param)
        assert converted == "model." + name, (converted, name)
        checkpoint[converted] = tensor

    # The truncated build's final norm must use the one spelling the loader
    # accepts as an omitted key.
    final_norm = converter.fastvideo_name("model.norm.weight")
    assert final_norm == "model.language_model.norm.weight"
    checkpoint[final_norm] = torch.zeros(128, dtype=torch.bfloat16)

    loaded = model.load_weights(iter(checkpoint.items()))
    assert loaded == {name for name, _ in model.named_parameters()}
