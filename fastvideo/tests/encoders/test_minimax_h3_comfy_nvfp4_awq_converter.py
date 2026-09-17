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


def test_fastvideo_name_maps_comfy_qwen3_vl_names(converter) -> None:
    assert converter.fastvideo_name("model.layers.3.self_attn.q_proj.weight") == \
        "model.language_model.layers.3.self_attn.q_proj.weight"
    assert converter.fastvideo_name("model.embed_tokens.weight") == \
        "model.language_model.embed_tokens.weight"
    assert converter.fastvideo_name("visual.blocks.0.norm1.weight") == "model.visual.blocks.0.norm1.weight"
    assert converter.fastvideo_name("model.norm.weight") == "model.norm.weight"


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
    tensors = {
        "model.embed_tokens.comfy_quant": int8,
        "model.embed_tokens.weight": torch.ones(2, 2, dtype=torch.int8),
        "model.embed_tokens.weight_scale": torch.ones(2, 1),
    }
    for layer in range(converter.EXPECTED_LAYERS):
        for projection in converter.LANGUAGE_PROJECTIONS:
            prefix = f"model.layers.{layer}.{projection}"
            tensors[prefix + ".comfy_quant"] = nvfp4
            tensors[prefix + ".weight"] = torch.ones(1, 1, dtype=torch.uint8)
            tensors[prefix + ".weight_scale"] = torch.ones(1, 1, dtype=torch.float8_e4m3fn)
            tensors[prefix + ".weight_scale_2"] = torch.ones(())
    tensors["model.layers.0.self_attn.o_proj.pre_quant_scale"] = torch.ones(2, dtype=torch.bfloat16)

    class Handle:
        def keys(self):
            return tensors.keys()

        def get_tensor(self, name):
            return tensors[name]

    quantized, pre_scaled = converter.inspect_source(Handle())
    assert len(quantized) == 350
    assert pre_scaled == {"model.layers.0.self_attn.o_proj"}

    del tensors["model.layers.49.mlp.down_proj.comfy_quant"]
    with pytest.raises(ValueError, match="Expected 350 H3 language linears"):
        converter.inspect_source(Handle())
