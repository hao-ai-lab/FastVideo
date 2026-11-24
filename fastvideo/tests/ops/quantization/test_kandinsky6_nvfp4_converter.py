# SPDX-License-Identifier: Apache-2.0
"""CPU smoke test for the Kandinsky 6 packed checkpoint format."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

import fastvideo.layers.quantization.nvfp4_config as nvfp4
from fastvideo.layers.linear import ReplicatedLinear

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/checkpoint_conversion/convert_kandinsky6_nvfp4.py"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
SOURCE_WEIGHT = "visual_transformer_blocks.0.video_dec_block.feed_forward.net.0.proj.weight"
DENSE_BIAS = "audio_out_layer.modulation.out_layer.bias"
PACKED_PREFIX = "visual_transformer_blocks.0.videoT.feed_forward.mlp.fc_in"


def test_indexed_kandinsky6_checkpoint_converts_and_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("convert_kandinsky6_nvfp4", SCRIPT)
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    source = tmp_path / "source"
    transformer = source / "transformer"
    transformer.mkdir(parents=True)
    (source / "model_index.json").write_text(json.dumps({"_class_name": "Kandinsky6TI2VAPipeline"}))
    config = {"_class_name": "Kandinsky6Transformer3DModel"}
    (transformer / "config.json").write_text(json.dumps(config))

    weight = torch.full((8, 16), 2, dtype=torch.bfloat16)
    bias = torch.arange(8, dtype=torch.float32)
    first = "diffusion_pytorch_model-00001-of-00002.safetensors"
    second = "diffusion_pytorch_model-00002-of-00002.safetensors"
    save_file({SOURCE_WEIGHT: weight}, transformer / first)
    save_file({DENSE_BIAS: bias}, transformer / second)
    save_file({SOURCE_WEIGHT: weight}, transformer / "old-merged.safetensors")
    (transformer / INDEX_NAME).write_text(json.dumps({"weight_map": {SOURCE_WEIGHT: first, DENSE_BIAS: second}}))

    packed_weight = torch.full((8, 8), 0x77, dtype=torch.uint8)
    block_scale = torch.full((8, 1), 448.0).to(torch.float8_e4m3fn)
    layout = SimpleNamespace(layout_128x4=object())
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get("device") == torch.device("cuda:0"):
            kwargs["device"] = torch.device("cpu")
        return original_to(tensor, *args, **kwargs)

    def pack(value, multiplier, *, sfLayout, do_shuffle):
        torch.testing.assert_close(value, weight)
        assert multiplier.item() == 1344.0
        assert sfLayout is layout.layout_128x4 and not do_shuffle
        return packed_weight, block_scale

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(nvfp4, "_require_flashinfer", lambda: (layout, None, None))
    monkeypatch.setattr(nvfp4, "_nvfp4_quantize", pack)
    destination = tmp_path / "packed"
    converter.export_checkpoint(source, destination, "cuda:0")

    assert json.loads((destination / "config.json").read_text()) == config
    dense_index = json.loads((destination / INDEX_NAME).read_text())["weight_map"]
    assert set(dense_index) == {DENSE_BIAS}
    dense_weights = load_file(destination / dense_index[DENSE_BIAS])
    assert set(dense_weights) == {DENSE_BIAS}
    torch.testing.assert_close(dense_weights[DENSE_BIAS], bias)

    packed_path = destination / nvfp4.H3_NVFP4_DIT_EXPORT_FILENAME
    with safe_open(packed_path, framework="pt", device="cpu") as reader:
        assert reader.metadata() == {"layer_profile": "kandinsky6"}
        assert set(reader.keys()) == {f"{PACKED_PREFIX}::{name}" for name in nvfp4.H3_NVFP4_DIT_BUFFER_NAMES}

    def init_cpu_quant_method(method, layer_prefix):
        method.layer_prefix = layer_prefix

    monkeypatch.setattr(nvfp4.NVFP4QuantizeMethod, "__init__", init_cpu_quant_method)
    quant = nvfp4.NVFP4Config(layer_profile="kandinsky6")
    linear = ReplicatedLinear(16, 8, bias=False, params_dtype=torch.bfloat16, quant_config=quant, prefix=PACKED_PREFIX)
    assert isinstance(linear.quant_method, nvfp4.NVFP4QuantizeMethod)
    assert quant.get_quant_method(linear, "audio_out_layer.out_layer") is None
    model = nn.Module()
    model.visual_transformer_blocks = nn.ModuleList([nn.Module()])
    block = model.visual_transformer_blocks[0]
    block.videoT = nn.Module()
    block.videoT.feed_forward = nn.Module()
    block.videoT.feed_forward.mlp = nn.Module()
    block.videoT.feed_forward.mlp.fc_in = linear

    assert nvfp4.load_minimax_h3_nvfp4_dit_export(model, str(packed_path), device="cpu") == 1
    assert linear.weight is None
    torch.testing.assert_close(linear._nvfp4_weight, packed_weight)
    torch.testing.assert_close(linear._nvfp4_weight_scale.float(), block_scale.float())
    assert linear._nvfp4_alpha.item() == pytest.approx(1 / 1344)
    assert linear._weight_global_sf.item() == 1344.0
