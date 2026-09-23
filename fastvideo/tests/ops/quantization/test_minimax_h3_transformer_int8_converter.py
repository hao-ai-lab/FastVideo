# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the int8 transformer converter's key handling and output layout."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "checkpoint_conversion" / \
    "convert_minimax_h3_transformer_int8.py"


@pytest.fixture(scope="module")
def converter():
    spec = importlib.util.spec_from_file_location("convert_minimax_h3_transformer_int8", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("key, native", [
    ("transformer_blocks.0.attn.to_q.weight", "attn.to_q"),
    ("transformer_blocks.12.attn.to_out.0.weight", "attn.to_out"),
    ("transformer_blocks.7.attn.to_gate_compress.weight", "attn.to_gate_compress"),
    ("transformer_blocks.49.ff.net.0.proj.weight", "ff.fc_in"),
    ("transformer_blocks.49.ff.net.2.weight", "ff.fc_out"),
])
def test_block_linear_regex_matches_the_seven_linears(converter, key: str, native: str):
    match = converter.BLOCK_LINEAR.match(key)
    assert match is not None
    assert converter.SOURCE_LINEARS[match["name"]] == native


@pytest.mark.parametrize("key", [
    "transformer_blocks.0.adaln_proj.linear.weight",
    "transformer_blocks.0.attn.norm_q.weight",
    "transformer_blocks.0.norm1.weight",
    "token_refiner.refiner_blocks.0.attn.to_q.weight",
    "token_refiner.refiner_blocks.0.ff.net.2.weight",
    "context_embedder.weight",
    "transformer_blocks.0.attn.to_q.bias",
])
def test_block_linear_regex_leaves_everything_else(converter, key: str):
    assert converter.BLOCK_LINEAR.match(key) is None


def _write_source(tmp_path: Path, blocks: int = 2, gate: bool = True) -> Path:
    src = tmp_path / "transformer"
    src.mkdir(parents=True)
    tensors = {}
    for index in range(blocks):
        prefix = f"transformer_blocks.{index}"
        tensors[f"{prefix}.attn.to_q.weight"] = torch.randn(32, 64, dtype=torch.bfloat16)
        tensors[f"{prefix}.attn.to_k.weight"] = torch.randn(32, 64, dtype=torch.bfloat16)
        tensors[f"{prefix}.attn.to_v.weight"] = torch.randn(32, 64, dtype=torch.bfloat16)
        tensors[f"{prefix}.attn.to_out.0.weight"] = torch.randn(64, 32, dtype=torch.bfloat16)
        if gate:
            tensors[f"{prefix}.attn.to_gate_compress.weight"] = torch.randn(32, 64, dtype=torch.bfloat16)
        tensors[f"{prefix}.ff.net.0.proj.weight"] = torch.randn(128, 64, dtype=torch.bfloat16)
        tensors[f"{prefix}.ff.net.2.weight"] = torch.randn(64, 64, dtype=torch.bfloat16)
        tensors[f"{prefix}.adaln_proj.linear.weight"] = torch.randn(96, 16, dtype=torch.float16)
        tensors[f"{prefix}.norm1.weight"] = torch.ones(64, dtype=torch.bfloat16)
    tensors["context_embedder.weight"] = torch.randn(64, 48, dtype=torch.bfloat16)
    save_file(tensors, str(src / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps({"_class_name": "MiniMaxH3Transformer3DModel", "num_layers": blocks,
                                                 "adaln_rank": 16}), encoding="utf-8")
    (src / "README.md").write_text("notes", encoding="utf-8")
    return src


def test_scan_source_counts_block_linears(converter, tmp_path: Path):
    src = _write_source(tmp_path, blocks=3)
    counts, blocks = converter.scan_source([str(src / "diffusion_pytorch_model.safetensors")])
    assert blocks == 3
    assert dict(counts) == {name: 3 for name in converter.TRANSFORMER_LINEARS}
    src_dense = _write_source(tmp_path / "dense", blocks=2, gate=False)
    counts, blocks = converter.scan_source([str(src_dense / "diffusion_pytorch_model.safetensors")])
    assert blocks == 2 and "attn.to_gate_compress" not in counts


def test_shard_writer_renames_and_indexes(converter, tmp_path: Path):
    writer = converter.ShardWriter(tmp_path, shard_bytes=64)
    writer.add("a", torch.zeros(16, dtype=torch.int8))
    writer.add("b", torch.zeros(48, dtype=torch.int8))  # fills the first shard exactly
    writer.add("c", torch.zeros(8, dtype=torch.float32))  # does not fit, opens the second
    writer.finish()
    names = sorted(path.name for path in tmp_path.glob("*.safetensors"))
    assert names == ["diffusion_pytorch_model-00001-of-00002.safetensors",
                     "diffusion_pytorch_model-00002-of-00002.safetensors"]
    index = json.loads((tmp_path / "diffusion_pytorch_model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 16 + 48 + 32
    assert index["weight_map"] == {
        "a": "diffusion_pytorch_model-00001-of-00002.safetensors",
        "b": "diffusion_pytorch_model-00001-of-00002.safetensors",
        "c": "diffusion_pytorch_model-00002-of-00002.safetensors",
    }
    writer.abort()
    assert not list(tmp_path.glob("*.safetensors"))


def test_end_to_end_conversion_on_cpu(converter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = _write_source(tmp_path, blocks=2)
    dst = tmp_path / "int8"
    monkeypatch.setattr(sys, "argv", [
        "convert", "--src", str(src), "--dst", str(dst), "--device", "cpu", "--probe-rows", "64",
        "--shard-size-gb", "0.0001"
    ])
    converter.main()
    config = json.loads((dst / "config.json").read_text())
    assert config["num_layers"] == 2
    assert config["quantization_config"]["quant_method"] == "int8"
    assert config["quantization_config"]["linears"] == list(converter.TRANSFORMER_LINEARS)
    assert config["quantization_config"]["producer"]["gate_compress"] is True
    assert (dst / "README.md").read_text() == "notes"
    index = json.loads((dst / "diffusion_pytorch_model.safetensors.index.json").read_text())
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors.update(load_file(str(dst / shard)))
    for index_ in range(2):
        prefix = f"transformer_blocks.{index_}"
        for name in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.to_gate_compress",
                     "ff.net.0.proj", "ff.net.2"):
            assert tensors[f"{prefix}.{name}.weight"].dtype == torch.int8
            assert tensors[f"{prefix}.{name}.weight_scale"].dtype == torch.float32
            assert tuple(tensors[f"{prefix}.{name}.weight_scale"].shape) == \
                (tensors[f"{prefix}.{name}.weight"].shape[0], )
        assert tensors[f"{prefix}.adaln_proj.linear.weight"].dtype == torch.float16
        assert tensors[f"{prefix}.norm1.weight"].dtype == torch.bfloat16
    assert tensors["context_embedder.weight"].dtype == torch.bfloat16
    assert not any(key.endswith(".weight_packed") for key in tensors)
    # A second run must refuse to overwrite.
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        converter.main()


def test_converter_refuses_a_serialized_source(converter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = _write_source(tmp_path)
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "int8"}
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["convert", "--src", str(src), "--dst", str(tmp_path / "out"), "--device", "cpu"])
    with pytest.raises(SystemExit, match="already carries a quantization_config"):
        converter.main()


def test_converter_refuses_an_incomplete_block(converter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = _write_source(tmp_path, blocks=2)
    tensors = load_file(str(src / "diffusion_pytorch_model.safetensors"))
    tensors.pop("transformer_blocks.1.ff.net.2.weight")
    save_file(tensors, str(src / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    monkeypatch.setattr(sys, "argv", ["convert", "--src", str(src), "--report-only", "--device", "cpu"])
    with pytest.raises(SystemExit, match="incomplete"):
        converter.main()


def test_arg_validation(converter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["convert", "--src", "x"])
    with pytest.raises(SystemExit):
        converter.parse_args()
    monkeypatch.setattr(sys, "argv", ["convert", "--src", "x", "--dst", "y", "--probe-rows", "-1"])
    with pytest.raises(SystemExit):
        converter.parse_args()
    monkeypatch.setattr(sys, "argv", ["convert", "--src", "x", "--report-only", "--max-probe-error", "0"])
    with pytest.raises(SystemExit):
        converter.parse_args()
