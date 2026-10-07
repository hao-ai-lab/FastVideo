# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the pure-Python parts of the NVFP4 text-encoder converter."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 import LANGUAGE_PROJECTIONS

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "checkpoint_conversion" / \
    "convert_minimax_h3_text_encoder_nvfp4.py"


@pytest.fixture(scope="module")
def converter():
    spec = importlib.util.spec_from_file_location("convert_minimax_h3_text_encoder_nvfp4", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_language_linear_regex_matches_only_the_seven_projections(converter) -> None:
    for proj in LANGUAGE_PROJECTIONS:
        match = converter.LANGUAGE_LINEAR.match(f"model.language_model.layers.3.{proj}.weight")
        assert match is not None and match["proj"] == proj and match["layer"] == "3"
    for key in (
            "model.language_model.layers.3.self_attn.q_proj.bias",
            "model.language_model.layers.3.self_attn.q_norm.weight",
            "model.language_model.embed_tokens.weight",
            "model.visual.blocks.0.attn.proj.weight",
            "model.layers.3.self_attn.q_proj.weight",
    ):
        assert converter.LANGUAGE_LINEAR.match(key) is None, key
    assert converter.LANGUAGE_LAYER.match("model.language_model.layers.12.mlp.down_proj.weight")["layer"] == "12"


def test_layer_plan_follows_the_conditioner_build_depth(converter) -> None:
    kept, total, tap = converter.layer_plan({"text_config": {}})
    assert (kept, total, tap) == (50, 64, 50)
    kept, total, tap = converter.layer_plan({"text_config": {"num_hidden_layers": 60}})
    assert (kept, total, tap) == (50, 60, 50)
    with pytest.raises(ValueError, match="output_hidden_state_index"):
        converter.layer_plan({"text_config": {"num_hidden_layers": 40}})


def test_scan_source_counts_only_language_linears_below_the_kept_depth(converter, tmp_path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    tensors = {
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        "model.language_model.layers.1.self_attn.q_proj.weight": torch.zeros(2, 2),
        "model.language_model.layers.1.mlp.down_proj.weight": torch.zeros(2, 2),
        "model.language_model.layers.1.self_attn.q_norm.weight": torch.zeros(2),
        "model.visual.blocks.0.attn.proj.weight": torch.zeros(2, 2),
        "lm_head.weight": torch.zeros(2, 2),
    }
    save_file(tensors, str(shard), metadata={"format": "pt"})

    counts, highest_layer, layers_seen = converter.scan_source([str(shard)], kept=2)
    assert counts == {"self_attn.q_proj": 2, "mlp.down_proj": 1}
    assert highest_layer == 1
    assert layers_seen["self_attn.q_proj"] == {0, 1}
    assert layers_seen["mlp.down_proj"] == {1}
    assert layers_seen["mlp.up_proj"] == set()

    counts, highest_layer, _ = converter.scan_source([str(shard)], kept=1)
    assert counts == {"self_attn.q_proj": 1}
    assert highest_layer == 1


def test_shard_writer_uses_hub_names_writes_an_index_and_can_abort(converter, tmp_path) -> None:
    writer = converter.ShardWriter(tmp_path, shard_bytes=8)
    writer.add("a", torch.zeros(4, dtype=torch.uint8))
    writer.add("b", torch.zeros(4, dtype=torch.uint8))
    writer.add("c", torch.zeros(4, dtype=torch.uint8))
    writer.finish()

    names = sorted(path.name for path in tmp_path.glob("*.safetensors"))
    assert names == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    index = json.loads((tmp_path / converter.SAFE_WEIGHTS_INDEX_NAME).read_text())
    assert index["metadata"]["total_size"] == 12
    assert index["weight_map"] == {
        "a": "model-00001-of-00002.safetensors",
        "b": "model-00001-of-00002.safetensors",
        "c": "model-00002-of-00002.safetensors",
    }

    writer.abort()
    assert not list(tmp_path.glob("*.safetensors"))

    dry = converter.ShardWriter(None, shard_bytes=8)
    dry.add("a", torch.zeros(4, dtype=torch.uint8))
    dry.finish()
    assert dry.total_bytes == 4 and not list(tmp_path.glob("*.safetensors"))


def test_argument_validation_rejects_unusable_requests(converter, monkeypatch, tmp_path) -> None:
    base = [str(SCRIPT), "--src", str(tmp_path), "--dst", str(tmp_path / "out")]

    monkeypatch.setattr(sys, "argv", base + ["--keep-bf16", "mlp.dwn_proj"])
    with pytest.raises(SystemExit):
        converter.parse_args()

    monkeypatch.setattr(sys, "argv", base + ["--keep-bf16", ",".join(LANGUAGE_PROJECTIONS)])
    with pytest.raises(SystemExit):
        converter.parse_args()

    monkeypatch.setattr(sys, "argv", base + ["--probe-rows", "-1"])
    with pytest.raises(SystemExit):
        converter.parse_args()

    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--src", str(tmp_path)])
    with pytest.raises(SystemExit):
        converter.parse_args()

    monkeypatch.setattr(sys, "argv", base + ["--keep-bf16", "mlp.down_proj,mlp.down_proj"])
    args = converter.parse_args()
    assert args.keep_bf16 == ("mlp.down_proj", )
    assert args.max_probe_error == 0.5
