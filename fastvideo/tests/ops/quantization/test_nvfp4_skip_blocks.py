# SPDX-License-Identifier: Apache-2.0
"""NVFP4 DiT exports that keep some MiniMax-H3 main blocks in bf16 (FastH3 OmniRef QAD: blocks 42-49)."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.quantization.nvfp4_config import (
    NVFP4Config,
    NVFP4QuantizeMethod,
    minimax_h3_block_index,
    read_minimax_h3_nvfp4_dit_skip_blocks,
)
from fastvideo.models.loader.fsdp_load import _adopt_nvfp4_export_skip_blocks


def _linear() -> ReplicatedLinear:
    return ReplicatedLinear(32, 32, bias=False)


def test_block_index_parses_main_blocks_only():
    assert minimax_h3_block_index("transformer_blocks.42.attn.to_q") == 42
    assert minimax_h3_block_index("minimax_h3.transformer_blocks.7.ff.fc_in") == 7
    assert minimax_h3_block_index("token_refiner.refiner_blocks.1.attn.to_q") is None
    assert minimax_h3_block_index("proj_out") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4QuantizeMethod allocates its unit scale on CUDA")
def test_skip_blocks_leave_listed_blocks_unquantized():
    config = NVFP4Config(layer_profile="h3_dit_vsa", skip_blocks=[49, 42, 42])
    assert config.skip_blocks == (42, 49)
    layer = _linear()
    assert isinstance(config.get_quant_method(layer, "transformer_blocks.41.attn.to_gate_compress"),
                      NVFP4QuantizeMethod)
    assert isinstance(config.get_quant_method(layer, "transformer_blocks.41.ff.fc_out"), NVFP4QuantizeMethod)
    for prefix in ("transformer_blocks.42.attn.to_q", "transformer_blocks.49.ff.fc_in",
                   "transformer_blocks.42.attn.to_gate_compress"):
        assert config.get_quant_method(layer, prefix) is None


def test_skip_blocks_round_trip_and_copy():
    config = NVFP4Config.from_config({"layer_profile": "h3_dit_vsa", "skip_blocks": [3, 1]})
    assert config.skip_blocks == (1, 3)
    copy = config.with_skip_blocks((5, ))
    assert copy.skip_blocks == (5, ) and config.skip_blocks == (1, 3)
    assert copy.layer_profile == "h3_dit_vsa"


@pytest.mark.parametrize("bad", [(-1, ), (True, ), ("3", )])
def test_skip_blocks_rejects_bad_indices(bad):
    with pytest.raises(ValueError):
        NVFP4Config(layer_profile="h3_dit", skip_blocks=bad)


def _export(tmp_path, metadata):
    path = tmp_path / "nvfp4_weights.safetensors"
    save_file({"transformer_blocks.0.ff.fc_in::_nvfp4_alpha": torch.ones(())}, str(path), metadata=metadata)
    return str(path)


def test_export_metadata_skip_blocks(tmp_path):
    assert read_minimax_h3_nvfp4_dit_skip_blocks(_export(tmp_path, None)) == ()
    assert read_minimax_h3_nvfp4_dit_skip_blocks(_export(tmp_path, {"skip_blocks": "[49, 42]"})) == (42, 49)
    with pytest.raises(ValueError):
        read_minimax_h3_nvfp4_dit_skip_blocks(_export(tmp_path, {"skip_blocks": "{\"a\": 1}"}))


def test_loader_adopts_export_skip_blocks_without_mutating_config(tmp_path):
    path = _export(tmp_path, {"skip_blocks": "[42, 43]"})
    quant_config = NVFP4Config(layer_profile="h3_dit_vsa")
    dit_config = SimpleNamespace(quant_config=quant_config, name="dit")
    params = {"config": dit_config, "hf_config": {}}
    adopted = _adopt_nvfp4_export_skip_blocks(params, quant_config, path)
    assert adopted["config"].quant_config.skip_blocks == (42, 43)
    assert dit_config.quant_config is quant_config and quant_config.skip_blocks == ()
    assert adopted["hf_config"] is params["hf_config"]


def test_loader_keeps_matching_and_rejects_conflicting_skip_blocks(tmp_path):
    path = _export(tmp_path, {"skip_blocks": "[42]"})
    matching = NVFP4Config(layer_profile="h3_dit_vsa", skip_blocks=(42, ))
    params = {"config": SimpleNamespace(quant_config=matching)}
    assert _adopt_nvfp4_export_skip_blocks(params, matching, path) is params
    conflicting = NVFP4Config(layer_profile="h3_dit_vsa", skip_blocks=(41, ))
    with pytest.raises(ValueError, match="skip_blocks"):
        _adopt_nvfp4_export_skip_blocks({"config": SimpleNamespace(quant_config=conflicting)}, conflicting, path)


def _converter():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[4] / "scripts/checkpoint_conversion/convert_minimax_h3_modelopt_nvfp4_dit.py"
    spec = importlib.util.spec_from_file_location("convert_minimax_h3_modelopt_nvfp4_dit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_parses_block_ranges():
    converter = _converter()
    assert converter.parse_blocks("42-49") == tuple(range(42, 50))
    assert converter.parse_blocks("3, 7,42-43") == (3, 7, 42, 43)
    assert converter.parse_blocks("3,3,7-7") == (3, 7)
    assert converter.block_of("transformer_blocks.12.ff.net.2") == 12
    assert converter.block_of("proj_out") is None


@pytest.mark.parametrize("text", ["49-42", "3,7,49-42", "7-3,42-49"])
def test_converter_rejects_inverted_block_ranges(text):
    with pytest.raises(ValueError, match=r"end \(\d+\) is below start \(\d+\)"):
        _converter().parse_blocks(text)


def test_converter_reads_full_name_amax_keys():
    converter = _converter()
    table = {"transformer_blocks.0.ff.fc_in": 3.0, "b1.ff.fc_out": 5.0}
    assert converter.act_amax_for("transformer_blocks.0.ff.fc_in", table, required=True) == 3.0
    assert converter.act_amax_for("transformer_blocks.1.ff.fc_out", table, required=True) == 5.0
    assert converter.act_amax_for("transformer_blocks.2.attn.to_q", table, required=False) is None
    with pytest.raises(SystemExit):
        converter.act_amax_for("transformer_blocks.2.ff.fc_in", table, required=True)
