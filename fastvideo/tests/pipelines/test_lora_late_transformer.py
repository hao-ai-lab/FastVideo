# SPDX-License-Identifier: Apache-2.0
"""A runtime adapter set before the transformer is in memory reaches it when it arrives (CPU only)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import fastvideo.envs as envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.lora import linear as lora_linear
from fastvideo.layers.lora.linear import get_lora_layer
from fastvideo.pipelines import lora_pipeline
from fastvideo.pipelines.lazy_module import LazyModule
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

DIM = 8
RANK = 2


class _Pipe(LoRAPipeline):

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        del fastvideo_args


class _Transformer(torch.nn.Module):
    param_names_mapping: dict = {}
    lora_param_names_mapping: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.layer = ReplicatedLinear(DIM, DIM, bias=False, params_dtype=torch.float32)


@pytest.fixture
def adapter_file(tmp_path: Path) -> str:
    path = tmp_path / "adapter.safetensors"
    save_file({"layer.lora_A.weight": torch.randn(RANK, DIM), "layer.lora_B.weight": torch.randn(DIM, RANK)}, str(path))
    return str(path)


@pytest.fixture
def make_pipeline(monkeypatch: pytest.MonkeyPatch, env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_LORA_SIDE_BRANCH.override(True))
    monkeypatch.setattr(lora_pipeline.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(lora_pipeline, "maybe_download_lora", lambda path: path)
    monkeypatch.setattr(lora_linear, "get_local_torch_device", lambda: torch.device("cpu"))

    def make(modules: dict[str, object]) -> _Pipe:
        pipeline = _Pipe.__new__(_Pipe)
        arch_config = SimpleNamespace(param_names_mapping={}, lora_param_names_mapping={})
        fastvideo_args = SimpleNamespace(training_mode=False,
                                         pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(
                                             arch_config=arch_config)))
        attributes = dict(modules=modules, lora_adapters={}, lora_adapter_paths={}, lora_layers={},
                          cur_adapter_name="", cur_adapter_path="", cur_adapter_strength=1.0, lora_initialized=False,
                          lora_path=None, fastvideo_args=fastvideo_args, training_mode=False,
                          device=torch.device("cpu"), _tracks_late_transformers=True, _coverage_deferred=set(),
                          trainable_transformer_modules=dict(modules))
        for name, value in attributes.items():
            setattr(pipeline, name, value)

        def convert_one(name: str, module: torch.nn.Module) -> None:
            module.layer = get_lora_layer(module.layer)
            pipeline.lora_layers[name] = SimpleNamespace(
                lora_layers_by_block=lambda: iter([(None, {"layer": module.layer})]),
                all_lora_layers=lambda: iter([("layer", module.layer)]))

        pipeline._convert_one_transformer = convert_one  # type: ignore[method-assign]
        return pipeline

    return make


def test_adapter_set_before_sequential_load_reaches_the_transformer(make_pipeline, adapter_file: str) -> None:
    pipeline = make_pipeline({})

    pipeline.set_lora_adapter("realism", adapter_file, strength=0.8)
    transformer = _Transformer()
    pipeline.add_module("transformer", transformer)

    assert transformer.layer.lora_A is not None
    assert transformer.layer.lora_strength == 0.8


def test_transformer_loaded_before_any_adapter_is_wrapped_by_the_first_one(make_pipeline, adapter_file: str) -> None:
    pipeline = make_pipeline({})
    transformer = _Transformer()
    pipeline.add_module("transformer", transformer)

    pipeline.set_lora_adapter("realism", adapter_file, strength=0.5)

    assert transformer.layer.lora_A is not None
    assert transformer.layer.lora_strength == 0.5


def test_loading_an_adapter_does_not_load_a_deferred_transformer(make_pipeline, adapter_file: str) -> None:
    transformer = LazyModule("transformer", _Transformer)
    pipeline = make_pipeline({"transformer": transformer})

    pipeline.set_lora_adapter("realism", adapter_file, strength=0.8)

    assert not transformer.is_materialized
    assert transformer.materialize().layer.lora_strength == 0.8


def test_a_change_that_fails_part_way_is_not_recorded(make_pipeline, adapter_file: str,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    transformer = _Transformer()
    pipeline = make_pipeline({"transformer": transformer})
    pipeline.set_lora_adapter("realism", adapter_file, strength=0.8)
    layer_cls = type(transformer.layer)
    set_weights = layer_cls.set_lora_weights

    def fail(*args, **kwargs) -> None:
        raise RuntimeError("device lost")

    monkeypatch.setattr(layer_cls, "set_lora_weights", fail)
    with pytest.raises(RuntimeError, match="device lost"):
        pipeline.set_lora_adapter("realism", strength=0.3)
    monkeypatch.setattr(layer_cls, "set_lora_weights", set_weights)

    pipeline.set_lora_adapter("realism", strength=0.3)

    assert transformer.layer.lora_strength == 0.3


def test_coverage_deferred_by_sequential_load_is_checked_when_the_transformer_arrives(
        make_pipeline, tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.safetensors"
    save_file({"unet.0.lora_A.weight": torch.randn(RANK, DIM), "unet.0.lora_B.weight": torch.randn(DIM, RANK)},
              str(foreign))
    pipeline = make_pipeline({})
    pipeline.lora_strict_coverage = True
    pipeline.set_lora_adapter("foreign", str(foreign), strength=0.8)

    transformer = _Transformer()
    with pytest.raises(ValueError, match="reaches 0 of 1"):
        pipeline.add_module("transformer", transformer)

    assert "foreign" not in pipeline.lora_adapters and pipeline._runtime_adapter is None
    assert transformer.layer.lora_A is None


def test_a_lazy_transformer_whose_deferred_coverage_fails_is_not_kept_alive(make_pipeline, tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.safetensors"
    save_file({"unet.0.lora_A.weight": torch.randn(RANK, DIM), "unet.0.lora_B.weight": torch.randn(DIM, RANK)},
              str(foreign))
    transformer = LazyModule("transformer", _Transformer)
    pipeline = make_pipeline({"transformer": transformer})
    pipeline.lora_strict_coverage = True
    pipeline.set_lora_adapter("foreign", str(foreign), strength=0.8)

    with pytest.raises(ValueError, match="reaches 0 of 1"):
        transformer.materialize()

    assert not transformer.is_materialized
    assert "transformer" not in pipeline.lora_layers
