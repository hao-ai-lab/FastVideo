# SPDX-License-Identifier: Apache-2.0
"""A runtime LoRA adapter survives the release and reload of a lazy transformer (CPU only)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import fastvideo.envs as envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.layers.linear import ReplicatedLinear
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

    def __init__(self) -> None:
        super().__init__()
        self.layer = ReplicatedLinear(DIM, DIM, bias=False, params_dtype=torch.bfloat16)


def _lazy_pipeline() -> tuple[_Pipe, LazyModule]:
    pipeline = _Pipe.__new__(_Pipe)
    transformer = LazyModule("transformer", _Transformer)
    adapter = {"layer.lora_A": torch.randn(RANK, DIM), "layer.lora_B": torch.randn(DIM, RANK)}
    attributes = dict(lora_adapters={"realism": adapter}, lora_adapter_paths={}, lora_layers={}, cur_adapter_name="",
                      cur_adapter_path="", cur_adapter_strength=1.0, lora_initialized=False, lora_path=None,
                      fastvideo_args=SimpleNamespace(training_mode=False), device=torch.device("cpu"),
                      trainable_transformer_modules={"transformer": transformer})
    for name, value in attributes.items():
        setattr(pipeline, name, value)

    def convert_one(name: str, module: torch.nn.Module) -> None:
        module.layer = get_lora_layer(module.layer)
        pipeline.lora_layers[name] = SimpleNamespace(lora_layers_by_block=lambda: iter([(None, {"layer": module.layer})]))

    pipeline._convert_one_transformer = convert_one  # type: ignore[method-assign]
    return pipeline, transformer


def test_runtime_adapter_is_reapplied_after_a_lazy_reload(monkeypatch, env_overrides) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_LORA_SIDE_BRANCH.override(True))
    monkeypatch.setattr(lora_pipeline.dist, "get_rank", lambda: 0)
    pipeline, transformer = _lazy_pipeline()
    pipeline.convert_to_lora_layers()

    first = transformer.materialize()
    pipeline.set_lora_adapter("realism", strength=0.8)
    assert first.layer.lora_A is not None
    assert transformer.release()

    second = transformer.materialize()
    assert second is not first
    assert second.layer.lora_A is not None
    assert second.layer.lora_strength == 0.8



def test_adapter_change_on_a_released_transformer_does_not_load_it(monkeypatch, env_overrides) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_LORA_SIDE_BRANCH.override(True))
    monkeypatch.setattr(lora_pipeline.dist, "get_rank", lambda: 0)
    pipeline, transformer = _lazy_pipeline()
    pipeline.convert_to_lora_layers()
    transformer.materialize()
    pipeline.set_lora_adapter("realism", strength=0.8)
    assert transformer.release()

    pipeline.set_lora_adapter("realism", strength=0.0)

    assert not transformer.is_materialized
    assert transformer.materialize().layer.lora_strength == 0.0
