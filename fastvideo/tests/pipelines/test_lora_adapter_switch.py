# SPDX-License-Identifier: Apache-2.0
"""Switching runtime adapters restores every merged layer the new adapter does not cover (CPU only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.lora import linear as lora_linear
from fastvideo.layers.lora.linear import get_lora_layer
from fastvideo.pipelines import lora_pipeline
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

DIM = 8
RANK = 2


class _Pipe(LoRAPipeline):

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        del fastvideo_args


def _factors(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {"lora_A": torch.randn(RANK, DIM, generator=generator), "lora_B": torch.randn(DIM, RANK, generator=generator)}


def _adapter(layers: list[str], seed: int) -> dict[str, torch.Tensor]:
    return {f"{layer}.{key}": value for layer in layers for key, value in _factors(seed).items()}


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch, env_overrides) -> tuple[_Pipe, torch.nn.Module]:
    env_overrides.enter_context(envs.FASTVIDEO_LORA_SIDE_BRANCH.override(False))
    monkeypatch.setattr(lora_pipeline.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(lora_linear, "get_local_torch_device", lambda: torch.device("cpu"))
    torch.manual_seed(0)
    transformer = torch.nn.Module()
    transformer.first = ReplicatedLinear(DIM, DIM, bias=False, params_dtype=torch.float32)
    transformer.second = ReplicatedLinear(DIM, DIM, bias=False, params_dtype=torch.float32)
    pipe = _Pipe.__new__(_Pipe)
    attributes = dict(lora_adapters={"a": _adapter(["first", "second"], 1), "b": _adapter(["first"], 2)},
                      lora_adapter_paths={}, lora_layers={}, cur_adapter_name="", cur_adapter_path="",
                      cur_adapter_strength=1.0, lora_initialized=False, lora_path=None,
                      fastvideo_args=SimpleNamespace(training_mode=False), device=torch.device("cpu"),
                      trainable_transformer_modules={"transformer": transformer})
    for name, value in attributes.items():
        setattr(pipe, name, value)

    def convert_one(name: str, module: torch.nn.Module) -> None:
        module.first, module.second = get_lora_layer(module.first), get_lora_layer(module.second)
        layers = {"first": module.first, "second": module.second}
        pipe.lora_layers[name] = SimpleNamespace(lora_layers_by_block=lambda: iter([(None, layers)]))

    pipe._convert_one_transformer = convert_one  # type: ignore[method-assign]
    return pipe, transformer


def test_switching_restores_a_merged_layer_the_new_adapter_does_not_cover(setup) -> None:
    pipeline, transformer = setup
    original = transformer.second.weight.detach().clone()

    pipeline.set_lora_adapter("a", strength=1.0)
    second = transformer.second
    assert not second.side_branch and second.merged
    assert not torch.equal(second.base_layer.weight, original)

    pipeline.set_lora_adapter("b", strength=1.0)
    assert torch.equal(second.base_layer.weight, original)
    assert not second.merged and second.disable_lora

    pipeline.set_lora_adapter("b", strength=0.0)
    assert torch.equal(second.base_layer.weight, original)


def test_accumulating_keeps_the_earlier_adapter_where_the_new_one_has_nothing(setup) -> None:
    pipeline, transformer = setup
    pipeline.set_lora_adapter("a", strength=1.0)
    merged = transformer.second.base_layer.weight.detach().clone()

    pipeline.set_lora_adapter("b", strength=1.0, accumulate=True)

    assert torch.equal(transformer.second.base_layer.weight, merged)


def test_clear_restores_base_weights_after_an_adapter(setup) -> None:
    pipeline, transformer = setup
    original = {name: getattr(transformer, name).weight.detach().clone() for name in ("first", "second")}

    pipeline.set_lora_adapter("a", strength=1.0)
    pipeline.clear_lora_adapter()

    for name in ("first", "second"):
        layer = getattr(transformer, name)
        assert torch.equal(layer.base_layer.weight, original[name])
        assert layer.lora_A is None and not layer.merged
    assert pipeline._runtime_adapter is None and pipeline.cur_adapter_name == ""
    pipeline.set_lora_adapter("b", strength=1.0)
    assert transformer.first.merged and torch.equal(transformer.second.base_layer.weight, original["second"])


def test_clear_restores_base_weights_after_a_change_that_failed_part_way(setup) -> None:
    pipeline, transformer = setup
    original = {name: getattr(transformer, name).weight.detach().clone() for name in ("first", "second")}
    pipeline.set_lora_adapter("a", strength=1.0)
    # B's factors for "second" have the wrong shape, so the change fails after "first" already switched to B.
    pipeline.lora_adapters["b"] = {**_adapter(["first"], 2), "second.lora_A": torch.randn(RANK, DIM + 1),
                                   "second.lora_B": torch.randn(DIM, RANK)}

    with pytest.raises(RuntimeError):
        pipeline.set_lora_adapter("b", strength=1.0)
    pipeline.clear_lora_adapter()

    for name in ("first", "second"):
        assert torch.equal(getattr(transformer, name).base_layer.weight, original[name])
