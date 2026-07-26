# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import pytest

from fastvideo.models.loader import fsdp_load
from fastvideo.train.utils import moduleloader
from fastvideo.train.utils.training_config import (
    DistributedConfig,
    ModelTrainingConfig,
    TrainingConfig,
)


class _RepeatedModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
        self.output = torch.nn.Linear(1, 1)
        self._compile_conditions = [lambda name, _module: name in {"blocks.0", "blocks.1"}]
        self._fsdp_shard_conditions = self._compile_conditions
        self.param_names_mapping = {}


def test_make_training_args_propagates_training_fsdp_options() -> None:
    config = TrainingConfig(
        distributed=DistributedConfig(
            hsdp_shard_dim=1,
            reduce_dtype="bf16",
            fsdp_modules_per_group=2,
        ),
        model=ModelTrainingConfig(enable_torch_compile=True),
    )

    args = moduleloader._make_training_args(config, model_path="fake/model")

    assert args.enable_torch_compile is True
    assert args.fsdp_reduce_dtype == "bf16"
    assert args.fsdp_modules_per_group == 2


def test_compile_matched_submodule_forwards_is_regional_and_allows_graph_breaks(monkeypatch) -> None:
    model = _RepeatedModel()
    output_forward = model.output.forward
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_compile(forward, **kwargs):
        calls.append((forward.__self__, kwargs))
        return forward

    monkeypatch.setattr(fsdp_load.torch, "compile", fake_compile)

    count = fsdp_load._compile_matched_submodule_forwards(model, {"backend": "eager"})

    assert count == 2
    assert [module for module, _ in calls] == list(model.blocks)
    assert [kwargs for _, kwargs in calls] == [
        {
            "backend": "eager",
            "fullgraph": False
        },
        {
            "backend": "eager",
            "fullgraph": False
        },
    ]
    assert model.output.forward == output_forward


@pytest.mark.parametrize(
    "conditions",
    [[], [lambda _name, _module: False]],
)
def test_compile_matched_submodule_forwards_rejects_missing_matches(monkeypatch, conditions) -> None:
    model = _RepeatedModel()
    model._compile_conditions = conditions
    monkeypatch.setattr(fsdp_load.torch, "compile", lambda forward, **kwargs: forward)

    with pytest.raises(ValueError, match="refusing to compile the whole FSDP model"):
        fsdp_load._compile_matched_submodule_forwards(model)


def test_compile_matched_submodule_forwards_rejects_fullgraph() -> None:
    with pytest.raises(ValueError, match="requires fullgraph=False"):
        fsdp_load._compile_matched_submodule_forwards(_RepeatedModel(), {"fullgraph": True})


def test_fsdp_loader_compiles_regions_before_sharding(monkeypatch) -> None:
    events: list[str] = []

    def fake_compile(forward, **_kwargs):
        events.append("compile")
        return forward

    def fake_shard_model(model, **_kwargs):
        del model
        events.append("shard")

    def fake_load_state(model, _iterator, device, _dtype, **_kwargs):
        model.to_empty(device=device)

    monkeypatch.setattr(fsdp_load.torch, "compile", fake_compile)
    monkeypatch.setattr(fsdp_load, "init_device_mesh", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(fsdp_load, "shard_model", fake_shard_model)
    monkeypatch.setattr(fsdp_load, "safetensors_weights_iterator", lambda *_args, **_kwargs: iter(()))
    monkeypatch.setattr(fsdp_load, "get_param_names_mapping", lambda _mapping: None)
    monkeypatch.setattr(fsdp_load, "load_model_from_full_model_state_dict", fake_load_state)
    monkeypatch.setattr(fsdp_load, "_maybe_quantize_model", lambda _model: None)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)

    fsdp_load.maybe_load_fsdp_model(
        model_cls=_RepeatedModel,
        init_params={},
        weight_dir_list=[],
        device=torch.device("cpu"),
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        default_dtype=torch.float32,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        enable_torch_compile=True,
    )

    assert events == ["compile", "compile", "shard"]
