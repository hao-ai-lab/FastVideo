# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import pytest

from fastvideo.attention.selector import get_global_forced_attn_backend
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.utils import moduleloader
from fastvideo.train.utils.training_config import (
    DistributedConfig,
    TrainingConfig,
)


def test_load_transformer_scopes_attention_backend(monkeypatch, tmp_path) -> None:
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )
    captured: list[AttentionBackendEnum | None] = []

    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "FakeTransformer")},
    )

    def _fake_load_module(**kwargs):
        del kwargs
        captured.append(get_global_forced_attn_backend())
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(
        moduleloader.PipelineComponentLoader,
        "load_module",
        _fake_load_module,
    )

    result = moduleloader.load_module_from_path(
        model_path="fake/model",
        module_type="transformer",
        training_config=training_config,
        attention_backend="ATTN_QAT_TRAIN",
    )

    assert isinstance(result, torch.nn.Module)
    assert captured == [AttentionBackendEnum.ATTN_QAT_TRAIN]
    assert get_global_forced_attn_backend() is None


def test_load_transformer_restores_backend_when_loading_fails(
    monkeypatch,
    tmp_path,
) -> None:
    training_config = TrainingConfig(
        distributed=DistributedConfig(hsdp_shard_dim=1),
        pipeline_config=PipelineConfig(),
    )
    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "FakeTransformer")},
    )

    def _raise_during_load(**kwargs):
        del kwargs
        assert get_global_forced_attn_backend() is AttentionBackendEnum.ATTN_QAT_TRAIN
        raise RuntimeError("load failed")

    monkeypatch.setattr(
        moduleloader.PipelineComponentLoader,
        "load_module",
        _raise_during_load,
    )

    with pytest.raises(RuntimeError, match="load failed"):
        moduleloader.load_module_from_path(
            model_path="fake/model",
            module_type="transformer",
            training_config=training_config,
            attention_backend="ATTN_QAT_TRAIN",
        )
    assert get_global_forced_attn_backend() is None


def test_load_transformer_configures_recursive_fsdp_reshard(
    monkeypatch,
    tmp_path,
) -> None:
    training_config = TrainingConfig(
        distributed=DistributedConfig(
            hsdp_shard_dim=1,
            reshard_after_forward=False,
        ),
        pipeline_config=PipelineConfig(),
    )
    calls: list[tuple[bool, bool]] = []

    class FakeTransformer(torch.nn.Linear):

        def set_reshard_after_forward(
            self,
            value: bool,
            *,
            recurse: bool,
        ) -> None:
            calls.append((value, recurse))

    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "FakeTransformer")},
    )
    monkeypatch.setattr(
        moduleloader.PipelineComponentLoader,
        "load_module",
        lambda **kwargs: FakeTransformer(1, 1),
    )

    moduleloader.load_module_from_path(
        model_path="fake/model",
        module_type="transformer",
        training_config=training_config,
    )

    assert calls == [(False, True)]


def test_load_transformer_configures_all_fsdp_symmetric_memory_modules(
    monkeypatch,
    tmp_path,
) -> None:
    training_config = TrainingConfig(
        distributed=DistributedConfig(
            hsdp_shard_dim=1,
            fsdp_symmetric_memory=True,
        ),
        pipeline_config=PipelineConfig(),
    )
    calls: list[tuple[str, str, object]] = []

    class FakeFSDPModule(torch.nn.Module):

        def __init__(self, name: str, child: torch.nn.Module | None = None) -> None:
            super().__init__()
            self.name = name
            if child is not None:
                self.child = child

        def set_force_sum_reduction_for_comms(self, value: bool) -> None:
            calls.append((self.name, "force_sum", value))

        def set_symm_mem_for_comm(self, backend: str) -> None:
            calls.append((self.name, "symmetric_memory", backend))

    transformer = FakeFSDPModule("root", FakeFSDPModule("block"))
    monkeypatch.setattr(moduleloader, "FSDPModule", FakeFSDPModule)
    monkeypatch.setattr(moduleloader, "maybe_download_model", lambda path: str(tmp_path))
    monkeypatch.setattr(
        moduleloader,
        "verify_model_config_and_directory",
        lambda path: {"transformer": ("diffusers", "FakeTransformer")},
    )
    monkeypatch.setattr(
        moduleloader.PipelineComponentLoader,
        "load_module",
        lambda **kwargs: transformer,
    )

    moduleloader.load_module_from_path(
        model_path="fake/model",
        module_type="transformer",
        training_config=training_config,
    )

    assert calls == [
        ("root", "force_sum", True),
        ("root", "symmetric_memory", "NCCL"),
        ("block", "force_sum", True),
        ("block", "symmetric_memory", "NCCL"),
    ]
