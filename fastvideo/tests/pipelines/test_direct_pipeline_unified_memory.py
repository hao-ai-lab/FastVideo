# SPDX-License-Identifier: Apache-2.0
"""Direct pipeline construction applies offload policy after device setup."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

import fastvideo.pipelines.composed_pipeline_base as composed_pipeline_base
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase


class _Profiler:

    def region(self, name):
        del name
        return nullcontext()


class _Pipeline(ComposedPipelineBase):
    events = []

    def load_modules(self, fastvideo_args, loaded_modules=None):
        del loaded_modules
        self.events.append(("load_modules", fastvideo_args.text_encoder_cpu_offload))
        return {}

    def create_pipeline_stages(self, fastvideo_args):
        del fastvideo_args


def test_direct_pipeline_applies_policy_after_device_initialization(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(_Pipeline, "events", events)
    args = FastVideoArgs(model_path="unused", text_encoder_cpu_offload=True)

    def apply_policy(device_id):
        events.append(("offload_policy", device_id))
        args.text_encoder_cpu_offload = False
        return True

    args.disable_offload_on_unified_memory = Mock(side_effect=apply_policy)

    monkeypatch.setattr(
        composed_pipeline_base,
        "maybe_init_distributed_environment_and_model_parallel",
        lambda *args: events.append(("distributed", None)),
    )
    monkeypatch.setattr(composed_pipeline_base, "get_local_torch_device", lambda: torch.device("cuda:4"))
    monkeypatch.setattr(composed_pipeline_base, "get_world_group", lambda: SimpleNamespace(local_rank=4))
    monkeypatch.setattr(composed_pipeline_base, "get_or_create_profiler", lambda trace_dir: _Profiler())

    pipeline = _Pipeline("unused", args, required_config_modules=[])

    assert pipeline.modules == {}
    assert events == [
        ("distributed", None),
        ("offload_policy", 4),
        ("load_modules", False),
    ]
