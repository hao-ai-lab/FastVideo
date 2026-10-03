# SPDX-License-Identifier: Apache-2.0
"""Direct pipeline construction applies offload policy after device setup."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

import fastvideo.pipelines.composed_pipeline_base as composed_pipeline_base
from fastvideo.api.device_policy import UNIFIED_MEMORY_OFFLOAD_PATHS
from fastvideo.api.inference_resolution import resolve_inference_config
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.tests.api.config_snapshot import isolated_environment


class _Profiler:

    def region(self, name):
        del name
        return nullcontext()


class _Pipeline(ComposedPipelineBase):
    events = []

    def load_modules(self, resolved_config, loaded_modules=None):
        del loaded_modules
        policy_state = {path: resolved_config.provenance(path).value for path in UNIFIED_MEMORY_OFFLOAD_PATHS.values()}
        policy_state["engine.use_fsdp_inference"] = resolved_config.engine.use_fsdp_inference
        self.events.append(("load_modules", policy_state))
        return {}

    def create_pipeline_stages(self, resolved_config):
        del resolved_config


def test_direct_pipeline_applies_policy_after_device_initialization(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(_Pipeline, "events", events)
    with isolated_environment():
        resolved_config = resolve_inference_config({
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "engine": {
                "use_fsdp_inference": True
            },
        })

    def classify_device(device_id):
        events.append(("offload_policy", device_id))
        return True

    monkeypatch.setattr(
        composed_pipeline_base,
        "maybe_init_distributed_environment_and_model_parallel",
        lambda *args: events.append(("distributed", None)),
    )
    monkeypatch.setattr(composed_pipeline_base, "get_local_torch_device", lambda: torch.device("cuda:4"))
    monkeypatch.setattr(composed_pipeline_base, "get_world_group", lambda: SimpleNamespace(local_rank=4))
    monkeypatch.setattr(composed_pipeline_base, "get_or_create_profiler", lambda trace_dir: _Profiler())
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", classify_device)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)

    pipeline = _Pipeline("unused", resolved_config, required_config_modules=[])

    assert pipeline.modules == {}
    assert events == [
        ("distributed", None),
        ("offload_policy", 4),
        (
            "load_modules",
            {
                **{path: False for path in UNIFIED_MEMORY_OFFLOAD_PATHS.values()},
                "engine.use_fsdp_inference": True,
            },
        ),
    ]
