# SPDX-License-Identifier: Apache-2.0
"""Direct pipeline construction runs on the resolved config, whose offload settings resolution already settled."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import fastvideo.pipelines.composed_pipeline_base as composed_pipeline_base
from fastvideo.api import device_policy
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


def test_direct_pipeline_loads_with_the_resolved_offload_policy(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(_Pipeline, "events", events)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda device_id: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda device_id: "NVIDIA GB10")
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    with isolated_environment(), patch.object(device_policy, "APPLY_DEVICE_POLICY", True):
        resolved_config = resolve_inference_config({
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "engine": {
                "use_fsdp_inference": True
            },
        })

    probe = Mock(side_effect=AssertionError("device probe ran while building the pipeline"))
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", probe)
    monkeypatch.setattr(
        composed_pipeline_base,
        "maybe_init_distributed_environment_and_model_parallel",
        lambda *args: events.append(("distributed", None)),
    )
    monkeypatch.setattr(composed_pipeline_base, "get_world_group", lambda: SimpleNamespace(local_rank=4))
    monkeypatch.setattr(composed_pipeline_base, "get_or_create_profiler", lambda trace_dir: _Profiler())

    pipeline = _Pipeline("unused", resolved_config, required_config_modules=[])

    assert pipeline.modules == {}
    assert pipeline.resolved_config is resolved_config
    assert events == [
        ("distributed", None),
        (
            "load_modules",
            {
                **{path: False for path in UNIFIED_MEMORY_OFFLOAD_PATHS.values()},
                "engine.use_fsdp_inference": True,
            },
        ),
    ]
    probe.assert_not_called()
