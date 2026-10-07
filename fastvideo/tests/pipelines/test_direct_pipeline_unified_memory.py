# SPDX-License-Identifier: Apache-2.0
"""Direct pipeline construction applies offload policy after device setup."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

import fastvideo.pipelines.composed_pipeline_base as composed_pipeline_base
from fastvideo.fastvideo_args import UNIFIED_MEMORY_OFFLOAD_FLAGS, FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase


class _Profiler:

    def region(self, name):
        del name
        return nullcontext()


class _Pipeline(ComposedPipelineBase):
    events = []

    def load_modules(self, fastvideo_args, loaded_modules=None):
        del loaded_modules
        policy_state = {flag: getattr(fastvideo_args, flag) for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS}
        policy_state["use_fsdp_inference"] = fastvideo_args.use_fsdp_inference
        self.events.append(("load_modules", policy_state))
        return {}

    def create_pipeline_stages(self, fastvideo_args):
        del fastvideo_args


def test_direct_pipeline_applies_policy_after_device_initialization(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(_Pipeline, "events", events)
    args = FastVideoArgs(model_path="unused", use_fsdp_inference=True)

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

    pipeline = _Pipeline("unused", args, required_config_modules=[])

    assert pipeline.modules == {}
    assert events == [
        ("distributed", None),
        ("offload_policy", 4),
        (
            "load_modules",
            {
                **{flag: False for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS},
                "use_fsdp_inference": True,
            },
        ),
    ]
