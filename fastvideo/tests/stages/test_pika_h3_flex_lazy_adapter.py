# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex's automatic adapter must not reload released lazy components.

CPU-only: the components are tiny stand-ins and the adapter load itself is
stubbed, so the test exercises the real LoRA constructor path, the lazy
materialize transform, and MiniMax-H3's pinned-copy invalidation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lazy_module import LazyModule


class _Component(torch.nn.Module):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])


def _args(lora_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        lora_target_modules=None,
        lora_path=lora_path,
        lora_nickname="default",
        lora_strength=1.0,
        training_mode=False,
        lora_training=False,
        dit_layerwise_offload=False,
        pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(arch_config=SimpleNamespace(
            exclude_lora_layers=[]))),
    )


@pytest.fixture
def lazy_pika(monkeypatch, tmp_path):
    """A Pika pipeline whose four heavy components are deferred, with load tracking."""
    from fastvideo.pipelines import lora_pipeline as lora_module
    from fastvideo.pipelines.basic.pika_h3_flex.pipeline import PikaH3FlexPipeline

    loads: list[str] = []
    modules = {
        name: LazyModule(name, lambda name=name: loads.append(name) or _Component(name))
        for name in ("text_encoder", "transformer", "vae", "audio_vae")
    }
    args = _args(str(tmp_path / "fl2va" / "adapter.safetensors"))

    def initialize_base(pipeline, *unused_args, **unused_kwargs):
        pipeline.fastvideo_args = args
        pipeline.modules = dict(modules)

    adapter_calls: list[str] = []

    def record_adapter(pipeline, lora_nickname, lora_path=None, strength=1.0, accumulate=False):
        adapter_calls.append(lora_path)

    monkeypatch.setattr(ComposedPipelineBase, "__init__", initialize_base)
    monkeypatch.setattr(lora_module, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(lora_module.DenseLoRAPatch, "from_adapter", staticmethod(lambda path: None))
    monkeypatch.setattr(lora_module.LoRAPipeline, "set_lora_adapter", record_adapter)

    pipeline = PikaH3FlexPipeline("unused", args)
    return SimpleNamespace(pipeline=pipeline, modules=modules, loads=loads, adapter_calls=adapter_calls)


def test_adapter_attach_after_release_keeps_released_components_released(lazy_pika):
    modules, loads = lazy_pika.modules, lazy_pika.loads
    # Conditioning loads the text encoder, then the release schedule drops it.
    modules["text_encoder"].materialize()
    assert modules["text_encoder"].release() is True
    assert loads == ["text_encoder"]

    # Denoising loads the transformer; its materialize transform attaches the adapter.
    modules["transformer"].materialize()

    assert lazy_pika.adapter_calls == [lazy_pika.pipeline.lora_path]
    assert loads == ["text_encoder", "transformer"], f"adapter attach reloaded {loads[2:]}"
    assert not modules["text_encoder"].is_materialized
    assert not modules["vae"].is_materialized
    assert not modules["audio_vae"].is_materialized


def test_adapter_attach_still_drops_pinned_copies_of_resident_components(lazy_pika):
    modules = lazy_pika.modules
    vae = modules["vae"].materialize()
    vae.__dict__["_pinned_host_tensors"] = {"weight": torch.zeros(1)}

    modules["transformer"].materialize()

    assert "_pinned_host_tensors" not in vae.__dict__
    assert lazy_pika.loads == ["vae", "transformer"]
