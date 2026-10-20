# SPDX-License-Identifier: Apache-2.0
"""The constructor LoRA waits until sequential load brings the transformer in."""

from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.lora_pipeline import LoRAPipeline


class _Pipe(LoRAPipeline):

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        del fastvideo_args


def test_sequential_load_attaches_lora_when_the_transformer_arrives() -> None:
    pipeline = _Pipe.__new__(_Pipe)
    pipeline.modules = {}
    pipeline._lazy_release_hooks_installed = False
    pipeline._constructor_adapter_pending = True
    pipeline.trainable_transformer_names = ["transformer"]
    pipeline.trainable_transformer_modules = {}
    pipeline.lora_initialized = False
    pipeline.lora_path = "/tmp/community_h3_lora.safetensors"
    pipeline.lora_layers = {}
    events: list[str] = []

    def convert_all() -> None:
        events.append("convert")
        pipeline.lora_initialized = True

    pipeline.convert_to_lora_layers = convert_all  # type: ignore[method-assign]
    pipeline._apply_constructor_adapter = lambda: events.append("apply")  # type: ignore[method-assign]

    pipeline.add_module("vae", torch.nn.Linear(1, 1))
    assert events == []
    assert pipeline._constructor_adapter_pending

    transformer = torch.nn.Linear(1, 1)
    pipeline.add_module("transformer", transformer)

    assert events == ["convert", "apply"]
    assert pipeline.modules["transformer"] is transformer
    assert pipeline.trainable_transformer_modules["transformer"] is transformer
    assert not pipeline._constructor_adapter_pending
