# SPDX-License-Identifier: Apache-2.0
"""PAN2 draws its initial noise in fp32 whatever dtype the prompt embeddings arrive in; no weights or CUDA needed."""

import types

import pytest
import torch

from fastvideo.pipelines.basic.pan2 import pan2_pipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages import LatentPreparationStage, latent_preparation


@pytest.fixture()
def cpu_device(monkeypatch):
    monkeypatch.setattr(latent_preparation, "get_local_torch_device", lambda: torch.device("cpu"))


def _args() -> types.SimpleNamespace:
    arch = types.SimpleNamespace(temporal_compression_ratio=4, spatial_compression_ratio=16)
    return types.SimpleNamespace(pipeline_config=types.SimpleNamespace(vae_config=types.SimpleNamespace(arch_config=arch)))


def _noise(stage, embeds_dtype: torch.dtype) -> torch.Tensor:
    batch = ForwardBatch(data_type="video",
                         prompt="a cat",
                         prompt_embeds=[torch.zeros(1, 4, 8, dtype=embeds_dtype)],
                         generator=torch.Generator("cpu").manual_seed(7),
                         height=32,
                         width=32,
                         num_frames=5)
    return stage.forward(batch, _args()).latents


def test_noise_is_fp32_and_independent_of_the_prompt_embedding_dtype(cpu_device):
    transformer = types.SimpleNamespace(num_channels_latents=3)
    stage = LatentPreparationStage(scheduler=types.SimpleNamespace(),
                                   transformer=transformer,
                                   latent_dtype=torch.float32)

    from_bf16, from_fp32 = _noise(stage, torch.bfloat16), _noise(stage, torch.float32)

    assert from_bf16.dtype == torch.float32
    assert torch.equal(from_bf16, from_fp32)
    assert _noise(LatentPreparationStage(types.SimpleNamespace(), transformer), torch.bfloat16).dtype == torch.bfloat16


def test_pan2_pipeline_requests_fp32_latents(monkeypatch):
    seen = {}

    class _Stage:

        def __init__(self, *args, **kwargs):
            seen.setdefault("kwargs", []).append(kwargs)

    for name in ("PAN2InputValidationStage", "PAN2TextEncodingStage", "PAN2TimestepPreparationStage",
                 "PAN2ConditioningStage", "PAN2DenoisingStage", "DecodingStage"):
        monkeypatch.setattr(pan2_pipeline, name, _Stage)
    monkeypatch.setattr(pan2_pipeline, "LatentPreparationStage", _Stage)
    pipeline = pan2_pipeline.PAN2Pipeline.__new__(pan2_pipeline.PAN2Pipeline)
    stages = {}
    pipeline.add_stage = lambda stage_name, stage: stages.__setitem__(stage_name, stage)
    pipeline.get_module = lambda name: name

    pipeline.create_pipeline_stages(None)

    assert any(kwargs.get("latent_dtype") is torch.float32 for kwargs in seen["kwargs"])
