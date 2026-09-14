# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.training import wan_i2v_distillation_pipeline
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.training.wan_i2v_distillation_pipeline import (
    WanI2VDistillationPipeline, )


def test_i2v_prepare_inputs_does_not_pre_shard_conditioning(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = WanI2VDistillationPipeline.__new__(WanI2VDistillationPipeline)
    arch_config = SimpleNamespace(temporal_compression_ratio=4)
    pipeline.training_args = SimpleNamespace(
        num_latent_t=4,
        pipeline_config=SimpleNamespace(vae_config=SimpleNamespace(arch_config=arch_config)),
    )
    pipeline.sp_world_size = 8
    pipeline.rank_in_sp_group = 0
    batch = SimpleNamespace(image_latents=torch.zeros(1, 16, 4, 2, 3))

    monkeypatch.setattr(DistillationPipeline, "_prepare_dit_inputs", lambda self, training_batch: training_batch)
    monkeypatch.setattr(wan_i2v_distillation_pipeline, "get_local_torch_device", lambda: torch.device("cpu"))

    result = pipeline._prepare_dit_inputs(batch)

    assert result.image_latents.shape == (1, 20, 4, 2, 3)


def test_i2v_conditioning_preserves_full_temporal_length_for_sequence_parallelism() -> None:
    image_latents = torch.zeros(1, 16, 4, 2, 3)

    conditioning = WanI2VDistillationPipeline._build_image_conditioning(
        image_latents,
        num_latent_t=4,
    )

    assert conditioning.shape == (1, 20, 4, 2, 3)
    torch.testing.assert_close(conditioning[:, :4, 0], torch.ones_like(conditioning[:, :4, 0]))
    torch.testing.assert_close(conditioning[:, :4, 1:], torch.zeros_like(conditioning[:, :4, 1:]))


def test_i2v_conditioning_rejects_temporal_length_mismatch() -> None:
    image_latents = torch.zeros(1, 16, 3, 2, 3)

    with pytest.raises(ValueError, match="temporal length"):
        WanI2VDistillationPipeline._build_image_conditioning(
            image_latents,
            num_latent_t=4,
        )


def test_i2v_distill_input_kwargs_concatenates_conditioning_in_transformer_layout(
        monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = WanI2VDistillationPipeline.__new__(WanI2VDistillationPipeline)
    monkeypatch.setattr(wan_i2v_distillation_pipeline, "get_local_torch_device", lambda: torch.device("cpu"))

    noise_input = torch.zeros(1, 4, 16, 2, 3)  # bs, t, c, h, w
    image_latents = torch.ones(1, 20, 4, 2, 3)  # bs, c, t, h, w
    batch = SimpleNamespace(image_embeds=torch.zeros(1, 4), image_latents=image_latents)
    text_dict = {
        "encoder_hidden_states": torch.zeros(1, 4),
        "encoder_attention_mask": torch.zeros(1, 4),
    }

    result = pipeline._build_distill_input_kwargs(noise_input, torch.zeros(1), text_dict, batch)

    hidden_states = result.input_kwargs["hidden_states"]
    assert hidden_states.shape == (1, 36, 4, 2, 3)
    torch.testing.assert_close(hidden_states[:, :16], torch.zeros_like(hidden_states[:, :16]))
    torch.testing.assert_close(hidden_states[:, 16:], torch.ones_like(hidden_states[:, 16:]))
    assert result.noise_latents is noise_input
