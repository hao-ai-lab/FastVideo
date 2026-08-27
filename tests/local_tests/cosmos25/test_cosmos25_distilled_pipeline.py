# SPDX-License-Identifier: Apache-2.0
"""Isolated pipeline contracts for the Cosmos Predict2.5 distilled student."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_cosmos25_distilled import Cosmos25DistilledScheduler
from fastvideo.pipelines.basic.cosmos.cosmos2_5_pipeline import Cosmos25DistilledInputValidationStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import Cosmos25DistilledT2WDenoisingStage
from fastvideo.pipelines.stages.latent_preparation import Cosmos25DistilledT2WLatentPreparationStage


class _Progress:

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def update(self) -> None:
        pass


class _RecordingTransformer(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.config = SimpleNamespace(in_channels=2)
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(self, **kwargs):
        self.calls.append({key: value.detach().clone() for key, value in kwargs.items() if torch.is_tensor(value)})
        return torch.zeros_like(kwargs["hidden_states"])


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        disable_autocast=False,
        model_loaded={"transformer": True},
        model_paths={},
    )


def test_distilled_latent_preparation_preserves_official_fp32_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fastvideo.pipelines.stages.latent_preparation.get_local_torch_device",
        lambda: torch.device("cpu"),
    )
    transformer = _RecordingTransformer()
    stage = Cosmos25DistilledT2WLatentPreparationStage(Cosmos25DistilledScheduler(), transformer)
    batch = ForwardBatch(
        data_type="video",
        prompt="test",
        prompt_embeds=[torch.zeros(1, 2, 4)],
        height=16,
        width=24,
        num_frames=5,
        seed=7,
        seeds=[7],
    )

    stage.forward(batch, _args())

    assert batch.latents is not None
    assert batch.latents.dtype is torch.float32
    assert batch.latents.shape == (1, 2, 2, 2, 3)
    expected = torch.randn((2, 2, 2, 3), generator=torch.Generator().manual_seed(7))
    assert_close(batch.latents[0], expected, rtol=0, atol=0)


def test_distilled_denoising_uses_per_frame_timesteps_and_official_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fastvideo.pipelines.stages.denoising.set_forward_context",
        lambda **_kwargs: nullcontext(),
    )
    scheduler = Cosmos25DistilledScheduler()
    scheduler.set_timesteps(2)
    transformer = _RecordingTransformer()
    stage = Cosmos25DistilledT2WDenoisingStage.__new__(Cosmos25DistilledT2WDenoisingStage)
    stage.transformer = transformer
    stage.scheduler = scheduler
    stage.pipeline = None
    stage.progress_bar = lambda **_kwargs: _Progress()

    initial = torch.arange(48, dtype=torch.float32).reshape(1, 2, 2, 3, 4) / 48
    batch = ForwardBatch(
        data_type="video",
        prompt_embeds=[torch.zeros(1, 3, 4)],
        latents=initial.clone(),
        timesteps=scheduler.timesteps.clone(),
        num_inference_steps=2,
        fps=24,
    )
    stage.forward(batch, _args())

    expected_scheduler = Cosmos25DistilledScheduler()
    expected_scheduler.set_timesteps(2)
    expected = initial.clone()
    for timestep in expected_scheduler.timesteps:
        expected_fp32 = expected.float()
        expected_scheduler.scale_model_input(expected_fp32, timestep)
        expected = expected_scheduler.step(torch.zeros_like(expected_fp32), timestep, expected_fp32).prev_sample

    assert batch.latents is not None
    assert batch.latents.dtype is torch.float32
    assert_close(batch.latents, expected.float(), rtol=0, atol=0)
    assert len(transformer.calls) == 2
    assert transformer.calls[0]["timestep"].shape == (1, 2)
    assert_close(transformer.calls[0]["timestep"].float(), torch.ones(1, 2))
    assert_close(transformer.calls[0]["condition_mask"], torch.zeros(1, 1, 2, 3, 4))
    assert_close(transformer.calls[0]["padding_mask"], torch.ones(1, 1, 3, 4))


@pytest.mark.parametrize("field", ["image_path", "pil_image", "preprocessed_image", "video_path", "video_latent"])
def test_distilled_validation_rejects_conditioning(field: str) -> None:
    batch = ForwardBatch(data_type="video", prompt="test", height=16, width=16, seed=0)
    setattr(batch, field, "provided")
    with pytest.raises(ValueError, match="text-to-world"):
        Cosmos25DistilledInputValidationStage().forward(batch, _args())


def test_distilled_validation_rejects_classic_cfg() -> None:
    batch = ForwardBatch(
        data_type="video",
        prompt="test",
        negative_prompt="bad",
        height=16,
        width=16,
        seed=0,
        guidance_scale=2,
    )
    with pytest.raises(ValueError, match="guidance_scale=1"):
        Cosmos25DistilledInputValidationStage().forward(batch, _args())


def test_distilled_validation_rejects_non_distilled_step_count() -> None:
    batch = ForwardBatch(
        data_type="video",
        prompt="test",
        height=16,
        width=16,
        seed=0,
        num_inference_steps=50,
    )
    with pytest.raises(ValueError, match="1 to 4"):
        Cosmos25DistilledInputValidationStage().forward(batch, _args())
