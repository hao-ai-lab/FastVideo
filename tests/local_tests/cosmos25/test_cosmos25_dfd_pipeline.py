# SPDX-License-Identifier: Apache-2.0
"""Isolated runtime contracts for Cosmos Predict2.5 DFD V2W."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_cosmos25_dfd import Cosmos25DFDScheduler
from fastvideo.pipelines.basic.cosmos.cosmos2_5_pipeline import Cosmos25DFDInputValidationStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import Cosmos25DFDV2WDenoisingStage
from fastvideo.pipelines.stages.latent_preparation import Cosmos25DFDV2WLatentPreparationStage


class _Progress:

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def update(self) -> None:
        pass


class _LatentDistribution:

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def sample(self, _generator=None) -> torch.Tensor:
        return self.value


class _RecordingVAE(torch.nn.Module):
    handles_latent_norm = True

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.encode_shapes: list[tuple[int, ...]] = []

    def encode(self, image: torch.Tensor) -> _LatentDistribution:
        self.encode_shapes.append(tuple(image.shape))
        latent = torch.full(
            (image.shape[0], 2, 1, image.shape[-2] // 8, image.shape[-1] // 8),
            0.25,
            device=image.device,
            dtype=image.dtype,
        )
        return _LatentDistribution(latent)


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


def test_dfd_latent_preparation_encodes_one_frame_and_scales_bf16_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fastvideo.pipelines.stages.latent_preparation.get_local_torch_device",
        lambda: torch.device("cpu"),
    )
    scheduler = Cosmos25DFDScheduler()
    transformer = _RecordingTransformer()
    vae = _RecordingVAE()
    stage = Cosmos25DFDV2WLatentPreparationStage(scheduler, transformer, vae)
    batch = ForwardBatch(
        data_type="video",
        prompt="test",
        prompt_embeds=[torch.zeros(1, 2, 4)],
        preprocessed_image=torch.zeros(1, 3, 3, 16, 24),
        height=16,
        width=24,
        num_frames=5,
        seed=7,
        seeds=[7],
        generator=[torch.Generator("cpu").manual_seed(7)],
    )

    stage.forward(batch, _args())

    expected_noise = torch.randn(
        (2, 2, 2, 3),
        generator=torch.Generator("cpu").manual_seed(7),
        dtype=torch.bfloat16,
    )
    assert batch.latents is not None
    assert_close(batch.latents[0], scheduler.scale_noise(torch.zeros_like(expected_noise), noise=expected_noise))
    assert vae.encode_shapes == [(1, 3, 1, 16, 24)]
    assert batch.conditioning_latents is not None
    assert batch.conditioning_latents.shape == (1, 2, 1, 2, 3)
    assert batch.cond_mask is not None
    assert_close(batch.cond_mask[:, :, 0], torch.ones(1, 1, 2, 3, dtype=torch.bfloat16))
    assert_close(batch.cond_mask[:, :, 1], torch.zeros(1, 1, 2, 3, dtype=torch.bfloat16))


def test_dfd_denoising_uses_clean_condition_and_per_frame_timesteps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fastvideo.pipelines.stages.denoising.set_forward_context",
        lambda **_kwargs: nullcontext(),
    )
    scheduler = Cosmos25DFDScheduler()
    scheduler.set_timesteps(4)
    transformer = _RecordingTransformer()
    stage = Cosmos25DFDV2WDenoisingStage.__new__(Cosmos25DFDV2WDenoisingStage)
    stage.transformer = transformer
    stage.scheduler = scheduler
    stage.pipeline = None
    stage.progress_bar = lambda **_kwargs: _Progress()

    noise = torch.arange(48, dtype=torch.bfloat16).reshape(1, 2, 2, 3, 4) / 48
    initial = scheduler.scale_noise(torch.zeros_like(noise), noise=noise)
    conditioning = torch.full((1, 2, 1, 3, 4), 0.25, dtype=torch.bfloat16)
    condition_mask = torch.zeros(1, 1, 2, 3, 4, dtype=torch.bfloat16)
    condition_mask[:, :, :1] = 1
    batch = ForwardBatch(
        data_type="video",
        prompt_embeds=[torch.zeros(1, 3, 4)],
        latents=initial.clone(),
        timesteps=scheduler.timesteps.clone(),
        num_inference_steps=4,
        guidance_scale=1.0,
        fps=24,
    )
    batch.conditioning_latents = conditioning
    batch.cond_mask = condition_mask
    batch.padding_mask = torch.zeros(1, 1, 3, 4, dtype=torch.bfloat16)

    stage.forward(batch, _args())

    assert len(transformer.calls) == 4
    first_call = transformer.calls[0]
    assert_close(first_call["hidden_states"][:, :, :1], conditioning)
    assert_close(first_call["timestep"][:, :1], torch.zeros(1, 1, dtype=torch.float64))
    assert_close(first_call["timestep"][:, 1:], torch.full((1, 1), 0.999, dtype=torch.float64))
    assert first_call["fps"].dtype is torch.float32
    assert batch.latents is not None
    assert_close(batch.latents[:, :, :1], conditioning)


def _production_batch(**overrides) -> ForwardBatch:
    values = {
        "data_type": "video",
        "prompt": "test",
        "image_path": "frame.png",
        "height": 704,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "fps": 24,
        "seed": 0,
    }
    values.update(overrides)
    return ForwardBatch(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image_path": None}, "conditioning image"),
        ({"video_path": "clip.mp4"}, "single image"),
        ({"num_inference_steps": 3}, "exactly 4"),
        ({"num_frames": 77}, "exactly 81"),
        ({"height": 720}, "height=704"),
        ({"fps": 16}, "24 fps"),
        ({"guidance_scale": 3.0}, "guidance_scale=1"),
    ],
)
def test_dfd_validation_rejects_unsupported_contract(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Cosmos25DFDInputValidationStage().forward(_production_batch(**overrides), _args())

