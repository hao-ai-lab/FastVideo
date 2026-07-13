# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for LingBot-Video's decoded-video MoE refiner stages."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch


class _ZeroDistribution:
    """Return a fixed VAE sample without consuming the supplied generator."""

    def __init__(self, sample: torch.Tensor) -> None:
        self.value = sample

    def sample(self, generator: torch.Generator) -> torch.Tensor:
        del generator
        return self.value


class _FakeVAE:
    """Record normalized pixels and emit zero latents in Wan geometry."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16)
        self.last_input: torch.Tensor | None = None
        self.last_device: torch.device | None = None

    def to(self, device: str | torch.device) -> _FakeVAE:
        """Record explicit offload moves used by the refiner stage."""
        self.last_device = torch.device(device)
        return self

    def encode(self, video: torch.Tensor) -> _ZeroDistribution:
        """Create one zero latent for each 4x8x8 video cell."""
        self.last_input = video
        shape = (video.shape[0], 16, (video.shape[2] - 1) // 4 + 1, video.shape[3] // 8, video.shape[4] // 8)
        return _ZeroDistribution(torch.zeros(shape, device=video.device, dtype=video.dtype))


class _FakeScheduler:
    """Capture the custom refiner schedule passed by the preparation stage."""

    sigma_max = 0.999
    sigma_min = 0.0

    def __init__(self) -> None:
        self.sigmas: np.ndarray | None = None
        self.shift: float | None = None
        self.timesteps = torch.empty(0)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device,
        sigmas: np.ndarray,
        shift: float,
    ) -> None:
        """Mirror the schedule surface needed by LingBot refiner preparation."""
        assert num_inference_steps == len(sigmas)
        self.sigmas = sigmas
        self.shift = shift
        self.timesteps = torch.from_numpy(sigmas * 1000.0).to(device=device, dtype=torch.int64)


def test_lingbot_video_refiner_sigmas_match_released_schedule() -> None:
    """Match the official eight-step, threshold-0.85 schedule including its tail."""
    from fastvideo.pipelines.basic.lingbot_video.stages import _compute_refiner_sigmas

    actual = _compute_refiner_sigmas(0.999, 0.0, 8, 3.0, 0.85)
    expected = np.asarray(
        [0.85, 0.83296275, 0.7496248, 0.6424896, 0.49966654, 0.29975995, 0.19983996, 0.09991998],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_lingbot_video_refiner_preparation_resizes_encodes_and_noises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise pixel resize, VAE normalization, seeded noise, and scheduler setup."""
    from fastvideo.pipelines.basic.lingbot_video import stages
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))
    vae = _FakeVAE()
    scheduler = _FakeScheduler()
    stage = stages.LingBotVideoRefinerPreparationStage(vae, scheduler)
    batch = ForwardBatch(
        data_type="video",
        output=torch.linspace(0.0, 1.0, 3 * 5 * 16 * 32).reshape(1, 3, 5, 16, 32),
        height=16,
        width=32,
        height_sr=32,
        width_sr=64,
        num_frames=5,
        num_inference_steps_sr=8,
        t_thresh=0.85,
        seed=42,
    )
    args = cast(
        Any,
        SimpleNamespace(
            pipeline_config=SimpleNamespace(flow_shift=3.0),
            vae_cpu_offload=True,
        ),
    )
    result = stage.forward(batch, args)

    assert vae.last_input is not None and tuple(vae.last_input.shape) == (1, 3, 5, 32, 64)
    assert float(vae.last_input.min()) >= -1.0 and float(vae.last_input.max()) <= 1.0
    assert result.output is None and result.raw_latent_shape == (1, 16, 2, 4, 8)
    expected_noise = torch.randn(result.raw_latent_shape, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(result.latents, expected_noise * 0.85)
    assert scheduler.shift == 1.0
    assert scheduler.sigmas is not None and scheduler.sigmas.shape == (8,)
    assert result.extra["lingbot_video_base_shape"] == (1, 3, 5, 16, 32)
    assert vae.last_device == torch.device("cpu")


def test_lingbot_video_refiner_uses_zero_cloned_null_condition() -> None:
    """Ignore encoded negative text and clone the positive mask for refiner CFG."""
    from fastvideo.pipelines.basic.lingbot_video.stages import LingBotVideoDenoisingStage
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    stage = LingBotVideoDenoisingStage(torch.nn.Linear(2, 2), SimpleNamespace(), refiner=True)
    prompt = torch.ones(1, 3, 2)
    prompt_mask = torch.tensor([[1, 1, 0]])
    batch = ForwardBatch(
        data_type="video",
        prompt_embeds=[prompt],
        prompt_attention_mask=[prompt_mask],
        negative_prompt_embeds=[torch.full_like(prompt, 7.0)],
        negative_attention_mask=[torch.zeros_like(prompt_mask)],
        guidance_scale=1.0,
        guidance_scale_2=3.0,
        batch_cfg=True,
    )
    condition, mask = stage._prepare_conditions(batch, torch.float32, torch.device("cpu"))
    torch.testing.assert_close(condition[0], prompt[0])
    torch.testing.assert_close(condition[1], torch.zeros_like(prompt[0]))
    torch.testing.assert_close(mask[0], prompt_mask[0])
    torch.testing.assert_close(mask[1], prompt_mask[0])

    base_stage = LingBotVideoDenoisingStage(torch.nn.Linear(2, 2), SimpleNamespace())
    base_condition, base_mask = base_stage._prepare_conditions(batch, torch.float32, torch.device("cpu"))
    assert base_condition.shape == prompt.shape
    torch.testing.assert_close(base_condition, prompt)
    torch.testing.assert_close(base_mask, prompt_mask)


def test_lingbot_video_pipeline_loads_refiner_and_vae_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require transformer_2 and the VAE encoder only for a declared refiner layout."""
    from fastvideo.pipelines import ComposedPipelineBase
    from fastvideo.pipelines.basic.lingbot_video.lingbot_video_pipeline import LingBotVideoPipeline

    captured: dict[str, Any] = {}

    def fake_super_load(self, fastvideo_args, loaded_modules=None):
        """Capture the required module list after LingBot-specific discovery."""
        del fastvideo_args, loaded_modules
        captured["required"] = list(self.required_config_modules)
        return {name: object() for name in self.required_config_modules}

    monkeypatch.setattr(ComposedPipelineBase, "load_modules", fake_super_load)
    pipe = object.__new__(LingBotVideoPipeline)
    pipe.model_path = "/model"
    monkeypatch.setattr(pipe, "_load_config", lambda _path: {"transformer_2": ["diffusers", "model"]})
    vae_config = SimpleNamespace(load_encoder=False)
    args = cast(Any, SimpleNamespace(pipeline_config=SimpleNamespace(vae_config=vae_config)))
    modules = pipe.load_modules(args)

    assert "transformer_2" in captured["required"]
    assert "transformer_2" in modules
    assert vae_config.load_encoder is True

    base_pipe = object.__new__(LingBotVideoPipeline)
    base_pipe.model_path = "/model"
    monkeypatch.setattr(base_pipe, "_load_config", lambda _path: {"transformer_2": ["diffusers", "model"]})
    base_vae_config = SimpleNamespace(load_encoder=False)
    base_args = cast(
        Any,
        SimpleNamespace(
            refine_enabled=False,
            pipeline_config=SimpleNamespace(vae_config=base_vae_config),
        ),
    )
    base_modules = base_pipe.load_modules(base_args)
    assert "transformer_2" not in base_modules
    assert base_vae_config.load_encoder is False
