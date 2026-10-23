# SPDX-License-Identifier: Apache-2.0
"""PAN2 image-to-video keeps the first latent frame out of classifier-free guidance; no weights or CUDA needed."""

import pytest
import torch

from fastvideo.tests.stages._denoising_fixtures import (
    NullProgressBar, TinyDenoiser, TinyScheduler, _patch_denoising_module, _tiny_args, _tiny_batch,
)


def _one_step(monkeypatch, env_overrides, *, image_to_video: bool, skip_first_latent: bool):
    _patch_denoising_module(monkeypatch, env_overrides, None)
    from fastvideo.pipelines.basic.pan2.stages import PAN2DenoisingStage

    stage = PAN2DenoisingStage(TinyDenoiser(), TinyScheduler())
    stage.progress_bar = lambda iterable=None, total=None: NullProgressBar()
    batch = _tiny_batch()
    batch.latents = torch.zeros(1, 1, 3, 1, 1)
    batch.raw_latent_shape = (1, 1, 3, 1, 1)
    batch.timesteps = torch.tensor([0.0])
    batch.num_inference_steps = 1
    batch.pil_image = object() if image_to_video else None
    args = _tiny_args()
    args.pipeline_config.i2v_skip_first_latent_cfg = skip_first_latent
    args.pipeline_config.scheduler_step_in_fp32 = True
    return stage.forward(batch, args).latents.flatten().tolist()


def test_image_to_video_takes_the_positive_prediction_for_the_first_latent_frame(monkeypatch, env_overrides):
    # With a zero latent and timestep the cond prediction is 0.5 and the uncond -0.5, so CFG 2 guides to 1.5.
    latents = _one_step(monkeypatch, env_overrides, image_to_video=True, skip_first_latent=True)

    assert latents == pytest.approx([-0.005, -0.015, -0.015])


@pytest.mark.parametrize("image_to_video,skip_first_latent", [(True, False), (False, True)])
def test_every_frame_is_guided_without_a_conditioning_image_or_with_the_skip_off(monkeypatch, env_overrides,
                                                                                image_to_video, skip_first_latent):
    latents = _one_step(monkeypatch, env_overrides, image_to_video=image_to_video, skip_first_latent=skip_first_latent)

    assert latents == pytest.approx([-0.015, -0.015, -0.015])
