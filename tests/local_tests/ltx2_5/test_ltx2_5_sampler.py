# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the LTX-2.5 distilled ancestral sampler."""

import torch

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising import (
    ANCESTRAL_NOISE_SEED_OFFSET,
    _ltx2_first_frame_keyframes_mask,
    _ltx2_euler_ancestral_step,
)


def test_ltx2_5_preset_enables_ancestral_sampling() -> None:
    params = SamplingParam.from_pretrained("FastVideo/LTX-2.5-Distilled-Diffusers")

    assert params.ltx2_use_ancestral_sampler is True
    assert params.num_inference_steps == 8
    assert params.ltx2_image_crf == 18.0


def test_ancestral_terminal_step_returns_denoised_prediction() -> None:
    sample = torch.tensor([3.0], dtype=torch.bfloat16)
    denoised = torch.tensor([1.25], dtype=torch.bfloat16)

    actual = _ltx2_euler_ancestral_step(
        sample,
        denoised,
        torch.tensor(0.421875),
        torch.tensor(0.0),
        noise=None,
    )

    torch.testing.assert_close(actual, denoised)


def test_ancestral_eta_zero_matches_deterministic_euler() -> None:
    sample = torch.tensor([2.0, -1.0], dtype=torch.float32)
    denoised = torch.tensor([0.5, 0.25], dtype=torch.float32)
    sigma = torch.tensor(0.8)
    sigma_next = torch.tensor(0.3)
    expected = sample + ((sample - denoised) / sigma) * (sigma_next - sigma)

    actual = _ltx2_euler_ancestral_step(
        sample,
        denoised,
        sigma,
        sigma_next,
        noise=None,
        eta=0.0,
    )

    torch.testing.assert_close(actual, expected)


def test_ancestral_noise_stream_is_reproducible_and_video_first() -> None:
    seed = 17 + ANCESTRAL_NOISE_SEED_OFFSET
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video_noise = torch.randn((1, 2), generator=generator)
    audio_noise = torch.randn((1, 3), generator=generator)

    replay = torch.Generator(device="cpu").manual_seed(seed)
    torch.testing.assert_close(video_noise, torch.randn((1, 2), generator=replay))
    torch.testing.assert_close(audio_noise, torch.randn((1, 3), generator=replay))


def test_keyframe_mask_marks_every_token_in_first_causal_latent_frame() -> None:
    mask = _ltx2_first_frame_keyframes_mask(
        batch_size=2,
        token_count=12,
        latent_frames=3,
        device=torch.device("cpu"),
    )

    expected = torch.zeros((2, 12, 1))
    expected[:, :4] = 1.0
    torch.testing.assert_close(mask, expected)
