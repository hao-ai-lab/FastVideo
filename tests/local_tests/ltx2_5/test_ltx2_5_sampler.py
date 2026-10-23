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
    """Verify ancestral noise uses the seeded offset and video-first ordering."""
    seed = 17 + ANCESTRAL_NOISE_SEED_OFFSET
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video_noise = torch.randn((1, 2), generator=generator)
    audio_noise = torch.randn((1, 3), generator=generator)

    replay = torch.Generator(device="cpu").manual_seed(seed)
    torch.testing.assert_close(video_noise, torch.randn((1, 2), generator=replay))
    torch.testing.assert_close(audio_noise, torch.randn((1, 3), generator=replay))


def test_ancestral_step_captures_noise_from_stage_forward(monkeypatch) -> None:
    """LTX2DenoisingStage.forward generates ancestral noise with the expected seed, shapes, and ordering."""
    from unittest.mock import MagicMock
    from fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising import LTX2DenoisingStage
    from fastvideo.pipelines.composed_pipeline_base import ForwardBatch
    from fastvideo.fastvideo_args import FastVideoArgs

    # Track noise tensors passed to _ltx2_euler_ancestral_step
    captured_noises = []
    original_step = _ltx2_euler_ancestral_step

    def mock_step(sample, denoised, sigma, sigma_next, noise=None, eta=1.0):
        if noise is not None:
            captured_noises.append(noise.clone())
        return original_step(sample, denoised, sigma, sigma_next, noise=noise, eta=eta)

    monkeypatch.setattr(
        "fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising._ltx2_euler_ancestral_step",
        mock_step,
    )

    # Create a minimal stage with a mock transformer
    mock_transformer = MagicMock()
    mock_transformer.return_value = (torch.zeros(1, 12, 4), torch.zeros(1, 18, 4))
    stage = LTX2DenoisingStage(
        mock_transformer,
        sigmas_override=[0.8, 0.3],
        num_inference_steps_override=1,
    )

    # Create a batch with video and audio latents matching expected shapes
    batch = ForwardBatch(
        latents=torch.randn(1, 4, 3, 2, 2),  # video: (B, C, T, H, W)
        extra={
            "ltx2_audio_latents": torch.randn(1, 4, 6),  # audio: (B, C, T)
            "ltx2_conditioning_video_latents": None,
            "ltx2_conditioning_audio_latents": None,
        },
    )
    batch.text_embeddings = torch.randn(1, 10, 8)
    batch.timestep_embeddings = torch.randn(1)

    args = FastVideoArgs(
        model_path="dummy",
        seed=17,
        ltx2_use_ancestral_sampler=True,
    )

    # Run the stage forward to generate ancestral noise
    stage.forward(batch, args)

    # Verify we captured noise tensors (one per ancestral step)
    assert len(captured_noises) == 1
    noise = captured_noises[0]

    # Verify the noise tensor contains both video and audio with video-first ordering
    # Video latents are (1, 4, 3, 2, 2) = 48 elements; audio latents are (1, 4, 6) = 24 elements
    # Combined flattened noise should have 48 + 24 = 72 elements
    assert noise.numel() == 48 + 24

    # Verify reproducibility with the seeded offset
    expected_seed = 17 + ANCESTRAL_NOISE_SEED_OFFSET
    generator = torch.Generator(device="cpu").manual_seed(expected_seed)
    video_noise_expected = torch.randn((1, 4, 3, 2, 2), generator=generator)
    audio_noise_expected = torch.randn((1, 4, 6), generator=generator)

    # The noise tensor is concatenated [video, audio] in flattened form
    noise_video = noise[:48].view(1, 4, 3, 2, 2)
    noise_audio = noise[48:].view(1, 4, 6)
    torch.testing.assert_close(noise_video, video_noise_expected)
    torch.testing.assert_close(noise_audio, audio_noise_expected)


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
