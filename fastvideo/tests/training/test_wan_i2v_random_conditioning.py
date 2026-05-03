# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.testing import assert_close

from fastvideo.fastvideo_args import TrainingArgs
from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.wan_i2v_training_pipeline import WanI2VTrainingPipeline


def _build_pipeline(
    min_condition_latents: int = 1,
    max_condition_latents: int = 1,
    num_latent_t: int = 5,
) -> WanI2VTrainingPipeline:
    pipeline = WanI2VTrainingPipeline.__new__(WanI2VTrainingPipeline)
    pipeline.training_args = TrainingArgs(
        model_path="dummy-model",
        num_latent_t=num_latent_t,
        min_condition_latents=min_condition_latents,
        max_condition_latents=max_condition_latents,
    )
    pipeline.noise_gen_cuda = torch.Generator(device="cpu").manual_seed(11)
    return pipeline


def test_sample_condition_latent_counts_inclusive_range() -> None:
    pipeline = _build_pipeline(
        min_condition_latents=1,
        max_condition_latents=5,
        num_latent_t=13,
    )

    counts = pipeline._sample_condition_latent_counts(
        batch_size=256,
        device=torch.device("cpu"),
    )

    assert counts.min().item() >= 1
    assert counts.max().item() <= 5
    assert set(counts.tolist()).issubset({1, 2, 3, 4, 5})


def test_apply_random_context_conditioning_clean_prefix_and_masks() -> None:
    pipeline = _build_pipeline(num_latent_t=5)
    pipeline._sample_condition_latent_counts = (  # type: ignore[method-assign]
        lambda batch_size, device: torch.tensor([1, 3], device=device)
    )

    latents = torch.arange(20, dtype=torch.float32).view(2, 2, 5, 1, 1)
    noisy_model_input = torch.full_like(latents, -1.0)
    training_batch = TrainingBatch(
        latents=latents,
        noisy_model_input=noisy_model_input,
    )

    training_batch = pipeline._apply_random_context_conditioning(training_batch)

    expected_noisy = noisy_model_input.clone()
    expected_noisy[0, :, :1] = latents[0, :, :1]
    expected_noisy[1, :, :3] = latents[1, :, :3]
    assert_close(training_batch.noisy_model_input, expected_noisy)

    expected_condition = torch.zeros_like(latents, dtype=torch.bfloat16)
    expected_condition[0, :, :1] = latents[0, :, :1].to(torch.bfloat16)
    expected_condition[1, :, :3] = latents[1, :, :3].to(torch.bfloat16)
    assert_close(training_batch.image_latents, expected_condition)

    assert training_batch.mask_lat_size is not None
    assert training_batch.mask_lat_size.shape == (2, 4, 5, 1, 1)
    assert training_batch.mask_lat_size.dtype == torch.bfloat16
    assert_close(
        training_batch.mask_lat_size[0, :, :1],
        torch.ones(4, 1, 1, 1, dtype=torch.bfloat16),
    )
    assert_close(
        training_batch.mask_lat_size[0, :, 1:],
        torch.zeros(4, 4, 1, 1, dtype=torch.bfloat16),
    )
    assert_close(
        training_batch.mask_lat_size[1, :, :3],
        torch.ones(4, 3, 1, 1, dtype=torch.bfloat16),
    )
    assert_close(
        training_batch.mask_lat_size[1, :, 3:],
        torch.zeros(4, 2, 1, 1, dtype=torch.bfloat16),
    )

    assert training_batch.generation_loss_mask is not None
    assert training_batch.generation_loss_mask.shape == (2, 1, 5, 1, 1)
    assert training_batch.generation_loss_mask[0, 0, :1].tolist() == [[[False]]]
    assert training_batch.generation_loss_mask[0, 0, 1:].all()
    assert not training_batch.generation_loss_mask[1, 0, :3].any()
    assert training_batch.generation_loss_mask[1, 0, 3:].all()


def test_masked_loss_ignores_conditioned_prefix() -> None:
    target = torch.zeros(1, 2, 4, 1, 1)
    model_pred = torch.zeros_like(target)
    model_pred[:, :, :2] = 100.0
    training_batch = TrainingBatch(
        generation_loss_mask=torch.tensor(
            [[[[[False]], [[False]], [[True]], [[True]]]]]
        )
    )

    loss = TrainingPipeline._compute_loss(model_pred, target, training_batch)

    assert_close(loss, torch.tensor(0.0))


def test_masked_loss_normalizes_over_generated_suffix() -> None:
    target = torch.zeros(1, 2, 4, 1, 1)
    model_pred = torch.zeros_like(target)
    model_pred[:, :, 2:] = 2.0
    training_batch = TrainingBatch(
        generation_loss_mask=torch.tensor(
            [[[[[False]], [[False]], [[True]], [[True]]]]]
        )
    )

    loss = TrainingPipeline._compute_loss(model_pred, target, training_batch)

    assert_close(loss, torch.tensor(4.0))


@pytest.mark.parametrize(
    "min_condition_latents,max_condition_latents,num_latent_t",
    [
        (0, 1, 13),
        (3, 2, 13),
        (1, 13, 13),
    ],
)
def test_invalid_condition_latent_args_raise(
    min_condition_latents: int,
    max_condition_latents: int,
    num_latent_t: int,
) -> None:
    training_args = TrainingArgs(
        model_path="dummy-model",
        num_latent_t=num_latent_t,
        min_condition_latents=min_condition_latents,
        max_condition_latents=max_condition_latents,
    )

    with pytest.raises(ValueError):
        WanI2VTrainingPipeline._validate_condition_latent_args(training_args)
