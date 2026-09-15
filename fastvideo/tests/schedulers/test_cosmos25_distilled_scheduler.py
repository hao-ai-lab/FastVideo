# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_cosmos25_distilled import (
    Cosmos25DistilledScheduler,
)
from fastvideo.models.registry import ModelRegistry


def test_registry_resolves_scheduler() -> None:
    scheduler_class, architecture = ModelRegistry.resolve_model_cls("Cosmos25DistilledScheduler")
    assert scheduler_class is Cosmos25DistilledScheduler
    assert architecture == "Cosmos25DistilledScheduler"


def test_official_schedule_and_model_timesteps() -> None:
    scheduler = Cosmos25DistilledScheduler()
    scheduler.set_timesteps(4)

    expected_angles = torch.tensor(
        [math.pi / 2, math.atan(15), math.atan(5), math.atan(5 / 3)],
        dtype=torch.float64,
    )
    expected_model_timesteps = torch.tensor(
        [1.0, 15 / 16, 5 / 6, 5 / 8],
        dtype=torch.float64,
    )
    assert_close(scheduler.trigflow_timesteps, expected_angles, rtol=0, atol=0)
    assert_close(scheduler.timesteps, expected_model_timesteps, rtol=1e-15, atol=1e-15)


def test_rollout_matches_official_fixed_noise_equations() -> None:
    scheduler = Cosmos25DistilledScheduler()
    scheduler.set_timesteps(4)
    initial_noise = torch.tensor([[[1.5, -0.25], [0.5, 2.0]]], dtype=torch.float32)
    sample = initial_noise.to(torch.float64)
    expected = sample.clone()

    for index, model_timestep in enumerate(scheduler.timesteps):
        angle = scheduler.trigflow_timesteps[index]
        denominator = torch.cos(angle) + torch.sin(angle)
        c_skip = 1 / denominator
        c_out = -torch.sin(angle) / denominator
        c_in = 1 / denominator

        assert_close(scheduler.scale_model_input(sample, model_timestep), expected * c_in)
        model_output = torch.full_like(sample, 0.125 * (index + 1))
        expected_x0 = c_skip * expected + c_out * model_output
        if index + 1 < len(scheduler.trigflow_timesteps):
            next_angle = scheduler.trigflow_timesteps[index + 1]
            expected = torch.cos(next_angle) * expected_x0 + torch.sin(next_angle) * initial_noise
        else:
            expected = expected_x0

        result = scheduler.step(model_output, model_timestep, sample)
        assert_close(result.pred_original_sample, expected_x0)
        assert_close(result.prev_sample, expected)
        sample = result.prev_sample


def test_reuses_initial_noise_instead_of_generator_noise() -> None:
    initial_noise = torch.tensor([1.0, -2.0], dtype=torch.float32)
    results = []
    for seed in (1, 999):
        scheduler = Cosmos25DistilledScheduler()
        scheduler.set_timesteps(2)
        sample = initial_noise.to(torch.float64)
        generator = torch.Generator().manual_seed(seed)
        for timestep in scheduler.timesteps:
            model_output = torch.full_like(sample, 0.25)
            sample = scheduler.step(model_output, timestep, sample, generator=generator).prev_sample
        results.append(sample)
    assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.mark.parametrize("num_steps", [0, 5])
def test_rejects_schedule_outside_released_checkpoint_range(num_steps: int) -> None:
    scheduler = Cosmos25DistilledScheduler()
    with pytest.raises(ValueError, match="1 to 4 steps"):
        scheduler.set_timesteps(num_steps)


def test_set_timesteps_resets_sampler_state() -> None:
    scheduler = Cosmos25DistilledScheduler()
    scheduler.set_timesteps(1)
    sample = torch.ones(2, dtype=torch.float64)
    scheduler.step(torch.zeros_like(sample), scheduler.timesteps[0], sample)
    assert scheduler.step_index == 1

    scheduler.set_timesteps(1)
    assert scheduler.step_index is None
    assert scheduler._initial_noise is None
