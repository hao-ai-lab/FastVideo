# SPDX-License-Identifier: Apache-2.0
"""Cosmos Predict2.5 distilled sampler parity (implementation_subcomponent)."""

import math

import torch
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_cosmos25_distilled import (
    Cosmos25DistilledScheduler,
)
from tests.local_tests.cosmos25._reference import load_official_scaling_module


def test_rectified_flow_preconditioning_matches_official() -> None:
    official_module = load_official_scaling_module()
    official = official_module.RectifiedFlow_sCMWrapper(sigma_data=1.0)
    actual = Cosmos25DistilledScheduler(sigma_data=1.0)
    times = torch.tensor(
        [math.pi / 2, math.atan(15), math.atan(5), math.atan(5 / 3), 0.37],
        dtype=torch.float32,
    ).view(1, 1, -1, 1, 1)

    for actual_value, expected_value in zip(actual._scalings(times), official(times), strict=True):
        assert_close(actual_value, expected_value, rtol=0, atol=0)


def test_four_step_x0_rollout_matches_official() -> None:
    official_module = load_official_scaling_module()
    official = official_module.RectifiedFlow_sCMWrapper(sigma_data=1.0)
    scheduler = Cosmos25DistilledScheduler(sigma_data=1.0)
    scheduler.set_timesteps(4)

    generator = torch.Generator().manual_seed(20260827)
    initial_noise = torch.randn((1, 2, 3, 2, 2), generator=generator, dtype=torch.float32)
    expected = initial_noise.to(torch.float64)
    actual = expected.clone()

    for index, model_timestep in enumerate(scheduler.timesteps):
        angle = scheduler.trigflow_timesteps[index]
        official_coefficients = official(angle)
        c_skip, c_out, c_in, c_noise = official_coefficients

        expected_model_input = (expected * c_in).float()
        expected_model_output = expected_model_input * 0.125 + c_noise.float()
        expected_x0 = c_skip * expected + c_out * expected_model_output
        if index + 1 < len(scheduler.trigflow_timesteps):
            next_angle = scheduler.trigflow_timesteps[index + 1]
            expected = torch.cos(next_angle) * expected_x0 + torch.sin(next_angle) * initial_noise
        else:
            expected = expected_x0

        actual_model_input = scheduler.scale_model_input(actual, model_timestep).float()
        actual_model_output = actual_model_input * 0.125 + model_timestep.float()
        result = scheduler.step(actual_model_output, model_timestep, actual)

        assert_close(actual_model_input, expected_model_input, rtol=0, atol=0)
        assert_close(result.pred_original_sample, expected_x0, rtol=0, atol=0)
        assert_close(result.prev_sample, expected, rtol=0, atol=0)
        actual = result.prev_sample
