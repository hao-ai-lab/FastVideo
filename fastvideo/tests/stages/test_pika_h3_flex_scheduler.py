# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex's re-noise scheduler step."""

import torch

from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler


def _scheduler() -> MiniMaxH3Scheduler:
    scheduler = MiniMaxH3Scheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True)
    scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.0], device="cpu")
    scheduler.set_noise_generator(torch.Generator().manual_seed(7))
    return scheduler


def test_renoised_step_uses_the_next_sigma_and_fresh_noise() -> None:
    scheduler = _scheduler()
    sample = torch.zeros(2, 3)
    velocity = torch.ones(2, 3)
    timestep = scheduler.timesteps[0]
    output = scheduler.step(velocity, timestep, sample).prev_sample

    expected_noise = torch.randn(sample.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float32)
    torch.testing.assert_close(output, 0.5 * torch.ones_like(sample) + 0.5 * expected_noise)


def test_terminal_renoised_step_returns_x0_without_drawing_noise() -> None:
    scheduler = _scheduler()
    _ = scheduler.step(torch.ones(2, 3), scheduler.timesteps[0], torch.zeros(2, 3))
    state_before = scheduler._noise_generator.get_state()
    output = scheduler.step(torch.ones(2, 3), scheduler.timesteps[1], torch.zeros(2, 3)).prev_sample
    state_after = scheduler._noise_generator.get_state()
    torch.testing.assert_close(output, torch.full((2, 3), 0.5))
    assert torch.equal(state_before, state_after)


def test_euler_step_is_unchanged_when_stochastic_sampling_is_off() -> None:
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.0], device="cpu")
    sample = torch.full((2, 3), 2.0)
    velocity = torch.full((2, 3), 4.0)
    output = scheduler.step(velocity, scheduler.timesteps[0], sample).prev_sample
    # x0 = 2 + 1 * 4 = 6; ratio = 0.5, so prev = 0.5 * 2 + 0.5 * 6 = 4.
    torch.testing.assert_close(output, torch.full((2, 3), 4.0))
