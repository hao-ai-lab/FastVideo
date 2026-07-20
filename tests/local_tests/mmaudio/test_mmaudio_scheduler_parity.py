# SPDX-License-Identifier: Apache-2.0
"""MMAudio flow-matching scheduler reuse parity."""

from __future__ import annotations

import torch


def test_mmaudio_euler_schedule_matches_fastvideo_flow_scheduler() -> None:
    from mmaudio.model.flow_matching import FlowMatching

    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    num_steps = 25
    official = FlowMatching(min_sigma=0.0, inference_mode="euler", num_steps=num_steps)
    fastvideo = FlowMatchEulerDiscreteScheduler(
        shift=1.0,
        invert_sigmas=True,
        sigma_min=0.0,
        use_reference_discrete_timesteps=True,
    )
    fastvideo.set_timesteps(num_steps, device="cpu")
    initial = torch.randn((2, 7, 4), generator=torch.Generator().manual_seed(1234))

    def flow(time: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        time = torch.as_tensor(time, dtype=sample.dtype, device=sample.device)
        return torch.tanh(sample * 0.125 + time)

    expected = official.to_data(flow, initial.clone())
    actual = initial.clone()
    for timestep in fastvideo.timesteps:
        model_output = flow(timestep / fastvideo.config.num_train_timesteps, actual)
        actual = fastvideo.step(model_output, timestep, actual).prev_sample

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
