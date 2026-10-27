# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch
from torch.testing import assert_close

from tests.local_tests.minimax_h3._reference import REFERENCE_SRC, assert_pinned_reference, assert_reference_source

assert_pinned_reference(
    "src/diffusers/schedulers/scheduling_minimax_h3.py",
    "b6fe1b29d5bc7f134213c175817803f569db77de433480e13f4c71c50e4b7758",
)
sys.path.insert(0, str(REFERENCE_SRC))

from diffusers import MiniMaxH3Scheduler as ReferenceScheduler  # noqa: E402

assert_reference_source(ReferenceScheduler, "src/diffusers/schedulers/scheduling_minimax_h3.py")

from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler  # noqa: E402


@pytest.mark.parametrize("shift", [3.0, 12.0])
def test_generated_schedule_and_steps_match_reference(shift: float) -> None:
    reference = ReferenceScheduler(shift=shift)
    actual = MiniMaxH3Scheduler(shift=shift)
    reference.set_timesteps(30)
    actual.set_timesteps(30)
    assert_close(actual.sigmas, reference.sigmas, rtol=0, atol=0)
    assert_close(actual.timesteps, reference.timesteps, rtol=0, atol=0)
    assert actual.num_inference_steps == reference.num_inference_steps == 29

    generator = torch.Generator().manual_seed(123)
    sample = torch.randn((2, 3, 4), generator=generator, dtype=torch.float32)
    for reference_timestep, actual_timestep in zip(reference.timesteps, actual.timesteps, strict=True):
        velocity = torch.randn(sample.shape, generator=generator, dtype=sample.dtype)
        reference_sample = reference.step(velocity, reference_timestep, sample).prev_sample
        actual_sample = actual.step(velocity, actual_timestep, sample).prev_sample
        assert_close(actual_sample, reference_sample, rtol=0, atol=0)
        sample = reference_sample


def test_explicit_schedule_noise_and_half_precision_match_reference() -> None:
    sigmas = [1.0, 0.5, 0.125, 0.0]
    reference = ReferenceScheduler(shift=12.0)
    actual = MiniMaxH3Scheduler(shift=12.0)
    reference.set_timesteps(sigmas=sigmas)
    actual.set_timesteps(sigmas=sigmas)

    sample = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    noise = torch.tensor([[0.5, 3.0]], dtype=torch.bfloat16)
    assert_close(actual.scale_noise(sample, 0.999, noise), reference.scale_noise(sample, 0.999, noise), rtol=0, atol=0)

    velocity = torch.tensor([[0.25, -0.75]], dtype=torch.bfloat16)
    expected = reference.step(velocity, reference.timesteps[1], sample).prev_sample
    result = actual.step(velocity, actual.timesteps[1], sample).prev_sample
    assert result.dtype == torch.bfloat16
    assert_close(result, expected, rtol=0, atol=0)
