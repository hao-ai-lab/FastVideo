# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_cosmos25_dfd import Cosmos25DFDScheduler
from fastvideo.models.registry import ModelRegistry


def test_registry_resolves_scheduler():
    scheduler_class, architecture = ModelRegistry.resolve_model_cls("Cosmos25DFDScheduler")
    assert scheduler_class is Cosmos25DFDScheduler
    assert architecture == "Cosmos25DFDScheduler"


def test_fixed_schedule_and_initial_noise_scale():
    scheduler = Cosmos25DFDScheduler()
    scheduler.set_timesteps(4, device="cpu")

    assert_close(
        scheduler.timesteps,
        torch.tensor([0.999, 0.937, 0.833, 0.624], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    assert_close(
        scheduler.sigmas,
        torch.tensor([0.999, 0.937, 0.833, 0.624, 0.0], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    assert scheduler.init_noise_sigma == 0.999


@pytest.mark.parametrize("steps", [1, 2, 3, 5, 35])
def test_rejects_non_checkpoint_step_counts(steps):
    with pytest.raises(ValueError, match="exactly 4"):
        Cosmos25DFDScheduler().set_timesteps(steps)


def test_ode_step_matches_rectified_flow_euler_update():
    scheduler = Cosmos25DFDScheduler()
    scheduler.set_timesteps(4)
    sample = torch.tensor([1.25, -0.75], dtype=torch.float32)
    flow = torch.tensor([0.4, -0.2], dtype=torch.float32)

    expected = sample + (0.937 - 0.999) * flow
    actual = scheduler.step(flow, scheduler.timesteps[0], sample, return_dict=False)[0]

    assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_final_step_returns_predicted_x0():
    scheduler = Cosmos25DFDScheduler()
    scheduler.set_timesteps(4)
    sample = torch.tensor([1.25, -0.75], dtype=torch.float32)
    flow = torch.tensor([0.4, -0.2], dtype=torch.float32)

    for index in range(3):
        sample = scheduler.step(flow, scheduler.timesteps[index], sample, return_dict=False)[0]

    result = scheduler.step(flow, scheduler.timesteps[3], sample)
    expected_x0 = sample - 0.624 * flow
    assert_close(result.prev_sample, expected_x0, rtol=1e-6, atol=1e-6)
    assert_close(result.pred_original_sample, expected_x0, rtol=1e-6, atol=1e-6)


def test_scale_noise_matches_upstream_initial_latents():
    scheduler = Cosmos25DFDScheduler()
    clean = torch.tensor([2.0], dtype=torch.float32)
    noise = torch.tensor([-1.0], dtype=torch.float32)

    expected = 0.999 * noise
    assert_close(scheduler.scale_noise(clean, noise=noise), expected)
