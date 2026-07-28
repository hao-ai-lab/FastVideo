# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the float64 schedule memoization in fastvideo.models.utils."""

import pytest
import torch

from fastvideo.models.utils import (
    _FLOAT64_SCHEDULE_ATTR,
    get_float64_schedule,
    pred_noise_to_pred_video,
    pred_noise_to_x_bound,
)


class FakeScheduler:
    """Minimal stand-in exposing the sigma and timestep tables the helper reads."""

    def __init__(self, num_steps: int = 4) -> None:
        self.set_timesteps(num_steps)

    def set_timesteps(self, num_steps: int) -> None:
        """Rebuild the schedule tables the way a real scheduler does per call."""
        self.timesteps = torch.linspace(1000.0, 0.0, num_steps)
        self.sigmas = self.timesteps / 1000.0


def _reference_schedule(scheduler: FakeScheduler,
                        device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the schedule the way the call sites did before memoization."""
    return (scheduler.sigmas.double().to(device),
            scheduler.timesteps.double().to(device))


def test_repeated_call_returns_cached_tensors():
    """A second call for the same schedule reuses the same tensor objects."""
    scheduler = FakeScheduler()
    device = torch.device("cpu")
    sigmas_first, timesteps_first = get_float64_schedule(scheduler, device)
    sigmas_second, timesteps_second = get_float64_schedule(scheduler, device)
    assert sigmas_first is sigmas_second
    assert timesteps_first is timesteps_second


def test_cached_values_match_direct_conversion():
    """Memoized tables are bitwise-equal to converting the schedule directly."""
    scheduler = FakeScheduler()
    device = torch.device("cpu")
    sigmas, timesteps = get_float64_schedule(scheduler, device)
    sigmas_reference, timesteps_reference = _reference_schedule(scheduler, device)
    assert torch.equal(sigmas, sigmas_reference)
    assert torch.equal(timesteps, timesteps_reference)


def test_tables_are_float64():
    """Both tables are promoted to float64 regardless of the source dtype."""
    scheduler = FakeScheduler()
    scheduler.sigmas = scheduler.sigmas.to(torch.float32)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32)
    sigmas, timesteps = get_float64_schedule(scheduler, torch.device("cpu"))
    assert sigmas.dtype == torch.float64
    assert timesteps.dtype == torch.float64


def test_set_timesteps_invalidates_cache():
    """Swapping the schedule tables rebuilds them instead of serving stale ones."""
    scheduler = FakeScheduler(num_steps=4)
    device = torch.device("cpu")
    sigmas_before, _ = get_float64_schedule(scheduler, device)

    scheduler.set_timesteps(8)
    sigmas_after, timesteps_after = get_float64_schedule(scheduler, device)

    assert sigmas_after is not sigmas_before
    assert sigmas_after.shape[0] == 8
    sigmas_reference, timesteps_reference = _reference_schedule(scheduler, device)
    assert torch.equal(sigmas_after, sigmas_reference)
    assert torch.equal(timesteps_after, timesteps_reference)


def test_cache_is_stored_on_the_scheduler():
    """The memo lives on the scheduler, so separate schedulers stay independent."""
    first, second = FakeScheduler(), FakeScheduler()
    device = torch.device("cpu")
    assert getattr(first, _FLOAT64_SCHEDULE_ATTR, None) is None

    first_sigmas, _ = get_float64_schedule(first, device)
    assert getattr(first, _FLOAT64_SCHEDULE_ATTR, None) is not None
    assert getattr(second, _FLOAT64_SCHEDULE_ATTR, None) is None

    second_sigmas, _ = get_float64_schedule(second, device)
    assert second_sigmas is not first_sigmas


def _reference_pred_noise_to_pred_video(pred_noise: torch.Tensor,
                                        noise_input_latent: torch.Tensor,
                                        timestep: torch.Tensor,
                                        scheduler: FakeScheduler) -> torch.Tensor:
    """Reproduce the pre-memoization conversion for a numerical parity check."""
    dtype = pred_noise.dtype
    device = pred_noise.device
    pred_noise = pred_noise.double().to(device)
    noise_input_latent = noise_input_latent.double().to(device)
    sigmas, timesteps = _reference_schedule(scheduler, device)
    timestep_id = torch.argmin(
        (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    return (noise_input_latent - sigma_t * pred_noise).to(dtype)


@pytest.mark.parametrize("num_steps", [2, 4, 8])
def test_pred_noise_to_pred_video_matches_reference(num_steps):
    """Memoizing the schedule leaves the converted latents bitwise unchanged."""
    torch.manual_seed(0)
    scheduler = FakeScheduler(num_steps=num_steps)
    timestep = scheduler.timesteps[:min(3, num_steps)].clone()
    batch_size = timestep.numel()
    pred_noise = torch.randn(batch_size, 4, 8, 8)
    noise_input_latent = torch.randn(batch_size, 4, 8, 8)

    expected = _reference_pred_noise_to_pred_video(pred_noise, noise_input_latent,
                                                   timestep, scheduler)
    actual = pred_noise_to_pred_video(pred_noise, noise_input_latent, timestep,
                                      scheduler)
    assert torch.equal(actual, expected)


def test_pred_noise_to_pred_video_is_stable_across_repeated_calls():
    """Reusing a cached schedule across steps keeps results identical."""
    torch.manual_seed(0)
    scheduler = FakeScheduler()
    pred_noise = torch.randn(2, 4, 8, 8)
    noise_input_latent = torch.randn(2, 4, 8, 8)
    timestep = scheduler.timesteps[:2].clone()

    first = pred_noise_to_pred_video(pred_noise, noise_input_latent, timestep,
                                     scheduler)
    second = pred_noise_to_pred_video(pred_noise, noise_input_latent, timestep,
                                      scheduler)
    assert torch.equal(first, second)


def test_pred_noise_to_x_bound_uses_the_same_schedule():
    """The boundary variant reads the memoized tables without altering them."""
    torch.manual_seed(0)
    scheduler = FakeScheduler()
    pred_noise = torch.randn(2, 4, 8, 8)
    noise_input_latent = torch.randn(2, 4, 8, 8)
    timestep = scheduler.timesteps[:2].clone()
    boundary_timestep = torch.full_like(timestep, float(scheduler.timesteps[-1]))

    sigmas_before, _ = get_float64_schedule(scheduler, pred_noise.device)
    result = pred_noise_to_x_bound(pred_noise, noise_input_latent, timestep,
                                   boundary_timestep, scheduler)
    sigmas_after, _ = get_float64_schedule(scheduler, pred_noise.device)

    assert sigmas_after is sigmas_before
    assert result.shape == pred_noise.shape
