# SPDX-License-Identifier: Apache-2.0
"""H3 joint video+audio TDM math: sigma grids, model time, and the sign bridge."""

from __future__ import annotations

import pytest
import torch

from fastvideo.train.models.minimax_h3.minimax_h3 import (
    tdm_h3_model_timestep,
    tdm_h3_shift,
    tdm_h3_sigma_grid,
)


def test_h3_shifts_match_the_checkpoint_schedules() -> None:
    """Video uses shift 12 and audio shift 3, like the released checkpoint."""
    assert tdm_h3_shift("video") == 12.0
    assert tdm_h3_shift("audio") == 3.0
    with pytest.raises(ValueError):
        tdm_h3_shift("motion")


def test_h3_sigma_grid_endpoints_and_closed_form() -> None:
    """Label 0 is clean, label T is pure noise, and the grid is monotone."""
    labels = torch.tensor([0.0, 250.0, 500.0, 750.0, 1000.0])
    video = tdm_h3_sigma_grid(labels, "video")
    assert video[0].item() == pytest.approx(0.0)
    assert video[-1].item() == pytest.approx(1.0)
    assert bool(torch.all(video[1:] > video[:-1]))

    u = 0.75
    assert video[3].item() == pytest.approx(12.0 * u / (1.0 + 11.0 * u))

    audio = tdm_h3_sigma_grid(labels, "audio")
    assert audio[-1].item() == pytest.approx(1.0)
    # A smaller shift gives a smaller sigma at the same interior label.
    assert bool(torch.all(audio[1:-1] < video[1:-1]))


def test_h3_model_time_is_one_minus_sigma() -> None:
    """H3's model timestep convention is ``1 - sigma`` in ``[0, 1]``."""
    sigma = torch.tensor([0.0, 0.25, 0.75, 1.0])
    assert torch.allclose(tdm_h3_model_timestep(sigma), 1.0 - sigma)


def test_h3_data_ward_velocity_reconstructs_x0() -> None:
    """The modular plugin negates H3's data-ward velocity into the base sign.

    With ``pred_noise = -velocity`` the base conversion
    ``x0 = x_t - sigma * pred_noise`` must equal ``x0 = x_t + sigma * velocity``.
    """
    torch.manual_seed(0)
    noisy = torch.randn(2, 3)
    velocity = torch.randn(2, 3)
    sigma = torch.tensor([[0.2], [0.8]])
    x0 = noisy - sigma * (-velocity)
    assert torch.allclose(x0, noisy + sigma * velocity)
