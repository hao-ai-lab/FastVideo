# SPDX-License-Identifier: Apache-2.0
"""Reference parity for the Cosmos Predict2.5 DFD ODE sampler.

Coverage scope: implementation_subcomponent. The upstream RF noise-schedule
operations are executed directly for every transition in the published
four-step schedule and compared with the FastVideo scheduler.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.testing import assert_close

REPO_ROOT = Path(__file__).resolve().parents[3]
DFD_REF_DIR = Path(os.getenv("COSMOS25_DFD_REF_DIR", REPO_ROOT / "DFDReference"))
DFD_TIMESTEPS = (0.999, 0.937, 0.833, 0.624, 0.0)


def _official_rf_schedule():
    if not DFD_REF_DIR.is_dir():
        pytest.skip(f"DFD reference checkout missing: {DFD_REF_DIR}")
    if str(DFD_REF_DIR) not in sys.path:
        sys.path.insert(0, str(DFD_REF_DIR))
    try:
        module = importlib.import_module("fastgen.networks.noise_schedule")
    except Exception as exc:  # noqa: BLE001 - local reference dependencies may be absent.
        pytest.skip(f"Cannot import the DFD RF noise schedule: {exc}")
    return module.RFNoiseSchedule(min_t=0.0, max_t=0.999, num_steps=1000)


def _official_step(schedule, sample, model_output, timestep, next_timestep):
    batch_size = sample.shape[0]
    t = torch.full((batch_size,), timestep, dtype=torch.float64)
    x0 = schedule.flow_to_x0(sample, model_output, t)
    if next_timestep == 0.0:
        return x0
    eps = schedule.x0_to_eps(sample, x0, t)
    t_next = torch.full((batch_size,), next_timestep, dtype=torch.float64)
    return schedule.forward_process(x0, eps, t_next)


def test_dfd_scheduler_matches_upstream_four_step_ode():
    official = _official_rf_schedule()
    try:
        from fastvideo.models.schedulers.scheduling_cosmos25_dfd import Cosmos25DFDScheduler
    except ImportError as exc:
        pytest.skip(f"FastVideo DFD scheduler is not implemented: {exc}")

    actual_scheduler = Cosmos25DFDScheduler()
    actual_scheduler.set_timesteps(4, device="cpu")

    assert_close(
        actual_scheduler.timesteps,
        torch.tensor(DFD_TIMESTEPS[:-1], dtype=torch.float64),
        rtol=0,
        atol=0,
    )

    generator = torch.Generator(device="cpu").manual_seed(123)
    noise = torch.randn((2, 3, 4, 5), generator=generator, dtype=torch.float32)
    expected = official.latents(noise, torch.tensor(DFD_TIMESTEPS[0], dtype=torch.float64))
    actual = actual_scheduler.scale_noise(torch.zeros_like(noise), noise=noise)
    assert_close(actual, expected, rtol=0, atol=0)

    for index, (timestep, next_timestep) in enumerate(zip(DFD_TIMESTEPS[:-1], DFD_TIMESTEPS[1:], strict=True)):
        model_output = torch.tanh(actual * (0.15 + index * 0.07) + timestep)
        expected = _official_step(official, actual, model_output, timestep, next_timestep)
        actual = actual_scheduler.step(
            model_output,
            actual_scheduler.timesteps[index],
            actual,
            return_dict=False,
        )[0]
        assert_close(actual, expected, rtol=1e-6, atol=1e-6)
