# SPDX-License-Identifier: Apache-2.0
"""DreamX-World default Flow scheduler parity.

Coverage scope: implementation_subcomponent. DreamX-World-5B-Cam defaults to
Diffusers FlowMatchEulerDiscreteScheduler for sampler_name=Flow. This test
checks that FastVideo's native FlowMatchEulerDiscreteScheduler matches the
timestep schedule and Euler step used by the official default path.
"""
from __future__ import annotations

from pathlib import Path
import inspect

import torch
from diffusers import FlowMatchEulerDiscreteScheduler as OfficialFlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler as FastVideoFlowMatchEulerDiscreteScheduler,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_SCOPE = "implementation_subcomponent"


def _scheduler_kwargs(cls):
    config_path = REPO_ROOT / "DreamX-World" / "configs" / "wan2.2" / "wan_ti2v_5b.yaml"
    config = OmegaConf.load(config_path)
    raw_kwargs = OmegaConf.to_container(config["scheduler_kwargs"])
    signature = inspect.signature(cls)
    return {key: value for key, value in raw_kwargs.items() if key in signature.parameters}


def test_dreamx_world_flow_scheduler_timesteps_and_step_match():
    official = OfficialFlowMatchEulerDiscreteScheduler(**_scheduler_kwargs(OfficialFlowMatchEulerDiscreteScheduler))
    fastvideo = FastVideoFlowMatchEulerDiscreteScheduler(**_scheduler_kwargs(FastVideoFlowMatchEulerDiscreteScheduler))

    official.set_timesteps(50, device="cpu", mu=1)
    fastvideo.set_timesteps(50, device="cpu", mu=1)
    assert_close(fastvideo.timesteps, official.timesteps, atol=0, rtol=0)
    assert_close(fastvideo.sigmas, official.sigmas, atol=0, rtol=0)

    torch.manual_seed(7)
    sample = torch.randn(1, 4, 2, 8, 8)
    model_output = torch.randn_like(sample)
    timestep = official.timesteps[3]

    official_prev = official.step(model_output, timestep, sample, return_dict=False)[0]
    fastvideo_prev = fastvideo.step(model_output, fastvideo.timesteps[3], sample, return_dict=False)[0]
    diff = (official_prev - fastvideo_prev).abs()
    print(f"scheduler diff_max={diff.max().item():.8f} diff_mean={diff.mean().item():.8f}")
    assert_close(fastvideo_prev, official_prev, atol=0, rtol=0)
