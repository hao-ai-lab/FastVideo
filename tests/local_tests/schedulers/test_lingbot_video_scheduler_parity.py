# SPDX-License-Identifier: Apache-2.0
"""LingBot-Video Flow-UniPC scheduler parity.

Coverage scope: implementation_subcomponent. The official repository vendors
the scheduler used by every Dense, MoE, and refiner denoising loop. These tests
compare its schedule and concrete multistep outputs with FastVideo's native
scheduler under the released checkpoint configuration.
"""
from __future__ import annotations

import json

import torch
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler as FastVideoFlowUniPCMultistepScheduler, )
from lingbot_video.scheduling_flow_unipc import (
    FlowUniPCMultistepScheduler as OfficialFlowUniPCMultistepScheduler, )
from tests.local_tests.lingbot_video.hf_assets import OFFICIAL_DENSE, download_components

PARITY_SCOPE = "implementation_subcomponent"


def _scheduler_kwargs() -> dict[str, object]:
    """Load the exact scheduler constructor arguments released with Dense."""
    model_path = download_components(OFFICIAL_DENSE, "scheduler")
    config_path = model_path / "scheduler" / "scheduler_config.json"
    config = json.loads(config_path.read_text())
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _scheduler_pair() -> tuple[OfficialFlowUniPCMultistepScheduler, FastVideoFlowUniPCMultistepScheduler]:
    """Construct independent official and FastVideo schedulers from one config."""
    kwargs = _scheduler_kwargs()
    return OfficialFlowUniPCMultistepScheduler(**kwargs), FastVideoFlowUniPCMultistepScheduler(**kwargs)


def test_lingbot_video_scheduler_timesteps_and_sigmas_match() -> None:
    """Require exact released 40-step timestep and sigma schedules."""
    official, fastvideo = _scheduler_pair()
    official.set_timesteps(40, device="cpu", shift=3.0)
    fastvideo.set_timesteps(40, device="cpu", shift=3.0)

    assert_close(fastvideo.timesteps, official.timesteps, atol=0, rtol=0)
    assert_close(fastvideo.sigmas, official.sigmas, atol=0, rtol=0)


def test_lingbot_video_scheduler_multistep_outputs_match() -> None:
    """Compare four deterministic updates so UniPC history is exercised."""
    official, fastvideo = _scheduler_pair()
    official.set_timesteps(4, device="cpu", shift=3.0)
    fastvideo.set_timesteps(4, device="cpu", shift=3.0)
    generator = torch.Generator(device="cpu").manual_seed(42)
    official_sample = torch.randn((1, 16, 3, 4, 4), generator=generator)
    fastvideo_sample = official_sample.clone()

    for index in range(4):
        model_generator = torch.Generator(device="cpu").manual_seed(100 + index)
        model_output = torch.randn(official_sample.shape, generator=model_generator)
        official_sample = official.step(model_output, official.timesteps[index], official_sample, return_dict=False)[0]
        fastvideo_sample = fastvideo.step(model_output, fastvideo.timesteps[index], fastvideo_sample,
                                          return_dict=False)[0]
        assert_close(fastvideo_sample, official_sample, atol=0, rtol=0)
