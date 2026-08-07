# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for modular RL sampling."""

from types import SimpleNamespace

import torch

from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler, )
from fastvideo.train.methods.rl.common.sampling import (
    DiffusionSampler, SamplingConfig)


def test_flow_unipc_accepts_list_sigmas() -> None:
    sampler = DiffusionSampler(
        SamplingConfig(scheduler="flow_unipc", sigmas=[1.0, 0.5]))
    model = SimpleNamespace(noise_scheduler=SimpleNamespace(shift=1.0))

    scheduler = sampler._prepare_scheduler(model, torch.device("cpu"))

    assert isinstance(scheduler, FlowUniPCMultistepScheduler)
    assert scheduler.timesteps.tolist() == [1000, 500]
    assert scheduler.sigmas.tolist() == [1.0, 0.5, 0.0]
