# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from fastvideo.train.methods.base import TrainingMethod
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler
from fastvideo.train.utils.training_config import (
    OptimizerConfig,
    TrainingLoopConfig,
)


def test_build_optimizer_forwards_fused_flag() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))

    optimizer, _ = build_optimizer_and_scheduler(
        params=[parameter],
        optimizer_config=OptimizerConfig(fused=True),
        loop_config=TrainingLoopConfig(max_train_steps=1),
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        scheduler_name="constant",
    )

    assert optimizer.defaults["fused"] is True


def test_resume_seed_places_fused_step_with_parameter() -> None:
    parameter = torch.nn.Parameter(torch.empty(1, device="meta"))
    optimizer = torch.optim.AdamW([parameter], fused=True)
    method = SimpleNamespace(get_optimizers=lambda _: [optimizer])

    TrainingMethod.seed_optimizer_state_for_resume(method)

    assert optimizer.state[parameter]["step"].device == parameter.device
    assert optimizer.state[parameter]["step"].dtype == torch.float32


def test_resume_seed_keeps_default_step_on_cpu() -> None:
    parameter = torch.nn.Parameter(torch.empty(1, device="meta"))
    optimizer = torch.optim.AdamW([parameter])
    method = SimpleNamespace(get_optimizers=lambda _: [optimizer])

    TrainingMethod.seed_optimizer_state_for_resume(method)

    assert optimizer.state[parameter]["step"].device.type == "cpu"
