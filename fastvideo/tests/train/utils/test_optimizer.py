# SPDX-License-Identifier: Apache-2.0
"""Tests for FastVideo modular trainer optimizer construction."""

import pytest
import torch

from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler,
    seed_adamw_parameter_state,
)
from fastvideo.train.utils.training_config import OptimizerConfig, TrainingLoopConfig
from fastvideo.training.checkpointing_utils import SchedulerWrapper


def test_multistep_with_warmup_matches_mmaudio_scheduler_composition() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    config = OptimizerConfig(
        learning_rate=1e-4,
        lr_scheduler="multistep_with_warmup",
        lr_warmup_steps=2,
        lr_milestones=(3, 5),
        lr_gamma=0.1,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        params=[parameter],
        optimizer_config=config,
        loop_config=TrainingLoopConfig(max_train_steps=10),
        learning_rate=config.learning_rate,
        betas=config.betas,
        scheduler_name=config.lr_scheduler,
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
    initial_lr = optimizer.param_groups[0]["lr"]
    assert initial_lr == pytest.approx(1e-4 / 3)

    values = []
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])

    assert values[1] == pytest.approx(1e-4)
    assert values[4] == pytest.approx(1e-5)
    assert values[6] == pytest.approx(1e-6)


def test_multistep_with_warmup_requires_milestones() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    with pytest.raises(ValueError, match="lr_milestones"):
        build_optimizer_and_scheduler(
            params=[parameter],
            optimizer_config=OptimizerConfig(
                learning_rate=1e-4,
                lr_scheduler="multistep_with_warmup",
                lr_warmup_steps=2,
            ),
            loop_config=TrainingLoopConfig(max_train_steps=10),
            learning_rate=1e-4,
            betas=(0.9, 0.95),
            scheduler_name="multistep_with_warmup",
        )


@pytest.mark.parametrize("fused", [False, True])
def test_seed_adamw_state_uses_expected_step_device(fused: bool) -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], fused=fused)
    seed_adamw_parameter_state(optimizer, parameter)
    expected = parameter.device if fused else torch.device("cpu")
    assert optimizer.state[parameter]["step"].device == expected


def test_seed_adamw_state_rejects_unowned_parameter() -> None:
    owned = torch.nn.Parameter(torch.zeros(()))
    unowned = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([owned])
    with pytest.raises(ValueError, match="unowned"):
        seed_adamw_parameter_state(optimizer, unowned)


def test_scheduler_wrapper_stabilizes_multistep_milestone_keys() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    config = OptimizerConfig(
        learning_rate=1e-4,
        lr_scheduler="multistep_with_warmup",
        lr_warmup_steps=2,
        lr_milestones=(3, 5),
    )
    _, scheduler = build_optimizer_and_scheduler(
        params=[parameter],
        optimizer_config=config,
        loop_config=TrainingLoopConfig(max_train_steps=10),
        learning_rate=config.learning_rate,
        betas=config.betas,
        scheduler_name=config.lr_scheduler,
    )
    wrapper = SchedulerWrapper(scheduler)
    state = wrapper.state_dict()
    milestones = state["scheduler"]["_schedulers"][1]["milestones"]
    assert milestones == {"3": 1, "5": 1}

    # Model what DCP returns after flattening mapping keys to strings.
    wrapper.load_state_dict(state)
    restored = scheduler.state_dict()["_schedulers"][1]["milestones"]
    assert restored == {3: 1, 5: 1}
