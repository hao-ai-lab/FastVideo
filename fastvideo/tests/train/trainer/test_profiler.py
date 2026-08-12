# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for modular Trainer profiler boundaries."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch

from fastvideo.profiler import list_profiler_regions
from fastvideo.train.trainer import Trainer
from fastvideo.train.utils.training_config import TrainingConfig


class _RecordingProfiler:

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    @property
    def has_profiler(self) -> bool:
        return True

    @contextmanager
    def region(self, name: str):
        self.events.append(("enter", name))
        try:
            yield
        finally:
            self.events.append(("exit", name))


class _DummyTracker:

    def log(self, metrics: dict[str, float], step: int) -> None:
        del metrics, step

    def finish(self) -> None:
        pass


class _DummyMethod:

    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def set_tracker(self, tracker: Any) -> None:
        del tracker

    def on_train_start(self) -> None:
        pass

    def manages_optimization(self) -> bool:
        return False

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, float]]:
        del batch, iteration
        return {"total_loss": self.weight.square()}, {}, {}

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int,
    ) -> None:
        del outputs
        (loss_map["total_loss"] / grad_accum_rounds).backward()

    def optimizers_schedulers_step(self, iteration: int) -> None:
        del iteration

    def optimizers_zero_grad(self, iteration: int) -> None:
        del iteration
        self.weight.grad = None


def test_modular_trainer_emits_nested_step_regions(monkeypatch) -> None:
    group = SimpleNamespace(rank=0, local_rank=0, rank_in_group=0, world_size=1)
    profiler = _RecordingProfiler()

    monkeypatch.setattr("fastvideo.train.trainer.get_world_group", lambda: group)
    monkeypatch.setattr("fastvideo.train.trainer.get_sp_group", lambda: group)
    monkeypatch.setattr(
        "fastvideo.train.trainer.build_tracker",
        lambda *args, **kwargs: _DummyTracker(),
    )
    monkeypatch.setattr("fastvideo.profiler._GLOBAL_CONTROLLER", profiler)

    trainer = Trainer(TrainingConfig())
    trainer.run(
        _DummyMethod(),
        dataloader=[{}],
        max_steps=1,
    )

    assert profiler.events == [
        ("enter", "profiler_region_training_train"),
        ("enter", "profiler_region_training_train_one_step"),
        ("enter", "profiler_region_training_dataloader"),
        ("exit", "profiler_region_training_dataloader"),
        ("enter", "profiler_region_training_forward"),
        ("exit", "profiler_region_training_forward"),
        ("enter", "profiler_region_training_backward"),
        ("exit", "profiler_region_training_backward"),
        ("enter", "profiler_region_training_optimizer"),
        ("exit", "profiler_region_training_optimizer"),
        ("enter", "profiler_region_training_callbacks"),
        ("exit", "profiler_region_training_callbacks"),
        ("exit", "profiler_region_training_train_one_step"),
        ("exit", "profiler_region_training_train"),
    ]


def test_modular_training_regions_are_registered() -> None:
    names = {region.name for region in list_profiler_regions()}
    assert {
        "profiler_region_model_loading",
        "profiler_region_training_train",
        "profiler_region_training_train_one_step",
        "profiler_region_training_dataloader",
        "profiler_region_training_forward",
        "profiler_region_training_backward",
        "profiler_region_training_optimizer",
        "profiler_region_training_callbacks",
        "profiler_region_training_save_checkpoint",
        "profiler_region_training_validation",
        "profiler_region_dmd2_student_rollout",
        "profiler_region_dmd2_generator_loss",
        "profiler_region_dmd2_critic_loss",
    } <= names
