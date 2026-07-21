# SPDX-License-Identifier: Apache-2.0
"""CPU-only integration tests for Trainer validation hook dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from fastvideo.train.callbacks.validation import ValidationCallback
from fastvideo.train.methods import base as method_base
from fastvideo.train.trainer import Trainer
from fastvideo.train.utils.training_config import TrainingConfig


class _RecordingValidationCallback(ValidationCallback):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_calls: list[int] = []

    def _run_validation(self, method: Any, step: int) -> None:  # type: ignore[override]
        del method
        self.run_calls.append(step)


class _DummyTracker:

    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, float], int]] = []
        self.finished = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.logs.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


class _DummyMethod:

    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.train_start_calls = 0
        self.zero_grad_steps: list[int] = []
        self.optimizer_steps: list[int] = []
        self.backward_calls = 0
        self.gradient_sync_calls: list[bool] = []
        self.tracker = None

    def set_tracker(self, tracker: Any) -> None:
        self.tracker = tracker

    def on_train_start(self) -> None:
        self.train_start_calls += 1

    def manages_optimization(self) -> bool:
        return False

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, float]]:
        assert batch["sample"] == "x"
        loss = self.weight * 0.0 + 1.0
        return {"total_loss": loss}, {}, {"iteration_seen": float(iteration)}

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int,
    ) -> None:
        del outputs
        self.backward_calls += 1
        (loss_map["total_loss"] / grad_accum_rounds).backward()

    def set_requires_gradient_sync(self, requires_gradient_sync: bool) -> None:
        self.gradient_sync_calls.append(requires_gradient_sync)

    def optimizers_schedulers_step(self, iteration: int) -> None:
        self.optimizer_steps.append(iteration)

    def optimizers_zero_grad(self, iteration: int) -> None:
        self.zero_grad_steps.append(iteration)
        self.weight.grad = None


def test_trainer_runs_validation_callback_during_training(
    monkeypatch,
) -> None:
    tracker = _DummyTracker()
    group = SimpleNamespace(rank=0, local_rank=0, rank_in_group=0, world_size=1)

    monkeypatch.setattr("fastvideo.train.trainer.get_world_group", lambda: group)
    monkeypatch.setattr("fastvideo.train.trainer.get_sp_group", lambda: group)
    monkeypatch.setattr(
        "fastvideo.train.callbacks.validation.get_world_group",
        lambda: group,
    )
    monkeypatch.setattr(
        "fastvideo.train.callbacks.validation.get_sp_group",
        lambda: group,
    )
    monkeypatch.setattr(
        "fastvideo.train.trainer.build_tracker",
        lambda *args, **kwargs: tracker,
    )

    cfg = TrainingConfig()
    cfg.tracker.project_name = ""
    cfg.loop.gradient_accumulation_steps = 1
    callback_configs = {
        "validation": {
            "_target_": f"{__name__}._RecordingValidationCallback",
            "pipeline_target": "unused.pipeline.Target",
            "dataset_file": "unused.json",
            "every_steps": 2,
        }
    }
    trainer = Trainer(
        cfg,
        callback_configs=callback_configs,
    )
    method = _DummyMethod()

    trainer.run(
        method,
        dataloader=[{"sample": "x"}],
        max_steps=3,
    )

    validation = trainer.callbacks._callbacks["validation"]
    assert isinstance(validation, _RecordingValidationCallback)
    assert validation.run_calls == [0, 2]
    assert method.train_start_calls == 1
    assert method.backward_calls == 3
    assert method.gradient_sync_calls == []
    assert method.zero_grad_steps == [0, 1, 2, 3]
    assert method.optimizer_steps == [1, 2, 3]
    assert [step for _, step in tracker.logs] == [1, 2, 3]
    assert tracker.finished is True


def test_trainer_syncs_fsdp_gradients_only_on_final_accumulation(
    monkeypatch,
) -> None:
    tracker = _DummyTracker()
    group = SimpleNamespace(rank=0, local_rank=0, rank_in_group=0, world_size=1)

    monkeypatch.setattr("fastvideo.train.trainer.get_world_group", lambda: group)
    monkeypatch.setattr("fastvideo.train.trainer.get_sp_group", lambda: group)
    monkeypatch.setattr(
        "fastvideo.train.trainer.build_tracker",
        lambda *args, **kwargs: tracker,
    )

    cfg = TrainingConfig()
    cfg.tracker.project_name = ""
    cfg.loop.gradient_accumulation_steps = 3
    trainer = Trainer(cfg)
    method = _DummyMethod()

    trainer.run(
        method,
        dataloader=[{"sample": "x"}],
        max_steps=1,
    )

    assert method.backward_calls == 3
    assert method.gradient_sync_calls == [False, False, True]
    assert method.optimizer_steps == [1]


def test_training_method_forwards_gradient_sync_to_trainable_fsdp_roots(
    monkeypatch,
) -> None:

    class _FakeFSDP(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, bool, bool | None]] = []

        def set_requires_gradient_sync(self, enabled: bool, *, recurse: bool) -> None:
            self.calls.append(("sync", enabled, recurse))

        def set_is_last_backward(self, enabled: bool) -> None:
            self.calls.append(("last", enabled, None))

    monkeypatch.setattr(method_base, "FSDPModule", _FakeFSDP)
    trainable = _FakeFSDP()
    frozen = _FakeFSDP()
    owner = SimpleNamespace(
        _role_models={
            "student": SimpleNamespace(transformer=trainable, _trainable=True),
            "teacher": SimpleNamespace(transformer=frozen, _trainable=False),
        }
    )

    method_base.TrainingMethod.set_requires_gradient_sync(owner, False)

    assert trainable.calls == [("sync", False, True), ("last", False, None)]
    assert frozen.calls == []
