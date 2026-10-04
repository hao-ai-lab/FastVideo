# SPDX-License-Identifier: Apache-2.0
"""CPU-only integration tests for Trainer validation hook dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastvideo.train.callbacks.validation import ValidationCallback
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
        self.synchronize_gradient_steps: list[int] = []
        self.backward_calls = 0
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

    def optimizers_schedulers_step(self, iteration: int) -> None:
        self.optimizer_steps.append(iteration)

    def synchronize_gradients(self, iteration: int) -> None:
        self.synchronize_gradient_steps.append(iteration)

    def optimizers_zero_grad(self, iteration: int) -> None:
        self.zero_grad_steps.append(iteration)
        self.weight.grad = None


def test_trainer_runs_validation_callback_during_training(monkeypatch, ) -> None:
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
        dataloader=[{
            "sample": "x"
        }],
        max_steps=3,
    )

    validation = trainer.callbacks._callbacks["validation"]
    assert isinstance(validation, _RecordingValidationCallback)
    assert validation.run_calls == [0, 2]
    assert method.train_start_calls == 1
    assert method.backward_calls == 3
    assert method.synchronize_gradient_steps == [1, 2, 3]
    assert method.zero_grad_steps == [0, 1, 2, 3]
    assert method.optimizer_steps == [1, 2, 3]
    assert [step for _, step in tracker.logs] == [1, 2, 3]
    assert tracker.finished is True


class _RecordingCheckpointManager:

    def __init__(self) -> None:
        self.maybe_save_steps: list[int] = []
        self.save_final_steps: list[int] = []

    def maybe_resume(self, *, resume_from_checkpoint: str | None) -> None:
        del resume_from_checkpoint
        return None

    def maybe_save(self, step: int) -> None:
        self.maybe_save_steps.append(step)

    def save_final(self, step: int) -> None:
        self.save_final_steps.append(step)

    def load_rng_snapshot(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def test_trainer_defers_final_step_interval_save_to_save_final(monkeypatch, ) -> None:
    """The final step must be saved after validation, not on the interval.

    Otherwise validation advances the checkpointed callback RNG (and any
    global RNG it consumes) after the interval save, and ``save_final``
    dedupes against that stale snapshot.
    """
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
            "every_steps": 3,
        }
    }
    trainer = Trainer(
        cfg,
        callback_configs=callback_configs,
    )
    method = _DummyMethod()
    checkpoint_manager = _RecordingCheckpointManager()

    trainer.run(
        method,
        dataloader=[{
            "sample": "x"
        }],
        max_steps=3,
        checkpoint_manager=checkpoint_manager,
    )

    assert checkpoint_manager.maybe_save_steps == [1, 2]
    assert checkpoint_manager.save_final_steps == [3]
    validation = trainer.callbacks._callbacks["validation"]
    assert isinstance(validation, _RecordingValidationCallback)
    assert validation.run_calls == [0, 3]


class _RaisingValidationCallback(_RecordingValidationCallback):

    def _run_validation(self, method: Any, step: int) -> None:  # type: ignore[override]
        del method
        self.run_calls.append(step)
        raise RuntimeError("validation boom")


def test_trainer_saves_final_checkpoint_when_validation_raises(monkeypatch, ) -> None:
    """A crash in final-step validation must not discard the terminal save.

    The final-step interval save is deferred to ``save_final`` so the
    terminal checkpoint captures post-validation callback RNG. That
    deferral must not drop the run's last unsaved interval when
    validation aborts before ``save_final`` runs.
    """
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
            "_target_": f"{__name__}._RaisingValidationCallback",
            "pipeline_target": "unused.pipeline.Target",
            "dataset_file": "unused.json",
            "every_steps": 3,
            "run_at_start": False,
        }
    }
    trainer = Trainer(
        cfg,
        callback_configs=callback_configs,
    )
    method = _DummyMethod()
    checkpoint_manager = _RecordingCheckpointManager()

    with pytest.raises(RuntimeError, match="validation boom"):
        trainer.run(
            method,
            dataloader=[{
                "sample": "x"
            }],
            max_steps=3,
            checkpoint_manager=checkpoint_manager,
        )

    assert checkpoint_manager.maybe_save_steps == [1, 2]
    assert checkpoint_manager.save_final_steps == [3]
