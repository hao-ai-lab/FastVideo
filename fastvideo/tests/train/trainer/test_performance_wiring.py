# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Trainer performance-metric wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from fastvideo.train.trainer import Trainer
from fastvideo.train.utils.performance import TrainingPerformanceMonitor
from fastvideo.train.utils.training_config import TrainingConfig


class _FakeTransformer(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            arch_config=SimpleNamespace(
                hidden_size=8,
                ffn_dim=16,
                num_layers=2,
                patch_size=(1, 2, 2),
            ))
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return hidden_states * self.weight


class _FakeRoleModel:

    def __init__(self) -> None:
        self.transformer = _FakeTransformer()


class _DummyTracker:

    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, float], int]] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.logs.append((metrics, step))

    def finish(self) -> None:
        pass


class _PerformanceMethod:

    def __init__(self) -> None:
        self._role_models = {"student": _FakeRoleModel()}
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.tracker = None
        self.pre_loop_forwards = 0

    def set_tracker(self, tracker: Any) -> None:
        self.tracker = tracker

    def on_train_start(self) -> None:
        # Stands in for teacher-cache generation or step-0 validation.
        self._forward()
        self.pre_loop_forwards += 1

    def manages_optimization(self) -> bool:
        return False

    def _forward(self) -> torch.Tensor:
        hidden = torch.ones(1, 3, 2, 4, 4)
        return self._role_models["student"].transformer(hidden_states=hidden)

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, float]]:
        del batch, iteration
        output = self._forward()
        return {"total_loss": output.sum() * 0.0 + 1.0}, {}, {}

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


def _run_trainer(
    monkeypatch: Any,
    *,
    performance_enabled: bool,
) -> tuple[_PerformanceMethod, _DummyTracker]:
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
    cfg.loop.gradient_accumulation_steps = 1
    cfg.performance.enabled = performance_enabled
    trainer = Trainer(cfg)
    method = _PerformanceMethod()

    trainer.run(
        method,
        dataloader=[{
            "sample": "x"
        }],
        max_steps=1,
    )
    return method, tracker


def test_pre_loop_forwards_are_not_counted(monkeypatch) -> None:
    method, tracker = _run_trainer(monkeypatch, performance_enabled=True)

    assert method.pre_loop_forwards == 1
    metrics = tracker.logs[-1][0]
    # Only the forward inside the training step is measured; the forward run
    # during on_train_start() was dropped by the pre-loop reset.
    assert metrics["perf/model_forward_calls"] == 1.0
    assert metrics["perf/role/student/forward_calls"] == 1.0


def test_pre_loop_forwards_are_dropped_before_the_first_step(monkeypatch) -> None:
    observed: list[int] = []
    original_reset = TrainingPerformanceMonitor.reset

    def recording_reset(self: TrainingPerformanceMonitor) -> None:
        observed.append(len(self._work))
        original_reset(self)

    monkeypatch.setattr(TrainingPerformanceMonitor, "reset", recording_reset)
    method, _ = _run_trainer(monkeypatch, performance_enabled=True)

    assert method.pre_loop_forwards == 1
    # The monitor must already be empty when the first training step starts, so
    # a long pre-loop phase (teacher cache generation, step-0 validation)
    # cannot accumulate per-forward records.
    assert observed[-1] == 0


def test_disabled_performance_logs_no_perf_metrics(monkeypatch) -> None:
    _, tracker = _run_trainer(monkeypatch, performance_enabled=False)

    metrics = tracker.logs[-1][0]
    assert not any(key.startswith("perf/") for key in metrics)
