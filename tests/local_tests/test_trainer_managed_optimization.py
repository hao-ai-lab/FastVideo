from dataclasses import dataclass, field

import torch

from fastvideo.train.trainer import Trainer
from fastvideo.train.utils.training_config import (
    CheckpointConfig,
    DistributedConfig,
    ModelTrainingConfig,
    OptimizerConfig,
    TrackerConfig,
    TrainingConfig,
    TrainingLoopConfig,
)


class _Tracker:

    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, metrics, step):
        self.logged.append((step, metrics))

    def finish(self):
        self.finished = True


class _Callbacks:

    def __init__(self):
        self.before_optimizer_steps = 0
        self.training_step_ends = 0

    def on_train_start(self, method, iteration=0):
        pass

    def on_before_optimizer_step(self, method, iteration=0):
        self.before_optimizer_steps += 1

    def on_training_step_end(self, method, metrics, iteration=0):
        self.training_step_ends += 1

    def on_validation_begin(self, method, iteration=0):
        pass

    def on_validation_end(self, method, iteration=0):
        pass

    def on_train_end(self, method, iteration=0):
        pass


class _World:
    rank = 0
    local_rank = 0


class _Method(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.backward_calls = 0
        self.optimizer_steps = 0
        self.tracker = None

    def set_tracker(self, tracker):
        self.tracker = tracker

    def on_train_start(self):
        pass

    def manages_optimization(self):
        return True

    def managed_train_step(self, data_stream, iteration):
        batch = next(data_stream)
        self.calls += 1
        return (
            {"total_loss": torch.tensor(float(batch["x"]))},
            {},
            {"managed_metric": float(iteration)},
        )

    def backward(self, *args, **kwargs):
        self.backward_calls += 1

    def optimizers_schedulers_step(self, iteration):
        self.optimizer_steps += 1

    def optimizers_zero_grad(self, iteration):
        pass


class _MethodWithValidation(_Method):

    def __init__(self):
        super().__init__()
        self.validation_iterations = []

    def on_validation_begin(self, iteration=0):
        self.validation_iterations.append(iteration)
        return {"validation/fake": float(iteration)}


def test_trainer_skips_default_optimizer_path_for_managed_methods(monkeypatch):
    monkeypatch.setattr("fastvideo.train.trainer.get_world_group", lambda: _World())
    monkeypatch.setattr("fastvideo.train.trainer.get_sp_group", lambda: _World())
    monkeypatch.setattr("fastvideo.train.trainer.build_tracker", lambda *args, **kwargs: _Tracker())

    cfg = TrainingConfig(
        distributed=DistributedConfig(),
        optimizer=OptimizerConfig(),
        loop=TrainingLoopConfig(max_train_steps=1, gradient_accumulation_steps=3),
        checkpoint=CheckpointConfig(),
        tracker=TrackerConfig(trackers=[]),
        model=ModelTrainingConfig(),
    )
    trainer = Trainer(cfg)
    trainer.callbacks = _Callbacks()
    method = _Method()
    dataloader = [{"x": 2}]

    trainer.run(method, dataloader=dataloader, max_steps=1)

    assert method.calls == 1
    assert method.backward_calls == 0
    assert method.optimizer_steps == 0
    assert trainer.callbacks.before_optimizer_steps == 0
    assert trainer.callbacks.training_step_ends == 1
    assert trainer.tracker.logged[0][1]["total_loss"] == 2.0


def test_trainer_logs_method_validation_at_step_zero(monkeypatch):
    monkeypatch.setattr("fastvideo.train.trainer.get_world_group", lambda: _World())
    monkeypatch.setattr("fastvideo.train.trainer.get_sp_group", lambda: _World())
    monkeypatch.setattr("fastvideo.train.trainer.build_tracker", lambda *args, **kwargs: _Tracker())

    cfg = TrainingConfig(
        distributed=DistributedConfig(),
        optimizer=OptimizerConfig(),
        loop=TrainingLoopConfig(max_train_steps=1, gradient_accumulation_steps=1),
        checkpoint=CheckpointConfig(),
        tracker=TrackerConfig(trackers=[]),
        model=ModelTrainingConfig(),
    )
    trainer = Trainer(cfg)
    trainer.callbacks = _Callbacks()
    method = _MethodWithValidation()
    dataloader = [{"x": 2}]

    trainer.run(method, dataloader=dataloader, max_steps=1)

    assert method.validation_iterations == [0, 1]
    assert trainer.tracker.logged[0] == (0, {"validation/fake": 0.0})
