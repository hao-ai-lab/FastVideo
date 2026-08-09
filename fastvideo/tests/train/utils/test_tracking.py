# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the modular trainer's wandb tracking guard.

The Trainer logs each step's loss dict through ``build_tracker``'s tracker on
rank 0; these tests pin the graceful no-op path (wandb missing / no
credentials) and the logged loss-dict shape with a monkeypatched wandb.
"""

import builtins
import sys
from types import SimpleNamespace

from fastvideo.train.utils import tracking
from fastvideo.train.utils.training_config import (
    CheckpointConfig,
    TrackerConfig,
)
from fastvideo.training.trackers import DummyTracker, WandbTracker


class _FakeRun:

    def __init__(self, **init_kwargs):
        self.init_kwargs = init_kwargs
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))

    def finish(self):
        pass


def _fake_wandb():
    runs = []

    def _init(**kwargs):
        run = _FakeRun(**kwargs)
        runs.append(run)
        return run

    return SimpleNamespace(init=_init, runs=runs, api=SimpleNamespace(api_key="key"))


def test_wandb_usable_false_when_not_importable(monkeypatch) -> None:
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    assert tracking._wandb_usable() is False


def test_wandb_usable_false_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(api=SimpleNamespace(api_key=None)),
    )
    assert tracking._wandb_usable() is False


def test_wandb_usable_with_env_key(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace())
    assert tracking._wandb_usable() is True


def test_build_tracker_noops_cleanly_without_wandb(monkeypatch, tmp_path) -> None:
    """A wandb-project config must degrade to the dummy tracker, not crash."""
    monkeypatch.setattr(tracking, "get_world_group", lambda: SimpleNamespace(rank=0))
    monkeypatch.setattr(tracking, "_wandb_usable", lambda: False)

    tracker = tracking.build_tracker(
        TrackerConfig(project_name="h3-dmd2-vsa"),
        CheckpointConfig(output_dir=str(tmp_path)),
        config={"method": {}},
    )

    assert isinstance(tracker, DummyTracker)
    tracker.log({"total_loss": 1.0}, 1)  # must not raise
    tracker.finish()


def test_build_tracker_logs_loss_dict_with_monkeypatched_wandb(monkeypatch, tmp_path) -> None:
    fake = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.setattr(tracking, "get_world_group", lambda: SimpleNamespace(rank=0))

    run_config = {"method": {"dmd_denoising_steps": [1000, 757, 522]}}
    tracker = tracking.build_tracker(
        TrackerConfig(project_name="h3-dmd2-vsa", run_name="dmd2_vsa0_overfit"),
        CheckpointConfig(output_dir=str(tmp_path)),
        config=run_config,
    )

    assert isinstance(tracker, WandbTracker)
    (run, ) = fake.runs
    assert run.init_kwargs["project"] == "h3-dmd2-vsa"
    assert run.init_kwargs["name"] == "dmd2_vsa0_overfit"
    assert run.init_kwargs["config"] == run_config

    # The per-step dict the Trainer logs on rank 0 (DMD2 loss map + metrics).
    metrics = {
        "total_loss": 0.5,
        "generator_loss": 0.25,
        "fake_score_loss": 0.25,
        "update_student": 1.0,
        "step_time_sec": 0.1,
        "vsa_sparsity": 0.0,
    }
    tracker.log(metrics, 7)
    assert run.logged == [(metrics, 7)]


def test_build_tracker_nonzero_rank_never_inits_wandb(monkeypatch, tmp_path) -> None:
    fake = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.setattr(tracking, "get_world_group", lambda: SimpleNamespace(rank=1))

    tracker = tracking.build_tracker(
        TrackerConfig(project_name="h3-dmd2-vsa"),
        CheckpointConfig(output_dir=str(tmp_path)),
        config=None,
    )

    assert isinstance(tracker, DummyTracker)
    assert fake.runs == []
