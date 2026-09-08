# SPDX-License-Identifier: Apache-2.0
"""TrainRoleBase plumbing: construction + checkpoint registration."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch

from fastvideo.train.methods.base import TrainingMethod
from fastvideo.train.roles.base import TrainRoleBase
from fastvideo.train.utils.training_config import (
    DataConfig,
    DistributedConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingLoopConfig,
)


class _FakeLMRole(TrainRoleBase):
    """Non-diffusion role registering a plain ``model`` module."""

    def __init__(self, trainable: bool = True) -> None:
        self._trainable = trainable
        self.model = torch.nn.Linear(4, 4)
        if not trainable:
            self.model.requires_grad_(False)

    def checkpoint_modules(self):
        return {"model": self.model}


class _FakeDiffusionLikeRole(TrainRoleBase):
    """Diffusion-style role registering a ``transformer`` module."""

    def __init__(self) -> None:
        self._trainable = True
        self.transformer = torch.nn.Linear(8, 8)
        self.device_checked = False

    def checkpoint_modules(self):
        return {"transformer": self.transformer}


class _TinyMethod(TrainingMethod):
    """Minimal concrete method for role-registration assertions."""

    def single_train_step(self, batch, iteration):
        raise NotImplementedError

    def get_optimizers(self, iteration):
        return []

    def get_lr_schedulers(self, iteration):
        return []

    @property
    def _optimizer_dict(self):
        return {}

    @property
    def _lr_scheduler_dict(self):
        return {}


def _make_cfg():
    return SimpleNamespace(
        training=TrainingConfig(
            distributed=DistributedConfig(),
            data=DataConfig(seed=0),
            optimizer=OptimizerConfig(),
            loop=TrainingLoopConfig(max_train_steps=1),
        ),
        method={},
        validation={},
    )


def test_role_modules_registered_from_checkpoint_modules():
    roles = {
        "student": _FakeDiffusionLikeRole(),
        "refiner": _FakeLMRole(),
    }
    method = _TinyMethod(cfg=_make_cfg(), role_models=roles)
    assert set(method.role_modules.keys()) == {"student", "refiner"}
    assert "transformer" in method.role_modules["student"]
    assert "model" in method.role_modules["refiner"]


def test_checkpoint_state_covers_trainable_roles_and_modules():
    roles = {
        "student": _FakeDiffusionLikeRole(),
        "refiner": _FakeLMRole(),
        "frozen_helper": _FakeLMRole(trainable=False),
    }
    method = _TinyMethod(cfg=_make_cfg(), role_models=roles)
    states = method.checkpoint_state()
    assert "roles.student.transformer" in states
    assert "roles.refiner.model" in states
    # Frozen roles are excluded from the trainable-role checkpoint path.
    assert not any(key.startswith("roles.frozen_helper") for key in states)


def test_trainable_parameters_default_collection():
    role = _FakeLMRole()
    params = role.trainable_parameters()
    assert params
    assert all(p.requires_grad for p in params)
    frozen = _FakeLMRole(trainable=False)
    assert frozen.trainable_parameters() == []


def test_builder_accepts_train_role_base():
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.config import RunConfig

    cfg = RunConfig(
        models={
            "student": {"_target_": "tests.local_tests.test_promptrl_roles._FakeDiffusionLikeRole"},
            "refiner": {"_target_": "tests.local_tests.test_promptrl_roles._FakeLMRole"},
        },
        method={"_target_": "tests.local_tests.test_promptrl_roles._TinyMethod"},
        training=_make_cfg().training,
        callbacks={},
        raw={},
    )
    _, method, _, _ = build_from_config(cfg)
    # Pytest may import this file as ``test_promptrl_roles`` while the
    # config target resolves the package-qualified module. Compare against
    # the class loaded through the same canonical path as the builder.
    expected_type = importlib.import_module(
        "tests.local_tests.test_promptrl_roles")._TinyMethod
    assert isinstance(method, expected_type)
    assert set(method._role_models) == {"student", "refiner"}


def test_builder_rejects_non_role_targets():
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.config import RunConfig
    import pytest

    cfg = RunConfig(
        models={"student": {
            "_target_": "torch.nn.Linear",
            "in_features": 2,
            "out_features": 2,
        }},
        method={"_target_": "tests.local_tests.test_promptrl_roles._TinyMethod"},
        training=_make_cfg().training,
        callbacks={},
        raw={},
    )
    with pytest.raises(TypeError, match="TrainRoleBase"):
        build_from_config(cfg)
