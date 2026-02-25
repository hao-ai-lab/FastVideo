# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastvideo.fastvideo_args import TrainingArgs

from fastvideo.distillation.bundle import ModelBundle
from fastvideo.distillation.methods.base import DistillMethod

RoleName = str


@dataclass(slots=True)
class RecipeSpec:
    """Selects the model family + training method.

    This is intentionally small: everything else (roles, training args, and
    pipeline config) lives in the run config.
    """

    family: str
    method: str


@dataclass(slots=True)
class RoleSpec:
    """Describes a role's model source and whether it should be trained."""

    family: str
    path: str
    trainable: bool = True
    disable_custom_init_weights: bool = False


@dataclass(slots=True)
class FamilyArtifacts:
    """Build-time outputs produced by a model family plugin.

    A family is responsible for loading modules, constructing a `ModelBundle`,
    and assembling shared components needed by runtime adapters, dataloaders,
    validators, and trackers.
    """

    training_args: TrainingArgs
    bundle: ModelBundle
    adapter: Any
    dataloader: Any
    tracker: Any
    validator: Any | None = None
    start_step: int = 0


@dataclass(slots=True)
class DistillRuntime:
    """Fully assembled runtime for `DistillTrainer.run()`."""

    training_args: TrainingArgs
    method: DistillMethod
    dataloader: Any
    tracker: Any
    start_step: int = 0

