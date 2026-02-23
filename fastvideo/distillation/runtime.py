# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastvideo.fastvideo_args import TrainingArgs

from fastvideo.distillation.bundle import ModelBundle
from fastvideo.distillation.methods.base import DistillMethod


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
    start_step: int = 0


@dataclass(slots=True)
class DistillRuntime:
    """Fully assembled runtime for `DistillTrainer.run()`."""

    training_args: TrainingArgs
    method: DistillMethod
    dataloader: Any
    tracker: Any
    start_step: int = 0

