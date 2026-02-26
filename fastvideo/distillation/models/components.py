# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.distillation.roles import RoleManager


@dataclass(slots=True)
class ModelComponents:
    """Build-time outputs produced by a model plugin.

    A model plugin is responsible for loading modules, constructing a
    role container (`RoleManager`), and assembling shared
    components needed by runtime adapters, dataloaders, validators, and
    trackers.
    """

    training_args: TrainingArgs
    bundle: RoleManager
    adapter: Any
    dataloader: Any
    validator: Any | None = None
    start_step: int = 0
