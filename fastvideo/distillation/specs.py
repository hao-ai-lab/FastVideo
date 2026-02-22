# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

RoleName = str


@dataclass(slots=True)
class DistillSpec:
    """Selects the model family + distillation method.

    This is intentionally small: everything else (roles, training args, and
    pipeline config) lives in the run config.
    """

    model: str
    method: str


@dataclass(slots=True)
class RoleSpec:
    """Describes a role's model source and whether it should be trained."""

    family: str
    path: str
    trainable: bool = True

