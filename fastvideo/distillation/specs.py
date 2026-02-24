# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

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
