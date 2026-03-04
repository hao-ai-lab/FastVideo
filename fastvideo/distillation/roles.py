# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

RoleName = str


@dataclass(slots=True)
class RoleHandle:
    modules: dict[str, torch.nn.Module] = field(default_factory=dict)
    optimizers: dict[str, torch.optim.Optimizer] = field(default_factory=dict)
    lr_schedulers: dict[str, Any] = field(default_factory=dict)
    trainable: bool = True

    def require_module(self, name: str) -> torch.nn.Module:
        if name not in self.modules:
            raise KeyError(f"Missing module '{name}'")
        return self.modules[name]


@dataclass(slots=True)
class RoleManager:
    roles: dict[RoleName, RoleHandle]

    def require_roles(self, roles: list[RoleName]) -> None:
        missing = [role for role in roles if role not in self.roles]
        if missing:
            raise KeyError(f"Missing roles: {missing}")

    def role(self, role: RoleName) -> RoleHandle:
        if role not in self.roles:
            raise KeyError(f"Unknown role: {role}")
        return self.roles[role]
