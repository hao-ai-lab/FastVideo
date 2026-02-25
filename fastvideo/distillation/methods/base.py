# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch

from fastvideo.distillation.roles import RoleManager


class DistillMethod(torch.nn.Module, ABC):
    def __init__(self, bundle: RoleManager) -> None:
        super().__init__()
        self.bundle = bundle
        self.role_modules = torch.nn.ModuleDict()
        for role, handle in bundle.roles.items():
            if handle.modules:
                self.role_modules[role] = torch.nn.ModuleDict(handle.modules)

    @abstractmethod
    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
        *,
        current_vsa_sparsity: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_optimizers(self, iteration: int) -> Sequence[torch.optim.Optimizer]:
        raise NotImplementedError

    @abstractmethod
    def get_lr_schedulers(self, iteration: int) -> Sequence[Any]:
        raise NotImplementedError

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        del outputs
        grad_accum_rounds = max(1, int(grad_accum_rounds))
        (loss_map["total_loss"] / grad_accum_rounds).backward()

    def optimizers_schedulers_step(self, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            optimizer.step()
        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    def optimizers_zero_grad(self, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()
