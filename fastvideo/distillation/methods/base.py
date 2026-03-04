# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch

from fastvideo.distillation.models.base import ModelBase

LogScalar = float | int | torch.Tensor


class DistillMethod(torch.nn.Module, ABC):
    """Base distillation method (algorithm layer).

    Subclasses own their role models (student, teacher, critic, …) as
    plain attributes and manage optimizers directly — no ``RoleManager``
    or ``RoleHandle``.

    The constructor receives *role_models* (a ``dict[str, ModelBase]``)
    and a *cfg* object.  It calls ``init_preprocessors`` on the student
    and builds ``self.role_modules`` for FSDP wrapping.
    """

    def __init__(
        self,
        *,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__()
        self.tracker: Any | None = None
        # Build nn.ModuleDict for FSDP / checkpoint visibility.
        self.role_modules = torch.nn.ModuleDict()
        for role, model in role_models.items():
            mods: dict[str, torch.nn.Module] = {}
            transformer = getattr(model, "transformer", None)
            if isinstance(transformer, torch.nn.Module):
                mods["transformer"] = transformer
            transformer_2 = getattr(model, "transformer_2", None)
            if isinstance(transformer_2, torch.nn.Module):
                mods["transformer_2"] = transformer_2
            if mods:
                self.role_modules[role] = torch.nn.ModuleDict(mods)

    def set_tracker(self, tracker: Any) -> None:
        self.tracker = tracker
        validator = getattr(self, "validator", None)
        if validator is None:
            return
        set_tracker = getattr(validator, "set_tracker", None)
        if callable(set_tracker):
            set_tracker(tracker)
            return
        if hasattr(validator, "tracker"):
            validator.tracker = tracker  # type: ignore[attr-defined]

    @abstractmethod
    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
        *,
        current_vsa_sparsity: float = 0.0,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, Any],
        dict[str, LogScalar],
    ]:
        raise NotImplementedError

    @abstractmethod
    def get_optimizers(
        self, iteration: int
    ) -> Sequence[torch.optim.Optimizer]:
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
