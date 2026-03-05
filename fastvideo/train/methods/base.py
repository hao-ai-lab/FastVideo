# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, cast

import torch

from fastvideo.train.models.base import ModelBase

LogScalar = float | int | torch.Tensor


class TrainingMethod(torch.nn.Module, ABC):
    """Base training method (algorithm layer).

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
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__()
        self.tracker: Any | None = None
        self._role_models: dict[str, ModelBase] = dict(role_models)

        self.student = role_models["student"]
        self.training_config = cfg.training
        self.method_config: dict[str, Any] = dict(cfg.method)
        self.validation_config: dict[str, Any] = dict(
            getattr(cfg, "validation", {}) or {}
        )

        # Build nn.ModuleDict for FSDP / checkpoint visibility.
        self.role_modules = torch.nn.ModuleDict()
        for role, model in role_models.items():
            mods: dict[str, torch.nn.Module] = {}
            transformer = getattr(model, "transformer", None)
            if isinstance(transformer, torch.nn.Module):
                mods["transformer"] = transformer
            if mods:
                self.role_modules[role] = torch.nn.ModuleDict(mods)

    def set_tracker(self, tracker: Any) -> None:
        self.tracker = tracker

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
        self, iteration: int,
    ) -> Sequence[torch.optim.Optimizer]:
        raise NotImplementedError

    @abstractmethod
    def get_lr_schedulers(
        self, iteration: int,
    ) -> Sequence[Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def _optimizer_dict(self) -> dict[str, Any]:
        ...

    @property
    @abstractmethod
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        ...

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

    def optimizers_schedulers_step(
        self, iteration: int,
    ) -> None:
        for optimizer in self.get_optimizers(iteration):
            optimizer.step()
        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    def optimizers_zero_grad(
        self, iteration: int,
    ) -> None:
        for optimizer in self.get_optimizers(iteration):
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()

    # -- Shared hooks (override in subclasses as needed) --

    def get_grad_clip_targets(
        self, iteration: int,
    ) -> dict[str, torch.nn.Module]:
        """Return modules whose gradients should be clipped.

        Override in subclasses to add/conditionally include
        modules (e.g. critic, conditionally student).
        Default: student transformer.
        """
        return {"student": self.student.transformer}

    def on_train_start(self) -> None:
        self.student.on_train_start()

    def get_rng_generators(
        self,
    ) -> dict[str, torch.Generator]:
        generators: dict[str, torch.Generator] = {}

        student_gens = self.student.get_rng_generators()
        generators.update(student_gens)

        return generators

    @staticmethod
    def _parse_attn_kind(
        raw: Any,
    ) -> Literal["dense", "vsa"]:
        if raw in (None, ""):
            return "dense"
        kind = str(raw).strip().lower()
        if kind not in {"dense", "vsa"}:
            raise ValueError(
                "method_config.attn_kind must be one of "
                f"{{'dense', 'vsa'}}, got {raw!r}."
            )
        return cast(Literal["dense", "vsa"], kind)
