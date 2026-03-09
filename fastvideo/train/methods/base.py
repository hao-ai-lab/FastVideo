# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, cast

import torch

from fastvideo.logger import init_logger
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.checkpoint import _RoleModuleContainer
from fastvideo.training.checkpointing_utils import (
    ModelWrapper,
    OptimizerWrapper,
    SchedulerWrapper,
)

logger = init_logger(__name__)

LogScalar = float | int | torch.Tensor


class TrainingMethod(torch.nn.Module, ABC):
    """Base training method (algorithm layer).

    Subclasses own their role models (student, teacher, critic, …) as
    plain attributes and manage optimizers directly — no ``RoleManager``
    or ``RoleHandle``.

    The constructor receives *role_models* (a ``dict[str, ModelBase]``)
    and a *cfg* object.  It calls ``init_preprocessors`` on the student
    and builds ``self.role_modules`` for FSDP wrapping.

    A single shared CUDA RNG generator (``cuda_generator``) is
    created in :meth:`on_train_start`.  All ``torch.randn`` /
    ``torch.randint`` calls in methods **and** models must use this
    generator instead of relying on global RNG state.
    """

    # Shared CUDA RNG generator (initialized in on_train_start).
    cuda_generator: torch.Generator | None = None

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

    # ------------------------------------------------------------------

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

    def checkpoint_state(self) -> dict[str, Any]:
        """Return DCP-ready checkpoint state for all trainable roles.

        Keys follow the convention:
        ``roles.<role>.<module>``, ``optimizers.<role>``,
        ``schedulers.<role>``, ``random_state.*``.

        EMA state is managed by the ``EMACallback`` and is
        checkpointed through the callback state mechanism.
        """
        states: dict[str, Any] = {}

        for role, model in self._role_models.items():
            if not getattr(model, "_trainable", False):
                continue

            modules: dict[str, torch.nn.Module] = {}
            if model.transformer is not None:
                modules["transformer"] = model.transformer

            container = _RoleModuleContainer(modules)

            for module_name, module in modules.items():
                states[
                    f"roles.{role}.{module_name}"
                ] = ModelWrapper(module)

            opt = self._optimizer_dict.get(role)
            if opt is not None:
                states[
                    f"optimizers.{role}"
                ] = OptimizerWrapper(container, opt)

            sched = self._lr_scheduler_dict.get(role)
            if sched is not None:
                states[
                    f"schedulers.{role}"
                ] = SchedulerWrapper(sched)

        return states

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

    def seed_optimizer_state_for_resume(self) -> None:
        """Seed optimizer state so DCP can load saved state.

        A fresh optimizer has empty state (exp_avg, exp_avg_sq,
        step are only created on the first optimizer.step()).
        DCP needs matching entries to load into; without them
        the saved optimizer state is silently dropped.
        """
        for opt in self.get_optimizers(0):
            for group in opt.param_groups:
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    if len(opt.state.get(p, {})) > 0:
                        continue
                    opt.state[p] = {
                        "step": torch.tensor(0.0),
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p),
                    }

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
        from fastvideo.distributed import (
            get_sp_group,
            get_world_group,
        )
        from fastvideo.utils import set_random_seed

        seed = self.training_config.data.seed
        if seed is None:
            raise ValueError(
                "training.data.seed must be set"
            )
        seed = int(seed)

        world_group = get_world_group()
        global_rank = int(world_group.rank)
        sp_size = int(
            self.training_config.distributed.sp_size
            or 1
        )

        # Ranks within the same SP group share a seed.
        if sp_size > 1:
            sp_group_seed = seed + (
                global_rank // sp_size
            )
        else:
            sp_group_seed = seed + global_rank

        set_random_seed(seed)  

        self.cuda_generator = torch.Generator(
            device=self.student.device
        ).manual_seed(sp_group_seed)

        self.student.on_train_start()

    def get_rng_generators(
        self,
    ) -> dict[str, torch.Generator]:
        generators: dict[str, torch.Generator] = {}
        if self.cuda_generator is not None:
            generators["cuda"] = self.cuda_generator
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
