# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from typing import Any, Literal, cast

import torch

from fastvideo.logger import init_logger
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.checkpoint import _RoleModuleContainer
from fastvideo.training.checkpointing_utils import (
    ModelWrapper,
    OptimizerWrapper,
    RandomStateWrapper,
    SchedulerWrapper,
)
from fastvideo.training.training_utils import EMA_FSDP

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
        self._use_ema: bool = bool(
            self.method_config.get("use_ema", False)
        )
        self._ema_decay: float = float(
            self.method_config.get("ema_decay", 0.9999)
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

        self._setup_ema()

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def _setup_ema(self) -> None:
        """Create ``EMA_FSDP`` shadow of student transformer.

        Uses the legacy local-shard approach: each rank keeps an
        fp32 CPU copy of its own FSDP shard.  Updates are purely
        local (no all-gather).  For inference, EMA weights are
        swapped into the live FSDP model via a context manager.
        """
        self.generator_ema: EMA_FSDP | None = None
        if not self._use_ema:
            return
        logger.info(
            "Initializing EMA (local_shard) with "
            "decay=%s from student transformer",
            self._ema_decay,
        )
        self.generator_ema = EMA_FSDP(
            self.student.transformer,
            decay=self._ema_decay,
            mode="local_shard",
        )

    @contextlib.contextmanager
    def ema_context(
        self,
    ) -> Generator[torch.nn.Module, None, None]:
        """Context manager: temporarily apply EMA weights.

        Swaps local EMA shards into the FSDP-wrapped student
        transformer, yields it, then restores originals.
        If EMA is disabled, yields the student transformer
        unchanged.
        """
        transformer = self.student.transformer
        if self.generator_ema is not None:
            with self.generator_ema.apply_to_model(
                transformer,
            ):
                yield transformer
        else:
            yield transformer

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

        EMA is maintained as local FSDP shards on CPU
        (``EMA_FSDP``), not as a separate ``nn.Module``, so
        it is not included here.
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

        # RNG states.
        states["random_state"] = RandomStateWrapper(None)
        for name, gen in (
            self.get_rng_generators() or {}
        ).items():
            if gen is not None:
                states[
                    f"random_state.{name}"
                ] = RandomStateWrapper(gen)

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
