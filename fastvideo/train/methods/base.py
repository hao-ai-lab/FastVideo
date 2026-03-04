# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, cast

import torch

from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.validation import (
    is_validation_enabled,
    parse_validation_dataset_file,
    parse_validation_every_steps,
    parse_validation_guidance_scale,
    parse_validation_num_frames,
    parse_validation_ode_solver,
    parse_validation_output_dir,
    parse_validation_rollout_mode,
    parse_validation_sampler_kind,
    parse_validation_sampling_steps,
)
from fastvideo.train.validators.base import ValidationRequest

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
        student = self._role_models.get("student")
        if student is None:
            return
        validator = getattr(student, "validator", None)
        if validator is None:
            return
        if hasattr(validator, "set_tracker"):
            validator.set_tracker(tracker)
        elif hasattr(validator, "tracker"):
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

    def on_train_start(self) -> None:
        self.student.on_train_start()

    def get_rng_generators(
        self,
    ) -> dict[str, torch.Generator]:
        generators: dict[str, torch.Generator] = {}

        student_gens = self.student.get_rng_generators()
        generators.update(student_gens)

        if is_validation_enabled(self.validation_config):
            validation_gen = (
                self.student.validator.validation_random_generator
            )
            if isinstance(validation_gen, torch.Generator):
                generators["validation_cpu"] = validation_gen

        return generators

    def log_validation(self, iteration: int) -> None:
        if not is_validation_enabled(self.validation_config):
            return

        every_steps = parse_validation_every_steps(
            self.validation_config,
        )
        if every_steps <= 0:
            return
        if iteration % every_steps != 0:
            return

        request = self._build_validation_request()
        self.student.validator.log_validation(
            iteration, request=request,
        )

    def _build_validation_request(self) -> ValidationRequest:
        """Build the ``ValidationRequest`` for validation.

        Override in subclasses that need custom parameters (e.g.
        ``sampling_timesteps`` for DMD2).
        """
        vc = self.validation_config
        sampling_steps = parse_validation_sampling_steps(vc)
        guidance_scale = parse_validation_guidance_scale(vc)
        sampler_kind = parse_validation_sampler_kind(
            vc, default="ode",
        )
        ode_solver = parse_validation_ode_solver(
            vc, sampler_kind=sampler_kind,
        )
        return ValidationRequest(
            sample_handle=self.student,
            dataset_file=parse_validation_dataset_file(vc),
            sampling_steps=sampling_steps,
            sampler_kind=sampler_kind,
            rollout_mode=parse_validation_rollout_mode(vc),
            ode_solver=ode_solver,
            sampling_timesteps=None,
            guidance_scale=guidance_scale,
            num_frames=parse_validation_num_frames(vc),
            output_dir=parse_validation_output_dir(vc),
        )

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
