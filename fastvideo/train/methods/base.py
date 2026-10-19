# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import torch

from fastvideo import envs
from fastvideo.logger import init_logger
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.checkpoint import _RoleModuleContainer
from fastvideo.train.utils.optimizer import seed_adamw_parameter_state
from fastvideo.training.checkpointing_utils import (
    ModelWrapper,
    OptimizerWrapper,
    SchedulerWrapper,
)

logger = init_logger(__name__)

LogScalar: TypeAlias = float | int | torch.Tensor


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
        validation = getattr(cfg, "validation", None)
        if validation is None:
            validation = (getattr(cfg, "callbacks", {}) or {}).get("validation", {})
        self.validation_config: dict[str, Any] = dict(validation or {})

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
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        raise NotImplementedError

    @abstractmethod
    def get_optimizers(
        self,
        iteration: int,
    ) -> Sequence[torch.optim.Optimizer]:
        raise NotImplementedError

    @abstractmethod
    def get_lr_schedulers(
        self,
        iteration: int,
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
                states[f"roles.{role}.{module_name}"] = ModelWrapper(module)

            opt = self._optimizer_dict.get(role)
            if opt is not None:
                states[f"optimizers.{role}"] = OptimizerWrapper(container, opt)

            sched = self._lr_scheduler_dict.get(role)
            if sched is not None:
                states[f"schedulers.{role}"] = SchedulerWrapper(sched)

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
        self,
        iteration: int,
    ) -> None:
        for optimizer in self.get_optimizers(iteration):
            optimizer.step()
        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    def optimizers_zero_grad(
        self,
        iteration: int,
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
        for opt in self._optimizer_dict.values():
            for group in opt.param_groups:
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    if len(opt.state.get(p, {})) > 0:
                        continue
                    seed_adamw_parameter_state(opt, p)

    # -- Shared hooks (override in subclasses as needed) --

    def manages_optimization(self) -> bool:
        """Whether the method owns backward/optimizer stepping internally.

        Most methods return loss tensors and let :class:`Trainer` handle
        gradient accumulation, callbacks, optimizer stepping, and scheduler
        stepping. RL-style methods such as DiffusionNFT need to preserve their
        own sample-then-inner-train loop, so they can provide a specific
        ``managed_train_step``.
        """
        return False

    def managed_train_step(
        self,
        data_stream: Any,
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        """Run one method-managed step.

        Subclasses that return ``True`` from :meth:`manages_optimization`
        should override this. The fallback consumes one dataloader batch and
        delegates to ``single_train_step`` so tests can exercise the hook with
        tiny fake methods.
        """
        return self.single_train_step(next(data_stream), iteration)

    def on_validation_begin(self, iteration: int = 0) -> dict[str, LogScalar]:
        """Run method-owned validation, if any.

        Pipeline-style validation should remain in callbacks. Methods that
        intentionally avoid inference pipelines, such as RL methods with their
        own sampler/reward loop, can override this hook and return metrics for
        the trainer to log at ``iteration``.
        """
        del iteration
        return {}

    def get_grad_clip_targets(
        self,
        iteration: int,
    ) -> dict[str, torch.nn.Module]:
        """Return modules whose gradients should be clipped.

        Override in subclasses to add/conditionally include
        modules (e.g. critic, conditionally student).
        Default: student transformer.
        """
        return {"student": self.student.transformer}

    def apply_configured_lrs(self) -> None:
        """Re-apply the config's learning rates to live optimizers/schedulers.

        Called after a checkpoint resume when
        ``training.checkpoint.reset_lr_on_resume`` is set: DCP restores
        param-group ``lr``/``initial_lr`` and scheduler ``base_lrs`` from the
        checkpoint, which would otherwise silently override an LR change in
        the YAML. Methods with per-role LRs override this.
        """
        return None

    def inference_checkpoint_modules(
        self,
        role: str = "student",
    ) -> dict[str, torch.nn.Module]:
        """Return the complete modules that form an inference checkpoint.

        This intentionally does not reuse :meth:`checkpoint_state`: resumable
        state includes optimizers and every trainable role, while inference
        exports select one explicit role and must retain frozen parameters too.
        Methods with a different deployable role contract may override this
        hook; EMA is never selected implicitly.
        """
        model = self._role_models.get(role)
        if model is None:
            raise ValueError(f"Inference checkpoint role {role!r} is not available; "
                             f"known roles: {sorted(self._role_models)}")
        transformer = getattr(model, "transformer", None)
        if not isinstance(transformer, torch.nn.Module):
            raise ValueError(f"Inference checkpoint role {role!r} has no transformer module")
        module_name = str(getattr(model, "transformer_module_type", "transformer"))
        return {module_name: transformer}

    def inference_checkpoint_base_model_path(
        self,
        role: str = "student",
    ) -> str:
        """Return the immutable model directory supplying non-trained parts."""
        model = self._role_models.get(role)
        if model is None:
            raise ValueError(f"Inference checkpoint role {role!r} is not available; "
                             f"known roles: {sorted(self._role_models)}")
        path = str(getattr(model, "_init_from", "") or "")
        if not path:
            raise ValueError(f"Inference checkpoint role {role!r} does not expose its init_from model path")
        return path

    def get_grad_sanitize_targets(
        self,
        iteration: int,
    ) -> dict[str, torch.nn.Module]:
        """Return active modules whose gradients must be made finite.

        Most methods sanitize and clip the same modules. Methods that leave
        an optimizer role unclipped can override this independently.
        """
        return self.get_grad_clip_targets(iteration)

    def set_is_last_backward(
        self,
        iteration: int,
        is_last_backward: bool,
    ) -> None:
        """Mark an FSDP2 accumulation boundary without changing reductions.

        Intermediate backwards still reduce-scatter their gradients. Only
        FSDP's pending-reduction waits and backward-prefetch cleanup are
        deferred. Use the active gradient targets, including roles whose
        gradients are sanitized but deliberately not clipped. Non-FSDP
        modules are unchanged.
        """
        seen: set[int] = set()
        for module in self.get_grad_sanitize_targets(iteration).values():
            if id(module) in seen:
                continue
            seen.add(id(module))
            set_last_backward = getattr(module, "set_is_last_backward", None)
            if callable(set_last_backward):
                set_last_backward(is_last_backward)

    def set_requires_gradient_sync(
        self,
        iteration: int,
        requires_gradient_sync: bool,
    ) -> None:
        """Control FSDP gradient communication for accumulation rounds.

        FSDP2 normally reduces gradients after every backward. During
        gradient accumulation only the final microbatch needs that
        communication; earlier rounds can accumulate local gradients. The
        active gradient targets include roles that are deliberately not clipped.

        Non-FSDP modules do not expose ``set_requires_gradient_sync`` and are
        intentionally left unchanged.
        """
        seen: set[int] = set()
        for module in self.get_grad_sanitize_targets(iteration).values():
            if id(module) in seen:
                continue
            seen.add(id(module))
            set_sync = getattr(module, "set_requires_gradient_sync", None)
            if callable(set_sync):
                set_sync(requires_gradient_sync, recurse=True)
                set_last_backward = getattr(module, "set_is_last_backward", None)
                if callable(set_last_backward):
                    set_last_backward(requires_gradient_sync)

    def supports_gradient_accumulation_no_sync(
        self,
        iteration: int,
    ) -> bool:
        """Whether every active role safely retains unsharded gradients.

        FSDP2 no-sync holds full, unreduced gradients between accumulation
        rounds. Model plugins must opt in explicitly; this prevents the
        optimization from adding roughly one full model's FP32 gradients to
        memory for ordinary full-weight training.
        """
        targets = self.get_grad_sanitize_targets(iteration)
        if not targets:
            return False
        for role in targets:
            model = self._role_models.get(role)
            if model is None or not bool(getattr(model, "supports_gradient_accumulation_no_sync", False)):
                return False
        return True

    def on_train_start(self) -> None:
        from fastvideo.distributed import (
            get_world_group, )
        from fastvideo.utils import set_random_seed

        seed = self.training_config.data.seed
        if seed is None:
            raise ValueError("training.data.seed must be set")
        seed = int(seed)

        world_group = get_world_group()
        global_rank = int(world_group.rank)
        sp_size = int(self.training_config.distributed.sp_size or 1)

        # Ranks within the same SP group share a seed.
        sp_group_seed = seed + (global_rank // sp_size) if sp_size > 1 else seed + global_rank

        set_random_seed(seed)

        self.cuda_generator = torch.Generator(device=self.student.device).manual_seed(sp_group_seed)

        self.student.on_train_start()

    def _infer_attn_kind(self) -> Literal["dense", "vsa"]:
        """Derive metadata mode from the student's resolved backend."""
        backend = (self.student.attention_backend_name or envs.FASTVIDEO_ATTENTION_BACKEND.get())
        if backend in {"VIDEO_SPARSE_ATTN", "VIDEO_SPARSE_ATTN_H3"}:
            return "vsa"
        return "dense"
