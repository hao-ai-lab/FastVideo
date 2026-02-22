# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn.functional as F

from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
)

from fastvideo.distillation.bundle import ModelBundle
from fastvideo.distillation.methods.base import DistillMethod


class _DMD2Adapter(Protocol):
    """Algorithm-specific adapter contract for :class:`DMD2Method`.

    The method layer is intentionally model-agnostic: it should not import or
    depend on any concrete pipeline/model implementation. Instead, all
    model-specific primitives (batch preparation, noise schedule helpers,
    forward-context management, and role-specific backward behavior) are
    provided by an adapter (e.g. ``WanAdapter``).

    This ``Protocol`` documents the required surface area and helps static type
    checkers/IDE tooling; it is not enforced at runtime (duck typing).
    """

    training_args: Any

    def on_train_start(self) -> None:
        ...

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        current_vsa_sparsity: float = 0.0,
    ) -> Any:
        ...

    def student_rollout(self, batch: Any) -> tuple[torch.Tensor, Any]:
        ...

    def sample_dmd_timestep(self, *, device: torch.device) -> torch.Tensor:
        ...

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def teacher_predict_x0(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
    ) -> torch.Tensor:
        ...

    def critic_predict_x0(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        ...

    def critic_flow_matching_loss(self, batch: Any) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        ...

    def backward_student(self, loss: torch.Tensor, ctx: Any, *, grad_accum_rounds: int) -> None:
        ...

    def backward_critic(self, loss: torch.Tensor, ctx: Any, *, grad_accum_rounds: int) -> None:
        ...

    def log_validation(self, iteration: int) -> None:
        ...


class DMD2Method(DistillMethod):
    """DMD2 distillation algorithm (method layer).

    Owns the algorithmic orchestration (loss construction + update policy) and
    stays independent of any specific model family. It requires a
    :class:`~fastvideo.distillation.bundle.ModelBundle` containing at least the
    roles ``student``, ``teacher``, and ``critic``.

    All model-family details (how to run student rollout, teacher CFG
    prediction, critic loss, and how to safely run backward under activation
    checkpointing/forward-context constraints) are delegated to the adapter
    passed in at construction time.
    """

    def __init__(
        self,
        *,
        bundle: ModelBundle,
        adapter: _DMD2Adapter,
    ) -> None:
        super().__init__(bundle)
        bundle.require_roles(["student", "teacher", "critic"])
        self.adapter = adapter
        self.training_args = adapter.training_args

    def on_train_start(self) -> None:
        self.adapter.on_train_start()

    def log_validation(self, iteration: int) -> None:
        if hasattr(self.adapter, "log_validation"):
            self.adapter.log_validation(iteration)

    def _should_update_student(self, iteration: int) -> bool:
        interval = int(getattr(self.training_args, "generator_update_interval", 1) or 1)
        if interval <= 0:
            return True
        return iteration % interval == 0

    def _clip_grad_norm(self, module: torch.nn.Module) -> float:
        max_grad_norm = getattr(self.training_args, "max_grad_norm", None)
        if not max_grad_norm:
            return 0.0
        grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for p in module.parameters()],
            float(max_grad_norm),
            foreach=None,
        )
        return float(grad_norm.item()) if grad_norm is not None else 0.0

    def _dmd_loss(self, generator_pred_x0: torch.Tensor, batch: Any) -> torch.Tensor:
        guidance_scale = float(getattr(self.training_args, "real_score_guidance_scale", 1.0))
        device = generator_pred_x0.device

        with torch.no_grad():
            timestep = self.adapter.sample_dmd_timestep(device=device)

            noise = torch.randn(
                generator_pred_x0.shape,
                device=device,
                dtype=generator_pred_x0.dtype,
            )
            noisy_latents = self.adapter.add_noise(generator_pred_x0, noise, timestep)

            faker_x0 = self.adapter.critic_predict_x0(noisy_latents, timestep, batch)
            real_cond_x0 = self.adapter.teacher_predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=True,
            )
            real_uncond_x0 = self.adapter.teacher_predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=False,
            )
            real_cfg_x0 = real_cond_x0 + (real_cond_x0 - real_uncond_x0) * guidance_scale

            denom = torch.abs(generator_pred_x0 - real_cfg_x0).mean()
            grad = (faker_x0 - real_cfg_x0) / denom
            grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(
            generator_pred_x0.float(),
            (generator_pred_x0.float() - grad.float()).detach(),
        )
        return loss

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
        *,
        current_vsa_sparsity: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        training_batch = self.adapter.prepare_batch(
            batch,
            current_vsa_sparsity=current_vsa_sparsity,
        )

        update_student = self._should_update_student(iteration)

        generator_loss = torch.zeros(
            (),
            device=training_batch.latents.device,
            dtype=training_batch.latents.dtype,
        )
        student_ctx = None
        if update_student:
            generator_pred_x0, student_ctx = self.adapter.student_rollout(training_batch)
            generator_loss = self._dmd_loss(generator_pred_x0, training_batch)

        fake_score_loss, critic_ctx, critic_outputs = self.adapter.critic_flow_matching_loss(
            training_batch
        )

        total_loss = generator_loss + fake_score_loss
        loss_map = {
            "total_loss": total_loss,
            "generator_loss": generator_loss,
            "fake_score_loss": fake_score_loss,
        }

        outputs: dict[str, Any] = dict(critic_outputs)
        outputs["_fv_backward"] = {
            "update_student": update_student,
            "student_ctx": student_ctx,
            "critic_ctx": critic_ctx,
        }
        return loss_map, outputs

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        grad_accum_rounds = max(1, int(grad_accum_rounds))
        backward_ctx = outputs.get("_fv_backward")
        if not isinstance(backward_ctx, dict):
            super().backward(loss_map, outputs, grad_accum_rounds=grad_accum_rounds)
            return

        update_student = bool(backward_ctx.get("update_student", False))
        if update_student:
            student_ctx = backward_ctx.get("student_ctx")
            if student_ctx is None:
                raise RuntimeError("Missing student backward context")
            self.adapter.backward_student(
                loss_map["generator_loss"],
                student_ctx,
                grad_accum_rounds=grad_accum_rounds,
            )

        critic_ctx = backward_ctx.get("critic_ctx")
        if critic_ctx is None:
            raise RuntimeError("Missing critic backward context")
        self.adapter.backward_critic(
            loss_map["fake_score_loss"],
            critic_ctx,
            grad_accum_rounds=grad_accum_rounds,
        )

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        optimizers: list[torch.optim.Optimizer] = []
        optimizers.extend(self.bundle.role("critic").optimizers.values())
        if self._should_update_student(iteration):
            optimizers.extend(self.bundle.role("student").optimizers.values())
        return optimizers

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        schedulers: list[Any] = []
        schedulers.extend(self.bundle.role("critic").lr_schedulers.values())
        if self._should_update_student(iteration):
            schedulers.extend(self.bundle.role("student").lr_schedulers.values())
        return schedulers

    def optimizers_schedulers_step(self, iteration: int) -> None:
        if self._should_update_student(iteration):
            for module in self.bundle.role("student").modules.values():
                self._clip_grad_norm(module)
        for module in self.bundle.role("critic").modules.values():
            self._clip_grad_norm(module)

        super().optimizers_schedulers_step(iteration)
