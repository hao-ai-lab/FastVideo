# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal, Protocol

import torch
import torch.nn.functional as F

from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
)

from fastvideo.distillation.bundle import ModelBundle
from fastvideo.distillation.bundle import RoleHandle
from fastvideo.distillation.methods.base import DistillMethod
from fastvideo.distillation.registry import register_method
from fastvideo.distillation.yaml_config import DistillRunConfig


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

    @property
    def num_train_timesteps(self) -> int:
        ...

    def shift_and_clamp_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        ...

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def predict_x0(
        self,
        handle: RoleHandle,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        ...

    def predict_noise(
        self,
        handle: RoleHandle,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        ...

    def backward(self, loss: torch.Tensor, ctx: Any, *, grad_accum_rounds: int) -> None:
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
        self.student = bundle.role("student")
        self.teacher = bundle.role("teacher")
        self.critic = bundle.role("critic")
        if getattr(self.teacher, "trainable", False):
            raise ValueError("DMD2Method requires models.teacher.trainable=false")
        self.adapter = adapter
        self.training_args = adapter.training_args
        self._simulate_generator_forward = bool(
            getattr(self.training_args, "simulate_generator_forward", False)
        )
        self._denoising_step_list: torch.Tensor | None = None

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

    def _get_denoising_step_list(self, device: torch.device) -> torch.Tensor:
        if self._denoising_step_list is not None and self._denoising_step_list.device == device:
            return self._denoising_step_list

        raw = getattr(self.training_args.pipeline_config, "dmd_denoising_steps", None)
        if not raw:
            raise ValueError("pipeline_config.dmd_denoising_steps must be set for DMD2 distillation")

        steps = torch.tensor(raw, dtype=torch.long, device=device)

        if getattr(self.training_args, "warp_denoising_step", False):
            noise_scheduler = getattr(self.adapter, "noise_scheduler", None)
            if noise_scheduler is None:
                raise ValueError("warp_denoising_step requires adapter.noise_scheduler.timesteps")

            timesteps = torch.cat(
                (
                    noise_scheduler.timesteps.to("cpu"),
                    torch.tensor([0], dtype=torch.float32),
                )
            ).to(device)
            steps = timesteps[1000 - steps]

        self._denoising_step_list = steps
        return steps

    def _sample_rollout_timestep(self, device: torch.device) -> torch.Tensor:
        step_list = self._get_denoising_step_list(device)
        index = torch.randint(
            0,
            len(step_list),
            [1],
            device=device,
            dtype=torch.long,
        )
        return step_list[index]

    def _student_rollout(self, batch: Any, *, with_grad: bool) -> torch.Tensor:
        latents = batch.latents
        device = latents.device
        dtype = latents.dtype
        step_list = self._get_denoising_step_list(device)

        if not self._simulate_generator_forward:
            timestep = self._sample_rollout_timestep(device)
            noise = torch.randn(latents.shape, device=device, dtype=dtype)
            noisy_latents = self.adapter.add_noise(latents, noise, timestep)
            pred_x0 = self.adapter.predict_x0(
                self.student,
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                attn_kind="vsa",
            )
            batch.dmd_latent_vis_dict["generator_timestep"] = timestep
            return pred_x0

        target_timestep_idx = torch.randint(
            0,
            len(step_list),
            [1],
            device=device,
            dtype=torch.long,
        )
        target_timestep_idx_int = int(target_timestep_idx.item())
        target_timestep = step_list[target_timestep_idx]

        current_noise_latents = torch.randn(latents.shape, device=device, dtype=dtype)
        current_noise_latents_copy = current_noise_latents.clone()

        max_target_idx = len(step_list) - 1
        noise_latents: list[torch.Tensor] = []
        noise_latent_index = target_timestep_idx_int - 1

        if max_target_idx > 0:
            with torch.no_grad():
                for step_idx in range(max_target_idx):
                    current_timestep = step_list[step_idx]
                    current_timestep_tensor = current_timestep * torch.ones(
                        1, device=device, dtype=torch.long
                    )

                    pred_clean = self.adapter.predict_x0(
                        self.student,
                        current_noise_latents,
                        current_timestep_tensor,
                        batch,
                        conditional=True,
                        attn_kind="vsa",
                    )

                    next_timestep = step_list[step_idx + 1]
                    next_timestep_tensor = next_timestep * torch.ones(
                        1, device=device, dtype=torch.long
                    )
                    noise = torch.randn(latents.shape, device=device, dtype=pred_clean.dtype)
                    current_noise_latents = self.adapter.add_noise(
                        pred_clean,
                        noise,
                        next_timestep_tensor,
                    )
                    noise_latents.append(current_noise_latents.clone())

        if noise_latent_index >= 0:
            if noise_latent_index >= len(noise_latents):
                raise RuntimeError("noise_latent_index is out of bounds")
            noisy_input = noise_latents[noise_latent_index]
        else:
            noisy_input = current_noise_latents_copy

        if with_grad:
            pred_x0 = self.adapter.predict_x0(
                self.student,
                noisy_input,
                target_timestep,
                batch,
                conditional=True,
                attn_kind="vsa",
            )
        else:
            with torch.no_grad():
                pred_x0 = self.adapter.predict_x0(
                    self.student,
                    noisy_input,
                    target_timestep,
                    batch,
                    conditional=True,
                    attn_kind="vsa",
                )

        batch.dmd_latent_vis_dict["generator_timestep"] = target_timestep.float().detach()
        return pred_x0

    def _critic_flow_matching_loss(self, batch: Any) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        with torch.no_grad():
            generator_pred_x0 = self._student_rollout(batch, with_grad=False)

        device = generator_pred_x0.device
        fake_score_timestep = torch.randint(
            0,
            int(self.adapter.num_train_timesteps),
            [1],
            device=device,
            dtype=torch.long,
        )
        fake_score_timestep = self.adapter.shift_and_clamp_timestep(fake_score_timestep)

        noise = torch.randn(
            generator_pred_x0.shape,
            device=device,
            dtype=generator_pred_x0.dtype,
        )
        noisy_x0 = self.adapter.add_noise(generator_pred_x0, noise, fake_score_timestep)

        pred_noise = self.adapter.predict_noise(
            self.critic,
            noisy_x0,
            fake_score_timestep,
            batch,
            conditional=True,
            attn_kind="dense",
        )
        target = noise - generator_pred_x0
        flow_matching_loss = torch.mean((pred_noise - target) ** 2)

        batch.fake_score_latent_vis_dict = {
            "generator_pred_video": generator_pred_x0,
            "fake_score_timestep": fake_score_timestep,
        }
        outputs = {"fake_score_latent_vis_dict": batch.fake_score_latent_vis_dict}
        return flow_matching_loss, (batch.timesteps, batch.attn_metadata), outputs

    def _dmd_loss(self, generator_pred_x0: torch.Tensor, batch: Any) -> torch.Tensor:
        guidance_scale = float(getattr(self.training_args, "real_score_guidance_scale", 1.0))
        device = generator_pred_x0.device

        with torch.no_grad():
            timestep = torch.randint(
                0,
                int(self.adapter.num_train_timesteps),
                [1],
                device=device,
                dtype=torch.long,
            )
            timestep = self.adapter.shift_and_clamp_timestep(timestep)

            noise = torch.randn(
                generator_pred_x0.shape,
                device=device,
                dtype=generator_pred_x0.dtype,
            )
            noisy_latents = self.adapter.add_noise(generator_pred_x0, noise, timestep)

            faker_x0 = self.adapter.predict_x0(
                self.critic,
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            )
            real_cond_x0 = self.adapter.predict_x0(
                self.teacher,
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            )
            real_uncond_x0 = self.adapter.predict_x0(
                self.teacher,
                noisy_latents,
                timestep,
                batch,
                conditional=False,
                attn_kind="dense",
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
            generator_pred_x0 = self._student_rollout(training_batch, with_grad=True)
            student_ctx = (training_batch.timesteps, training_batch.attn_metadata_vsa)
            generator_loss = self._dmd_loss(generator_pred_x0, training_batch)

        fake_score_loss, critic_ctx, critic_outputs = self._critic_flow_matching_loss(training_batch)

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
            self.adapter.backward(
                loss_map["generator_loss"],
                student_ctx,
                grad_accum_rounds=grad_accum_rounds,
            )

        critic_ctx = backward_ctx.get("critic_ctx")
        if critic_ctx is None:
            raise RuntimeError("Missing critic backward context")
        self.adapter.backward(
            loss_map["fake_score_loss"],
            critic_ctx,
            grad_accum_rounds=grad_accum_rounds,
        )

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        optimizers: list[torch.optim.Optimizer] = []
        optimizers.extend(self.critic.optimizers.values())
        if self._should_update_student(iteration):
            optimizers.extend(self.student.optimizers.values())
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


@register_method("dmd2")
def build_dmd2_method(
    *,
    cfg: DistillRunConfig,
    bundle: ModelBundle,
    adapter: _DMD2Adapter,
) -> DistillMethod:
    del cfg
    return DMD2Method(bundle=bundle, adapter=adapter)
