# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal, Protocol

import torch
import torch.nn.functional as F

from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
)

from fastvideo.distillation.bundle import ModelBundle
from fastvideo.distillation.bundle import RoleHandle
from fastvideo.distillation.methods.base import DistillMethod
from fastvideo.distillation.registry import register_method
from fastvideo.distillation.validators.base import ValidationRequest
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

    def backward(self, loss: torch.Tensor, ctx: Any, *,
                 grad_accum_rounds: int) -> None:
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
        validator: Any | None = None,
    ) -> None:
        super().__init__(bundle)
        bundle.require_roles(["student", "teacher", "critic"])
        self.student = bundle.role("student")
        self.teacher = bundle.role("teacher")
        self.critic = bundle.role("critic")
        if not getattr(self.student, "trainable", False):
            raise ValueError(
                "DMD2Method requires models.student.trainable=true")
        if getattr(self.teacher, "trainable", False):
            raise ValueError(
                "DMD2Method requires models.teacher.trainable=false")
        if not getattr(self.critic, "trainable", False):
            raise ValueError("DMD2Method requires models.critic.trainable=true")
        self.adapter = adapter
        self.validator = validator
        self.training_args = adapter.training_args
        self._simulate_generator_forward = bool(
            getattr(self.training_args, "simulate_generator_forward", False))
        self._denoising_step_list: torch.Tensor | None = None
        self._init_optimizers_and_schedulers()

    def _parse_betas(self, raw: Any, *, where: str) -> tuple[float, float]:
        if raw is None:
            raise ValueError(f"Missing betas for {where}")
        if isinstance(raw, tuple | list) and len(raw) == 2:
            return float(raw[0]), float(raw[1])
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError(
                    f"Expected betas as 'b1,b2' at {where}, got {raw!r}")
            return float(parts[0]), float(parts[1])
        raise ValueError(
            f"Expected betas as 'b1,b2' at {where}, got {type(raw).__name__}")

    def _build_role_optimizer_and_scheduler(
        self,
        *,
        role: str,
        handle: RoleHandle,
        learning_rate: float,
        betas: tuple[float, float],
        scheduler_name: str,
    ) -> None:
        modules = handle.modules
        params: list[torch.nn.Parameter] = []
        for module in modules.values():
            params.extend([p for p in module.parameters() if p.requires_grad])
        if not params:
            raise ValueError(
                f"Role {role!r} is trainable but has no trainable parameters")

        optimizer = torch.optim.AdamW(
            params,
            lr=float(learning_rate),
            betas=betas,
            weight_decay=float(
                getattr(self.training_args, "weight_decay", 0.0) or 0.0),
            eps=1e-8,
        )

        scheduler = get_scheduler(
            str(scheduler_name),
            optimizer=optimizer,
            num_warmup_steps=int(
                getattr(self.training_args, "lr_warmup_steps", 0) or 0),
            num_training_steps=int(
                getattr(self.training_args, "max_train_steps", 0) or 0),
            num_cycles=int(
                getattr(self.training_args, "lr_num_cycles", 0) or 0),
            power=float(getattr(self.training_args, "lr_power", 0.0) or 0.0),
            min_lr_ratio=float(
                getattr(self.training_args, "min_lr_ratio", 0.5) or 0.5),
            last_epoch=-1,
        )

        handle.optimizers = {"main": optimizer}
        handle.lr_schedulers = {"main": scheduler}

    def _init_optimizers_and_schedulers(self) -> None:
        training_args = self.training_args

        # Student optimizer/scheduler (default training hyperparams).
        student_lr = float(getattr(training_args, "learning_rate", 0.0) or 0.0)
        student_betas = self._parse_betas(
            getattr(training_args, "betas", None),
            where="training.betas",
        )
        student_sched = str(getattr(training_args, "lr_scheduler", "constant"))
        self._build_role_optimizer_and_scheduler(
            role="student",
            handle=self.student,
            learning_rate=student_lr,
            betas=student_betas,
            scheduler_name=student_sched,
        )

        # Critic optimizer/scheduler (DMD2-specific overrides).
        critic_lr = float(
            getattr(training_args, "fake_score_learning_rate", 0.0) or 0.0)
        if critic_lr == 0.0:
            critic_lr = student_lr

        critic_betas_raw = getattr(training_args, "fake_score_betas", None)
        if critic_betas_raw is None:
            critic_betas_raw = getattr(training_args, "betas", None)
        critic_betas = self._parse_betas(critic_betas_raw,
                                         where="training.fake_score_betas")

        critic_sched = str(
            getattr(training_args, "fake_score_lr_scheduler", None)
            or student_sched)
        self._build_role_optimizer_and_scheduler(
            role="critic",
            handle=self.critic,
            learning_rate=critic_lr,
            betas=critic_betas,
            scheduler_name=critic_sched,
        )

    def on_train_start(self) -> None:
        self.adapter.on_train_start()

    def log_validation(self, iteration: int) -> None:
        validator = getattr(self, "validator", None)
        if validator is None:
            return
        if not getattr(self.training_args, "log_validation", False):
            return

        raw_steps = str(
            getattr(self.training_args, "validation_sampling_steps", "") or "")
        sampling_steps = [int(s) for s in raw_steps.split(",") if s.strip()]
        sampling_steps = [s for s in sampling_steps if s > 0]
        if not sampling_steps:
            # Default to the few-step student rollout step count for DMD2.
            raw_rollout = getattr(self.training_args.pipeline_config,
                                  "dmd_denoising_steps", None)
            if not raw_rollout:
                return
            sampling_steps = [int(len(raw_rollout))]

        raw_guidance = getattr(self.training_args, "validation_guidance_scale",
                               None)
        guidance_scale = float(raw_guidance) if raw_guidance not in (
            None, "") else None

        request = ValidationRequest(
            sample_handle=self.student,
            sampling_steps=sampling_steps,
            guidance_scale=guidance_scale,
        )
        validator.log_validation(iteration, request=request)

    def get_rng_generators(self) -> dict[str, torch.Generator]:
        """Return RNG generators that should be checkpointed for exact resume."""

        generators: dict[str, torch.Generator] = {}

        adapter = getattr(self, "adapter", None)
        get_adapter_generators = getattr(adapter, "get_rng_generators", None)
        if callable(get_adapter_generators):
            generators.update(get_adapter_generators())

        validator = getattr(self, "validator", None)
        validation_gen = getattr(validator, "validation_random_generator", None)
        if isinstance(validation_gen, torch.Generator):
            generators["validation_cpu"] = validation_gen

        return generators

    def _should_update_student(self, iteration: int) -> bool:
        interval = int(
            getattr(self.training_args, "generator_update_interval", 1) or 1)
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

        raw = getattr(self.training_args.pipeline_config, "dmd_denoising_steps",
                      None)
        if not raw:
            raise ValueError(
                "pipeline_config.dmd_denoising_steps must be set for DMD2 distillation"
            )

        steps = torch.tensor(raw, dtype=torch.long, device=device)

        if getattr(self.training_args, "warp_denoising_step", False):
            noise_scheduler = getattr(self.adapter, "noise_scheduler", None)
            if noise_scheduler is None:
                raise ValueError(
                    "warp_denoising_step requires adapter.noise_scheduler.timesteps"
                )

            timesteps = torch.cat((
                noise_scheduler.timesteps.to("cpu"),
                torch.tensor([0], dtype=torch.float32),
            )).to(device)
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

        current_noise_latents = torch.randn(latents.shape,
                                            device=device,
                                            dtype=dtype)
        current_noise_latents_copy = current_noise_latents.clone()

        max_target_idx = len(step_list) - 1
        noise_latents: list[torch.Tensor] = []
        noise_latent_index = target_timestep_idx_int - 1

        if max_target_idx > 0:
            with torch.no_grad():
                for step_idx in range(max_target_idx):
                    current_timestep = step_list[step_idx]
                    current_timestep_tensor = current_timestep * torch.ones(
                        1, device=device, dtype=torch.long)

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
                        1, device=device, dtype=torch.long)
                    noise = torch.randn(latents.shape,
                                        device=device,
                                        dtype=pred_clean.dtype)
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

        batch.dmd_latent_vis_dict["generator_timestep"] = target_timestep.float(
        ).detach()
        return pred_x0

    def _critic_flow_matching_loss(
            self, batch: Any) -> tuple[torch.Tensor, Any, dict[str, Any]]:
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
        fake_score_timestep = self.adapter.shift_and_clamp_timestep(
            fake_score_timestep)

        noise = torch.randn(
            generator_pred_x0.shape,
            device=device,
            dtype=generator_pred_x0.dtype,
        )
        noisy_x0 = self.adapter.add_noise(generator_pred_x0, noise,
                                          fake_score_timestep)

        pred_noise = self.adapter.predict_noise(
            self.critic,
            noisy_x0,
            fake_score_timestep,
            batch,
            conditional=True,
            attn_kind="dense",
        )
        target = noise - generator_pred_x0
        flow_matching_loss = torch.mean((pred_noise - target)**2)

        batch.fake_score_latent_vis_dict = {
            "generator_pred_video": generator_pred_x0,
            "fake_score_timestep": fake_score_timestep,
        }
        outputs = {
            "fake_score_latent_vis_dict": batch.fake_score_latent_vis_dict
        }
        return flow_matching_loss, (batch.timesteps,
                                    batch.attn_metadata), outputs

    def _dmd_loss(self, generator_pred_x0: torch.Tensor,
                  batch: Any) -> torch.Tensor:
        guidance_scale = float(
            getattr(self.training_args, "real_score_guidance_scale", 1.0))
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
            noisy_latents = self.adapter.add_noise(generator_pred_x0, noise,
                                                   timestep)

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
            real_cfg_x0 = real_cond_x0 + (real_cond_x0 -
                                          real_uncond_x0) * guidance_scale

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
            generator_pred_x0 = self._student_rollout(training_batch,
                                                      with_grad=True)
            student_ctx = (training_batch.timesteps,
                           training_batch.attn_metadata_vsa)
            generator_loss = self._dmd_loss(generator_pred_x0, training_batch)

        fake_score_loss, critic_ctx, critic_outputs = self._critic_flow_matching_loss(
            training_batch)

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
            super().backward(loss_map,
                             outputs,
                             grad_accum_rounds=grad_accum_rounds)
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
            schedulers.extend(
                self.bundle.role("student").lr_schedulers.values())
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
    validator: Any | None,
) -> DistillMethod:
    del cfg
    return DMD2Method(bundle=bundle, adapter=adapter, validator=validator)
