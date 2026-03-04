# SPDX-License-Identifier: Apache-2.0
"""DMD2 distillation method (algorithm layer)."""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastvideo.distillation.methods.base import DistillMethod, LogScalar
from fastvideo.distillation.models.base import ModelBase
from fastvideo.distillation.utils.optimizer import (
    build_optimizer_and_scheduler,
    clip_grad_norm_if_needed,
)
from fastvideo.distillation.utils.validation import (
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
from fastvideo.distillation.validators.base import ValidationRequest
from fastvideo.distillation.utils.config import (
    get_optional_float,
    get_optional_int,
    parse_betas,
)

if TYPE_CHECKING:
    pass


class DMD2Method(DistillMethod):
    """DMD2 distillation algorithm (method layer).

    Owns role model instances directly:
    - ``self.student`` — trainable student :class:`ModelBase`
    - ``self.teacher`` — frozen teacher :class:`ModelBase`
    - ``self.critic`` — trainable critic :class:`ModelBase`
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
        validator: Any | None = None,
    ) -> None:
        super().__init__(role_models=role_models)

        if "student" not in role_models:
            raise ValueError("DMD2Method requires role 'student'")
        if "teacher" not in role_models:
            raise ValueError("DMD2Method requires role 'teacher'")
        if "critic" not in role_models:
            raise ValueError("DMD2Method requires role 'critic'")

        self.student = role_models["student"]
        self.teacher = role_models["teacher"]
        self.critic = role_models["critic"]

        if not getattr(self.student, "_trainable", True):
            raise ValueError("DMD2Method requires student to be trainable")
        if getattr(self.teacher, "_trainable", True):
            raise ValueError("DMD2Method requires teacher to be non-trainable")
        if not getattr(self.critic, "_trainable", True):
            raise ValueError("DMD2Method requires critic to be trainable")

        self.validator = validator
        self.training_args = cfg.training_args
        self.method_config: dict[str, Any] = dict(cfg.method)
        self.validation_config: dict[str, Any] = dict(
            getattr(cfg, "validation", {}) or {})
        self._cfg_uncond = self._parse_cfg_uncond()
        self._rollout_mode = self._parse_rollout_mode()
        self._denoising_step_list: torch.Tensor | None = None

        # Initialize preprocessors on student.
        self.student.init_preprocessors(self.training_args)

        self._init_optimizers_and_schedulers()

    # DistillMethod override: single_train_step
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
        latents_source: Literal["data", "zeros"] = "data"
        if self._rollout_mode == "simulate":
            latents_source = "zeros"

        training_batch = self.student.prepare_batch(
            batch,
            current_vsa_sparsity=current_vsa_sparsity,
            latents_source=latents_source,
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
            student_ctx = (
                training_batch.timesteps,
                training_batch.attn_metadata_vsa,
            )
            generator_loss = self._dmd_loss(generator_pred_x0, training_batch)

        (
            fake_score_loss,
            critic_ctx,
            critic_outputs,
        ) = self._critic_flow_matching_loss(training_batch)

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
        metrics: dict[str, LogScalar] = {
            "update_student": float(update_student)
        }
        return loss_map, outputs, metrics

    # DistillMethod override: backward
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
            super().backward(
                loss_map,
                outputs,
                grad_accum_rounds=grad_accum_rounds,
            )
            return

        update_student = bool(backward_ctx.get("update_student", False))
        if update_student:
            student_ctx = backward_ctx.get("student_ctx")
            if student_ctx is None:
                raise RuntimeError("Missing student backward context")
            self.student.backward(
                loss_map["generator_loss"],
                student_ctx,
                grad_accum_rounds=grad_accum_rounds,
            )

        critic_ctx = backward_ctx.get("critic_ctx")
        if critic_ctx is None:
            raise RuntimeError("Missing critic backward context")
        self.student.backward(
            loss_map["fake_score_loss"],
            critic_ctx,
            grad_accum_rounds=grad_accum_rounds,
        )

    # DistillMethod override: get_optimizers
    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        optimizers: list[torch.optim.Optimizer] = []
        optimizers.append(self._critic_optimizer)
        if self._should_update_student(iteration):
            optimizers.append(self._student_optimizer)
        return optimizers

    # DistillMethod override: get_lr_schedulers
    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        schedulers: list[Any] = []
        schedulers.append(self._critic_lr_scheduler)
        if self._should_update_student(iteration):
            schedulers.append(self._student_lr_scheduler)
        return schedulers

    # DistillMethod override: optimizers_schedulers_step
    def optimizers_schedulers_step(self, iteration: int) -> None:
        if self._should_update_student(iteration):
            clip_grad_norm_if_needed(self.student.transformer,
                                     self.training_args)
        clip_grad_norm_if_needed(self.critic.transformer, self.training_args)
        super().optimizers_schedulers_step(iteration)

    # DistillTrainer hook: on_train_start
    def on_train_start(self) -> None:
        self.student.on_train_start()

    # DistillTrainer hook: log_validation
    def log_validation(self, iteration: int) -> None:
        validator = getattr(self, "validator", None)
        if validator is None:
            return
        if not is_validation_enabled(self.validation_config):
            return

        every_steps = parse_validation_every_steps(self.validation_config)
        if every_steps <= 0:
            return
        if iteration % every_steps != 0:
            return

        dataset_file = parse_validation_dataset_file(self.validation_config)
        sampling_steps = parse_validation_sampling_steps(self.validation_config)

        sampling_timesteps: list[int] | None = None
        raw_timesteps = self.validation_config.get("sampling_timesteps", None)
        if raw_timesteps is None:
            raw_timesteps = self.method_config.get("dmd_denoising_steps", None)
        if isinstance(raw_timesteps, list) and raw_timesteps:
            sampling_timesteps = [int(s) for s in raw_timesteps]

        if not sampling_steps:
            if sampling_timesteps is None:
                return
            sampling_steps = [int(len(sampling_timesteps))]

        sampler_kind = parse_validation_sampler_kind(self.validation_config,
                                                     default="sde")
        ode_solver = parse_validation_ode_solver(self.validation_config,
                                                 sampler_kind=sampler_kind)
        if (sampling_timesteps is not None and sampler_kind != "sde"):
            raise ValueError("method_config.validation.sampling_timesteps is "
                             "only valid when sampler_kind='sde'")

        rollout_mode = parse_validation_rollout_mode(self.validation_config)
        guidance_scale = parse_validation_guidance_scale(self.validation_config)
        output_dir = parse_validation_output_dir(self.validation_config)
        num_actions = parse_validation_num_frames(self.validation_config)

        request = ValidationRequest(
            sample_handle=self.student,
            dataset_file=dataset_file,
            sampling_steps=sampling_steps,
            sampler_kind=sampler_kind,
            rollout_mode=rollout_mode,
            ode_solver=ode_solver,
            sampling_timesteps=sampling_timesteps,
            guidance_scale=guidance_scale,
            num_frames=num_actions,
            output_dir=output_dir,
        )
        validator.log_validation(iteration, request=request)

    # Checkpoint hook: get_rng_generators
    def get_rng_generators(self) -> dict[str, torch.Generator]:
        generators: dict[str, torch.Generator] = {}

        student_gens = self.student.get_rng_generators()
        generators.update(student_gens)

        validator = getattr(self, "validator", None)
        validation_gen = getattr(validator, "validation_random_generator", None)
        if isinstance(validation_gen, torch.Generator):
            generators["validation_cpu"] = validation_gen

        return generators

    def _parse_rollout_mode(self, ) -> Literal["simulate", "data_latent"]:
        raw = self.method_config.get("rollout_mode", None)
        if raw is None:
            raise ValueError("method_config.rollout_mode must be set for DMD2")
        if not isinstance(raw, str):
            raise ValueError("method_config.rollout_mode must be a string, "
                             f"got {type(raw).__name__}")
        mode = raw.strip().lower()
        if mode in ("simulate", "sim"):
            return "simulate"
        if mode in ("data_latent", "data", "vae_latent"):
            return "data_latent"
        raise ValueError("method_config.rollout_mode must be one of "
                         "{simulate, data_latent}, got "
                         f"{raw!r}")

    def _parse_cfg_uncond(self) -> dict[str, Any] | None:
        raw = self.method_config.get("cfg_uncond", None)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("method_config.cfg_uncond must be a dict when "
                             f"set, got {type(raw).__name__}")

        cfg: dict[str, Any] = dict(raw)

        on_missing_raw = cfg.get("on_missing", "error")
        if on_missing_raw is None:
            on_missing_raw = "error"
        if not isinstance(on_missing_raw, str):
            raise ValueError("method_config.cfg_uncond.on_missing must be a "
                             f"string, got {type(on_missing_raw).__name__}")
        on_missing = on_missing_raw.strip().lower()
        if on_missing not in {"error", "ignore"}:
            raise ValueError("method_config.cfg_uncond.on_missing must be one "
                             "of {error, ignore}, got "
                             f"{on_missing_raw!r}")
        cfg["on_missing"] = on_missing

        for channel, policy_raw in list(cfg.items()):
            if channel == "on_missing":
                continue
            if policy_raw is None:
                continue
            if not isinstance(policy_raw, str):
                raise ValueError("method_config.cfg_uncond values must be "
                                 "strings, got "
                                 f"{channel}={type(policy_raw).__name__}")
            policy = policy_raw.strip().lower()
            allowed = {"keep", "zero", "drop"}
            if channel == "text":
                allowed = {*allowed, "negative_prompt"}
            if policy not in allowed:
                raise ValueError("method_config.cfg_uncond values must be one "
                                 f"of {sorted(allowed)}, got "
                                 f"{channel}={policy_raw!r}")
            cfg[channel] = policy

        return cfg

    def _init_optimizers_and_schedulers(self) -> None:
        training_args = self.training_args

        # Student optimizer/scheduler.
        student_lr = float(getattr(training_args, "learning_rate", 0.0) or 0.0)
        student_betas = parse_betas(
            getattr(training_args, "betas", None),
            where="training.betas",
        )
        student_sched = str(getattr(training_args, "lr_scheduler", "constant"))
        student_params = [
            p for p in self.student.transformer.parameters() if p.requires_grad
        ]
        (
            self._student_optimizer,
            self._student_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=student_params,
            training_args=self.training_args,
            learning_rate=student_lr,
            betas=student_betas,
            scheduler_name=student_sched,
        )

        # Critic optimizer/scheduler — read from method config.
        critic_lr_raw = get_optional_float(
            self.method_config,
            "fake_score_learning_rate",
            where="method.fake_score_learning_rate",
        )
        critic_lr = float(critic_lr_raw or 0.0)
        if critic_lr == 0.0:
            critic_lr = student_lr

        critic_betas_raw = self.method_config.get("fake_score_betas", None)
        if critic_betas_raw is None:
            critic_betas_raw = getattr(training_args, "betas", None)
        critic_betas = parse_betas(critic_betas_raw,
                                   where="method.fake_score_betas")

        critic_sched_raw = self.method_config.get("fake_score_lr_scheduler",
                                                  None)
        critic_sched = str(critic_sched_raw or student_sched)
        critic_params = [
            p for p in self.critic.transformer.parameters() if p.requires_grad
        ]
        (
            self._critic_optimizer,
            self._critic_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=critic_params,
            training_args=self.training_args,
            learning_rate=critic_lr,
            betas=critic_betas,
            scheduler_name=critic_sched,
        )

    def _should_update_student(self, iteration: int) -> bool:
        interval = get_optional_int(
            self.method_config,
            "generator_update_interval",
            where="method.generator_update_interval",
        )
        if interval is None:
            interval = 1
        if interval <= 0:
            return True
        return iteration % interval == 0

    def _get_denoising_step_list(self, device: torch.device) -> torch.Tensor:
        if (self._denoising_step_list is not None
                and self._denoising_step_list.device == device):
            return self._denoising_step_list

        raw = self.method_config.get("dmd_denoising_steps", None)
        if not isinstance(raw, list) or not raw:
            raise ValueError("method_config.dmd_denoising_steps must be set "
                             "for DMD2 distillation")

        steps = torch.tensor(
            [int(s) for s in raw],
            dtype=torch.long,
            device=device,
        )

        warp = self.method_config.get("warp_denoising_step", None)
        if warp is None:
            warp = getattr(self.training_args, "warp_denoising_step", False)
        if bool(warp):
            timesteps = torch.cat((
                self.student.noise_scheduler.timesteps.to("cpu"),
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

        if self._rollout_mode != "simulate":
            timestep = self._sample_rollout_timestep(device)
            noise = torch.randn(latents.shape, device=device, dtype=dtype)
            noisy_latents = self.student.add_noise(latents, noise, timestep)
            pred_x0 = self.student.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
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
                    current_timestep_tensor = (
                        current_timestep *
                        torch.ones(1, device=device, dtype=torch.long))

                    pred_clean = self.student.predict_x0(
                        current_noise_latents,
                        current_timestep_tensor,
                        batch,
                        conditional=True,
                        cfg_uncond=self._cfg_uncond,
                        attn_kind="vsa",
                    )

                    next_timestep = step_list[step_idx + 1]
                    next_timestep_tensor = (
                        next_timestep *
                        torch.ones(1, device=device, dtype=torch.long))
                    noise = torch.randn(
                        latents.shape,
                        device=device,
                        dtype=pred_clean.dtype,
                    )
                    current_noise_latents = (self.student.add_noise(
                        pred_clean,
                        noise,
                        next_timestep_tensor,
                    ))
                    noise_latents.append(current_noise_latents.clone())

        if noise_latent_index >= 0:
            if noise_latent_index >= len(noise_latents):
                raise RuntimeError("noise_latent_index is out of bounds")
            noisy_input = noise_latents[noise_latent_index]
        else:
            noisy_input = current_noise_latents_copy

        if with_grad:
            pred_x0 = self.student.predict_x0(
                noisy_input,
                target_timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="vsa",
            )
        else:
            with torch.no_grad():
                pred_x0 = self.student.predict_x0(
                    noisy_input,
                    target_timestep,
                    batch,
                    conditional=True,
                    cfg_uncond=self._cfg_uncond,
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
            int(self.student.num_train_timesteps),
            [1],
            device=device,
            dtype=torch.long,
        )
        fake_score_timestep = (
            self.student.shift_and_clamp_timestep(fake_score_timestep))

        noise = torch.randn(
            generator_pred_x0.shape,
            device=device,
            dtype=generator_pred_x0.dtype,
        )
        noisy_x0 = self.student.add_noise(generator_pred_x0, noise,
                                          fake_score_timestep)

        pred_noise = self.critic.predict_noise(
            noisy_x0,
            fake_score_timestep,
            batch,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind="dense",
        )
        target = noise - generator_pred_x0
        flow_matching_loss = torch.mean((pred_noise - target)**2)

        batch.fake_score_latent_vis_dict = {
            "generator_pred_video": generator_pred_x0,
            "fake_score_timestep": fake_score_timestep,
        }
        outputs = {
            "fake_score_latent_vis_dict": (batch.fake_score_latent_vis_dict)
        }
        return (
            flow_matching_loss,
            (batch.timesteps, batch.attn_metadata),
            outputs,
        )

    def _dmd_loss(
        self,
        generator_pred_x0: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        guidance_scale = get_optional_float(
            self.method_config,
            "real_score_guidance_scale",
            where="method.real_score_guidance_scale",
        )
        if guidance_scale is None:
            guidance_scale = 1.0
        device = generator_pred_x0.device

        with torch.no_grad():
            timestep = torch.randint(
                0,
                int(self.student.num_train_timesteps),
                [1],
                device=device,
                dtype=torch.long,
            )
            timestep = self.student.shift_and_clamp_timestep(timestep)

            noise = torch.randn(
                generator_pred_x0.shape,
                device=device,
                dtype=generator_pred_x0.dtype,
            )
            noisy_latents = self.student.add_noise(generator_pred_x0, noise,
                                                   timestep)

            faker_x0 = self.critic.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            real_cond_x0 = self.teacher.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            real_uncond_x0 = self.teacher.predict_x0(
                noisy_latents,
                timestep,
                batch,
                conditional=False,
                cfg_uncond=self._cfg_uncond,
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
