# SPDX-License-Identifier: Apache-2.0
"""Trajectory Distribution Matching for flow-matching video models."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.config import (
    get_optional_float,
    require_bool,
    require_choice,
)
from fastvideo.train.utils.lora import synchronize_lora_gradients
from fastvideo.train.utils.optimizer import clip_grad_norm_if_needed

if TYPE_CHECKING:
    from fastvideo.pipelines import TrainingBatch


@dataclass(slots=True)
class TDMTrajectory:
    """Few-step student trajectory used by TDM losses."""

    noisy_latents: list[torch.Tensor]
    clean_latents: list[torch.Tensor]
    timesteps: torch.Tensor
    sigmas: torch.Tensor

    @property
    def final_clean(self) -> torch.Tensor:
        return self.clean_latents[-1]


@dataclass(slots=True)
class TDMSampleContext:
    """Per-sample source, reconstruction, and score-target state."""

    clean_latents: torch.Tensor
    noisy_source: torch.Tensor
    noisy_intermediate: torch.Tensor
    noisy_target: torch.Tensor
    timestep_source: torch.Tensor
    timestep_target: torch.Tensor
    sigma_source: torch.Tensor
    sigma_intermediate: torch.Tensor
    sigma_target: torch.Tensor
    eps_source: torch.Tensor
    mixed_noise: torch.Tensor
    proposal_noise: torch.Tensor
    transition_beta: torch.Tensor
    trajectory_indices: torch.Tensor


def _expand_sigma_for_latents(
    sigma: torch.Tensor,
    latents: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Broadcast scalar, batch, or frame-level sigma tensors to *latents*."""
    sigma = sigma.to(device=latents.device, dtype=dtype or latents.dtype)
    if sigma.ndim == 0:
        sigma = sigma.reshape(1)

    if sigma.ndim == latents.ndim:
        return sigma

    if sigma.ndim == 1:
        if sigma.numel() == 1:
            return sigma.reshape((1, ) + (1, ) * (latents.ndim - 1))
        if sigma.numel() == latents.shape[0]:
            return sigma.reshape((latents.shape[0], ) + (1, ) * (latents.ndim - 1))
        if latents.ndim >= 2 and sigma.numel() == latents.shape[0] * latents.shape[1]:
            return sigma.reshape(latents.shape[0], latents.shape[1], *([1] * (latents.ndim - 2)))
        raise ValueError("Cannot broadcast sigma with shape "
                         f"{tuple(sigma.shape)} to latents {tuple(latents.shape)}")

    if sigma.ndim == 2:
        if latents.ndim < 2 or tuple(sigma.shape) != tuple(latents.shape[:2]):
            raise ValueError("Frame-level sigma shape must match latent batch/frame "
                             f"prefix, got {tuple(sigma.shape)} for {tuple(latents.shape)}")
        return sigma.reshape(*sigma.shape, *([1] * (latents.ndim - 2)))

    while sigma.ndim < latents.ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _mean_except_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 1:
        return tensor
    return tensor.mean(dim=tuple(range(1, tensor.ndim)))


def flow_effective_noise(
    noisy_latents: torch.Tensor,
    clean_latents: torch.Tensor,
    sigma: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Recover effective flow noise from ``x_sigma = (1-sigma)x0 + sigma eps``."""
    sigma_b = _expand_sigma_for_latents(sigma, noisy_latents)
    return (noisy_latents - (1.0 - sigma_b) * clean_latents) / sigma_b.clamp_min(eps)


def flow_snr(
    sigma: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Flow-matching analogue of diffusion SNR, ``((1-sigma)/sigma)^2``."""
    sigma = sigma.float()
    return ((1.0 - sigma).clamp_min(0.0) / sigma.clamp_min(eps)).square()


def flow_transition_to_noisier_sigma(
    *,
    noisy_from: torch.Tensor,
    clean_latents: torch.Tensor,
    eps_from: torch.Tensor,
    sigma_from: torch.Tensor,
    sigma_to: torch.Tensor,
    proposal_noise: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move a flow-matching point from ``sigma_from`` to ``sigma_to``.

    This is the Wan/flow analogue of TDM's diffusion between-timestep noising.
    The returned ``mixed_noise`` satisfies:

    ``noisy_to == (1 - sigma_to) * clean_latents + sigma_to * mixed_noise``.
    """
    s1 = _expand_sigma_for_latents(sigma_from, noisy_from, dtype=torch.float32)
    s2 = _expand_sigma_for_latents(sigma_to, noisy_from, dtype=torch.float32)
    if bool(torch.any(s2 + eps < s1).item()):
        raise ValueError("TDM flow transition requires sigma_to >= sigma_from")
    if bool(torch.any((1.0 - s1).abs() < eps).item()):
        raise ValueError("TDM flow transition cannot start from sigma=1")
    if bool(torch.any(s2 <= eps).item()):
        raise ValueError("TDM flow transition requires sigma_to > 0")

    a = (1.0 - s2) / (1.0 - s1).clamp_min(eps)
    beta_sq = s2.square() - (a * s1).square()
    min_beta_sq = beta_sq.min()
    if bool((min_beta_sq < -1e-6).item()):
        raise ValueError("TDM flow transition produced negative beta_sq")
    beta_sq = beta_sq.clamp_min(0.0)
    beta = beta_sq.sqrt()
    coefficient_dtype = noisy_from.dtype
    a_latent = a.to(dtype=coefficient_dtype)
    beta_latent = beta.to(dtype=coefficient_dtype)
    s1_latent = s1.to(dtype=coefficient_dtype)
    s2_latent = s2.to(dtype=coefficient_dtype)

    noisy_to = a_latent * noisy_from + beta_latent * proposal_noise
    mixed_noise = (a_latent * s1_latent * eps_from + beta_latent * proposal_noise) / s2_latent.clamp_min(eps)
    return noisy_to, mixed_noise, beta_latent


class TDMMethod(DMD2Method):
    """Trajectory Distribution Matching adapted to Wan-style flow matching.

    TDM keeps DMD2's three-role training layout but changes the generated
    trajectory and fake-score objectives to match the reference TDM algorithm.
    Diffusion alpha-bar math is replaced with Wan's linear flow noising family:
    ``x_sigma = (1 - sigma) * x0 + sigma * eps``.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(
            cfg=cfg,
            role_models=role_models,
        )

        if self._rollout_mode != "simulate":
            raise ValueError("TDMMethod currently requires method.rollout_mode='simulate'")

        mcfg = self.method_config
        self._denoising_step_list: torch.Tensor | None = None
        self._denoising_sigma_list: torch.Tensor | None = None
        self._rollout_sample_type: Literal["sde", "ode"] = require_choice(
            mcfg,
            "student_sample_type",
            {"sde", "ode"},
            default="sde",
            where="method.student_sample_type",
        )  # type: ignore[assignment]
        self._noise_interval_mode: Literal["separate", "to_terminal"] = require_choice(
            mcfg,
            "noise_interval_mode",
            {"separate", "to_terminal"},
            default="separate",
            where="method.noise_interval_mode",
        )  # type: ignore[assignment]
        self._use_randmid = require_bool(
            mcfg,
            "use_randmid",
            default=False,
            where="method.use_randmid",
        )

        self._use_huber = require_bool(
            mcfg,
            "use_huber",
            default=False,
            where="method.use_huber",
        )

        huber_c = get_optional_float(
            mcfg,
            "huber_c",
            where="method.huber_c",
        )
        if huber_c is None:
            huber_c = 0.001
        if huber_c <= 0:
            raise ValueError("method.huber_c must be positive")
        self._huber_c = float(huber_c)

        self._use_pseudo_huber = require_bool(
            mcfg,
            "use_pseudo_huber",
            default=False,
            where="method.use_pseudo_huber",
        )
        if self._use_pseudo_huber and self._use_huber:
            raise ValueError("method.use_huber and method.use_pseudo_huber are mutually exclusive")

        snr_clip = get_optional_float(
            mcfg,
            "snr_clip",
            where="method.snr_clip",
        )
        if snr_clip is None:
            snr_clip = 5.0
        if snr_clip <= 0:
            raise ValueError("method.snr_clip must be positive")
        self._snr_clip = float(snr_clip)

        importance_clip = get_optional_float(
            mcfg,
            "importance_weight_clip",
            where="method.importance_weight_clip",
        )
        if importance_clip is None:
            importance_clip = 10.0
        if importance_clip <= 0:
            raise ValueError("method.importance_weight_clip must be positive")
        self._importance_weight_clip = float(importance_clip)

        self._normalize_generator_delta = require_bool(
            mcfg,
            "normalize_generator_delta",
            default=True,
            where="method.normalize_generator_delta",
        )
        max_grad_norm = get_optional_float(
            mcfg,
            "max_grad_norm",
            where="method.max_grad_norm",
        )
        if max_grad_norm is None:
            max_grad_norm = 1.0
        if max_grad_norm < 0.0:
            raise ValueError("method.max_grad_norm must be non-negative")
        self._max_grad_norm = float(max_grad_norm)
        self._sigma_eps = 1e-8

    def manages_optimization(self) -> bool:
        return True

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        del batch, iteration
        raise RuntimeError("TDMMethod uses managed_train_step() to preserve fake-score-before-generator ordering")

    def managed_train_step(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        grad_accum = max(1, int(self.training_config.loop.gradient_accumulation_steps or 1))
        raw_batches = [next(data_stream) for _ in range(grad_accum)]

        fake_score_losses: list[torch.Tensor] = []
        fake_score_metric_maps: list[dict[str, LogScalar]] = []
        for raw_batch in raw_batches:
            training_batch = self.student.prepare_batch(
                raw_batch,
                generator=self.cuda_generator,
                latents_source="zeros",
            )
            if training_batch.latents is None:
                raise RuntimeError("TDM requires student.prepare_batch to populate latents")
            fake_score_loss, critic_ctx, _, fake_score_metrics = self._tdm_fake_score_loss(training_batch)
            self.critic.backward(
                fake_score_loss,
                critic_ctx,
                grad_accum_rounds=grad_accum,
            )
            fake_score_losses.append(fake_score_loss.detach())
            fake_score_metric_maps.append(fake_score_metrics)

        critic_grad_norm = self._finish_role_update(
            model=self.critic,
            optimizer=self._critic_optimizer,
            lr_scheduler=self._critic_lr_scheduler,
        )

        update_student = self._should_update_student(iteration)
        generator_losses: list[torch.Tensor] = []
        generator_metric_maps: list[dict[str, LogScalar]] = []
        if update_student:
            for raw_batch in raw_batches:
                training_batch = self.student.prepare_batch(
                    raw_batch,
                    generator=self.cuda_generator,
                    latents_source="zeros",
                )
                if training_batch.latents is None:
                    raise RuntimeError("TDM requires student.prepare_batch to populate latents")
                with torch.no_grad():
                    trajectory = self._student_trajectory(training_batch)
                generator_loss, generator_metrics, student_ctx = self._tdm_generator_loss(
                    trajectory,
                    training_batch,
                )
                self.student.backward(
                    generator_loss,
                    student_ctx,
                    grad_accum_rounds=grad_accum,
                )
                generator_losses.append(generator_loss.detach())
                generator_metric_maps.append(generator_metrics)
            student_grad_norm = self._finish_role_update(
                model=self.student,
                optimizer=self._student_optimizer,
                lr_scheduler=self._student_lr_scheduler,
            )
        else:
            student_grad_norm = 0.0

        fake_score_loss = torch.stack(fake_score_losses).mean()
        generator_loss = (torch.stack(generator_losses).mean()
                          if generator_losses else torch.zeros_like(fake_score_loss))
        loss_map = {
            "total_loss": generator_loss + fake_score_loss,
            "generator_loss": generator_loss,
            "fake_score_loss": fake_score_loss,
        }
        metrics: dict[str, LogScalar] = {
            "update_student": float(update_student),
            "grad_norm/critic": critic_grad_norm,
        }
        if update_student:
            metrics["grad_norm/student"] = student_grad_norm
        metrics.update(self._mean_metric_maps(fake_score_metric_maps))
        metrics.update(self._mean_metric_maps(generator_metric_maps))
        return loss_map, {}, metrics

    def _finish_role_update(
        self,
        *,
        model: ModelBase,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
    ) -> float:
        synchronize_lora_gradients(model.transformer)
        grad_norm = clip_grad_norm_if_needed(model.transformer, self._max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return grad_norm

    @staticmethod
    def _mean_metric_maps(metric_maps: list[dict[str, LogScalar]]) -> dict[str, LogScalar]:
        if not metric_maps:
            return {}
        averaged: dict[str, LogScalar] = {}
        for key in metric_maps[0]:
            values = [metrics[key] for metrics in metric_maps]
            first = values[0]
            if isinstance(first, torch.Tensor):
                averaged[key] = torch.stack([
                    value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=first.device)
                    for value in values
                ]).mean()
            else:
                averaged[key] = sum(float(value) for value in values) / len(values)
        return averaged

    def _get_denoising_step_list(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        if (self._denoising_step_list is not None and self._denoising_step_list.device == device):
            return self._denoising_step_list

        raw = self.method_config.get("tdm_denoising_steps", None)
        if raw is None:
            raw = self.method_config.get("dmd_denoising_steps", None)
        if not isinstance(raw, list) or not raw:
            raise ValueError("method.tdm_denoising_steps must be set for TDM")

        raw_steps = torch.tensor(
            [int(s) for s in raw],
            dtype=torch.long,
            device=device,
        )
        if raw_steps.numel() < 2:
            raise ValueError("method.tdm_denoising_steps must contain at least two steps for TDM")
        steps = raw_steps.to(dtype=torch.float32)

        warp = self.method_config.get("warp_denoising_step", None)
        if warp is None:
            warp = False
        if bool(warp):
            timesteps = torch.cat((
                self.student.noise_scheduler.timesteps.to("cpu"),
                torch.tensor([0], dtype=torch.float32),
            )).to(device)
            step_indices = int(self.student.num_train_timesteps) - raw_steps
            if bool(torch.any((step_indices < 0) | (step_indices >= len(timesteps))).item()):
                raise ValueError("method.tdm_denoising_steps contains values outside the scheduler training range")
            steps = timesteps[step_indices]

        sigmas = self._timestep_to_sigma(steps, scheduler_space=bool(warp))
        scheduler_sigmas = self.student.noise_scheduler.sigmas.to(device=device, dtype=torch.float32)
        terminal_sigma = scheduler_sigmas.max()
        if not bool(torch.isclose(
                sigmas[0],
                terminal_sigma,
                rtol=0.0,
                atol=1e-6,
        ).item()):
            raise ValueError("method.tdm_denoising_steps must start at the scheduler terminal sigma because "
                             "TDM rollout starts from pure noise")
        if not bool(torch.all(sigmas[:-1] > sigmas[1:] + 1e-6).item()):
            raise ValueError("method.tdm_denoising_steps must map to strictly decreasing scheduler sigmas")

        self._denoising_step_list = steps
        self._denoising_sigma_list = sigmas
        return steps

    def _timestep_to_sigma(
        self,
        timestep: torch.Tensor,
        *,
        scheduler_space: bool = False,
    ) -> torch.Tensor:
        scheduler = self.student.noise_scheduler
        t = timestep.to(device=timestep.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.reshape(1)
        elif t.ndim == 2:
            t = t.flatten(0, 1)
        elif t.ndim != 1:
            raise ValueError(f"Invalid timestep shape: {tuple(timestep.shape)}")

        config = getattr(scheduler, "config", None)
        shift = getattr(scheduler, "shift", None)
        num_train_timesteps = getattr(
            config,
            "num_train_timesteps",
            getattr(scheduler, "num_train_timesteps", None),
        )
        sigmas = scheduler.sigmas.to(device=timestep.device, dtype=torch.float32)
        timesteps = scheduler.timesteps.to(device=timestep.device, dtype=torch.float32)
        if scheduler_space:
            idx = torch.argmin(
                (timesteps.unsqueeze(0) - t.unsqueeze(1)).abs(),
                dim=1,
            )
            return sigmas[idx]

        has_static_flow_schedule = (shift is not None and num_train_timesteps is not None
                                    and not bool(getattr(config, "use_dynamic_shifting", False))
                                    and not getattr(config, "shift_terminal", None)
                                    and not bool(getattr(config, "use_karras_sigmas", False))
                                    and not bool(getattr(config, "use_exponential_sigmas", False))
                                    and not bool(getattr(config, "use_beta_sigmas", False)))
        if has_static_flow_schedule:
            assert shift is not None
            assert num_train_timesteps is not None
            flow_shift = float(shift)
            sigma = t / float(num_train_timesteps)
            return flow_shift * sigma / (1.0 + (flow_shift - 1.0) * sigma)

        idx = torch.argmin(
            (timesteps.unsqueeze(0) - t.unsqueeze(1)).abs(),
            dim=1,
        )
        return sigmas[idx]

    def _model_timestep_for_sigma(
        self,
        sigma: torch.Tensor,
        model: ModelBase,
    ) -> torch.Tensor:
        """Resolve an authoritative flow sigma to a model scheduler label."""
        scheduler = model.noise_scheduler
        model_timesteps = scheduler.timesteps.to(device=sigma.device, dtype=torch.float32)
        model_sigmas = scheduler.sigmas[:model_timesteps.numel()].to(device=sigma.device, dtype=torch.float32)
        flat_sigma = sigma.to(dtype=torch.float32).reshape(-1)
        indices = torch.argmin(
            (model_sigmas.unsqueeze(0) - flat_sigma.unsqueeze(1)).abs(),
            dim=1,
        )
        resolved_sigmas = model_sigmas[indices]
        if not bool(torch.allclose(
                resolved_sigmas,
                flat_sigma,
                rtol=1e-5,
                atol=1e-6,
        )):
            raise ValueError("TDM role scheduler does not contain the requested trajectory sigma")
        return model_timesteps[indices].reshape(sigma.shape)

    def _student_trajectory(
        self,
        batch: TrainingBatch,
    ) -> TDMTrajectory:
        latents = batch.latents
        if latents is None:
            raise RuntimeError("TDM requires prepared batch latents")
        device = latents.device
        dtype = latents.dtype
        step_list = self._get_denoising_step_list(device)
        sigma_list = self._denoising_sigma_list
        if sigma_list is None:
            raise RuntimeError("TDM denoising sigmas were not initialized with the step schedule")
        if len(step_list) < 2:
            raise ValueError("TDM requires at least two denoising steps")

        current_noisy = torch.randn(
            latents.shape,
            device=device,
            dtype=dtype,
            generator=self.cuda_generator,
        )
        noisy_latents: list[torch.Tensor] = []
        clean_latents: list[torch.Tensor] = []
        sigmas: list[torch.Tensor] = []

        for step_idx in range(len(step_list)):
            sigma = sigma_list[step_idx].reshape(1)
            model_timestep = self._model_timestep_for_sigma(sigma, self.student)
            batch.timesteps = model_timestep
            with torch.no_grad():
                pred_x0 = self.student.predict_x0(
                    current_noisy,
                    model_timestep,
                    batch,
                    conditional=True,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind="vsa",
                )

            noisy_latents.append(current_noisy)
            clean_latents.append(pred_x0)
            sigmas.append(sigma[0])

            if step_idx + 1 >= len(step_list):
                break

            sigma_next = sigma_list[step_idx + 1].reshape(1)
            if self._rollout_sample_type == "sde":
                noise = torch.randn(
                    latents.shape,
                    device=device,
                    dtype=dtype,
                    generator=self.cuda_generator,
                )
                sigma_next_b = _expand_sigma_for_latents(sigma_next, current_noisy)
                current_noisy = (1.0 - sigma_next_b) * pred_x0 + sigma_next_b * noise
            else:
                eps = flow_effective_noise(
                    current_noisy,
                    pred_x0,
                    sigma,
                    eps=self._sigma_eps,
                )
                sigma_next_b = _expand_sigma_for_latents(sigma_next, current_noisy)
                current_noisy = (1.0 - sigma_next_b) * pred_x0 + sigma_next_b * eps

        timesteps = step_list.to(device=device)
        sigma_tensor = torch.stack(sigmas).to(device=device)
        batch.dmd_latent_vis_dict["generator_timestep"] = timesteps[-1].float().detach()
        return TDMTrajectory(
            noisy_latents=noisy_latents,
            clean_latents=clean_latents,
            timesteps=timesteps,
            sigmas=sigma_tensor,
        )

    def _sample_tdm_context(
        self,
        trajectory: TDMTrajectory,
    ) -> TDMSampleContext:
        device = trajectory.sigmas.device
        interval_eps = 1e-6
        batch_size = trajectory.noisy_latents[0].shape[0]
        batch_indices = torch.arange(batch_size, device=device)
        trajectory_indices = torch.randint(
            0,
            len(trajectory.sigmas),
            [batch_size],
            device=device,
            dtype=torch.long,
            generator=self.cuda_generator,
        )
        clean_latents = torch.stack(trajectory.clean_latents)[trajectory_indices, batch_indices].detach()
        noisy_source = torch.stack(trajectory.noisy_latents)[trajectory_indices, batch_indices].detach()
        sigma_source = trajectory.sigmas[trajectory_indices]
        timestep_source = self._model_timestep_for_sigma(sigma_source, self.student)

        scheduled_intermediate_sigmas = torch.cat((trajectory.sigmas[1:], trajectory.sigmas.new_zeros(1)))
        sigma_intermediate = scheduled_intermediate_sigmas[trajectory_indices]
        scheduler = self.student.noise_scheduler
        scheduler_timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)
        scheduler_sigmas = scheduler.sigmas[:scheduler_timesteps.numel()].to(device=device, dtype=torch.float32)

        def sample_scheduler_point(lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            sampled_sigmas: list[torch.Tensor] = []
            sampled_timesteps: list[torch.Tensor] = []
            for lower_i, upper_i in zip(lower, upper, strict=False):
                candidates = torch.nonzero(
                    (scheduler_sigmas >= lower_i - interval_eps)
                    & (scheduler_sigmas < upper_i - interval_eps)
                    & (scheduler_sigmas > self._sigma_eps),
                    as_tuple=False,
                ).flatten()
                if candidates.numel() == 0:
                    raise ValueError("TDM schedule has no scheduler point in the requested noise interval")
                position = torch.randint(
                    0,
                    candidates.numel(),
                    [1],
                    device=device,
                    dtype=torch.long,
                    generator=self.cuda_generator,
                )
                selected = candidates[position].reshape(())
                sampled_sigmas.append(scheduler_sigmas[selected])
                sampled_timesteps.append(scheduler_timesteps[selected])
            return torch.stack(sampled_sigmas), torch.stack(sampled_timesteps)

        if self._use_randmid:
            sigma_intermediate, _ = sample_scheduler_point(sigma_intermediate, sigma_source)

        target_upper = sigma_source
        if self._noise_interval_mode == "to_terminal":
            target_upper = torch.full_like(sigma_source, trajectory.sigmas.max())
        sigma_target, timestep_target = sample_scheduler_point(sigma_intermediate, target_upper)

        eps_source = flow_effective_noise(
            noisy_source,
            clean_latents,
            sigma_source,
            eps=self._sigma_eps,
        )
        sigma_intermediate_b = _expand_sigma_for_latents(sigma_intermediate, clean_latents)
        noisy_intermediate = (1.0 - sigma_intermediate_b) * clean_latents + sigma_intermediate_b * eps_source
        proposal_noise = torch.randn(
            clean_latents.shape,
            device=clean_latents.device,
            dtype=clean_latents.dtype,
            generator=self.cuda_generator,
        )
        noisy_target, mixed_noise, beta = flow_transition_to_noisier_sigma(
            noisy_from=noisy_intermediate,
            clean_latents=clean_latents,
            eps_from=eps_source,
            sigma_from=sigma_intermediate,
            sigma_to=sigma_target,
            proposal_noise=proposal_noise,
            eps=self._sigma_eps,
        )

        return TDMSampleContext(
            clean_latents=clean_latents,
            noisy_source=noisy_source,
            noisy_intermediate=noisy_intermediate,
            noisy_target=noisy_target,
            timestep_source=timestep_source,
            timestep_target=timestep_target,
            sigma_source=sigma_source,
            sigma_intermediate=sigma_intermediate,
            sigma_target=sigma_target,
            eps_source=eps_source,
            mixed_noise=mixed_noise,
            proposal_noise=proposal_noise,
            transition_beta=beta,
            trajectory_indices=trajectory_indices,
        )

    def _tdm_fake_score_loss(
        self,
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, Any, dict[str, Any], dict[str, LogScalar]]:
        with torch.no_grad():
            trajectory = self._student_trajectory(batch)
            context = self._sample_tdm_context(trajectory)

        critic_timestep = self._model_timestep_for_sigma(context.sigma_target, self.critic)
        batch.timesteps = critic_timestep
        fake_x0 = self.critic.predict_x0(
            context.noisy_target,
            critic_timestep,
            batch,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind="dense",
        )
        elementwise = (fake_x0.float() - context.clean_latents.float()).square()
        per_sample = _mean_except_batch(elementwise)
        weight_components = self._tdm_fake_score_weight_components(context)
        weights = weight_components["weights"].to(device=per_sample.device, dtype=per_sample.dtype)
        fake_score_loss = (per_sample * weights).mean()

        batch.fake_score_latent_vis_dict = {
            "generator_pred_video": context.clean_latents,
            "fake_score_timestep": context.timestep_target.float().detach(),
            "fake_score_source_timestep": context.timestep_source.float().detach(),
        }
        outputs = {"fake_score_latent_vis_dict": (batch.fake_score_latent_vis_dict)}
        metrics = self._tdm_fake_score_metrics(
            context,
            per_sample,
            weight_components,
        )
        return (
            fake_score_loss,
            (critic_timestep, batch.attn_metadata),
            outputs,
            metrics,
        )

    def _tdm_fake_score_weight_components(
        self,
        context: TDMSampleContext,
    ) -> dict[str, torch.Tensor]:
        snr = flow_snr(context.sigma_target, eps=self._sigma_eps)
        snr_weight = torch.minimum(snr, torch.full_like(snr, self._snr_clip))

        mixed_sq = _mean_except_batch(context.mixed_noise.float().square())
        proposal_sq = _mean_except_batch(context.proposal_noise.float().square())
        log_importance = 0.5 * (proposal_sq - mixed_sq)
        importance = torch.exp(log_importance.clamp(
            min=-20.0,
            max=20.0,
        ))
        importance = importance.clamp(max=self._importance_weight_clip)
        weights = snr_weight.reshape(-1) * importance.detach()
        return {
            "snr": snr.detach(),
            "snr_weight": snr_weight.detach(),
            "importance": importance.detach(),
            "weights": weights.detach(),
            "mixed_noise_sq": mixed_sq.detach(),
            "proposal_noise_sq": proposal_sq.detach(),
        }

    def _tdm_fake_score_weights(
        self,
        context: TDMSampleContext,
    ) -> torch.Tensor:
        return self._tdm_fake_score_weight_components(context)["weights"]

    def _tdm_fake_score_metrics(
        self,
        context: TDMSampleContext,
        per_sample_loss: torch.Tensor,
        weight_components: dict[str, torch.Tensor],
    ) -> dict[str, LogScalar]:
        weights = weight_components["weights"].float()
        importance = weight_components["importance"].float()
        per_sample_loss = per_sample_loss.detach().float()
        snr = weight_components["snr"].float()
        snr_weight = weight_components["snr_weight"].float()
        sigma_target = context.sigma_target.detach().float()
        max_sigma = torch.ones_like(sigma_target)
        terminal = torch.isclose(
            sigma_target,
            max_sigma,
            rtol=0.0,
            atol=1e-6,
        ).float()
        return {
            "tdm/fake_score/source_sigma": context.sigma_source.detach().float().mean(),
            "tdm/fake_score/intermediate_sigma": context.sigma_intermediate.detach().float().mean(),
            "tdm/fake_score/target_sigma": sigma_target.mean(),
            "tdm/fake_score/source_timestep": context.timestep_source.detach().float().mean(),
            "tdm/fake_score/target_timestep": context.timestep_target.detach().float().mean(),
            "tdm/fake_score/source_trajectory_index": context.trajectory_indices.detach().float().mean(),
            "tdm/fake_score/sigma_to_is_terminal": terminal.mean(),
            "tdm/fake_score/snr": snr.mean(),
            "tdm/fake_score/snr_weight": snr_weight.mean(),
            "tdm/fake_score/importance_min": importance.min(),
            "tdm/fake_score/importance_mean": importance.mean(),
            "tdm/fake_score/importance_max": importance.max(),
            "tdm/fake_score/weight_min": weights.min(),
            "tdm/fake_score/weight_mean": weights.mean(),
            "tdm/fake_score/weight_max": weights.max(),
            "tdm/fake_score/per_sample_loss_min": per_sample_loss.min(),
            "tdm/fake_score/per_sample_loss_mean": per_sample_loss.mean(),
            "tdm/fake_score/per_sample_loss_max": per_sample_loss.max(),
            "tdm/fake_score/mixed_noise_sq_mean": weight_components["mixed_noise_sq"].float().mean(),
            "tdm/fake_score/proposal_noise_sq_mean": weight_components["proposal_noise_sq"].float().mean(),
            "tdm/fake_score/transition_beta_mean": context.transition_beta.detach().float().mean(),
        }

    def _tdm_generator_loss(
        self,
        trajectory: TDMTrajectory,
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, dict[str, LogScalar], tuple[torch.Tensor, Any]]:
        guidance_scale = get_optional_float(
            self.method_config,
            "real_score_guidance_scale",
            where="method.real_score_guidance_scale",
        )
        if guidance_scale is None:
            guidance_scale = 1.0

        context = self._sample_tdm_context(trajectory)
        source_timestep = context.timestep_source
        target_timestep = context.timestep_target

        batch.timesteps = source_timestep
        generator_pred_x0 = self.student.predict_x0(
            context.noisy_source,
            source_timestep,
            batch,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind="vsa",
        )
        device = generator_pred_x0.device

        with torch.no_grad():
            eps_source = flow_effective_noise(
                context.noisy_source,
                generator_pred_x0.detach(),
                context.sigma_source,
                eps=self._sigma_eps,
            )
            sigma_intermediate_b = _expand_sigma_for_latents(context.sigma_intermediate, generator_pred_x0)
            noisy_intermediate = ((1.0 - sigma_intermediate_b) * generator_pred_x0.detach() +
                                  sigma_intermediate_b * eps_source)
            target_noisy_latents, _, _ = flow_transition_to_noisier_sigma(
                noisy_from=noisy_intermediate,
                clean_latents=generator_pred_x0.detach(),
                eps_from=eps_source,
                sigma_from=context.sigma_intermediate,
                sigma_to=context.sigma_target,
                proposal_noise=context.proposal_noise,
                eps=self._sigma_eps,
            )
            critic_timestep = self._model_timestep_for_sigma(context.sigma_target, self.critic)
            teacher_timestep = self._model_timestep_for_sigma(context.sigma_target, self.teacher)
            batch.timesteps = critic_timestep
            faker_x0 = self.critic.predict_x0(
                target_noisy_latents,
                critic_timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            batch.timesteps = teacher_timestep
            real_cond_x0 = self.teacher.predict_x0(
                target_noisy_latents,
                teacher_timestep,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            real_uncond_x0 = self.teacher.predict_x0(
                target_noisy_latents,
                teacher_timestep,
                batch,
                conditional=False,
                cfg_uncond=self._cfg_uncond,
                attn_kind="dense",
            )
            real_cfg_x0 = real_uncond_x0 + (real_cond_x0 - real_uncond_x0) * float(guidance_scale)
            delta = real_cfg_x0 - faker_x0
            raw_delta_abs_mean = delta.detach().float().abs().mean()
            denom = torch.ones((), device=device, dtype=torch.float32)
            if self._normalize_generator_delta:
                reduce_dims = tuple(range(1, generator_pred_x0.ndim))
                denom = torch.abs(generator_pred_x0.detach() - real_cfg_x0).mean(
                    dim=reduce_dims,
                    keepdim=True,
                )
            target_delta = torch.nan_to_num(delta)
            target = generator_pred_x0.detach() + target_delta

        if self._use_pseudo_huber:
            loss = self._generator_pseudo_huber_loss(generator_pred_x0, target)
        else:
            loss = self._generator_elementwise_loss(generator_pred_x0, target)
            if self._normalize_generator_delta:
                loss = loss / denom.clamp_min(self._sigma_eps)
        batch.dmd_latent_vis_dict.update({
            "dmd_timestep": target_timestep.float().detach(),
            "generator_timestep": source_timestep.float().detach(),
            "generator_pred_video": generator_pred_x0.detach(),
        })
        metrics: dict[str, LogScalar] = {
            "tdm/generator/source_trajectory_index": context.trajectory_indices.detach().float().mean(),
            "tdm/generator/source_timestep": source_timestep.detach().float().mean(),
            "tdm/generator/target_timestep": target_timestep.detach().float().mean(),
            "tdm/generator/source_sigma": context.sigma_source.detach().float().mean(),
            "tdm/generator/intermediate_sigma": context.sigma_intermediate.detach().float().mean(),
            "tdm/generator/target_sigma": context.sigma_target.detach().float().mean(),
            "tdm/generator/raw_delta_abs_mean": raw_delta_abs_mean,
            "tdm/generator/target_delta_abs_mean": target_delta.detach().float().abs().mean(),
            "tdm/generator/normalization_denom": denom.detach().float().mean(),
            "tdm/generator/normalize_delta": float(self._normalize_generator_delta),
            "tdm/generator/use_pseudo_huber": float(self._use_pseudo_huber),
        }
        return loss.mean(), metrics, (source_timestep, batch.attn_metadata_vsa)

    def _generator_pseudo_huber_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Paper Eq. 11 surrogate: per-sample vector-norm pseudo-Huber.

        ``sqrt(||pred - target||_2^2 + c^2) - c`` with ``c = 0.00054 * sqrt(d)``
        and ``d`` the flattened per-sample latent size. No DMD delta
        normalization is applied.
        """
        error = pred.float() - target.float()
        per_sample_dim = error[0].numel()
        huber_c = 0.00054 * math.sqrt(float(per_sample_dim))
        error_norm = error.flatten(1).norm(dim=1)
        return torch.sqrt(error_norm.square() + huber_c**2) - huber_c

    def _generator_elementwise_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        error = pred.float() - target.float()
        if self._use_huber:
            return torch.sqrt(error.square() + self._huber_c**2) - self._huber_c
        return error.square()
