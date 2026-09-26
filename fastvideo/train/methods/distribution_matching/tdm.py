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
        # One joint model can carry several modalities (MiniMax H3 video +
        # audio); the same code paths loop over this tuple, and video-only
        # models resolve to a single "video" entry.
        self._modalities: tuple[str, ...] = tuple(self.student.tdm_modalities())
        self._denoising_step_list: torch.Tensor | None = None
        self._denoising_sigma_lists: dict[str, torch.Tensor] | None = None
        self._step_schedules = self._parse_step_schedules()
        self._active_schedule_index = 0
        self._cached_schedule_index: int | None = None
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
        # Sparse-attention scope for the roles TDM drives. The standalone H3
        # recipe validated sparse attention on the student only; "all" runs the
        # critic and teacher through the sparse metadata too (their attention
        # backends must then be the matching sparse backend).
        self._vsa_apply_to: Literal["student", "all"] = require_choice(
            mcfg,
            "tdm_vsa_apply_to",
            {"student", "all"},
            default="student",
            where="method.tdm_vsa_apply_to",
        )  # type: ignore[assignment]

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
        self._warmup_steps = self._resolve_warmup_steps()
        self.method_config["tdm_warmup_steps"] = self._warmup_steps
        # Validation and inference diagnostics should follow the final
        # (deployment) ladder stage, which is what the run is training for.
        if self._step_schedules[-1][1] is not None:
            self.method_config["dmd_denoising_steps"] = list(self._step_schedules[-1][0])
        else:
            self.method_config.setdefault("dmd_denoising_steps", list(self._step_schedules[-1][0]))

        if self._model_family() == "minimax_h3" and self._real_score_guidance() != 1.0:
            raise ValueError("MiniMax H3 TDM requires method.real_score_guidance_scale == 1.0 "
                             "because the model has no unconditional branch for CFG guidance")
        self._validate_wan_validation_shift()

    def _validate_wan_validation_shift(self) -> None:
        """Reject a Wan config whose validation cannot match the trained grid.

        Wan TDM validation runs the dense DMD sampler, whose training-noise
        scheduler is pinned to ``DMD_TRAINING_NOISE_SHIFT``. When the DMD
        ladder is read as raw labels (``dmd_denoising_steps_are_scheduler_space``
        false), the sampler shifts those labels by that fixed amount, so it
        matches TDM's student grid only when the student's *effective*
        scheduler shift is the same value. When the labels are already
        scheduler-space, the sampler reads them as raw sigmas (``label / T``)
        and disregards its own shift, so the grid matches only an identity
        student shift. The effective shift is resolved by
        ``WanModel._resolve_flow_shift`` from ``models.<role>.flow_shift``
        first, then ``pipeline.flow_shift``, then 3.0, so compare the resolved
        value rather than the pipeline field alone.
        """
        if self._model_family() != "wan":
            return
        pipeline_config = getattr(self.training_config, "pipeline_config", None)
        if pipeline_config is None:
            return
        scheduler = getattr(self.student, "noise_scheduler", None)
        shift = getattr(scheduler, "shift", None)
        if shift is None:
            # Model plugins without a resolved scheduler mirror the same
            # resolution order: pipeline shift, else the 3.0 fallback.
            pipeline_flow_shift = getattr(pipeline_config, "flow_shift", None)
            shift = 3.0 if pipeline_flow_shift is None else float(pipeline_flow_shift)
        from fastvideo.models.wan.definition import DMD_TRAINING_NOISE_SHIFT
        if bool(getattr(pipeline_config, "dmd_denoising_steps_are_scheduler_space", True)):
            if float(shift) != 1.0:
                raise ValueError(
                    "Wan TDM validation reads dmd_denoising_steps as scheduler-space "
                    "labels (dmd_denoising_steps_are_scheduler_space=True), so the dense "
                    "DMD sampler samples raw sigmas (label / T) and disregards its pinned "
                    "training-noise shift. That matches TDM's shifted grid only when the "
                    "student's resolved flow_shift is 1.0; got "
                    f"{float(shift)}. Unset dmd_denoising_steps_are_scheduler_space so the "
                    "training-noise shift is applied, or set the student's flow shift to 1.0.")
            return
        if float(shift) != DMD_TRAINING_NOISE_SHIFT:
            raise ValueError("Wan TDM validation samples through the dense DMD sampler, whose "
                             f"training-noise shift is fixed at {DMD_TRAINING_NOISE_SHIFT}; "
                             f"the student's resolved flow_shift={float(shift)} would make "
                             "validation disagree with the trained sigma grid. Set the student's "
                             "flow shift (models.student.flow_shift overrides pipeline.flow_shift) "
                             f"to {DMD_TRAINING_NOISE_SHIFT}.")

    def manages_optimization(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # phase gating (warmup)
    # ------------------------------------------------------------------

    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        if self._in_warmup(iteration):
            return [self._student_optimizer]
        return super().get_optimizers(iteration)

    def get_lr_schedulers(
        self,
        iteration: int,
    ) -> list[Any]:
        if self._in_warmup(iteration):
            return [self._student_lr_scheduler]
        return super().get_lr_schedulers(iteration)

    def get_grad_clip_targets(
        self,
        iteration: int,
    ) -> dict[str, torch.nn.Module]:
        if self._in_warmup(iteration):
            return {"student": self.student.transformer}
        return super().get_grad_clip_targets(iteration)

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
        self._select_schedule_stage(iteration)
        if self._in_warmup(iteration):
            return self._managed_warmup_step(data_stream, iteration, grad_accum)
        raw_batches = [next(data_stream) for _ in range(grad_accum)]

        fake_score_losses: list[torch.Tensor] = []
        fake_score_metric_maps: list[dict[str, LogScalar]] = []
        prepared_trajectories: list[tuple[TrainingBatch, dict[str, TDMTrajectory]]] = []
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
            prepared_trajectories.append((training_batch, trajectory))
            fake_score_loss, critic_ctx, _, fake_score_metrics = self._tdm_fake_score_loss(
                trajectory,
                training_batch,
            )
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
            for training_batch, trajectory in prepared_trajectories:
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
            "tdm/warmup": 0.0,
            "tdm/warmup_steps": float(self._warmup_steps),
            "tdm/step_ladder_stage": float(self._active_schedule_index),
        }
        if update_student:
            metrics["grad_norm/student"] = student_grad_norm
        metrics.update(self._mean_metric_maps(fake_score_metric_maps))
        metrics.update(self._mean_metric_maps(generator_metric_maps))
        return loss_map, {}, metrics

    def _sample_warmup_source(
        self,
        trajectories: dict[str, TDMTrajectory],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Pick the rollout state each sample regresses at (detached).

        One trajectory index drives every modality so the joint state stays
        synchronized at the same denoising step.
        """
        reference = trajectories["video"].sigmas
        device = reference.device
        batch_size = trajectories["video"].noisy_latents[0].shape[0]
        batch_indices = torch.arange(batch_size, device=device)
        trajectory_indices = torch.randint(
            0,
            len(reference),
            [batch_size],
            device=device,
            dtype=torch.long,
            generator=self.cuda_generator,
        )
        noisy_source = {
            name: torch.stack(trajectory.noisy_latents)[trajectory_indices, batch_indices].detach()
            for name, trajectory in trajectories.items()
        }
        sigma_source = {name: trajectory.sigmas[trajectory_indices] for name, trajectory in trajectories.items()}
        return noisy_source, sigma_source, trajectory_indices

    def _tdm_warmup_loss(
        self,
        trajectories: dict[str, TDMTrajectory],
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, dict[str, LogScalar], tuple[torch.Tensor, Any]]:
        """Regression warmup step toward the CFG-combined teacher x0.

        The standalone-validated warmup: the student's x0 at its own
        rollout states is regressed onto the guidance-combined teacher x0
        at those same states, with no critic update. This bakes CFG
        sharpness into the unguided student and puts it inside the
        teacher's basin before the TDM correction is enabled. Each
        modality contributes its own squared error and the losses are
        summed, matching the standalone joint warmup.
        """
        noisy_source, sigma_source, trajectory_indices = self._sample_warmup_source(trajectories)
        guidance = self._real_score_guidance()
        pred_x0 = self.student.tdm_predict_x0(
            noisy_source,
            sigma_source,
            batch,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind=self._attn_kind_for("student"),
        )
        student_timestep = batch.timesteps
        with torch.no_grad():
            real_cond_x0 = self.teacher.tdm_predict_x0(
                noisy_source,
                sigma_source,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind=self._attn_kind_for("teacher"),
            )
            if guidance == 1.0:
                real_cfg_x0 = real_cond_x0
            else:
                real_uncond_x0 = self.teacher.tdm_predict_x0(
                    noisy_source,
                    sigma_source,
                    batch,
                    conditional=False,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind=self._attn_kind_for("teacher"),
                )
                real_cfg_x0 = {
                    name: real_uncond_x0[name] + (real_cond_x0[name] - real_uncond_x0[name]) * guidance
                    for name in real_cond_x0
                }
        loss = sum((pred_x0[name].float() - real_cfg_x0[name].float()).square().mean() for name in pred_x0)
        video_sigma = sigma_source["video"]
        metrics: dict[str, LogScalar] = {
            "tdm/warmup/loss": loss.detach(),
            "tdm/warmup/guidance": guidance,
            "tdm/warmup/source_timestep": student_timestep.detach().float().mean(),
            "tdm/warmup/source_sigma": video_sigma.detach().float().mean(),
            "tdm/warmup/source_trajectory_index": trajectory_indices.detach().float().mean(),
        }
        return loss, metrics, (student_timestep, self._attn_metadata_for("student", batch))

    def _managed_warmup_step(
        self,
        data_stream: Iterator[dict[str, Any]],
        iteration: int,
        grad_accum: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del iteration
        losses: list[torch.Tensor] = []
        metric_maps: list[dict[str, LogScalar]] = []
        for _ in range(grad_accum):
            raw_batch = next(data_stream)
            training_batch = self.student.prepare_batch(
                raw_batch,
                generator=self.cuda_generator,
                latents_source="zeros",
            )
            if training_batch.latents is None:
                raise RuntimeError("TDM requires student.prepare_batch to populate latents")
            with torch.no_grad():
                trajectory = self._student_trajectory(training_batch)
            warmup_loss, warmup_metrics, student_ctx = self._tdm_warmup_loss(trajectory, training_batch)
            self.student.backward(
                warmup_loss,
                student_ctx,
                grad_accum_rounds=grad_accum,
            )
            losses.append(warmup_loss.detach())
            metric_maps.append(warmup_metrics)
        student_grad_norm = self._finish_role_update(
            model=self.student,
            optimizer=self._student_optimizer,
            lr_scheduler=self._student_lr_scheduler,
        )
        warmup_loss = torch.stack(losses).mean()
        loss_map = {
            "total_loss": warmup_loss,
            "generator_loss": warmup_loss,
            "fake_score_loss": torch.zeros_like(warmup_loss),
        }
        metrics: dict[str, LogScalar] = {
            "update_student": 1.0,
            "grad_norm/student": student_grad_norm,
            "tdm/warmup": 1.0,
            "tdm/warmup_steps": float(self._warmup_steps),
            "tdm/step_ladder_stage": float(self._active_schedule_index),
        }
        metrics.update(self._mean_metric_maps(metric_maps))
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

    def _attn_kind_for(self, role: str) -> Literal["dense", "vsa"]:
        """Attention kind TDM uses for one role's forward passes."""
        if role == "student" or self._vsa_apply_to == "all":
            return "vsa"
        return "dense"

    def _attn_metadata_for(self, role: str, batch: TrainingBatch) -> Any:
        """Metadata matching the forward ``_attn_kind_for(role)`` selects.

        The backward context must restore the exact attention metadata the
        forward used, because activation-checkpointed layers re-read it from
        the forward context during the recompute.
        """
        if self._attn_kind_for(role) == "vsa":
            return batch.attn_metadata_vsa
        return batch.attn_metadata

    def _model_family(self) -> str:
        """Best-effort family name of the student plugin (``wan``, ...)."""
        module = type(self.student).__module__
        for family in ("wan", "minimax_h3"):
            if f".{family}" in module:
                return family
        return "unknown"

    def _default_warmup_steps(self) -> int:
        """Regression-warmup length defaults by family.

        The standalone TDM experiments showed that TDM alone drifts out of
        the teacher distribution on Wan while warmup alone collapses
        visually; the working recipe is a regression warmup to the
        CFG-combined teacher followed by TDM. Wan therefore defaults to
        ``200`` warmup updates. Guidance-distilled bases (H3) already
        start inside the teacher's basin, so they default to no warmup.
        """
        return 200 if self._model_family() == "wan" else 0

    def _resolve_warmup_steps(self) -> int:
        raw = self.method_config.get("warmup_steps", None)
        if raw is None:
            return self._default_warmup_steps()
        warmup_steps = int(raw)
        if warmup_steps < 0:
            raise ValueError("method.warmup_steps must be non-negative")
        return warmup_steps

    def _in_warmup(self, iteration: int) -> bool:
        return iteration < self._warmup_steps

    def _real_score_guidance(self) -> float:
        guidance = get_optional_float(
            self.method_config,
            "real_score_guidance_scale",
            where="method.real_score_guidance_scale",
        )
        return 1.0 if guidance is None else float(guidance)

    def _parse_step_schedules(self) -> list[tuple[list[int], int | None]]:
        """Resolve the student step schedule, optionally as a ladder.

        ``method.tdm_step_ladder`` is a list of stages
        ``{denoising_steps: [...], until_iteration: int}``; the final
        stage omits ``until_iteration``. Stage step counts must strictly
        decrease (the curriculum direction validated by the standalone
        experiments: high step counts first, deployment count last).
        Without a ladder the single schedule comes from
        ``method.tdm_denoising_steps`` (or ``dmd_denoising_steps``).
        """
        mcfg = self.method_config
        ladder = mcfg.get("tdm_step_ladder", None)
        if ladder is None:
            raw = mcfg.get("tdm_denoising_steps", None)
            if raw is None:
                raw = mcfg.get("dmd_denoising_steps", None)
            if not isinstance(raw, list) or not raw:
                raise ValueError("method.tdm_denoising_steps must be set for TDM")
            return [([int(s) for s in raw], None)]

        if not isinstance(ladder, list) or not ladder:
            raise ValueError("method.tdm_step_ladder must be a non-empty list of stages")
        stages: list[tuple[list[int], int | None]] = []
        previous_until = 0
        previous_count: int | None = None
        for position, stage in enumerate(ladder):
            if not isinstance(stage, dict):
                raise ValueError("method.tdm_step_ladder entries must be mappings")
            steps = stage.get("denoising_steps", None)
            if not isinstance(steps, list) or not steps:
                raise ValueError(f"method.tdm_step_ladder[{position}].denoising_steps must be a non-empty list")
            if previous_count is not None and len(steps) >= previous_count:
                raise ValueError("method.tdm_step_ladder stages must strictly decrease the student step count")
            previous_count = len(steps)
            until = stage.get("until_iteration", None)
            if until is None:
                if position != len(ladder) - 1:
                    raise ValueError("only the last method.tdm_step_ladder stage may omit until_iteration")
                stages.append(([int(s) for s in steps], None))
                continue
            until = int(until)
            if until <= previous_until:
                raise ValueError("method.tdm_step_ladder until_iteration values must be strictly increasing")
            previous_until = until
            stages.append(([int(s) for s in steps], until))
        return stages

    def _select_schedule_stage(self, iteration: int) -> None:
        for index, (_, until) in enumerate(self._step_schedules):
            if until is None or iteration < until:
                self._active_schedule_index = index
                return
        self._active_schedule_index = len(self._step_schedules) - 1

    def _get_denoising_step_list(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        if (self._denoising_step_list is not None and self._denoising_step_list.device == device
                and self._cached_schedule_index == self._active_schedule_index):
            return self._denoising_step_list

        raw = self._step_schedules[self._active_schedule_index][0]
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

        sigmas_by_modality = ({
            modality: self._timestep_to_sigma(steps, scheduler_space=True)
            for modality in self._modalities
        } if bool(warp) else {
            modality:
            self.student.tdm_sigma_grid(steps, modality).to(device=device, dtype=torch.float32)
            for modality in self._modalities
        })
        for modality, sigmas in sigmas_by_modality.items():
            terminal_sigma = self.student.tdm_terminal_sigma(modality).to(device=device, dtype=torch.float32)
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
        self._denoising_sigma_lists = sigmas_by_modality
        self._cached_schedule_index = self._active_schedule_index
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
        modality: str = "video",
    ) -> torch.Tensor:
        """Resolve an authoritative flow sigma to a model scheduler label."""
        return model.tdm_sigma_to_model_timestep(sigma, modality)

    def _student_trajectory(
        self,
        batch: TrainingBatch,
    ) -> dict[str, TDMTrajectory]:
        """Roll the few-step student trajectory out for every modality.

        One joint forward per step produces all modalities, so video and
        audio stay synchronized on a shared denoising schedule while each
        modality advances with its own sigma grid.
        """
        if batch.latents is None:
            raise RuntimeError("TDM requires prepared batch latents")
        clean = self.student.tdm_clean_latents(batch)
        device = batch.latents.device
        dtype = batch.latents.dtype
        step_list = self._get_denoising_step_list(device)
        sigma_lists = self._denoising_sigma_lists
        if sigma_lists is None:
            raise RuntimeError("TDM denoising sigmas were not initialized with the step schedule")
        if len(step_list) < 2:
            raise ValueError("TDM requires at least two denoising steps")

        current = self.student.tdm_initial_noise(
            clean,
            generator=self.cuda_generator,
            dtype=dtype,
        )
        noisy_latents: dict[str, list[torch.Tensor]] = {name: [] for name in self._modalities}
        clean_latents: dict[str, list[torch.Tensor]] = {name: [] for name in self._modalities}
        sigmas: dict[str, list[torch.Tensor]] = {name: [] for name in self._modalities}

        for step_idx in range(len(step_list)):
            step_sigmas = {name: sigma_lists[name][step_idx].reshape(1) for name in self._modalities}
            with torch.no_grad():
                pred_x0 = self.student.tdm_predict_x0(
                    current,
                    step_sigmas,
                    batch,
                    conditional=True,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind=self._attn_kind_for("student"),
                )
            for name in self._modalities:
                noisy_latents[name].append(current[name])
                clean_latents[name].append(pred_x0[name])
                sigmas[name].append(step_sigmas[name][0])

            if step_idx + 1 >= len(step_list):
                break

            for name in self._modalities:
                sigma = step_sigmas[name]
                sigma_next = sigma_lists[name][step_idx + 1].reshape(1)
                if self._rollout_sample_type == "sde":
                    noise = torch.randn(
                        current[name].shape,
                        device=current[name].device,
                        dtype=current[name].dtype,
                        generator=self.cuda_generator,
                    )
                    sigma_next_b = _expand_sigma_for_latents(sigma_next, current[name])
                    current[name] = (1.0 - sigma_next_b) * pred_x0[name] + sigma_next_b * noise
                else:
                    eps = flow_effective_noise(
                        current[name],
                        pred_x0[name],
                        sigma,
                        eps=self._sigma_eps,
                    )
                    sigma_next_b = _expand_sigma_for_latents(sigma_next, current[name])
                    current[name] = (1.0 - sigma_next_b) * pred_x0[name] + sigma_next_b * eps

        timesteps = step_list.to(device=device)
        batch.dmd_latent_vis_dict["generator_timestep"] = timesteps[-1].float().detach()
        return {
            name:
            TDMTrajectory(
                noisy_latents=noisy_latents[name],
                clean_latents=clean_latents[name],
                timesteps=timesteps,
                sigmas=torch.stack(sigmas[name]).to(device=device),
            )
            for name in self._modalities
        }

    def _sample_tdm_context(
        self,
        trajectories: dict[str, TDMTrajectory],
    ) -> dict[str, TDMSampleContext]:
        """Sample one source/intermediate/target context per modality.

        Video and audio share the sampled integer trajectory labels, so the
        joint state stays synchronized; each modality maps those labels to
        its own sigma grid and builds its own noise tensors.
        """
        reference = trajectories["video"]
        device = reference.sigmas.device
        batch_size = reference.noisy_latents[0].shape[0]
        batch_indices = torch.arange(batch_size, device=device)
        trajectory_indices = torch.randint(
            0,
            len(reference.sigmas),
            [batch_size],
            device=device,
            dtype=torch.long,
            generator=self.cuda_generator,
        )
        step_labels = reference.timesteps
        max_label = int(self.student.tdm_max_trajectory_label("video"))
        # Descending labels keep the candidate order identical to the
        # scheduler-sigma sampling this replaces (largest sigma first).
        all_labels = torch.arange(max_label, 0, -1, device=device)
        label_source = step_labels[trajectory_indices]
        next_labels = torch.cat((step_labels[1:], step_labels.new_zeros(1)))
        label_intermediate = next_labels[trajectory_indices]

        def sample_label(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
            sampled: list[torch.Tensor] = []
            for lower_i, upper_i in zip(lower, upper, strict=False):
                candidates = torch.nonzero(
                    (all_labels >= lower_i) & (all_labels < upper_i) & (all_labels > 0),
                    as_tuple=False,
                ).flatten()
                if candidates.numel() == 0:
                    raise ValueError("TDM schedule has no step in the requested noise interval")
                position = torch.randint(
                    0,
                    candidates.numel(),
                    [1],
                    device=device,
                    dtype=torch.long,
                    generator=self.cuda_generator,
                )
                sampled.append(all_labels[candidates[position]].reshape(()))
            return torch.stack(sampled).to(dtype=torch.float32)

        if self._use_randmid:
            label_intermediate = sample_label(label_intermediate, label_source)

        target_upper = label_source
        if self._noise_interval_mode == "to_terminal":
            target_upper = torch.full_like(label_source, float(max_label))
        label_target = sample_label(label_intermediate, target_upper)

        contexts: dict[str, TDMSampleContext] = {}
        for name, trajectory in trajectories.items():
            clean_latents = torch.stack(trajectory.clean_latents)[trajectory_indices, batch_indices].detach()
            noisy_source = torch.stack(trajectory.noisy_latents)[trajectory_indices, batch_indices].detach()
            sigma_source = trajectory.sigmas[trajectory_indices]
            sigma_intermediate = self.student.tdm_sigma_grid(label_intermediate, name).to(
                device=device,
                dtype=torch.float32,
            )
            sigma_target = self.student.tdm_sigma_grid(label_target, name).to(
                device=device,
                dtype=torch.float32,
            )

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

            contexts[name] = TDMSampleContext(
                clean_latents=clean_latents,
                noisy_source=noisy_source,
                noisy_intermediate=noisy_intermediate,
                noisy_target=noisy_target,
                timestep_source=self.student.tdm_sigma_to_model_timestep(sigma_source, name),
                timestep_target=self.student.tdm_sigma_to_model_timestep(sigma_target, name),
                sigma_source=sigma_source,
                sigma_intermediate=sigma_intermediate,
                sigma_target=sigma_target,
                eps_source=eps_source,
                mixed_noise=mixed_noise,
                proposal_noise=proposal_noise,
                transition_beta=beta,
                trajectory_indices=trajectory_indices,
            )
        return contexts

    def _tdm_fake_score_loss(
        self,
        trajectories: dict[str, TDMTrajectory],
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, Any, dict[str, Any], dict[str, LogScalar]]:
        with torch.no_grad():
            contexts = self._sample_tdm_context(trajectories)

        fake_x0 = self.critic.tdm_predict_x0(
            {
                name: context.noisy_target
                for name, context in contexts.items()
            },
            {
                name: context.sigma_target
                for name, context in contexts.items()
            },
            batch,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind=self._attn_kind_for("critic"),
        )
        critic_timestep = batch.timesteps
        per_sample_by_modality: dict[str, torch.Tensor] = {}
        weight_by_modality: dict[str, dict[str, torch.Tensor]] = {}
        fake_score_loss: torch.Tensor | None = None
        for name, context in contexts.items():
            elementwise = (fake_x0[name].float() - context.clean_latents.float()).square()
            per_sample = _mean_except_batch(elementwise)
            weight_components = self._tdm_fake_score_weight_components(context)
            weights = weight_components["weights"].to(device=per_sample.device, dtype=per_sample.dtype)
            component = (per_sample * weights).mean()
            fake_score_loss = component if fake_score_loss is None else fake_score_loss + component
            per_sample_by_modality[name] = per_sample
            weight_by_modality[name] = weight_components
        assert fake_score_loss is not None

        video_context = contexts["video"]
        batch.fake_score_latent_vis_dict = {
            "generator_pred_video": video_context.clean_latents,
            "fake_score_timestep": video_context.timestep_target.float().detach(),
            "fake_score_source_timestep": video_context.timestep_source.float().detach(),
        }
        outputs = {"fake_score_latent_vis_dict": (batch.fake_score_latent_vis_dict)}
        metrics = self._tdm_fake_score_metrics(
            contexts,
            per_sample_by_modality,
            weight_by_modality,
        )
        return (
            fake_score_loss,
            (critic_timestep, self._attn_metadata_for("critic", batch)),
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
        contexts: dict[str, TDMSampleContext],
        per_sample_by_modality: dict[str, torch.Tensor],
        weight_by_modality: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, LogScalar]:
        """Per-modality fake-score diagnostics averaged for logging."""
        per_modality = [
            self._tdm_fake_score_metrics_single(
                context,
                per_sample_by_modality[name],
                weight_by_modality[name],
            ) for name, context in contexts.items()
        ]
        if len(per_modality) == 1:
            return per_modality[0]
        return self._mean_metric_maps(per_modality)

    def _tdm_fake_score_metrics_single(
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
        trajectories: dict[str, TDMTrajectory],
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, dict[str, LogScalar], tuple[torch.Tensor, Any]]:
        guidance_scale = self._real_score_guidance()

        contexts = self._sample_tdm_context(trajectories)
        generator_pred_x0 = self.student.tdm_predict_x0(
            {
                name: context.noisy_source
                for name, context in contexts.items()
            },
            {
                name: context.sigma_source
                for name, context in contexts.items()
            },
            batch,
            conditional=True,
            cfg_uncond=self._cfg_uncond,
            attn_kind=self._attn_kind_for("student"),
        )
        source_timestep = batch.timesteps
        target_timestep = contexts["video"].timestep_target
        device = generator_pred_x0["video"].device

        with torch.no_grad():
            target_noisy_latents: dict[str, torch.Tensor] = {}
            for name, context in contexts.items():
                prediction = generator_pred_x0[name]
                eps_source = flow_effective_noise(
                    context.noisy_source,
                    prediction.detach(),
                    context.sigma_source,
                    eps=self._sigma_eps,
                )
                sigma_intermediate_b = _expand_sigma_for_latents(context.sigma_intermediate, prediction)
                noisy_intermediate = ((1.0 - sigma_intermediate_b) * prediction.detach() +
                                      sigma_intermediate_b * eps_source)
                target_noisy_latents[name], _, _ = flow_transition_to_noisier_sigma(
                    noisy_from=noisy_intermediate,
                    clean_latents=prediction.detach(),
                    eps_from=eps_source,
                    sigma_from=context.sigma_intermediate,
                    sigma_to=context.sigma_target,
                    proposal_noise=context.proposal_noise,
                    eps=self._sigma_eps,
                )
            target_sigmas = {name: context.sigma_target for name, context in contexts.items()}
            faker_x0 = self.critic.tdm_predict_x0(
                target_noisy_latents,
                target_sigmas,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind=self._attn_kind_for("critic"),
            )
            real_cond_x0 = self.teacher.tdm_predict_x0(
                target_noisy_latents,
                target_sigmas,
                batch,
                conditional=True,
                cfg_uncond=self._cfg_uncond,
                attn_kind=self._attn_kind_for("teacher"),
            )
            if float(guidance_scale) == 1.0:
                real_cfg_x0 = real_cond_x0
            else:
                real_uncond_x0 = self.teacher.tdm_predict_x0(
                    target_noisy_latents,
                    target_sigmas,
                    batch,
                    conditional=False,
                    cfg_uncond=self._cfg_uncond,
                    attn_kind=self._attn_kind_for("teacher"),
                )
                real_cfg_x0 = {
                    name: real_uncond_x0[name] + (real_cond_x0[name] - real_uncond_x0[name]) * float(guidance_scale)
                    for name in real_cond_x0
                }
            target_by_modality: dict[str, torch.Tensor] = {}
            denom_by_modality: dict[str, torch.Tensor] = {}
            delta_nan_by_modality: dict[str, torch.Tensor] = {}
            for name in real_cfg_x0:
                prediction = generator_pred_x0[name]
                delta = real_cfg_x0[name] - faker_x0[name]
                delta_nan = torch.nan_to_num(delta)
                delta_nan_by_modality[name] = delta_nan
                denom = torch.ones((), device=device, dtype=torch.float32)
                if self._normalize_generator_delta:
                    reduce_dims = tuple(range(1, prediction.ndim))
                    denom = torch.abs(prediction.detach() - real_cfg_x0[name]).mean(
                        dim=reduce_dims,
                        keepdim=True,
                    )
                denom_by_modality[name] = denom
                target_by_modality[name] = prediction.detach() + delta_nan
            raw_delta_abs_mean = sum((real_cfg_x0[name] - faker_x0[name]).detach().float().abs().mean()
                                     for name in real_cfg_x0) / len(real_cfg_x0)
            target_delta_abs_mean = sum(delta.detach().float().abs().mean()
                                        for delta in delta_nan_by_modality.values()) / len(delta_nan_by_modality)
            normalization_denom = sum(denom.detach().float().mean()
                                      for denom in denom_by_modality.values()) / len(denom_by_modality)

        loss: torch.Tensor | None = None
        for name in generator_pred_x0:
            if self._use_pseudo_huber:
                component = self._generator_pseudo_huber_loss(generator_pred_x0[name], target_by_modality[name])
            else:
                component = self._generator_elementwise_loss(generator_pred_x0[name], target_by_modality[name])
                if self._normalize_generator_delta:
                    component = component / denom_by_modality[name].clamp_min(self._sigma_eps)
            component = component.mean()
            loss = component if loss is None else loss + component
        assert loss is not None

        batch.dmd_latent_vis_dict.update({
            "dmd_timestep": target_timestep.float().detach(),
            "generator_timestep": source_timestep.float().detach(),
            "generator_pred_video": generator_pred_x0["video"].detach(),
        })
        metrics: dict[str, LogScalar] = {
            "tdm/generator/source_trajectory_index": contexts["video"].trajectory_indices.detach().float().mean(),
            "tdm/generator/source_timestep": source_timestep.detach().float().mean(),
            "tdm/generator/target_timestep": target_timestep.detach().float().mean(),
            "tdm/generator/source_sigma": contexts["video"].sigma_source.detach().float().mean(),
            "tdm/generator/intermediate_sigma": contexts["video"].sigma_intermediate.detach().float().mean(),
            "tdm/generator/target_sigma": contexts["video"].sigma_target.detach().float().mean(),
            "tdm/generator/raw_delta_abs_mean": raw_delta_abs_mean,
            "tdm/generator/target_delta_abs_mean": target_delta_abs_mean,
            "tdm/generator/normalization_denom": normalization_denom,
            "tdm/generator/normalize_delta": float(self._normalize_generator_delta),
            "tdm/generator/use_pseudo_huber": float(self._use_pseudo_huber),
        }
        return loss, metrics, (source_timestep, self._attn_metadata_for("student", batch))

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
