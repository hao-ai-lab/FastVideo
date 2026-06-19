# SPDX-License-Identifier: Apache-2.0
"""Configurable diffusion samplers for RL training methods."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import torch

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler, )
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.base import ModelBase

SchedulerName = Literal["flow_match_euler", "flow_unipc", "model_default"]
TrajectoryName = Literal["ode", "sde_reflow"]


@dataclass(slots=True)
class SamplingConfig:
    """YAML-backed sampling knobs shared by RL methods."""

    num_steps: int = 25
    scheduler: SchedulerName = "model_default"
    trajectory: TrajectoryName = "ode"
    flow_shift: float | None = None
    guidance_scale: float = 1.0
    timesteps: list[float] | None = None
    sigmas: list[float] | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> SamplingConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"method.sampling must be a mapping, got {type(raw).__name__}")
        supported_keys = {
            "flow_shift",
            "guidance_scale",
            "num_steps",
            "scheduler",
            "sigmas",
            "timesteps",
            "trajectory",
        }
        unsupported_keys = sorted(set(raw) - supported_keys)
        if unsupported_keys:
            raise ValueError(f"Unsupported method.sampling key(s): {unsupported_keys}. "
                             f"Supported keys: {sorted(supported_keys)}")
        scheduler = str(raw.get("scheduler", "model_default") or "model_default").strip().lower()
        if scheduler not in {"flow_match_euler", "flow_unipc", "model_default"}:
            raise ValueError("method.sampling.scheduler must be one of "
                             "{flow_match_euler, flow_unipc, model_default}, got "
                             f"{raw.get('scheduler')!r}")
        trajectory = str(raw.get("trajectory", "ode") or "ode").strip().lower()
        if trajectory not in {"ode", "sde_reflow"}:
            raise ValueError("method.sampling.trajectory must be one of "
                             "{ode, sde_reflow}, got "
                             f"{raw.get('trajectory')!r}")
        timesteps = raw.get("timesteps")
        sigmas = raw.get("sigmas")
        if timesteps is not None:
            if not isinstance(timesteps, list) or not timesteps:
                raise ValueError("method.sampling.timesteps must be a non-empty list when set")
            timesteps = [float(t) for t in timesteps]
        if sigmas is not None:
            if not isinstance(sigmas, list) or not sigmas:
                raise ValueError("method.sampling.sigmas must be a non-empty list when set")
            sigmas = [float(s) for s in sigmas]
        if timesteps is not None and sigmas is not None and len(timesteps) != len(sigmas):
            raise ValueError("method.sampling.timesteps and method.sampling.sigmas must have the same length")
        if scheduler == "flow_unipc" and timesteps is not None:
            raise ValueError("method.sampling.timesteps is not supported with flow_unipc; "
                             "use num_steps or sigmas instead")
        num_steps = int(raw.get("num_steps", 25) or 25)
        if num_steps <= 0:
            raise ValueError("method.sampling.num_steps must be positive")
        guidance_scale = float(raw.get("guidance_scale", 1.0) or 1.0)
        if guidance_scale < 0.0:
            raise ValueError("method.sampling.guidance_scale must be non-negative")
        return cls(
            num_steps=num_steps,
            scheduler=scheduler,  # type: ignore[arg-type]
            trajectory=trajectory,  # type: ignore[arg-type]
            flow_shift=(None if raw.get("flow_shift", None) in (None, "inherit") else float(raw["flow_shift"])),
            guidance_scale=guidance_scale,
            timesteps=timesteps,
            sigmas=sigmas,
        )


@dataclass(slots=True)
class SamplingResult:
    latents: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor


class DiffusionSampler:
    """Thin model/scheduler sampler used by RL methods.

    This intentionally does not call FastVideo's full inference pipelines.
    RL training needs a reusable sampling primitive that works with
    ``ModelBase`` wrappers and scheduler math without binding a method to
    model-family pipeline classes such as ``WanDMDPipeline``.
    """

    def __init__(self, config: SamplingConfig) -> None:
        self.config = config

    @torch.no_grad()
    def sample(
        self,
        model: ModelBase,
        batch: TrainingBatch,
        *,
        generator: torch.Generator | None,
    ) -> SamplingResult:
        latents = batch.latents
        if latents is None:
            raise RuntimeError("TrainingBatch.latents is required for RL sampling")
        current = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=generator,
        )

        scheduler = self._prepare_scheduler(model, current.device)
        timesteps = scheduler.timesteps.to(device=current.device)
        sigmas = scheduler.sigmas.to(device=current.device)

        original_timesteps = batch.timesteps
        try:
            if self.config.trajectory == "ode":
                pred_clean = current
                for timestep in timesteps:
                    model_timestep = self._model_timestep(timestep, current)
                    batch.timesteps = model_timestep
                    pred_noise = self._predict_with_cfg(model, current, model_timestep, batch)
                    current = scheduler.step(
                        pred_noise.flatten(0, 1),
                        timestep,
                        current.flatten(0, 1),
                        return_dict=False,
                    )[0].unflatten(0, pred_noise.shape[:2])
                    pred_clean = current
                return SamplingResult(latents=pred_clean, timesteps=timesteps, sigmas=sigmas)

            return SamplingResult(
                latents=self._sample_sde_reflow(
                    model,
                    batch,
                    current,
                    timesteps,
                    generator=generator,
                ),
                timesteps=timesteps,
                sigmas=sigmas,
            )
        finally:
            batch.timesteps = original_timesteps

    def _prepare_scheduler(
        self,
        model: ModelBase,
        device: torch.device,
    ) -> Any:
        if self.config.scheduler == "flow_match_euler":
            shift = self.config.flow_shift
            if shift is None:
                shift = float(getattr(model.noise_scheduler, "shift", 1.0))
            scheduler = FlowMatchEulerDiscreteScheduler(shift=float(shift))
        elif self.config.scheduler == "flow_unipc":
            shift = self.config.flow_shift
            if shift is None:
                shift = float(getattr(model.noise_scheduler, "shift", 1.0))
            scheduler = FlowUniPCMultistepScheduler(shift=float(shift))
        else:
            scheduler = copy.deepcopy(model.noise_scheduler)
        kwargs: dict[str, Any] = {"device": device}
        if self.config.timesteps is not None:
            kwargs["timesteps"] = self.config.timesteps
            kwargs["num_inference_steps"] = len(self.config.timesteps)
        if self.config.sigmas is not None:
            kwargs["sigmas"] = self.config.sigmas
            kwargs["num_inference_steps"] = len(self.config.sigmas)
        if "num_inference_steps" not in kwargs:
            kwargs["num_inference_steps"] = self.config.num_steps
        scheduler.set_timesteps(**kwargs)
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(0)
        return scheduler

    def _sample_sde_reflow(
        self,
        model: ModelBase,
        batch: TrainingBatch,
        current: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        pred_clean = current
        for step_idx, timestep in enumerate(timesteps):
            timestep_tensor = self._model_timestep(timestep, current)
            batch.timesteps = timestep_tensor
            pred_clean = self._predict_x0_with_cfg(model, current, timestep_tensor, batch)
            if step_idx < len(timesteps) - 1:
                next_timestep = timesteps[step_idx + 1].reshape(1).to(device=current.device)
                noise = torch.randn(
                    pred_clean.shape,
                    device=pred_clean.device,
                    dtype=pred_clean.dtype,
                    generator=generator,
                )
                current = model.add_noise(pred_clean, noise, next_timestep)
        return pred_clean

    def _predict_with_cfg(
        self,
        model: ModelBase,
        current: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
    ) -> torch.Tensor:
        cond = model.predict_noise(
            current,
            timestep,
            batch,
            conditional=True,
            attn_kind="dense",
        )
        guidance_scale = float(self.config.guidance_scale)
        if guidance_scale == 1.0:
            return cond
        uncond = model.predict_noise(
            current,
            timestep,
            batch,
            conditional=False,
            attn_kind="dense",
        )
        return uncond + guidance_scale * (cond - uncond)

    def _predict_x0_with_cfg(
        self,
        model: ModelBase,
        current: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
    ) -> torch.Tensor:
        cond = model.predict_x0(
            current,
            timestep,
            batch,
            conditional=True,
            attn_kind="dense",
        )
        guidance_scale = float(self.config.guidance_scale)
        if guidance_scale == 1.0:
            return cond
        uncond = model.predict_x0(
            current,
            timestep,
            batch,
            conditional=False,
            attn_kind="dense",
        )
        return uncond + guidance_scale * (cond - uncond)

    @staticmethod
    def _model_timestep(
        timestep: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        return timestep.reshape(1).to(device=current.device).expand(current.shape[0]).contiguous()
