# SPDX-License-Identifier: Apache-2.0
"""Exact few-step sampler used by MiniMax H3 RVM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.minimax_h3.minimax_h3_rvm import MiniMaxH3RVMModel

_DEFAULT_DENOISING_STEPS = (1000, 750, 500, 250)


@dataclass(frozen=True, slots=True)
class H3RVMSamplingConfig:
    """The released FastH3 schedule uses four forwards and terminal zero."""

    denoising_steps: tuple[int, ...] = _DEFAULT_DENOISING_STEPS
    attn_kind: Literal["dense", "vsa"] = "vsa"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "H3RVMSamplingConfig":
        raw = dict(raw or {})
        unknown = sorted(set(raw) - {"denoising_steps", "attn_kind"})
        if unknown:
            raise ValueError(f"Unsupported H3 RVM sampling keys: {unknown}")
        steps = tuple(int(value) for value in raw.get("denoising_steps", _DEFAULT_DENOISING_STEPS))
        if not steps:
            raise ValueError("denoising_steps must be nonempty")
        if steps[0] != 1000 or any(a <= b for a, b in zip(steps, steps[1:], strict=False)):
            raise ValueError("denoising_steps must start at 1000 and be strictly decreasing")
        if steps[-1] <= 0:
            raise ValueError("the last denoising step must be positive; terminal zero is appended internally")
        attn_kind = str(raw.get("attn_kind", "vsa")).strip().lower()
        if attn_kind not in {"dense", "vsa"}:
            raise ValueError("attn_kind must be 'dense' or 'vsa'")
        return cls(
            denoising_steps=steps,
            attn_kind=attn_kind,  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class H3RVMSamplingResult:
    endpoint: torch.Tensor
    source_noise: torch.Tensor


class MiniMaxH3RVMSampler:
    """Deterministic Euler sampler over H3's paired video/audio noise grids."""

    def __init__(self, config: H3RVMSamplingConfig) -> None:
        self.config = config

    @torch.no_grad()
    def sample(
        self,
        model: MiniMaxH3RVMModel,
        batch: TrainingBatch,
        *,
        generator: torch.Generator,
    ) -> H3RVMSamplingResult:
        if batch.latents is None:
            raise RuntimeError("prepare_batch() must provide packed latent geometry")
        source_noise = torch.randn(
            batch.latents.shape,
            device=batch.latents.device,
            dtype=batch.latents.dtype,
            generator=generator,
        )
        current = source_noise.clone()
        steps = (*self.config.denoising_steps, 0)

        original = (
            batch.timesteps,
            batch.audio_timesteps,
            batch.audio_noisy_model_input,
            batch.attn_metadata_vsa,
            batch.current_timestep,
        )
        try:
            for step_index, (step, next_step) in enumerate(zip(steps[:-1], steps[1:], strict=True)):
                timestep = torch.tensor([step], device=current.device, dtype=torch.long)
                model.refresh_vsa_metadata(batch, current_timestep=step_index)
                clean = model.predict_x0(
                    current,
                    timestep,
                    batch,
                    conditional=True,
                    attn_kind=self.config.attn_kind,
                )
                if next_step == 0:
                    current = clean
                    continue
                next_timestep = torch.tensor([next_step], device=current.device, dtype=torch.long)
                current = self._bridge_to_next_noise(model, current, clean, timestep, next_timestep)
        finally:
            (
                batch.timesteps,
                batch.audio_timesteps,
                batch.audio_noisy_model_input,
                batch.attn_metadata_vsa,
                batch.current_timestep,
            ) = original
        return H3RVMSamplingResult(endpoint=current.detach(), source_noise=source_noise.detach())

    @staticmethod
    def _bridge_to_next_noise(
        model: MiniMaxH3RVMModel,
        current: torch.Tensor,
        clean: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
    ) -> torch.Tensor:
        current_video, current_audio = model.unpack_latents(current)
        clean_video, clean_audio = model.unpack_latents(clean)
        sigma_video, sigma_audio = model.noise_amounts(timestep)
        sigma_video_next, sigma_audio_next = model.noise_amounts(next_timestep)

        def bridge(sample: torch.Tensor, x0: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
            sigma = sigma.to(device=sample.device, dtype=torch.float32)
            sigma_next = sigma_next.to(device=sample.device, dtype=torch.float32)
            if bool((sigma <= 0).any()):
                raise ValueError("current sigma must be positive before the terminal step")
            ratio = sigma_next / sigma
            return (ratio * sample.float() + (1.0 - ratio) * x0.float()).to(sample.dtype)

        return model.pack_latents(
            bridge(current_video, clean_video, sigma_video, sigma_video_next),
            bridge(current_audio, clean_audio, sigma_audio, sigma_audio_next),
        )
