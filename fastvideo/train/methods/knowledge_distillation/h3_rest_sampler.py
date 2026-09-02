# SPDX-License-Identifier: Apache-2.0
"""Dense full-H3 sampler that records FastH3-aligned REST trajectory anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.knowledge_distillation.h3_rest_utils import (
    build_piecewise_teacher_schedule,
    teacher_anchor_indices,
)
from fastvideo.train.models.minimax_h3.minimax_h3_rest import (
    MiniMaxH3RESTTeacherModel,
)


@dataclass(frozen=True, slots=True)
class H3RESTSamplingConfig:
    """Dense teacher schedule aligned to the deployed FastH3 boundaries."""

    student_timesteps: tuple[float, ...] = (1000.0, 750.0, 500.0, 250.0, 0.0)
    substeps_per_segment: int = 12
    attn_kind: Literal["dense"] = "dense"

    def __post_init__(self) -> None:
        schedule = build_piecewise_teacher_schedule(
            self.student_timesteps,
            self.substeps_per_segment,
        )
        if self.student_timesteps[0] != 1000.0 or self.student_timesteps[-1] != 0.0:
            raise ValueError(
                "H3 REST student_timesteps must start at 1000 and end at 0"
            )
        if self.attn_kind != "dense":
            raise ValueError("Full-H3 REST cache generation must use dense attention")
        if len(schedule) - 1 != (
            len(self.student_timesteps) - 1
        ) * self.substeps_per_segment:
            raise AssertionError("Dense H3 REST schedule length is inconsistent")

    @property
    def dense_schedule(self) -> tuple[float, ...]:
        return build_piecewise_teacher_schedule(
            self.student_timesteps,
            self.substeps_per_segment,
        )

    @property
    def anchor_indices(self) -> tuple[int, ...]:
        return teacher_anchor_indices(
            len(self.student_timesteps) - 1,
            self.substeps_per_segment,
        )


@dataclass(frozen=True, slots=True)
class H3RESTSamplingResult:
    """One complete dense teacher rollout reduced to student-grid anchors."""

    anchor_states: torch.Tensor
    anchor_timesteps: torch.Tensor
    source_noise: torch.Tensor

    @property
    def endpoint(self) -> torch.Tensor:
        return self.anchor_states[-1:].contiguous()


class MiniMaxH3RESTSampler:
    """Euler sampler over H3's paired shifted video/audio noise schedules."""

    def __init__(self, config: H3RESTSamplingConfig) -> None:
        self.config = config

    @torch.no_grad()
    def sample(
        self,
        model: MiniMaxH3RESTTeacherModel,
        batch: TrainingBatch,
        *,
        generator: torch.Generator,
    ) -> H3RESTSamplingResult:
        if batch.latents is None:
            raise RuntimeError("prepare_batch() must provide packed latent geometry")
        source_noise = torch.randn(
            batch.latents.shape,
            device=batch.latents.device,
            dtype=batch.latents.dtype,
            generator=generator,
        )
        current = source_noise.clone()
        schedule = self.config.dense_schedule
        anchor_indices = set(self.config.anchor_indices)
        anchors: list[torch.Tensor] = [current.detach().clone()]

        original = (
            batch.timesteps,
            batch.audio_timesteps,
            batch.audio_noisy_model_input,
            batch.attn_metadata,
            batch.attn_metadata_vsa,
            batch.current_timestep,
        )
        try:
            for dense_index, (step, next_step) in enumerate(
                zip(schedule[:-1], schedule[1:], strict=True)
            ):
                timestep = torch.tensor(
                    [step], device=current.device, dtype=torch.float32
                )
                clean = model.predict_x0(
                    current,
                    timestep,
                    batch,
                    conditional=True,
                    attn_kind="dense",
                )
                if next_step == 0.0:
                    current = clean
                else:
                    next_timestep = torch.tensor(
                        [next_step], device=current.device, dtype=torch.float32
                    )
                    current = self._bridge_to_next_noise(
                        model,
                        current,
                        clean,
                        timestep,
                        next_timestep,
                    )
                point_index = dense_index + 1
                if point_index in anchor_indices:
                    anchors.append(current.detach().clone())
        finally:
            (
                batch.timesteps,
                batch.audio_timesteps,
                batch.audio_noisy_model_input,
                batch.attn_metadata,
                batch.attn_metadata_vsa,
                batch.current_timestep,
            ) = original

        if len(anchors) != len(self.config.student_timesteps):
            raise RuntimeError(
                "Dense H3 sampler failed to record every student boundary: "
                f"expected={len(self.config.student_timesteps)}, observed={len(anchors)}"
            )
        anchor_states = torch.cat(anchors, dim=0)
        return H3RESTSamplingResult(
            anchor_states=anchor_states,
            anchor_timesteps=torch.tensor(
                self.config.student_timesteps,
                device="cpu",
                dtype=torch.float32,
            ),
            source_noise=source_noise.detach(),
        )

    @staticmethod
    def _bridge_to_next_noise(
        model: MiniMaxH3RESTTeacherModel,
        current: torch.Tensor,
        clean: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
    ) -> torch.Tensor:
        current_video, current_audio = model.unpack_latents(current)
        clean_video, clean_audio = model.unpack_latents(clean)
        sigma_video, sigma_audio = model.noise_amounts(timestep)
        next_sigma_video, next_sigma_audio = model.noise_amounts(next_timestep)

        def bridge(
            sample: torch.Tensor,
            x0: torch.Tensor,
            sigma: torch.Tensor,
            sigma_next: torch.Tensor,
        ) -> torch.Tensor:
            sigma = sigma.to(device=sample.device, dtype=torch.float32)
            sigma_next = sigma_next.to(device=sample.device, dtype=torch.float32)
            if bool((sigma <= 0).any()):
                raise ValueError("Current H3 sigma must be positive before terminal zero")
            if bool((sigma_next < 0).any()) or bool((sigma_next >= sigma).any()):
                raise ValueError(
                    "H3 REST sigma schedule must strictly decrease and remain nonnegative"
                )
            ratio = sigma_next / sigma
            return (
                ratio * sample.float() + (1.0 - ratio) * x0.float()
            ).to(sample.dtype)

        return model.pack_latents(
            bridge(current_video, clean_video, sigma_video, next_sigma_video),
            bridge(current_audio, clean_audio, sigma_audio, next_sigma_audio),
        )


__all__ = [
    "H3RESTSamplingConfig",
    "H3RESTSamplingResult",
    "MiniMaxH3RESTSampler",
]
