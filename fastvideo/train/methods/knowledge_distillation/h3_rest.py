# SPDX-License-Identifier: Apache-2.0
"""REST/AMD scored full-H3 trajectory distillation for four-step FastH3."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.knowledge_distillation.h3_rest_ema import TrainableShardEMA
from fastvideo.train.methods.knowledge_distillation.h3_rest_utils import (
    amd_coefficients,
    segment_velocity_target,
    signed_loss_surrogate,
)
from fastvideo.train.models.minimax_h3.minimax_h3_rest import MiniMaxH3RESTModel
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler

logger = init_logger(__name__)


def compute_h3_rest_losses(
    prediction: torch.Tensor,
    *,
    video_target: torch.Tensor,
    audio_target: torch.Tensor,
    ema_prediction: torch.Tensor,
    video_slice: slice,
    audio_slice: slice,
    coefficient: torch.Tensor,
    audio_loss_weight: float,
    ema_regularization_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute signed video AMD, unsigned audio imitation, and EMA anchor."""
    if prediction.ndim != 2 or ema_prediction.shape != prediction.shape:
        raise ValueError(
            "prediction and ema_prediction must share packed shape [B, N], got "
            f"{tuple(prediction.shape)} and {tuple(ema_prediction.shape)}"
        )
    pred_video = prediction[:, video_slice].float()
    pred_audio = prediction[:, audio_slice].float()
    if pred_video.shape != video_target.shape or pred_audio.shape != audio_target.shape:
        raise ValueError(
            "REST target shape mismatch: "
            f"video prediction/target={tuple(pred_video.shape)}/{tuple(video_target.shape)}, "
            f"audio prediction/target={tuple(pred_audio.shape)}/{tuple(audio_target.shape)}"
        )
    if audio_loss_weight < 0.0 or ema_regularization_weight < 0.0:
        raise ValueError("REST loss weights must be nonnegative")

    video_mse = (pred_video - video_target.float()).square().mean(dim=1)
    audio_mse = (pred_audio - audio_target.float()).square().mean(dim=1)
    video_amd = signed_loss_surrogate(video_mse, coefficient)
    audio_teacher = audio_mse.mean()

    ema_video = ema_prediction[:, video_slice].float()
    ema_audio = ema_prediction[:, audio_slice].float()
    video_ema = (pred_video - ema_video).square().mean(dim=1).mean()
    audio_ema = (pred_audio - ema_audio).square().mean(dim=1).mean()
    ema_loss = video_ema + float(audio_loss_weight) * audio_ema
    total = (
        video_amd
        + float(audio_loss_weight) * audio_teacher
        + float(ema_regularization_weight) * ema_loss
    )
    return {
        "total_loss": total,
        "video_amd_loss": video_amd,
        "audio_teacher_loss": audio_teacher,
        "video_ema_loss": video_ema,
        "audio_ema_loss": audio_ema,
        "video_mse": video_mse.mean(),
        "audio_mse": audio_mse.mean(),
        "video_signed_objective": (
            coefficient.detach().to(video_mse) * video_mse.detach()
        ).mean(),
    }


class H3RESTMethod(TrainingMethod):
    """Offline reward-enhanced segment distillation from full H3 to FastH3.

    Rewards and dense teacher trajectories are frozen in an immutable cache.
    Training loads one student-aligned segment, regresses the H3 finite-
    difference velocity, applies REST's signed AMD coefficient only to video,
    and always gives audio a positive teacher-imitation target.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, Any],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if set(role_models) != {"student"}:
            raise ValueError("H3RESTMethod requires exactly one role: models.student")
        if not isinstance(self.student, MiniMaxH3RESTModel):
            raise TypeError(
                "H3RESTMethod requires models.student._target_=MiniMaxH3RESTModel"
            )
        if not self.student._trainable:
            raise ValueError("H3RESTMethod requires a trainable student")

        self.student.init_preprocessors(self.training_config)
        self._attention_kind = self._read_attention_kind()
        self._amd_scale = self._read_nonnegative_float("amd_scale", 1.0)
        self._amd_bias = self._read_float("amd_bias", 0.5)
        coefficient_clip = self.method_config.get("amd_coefficient_clip")
        self._amd_coefficient_clip = (
            None if coefficient_clip in (None, "") else float(coefficient_clip)
        )
        if self._amd_coefficient_clip is not None:
            if (
                not math.isfinite(self._amd_coefficient_clip)
                or self._amd_coefficient_clip <= 0.0
            ):
                raise ValueError(
                    "method.amd_coefficient_clip must be finite and positive when set"
                )
        self._audio_loss_weight = self._read_nonnegative_float(
            "audio_loss_weight", 1.0
        )
        self._ema_regularization_weight = self._read_nonnegative_float(
            "ema_regularization_weight", 0.2
        )
        self._ema_decay = self._read_float("ema_decay", 0.9)
        if not 0.0 <= self._ema_decay < 1.0:
            raise ValueError("method.ema_decay must satisfy 0 <= decay < 1")
        self._ema: TrainableShardEMA | None = None
        self._init_optimizer_and_scheduler()

    @property
    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        return {"student": self._optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._lr_scheduler}

    @property
    def ema(self) -> TrainableShardEMA:
        if self._ema is None:
            raise RuntimeError("REST EMA is not initialized; call on_train_start() first")
        return self._ema

    def on_train_start(self) -> None:
        super().on_train_start()
        self._ema = TrainableShardEMA(
            self.student.transformer,
            decay=self._ema_decay,
        )
        logger.info(
            "H3 REST initialized: cache=%s, grid=%s, AMD=lambda*(A+b) with "
            "lambda=%s b=%s, EMA=%s beta=%s",
            self.student.rest_cache_fingerprint,
            list(self.student.rest_student_timesteps),
            self._amd_scale,
            self._amd_bias,
            self._ema_decay,
            self._ema_regularization_weight,
        )

    def checkpoint_state(self) -> dict[str, Any]:
        state = super().checkpoint_state()
        if self._ema is not None:
            state["rest_ema"] = self._ema
        return state

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del iteration
        generator = self._require_generator()
        prepared = self.student.prepare_batch(
            batch,
            generator=generator,
            latents_source="zeros",
        )
        if prepared.latents is None:
            raise RuntimeError("H3 REST batch preparation returned no packed geometry")
        dtype = prepared.latents.dtype
        current = self._batch_tensor(batch, "trajectory_current", dtype=dtype)
        next_state = self._batch_tensor(batch, "trajectory_next", dtype=dtype)
        timestep = self._batch_tensor(
            batch, "trajectory_timestep", dtype=torch.float32
        )
        next_timestep = self._batch_tensor(
            batch, "trajectory_next_timestep", dtype=torch.float32
        )
        mixed_advantage = self._batch_tensor(
            batch, "rest_mixed_advantage", dtype=torch.float32
        ).reshape(-1)
        segment_index_tensor = self._batch_tensor(
            batch, "rest_segment_index", dtype=torch.long
        ).reshape(-1)
        if segment_index_tensor.numel() != 1:
            raise ValueError("H3 REST requires exactly one segment per rank")
        segment_index = int(segment_index_tensor.item())
        self._validate_segment_contract(
            current,
            next_state,
            timestep,
            next_timestep,
            segment_index,
        )

        self.student.refresh_vsa_metadata(
            prepared,
            current_timestep=segment_index,
        )
        video_slice, audio_slice = self._modality_slices()
        sigma_video, sigma_audio = self.student.noise_amounts(timestep)
        next_sigma_video, next_sigma_audio = self.student.noise_amounts(next_timestep)
        video_target = segment_velocity_target(
            current[:, video_slice],
            next_state[:, video_slice],
            sigma_video,
            next_sigma_video,
        )
        audio_target = segment_velocity_target(
            current[:, audio_slice],
            next_state[:, audio_slice],
            sigma_audio,
            next_sigma_audio,
        )
        coefficient = amd_coefficients(
            mixed_advantage,
            scale=self._amd_scale,
            bias=self._amd_bias,
            clip=self._amd_coefficient_clip,
        ).to(self.student.device)

        # The EMA pass comes first. Swapping parameters after constructing the
        # online autograd graph would invalidate saved-tensor version counters.
        with torch.no_grad(), self.ema.apply_to_model(self.student.transformer):
            ema_prediction = self.student.predict_noise(
                current,
                timestep,
                prepared,
                conditional=True,
                attn_kind=self._attention_kind,
            ).detach()

        prediction = self.student.predict_noise(
            current,
            timestep,
            prepared,
            conditional=True,
            attn_kind=self._attention_kind,
        )
        backward_metadata = (
            prepared.attn_metadata_vsa
            if self._attention_kind == "vsa"
            else prepared.attn_metadata
        )
        backward_context = (prepared.timesteps, backward_metadata)
        losses = compute_h3_rest_losses(
            prediction,
            video_target=video_target,
            audio_target=audio_target,
            ema_prediction=ema_prediction,
            video_slice=video_slice,
            audio_slice=audio_slice,
            coefficient=coefficient,
            audio_loss_weight=self._audio_loss_weight,
            ema_regularization_weight=self._ema_regularization_weight,
        )
        metrics: dict[str, LogScalar] = {
            "rest/mixed_advantage": mixed_advantage.mean(),
            "rest/amd_coefficient": coefficient.mean(),
            "rest/negative_coefficient": (coefficient < 0).float().mean(),
            "rest/segment_index": float(segment_index),
            "rest/ema_updates": float(self.ema.num_updates),
            "rest/cache_examples": float(self._cache_num_examples()),
        }
        reward_scores = batch.get("rest_reward_scores")
        if isinstance(reward_scores, Mapping):
            for name, value in reward_scores.items():
                metrics[f"rest/reward/{name}"] = float(value)
        reward_advantages = batch.get("rest_reward_advantages")
        if isinstance(reward_advantages, Mapping):
            for name, value in reward_advantages.items():
                metrics[f"rest/reward_advantage/{name}"] = float(value)
        return losses, {"student_ctx": backward_context}, metrics

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        student_ctx = outputs.get("student_ctx")
        if student_ctx is None:
            raise RuntimeError("H3 REST backward is missing the student forward context")
        self.student.backward(
            loss_map["total_loss"],
            student_ctx,
            grad_accum_rounds=max(1, int(grad_accum_rounds)),
        )

    def get_optimizers(self, iteration: int) -> Sequence[torch.optim.Optimizer]:
        del iteration
        return (self._optimizer,)

    def get_lr_schedulers(self, iteration: int) -> Sequence[Any]:
        del iteration
        return (self._lr_scheduler,)

    def get_grad_clip_targets(self, iteration: int) -> dict[str, torch.nn.Module]:
        del iteration
        return {"student": self.student.transformer}

    def optimizers_schedulers_step(self, iteration: int) -> None:
        super().optimizers_schedulers_step(iteration)
        self.ema.update(self.student.transformer)

    def apply_configured_lrs(self) -> None:
        learning_rate = float(self.training_config.optimizer.learning_rate)
        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate
            if "initial_lr" in group:
                group["initial_lr"] = learning_rate
        if hasattr(self._lr_scheduler, "base_lrs"):
            self._lr_scheduler.base_lrs = [
                learning_rate for _ in self._lr_scheduler.base_lrs
            ]

    def _init_optimizer_and_scheduler(self) -> None:
        parameters = [
            parameter
            for parameter in self.student.transformer.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("H3 REST student has no trainable parameters")
        self._optimizer, self._lr_scheduler = build_optimizer_and_scheduler(
            params=parameters,
            optimizer_config=self.training_config.optimizer,
            loop_config=self.training_config.loop,
            learning_rate=float(self.training_config.optimizer.learning_rate),
            betas=tuple(self.training_config.optimizer.betas),
            scheduler_name=str(self.training_config.optimizer.lr_scheduler),
        )

    def _cache_num_examples(self) -> int:
        dataset = self.student.rest_cache_dataset
        if dataset is None:
            raise RuntimeError("H3 REST cache dataset is not initialized")
        return int(dataset.summary.num_examples)

    def _validate_segment_contract(
        self,
        current: torch.Tensor,
        next_state: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        segment_index: int,
    ) -> None:
        if current.shape != next_state.shape or current.ndim != 2:
            raise ValueError(
                "Cached REST states must share packed shape [B, N], got "
                f"{tuple(current.shape)} and {tuple(next_state.shape)}"
            )
        if current.shape[0] != 1:
            raise ValueError("H3 REST requires batch size one per SP group")
        grid = self.student.rest_student_timesteps
        if not 0 <= segment_index < len(grid) - 1:
            raise ValueError(
                f"REST segment_index out of range: {segment_index} for grid {grid}"
            )
        observed = (float(timestep.item()), float(next_timestep.item()))
        expected = (float(grid[segment_index]), float(grid[segment_index + 1]))
        if observed != expected:
            raise ValueError(
                "Cached REST segment timestep mismatch: "
                f"segment={segment_index}, observed={observed}, expected={expected}"
            )
        self.student.unpack_latents(current)
        self.student.unpack_latents(next_state)

    def _modality_slices(self) -> tuple[slice, slice]:
        slices = dict(self.student.modality_slices())
        if "video" not in slices or "audio" not in slices:
            raise RuntimeError(
                f"H3 REST requires video/audio slices, got {sorted(slices)}"
            )
        return slices["video"], slices["audio"]

    def _batch_tensor(
        self,
        batch: Mapping[str, Any],
        key: str,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = batch.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"H3 REST cache batch is missing tensor {key!r}")
        tensor = value.to(device=self.student.device, dtype=dtype, non_blocking=True)
        if not bool(torch.isfinite(tensor.float()).all()):
            raise ValueError(f"H3 REST cache tensor {key!r} contains NaN or Inf")
        return tensor

    def _require_generator(self) -> torch.Generator:
        if self.cuda_generator is None:
            raise RuntimeError("H3 REST CUDA generator is not initialized")
        return self.cuda_generator

    def _read_attention_kind(self) -> Literal["dense", "vsa"]:
        value = str(self.method_config.get("attn_kind", "vsa")).strip().lower()
        if value not in {"dense", "vsa"}:
            raise ValueError("method.attn_kind must be one of {dense, vsa}")
        return value  # type: ignore[return-value]

    def _read_float(self, key: str, default: float) -> float:
        raw = self.method_config.get(key, default)
        value = float(default if raw is None else raw)
        if not math.isfinite(value):
            raise ValueError(f"method.{key} must be finite")
        return value

    def _read_nonnegative_float(self, key: str, default: float) -> float:
        value = self._read_float(key, default)
        if value < 0.0:
            raise ValueError(f"method.{key} must be nonnegative")
        return value


__all__ = ["H3RESTMethod", "compute_h3_rest_losses"]
