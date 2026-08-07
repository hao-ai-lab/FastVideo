# SPDX-License-Identifier: Apache-2.0
"""Supervised finetuning method (algorithm layer)."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F

from fastvideo.train.methods.base import TrainingMethod, LogScalar
from fastvideo.train.models.base import ModelBase, NoisePrediction
from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler, )


def _compute_finetune_loss_map(
    prediction: NoisePrediction,
    clean_video_latents: torch.Tensor,
    noisy_video_latents: torch.Tensor,
    video_noise: torch.Tensor,
    video_sigmas: torch.Tensor,
    training_batch: Any,
    *,
    precondition_outputs: bool,
) -> dict[str, torch.Tensor]:
    """Compute supervised flow-matching losses for one or two modalities.

    A tensor prediction follows the video-only model contract. An ordered
    ``(video, audio)`` prediction uses each modality's independently shifted
    noise level and sums both mean-squared errors so one backward pass trains
    both output branches.
    """
    if isinstance(prediction, torch.Tensor):
        if precondition_outputs:
            predicted_clean_video = noisy_video_latents - prediction * video_sigmas
            loss = F.mse_loss(predicted_clean_video.float(), clean_video_latents.float())
        else:
            video_target = video_noise - clean_video_latents
            loss = F.mse_loss(prediction.float(), video_target.float())
        return {
            "total_loss": loss,
            "finetune_loss": loss,
        }

    if len(prediction) != 2:
        raise ValueError("A multimodal prediction must contain video and audio tensors")
    video_prediction, audio_prediction = prediction
    required_audio_tensors = {
        "audio_latents": training_batch.audio_latents,
        "audio_noisy_model_input": training_batch.audio_noisy_model_input,
        "audio_noise": training_batch.audio_noise,
        "audio_sigmas": training_batch.audio_sigmas,
    }
    missing = [name for name, value in required_audio_tensors.items() if value is None]
    if missing:
        raise RuntimeError("prepare_batch() must set " + ", ".join(f"TrainingBatch.{name}" for name in missing))

    clean_audio_latents = required_audio_tensors["audio_latents"]
    noisy_audio_latents = required_audio_tensors["audio_noisy_model_input"]
    audio_noise = required_audio_tensors["audio_noise"]
    audio_sigmas = required_audio_tensors["audio_sigmas"]
    assert isinstance(clean_audio_latents, torch.Tensor)
    assert isinstance(noisy_audio_latents, torch.Tensor)
    assert isinstance(audio_noise, torch.Tensor)
    assert isinstance(audio_sigmas, torch.Tensor)

    if precondition_outputs:
        predicted_clean_video = noisy_video_latents - video_prediction * video_sigmas
        predicted_clean_audio = noisy_audio_latents - audio_prediction * audio_sigmas
        video_loss = F.mse_loss(predicted_clean_video.float(), clean_video_latents.float())
        audio_loss = F.mse_loss(predicted_clean_audio.float(), clean_audio_latents.float())
    else:
        video_target = video_noise - clean_video_latents
        audio_target = audio_noise - clean_audio_latents
        video_loss = F.mse_loss(video_prediction.float(), video_target.float())
        audio_loss = F.mse_loss(audio_prediction.float(), audio_target.float())
    total_loss = video_loss + audio_loss
    return {
        "total_loss": total_loss,
        "finetune_loss": total_loss,
        "video_finetune_loss": video_loss,
        "audio_finetune_loss": audio_loss,
    }


class FineTuneMethod(TrainingMethod):
    """Supervised finetuning: only ``student`` participates."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        if "student" not in role_models:
            raise ValueError("FineTuneMethod requires role 'student'")
        if not self.student._trainable:
            raise ValueError("FineTuneMethod requires student to be "
                             "trainable")
        self._attn_kind: Literal["dense", "vsa"] = (self._infer_attn_kind())

        # Initialize preprocessors on student.
        self.student.init_preprocessors(self.training_config)

        self._init_optimizers_and_schedulers()

    @property
    def _optimizer_dict(self) -> dict[str, Any]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    # TrainingMethod override: single_train_step
    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        """Prepare synchronized targets and compute supervised flow loss.

        The returned forward context lets model-specific backward methods
        restore activation-checkpoint metadata during recomputation.
        """
        del iteration
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="data",
        )

        if training_batch.latents is None:
            raise RuntimeError("prepare_batch() must set "
                               "TrainingBatch.latents")
        if training_batch.noisy_model_input is None:
            raise RuntimeError("prepare_batch() must set "
                               "TrainingBatch.noisy_model_input")
        if training_batch.noise is None:
            raise RuntimeError("prepare_batch() must set "
                               "TrainingBatch.noise")
        if training_batch.sigmas is None:
            raise RuntimeError("prepare_batch() must set "
                               "TrainingBatch.sigmas")
        if training_batch.timesteps is None:
            raise RuntimeError("prepare_batch() must set "
                               "TrainingBatch.timesteps")

        clean_latents = training_batch.latents
        noisy_latents = (training_batch.noisy_model_input.permute(0, 2, 1, 3, 4))
        noise = training_batch.noise.permute(0, 2, 1, 3, 4)
        sigmas = training_batch.sigmas
        timesteps = training_batch.timesteps

        pred = self.student.predict_noise(
            noisy_latents,
            timesteps,
            training_batch,
            conditional=True,
            attn_kind=self._attn_kind,
        )

        loss_map = _compute_finetune_loss_map(
            pred,
            clean_latents,
            noisy_latents,
            noise,
            sigmas,
            training_batch,
            precondition_outputs=bool(self.training_config.model.precondition_outputs),
        )

        attn_metadata = training_batch.attn_metadata_vsa if self._attn_kind == "vsa" else training_batch.attn_metadata

        outputs: dict[str, Any] = {
            "_fv_backward": (
                training_batch.timesteps,
                attn_metadata,
            )
        }
        metrics: dict[str, LogScalar] = {}
        return loss_map, outputs, metrics

    # TrainingMethod override: backward
    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        """Backpropagate an accumulation-scaled loss through the student model.

        Delegating to ``ModelBase.backward`` lets each model restore its forward
        context before the distributed wrapper synchronizes parameter gradients.
        """
        grad_accum_rounds = max(1, int(grad_accum_rounds))
        ctx = outputs.get("_fv_backward")
        if ctx is None:
            super().backward(
                loss_map,
                outputs,
                grad_accum_rounds=grad_accum_rounds,
            )
            return
        self.student.backward(
            loss_map["total_loss"],
            ctx,
            grad_accum_rounds=grad_accum_rounds,
        )

    # TrainingMethod override: get_optimizers
    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    # TrainingMethod override: get_lr_schedulers
    def get_lr_schedulers(
        self,
        iteration: int,
    ) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config

        student_lr = float(tc.optimizer.learning_rate)
        if student_lr <= 0.0:
            raise ValueError("training.learning_rate must be > 0 "
                             "for finetune")

        student_betas = tc.optimizer.betas
        student_sched = str(tc.optimizer.lr_scheduler)
        student_params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        (
            self._student_optimizer,
            self._student_lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=student_params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=student_lr,
            betas=student_betas,
            scheduler_name=student_sched,
        )
