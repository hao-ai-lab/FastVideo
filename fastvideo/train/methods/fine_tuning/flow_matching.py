# SPDX-License-Identifier: Apache-2.0
"""Shape-agnostic conditional flow-matching fine-tuning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod


class FlowMatchingFineTuneMethod(FineTuneMethod):
    """Fine-tune a model against a velocity target supplied by its adapter.

    Unlike :class:`FineTuneMethod`, this method does not assume a five-
    dimensional video latent layout. The model plugin owns sampling the flow
    path and stores its target in ``TrainingBatch.training_target``.
    """

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, Any],
            dict[str, LogScalar],
    ]:
        del iteration
        if self.cuda_generator is None:
            raise RuntimeError("on_train_start() must run before training")

        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="data",
        )
        noisy_latents = training_batch.noisy_model_input
        timesteps = training_batch.timesteps
        if noisy_latents is None:
            raise RuntimeError("prepare_batch() must set noisy_model_input")
        if timesteps is None:
            raise RuntimeError("prepare_batch() must set timesteps")

        target = training_batch.training_target
        if not isinstance(target, torch.Tensor):
            raise RuntimeError("prepare_batch() must set training_target")
        if target.shape != noisy_latents.shape:
            raise RuntimeError("Flow-matching target and model input shapes must match, got "
                               f"{tuple(target.shape)} and {tuple(noisy_latents.shape)}")

        pred = self.student.predict_noise(
            noisy_latents,
            timesteps,
            training_batch,
            conditional=True,
            attn_kind=self._attn_kind,
        )
        if pred.shape != target.shape:
            raise RuntimeError("Flow-matching prediction and target shapes must match, got "
                               f"{tuple(pred.shape)} and {tuple(target.shape)}")
        loss = F.mse_loss(pred.float(), target.float())

        attn_metadata = (training_batch.attn_metadata_vsa if self._attn_kind == "vsa" else training_batch.attn_metadata)
        loss_map = {
            "total_loss": loss,
            "flow_matching_loss": loss,
        }
        outputs: dict[str, Any] = {
            "_fv_backward": (timesteps, attn_metadata),
        }
        metrics: dict[str, LogScalar] = {}
        return loss_map, outputs, metrics


__all__ = ["FlowMatchingFineTuneMethod"]
