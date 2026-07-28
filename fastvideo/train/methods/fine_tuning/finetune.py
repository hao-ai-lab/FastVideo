# SPDX-License-Identifier: Apache-2.0
"""Supervised finetuning method (algorithm layer)."""

from __future__ import annotations

import os
from typing import Any, Literal

import torch
import torch.nn.functional as F

from fastvideo.logger import init_logger
from fastvideo.train.methods.base import TrainingMethod, LogScalar
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler, )

logger = init_logger(__name__)


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

        if bool(self.training_config.model.precondition_outputs):
            pred_x0 = noisy_latents - pred * sigmas
            loss = F.mse_loss(pred_x0.float(), clean_latents.float())
        else:
            target = noise - clean_latents
            loss = F.mse_loss(pred.float(), target.float())

        attn_metadata = training_batch.attn_metadata_vsa if self._attn_kind == "vsa" else training_batch.attn_metadata

        loss_map = {
            "total_loss": loss,
            "finetune_loss": loss,
        }
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

    # Names that make up the track pathway. patch_embedding is matched on the CHECKPOINT name
    # used by the live module ('patch_embedding.proj.weight'); an earlier experiment silently
    # matched nothing by using 'patch_embedding.weight' (the converter-side name) and so never
    # actually applied its layer-wise LR.
    _TRACK_ENCODER_HINT = "track_encoder"
    _PATCH_EMBED_HINTS = ("patch_embedding.proj.weight", "patch_embedding.weight")
    _BASE_IN_CHANNELS = 36

    def _build_track_param_groups(self, base_lr: float) -> list[dict]:
        """Split params into a boosted 'track' group and the normal 'base' group."""
        mult = float(os.getenv("WANTRACK_TRACK_LR_MULT", "1"))
        track, base = [], []
        pe_masked = 0
        for name, p in self.student.transformer.named_parameters():
            if not p.requires_grad:
                continue
            is_pe = any(name.endswith(h) for h in self._PATCH_EMBED_HINTS)
            if self._TRACK_ENCODER_HINT in name or is_pe:
                if is_pe and p.dim() >= 2 and p.shape[1] > self._BASE_IN_CHANNELS:
                    # Keep the pretrained input channels out of the boosted group entirely.
                    base_in = self._BASE_IN_CHANNELS

                    def _mask_pretrained(grad, _base_in=base_in):
                        g = grad.clone()
                        g[:, :_base_in] = 0
                        return g

                    p.register_hook(_mask_pretrained)
                    pe_masked += 1
                track.append(p)
            else:
                base.append(p)

        if not track:
            raise ValueError(
                "WANTRACK_TRACK_GROUP=1 but no track params matched "
                f"({self._TRACK_ENCODER_HINT!r} / {self._PATCH_EMBED_HINTS!r}) — refusing to "
                "run an experiment that would silently be a plain finetune")

        n_track = sum(p.numel() for p in track)
        logger.info(
            "[WANTRACK] track param group: %d tensors (%d params) lr=%.3g (mult %.3g); "
            "base group: %d tensors lr=%.3g; patch_embedding grad-masked to [:, %d:] on %d tensor(s)",
            len(track), n_track, base_lr * mult, mult, len(base), base_lr,
            self._BASE_IN_CHANNELS, pe_masked)
        # 'name' is what TrackWarmupCallback looks for to zero the base group during warmup.
        return [
            {"params": track, "lr": base_lr * mult, "name": "track"},
            {"params": base, "lr": base_lr, "name": "base"},
        ]

    def _init_optimizers_and_schedulers(self) -> None:
        tc = self.training_config

        student_lr = float(tc.optimizer.learning_rate)
        if student_lr <= 0.0:
            raise ValueError("training.learning_rate must be > 0 "
                             "for finetune")

        student_betas = tc.optimizer.betas
        student_sched = str(tc.optimizer.lr_scheduler)
        student_params = [p for p in self.student.transformer.parameters() if p.requires_grad]

        # Optional: split the track pathway into its OWN param group so it can be given a
        # different LR from the pretrained DiT (WANTRACK_TRACK_GROUP=1). Used to test whether a
        # random-init encoder can be bootstrapped directly on stage-1 data — warming up the
        # track pathway first, or running it at a higher LR — instead of via the overfit+merge.
        #
        # Two things make this non-trivial and are handled here:
        #  * Adam is scale-invariant per parameter, so scaling GRADIENTS does not emulate a
        #    higher LR. It has to be a real param group with its own lr.
        #  * patch_embedding.proj.weight is ONE tensor covering both the pretrained input
        #    channels [:, :36] and the track slot [:, 36:]. Putting it in a boosted group would
        #    blast the pretrained channels too, so we mask its gradient to the track slot.
        if os.getenv("WANTRACK_TRACK_GROUP", "0") not in ("0", "false", "False"):
            student_params = self._build_track_param_groups(student_lr)
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
