# SPDX-License-Identifier: Apache-2.0
"""Shape-agnostic conditional flow-matching fine-tuning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.logger import init_logger
from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.distributed_strategy import is_ddp_strategy

logger = init_logger(__name__)

TrainStepResult = tuple[
    dict[str, torch.Tensor],
    dict[str, Any],
    dict[str, LogScalar],
]

TensorTrainFn = Callable[..., torch.Tensor]


class FlowMatchingFineTuneMethod(FineTuneMethod):
    """Fine-tune a model against a velocity target supplied by its adapter.

    Unlike :class:`FineTuneMethod`, this method does not assume a five-
    dimensional video latent layout. The model plugin owns sampling the flow
    path and stores its target in ``TrainingBatch.training_target``.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        flow_matching_forward = getattr(self.student, "flow_matching_forward", None)
        if not callable(flow_matching_forward):
            raise TypeError("FlowMatchingFineTuneMethod requires the student model to "
                            "implement flow_matching_forward()")
        self._flow_matching_forward: TensorTrainFn = flow_matching_forward
        self._train_fn: TensorTrainFn = self._eager_train_fn

        if bool(self.training_config.model.compile_train_fn):
            if not is_ddp_strategy(self.training_config):
                raise ValueError("training.model.compile_train_fn is currently supported "
                                 "only with training.distributed.strategy=ddp")
            compile_kwargs = dict(self.training_config.model.torch_compile_kwargs)
            compile_kwargs.setdefault("fullgraph", True)
            compile_training_forward = getattr(self.student, "compile_training_forward", None)
            if not callable(compile_training_forward):
                raise TypeError("compile_train_fn requires the student model to implement "
                                "compile_training_forward()")
            logger.info(
                "Enabling fullgraph transformer compilation with kwargs=%s",
                compile_kwargs,
            )
            compile_training_forward(compile_kwargs)

    def _eager_train_fn(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        target: torch.Tensor,
        clip_features: torch.Tensor,
        sync_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self._flow_matching_forward(
            noisy_latents,
            timesteps,
            clip_features,
            sync_features,
            text_features,
        )
        return F.mse_loss(prediction.float(), target.float())

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> TrainStepResult:
        del iteration
        if self.cuda_generator is None:
            raise RuntimeError("on_train_start() must run before training")

        # RNG, existence masks, and CFG masks stay eager so the explicit
        # per-rank CUDA Generator advances exactly as in official MMAudio.
        training_batch = self.student.prepare_batch(
            batch,
            generator=self.cuda_generator,
            latents_source="data",
        )
        noisy_latents = training_batch.noisy_model_input
        timesteps = training_batch.timesteps
        target = training_batch.training_target
        if not all(isinstance(value, torch.Tensor) for value in (noisy_latents, timesteps, target)):
            raise RuntimeError("prepare_batch() returned incomplete flow inputs")
        assert isinstance(noisy_latents, torch.Tensor)
        assert isinstance(timesteps, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        if target.shape != noisy_latents.shape:
            raise RuntimeError("Flow-matching target and model input shapes must match, got "
                               f"{tuple(target.shape)} and {tuple(noisy_latents.shape)}")

        conditions = training_batch.conditional_dict
        if not isinstance(conditions, dict):
            raise RuntimeError("prepare_batch() must set conditional_dict")
        condition_values = tuple(conditions.get(key) for key in ("clip_features", "sync_features", "text_features"))
        if not all(isinstance(value, torch.Tensor) for value in condition_values):
            raise RuntimeError("MMAudio flow conditions must all be tensors")
        clip_features, sync_features, text_features = condition_values
        assert isinstance(clip_features, torch.Tensor)
        assert isinstance(sync_features, torch.Tensor)
        assert isinstance(text_features, torch.Tensor)

        loss = self._train_fn(
            noisy_latents,
            timesteps,
            target,
            clip_features,
            sync_features,
            text_features,
        )
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
