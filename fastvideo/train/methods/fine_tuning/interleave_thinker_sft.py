# SPDX-License-Identifier: Apache-2.0
"""Supervised fine-tuning method for InterleaveThinker Qwen actors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler


class InterleaveThinkerSFTMethod(TrainingMethod):
    """SFT for planner/critic Qwen3-VL actors with response-token masking."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if "student" not in role_models:
            raise ValueError("InterleaveThinkerSFTMethod requires role 'student'")
        if not self.student._trainable:
            raise ValueError("InterleaveThinkerSFTMethod requires student to be trainable")
        if not callable(getattr(self.student, "compute_interleave_sft_loss", None)):
            raise TypeError("InterleaveThinkerSFTMethod requires student.compute_interleave_sft_loss()")

        self.student.init_preprocessors(self.training_config)
        self._init_optimizer_and_scheduler()

    @property
    def _optimizer_dict(self) -> dict[str, Any]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

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
        compute_loss = self.student.compute_interleave_sft_loss
        result = compute_loss(batch)
        loss_map, metrics = _coerce_sft_result(result)
        return loss_map, {}, metrics

    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        del iteration
        return [self._student_optimizer]

    def get_lr_schedulers(
        self,
        iteration: int,
    ) -> list[Any]:
        del iteration
        return [self._student_lr_scheduler]

    def _init_optimizer_and_scheduler(self) -> None:
        tc = self.training_config
        lr = float(tc.optimizer.learning_rate)
        if lr <= 0.0:
            raise ValueError("training.optimizer.learning_rate must be > 0 for InterleaveThinker SFT")
        params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        if not params:
            raise ValueError("InterleaveThinkerSFTMethod found no trainable student parameters")
        self._student_optimizer, self._student_lr_scheduler = build_optimizer_and_scheduler(
            params=params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=lr,
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )


def _coerce_sft_result(result: Any) -> tuple[dict[str, torch.Tensor], dict[str, LogScalar]]:
    if isinstance(result, tuple) and len(result) == 2:
        loss_map_raw, metrics_raw = result
    elif isinstance(result, Mapping):
        loss_map_raw = result.get("loss_map")
        metrics_raw = result.get("metrics", {})
        if loss_map_raw is None and "loss" in result:
            loss_map_raw = {"total_loss": result["loss"]}
    else:
        raise TypeError("compute_interleave_sft_loss() must return (loss_map, metrics) or a mapping")

    if not isinstance(loss_map_raw, Mapping):
        raise TypeError("compute_interleave_sft_loss() result must include a loss_map mapping")
    loss_map = {str(key): _coerce_tensor(value) for key, value in loss_map_raw.items()}
    if "total_loss" not in loss_map:
        if len(loss_map) != 1:
            raise ValueError("SFT loss_map must include total_loss when multiple losses are returned")
        loss_map["total_loss"] = next(iter(loss_map.values()))

    if metrics_raw is None:
        metrics_raw = {}
    if not isinstance(metrics_raw, Mapping):
        raise TypeError("compute_interleave_sft_loss() metrics must be a mapping")
    return loss_map, {str(key): value for key, value in metrics_raw.items()}


def _coerce_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(float(value), dtype=torch.float32)


__all__ = ["InterleaveThinkerSFTMethod"]
