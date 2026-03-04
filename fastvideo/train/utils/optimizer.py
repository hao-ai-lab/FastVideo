# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
)

if TYPE_CHECKING:
    from fastvideo.distillation.utils.distill_config import (
        OptimizerConfig,
        TrainingLoopConfig,
    )


def build_optimizer_and_scheduler(
    *,
    params: list[torch.nn.Parameter],
    optimizer_config: OptimizerConfig,
    loop_config: TrainingLoopConfig,
    learning_rate: float,
    betas: tuple[float, float],
    scheduler_name: str,
) -> tuple[torch.optim.Optimizer, object]:
    """Build an AdamW optimizer and LR scheduler.

    Returns ``(optimizer, lr_scheduler)`` so the caller can store them
    as method-level attributes.
    """
    if not params:
        raise ValueError("No trainable parameters passed to "
                         "build_optimizer_and_scheduler")

    optimizer = torch.optim.AdamW(
        params,
        lr=float(learning_rate),
        betas=betas,
        weight_decay=float(optimizer_config.weight_decay),
        eps=1e-8,
    )

    scheduler = get_scheduler(
        str(scheduler_name),
        optimizer=optimizer,
        num_warmup_steps=int(optimizer_config.lr_warmup_steps),
        num_training_steps=int(loop_config.max_train_steps),
        num_cycles=int(optimizer_config.lr_num_cycles),
        power=float(optimizer_config.lr_power),
        min_lr_ratio=float(optimizer_config.min_lr_ratio),
        last_epoch=-1,
    )

    return optimizer, scheduler


def clip_grad_norm_if_needed(
    module: torch.nn.Module,
    max_grad_norm: float,
) -> float:
    if max_grad_norm <= 0.0:
        return 0.0
    grad_norm = (clip_grad_norm_while_handling_failing_dtensor_cases(
        [p for p in module.parameters()],
        max_grad_norm,
        foreach=None,
    ))
    return (float(grad_norm.item()) if grad_norm is not None else 0.0)
