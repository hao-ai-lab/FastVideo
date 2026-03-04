# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
)

if TYPE_CHECKING:
    from fastvideo.fastvideo_args import TrainingArgs


def build_optimizer_and_scheduler(
    *,
    params: list[torch.nn.Parameter],
    training_args: TrainingArgs,
    learning_rate: float,
    betas: tuple[float, float],
    scheduler_name: str,
) -> tuple[torch.optim.Optimizer, object]:
    """Build an AdamW optimizer and LR scheduler.

    Returns ``(optimizer, lr_scheduler)`` so the caller can store them
    as method-level attributes.
    """
    if not params:
        raise ValueError(
            "No trainable parameters passed to "
            "build_optimizer_and_scheduler"
        )

    optimizer = torch.optim.AdamW(
        params,
        lr=float(learning_rate),
        betas=betas,
        weight_decay=float(
            getattr(training_args, "weight_decay", 0.0) or 0.0
        ),
        eps=1e-8,
    )

    scheduler = get_scheduler(
        str(scheduler_name),
        optimizer=optimizer,
        num_warmup_steps=int(
            getattr(training_args, "lr_warmup_steps", 0) or 0
        ),
        num_training_steps=int(
            getattr(training_args, "max_train_steps", 0) or 0
        ),
        num_cycles=int(
            getattr(training_args, "lr_num_cycles", 0) or 0
        ),
        power=float(
            getattr(training_args, "lr_power", 0.0) or 0.0
        ),
        min_lr_ratio=float(
            getattr(training_args, "min_lr_ratio", 0.5) or 0.5
        ),
        last_epoch=-1,
    )

    return optimizer, scheduler


def clip_grad_norm_if_needed(
    module: torch.nn.Module, training_args: TrainingArgs
) -> float:
    max_grad_norm_raw = getattr(
        training_args, "max_grad_norm", None
    )
    if max_grad_norm_raw is None:
        return 0.0
    try:
        max_grad_norm = float(max_grad_norm_raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "training.max_grad_norm must be a number when set, "
            f"got {max_grad_norm_raw!r}"
        ) from e
    if max_grad_norm <= 0.0:
        return 0.0
    grad_norm = (
        clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for p in module.parameters()],
            max_grad_norm,
            foreach=None,
        )
    )
    return (
        float(grad_norm.item()) if grad_norm is not None else 0.0
    )
