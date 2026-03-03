# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fastvideo.distillation.roles import RoleHandle

from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
)

if TYPE_CHECKING:
    from fastvideo.fastvideo_args import TrainingArgs


def build_role_optimizer_and_scheduler(
    *,
    role: str,
    handle: RoleHandle,
    training_args: TrainingArgs,
    learning_rate: float,
    betas: tuple[float, float],
    scheduler_name: str,
) -> None:
    modules = handle.modules
    params: list[torch.nn.Parameter] = []
    for module in modules.values():
        params.extend([p for p in module.parameters() if p.requires_grad])
    if not params:
        raise ValueError(f"Role {role!r} is trainable but has no trainable parameters")

    optimizer = torch.optim.AdamW(
        params,
        lr=float(learning_rate),
        betas=betas,
        weight_decay=float(getattr(training_args, "weight_decay", 0.0) or 0.0),
        eps=1e-8,
    )

    scheduler = get_scheduler(
        str(scheduler_name),
        optimizer=optimizer,
        num_warmup_steps=int(getattr(training_args, "lr_warmup_steps", 0) or 0),
        num_training_steps=int(getattr(training_args, "max_train_steps", 0) or 0),
        num_cycles=int(getattr(training_args, "lr_num_cycles", 0) or 0),
        power=float(getattr(training_args, "lr_power", 0.0) or 0.0),
        min_lr_ratio=float(getattr(training_args, "min_lr_ratio", 0.5) or 0.5),
        last_epoch=-1,
    )

    handle.optimizers = {"main": optimizer}
    handle.lr_schedulers = {"main": scheduler}


def clip_grad_norm_if_needed(module: torch.nn.Module, training_args: TrainingArgs) -> float:
    max_grad_norm_raw = getattr(training_args, "max_grad_norm", None)
    if max_grad_norm_raw is None:
        return 0.0
    try:
        max_grad_norm = float(max_grad_norm_raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "training.max_grad_norm must be a number when set, got "
            f"{max_grad_norm_raw!r}"
        ) from e
    if max_grad_norm <= 0.0:
        return 0.0
    grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
        [p for p in module.parameters()],
        max_grad_norm,
        foreach=None,
    )
    return float(grad_norm.item()) if grad_norm is not None else 0.0

