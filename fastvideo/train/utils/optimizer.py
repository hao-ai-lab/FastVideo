# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor

from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
)

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
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
    weight_decay: float | None = None,
) -> tuple[torch.optim.Optimizer, object]:
    """Build an AdamW optimizer and LR scheduler.

    Returns ``(optimizer, lr_scheduler)`` so the caller can store them
    as method-level attributes.
    """
    if not params:
        raise ValueError("No trainable parameters passed to "
                         "build_optimizer_and_scheduler")

    optimizer_kwargs = {}
    if bool(getattr(optimizer_config, "fused", False)):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        params,
        lr=float(learning_rate),
        betas=betas,
        weight_decay=float(optimizer_config.weight_decay if weight_decay is None else weight_decay),
        eps=float(optimizer_config.eps),
        **optimizer_kwargs,
    )

    if str(scheduler_name) == "multistep_with_warmup":
        milestones = list(optimizer_config.lr_milestones)
        if not milestones:
            raise ValueError("multistep_with_warmup requires training.optimizer.lr_milestones")
        warmup_steps = int(optimizer_config.lr_warmup_steps)
        if warmup_steps <= 0:
            raise ValueError("multistep_with_warmup requires lr_warmup_steps > 0")

        def warmup(current_step: int) -> float:
            return (current_step + 1) / (warmup_steps + 1)

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup,
        )
        step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(optimizer_config.lr_gamma),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[warmup_steps],
        )
    else:
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


def seed_adamw_parameter_state(
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
) -> None:
    """Create AdamW state using PyTorch's fused/capturable device contract."""
    if optimizer.state.get(parameter):
        return
    owner_group = next(
        (group for group in optimizer.param_groups if any(candidate is parameter for candidate in group["params"])),
        None,
    )
    if owner_group is None:
        raise ValueError("Cannot seed optimizer state for an unowned parameter")
    step_on_parameter = bool(owner_group.get("capturable", False) or owner_group.get("fused", False))
    step_device = parameter.device if step_on_parameter else torch.device("cpu")
    optimizer.state[parameter] = {
        "step": torch.zeros((), dtype=torch.float32, device=step_device),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }


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


@torch.no_grad()
def measure_grad_norm(module: torch.nn.Module) -> float:
    """Measure the global L2 gradient norm without changing gradients.

    This uses the same DTensor-aware reduction as gradient clipping, but it
    deliberately does not apply a clip coefficient. In particular, an
    overflowing norm must remain diagnostic evidence rather than multiplying
    otherwise-finite gradients by ``inf / inf``.
    """
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not gradients:
        return 0.0
    grad_norm = torch.nn.utils.get_total_norm(
        gradients,
        norm_type=2.0,
        error_if_nonfinite=False,
        foreach=None,
    )
    if isinstance(grad_norm, DTensor):
        grad_norm = grad_norm.full_tensor()
    return float(grad_norm.item())


@torch.no_grad()
def sanitize_non_finite_gradients(module: torch.nn.Module) -> int:
    """Replace non-finite local gradient values using ``torch.nan_to_num``.

    FSDP2 exposes gradients as DTensors. Mutating their local tensor avoids a
    collective while applying the same per-rank sanitation as FastGen's
    gradient callback.
    """
    non_finite_count = 0
    for parameter in module.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        local_grad = grad._local_tensor if isinstance(grad, DTensor) else grad
        count = int((~torch.isfinite(local_grad)).sum().item())
        if count:
            non_finite_count += count
            torch.nan_to_num_(local_grad)
    return non_finite_count
