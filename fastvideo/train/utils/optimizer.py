# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
    module: torch.nn.Module | None = None,
) -> tuple[torch.optim.Optimizer, object]:
    """Build the optimizer (AdamW or Muon) and LR scheduler.

    ``optimizer_config.optimizer_type`` selects the optimizer. For ``"muon"``,
    ``module`` must be provided so the 2-D hidden weight matrices can be split
    from embeddings / output head / 1-D params by name (the latter fall back to
    an auxiliary AdamW group). Returns ``(optimizer, lr_scheduler)``.
    """
    if not params:
        raise ValueError("No trainable parameters passed to "
                         "build_optimizer_and_scheduler")

    opt_type = str(getattr(optimizer_config, "optimizer_type", "adamw")).lower()
    if opt_type == "muon":
        if module is None:
            raise ValueError("optimizer_type='muon' requires the `module` argument so Muon "
                             "can classify 2-D weights vs embeddings/head/1-D params")
        from fastvideo.train.utils.muon import (
            MuonWithAuxAdam,
            split_params_for_muon,
        )
        muon_params, aux_params = split_params_for_muon([(n, p) for n, p in module.named_parameters()
                                                         if p.requires_grad])
        muon_lr = float(optimizer_config.muon_lr) or float(learning_rate)
        optimizer: torch.optim.Optimizer = MuonWithAuxAdam(
            muon_params,
            aux_params,
            lr=muon_lr,
            momentum=float(optimizer_config.muon_momentum),
            weight_decay=float(optimizer_config.muon_weight_decay),
            ns_steps=int(optimizer_config.muon_ns_steps),
            aux_lr=float(learning_rate),
            aux_betas=betas,
            aux_eps=1e-8,
            aux_weight_decay=float(optimizer_config.weight_decay),
        )
    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=float(learning_rate),
            betas=betas,
            weight_decay=float(optimizer_config.weight_decay),
            eps=1e-8,
        )
    else:
        raise ValueError(f"Unknown training.optimizer.optimizer_type={opt_type!r} "
                         "(expected 'adamw' or 'muon')")

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
