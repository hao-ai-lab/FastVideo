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


class AdamWBeta1Zero(torch.optim.Optimizer):
    """AdamW specialized for ``beta1 == 0``: identical update, no ``exp_avg``.

    With ``beta1 = 0`` Adam's first moment reduces to the raw gradient
    (``m_t = g_t``, bias correction 1), so the buffer only doubles optimizer
    state for nothing — one full parameter-sized tensor per model. The op
    sequence below mirrors ``torch.optim.AdamW``'s single-tensor path
    exactly, so the parameter trajectory is bitwise-equivalent to
    ``AdamW(betas=(0.0, beta2))``.
    """

    def __init__(
        self,
        params,
        lr: float,
        beta2: float,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        defaults = dict(lr=float(lr), beta2=float(beta2), eps=float(eps), weight_decay=float(weight_decay))
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    # Low-precision params must not starve the second moment:
                    # bf16 v loses g^2 increments below ~0.4% of its running
                    # magnitude. Keep v in fp32 whenever p is not fp32 (the
                    # fp32-master path is unchanged and stays bitwise-equal
                    # to torch.optim.AdamW).
                    if p.dtype == torch.float32:
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                exp_avg_sq = state["exp_avg_sq"]
                p.mul_(1 - lr * weight_decay)
                if p.dtype == torch.float32:
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction2_sqrt = (1 - beta2**state["step"])**0.5
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                    p.addcdiv_(grad, denom, value=-lr)
                else:
                    grad_f = grad.float()
                    exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1 - beta2)
                    bias_correction2_sqrt = (1 - beta2**state["step"])**0.5
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                    update = grad_f.div_(denom).mul_(-lr)
                    p.add_(update.to(p.dtype))
        return loss


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

    if float(betas[0]) == 0.0:
        # beta1 == 0 makes Adam's first-moment buffer redundant; skip it to
        # save one parameter-sized state tensor per trainable model.
        optimizer: torch.optim.Optimizer = AdamWBeta1Zero(
            params,
            lr=float(learning_rate),
            beta2=float(betas[1]),
            weight_decay=float(optimizer_config.weight_decay),
            eps=1e-8,
        )
    else:
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
