# SPDX-License-Identifier: Apache-2.0
"""Stochastic (SDE) flow-matching transitions with log probabilities.

Adapted from the Flow-GRPO ``sde_step_with_logprob`` patch
(Apache-2.0), converted to stateless functions over explicit sigma
pairs so transitions can be recomputed later from stored states:

* ``sde_step_from_model_output`` — rollout-time stochastic step that
  also returns the transition log probability under the behavior
  policy.
* ``transition_log_prob`` — recompute the log probability of a stored
  transition under a (new or reference) policy.
* ``transition_kl_to_reference`` — Gaussian KL from the current
  transition to the frozen-base transition at the same step.

Conventions match ``FlowMatchEulerDiscreteScheduler``:
``x = (1 - sigma) * x0 + sigma * noise``, Euler step
``x' = x + (sigma_next - sigma) * model_output`` where ``model_output``
is the predicted velocity.  Latents may be 5D video tensors
``[B, C, T, H, W]``; log probs are reduced over all non-batch dims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SDETransition:
    """One stored stochastic transition from rollout time."""

    sample: torch.Tensor  # x_t (detached)
    prev_sample: torch.Tensor  # x_{t+1} actually sampled (detached)
    timestep: float
    sigma: float
    sigma_next: float
    old_log_prob: torch.Tensor  # [B] behavior-policy log prob (detached)


def _std_dev_t(sigma: float, sigma_max: float, noise_scale: float) -> float:
    denom = 1.0 - (sigma_max if sigma == 1.0 else sigma)
    denom = max(denom, 1e-6)
    return math.sqrt(max(sigma, 0.0) / denom) * float(noise_scale)


def sde_step_from_model_output(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    *,
    sigma: float,
    sigma_next: float,
    noise_scale: float,
    sigma_max: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take one stochastic reverse step; return (prev, log_prob, mean).

    ``log_prob`` is the per-sample transition log probability of the
    sampled ``prev`` under the Gaussian transition kernel, reduced over
    all non-batch dimensions.
    """
    model_output = model_output.float()
    sample = sample.float()
    dt = float(sigma_next) - float(sigma)  # negative
    std = _std_dev_t(float(sigma), float(sigma_max), noise_scale)

    std_sq = std * std
    prev_mean = (sample * (1.0 + std_sq / (2.0 * sigma) * dt) +
                 model_output * (1.0 + std_sq * (1.0 - sigma) / (2.0 * sigma)) * dt)

    noise = torch.randn(
        prev_mean.shape,
        device=prev_mean.device,
        dtype=prev_mean.dtype,
        generator=generator,
    )
    step_std = std * math.sqrt(-dt)
    prev_sample = prev_mean + step_std * noise
    log_prob = _gaussian_log_prob(prev_sample.detach(), prev_mean, step_std)
    return prev_sample, log_prob, prev_mean


def transition_log_prob(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    prev_sample: torch.Tensor,
    *,
    sigma: float,
    sigma_next: float,
    noise_scale: float,
    sigma_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the log prob of a stored transition.

    Returns ``(log_prob, prev_mean)`` where ``prev_mean`` keeps its
    gradient (for the policy loss) and ``prev_sample`` is treated as a
    constant.
    """
    model_output = model_output.float()
    sample = sample.float()
    prev_sample = prev_sample.float()
    dt = float(sigma_next) - float(sigma)
    std = _std_dev_t(float(sigma), float(sigma_max), noise_scale)
    std_sq = std * std
    prev_mean = (sample * (1.0 + std_sq / (2.0 * sigma) * dt) +
                 model_output * (1.0 + std_sq * (1.0 - sigma) / (2.0 * sigma)) * dt)
    step_std = std * math.sqrt(-dt)
    log_prob = _gaussian_log_prob(prev_sample, prev_mean, step_std)
    return log_prob, prev_mean


def transition_kl_to_reference(
    prev_mean: torch.Tensor,
    ref_prev_mean: torch.Tensor,
    *,
    sigma: float,
    sigma_next: float,
    noise_scale: float,
    sigma_max: float,
) -> torch.Tensor:
    """Gaussian KL(current || reference) for one transition.

    Both kernels share the same (policy-independent) scale
    ``step_std = std * sqrt(-dt)``, so the KL reduces to the squared
    mean gap over ``2 * step_std^2``, summed over non-batch dims.
    """
    dt = float(sigma_next) - float(sigma)
    std = _std_dev_t(float(sigma), float(sigma_max), noise_scale)
    step_std_sq = max(std * std * (-dt), 1e-12)
    gap = (prev_mean.float() - ref_prev_mean.detach().float()) ** 2
    reduce_dims = tuple(range(1, gap.ndim))
    return (gap.sum(dim=reduce_dims) / (2.0 * step_std_sq))


def _gaussian_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Diagonal-Gaussian log density, mean over non-batch dims.

    Mirrors the Flow-GRPO reduction (mean over pixels/channels), which
    keeps per-step log probs on a comparable scale across resolutions.
    """
    std = max(float(std), 1e-8)
    log_prob = (-((value.detach() - mean) ** 2) / (2.0 * std * std) - math.log(std) -
                math.log(math.sqrt(2.0 * math.pi)))
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
