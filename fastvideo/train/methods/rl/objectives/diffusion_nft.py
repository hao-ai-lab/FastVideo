# SPDX-License-Identifier: Apache-2.0
"""DiffusionNFT forward-process objective."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.train.models.base import ModelBase

AdvMode = Literal[
    "all",
    "positive_only",
    "negative_only",
    "one_only",
    "binary",
]


def prediction_to_x0(
    model: ModelBase,
    pred_noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Convert a Wan noise/flow prediction into x0 latents."""
    if timestep.ndim == 1 and pred_noise.ndim >= 5:
        batch, frames = pred_noise.shape[:2]
        if timestep.shape[0] == batch:
            timestep = timestep.view(batch, 1).expand(batch, frames)
    return pred_noise_to_pred_video(
        pred_noise=pred_noise.flatten(0, 1),
        noise_input_latent=noisy_latents.flatten(0, 1),
        timestep=timestep,
        scheduler=model.noise_scheduler,
    ).unflatten(0, pred_noise.shape[:2])


def shape_advantages(
    advantages: torch.Tensor,
    *,
    adv_clip_max: float,
    adv_mode: AdvMode,
) -> torch.Tensor:
    """Clip and optionally sparsify DiffusionNFT advantages."""
    clipped = advantages.float().clamp(-adv_clip_max, adv_clip_max)
    if adv_mode == "positive_only":
        clipped = clipped.clamp(0.0, adv_clip_max)
    elif adv_mode == "negative_only":
        clipped = clipped.clamp(-adv_clip_max, 0.0)
    elif adv_mode == "one_only":
        clipped = torch.where(
            clipped > 0,
            torch.ones_like(clipped),
            torch.zeros_like(clipped),
        )
    elif adv_mode == "binary":
        clipped = torch.sign(clipped)
    return clipped


def compute_diffusion_nft_loss(
    *,
    student: ModelBase,
    forward_prediction: torch.Tensor,
    old_prediction: torch.Tensor,
    noisy_latents: torch.Tensor,
    clean_latents: torch.Tensor,
    timestep: torch.Tensor,
    advantages: torch.Tensor,
    adv_clip_max: float,
    adv_mode: AdvMode,
    nft_beta: float,
    reference_prediction: torch.Tensor | None = None,
    kl_weight: float = 0.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute the DiffusionNFT policy loss and diagnostics."""
    shaped_advantages = shape_advantages(
        advantages,
        adv_clip_max=adv_clip_max,
        adv_mode=adv_mode,
    )
    r = ((shaped_advantages / adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)

    old_detached = old_prediction.detach()
    positive_prediction = (nft_beta * forward_prediction + (1.0 - nft_beta) * old_detached)
    negative_prediction = ((1.0 + nft_beta) * old_detached - nft_beta * forward_prediction)

    positive_x0 = prediction_to_x0(student, positive_prediction, noisy_latents, timestep)
    negative_x0 = prediction_to_x0(student, negative_prediction, noisy_latents, timestep)

    reduce_dims = tuple(range(1, clean_latents.ndim))
    with torch.no_grad():
        positive_weight = ((positive_x0.float() - clean_latents.float()).abs().mean(dim=reduce_dims,
                                                                                    keepdim=True).clamp_min(1e-5))
        negative_weight = ((negative_x0.float() - clean_latents.float()).abs().mean(dim=reduce_dims,
                                                                                    keepdim=True).clamp_min(1e-5))

    positive_loss = ((positive_x0 - clean_latents)**2 / positive_weight).mean(dim=reduce_dims)
    negative_loss = ((negative_x0 - clean_latents)**2 / negative_weight).mean(dim=reduce_dims)
    unweighted_policy_loss = (r * positive_loss / nft_beta + (1.0 - r) * negative_loss / nft_beta)
    policy_loss = (unweighted_policy_loss * adv_clip_max).mean()

    kl_loss = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
    if reference_prediction is not None and kl_weight > 0.0:
        kl_loss = F.mse_loss(forward_prediction.float(), reference_prediction.float())

    total_loss = policy_loss + kl_weight * kl_loss
    losses = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
    }
    metrics = {
        "nft/unweighted_policy_loss": (unweighted_policy_loss.mean().detach()),
        "nft/old_deviate": ((forward_prediction - old_detached)**2).mean().detach(),
        "nft/advantage_abs_mean": (shaped_advantages.abs().mean().detach()),
        "nft/r_mean": r.mean().detach(),
        "nft/timestep_mean": timestep.float().mean().detach(),
    }
    return losses, metrics
