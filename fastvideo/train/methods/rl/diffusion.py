# SPDX-License-Identifier: Apache-2.0
"""Single diffusion step for PPO training phase."""

from __future__ import annotations

import torch

from fastvideo.train.methods.rl.sde import (
    sde_step_with_logprob,
)


def compute_log_prob(
    model,
    scheduler,
    sample: dict[str, torch.Tensor],
    j: int,
    embeds: torch.Tensor,
    negative_embeds: torch.Tensor | None,
    guidance_scale: float,
    use_cfg: bool,
    noise_level: float,
    sde_type: str,
    diffusion_clip: bool = False,
    diffusion_clip_value: float = 0.45,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    """Run one diffusion step and return log-probability.

    Uses model.forward_transformer_raw() for the forward
    pass.

    Args:
        model: WanModel instance.
        scheduler: Noise scheduler.
        sample: Dict with latents, next_latents, timesteps.
        j: Timestep index within the trajectory.
        embeds: Conditional text embeddings.
        negative_embeds: Unconditional embeddings (or None).
        guidance_scale: CFG scale.
        use_cfg: Whether to use classifier-free guidance.
        noise_level: SDE noise level.
        sde_type: 'flow_sde' or 'flow_cps'.
        diffusion_clip: Clip SDE variance.
        diffusion_clip_value: Clip threshold.

    Returns:
        (prev_sample, log_prob, prev_sample_mean,
         std_dev_t, dt_sqrt, sigma, sigma_max)
    """
    dtype = embeds.dtype
    latents_j = sample["latents"][:, j]
    timestep_j = sample["timesteps"][:, j]

    if use_cfg and negative_embeds is not None:
        noise_pred_text = model.forward_transformer_raw(
            latents_j.to(dtype),
            timestep_j,
            embeds,
        )
        noise_pred_uncond = model.forward_transformer_raw(
            latents_j.to(dtype),
            timestep_j,
            negative_embeds,
        )
        noise_pred = (
            noise_pred_uncond
            + guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = model.forward_transformer_raw(
            latents_j.to(dtype),
            timestep_j,
            embeds,
        )

    (
        prev_sample,
        log_prob,
        prev_sample_mean,
        std_dev_t,
        dt_sqrt,
        sigma,
        sigma_max,
    ) = sde_step_with_logprob(
        scheduler,
        noise_pred.float(),
        timestep_j,
        latents_j.float(),
        noise_level=noise_level,
        prev_sample=sample["next_latents"][:, j].float(),
        sde_type=sde_type,
        diffusion_clip=diffusion_clip,
        diffusion_clip_value=diffusion_clip_value,
        return_sqrt_dt_and_std_dev_t=True,
    )
    return (
        prev_sample,
        log_prob,
        prev_sample_mean,
        std_dev_t,
        dt_sqrt,
        sigma,
        sigma_max,
    )
