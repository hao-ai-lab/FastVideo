# SPDX-License-Identifier: Apache-2.0
"""SDE step with log-probability computation for RL training."""

from __future__ import annotations

import math

import torch
from diffusers.utils.torch_utils import randn_tensor


def sde_step_with_logprob(
    scheduler,
    model_output: torch.FloatTensor,
    timestep: float | torch.FloatTensor,
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: torch.FloatTensor | None = None,
    generator: torch.Generator | None = None,
    sde_type: str | None = "flow_sde",
    deterministic: bool = False,
    return_sqrt_dt_and_std_dev_t: bool = False,
    diffusion_clip: bool = False,
    diffusion_clip_value: float = 0.45,
):
    """Predict the sample from the previous timestep by reversing
    the SDE, returning log-probability of the transition.

    Args:
        scheduler: Noise scheduler with sigmas and timestep index.
        model_output: Predicted noise/velocity.
        timestep: Current timestep(s).
        sample: Current latents.
        noise_level: Noise level for SDE/CPS computation.
        prev_sample: Optional precomputed previous sample.
        generator: Optional RNG for sampling prev_sample.
        sde_type: 'flow_sde' or 'flow_cps'.
        deterministic: If True, no noise added.
        return_sqrt_dt_and_std_dev_t: If True, return extra terms.
        diffusion_clip: If True, clip std_dev_t.
        diffusion_clip_value: Clipping threshold.

    Returns:
        If return_sqrt_dt_and_std_dev_t:
            (prev_sample, log_prob, prev_sample_mean,
             std_dev_t, sqrt_neg_dt, sigma, sigma_max)
        Else:
            (prev_sample, log_prob, prev_sample_mean,
             std_dev_t * sqrt_neg_dt, sigma, sigma_max)
    """
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    step_index = [
        scheduler.index_for_timestep(t) for t in timestep
    ]
    prev_step_index = [step + 1 for step in step_index]

    scheduler.sigmas = scheduler.sigmas.to(sample.device)
    sigma = scheduler.sigmas[step_index].view(-1, 1, 1, 1, 1)
    sigma_prev = (
        scheduler.sigmas[prev_step_index].view(-1, 1, 1, 1, 1)
    )
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma

    if sde_type == "flow_sde":
        std_dev_t = (
            torch.sqrt(
                sigma
                / (
                    1
                    - torch.where(
                        sigma == 1,
                        torch.tensor(
                            sigma_max,
                            device=sigma.device,
                            dtype=sigma.dtype,
                        ),
                        sigma,
                    )
                )
            )
            * noise_level
        )

        if diffusion_clip:
            max_std_dev_t = (
                diffusion_clip_value / torch.sqrt(-1 * dt)
            )
            std_dev_t = torch.minimum(std_dev_t, max_std_dev_t)

        prev_sample_mean = (
            sample
            * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output
            * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma))
            * dt
        )

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = (
                prev_sample_mean
                + std_dev_t * torch.sqrt(-1 * dt) * variance_noise
            )

        if deterministic:
            prev_sample = sample + dt * model_output

        log_prob = (
            -(
                (prev_sample.detach() - prev_sample_mean) ** 2
            )
            / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(
                torch.sqrt(2 * torch.as_tensor(math.pi))
            )
        )

    elif sde_type == "flow_cps":
        std_dev_t = sigma_prev * math.sin(
            noise_level * math.pi / 2
        )
        pred_original_sample = sample - sigma * model_output
        noise_estimate = (
            sample + model_output * (1 - sigma)
        )
        prev_sample_mean = pred_original_sample * (
            1 - sigma_prev
        ) + noise_estimate * torch.sqrt(
            sigma_prev**2 - std_dev_t**2
        )

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = (
                prev_sample_mean + std_dev_t * variance_noise
            )

        if deterministic:
            prev_sample = (
                pred_original_sample * (1 - sigma_prev)
                + noise_estimate * sigma_prev
            )

        log_prob = -(
            (prev_sample.detach() - prev_sample_mean) ** 2
        )

    else:
        msg = (
            f"Unknown sde_type: {sde_type}. "
            "Must be 'flow_sde' or 'flow_cps'."
        )
        raise ValueError(msg)

    log_prob = log_prob.mean(
        dim=tuple(range(1, log_prob.ndim))
    )

    if return_sqrt_dt_and_std_dev_t:
        return (
            prev_sample,
            log_prob,
            prev_sample_mean,
            std_dev_t,
            torch.sqrt(-1 * dt),
            sigma,
            sigma_max,
        )
    return (
        prev_sample,
        log_prob,
        prev_sample_mean,
        std_dev_t * torch.sqrt(-1 * dt),
        sigma,
        sigma_max,
    )
