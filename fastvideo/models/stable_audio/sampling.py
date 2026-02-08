# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio k-diffusion v-prediction sampling.

Uses k-diffusion only (VDenoiser, get_sigmas_polyexponential, sample_dpmpp_2m_sde).
No dependency on stable-audio-tools.
"""
from __future__ import annotations

from typing import Any

import torch


# Default sampler config matching stable-audio-tools
DEFAULT_SAMPLER_TYPE = "dpmpp-2m-sde"
DEFAULT_SIGMA_MIN = 0.01
DEFAULT_SIGMA_MAX = 100.0
DEFAULT_RHO = 1.0


def sample_stable_audio(
    model_fn: torch.nn.Module,
    noise: torch.Tensor,
    *,
    init_data: torch.Tensor | None = None,
    steps: int = 100,
    sampler_type: str = DEFAULT_SAMPLER_TYPE,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    sigma_max: float = DEFAULT_SIGMA_MAX,
    rho: float = DEFAULT_RHO,
    device: str | torch.device = "cuda",
    **extra_args: Any,
) -> torch.Tensor:
    """
    Run k-diffusion sampling (v-prediction) compatible with Stable Audio.

    Uses get_sigmas_polyexponential and VDenoiser; default sampler is dpmpp-2m-sde.
    Requires: pip install k-diffusion (or pip install .[stable-audio]).
    """
    import k_diffusion as K

    device_str = str(device)
    denoiser = K.external.VDenoiser(model_fn)
    sigmas = K.sampling.get_sigmas_polyexponential(
        steps, sigma_min, sigma_max, rho, device=device_str
    )
    noise = noise.to(device_str) * sigmas[0]
    if init_data is not None:
        x = init_data.to(device_str) + noise
    else:
        x = noise

    if sampler_type == "dpmpp-2m-sde":
        return K.sampling.sample_dpmpp_2m_sde(
            denoiser, x, sigmas, disable=False, callback=None, extra_args=extra_args
        )
    if sampler_type == "dpmpp-2m":
        return K.sampling.sample_dpmpp_2m(
            denoiser, x, sigmas, disable=False, callback=None, extra_args=extra_args
        )
    if sampler_type == "dpmpp-3m-sde":
        return K.sampling.sample_dpmpp_3m_sde(
            denoiser, x, sigmas, disable=False, callback=None, extra_args=extra_args
        )
    if sampler_type == "k-heun":
        return K.sampling.sample_heun(
            denoiser, x, sigmas, disable=False, callback=None, extra_args=extra_args
        )
    if sampler_type == "k-lms":
        return K.sampling.sample_lms(
            denoiser, x, sigmas, disable=False, callback=None, extra_args=extra_args
        )
    raise ValueError(f"Unsupported sampler_type: {sampler_type}")
