# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio k-diffusion v-prediction sampling.

Uses same sigma schedule and sampler as stable-audio-tools.
"""
from __future__ import annotations

import sys
from typing import Any

import torch


def _get_sample_k():
    """Import sample_k from stable-audio-tools."""
    import os
    proj_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    sat_path = os.path.join(proj_root, "stable-audio-tools")
    if os.path.isdir(sat_path) and sat_path not in sys.path:
        sys.path.insert(0, sat_path)
    from stable_audio_tools.inference.sampling import sample_k
    return sample_k


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

    Uses get_sigmas_polyexponential and VDenoiser like stable-audio-tools.
    """
    sample_k_fn = _get_sample_k()
    return sample_k_fn(
        model_fn,
        noise,
        init_data=init_data,
        steps=steps,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=str(device),
        **extra_args,
    )
