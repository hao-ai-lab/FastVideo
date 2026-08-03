# SPDX-License-Identifier: Apache-2.0
"""Shared VAE conditioning numerics for MiniMax H3."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.pipelines.basic.minimax_h3.packing import randn_tensor


def arch_value(module: Any, name: str) -> Any:
    value = getattr(module, name, None)
    if value is None:
        config = getattr(module, "config", None)
        arch = getattr(config, "arch_config", config)
        value = getattr(arch, name, None)
    if value is None:
        raise ValueError(f"MiniMax-H3 component {type(module).__name__} does not expose `{name}`.")
    return value


def latent_stats(module: Any, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(arch_value(module, "latents_mean"), dtype=torch.float32).detach().cpu().reshape(shape)
    std = torch.as_tensor(arch_value(module, "latents_std"), dtype=torch.float32).detach().cpu().reshape(shape)
    return mean, std


def sample_posterior(posterior: Any, seed: int) -> torch.Tensor:
    """Match Diffusers' CPU-generator draw even when the posterior lives on CUDA."""
    generator = torch.Generator("cpu").manual_seed(seed)
    noise = randn_tensor(
        tuple(posterior.mean.shape),
        generator=generator,
        device=posterior.mean.device,
        dtype=posterior.mean.dtype,
    )
    return posterior.mean + posterior.std * noise


__all__ = ["arch_value", "latent_stats", "sample_posterior"]
