# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def resolve_denoising_steps(
    raw_steps: Iterable[int] | None,
    scheduler_timesteps: torch.Tensor,
    warp_denoising_step: bool,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert configured denoising steps into runtime scheduler timesteps."""
    if raw_steps is None:
        raise ValueError("raw_steps must be provided")

    steps = torch.tensor(list(raw_steps), dtype=torch.long)
    if warp_denoising_step:
        timesteps = torch.cat(
            (
                scheduler_timesteps.detach().cpu(),
                torch.tensor([0], dtype=torch.float32),
            )
        )
        steps = timesteps[1000 - steps]

    if device is not None:
        steps = steps.to(device)
    return steps


def build_block_denoising_steps(
    base_steps: torch.Tensor,
    block_index: int,
    use_diagonal_denoising: bool,
    warmup_mid_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build the per-block denoising schedule for diagonal denoising.

    - block 0: first + 2 mids + last
    - block 1: first + 1 mid + last
    - block >= 2: first + last
    """
    if (not use_diagonal_denoising) or base_steps.numel() <= 1:
        return base_steps

    first = base_steps[0:1]
    last = base_steps[-1:]
    mids = base_steps[1:-1]

    if warmup_mid_steps is not None and warmup_mid_steps.numel() > 0:
        mids = warmup_mid_steps.to(
            device=base_steps.device,
            dtype=base_steps.dtype,
        )

    if block_index <= 0:
        if mids.numel() >= 2:
            selected_mid = mids[:2]
        elif mids.numel() == 1:
            selected_mid = mids
        else:
            selected_mid = torch.empty(
                0,
                dtype=base_steps.dtype,
                device=base_steps.device,
            )
        return torch.cat([first, selected_mid, last], dim=0)

    if block_index == 1:
        if mids.numel() >= 1:
            return torch.cat([first, mids[:1], last], dim=0)
        return torch.cat([first, last], dim=0)

    return torch.cat([first, last], dim=0)


def resolve_base_denoising_timesteps(
    pipeline_config: Any,
    scheduler_timesteps: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Resolve the base denoising steps and optional diagonal warmup mids."""
    timesteps = resolve_denoising_steps(
        pipeline_config.dmd_denoising_steps,
        scheduler_timesteps,
        getattr(pipeline_config, "warp_denoising_step", False),
        device=device,
    )
    warmup_mid_steps = None
    raw_warmup_mid_steps = getattr(
        pipeline_config,
        "diagonal_warmup_mid_steps",
        None,
    )
    if raw_warmup_mid_steps:
        warmup_mid_steps = resolve_denoising_steps(
            raw_warmup_mid_steps,
            scheduler_timesteps,
            getattr(pipeline_config, "warp_denoising_step", False),
            device=device,
        )
    return timesteps, warmup_mid_steps


def resolve_block_denoising_timesteps(
    base_timesteps: torch.Tensor,
    block_idx: int,
    use_diagonal_denoising: bool,
    diagonal_warmup_mid_steps: torch.Tensor | None,
) -> torch.Tensor:
    """Resolve the per-block denoising timesteps for causal rollout."""
    return build_block_denoising_steps(
        base_steps=base_timesteps,
        block_index=block_idx,
        use_diagonal_denoising=use_diagonal_denoising,
        warmup_mid_steps=diagonal_warmup_mid_steps,
    )
