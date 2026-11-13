# SPDX-License-Identifier: Apache-2.0
"""Fixed Open-H3-Regenerate-2K refine schedule: Euler, CFG 1, model time ``1 - sigma``."""

from __future__ import annotations

import torch

REFINE_SHIFT = 6.0
DEFAULT_REFINE_STEPS = 3
# Start noise of the two-step variant; the three-step grid starts at 0.909375.
TWO_STEP_START_SIGMA = 0.7


def shifted_grid(start: float, steps: int, shift: float = REFINE_SHIFT) -> list[float]:
    """Euler grid from ``start`` to 0: even spacing in unshifted sigma, then ``s * x / (1 + (s - 1) * x)``."""
    if not 0.0 < start < 1.0:
        raise ValueError(f"start sigma must be in (0, 1), got {start}.")
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}.")
    raw = start / (shift - (shift - 1) * start)
    mapped = [shift * x / (1 + (shift - 1) * x) for x in (raw * (1 - i / steps) for i in range(1, steps))]
    return [start, *mapped, 0.0]


REFINE_SCHEDULES: dict[int, tuple[float, ...]] = {
    3: (0.909375, 6 / 7, 2 / 3, 0.0),
    2: tuple(shifted_grid(TWO_STEP_START_SIGMA, 2)),
}


def refine_sigmas(steps: int = DEFAULT_REFINE_STEPS, start_sigma: float | None = None) -> list[float]:
    """Sigma grid of ``steps`` Euler steps, ending at 0."""
    if start_sigma is not None:
        return shifted_grid(start_sigma, steps)
    if steps not in REFINE_SCHEDULES:
        raise ValueError(f"Open-H3-Regenerate-2K supports {sorted(REFINE_SCHEDULES)} refine steps, got {steps}.")
    return list(REFINE_SCHEDULES[steps])


def mix_initial_state(upscaled: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
    """Re-noise the upscaled draft latent to the first sigma: ``(1 - sigma) * upscaled + sigma * noise``."""
    return (1.0 - sigma) * upscaled + sigma * noise
