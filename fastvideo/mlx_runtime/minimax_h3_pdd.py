# SPDX-License-Identifier: Apache-2.0
"""Parallel Decoding Distillation (PDD) sampling math for the MLX H3 runtime.

A numpy/MLX port of ``fastvideo.layers.pdd`` (no torch). A PDD student widens
the transformer's final output projections so one forward emits ``pdd_steps``
head predictions, one per interval of a fixed fine time grid. Head ``j`` is the
modality's rectified-flow direction frozen on interval ``j``, so a runtime block
``[start, end)`` of heads advances the state by
``sum_j (sigma(u_{j+1}) - sigma(u_j)) * q_j``.

The sampler never materializes every head: for each runtime step it fuses one
block of head parameters inside the widened projection into their
integration-weighted *normalized* mean, and the ordinary Euler step over the
block's two node sigmas supplies the block's total weight. The projection
therefore emits ``sum_j w_j * head_j / sum(w[start:end])`` and the caller's Euler
step scales it by ``sum(w[start:end])``, together equal to ``sum_j w_j head_j``.

The fine grid lives on the base (unshifted) clock ``[0, 0.999]``; each modality
reaches its own noise level through its rational time shift. Every formula
matches ``fastvideo/layers/pdd.py`` exactly, including the cancellation-free
``shifted_noise_delta`` and the ``shift == 1.0`` early returns — the MLX runtime
must reproduce the CUDA sampler's numbers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Rectified-flow students end their finite time domain at 0.999 and build the
# PDD fine grid on ``[0, 0.999]``.
PDD_GRID_MAX_T = 0.999


def validate_finite_nonzero(value: np.ndarray, name: str) -> None:
    if not bool(np.all(np.isfinite(value) & (value != 0))):
        raise ValueError(f"{name} must be finite and non-zero")


def validate_pdd_steps(value: Any, name: str = "pdd_steps") -> int:
    """A PDD grid needs at least two intervals to have a block structure."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"{name} must be an int >= 2, got {value!r}")
    return value


def shifted_noise_amount(base: np.ndarray, shift: float, *, max_t: float = PDD_GRID_MAX_T) -> np.ndarray:
    """Rational time shift ``f_s(u) = s * u * M / (u * (s - 1) + M)`` on ``[0, M]``.

    ``M = max_t`` is a fixed point of every shift, so the first grid node has
    the same noise level in every modality.
    """
    if shift == 1.0:
        return base
    return shift * base * max_t / (base * (shift - 1.0) + max_t)


def shifted_noise_delta(
    base_start: np.ndarray,
    base_end: np.ndarray,
    shift: float,
    *,
    max_t: float = PDD_GRID_MAX_T,
) -> np.ndarray:
    """Return ``f_s(base_end) - f_s(base_start)`` without subtractive cancellation.

    The rational difference simplifies to ``s * M^2 * (b - a) / (D(a) * D(b))``
    with ``D(u) = M + (s - 1) * u``; the identity shift keeps the direct
    difference bit-exact.
    """
    if shift == 1.0:
        return base_end - base_start
    denominator_start = max_t + (shift - 1.0) * base_start
    denominator_end = max_t + (shift - 1.0) * base_end
    return shift * max_t**2 * (base_end - base_start) / (denominator_start * denominator_end)


@dataclass(frozen=True)
class PDDModalitySchedule:
    """One modality's noise level as a function of the base clock: ``sigma(u) = f_shift(u)``."""

    shift: float
    max_t: float = PDD_GRID_MAX_T

    def sigma(self, base: np.ndarray) -> np.ndarray:
        return shifted_noise_amount(base, self.shift, max_t=self.max_t)

    def integration_weights(self, fine_grid: np.ndarray) -> np.ndarray:
        """``sigma(u_{j+1}) - sigma(u_j)`` for every fine-grid interval ``j``.

        Freezing head ``j`` on its interval advances the state by exactly this
        coefficient. Negative for a descending grid.
        """
        return shifted_noise_delta(fine_grid[:-1], fine_grid[1:], self.shift, max_t=self.max_t)


def pdd_fine_grid(pdd_steps: int, *, max_t: float = PDD_GRID_MAX_T) -> np.ndarray:
    """The descending base-clock grid ``u_0 = max_t > ... > u_N = 0`` in float64."""
    grid = np.linspace(max_t, 0.0, pdd_steps + 1, dtype=np.float64)
    return np.minimum(grid, max_t)


@dataclass(frozen=True)
class PDDSamplingPlan:
    """One runtime partition of the fine grid plus every modality's coefficients."""

    fine_grid: np.ndarray
    indices: np.ndarray
    node_sigmas: dict[str, np.ndarray]
    integration_weights: dict[str, np.ndarray]

    @property
    def pdd_steps(self) -> int:
        return int(self.fine_grid.size) - 1

    @property
    def num_steps(self) -> int:
        return int(self.indices.size) - 1

    @property
    def block_sizes(self) -> list[int]:
        return (self.indices[1:] - self.indices[:-1]).tolist()

    def block(self, step: int) -> tuple[int, int]:
        """Half-open fine-grid block ``[start, end)`` of runtime step *step*."""
        return int(self.indices[step]), int(self.indices[step + 1])


def build_pdd_sampling_plan(
    step_indices: Sequence[int],
    schedules: Mapping[str, PDDModalitySchedule],
) -> PDDSamplingPlan:
    """Partition the fine grid into fused blocks for every modality.

    *step_indices* is the checkpoint's trained partition: fine-grid nodes that
    increase strictly from 0 to the grid size, one fused block per consecutive
    pair. The CUDA reference validates it at contract-resolution time; the MLX
    contract loader does the same here.
    """
    max_ts = {schedule.max_t for schedule in schedules.values()}
    if len(max_ts) != 1:
        raise ValueError(f"Every PDD modality schedule must share max_t, got {sorted(max_ts)}")
    fine_grid = pdd_fine_grid(int(step_indices[-1]), max_t=next(iter(max_ts)))
    node_indices = np.asarray(list(step_indices), dtype=np.int64)
    weights = {}
    for name, schedule in schedules.items():
        weights[name] = schedule.integration_weights(fine_grid)
        validate_finite_nonzero(weights[name], f"PDD {name} integration weight")
    return PDDSamplingPlan(
        fine_grid=fine_grid,
        indices=node_indices,
        node_sigmas={
            name: schedule.sigma(fine_grid[node_indices])
            for name, schedule in schedules.items()
        },
        integration_weights=weights,
    )


def resolved_grid_size(weight: np.ndarray, bias: np.ndarray | None, output_size: int) -> int | None:
    """The widened head count implied by a projection, or None when it is single-headed.

    Output features are laid out head-major, ``(grid, output_size)``, so a
    widened head's leading dimension is an exact multiple of ``output_size``.
    A checkpoint that records ``pdd_steps`` resolves through that field
    (``declared_grid``); this shape check is the fallback, and the two must
    agree.
    """
    rows = int(bias.shape[0]) if bias is not None else int(weight.shape[0])
    if output_size <= 0 or rows % output_size:
        raise ValueError(f"Output projection width {rows} is not a multiple of the expected {output_size}.")
    grid = rows // output_size
    return grid if grid > 1 else None


def fuse_block_params(
    weight: Any,
    bias: Any | None,
    *,
    grid_size: int,
    start: int,
    end: int,
    weights: np.ndarray,
) -> tuple[Any, Any | None]:
    """The fused ``[output, input]`` weight (and bias) of head block ``[start, end)``.

    Mirrors ``PDDReplicatedLinear._fused_params``: the block's integration
    weights are normalized in the schedule's (float64) precision, the heads and
    the normalized weights are accumulated in fp32, and the fused parameters
    return to the projection's dtype before the projection runs. The layer emits
    the weighted mean; the caller's Euler step over the block's node sigmas
    supplies the block total, so the two together yield ``sum_j w_j head_j``.
    """
    import mlx.core as mx

    block = np.asarray(weights, dtype=np.float64)[start:end]
    if block.size == 0:
        raise ValueError(f"PDD head block [{start}, {end}) is empty.")
    block_scale = block.sum()
    normalized = mx.array((block / block_scale).astype(np.float32))

    def fuse_parameter(parameter: Any) -> Any:
        heads = parameter.reshape(grid_size, -1)[start:end].astype(mx.float32)
        fused = (normalized[:, None] * heads).sum(axis=0)
        if parameter.ndim == 2:
            fused = fused.reshape(-1, parameter.shape[1])
        return fused.astype(parameter.dtype)

    return fuse_parameter(weight), None if bias is None else fuse_parameter(bias)


__all__ = [
    "PDD_GRID_MAX_T",
    "PDDModalitySchedule",
    "PDDSamplingPlan",
    "build_pdd_sampling_plan",
    "fuse_block_params",
    "pdd_fine_grid",
    "resolved_grid_size",
    "shifted_noise_amount",
    "shifted_noise_delta",
    "validate_finite_nonzero",
    "validate_pdd_steps",
]
