# SPDX-License-Identifier: Apache-2.0
"""Parallel Decoding Distillation (PDD) sampling primitives.

A PDD student widens a diffusion transformer's final output projection so one
forward emits ``pdd_steps`` head predictions, one per interval of a fixed fine
time grid. Head ``j`` is the modality's rectified-flow direction frozen on
interval ``j``, so a block ``[start, end)`` of heads advances the state by
``sum_j (sigma(u_{j+1}) - sigma(u_j)) * q_j``.

The sampler never materializes every head. For each runtime step it fuses one
block of head parameters inside the linear layer (:meth:`PDDReplicatedLinear.fuse`)
into their integration-weighted mean, and an ordinary Euler step over the
block's two node sigmas applies it: the block's total weight is exactly its
node-sigma increment.

The fine grid lives on the base (unshifted) clock ``[0, 0.999]``; every
modality reaches its own noise level through its rational time shift. The
arithmetic matches the rectified-flow schedule the students are trained with.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from fastvideo.layers.quantization.base_config import QuantizationConfig

# Rectified-flow students end their finite time domain at 0.999 and build the
# PDD fine grid on ``[0, 0.999]``.
PDD_GRID_MAX_T = 0.999


def validate_finite_nonzero(value: torch.Tensor, name: str) -> None:
    if not bool(torch.all(torch.isfinite(value) & (value != 0))):
        raise ValueError(f"{name} must be finite and non-zero")


def validate_pdd_steps(value: Any, name: str = "pdd_steps") -> int:
    """A PDD grid needs at least two intervals to have a block structure."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"{name} must be an int >= 2, got {value!r}")
    return value


# ----------------------------------------------------------------------------
# Shifted rectified-flow arithmetic on the base clock
# ----------------------------------------------------------------------------


def shifted_noise_amount(base: torch.Tensor, shift: float, *, max_t: float = PDD_GRID_MAX_T) -> torch.Tensor:
    """Rational time shift ``f_s(u) = s * u * M / (u * (s - 1) + M)`` on ``[0, M]``.

    ``M = max_t`` is a fixed point of every shift, so the first grid node has
    the same noise level in every modality.
    """
    if shift == 1.0:
        return base
    return shift * base * max_t / (base * (shift - 1.0) + max_t)


def shifted_noise_delta(
    base_start: torch.Tensor,
    base_end: torch.Tensor,
    shift: float,
    *,
    max_t: float = PDD_GRID_MAX_T,
) -> torch.Tensor:
    """Return ``f_s(base_end) - f_s(base_start)`` without subtractive cancellation.

    The rational difference simplifies to
    ``s * M^2 * (b - a) / (D(a) * D(b))`` with ``D(u) = M + (s - 1) * u``; the
    identity shift keeps the direct difference bit-exact.
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

    def sigma(self, base: torch.Tensor) -> torch.Tensor:
        return shifted_noise_amount(base, self.shift, max_t=self.max_t)

    def integration_weights(self, fine_grid: torch.Tensor) -> torch.Tensor:
        """``sigma(u_{j+1}) - sigma(u_j)`` for every fine-grid interval ``j``.

        Freezing head ``j`` on its interval advances the state by exactly this
        coefficient. Negative for a descending grid.
        """
        return shifted_noise_delta(fine_grid[:-1], fine_grid[1:], self.shift, max_t=self.max_t)


# ----------------------------------------------------------------------------
# Fine grid and runtime partitions
# ----------------------------------------------------------------------------


def pdd_fine_grid(
    pdd_steps: int,
    *,
    max_t: float = PDD_GRID_MAX_T,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """The descending base-clock grid ``u_0 = max_t > ... > u_N = 0`` in float64."""
    grid = torch.linspace(max_t, 0.0, pdd_steps + 1, dtype=torch.float64, device=device)
    return grid.clamp(max=max_t)


@dataclass(frozen=True)
class PDDSamplingPlan:
    """One runtime partition of the fine grid plus every modality's coefficients."""

    fine_grid: torch.Tensor
    indices: torch.Tensor
    node_sigmas: dict[str, torch.Tensor]
    integration_weights: dict[str, torch.Tensor]

    @property
    def pdd_steps(self) -> int:
        return int(self.fine_grid.numel()) - 1

    @property
    def num_steps(self) -> int:
        return int(self.indices.numel()) - 1

    @property
    def block_sizes(self) -> list[int]:
        return (self.indices[1:] - self.indices[:-1]).tolist()

    def block(self, step: int) -> tuple[int, int]:
        """Half-open fine-grid block ``[start, end)`` of runtime step *step*."""
        return int(self.indices[step]), int(self.indices[step + 1])


def build_pdd_sampling_plan(
    step_indices: Sequence[int],
    schedules: Mapping[str, PDDModalitySchedule],
    *,
    device: torch.device | str | None = None,
) -> PDDSamplingPlan:
    """Partition the fine grid into fused blocks for every modality.

    *step_indices* is the checkpoint's trained partition: fine-grid nodes that
    increase strictly from 0 to the grid size, one fused block per consecutive
    pair. ``MiniMaxH3PipelineConfig.resolve_checkpoint_settings`` validates it.
    """
    max_ts = {schedule.max_t for schedule in schedules.values()}
    if len(max_ts) != 1:
        raise ValueError(f"Every PDD modality schedule must share max_t, got {sorted(max_ts)}")
    fine_grid = pdd_fine_grid(int(step_indices[-1]), max_t=next(iter(max_ts)), device=device)
    node_indices = torch.as_tensor(list(step_indices), dtype=torch.long, device=fine_grid.device)
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


# ----------------------------------------------------------------------------
# Widened output projection
# ----------------------------------------------------------------------------


class PDDReplicatedLinear(ReplicatedLinear):
    """Final linear layer widened to ``grid_size`` PDD heads.

    Output features are laid out head-major, ``(grid, output_size)``, so after
    the architecture's own unpatchify the output channel axis reads
    ``(grid, C)``. An ordinary ``forward(input)`` returns every head.

    Inside :meth:`fuse`, ``forward(input)`` instead returns the normalized
    weighted average over the heads of one contiguous block ``[start, end)``,
    computed by collapsing that block's parameter slices into one effective
    linear. The architecture's inline final projection therefore needs no
    seam, and the fused parameters are built inside ``forward``, where a
    sharded weight is already gathered.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        grid_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            input_size,
            output_size * grid_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            raise NotImplementedError("PDDReplicatedLinear supports only unquantized weights")
        self.head_output_size = int(output_size)
        self.grid_size = grid_size
        # Transient fusion state is a plain tuple, never a buffer, so it stays
        # out of state_dict and can be restored as one value.
        self._fusion_state: tuple[int, int, torch.Tensor, torch.dtype] | None = None

    @contextlib.contextmanager
    def fuse(
        self,
        start: int,
        end: int,
        weights: torch.Tensor,
        precision_decoding: torch.dtype,
    ) -> Iterator[PDDReplicatedLinear]:
        """Temporarily make ``forward(input)`` return the fused block output.

        *weights* are this modality's per-interval integration weights over the
        whole grid (``[grid_size]``); the block ``[start, end)`` is fused with
        their normalized form, so the layer emits the weighted mean and the
        caller scales by the block's total weight. ``precision_decoding`` is
        used only for the fused-parameter accumulation; the fused parameters
        return to the layer's dtype before the projection runs. The previous
        fusion state is restored on exit.
        """
        previous = self._fusion_state
        self._fusion_state = (start, end, weights, precision_decoding)
        try:
            yield self
        finally:
            self._fusion_state = previous

    def _fused_params(
        self,
        start: int,
        end: int,
        weights: torch.Tensor,
        precision_decoding: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The ``[O, I]`` fused weight (and bias) of block ``[start, end)``."""
        # Normalize in the incoming schedule precision, then cast only the
        # fused-parameter accumulation. The layer emits the weighted mean and
        # the caller multiplies by the block total, so together they represent
        # ``sum_j w_j pred_j``.
        block_weights = weights[start:end].to(device=self.weight.device)
        block_scale = block_weights.sum()
        normalized = (block_weights / block_scale).to(dtype=precision_decoding)

        def fuse_parameter(parameter: torch.Tensor) -> torch.Tensor:
            heads = parameter.unflatten(0, (self.grid_size, -1))[start:end].to(dtype=precision_decoding)
            return torch.einsum("n,n...->...", normalized, heads).to(dtype=parameter.dtype)

        return fuse_parameter(self.weight), None if self.bias is None else fuse_parameter(self.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:  # type: ignore[override]
        fusion_state = self._fusion_state
        if fusion_state is None:
            return super().forward(x)
        fused_weight, fused_bias = self._fused_params(*fusion_state)
        output = F.linear(x, fused_weight, None if self.skip_bias_add else fused_bias)
        return output, fused_bias if self.skip_bias_add else None


@contextlib.contextmanager
def fuse_pdd_heads(
    linears: Mapping[str, PDDReplicatedLinear],
    start: int,
    end: int,
    integration_weights: Mapping[str, torch.Tensor],
    precision_decoding: torch.dtype,
) -> Iterator[None]:
    """Fuse block ``[start, end)`` on every modality head at once."""
    with contextlib.ExitStack() as stack:
        for name, linear in linears.items():
            stack.enter_context(linear.fuse(start, end, integration_weights[name], precision_decoding))
        yield


__all__ = [
    "PDD_GRID_MAX_T",
    "PDDModalitySchedule",
    "PDDReplicatedLinear",
    "PDDSamplingPlan",
    "build_pdd_sampling_plan",
    "fuse_pdd_heads",
    "pdd_fine_grid",
    "shifted_noise_amount",
    "shifted_noise_delta",
    "validate_finite_nonzero",
    "validate_pdd_steps",
]
