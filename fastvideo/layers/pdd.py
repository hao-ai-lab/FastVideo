# SPDX-License-Identifier: Apache-2.0
"""Parallel Decoding Distillation (PDD) primitives shared by training and inference.

PDD widens a diffusion transformer's final output projection so one forward
emits ``pdd_steps`` head predictions, one per interval of a fixed fine time
grid. Head ``j`` is parameterized as its modality's native rectified-flow
direction ``q_j = eps - x0 = dx/dsigma_m`` frozen on interval ``j``, so a block
``[start, end)`` of heads advances the state by
``x += sum_j (sigma_m(u_{j+1}) - sigma_m(u_j)) * q_j``. A fused forward mixes
the head parameters of one block inside the linear layer instead of
materializing every head; that is how the distilled sampler runs at inference.

This module holds the pieces that the model, the training methods, and the
inference denoising stage all need: the widened :class:`PDDReplicatedLinear`,
the fine grid with its runtime partitions, and the per-modality shifted
rectified-flow arithmetic. The math mirrors FastGen's
``fastgen/networks/utils/pdd_mixin.py`` and the rectified-flow schedule in
``fastgen/networks/noise_schedule.py``; the fine grid lives on the *base*
(unshifted) clock, and every modality reaches its own noise level through its
rational shift, exactly as ``MiniMaxH3DMDModel._noise_amounts`` does.
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
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Widened output-feature layouts.
#   - "patch_major":   teacher rows are (patch, C); the grid is inserted between
#                       them -> (patch, grid, C).
#   - "channel_major": teacher rows are (C, patch); the grid is inserted before
#                       the channel axis -> (grid, C, patch).
# Either way the grid sits immediately *before* the channel axis, so after the
# architecture's own unpatchify the output channel axis reads ``(grid, C)``.
CHANNEL_LAYOUTS = ("patch_major", "channel_major")

# FastGen's rectified-flow schedules end their finite time domain at 0.999 and
# construct the PDD fine grid on ``[0, 0.999]``. MiniMax-H3 uses the same value
# for its continuous (float) score times.
PDD_GRID_MAX_T = 0.999


def validate_finite_nonzero(value: torch.Tensor, name: str) -> None:
    if not bool(torch.all(torch.isfinite(value) & (value != 0))):
        raise ValueError(f"{name} must be finite and non-zero")


def validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


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

    ``M = max_t`` is a fixed point, so the shifts form a group under
    composition (``f_a . f_b = f_ab``). This is FastGen's ``time_shift`` and
    MiniMax-H3's ``shift_noise_amount`` with an explicit ``max_noise_amount``.
    """
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    if not 0.0 < max_t <= 1.0:
        raise ValueError(f"max_t must satisfy 0 < max_t <= 1, got {max_t}")
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
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    if shift == 1.0:
        return base_end - base_start
    denominator_start = max_t + (shift - 1.0) * base_start
    denominator_end = max_t + (shift - 1.0) * base_end
    return shift * max_t**2 * (base_end - base_start) / (denominator_start * denominator_end)


def unshifted_noise_amount(sigma: torch.Tensor, shift: float, *, max_t: float = PDD_GRID_MAX_T) -> torch.Tensor:
    """Base time whose shifted noise amount is *sigma*: ``f_(1/s)``."""
    return shifted_noise_amount(sigma, 1.0 / shift, max_t=max_t)


@dataclass(frozen=True)
class PDDModalitySchedule:
    """One modality's rectified-flow noise level as a function of the base clock.

    ``sigma(u) = f_shift(u)`` and ``x_u = (1 - sigma) x0 + sigma eps``. The
    primary modality's schedule additionally supplies the shared clock used to
    place multi-stage teacher evaluations (see :func:`unwarp`).
    """

    shift: float
    max_t: float = PDD_GRID_MAX_T

    def __post_init__(self) -> None:
        if self.shift <= 0:
            raise ValueError(f"shift must be positive, got {self.shift}")
        if not 0.0 < self.max_t <= 1.0:
            raise ValueError(f"max_t must satisfy 0 < max_t <= 1, got {self.max_t}")

    def sigma(self, base: torch.Tensor) -> torch.Tensor:
        return shifted_noise_amount(base, self.shift, max_t=self.max_t)

    def sigma_delta(self, base_start: torch.Tensor, base_end: torch.Tensor) -> torch.Tensor:
        return shifted_noise_delta(base_start, base_end, self.shift, max_t=self.max_t)

    def unwarp(self, sigma: torch.Tensor) -> torch.Tensor:
        return unshifted_noise_amount(sigma, self.shift, max_t=self.max_t)

    def integration_weights(self, fine_grid: torch.Tensor) -> torch.Tensor:
        """``sigma(u_{j+1}) - sigma(u_j)`` for every fine-grid interval ``j``.

        Freezing head ``j`` (``eps - x0``) on its interval advances the state
        by exactly this coefficient. Negative for a descending grid. Callers
        validate the weights once per grid (:func:`validate_finite_nonzero`)
        rather than on every use.
        """
        return self.sigma_delta(fine_grid[:-1], fine_grid[1:])


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
    pdd_steps = validate_pdd_steps(pdd_steps)
    grid = torch.linspace(max_t, 0.0, pdd_steps + 1, dtype=torch.float64, device=device)
    return grid.clamp(max=max_t)


def pdd_student_step_indices(
    source_steps: int,
    target_steps: int,
    *,
    configured_indices: Mapping[int, Sequence[int]] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Fine-grid node indices of one ``target_steps``-interval runtime partition.

    Without an explicit entry in *configured_indices* the balanced subset
    ``arange(n + 1) * N // n`` is used, matching FastGen's
    ``BaseNoiseSchedule._student_step_indices``.
    """
    source_steps = validate_positive_int(source_steps, "source_steps")
    target_steps = validate_positive_int(target_steps, "target_steps")
    if target_steps > source_steps:
        raise ValueError(f"Cannot select {target_steps} steps from a {source_steps}-step PDD grid")

    selected = None if configured_indices is None else configured_indices.get(target_steps)
    if selected is None:
        return torch.arange(target_steps + 1, device=device, dtype=torch.long) * source_steps // target_steps
    return validate_grid_indices(selected, source_steps, target_steps, device=device)


def validate_grid_indices(
    selected: Any,
    source_steps: int,
    target_steps: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Check an explicit ``target_steps``-interval partition of a ``source_steps`` grid."""
    if isinstance(selected, str | bytes):
        raise TypeError("PDD grid indices must be a one-dimensional sequence of integers")
    indices = torch.as_tensor(list(selected) if not torch.is_tensor(selected) else selected, device=device)
    if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
        raise TypeError(f"PDD grid indices must have an integer dtype, got {indices.dtype}")
    indices = indices.to(dtype=torch.long)
    if indices.shape != (target_steps + 1, ):
        raise ValueError(f"PDD grid indices for {target_steps} target steps must have shape "
                         f"({target_steps + 1},), got {tuple(indices.shape)}")
    if int(indices[0]) != 0 or int(indices[-1]) != source_steps:
        raise ValueError(f"PDD grid indices must start at 0 and end at {source_steps}, got {indices.tolist()}")
    if not bool((indices[:-1] < indices[1:]).all()):
        raise ValueError(f"PDD grid indices must be strictly increasing, got {indices.tolist()}")
    return indices


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
        if not 0 <= step < self.num_steps:
            raise ValueError(f"step must be in [0, {self.num_steps - 1}], got {step}")
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

    The output features of a teacher ``ReplicatedLinear`` are repeated
    ``grid_size`` times (one native head prediction per fine-grid interval).
    An optional ``teacher_scale`` multiplies each copied head's parameters by a
    fixed per-node factor, converting the teacher's output convention into the
    heads' native prediction space at their grid times; MiniMax-H3's raw heads
    already emit ``eps - x0`` and use no fold.

    The layer can also return a *fused* output: the normalized weighted average
    of the parameter slices of a contiguous block ``[start, end)`` of grid
    steps, collapsing that block into one effective linear. Fusion is
    activated by the :meth:`fuse` context manager and read by an ordinary
    ``forward(input)`` call, so the architecture's inline final projection
    needs no seam. It is FSDP-safe because the fused parameters are computed
    inside ``forward``, where the full weight is all-gathered.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        grid_size: int,
        channel_layout: str = "channel_major",
        patch_size: int = 1,
        teacher_scale: torch.Tensor | Sequence[float] | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        grid_size = validate_pdd_steps(grid_size, "grid_size")
        if channel_layout not in CHANNEL_LAYOUTS:
            raise ValueError(f"channel_layout must be one of {CHANNEL_LAYOUTS}, got {channel_layout!r}")
        patch_size = validate_positive_int(patch_size, "patch_size")
        if channel_layout == "patch_major" and output_size % patch_size:
            raise ValueError(f"output_size={output_size} is not divisible by patch_size={patch_size}")
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
        self.channel_layout = channel_layout
        # ``patch_major`` groups rows as (patch, grid, C); ``channel_major`` is
        # its patch = 1 case, (grid, rest).
        self.patch_size = patch_size if channel_layout == "patch_major" else 1

        if teacher_scale is None:
            self._teacher_scale: tuple[float, ...] | None = None
        else:
            # Explicit CPU placement keeps the fold real inside a meta-device
            # construction, and a plain tuple keeps it out of state_dict.
            scale = torch.as_tensor(teacher_scale, device="cpu", dtype=torch.float64).detach()
            if scale.shape != (self.grid_size, ):
                raise ValueError(f"teacher_scale must have shape [{self.grid_size}], got {tuple(scale.shape)}")
            validate_finite_nonzero(scale, "teacher_scale")
            self._teacher_scale = tuple(scale.tolist())

        # Transient fusion state is a plain tuple, never a buffer, so it stays
        # out of state_dict and can be restored as one value.
        self._fusion_state: tuple[int, int, torch.Tensor, torch.dtype] | None = None

    # -- teacher parameter widening -------------------------------------------

    def repeat_teacher_parameter(self, tensor: torch.Tensor) -> torch.Tensor:
        """Repeat one single-head parameter in this layer's widened layout."""
        if tensor.shape[0] != self.head_output_size:
            raise ValueError(f"Expected a single-head parameter with {self.head_output_size} output rows, "
                             f"got {tuple(tensor.shape)}")
        # (patch, rest, ...) -> (patch, grid, rest, ...)
        head = tensor.unflatten(0, (self.patch_size, -1)).unsqueeze(1)
        wide = head.expand(-1, self.grid_size, *head.shape[2:])
        if self._teacher_scale is not None:
            scale = torch.tensor(self._teacher_scale, device=tensor.device, dtype=tensor.dtype)
            wide = wide * scale.reshape(1, self.grid_size, *((1, ) * (wide.ndim - 2)))
        return wide.reshape(-1, *tensor.shape[1:])

    def expand_loaded_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Widen a teacher checkpoint tensor; pass a grid-wide PDD tensor through."""
        if tensor.shape[0] == self.output_size:
            return tensor
        if tensor.shape[0] == self.head_output_size:
            logger.info("Repeating %s grid_size=%s times (%s)", name, self.grid_size, self.channel_layout)
            return self.repeat_teacher_parameter(tensor)
        raise ValueError(f"Cannot load {name!r} of shape {tuple(tensor.shape)} into a PDD head with "
                         f"{self.head_output_size} teacher rows x {self.grid_size} heads")

    # -- fusion ---------------------------------------------------------------

    @contextlib.contextmanager
    def fuse(
        self,
        start: int,
        end: int,
        weights: torch.Tensor,
        precision_decoding: torch.dtype,
    ) -> Iterator[PDDReplicatedLinear]:
        """Temporarily make ``forward(input)`` return the fused block output.

        *weights* are this modality's per-node integration weights over the
        whole grid (``[grid_size]``); the block ``[start, end)`` is fused with
        their normalized form, so the layer emits the weighted mean and the
        caller scales by the block's total weight. ``precision_decoding`` is
        used only for the fused-parameter head-axis accumulation; the fused
        parameters return to the layer's dtype before the projection runs. The
        previous fusion state is restored on exit.
        """
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= self.grid_size):
            raise ValueError(f"PDD fusion requires 0 <= start < end <= grid_size, got start={start}, end={end}, "
                             f"grid_size={self.grid_size}")
        if weights.ndim != 1 or weights.shape[0] != self.grid_size:
            raise ValueError(f"fusion weights must have shape [grid_size], got {tuple(weights.shape)}")
        if not weights.is_floating_point():
            raise TypeError(f"fusion weights must be floating point, got {weights.dtype}")
        previous = self._fusion_state
        self._fusion_state = (start, end, weights, precision_decoding)
        try:
            yield self
        finally:
            self._fusion_state = previous

    def _head_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """View a widened parameter as ``[grid, head_features, ...]``."""
        if self.channel_layout == "patch_major":
            # (p n c) ... -> n (p c) ...
            view = tensor.unflatten(0, (self.patch_size, self.grid_size, -1))
            view = view.movedim(1, 0)
            return view.flatten(1, 2)
        return tensor.unflatten(0, (self.grid_size, -1))

    def _fused_params(
        self,
        start: int,
        end: int,
        weights: torch.Tensor,
        precision_decoding: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The ``[O, I]`` fused weight (and bias) of block ``[start, end)``."""
        # Normalize in the incoming scheduler precision, then cast only the
        # fused-parameter accumulation. The layer emits the weighted mean and
        # the caller multiplies by the block total, so together they represent
        # ``sum_j w_j pred_j``.
        block_weights = weights[start:end].to(device=self.weight.device)
        block_scale = block_weights.sum()
        validate_finite_nonzero(block_scale, "PDD fusion block total weight")
        normalized = (block_weights / block_scale).to(dtype=precision_decoding)

        def fuse_parameter(parameter: torch.Tensor) -> torch.Tensor:
            heads = self._head_view(parameter)[start:end].to(dtype=precision_decoding)
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
    missing = sorted(set(linears) - set(integration_weights))
    if missing:
        raise ValueError(f"PDD fusion needs integration weights for every head, missing {missing}")
    with contextlib.ExitStack() as stack:
        for name, linear in linears.items():
            stack.enter_context(linear.fuse(start, end, integration_weights[name], precision_decoding))
        yield


__all__ = [
    "CHANNEL_LAYOUTS",
    "PDD_GRID_MAX_T",
    "PDDModalitySchedule",
    "PDDReplicatedLinear",
    "PDDSamplingPlan",
    "build_pdd_sampling_plan",
    "fuse_pdd_heads",
    "pdd_fine_grid",
    "pdd_student_step_indices",
    "shifted_noise_amount",
    "shifted_noise_delta",
    "unshifted_noise_amount",
    "validate_finite_nonzero",
    "validate_grid_indices",
    "validate_pdd_steps",
    "validate_positive_int",
]
