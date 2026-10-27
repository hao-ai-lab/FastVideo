# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic Parallel Decoding Distillation math on packed latents.

The PDD methods drive one packed ``[B, numel]`` latent per sample whose
modalities occupy contiguous column ranges (:class:`PDDModalityBlock`), each
with its own shifted rectified-flow noise level on one shared base clock. A
model plugin materializes ``pdd_steps`` heads as ``[B, pdd_steps, numel]``
(head ``j`` predicts each modality's native ``eps - x0`` frozen on fine-grid
interval ``j``), and everything below integrates, supervises, and inverts
those heads exactly as FastGen's ``PDDMixin`` (``pdd_mixin.py``) and
``PDDModelMixin`` (``pdd.py``) do, with the per-modality dict dispatch
replaced by column slices.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from fastvideo.layers.pdd import PDDModalitySchedule, validate_pdd_steps

# Single-predecessor explicit Runge-Kutta tableaus on the shared primary clock.
# Each stage ``(fraction, source)`` evaluates the teacher at the shared time
# whose primary sigma is ``sigma_p + fraction * delta_sigma_p`` after an Euler
# step along the ``source``-th previously computed native velocity;
# ``output_weights`` mix all stage velocities into the interval mean.
# ``terminal_fallback_stage`` handles solvers whose trailing ``fraction == 1``
# stage would supervise from a teacher evaluation at ``t = 0``: samples whose
# interval ends on the clean endpoint blend to that earlier stage instead.
RK_STAGES: dict[str, dict[str, Any]] = {
    "euler": {
        "stages": (),
        "output_weights": (1.0, ),
        "terminal_fallback_stage": None,
    },
    "midpoint": {
        "stages": ((0.5, 0), ),
        "output_weights": (0.0, 1.0),
        "terminal_fallback_stage": None,
    },
    "heun": {
        "stages": ((1.0, 0), ),
        "output_weights": (0.5, 0.5),
        "terminal_fallback_stage": 0,
    },
    "kutta4": {
        "stages": ((0.5, 0), (0.5, 1), (1.0, 2)),
        "output_weights": (1 / 6, 2 / 6, 2 / 6, 1 / 6),
        "terminal_fallback_stage": 1,
    },
}


@dataclass(frozen=True)
class PDDModalityBlock:
    """One modality's packed column range and shifted noise schedule."""

    name: str
    columns: slice
    schedule: PDDModalitySchedule


class PDDStudent(Protocol):
    """The contract a model plugin exposes to the PDD methods.

    Latents are packed ``[B, numel]`` tensors; heads are ``[B, heads, numel]``
    in the model's native output dtype, every head or only a contiguous
    ``head_window``. ``pdd_timestep`` maps a base-clock time ``u`` to the
    method timestep that ``predict_noise`` and ``add_noise_for_batch``
    understand; ``pdd_model_dtype`` is the model-boundary dtype of student and
    teacher forwards.
    """

    pdd_steps: int
    pdd_model_dtype: torch.dtype

    def pdd_fine_grid(self, device: torch.device) -> torch.Tensor:
        ...

    def pdd_modality_schedules(self, batch: Any) -> tuple[PDDModalityBlock, ...]:
        ...

    def pdd_timestep(self, base: torch.Tensor) -> torch.Tensor:
        ...

    def predict_pdd_heads(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None,
        attn_kind: str,
        head_window: tuple[int, int] | None,
    ) -> torch.Tensor:
        ...

    def predict_pdd_fused(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        start: int,
        end: int,
        fine_grid: torch.Tensor,
        batch: Any,
        *,
        precision_decoding: torch.dtype,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None,
        attn_kind: str,
    ) -> torch.Tensor:
        ...


_PDD_STUDENT_ATTRIBUTES = (
    "pdd_steps",
    "pdd_model_dtype",
    "pdd_fine_grid",
    "pdd_modality_schedules",
    "pdd_timestep",
    "predict_pdd_heads",
    "predict_pdd_fused",
    "add_noise_for_batch",
    "prepare_batch",
)


def require_pdd_student(model: Any, *, role: str = "student") -> None:
    """Fail fast when a role model does not implement the PDD contract."""
    missing = [name for name in _PDD_STUDENT_ATTRIBUTES if not hasattr(model, name)]
    if missing:
        raise TypeError(f"PDD methods require a PDD-widened {role} model plugin exposing "
                        f"{list(_PDD_STUDENT_ATTRIBUTES)}; {type(model).__name__} lacks {missing}")
    validate_pdd_steps(getattr(model, "pdd_steps", None), f"PDD {role} pdd_steps")


def validate_block_indices(
    step_idx: torch.Tensor,
    end_idx: torch.Tensor,
    pdd_steps: int,
    *,
    allow_empty: bool = False,
    name: str = "end_idx",
    lower: int = 0,
) -> None:
    """Validate batched half-open fine-grid blocks ``lower <= step_idx < end_idx <= pdd_steps``.

    One host synchronization for the whole batch.
    """
    if not torch.is_tensor(step_idx) or step_idx.dtype != torch.long or step_idx.ndim != 1:
        raise TypeError("PDD step indices must be a one-dimensional torch.long tensor")
    if not torch.is_tensor(end_idx) or end_idx.dtype != torch.long or tuple(end_idx.shape) != tuple(step_idx.shape):
        raise ValueError(f"{name} must be a torch.long tensor of shape {tuple(step_idx.shape)}")
    ordered = step_idx <= end_idx if allow_empty else step_idx < end_idx
    if not bool(torch.all((step_idx >= lower) & ordered & (end_idx <= pdd_steps))):
        relation = "<=" if allow_empty else "<"
        raise ValueError(f"Every PDD block must satisfy {lower} <= step_idx {relation} {name} <= {pdd_steps}")


def block_mask(step_idx: torch.Tensor, end_idx: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Boolean ``[B, num_heads]`` mask of each sample's half-open block."""
    grid = torch.arange(num_heads, device=step_idx.device)[None]
    return (grid >= step_idx[:, None]) & (grid < end_idx[:, None])


def _in_precision(precision: torch.dtype | None, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tensors if precision is None else tuple(tensor.to(precision) for tensor in tensors)


def advance_block(
    state: torch.Tensor,
    pred: torch.Tensor,
    step_idx: torch.Tensor,
    end_idx: torch.Tensor,
    fine_grid: torch.Tensor,
    blocks: Sequence[PDDModalityBlock],
    *,
    allow_empty: bool = False,
    precision_decoding: torch.dtype | None = None,
    head_offset: int = 0,
) -> torch.Tensor:
    """Advance one rectified-flow block from materialized heads.

    Heads inside each sample's ``[step_idx, end_idx)`` block are reduced to
    their weighted mean sigma velocity per modality and applied with the
    block's total sigma increment. Equal start/end rows are allowed only with
    ``allow_empty`` and produce an identity update. ``precision_decoding``
    scopes the block arithmetic: state and prediction are promoted for the
    integration and the advanced state is cast back to the incoming dtype.
    ``pred`` may hold only the heads ``[head_offset, head_offset + pred.shape[1])``
    of the grid; every block must lie inside that window.
    """
    input_dtype = state.dtype
    state, pred = _in_precision(precision_decoding, state, pred)
    if state.dtype != pred.dtype:
        raise TypeError(f"PDD state and prediction must share a dtype, got {state.dtype} and {pred.dtype}")
    if pred.ndim != 3 or state.ndim != 2 or pred.shape[0] != state.shape[0] or pred.shape[2] != state.shape[1]:
        raise ValueError("Materialized PDD prediction must have shape [batch, heads, numel] matching the "
                         f"[batch, numel] state, got {tuple(pred.shape)} and {tuple(state.shape)}")
    num_heads = int(pred.shape[1])
    pdd_steps = int(fine_grid.numel()) - 1
    if head_offset < 0 or head_offset + num_heads > pdd_steps:
        raise ValueError(f"PDD heads [{head_offset}, {head_offset + num_heads}) fall outside the "
                         f"{pdd_steps}-interval fine grid")
    validate_block_indices(step_idx, end_idx, head_offset + num_heads, allow_empty=allow_empty, lower=head_offset)

    mask = block_mask(step_idx - head_offset, end_idx - head_offset, num_heads)
    # Legitimately empty rows (identity update) are exempt from normalization.
    row_scale_guard = (end_idx > step_idx)
    advanced = state.clone()
    for block in blocks:
        weights = block.schedule.integration_weights(fine_grid)[head_offset:head_offset + num_heads]
        block_weights = torch.where(mask, weights.to(device=mask.device)[None], torch.zeros_like(weights[:1]))
        block_scale = block_weights.sum(dim=-1)
        safe_scale = torch.where(row_scale_guard, block_scale, torch.ones_like(block_scale))
        normalized = (block_weights / safe_scale[:, None]).to(dtype=pred.dtype)
        mean_prediction = torch.einsum("bn,bnk->bk", normalized, pred[:, :, block.columns])
        state_scale = block_scale.to(device=state.device, dtype=state.dtype)
        advanced[:, block.columns] = state[:, block.columns] + mean_prediction * state_scale[:, None]
    return advanced if precision_decoding is None else advanced.to(input_dtype)


def advance_fused(
    state: torch.Tensor,
    mean_prediction: torch.Tensor,
    start: int,
    end: int,
    fine_grid: torch.Tensor,
    blocks: Sequence[PDDModalityBlock],
    *,
    precision_decoding: torch.dtype | None = None,
) -> torch.Tensor:
    """Advance one shared block ``[start, end)`` from its fused mean velocity ``[B, numel]``."""
    input_dtype = state.dtype
    state, mean_prediction = _in_precision(precision_decoding, state, mean_prediction)
    if state.dtype != mean_prediction.dtype:
        raise TypeError("PDD state and fused prediction must share a dtype, "
                        f"got {state.dtype} and {mean_prediction.dtype}")
    pdd_steps = int(fine_grid.numel()) - 1
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= pdd_steps):
        raise ValueError(f"PDD block indices must satisfy 0 <= start < end <= {pdd_steps}, got {start=} and {end=}")
    advanced = state.clone()
    for block in blocks:
        scale = block.schedule.integration_weights(fine_grid)[start:end].sum()
        state_scale = scale.to(device=state.device, dtype=state.dtype)
        advanced[:, block.columns] = state[:, block.columns] + mean_prediction[:, block.columns] * state_scale
    return advanced if precision_decoding is None else advanced.to(input_dtype)


def carried_step_count(pdd_steps: int, block_size_min: int) -> int:
    """Highest PDD NFE implied by the minimum training block size (last stride may be partial)."""
    return (pdd_steps + block_size_min - 1) // block_size_min


def aligned_block_starts(
    batch_size: int,
    *,
    pdd_steps: int,
    block_size_min: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample fine-grid block starts aligned to ``block_size_min``."""
    return block_size_min * torch.randint(
        carried_step_count(pdd_steps, block_size_min),
        (batch_size, ),
        device=device,
        dtype=torch.long,
        generator=generator,
    )


def sample_supervision_index(
    step_idx: torch.Tensor,
    *,
    pdd_steps: int,
    block_size_max: int,
    block_end_idx: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the fine-grid head supervised by the PDD loss.

    Without ``block_end_idx`` the head is drawn uniformly from the
    ``block_size_max`` window after each start; with it (exclusive) the head
    is confined to that rollout segment.
    """
    if block_end_idx is None:
        max_idx = torch.clamp(step_idx + block_size_max - 1, max=pdd_steps - 1)
    else:
        validate_block_indices(step_idx, block_end_idx, pdd_steps, name="block_end_idx")
        max_idx = block_end_idx - 1
    mask = block_mask(step_idx, max_idx + 1, pdd_steps)
    return torch.multinomial(mask.float(), 1, replacement=False, generator=generator).squeeze(1)


def teacher_block_target(
    x_t: torch.Tensor,
    step_idx: torch.Tensor,
    fine_grid: torch.Tensor,
    blocks: Sequence[PDDModalityBlock],
    *,
    solver: str,
    predict_teacher: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """The teacher's mean native sigma velocity over one fine-grid interval.

    ``predict_teacher(state, base)`` returns every modality's ``eps - x0`` at
    base time ``base`` (guidance and the model/decoding precision boundary are
    the caller's). Multi-stage solvers place their stages at shared times
    chosen by the primary (first) modality's clock, while every modality
    advances by its own exact sigma increment per stage, so the mean is native
    by construction. Samples whose interval ends on the clean endpoint blend
    to the solver's terminal fallback stage: the trailing stage still runs, so
    every rank issues identical forwards, but a teacher evaluation at
    ``t = 0`` never supervises.
    """
    if solver not in RK_STAGES:
        raise ValueError(f"RK type: {solver} is undefined")
    tableau = RK_STAGES[solver]
    stages, output_weights = tableau["stages"], tableau["output_weights"]
    primary = blocks[0].schedule
    validate_block_indices(step_idx, step_idx + 1, int(fine_grid.numel()) - 1, name="step_idx + 1")

    base = fine_grid[step_idx]
    base_next = fine_grid[step_idx + 1]
    delta_sigma = primary.sigma_delta(base, base_next)

    def stage_time(fraction: float) -> torch.Tensor:
        if fraction == 1.0:
            return base_next
        if primary.shift == 1.0:
            return base + delta_sigma * fraction
        stage_sigma = primary.sigma(base) + delta_sigma * fraction
        return primary.unwarp(stage_sigma).clamp(min=0.0, max=primary.max_t)

    velocities = [predict_teacher(x_t, base)]
    for fraction, source in stages:
        time = stage_time(fraction)
        velocity = velocities[source]
        stage_state = x_t.to(dtype=velocity.dtype).clone()
        for block in blocks:
            stage_delta = block.schedule.sigma_delta(base, time).to(device=stage_state.device, dtype=stage_state.dtype)
            stage_state[:,
                        block.columns] = stage_state[:,
                                                     block.columns] + velocity[:, block.columns] * stage_delta[:, None]
        velocities.append(predict_teacher(stage_state, time))

    mean_velocity = sum(weight * value for weight, value in zip(output_weights, velocities, strict=True))
    fallback_stage = tableau["terminal_fallback_stage"]
    if fallback_stage is None:
        return mean_velocity
    # The grid is descending; the terminal interval ends at ``fine_grid[-1]``.
    mask_last_step = (base_next <= fine_grid[-1])[:, None]
    return torch.where(mask_last_step, velocities[fallback_stage], mean_velocity)


def latent_loss_valid_mask(batch: Any) -> torch.Tensor | None:
    """Scalar rollout validity, never the differently sized joint-token mask.

    Fixed transformer capacities leave the rollout real-only. An adapter may
    also expose trailing *scalar* padding through its DMD latent layout.
    Models without that layout retain their existing unmasked loss path.
    """
    layout = getattr(batch, "minimax_h3_dmd_layout", None)
    if layout is None or getattr(layout, "pad_tokens", None) == 0:
        # Joint capacity normally leaves millions of scalar rollout columns
        # real-only. Their layout certifies all-valid; do not allocate/scan a
        # large CPU mask at every teacher/student loss in that common case.
        return None
    valid = getattr(layout, "valid", None)
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.ndim != 1:
        raise ValueError("Latent loss validity must be a one-dimensional torch.bool scalar-column mask")
    return valid


def validate_latent_loss_regions(valid: torch.Tensor | None, regions: Sequence[slice]) -> None:
    """Named modality/document regions must not claim invalid scalar columns."""
    if valid is not None:
        for region in regions:
            region_valid = valid[region]
            if region_valid.numel() == 0 or not bool(region_valid.all()):
                raise ValueError("Latent loss regions must be non-empty and exclude invalid padding columns")


def latent_loss_region(
    tensor: torch.Tensor,
    columns: slice,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    """Select valid scalars *before* residuals, squares, or normalizers.

    Multiplying a squared error by zero is not sufficient: NaN/Inf padding
    would still contaminate both the loss and its gradients. Preserve the
    original slice (and reduction order) when it is entirely real.
    """
    if valid is None:
        return tensor[:, columns]
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.ndim != 1:
        raise ValueError("Latent loss validity must be a one-dimensional torch.bool scalar-column mask")
    if tensor.ndim != 2 or tensor.shape[1] != valid.numel():
        raise ValueError("Latent loss validity must match the [batch, scalar columns] tensor, not joint tokens")
    region_valid = valid[columns]
    if region_valid.numel() == 0:
        raise ValueError("A latent loss region must contain at least one valid scalar column")
    region = tensor[:, columns]
    if bool(region_valid.all()):
        return region
    if not bool(region_valid.any()):
        raise ValueError("A latent loss region must contain at least one valid scalar column")
    return region[:, region_valid.to(device=tensor.device)]


def head_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    blocks: Sequence[PDDModalityBlock],
    *,
    valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-modality MSE between one student head and its teacher target (>= FP32).

    ``valid`` indexes scalar rollout columns, not joint transformer tokens.
    Padding is selected out before arithmetic and never changes the modality
    normalizers. Without a mask, the established block-slice path is unchanged.
    """
    dtype = torch.promote_types(prediction.dtype, torch.float32)
    return {
        block.name:
        F.mse_loss(
            latent_loss_region(prediction, block.columns, valid).to(dtype),
            latent_loss_region(target, block.columns, valid).to(dtype),
        )
        for block in blocks
    }


@dataclass
class PDDHeadLoss:
    """One supervised head: its loss terms and the point it was supervised at."""

    total: torch.Tensor
    per_modality: dict[str, torch.Tensor]
    # The state at the supervised head's node, the head's grid index, and the
    # student's prediction there (for an x0 preview without another inversion).
    x_k: torch.Tensor
    k: torch.Tensor
    p_k: torch.Tensor


def x0_from_local_head(
    x_t: torch.Tensor,
    step_idx: torch.Tensor,
    fine_grid: torch.Tensor,
    blocks: Sequence[PDDModalityBlock],
    prediction: torch.Tensor,
) -> torch.Tensor:
    """Invert one head: ``x0 = x_t - sigma_m(u) * (eps - x0)`` per modality."""
    original_dtype = x_t.dtype
    base = fine_grid[step_idx]
    x0 = x_t.to(torch.float64).clone()
    pred64 = prediction.to(torch.float64)
    for block in blocks:
        sigma = block.schedule.sigma(base).to(device=x0.device)[:, None]
        x0[:, block.columns] = x0[:, block.columns] - sigma * pred64[:, block.columns]
    return x0.to(original_dtype)


def x0_from_points(
    x_start: torch.Tensor,
    x_end: torch.Tensor,
    start_idx: torch.Tensor,
    end_idx: torch.Tensor,
    fine_grid: torch.Tensor,
    blocks: Sequence[PDDModalityBlock],
) -> torch.Tensor:
    """Invert an integrated endpoint pair through each modality's chord.

    The conditional path is affine in sigma, so
    ``x0 = x_start - sigma_start * (x_end - x_start) / (sigma_end - sigma_start)``;
    the clean endpoint is returned exactly where ``sigma_end == 0``.
    """
    original_dtype = x_start.dtype
    validate_block_indices(start_idx, end_idx, int(fine_grid.numel()) - 1)
    base_start = fine_grid[start_idx]
    base_end = fine_grid[end_idx]
    start64 = x_start.to(torch.float64)
    end64 = x_end.to(torch.float64)
    x0 = start64.clone()
    for block in blocks:
        schedule = block.schedule
        delta = schedule.sigma_delta(base_start, base_end).to(device=x0.device)[:, None]
        sigma_start = schedule.sigma(base_start).to(device=x0.device)[:, None]
        sigma_end = schedule.sigma(base_end).to(device=x0.device)[:, None]
        columns = block.columns
        chord = start64[:, columns] - sigma_start * (end64[:, columns] - start64[:, columns]) / delta
        x0[:, columns] = torch.where(sigma_end == 0, end64[:, columns], chord)
    return x0.to(original_dtype)


__all__ = [
    "PDDHeadLoss",
    "PDDModalityBlock",
    "PDDStudent",
    "RK_STAGES",
    "advance_block",
    "advance_fused",
    "aligned_block_starts",
    "block_mask",
    "carried_step_count",
    "head_losses",
    "latent_loss_region",
    "latent_loss_valid_mask",
    "require_pdd_student",
    "sample_supervision_index",
    "teacher_block_target",
    "validate_block_indices",
    "validate_latent_loss_regions",
    "x0_from_local_head",
    "x0_from_points",
]
