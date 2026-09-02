# SPDX-License-Identifier: Apache-2.0
"""Pure utilities for H3 reward-enhanced scored-trajectory distillation.

The functions in this module intentionally contain no FastVideo runtime or
model dependencies.  Cache builders and the training method share this exact
math so the offline reward contract cannot drift from the optimizer contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def normalize_reward_weights(
    reward_weights: Mapping[str, float],
) -> dict[str, float]:
    """Return finite nonnegative reward weights normalized to sum to one."""
    normalized: dict[str, float] = {}
    for raw_name, raw_weight in reward_weights.items():
        name = str(raw_name).strip().lower()
        if not name:
            raise ValueError("Reward names must be nonempty")
        if name in normalized:
            raise ValueError(f"Duplicate canonical reward name: {name!r}")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"Reward weight for {name!r} must be finite and nonnegative, got {raw_weight!r}"
            )
        normalized[name] = weight

    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("At least one reward weight must be positive")
    return {name: weight / total for name, weight in normalized.items()}


def group_relative_advantages(
    reward_scores: Mapping[str, torch.Tensor],
    reward_weights: Mapping[str, float],
    *,
    eps: float = 1e-6,
    clip: float = 1.0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Compute REST per-reward group advantages and their weighted fusion.

    For each reward source ``j`` and candidate ``i`` in one prompt group,

    ``A[j, i] = clip((r[j, i] - mean_j) / (std_j + eps), -clip, clip)``.

    Constant reward components contribute exactly zero. The returned mixed
    advantage is a convex combination using normalized nonnegative weights.
    Population standard deviation is used because the candidates are the full
    group whose relative scores define the update.
    """
    if eps <= 0.0 or not math.isfinite(float(eps)):
        raise ValueError(f"eps must be finite and positive, got {eps!r}")
    if clip <= 0.0 or not math.isfinite(float(clip)):
        raise ValueError(f"clip must be finite and positive, got {clip!r}")

    weights = normalize_reward_weights(reward_weights)
    canonical_scores: dict[str, torch.Tensor] = {}
    for raw_name, scores in reward_scores.items():
        name = str(raw_name).strip().lower()
        if not name:
            raise ValueError("Reward score names must be nonempty")
        if name in canonical_scores:
            raise ValueError(f"Duplicate canonical reward score name: {name!r}")
        canonical_scores[name] = scores

    missing = sorted(set(weights) - set(canonical_scores))
    extra = sorted(set(canonical_scores) - set(weights))
    if missing or extra:
        raise ValueError(
            "Reward score/weight keys must match exactly: "
            f"missing_scores={missing}, unexpected_scores={extra}"
        )

    expected_shape: tuple[int, ...] | None = None
    expected_device: torch.device | None = None
    advantages: dict[str, torch.Tensor] = {}
    mixed: torch.Tensor | None = None
    for name, weight in weights.items():
        scores = torch.as_tensor(canonical_scores[name]).detach().float()
        if scores.ndim != 1 or scores.numel() < 2:
            raise ValueError(
                f"Reward {name!r} must have shape [K] with K >= 2, got {tuple(scores.shape)}"
            )
        if not bool(torch.isfinite(scores).all()):
            raise ValueError(f"Reward {name!r} contains NaN or Inf")
        if expected_shape is None:
            expected_shape = tuple(scores.shape)
            expected_device = scores.device
        elif tuple(scores.shape) != expected_shape:
            raise ValueError(
                f"All reward vectors must share one shape: expected={expected_shape}, "
                f"got {name}={tuple(scores.shape)}"
            )
        elif scores.device != expected_device:
            raise ValueError(
                f"All reward vectors must share one device: expected={expected_device}, "
                f"got {name}={scores.device}"
            )

        centered = scores - scores.mean()
        std = scores.std(unbiased=False)
        if float(std) <= float(eps):
            advantage = torch.zeros_like(scores)
        else:
            advantage = (centered / (std + float(eps))).clamp(-float(clip), float(clip))
        advantages[name] = advantage
        weighted = advantage * float(weight)
        mixed = weighted if mixed is None else mixed + weighted

    assert mixed is not None
    return advantages, mixed


def amd_coefficients(
    mixed_advantage: torch.Tensor,
    *,
    scale: float = 1.0,
    bias: float = 0.5,
    clip: float | None = None,
) -> torch.Tensor:
    """Return REST's signed advantage-modulated distillation coefficient.

    The paper uses ``lambda * (A + b)`` with default ``lambda=1`` and
    ``b=0.5``. An optional symmetric coefficient clip is an explicit
    numerical-safety ablation; ``None`` preserves the published expression.
    """
    scale = float(scale)
    bias = float(bias)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(f"scale must be finite and nonnegative, got {scale!r}")
    if not math.isfinite(bias):
        raise ValueError(f"bias must be finite, got {bias!r}")
    value = torch.as_tensor(mixed_advantage).float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("mixed_advantage contains NaN or Inf")
    coefficient = scale * (value + bias)
    if clip is not None:
        clip = float(clip)
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError(f"clip must be finite and positive, got {clip!r}")
        coefficient = coefficient.clamp(-clip, clip)
    return coefficient


def signed_loss_surrogate(
    per_sample_loss: torch.Tensor,
    coefficient: torch.Tensor,
) -> torch.Tensor:
    """Keep a nonnegative forward value while preserving a signed gradient.

    The returned scalar has forward value ``mean(abs(c) * L)`` and derivative
    ``mean(c * dL)``. This avoids exposing optimizers/loggers to an unbounded
    negative scalar loss while producing the exact REST/AMD signed update.
    ``coefficient`` is always detached: rewards never receive gradients.
    """
    if per_sample_loss.ndim == 0:
        per_sample_loss = per_sample_loss.reshape(1)
    coefficient = torch.as_tensor(
        coefficient,
        device=per_sample_loss.device,
        dtype=per_sample_loss.dtype,
    ).detach()
    if coefficient.ndim == 0:
        coefficient = coefficient.reshape(1)
    try:
        coefficient = torch.broadcast_to(coefficient, per_sample_loss.shape)
    except RuntimeError as exc:
        raise ValueError(
            "coefficient must broadcast to per_sample_loss: "
            f"loss={tuple(per_sample_loss.shape)}, coefficient={tuple(coefficient.shape)}"
        ) from exc
    if not bool(torch.isfinite(per_sample_loss).all()):
        raise ValueError("per_sample_loss contains NaN or Inf")
    if not bool(torch.isfinite(coefficient).all()):
        raise ValueError("coefficient contains NaN or Inf")

    signed = coefficient * per_sample_loss
    forward_value = coefficient.abs() * per_sample_loss.detach()
    return (signed + (forward_value - signed.detach())).mean()


def segment_velocity_target(
    current_state: torch.Tensor,
    next_state: torch.Tensor,
    sigma_current: torch.Tensor | float,
    sigma_next: torch.Tensor | float,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert two trajectory anchors into a finite-difference flow target.

    ``v = (x_next - x_current) / (sigma_next - sigma_current)``.

    H3 video and audio use different shifted sigma schedules, so callers must
    invoke this function separately for each modality with that modality's
    sigma pair. Using the raw base-timestep delta is incorrect.
    """
    if current_state.shape != next_state.shape:
        raise ValueError(
            "Trajectory anchors must share a shape, got "
            f"{tuple(current_state.shape)} and {tuple(next_state.shape)}"
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps!r}")
    sigma_current = torch.as_tensor(
        sigma_current,
        device=current_state.device,
        dtype=torch.float32,
    )
    sigma_next = torch.as_tensor(
        sigma_next,
        device=current_state.device,
        dtype=torch.float32,
    )
    delta = sigma_next - sigma_current
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("Sigma delta contains NaN or Inf")
    if bool((delta.abs() <= float(eps)).any()):
        raise ValueError(
            "Trajectory segment has a zero/degenerate sigma interval: "
            f"sigma_current={sigma_current}, sigma_next={sigma_next}"
        )
    while delta.ndim < current_state.ndim:
        delta = delta.unsqueeze(-1)
    return (next_state.float() - current_state.float()) / delta


def build_piecewise_teacher_schedule(
    student_timesteps: Sequence[int | float],
    substeps_per_segment: int,
) -> tuple[float, ...]:
    """Build a dense schedule whose boundaries exactly match the student grid."""
    anchors = tuple(float(value) for value in student_timesteps)
    if len(anchors) < 2:
        raise ValueError("student_timesteps must contain at least two boundaries")
    if substeps_per_segment <= 0:
        raise ValueError("substeps_per_segment must be positive")
    if any(not math.isfinite(value) for value in anchors):
        raise ValueError("student_timesteps must be finite")
    if any(
        left <= right
        for left, right in zip(anchors[:-1], anchors[1:], strict=True)
    ):
        raise ValueError(
            "student_timesteps must be strictly descending, got "
            f"{list(student_timesteps)}"
        )

    schedule: list[float] = [anchors[0]]
    for segment_index in range(len(anchors) - 1):
        start = anchors[segment_index]
        end = anchors[segment_index + 1]
        for substep in range(1, substeps_per_segment + 1):
            fraction = substep / substeps_per_segment
            schedule.append(start + fraction * (end - start))
    return tuple(schedule)


def teacher_anchor_indices(
    num_student_segments: int,
    substeps_per_segment: int,
) -> tuple[int, ...]:
    """Indices of student boundaries in a piecewise dense teacher schedule."""
    if num_student_segments <= 0:
        raise ValueError("num_student_segments must be positive")
    if substeps_per_segment <= 0:
        raise ValueError("substeps_per_segment must be positive")
    return tuple(
        index * substeps_per_segment for index in range(num_student_segments + 1)
    )


def canonical_json_hash(payload: Any) -> str:
    """Return SHA-256 over deterministic UTF-8 JSON for provenance locks."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "amd_coefficients",
    "build_piecewise_teacher_schedule",
    "canonical_json_hash",
    "group_relative_advantages",
    "normalize_reward_weights",
    "segment_velocity_target",
    "signed_loss_surrogate",
    "teacher_anchor_indices",
]
