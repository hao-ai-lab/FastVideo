# SPDX-License-Identifier: Apache-2.0
"""Pure utilities for reward-enhanced H3 trajectory distillation.

The functions in this module deliberately have no FastVideo model dependency so
that the REST/AMD mathematics and cache fingerprints can be audited on CPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import torch


def normalize_reward_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Return non-negative reward weights normalized to sum to one."""
    if not weights:
        raise ValueError("reward weights must be nonempty")
    normalized: dict[str, float] = {}
    total = 0.0
    for raw_name, raw_weight in weights.items():
        name = str(raw_name).strip().lower()
        if not name:
            raise ValueError("reward names must be nonempty")
        weight = float(raw_weight)
        if not torch.isfinite(torch.tensor(weight)):
            raise ValueError(f"reward weight for {name!r} must be finite")
        if weight < 0.0:
            raise ValueError(f"reward weight for {name!r} must be nonnegative")
        normalized[name] = weight
        total += weight
    if total <= 0.0:
        raise ValueError("at least one reward weight must be positive")
    return {name: weight / total for name, weight in normalized.items()}


def group_relative_advantages(
    reward_scores: Mapping[str, torch.Tensor],
    reward_weights: Mapping[str, float],
    *,
    eps: float = 1e-6,
    clip: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute REST/AMD group-normalized advantages for one prompt.

    Every reward source is standardized independently across the ``K`` teacher
    rollouts, clipped to ``[-clip, clip]``, and only then mixed using normalized
    non-negative weights. Constant reward components contribute exactly zero.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if clip <= 0.0:
        raise ValueError("clip must be positive")
    weights = normalize_reward_weights(reward_weights)
    missing = sorted(set(weights) - set(reward_scores))
    if missing:
        raise ValueError(f"missing reward scores for {missing}")

    expected_k: int | None = None
    per_reward: dict[str, torch.Tensor] = {}
    for name in weights:
        scores = torch.as_tensor(reward_scores[name], dtype=torch.float32)
        if scores.ndim != 1:
            raise ValueError(f"reward {name!r} must have shape [K], got {tuple(scores.shape)}")
        if scores.numel() < 2:
            raise ValueError("REST/AMD requires at least two rollouts per prompt")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError(f"reward {name!r} contains non-finite values")
        if expected_k is None:
            expected_k = int(scores.numel())
        elif scores.numel() != expected_k:
            raise ValueError("all reward components must have the same K")

        centered = scores - scores.mean()
        std = centered.square().mean().sqrt()
        if float(std) <= eps:
            advantage = torch.zeros_like(scores)
        else:
            advantage = (centered / std).clamp(-clip, clip)
        per_reward[name] = advantage

    assert expected_k is not None
    mixed = torch.zeros(expected_k, dtype=torch.float32)
    for name, weight in weights.items():
        mixed = mixed + float(weight) * per_reward[name]
    return mixed, per_reward


def amd_coefficients(
    mixed_advantage: torch.Tensor,
    *,
    scale: float = 1.0,
    bias: float = 0.5,
) -> torch.Tensor:
    """Apply the REST affine modulation ``lambda * (A_mix + b)``."""
    if scale < 0.0:
        raise ValueError("AMD scale must be nonnegative")
    if bias < 0.0:
        raise ValueError("AMD bias must be nonnegative")
    advantage = torch.as_tensor(mixed_advantage, dtype=torch.float32)
    if not bool(torch.isfinite(advantage).all()):
        raise ValueError("mixed advantages must be finite")
    return float(scale) * (advantage + float(bias))


def signed_loss_surrogate(
    per_sample_loss: torch.Tensor,
    coefficient: torch.Tensor | float,
) -> torch.Tensor:
    """Return a non-negative scalar whose gradient is signed AMD regression.

    A literal negative-weight MSE makes logged losses negative and can confuse
    generic trainer checks. This detached-value surrogate has value
    ``|c| * loss`` but derivative ``c * d(loss)/d(theta)``.
    """
    losses = torch.as_tensor(per_sample_loss)
    coefficients = torch.as_tensor(coefficient, device=losses.device, dtype=losses.dtype)
    try:
        coefficients = torch.broadcast_to(coefficients, losses.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"coefficient shape {tuple(coefficients.shape)} cannot broadcast to loss shape {tuple(losses.shape)}"
        ) from exc
    return (coefficients.abs() * losses.detach() + coefficients * (losses - losses.detach())).mean()


def segment_velocity_target(
    current: torch.Tensor,
    next_state: torch.Tensor,
    sigma_current: torch.Tensor | float,
    sigma_next: torch.Tensor | float,
) -> torch.Tensor:
    """Compute a teacher segment slope in the modality's shifted H3 sigma."""
    if current.shape != next_state.shape:
        raise ValueError(f"segment states must match, got {tuple(current.shape)} and {tuple(next_state.shape)}")
    sigma0 = torch.as_tensor(sigma_current, device=current.device, dtype=torch.float32)
    sigma1 = torch.as_tensor(sigma_next, device=current.device, dtype=torch.float32)
    delta = sigma1 - sigma0
    if bool((delta.abs() <= torch.finfo(torch.float32).eps).any()):
        raise ValueError("segment sigma endpoints must be distinct")
    while delta.ndim < current.ndim:
        delta = delta.unsqueeze(-1)
    return ((next_state.float() - current.float()) / delta).to(current.dtype)


def build_piecewise_teacher_schedule(
    student_timesteps: Sequence[float | int],
    *,
    substeps_per_segment: int,
) -> tuple[float, ...]:
    """Build a dense schedule that contains every deployed student boundary."""
    anchors = tuple(float(value) for value in student_timesteps)
    if len(anchors) < 2:
        raise ValueError("student_timesteps must contain at least two boundaries")
    if any(left <= right for left, right in zip(anchors, anchors[1:], strict=True)):
        raise ValueError("student_timesteps must be strictly decreasing")
    if anchors[-1] != 0.0:
        raise ValueError("student_timesteps must end at terminal zero")
    if substeps_per_segment < 1:
        raise ValueError("substeps_per_segment must be at least one")

    schedule: list[float] = []
    for start, end in zip(anchors, anchors[1:], strict=True):
        width = end - start
        for index in range(substeps_per_segment):
            schedule.append(start + width * (index / substeps_per_segment))
    schedule.append(anchors[-1])
    return tuple(schedule)


def teacher_anchor_indices(num_student_segments: int, substeps_per_segment: int) -> tuple[int, ...]:
    """Return dense-trajectory indices corresponding to student boundaries."""
    if num_student_segments < 1:
        raise ValueError("num_student_segments must be positive")
    if substeps_per_segment < 1:
        raise ValueError("substeps_per_segment must be positive")
    return tuple(index * substeps_per_segment for index in range(num_student_segments + 1))


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    """SHA-256 of a JSON-compatible mapping with deterministic formatting."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
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
