# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from fastvideo.train.methods.knowledge_distillation.h3_rest_utils import (
    amd_coefficients,
    build_piecewise_teacher_schedule,
    canonical_json_hash,
    group_relative_advantages,
    normalize_reward_weights,
    segment_velocity_target,
    signed_loss_surrogate,
    teacher_anchor_indices,
)


def test_advantages_standardize_each_reward_before_mixing() -> None:
    mixed, components = group_relative_advantages(
        {
            "alignment": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "motion": torch.tensor([40.0, 30.0, 20.0, 10.0]),
        },
        {"alignment": 3.0, "motion": 1.0},
    )
    assert torch.allclose(components["alignment"].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(components["motion"].mean(), torch.tensor(0.0), atol=1e-6)
    expected = 0.75 * components["alignment"] + 0.25 * components["motion"]
    assert torch.allclose(mixed, expected)
    assert mixed.shape == (4,)


def test_constant_reward_component_contributes_zero() -> None:
    mixed, components = group_relative_advantages(
        {"constant": torch.ones(8), "signal": torch.arange(8, dtype=torch.float32)},
        {"constant": 0.4, "signal": 0.6},
    )
    assert torch.equal(components["constant"], torch.zeros(8))
    assert torch.allclose(mixed, 0.6 * components["signal"])


def test_amd_defaults_match_rest_affine_transform() -> None:
    advantage = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.equal(amd_coefficients(advantage), torch.tensor([-0.5, 0.5, 1.5]))


def test_signed_surrogate_has_nonnegative_value_and_signed_gradient() -> None:
    parameter = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    coefficients = torch.tensor([-0.5, 0.5, 1.5])
    per_sample = parameter.square()
    surrogate = signed_loss_surrogate(per_sample, coefficients)
    surrogate.backward()

    assert float(surrogate) >= 0.0
    expected = coefficients * (2.0 * parameter.detach()) / parameter.numel()
    assert torch.allclose(parameter.grad, expected)


def test_segment_target_uses_shifted_sigma_not_raw_timestep() -> None:
    velocity = torch.tensor([[2.0, -3.0]])
    sigma_current = torch.tensor([0.8])
    sigma_next = torch.tensor([0.3])
    current = torch.tensor([[4.0, 5.0]])
    next_state = current + (sigma_next - sigma_current).view(1, 1) * velocity
    recovered = segment_velocity_target(current, next_state, sigma_current, sigma_next)
    assert torch.allclose(recovered, velocity)


def test_dense_teacher_schedule_contains_exact_four_step_boundaries() -> None:
    anchors = (1000, 750, 500, 250, 0)
    schedule = build_piecewise_teacher_schedule(anchors, substeps_per_segment=12)
    indices = teacher_anchor_indices(4, 12)
    assert len(schedule) == 49
    assert tuple(schedule[index] for index in indices) == tuple(float(x) for x in anchors)
    assert all(left > right for left, right in zip(schedule, schedule[1:], strict=True))


def test_reward_weights_are_normalized_and_validated() -> None:
    assert normalize_reward_weights({"a": 3.0, "b": 1.0}) == {"a": 0.75, "b": 0.25}
    with pytest.raises(ValueError, match="nonnegative"):
        normalize_reward_weights({"a": -1.0})
    with pytest.raises(ValueError, match="positive"):
        normalize_reward_weights({"a": 0.0})


def test_canonical_hash_is_order_invariant_but_content_sensitive() -> None:
    assert canonical_json_hash({"a": 1, "b": [2, 3]}) == canonical_json_hash({"b": [2, 3], "a": 1})
    assert canonical_json_hash({"a": 1}) != canonical_json_hash({"a": 2})
