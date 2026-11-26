# SPDX-License-Identifier: Apache-2.0
"""Group-relative advantage computation tests."""

from __future__ import annotations

import pytest
import torch

from fastvideo.train.methods.rl.promptrl.advantages import (
    group_relative_advantages,
    route_generator_advantages,
    route_refiner_advantages,
)


class TestGroupRelativeAdvantages:
    def test_normalized_within_group(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        keys = ["g"] * 4
        adv = group_relative_advantages(rewards, keys)
        assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(adv.std(unbiased=False), torch.tensor(1.0), atol=1e-6)
        assert adv[0] < 0 < adv[-1]

    def test_zero_variance_group_gives_zero_advantages(self):
        rewards = torch.tensor([2.5, 2.5, 2.5, 2.5])
        adv = group_relative_advantages(rewards, ["g"] * 4)
        assert torch.equal(adv, torch.zeros(4))

    def test_reward_tag_grouping_keeps_basins_separate(self):
        # Same group id prefix, different reward tags: each tag forms its
        # own normalization basin even when value scales differ.
        rewards = torch.tensor([0.0, 1.0, 100.0, 200.0])
        keys = ["g|tagA", "g|tagA", "g|tagB", "g|tagB"]
        adv = group_relative_advantages(rewards, keys)
        assert torch.allclose(adv, torch.tensor([-1.0, 1.0, -1.0, 1.0]))

    def test_mixed_zero_and_nonzero_variance_groups(self):
        rewards = torch.tensor([5.0, 5.0, 1.0, 3.0])
        keys = ["flat", "flat", "varied", "varied"]
        adv = group_relative_advantages(rewards, keys)
        assert adv[0] == 0 and adv[1] == 0
        assert torch.allclose(adv[2:], torch.tensor([-1.0, 1.0]))

    def test_shape_and_length_validation(self):
        with pytest.raises(ValueError):
            group_relative_advantages(torch.zeros(2, 2), ["a", "b"])
        with pytest.raises(ValueError):
            group_relative_advantages(torch.zeros(3), ["a"])

    def test_advantages_are_detached(self):
        rewards = torch.tensor([1.0, 2.0], requires_grad=True)
        adv = group_relative_advantages(rewards, ["g", "g"])
        assert not adv.requires_grad


class TestAdvantageRouting:
    def test_refiner_route_zeroes_retained_originals(self):
        adv = torch.tensor([0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75])
        participation = [False, False] + [True] * 6
        routed = route_refiner_advantages(adv, participation)
        assert routed[0] == 0 and routed[1] == 0
        assert torch.equal(routed[2:], adv[2:])
        assert not routed.requires_grad

    def test_refiner_route_length_validation(self):
        with pytest.raises(ValueError):
            route_refiner_advantages(torch.zeros(3), [True])

    def test_generator_route_consumes_all_samples(self):
        adv = torch.tensor([1.0, -1.0, 2.0, -2.0])
        routed = route_generator_advantages(adv)
        assert torch.equal(routed, adv)
        assert not routed.requires_grad
