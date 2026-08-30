# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.train.methods.rl.common.rvm_utils import (
    detached_rvm_surrogate,
    five_percent_interval,
    standardize_group_rewards,
    visual_text_from_h3_prompt,
)


def test_visual_text_strips_audio_fields() -> None:
    prompt = (
        "integrated_multimodal_description: A red car drives left. "
        "overall_soundscape: engine noise. non_diegetic_music: drums."
    )
    assert visual_text_from_h3_prompt(prompt) == "A red car drives left."


def test_equal_rewards_have_zero_rvm_signal() -> None:
    advantages, std = standardize_group_rewards(torch.ones(8))
    assert torch.equal(advantages, torch.zeros_like(advantages))
    assert std.item() == 0.0


def test_signed_surrogate_matches_expected_gradient() -> None:
    prediction = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([0.0, 0.0])
    loss = detached_rvm_surrogate(prediction, target, coefficient=-0.5)
    loss.backward()
    expected = -0.5 * (prediction.detach() - target) / prediction.numel()
    assert torch.allclose(prediction.grad, expected)


def test_reference_anchor_adds_gradient() -> None:
    prediction = torch.tensor([1.0, 3.0], requires_grad=True)
    target = torch.zeros(2)
    reference = torch.tensor([0.5, 2.0])
    loss = detached_rvm_surrogate(
        prediction,
        target,
        coefficient=0.0,
        reference=reference,
        anchor_beta=0.2,
    )
    loss.backward()
    expected = 0.2 * (prediction.detach() - reference) / prediction.numel()
    assert torch.allclose(prediction.grad, expected)


def test_five_percent_interval() -> None:
    assert five_percent_interval(180) == 9
    assert five_percent_interval(16) == 1
