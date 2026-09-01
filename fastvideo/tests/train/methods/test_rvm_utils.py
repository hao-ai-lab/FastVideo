# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from fastvideo.train.methods.rl.common.rvm_utils import (
    detached_rvm_surrogate,
    five_percent_interval,
    standardize_group_rewards,
    visual_text_from_h3_prompt,
)
from fastvideo.train.methods.rl.rewards import _parse_reward_specs
from fastvideo.train.methods.rl.rewards import hpsv3
from fastvideo.train.methods.rl.rewards.dynamic_tracking import (
    DynamicTrackingScorer,
)


def test_visual_text_strips_audio_fields() -> None:
    prompt = (
        "integrated_multimodal_description: A red car drives left. "
        "overall_soundscape: engine noise. "
        "non_diegetic_music: drums."
    )
    assert (
        visual_text_from_h3_prompt(prompt)
        == "A red car drives left."
    )


def test_equal_rewards_have_zero_rvm_signal() -> None:
    advantages, std = standardize_group_rewards(
        torch.ones(8)
    )
    assert torch.equal(
        advantages,
        torch.zeros_like(advantages),
    )
    assert std.item() == 0.0


def test_batch_global_std_preserves_relative_group_signal() -> None:
    small = torch.tensor([-0.01, 0.01])
    large = torch.tensor([-1.0, 1.0])
    global_std = torch.cat(
        [small, large]
    ).std(unbiased=False)

    small_advantages, _ = standardize_group_rewards(
        small,
        scale=1.0,
        clip=None,
        normalization_std=global_std,
    )
    large_advantages, _ = standardize_group_rewards(
        large,
        scale=1.0,
        clip=None,
        normalization_std=global_std,
    )

    ratio = (
        large_advantages.abs().max()
        / small_advantages.abs().max()
    )
    assert torch.allclose(
        ratio,
        torch.tensor(100.0),
        atol=1e-4,
    )


def test_legacy_group_std_remains_available_for_ablation() -> None:
    small = torch.tensor([-0.01, 0.01])
    large = torch.tensor([-1.0, 1.0])

    small_advantages, _ = standardize_group_rewards(
        small,
        scale=1.0,
        clip=None,
    )
    large_advantages, _ = standardize_group_rewards(
        large,
        scale=1.0,
        clip=None,
    )

    assert torch.allclose(
        small_advantages,
        large_advantages,
        atol=1e-5,
    )


def test_signed_surrogate_matches_expected_gradient() -> None:
    prediction = torch.tensor(
        [1.0, 2.0],
        requires_grad=True,
    )
    target = torch.tensor([0.0, 0.0])
    loss = detached_rvm_surrogate(
        prediction,
        target,
        coefficient=-0.5,
    )
    loss.backward()
    expected = (
        -0.5
        * (prediction.detach() - target)
        / prediction.numel()
    )
    assert torch.allclose(
        prediction.grad,
        expected,
    )


def test_reference_anchor_adds_gradient() -> None:
    prediction = torch.tensor(
        [1.0, 3.0],
        requires_grad=True,
    )
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
    expected = (
        0.2
        * (prediction.detach() - reference)
        / prediction.numel()
    )
    assert torch.allclose(
        prediction.grad,
        expected,
    )


def test_five_percent_interval() -> None:
    assert five_percent_interval(180) == 9
    assert five_percent_interval(16) == 1


def test_reward_specs_merge_top_level_options_and_inline_overrides() -> None:
    weights, options = _parse_reward_specs(
        {
            "rewards": {
                "hpsv3_general": 0.1,
                "dynamic_tracking": {
                    "weight": 0.7,
                    "frame_pairs": 6,
                },
            },
            "options": {
                "hpsv3_general": {
                    "device": "cpu",
                    "max_frames": 4,
                },
                "dynamic_tracking": {
                    "frame_pairs": 4,
                    "top_fraction": 0.05,
                },
            },
        }
    )

    assert weights == {
        "hpsv3_general": 0.1,
        "dynamic_tracking": 0.7,
    }
    assert options["hpsv3_general"] == {
        "device": "cpu",
        "max_frames": 4,
    }
    assert options["dynamic_tracking"] == {
        "frame_pairs": 6,
        "top_fraction": 0.05,
    }


def test_dynamic_tracking_accepts_h3_config_option_names() -> None:
    scorer = DynamicTrackingScorer(
        device="cpu",
        frame_pairs=4,
        resize_short_edge=256,
        pretrained=True,
    )

    assert scorer.num_pairs == 4
    assert scorer.resize_short_edge == 256


def test_hpsv3_chunks_frames_without_changing_per_video_aggregation(
    monkeypatch,
) -> None:
    class FakeInferencer:
        def __init__(self) -> None:
            self.batch_sizes = []
            self.next_value = 0

        def reward(self, paths, prompts):
            assert len(paths) == len(prompts)
            self.batch_sizes.append(len(paths))
            values = list(
                range(
                    self.next_value,
                    self.next_value + len(paths),
                )
            )
            self.next_value += len(paths)
            return values

    inferencer = FakeInferencer()
    videos = np.zeros(
        (2, 5, 2, 2, 3),
        dtype=np.uint8,
    )
    monkeypatch.setattr(
        hpsv3,
        "_get_inferencer",
        lambda device: inferencer,
    )
    monkeypatch.setattr(
        hpsv3,
        "media_to_uint8_array",
        lambda media: videos,
    )

    scorer = hpsv3.HPSv3GeneralScorer(
        device="cpu",
        max_frames=None,
        batch_size=4,
    )
    scores = scorer(
        torch.empty(0),
        ["first", "second"],
    )

    assert inferencer.batch_sizes == [4, 4, 2]
    assert torch.equal(
        scores,
        torch.tensor([2.0, 7.0]),
    )
