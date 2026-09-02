# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.train.methods.rl.rewards.media import MultiRewardScorer


def test_multi_reward_exposes_diagnostics_without_changing_total() -> None:
    class DiagnosticScorer:
        diagnostic_names = ("raw", "saturation")

        def __init__(self) -> None:
            self.last_diagnostics: dict[str, torch.Tensor] = {}

        def __call__(self, media, prompts) -> torch.Tensor:
            del media, prompts
            values = torch.tensor([1.0, 2.0])
            self.last_diagnostics = {
                "raw": torch.tensor([1.5, 3.0]),
                "saturation": torch.tensor([0.5, 1.0]),
            }
            return values

    scorer = MultiRewardScorer(
        {"dynamic_tracking": 0.7},
        scorers={"dynamic_tracking": DiagnosticScorer()},
    )
    result = scorer(
        torch.zeros(2, 3, 1, 2, 2),
        ["first", "second"],
    )

    assert scorer.output_keys == (
        "dynamic_tracking",
        "dynamic_tracking_raw",
        "dynamic_tracking_saturation",
        "avg",
    )
    assert torch.allclose(
        result["avg"],
        torch.tensor([0.7, 1.4]),
    )
    assert torch.equal(
        result["dynamic_tracking_raw"],
        torch.tensor([1.5, 3.0]),
    )
    assert torch.equal(
        result["dynamic_tracking_saturation"],
        torch.tensor([0.5, 1.0]),
    )
