# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
import torch

from fastvideo.train.methods.rl.rewards import (
    CalibratedRewardScorer,
    MultiRewardScorer,
    RewardCalibrationEntry,
    build_multi_reward_scorer,
    load_reward_calibration,
)


class _FakeScorer:
    diagnostic_names = ("aux",)

    def __init__(self, values: torch.Tensor) -> None:
        self.values = values
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def __call__(self, media, prompts) -> torch.Tensor:
        del media
        if len(prompts) != self.values.numel():
            raise ValueError("prompt count mismatch")
        self.last_diagnostics = {
            "aux": self.values + 10.0,
        }
        return self.values.clone()


def test_calibrated_scorer_applies_fixed_affine_transform() -> None:
    base = _FakeScorer(torch.tensor([2.0, 6.0]))
    scorer = CalibratedRewardScorer(
        base,
        RewardCalibrationEntry(
            center=2.0,
            scale=2.0,
        ),
        clip=1.5,
    )

    values = scorer(
        torch.empty(2, 3, 1, 1),
        ["a", "b"],
    )

    assert torch.equal(
        values,
        torch.tensor([0.0, 1.5]),
    )
    assert scorer.diagnostic_names == (
        "unnormalized",
        "aux",
    )
    assert torch.equal(
        scorer.last_diagnostics["unnormalized"],
        torch.tensor([2.0, 6.0]),
    )
    assert torch.equal(
        scorer.last_diagnostics["aux"],
        torch.tensor([12.0, 16.0]),
    )


def test_load_inline_calibration_requires_every_configured_reward() -> None:
    raw = {
        "rewards": {
            "first": 1.0,
            "second": 1.0,
        },
        "calibration": {
            "required": True,
            "entries": {
                "first": {
                    "center": 0.0,
                    "scale": 1.0,
                }
            },
        },
    }
    with pytest.raises(
        ValueError,
        match="missing.*second",
    ):
        load_reward_calibration(
            raw,
            reward_names=["first", "second"],
        )


def test_load_calibration_artifact_records_metadata(
    tmp_path,
) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_by": "unit-test",
                "components": {
                    "first": {
                        "center": 3.0,
                        "scale": 2.0,
                        "count": 100,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    calibration, required, clip = load_reward_calibration(
        {
            "calibration": {
                "path": str(path),
                "required": True,
                "clip": 4.0,
            }
        },
        reward_names=["first"],
    )

    assert calibration is not None
    assert required is True
    assert clip == 4.0
    assert calibration.source_path == str(path)
    assert calibration.metadata["created_by"] == "unit-test"
    assert calibration.entries["first"].count == 100


@pytest.mark.parametrize(
    "scale",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_invalid_calibration_scale_fails(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        load_reward_calibration(
            {
                "calibration": {
                    "required": True,
                    "entries": {
                        "first": {
                            "center": 0.0,
                            "scale": scale,
                        }
                    },
                }
            },
            reward_names=["first"],
        )


def test_multi_reward_output_contract_includes_diagnostics() -> None:
    first = _FakeScorer(torch.tensor([1.0, 2.0]))
    scorer = MultiRewardScorer(
        {"first": 0.5},
        scorers={"first": first},
    )

    assert scorer.output_keys == (
        "first",
        "first_aux",
        "avg",
    )
    result = scorer(
        torch.empty(2, 3, 1, 1),
        ["a", "b"],
    )
    assert tuple(result) == scorer.output_keys
    assert torch.equal(
        result["avg"],
        torch.tensor([0.5, 1.0]),
    )


def test_builder_preserves_raw_profile_without_calibration() -> None:
    base = _FakeScorer(torch.tensor([2.0, 4.0]))
    scorer = build_multi_reward_scorer(
        {
            "rewards": {
                "first": 0.25,
            }
        },
        device="cpu",
        scorers={"first": base},
    )

    result = scorer(
        torch.empty(2, 3, 1, 1),
        ["a", "b"],
    )
    assert torch.equal(
        result["first"],
        torch.tensor([2.0, 4.0]),
    )
    assert torch.equal(
        result["avg"],
        torch.tensor([0.5, 1.0]),
    )


def test_builder_applies_inline_calibration_and_logs_raw_values() -> None:
    base = _FakeScorer(torch.tensor([2.0, 4.0]))
    scorer = build_multi_reward_scorer(
        {
            "rewards": {
                "first": 0.25,
            },
            "calibration": {
                "required": True,
                "entries": {
                    "first": {
                        "center": 2.0,
                        "scale": 2.0,
                    }
                },
            },
        },
        device="cpu",
        scorers={"first": base},
    )

    assert scorer.output_keys == (
        "first",
        "first_unnormalized",
        "first_aux",
        "avg",
    )
    result = scorer(
        torch.empty(2, 3, 1, 1),
        ["a", "b"],
    )
    assert torch.equal(
        result["first"],
        torch.tensor([0.0, 1.0]),
    )
    assert torch.equal(
        result["first_unnormalized"],
        torch.tensor([2.0, 4.0]),
    )
    assert torch.equal(
        result["avg"],
        torch.tensor([0.0, 0.25]),
    )
