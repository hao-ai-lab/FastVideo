# SPDX-License-Identifier: Apache-2.0
"""Reusable reward models for training methods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from fastvideo.train.methods.rl.rewards.calibration import (
    CalibratedRewardScorer,
    RewardCalibration,
    RewardCalibrationEntry,
    load_reward_calibration,
)
from fastvideo.train.methods.rl.rewards.frame_rewards import (
    ClipScoreScorer,
    PickScoreScorer,
)
from fastvideo.train.methods.rl.rewards.media import (
    MultiRewardScorer,
    RewardScorer,
    media_to_float_tensor,
    media_to_uint8_array,
    select_first_frame,
)


class MeanLuminanceScorer:
    """Dependency-free scorer used only by correctness smoke tests."""

    @torch.no_grad()
    def __call__(self, media: torch.Tensor, prompts) -> torch.Tensor:
        del prompts
        value = media_to_float_tensor(media)
        return value.mean(dim=tuple(range(1, value.ndim)))


def _parse_reward_specs(
    raw: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    configured_options: Mapping[str, Any] = {}
    if "rewards" in raw:
        candidate_options = raw.get("options", {})
        if not isinstance(candidate_options, Mapping):
            raise ValueError("reward config options must be a mapping")
        configured_options = candidate_options
        raw = raw["rewards"]
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("reward config must be a nonempty mapping")
    weights: dict[str, float] = {}
    options: dict[str, dict[str, Any]] = {}
    for raw_name, value in raw.items():
        name = str(raw_name).strip().lower()
        base_options = configured_options.get(name, {})
        if not isinstance(base_options, Mapping):
            raise ValueError(
                f"reward options for {name!r} must be a mapping"
            )
        merged_options = dict(base_options)
        if isinstance(value, Mapping):
            spec = dict(value)
            if "weight" not in spec:
                raise ValueError(
                    f"reward {name!r} must define a numeric weight"
                )
            weights[name] = float(spec.pop("weight"))
            merged_options.update(spec)
        else:
            weights[name] = float(value)
        options[name] = merged_options
    return weights, options


def _build_scorer(
    name: str,
    *,
    device: torch.device | str,
    options: dict[str, Any],
) -> RewardScorer:
    options = dict(options)
    scorer_device = options.pop("device", device)
    if name == "pickscore":
        return PickScoreScorer(device=device)
    if name == "clipscore":
        return ClipScoreScorer(device=device)
    if name == "mean_luminance":
        if options:
            raise ValueError(
                "mean_luminance does not accept options"
            )
        return MeanLuminanceScorer()
    if name == "videoalign_ta":
        from fastvideo.train.methods.rl.rewards.videoalign import (
            VideoAlignTextAlignmentScorer,
        )

        return VideoAlignTextAlignmentScorer(
            device=scorer_device,
            **options,
        )
    if name == "videoalign_mq":
        from fastvideo.train.methods.rl.rewards.videoalign import (
            VideoAlignMotionQualityScorer,
        )

        return VideoAlignMotionQualityScorer(
            device=scorer_device,
            **options,
        )
    if name == "videoalign_vq":
        from fastvideo.train.methods.rl.rewards.videoalign import (
            VideoAlignVisualQualityScorer,
        )

        return VideoAlignVisualQualityScorer(
            device=scorer_device,
            **options,
        )
    if name == "hpsv3_general":
        from fastvideo.train.methods.rl.rewards.hpsv3 import (
            HPSv3GeneralScorer,
        )

        return HPSv3GeneralScorer(
            device=scorer_device,
            **options,
        )
    if name == "hpsv3_percentile":
        from fastvideo.train.methods.rl.rewards.hpsv3 import (
            HPSv3PercentileScorer,
        )

        return HPSv3PercentileScorer(
            device=scorer_device,
            **options,
        )
    if name == "dynamic_tracking":
        from fastvideo.train.methods.rl.rewards.dynamic_tracking import (
            DynamicTrackingScorer,
        )

        scorer = DynamicTrackingScorer(
            device=scorer_device,
            **options,
        )
        scorer.diagnostic_names = ("raw", "saturation")
        return scorer
    if name in {"mjvideo_cc", "mjvideo_fineness"}:
        from fastvideo.train.methods.rl.rewards.mj_video_compat import (
            install_mj_video_transformers_compat,
        )

        install_mj_video_transformers_compat()
        from fastvideo.train.methods.rl.rewards.mj_video import (
            MJVideoAspectScorer,
        )

        aspect = (
            "cc"
            if name == "mjvideo_cc"
            else "fineness"
        )
        return MJVideoAspectScorer(
            aspect=aspect,
            device=scorer_device,
            **options,
        )
    raise ValueError(
        f"Unsupported reward {name!r}. Available: clipscore, "
        "pickscore, mean_luminance, videoalign_ta, videoalign_mq, "
        "videoalign_vq, hpsv3_general, hpsv3_percentile, "
        "dynamic_tracking, mjvideo_cc, mjvideo_fineness"
    )


def build_multi_reward_scorer(
    reward_weights,
    *,
    device="cuda",
    scorers: dict[str, RewardScorer] | None = None,
) -> MultiRewardScorer:
    weights, options = _parse_reward_specs(reward_weights)
    available: dict[str, RewardScorer] = dict(scorers or {})
    for name in weights:
        if name not in available:
            available[name] = _build_scorer(
                name,
                device=device,
                options=options[name],
            )

    calibration, _required, clip = load_reward_calibration(
        reward_weights,
        reward_names=list(weights),
    )
    if calibration is not None:
        for name, entry in calibration.entries.items():
            available[name] = CalibratedRewardScorer(
                available[name],
                entry,
                clip=clip,
            )

    return MultiRewardScorer(
        weights,
        scorers=available,
    )


__all__ = [
    "CalibratedRewardScorer",
    "ClipScoreScorer",
    "MeanLuminanceScorer",
    "MultiRewardScorer",
    "PickScoreScorer",
    "RewardCalibration",
    "RewardCalibrationEntry",
    "RewardScorer",
    "build_multi_reward_scorer",
    "load_reward_calibration",
    "media_to_float_tensor",
    "media_to_uint8_array",
    "select_first_frame",
]
