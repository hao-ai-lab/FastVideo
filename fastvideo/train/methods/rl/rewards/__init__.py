# SPDX-License-Identifier: Apache-2.0
"""Reusable reward models for training methods."""

from typing import Any

from fastvideo.train.methods.rl.rewards.diffusion_nft import (
    BUILTIN_DEBUG_REWARD_SCORERS,
    ExternalDiffusionNFTScorer,
    JpegCompressibilityScorer,
    JpegIncompressibilityScorer,
    MeanLuminanceScorer,
    normalize_reward_weights,
)
from fastvideo.train.methods.rl.rewards.frame_rewards import (
    ClipScoreScorer,
    PickScoreScorer,
)
from fastvideo.train.methods.rl.rewards.media import (
    MultiRewardScorer,
    RewardScorer,
    media_to_uint8_array,
    select_first_frame,
)

GENRL_REWARD_NAMES = frozenset({
    "video_ocr",
    "hpsv3_general",
    "hpsv3_percentile",
    "videoalign_mq",
    "videoalign_ta",
    "videoalign_vq",
})

_NATIVE_SCORER_CLASSES: dict[str, Any] = {
    "pickscore": PickScoreScorer,
    "clipscore": ClipScoreScorer,
    **BUILTIN_DEBUG_REWARD_SCORERS,
}


def _build_lazy_genrl_scorer(
    name: str,
    *,
    device,
) -> RewardScorer:
    if name == "hpsv3_general":
        from fastvideo.train.methods.rl.rewards.hpsv3 import HPSv3GeneralScorer

        return HPSv3GeneralScorer(device=device)
    if name == "hpsv3_percentile":
        from fastvideo.train.methods.rl.rewards.hpsv3 import HPSv3PercentileScorer

        return HPSv3PercentileScorer(device=device)
    if name == "videoalign_mq":
        from fastvideo.train.methods.rl.rewards.videoalign import VideoAlignMotionQualityScorer

        return VideoAlignMotionQualityScorer(device=device)
    if name == "videoalign_ta":
        from fastvideo.train.methods.rl.rewards.videoalign import VideoAlignTextAlignmentScorer

        return VideoAlignTextAlignmentScorer(device=device)
    if name == "videoalign_vq":
        from fastvideo.train.methods.rl.rewards.videoalign import VideoAlignVisualQualityScorer

        return VideoAlignVisualQualityScorer(device=device)
    if name == "video_ocr":
        from fastvideo.train.methods.rl.rewards.ocr import VideoOCRScorer

        return VideoOCRScorer()
    raise ValueError(f"Unsupported GenRL reward {name!r}. "
                     f"Available GenRL rewards: {sorted(GENRL_REWARD_NAMES)}")


def build_multi_reward_scorer(
    reward_weights,
    *,
    device="cuda",
    backend: str = "auto",
    scorers: dict[str, RewardScorer] | None = None,
) -> MultiRewardScorer:
    reward_weights, reward_backend = normalize_reward_weights(reward_weights)
    backend = reward_backend or str(backend or "auto").strip().lower()
    if backend not in {"auto", "diffusion_nft", "genrl"}:
        raise ValueError("method.reward_backend must be one of auto, diffusion_nft, or genrl, "
                         f"got {backend!r}")

    available: dict[str, RewardScorer] = dict(scorers or {})
    for name in reward_weights:
        if name in available:
            continue
        if backend == "diffusion_nft" and name not in _NATIVE_SCORER_CLASSES:
            available[name] = ExternalDiffusionNFTScorer(name, device=device)
            continue
        if name in GENRL_REWARD_NAMES:
            available[name] = _build_lazy_genrl_scorer(name, device=device)
            continue
        scorer_cls = _NATIVE_SCORER_CLASSES.get(name)
        if scorer_cls is None:
            if backend == "genrl":
                raise ValueError(f"Unsupported GenRL reward {name!r}. "
                                 f"Available GenRL rewards: {sorted(GENRL_REWARD_NAMES)}")
            available[name] = ExternalDiffusionNFTScorer(name, device=device)
        elif name in BUILTIN_DEBUG_REWARD_SCORERS:
            available[name] = scorer_cls()
        else:
            available[name] = scorer_cls(device=device)
    return MultiRewardScorer(reward_weights, scorers=available)


__all__ = [
    "ClipScoreScorer",
    "ExternalDiffusionNFTScorer",
    "JpegCompressibilityScorer",
    "JpegIncompressibilityScorer",
    "MeanLuminanceScorer",
    "MultiRewardScorer",
    "PickScoreScorer",
    "RewardScorer",
    "build_multi_reward_scorer",
    "media_to_uint8_array",
    "normalize_reward_weights",
    "select_first_frame",
]
