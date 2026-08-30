# SPDX-License-Identifier: Apache-2.0
"""Adapters for public GenRL video reward implementations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from fastvideo.train.methods.rl.rewards.media import RewardScorer, media_to_uint8_array

_GENRL_REWARD_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "rl_rewards" / "GenRL"


class GenRLRewardScorer:
    """Wrap a GenRL reward factory as FastVideo's media scorer contract."""

    def __init__(
        self,
        factory_name: str,
        *,
        device: torch.device | str = "cuda",
        factory_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.factory_name = str(factory_name)
        self.factory_kwargs = dict(factory_kwargs or {})
        self._score_fn: Callable[..., Any] | None = None

    def _load(self) -> Callable[..., Any]:
        if self._score_fn is not None:
            return self._score_fn
        if _GENRL_REWARD_ROOT.is_dir() and str(_GENRL_REWARD_ROOT) not in sys.path:
            sys.path.insert(0, str(_GENRL_REWARD_ROOT))
        try:
            from genrl.reward import (
                hpsv3_general_score,
                hpsv3_percentile_score,
                videoalign_mq_score,
                videoalign_ta_score,
            )
        except ImportError as exc:
            raise ImportError(
                "GenRL rewards require a checkout under "
                "fastvideo/third_party/rl_rewards/GenRL or an installed "
                "genrl package with reward dependencies."
            ) from exc
        factories = {
            "videoalign_mq": videoalign_mq_score,
            "videoalign_ta": videoalign_ta_score,
            "hpsv3_general": hpsv3_general_score,
            "hpsv3_percentile": hpsv3_percentile_score,
        }
        if self.factory_name not in factories:
            raise ValueError(f"Unknown GenRL reward factory: {self.factory_name}")
        self._score_fn = factories[self.factory_name](self.device, **self.factory_kwargs)
        return self._score_fn

    @torch.no_grad()
    def __call__(self, media: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        score_fn = self._load()
        images = media_to_uint8_array(media)
        result, _metadata = score_fn(images, list(prompts), metadata=None, only_strict=True)
        value = result["avg"] if isinstance(result, dict) else result
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)


GENRL_REWARD_FACTORIES = {
    "videoalign_mq": "videoalign_mq",
    "videoalign_ta": "videoalign_ta",
    "hpsv3_general": "hpsv3_general",
    "hpsv3_percentile": "hpsv3_percentile",
}


def build_genrl_reward_scorer(
    name: str,
    *,
    device: torch.device | str,
    options: dict[str, Any] | None = None,
) -> RewardScorer:
    factory_name = GENRL_REWARD_FACTORIES.get(name)
    if factory_name is None:
        raise ValueError(f"Unsupported GenRL reward {name!r}; available: {sorted(GENRL_REWARD_FACTORIES)}")
    return GenRLRewardScorer(
        factory_name,
        device=device,
        factory_kwargs=options,
    )


__all__ = ["GENRL_REWARD_FACTORIES", "GenRLRewardScorer", "build_genrl_reward_scorer"]
