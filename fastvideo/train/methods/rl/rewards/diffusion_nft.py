# SPDX-License-Identifier: Apache-2.0
"""DiffusionNFT-compatible reward adapters."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from fastvideo.train.methods.rl.rewards.media import media_to_uint8_array


def normalize_reward_weights(
    reward_config: Any,
) -> tuple[dict[str, float], str | None]:
    """Accept flat or DiffusionNFT-style nested reward mappings."""
    backend = None
    raw_rewards = reward_config
    if isinstance(raw_rewards, Mapping) and "backend" in raw_rewards:
        backend = str(raw_rewards["backend"]).strip().lower()
    if isinstance(raw_rewards, Mapping) and "rewards" in raw_rewards:
        raw_rewards = raw_rewards["rewards"]
    if not isinstance(raw_rewards, Mapping) or not raw_rewards:
        raise ValueError("method.reward_fn must be a non-empty mapping, "
                         "for example {pickscore: 1.0, clipscore: 1.0} "
                         "or {rewards: {videoalign_vq: 1.0}}")
    return {str(key).strip().lower(): float(value) for key, value in raw_rewards.items()}, backend


class JpegIncompressibilityScorer:

    def __call__(
        self,
        media: torch.Tensor,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        del prompts
        arr = media_to_uint8_array(media)
        sizes: list[float] = []
        for sample in arr:
            frames = sample[np.newaxis] if sample.ndim == 3 else sample
            frame_sizes: list[float] = []
            for frame in frames:
                buffer = BytesIO()
                Image.fromarray(frame).save(buffer, format="JPEG", quality=95)
                frame_sizes.append(buffer.tell() / 1000.0)
            sizes.append(float(np.mean(frame_sizes)))
        return torch.tensor(sizes, device=media.device, dtype=torch.float32)


class JpegCompressibilityScorer:

    def __init__(self) -> None:
        self._incompressibility = JpegIncompressibilityScorer()

    def __call__(
        self,
        media: torch.Tensor,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        return -self._incompressibility(media, prompts) / 500.0


class MeanLuminanceScorer:

    def __call__(
        self,
        media: torch.Tensor,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        del prompts
        reduce_dims = tuple(range(1, media.ndim))
        return media.detach().float().mean(dim=reduce_dims)


def _ensure_local_diffusion_nft_on_path() -> None:
    explicit_root = os.environ.get("DIFFUSION_NFT_ROOT")
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    candidates.append(Path("/cache/DiffusionNFT"))
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "DiffusionNFT")

    for root_path in candidates:
        if (root_path / "flow_grpo").is_dir():
            root = str(root_path)
            if root not in sys.path:
                sys.path.insert(0, root)
            return


def _flatten_video_for_image_reward(
    media: torch.Tensor,
    prompts: Sequence[str],
) -> tuple[torch.Tensor, list[str], int | None]:
    if media.ndim != 5:
        return media, list(prompts), None
    batch, channels, frames, height, width = media.shape
    flat = media.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    flat_prompts = [prompt for prompt in prompts for _ in range(frames)]
    return flat, flat_prompts, frames


class ExternalDiffusionNFTScorer:
    """Adapter for rewards from a local DiffusionNFT/Flow-GRPO checkout."""

    def __init__(
        self,
        name: str,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        self.name = str(name).strip().lower()
        self.device = torch.device(device)
        _ensure_local_diffusion_nft_on_path()
        try:
            from flow_grpo import rewards as flow_rewards
        except ImportError as exc:
            raise ImportError(f"Reward {self.name!r} requires DiffusionNFT's "
                              "Flow-GRPO reward package. Set DIFFUSION_NFT_ROOT "
                              "to a checkout containing flow_grpo, or choose a "
                              "built-in reward.") from exc
        self._score_fn = flow_rewards.multi_score(self.device, {self.name: 1.0})

    @torch.no_grad()
    def __call__(
        self,
        media: torch.Tensor,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        reward_media, reward_prompts, frames = _flatten_video_for_image_reward(media, prompts)
        metadata: list[dict[str, Any]] = [{} for _ in reward_prompts]
        scores, _ = self._score_fn(
            reward_media,
            reward_prompts,
            metadata,
            only_strict=True,
        )
        value = torch.as_tensor(scores[self.name], device=self.device, dtype=torch.float32)
        if frames is not None:
            if value.numel() % frames != 0:
                raise RuntimeError(f"Reward {self.name!r} returned {value.numel()} scores, "
                                   f"which is not divisible by num_frames={frames}.")
            value = value.reshape(-1, frames).mean(dim=1)
        return value


BUILTIN_DEBUG_REWARD_SCORERS = {
    "jpeg_incompressibility": JpegIncompressibilityScorer,
    "jpeg_compressibility": JpegCompressibilityScorer,
    "mean_luminance": MeanLuminanceScorer,
}
