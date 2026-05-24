# SPDX-License-Identifier: Apache-2.0
"""Reward helpers for DiffusionNFT-style RL methods."""

from __future__ import annotations

import sys
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

RewardFn = Callable[
    [torch.Tensor, list[str], list[dict[str, Any]]],
    dict[str, torch.Tensor],
]


def _to_uint8_nhwc(images: torch.Tensor) -> np.ndarray:
    images = images.detach().float().clamp(0, 1).cpu()
    images = (images * 255.0).round().to(torch.uint8).numpy()
    if images.ndim == 4:
        return np.transpose(images, (0, 2, 3, 1))
    if images.ndim == 5:
        # (N, C, F, H, W) -> (N, F, H, W, C)
        return np.transpose(images, (0, 2, 3, 4, 1))
    raise ValueError(
        "Expected image/video tensor with 4 or 5 dims, got "
        f"shape={images.shape}"
    )


def _jpeg_incompressibility(images: torch.Tensor) -> torch.Tensor:
    arr = _to_uint8_nhwc(images)
    sizes: list[float] = []
    for sample in arr:
        frames = sample[np.newaxis] if sample.ndim == 3 else sample
        frame_sizes: list[float] = []
        for frame in frames:
            buffer = BytesIO()
            Image.fromarray(frame).save(
                buffer, format="JPEG", quality=95
            )
            frame_sizes.append(buffer.tell() / 1000.0)
        sizes.append(float(np.mean(frame_sizes)))
    return torch.tensor(
        sizes, device=images.device, dtype=torch.float32
    )


def _mean_luminance(images: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, images.ndim))
    return images.detach().float().mean(dim=reduce_dims)


def _build_genrl_reward_fn(
    reward_config: dict[str, Any],
    *,
    device: torch.device,
) -> RewardFn:
    from fastvideo.train.methods.rl.utils.rewards import multi_score

    score_fn = multi_score(
        device,
        {str(k): float(v) for k, v in reward_config.items()},
        return_raw_scores=True,
    )

    def _fn(
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        scores, _ = score_fn(images, prompts, metadata, True)
        return {
            key: torch.as_tensor(
                value, device=device, dtype=torch.float32
            )
            for key, value in scores.items()
        }

    return _fn


def _ensure_local_diffusion_nft_on_path() -> None:
    """Let configs use a sibling ``DiffusionNFT`` checkout if present."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "DiffusionNFT" / "flow_grpo"
        if candidate.is_dir():
            root = str(candidate.parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return


def _external_diffusion_nft_reward(
    reward_name: str,
    *,
    weight: float,
    device: torch.device,
) -> RewardFn:
    _ensure_local_diffusion_nft_on_path()
    try:
        from flow_grpo import rewards as flow_rewards
    except ImportError as exc:
        raise ImportError(
            f"Reward {reward_name!r} requires DiffusionNFT's "
            "Flow-GRPO reward package. Keep the local DiffusionNFT "
            "checkout at the repo root or use a built-in reward such "
            "as 'jpeg_incompressibility'."
        ) from exc

    score_fn = flow_rewards.multi_score(
        device, {reward_name: float(weight)}
    )

    def _fn(
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        score_details, _ = score_fn(
            images, prompts, metadata, only_strict=True
        )
        return {
            key: torch.as_tensor(
                value, device=device, dtype=torch.float32
            )
            for key, value in score_details.items()
        }

    return _fn


def build_diffusion_nft_reward_fn(
    reward_config: dict[str, Any] | None,
    *,
    device: torch.device,
    backend: str = "auto",
) -> RewardFn:
    """Build a weighted image reward callable.

    ``reward_config`` accepts either ``{"name": weight}`` or
    ``{"rewards": {"name": weight}}``. Built-ins are
    ``jpeg_incompressibility``, ``jpeg_compressibility``, and
    ``mean_luminance``. Unknown names delegate to the local
    DiffusionNFT/Flow-GRPO reward package.
    """
    raw_rewards: Any = reward_config or {
        "jpeg_incompressibility": 1.0
    }
    if isinstance(raw_rewards, dict) and "backend" in raw_rewards:
        backend = str(raw_rewards["backend"]).strip().lower()
    if isinstance(raw_rewards, dict) and "rewards" in raw_rewards:
        raw_rewards = raw_rewards["rewards"]
    if not isinstance(raw_rewards, dict) or not raw_rewards:
        raise ValueError("method.reward_fn must be a non-empty mapping")

    genrl_reward_names = {
        "video_ocr",
        "hpsv3_general",
        "hpsv3_percentile",
        "videoalign_mq",
        "videoalign_ta",
    }
    if backend not in {"auto", "diffusion_nft", "genrl"}:
        raise ValueError(
            "method.reward_backend must be one of auto, diffusion_nft, "
            f"or genrl, got {backend!r}"
        )
    if backend == "genrl" or (
        backend == "auto"
        and any(str(name) in genrl_reward_names for name in raw_rewards)
    ):
        return _build_genrl_reward_fn(raw_rewards, device=device)

    reward_fns: list[tuple[str, float, RewardFn]] = []
    for name, weight_raw in raw_rewards.items():
        reward_name = str(name).strip().lower()
        weight = float(weight_raw)

        if reward_name == "jpeg_incompressibility":

            def reward_fn(
                images: torch.Tensor,
                prompts: list[str],
                metadata: list[dict[str, Any]],
            ) -> dict[str, torch.Tensor]:
                del prompts, metadata
                return {
                    "jpeg_incompressibility": (
                        _jpeg_incompressibility(images)
                    )
                }

        elif reward_name == "jpeg_compressibility":

            def reward_fn(
                images: torch.Tensor,
                prompts: list[str],
                metadata: list[dict[str, Any]],
            ) -> dict[str, torch.Tensor]:
                del prompts, metadata
                return {
                    "jpeg_compressibility": (
                        -_jpeg_incompressibility(images) / 500.0
                    )
                }

        elif reward_name == "mean_luminance":

            def reward_fn(
                images: torch.Tensor,
                prompts: list[str],
                metadata: list[dict[str, Any]],
            ) -> dict[str, torch.Tensor]:
                del prompts, metadata
                return {"mean_luminance": _mean_luminance(images)}

        else:
            reward_fn = _external_diffusion_nft_reward(
                reward_name,
                weight=weight,
                device=device,
            )

        reward_fns.append((reward_name, weight, reward_fn))

    def _multi_reward(
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        details: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for reward_name, weight, reward_fn in reward_fns:
            scores = reward_fn(images, prompts, metadata)
            if reward_name in scores:
                base_score = scores[reward_name]
            elif "avg" in scores:
                base_score = scores["avg"]
            else:
                base_score = scores[next(iter(scores))]
            for key, value in scores.items():
                details[key] = value.to(
                    device=device, dtype=torch.float32
                )
            weighted = (
                base_score.to(device=device, dtype=torch.float32)
                * weight
            )
            total = weighted if total is None else total + weighted
        if total is None:
            raise RuntimeError("No rewards were computed")
        details["avg"] = total
        return details

    return _multi_reward
