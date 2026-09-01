# SPDX-License-Identifier: Apache-2.0
"""Frame-aggregated HPSv3 rewards used by the RVM video recipe."""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from fastvideo.train.methods.rl.rewards.media import media_to_uint8_array

_INFERENCERS: dict[str, Any] = {}


def _patch_transformers5_imports() -> None:
    """Expose HPSv3's training-only legacy imports without downgrading Transformers."""
    from transformers import trainer, trainer_pt_utils

    if not hasattr(trainer, "nested_concat"):
        trainer.nested_concat = trainer_pt_utils.nested_concat

    class _TrainingOnlyLegacyUtility:

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError("This removed Transformers utility is only available in HPSv3 training code")

    for name in ("DistributedTensorGatherer", "SequentialDistributedSampler"):
        if not hasattr(trainer, name):
            setattr(trainer, name, _TrainingOnlyLegacyUtility)
    differentiable_module = "hpsv3.model.differentiable_image_processor"
    if differentiable_module not in sys.modules:
        module: Any = types.ModuleType(differentiable_module)
        module.Qwen2VLImageProcessor = _TrainingOnlyLegacyUtility
        sys.modules[differentiable_module] = module


def _patch_hpsv3_model() -> None:
    from hpsv3.model import qwen2vl_trainer

    from fastvideo.train.methods.rl.rewards.videoalign import (
        _patch_load_state_dict,
        _patch_reward_model_forward,
        _patch_reward_model_from_pretrained,
        _patch_reward_model_init,
    )

    cls = qwen2vl_trainer.Qwen2VLRewardModelBT
    _patch_reward_model_init(cls)
    _patch_reward_model_from_pretrained(cls)
    _patch_reward_model_forward(cls)
    _patch_load_state_dict(cls)


def _get_inferencer(device: torch.device) -> Any:
    key = str(device)
    if key not in _INFERENCERS:
        try:
            _patch_transformers5_imports()
            from hpsv3 import HPSv3RewardInferencer
            _patch_hpsv3_model()
        except ImportError as exc:
            raise ImportError("HPSv3 rewards require `pip install hpsv3==1.0.0`.") from exc
        _INFERENCERS[key] = HPSv3RewardInferencer(device=device)
    return _INFERENCERS[key]


def _scalar(value: Any) -> float:
    if isinstance(value, list | tuple) and value:
        value = value[0]
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.flatten()[0].item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _sample_frame_indices(num_frames: int, max_frames: int | None) -> list[int]:
    if num_frames <= 0:
        raise ValueError("video contains no frames")
    if max_frames is None or max_frames <= 0 or num_frames <= max_frames:
        return list(range(num_frames))
    return torch.linspace(0, num_frames - 1, int(max_frames)).round().long().tolist()


class _HPSv3VideoScorer:

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        max_frames: int | None = 53,
        batch_size: int = 16,
    ) -> None:
        self.device = torch.device(device)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def _frame_prompts(self, prompt: str, count: int) -> list[str]:
        raise NotImplementedError

    def _aggregate(self, values: list[float]) -> float:
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, media: torch.Tensor, prompts) -> torch.Tensor:
        inferencer = _get_inferencer(self.device)
        videos = media_to_uint8_array(media)
        all_paths: list[str] = []
        all_prompts: list[str] = []
        owners: list[int] = []
        with tempfile.TemporaryDirectory(prefix="fastvideo_hpsv3_") as temp_dir:
            root = Path(temp_dir)
            for sample_index, (video, prompt) in enumerate(zip(videos, prompts, strict=True)):
                frames = video[None] if video.ndim == 3 else video
                indices = _sample_frame_indices(len(frames), self.max_frames)
                selected_prompts = self._frame_prompts(str(prompt), len(indices))
                for local_index, (frame_index, frame_prompt) in enumerate(zip(indices, selected_prompts, strict=True)):
                    path = root / f"sample_{sample_index:04d}_frame_{local_index:04d}.png"
                    Image.fromarray(frames[frame_index]).save(path)
                    all_paths.append(str(path))
                    all_prompts.append(frame_prompt)
                    owners.append(sample_index)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                raw = []
                for start in range(0, len(all_paths), self.batch_size):
                    raw.extend(
                        inferencer.reward(
                            all_paths[start:start + self.batch_size],
                            all_prompts[start:start + self.batch_size],
                        ))
            grouped: list[list[float]] = [[] for _ in prompts]
            for owner, value in zip(owners, raw, strict=True):
                grouped[owner].append(_scalar(value))
        scores = [self._aggregate(values) if values else 0.0 for values in grouped]
        return torch.tensor(scores, device=self.device, dtype=torch.float32)


class HPSv3GeneralScorer(_HPSv3VideoScorer):
    """Mean frame preference under the generic quality prompt."""

    def _frame_prompts(self, prompt: str, count: int) -> list[str]:
        del prompt
        return ["A high-quality image"] * count

    def _aggregate(self, values: list[float]) -> float:
        return float(np.mean(values))


class HPSv3PercentileScorer(_HPSv3VideoScorer):
    """Mean of the top 30% prompt-conditioned frame scores."""

    def _frame_prompts(self, prompt: str, count: int) -> list[str]:
        return [prompt] * count

    def _aggregate(self, values: list[float]) -> float:
        count = max(1, int(len(values) * 0.3))
        return float(np.mean(sorted(values, reverse=True)[:count]))


__all__ = ["HPSv3GeneralScorer", "HPSv3PercentileScorer"]
