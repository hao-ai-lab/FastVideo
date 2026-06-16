# SPDX-License-Identifier: Apache-2.0
"""HPSv3 reward functions for visual quality assessment."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.rl.reward.utils import (
    prepare_images, )

logger = init_logger(__name__)

# Global cache of HPSv3 inferencers keyed by device.
_HPSV3_INFERENCERS: dict[str, Any] = {}
_HPSV3_LOAD_PATCHED = False


def _patch_transformers_video_input_alias() -> None:
    """Keep HPSv3 compatible with newer transformers releases.

    HPSv3 imports ``VideoInput`` from ``transformers.image_utils`` for type
    annotations. Some transformers versions used by FastVideo no longer
    export that alias, even though the runtime image utilities HPSv3 needs are
    still present.
    """
    from transformers import image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = image_utils.ImageInput


def _remap_hpsv3_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Adapt HPSv3 checkpoints saved with older Qwen2-VL key names."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("visual."):
            key = f"model.{key}"
        elif key.startswith("model.layers.") or key.startswith("model.embed_tokens.") or key.startswith("model.norm."):
            key = f"model.language_model.{key[len('model.'):]}"

        key = key.replace(
            "base_model.model.visual.",
            "base_model.model.model.visual.",
            1,
        )
        key = key.replace(
            "base_model.model.model.layers.",
            "base_model.model.model.language_model.layers.",
            1,
        )
        key = key.replace(
            "base_model.model.model.embed_tokens.",
            "base_model.model.model.language_model.embed_tokens.",
            1,
        )
        key = key.replace(
            "base_model.model.model.norm.",
            "base_model.model.model.language_model.norm.",
            1,
        )
        remapped[key] = value
    return remapped


def _walk_model_graph(model: Any):
    """Yield common wrapper/base model objects without importing PEFT."""
    stack = [model]
    seen = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("base_model", "model"):
            child = getattr(current, attr, None)
            if child is not None:
                stack.append(child)


def _patch_load_state_dict(cls: Any) -> None:
    """Patch a model class to accept old Qwen2-VL checkpoint keys."""
    if getattr(cls, "_fastvideo_qwen2vl_key_remap", False):
        return

    original_load_state_dict = cls.load_state_dict

    def load_state_dict_with_key_remap(
        self,
        state_dict,
        strict=True,
        assign=False,
    ):
        state_dict = _remap_hpsv3_state_dict(state_dict)
        if not assign:
            try:
                assign = any(getattr(param, "is_meta", False) for param in self.parameters())
            except Exception:
                assign = False
        return original_load_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
        )

    cls.load_state_dict = load_state_dict_with_key_remap
    cls._fastvideo_qwen2vl_key_remap = True


def _patch_hpsv3_state_dict_loader() -> None:
    """Patch HPSv3 reward model loading for transformers key drift."""
    global _HPSV3_LOAD_PATCHED
    if _HPSV3_LOAD_PATCHED:
        return

    from fastvideo.train.methods.rl.reward.HPSv3.hpsv3.model.qwen2vl_trainer import (
        Qwen2VLRewardModelBT, )

    _patch_load_state_dict(Qwen2VLRewardModelBT)
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None
    if PeftModel is not None:
        _patch_load_state_dict(PeftModel)
    _HPSV3_LOAD_PATCHED = True


def _patch_hpsv3_runtime_model(model: Any) -> None:
    """Add aliases expected by HPSv3's older Qwen2-VL forward."""
    for candidate in _walk_model_graph(model):
        language_model = getattr(candidate, "language_model", None)
        if (language_model is not None and not hasattr(candidate, "embed_tokens")
                and hasattr(language_model, "embed_tokens")):
            candidate.__dict__["embed_tokens"] = language_model.embed_tokens


def _normalize_device(device: torch.device | str) -> str:
    if isinstance(device, torch.device):
        return str(device)
    return str(torch.device(device))


def _move_hpsv3_inferencer(inferencer: Any, device: torch.device | str) -> None:
    """Move an HPSv3 inferencer across devices.

    HPSv3RewardInferencer does not expose ``.to()``, but it stores its torch
    module on ``.model`` and reads ``.device`` when preparing batches.
    """
    device_str = _normalize_device(device)
    model = getattr(inferencer, "model", None)
    if model is not None and hasattr(model, "to"):
        model.to(device)
    inferencer.device = device_str


def set_hpsv3_device(device: torch.device | str) -> None:
    """Move cached HPSv3 inferencer to given device."""
    key = _normalize_device(device)
    if key in _HPSV3_INFERENCERS:
        return
    # Move from any existing device.
    for old_key, inf in list(_HPSV3_INFERENCERS.items()):
        if old_key != key:
            _move_hpsv3_inferencer(inf, device)
            _HPSV3_INFERENCERS[key] = inf
            del _HPSV3_INFERENCERS[old_key]
            return


def _get_hpsv3_inferencer(device: torch.device | str) -> Any:
    """Get or create HPSv3 inferencer for device."""
    key = _normalize_device(device)
    if key not in _HPSV3_INFERENCERS:
        try:
            _patch_transformers_video_input_alias()
            from fastvideo.train.methods.rl.reward.HPSv3.hpsv3 import HPSv3RewardInferencer

            _patch_hpsv3_state_dict_loader()
        except ImportError as exc:
            msg = ("Failed to import the vendored HPSv3 reward runtime under "
                   "fastvideo/train/methods/rl/reward/HPSv3. Install "
                   "FastVideo with the GenRL extra and verify the vendored "
                   "runtime files are present.")
            raise ImportError(msg) from exc
        inf = HPSv3RewardInferencer(device=device)
        _patch_hpsv3_runtime_model(inf.model)
        _HPSV3_INFERENCERS[key] = inf
    return _HPSV3_INFERENCERS[key]


def _save_frame_to_temp(frame: np.ndarray) -> str:
    """Save a frame to a temporary PNG file."""
    from PIL import Image

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.fromarray(frame).save(path)
    return path


def _extract_reward_scalar(result) -> float:
    """Extract a float from HPSv3 result."""
    if isinstance(result, torch.Tensor):
        return float(result.item())
    if isinstance(result, float | int):
        return float(result)
    if isinstance(result, list | np.ndarray):
        return float(np.mean(result))
    return float(result)


def hpsv3_general_score(device):
    """Return a reward fn that scores frames with
    'A high-quality image' as prompt.

    Returns mean score across all frames."""

    def _score(images, prompts, metadata, only_strict=False):
        inf = _get_hpsv3_inferencer(device)
        images_np = prepare_images(images)
        batch_scores = []

        for b in range(len(images_np)):
            frames = images_np[b]
            if frames.ndim == 3:
                frames = frames[np.newaxis]
            frame_scores = []
            for frame in frames:
                path = _save_frame_to_temp(frame)
                try:
                    rewards = inf.reward(["A high-quality image"], [path])
                    frame_scores.append(_extract_reward_scalar(rewards[0][0]))
                finally:
                    os.remove(path)
            batch_scores.append(np.mean(frame_scores))

        reward = torch.tensor(batch_scores, device=device).float()
        return {"avg": reward}, {}

    return _score


def hpsv3_percentile_score(device):
    """Return a reward fn that scores frames with per-prompt
    text. Returns mean of top 30% frame scores."""

    def _score(images, prompts, metadata, only_strict=False):
        inf = _get_hpsv3_inferencer(device)
        images_np = prepare_images(images)
        batch_scores = []

        for b in range(len(images_np)):
            frames = images_np[b]
            if frames.ndim == 3:
                frames = frames[np.newaxis]
            prompt = (prompts[b] if prompts and b < len(prompts) else "A high-quality image")
            frame_scores = []
            for frame in frames:
                path = _save_frame_to_temp(frame)
                try:
                    rewards = inf.reward([prompt], [path])
                    frame_scores.append(_extract_reward_scalar(rewards[0][0]))
                finally:
                    os.remove(path)
            # Top 30% percentile.
            if frame_scores:
                k = max(1, int(len(frame_scores) * 0.3))
                top_k = sorted(frame_scores, reverse=True)[:k]
                batch_scores.append(np.mean(top_k))
            else:
                batch_scores.append(0.0)

        reward = torch.tensor(batch_scores, device=device).float()
        return {"avg": reward}, {}

    return _score
