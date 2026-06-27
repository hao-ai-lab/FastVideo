# SPDX-License-Identifier: Apache-2.0
"""HPSv3 reward scorers for RL training."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastvideo.train.methods.rl.rewards.media import media_to_uint8_array

_HPSV3_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "rl_rewards" / "HPSv3"
if _HPSV3_ROOT.exists() and str(_HPSV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_HPSV3_ROOT))

_HPSV3_INFERENCERS: dict[str, Any] = {}
_HPSV3_LOAD_PATCHED = False


def _patch_transformers_video_input_alias() -> None:
    from transformers import image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = image_utils.ImageInput


def _remap_hpsv3_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("visual."):
            key = f"model.{key}"
        elif key.startswith("model.layers.") or key.startswith("model.embed_tokens.") or key.startswith("model.norm."):
            key = f"model.language_model.{key[len('model.'):]}"
        key = key.replace("base_model.model.visual.", "base_model.model.model.visual.", 1)
        key = key.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers.", 1)
        key = key.replace("base_model.model.model.embed_tokens.",
                          "base_model.model.model.language_model.embed_tokens.", 1)
        key = key.replace("base_model.model.model.norm.", "base_model.model.model.language_model.norm.", 1)
        remapped[key] = value
    return remapped


def _walk_model_graph(model: Any):
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
    if getattr(cls, "_fastvideo_qwen2vl_key_remap", False):
        return
    original_load_state_dict = cls.load_state_dict

    def load_state_dict_with_key_remap(self, state_dict, strict=True, assign=False):
        state_dict = _remap_hpsv3_state_dict(state_dict)
        return original_load_state_dict(self, state_dict, strict=strict, assign=assign)

    cls.load_state_dict = load_state_dict_with_key_remap
    cls._fastvideo_qwen2vl_key_remap = True


def _patch_hpsv3_state_dict_loader() -> None:
    global _HPSV3_LOAD_PATCHED
    if _HPSV3_LOAD_PATCHED:
        return
    from hpsv3.model.qwen2vl_trainer import Qwen2VLRewardModelBT

    _patch_load_state_dict(Qwen2VLRewardModelBT)
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None
    if PeftModel is not None:
        _patch_load_state_dict(PeftModel)
    _HPSV3_LOAD_PATCHED = True


def _patch_hpsv3_runtime_model(model: Any) -> None:
    for candidate in _walk_model_graph(model):
        language_model = getattr(candidate, "language_model", None)
        if language_model is not None and not hasattr(candidate, "embed_tokens") and hasattr(language_model,
                                                                                            "embed_tokens"):
            candidate.embed_tokens = language_model.embed_tokens


def _normalize_device(device: torch.device | str) -> str:
    return str(torch.device(device))


def _move_hpsv3_inferencer(inferencer: Any, device: torch.device | str) -> None:
    device_str = _normalize_device(device)
    model = getattr(inferencer, "model", None)
    if model is not None and hasattr(model, "to"):
        model.to(device)
    inferencer.device = device_str


def set_hpsv3_device(device: torch.device | str) -> None:
    key = _normalize_device(device)
    if key in _HPSV3_INFERENCERS:
        return
    for old_key, inferencer in list(_HPSV3_INFERENCERS.items()):
        if old_key != key:
            _move_hpsv3_inferencer(inferencer, device)
            _HPSV3_INFERENCERS[key] = inferencer
            del _HPSV3_INFERENCERS[old_key]
            return


def _get_hpsv3_inferencer(device: torch.device | str) -> Any:
    key = _normalize_device(device)
    if key not in _HPSV3_INFERENCERS:
        try:
            _patch_transformers_video_input_alias()
            from hpsv3 import HPSv3RewardInferencer
            _patch_hpsv3_state_dict_loader()
        except ImportError as exc:
            raise ImportError("HPSv3 rewards require the HPSv3 package or the vendored "
                              "fastvideo/third_party/rl_rewards/HPSv3 directory.") from exc
        inferencer = HPSv3RewardInferencer(device=device)
        _patch_hpsv3_runtime_model(inferencer.model)
        _HPSV3_INFERENCERS[key] = inferencer
    return _HPSV3_INFERENCERS[key]


def _save_frame_to_temp(frame: np.ndarray) -> str:
    from PIL import Image

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.fromarray(frame).save(path)
    return path


def _extract_reward_scalar(result: Any) -> float:
    if isinstance(result, torch.Tensor):
        return float(result.item())
    if isinstance(result, float | int):
        return float(result)
    if isinstance(result, list | np.ndarray):
        return float(np.mean(result))
    return float(result)


class HPSv3GeneralScorer:
    """Score every frame with a generic quality prompt and average."""

    def __init__(self, *, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, media: torch.Tensor, prompts) -> torch.Tensor:
        del prompts
        inferencer = _get_hpsv3_inferencer(self.device)
        images_np = media_to_uint8_array(media)
        batch_scores = []
        for sample in images_np:
            frames = sample[np.newaxis] if sample.ndim == 3 else sample
            frame_scores = []
            for frame in frames:
                path = _save_frame_to_temp(frame)
                try:
                    rewards = inferencer.reward(["A high-quality image"], [path])
                    frame_scores.append(_extract_reward_scalar(rewards[0][0]))
                finally:
                    os.remove(path)
            batch_scores.append(float(np.mean(frame_scores)))
        return torch.tensor(batch_scores, device=self.device, dtype=torch.float32)


class HPSv3PercentileScorer:
    """Score frames with per-prompt text and average the top 30 percent."""

    def __init__(self, *, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, media: torch.Tensor, prompts) -> torch.Tensor:
        inferencer = _get_hpsv3_inferencer(self.device)
        images_np = media_to_uint8_array(media)
        batch_scores = []
        for sample_idx, sample in enumerate(images_np):
            frames = sample[np.newaxis] if sample.ndim == 3 else sample
            prompt = prompts[sample_idx] if sample_idx < len(prompts) else "A high-quality image"
            frame_scores = []
            for frame in frames:
                path = _save_frame_to_temp(frame)
                try:
                    rewards = inferencer.reward([prompt], [path])
                    frame_scores.append(_extract_reward_scalar(rewards[0][0]))
                finally:
                    os.remove(path)
            if not frame_scores:
                batch_scores.append(0.0)
                continue
            top_k = max(1, int(len(frame_scores) * 0.3))
            batch_scores.append(float(np.mean(sorted(frame_scores, reverse=True)[:top_k])))
        return torch.tensor(batch_scores, device=self.device, dtype=torch.float32)
