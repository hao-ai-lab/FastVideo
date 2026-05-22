# SPDX-License-Identifier: Apache-2.0
"""VideoAlign reward functions for motion quality and
text-video alignment."""

from __future__ import annotations

import os
import sys
import tempfile
from importlib import import_module, util
from typing import Any

import numpy as np
import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.rl.reward.utils import (
    prepare_images,
)

logger = init_logger(__name__)

# Add VideoAlign submodule to path for importing.
_VIDEOALIGN_ROOT = os.path.join(
    os.path.dirname(__file__), "VideoAlign"
)
if os.path.isdir(_VIDEOALIGN_ROOT):
    if _VIDEOALIGN_ROOT not in sys.path:
        sys.path.insert(0, _VIDEOALIGN_ROOT)

# Global cache of VideoAlign inferencers.
_VIDEOALIGN_INFERENCERS: dict[str, Any] = {}
_VIDEOALIGN_PATCHED = False


def _normalize_device_str(device) -> str:
    if isinstance(device, torch.device):
        return str(device)
    return str(torch.device(device))


def _move_videoalign_inferencer(inferencer: Any, device) -> None:
    """Move a VideoAlign inferencer across devices."""
    device_str = _normalize_device_str(device)
    model = getattr(inferencer, "model", None)
    if model is not None and hasattr(model, "to"):
        model.to(device)
    inferencer.device = device_str


def _remap_qwen2vl_state_dict_keys(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Adapt checkpoints saved with older Qwen2-VL key names."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("visual."):
            key = f"model.{key}"
        elif key.startswith("model.layers."):
            key = f"model.language_model.{key[len('model.'):]}"
        elif key.startswith("model.embed_tokens."):
            key = f"model.language_model.{key[len('model.'):]}"
        elif key.startswith("model.norm."):
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


def _patch_load_state_dict(cls: Any) -> None:
    """Patch a model class to accept old VideoAlign checkpoint keys."""
    if getattr(cls, "_fastvideo_qwen2vl_key_remap", False):
        return

    original_load_state_dict = cls.load_state_dict

    def load_state_dict_with_key_remap(
        self,
        state_dict,
        strict=True,
        assign=False,
    ):
        state_dict = _remap_qwen2vl_state_dict_keys(state_dict)
        return original_load_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
        )

    cls.load_state_dict = load_state_dict_with_key_remap
    cls._fastvideo_qwen2vl_key_remap = True


def _patch_videoalign_modules() -> Any:
    """Patch VideoAlign for the FastVideo dependency set."""
    global _VIDEOALIGN_PATCHED

    inference_mod = import_module("inference")
    if _VIDEOALIGN_PATCHED:
        return inference_mod

    train_reward_mod = import_module("train_reward")
    trainer_mod = import_module("trainer")

    if util.find_spec("flash_attn") is None:
        for mod in (train_reward_mod, inference_mod):
            original_create = mod.create_model_and_processor

            def create_model_and_processor_sdpa(
                *args,
                _original_create=original_create,
                **kwargs,
            ):
                training_args = kwargs.get("training_args")
                if training_args is not None:
                    training_args.disable_flash_attn2 = True
                return _original_create(*args, **kwargs)

            mod.create_model_and_processor = create_model_and_processor_sdpa

    _patch_load_state_dict(trainer_mod.Qwen2VLRewardModelBT)
    try:
        peft_mod = import_module("peft")
    except ImportError:
        peft_mod = None
    if peft_mod is not None:
        _patch_load_state_dict(peft_mod.PeftModel)

    _VIDEOALIGN_PATCHED = True
    return inference_mod


def _patch_videoalign_runtime_model(model: Any) -> None:
    """Add aliases expected by VideoAlign's older Qwen2-VL forward."""
    inner = getattr(model, "model", None)
    language_model = getattr(inner, "language_model", None)
    if (
        inner is not None
        and language_model is not None
        and not hasattr(inner, "embed_tokens")
        and hasattr(language_model, "embed_tokens")
    ):
        inner.embed_tokens = language_model.embed_tokens


def set_videoalign_device(device) -> None:
    """Move cached VideoAlign inferencers to device."""
    key = _normalize_device_str(device)
    for old_key, inf in list(
        _VIDEOALIGN_INFERENCERS.items()
    ):
        if old_key != key and old_key.split(":")[0] != key:
            new_key = inf._key_prefix + ":" + key
            _move_videoalign_inferencer(inf, device)
            _VIDEOALIGN_INFERENCERS[new_key] = inf
            del _VIDEOALIGN_INFERENCERS[old_key]


def _get_inferencer(
    device,
    checkpoint_path: str | None = None,
):
    """Get or create VideoAlign inferencer."""
    if checkpoint_path is None:
        checkpoint_path = os.environ.get(
            "VIDEOALIGN_CHECKPOINT_PATH",
            os.path.join(
                os.path.dirname(__file__),
                "VideoAlign",
                "checkpoints",
            ),
        )
    checkpoint_path = os.path.abspath(checkpoint_path)

    key = _normalize_device_str(device)
    cache_key = f"{checkpoint_path}:{key}"
    if cache_key not in _VIDEOALIGN_INFERENCERS:
        try:
            inference_mod = _patch_videoalign_modules()
            VideoVLMRewardInference = inference_mod.VideoVLMRewardInference
        except ImportError as exc:
            msg = (
                "VideoAlign not found. Ensure the "
                "VideoAlign submodule is checked out "
                "under fastvideo/train/methods/rl/"
                "reward/VideoAlign"
            )
            raise ImportError(msg) from exc

        inf = VideoVLMRewardInference(
            load_from_pretrained=checkpoint_path,
            device=device,
        )
        _patch_videoalign_runtime_model(inf.model)
        inf._key_prefix = checkpoint_path or "default"
        _VIDEOALIGN_INFERENCERS[cache_key] = inf
    return _VIDEOALIGN_INFERENCERS[cache_key]


def _convert_to_grayscale(
    frames: np.ndarray,
) -> np.ndarray:
    """Convert FHWC frames to grayscale FHWC."""
    if frames.ndim == 4 and frames.shape[-1] == 3:
        gray = np.mean(frames, axis=-1, keepdims=True)
        return np.repeat(gray.astype(np.uint8), 3, axis=-1)
    return frames


def _save_video_to_temp(
    frames: np.ndarray,
    fps: int = 8,
) -> str:
    """Save frames to a temporary MP4 file."""
    import cv2

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    return path


def videoalign_mq_score(
    device,
    checkpoint_path: str | None = None,
):
    """Return Motion Quality reward fn (grayscale)."""

    def _score(images, prompts, metadata, only_strict=False):
        inf = _get_inferencer(device, checkpoint_path)
        images_np = prepare_images(images)
        batch_scores = []

        for b in range(len(images_np)):
            frames = images_np[b]
            if frames.ndim == 3:
                frames = frames[np.newaxis]
            gray_frames = _convert_to_grayscale(frames)
            path = _save_video_to_temp(gray_frames)
            try:
                results = inf.reward(
                    [path], [""], use_norm=True
                )
                mq = float(results[0].get("MQ", 0))
                batch_scores.append(mq)
            finally:
                os.remove(path)

        reward = torch.tensor(
            batch_scores, device=device
        ).float()
        return {"avg": reward}, {}

    return _score


def videoalign_ta_score(
    device,
    checkpoint_path: str | None = None,
):
    """Return Text-Video Alignment reward fn (color)."""

    def _score(images, prompts, metadata, only_strict=False):
        inf = _get_inferencer(device, checkpoint_path)
        images_np = prepare_images(images)
        batch_scores = []

        for b in range(len(images_np)):
            frames = images_np[b]
            if frames.ndim == 3:
                frames = frames[np.newaxis]
            prompt = (
                prompts[b] if prompts and b < len(prompts)
                else ""
            )
            path = _save_video_to_temp(frames)
            try:
                results = inf.reward(
                    [path], [prompt], use_norm=True
                )
                ta = float(results[0].get("TA", 0))
                batch_scores.append(ta)
            finally:
                os.remove(path)

        reward = torch.tensor(
            batch_scores, device=device
        ).float()
        return {"avg": reward}, {}

    return _score
