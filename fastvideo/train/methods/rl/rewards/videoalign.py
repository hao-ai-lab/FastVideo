# SPDX-License-Identifier: Apache-2.0
"""VideoAlign reward scorers for motion quality and text alignment."""

from __future__ import annotations

import os
import sys
import tempfile
from importlib import import_module, util
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from fastvideo.train.methods.rl.rewards.media import media_to_uint8_array

_DEFAULT_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "rl_rewards" / "VideoAlign"
_VIDEOALIGN_ROOT = Path(
    os.environ.get("VIDEOALIGN_RUNTIME_PATH")
    or os.environ.get("VIDEOALIGN_ROOT")
    or _DEFAULT_ROOT
)
if _VIDEOALIGN_ROOT.is_dir() and str(_VIDEOALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIDEOALIGN_ROOT))

_INFERENCERS: dict[str, Any] = {}
_PATCHED = False


def _remap_qwen2vl_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Bridge VideoAlign's Qwen2-VL checkpoints across transformers layouts."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("visual."):
            key = f"model.{key}"
        elif key.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
            key = f"model.language_model.{key[len('model.') :]}"
        key = key.replace("base_model.model.visual.", "base_model.model.model.visual.", 1)
        key = key.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers.", 1)
        key = key.replace(
            "base_model.model.model.embed_tokens.",
            "base_model.model.model.language_model.embed_tokens.",
            1,
        )
        key = key.replace("base_model.model.model.norm.", "base_model.model.model.language_model.norm.", 1)
        remapped[key] = value
    return remapped


def _patch_load_state_dict(cls: Any) -> None:
    if getattr(cls, "_fastvideo_qwen2vl_key_remap", False):
        return
    original = cls.load_state_dict

    def wrapped(self, state_dict, strict=True, assign=False):
        state_dict = _remap_qwen2vl_state_dict_keys(state_dict)
        if not assign:
            try:
                assign = any(getattr(param, "is_meta", False) for param in self.parameters())
            except Exception:
                assign = False
        return original(self, state_dict, strict=strict, assign=assign)

    cls.load_state_dict = wrapped
    cls._fastvideo_qwen2vl_key_remap = True


def _select_frame_indices(vision_mod: Any, ele: dict[str, Any], total_frames: int, fps: float) -> list[int]:
    sample_type = ele.get("sample_type", "uniform")
    if sample_type == "uniform":
        nframes = vision_mod.smart_nframes(ele, total_frames=total_frames, video_fps=fps)
        return torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    if sample_type == "multi_pts":
        frames_each_pts = 6
        num_pts = 4
        target_fps = 8
        nframes = max(frames_each_pts, int(total_frames * target_fps // fps))
        frame_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        start_pt = frames_each_pts // 2
        end_pt = nframes - frames_each_pts // 2 - 1
        points = torch.linspace(start_pt, end_pt, num_pts).round().long().tolist()
        selected: list[int] = []
        for point in points:
            selected.extend(frame_idx[point - frames_each_pts // 2 : point + frames_each_pts // 2])
        return selected
    raise ValueError(f"Unsupported VideoAlign sample_type: {sample_type}")


def _read_video_opencv(vision_mod: Any, ele: dict[str, Any]) -> torch.Tensor:
    path = str(ele["video"])
    if path.startswith("file://"):
        path = path[7:]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames were read from video: {path}")
    indices = _select_frame_indices(vision_mod, ele, len(frames), fps)
    return torch.from_numpy(np.stack([frames[index] for index in indices])).permute(0, 3, 1, 2)


def _patch_runtime() -> Any:
    global _PATCHED
    inference_mod = import_module("inference")
    if _PATCHED:
        return inference_mod
    train_reward_mod = import_module("train_reward")
    trainer_mod = import_module("trainer")
    vision_mod = import_module("vision_process")
    if "opencv" not in vision_mod.VIDEO_READER_BACKENDS:
        vision_mod.VIDEO_READER_BACKENDS["opencv"] = lambda ele: _read_video_opencv(vision_mod, ele)
    if util.find_spec("torchvision.io") is None or not hasattr(import_module("torchvision.io"), "read_video"):
        vision_mod.__dict__["FORCE_QWENVL_VIDEO_READER"] = "opencv"
        if hasattr(vision_mod.get_video_reader_backend, "cache_clear"):
            vision_mod.get_video_reader_backend.cache_clear()
    if util.find_spec("flash_attn") is None:
        for module in (train_reward_mod, inference_mod):
            original_create = module.__dict__["create_model_and_processor"]

            def create_sdpa(*args, _original=original_create, **kwargs):
                training_args = kwargs.get("training_args")
                if training_args is not None:
                    training_args.disable_flash_attn2 = True
                return _original(*args, **kwargs)

            module.__dict__["create_model_and_processor"] = create_sdpa
    _patch_load_state_dict(trainer_mod.Qwen2VLRewardModelBT)
    try:
        peft_mod = import_module("peft")
    except ImportError:
        peft_mod = None
    if peft_mod is not None:
        _patch_load_state_dict(peft_mod.PeftModel)
    _PATCHED = True
    return inference_mod


def _get_inferencer(device: torch.device, checkpoint_path: str | None) -> Any:
    checkpoint = os.path.abspath(
        checkpoint_path
        or os.environ.get("VIDEOALIGN_CHECKPOINT_PATH")
        or str(_VIDEOALIGN_ROOT / "checkpoints")
    )
    key = f"{checkpoint}:{device}"
    if key not in _INFERENCERS:
        try:
            cls = _patch_runtime().VideoVLMRewardInference
        except ImportError as exc:
            raise ImportError(
                "VideoAlign requires VIDEOALIGN_RUNTIME_PATH to point to its source checkout and "
                "VIDEOALIGN_CHECKPOINT_PATH to its downloaded VideoReward checkpoint."
            ) from exc
        _INFERENCERS[key] = cls(load_from_pretrained=checkpoint, device=str(device))
    return _INFERENCERS[key]


def _write_mp4(frames: np.ndarray, *, fps: int) -> str:
    handle, path = tempfile.mkstemp(suffix=".mp4")
    os.close(handle)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        os.remove(path)
        raise RuntimeError("OpenCV could not create a temporary MP4 for VideoAlign")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def _grayscale(frames: np.ndarray) -> np.ndarray:
    gray = np.dot(frames[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


class _VideoAlignScorer:
    score_key: str

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        checkpoint_path: str | None = None,
        fps: int = 8,
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.fps = int(fps)

    def _frames(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def _prompt(self, prompt: str) -> str:
        return prompt

    @torch.no_grad()
    def __call__(self, media: torch.Tensor, prompts) -> torch.Tensor:
        inferencer = _get_inferencer(self.device, self.checkpoint_path)
        samples = media_to_uint8_array(media)
        paths: list[str] = []
        try:
            for sample in samples:
                frames = sample[None] if sample.ndim == 3 else sample
                paths.append(_write_mp4(self._frames(frames), fps=self.fps))
            score_prompts = [self._prompt(str(prompt)) for prompt in prompts]
            results = inferencer.reward(paths, score_prompts, use_norm=True)
            values = [float(result.get(self.score_key, 0.0)) for result in results]
            return torch.tensor(values, device=self.device, dtype=torch.float32)
        finally:
            for path in paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass


class VideoAlignMotionQualityScorer(_VideoAlignScorer):
    score_key = "MQ"

    def _frames(self, frames: np.ndarray) -> np.ndarray:
        return _grayscale(frames)

    def _prompt(self, prompt: str) -> str:
        del prompt
        return ""


class VideoAlignTextAlignmentScorer(_VideoAlignScorer):
    score_key = "TA"


class VideoAlignVisualQualityScorer(_VideoAlignScorer):
    score_key = "VQ"

    def _prompt(self, prompt: str) -> str:
        del prompt
        return ""


__all__ = [
    "VideoAlignMotionQualityScorer",
    "VideoAlignTextAlignmentScorer",
    "VideoAlignVisualQualityScorer",
]
