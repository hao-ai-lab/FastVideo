# SPDX-License-Identifier: Apache-2.0
"""VideoAlign reward scorers for motion, visual quality, and text alignment."""

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

_VIDEOALIGN_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "rl_rewards" / "VideoAlign"
if _VIDEOALIGN_ROOT.is_dir() and str(_VIDEOALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIDEOALIGN_ROOT))

_VIDEOALIGN_INFERENCERS: dict[str, Any] = {}
_VIDEOALIGN_PATCHED = False


def _normalize_device_str(device: torch.device | str) -> str:
    return str(torch.device(device))


def _move_videoalign_inferencer(inferencer: Any, device: torch.device | str) -> None:
    device_str = _normalize_device_str(device)
    model = getattr(inferencer, "model", None)
    if model is not None and hasattr(model, "to"):
        model.to(device)
    inferencer.device = device_str


def _remap_qwen2vl_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
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
        state_dict = _remap_qwen2vl_state_dict_keys(state_dict)
        if not assign:
            try:
                assign = any(getattr(param, "is_meta", False) for param in self.parameters())
            except Exception:
                assign = False
        return original_load_state_dict(self, state_dict, strict=strict, assign=assign)

    cls.load_state_dict = load_state_dict_with_key_remap
    cls._fastvideo_qwen2vl_key_remap = True


def _select_videoalign_frame_indices(
    vision_mod: Any,
    ele: dict[str, Any],
    total_frames: int,
    video_fps: float,
) -> list[int]:
    sample_type = ele.get("sample_type", "uniform")
    if sample_type == "uniform":
        nframes = vision_mod.smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        return torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    if sample_type == "multi_pts":
        frames_each_pts = 6
        num_pts = 4
        fps = 8
        nframes = max(frames_each_pts, int(total_frames * fps // video_fps))
        frame_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        start_pt = int(frames_each_pts // 2)
        end_pt = int(nframes - frames_each_pts // 2 - 1)
        pts = torch.linspace(start_pt, end_pt, num_pts).round().long().tolist()
        idx = []
        for pt in pts:
            idx.extend(frame_idx[pt - frames_each_pts // 2:pt + frames_each_pts // 2])
        return idx
    raise ValueError(f"Unsupported VideoAlign sample_type: {sample_type}")


def _read_video_opencv(vision_mod: Any, ele: dict[str, Any]) -> torch.Tensor:
    video_path = ele["video"]
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_file_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(float(ele.get("video_start", 0.0) or 0.0) * video_fps)))
    end_sec = ele.get("video_end")
    if end_sec is None:
        end_frame = total_file_frames if total_file_frames > 0 else None
    else:
        end_frame = int(round(float(end_sec) * video_fps))
        if total_file_frames > 0:
            end_frame = min(end_frame, total_file_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    current_frame = start_frame
    while end_frame is None or current_frame < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current_frame += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames were read from video: {video_path}.")

    idx = _select_videoalign_frame_indices(vision_mod, ele, total_frames=len(frames), video_fps=video_fps)
    video = np.stack([frames[i] for i in idx], axis=0)
    return torch.from_numpy(video).permute(0, 3, 1, 2)


def _torchvision_read_video_available() -> bool:
    try:
        torchvision_io = import_module("torchvision.io")
    except Exception:
        return False
    return hasattr(torchvision_io, "read_video")


def _patch_videoalign_video_reader() -> None:
    vision_mod = import_module("vision_process")
    if "opencv" not in vision_mod.VIDEO_READER_BACKENDS:

        def read_video_opencv(ele):
            return _read_video_opencv(vision_mod, ele)

        vision_mod.VIDEO_READER_BACKENDS["opencv"] = read_video_opencv
    if _torchvision_read_video_available():
        return
    vision_mod.__dict__["FORCE_QWENVL_VIDEO_READER"] = "opencv"
    if hasattr(vision_mod.get_video_reader_backend, "cache_clear"):
        vision_mod.get_video_reader_backend.cache_clear()


def _patch_videoalign_modules() -> Any:
    global _VIDEOALIGN_PATCHED
    inference_mod = import_module("inference")
    if _VIDEOALIGN_PATCHED:
        return inference_mod

    train_reward_mod = import_module("train_reward")
    trainer_mod = import_module("trainer")
    _patch_videoalign_video_reader()
    if util.find_spec("flash_attn") is None:
        for mod in (train_reward_mod, inference_mod):
            original_create = mod.__dict__["create_model_and_processor"]

            def create_model_and_processor_sdpa(*args, _original_create=original_create, **kwargs):
                training_args = kwargs.get("training_args")
                if training_args is not None:
                    training_args.disable_flash_attn2 = True
                return _original_create(*args, **kwargs)

            mod.__dict__["create_model_and_processor"] = create_model_and_processor_sdpa

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
    for candidate in _walk_model_graph(model):
        language_model = getattr(candidate, "language_model", None)
        if language_model is not None and not hasattr(candidate, "embed_tokens") and hasattr(language_model,
                                                                                            "embed_tokens"):
            candidate.embed_tokens = language_model.embed_tokens


def set_videoalign_device(device: torch.device | str) -> None:
    key = _normalize_device_str(device)
    for old_key, inferencer in list(_VIDEOALIGN_INFERENCERS.items()):
        if old_key != key and old_key.split(":")[-1] != key:
            new_key = f"{inferencer._key_prefix}:{key}"
            _move_videoalign_inferencer(inferencer, device)
            _VIDEOALIGN_INFERENCERS[new_key] = inferencer
            del _VIDEOALIGN_INFERENCERS[old_key]


def _get_inferencer(
    device: torch.device | str,
    checkpoint_path: str | None = None,
) -> Any:
    if checkpoint_path is None:
        checkpoint_path = os.environ.get(
            "VIDEOALIGN_CHECKPOINT_PATH",
            str(_VIDEOALIGN_ROOT / "checkpoints"),
        )
    checkpoint_path = os.path.abspath(checkpoint_path)
    key = _normalize_device_str(device)
    cache_key = f"{checkpoint_path}:{key}"
    if cache_key not in _VIDEOALIGN_INFERENCERS:
        try:
            inference_mod = _patch_videoalign_modules()
            video_reward_cls = inference_mod.VideoVLMRewardInference
        except ImportError as exc:
            raise ImportError("VideoAlign rewards require the VideoAlign runtime files under "
                              "fastvideo/third_party/rl_rewards/VideoAlign and a "
                              "VIDEOALIGN_CHECKPOINT_PATH checkpoint.") from exc
        inferencer = video_reward_cls(load_from_pretrained=checkpoint_path, device=device)
        _patch_videoalign_runtime_model(inferencer.model)
        inferencer._key_prefix = checkpoint_path or "default"
        _VIDEOALIGN_INFERENCERS[cache_key] = inferencer
    return _VIDEOALIGN_INFERENCERS[cache_key]


def _convert_to_grayscale(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 4 and frames.shape[-1] == 3:
        gray = np.mean(frames, axis=-1, keepdims=True)
        return np.repeat(gray.astype(np.uint8), 3, axis=-1)
    return frames


def _save_video_to_temp(frames: np.ndarray, fps: int = 8) -> str:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


class _VideoAlignScorer:
    score_key: str

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        checkpoint_path: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

    def _prompt(self, prompts, index: int) -> str:
        return prompts[index] if prompts and index < len(prompts) else ""

    def _frames(self, frames: np.ndarray) -> np.ndarray:
        return frames

    @torch.no_grad()
    def __call__(self, media: torch.Tensor, prompts) -> torch.Tensor:
        inferencer = _get_inferencer(self.device, self.checkpoint_path)
        images_np = media_to_uint8_array(media)
        batch_scores = []
        for sample_idx, sample in enumerate(images_np):
            frames = sample[np.newaxis] if sample.ndim == 3 else sample
            path = _save_video_to_temp(self._frames(frames))
            try:
                results = inferencer.reward([path], [self._prompt(prompts, sample_idx)], use_norm=True)
                batch_scores.append(float(results[0].get(self.score_key, 0.0)))
            finally:
                os.remove(path)
        return torch.tensor(batch_scores, device=self.device, dtype=torch.float32)


class VideoAlignMotionQualityScorer(_VideoAlignScorer):
    score_key = "MQ"

    def _frames(self, frames: np.ndarray) -> np.ndarray:
        return _convert_to_grayscale(frames)

    def _prompt(self, prompts, index: int) -> str:
        del prompts, index
        return ""


class VideoAlignVisualQualityScorer(_VideoAlignScorer):
    score_key = "VQ"

    def _prompt(self, prompts, index: int) -> str:
        del prompts, index
        return ""


class VideoAlignTextAlignmentScorer(_VideoAlignScorer):
    score_key = "TA"
