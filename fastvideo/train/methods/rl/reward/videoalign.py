# SPDX-License-Identifier: Apache-2.0
"""VideoAlign reward functions for motion quality and
text-video alignment."""

from __future__ import annotations

import os
import tempfile
from importlib import import_module, util
from typing import Any

import numpy as np
import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.rl.reward.utils import (
    prepare_images, )

logger = init_logger(__name__)

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


def _remap_qwen2vl_state_dict_keys(state_dict: dict[str, Any], ) -> dict[str, Any]:
    """Adapt checkpoints saved with older Qwen2-VL key names."""
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


def _select_videoalign_frame_indices(
    vision_mod: Any,
    ele: dict[str, Any],
    total_frames: int,
    video_fps: float,
) -> list[int]:
    sample_type = ele.get("sample_type", "uniform")
    if sample_type == "uniform":
        nframes = vision_mod.smart_nframes(
            ele,
            total_frames=total_frames,
            video_fps=video_fps,
        )
        return torch.linspace(
            0,
            total_frames - 1,
            nframes,
        ).round().long().tolist()
    if sample_type == "multi_pts":
        frames_each_pts = 6
        num_pts = 4
        fps = 8
        nframes = max(
            frames_each_pts,
            int(total_frames * fps // video_fps),
        )
        frame_idx = torch.linspace(
            0,
            total_frames - 1,
            nframes,
        ).round().long().tolist()
        start_pt = int(frames_each_pts // 2)
        end_pt = int(nframes - frames_each_pts // 2)
        pts = torch.linspace(
            start_pt,
            end_pt,
            num_pts,
        ).round().long().tolist()
        idx = []
        for pt in pts:
            idx.extend(frame_idx[pt - frames_each_pts // 2:pt + frames_each_pts // 2])
        return idx
    raise ValueError(f"Unsupported VideoAlign sample_type: {sample_type}")


def _read_video_opencv(
    vision_mod: Any,
    ele: dict[str, Any],
) -> torch.Tensor:
    """Read local MP4s without relying on torchvision.io.read_video."""
    import cv2

    video_path = ele["video"]
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_file_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(
        0,
        int(round(float(ele.get("video_start", 0.0) or 0.0) * video_fps)),
    )
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

    idx = _select_videoalign_frame_indices(
        vision_mod,
        ele,
        total_frames=len(frames),
        video_fps=video_fps,
    )
    video = np.stack([frames[i] for i in idx], axis=0)
    return torch.from_numpy(video).permute(0, 3, 1, 2)


def _torchvision_read_video_available() -> bool:
    try:
        torchvision_io = import_module("torchvision.io")
    except Exception:
        return False
    return hasattr(torchvision_io, "read_video")


def _patch_videoalign_video_reader() -> None:
    """Register an OpenCV reader for torchvision builds without read_video."""
    from fastvideo.train.methods.rl.reward.VideoAlign import vision_process as vision_mod

    if "opencv" not in vision_mod.VIDEO_READER_BACKENDS:

        def read_video_opencv(ele):
            return _read_video_opencv(vision_mod, ele)

        vision_mod.VIDEO_READER_BACKENDS["opencv"] = read_video_opencv

    if _torchvision_read_video_available():
        return

    vision_mod.FORCE_QWENVL_VIDEO_READER = "opencv"
    if hasattr(vision_mod.get_video_reader_backend, "cache_clear"):
        vision_mod.get_video_reader_backend.cache_clear()


def _patch_videoalign_modules() -> Any:
    """Patch VideoAlign for the FastVideo dependency set."""
    global _VIDEOALIGN_PATCHED

    from fastvideo.train.methods.rl.reward.VideoAlign import inference as inference_mod

    if _VIDEOALIGN_PATCHED:
        return inference_mod

    from fastvideo.train.methods.rl.reward.VideoAlign import train_reward as train_reward_mod
    from fastvideo.train.methods.rl.reward.VideoAlign import trainer as trainer_mod

    _patch_videoalign_video_reader()

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
    for candidate in _walk_model_graph(model):
        language_model = getattr(candidate, "language_model", None)
        if (language_model is not None and not hasattr(candidate, "embed_tokens")
                and hasattr(language_model, "embed_tokens")):
            candidate.__dict__["embed_tokens"] = language_model.embed_tokens


def set_videoalign_device(device) -> None:
    """Move cached VideoAlign inferencers to device."""
    key = _normalize_device_str(device)
    for old_key, inf in list(_VIDEOALIGN_INFERENCERS.items()):
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
            msg = ("VideoAlign not found. Ensure the "
                   "VideoAlign submodule is checked out "
                   "under fastvideo/train/methods/rl/"
                   "reward/VideoAlign")
            raise ImportError(msg) from exc

        inf = VideoVLMRewardInference(
            load_from_pretrained=checkpoint_path,
            device=device,
        )
        _patch_videoalign_runtime_model(inf.model)
        inf._key_prefix = checkpoint_path or "default"
        _VIDEOALIGN_INFERENCERS[cache_key] = inf
    return _VIDEOALIGN_INFERENCERS[cache_key]


def _convert_to_grayscale(frames: np.ndarray, ) -> np.ndarray:
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
                results = inf.reward([path], [""], use_norm=True)
                mq = float(results[0].get("MQ", 0))
                batch_scores.append(mq)
            finally:
                os.remove(path)

        reward = torch.tensor(batch_scores, device=device).float()
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
            prompt = (prompts[b] if prompts and b < len(prompts) else "")
            path = _save_video_to_temp(frames)
            try:
                results = inf.reward([path], [prompt], use_norm=True)
                ta = float(results[0].get("TA", 0))
                batch_scores.append(ta)
            finally:
                os.remove(path)

        reward = torch.tensor(batch_scores, device=device).float()
        return {"avg": reward}, {}

    return _score
