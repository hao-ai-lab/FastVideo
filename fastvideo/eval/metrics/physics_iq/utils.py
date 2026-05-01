from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import tempfile

import cv2
import numpy as np
import torch

from fastvideo.eval.metrics.physics_iq.models import DEFAULT_DURATION_SECONDS, DEFAULT_TARGET_FPS


@dataclass(frozen=True)
class PreparedPhysicsIQPair:
    generated_quarter: np.ndarray
    reference_quarter: np.ndarray
    generated_masks: np.ndarray
    reference_masks: np.ndarray


@dataclass(frozen=True)
class PreparedPhysicsIQTriplet:
    generated_quarter: np.ndarray
    reference_quarter: np.ndarray
    reference_take2_quarter: np.ndarray
    generated_masks: np.ndarray
    reference_masks: np.ndarray
    reference_take2_masks: np.ndarray


def tensor_to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    arr = video.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def read_video_frames(
    source: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")

    frames: list[np.ndarray] = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= start_frame and (end_frame is None or frame_idx < end_frame):
            frames.append(frame)
        if end_frame is not None and frame_idx >= end_frame:
            break
        frame_idx += 1
    cap.release()

    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def as_numpy_video(source: Any) -> tuple[np.ndarray, str]:
    if isinstance(source, torch.Tensor):
        if source.ndim != 4:
            raise ValueError(f"Expected 4D video tensor (T,C,H,W), got shape {tuple(source.shape)}")
        return tensor_to_uint8_frames(source), "rgb"
    if isinstance(source, np.ndarray):
        if source.ndim != 4:
            raise ValueError(f"Expected 4D ndarray video, got shape {source.shape}")
        if source.shape[-1] == 3:
            return source.astype(np.uint8), "rgb"
        if source.shape[1] == 3:
            return np.transpose(source, (0, 2, 3, 1)).astype(np.uint8), "rgb"
        raise ValueError(f"Unsupported ndarray video shape: {source.shape}")
    if isinstance(source, (str, Path)):
        return read_video_frames(source), "bgr"
    raise TypeError(f"Unsupported Physics-IQ video source type: {type(source)!r}")


def unpack_batch_value(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 4:
            return [value]
        if value.ndim == 5:
            return [value[idx] for idx in range(value.shape[0])]
        raise ValueError(f"Unsupported tensor rank for Physics-IQ metric: {value.ndim}")
    if isinstance(value, np.ndarray):
        if value.ndim == 4:
            return [value]
        if value.ndim == 5:
            return [value[idx] for idx in range(value.shape[0])]
        raise ValueError(f"Unsupported ndarray rank for Physics-IQ metric: {value.ndim}")
    if isinstance(value, (str, Path)):
        return [value]
    if isinstance(value, list):
        return value
    raise TypeError(f"Unsupported batched Physics-IQ value type: {type(value)!r}")


def prepare_pair_batch(
    sample: dict[str, Any],
    *,
    prep_kwargs: dict[str, Any] | None = None,
) -> list[PreparedPhysicsIQPair]:
    prepared = sample.get("_physics_iq_pair")
    if prepared is not None:
        return [prepared]

    if "reference" not in sample:
        raise KeyError("Physics-IQ pair metrics require sample['reference'].")

    videos = unpack_batch_value(sample["video"])
    references = unpack_batch_value(sample["reference"])
    generated_masks = unpack_batch_value(sample["video_mask"]) if "video_mask" in sample else [None] * len(videos)
    reference_masks = unpack_batch_value(sample["reference_mask"]) if "reference_mask" in sample else [None] * len(videos)

    if not (len(videos) == len(references) == len(generated_masks) == len(reference_masks)):
        raise ValueError("Physics-IQ pair metric inputs must have the same batch size.")

    prep_kwargs = prep_kwargs or {}
    return [
        prepare_pair_inputs(
            video,
            reference,
            generated_mask=generated_mask,
            reference_mask=reference_mask,
            **prep_kwargs,
        )
        for video, reference, generated_mask, reference_mask in zip(
            videos,
            references,
            generated_masks,
            reference_masks,
        )
    ]


def select_window(frames: np.ndarray, *, target_frames: int, selection: str = "first") -> np.ndarray:
    if selection != "first":
        start = max(frames.shape[0] - target_frames, 0)
        return frames[start:start + target_frames]
    return frames[:target_frames]


def resize_frames(frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    if frames.size == 0:
        return frames
    resized = [cv2.resize(frame, target_size) for frame in frames]
    return np.stack(resized, axis=0)


def rebinarize_masks(mask_frames: np.ndarray) -> np.ndarray:
    if mask_frames.ndim == 4 and mask_frames.shape[-1] == 3:
        mask_frames = mask_frames[..., 0]
    return (mask_frames > 127).astype(np.uint8)


def load_mask_frames(
    mask_source: Any,
    *,
    target_frames: int,
    target_size: tuple[int, int],
) -> np.ndarray:
    if mask_source is None:
        raise ValueError("mask_source cannot be None when loading mask frames")
    mask_frames, _ = as_numpy_video(mask_source)
    mask_frames = select_window(mask_frames, target_frames=target_frames, selection="first")
    mask_frames = resize_frames(mask_frames, target_size)
    return rebinarize_masks(mask_frames)


def roundtrip_mask_frames(mask_frames: np.ndarray, *, fps: int) -> np.ndarray:
    if mask_frames.size == 0:
        return mask_frames
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    output_path = Path(tmp_path)
    output_path.unlink(missing_ok=True)
    try:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (mask_frames.shape[2], mask_frames.shape[1]),
            isColor=False,
        )
        for frame in mask_frames:
            writer.write(frame)
        writer.release()
        return read_video_frames(output_path)
    finally:
        output_path.unlink(missing_ok=True)


def infer_real_mask_path(video_source: Any) -> str | None:
    if not isinstance(video_source, (str, Path)):
        return None

    video_path = Path(video_source)
    filename = video_path.name
    if "_testing-videos_" not in filename:
        return None

    mask_name = filename.replace("_testing-videos_", "_video-masks_")
    candidates: list[Path] = []
    fps_dir = video_path.parent.name
    for ancestor in video_path.parents:
        if ancestor.name == "split-videos":
            candidates.extend(
                [
                    ancestor.parent / "video-masks" / "real" / fps_dir / mask_name,
                    ancestor.parent / "video_masks" / "real" / fps_dir / mask_name,
                ]
            )
            break

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def prepare_grayscale_frame(frame: np.ndarray, *, color_order: str) -> np.ndarray:
    if frame.ndim == 2:
        gray = frame
    elif color_order == "bgr":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif color_order == "rgb":
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unsupported color order: {color_order}")
    return cv2.GaussianBlur(gray, (5, 5), 0)


def generate_motion_mask(
    video_frames: np.ndarray,
    *,
    threshold: int = 10,
    alpha: float = 0.3,
    color_order: str = "rgb",
) -> np.ndarray:
    if video_frames.size == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8)

    first_gray = prepare_grayscale_frame(video_frames[0], color_order=color_order)
    avg_frame = first_gray.astype("float")
    masks = [np.zeros_like(first_gray, dtype=np.uint8)]
    kernel = np.ones((5, 5), np.uint8)

    for frame in video_frames[1:]:
        gray_frame = prepare_grayscale_frame(frame, color_order=color_order)
        cv2.accumulateWeighted(gray_frame, avg_frame, alpha)
        avg_gray_frame = cv2.convertScaleAbs(avg_frame)
        frame_diff = cv2.absdiff(gray_frame, avg_gray_frame)
        _, binary_frame = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
        masks.append(binary_frame)
    return np.stack(masks, axis=0)


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_mse(video1_frames: np.ndarray, video2_frames: np.ndarray) -> list[float]:
    if len(video1_frames) != len(video2_frames):
        raise ValueError("Videos must have the same number of frames.")
    frame_mses: list[float] = []
    for frame1, frame2 in zip(video1_frames, video2_frames):
        if frame1.shape != frame2.shape:
            raise ValueError("Frames must have the same dimensions.")
        mse = np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2)
        frame_mses.append(round(float(mse), 4))
    return frame_mses


def compute_spatiotemporal_iou(mask1_frames: np.ndarray, mask2_frames: np.ndarray) -> list[float]:
    values: list[float] = []
    for mask1, mask2 in zip(mask1_frames, mask2_frames):
        values.append(round(compute_iou(mask1, mask2), 4))
    return values


def compute_spatial_iou(mask1_frames: np.ndarray, mask2_frames: np.ndarray) -> float:
    spatial_mask1 = (np.max(mask1_frames, axis=0) > 0).astype(np.uint8) * 255
    spatial_mask2 = (np.max(mask2_frames, axis=0) > 0).astype(np.uint8) * 255
    return compute_iou(spatial_mask1, spatial_mask2)


def compute_weighted_spatial_iou(mask1_frames: np.ndarray, mask2_frames: np.ndarray) -> float:
    weighted_spatial_1 = np.sum(mask1_frames, axis=0, dtype=np.uint16) / len(mask1_frames)
    weighted_spatial_2 = np.sum(mask2_frames, axis=0, dtype=np.uint16) / len(mask2_frames)
    intersection = np.minimum(weighted_spatial_1, weighted_spatial_2)
    union = np.maximum(weighted_spatial_1, weighted_spatial_2)
    valid_pixels = union > 0
    if np.sum(valid_pixels) == 0:
        return 1.0
    return float(np.sum(intersection[valid_pixels]) / np.sum(union[valid_pixels]))


def mean(values: list[float] | tuple[float, ...]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot aggregate empty Physics-IQ values.")
    return float(arr.mean())


def quarter_resolution_target(reference_frames: np.ndarray) -> tuple[int, int]:
    return (
        max(reference_frames[0].shape[1] // 4, 1),
        max(reference_frames[0].shape[0] // 4, 1),
    )


def prepare_pair_inputs(
    generated: Any,
    reference: Any,
    *,
    generated_mask: Any | None = None,
    reference_mask: Any | None = None,
    target_fps: int = DEFAULT_TARGET_FPS,
    duration_seconds: int = DEFAULT_DURATION_SECONDS,
    video_time_selection: str = "first",
    threshold: int = 10,
    alpha: float = 0.3,
    roundtrip_generated_masks: bool = True,
) -> PreparedPhysicsIQPair:
    generated_frames, generated_color = as_numpy_video(generated)
    reference_frames, reference_color = as_numpy_video(reference)
    consider_frames = target_fps * duration_seconds

    generated_frames = select_window(
        generated_frames,
        target_frames=consider_frames,
        selection=video_time_selection,
    )
    reference_frames = reference_frames[:consider_frames]
    if not len(generated_frames) or not len(reference_frames):
        raise ValueError("Physics-IQ pair metrics require non-empty generated and reference videos.")

    target_size = quarter_resolution_target(reference_frames)
    reference_mask = reference_mask or infer_real_mask_path(reference)

    generated_quarter = resize_frames(generated_frames, target_size).astype(np.float32) / 255.0
    reference_quarter = resize_frames(reference_frames, target_size).astype(np.float32) / 255.0

    generated_masks = (
        load_mask_frames(generated_mask, target_frames=consider_frames, target_size=target_size)
        if generated_mask is not None
        else rebinarize_masks(
            resize_frames(
                roundtrip_mask_frames(
                    generate_motion_mask(
                        generated_frames,
                        threshold=threshold,
                        alpha=alpha,
                        color_order=generated_color,
                    ),
                    fps=target_fps,
                )
                if roundtrip_generated_masks
                else generate_motion_mask(
                    generated_frames,
                    threshold=threshold,
                    alpha=alpha,
                    color_order=generated_color,
                ),
                target_size,
            )
        )
    )
    reference_masks = (
        load_mask_frames(reference_mask, target_frames=consider_frames, target_size=target_size)
        if reference_mask is not None
        else rebinarize_masks(
            resize_frames(
                generate_motion_mask(
                    reference_frames,
                    threshold=threshold,
                    alpha=alpha,
                    color_order=reference_color,
                ),
                target_size,
            )
        )
    )
    return PreparedPhysicsIQPair(
        generated_quarter=generated_quarter,
        reference_quarter=reference_quarter,
        generated_masks=generated_masks,
        reference_masks=reference_masks,
    )


def prepare_triplet_inputs(
    generated: Any,
    reference: Any,
    reference_take2: Any,
    *,
    generated_mask: Any | None = None,
    reference_mask: Any | None = None,
    reference_take2_mask: Any | None = None,
    target_fps: int = DEFAULT_TARGET_FPS,
    duration_seconds: int = DEFAULT_DURATION_SECONDS,
    video_time_selection: str = "first",
    threshold: int = 10,
    alpha: float = 0.3,
    roundtrip_generated_masks: bool = True,
) -> PreparedPhysicsIQTriplet:
    pair = prepare_pair_inputs(
        generated,
        reference,
        generated_mask=generated_mask,
        reference_mask=reference_mask,
        target_fps=target_fps,
        duration_seconds=duration_seconds,
        video_time_selection=video_time_selection,
        threshold=threshold,
        alpha=alpha,
        roundtrip_generated_masks=roundtrip_generated_masks,
    )
    reference_take2_frames, reference_take2_color = as_numpy_video(reference_take2)
    consider_frames = target_fps * duration_seconds
    reference_take2_frames = reference_take2_frames[:consider_frames]
    if not len(reference_take2_frames):
        raise ValueError("Physics-IQ requires a non-empty take-2 reference video.")

    target_size = (pair.reference_quarter.shape[2], pair.reference_quarter.shape[1])
    reference_take2_mask = reference_take2_mask or infer_real_mask_path(reference_take2)
    reference_take2_quarter = resize_frames(reference_take2_frames, target_size).astype(np.float32) / 255.0
    reference_take2_masks = (
        load_mask_frames(reference_take2_mask, target_frames=consider_frames, target_size=target_size)
        if reference_take2_mask is not None
        else rebinarize_masks(
            resize_frames(
                generate_motion_mask(
                    reference_take2_frames,
                    threshold=threshold,
                    alpha=alpha,
                    color_order=reference_take2_color,
                ),
                target_size,
            )
        )
    )
    return PreparedPhysicsIQTriplet(
        generated_quarter=pair.generated_quarter,
        reference_quarter=pair.reference_quarter,
        reference_take2_quarter=reference_take2_quarter,
        generated_masks=pair.generated_masks,
        reference_masks=pair.reference_masks,
        reference_take2_masks=reference_take2_masks,
    )
