# SPDX-License-Identifier: Apache-2.0
"""Track-conditioned validation shared by bidirectional and causal WanTrack."""

from __future__ import annotations

import colorsys
import glob
import os
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw

from fastvideo.distributed import get_world_group
from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.models.wantrack.inference import (
    prepare_wantrack_batch,
    sample_wantrack,
)
from fastvideo.training.trackers import DummyTracker

logger = init_logger(__name__)


def _track_colors(count: int) -> np.ndarray:
    colors = np.empty((count, 3), dtype=np.uint8)
    for index in range(count):
        hue = index / max(count, 1)
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        colors[index] = (
            int(red * 255),
            int(green * 255),
            int(blue * 255),
        )
    return colors


def _overlay_tracks(
    frames: np.ndarray,
    track_points: torch.Tensor,
    track_visibility: torch.Tensor,
    *,
    stride: int,
    tail: int,
    radius: int,
    visibility_threshold: float,
) -> list[np.ndarray]:
    points = track_points[0].float().cpu().numpy()
    visibility = track_visibility[0].float().cpu().numpy()
    frame_count = min(
        int(frames.shape[0]),
        int(points.shape[0]),
        int(visibility.shape[0]),
    )
    frames = frames[:frame_count]
    points = points[:frame_count, ::stride]
    visibility = visibility[:frame_count, ::stride]
    colors = _track_colors(int(points.shape[1]))

    height, width = int(frames.shape[1]), int(frames.shape[2])
    points = points.copy()
    points[..., 0] *= width
    points[..., 1] *= height

    output: list[np.ndarray] = []
    for frame_index, frame in enumerate(frames):
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        start = max(0, frame_index - tail)
        for track_index, color_value in enumerate(colors):
            color = tuple(int(value) for value in color_value)
            trail: list[tuple[float, float]] = []
            for history_index in range(start, frame_index + 1):
                if visibility[history_index, track_index] < visibility_threshold:
                    trail = []
                    continue
                x, y = points[history_index, track_index]
                if 0 <= x < width and 0 <= y < height:
                    trail.append((float(x), float(y)))
            if len(trail) >= 2:
                draw.line(trail, fill=color, width=1)
            if visibility[frame_index, track_index] >= visibility_threshold:
                x, y = points[frame_index, track_index]
                if 0 <= x < width and 0 <= y < height:
                    draw.ellipse(
                        (
                            float(x - radius),
                            float(y - radius),
                            float(x + radius),
                            float(y + radius),
                        ),
                        fill=color,
                    )
        output.append(np.asarray(image))
    return output


class TrackValidationCallback(Callback):
    """Generate fixed track-conditioned samples through the live student."""

    def __init__(
        self,
        *,
        every_steps: int = 250,
        val_data_path: str | None = None,
        num_val_samples: int = 1,
        num_inference_steps: int = 30,
        guidance_scale: float = 1.0,
        motion_guidance_scale: float = 1.0,
        output_dir: str | None = None,
        fps: int = 24,
        grid_stride: int = 3,
        tail: int = 12,
        radius: int = 2,
        visibility_threshold: float = 0.5,
        seed: int = 1000,
        validate_at_start: bool = False,
    ) -> None:
        self.every_steps = int(every_steps)
        self.val_data_path = (str(val_data_path) if val_data_path is not None else None)
        self.num_val_samples = int(num_val_samples)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.motion_guidance_scale = float(motion_guidance_scale)
        self.output_dir = (str(output_dir) if output_dir is not None else None)
        self.fps = int(fps)
        self.grid_stride = int(grid_stride)
        self.tail = int(tail)
        self.radius = int(radius)
        self.visibility_threshold = float(visibility_threshold)
        self.seed = int(seed)
        self.validate_at_start = bool(validate_at_start)

        if self.every_steps <= 0:
            raise ValueError("every_steps must be positive")
        if self.num_val_samples <= 0:
            raise ValueError("num_val_samples must be positive")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.grid_stride <= 0:
            raise ValueError("grid_stride must be positive")

        self.tracker: Any = DummyTracker()
        self._samples: list[dict[str, Any]] = []
        self._is_main = False
        self._did_start_validation = False

    def on_train_start(self, method: Any, iteration: int = 0) -> None:
        del iteration
        self.tracker = getattr(method, "tracker", None) or DummyTracker()
        self._is_main = int(get_world_group().rank) == 0
        try:
            self._samples = self._load_samples()
            logger.info(
                "WanTrack validation loaded %d fixed sample(s)",
                len(self._samples),
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "WanTrack validation setup failed; disabling callback: %s",
                error,
            )
            self._samples = []

    def on_validation_begin(self, method: Any, iteration: int = 0) -> None:
        if not self._samples:
            return
        run_at_start = (self.validate_at_start and not self._did_start_validation and iteration == 0)
        run_periodic = iteration > 0 and iteration % self.every_steps == 0
        if not run_at_start and not run_periodic:
            return
        self._did_start_validation = True
        try:
            self._run(method, iteration)
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "WanTrack validation failed at step %d: %s",
                iteration,
                error,
            )

    def _load_samples(self) -> list[dict[str, Any]]:
        from fastvideo.dataset.dataloader.schema import (
            pyarrow_schema_i2v_track, )
        from fastvideo.dataset.utils import (
            collate_rows_from_parquet_schema, )

        data_path = self.val_data_path or str(self.training_config.data.data_path)
        files = sorted(glob.glob(
            os.path.join(data_path, "**", "*.parquet"),
            recursive=True,
        ))
        if not files:
            raise FileNotFoundError(f"No WanTrack validation parquet under {data_path}")

        rows: list[dict[str, Any]] = []
        for path in files:
            table = pq.read_table(
                path,
                columns=pyarrow_schema_i2v_track.names,
            )
            rows.extend(table.to_pylist())
            if len(rows) >= self.num_val_samples:
                break

        text_len = int(self.training_config.pipeline_config.text_encoder_configs[0].arch_config.text_len)
        collated = collate_rows_from_parquet_schema(
            rows[:self.num_val_samples],
            pyarrow_schema_i2v_track,
            text_padding_length=text_len,
            cfg_rate=0.0,
            seed=self.seed,
        )

        infos = collated.get("info_list") or [{} for _ in rows]
        samples: list[dict[str, Any]] = []
        batch_size = int(collated["text_embedding"].shape[0])
        for index in range(batch_size):
            sample = {
                name: value[index:index + 1].clone()
                for name, value in collated.items() if torch.is_tensor(value)
            }
            sample["info_list"] = [infos[index]]
            samples.append(sample)
        return samples

    @torch.no_grad()
    def _run(self, method: Any, iteration: int) -> None:
        student = method.student
        transformer = student.transformer
        was_training = bool(transformer.training)
        transformer.eval()
        try:
            artifacts: list[Any] = []
            for sample_index, raw_sample in enumerate(self._samples):
                seed = self.seed + sample_index
                batch = prepare_wantrack_batch(
                    student,
                    raw_sample,
                    seed=seed,
                    latents_source="zeros",
                )
                sampled = sample_wantrack(
                    student,
                    batch,
                    num_inference_steps=self.num_inference_steps,
                    seed=seed,
                    text_guidance_scale=self.guidance_scale,
                    motion_guidance_scale=self.motion_guidance_scale,
                )
                decoded = student.decode_latents(sampled)

                if not self._is_main:
                    continue
                output_dir = self.output_dir or os.path.join(
                    self.training_config.checkpoint.output_dir,
                    "track_validation",
                )
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir,
                    f"step{iteration:06d}_sample{sample_index}.mp4",
                )
                video = (decoded[0].clamp(0, 1).float().cpu().numpy() * 255.0).astype(np.uint8)
                frames = video.transpose(1, 2, 3, 0)
                conditional = batch.conditional_dict
                if conditional is None:
                    raise RuntimeError("WanTrack validation batch has no conditioning")
                frames_with_tracks = _overlay_tracks(
                    frames,
                    conditional["track_points"],
                    conditional["track_visibility"],
                    stride=self.grid_stride,
                    tail=self.tail,
                    radius=self.radius,
                    visibility_threshold=self.visibility_threshold,
                )
                imageio.mimsave(
                    output_path,
                    frames_with_tracks,
                    fps=self.fps,
                    macro_block_size=1,
                )
                info = raw_sample.get("info_list") or [{}]
                caption = str(info[0].get("caption", ""))
                artifact = self.tracker.video(
                    output_path,
                    caption=(f"WanTrack step {iteration}: "
                             f"{caption[:120]}"),
                )
                if artifact is not None:
                    artifacts.append(artifact)

            if self._is_main and artifacts:
                self.tracker.log_artifacts(
                    {"track_validation/generated": artifacts},
                    iteration,
                )
        finally:
            if was_training:
                transformer.train()
