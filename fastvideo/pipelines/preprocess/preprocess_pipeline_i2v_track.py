# SPDX-License-Identifier: Apache-2.0
"""I2V preprocessing with point-track conditioning."""

from typing import Any

import numpy as np

from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v_track
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.preprocess.preprocess_pipeline_i2v import (
    PreprocessPipeline_I2V, )


class PreprocessPipeline_I2V_Track(PreprocessPipeline_I2V):
    """Store I2V features and spatially aligned MotionStream point tracks."""

    def get_pyarrow_schema(self):
        return pyarrow_schema_i2v_track

    @staticmethod
    def _get_source_size(
        sidecar: Any,
        valid_data: dict[str, Any],
        idx: int,
        points_path: str,
    ) -> tuple[float, float]:
        if "width" in sidecar and "height" in sidecar:
            width = float(np.asarray(sidecar["width"]).item())
            height = float(np.asarray(sidecar["height"]).item())
        elif "source_width" in valid_data and "source_height" in valid_data:
            width = float(valid_data["source_width"][idx])
            height = float(valid_data["source_height"][idx])
        else:
            raise ValueError(f"{points_path}: source width and height are required to align "
                             "pixel-space tracks. Store them in the sidecar or manifest resolution.")

        if width <= 0 or height <= 0:
            raise ValueError(f"{points_path}: invalid source size ({width}, {height})")
        return width, height

    @staticmethod
    def _normalize_after_center_crop(
        tracks: np.ndarray,
        visibility: np.ndarray,
        *,
        source_width: float,
        source_height: float,
        target_width: int,
        target_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply ``CenterCropResizeVideo`` geometry and normalize coordinates."""
        if target_width <= 0 or target_height <= 0:
            raise ValueError(f"Invalid target size ({target_width}, {target_height})")
        target_ratio = target_height / target_width
        if source_height / source_width > target_ratio:
            crop_height = int(source_width * target_ratio)
            crop_width = int(source_width)
        else:
            crop_height = int(source_height)
            crop_width = int(source_height / target_ratio)
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError(f"Center crop is empty for source size "
                             f"({source_width}, {source_height}) and target size "
                             f"({target_width}, {target_height})")

        crop_top = int(round((source_height - crop_height) / 2.0))
        crop_left = int(round((source_width - crop_width) / 2.0))

        x = tracks[..., 0]
        y = tracks[..., 1]
        finite = np.isfinite(x) & np.isfinite(y)
        in_crop = (finite
                   & (x >= crop_left)
                   & (x < crop_left + crop_width)
                   & (y >= crop_top)
                   & (y < crop_top + crop_height))

        normalized = np.empty_like(tracks, dtype=np.float32)
        normalized[..., 0] = (x - crop_left) / crop_width
        normalized[..., 1] = (y - crop_top) / crop_height
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(normalized, 0.0, 1.0, out=normalized)

        aligned_visibility = np.asarray(visibility, dtype=np.float32) * in_crop
        aligned_visibility = np.nan_to_num(aligned_visibility, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(aligned_visibility, 0.0, 1.0, out=aligned_visibility)
        return (
            np.ascontiguousarray(normalized, dtype=np.float32),
            np.ascontiguousarray(aligned_visibility, dtype=np.float32),
        )

    def get_extra_features(
        self,
        valid_data: dict[str, Any],
        fastvideo_args: FastVideoArgs,
    ) -> dict[str, Any]:
        features = super().get_extra_features(valid_data, fastvideo_args)

        points_paths = valid_data.get("points_path")
        frame_indices_batch = valid_data.get("sample_frame_index")
        if not points_paths or frame_indices_batch is None:
            raise ValueError("i2v_track preprocessing requires points_path and sampled frame "
                             "indices for every video.")
        non_video_paths = [path for path in valid_data["path"] if not str(path).lower().endswith(".mp4")]
        if non_video_paths:
            raise ValueError("i2v_track preprocessing currently supports videos only; "
                             f"received {non_video_paths[0]!r}")

        target_height = int(valid_data["pixel_values"].shape[-2])
        target_width = int(valid_data["pixel_values"].shape[-1])
        expected_frames = int(valid_data["pixel_values"].shape[2])
        track_points: list[np.ndarray] = []
        track_visibility: list[np.ndarray] = []
        object_ids: list[np.ndarray] = []
        track_weights: list[np.ndarray] = []

        samples = zip(points_paths, frame_indices_batch, strict=True)
        for idx, (points_path, frame_indices) in enumerate(samples):
            indices = np.asarray(frame_indices, dtype=np.int64)
            if indices.ndim != 1 or indices.size != expected_frames:
                raise ValueError(f"{points_path}: expected {expected_frames} sampled frame "
                                 f"indices, got shape {indices.shape}")
            if indices.size == 0 or indices.min() < 0:
                raise ValueError(f"{points_path}: sampled frame indices must be non-negative")

            with np.load(points_path) as sidecar:
                if "tracks" not in sidecar or "visibility" not in sidecar:
                    raise ValueError(f"{points_path}: sidecar must contain 'tracks' and 'visibility'")
                tracks = np.asarray(sidecar["tracks"], dtype=np.float32)
                visibility = np.asarray(sidecar["visibility"], dtype=np.float32)
                if tracks.ndim != 3 or tracks.shape[-1] != 2:
                    raise ValueError(f"{points_path}: tracks must have shape [T, N, 2], got "
                                     f"{tracks.shape}")
                if visibility.shape != tracks.shape[:2]:
                    raise ValueError(f"{points_path}: visibility shape {visibility.shape} does "
                                     f"not match tracks {tracks.shape[:2]}")
                if indices.max() >= tracks.shape[0]:
                    raise ValueError(f"{points_path}: sampled frame {indices.max()} exceeds "
                                     f"{tracks.shape[0]} track frames")

                source_width, source_height = self._get_source_size(sidecar, valid_data, idx, points_path)
                num_tracks = tracks.shape[1]
                sample_object_ids = np.asarray(
                    sidecar["object_ids"] if "object_ids" in sidecar else np.full(num_tracks, -1, dtype=np.float32),
                    dtype=np.float32,
                )
                sample_track_weights = np.asarray(
                    sidecar["track_weights"] if "track_weights" in sidecar else np.zeros(num_tracks, dtype=np.float32),
                    dtype=np.float32,
                )
                if sample_object_ids.shape != (num_tracks, ):
                    raise ValueError(f"{points_path}: object_ids must have shape "
                                     f"[{num_tracks}], got {sample_object_ids.shape}")
                if sample_track_weights.shape != (num_tracks, ):
                    raise ValueError(f"{points_path}: track_weights must have shape "
                                     f"[{num_tracks}], got {sample_track_weights.shape}")
                tracks = tracks[indices]
                visibility = visibility[indices]

            tracks, visibility = self._normalize_after_center_crop(
                tracks,
                visibility,
                source_width=source_width,
                source_height=source_height,
                target_width=target_width,
                target_height=target_height,
            )
            track_points.append(tracks)
            track_visibility.append(visibility)
            object_ids.append(np.ascontiguousarray(sample_object_ids, dtype=np.float32))
            track_weights.append(np.ascontiguousarray(sample_track_weights, dtype=np.float32))

        features["track_points"] = track_points
        features["track_visibility"] = track_visibility
        features["object_ids"] = object_ids
        features["track_weights"] = track_weights
        return features

    def create_record(
        self,
        video_name: str,
        vae_latent: np.ndarray,
        text_embedding: np.ndarray,
        valid_data: dict[str, Any],
        idx: int,
        extra_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = super().create_record(
            video_name=video_name,
            vae_latent=vae_latent,
            text_embedding=text_embedding,
            valid_data=valid_data,
            idx=idx,
            extra_features=extra_features,
        )

        for name in (
                "track_points",
                "track_visibility",
                "object_ids",
                "track_weights",
        ):
            if extra_features is None or name not in extra_features:
                raise ValueError(f"Missing required WanTrack feature {name!r} for {video_name}")
            array = np.ascontiguousarray(extra_features[name], dtype=np.float32)
            record[f"{name}_bytes"] = array.tobytes()
            record[f"{name}_shape"] = list(array.shape)
            record[f"{name}_dtype"] = str(array.dtype)
            record.pop(name, None)

        return record


EntryClass = PreprocessPipeline_I2V_Track
