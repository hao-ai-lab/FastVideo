# SPDX-License-Identifier: Apache-2.0
"""I2V + MotionStream point-track preprocessing pipeline.

Turns (video.mp4 + CoTracker tracks.npz + text prompt) into a FastVideo training
parquet for the WanTrack model. Compared to the plain I2V pipeline this:
  - keeps the T5 text embedding (WanTrack conditions on the prompt),
  - computes the first-frame conditioning latent (concatenated I2V, no CLIP),
  - adds the dense CoTracker tracks (normalized coords) + visibility.

The track embedding itself (MotionStream scatter / track head) runs *inside the
model* at train/inference time, so here we only store the raw normalized tracks.
"""
from typing import Any

import numpy as np
import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v_track
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.preprocess.preprocess_pipeline_base import (BasePreprocessPipeline)
from fastvideo.pipelines.stages import TextEncodingStage

logger = init_logger(__name__)


class PreprocessPipeline_I2V_Track(BasePreprocessPipeline):
    """I2V + point-track preprocessing (first-frame latent concat, no CLIP)."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

    def get_pyarrow_schema(self):
        return pyarrow_schema_i2v_track

    def _encode_first_frame_latent(self, valid_data: dict[str, Any]) -> torch.Tensor:
        """First-frame I2V conditioning latent: VAE-encode [frame0, zeros...]."""
        self.get_module("vae").to(get_local_torch_device())
        first_frame = valid_data["pixel_values"][:, :, 0, :, :].permute(0, 2, 3, 1)  # (B,H,W,C)
        _, _, num_frames, height, width = valid_data["pixel_values"].shape

        video_conditions = []
        for frame in first_frame:
            processed_img = frame.to(device="cpu", dtype=torch.float32)
            processed_img = processed_img.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(2)  # (B,C,1,H,W)
            video_condition = torch.cat([
                processed_img,
                processed_img.new_zeros(processed_img.shape[0], processed_img.shape[1], num_frames - 1, height, width),
            ], dim=2)
            video_conditions.append(video_condition.to(device=get_local_torch_device(), dtype=torch.float32))
        video_conditions = torch.cat(video_conditions, dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
            latent_condition = self.get_module("vae").encode(video_conditions).mean

        # NOTE: the FastVideo Wan VAE's encode() already returns latents in the
        # normalized (latents_mean/std) space, so this conditioning is in the same
        # space as the (normalized) main latents. Verified empirically:
        # decode_latents(first_frame_latent) reconstructs the GT first frame (MSE~0.1).
        # Wan exposes no scaling_factor/shift_factor, so the block below is a no-op
        # (scale=1.0, shift=None); kept for parity with other I2V preprocessors.
        vae = self.get_module("vae")
        if getattr(vae, "shift_factor", None) is not None:
            shift = vae.shift_factor
            shift = shift.to(latent_condition.device, latent_condition.dtype) if isinstance(shift, torch.Tensor) else shift
            latent_condition = latent_condition - shift
        scale = getattr(vae, "scaling_factor", 1.0)
        scale = scale.to(latent_condition.device, latent_condition.dtype) if isinstance(scale, torch.Tensor) else scale
        latent_condition = latent_condition * scale
        return latent_condition

    def get_extra_features(self, valid_data: dict[str, Any], fastvideo_args: FastVideoArgs) -> dict[str, Any]:
        features: dict[str, Any] = {}
        features["first_frame_latent"] = self._encode_first_frame_latent(valid_data)

        num_frames = valid_data["pixel_values"].shape[2]
        points_paths = valid_data.get("points_path")
        if not points_paths:
            raise ValueError("PreprocessPipeline_I2V_Track requires a 'points_path' sidecar per sample; "
                             "none found in the manifest. Run extract_tracks.py first.")

        track_points_list: list[np.ndarray] = []
        track_visibility_list: list[np.ndarray] = []
        for points_path in points_paths:
            d = np.load(points_path)
            tracks = d["tracks"].astype(np.float32)          # [T, N, 2] in pixel coords
            visibility = d["visibility"].astype(np.float32)  # [T, N]
            width = float(d["width"]) if "width" in d else float(valid_data["pixel_values"].shape[-1])
            height = float(d["height"]) if "height" in d else float(valid_data["pixel_values"].shape[-2])
            if tracks.shape[0] < num_frames:
                raise ValueError(f"{points_path}: {tracks.shape[0]} track frames < {num_frames} video frames")
            tracks = tracks[:num_frames].copy()
            visibility = visibility[:num_frames]
            # Normalize coords to [0, 1] so they're resolution-independent.
            tracks[..., 0] /= width
            tracks[..., 1] /= height
            track_points_list.append(np.ascontiguousarray(tracks, dtype=np.float32))
            track_visibility_list.append(np.ascontiguousarray(visibility, dtype=np.float32))

        features["track_points"] = track_points_list
        features["track_visibility"] = track_visibility_list
        return features

    def create_record(self,
                      video_name: str,
                      vae_latent: np.ndarray,
                      text_embedding: np.ndarray,
                      valid_data: dict[str, Any],
                      idx: int,
                      extra_features: dict[str, Any] | None = None) -> dict[str, Any]:
        record = super().create_record(video_name=video_name,
                                       vae_latent=vae_latent,
                                       text_embedding=text_embedding,
                                       valid_data=valid_data,
                                       idx=idx,
                                       extra_features=extra_features)

        def _put(name: str, arr) -> None:
            if extra_features is not None and name in extra_features and extra_features[name] is not None:
                a = np.asarray(extra_features[name])
                record[f"{name}_bytes"] = a.tobytes()
                record[f"{name}_shape"] = list(a.shape)
                record[f"{name}_dtype"] = str(a.dtype)
            else:
                record[f"{name}_bytes"] = b""
                record[f"{name}_shape"] = []
                record[f"{name}_dtype"] = ""

        _put("first_frame_latent", None)
        _put("track_points", None)
        _put("track_visibility", None)
        return record


EntryClass = PreprocessPipeline_I2V_Track
