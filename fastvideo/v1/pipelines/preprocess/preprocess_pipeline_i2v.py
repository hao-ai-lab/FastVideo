# SPDX-License-Identifier: Apache-2.0
"""
I2V Data Preprocessing pipeline implementation.

This module contains an implementation of the I2V Data Preprocessing pipeline
using the modular pipeline architecture.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from fastvideo.v1.dataset.dataloader.schema import pyarrow_schema_i2v
from fastvideo.v1.distributed import get_torch_device
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline)


class PreprocessPipeline_I2V(BasePreprocessPipeline):
    """I2V preprocessing pipeline implementation."""

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "image_encoder", "image_processor"
    ]

    def get_schema_fields(self) -> List[str]:
        """Get the schema fields for I2V pipeline."""
        return [f.name for f in pyarrow_schema_i2v]

    def get_extra_features(self, valid_data: Dict[str, Any],
                           fastvideo_args: FastVideoArgs) -> Dict[str, Any]:
        """Get CLIP features from the first frame of each video."""
        first_frame = valid_data["pixel_values"][:, :, 0, :, :].permute(
            0, 2, 3, 1)  # (B, C, T, H, W) -> (B, H, W, C)

        processed_images = []
        for frame in first_frame:
            frame_pil = Image.fromarray(frame.cpu().numpy().astype(np.uint8))
            processed_img = self.get_module("image_processor")(
                images=frame_pil, return_tensors="pt")
            processed_images.append(processed_img)

        # Get CLIP features
        pixel_values = torch.cat(
            [img['pixel_values'] for img in processed_images],
            dim=0).to(get_torch_device())
        with torch.no_grad():
            image_inputs = {'pixel_values': pixel_values}
            with set_forward_context(current_timestep=0, attn_metadata=None):
                clip_features = self.get_module("image_encoder")(**image_inputs)
            clip_features = clip_features.last_hidden_state

        return {"clip_feature": clip_features}

    def create_record(
            self,
            video_name: str,
            vae_latent: np.ndarray,
            text_embedding: np.ndarray,
            text_attention_mask: np.ndarray,
            valid_data: Optional[Dict[str, Any]],
            idx: int,
            extra_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a record for the Parquet dataset with CLIP features."""
        record = super().create_record(video_name=video_name,
                                       vae_latent=vae_latent,
                                       text_embedding=text_embedding,
                                       text_attention_mask=text_attention_mask,
                                       valid_data=valid_data,
                                       idx=idx,
                                       extra_features=extra_features)

        if extra_features and "clip_feature" in extra_features:
            clip_feature = extra_features["clip_feature"]
            record.update({
                "clip_feature_bytes": clip_feature.tobytes(),
                "clip_feature_shape": list(clip_feature.shape),
                "clip_feature_dtype": str(clip_feature.dtype),
            })
        else:
            record.update({
                "clip_feature_bytes": b"",
                "clip_feature_shape": [],
                "clip_feature_dtype": "",
            })

        return record  # type: ignore


EntryClass = PreprocessPipeline_I2V
