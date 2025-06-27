# SPDX-License-Identifier: Apache-2.0
"""
I2V Data Preprocessing pipeline implementation.

This module contains an implementation of the I2V Data Preprocessing pipeline
using the modular pipeline architecture.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import PIL
import torch
from PIL import Image

from fastvideo.v1.dataset.dataloader.schema import pyarrow_schema_i2v
from fastvideo.v1.distributed import get_torch_device
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.models.vision_utils import (get_default_height_width,
                                              normalize, numpy_to_pt,
                                              pil_to_numpy, resize)
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline)
from fastvideo.v1.pipelines.stages import ImageEncodingStage, TextEncodingStage


class PreprocessPipeline_I2V(BasePreprocessPipeline):
    """I2V preprocessing pipeline implementation."""

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "image_encoder", "image_processor"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="image_encoding_stage",
                       stage=ImageEncodingStage(
                           image_encoder=self.get_module("image_encoder"),
                           image_processor=self.get_module("image_processor"),
                       ))

    def preprocess_image(self, image: PIL.Image.Image, height: int, width: int,
                         fastvideo_args: FastVideoArgs) -> Dict[str, Any]:
        assert hasattr(
            self,
            "image_encoding_stage"), "Image encoding stage must be created"

        batch = ForwardBatch(
            data_type="video",
            pil_image=image,
        )
        result_batch = self.image_encoding_stage(batch, fastvideo_args)
        clip_features = result_batch.image_embeds[0]

        image = self.preprocess(
            image,
            vae_scale_factor=self.get_module("vae").spatial_compression_ratio,
            height=height,
            width=width)

        return {
            "clip_feature": clip_features[0],
            "pil_image": image,
        }

    def preprocess_video(self, video: list[PIL.Image.Image], height: int,
                         width: int,
                         fastvideo_args: FastVideoArgs) -> Dict[str, Any]:
        return self.preprocess_image(video[0], height, width, fastvideo_args)

    def get_schema_fields(self) -> List[str]:
        """Get the schema fields for I2V pipeline."""
        return [f.name for f in pyarrow_schema_i2v]

    def get_extra_features(self, valid_data: Dict[str, Any],
                           fastvideo_args: FastVideoArgs) -> Dict[str, Any]:

        # TODO(will): move these to cpu at some point
        self.get_module("image_encoder").to(get_torch_device())
        self.get_module("vae").to(get_torch_device())

        features = {}
        """Get CLIP features from the first frame of each video."""
        first_frame = valid_data["pixel_values"][:, :, 0, :, :].permute(
            0, 2, 3, 1)  # (B, C, T, H, W) -> (B, H, W, C)
        batch_size, _, num_frames, height, width = valid_data[
            "pixel_values"].shape
        latent_height = height // self.get_module(
            "vae").spatial_compression_ratio
        latent_width = width // self.get_module("vae").spatial_compression_ratio

        processed_images = []
        # Frame has values between -1 and 1
        for frame in first_frame:
            frame = (frame + 1) * 127.5
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

        features["clip_feature"] = clip_features
        """Get VAE features from the first frame of each video"""
        video_conditions = []
        for frame in first_frame:
            processed_img = frame.to(device="cpu", dtype=torch.float32)
            processed_img = processed_img.unsqueeze(0).permute(0, 3, 1,
                                                               2).unsqueeze(2)
            # (B, H, W, C) -> (B, C, 1, H, W)
            video_condition = torch.cat([
                processed_img,
                processed_img.new_zeros(processed_img.shape[0],
                                        processed_img.shape[1], num_frames - 1,
                                        height, width)
            ],
                                        dim=2)
            video_condition = video_condition.to(device=get_torch_device(),
                                                 dtype=torch.float32)
            video_conditions.append(video_condition)

        video_conditions = torch.cat(video_conditions, dim=0)

        with torch.autocast(device_type="cuda",
                            dtype=torch.float32,
                            enabled=True):
            encoder_outputs = self.get_module("vae").encode(video_conditions)

        latent_condition = encoder_outputs.mean
        if (hasattr(self.get_module("vae"), "shift_factor")
                and self.get_module("vae").shift_factor is not None):
            if isinstance(self.get_module("vae").shift_factor, torch.Tensor):
                latent_condition -= self.get_module("vae").shift_factor.to(
                    latent_condition.device, latent_condition.dtype)
            else:
                latent_condition -= self.get_module("vae").shift_factor

        if isinstance(self.get_module("vae").scaling_factor, torch.Tensor):
            latent_condition = latent_condition * self.get_module(
                "vae").scaling_factor.to(latent_condition.device,
                                         latent_condition.dtype)
        else:
            latent_condition = latent_condition * self.get_module(
                "vae").scaling_factor

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height,
                                   latent_width)
        mask_lat_size[:, :, list(range(1, num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask,
            dim=2,
            repeats=self.get_module("vae").temporal_compression_ratio)
        mask_lat_size = torch.concat(
            [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1,
            self.get_module("vae").temporal_compression_ratio, latent_height,
            latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        image_latent = torch.concat([mask_lat_size, latent_condition], dim=1)

        features["first_frame_latent"] = image_latent

        return features

    def create_record(
            self,
            video_name: str,
            vae_latent: np.ndarray,
            text_embedding: np.ndarray,
            valid_data: Dict[str, Any],
            idx: int,
            extra_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a record for the Parquet dataset with CLIP features."""
        record = super().create_record(video_name=video_name,
                                       vae_latent=vae_latent,
                                       text_embedding=text_embedding,
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

        if extra_features and "first_frame_latent" in extra_features:
            first_frame_latent = extra_features["first_frame_latent"]
            record.update({
                "first_frame_latent_bytes":
                first_frame_latent.tobytes(),
                "first_frame_latent_shape":
                list(first_frame_latent.shape),
                "first_frame_latent_dtype":
                str(first_frame_latent.dtype),
            })
        else:
            record.update({
                "first_frame_latent_bytes": b"",
                "first_frame_latent_shape": [],
                "first_frame_latent_dtype": "",
            })

        if extra_features and "pil_image" in extra_features:
            pil_image = extra_features["pil_image"]
            record.update({
                "pil_image_bytes": pil_image.tobytes(),
                "pil_image_shape": list(pil_image.shape),
                "pil_image_dtype": str(pil_image.dtype),
            })
        else:
            record.update({
                "pil_image_bytes": b"",
                "pil_image_shape": [],
                "pil_image_dtype": "",
            })

        return record

    def pil_to_tensor(self, image: PIL.Image.Image) -> torch.Tensor:
        image = image

        image = np.array(image).astype(np.float32)
        image = torch.from_numpy(image)
        return image

    def preprocess(self,
                   image: PIL.Image.Image,
                   vae_scale_factor: int,
                   height: int,
                   width: int,
                   resize_mode: str = "default") -> torch.Tensor:
        image = [image]

        height, width = get_default_height_width(image[0], vae_scale_factor,
                                                 height, width)
        image = [
            resize(i, height, width, resize_mode=resize_mode) for i in image
        ]
        image = pil_to_numpy(image)  # to np
        image = numpy_to_pt(image)  # to pt

        do_normalize = True
        if image.min() < 0:
            do_normalize = False
        if do_normalize:
            image = normalize(image)

        return image


EntryClass = PreprocessPipeline_I2V
