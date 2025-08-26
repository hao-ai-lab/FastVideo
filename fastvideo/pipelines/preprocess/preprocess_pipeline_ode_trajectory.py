# SPDX-License-Identifier: Apache-2.0
"""
ODE Trajectory Data Preprocessing pipeline implementation.

This module contains an implementation of the ODE Trajectory Data Preprocessing pipeline
using the modular pipeline architecture.

Sec 4.3 of CausVid paper: https://arxiv.org/pdf/2412.07772
"""

import os
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from fastvideo.dataset import getdataset
from fastvideo.dataset.dataloader.schema import pyarrow_schema_ode_trajectory
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline)
from fastvideo.pipelines.stages import (DenoisingStage, ImageVAEEncodingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage)

logger = init_logger(__name__)

class PreprocessPipeline_ODE_Trajectory(BasePreprocessPipeline):
    """ODE Trajectory preprocessing pipeline implementation."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]
    preprocess_dataloader: StatefulDataLoader
    preprocess_loader_iter: Iterator[dict[str, Any]]

    def get_schema_fields(self):
        """Get the schema fields for ODE Trajectory pipeline."""
        return [f.name for f in pyarrow_schema_ode_trajectory]
    
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))
        self.add_stage(stage_name="vae_encoding_stage",
                       stage=ImageVAEEncodingStage(
                           vae=self.get_module("vae"),
                       ))
        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer", None)))
        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           transformer_2=self.get_module("transformer_2", None),
                           scheduler=self.get_module("scheduler"),
                           pipeline=self,
                       ))

    
    def preprocess_video_and_text_and_trajectory(self, fastvideo_args: FastVideoArgs, args):

        for batch_idx, data in enumerate(self.pbar):
            if data is None:
                continue

            with torch.inference_mode():
                # Filter out invalid samples (those with all zeros)
                valid_indices = []
                for i, pixel_values in enumerate(data["pixel_values"]):
                    if not torch.all(
                            pixel_values == 0):  # Check if all values are zero
                        valid_indices.append(i)
                self.num_processed_samples += len(valid_indices)

                if not valid_indices:
                    continue

                # Create new batch with only valid samples
                valid_data = {
                    "pixel_values":
                    torch.stack(
                        [data["pixel_values"][i] for i in valid_indices]),
                    "text": [data["text"][i] for i in valid_indices],
                    "path": [data["path"][i] for i in valid_indices],
                    "fps": [data["fps"][i] for i in valid_indices],
                    "duration": [data["duration"][i] for i in valid_indices],
                }

                # VAE
                with torch.autocast("cuda", dtype=torch.float32):
                    latents = self.get_module("vae").encode(
                        valid_data["pixel_values"].to(
                            get_local_torch_device())).mean

                # Get extra features if needed
                extra_features = self.get_extra_features(
                    valid_data, fastvideo_args)

                batch_captions = valid_data["text"]
                batch = ForwardBatch(
                    data_type="video",
                    prompt=batch_captions,
                    prompt_embeds=[],
                    prompt_attention_mask=[],
                )
                assert hasattr(self, "prompt_encoding_stage")
                result_batch = self.prompt_encoding_stage(batch, fastvideo_args)
                prompt_embeds, prompt_attention_mask = result_batch.prompt_embeds[
                    0], result_batch.prompt_attention_mask[0]
                assert prompt_embeds.shape[0] == prompt_attention_mask.shape[0]

                # Get sequence lengths from attention masks (number of 1s)
                seq_lens = prompt_attention_mask.sum(dim=1)

                non_padded_embeds = []
                non_padded_masks = []

                # Process each item in the batch
                for i in range(prompt_embeds.size(0)):
                    seq_len = seq_lens[i].item()
                    # Slice the embeddings and masks to keep only non-padding parts
                    non_padded_embeds.append(prompt_embeds[i, :seq_len])
                    non_padded_masks.append(prompt_attention_mask[i, :seq_len])

                # Update the tensors with non-padded versions
                prompt_embeds = non_padded_embeds
                prompt_attention_mask = non_padded_masks


                # Collect the trajectory data
                batch = ForwardBatch(
                    data_type="video",
                    prompt=batch_captions,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    height=args.max_height,
                    width=args.max_width,
                    num_frames=81
                    fps=args.train_fps,
                )
                fastvideo_args.pipeline_config.ti2v_task = True

                result_batch = self.input_validation_stage(batch, fastvideo_args)
                # result_batch = self.prompt_encoding_stage(result_batch, fastvideo_args)
                # result_batch = self.vae_encoding_stage(result_batch, fastvideo_args)
                result_batch = self.timestep_preparation_stage(batch, fastvideo_args)
                result_batch = self.latent_preparation_stage(result_batch, fastvideo_args)
                result_batch = self.denoising_stage(result_batch, fastvideo_args)
                trajectory_latents = result_batch.trajectory_latents

            # Prepare batch data for Parquet dataset
            batch_data = []

            # Add progress bar for saving outputs
            save_pbar = tqdm(enumerate(valid_data["path"]),
                             desc="Saving outputs",
                             unit="item",
                             leave=False)
            for idx, video_path in save_pbar:
                # Get the corresponding latent and info using video name
                latent = latents[idx].cpu()
                video_name = os.path.basename(video_path).split(".")[0]

                # Convert tensors to numpy arrays
                vae_latent = latent.cpu().numpy()
                text_embedding = prompt_embeds[idx].cpu().numpy()

                # Get extra features for this sample if needed
                sample_extra_features = {}
                if extra_features:
                    for key, value in extra_features.items():
                        if isinstance(value, torch.Tensor):
                            sample_extra_features[key] = value[idx].cpu().numpy(
                            )
                        else:
                            sample_extra_features[key] = value[idx]

                # Create record for Parquet dataset
                record = self.create_record(
                    video_name=video_name,
                    vae_latent=vae_latent,
                    text_embedding=text_embedding,
                    valid_data=valid_data,
                    idx=idx,
                    extra_features=sample_extra_features)
                batch_data.append(record)

            if batch_data:
                # Add progress bar for writing to Parquet dataset
                write_pbar = tqdm(total=1,
                                  desc="Writing to Parquet dataset",
                                  unit="batch")
                # Convert batch data to PyArrow arrays
                arrays = []
                for field in self.get_schema_fields():
                    if field.endswith('_bytes'):
                        arrays.append(
                            pa.array([record[field] for record in batch_data],
                                     type=pa.binary()))
                    elif field.endswith('_shape'):
                        arrays.append(
                            pa.array([record[field] for record in batch_data],
                                     type=pa.list_(pa.int32())))
                    elif field in ['width', 'height', 'num_frames']:
                        arrays.append(
                            pa.array([record[field] for record in batch_data],
                                     type=pa.int32()))
                    elif field in ['duration_sec', 'fps']:
                        arrays.append(
                            pa.array([record[field] for record in batch_data],
                                     type=pa.float32()))
                    else:
                        arrays.append(
                            pa.array([record[field] for record in batch_data]))

                table = pa.Table.from_arrays(arrays,
                                             names=self.get_schema_fields())
                write_pbar.update(1)
                write_pbar.close()

                # Store the table in a list for later processing
                if not hasattr(self, 'all_tables'):
                    self.all_tables = []
                self.all_tables.append(table)

                logger.info("Collected batch with %s samples", len(table))

            if self.num_processed_samples >= args.flush_frequency:
                self._flush_tables(self.num_processed_samples, args,
                                   self.combined_parquet_dir)
                self.num_processed_samples = 0
                self.all_tables = []

    def get_extra_features(self, valid_data: dict[str, Any],
                           fastvideo_args: FastVideoArgs) -> dict[str, Any]:

        # TODO(will): move these to cpu at some point
        self.get_module("vae").to(get_local_torch_device())

        # generator = torch.Generator(device=get_local_torch_device(), seed=42)
        generator = torch.Generator("cpu").manual_seed(42)

        features = {}
        """Get CLIP features from the first frame of each video."""
        first_frame = valid_data["pixel_values"][:, :, 0, :, :].permute(
            0, 2, 3, 1)  # (B, C, T, H, W) -> (B, H, W, C)
        _, _, num_frames, height, width = valid_data["pixel_values"].shape
        # latent_height = height // self.get_module(
        #     "vae").spatial_compression_ratio
        # latent_width = width // self.get_module("vae").spatial_compression_ratio

        unprocessed_images = []
        # Frame has values between -1 and 1
        for frame in first_frame:
            frame = (frame + 1) * 127.5
            frame_pil = Image.fromarray(frame.cpu().numpy().astype(np.uint8))
            # processed_img = self.get_module("image_processor")(
            #     images=frame_pil, return_tensors="pt")
            unprocessed_images.append(frame_pil)

        """Get VAE features from the first frame of each video"""
        video_conditions = []
        for frame in unprocessed_images:

            latent = self.vae_encoding_stage.encode_image(frame, height, width, fastvideo_args, generator)
            video_conditions.append(latent)


            # processed_img = frame.to(device="cpu", dtype=torch.float32)
            # processed_img = processed_img.unsqueeze(0).permute(0, 3, 1,
            #                                                    2).unsqueeze(2)
            # # (B, H, W, C) -> (B, C, 1, H, W)
            # video_condition = processed_img.unsqueeze(2)
            # video_condition = video_condition.to(
            #     device=get_local_torch_device(), dtype=torch.float32)
            # video_conditions.append(video_condition)

        video_conditions = torch.cat(video_conditions, dim=0)

        # with torch.autocast(device_type="cuda",
        #                     dtype=torch.float32,
        #                     enabled=True):
        #     encoder_outputs = self.get_module("vae").encode(video_conditions)

        # latent_condition = encoder_outputs.mean
        # if (hasattr(self.get_module("vae"), "shift_factor")
        #         and self.get_module("vae").shift_factor is not None):
        #     if isinstance(self.get_module("vae").shift_factor, torch.Tensor):
        #         latent_condition -= self.get_module("vae").shift_factor.to(
        #             latent_condition.device, latent_condition.dtype)
        #     else:
        #         latent_condition -= self.get_module("vae").shift_factor

        # if isinstance(self.get_module("vae").scaling_factor, torch.Tensor):
        #     latent_condition = latent_condition * self.get_module(
        #         "vae").scaling_factor.to(latent_condition.device,
        #                                  latent_condition.dtype)
        # else:
        #     latent_condition = latent_condition * self.get_module(
        #         "vae").scaling_factor

        features["first_frame_latent"] = video_conditions

        return features

    def create_record(
            self,
            video_name: str,
            vae_latent: np.ndarray,
            text_embedding: np.ndarray,
            valid_data: dict[str, Any],
            idx: int,
            extra_features: dict[str, Any] | None = None) -> dict[str, Any]:
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

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs, args):
        if not self.post_init_called:
            self.post_init()

        self.local_rank = int(os.getenv("RANK", 0))
        os.makedirs(args.output_dir, exist_ok=True)
        # Create directory for combined data
        self.combined_parquet_dir = os.path.join(args.output_dir,
                                            "combined_parquet_dataset")
        os.makedirs(self.combined_parquet_dir, exist_ok=True)

        # Loading dataset
        train_dataset = getdataset(args)

        self.preprocess_dataloader = DataLoader(
            train_dataset,
            batch_size=args.preprocess_video_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        self.preprocess_loader_iter = iter(self.preprocess_dataloader)

        self.num_processed_samples = 0
        # Add progress bar for video preprocessing
        self.pbar = tqdm(self.preprocess_loader_iter,
                    desc="Processing videos",
                    unit="batch",
                    disable=self.local_rank != 0)

        # Initialize class variables for data sharing
        self.video_data: dict[str, Any] = {}  # Store video metadata and paths
        self.latent_data: dict[str, Any] = {}  # Store latent tensors
        self.preprocess_video_and_text_and_trajectory(fastvideo_args, args)

EntryClass = PreprocessPipeline_ODE_Trajectory
