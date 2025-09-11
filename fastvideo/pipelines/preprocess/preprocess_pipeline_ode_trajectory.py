# SPDX-License-Identifier: Apache-2.0
"""
ODE Trajectory Data Preprocessing pipeline implementation.

This module contains an implementation of the ODE Trajectory Data Preprocessing pipeline
using the modular pipeline architecture.

Sec 4.3 of CausVid paper: https://arxiv.org/pdf/2412.07772
"""

import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset import getdataset, gettextdataset
from fastvideo.dataset.dataloader.schema import pyarrow_schema_ode_trajectory, pyarrow_schema_ode_trajectory_text_only
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.utils import shallow_asdict, save_decoded_latents_as_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline)
from fastvideo.pipelines.stages import (DenoisingStage, ImageVAEEncodingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage,
                                        DecodingStage)

logger = init_logger(__name__)


class PreprocessPipeline_ODE_Trajectory(BasePreprocessPipeline):
    """ODE Trajectory preprocessing pipeline implementation."""

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]
    preprocess_dataloader: StatefulDataLoader
    preprocess_loader_iter: Iterator[dict[str, Any]]

    def get_schema_fields(self):
        """Get the schema fields for ODE Trajectory pipeline."""
        # Check if we're using text dataset by checking if the dataset is TextDataset
        if hasattr(self, 'preprocess_dataloader') and hasattr(self.preprocess_dataloader.dataset, '_process_text_data'):
            return [f.name for f in pyarrow_schema_ode_trajectory_text_only]
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
                           vae=self.get_module("vae"), ))
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
        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

    def preprocess_video_and_text_and_trajectory(self,
                                                 fastvideo_args: FastVideoArgs,
                                                 args):

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
                # Encode text using the standalone TextEncodingStage API
                prompt_embeds_list, prompt_masks_list = self.prompt_encoding_stage.encode_text(
                    batch_captions,
                    fastvideo_args,
                    encoder_index=[0],
                    return_attention_mask=True,
                )
                prompt_embeds = prompt_embeds_list[0]
                prompt_attention_masks = prompt_masks_list[0]
                assert prompt_embeds.shape[0] == prompt_attention_masks.shape[0]

                # # Get sequence lengths from attention masks (number of 1s)
                # seq_lens = prompt_attention_mask.sum(dim=1)

                # non_padded_embeds = []
                # non_padded_masks = []

                # # Process each item in the batch
                # for i in range(prompt_embeds.size(0)):
                #     seq_len = seq_lens[i].item()
                #     # Slice the embeddings and masks to keep only non-padding parts
                #     non_padded_embeds.append(prompt_embeds[i, :seq_len])
                #     non_padded_masks.append(prompt_attention_mask[i, :seq_len])

                # Update the tensors with non-padded versions
                # prompt_embeds = non_padded_embeds
                # prompt_attention_masks = non_padded_masks
                # prompt_embeds = prompt_embeds

                # logger.info(f"===== prompt_embeds: {prompt_embeds[0].shape}")
                # logger.info(f"===== prompt_attention_masks: {prompt_attention_masks[0].shape}")

                sampling_params = SamplingParam.from_pretrained(
                    args.model_path)

                # encode negative prompt for trajectory collection
                if sampling_params.guidance_scale > 1 and sampling_params.negative_prompt is not None:
                    negative_prompt_embeds_list, negative_prompt_masks_list = self.prompt_encoding_stage.encode_text(
                        sampling_params.negative_prompt,
                        fastvideo_args,
                        encoder_index=[0],
                        return_attention_mask=True,
                    )
                    negative_prompt_embed = negative_prompt_embeds_list[0][0]
                    negative_prompt_attention_mask = negative_prompt_masks_list[0][0]
                else:
                    negative_prompt_embed = None
                    negative_prompt_attention_mask = None

                trajectory_latents = []
                trajectory_timesteps = []
                trajectory_decoded = []
                for i, (prompt_embed, prompt_attention_mask) in enumerate(zip(prompt_embeds, prompt_attention_masks)):
                    prompt_embed = prompt_embed.unsqueeze(0)
                    prompt_attention_mask = prompt_attention_mask.unsqueeze(0)
                    logger.info(f"what")
                    logger.info(f"===== prompt_embed: {prompt_embed.shape}")
                    logger.info(f"===== prompt_attention_mask: {prompt_attention_mask.shape}")
                    # Collect the trajectory data
                    batch = ForwardBatch(
                        **shallow_asdict(sampling_params),
                        # data_type="video",
                        # seed=args.seed,
                        # prompt=batch_captions[i],
                        # prompt_embeds=[prompt_embed],
                        # prompt_attention_mask=[prompt_attention_mask],
                        # height=args.max_height,
                        # width=args.max_width,
                        # num_frames=81,
                        # fps=args.train_fps,
                        # return_trajectory_latents=True,
                        # guidance_scale=3.0,
                        # do_classifier_free_guidance=True,
                    )
                    batch.prompt_embeds = [prompt_embed]
                    batch.prompt_attention_mask = [prompt_attention_mask]
                    batch.negative_prompt_embeds = [negative_prompt_embed]
                    batch.negative_attention_mask = [negative_prompt_attention_mask]
                    batch.return_trajectory_latents = True
                    batch.return_trajectory_decoded = False
                    batch.height = args.max_height
                    batch.width = args.max_width
                    # batch.num_frames = 81
                    batch.fps = args.train_fps
                    batch.guidance_scale = 3.0
                    batch.do_classifier_free_guidance = True
                    # fastvideo_args.pipeline_config.ti2v_task = True

                    result_batch = self.input_validation_stage(
                        batch, fastvideo_args)
                    # result_batch = self.prompt_encoding_stage(result_batch, fastvideo_args)
                    # result_batch = self.vae_encoding_stage(result_batch, fastvideo_args)
                    result_batch = self.timestep_preparation_stage(
                        batch, fastvideo_args)
                    result_batch = self.latent_preparation_stage(
                        result_batch, fastvideo_args)
                    result_batch = self.denoising_stage(result_batch,
                                                        fastvideo_args)
                    result_batch = self.decoding_stage(result_batch, fastvideo_args)
                    # trajectory_latents = result_batch.trajectory_latents
                    trajectory_latents.append(result_batch.trajectory_latents.cpu())
                    trajectory_timesteps.append(result_batch.trajectory_timesteps.cpu())
                    trajectory_decoded.append(result_batch.trajectory_decoded)

            extra_features["trajectory_latents"] = trajectory_latents
            extra_features["trajectory_timesteps"] = trajectory_timesteps
            logger.info(f"===== trajectory_latents: {trajectory_latents[0].shape}")
            logger.info(f"===== trajectory_latents len: {len(trajectory_latents)}")
            logger.info(f"===== trajectory_timesteps: {trajectory_timesteps}")
            logger.info(f"===== trajectory_timesteps len: {len(trajectory_timesteps)}")

            if batch.return_trajectory_decoded:
                logger.info(f"===== SAVING TRAJECTORY DECODED")
                for i, decoded_frames in enumerate(trajectory_decoded):
                    for j, decoded_frame in enumerate(decoded_frames):
                        logger.info(f"===== SAVING TRAJECTORY DECODED {i} for prompt {batch_captions[i]}")
                        save_decoded_latents_as_video(decoded_frame, f"decoded_videos/trajectory_decoded_{i}_{j}.mp4", args.train_fps)
            # assert False
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
                        logger.info(f"===== key: {key}")
                        if isinstance(value, torch.Tensor):
                            logger.info(f"===== value: {value[idx].shape}")
                            sample_extra_features[key] = value[idx].cpu().numpy(
                            )
                        else:
                            assert isinstance(value, list)
                            if isinstance(value[idx], torch.Tensor):
                                logger.info(f"===== value in list: {value[idx].shape}")
                                sample_extra_features[key] = value[idx].cpu().float().numpy(
                                )
                            else:
                                logger.info(f"===== value in list: not tensor")
                                sample_extra_features[key] = value[idx]
                            # logger.info(f"===== value: not tensor")
                            # sample_extra_features[key] = value[idx]

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

    def preprocess_text_and_trajectory(self,
                                     fastvideo_args: FastVideoArgs,
                                     args):
        """Preprocess text-only data and generate trajectory information."""
        
        for batch_idx, data in enumerate(self.pbar):
            if data is None:
                continue

            with torch.inference_mode():
                # For text-only processing, we only need text data
                # Filter out samples without text
                valid_indices = []
                for i, text in enumerate(data["text"]):
                    if text and text.strip():  # Check if text is not empty
                        valid_indices.append(i)
                self.num_processed_samples += len(valid_indices)

                if not valid_indices:
                    continue

                # Create new batch with only valid samples (text-only)
                valid_data = {
                    "text": [data["text"][i] for i in valid_indices],
                    "path": [data["path"][i] for i in valid_indices],
                }
                
                # Add fps and duration if available in data
                if "fps" in data:
                    valid_data["fps"] = [data["fps"][i] for i in valid_indices]
                if "duration" in data:
                    valid_data["duration"] = [data["duration"][i] for i in valid_indices]

                batch_captions = valid_data["text"]
                # Encode text using the standalone TextEncodingStage API
                prompt_embeds_list, prompt_masks_list = self.prompt_encoding_stage.encode_text(
                    batch_captions,
                    fastvideo_args,
                    encoder_index=[0],
                    return_attention_mask=True,
                )
                prompt_embeds = prompt_embeds_list[0]
                prompt_attention_masks = prompt_masks_list[0]
                assert prompt_embeds.shape[0] == prompt_attention_masks.shape[0]

                sampling_params = SamplingParam.from_pretrained(
                    args.model_path)

                # encode negative prompt for trajectory collection
                if sampling_params.guidance_scale > 1 and sampling_params.negative_prompt is not None:
                    negative_prompt_embeds_list, negative_prompt_masks_list = self.prompt_encoding_stage.encode_text(
                        sampling_params.negative_prompt,
                        fastvideo_args,
                        encoder_index=[0],
                        return_attention_mask=True,
                    )
                    negative_prompt_embed = negative_prompt_embeds_list[0][0]
                    negative_prompt_attention_mask = negative_prompt_masks_list[0][0]
                else:
                    negative_prompt_embed = None
                    negative_prompt_attention_mask = None

                trajectory_latents = []
                trajectory_timesteps = []
                trajectory_decoded = []
                
                for i, (prompt_embed, prompt_attention_mask) in enumerate(zip(prompt_embeds, prompt_attention_masks)):
                    prompt_embed = prompt_embed.unsqueeze(0)
                    prompt_attention_mask = prompt_attention_mask.unsqueeze(0)
                    
                    # Collect the trajectory data (text-to-video generation)
                    batch = ForwardBatch(
                        **shallow_asdict(sampling_params),
                    )
                    batch.prompt_embeds = [prompt_embed]
                    batch.prompt_attention_mask = [prompt_attention_mask]
                    batch.negative_prompt_embeds = [negative_prompt_embed]
                    batch.negative_attention_mask = [negative_prompt_attention_mask]
                    batch.return_trajectory_latents = True
                    batch.return_trajectory_decoded = False
                    batch.height = args.max_height
                    batch.width = args.max_width
                    batch.fps = args.train_fps
                    batch.guidance_scale = 3.0
                    batch.do_classifier_free_guidance = True

                    result_batch = self.input_validation_stage(
                        batch, fastvideo_args)
                    result_batch = self.timestep_preparation_stage(
                        batch, fastvideo_args)
                    result_batch = self.latent_preparation_stage(
                        result_batch, fastvideo_args)
                    result_batch = self.denoising_stage(result_batch,
                                                        fastvideo_args)
                    result_batch = self.decoding_stage(result_batch, fastvideo_args)
                    
                    trajectory_latents.append(result_batch.trajectory_latents.cpu())
                    trajectory_timesteps.append(result_batch.trajectory_timesteps.cpu())
                    trajectory_decoded.append(result_batch.trajectory_decoded)

                # Prepare extra features for text-only processing
                extra_features = {
                    "trajectory_latents": trajectory_latents,
                    "trajectory_timesteps": trajectory_timesteps
                }

                logger.info(f"===== trajectory_latents: {trajectory_latents[0].shape}")
                logger.info(f"===== trajectory_latents len: {len(trajectory_latents)}")
                logger.info(f"===== trajectory_timesteps: {trajectory_timesteps}")
                logger.info(f"===== trajectory_timesteps len: {len(trajectory_timesteps)}")

                if batch.return_trajectory_decoded:
                    logger.info(f"===== SAVING TRAJECTORY DECODED")
                    for i, decoded_frames in enumerate(trajectory_decoded):
                        for j, decoded_frame in enumerate(decoded_frames):
                            logger.info(f"===== SAVING TRAJECTORY DECODED {i} for prompt {batch_captions[i]}")
                            save_decoded_latents_as_video(decoded_frame, f"decoded_videos/trajectory_decoded_{i}_{j}.mp4", args.train_fps)

                # Prepare batch data for Parquet dataset
                batch_data = []

                # Add progress bar for saving outputs
                save_pbar = tqdm(enumerate(valid_data["path"]),
                                 desc="Saving outputs",
                                 unit="item",
                                 leave=False)
                
                for idx, video_path in save_pbar:
                    video_name = os.path.basename(video_path).split(".")[0]

                    # Convert tensors to numpy arrays
                    text_embedding = prompt_embeds[idx].cpu().numpy()

                    # Get extra features for this sample
                    sample_extra_features = {}
                    if extra_features:
                        for key, value in extra_features.items():
                            logger.info(f"===== key: {key}")
                            if isinstance(value, torch.Tensor):
                                logger.info(f"===== value: {value[idx].shape}")
                                sample_extra_features[key] = value[idx].cpu().numpy()
                            else:
                                assert isinstance(value, list)
                                if isinstance(value[idx], torch.Tensor):
                                    logger.info(f"===== value in list: {value[idx].shape}")
                                    sample_extra_features[key] = value[idx].cpu().float().numpy()
                                else:
                                    logger.info(f"===== value in list: not tensor")
                                    sample_extra_features[key] = value[idx]

                    # Create record for Parquet dataset (without VAE latents for text-only)
                    record = self.create_text_only_record(
                        args,
                        video_name=video_name,
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
        
        # Final flush for any remaining samples
        if hasattr(self, 'all_tables') and self.all_tables and self.num_processed_samples > 0:
            logger.info(f"Final flush with {self.num_processed_samples} remaining samples")
            self._flush_tables(self.num_processed_samples, args, self.combined_parquet_dir)
            self.num_processed_samples = 0
            self.all_tables = []

    def create_text_only_record(
            self,
            args,
            video_name: str,
            text_embedding: np.ndarray,
            valid_data: dict[str, Any],
            idx: int,
            extra_features: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create a record for text-only preprocessing using text-only schema."""
        
        # Create base record using only fields from text-only schema
        record = {
            "id": f"text_{video_name}_{idx}",
            "text_embedding_bytes": text_embedding.tobytes(),
            "text_embedding_shape": list(text_embedding.shape),
            "text_embedding_dtype": str(text_embedding.dtype),
            "file_name": video_name,
            "caption": valid_data["text"][idx],
            "media_type": "text",
        }

        # Add trajectory data if available
        if extra_features and "trajectory_latents" in extra_features:
            trajectory_latents = extra_features["trajectory_latents"][idx] if isinstance(extra_features["trajectory_latents"], list) else extra_features["trajectory_latents"]
            record.update({
                "trajectory_latents_bytes": trajectory_latents.tobytes(),
                "trajectory_latents_shape": list(trajectory_latents.shape),
                "trajectory_latents_dtype": str(trajectory_latents.dtype),
            })
        else:
            record.update({
                "trajectory_latents_bytes": b"",
                "trajectory_latents_shape": [],
                "trajectory_latents_dtype": "",
            })

        if extra_features and "trajectory_timesteps" in extra_features:
            trajectory_timesteps = extra_features["trajectory_timesteps"][idx] if isinstance(extra_features["trajectory_timesteps"], list) else extra_features["trajectory_timesteps"]
            record.update({
                "trajectory_timesteps_bytes": trajectory_timesteps.tobytes(),
                "trajectory_timesteps_shape": list(trajectory_timesteps.shape),
                "trajectory_timesteps_dtype": str(trajectory_timesteps.dtype),
            })
        else:
            record.update({
                "trajectory_timesteps_bytes": b"",
                "trajectory_timesteps_shape": [],
                "trajectory_timesteps_dtype": "",
            })

        return record


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
        pil_images = []
        # Frame has values between -1 and 1
        for frame in first_frame:
            frame = (frame + 1) * 127.5
            frame_pil = Image.fromarray(frame.cpu().numpy().astype(np.uint8))
            pil_images.append(frame_pil)
            # processed_img = self.get_module("image_processor")(
            #     images=frame_pil, return_tensors="pt")
            unprocessed_images.append(frame_pil)
        """Get VAE features from the first frame of each video"""
        video_conditions = []
        for frame in unprocessed_images:

            latent = self.vae_encoding_stage.encode_image(
                frame, height, width, fastvideo_args, generator)
            video_conditions.append(latent)

        features["image_condition_latents"] = video_conditions
        features["pil_images"] = pil_images
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

        if extra_features and "image_condition_latents" in extra_features:
            image_condition_latents = extra_features["image_condition_latents"]
            record.update({
                "image_condition_latents_bytes":
                image_condition_latents.tobytes(),
                "image_condition_latents_shape":
                list(image_condition_latents.shape),
                "image_condition_latents_dtype":
                str(image_condition_latents.dtype),
            })
        else:
            record.update({
                "image_condition_latents_bytes": b"",
                "image_condition_latents_shape": [],
                "image_condition_latents_dtype": "",
            })

        if extra_features and "trajectory_latents" in extra_features:
            trajectory_latents = extra_features["trajectory_latents"]
            record.update({
                "trajectory_latents_bytes": trajectory_latents.tobytes(),
                "trajectory_latents_shape": list(trajectory_latents.shape),
                "trajectory_latents_dtype": str(trajectory_latents.dtype),
            })
        else:
            record.update({
                "trajectory_latents_bytes": b"",
                "trajectory_latents_shape": [],
                "trajectory_latents_dtype": "",
            })

        if extra_features and "trajectory_timesteps" in extra_features:
            trajectory_timesteps = extra_features["trajectory_timesteps"]
            record.update({
                "trajectory_timesteps_bytes": trajectory_timesteps.tobytes(),
                "trajectory_timesteps_shape": list(trajectory_timesteps.shape),
                "trajectory_timesteps_dtype": str(trajectory_timesteps.dtype),
            })
        else:
            record.update({
                "trajectory_timesteps_bytes": b"",
                "trajectory_timesteps_shape": [],
                "trajectory_timesteps_dtype": "",
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
        #train_dataset = getdataset(args)
        train_dataset = gettextdataset(args)

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
        #self.preprocess_video_and_text_and_trajectory(fastvideo_args, args)
        self.preprocess_text_and_trajectory(fastvideo_args, args)


EntryClass = PreprocessPipeline_ODE_Trajectory