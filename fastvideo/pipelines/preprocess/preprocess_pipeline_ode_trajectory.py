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
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset import gettextdataset
from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_ode_trajectory_text_only)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline)
from fastvideo.pipelines.stages import (DecodingStage, DenoisingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        TextEncodingStage,
                                        TimestepPreparationStage)
from fastvideo.utils import save_decoded_latents_as_video, shallow_asdict
from fastvideo.distributed import get_local_torch_device
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)

logger = init_logger(__name__)

class FlowMatchScheduler():

    order = 1

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, device=None):
        sigma_start = self.sigma_min + \
            (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / \
            (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) /
                          num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * \
                (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def step(self, model_output, timestep, sample, to_final=False, return_dict=False, **kwargs):
        assert return_dict is False
        assert kwargs == {}
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        logger.info('step timestep: %s', timestep)
        logger.info('step timestep: %s', timestep.shape)
        # timestep  is [num_frames]
        # timestep_id = torch.argmin(
        #     (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        # assert timestep.ndim == 1
        # assert timestep.shape[0] == 1
        timestep_id = torch.argmin(
            (self.timesteps - timestep).abs(), dim=0)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_ = 1 if (
                self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return (prev_sample,)

    def scale_model_input(self, sample: torch.Tensor, *args,
                          **kwargs) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B, C, H, W]
            - noise: the noise with shape [B, C, H, W]
            - timestep: the timestep with shape [B]
        Output: the corrupted latent with shape [B, C, H, W]
        """
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights


class PreprocessPipeline_ODE_Trajectory(BasePreprocessPipeline):
    """ODE Trajectory preprocessing pipeline implementation."""

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]
    preprocess_dataloader: StatefulDataLoader
    preprocess_loader_iter: Iterator[dict[str, Any]]
    pbar: Any
    num_processed_samples: int

    def get_schema_fields(self) -> list[str]:
        """Get the schema fields for ODE Trajectory pipeline."""
        return [f.name for f in pyarrow_schema_ode_trajectory_text_only]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        logger.info('WTF flow_shift: %s', fastvideo_args.pipeline_config.flow_shift)
        assert fastvideo_args.pipeline_config.flow_shift  == 5
        # self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            # shift=fastvideo_args.pipeline_config.flow_shift)
        self.modules["scheduler"] = FlowMatchScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.modules["scheduler"].set_timesteps(num_inference_steps=48, denoising_strength=1.0)
        logger.info('WTF scheduler timesteps: %s', self.modules["scheduler"].timesteps)
        # scheduler = FlowMatchScheduler(
        #     shift=8.0, sigma_min=0.0, extra_one_step=True)
        # device = get_local_torch_device()
        # # scheduler.num_train_timesteps = 100
        # scheduler.set_timesteps(num_inference_steps=50, denoising_strength=1.0)
        # scheduler.sigmas = scheduler.sigmas.to(device)
        # self.modules["scheduler"] = scheduler

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
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
                           scheduler=self.get_module("scheduler"),
                           pipeline=self,
                       ))
        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))

    def preprocess_text_and_trajectory(self, fastvideo_args: FastVideoArgs,
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
                    valid_data["duration"] = [
                        data["duration"][i] for i in valid_indices
                    ]

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

                sampling_params = SamplingParam.from_pretrained(args.model_path)

                # encode negative prompt for trajectory collection
                if sampling_params.guidance_scale > 1 and sampling_params.negative_prompt is not None:
                    negative_prompt_embeds_list, negative_prompt_masks_list = self.prompt_encoding_stage.encode_text(
                        sampling_params.negative_prompt,
                        fastvideo_args,
                        encoder_index=[0],
                        return_attention_mask=True,
                    )
                    negative_prompt_embed = negative_prompt_embeds_list[0][0]
                    negative_prompt_attention_mask = negative_prompt_masks_list[
                        0][0]
                else:
                    negative_prompt_embed = None
                    negative_prompt_attention_mask = None

                trajectory_latents = []
                trajectory_timesteps = []
                trajectory_decoded = []

                for i, (prompt_embed, prompt_attention_mask) in enumerate(
                        zip(prompt_embeds, prompt_attention_masks,
                            strict=False)):
                    prompt_embed = prompt_embed.unsqueeze(0)
                    prompt_attention_mask = prompt_attention_mask.unsqueeze(0)

                    # Collect the trajectory data (text-to-video generation)
                    batch = ForwardBatch(**shallow_asdict(sampling_params), )
                    batch.prompt_embeds = [prompt_embed]
                    batch.prompt_attention_mask = [prompt_attention_mask]
                    batch.negative_prompt_embeds = [negative_prompt_embed]
                    batch.negative_attention_mask = [
                        negative_prompt_attention_mask
                    ]
                    batch.return_trajectory_latents = True
                    batch.return_trajectory_decoded = True
                    batch.height = args.max_height
                    batch.width = args.max_width
                    batch.fps = args.train_fps
                    batch.guidance_scale = 6.0
                    batch.do_classifier_free_guidance = True

                    result_batch = self.input_validation_stage(
                        batch, fastvideo_args)
                    result_batch = self.timestep_preparation_stage(
                        batch, fastvideo_args)
                    result_batch = self.latent_preparation_stage(
                        result_batch, fastvideo_args)
                    result_batch = self.denoising_stage(result_batch,
                                                        fastvideo_args)
                    result_batch = self.decoding_stage(result_batch,
                                                       fastvideo_args)

                    trajectory_latents.append(
                        result_batch.trajectory_latents.cpu())
                    trajectory_timesteps.append(
                        result_batch.trajectory_timesteps.cpu())
                    trajectory_decoded.append(result_batch.trajectory_decoded)

                # Prepare extra features for text-only processing
                extra_features = {
                    "trajectory_latents": trajectory_latents,
                    "trajectory_timesteps": trajectory_timesteps
                }

                if batch.return_trajectory_decoded:
                    for i, decoded_frames in enumerate(trajectory_decoded):
                        for j, decoded_frame in enumerate(decoded_frames):
                            save_decoded_latents_as_video(
                                decoded_frame,
                                f"decoded_videos/trajectory_decoded_{i}_{j}.mp4",
                                args.train_fps)

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
                            if isinstance(value, torch.Tensor):
                                sample_extra_features[key] = value[idx].cpu(
                                ).numpy()
                            else:
                                assert isinstance(value, list)
                                if isinstance(value[idx], torch.Tensor):
                                    sample_extra_features[key] = value[idx].cpu(
                                    ).float().numpy()
                                else:
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
                                pa.array(
                                    [record[field] for record in batch_data],
                                    type=pa.binary()))
                        elif field.endswith('_shape'):
                            arrays.append(
                                pa.array(
                                    [record[field] for record in batch_data],
                                    type=pa.list_(pa.int32())))
                        elif field in ['width', 'height', 'num_frames']:
                            arrays.append(
                                pa.array(
                                    [record[field] for record in batch_data],
                                    type=pa.int32()))
                        elif field in ['duration_sec', 'fps']:
                            arrays.append(
                                pa.array(
                                    [record[field] for record in batch_data],
                                    type=pa.float32()))
                        else:
                            arrays.append(
                                pa.array(
                                    [record[field] for record in batch_data]))

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
        if hasattr(self, 'all_tables'
                   ) and self.all_tables and self.num_processed_samples > 0:
            logger.info("Final flush with %s remaining samples",
                        self.num_processed_samples)
            self._flush_tables(self.num_processed_samples, args,
                               self.combined_parquet_dir)
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

        assert extra_features is not None, "extra_features is required"
        assert "trajectory_latents" in extra_features, "trajectory_latents is required"
        assert "trajectory_timesteps" in extra_features, "trajectory_timesteps is required"

        # Add trajectory data if available
        if extra_features and "trajectory_latents" in extra_features:
            trajectory_latents = extra_features[
                "trajectory_latents"][idx] if isinstance(
                    extra_features["trajectory_latents"],
                    list) else extra_features["trajectory_latents"]
            record.update({
                "trajectory_latents_bytes":
                trajectory_latents.tobytes(),
                "trajectory_latents_shape":
                list(trajectory_latents.shape),
                "trajectory_latents_dtype":
                str(trajectory_latents.dtype),
            })
        else:
            record.update({
                "trajectory_latents_bytes": b"",
                "trajectory_latents_shape": [],
                "trajectory_latents_dtype": "",
            })

        if extra_features and "trajectory_timesteps" in extra_features:
            trajectory_timesteps = extra_features[
                "trajectory_timesteps"][idx] if isinstance(
                    extra_features["trajectory_timesteps"],
                    list) else extra_features["trajectory_timesteps"]
            record.update({
                "trajectory_timesteps_bytes":
                trajectory_timesteps.tobytes(),
                "trajectory_timesteps_shape":
                list(trajectory_timesteps.shape),
                "trajectory_timesteps_dtype":
                str(trajectory_timesteps.dtype),
            })
        else:
            record.update({
                "trajectory_timesteps_bytes": b"",
                "trajectory_timesteps_shape": [],
                "trajectory_timesteps_dtype": "",
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
        self.preprocess_text_and_trajectory(fastvideo_args, args)


EntryClass = PreprocessPipeline_ODE_Trajectory
