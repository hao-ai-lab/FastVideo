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
from fastvideo.dataset import getdataset
from fastvideo.dataset.dataloader.parquet_io import (ParquetDatasetWriter,
                                                     records_to_table)
from fastvideo.dataset.dataloader.record_schema import (
    wangame_ode_record_creator)
from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_ode_trajectory_wangame)
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline)
from fastvideo.pipelines.stages import (DecodingStage, DenoisingStage,
                                        InputValidationStage,
                                        LatentPreparationStage,
                                        ImageEncodingStage,
                                        TimestepPreparationStage)
from fastvideo.utils import save_decoded_latents_as_video, shallow_asdict

logger = init_logger(__name__)


class PreprocessPipeline_WanGame_ODE_Trajectory(BasePreprocessPipeline):
    """ODE Trajectory preprocessing pipeline implementation."""

    _required_config_modules = [
        "vae", "image_encoder", "image_processor", "transformer", "scheduler"
    ]

    preprocess_dataloader: StatefulDataLoader
    preprocess_loader_iter: Iterator[dict[str, Any]]
    pbar: Any
    num_processed_samples: int

    def get_pyarrow_schema(self) -> pa.Schema:
        """Return the PyArrow schema for ODE Trajectory pipeline."""
        return pyarrow_schema_ode_trajectory_wangame

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        assert fastvideo_args.pipeline_config.flow_shift == 5
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.modules["scheduler"].set_timesteps(num_inference_steps=48,
                                                denoising_strength=1.0)

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())
        self.add_stage(stage_name="image_encoding_stage",
                       stage=ImageEncodingStage(
                           image_encoder=self.get_module("image_encoder"),
                           image_processor=self.get_module("image_processor"),
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

    def get_extra_features(self, valid_data: dict[str, Any],
                           fastvideo_args: FastVideoArgs) -> dict[str, Any]:

        # TODO(will): move these to cpu at some point
        self.get_module("image_encoder").to(get_local_torch_device())
        self.get_module("vae").to(get_local_torch_device())

        features = {}
        """Get CLIP features from the first frame of each video."""
        first_frame = valid_data["pixel_values"][:, :, 0, :, :].permute(
            0, 2, 3, 1)  # (B, C, T, H, W) -> (B, H, W, C)
        _, _, num_frames, height, width = valid_data["pixel_values"].shape
        # latent_height = height // self.get_module(
        #     "vae").spatial_compression_ratio
        # latent_width = width // self.get_module("vae").spatial_compression_ratio

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
            dim=0).to(get_local_torch_device())
        with torch.no_grad():
            image_inputs = {'pixel_values': pixel_values}
            with set_forward_context(current_timestep=0, attn_metadata=None):
                clip_features = self.get_module("image_encoder")(**image_inputs)
            clip_features = clip_features.last_hidden_state

        features["clip_feature"] = clip_features
        features["pil_image"] = first_frame
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
            video_condition = video_condition.to(
                device=get_local_torch_device(), dtype=torch.float32)
            video_conditions.append(video_condition)

        video_conditions = torch.cat(video_conditions, dim=0)

        with torch.autocast(device_type="cuda",
                            dtype=torch.float32,
                            enabled=True):
            encoder_outputs = self.get_module("vae").encode(video_conditions)

        # Use mode() instead of mean
        latent_condition = encoder_outputs.mode()

        # Use latents_mean/latents_std normalization to match
        vae = self.get_module("vae")
        if (hasattr(vae.config, 'latents_mean')
                and hasattr(vae.config, 'latents_std')):
            latents_mean = torch.tensor(vae.config.latents_mean,
                                        device=latent_condition.device,
                                        dtype=latent_condition.dtype).view(
                                            1, -1, 1, 1, 1)
            latents_std = torch.tensor(vae.config.latents_std,
                                       device=latent_condition.device,
                                       dtype=latent_condition.dtype).view(
                                           1, -1, 1, 1, 1)
            latent_condition = (latent_condition - latents_mean) / latents_std
        elif (hasattr(vae, "shift_factor") and vae.shift_factor is not None):
            if isinstance(vae.shift_factor, torch.Tensor):
                latent_condition -= vae.shift_factor.to(
                    latent_condition.device, latent_condition.dtype)
            else:
                latent_condition -= vae.shift_factor

            if isinstance(vae.scaling_factor, torch.Tensor):
                latent_condition = latent_condition * vae.scaling_factor.to(
                    latent_condition.device, latent_condition.dtype)
            else:
                latent_condition = latent_condition * vae.scaling_factor

        # mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height,
        #                            latent_width)
        # mask_lat_size[:, :, list(range(1, num_frames))] = 0
        # first_frame_mask = mask_lat_size[:, :, 0:1]
        # first_frame_mask = torch.repeat_interleave(
        #     first_frame_mask,
        #     dim=2,
        #     repeats=self.get_module("vae").temporal_compression_ratio)
        # mask_lat_size = torch.concat(
        #     [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        # mask_lat_size = mask_lat_size.view(
        #     batch_size, -1,
        #     self.get_module("vae").temporal_compression_ratio, latent_height,
        #     latent_width)
        # mask_lat_size = mask_lat_size.transpose(1, 2)
        # mask_lat_size = mask_lat_size.to(latent_condition.device)

        # image_latent = torch.concat([mask_lat_size, latent_condition], dim=1)

        # Create mask_cond: ones for first frame, zeros for rest
        # Shape: (B, 16, latent_frames, latent_height, latent_width)
        mask_cond = torch.ones_like(latent_condition)
        mask_cond[:, :, 1:] = 0  # Set all frames except first to 0
        # Create cond_concat: first 4 channels of mask + all 16 channels of img_cond
        # Shape: (B, 20, latent_frames, latent_height, latent_width)
        cond_concat = torch.cat([mask_cond[:, :4], latent_condition], dim=1)
        features["first_frame_latent"] = cond_concat

        if "action_path" in valid_data and valid_data["action_path"]:
            keyboard_cond_list = []
            mouse_cond_list = []
            arch_cfg = self.get_module("transformer").config.arch_config
            action_cfg = getattr(arch_cfg, "action_config", {}) or {}
            keyboard_dim = action_cfg.get("keyboard_dim_in", None)
            for action_path in valid_data["action_path"]:
                if action_path:
                    action_data = np.load(action_path, allow_pickle=True)
                    if isinstance(action_data,
                                  np.ndarray) and action_data.dtype == np.dtype('O'):
                        action_dict = action_data.item()
                        if "keyboard" in action_dict:
                            keyboard = action_dict["keyboard"].astype(np.float32)
                            if keyboard_dim is not None:
                                if keyboard.ndim >= 2:
                                    keyboard = keyboard[:, :keyboard_dim]
                                else:
                                    keyboard = keyboard[:keyboard_dim]
                            keyboard_cond_list.append(keyboard)
                        if "mouse" in action_dict:
                            mouse_cond_list.append(action_dict["mouse"])
                    else:
                        keyboard = action_data.astype(np.float32)
                        if keyboard_dim is not None:
                            if keyboard.ndim >= 2:
                                keyboard = keyboard[:, :keyboard_dim]
                            else:
                                keyboard = keyboard[:keyboard_dim]
                        keyboard_cond_list.append(keyboard)
            if keyboard_cond_list:
                features["keyboard_cond"] = keyboard_cond_list
            if mouse_cond_list:
                features["mouse_cond"] = mouse_cond_list
        return features

    def preprocess_action_and_trajectory(self, fastvideo_args: FastVideoArgs,
                                         args):
        """Preprocess data and generate trajectory information."""

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
                    "path": [data["path"][i] for i in valid_indices],
                }

                if "fps" in data:
                    valid_data["fps"] = [data["fps"][i] for i in valid_indices]
                if "duration" in data:
                    valid_data["duration"] = [
                        data["duration"][i] for i in valid_indices
                    ]
                if "action_path" in data:
                    valid_data["action_path"] = [
                        data["action_path"][i] for i in valid_indices
                    ]

                pixel_values = valid_data["pixel_values"]
                if pixel_values.shape[2] == 1 and args.num_frames is not None:
                    pixel_values = pixel_values.repeat(
                        1, 1, args.num_frames, 1, 1)
                    valid_data["pixel_values"] = pixel_values

                # Get extra features if needed
                extra_features = self.get_extra_features(
                    valid_data, fastvideo_args)

                clip_features = extra_features['clip_feature']
                image_latents = extra_features['first_frame_latent']
                image_latents = image_latents[:, :, :args.num_latent_t]
                pil_image = extra_features['pil_image']
                if "keyboard_cond" in extra_features:
                    keyboard_cond = extra_features['keyboard_cond']
                else:
                    keyboard_cond = None
                if "mouse_cond" in extra_features:
                    mouse_cond = extra_features['mouse_cond']
                else:
                    mouse_cond = None

                sampling_params = SamplingParam.from_pretrained(args.model_path)

                trajectory_latents = []
                trajectory_timesteps = []
                trajectory_decoded = []

                device = get_local_torch_device()
                for i in range(len(valid_indices)):
                    # Collect the trajectory data
                    batch = ForwardBatch(**shallow_asdict(sampling_params), )
                    batch.image_embeds = [clip_features[i].unsqueeze(0)]
                    batch.image_latent = image_latents[i].unsqueeze(0)
                    batch.keyboard_cond = (torch.from_numpy(
                        keyboard_cond[i]).unsqueeze(0).to(device) if
                                           keyboard_cond is not None else None)
                    batch.mouse_cond = (torch.from_numpy(
                        mouse_cond[i]).unsqueeze(0).to(device)
                                        if mouse_cond is not None else None)
                    batch.num_inference_steps = 48
                    batch.return_trajectory_latents = True
                    # Enabling this will save the decoded trajectory videos.
                    # Used for debugging.
                    batch.return_trajectory_decoded = False
                    batch.height = args.max_height
                    batch.width = args.max_width
                    batch.fps = args.train_fps
                    batch.num_frames = valid_data["pixel_values"].shape[2]
                    batch.guidance_scale = 6.0
                    batch.do_classifier_free_guidance = False
                    batch.prompt = ""
                    batch.prompt_embeds = [
                        torch.zeros((1, 0, self.get_module("transformer").hidden_size),
                                    dtype=torch.bfloat16,
                                    device=device)
                    ]

                    result_batch = self.input_validation_stage(
                        batch, fastvideo_args)
                    result_batch = self.timestep_preparation_stage(
                        result_batch, fastvideo_args)
                    result_batch.timesteps = result_batch.timesteps.to(device)
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

                # Prepare extra features
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
                batch_data: list[dict[str, Any]] = []

                # Add progress bar for saving outputs
                save_pbar = tqdm(enumerate(valid_data["path"]),
                                 desc="Saving outputs",
                                 unit="item",
                                 leave=False)

                for idx, video_path in save_pbar:
                    video_name = os.path.basename(video_path).split(".")[0]

                    clip_feature_np = clip_features[idx].cpu().numpy()
                    first_frame_latent_np = image_latents[idx].cpu().numpy()
                    pil_image_np = pil_image[idx].cpu().numpy()
                    keyboard_cond_np = keyboard_cond[
                        idx] if keyboard_cond is not None else None
                    mouse_cond_np = mouse_cond[
                        idx] if mouse_cond is not None else None

                    # Get trajectory features for this sample
                    traj_latents = extra_features["trajectory_latents"][idx]
                    traj_timesteps = extra_features["trajectory_timesteps"][idx]
                    if isinstance(traj_latents, torch.Tensor):
                        traj_latents = traj_latents.cpu().float().numpy()
                    if isinstance(traj_timesteps, torch.Tensor):
                        traj_timesteps = traj_timesteps.cpu().float().numpy()

                    # Create record for Parquet dataset
                    record: dict[str, Any] = wangame_ode_record_creator(
                        video_name=video_name,
                        clip_feature=clip_feature_np,
                        first_frame_latent=first_frame_latent_np,
                        trajectory_latents=traj_latents,
                        trajectory_timesteps=traj_timesteps,
                        pil_image=pil_image_np,
                        keyboard_cond=keyboard_cond_np,
                        mouse_cond=mouse_cond_np,
                        caption="")
                    batch_data.append(record)

                if batch_data:
                    write_pbar = tqdm(total=1,
                                      desc="Writing to Parquet dataset",
                                      unit="batch")
                    table = records_to_table(batch_data,
                                             self.get_pyarrow_schema())
                    write_pbar.update(1)
                    write_pbar.close()

                    if not hasattr(self, 'dataset_writer'):
                        self.dataset_writer = ParquetDatasetWriter(
                            out_dir=self.combined_parquet_dir,
                            samples_per_file=args.samples_per_file,
                        )
                    self.dataset_writer.append_table(table)

                    logger.info("Collected batch with %s samples", len(table))

                if self.num_processed_samples >= args.flush_frequency:
                    written = self.dataset_writer.flush()
                    logger.info("Flushed %s samples to parquet", written)
                    self.num_processed_samples = 0

        # Final flush for any remaining samples
        if hasattr(self, 'dataset_writer'):
            written = self.dataset_writer.flush(write_remainder=True)
            if written:
                logger.info("Final flush wrote %s samples", written)

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
        self.preprocess_action_and_trajectory(fastvideo_args, args)


EntryClass = PreprocessPipeline_WanGame_ODE_Trajectory
