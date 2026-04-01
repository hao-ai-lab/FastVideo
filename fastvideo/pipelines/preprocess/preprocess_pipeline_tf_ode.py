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
from unittest import result

import pyarrow as pa
import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset import build_parquet_map_style_dataloader
from fastvideo.dataset.dataloader.parquet_io import (ParquetDatasetWriter,
                                                     records_to_table)
from fastvideo.dataset.dataloader.record_schema import (
    ode_text_only_record_creator)
from fastvideo.dataset.dataloader.schema import pyarrow_schema_ode_trajectory_text_only, pyarrow_schema_t2v
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler)
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

logger = init_logger(__name__)


class PreprocessPipeline_TF_ODE(BasePreprocessPipeline):
    """ODE Trajectory preprocessing pipeline implementation."""

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]
    preprocess_dataloader: StatefulDataLoader
    preprocess_loader_iter: Iterator[dict[str, Any]]
    pbar: Any
    num_processed_samples: int

    def get_pyarrow_schema(self) -> pa.Schema:
        """Return the PyArrow schema for ODE Trajectory pipeline."""
        return pyarrow_schema_ode_trajectory_text_only

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

    def _process_batch(self, batch):
        # Required fields from parquet (ODE trajectory schema)
        encoder_hidden_states = batch['text_embedding']
        encoder_attention_mask = batch['text_attention_mask']
        latent = batch['vae_latent'].float()
        infos = batch['info_list'][0]

        if (hasattr(self.modules["vae"], "shift_factor")
                    and self.modules["vae"].shift_factor is not None):
            if isinstance(self.modules["vae"].shift_factor, torch.Tensor):
                latent -= self.modules["vae"].shift_factor.to(
                    latent.device, latent.dtype)
            else:
                latent -= self.modules["vae"].shift_factor
        else:
            raise ValueError("vae.shift_factor is not set")

        if isinstance(self.modules["vae"].scaling_factor, torch.Tensor):
            latent = latent * self.modules["vae"].scaling_factor.to(
                latent.device, latent.dtype)
        else:
            latent = latent * self.modules["vae"].scaling_factor

        # Move to device
        device = get_local_torch_device()
        encoder_hidden_states = encoder_hidden_states.to(
            device, dtype=torch.bfloat16)
        encoder_attention_mask = encoder_attention_mask.to(
            device, dtype=torch.bfloat16)

        return encoder_hidden_states, encoder_attention_mask, latent.to(device, dtype=torch.bfloat16), infos

    def preprocess_text_and_trajectory(self, fastvideo_args: FastVideoArgs,
                                       args):
        """Preprocess text-only data and generate trajectory information."""

        for batch_idx, data in enumerate(self.pbar):
            if data is None:
                continue

            with torch.inference_mode():
                sampling_params = SamplingParam.from_pretrained(args.model_path)

                # encode negative prompt for trajectory collection
                if sampling_params.guidance_scale > 1 and sampling_params.negative_prompt is not None:
                    negative_prompt_embeds_list, negative_prompt_masks_list = self.prompt_encoding_stage.encode_text(
                        sampling_params.negative_prompt,
                        fastvideo_args,
                        encoder_index=[0],
                        return_attention_mask=True,
                    )
                    negative_prompt_embed = negative_prompt_embeds_list[0]
                    negative_prompt_attention_mask = negative_prompt_masks_list[
                        0]
                else:
                    negative_prompt_embed = None
                    negative_prompt_attention_mask = None

                trajectory_latents = []
                trajectory_timesteps = []
                trajectory_decoded = []

                prompt_embeds, prompt_attention_masks, clean_latents, infos = self._process_batch(data)
                assert prompt_embeds.shape[0] == 1, "Only one prompt embeds are supported"
                self.num_processed_samples += prompt_embeds.shape[0]

                # Collect the trajectory data (text-to-video generation)
                batch = ForwardBatch(**shallow_asdict(sampling_params), )
                batch.prompt_embeds = [prompt_embeds]
                batch.prompt_attention_mask = [prompt_attention_masks]
                batch.negative_prompt_embeds = [negative_prompt_embed]
                batch.negative_attention_mask = [
                    negative_prompt_attention_mask
                ]
                batch.clean_latents = clean_latents
                batch.num_inference_steps = 48
                batch.return_trajectory_latents = True
                # Enabling this will save the decoded trajectory videos.
                # Used for debugging.
                batch.return_trajectory_decoded = False
                batch.height = args.max_height
                batch.width = args.max_width
                batch.fps = args.train_fps
                batch.guidance_scale = 6.0
                batch.do_classifier_free_guidance = True

                result_batch = self.input_validation_stage(
                    batch, fastvideo_args)
                # result_batch = self.timestep_preparation_stage(
                #     result_batch, fastvideo_args)
                result_batch.timesteps = self.get_module("scheduler").timesteps
                result_batch = self.latent_preparation_stage(
                    result_batch, fastvideo_args)
                result_batch = self.denoising_stage(result_batch,
                                                    fastvideo_args)
                if batch.return_trajectory_decoded:
                    result_batch = self.decoding_stage(result_batch,
                                                    fastvideo_args)
                    trajectory_decoded.append(result_batch.trajectory_decoded)

                result_batch.trajectory_latents = result_batch.trajectory_latents[:, [0, 12, 24, 36, -2, -1]]
                result_batch.trajectory_timesteps = result_batch.trajectory_timesteps[[0, 12, 24, 36, -2, -1]]
                trajectory_latents.append(
                    result_batch.trajectory_latents.cpu())
                trajectory_timesteps.append(
                    result_batch.trajectory_timesteps.cpu())

                # Prepare extra features for text-only processing
                extra_features = {
                    "trajectory_latents": trajectory_latents,
                    "trajectory_timesteps": trajectory_timesteps
                }

                if batch.return_trajectory_decoded:
                    for i, decoded_frames in enumerate(trajectory_decoded):
                        for j, decoded_frame in enumerate(decoded_frames):
                            if j in [50, 51]:
                                local_rank = int(os.getenv("RANK", 0))
                                save_decoded_latents_as_video(
                                    decoded_frame,
                                    f"decoded_videos/trajectory_decoded_{local_rank}_{j}.mp4",
                                    args.train_fps)

                # Prepare batch data for Parquet dataset
                batch_data: list[dict[str, Any]] = []

                # Add progress bar for saving outputs
                save_pbar = tqdm(enumerate([infos["file_name"]]),
                                 desc="Saving outputs",
                                 unit="item",
                                 leave=False)

                for idx, video_path in save_pbar:
                    video_name = os.path.basename(video_path).split(".")[0]

                    # Convert tensors to numpy arrays
                    text_embedding = prompt_embeds.float().cpu().numpy()

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

                    # Create record for Parquet dataset (text-only ODE schema)
                    record: dict[str, Any] = ode_text_only_record_creator(
                        video_name=video_name,
                        text_embedding=text_embedding,
                        caption=infos["caption"],
                        trajectory_latents=sample_extra_features[
                            "trajectory_latents"],
                        trajectory_timesteps=sample_extra_features[
                            "trajectory_timesteps"],
                    )
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
        self.train_dataset, self.preprocess_dataloader = build_parquet_map_style_dataloader(
            args.data_merge_path,
            args.preprocess_video_batch_size,
            # parquet_schema=pyarrow_schema_ode_trajectory_text_only,
            parquet_schema=pyarrow_schema_t2v,
            num_data_workers=args.dataloader_num_workers,
            cfg_rate=0.0,
            drop_last=True,
            text_padding_length=fastvideo_args.pipeline_config.
            text_encoder_configs[0].arch_config.
            text_len,  # type: ignore[attr-defined]
            seed=args.seed)

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


EntryClass = PreprocessPipeline_TF_ODE
