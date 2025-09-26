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
from fastvideo.dataset import gettextdataset
from fastvideo.dataset.dataloader.parquet_io import (ParquetDatasetWriter,
                                                     records_to_table)
from fastvideo.dataset.dataloader.record_schema import (
    ode_text_only_record_creator)
from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_ode_trajectory_text_only)
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

from fastvideo.forward_context import set_forward_context
from fastvideo.models.utils import pred_noise_to_pred_video, pred_video_to_pred_noise
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.distributed import get_local_torch_device

logger = init_logger(__name__)


class PreprocessPipeline_ODE_Trajectory(BasePreprocessPipeline):
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

        fastvideo_args.model_loaded["transformer"] = False
        loader = TransformerLoader()
        fastvideo_args.pipeline_config.dit_precision = "fp32" # Overwrite the precision to fp32 for transformer
        fastvideo_args.pipeline_config.dit_forward_precision = "fp32"
        self.transformer = loader.load(
            fastvideo_args.model_paths["transformer"], fastvideo_args)
        self.add_module("transformer", self.transformer)
        fastvideo_args.model_loaded["transformer"] = True

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
            # logger.info("transformer weight sum: %s", sum(p.float().sum().item() for p in self.transformer.parameters()))
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
                self.prompt_encoding_stage.text_encoders[0] = self.prompt_encoding_stage.text_encoders[0].to(dtype=torch.bfloat16).to(dtype=torch.float32)
                # Encode text using the standalone TextEncodingStage API
                prompt_embeds_list, prompt_masks_list = self.prompt_encoding_stage.encode_text(
                    batch_captions,
                    fastvideo_args,
                    encoder_index=[0],
                    return_attention_mask=True,
                )
                prompt_embeds = prompt_embeds_list[0]
                logger.info("prompt_embeds sum: %s, prompt_embeds shape: %s, prompt_embeds dtype: %s", prompt_embeds.float().sum(), prompt_embeds.shape, prompt_embeds.dtype)
                prompt_attention_masks = prompt_masks_list[0]
                assert prompt_embeds.shape[0] == prompt_attention_masks.shape[0]

                sampling_params = SamplingParam.from_pretrained(args.model_path)

                negative_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

                # encode negative prompt for trajectory collection
                if sampling_params.guidance_scale > 1 and sampling_params.negative_prompt is not None:
                    negative_prompt_embeds_list, negative_prompt_masks_list = self.prompt_encoding_stage.encode_text(
                        negative_prompt,
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
                    batch.num_inference_steps = 48
                    batch.return_trajectory_latents = True
                    # Enabling this will save the decoded trajectory videos.
                    # Used for debugging.
                    batch.return_trajectory_decoded = True
                    batch.height = args.max_height
                    batch.width = args.max_width
                    batch.fps = args.train_fps
                    batch.guidance_scale = 3.0
                    batch.do_classifier_free_guidance = True

                    result_batch = self.input_validation_stage(
                        batch, fastvideo_args)
                    result_batch = self.timestep_preparation_stage(
                        batch, fastvideo_args)
                    # result_batch = self.latent_preparation_stage(
                    #     result_batch, fastvideo_args)
                    # result_batch = self.denoising_stage(result_batch,
                    #                                     fastvideo_args)
                    noisy_input = []
                    # latents = result_batch.latents.permute(0, 2, 1, 3, 4)
                    latents = torch.randn(
                        [1, 21, 16, 60, 104], dtype=torch.float32, device=get_local_torch_device()
                    )
                    # logger.info("transformer weight sum: %s", sum(p.float().sum().item() for p in self.transformer.parameters()))
                    logger.info("latents sum: %s, latents shape: %s, latents dtype: %s", latents.float().sum(), latents.shape, latents.dtype)

                    logger.info("scheduler timesteps: %s", self.get_module("scheduler").timesteps)
                    for progress_id, t in enumerate(tqdm(self.get_module("scheduler").timesteps)):
                        timestep = t * \
                            torch.ones([1, 21], device=latents.device, dtype=torch.float32)

                        noisy_input.append(latents)

                        with set_forward_context(
                            current_timestep=0,
                            attn_metadata=None,
                            forward_batch=None,
                        ):
                            # logger.info("prompt_embed sum: %s, prompt_embed shape: %s, prompt_embed dtype: %s", prompt_embed.float().sum(), prompt_embed.shape, prompt_embed.dtype)
                            # logger.info("timestep: %s", timestep[:, 0])
                            # Run transformer
                            cond_pred_noise_btchw = self.transformer(
                                hidden_states=latents.permute(0, 2, 1, 3, 4),
                                encoder_hidden_states=prompt_embed,
                                timestep=timestep[:, 0]
                            ).permute(0, 2, 1, 3, 4)

                        # logger.info("cond_pred_noise_btchw sum: %s, cond_pred_noise_btchw shape: %s, cond_pred_noise_btchw dtype: %s", cond_pred_noise_btchw.float().sum(), cond_pred_noise_btchw.shape, cond_pred_noise_btchw.dtype)

                        cond_pred_video_btchw = pred_noise_to_pred_video(
                            pred_noise=cond_pred_noise_btchw.flatten(0, 1),
                            noise_input_latent=latents.flatten(0, 1),
                            timestep=timestep.flatten(0, 1),
                            scheduler=self.get_module("scheduler")).unflatten(
                                0, cond_pred_noise_btchw.shape[:2])

                        # logger.info("cond_pred_video_btchw sum: %s, cond_pred_video_btchw shape: %s, cond_pred_video_btchw dtype: %s", cond_pred_video_btchw.float().sum(), cond_pred_video_btchw.shape, cond_pred_video_btchw.dtype)

                        with set_forward_context(
                            current_timestep=t,
                            attn_metadata=None,
                            forward_batch=result_batch,
                        ):
                            # logger.info("latents sum: %s, latents shape: %s, latents dtype: %s", latents.float().sum(), latents.shape, latents.dtype)
                            # Run transformer
                            uncond_pred_noise_btchw = self.transformer(
                                latents.permute(0, 2, 1, 3, 4),
                                negative_prompt_embed,
                                timestep[:, 0]
                            ).permute(0, 2, 1, 3, 4)

                        # logger.info("uncond_pred_noise_btchw sum: %s, uncond_pred_noise_btchw shape: %s, uncond_pred_noise_btchw dtype: %s", uncond_pred_noise_btchw.float().sum(), uncond_pred_noise_btchw.shape, uncond_pred_noise_btchw.dtype)

                        uncond_pred_video_btchw = pred_noise_to_pred_video(
                            pred_noise=uncond_pred_noise_btchw.flatten(0, 1),
                            noise_input_latent=latents.flatten(0, 1),
                            timestep=timestep.flatten(0, 1),
                            scheduler=self.get_module("scheduler")).unflatten(
                                0, uncond_pred_noise_btchw.shape[:2])

                        pred_video_btchw = uncond_pred_video_btchw + batch.guidance_scale * (
                            cond_pred_video_btchw - uncond_pred_video_btchw
                        )

                        # logger.info("pred_video_btchw sum: %s, pred_video_btchw shape: %s, pred_video_btchw dtype: %s", pred_video_btchw.float().sum(), pred_video_btchw.shape, pred_video_btchw.dtype)

                        pred_noise_btchw = pred_video_to_pred_noise(
                            x0_pred=pred_video_btchw.flatten(0, 1),
                            xt=latents.flatten(0, 1),
                            timestep=timestep.flatten(0, 1),
                            scheduler=self.get_module("scheduler")).unflatten(
                                0, pred_video_btchw.shape[:2])

                        # logger.info("pred_noise_btchw sum: %s, pred_noise_btchw shape: %s, pred_noise_btchw dtype: %s", pred_noise_btchw.float().sum(), pred_noise_btchw.shape, pred_noise_btchw.dtype)

                        latents = self.get_module("scheduler").step(
                            pred_noise_btchw.flatten(0, 1),
                            self.get_module("scheduler").timesteps[progress_id] * torch.ones(
                                [1, 21], device=latents.device, dtype=torch.long).flatten(0, 1),
                            latents.flatten(0, 1)
                        )[0].unflatten(dim=0, sizes=pred_noise_btchw.shape[:2])

                        # logger.info("latents sum: %s, latents shape: %s, latents dtype: %s", latents.float().sum(), latents.shape, latents.dtype)

                    noisy_input.append(latents)

                    noisy_inputs = torch.stack(noisy_input, dim=1)

                    noisy_inputs = noisy_inputs[:, [0, 12, 24, 36, -1]].half()

                    logger.info("noisy inputs sum: %s, noisy inputs shape: %s, noisy inputs dtype: %s", noisy_inputs.float().sum(), noisy_inputs.shape, noisy_inputs.dtype)

                    result_batch.trajectory_latents = noisy_inputs.permute(0, 1, 3, 2, 4, 5)
                    result_batch.trajectory_timesteps = torch.tensor([self.get_module("scheduler").timesteps[i] for i in [0, 12, 24, 36, -1]])
                    result_batch.latents = latents.permute(0, 2, 1, 3, 4)
                    result_batch = self.decoding_stage(result_batch,
                                                       fastvideo_args)
                    trajectory_latents.append(
                        result_batch.trajectory_latents.cpu())
                    trajectory_timesteps.append(
                        result_batch.trajectory_timesteps.cpu())
                    trajectory_decoded.append(result_batch.trajectory_decoded)

                trajectory_latents = torch.stack(trajectory_latents, dim=0).squeeze(0)
                # trajecotry_latents = trajectory_latents[:, [0, 12, 24, 36, -1]]

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
                batch_data: list[dict[str, Any]] = []

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

                    # Create record for Parquet dataset (text-only ODE schema)
                    record: dict[str, Any] = ode_text_only_record_creator(
                        video_name=video_name,
                        text_embedding=text_embedding,
                        caption=valid_data["text"][idx],
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
