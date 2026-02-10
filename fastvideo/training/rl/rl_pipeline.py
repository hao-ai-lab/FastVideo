# SPDX-License-Identifier: Apache-2.0
"""
RL training pipeline for FastVideo.

This module implements GRPO (Group Relative Policy Optimization) training for video generation models.
It extends the base TrainingPipeline with RL-specific functionality including trajectory collection,
reward computation, advantage estimation, and GRPO loss computation.

Reference:
    Flow-GRPO: https://github.com/yifan123/flow_grpo
"""

import json
import math
import os
from types import SimpleNamespace
from typing import Any
from copy import deepcopy

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader

from fastvideo.dataset.validation_dataset import ValidationDataset
import torch.distributed as dist
import torch.nn as nn

from fastvideo.pipelines.basic.wan.wan_pipeline import WanPipeline
from fastvideo.configs.sample import SamplingParam
from fastvideo.distributed import get_world_group
from fastvideo.fastvideo_args import TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch, TrainingBatch
from fastvideo.pipelines.stages.denoising import sde_step_with_logprob
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.rl.rewards import (create_reward_models,
                                           MultiRewardAggregator, ValueModel)
from .rl_utils import (
    compute_reward_statistics, )
from fastvideo.training.rl.stat_tracking import PerPromptStatTracker
from fastvideo.training.training_utils import (get_scheduler)
from fastvideo.utils import get_compute_dtype, shallow_asdict
from fastvideo.dataset.rl_prompt_dataset import build_rl_prompt_dataloader

from fastvideo.forward_context import set_forward_context

logger = init_logger(__name__)


def _to_device_dtype(
    d: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Move all tensors (and list-of-tensors) in d to device and dtype."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, dtype=dtype)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [x.to(device, dtype=dtype) for x in v]
        else:
            out[k] = v
    return out


class RLPipeline(TrainingPipeline):
    """
    RL training pipeline for flow matching models.

    This pipeline implements GRPO (Group Relative Policy Optimization) for video generation models.
    It handles:
    - Trajectory collection with log probability computation
    - Reward computation using reward models
    - Advantage computation with per-prompt stat tracking
    - GRPO policy loss with clipping and KL regularization
    """

    def __init__(self,
                 model_path: str,
                 fastvideo_args: TrainingArgs,
                 required_config_modules: list[str] | None = None,
                 loaded_modules: dict[str, nn.Module] | None = None) -> None:
        """Initialize RL pipeline."""
        if not fastvideo_args.rl_args.rl_mode:
            logger.warning(
                "rl_mode is False, but RLPipeline is being initialized. "
                "Setting rl_mode=True.")
            fastvideo_args.rl_args.rl_mode = True

        super().__init__(model_path, fastvideo_args, required_config_modules,
                         loaded_modules)

        # RL-specific components (will be initialized in initialize_training_pipeline)
        self.reward_models: MultiRewardAggregator | None = None
        self.value_model: ValueModel | None = None
        self.value_optimizer: torch.optim.Optimizer | None = None
        self.value_scheduler: Any | None = None
        self.sampling_pipeline = None

        # Set CFG guidace scale
        self.guidance_scale = fastvideo_args.rl_args.guidance_scale

        # Per-prompt stat tracker for advantage normalization
        # Will be initialized in initialize_training_pipeline
        self.stat_tracker: PerPromptStatTracker | None = None

        logger.info("Initialized RLPipeline with algorithm: %s",
                    fastvideo_args.rl_args.rl_algorithm)

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the RL training pipeline with algorithm, reward and value models."""
        logger.info("==== RL pipeline: initialize_training_pipeline START ====")
        # Call parent initialization for basic setup (optimizer, scheduler, etc.)
        # But we'll override the dataloader initialization
        super().initialize_training_pipeline(training_args)

        # Override dataloader with RL prompt dataloader
        # Get RL dataset configuration from training_args
        rl_dataset_path = training_args.rl_dataset_path if training_args.rl_dataset_path else training_args.data_path
        rl_dataset_type = training_args.rl_dataset_type  # "text" or "geneval"
        rl_num_image_per_prompt = training_args.rl_num_image_per_prompt  # k parameter
        num_replicas = self.world_size  # number of ranks
        rank = self.global_rank  # current rank

        # Build RL prompt dataloader
        train_dataloader, test_dataloader, train_dataset, test_dataset, train_sampler = build_rl_prompt_dataloader(
            dataset_path=rl_dataset_path,
            dataset_type=rl_dataset_type,
            split='train',
            train_batch_size=training_args.train_batch_size,
            test_batch_size=4,  # Hardcoded for now
            k=rl_num_image_per_prompt,
            seed=training_args.seed if training_args.seed is not None else 42,
            train_num_workers=training_args.dataloader_num_workers,
            test_num_workers=0,
            num_replicas=num_replicas,
            rank=rank)

        self.train_dataloader = train_dataloader
        self.train_dataset = train_dataset
        self.test_dataloader = test_dataloader
        self.test_dataset = test_dataset

        self.train_sampler = train_sampler
        self.train_loader_iter = iter(self.train_dataloader)
        self.current_step = 0
        self.train_sampler.set_step(self.current_step)

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) /
            training_args.gradient_accumulation_steps * training_args.sp_size /
            training_args.train_sp_batch_size)
        self.num_train_epochs = math.ceil(training_args.max_train_steps /
                                          self.num_update_steps_per_epoch)

        # Initialize reward models (GRPO always uses reward models)
        self.reward_models = create_reward_models(
            reward_models=training_args.rl_args.reward_models,
            device=str(self.device))

        # # define self.transformer_dtype
        # transformer = self.get_module("transformer")
        # if hasattr(transformer, 'module'):
        #     self.transformer_dtype = next(transformer.module.parameters()).dtype
        # else:
        #     self.transformer_dtype = next(transformer.parameters()).dtype
        # logger.info("Transformer dtype: %s", self.transformer_dtype)

        # Initialize per-prompt stat tracker for advantage normalization
        global_std = getattr(training_args.rl_args, 'rl_global_std', False)
        self.stat_tracker = PerPromptStatTracker(global_std=global_std)
        logger.info("Initialized PerPromptStatTracker with global_std=%s",
                    global_std)

        logger.info("RL pipeline initialization complete")
        self.sampling_pipeline = self._build_sampling_pipeline(training_args)

    def _build_sampling_pipeline(self, training_args: TrainingArgs):
        return self._create_inference_pipeline(training_args,
                                               dit_cpu_offload=False)

    def _create_inference_pipeline(self, training_args: TrainingArgs,
                                   dit_cpu_offload: bool):
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True
        loaded_modules = {
            "transformer": self.get_module("transformer"),
        }
        transformer_2 = self.get_module("transformer_2", None)
        if transformer_2 is not None:
            loaded_modules["transformer_2"] = transformer_2
        text_encoder = self.get_module("text_encoder", None)
        if text_encoder is not None:
            loaded_modules["text_encoder"] = text_encoder
        tokenizer = self.get_module("tokenizer", None)
        if tokenizer is not None:
            loaded_modules["tokenizer"] = tokenizer
        vae = self.get_module("vae", None)
        if vae is not None:
            loaded_modules["vae"] = vae
        # Use UniPCMultistepScheduler for RL sampling to match flow_grpo
        scheduler = self.get_module("scheduler", None)
        if scheduler is not None:
            loaded_modules["scheduler"] = scheduler

        pipeline = WanPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules=loaded_modules,
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=dit_cpu_offload)
        # # Override scheduler to use UniPCMultistepScheduler
        # if scheduler is not None:
        #     pipeline.modules["scheduler"] = scheduler
        return pipeline

    def _initialize_value_model(self, training_args: TrainingArgs) -> None:
        """Initialize the value model and its optimizer."""
        if training_args.rl_args.value_model_share_backbone:
            # Share transformer backbone with policy
            self.value_model = ValueModel(self.transformer, share_backbone=True)
        else:
            # Separate value model (clone transformer architecture)
            # TODO: Implement separate value model initialization
            # For now, use shared backbone
            self.value_model = ValueModel(self.transformer, share_backbone=True)

        # Create optimizer and scheduler for value model
        if not training_args.rl_args.value_model_share_backbone:
            value_params = list(self.value_model.parameters())
            self.value_optimizer = torch.optim.AdamW(
                value_params,
                lr=training_args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=training_args.weight_decay,
                eps=1e-8,
            )

            self.value_scheduler = get_scheduler(
                training_args.lr_scheduler,
                optimizer=self.value_optimizer,
                num_warmup_steps=training_args.lr_warmup_steps,
                num_training_steps=training_args.max_train_steps,
                num_cycles=training_args.lr_num_cycles,
                power=training_args.lr_power,
                min_lr_ratio=training_args.min_lr_ratio,
                last_epoch=self.init_steps - 1,
            )

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Get next batch of prompts from RL prompt dataloader.
        
        The RL prompt dataloader returns (prompts, metadatas) tuples from the collate function.
        This method extracts prompts and stores them in training_batch for use in collect_trajectories.
        
        Args:
            training_batch: Current training batch
        
        Returns:
            Updated training_batch with prompts in input_kwargs
        """
        with self.tracker.timed("timing/get_next_batch"):
            try:
                batch = next(self.train_loader_iter)
                self.current_step += 1
                self.train_sampler.set_step(self.current_step)
            except StopIteration:
                # Reset iterator for next epoch
                self.train_sampler.set_step(0)
                self.train_loader_iter = iter(self.train_dataloader)
                batch = next(self.train_loader_iter)

                batch = next(self.train_loader_iter)
                self.current_step += 1
                self.train_sampler.set_step(self.current_step)

            # RL prompt dataloader returns (prompts, metadatas) tuple
            prompts, metadatas = batch

            # Store prompts and metadatas in training_batch for use in collect_trajectories
            if training_batch.input_kwargs is None:
                training_batch.input_kwargs = {}
            training_batch.input_kwargs["prompts"] = prompts
            training_batch.input_kwargs["metadata"] = metadatas

            # Also store in infos for compatibility (convert metadatas to info_list format)
            if metadatas:
                training_batch.infos = [{
                    "prompt": prompt,
                    "metadata": metadata
                } for prompt, metadata in zip(prompts, metadatas, strict=False)]
            else:
                training_batch.infos = [{
                    "prompt": prompt,
                    "caption": prompt
                } for prompt in prompts]

        return training_batch

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict[str, Any],
                                  num_inference_steps: int) -> ForwardBatch:
        sampling_param.prompt = validation_batch['prompt']
        sampling_param.height = 480 #training_args.num_height
        sampling_param.width = 832 #training_args.num_width
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        if training_args.validation_guidance_scale:
            sampling_param.guidance_scale = float(
                training_args.validation_guidance_scale)
        assert self.seed is not None
        sampling_param.seed = self.seed

        # temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        # num_frames = (training_args.num_latent_t - 1) * temporal_compression_factor + 1
        sampling_param.num_frames = training_args.num_frames

        # Calculate n_tokens AFTER updating num_frames (aligns with sampling pipeline)
        # latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
        #                 sampling_param.height // 8, sampling_param.width // 8]
        # n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=self.validation_random_generator,
            # n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )

        return batch

    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        """
        Generate validation videos, log them to the tracker, and compute mean
        reward on validation videos (logged as validation_reward_mean).
        """
        training_args.inference_mode = True
        training_args.dit_cpu_offload = False
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation", local_main_process_only=False)

        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

        logger.info(
            "rank: %s: fastvideo_args.validation_dataset_file: %s",
            self.global_rank,
            training_args.validation_dataset_file,
            local_main_process_only=False,
        )
        validation_dataset = ValidationDataset(
            training_args.validation_dataset_file)
        validation_dataloader = DataLoader(validation_dataset,
                                           batch_size=None,
                                           num_workers=0)

        self.transformer.eval()
        if getattr(self, "transformer_2", None) is not None:
            self.transformer_2.eval()

        validation_steps = training_args.validation_sampling_steps.split(",")
        validation_steps = [int(step) for step in validation_steps]
        validation_steps = [step for step in validation_steps if step > 0]
        world_group = get_world_group()
        num_sp_groups = world_group.world_size // self.sp_group.world_size

        for num_inference_steps in validation_steps:
            logger.info(
                "rank: %s: num_inference_steps: %s",
                self.global_rank,
                num_inference_steps,
                local_main_process_only=False,
            )
            step_videos: list[list[np.ndarray]] = []
            step_captions: list[str] = []

            for validation_batch in validation_dataloader:
                batch = self._prepare_validation_batch(
                    sampling_param,
                    training_args,
                    validation_batch,
                    num_inference_steps,
                )
                logger.info(
                    "rank: %s: rank_in_sp_group: %s, batch.prompt: %s",
                    self.global_rank,
                    self.rank_in_sp_group,
                    batch.prompt,
                    local_main_process_only=False,
                )

                assert batch.prompt is not None and isinstance(
                    batch.prompt, str)
                step_captions.append(batch.prompt)

                output_batch = self.validation_pipeline.forward(
                    batch, training_args)
                samples = output_batch.output

                if self.rank_in_sp_group != 0:
                    continue

                video = rearrange(samples, "b c t h w -> t b c h w")
                frames = []
                for x in video:
                    x = torchvision.utils.make_grid(x, nrow=6)
                    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    frames.append((x * 255).numpy().astype(np.uint8))
                step_videos.append(frames)

            if self.rank_in_sp_group == 0:
                if self.global_rank == 0:
                    all_videos = step_videos
                    all_captions = step_captions

                    for sp_group_idx in range(1, num_sp_groups):
                        src_rank = sp_group_idx * self.sp_world_size
                        recv_videos = world_group.recv_object(src=src_rank)
                        recv_captions = world_group.recv_object(src=src_rank)
                        all_videos.extend(recv_videos)
                        all_captions.extend(recv_captions)

                    video_filenames = []
                    for i, (video_frames, caption) in enumerate(
                            zip(all_videos, all_captions, strict=True)):
                        os.makedirs(training_args.output_dir, exist_ok=True)
                        filename = os.path.join(
                            training_args.output_dir,
                            f"validation_step_{global_step}_inference_steps_{num_inference_steps}_video_{i}.mp4",
                        )
                        imageio.mimsave(filename,
                                        video_frames,
                                        fps=sampling_param.fps)
                        video_filenames.append(filename)

                    artifacts = []
                    video_logs: dict[str, Any] = {}
                    for i, (filename, caption) in enumerate(
                            zip(video_filenames, all_captions, strict=True)):
                        video_artifact = self.tracker.video(filename,
                                                            caption=caption)
                        if video_artifact is not None:
                            artifacts.append(video_artifact)
                            video_logs[
                                f"validation_video_{num_inference_steps}_steps_{i}"] = video_artifact
                    if artifacts:
                        logs = {
                            f"validation_videos_{num_inference_steps}_steps":
                            artifacts
                        }
                        self.tracker.log_artifacts(logs, global_step)
                        if video_logs:
                            self.tracker.log(video_logs, global_step)

                    # Compute mean reward on validation videos and log to tracker
                    if (self.reward_models is not None and all_videos
                            and all_captions):
                        # Convert all_videos (list of list of [H,W,C] frames) to [B, C, T, H, W]
                        video_tensors = []
                        for frames_list in all_videos:
                            # frames_list: list of (H, W, C) uint8
                            arr = np.stack(frames_list, axis=0)
                            t = torch.from_numpy(arr).float() / 255.0
                            t = t.permute(3, 0, 1, 2)
                            video_tensors.append(t)
                        videos_batch = torch.stack(video_tensors)
                        reward_scores = self.reward_models.compute_reward(
                            videos_batch, all_captions)
                        validation_reward_mean = reward_scores.mean().item()
                        self.tracker.log(
                            {"validation_reward_mean": validation_reward_mean},
                            global_step,
                        )
                else:
                    world_group.send_object(step_videos, dst=0)
                    world_group.send_object(step_captions, dst=0)

        training_args.inference_mode = False
        self.transformer.train()
        if getattr(self, "transformer_2", None) is not None:
            self.transformer_2.train()

    def collect_trajectories(self,
                             training_batch: TrainingBatch) -> TrainingBatch:
        """
        Collect on-policy trajectories by generating videos with log probabilities.
        
        This method implements the GRPO sampling phase:
        1. Gets prompts from the training batch
        2. Uses the inference pipeline denoising stage to generate trajectories
           with log probabilities
        3. Stores latents, log_probs, timesteps, and KL divergences in TrainingBatch
        
        Ported from FlowGRPO's sampling loop to work with FastVideo's TrainingBatch structure.
        
        Args:
            training_batch: Current training batch (should contain prompts in input_kwargs or infos)
        
        Returns:
            Updated training_batch with sampling results:
            - latents: [B, num_steps+1, C, T, H, W] - latents at each denoising step
            - log_probs: [B, num_steps] - log probabilities at each step
            - timesteps: [B, num_steps] - timesteps used
            - old_log_probs: [B, num_steps] - copy of log_probs for importance ratio
            - kl: [B, num_steps] - KL divergences (if kl_reward > 0)
            - prompt_embeds/negative_prompt_embeds: recomputed later as needed
        """

        logger.info("==== RL pipeline: collect_trajectories START ====")

        # Get prompts from batch
        # Prompts can be in input_kwargs["prompts"] or in infos
        prompts = None
        if training_batch.input_kwargs is not None and "prompts" in training_batch.input_kwargs:
            prompts = training_batch.input_kwargs["prompts"]
        elif training_batch.infos is not None:
            # Extract prompts from infos (each info dict should have 'prompt' or 'caption')
            prompts = []
            for info in training_batch.infos:
                prompt = info.get("prompt") or info.get("caption", "")
                prompts.append(prompt)
        else:
            raise ValueError(
                "Cannot find prompts in training_batch. "
                "Prompts should be in input_kwargs['prompts'] or infos[]['prompt'/'caption']"
            )

        # Normalize to list
        if isinstance(prompts, str):
            prompts = [prompts]
        # Get sampling configuration - align with validation pipeline
        # Use validation_sampling_steps if available, otherwise fall back to num_latent_t
        # if hasattr(self.training_args, 'validation_sampling_steps') and self.training_args.validation_sampling_steps:
        #     validation_steps = self.training_args.validation_sampling_steps.split(",")
        #     validation_steps = [int(step) for step in validation_steps if step.strip()]
        #     num_inference_steps = validation_steps[0] if validation_steps else self.training_args.num_latent_t
        # else:
        #     num_inference_steps = self.training_args.num_latent_t

        # # Use validation_guidance_scale if available (aligns with validation pipeline)
        # if hasattr(self.training_args, 'validation_guidance_scale') and self.training_args.validation_guidance_scale:
        #     guidance_scale = float(self.training_args.validation_guidance_scale)
        # else:
        #     # Fall back to flow_grpo default for training
        #     guidance_scale = 4.5

        # Create SamplingParam like validation pipeline does
        # This ensures all fields from SamplingParam are included in ForwardBatch
        sampling_param = SamplingParam.from_pretrained(
            self.training_args.model_path)

        height = self.training_args.num_height
        width = self.training_args.num_width
        num_videos_per_prompt = 1  # Each prompt in batch generates one video (batch already has repeated prompts if needed)
        sample_time_per_prompt = 1  # config.sample.sample_time_per_prompt - hardcoded
        kl_reward = getattr(self.training_args.rl_args, 'kl_reward', 0.0)
        collect_kl = kl_reward > 0

        num_inference_steps = self.training_args.num_latent_t
        # Use validation_guidance_scale if available (aligns with validation pipeline)
        guidance_scale = float(self.training_args.validation_guidance_scale)

        # latents_size = [(num_frames - 1) // 4 + 1,
        #                 height // 8, width // 8]
        # n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        # temporal_compression_factor = self.training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        # num_frames = (num_inference_steps - 1) * temporal_compression_factor + 1
        num_frames = self.training_args.num_frames

        # Set sampling_param fields to match validation pipeline pattern
        sampling_param.prompt = prompts  # Will be set per-batch in loop
        sampling_param.height = height
        sampling_param.width = width
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        sampling_param.guidance_scale = guidance_scale
        sampling_param.num_frames = num_frames
        sampling_param.num_videos_per_prompt = num_videos_per_prompt

        if self.sampling_pipeline is None:
            raise RuntimeError("Sampling pipeline is not initialized")

        # Collect samples (multiple samples per prompt if sample_time_per_prompt > 1)
        all_latents_list = []
        all_log_probs_list = []
        all_kl_list = []
        all_timesteps_list = []
        all_decoded_videos_list = [
        ]  # Store decoded videos from pipeline output (like validation)
        # Placeholder for compatibility with older trajectory collection logic.
        # Kept to avoid NameError if referenced; currently prompt_ids are stored as None.
        all_prompt_ids_list = []

        # Set transformer to eval mode for sampling
        self.transformer.eval()

        with torch.no_grad():
            # Sample multiple times per prompt if needed
            for _ in range(sample_time_per_prompt):

                # Set seed in sampling_param - vary per batch so rollouts differ across batches (align with flow_grpo advancing RNG)
                sampling_param.seed = self.seed + getattr(self, "current_step", 0)

                rl_data = ForwardBatch.RLData(
                    enabled=True,
                    collect_log_probs=True,
                    collect_kl=collect_kl,
                    kl_reward=kl_reward,
                    store_trajectory=True,
                    keep_trajectory_on_cpu=False,
                )

                # Create ForwardBatch using same pattern as validation: **shallow_asdict(sampling_param)
                # This ensures all fields from SamplingParam are included
                # Note: generator=None lets InputValidationStage create proper list of generators
                # (one per batch item) from the seed, ensuring each video gets unique randomness
                forward_batch = ForwardBatch(
                    **shallow_asdict(sampling_param),
                    latents=None,
                    generator=
                    None,  # Let InputValidationStage create generators from seed
                    # n_tokens=n_tokens,  # Add n_tokens like validation
                    eta=0.0,  # Add eta like validation
                    VSA_sparsity=self.training_args.
                    VSA_sparsity,  # Add VSA_sparsity like validation
                    rl_data=rl_data,  # RL-specific field (not in validation)
                )

                orig_output_type = getattr(self.training_args, "output_type",
                                           None)
                orig_inference_mode = getattr(self.training_args,
                                              "inference_mode", None)
                orig_dit_cpu_offload = getattr(self.training_args,
                                               "dit_cpu_offload", None)
                # Use full pipeline with decoding (like validation) instead of "latent"
                # This ensures we use the same decoding path as validation
                if orig_output_type == "latent":
                    # Set to "pt" to enable decoding (same as validation pipeline)
                    self.training_args.output_type = "pt"
                elif orig_output_type is None:
                    # If not set, explicitly set to "pt" to ensure decoding
                    self.training_args.output_type = "pt"
                # If orig_output_type is already "pt" or something else, keep it
                if orig_inference_mode is not None:
                    self.training_args.inference_mode = True
                if orig_dit_cpu_offload is not None:
                    # Mirror validation: we run sampling fully on GPU.
                    self.training_args.dit_cpu_offload = False

                # Run sampling pipeline with full decoding
                try:
                    output_batch = self.sampling_pipeline.forward(
                        forward_batch, self.training_args)
                finally:
                    if orig_output_type is not None:
                        self.training_args.output_type = orig_output_type
                    if orig_inference_mode is not None:
                        self.training_args.inference_mode = orig_inference_mode
                    if orig_dit_cpu_offload is not None:
                        self.training_args.dit_cpu_offload = orig_dit_cpu_offload
                if output_batch.rl_data.trajectory_latents is None:
                    raise RuntimeError(
                        "RL trajectory latents were not collected")

                # Copy transformer forward context from rollout so GRPO loss uses same forward pass
                if output_batch.rl_data.transformer_forward_contexts is not None and output_batch.rl_data.transformer_forward_kwargs is not None:
                    training_batch.rl_transformer_forward_contexts = output_batch.rl_data.transformer_forward_contexts
                    training_batch.rl_transformer_forward_kwargs = output_batch.rl_data.transformer_forward_kwargs
                # Copy prompt embeddings from pipeline output so we use same embeddings in GRPO loss
                if output_batch.prompt_embeds and len(
                        output_batch.prompt_embeds) > 0:
                    training_batch.prompt_embeds = output_batch.prompt_embeds[0]
                if output_batch.negative_prompt_embeds and len(
                        output_batch.negative_prompt_embeds) > 0:
                    training_batch.negative_prompt_embeds = output_batch.negative_prompt_embeds[
                        0]

                latents = output_batch.rl_data.trajectory_latents
                log_probs = output_batch.rl_data.log_probs
                if log_probs is None:
                    raise RuntimeError(
                        "RL log probabilities were not collected")
                kl = output_batch.rl_data.kl
                timesteps = output_batch.rl_data.trajectory_timesteps
                if timesteps is None:
                    raise RuntimeError("RL timesteps were not collected")
                timesteps = timesteps.repeat(latents.shape[0], 1)

                # Extract decoded videos from pipeline output (same as validation pipeline)
                # output_batch.output contains decoded videos [B, C, T, H, W] if output_type != "latent"
                decoded_videos = output_batch.output
                if decoded_videos is None:
                    raise RuntimeError(
                        "Decoded videos not found in pipeline output. "
                        "Make sure output_type is not set to 'latent'.")

                # Set raw_latent_shape for metrics (used by training_pipeline.py)
                training_batch.raw_latent_shape = latents.shape

                all_latents_list.append(latents)
                if log_probs is not None:
                    all_log_probs_list.append(log_probs)
                if kl is not None:
                    all_kl_list.append(kl)
                all_timesteps_list.append(timesteps)
                all_decoded_videos_list.append(decoded_videos)
                all_prompt_ids_list.append(None)

        # Concatenate across sample_time_per_prompt dimension (if sample_time_per_prompt > 1)
        if sample_time_per_prompt > 1:
            # Shape: [B * sample_time_per_prompt, num_steps+1, C, T, H, W]
            training_batch.latents = torch.cat(all_latents_list, dim=0)
            # Shape: [B * sample_time_per_prompt, num_steps]
            training_batch.log_probs = torch.cat(all_log_probs_list, dim=0)
            # Shape: [B * sample_time_per_prompt, num_steps]
            training_batch.timesteps = torch.cat(all_timesteps_list, dim=0)
            # Store KL if computed
            training_batch.kl = torch.cat(all_kl_list,
                                          dim=0) if all_kl_list else None
            # Store decoded videos [B * sample_time_per_prompt, C, T, H, W]
            decoded_videos = torch.cat(all_decoded_videos_list, dim=0)
            # Store prompt_ids (repeat for each sample)
            training_batch.prompt_ids = None
        else:
            # Single sample per prompt
            training_batch.latents = all_latents_list[
                0]  # [B, num_steps+1, C, T, H, W]
            training_batch.log_probs = all_log_probs_list[0]  # [B, num_steps]
            training_batch.timesteps = all_timesteps_list[0]  # [B, num_steps]
            training_batch.kl = all_kl_list[0] if len(
                all_kl_list) > 0 and all_kl_list[0] is not None else None
            decoded_videos = all_decoded_videos_list[0]  # [B, C, T, H, W]
            training_batch.prompt_ids = None

        # Store old log probs for importance ratio computation
        training_batch.old_log_probs = training_batch.log_probs.clone()

        # When sample_time_per_prompt > 1, batch size expands so recompute embeddings in GRPO; else keep copied from pipeline
        if sample_time_per_prompt > 1:
            training_batch.prompt_embeds = None
            training_batch.negative_prompt_embeds = None

        # Store prompts and decoded videos in input_kwargs (like validation pipeline pattern)
        if training_batch.input_kwargs is None:
            training_batch.input_kwargs = {}
        # Repeat prompts for each sample if sample_time_per_prompt > 1
        if sample_time_per_prompt > 1:
            repeated_prompts = []
            for prompt in prompts:
                for _ in range(sample_time_per_prompt):
                    repeated_prompts.append(prompt)
            training_batch.input_kwargs["prompts"] = repeated_prompts
        else:
            training_batch.input_kwargs["prompts"] = prompts
        # Store decoded videos in input_kwargs (same pattern as validation)
        training_batch.input_kwargs["decoded_videos"] = decoded_videos

        # myregion: Debug: Save decoded video for visual verification (rank 0 only).
        # decoded_videos is already gathered along temporal dim above when sp_world_size > 1.
        # if self.global_rank == 0:
        #     from contextlib import nullcontext
        #     import numpy as np
        #     import imageio
        #     controller = getattr(self, "profiler_controller", None)
        #     region_cm = (controller.region("my region") if controller is not None
        #                 and getattr(controller, "has_profiler", False) else
        #                 nullcontext())
        #     with region_cm:
        #         out_dir = "/mnt/fast-disks/hao_lab/shijie/mylogs"
        #         os.makedirs(out_dir, exist_ok=True)
        #         batch_size = decoded_videos.shape[0]
        #         logger.info(f"Debug region: Saving {batch_size} videos from batch")
        #         fps = 24
        #         videos_np_list = []
        #         for batch_idx in range(batch_size):
        #             vid = decoded_videos[batch_idx].detach().to(torch.float32).cpu()
        #             vid = vid.permute(1, 2, 3, 0).contiguous()
        #             vid_np = vid.numpy()
        #             if vid_np.min() < 0.0:
        #                 vid_np = (vid_np + 1.0) / 2.0
        #             vid_np = np.clip(vid_np, 0.0, 1.0)
        #             vid_np = (vid_np * 255.0).round().astype(np.uint8)
        #             vid_fp64 = vid.detach().to(torch.float64).cpu().numpy()
        #             if vid_fp64.min() < 0.0:
        #                 vid_fp64 = (vid_fp64 + 1.0) / 2.0
        #             vid_fp64 = np.clip(vid_fp64, 0.0, 1.0)
        #             videos_np_list.append(vid_fp64)
        #             out_path = os.path.join(out_dir, f"debug_step0_batch_{batch_idx}.mp4")
        #             frames = [vid_np[t] for t in range(vid_np.shape[0])]
        #             imageio.mimsave(out_path, frames, fps=fps)
        #             logger.info(f"Saved debug video batch_{batch_idx} with {vid_np.shape[0]} frames at {fps} fps (duration: {vid_np.shape[0]/fps:.2f}s) to {out_path}")
        #         if batch_size >= 2:
        #             logger.info("=" * 80)
        #             logger.info("Video Difference Statistics (calculated in float64 for precision):")
        #             logger.info("=" * 80)
        #             for i in range(batch_size - 1):
        #                 vid_i, vid_j = videos_np_list[i], videos_np_list[i + 1]
        #                 diff = np.abs(vid_i.astype(np.float64) - vid_j.astype(np.float64))
        #                 avg_diff, max_diff = np.mean(diff), np.max(diff)
        #                 min_diff, sum_diff = np.min(diff), np.sum(diff)
        #                 total_elements = diff.size
        #                 logger.info(f"Difference between video {i} and {i+1}:")
        #                 logger.info(f"  Total elements: {total_elements:,}, Average: {avg_diff:.10f}, Max: {max_diff:.10f}, Min: {min_diff:.10f}, Sum: {sum_diff:.10f}")
        #                 logger.info("-" * 80)
        #             logger.info("=" * 80)
        #     raise KeyboardInterrupt("Debug stop after saving decoded video (my region).")
        # endregion

        logger.info("==== RL pipeline: collect_trajectories FINISH ====")
        return training_batch

    def compute_rewards(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Compute rewards using decoded videos from collect_trajectories and calling reward models.
        
        This method implements Step 5 of GRPO training:
        1. Uses decoded videos from collect_trajectories (decoded using full pipeline, same as validation)
        2. Calls reward models with decoded videos
        3. Applies KL reward penalty if configured
        4. Stores reward scores and statistics in TrainingBatch
        
        Ported from FlowGRPO's reward computation to work with FastVideo's TrainingBatch.
        
        Args:
            training_batch: Training batch with decoded_videos in input_kwargs from collect_trajectories
        
        Returns:
            Updated training_batch with:
            - reward_scores: Computed rewards [B]
            - reward_mean: Mean reward
            - reward_std: Std of rewards
        """
        if self.reward_models is None:
            raise RuntimeError(
                "Reward models not initialized. Call initialize_training_pipeline first."
            )

        # Get decoded videos from input_kwargs (same pattern as validation pipeline)
        if training_batch.input_kwargs is None or "decoded_videos" not in training_batch.input_kwargs:
            raise RuntimeError(
                "Decoded videos not found in training_batch.input_kwargs. "
                "Make sure collect_trajectories stores decoded videos in input_kwargs."
            )

        videos = training_batch.input_kwargs[
            "decoded_videos"]  # [B, C, T, H, W]
        logger.info(
            f"Using decoded videos from collect_trajectories: shape={videos.shape}"
        )

        # Get prompts for reward computation
        prompts = training_batch.input_kwargs.get(
            "prompts") if training_batch.input_kwargs else None
        if prompts is None:
            raise ValueError(
                "Prompts not found in training_batch.input_kwargs. Required for reward computation."
            )

        # Compute rewards using reward models
        # Note: reward_models.compute_reward expects videos [B, C, T, H, W] and prompts [B]
        reward_scores = self.reward_models.compute_reward(videos, prompts)

        # Apply KL reward penalty if configured
        # In FlowGRPO: rewards["avg"] = rewards["avg"] - kl_reward * kl
        kl_reward = getattr(self.training_args.rl_args, 'kl_reward', 0.0)
        if kl_reward > 0 and training_batch.kl is not None:
            # training_batch.kl is [B, num_steps], we need to aggregate across timesteps
            # FlowGRPO uses the mean KL across timesteps
            kl_penalty = training_batch.kl.mean(dim=1)  # [B]
            reward_scores = reward_scores - kl_reward * kl_penalty

        # Store reward scores
        training_batch.reward_scores = reward_scores.float()

        # Compute reward statistics (local)
        reward_stats = compute_reward_statistics(training_batch.reward_scores)
        training_batch.reward_mean = reward_stats["reward_mean"]
        training_batch.reward_std = reward_stats["reward_std"]

        # Multi-GPU: allreduce reward sum and count so rank 0 logs mean across all ranks
        if reward_scores.numel() > 0:
            world_size = getattr(self, "world_size", 1)
            if world_size > 1:
                wg = get_world_group()
                local_sum = reward_scores.sum().to(self.device)
                local_count = torch.tensor(reward_scores.numel(),
                                           device=self.device,
                                           dtype=local_sum.dtype)
                wg.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                wg.all_reduce(local_count, op=dist.ReduceOp.SUM)
                global_mean = (local_sum / local_count).item()
                global_count = int(local_count.item())
                training_batch.reward_mean = global_mean
            else:
                global_mean = training_batch.reward_mean
                global_count = reward_scores.numel()
            # reward_mean (global_mean) is logged via self.tracker.log in train_one_step

        return training_batch

    def compute_values(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Compute value predictions using value model.

        Args:
            training_batch: Training batch

        Returns:
            Updated training_batch with value predictions
        """
        # GRPO doesn't use value models, so this method is a no-op
        # Keeping for API compatibility but returning early
        return training_batch

    def compute_advantages(self,
                           training_batch: TrainingBatch) -> TrainingBatch:
        """
        Compute advantages using per-prompt stat tracking and normalization.
        
        This method implements Step 6 of GRPO training:
        1. Uses PerPromptStatTracker for per-prompt normalization (if enabled)
        2. Supports global std option
        3. Computes normalized advantages from rewards
        4. Computes returns (advantages + values if value model is used)
        5. Stores advantages, returns, and statistics in TrainingBatch
        
        Ported from FlowGRPO's advantage computation to work with FastVideo's TrainingBatch.
        
        Args:
            training_batch: Training batch with reward_scores [B] and prompt_ids [B, seq_len]
        
        Returns:
            Updated training_batch with:
            - advantages: Computed advantages [B]
            - returns: TD returns [B] (advantages if no value model)
            - advantage_mean: Mean advantage
            - advantage_std: Std of advantages
        """
        logger.info("==== RL pipeline: compute_advantages START ====")
        assert training_batch.reward_scores is not None, "Rewards must be computed before advantages"

        assert self.stat_tracker is not None, "Stat tracker not initialized. Call initialize_training_pipeline first."

        # Get prompts from prompt_ids (decode token IDs to strings)
        if training_batch.prompt_ids is not None:
            tokenizer = self.get_module("tokenizer")
            prompt_ids_np = training_batch.prompt_ids.cpu().numpy()
            prompts = tokenizer.batch_decode(prompt_ids_np,
                                             skip_special_tokens=True)
        elif training_batch.input_kwargs is not None and "prompts" in training_batch.input_kwargs:
            prompts = training_batch.input_kwargs["prompts"]
        else:
            raise ValueError(
                "Cannot find prompts for stat tracking. "
                "Need either prompt_ids or input_kwargs['prompts']")

        # Get rewards
        rewards = training_batch.reward_scores
        world_size = getattr(self, "world_size", 1)
        global_rank = getattr(self, "global_rank", 0)

        # Check if per-prompt stat tracking is enabled
        per_prompt_stat_tracking = getattr(
            self.training_args.rl_args,
            'rl_per_prompt_stat_tracking',
            True  # Default to True for GRPO
        )

        if per_prompt_stat_tracking:
            # Multi-GPU: all-gather rewards and prompts so per-prompt mean/std
            # are computed across all ranks (same as flow_grpo).
            if world_size > 1:
                wg = get_world_group()
                # All-gather rewards: each rank has [B], result is [B * world_size]
                rewards_1d = rewards.view(-1) if rewards.dim() > 1 else rewards
                rewards_1d = rewards_1d.contiguous().to(self.device)
                gathered_rewards = wg.all_gather(rewards_1d, dim=0)
                # Gather prompts: broadcast each rank's list in turn
                gathered_prompts_lists = []
                for src in range(world_size):
                    if global_rank == src:
                        obj_list = [prompts]
                    else:
                        obj_list = [None]
                    wg.broadcast_object_list(obj_list, src=src)
                    gathered_prompts_lists.append(obj_list[0])
                prompts_global = [
                    p for plist in gathered_prompts_lists for p in plist
                ]
                rewards_global = gathered_rewards
                # Run stat tracker on global data
                advantages_np = self.stat_tracker.update(prompts=prompts_global,
                                                         rewards=rewards_global,
                                                         type='grpo')

                # myregion debug
                logger.info(f"gathered_rewards: {gathered_rewards}")
                logger.info(f"advantages_np: {advantages_np}")
                logger.info(f"prompts_global: {prompts_global}")
                logger.info(f"prompts: {prompts}")
                # endregion

                # Slice back to this rank's indices (same as flow_grpo reshape + [rank])
                local_batch_size = len(prompts)
                advantages_flat = np.asarray(advantages_np).ravel()
                advantages_np_local = advantages_flat[global_rank *
                                                      local_batch_size:
                                                      (global_rank + 1) *
                                                      local_batch_size]
                advantages = torch.as_tensor(
                    advantages_np_local,
                    device=rewards.device,
                    dtype=rewards.dtype,
                )
            else:
                advantages_np = self.stat_tracker.update(prompts=prompts,
                                                         rewards=rewards,
                                                         type='grpo')
                advantages = torch.as_tensor(advantages_np,
                                             device=rewards.device,
                                             dtype=rewards.dtype)
        else:
            # Global normalization: (reward - global_mean) / global_std
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        # Get values (use zeros if algorithm doesn't use value model)
        if training_batch.values is not None:
            values = training_batch.values
        else:
            values = torch.zeros_like(rewards)

        # Compute returns: returns = advantages + values
        # For GRPO, we typically don't use value models, so returns = advantages
        returns = advantages + values

        # Store in training batch
        training_batch.advantages = advantages
        training_batch.returns = returns
        training_batch.advantage_mean = advantages.mean().item()
        training_batch.advantage_std = advantages.std().item()

        # Reset stats so next step uses current-batch-only mean/std (same as flow_grpo per epoch)
        self.stat_tracker.clear()

        logger.info("==== RL pipeline: compute_advantages FINISH ====")
        return training_batch

    def _compute_log_prob_for_timestep(
        self,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        timesteps: torch.Tensor,
        current_timestep: int,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None = None,
        guidance_scale: float = 4.5,
        step_context: dict[str, Any] | None = None,
        transformer_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ...]:
        """
        Compute log probability for a given timestep using current transformer.
        
        When step_context and transformer_kwargs are provided (from DenoisingStage
        via training_batch), uses the same set_forward_context and transformer call
        as trajectory collection so log_prob matches.
        
        Args:
            latents: Current latents [B, C, T, H, W]
            next_latents: Next latents (target) [B, C, T, H, W]
            timesteps: Timesteps [B]
            current_timestep: Step index (0..num_steps-1)
            prompt_embeds: Prompt embeddings [B, seq_len, hidden_dim]
            negative_prompt_embeds: Negative prompt embeddings for CFG [B, seq_len, hidden_dim]
            guidance_scale: Classifier-free guidance scale
            step_context: Optional per-step context from DenoisingStage (current_timestep, attn_metadata)
            transformer_kwargs: Optional batch-level kwargs from DenoisingStage (image_kwargs, pos_cond_kwargs, neg_cond_kwargs, action_kwargs, guidance_expand)
        
        Returns:
            (prev_sample, log_prob, prev_sample_mean, std_dev_t, dt)
        """
        scheduler = self.get_module("scheduler")
        transformer = self.get_module("transformer")

        self.transformer.train()
        use_saved_context = (step_context is not None
                             and transformer_kwargs is not None)
        # When replaying: match DenoisingStage (eval + bfloat16 + autocast) so ratio ~1.0
        if use_saved_context:
            target_dtype = torch.bfloat16
            autocast_enabled = (
                target_dtype != torch.float32
                and not getattr(self.training_args, "disable_autocast", False))
        else:
            target_dtype = get_compute_dtype()
            autocast_enabled = False

        num_inference_steps = self.training_args.num_latent_t
        if not hasattr(scheduler,
                       'timesteps') or scheduler.timesteps is None or len(
                           scheduler.timesteps) != num_inference_steps + 1:
            scheduler.set_timesteps(num_inference_steps, device=self.device)

        dev = latents.device
        latent_model_input = latents.to(target_dtype)
        t_expand = timesteps.to(dev)
        prompt_embeds = prompt_embeds.to(target_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(target_dtype)

        # Match DenoisingStage: scale latent input by scheduler
        t_scalar = timesteps[0] if timesteps.dim() > 0 else timesteps
        if isinstance(t_scalar, torch.Tensor):
            t_scalar = t_scalar.item() if t_scalar.numel() == 1 else t_scalar
        latent_model_input = scheduler.scale_model_input(
            latent_model_input, t_scalar)

        if use_saved_context:
            ctx_timestep = step_context.get("current_timestep",
                                            current_timestep)
            attn_metadata = step_context.get("attn_metadata")
            # Transformer may read enable_teacache, num_inference_steps, teacache_params from get_forward_context().forward_batch
            forward_batch_ref = SimpleNamespace(
                is_cfg_negative=False,
                enable_teacache=False,
                num_inference_steps=num_inference_steps,
                teacache_params=None,
            )
            image_kwargs = _to_device_dtype(
                transformer_kwargs.get("image_kwargs") or {}, dev, target_dtype)
            pos_cond_kwargs = _to_device_dtype(
                transformer_kwargs.get("pos_cond_kwargs") or {}, dev,
                target_dtype)
            neg_cond_kwargs = _to_device_dtype(
                transformer_kwargs.get("neg_cond_kwargs") or {}, dev,
                target_dtype)
            action_kwargs = _to_device_dtype(
                transformer_kwargs.get("action_kwargs") or {}, dev,
                target_dtype)
            guidance_expand = transformer_kwargs.get("guidance_expand")
            if guidance_expand is not None:
                guidance_expand = guidance_expand.to(dev, dtype=target_dtype)
        else:
            ctx_timestep = current_timestep
            attn_metadata = None
            forward_batch_ref = None
            image_kwargs = {}
            pos_cond_kwargs = {}
            neg_cond_kwargs = {}
            action_kwargs = {}
            guidance_expand = None

        def run_transformer_with_context(encoder_hidden_states, cond_kwargs,
                                         is_cfg_negative: bool):
            if forward_batch_ref is not None:
                forward_batch_ref.is_cfg_negative = is_cfg_negative
            with set_forward_context(
                    current_timestep=ctx_timestep,
                    attn_metadata=attn_metadata,
                    forward_batch=forward_batch_ref,
            ):
                with torch.autocast(
                        device_type="cuda",
                        dtype=target_dtype,
                        enabled=autocast_enabled,
                ):
                    if use_saved_context:
                        out = transformer(
                            latent_model_input,
                            encoder_hidden_states,
                            t_expand,
                            guidance=guidance_expand,
                            **image_kwargs,
                            **cond_kwargs,
                            **action_kwargs,
                        )
                    else:
                        out = transformer(
                            hidden_states=latent_model_input,
                            timestep=t_expand,
                            encoder_hidden_states=encoder_hidden_states,
                            return_dict=False,
                        )
                return out[0] if isinstance(out, tuple) else out

        if guidance_scale > 1.0:
            noise_pred_text = run_transformer_with_context(
                prompt_embeds, pos_cond_kwargs, False)
            noise_pred_uncond = run_transformer_with_context(
                negative_prompt_embeds, neg_cond_kwargs, True)
            noise_pred = (noise_pred_uncond + guidance_scale *
                          (noise_pred_text - noise_pred_uncond))
        else:
            noise_pred = run_transformer_with_context(prompt_embeds,
                                                      pos_cond_kwargs, False)

        if use_saved_context:
            self.transformer.train()

        return sde_step_with_logprob(scheduler,
                                     noise_pred.float(),
                                     timesteps,
                                     latents.float(),
                                     prev_sample=next_latents.float(),
                                     return_dt_and_std_dev_t=True)

    def _compute_grpo_loss(
            self, training_batch: TrainingBatch
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute GRPO loss (policy loss + KL loss) with per-timestep backward to avoid OOM.
        
        This function implements the GRPO training objective:
        1. Recomputes log probabilities for current policy at each timestep
        2. Computes reference log probabilities with adapter disabled (if using LoRA)
        3. Computes policy loss with clipping
        4. Computes KL loss using reference model
        5. Does backward() after each timestep to free activations (aligned with flow_grpo)
        6. Returns accumulated metrics
        
        Ported from FlowGRPO's training loop to work with FastVideo's TrainingBatch.
        Key difference: Does backward() after each timestep to prevent OOM.
        
        Args:
            training_batch: Training batch with:
                - latents: [B, num_steps+1, C, T, H, W] - latents at each step
                - timesteps: [B, num_steps] - timesteps used
                - log_probs: [B, num_steps] - old log probs from sampling
                - advantages: [B, num_steps] or [B] - advantages
                - prompt_embeds: [B, seq_len, hidden_dim] - prompt embeddings
                - negative_prompt_embeds: [B, seq_len, hidden_dim] - negative embeddings (optional)
        
        Returns:
            total_loss: Average total loss (for logging/metrics only, backward already done)
            metrics: Dictionary with loss components and diagnostics
        """

        # Get configuration from RLArgs
        # Note: CLI arguments map to RLArgs fields:
        # --rl-policy-clip-range -> grpo_policy_clip_range (via dest)
        # --rl-kl-beta -> kl_beta (via dest)
        clip_range = self.training_args.rl_args.grpo_policy_clip_range
        kl_beta = self.training_args.rl_args.kl_beta
        guidance_scale = 4.5
        adv_clip_max = 5.0  # Aligned with flow_grpo config.train.adv_clip_max = 5

        # Get data from training batch
        latents = training_batch.latents  # [B, num_steps+1, C, T, H, W]
        timesteps = training_batch.timesteps  # [B, num_steps]
        old_log_probs = training_batch.old_log_probs  # [B, num_steps]
        advantages = training_batch.advantages  # [B, num_steps] or [B]

        # Get prompt embeddings
        # If not stored, recompute from prompts
        if training_batch.prompt_embeds is not None:
            prompt_embeds = training_batch.prompt_embeds
            # Also set encoder_hidden_states for metrics compatibility
            if training_batch.encoder_hidden_states is None:
                training_batch.encoder_hidden_states = prompt_embeds
        elif training_batch.encoder_hidden_states is not None:
            prompt_embeds = training_batch.encoder_hidden_states
        else:
            # Recompute prompt embeddings from prompts
            prompts = training_batch.input_kwargs.get(
                "prompts") if training_batch.input_kwargs else None
            if prompts is None:
                raise ValueError(
                    "Cannot find prompts or prompt embeddings in training_batch"
                )

            # Encode prompts
            text_encoder = self.get_module("text_encoder")
            tokenizer = self.get_module("tokenizer")
            # device = next(self.transformer.parameters()).device
            device = self.device

            # Normalize to list
            if isinstance(prompts, str):
                prompts = [prompts]

            # Tokenize and encode
            text_inputs = tokenizer(prompts,
                                    padding="max_length",
                                    max_length=512,
                                    truncation=True,
                                    return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = text_encoder(
                    text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    output_hidden_states=True,
                )
                # prompt_embeds = outputs.last_hidden_state.to(self.transformer.dtype)
                prompt_embeds = outputs.last_hidden_state
                # Set encoder_hidden_states for metrics compatibility
                training_batch.encoder_hidden_states = prompt_embeds

        # Get negative prompt embeddings
        negative_prompt_embeds = training_batch.negative_prompt_embeds
        if negative_prompt_embeds is None and guidance_scale > 1.0:
            # Generate negative prompt embeddings if needed
            text_encoder = self.get_module("text_encoder")
            tokenizer = self.get_module("tokenizer")
            # device = next(self.transformer.parameters()).device
            device = self.device

            batch_size = prompt_embeds.shape[0]
            negative_prompts = [""] * batch_size

            neg_text_inputs = tokenizer(negative_prompts,
                                        padding="max_length",
                                        max_length=512,
                                        truncation=True,
                                        return_tensors="pt").to(device)

            with torch.no_grad():
                neg_outputs = text_encoder(
                    neg_text_inputs["input_ids"],
                    attention_mask=neg_text_inputs["attention_mask"],
                    output_hidden_states=True,
                )
                # negative_prompt_embeds = neg_outputs.last_hidden_state.to(self.transformer.dtype)
                negative_prompt_embeds = neg_outputs.last_hidden_state

        # Handle advantages shape: if [B], expand to [B, num_steps]
        # In flow_grpo, advantages are [B, num_steps] or [B, 1] - they should match timesteps
        if advantages.dim() == 1:
            # Expand to [B, num_steps] to match timesteps shape
            advantages = advantages.unsqueeze(1).expand(-1, timesteps.shape[1])
        elif advantages.dim() == 2 and advantages.shape[1] == 1:
            # If [B, 1], expand to [B, num_steps]
            advantages = advantages.expand(-1, timesteps.shape[1])

        batch_size, num_steps = timesteps.shape

        # Accumulate metrics across timesteps (for logging only, losses are backpropped immediately)
        policy_losses = []
        kl_losses = []
        total_losses = []
        clip_fractions = []
        importance_ratios = []
        approx_kls = []

        # Get transformer for reference model computation
        transformer = self.get_module("transformer")

        # Scale factor for loss: average across timesteps, gradient accumulation, and ranks (multi-GPU)
        world_size = max(1, getattr(self, "world_size", 1))
        # Loop over timesteps - do backward() after each timestep to free activations
        # This matches flow_grpo's approach and prevents OOM from accumulating activations
        for j in range(int(num_steps*0.99)):
            # Get latents and next_latents for this timestep
            latents_j = latents[:, j]  # [B, C, T, H, W]
            next_latents_j = latents[:, j + 1]  # [B, C, T, H, W]
            timesteps_j = timesteps[:, j]  # [B]
            old_log_probs_j = old_log_probs[:, j]  # [B]
            advantages_j = advantages[:, j]  # [B]
            # Per-step context from trajectory collection (same forward pass as DenoisingStage)
            step_context = None
            transformer_kwargs = None
            if training_batch.rl_transformer_forward_contexts is not None and training_batch.rl_transformer_forward_kwargs is not None:
                if j < len(training_batch.rl_transformer_forward_contexts):
                    step_context = training_batch.rl_transformer_forward_contexts[
                        j]
                transformer_kwargs = training_batch.rl_transformer_forward_kwargs

            # Compute log probability with current policy
            prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = self._compute_log_prob_for_timestep(
                latents_j,
                next_latents_j,
                timesteps_j,
                j,
                prompt_embeds,
                negative_prompt_embeds,
                guidance_scale,
                step_context=step_context,
                transformer_kwargs=transformer_kwargs,
            )

            # Compute reference log probability with adapter disabled (if using LoRA)
            # Aligned with flow_grpo: use transformer.module.disable_adapter() if wrapped, or pipeline.disable_adapter()
            if kl_beta > 0:
                with torch.no_grad():
                    # Try to disable adapter through pipeline (FastVideo's LoRA implementation)
                    # or through transformer if it's a PeftModel (flow_grpo style)
                    disable_adapter_ctx = None
                    if hasattr(self, 'disable_adapter'):
                        # FastVideo LoRA pipeline has disable_adapter method
                        disable_adapter_ctx = self.disable_adapter()
                    elif hasattr(transformer, 'module') and hasattr(
                            transformer.module, 'disable_adapter'):
                        # Wrapped transformer with PeftModel (like in flow_grpo with Accelerate)
                        disable_adapter_ctx = transformer.module.disable_adapter(
                        )
                    elif hasattr(transformer, 'disable_adapter'):
                        # Direct PeftModel (not wrapped)
                        disable_adapter_ctx = transformer.disable_adapter()

                    # if disable_adapter_ctx is not None:
                    with disable_adapter_ctx:
                        _, _, prev_sample_mean_ref, _, dt_ref = self._compute_log_prob_for_timestep(
                            latents_j,
                            next_latents_j,
                            timesteps_j,
                            j,
                            prompt_embeds,
                            negative_prompt_embeds,
                            guidance_scale,
                            step_context=step_context,
                            transformer_kwargs=transformer_kwargs,
                        )

                # Compute KL loss: KL = (mean_diff)^2 / (2 * (std_dev_t * dt_ref)^2)
                # FlowGRPO uses: kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * (std_dev_t * dt_ref) ** 2)
                # For videos [B, C, T, H, W], we average over all spatial/channel/temporal dims except batch
                # Note: std_dev_t and dt_ref are already broadcast to [B, 1, 1, 1, 1]
                # Aligned with flow_grpo: no epsilon added to match exactly
                kl_loss_j = ((prev_sample_mean - prev_sample_mean_ref)**2).mean(
                    dim=(1, 2, 3, 4), keepdim=True) / (2 *
                                                       (std_dev_t * dt_ref)**2)
                kl_loss_j = kl_loss_j.mean()  # Average over batch dimension
            else:
                kl_loss_j = torch.tensor(0.0, device=log_prob.device)

            # GRPO policy loss computation
            # Clip advantages
            advantages_j_clipped = torch.clamp(advantages_j, -adv_clip_max,
                                               adv_clip_max)

            # Compute importance ratio
            ratio = torch.exp(log_prob - old_log_probs_j)

            # myregion debug
            
            if j == 0:
                logger.info(f"guidance_scale: {guidance_scale}")
            # if j == 0:
            logger.info(
                f"RL_METRIC: GRPO first timestep {j}: ratio mean=%.6f min=%.6f max=%.6f | "
                "log_prob mean=%.6f min=%.6f max=%.6f | "
                "old_log_probs_j mean=%.6f min=%.6f max=%.6f",
                ratio.mean().item(),
                ratio.min().item(),
                ratio.max().item(),
                log_prob.mean().item(),
                log_prob.min().item(),
                log_prob.max().item(),
                old_log_probs_j.mean().item(),
                old_log_probs_j.min().item(),
                old_log_probs_j.max().item(),
            )

            # endregion

            # Clipped surrogate objective
            unclipped_loss = -advantages_j_clipped * ratio
            clipped_loss = -advantages_j_clipped * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            policy_loss_j = torch.maximum(unclipped_loss, clipped_loss).mean()

            # Total loss for this timestep (scaled for averaging)
            total_loss_j = (policy_loss_j + kl_beta * kl_loss_j)

            # Backward pass after each timestep to free activations (aligned with flow_grpo)
            # This prevents OOM by not accumulating activations across all timesteps
            with self.tracker.timed(
                    "timing/forward_backward"), set_forward_context(
                        current_timestep=j,
                        attn_metadata=None,
                        forward_batch=None):
                total_loss_j.backward()

            # Store metrics for logging (detached to avoid keeping computation graph)
            with torch.no_grad():
                policy_losses.append(policy_loss_j.detach())
                kl_losses.append(kl_loss_j.detach())
                total_losses.append(total_loss_j.detach())

                # Clip fraction
                clip_fraction_j = ((ratio < 1.0 - clip_range) |
                                   (ratio > 1.0 + clip_range)).float().mean()
                clip_fractions.append(clip_fraction_j)

                # Importance ratio
                importance_ratios.append(ratio.mean())

                # Approximate KL (using log prob difference)
                approx_kl_j = 0.5 * torch.mean((log_prob - old_log_probs_j)**2)
                approx_kls.append(approx_kl_j)

            # Explicitly delete intermediate tensors to free memory (only removes local
            # variable names; list contents above are separate tensors and are unchanged)
            # del prev_sample, log_prob, prev_sample_mean, std_dev_t, dt
            # del advantages_j_clipped, ratio, unclipped_loss, clipped_loss, policy_loss_j, total_loss_j, kl_loss_j
            if kl_beta > 0:
                del prev_sample_mean_ref, dt_ref

        # Average metrics across timesteps (stack then mean; backward already done per timestep)
        policy_loss = torch.stack(policy_losses).mean()
        kl_loss = torch.stack(kl_losses).mean()
        if kl_beta == 0:
            kl_loss = torch.tensor(0.0, device=policy_loss.device)
        total_loss = torch.stack(total_losses).mean()

        # myregion debug
        logger.info(f"importance_ratios: {importance_ratios}")
        # endregion

        # Compute metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "clip_fraction": torch.stack(clip_fractions).mean().item(),
            "importance_ratio_mean":
            torch.stack(importance_ratios).mean().item(),
            "approx_kl": torch.stack(approx_kls).mean().item(),
        }

        return total_loss, metrics

    def _log_grpo_metrics(
        self,
        tb: TrainingBatch,
        total_loss: torch.Tensor,
        metrics: dict[str, Any],
        log_step: int,
    ) -> None:
        """Allreduce loss/metrics and log to tracker on rank 0."""
        total_loss_t = torch.tensor(total_loss.item(), device=self.device)
        policy_loss_t = torch.tensor(metrics.get("policy_loss", 0.0), device=self.device)
        kl_loss_t = torch.tensor(metrics.get("kl_loss", 0.0), device=self.device)
        if getattr(self, "world_size", 1) > 1:
            wg = get_world_group()
            wg.all_reduce(total_loss_t, op=dist.ReduceOp.AVG)
            wg.all_reduce(policy_loss_t, op=dist.ReduceOp.AVG)
            wg.all_reduce(kl_loss_t, op=dist.ReduceOp.AVG)
        if getattr(self, "global_rank", 0) == 0:
            self.tracker.log(
                {
                    "reward_mean": getattr(tb, "reward_mean", 0.0),
                    "reward_std": getattr(tb, "reward_std", 0.0),
                    "total_loss": total_loss_t.item(),
                    "policy_loss": policy_loss_t.item(),
                    "kl_loss": kl_loss_t.item(),
                    "importance_ratio": metrics.get("importance_ratio_mean", 1.0),
                    "clip_fraction": metrics.get("clip_fraction", 0.0),
                },
                log_step,
            )

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Train one step using GRPO: sample M batches (each with get_batch -> collect -> reward -> advantage),
        then backward and optimizer step(s). M = num_batches_per_step when > 1, else gradient_accumulation_steps.
        When num_batches_per_step > 1, one optimizer step and one log per batch; otherwise one step and one log after M batches.
        """
        training_batch = self._prepare_training(training_batch)
        num_batches_per_step = getattr(
            self.training_args.rl_args, "num_batches_per_step", 1
        )
        M = (
            num_batches_per_step
            if num_batches_per_step > 1
            else self.training_args.gradient_accumulation_steps
        )
        base_step = getattr(
            training_batch, "current_timestep", getattr(self, "current_trainstep", 0)
        )

        # Sampling: M batches, advantage computed per batch (per group)
        # Use a new TrainingBatch per iteration so each collected[i] holds distinct data (avoids identical reward/loss per step).
        collected: list[TrainingBatch] = []
        for _ in range(M):
            tb = TrainingBatch()
            tb.current_timestep = getattr(training_batch, "current_timestep", 0)
            tb.current_vsa_sparsity = getattr(training_batch, "current_vsa_sparsity", 0.0)
            tb = self._get_next_batch(tb)
            tb = self.collect_trajectories(tb)
            tb = self.compute_rewards(tb)
            tb = self.compute_advantages(tb)
            collected.append(tb)

        # Training: backward, then optimizer step and log per batch
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, tb in enumerate(collected):
            total_loss, metrics = self._compute_grpo_loss(tb)
            tb.policy_loss = metrics.get("policy_loss", 0.0)
            tb.kl_divergence = metrics.get("kl_loss", 0.0)
            tb.importance_ratio = metrics.get("importance_ratio_mean", 1.0)
            tb.clip_fraction = metrics.get("clip_fraction", 0.0)
            tb.value_loss = 0.0
            tb.entropy = 0.0
            tb.total_loss = total_loss.item()
            tb = self._clip_grad_norm(tb)

            with self.tracker.timed("timing/optimizer_step"):
                self.optimizer.step()
                self.lr_scheduler.step()
                if self.value_optimizer is not None:
                    self.value_optimizer.step()
                    self.value_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._log_grpo_metrics(
                tb, total_loss, metrics, base_step * M + batch_idx
            )

        training_batch = collected[-1]
        kl_threshold = 0.1
        if training_batch.kl_divergence > kl_threshold:
            logger.warning(
                "High KL divergence at step %d: %.4f > %.4f",
                getattr(training_batch, "current_timestep", 0),
                training_batch.kl_divergence,
                kl_threshold,
            )
        return training_batch

    def set_trainable(self) -> None:
        """Set which parameters should be trainable."""
        # Policy (transformer) is trainable
        super().set_trainable()

        # GRPO doesn't use value models, so no value model training needed

        # Freeze reward models (they should not be trained)
        if self.reward_models is not None:
            for param in self.reward_models.parameters():
                param.requires_grad = False


def create_rl_pipeline(model_path: str,
                       training_args: TrainingArgs) -> RLPipeline:
    """
    Factory function to create RL pipeline.

    Args:
        model_path: Path to pretrained model
        training_args: Training arguments with RL configuration

    Returns:
        Initialized RLPipeline
    """
    pipeline = RLPipeline(model_path, training_args)
    pipeline.initialize_training_pipeline(training_args)
    return pipeline
