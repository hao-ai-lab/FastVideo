# SPDX-License-Identifier: Apache-2.0
import copy
import gc
import os
import time
from abc import abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import Any

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from torchmetrics.image.lpip import (
    LearnedPerceptualImagePatchSimilarity as LPIPSimilarity)
from tqdm.auto import tqdm

import fastvideo.envs as envs
from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.validation_dataset import ValidationDataset
from fastvideo.distributed import (cleanup_dist_env_and_memory,
                                   get_local_torch_device, get_sp_group,
                                   get_world_group)
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines import (ComposedPipelineBase, ForwardBatch,
                                 TrainingBatch)
from fastvideo.training.activation_checkpoint import (
    apply_activation_checkpointing)
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import (
    EMA_FSDP, clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler, load_checkpoint, pred_noise_to_pred_video, save_checkpoint,
    shift_timestep)
from fastvideo.utils import is_vsa_available, set_random_seed

import wandb  # isort: skip

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class CMDistillationPipeline(TrainingPipeline):
    """
    A distillation pipeline for training a 3 step model.
    Inherits from TrainingPipeline to reuse training infrastructure.
    """
    _required_config_modules = [
        "scheduler", "transformer", "vae"
    ]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[dict[str, Any]]
    current_epoch: int = 0
    init_steps: int
    current_trainstep: int
    num_generator_updates: int = 0
    video_latent_shape: tuple[int, ...]
    video_latent_shape_sp: tuple[int, ...]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the distillation training pipeline with multiple models."""
        logger.info("Initializing distillation pipeline...")

        super().initialize_training_pipeline(training_args)

        self.noise_scheduler = self.get_module("scheduler")
        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)

        self.timestep_shift = self.training_args.pipeline_config.flow_shift
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=self.timestep_shift)

        if self.training_args.warp_denoising_step:
            # timesteps = self.noise_scheduler.timesteps.cpu()
            timesteps = torch.cat((self.noise_scheduler.timesteps.cpu(),
                                   torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = torch.tensor(
                self.training_args.pipeline_config.dmd_denoising_steps,
                dtype=torch.long,
                device=torch.device("cpu"))
            self.denoising_step_list = timesteps[1000 -
                                                 self.denoising_step_list].to(
                                                     get_local_torch_device())
            logger.info(
                "Warp denoising step is enabled, using %s denoising steps",
                self.denoising_step_list)
        else:
            self.denoising_step_list = torch.tensor(
                self.training_args.pipeline_config.dmd_denoising_steps,
                dtype=torch.long,
                device=get_local_torch_device())
            logger.info(
                "Warp denoising step is disabled, using %s denoising steps",
                self.denoising_step_list)
        logger.info("Distillation generator model to %s denoising steps",
                    len(self.denoising_step_list))
        self.num_train_timestep = self.noise_scheduler.num_train_timesteps

        self.min_timestep = int(self.training_args.min_timestep_ratio *
                                self.num_train_timestep)
        self.max_timestep = int(self.training_args.max_timestep_ratio *
                                self.num_train_timestep)

        # self.real_score_guidance_scale = self.training_args.real_score_guidance_scale

        # Initialize EMA teacher for CM if enabled
        self.use_cm_with_ema = getattr(self.training_args, "cm_use_ema_teacher",
                                       False)
        if self.use_cm_with_ema:
            # Use local_shard mode for teacher forward compatibility with FSDP2
            self.ema_teacher = EMA_FSDP(self.transformer,
                                        decay=self.training_args.ema_decay,
                                        mode="local_shard")
        else:
            self.ema_teacher = None
        self.lpips = LPIPSimilarity().to(self.device)

    def _compute_lpips_loss(self, pred_video: torch.Tensor,
                            latents: torch.Tensor) -> torch.Tensor:
        """
        Helper method to compute LPIPS loss between predicted video and ground truth latents.
        This method handles VAE scaling, shifting, decoding, and frame processing consistently.
        """
        print("Computing LPIPS loss...")
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            # Apply VAE scaling factor and shift factor before decoding (same as visualize_intermediate_latents)
            pred_video = pred_video.permute(0, 2, 1, 3, 4)
            latents = latents.permute(0, 2, 1, 3, 4)
            if isinstance(self.vae.scaling_factor, torch.Tensor):
                pred_video_scaled = pred_video / self.vae.scaling_factor.to(
                    pred_video.device, pred_video.dtype)
                latents_scaled = latents / self.vae.scaling_factor.to(
                    latents.device, latents.dtype)
            else:
                pred_video_scaled = pred_video / self.vae.scaling_factor
                latents_scaled = latents / self.vae.scaling_factor

            # Apply shifting if needed
            if (hasattr(self.vae, "shift_factor")
                    and self.vae.shift_factor is not None):
                if isinstance(self.vae.shift_factor, torch.Tensor):
                    pred_video_scaled += self.vae.shift_factor.to(
                        pred_video.device, pred_video.dtype)
                    latents_scaled += self.vae.shift_factor.to(
                        latents.device, latents.dtype)
                else:
                    pred_video_scaled += self.vae.shift_factor
                    latents_scaled += self.vae.shift_factor

            # Permute from [batch, channels, frames, height, width] to [batch, frames, channels, height, width] for VAE decode
            # pred_video_scaled = pred_video_scaled.permute(0, 2, 1, 3, 4)
            # latents_scaled = latents_scaled.permute(0, 2, 1, 3, 4)

            # print(f"pred_video_scaled shape after permute: {pred_video_scaled.shape}")
            # print(f"latents_scaled shape after permute: {latents_scaled.shape}")
            pred_video_frames = self.vae.decode(pred_video_scaled)
            latents_frames = self.vae.decode(latents_scaled)

        # VAE output is already in [-1, 1] range, keep it for LPIPS
        # Just permute back to [B, T, C, H, W] format for frame processing
        pred_video_frames = pred_video_frames.permute(0, 2, 1, 3, 4)
        latents_frames = latents_frames.permute(0, 2, 1, 3, 4)

        pred_video_frames = rearrange(pred_video_frames,
                                      "b n c h w -> (b n) c h w")
        latents_frames = rearrange(latents_frames, "b n c h w -> (b n) c h w")

        return self.lpips(pred_video_frames, latents_frames)

    @abstractmethod
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Initialize validation pipeline - must be implemented by subclasses."""
        raise NotImplementedError(
            "Distillation pipelines must implement this method")

    def _prepare_distillation(self,
                              training_batch: TrainingBatch) -> TrainingBatch:
        """Prepare training environment for distillation."""
        self.transformer.requires_grad_(True)
        self.transformer.train()

        return training_batch

    def _build_distill_input_kwargs(
            self, noise_input: torch.Tensor, timestep: torch.Tensor,
            text_dict: dict[str, torch.Tensor] | None,
            training_batch: TrainingBatch) -> TrainingBatch:
        if text_dict is None:
            raise ValueError(
                "text_dict cannot be None for distillation pipeline")

        training_batch.input_kwargs = {
            "hidden_states": noise_input.permute(0, 2, 1, 3, 4),
            "encoder_hidden_states": text_dict["encoder_hidden_states"],
            "encoder_attention_mask": text_dict["encoder_attention_mask"],
            "timestep": timestep,
            "return_dict": False,
        }

        return training_batch

    def _generator_forward(self, training_batch: TrainingBatch) -> torch.Tensor:
        """
        Forward pass through student transformer for a single randomly sampled
        denoising step; returns predicted clean video latents and records
        auxiliary info/losses on training_batch.
        """
        latents = training_batch.latents
        dtype = latents.dtype
        index = torch.randint(0,
                              len(self.denoising_step_list), [1],
                              device=self.device,
                              dtype=torch.long)
        timestep = self.denoising_step_list[index]
        training_batch.dmd_latent_vis_dict["generator_timestep"] = timestep

        noise = torch.randn(self.video_latent_shape, device=self.device, dtype=dtype)
        if self.sp_world_size > 1:
            noise = rearrange(noise, "b (n t) c h w -> b n t c h w", n=self.sp_world_size).contiguous()
            noise = noise[:, self.rank_in_sp_group, :, :, :, :]

        noisy_latent = self.noise_scheduler.add_noise(latents.flatten(0, 1),
                                                      noise.flatten(0, 1),
                                                      timestep).unflatten(
                                                          0, (1, latents.shape[1]))

        training_batch = self._build_distill_input_kwargs(noisy_latent, timestep,
                                                          training_batch.conditional_dict,
                                                          training_batch)
        pred_noise = self.transformer(**training_batch.input_kwargs).permute(0, 2, 1, 3, 4)
        pred_video = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_latent.flatten(0, 1),
            timestep=timestep,
            scheduler=self.noise_scheduler,
        ).unflatten(0, pred_noise.shape[:2])

        # Compute regression loss (LPIPS) for optional auxiliary loss/logging
        # regression_loss = self._compute_lpips_loss(pred_video, latents)
        # training_batch.regression_loss = regression_loss
        training_batch.dmd_latent_vis_dict.update({
            "generator_pred_video": pred_video.detach().clone(),
        })
        return pred_video

    def _calculate_consistency_loss(
            self, pred_video: torch.Tensor,
            training_batch: TrainingBatch) -> torch.Tensor:
        """
        Consistency loss (CM-like): two variants
          - Without EMA: teacher = model at higher noise with stop-grad
          - With EMA: teacher = EMA(model) at lower noise or higher noise depending on schedule (use higher noise as teacher)
        """
        # Need at least two steps to form a pair
        if len(self.denoising_step_list) < 2:
            return torch.tensor(0.0, device=self.device, dtype=pred_video.dtype)

        # Choose a random index k in [1, N-1] so that (k-1, k) is valid
        idx_high = torch.randint(1,
                                 len(self.denoising_step_list), [1],
                                 device=self.device).item()
        t_high = self.denoising_step_list[idx_high]
        t_low = self.denoising_step_list[idx_high - 1]
        t_low = t_low * torch.ones(1, device=self.device, dtype=torch.long)
        t_high = t_high * torch.ones(1, device=self.device, dtype=torch.long)
        logger.info("t_high: %s, t_low: %s", t_high, t_low)

        # Use student output as base; construct noisy sample at higher timestep
        base_student = pred_video
        base_noise = torch.randn(self.video_latent_shape, device=self.device, dtype=base_student.dtype)
        if self.sp_world_size > 1:
            base_noise = rearrange(base_noise, "b (n t) c h w -> b n t c h w", n=self.sp_world_size).contiguous()
            base_noise = base_noise[:, self.rank_in_sp_group, :, :, :, :]

        noisy_high = self.noise_scheduler.add_noise(base_student.flatten(0, 1),
                                                    base_noise.flatten(0, 1),
                                                    t_high).unflatten(0, (1, base_student.shape[1]))

        # Compute ODE-adjacent lower-t sample via a single Euler step using teacher output at t_high
        # Temporarily switch to eval and no-grad for teacher forward(s)
        was_training = self.transformer.training
        self.transformer.eval()
        with torch.no_grad():
            tb_high_teacher = self._build_distill_input_kwargs(noisy_high, t_high,
                                                               training_batch.conditional_dict, training_batch)
            pred_flow_high_teacher = self.transformer(**tb_high_teacher.input_kwargs).permute(0, 2, 1, 3, 4)

            # Prepare flat tensors for scheduler stepping
            sample_flat = noisy_high.flatten(0, 1).to(dtype=torch.float32)
            flow_flat = pred_flow_high_teacher.flatten(0, 1).to(dtype=torch.float32)

            # Reset step index so stepping starts at the provided timestep
            prev_step_index = getattr(self.noise_scheduler, "_step_index", None)
            self.noise_scheduler._step_index = None
            stepped = self.noise_scheduler.step(model_output=flow_flat,
                                                timestep=t_high,
                                                sample=sample_flat,
                                                return_dict=True)
            # Restore step index state
            self.noise_scheduler._step_index = prev_step_index

            noisy_low_ode = stepped.prev_sample.unflatten(0, (1, noisy_high.shape[1])).to(noisy_high.dtype)

            # Teacher target at lower t (clean latent)
            tb_low_teacher = self._build_distill_input_kwargs(noisy_low_ode, t_low,
                                                              training_batch.conditional_dict, training_batch)
            pred_noise_low_teacher = self.transformer(**tb_low_teacher.input_kwargs).permute(0, 2, 1, 3, 4)
            y_low_teacher = pred_noise_to_pred_video(
                pred_noise=pred_noise_low_teacher.flatten(0, 1),
                noise_input_latent=noisy_low_ode.flatten(0, 1),
                timestep=t_low,
                scheduler=self.noise_scheduler,
            ).unflatten(0, pred_noise_low_teacher.shape[:2])
        if was_training:
            self.transformer.train()

        # Student prediction at lower t on the ODE-stepped sample
        tb_low_student = self._build_distill_input_kwargs(noisy_low_ode, t_low,
                                                          training_batch.conditional_dict, training_batch)
        pred_noise_low_student = self.transformer(**tb_low_student.input_kwargs).permute(0, 2, 1, 3, 4)
        y_low_student = pred_noise_to_pred_video(
            pred_noise=pred_noise_low_student.flatten(0, 1),
            noise_input_latent=noisy_low_ode.flatten(0, 1),
            timestep=t_low,
            scheduler=self.noise_scheduler,
        ).unflatten(0, pred_noise_low_student.shape[:2])

        cm_loss_weight = self.training_args.cm_loss_weight
        weighing_fn = getattr(self.training_args, "cm_weighing_function", "constant")

        if weighing_fn == "constant":
            cm_weighing_function = lambda x: cm_loss_weight
        elif weighing_fn == "sigma_sqrt":
            cm_weighing_function = lambda x: cm_loss_weight * (x**0.5)
        else:
            raise ValueError(f"Invalid cm_weighing_function: {weighing_fn}")

        # Match student at lower t to teacher at lower t (adjacent PF ODE point)
        cm_loss = F.mse_loss(y_low_student, y_low_teacher.detach())
        cm_loss = cm_loss * cm_weighing_function(t_low)

        # Optionally record for logging
        training_batch.dmd_latent_vis_dict.update({
            "cm_timestep_high": t_high,
            "cm_timestep_low": t_low,
        })

        return cm_loss

    def _clip_model_grad_norm_(self, training_batch: TrainingBatch,
                               transformer) -> TrainingBatch:

        max_grad_norm = self.training_args.max_grad_norm

        if max_grad_norm is not None:
            model_parts = [transformer]
            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                max_grad_norm,
                foreach=None,
            )
            assert grad_norm is not float('nan') or grad_norm is not float(
                'inf')
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        else:
            grad_norm = 0.0
        training_batch.grad_norm = grad_norm
        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        super()._prepare_dit_inputs(training_batch)
        conditional_dict = {
            "encoder_hidden_states": training_batch.encoder_hidden_states,
            "encoder_attention_mask": training_batch.encoder_attention_mask,
        }
        unconditional_dict = {
            "encoder_hidden_states": self.negative_prompt_embeds,
            "encoder_attention_mask": self.negative_prompt_attention_mask,
        }

        training_batch.dmd_latent_vis_dict = {}
        training_batch.fake_score_latent_vis_dict = {}

        training_batch.conditional_dict = conditional_dict
        training_batch.unconditional_dict = unconditional_dict
        training_batch.raw_latent_shape = training_batch.latents.shape
        training_batch.latents = training_batch.latents.permute(0, 2, 1, 3, 4)
        self.video_latent_shape = training_batch.latents.shape

        if self.sp_world_size > 1:
            training_batch.latents = rearrange(
                training_batch.latents,
                "b (n t) c h w -> b n t c h w",
                n=self.sp_world_size).contiguous()
            training_batch.latents = training_batch.latents[:, self.
                                                            rank_in_sp_group, :, :, :, :]

        self.video_latent_shape_sp = training_batch.latents.shape

        return training_batch

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        gradient_accumulation_steps = getattr(self.training_args, 'gradient_accumulation_steps', 1)
        batches: list[TrainingBatch] = []
        # Collect N batches for gradient accumulation
        for _ in range(gradient_accumulation_steps):
            batch = self._prepare_distillation(training_batch)
            batch = self._get_next_batch(batch)
            batch = self._normalize_dit_input(batch)
            batch = self._prepare_dit_inputs(batch)
            batch = self._build_attention_metadata(batch)
            batch.attn_metadata_vsa = copy.deepcopy(batch.attn_metadata)
            if batch.attn_metadata is not None:
                batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore
            batches.append(batch)

        self.optimizer.zero_grad()
        total_loss_value = 0.0
        dmd_latent_vis_dict: dict[str, torch.Tensor] = {}
        batch_gen = None

        for batch in batches:
            batch_gen = copy.deepcopy(batch)
            # Forward student once to obtain a clean prediction to anchor CM pair
            with set_forward_context(current_timestep=batch_gen.timesteps,
                                     attn_metadata=batch_gen.attn_metadata_vsa):
                generator_pred_video = self._generator_forward(batch_gen)

            # Consistency loss
            with set_forward_context(current_timestep=batch_gen.timesteps,
                                     attn_metadata=batch_gen.attn_metadata):
                cm_loss = self._calculate_consistency_loss(
                    pred_video=generator_pred_video, training_batch=batch_gen)

                # Optional auxiliary regression loss
                if getattr(self.training_args, "use_regression_loss", False):
                    cm_loss = cm_loss + batch_gen.regression_loss * self.training_args.regression_loss_weight

            with set_forward_context(current_timestep=batch_gen.timesteps,
                                     attn_metadata=batch_gen.attn_metadata_vsa):
                (cm_loss / gradient_accumulation_steps).backward()
            total_loss_value += cm_loss.detach().item()
            dmd_latent_vis_dict.update(batch_gen.dmd_latent_vis_dict)

        # Clip and step
        assert batch_gen is not None
        self._clip_model_grad_norm_(batch_gen, self.transformer)
        self.optimizer.step()
        self.lr_scheduler.step()

        # Update EMA teacher after parameter update
        if self.use_cm_with_ema:
            assert self.ema_teacher is not None
            self.ema_teacher.update(self.transformer)

        self.optimizer.zero_grad(set_to_none=True)
        avg_loss = torch.tensor(total_loss_value / max(1, gradient_accumulation_steps), device=self.device)
        world_group = get_world_group()
        world_group.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)

        training_batch.total_loss = avg_loss.item()
        training_batch.dmd_latent_vis_dict = dmd_latent_vis_dict
        training_batch.grad_norm = getattr(training_batch, "grad_norm", 0.0)
        return training_batch

    def _resume_from_checkpoint(self) -> None:
        """Resume training from checkpoint for the generator model only."""

        logger.info("Loading checkpoint from %s",
                    self.training_args.resume_from_checkpoint)

        resumed_step = load_checkpoint(self.transformer, self.global_rank,
                                       self.training_args.resume_from_checkpoint,
                                       self.optimizer, self.train_dataloader,
                                       self.lr_scheduler, self.noise_random_generator)

        if resumed_step > 0:
            self.init_steps = resumed_step
            logger.info("Successfully resumed from step %s", resumed_step)
        else:
            logger.warning("Failed to load checkpoint, starting from step 0")
            self.init_steps = -1

    def _log_training_info(self) -> None:
        """Log distillation-specific training information."""
        # First call parent class method to get basic training info
        super()._log_training_info()

        # Then add distillation-specific information
        logger.info("Distillation-specific settings:")
        assert isinstance(self.training_args, TrainingArgs)
        logger.info("  Max gradient norm: %s", self.training_args.max_grad_norm)


    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        training_args.inference_mode = True
        training_args.dit_cpu_offload = True
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation")

        # Create sampling parameters if not provided
        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

        # Set deterministic seed for validation

        logger.info("Using validation seed: %s", self.seed)

        # Prepare validation prompts
        logger.info('rank: %s: fastvideo_args.validation_dataset_file: %s',
                    self.global_rank,
                    training_args.validation_dataset_file,
                    local_main_process_only=False)
        validation_dataset = ValidationDataset(
            training_args.validation_dataset_file)
        validation_dataloader = DataLoader(validation_dataset,
                                           batch_size=None,
                                           num_workers=0)

        transformer.eval()

        validation_steps = training_args.validation_sampling_steps.split(",")
        validation_steps = [int(step) for step in validation_steps]
        validation_steps = [step for step in validation_steps if step > 0]
        # Log validation results for this step
        world_group = get_world_group()
        num_sp_groups = world_group.world_size // self.sp_group.world_size
        # Process each validation prompt for each validation step
        for num_inference_steps in validation_steps:
            logger.info("rank: %s: num_inference_steps: %s",
                        self.global_rank,
                        num_inference_steps,
                        local_main_process_only=False)
            step_videos: list[np.ndarray] = []
            step_captions: list[str] = []

            for validation_batch in validation_dataloader:
                batch = self._prepare_validation_batch(sampling_param,
                                                       training_args,
                                                       validation_batch,
                                                       num_inference_steps)

                negative_prompt = batch.negative_prompt
                batch_negative = ForwardBatch(
                    data_type="video",
                    prompt=negative_prompt,
                    prompt_embeds=[],
                    prompt_attention_mask=[],
                )
                result_batch = self.validation_pipeline.prompt_encoding_stage(  # type: ignore
                    batch_negative, training_args)
                self.negative_prompt_embeds, self.negative_prompt_attention_mask = result_batch.prompt_embeds[
                    0], result_batch.prompt_attention_mask[0]

                logger.info("rank: %s: rank_in_sp_group: %s, batch.prompt: %s",
                            self.global_rank,
                            self.rank_in_sp_group,
                            batch.prompt,
                            local_main_process_only=False)

                assert batch.prompt is not None and isinstance(
                    batch.prompt, str)
                step_captions.append(batch.prompt)

                # Run validation inference
                with torch.no_grad():
                    output_batch = self.validation_pipeline.forward(
                        batch, training_args)
                samples = output_batch.output
                if self.rank_in_sp_group != 0:
                    continue

                # Process outputs
                video = rearrange(samples, "b c t h w -> t b c h w")
                frames = []
                for x in video:
                    x = torchvision.utils.make_grid(x, nrow=6)
                    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    frames.append((x * 255).numpy().astype(np.uint8))
                step_videos.append(frames)

            # Log validation results for this step
            world_group = get_world_group()
            num_sp_groups = world_group.world_size // self.sp_group.world_size

            # Only sp_group leaders (rank_in_sp_group == 0) need to send their
            # results to global rank 0
            if self.rank_in_sp_group == 0:
                if self.global_rank == 0:
                    # Global rank 0 collects results from all sp_group leaders
                    all_videos = step_videos  # Start with own results
                    all_captions = step_captions

                    # Receive from other sp_group leaders
                    for sp_group_idx in range(1, num_sp_groups):
                        src_rank = sp_group_idx * self.sp_world_size  # Global rank of other sp_group leaders
                        recv_videos = world_group.recv_object(src=src_rank)
                        recv_captions = world_group.recv_object(src=src_rank)
                        all_videos.extend(recv_videos)
                        all_captions.extend(recv_captions)

                    video_filenames = []
                    for i, (video, caption) in enumerate(
                            zip(all_videos, all_captions, strict=True)):
                        os.makedirs(training_args.output_dir, exist_ok=True)
                        filename = os.path.join(
                            training_args.output_dir,
                            f"validation_step_{global_step}_inference_steps_{num_inference_steps}_video_{i}.mp4"
                        )
                        imageio.mimsave(filename, video, fps=sampling_param.fps)
                        video_filenames.append(filename)

                    logs = {
                        f"validation_videos_{num_inference_steps}_steps": [
                            wandb.Video(filename, caption=caption)
                            for filename, caption in zip(
                                video_filenames, all_captions, strict=True)
                        ]
                    }
                    wandb.log(logs, step=global_step)
                else:
                    # Other sp_group leaders send their results to global rank 0
                    world_group.send_object(step_videos, dst=0)
                    world_group.send_object(step_captions, dst=0)

        # Re-enable gradients for training
        transformer.train()
        gc.collect()

    def visualize_intermediate_latents(self, training_batch: TrainingBatch,
                                       training_args: TrainingArgs, step: int):
        """Add visualization data to wandb logging and save frames to disk."""
        wandb_loss_dict = {}
        dmd_latents_vis_dict = training_batch.dmd_latent_vis_dict

        # Only log generator output for CM pipeline
        if 'generator_pred_video' in dmd_latents_vis_dict:
            latents = dmd_latents_vis_dict['generator_pred_video']
            latents = latents.permute(0, 2, 1, 3, 4)

            if isinstance(self.vae.scaling_factor, torch.Tensor):
                latents = latents / self.vae.scaling_factor.to(
                    latents.device, latents.dtype)
            else:
                latents = latents / self.vae.scaling_factor

            # Apply shifting if needed
            if (hasattr(self.vae, "shift_factor")
                    and self.vae.shift_factor is not None):
                if isinstance(self.vae.shift_factor, torch.Tensor):
                    latents += self.vae.shift_factor.to(latents.device,
                                                        latents.dtype)
                else:
                    latents += self.vae.shift_factor
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video = self.vae.decode(latents)
            video = (video / 2 + 0.5).clamp(0, 1)
            video = video.cpu().float()
            video = video.permute(0, 2, 1, 3, 4)
            video = (video * 255).numpy().astype(np.uint8)
            wandb_loss_dict['generator_pred_video'] = wandb.Video(
                video, fps=24, format="mp4")
            # Clean up references
            del video, latents

        # Log to wandb
        if self.global_rank == 0:
            wandb.log(wandb_loss_dict, step=step)

    def train(self) -> None:
        """Main training loop with distillation-specific logging."""
        assert self.training_args.seed is not None, "seed must be set"
        seed = self.training_args.seed

        # Set the same seed within each SP group to ensure reproducibility
        if self.sp_world_size > 1:
            # Use the same seed for all processes within the same SP group
            sp_group_seed = seed + (self.global_rank // self.sp_world_size)
            set_random_seed(sp_group_seed)
            logger.info("Rank %s: Using SP group seed %s", self.global_rank,
                        sp_group_seed)
        else:
            set_random_seed(seed + self.global_rank)

        # Set random seeds for deterministic training
        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed)
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(
            self.seed)
        self.validation_random_generator = torch.Generator(
            device="cpu").manual_seed(self.seed)
        logger.info("Initialized random seeds with seed: %s", seed)

        # Resume from checkpoint if specified (this will restore random states)
        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()
            logger.info("Resumed from checkpoint, random states restored")
        else:
            logger.info("Starting training from scratch")

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()
        self._log_validation(self.transformer, self.training_args,
                             self.init_steps)

        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            disable=self.local_rank > 0,
        )

        use_vsa = vsa_available and envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN"
        for step in range(self.init_steps + 1,
                          self.training_args.max_train_steps + 1):
            start_time = time.perf_counter()
            if use_vsa:
                vsa_sparsity = self.training_args.VSA_sparsity
                vsa_decay_rate = self.training_args.VSA_decay_rate
                vsa_decay_interval_steps = self.training_args.VSA_decay_interval_steps
                if vsa_decay_interval_steps > 1:
                    current_decay_times = min(step // vsa_decay_interval_steps,
                                              vsa_sparsity // vsa_decay_rate)
                    current_vsa_sparsity = current_decay_times * vsa_decay_rate
                else:
                    current_vsa_sparsity = vsa_sparsity
            else:
                current_vsa_sparsity = 0.0

            training_batch = TrainingBatch()
            self.current_trainstep = step
            training_batch.current_vsa_sparsity = current_vsa_sparsity

            with torch.autocast("cuda", dtype=torch.bfloat16):
                training_batch = self.train_one_step(training_batch)

            total_loss = training_batch.total_loss
            grad_norm = training_batch.grad_norm

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "total_loss":
                f"{total_loss:.4f}",
                "lpips_loss":
                f"{training_batch.regression_loss:.4f}"
                if hasattr(training_batch, 'regression_loss')
                and training_batch.regression_loss is not None else "N/A",
                "step_time":
                f"{step_time:.2f}s",
                "grad_norm":
                grad_norm,
            })
            progress_bar.update(1)

            if self.global_rank == 0:
                # Prepare logging data
                log_data = {
                    "train_total_loss": total_loss,
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                }
                if use_vsa:
                    log_data["VSA_train_sparsity"] = current_vsa_sparsity

                if training_batch.dmd_latent_vis_dict:
                    dmd_additional_logs = {
                        "generator_timestep": training_batch.dmd_latent_vis_dict["generator_timestep"].item(),
                        "regression_loss": training_batch.regression_loss,
                    }
                    log_data.update(dmd_additional_logs)

                wandb.log(log_data, step=step)

            # Save training state checkpoint (for resuming training)
            if (self.training_args.training_state_checkpointing_steps > 0
                    and step % self.training_args.training_state_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save training state checkpoint at step", step)
                save_checkpoint(self.transformer, self.global_rank,
                                self.training_args.output_dir, step,
                                self.optimizer, self.train_dataloader,
                                self.lr_scheduler, self.noise_random_generator)

                if self.transformer:
                    self.transformer.train()
                self.sp_group.barrier()

            # Save weight-only checkpoint (export consolidated generator weights)
            if (self.training_args.weight_only_checkpointing_steps > 0
                    and step % self.training_args.weight_only_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save weight-only checkpoint at step", step)
                save_checkpoint(self.transformer, self.global_rank,
                                self.training_args.output_dir,
                                f"{step}", self.optimizer, self.train_dataloader,
                                self.lr_scheduler, self.noise_random_generator)

            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                if self.training_args.log_visualization:
                    self.visualize_intermediate_latents(training_batch,
                                                        self.training_args,
                                                        step)
                self._log_validation(self.transformer, self.training_args, step)

        wandb.finish()

        # Save final training state checkpoint
        print("rank", self.global_rank,
              "save final training state checkpoint at step",
              self.training_args.max_train_steps)
        save_checkpoint(self.transformer, self.global_rank,
                        self.training_args.output_dir,
                        self.training_args.max_train_steps, self.optimizer,
                        self.train_dataloader, self.lr_scheduler,
                        self.noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()
