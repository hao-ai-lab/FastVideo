# SPDX-License-Identifier: Apache-2.0
import gc
import math
import os
import time
from abc import abstractmethod
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers.optimization import get_scheduler
from einops import rearrange
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

import fastvideo.v1.envs as envs
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset import build_parquet_map_style_dataloader
from fastvideo.v1.distributed import (cleanup_dist_env_and_memory, get_sp_group,
                                      get_local_torch_device, get_world_group)
from fastvideo.v1.fastvideo_args import FastVideoArgs,TrainingArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import (ComposedPipelineBase, ForwardBatch,
                                    TrainingBatch)
from fastvideo.v1.training.training_pipeline import TrainingPipeline
from fastvideo.v1.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases, load_checkpoint, save_checkpoint, prepare_for_saving)
from fastvideo.v1.utils import set_random_seed, is_vsa_available
from fastvideo.v1.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.v1.training.activation_checkpoint import (
    apply_activation_checkpointing)
from fastvideo.v1.dataset.validation_dataset import ValidationDataset
from fastvideo.v1.training.training_utils import pred_noise_to_pred_video, shift_timestep

import wandb  # isort: skip

vsa_available = is_vsa_available()

logger = init_logger(__name__)
    
class DistillationPipeline(TrainingPipeline):
    _required_config_modules = ["scheduler", "transformer", "vae", "realscore_transformer", "fakescore_transformer"]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      Dict[str, Any]]]
    current_epoch: int = 0

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the distillation training pipeline with multiple models."""
        logger.info("Initializing distillation training pipeline...")
        super().initialize_training_pipeline(training_args)

        self.noise_scheduler = self.get_module("scheduler")
        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)
        
        self.timestep_shift = self.training_args.pipeline_config.flow_shift
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=self.timestep_shift)
        
        # self.transformer is the generator model
        self.real_score_transformer = self.get_module("realscore_transformer")
        self.fake_score_transformer = self.get_module("fakescore_transformer")

        self.real_score_transformer.requires_grad_(False)
        self.real_score_transformer.eval()
        self.fake_score_transformer.requires_grad_(True)
        self.fake_score_transformer.train()

        if training_args.enable_gradient_checkpointing_type is not None:
            self.fake_score_transformer = apply_activation_checkpointing(
                self.fake_score_transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)
            self.real_score_transformer = apply_activation_checkpointing(
                self.real_score_transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)

        # Initialize optimizers
        fake_score_transformer_params = list(filter(lambda p: p.requires_grad, self.fake_score_transformer.parameters()))
        self.fake_score_transformer_optimizer = torch.optim.AdamW(
            fake_score_transformer_params,
            lr=training_args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )
        
        self.fake_score_transformer_lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.fake_score_transformer_optimizer,
            num_warmup_steps=training_args.lr_warmup_steps * self.world_size,
            num_training_steps=training_args.max_train_steps * self.world_size,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            last_epoch=self.init_steps - 1,
        )

        self.generator_update_interval = self.training_args.generator_update_interval
        logger.info(f"Distillation pipeline initialized with generator_update_interval={self.generator_update_interval}")

        self.dmd_denoising_steps = torch.tensor(
            self.training_args.dmd_denoising_steps, dtype=torch.long, device=get_local_torch_device())
        logger.info(f"Distillation Generator model to {len(self.dmd_denoising_steps)} denoising steps")
        self.num_train_timestep = self.noise_scheduler.num_train_timesteps

        self.min_timestep = int(self.training_args.min_timestep_ratio * self.num_train_timestep)
        self.max_timestep = int(self.training_args.max_timestep_ratio * self.num_train_timestep)

        self.real_score_guidance_scale = self.training_args.real_score_guidance_scale

    @abstractmethod
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Initialize validation pipeline - must be implemented by subclasses."""
        raise NotImplementedError(
            "Distillation pipelines must implement this method") 

    def _prepare_distillation(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Prepare training environment for distillation."""
        self.transformer.requires_grad_(True)
        self.transformer.train()
        self.fake_score_transformer.requires_grad_(True)
        self.fake_score_transformer.train()
         
        return training_batch

    def _generator_forward(self, training_batch: TrainingBatch) -> torch.Tensor:
        """Forward pass through student transformer and compute student losses."""
        latents = training_batch.latents
        dtype = latents.dtype
        device = latents.device
        index = torch.randint(0, len(self.dmd_denoising_steps), [1], device=device, dtype=torch.long)
        timestep = self.dmd_denoising_steps[index] 

        noise = torch.randn(self.video_latent_shape, device=self.device, dtype=dtype)
        if self.sp_world_size > 1:
            noise = rearrange(noise,
                              "b (n t) c h w -> b n t c h w",
                              n=self.sp_world_size).contiguous()
            noise = noise[:, self.rank_in_sp_group, :, :, :, :]
        noisy_latent = self.noise_scheduler.add_noise(latents.flatten(0, 1), noise.flatten(0, 1), timestep).unflatten(0, self.video_latent_shape_sp[:2])

        training_batch = self._build_input_kwargs(noisy_latent, timestep, training_batch.conditional_dict, training_batch)

        pred_noise = self.transformer(**training_batch.input_kwargs).permute(0, 2, 1, 3, 4)
        pred_video = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=training_batch.noise_latents.flatten(0, 1),
            timestep=timestep,
            scheduler=self.noise_scheduler
        ).unflatten(0, pred_noise.shape[:2])

        return training_batch, pred_video, timestep.float()

    def _dmd_forward(self, pred_video: torch.Tensor, training_batch: TrainingBatch) -> Tuple[torch.Tensor, dict]:
        """Compute DMD (Diffusion Model Distillation) loss."""
        
        original_latent = pred_video
        with torch.no_grad():
            timestep = torch.randint(0, self.num_train_timestep, [1], device=self.device, dtype=torch.long)        
            timestep = shift_timestep(timestep, self.timestep_shift, self.num_train_timestep)
            timestep = timestep.clamp(self.min_timestep, self.max_timestep)

            noise = torch.randn(self.video_latent_shape, device=self.device, dtype=pred_video.dtype)

            if self.sp_world_size > 1:
                noise = rearrange(noise,
                                    "b (n t) c h w -> b n t c h w",
                                    n=self.sp_world_size).contiguous()
                noise = noise[:, self.rank_in_sp_group, :, :, :, :]
                
            noisy_latent = self.noise_scheduler.add_noise(pred_video.flatten(0, 1), noise.flatten(0, 1), timestep).unflatten(0, self.video_latent_shape_sp[:2])

            # fake_score_transformer forward
            training_batch = self._build_input_kwargs(noisy_latent, timestep, training_batch.conditional_dict, training_batch)
            pred_noise_fake_score = self.fake_score_transformer(**training_batch.input_kwargs).permute(0, 2, 1, 3, 4)
            pred_fake_video = pred_noise_to_pred_video(
                pred_noise=pred_noise_fake_score.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler
            ).unflatten(0, pred_noise_fake_score.shape[:2])
            
            # real_score_transformer cond forward
            training_batch = self._build_input_kwargs(noisy_latent, timestep, training_batch.conditional_dict, training_batch)
            pred_noise_real_score_cond = self.real_score_transformer(**training_batch.input_kwargs).permute(0, 2, 1, 3, 4)
            pred_real_video_cond = pred_noise_to_pred_video(
                pred_noise=pred_noise_real_score_cond.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler
            ).unflatten(0, pred_noise_real_score_cond.shape[:2])
            
            # real_score_transformer uncond forward
            training_batch = self._build_input_kwargs(noisy_latent, timestep, training_batch.unconditional_dict, training_batch)
            pred_noise_real_score_uncond = self.real_score_transformer(**training_batch.input_kwargs).permute(0, 2, 1, 3, 4)
            pred_real_video_uncond = pred_noise_to_pred_video(
                pred_noise=pred_noise_real_score_uncond.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler
            ).unflatten(0, pred_noise_real_score_uncond.shape[:2])
            
            pred_real_video = pred_real_video_cond + (
                pred_real_video_cond - pred_real_video_uncond
            ) * self.real_score_guidance_scale

            grad = (pred_fake_video - pred_real_video) / torch.abs(original_latent - pred_real_video).mean()
            grad = torch.nan_to_num(grad).detach()

        dmd_log_dict = {
            "dmdtrain_latents": original_latent.detach(),
            "dmdtrain_noisy_latent": noisy_latent.detach(),
            "dmdtrain_pred_real_video": pred_real_video.detach(),
            "dmdtrain_pred_fake_video": pred_fake_video.detach(),
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.float().detach()
        }

        dmd_loss = 0.5 * F.mse_loss(original_latent.float(), (original_latent.float() - grad.float()))
        
        return training_batch, dmd_loss, dmd_log_dict

    def faker_score_forward(self, training_batch: TrainingBatch) -> Tuple[TrainingBatch, torch.Tensor, dict]:
        with torch.no_grad():
            with set_forward_context(
                current_timestep=training_batch.timesteps, attn_metadata=training_batch.attn_metadata_vsa):
                training_batch, generated_video, timestep_gen = self._generator_forward(training_batch)

        fake_score_timestep = torch.randint(0, self.num_train_timestep, [1], device=self.device, dtype=torch.long)
        fake_score_timestep = shift_timestep(fake_score_timestep, self.timestep_shift, self.num_train_timestep)

        fake_score_timestep = fake_score_timestep.clamp(self.min_timestep, self.max_timestep)

        # Use cross-codebase generator for reproducible noise generation
        fake_score_noise = torch.randn(
            self.video_latent_shape, device=self.device, dtype=generated_video.dtype)
        if self.sp_world_size > 1:
            fake_score_noise = rearrange(fake_score_noise,
                                    "b (n t) c h w -> b n t c h w",
                                    n=self.sp_world_size).contiguous()
            fake_score_noise = fake_score_noise[:, self.rank_in_sp_group, :, :, :, :]
        
        noisy_generated_video = self.noise_scheduler.add_noise(
            generated_video.flatten(0, 1),
            fake_score_noise.flatten(0, 1),
            fake_score_timestep
        ).unflatten(0, self.video_latent_shape_sp[:2])

        with set_forward_context(
                current_timestep=training_batch.timesteps, attn_metadata=training_batch.attn_metadata):
            training_batch = self._build_input_kwargs(noisy_generated_video, fake_score_timestep, training_batch.conditional_dict, training_batch)
            
            pred_noise_fake_score = self.fake_score_transformer(**training_batch.input_kwargs).permute(0, 2, 1, 3, 4)

        target = fake_score_noise - generated_video
        denoising_loss = torch.mean((pred_noise_fake_score - target) ** 2)

        fake_score_log_dict = {
            "fake_scoretrain_latent": generated_video.detach(),
            "fake_scoretrain_noisy_latent": noisy_generated_video.detach(),
            "fake_score_timestep": fake_score_timestep.float().detach(),
            "fake_score_timestep_gen": timestep_gen.float().detach(),
        }

        return training_batch, denoising_loss, fake_score_log_dict
        
    def _clip_grad_norm(self, training_batch: TrainingBatch, transformer) -> TrainingBatch:
        assert self.training_args is not None
        max_grad_norm = self.training_args.max_grad_norm

        # TODO(will): perhaps move this into transformer api so that we can do
        # the following:
        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        if max_grad_norm is not None:
            # Clip gradients for both student and fake_score models
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
        
        training_batch.conditional_dict = conditional_dict
        training_batch.unconditional_dict = unconditional_dict
        assert training_batch.latents is not None
        training_batch.raw_latent_shape = training_batch.latents.shape
        training_batch.latents = training_batch.latents.permute(0, 2, 1, 3, 4)
        self.video_latent_shape = training_batch.latents.shape # [B, T, C, H, W]

        if self.sp_world_size > 1:
            training_batch.latents = rearrange(training_batch.latents,
                                "b (n t) c h w -> b n t c h w",
                                n=self.sp_world_size).contiguous()
            training_batch.latents = training_batch.latents[:, self.rank_in_sp_group, :, :, :, :]

        self.video_latent_shape_sp = training_batch.latents.shape

        return training_batch

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Train one step with alternating student and fake_score updates, supporting gradient accumulation."""
        import copy
        gradient_accumulation_steps = getattr(self.training_args, 'gradient_accumulation_steps', 1)
        batches = []
        # Collect N batches for gradient accumulation
        for _ in range(gradient_accumulation_steps):
            batch = self._prepare_distillation(training_batch)
            batch = self._get_next_batch(batch)
            batch = self._normalize_dit_input(batch)
            batch = self._prepare_dit_inputs(batch)
            batch = self._build_attention_metadata(batch)
            batch.attn_metadata_vsa = copy.deepcopy(batch.attn_metadata)
            if batch.attn_metadata is not None:
                batch.attn_metadata.VSA_sparsity = 0.0
            batches.append(batch)

        self.optimizer.zero_grad()
        total_dmd_loss = 0.0
        total_dmd_log_dict = None
        if (self.current_trainstep % self.generator_update_interval == 0) or (self.current_trainstep == 1):
            for batch in batches:
                batch_gen = copy.deepcopy(batch)
                with set_forward_context(
                        current_timestep=batch_gen.timesteps, attn_metadata=batch_gen.attn_metadata_vsa):
                    batch_gen, pred_video, timestep_gen = self._generator_forward(batch_gen)
                
                with set_forward_context(
                        current_timestep=batch_gen.timesteps, attn_metadata=batch_gen.attn_metadata):
                    batch_gen, dmd_loss, dmd_log_dict = self._dmd_forward(
                        pred_video=pred_video,
                        training_batch=batch_gen
                    )
                dmd_log_dict['dmd_timestep_gen'] = timestep_gen
                # Ensure backward is under the correct forward context
                dmd_loss = dmd_loss / gradient_accumulation_steps
                with set_forward_context(
                    current_timestep=batch_gen.timesteps, attn_metadata=batch_gen.attn_metadata_vsa):
                    dmd_loss.backward()
                total_dmd_loss += dmd_loss.detach().item()
                if total_dmd_log_dict is None:
                    total_dmd_log_dict = dmd_log_dict
                # Only keep the first log dict, ignore subsequent ones
            self._clip_grad_norm(batch_gen, self.transformer)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            avg_dmd_loss = torch.tensor(total_dmd_loss / gradient_accumulation_steps, device=self.device)
            world_group = get_world_group()
            world_group.all_reduce(avg_dmd_loss, op=torch.distributed.ReduceOp.AVG)
            training_batch.generator_loss = avg_dmd_loss.item()
            training_batch.dmd_log_dict = total_dmd_log_dict if total_dmd_log_dict is not None else {}
        else:
            training_batch.generator_loss = 0.0
            training_batch.dmd_log_dict = {}

        self.fake_score_transformer_optimizer.zero_grad()
        total_fake_score_loss = 0.0
        total_fake_score_log_dict = None
        for batch in batches:
            batch_fake = copy.deepcopy(batch)
            batch_fake, fake_score_loss, fake_score_log_dict = self.faker_score_forward(batch_fake)
            fake_score_loss = fake_score_loss / gradient_accumulation_steps
            with set_forward_context(
                current_timestep=batch_fake.timesteps, attn_metadata=batch_fake.attn_metadata):
                fake_score_loss.backward()
            total_fake_score_loss += fake_score_loss.detach().item()
            if total_fake_score_log_dict is None:
                total_fake_score_log_dict = fake_score_log_dict
            # Only keep the first log dict, ignore subsequent ones
        self._clip_grad_norm(batch_fake, self.fake_score_transformer)
        self.fake_score_transformer_optimizer.step()
        self.fake_score_transformer_lr_scheduler.step()
        self.fake_score_transformer_optimizer.zero_grad(set_to_none=True)
        avg_fake_score_loss = torch.tensor(total_fake_score_loss / gradient_accumulation_steps, device=self.device)
        world_group = get_world_group()
        world_group.all_reduce(avg_fake_score_loss, op=torch.distributed.ReduceOp.AVG)
        training_batch.fake_score_loss = avg_fake_score_loss.item()
        training_batch.fake_score_log_dict = total_fake_score_log_dict if total_fake_score_log_dict is not None else {}

        training_batch.total_loss = training_batch.generator_loss + training_batch.fake_score_loss
        return training_batch

    def _resume_from_checkpoint(self) -> None: #TODO(yongqi)
        """Resume training from checkpoint with distillation models."""
        assert self.training_args is not None
        logger.info("Loading distillation checkpoint from %s",
                    self.training_args.resume_from_checkpoint)
        
        resumed_step = load_checkpoint(
            self.transformer.model, self.global_rank,
            self.training_args.resume_from_checkpoint, self.optimizer,
            self.train_dataloader, self.lr_scheduler,
            self.noise_random_generator)
            
        # TODO: Add checkpoint loading for fake_score and teacher models
        
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
        logger.info("  Generator update interval: %s", self.generator_update_interval)
        assert isinstance(self.training_args, TrainingArgs)
        logger.info("  Max gradient norm: %s", self.training_args.max_grad_norm)
        assert self.real_score_transformer is not None
        logger.info("  Real score transformer parameters: %s B",
                    sum(p.numel() for p in self.real_score_transformer.parameters()) / 1e9)
        assert self.fake_score_transformer is not None
        logger.info("  Fake score transformer parameters: %s B",
                    sum(p.numel() for p in self.fake_score_transformer.parameters()) / 1e9)
        
    def add_visualization(self, generator_log_dict: Dict[str, Any], fake_score_log_dict: Dict[str, Any], training_args: TrainingArgs, step: int):
        """Add visualization data to wandb logging and save frames to disk."""
        wandb_loss_dict = {}
        
        # Clear GPU cache before VAE decoding to prevent OOM
        torch.cuda.empty_cache()
        
        # # Use consistent decoding approach - use decode_stage for all
        # decode_stage = self.validation_pipeline._stages[-1]
        
        # Process fake_score training data
        fake_score_latents_name = ['fake_scoretrain_latent', 'fake_scoretrain_noisy_latent']

        for latent_key in fake_score_latents_name:
            latents = fake_score_log_dict[latent_key] # bs, t,c, h, w
            latents = latents.permute(0, 2, 1, 3, 4)
            # decoded_latent = decode_stage(ForwardBatch(data_type="video", latents=latents), training_args)
            
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

            video = self.vae.decode(latents)
            video = (video / 2 + 0.5).clamp(0, 1)
            video = video.cpu().float()
            video = video.permute(0, 2, 1, 3, 4)

            wandb_loss_dict[latent_key] = prepare_for_saving(video)
            # Clean up references
            del video, latents
            torch.cuda.empty_cache()

        # Process DMD training data if available - use decode_stage instead of self.vae.decode
        if 'dmdtrain_pred_fake_video' in generator_log_dict:
            dmd_latents_name = ['dmdtrain_pred_fake_video', 'dmdtrain_pred_real_video', 'dmdtrain_latents', 'dmdtrain_noisy_latent']
            for latent_key in dmd_latents_name:
                latents = generator_log_dict[latent_key]
                latents = latents.permute(0, 2, 1, 3, 4)
                # decoded_latent = decode_stage(ForwardBatch(data_type="video", latents=latents), training_args)
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
                video = self.vae.decode(latents)
                video = (video / 2 + 0.5).clamp(0, 1)
                video = video.cpu().float()
                video = video.permute(0, 2, 1, 3, 4)
                
                wandb_loss_dict[latent_key] = prepare_for_saving(video)
                # Clean up references
                del video, latents
                torch.cuda.empty_cache()
        
        # Log to wandb
        if self.global_rank == 0:
            wandb.log(wandb_loss_dict, step=step)

    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        assert training_args is not None
        training_args.inference_mode = True
        training_args.use_cpu_offload = True
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation")

        # Create sampling parameters if not provided
        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

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
                result_batch = self.validation_pipeline.prompt_encoding_stage(batch_negative, training_args)
                self.negative_prompt_embeds, self.negative_prompt_attention_mask = result_batch.prompt_embeds[
                    0], result_batch.prompt_attention_mask[0]

                # logger.info("rank: %s: rank_in_sp_group: %s, batch.prompt: %s",
                #             self.global_rank,
                #             self.rank_in_sp_group,
                #             batch.prompt,
                #             local_main_process_only=False)

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
                    
                    # Save all prompts from all cards to txt file
                    # prompt_filename = os.path.join(
                    #     training_args.output_dir,
                    #     f"validation_step_{global_step}_inference_steps_{num_inference_steps}_prompts.txt"
                    # )
                    # with open(prompt_filename, 'w', encoding='utf-8') as f:
                    #     for i, caption in enumerate(all_captions):
                    #         f.write(f"Video_{i}: {caption}\n")
                    # logger.info(f"Saved {len(all_captions)} prompts to {prompt_filename}")
                    
                else:
                    # Other sp_group leaders send their results to global rank 0
                    world_group.send_object(step_videos, dst=0)
                    world_group.send_object(step_captions, dst=0)

        # Re-enable gradients for training
        transformer.train()
        gc.collect()
        torch.cuda.empty_cache()
        
    def train(self) -> None:
        """Main training loop with distillation-specific logging."""
        assert self.training_args is not None

        assert self.training_args.seed is not None, "seed must be set"
        seed = self.training_args.seed
        
        # Set the same seed within each SP group to ensure reproducibility
        if self.sp_world_size > 1:
            # Use the same seed for all processes within the same SP group
            sp_group_seed = seed + (self.global_rank // self.sp_world_size)
            set_random_seed(sp_group_seed)
            logger.info(f"Rank {self.global_rank}: Using SP group seed {sp_group_seed}")
        else:
            set_random_seed(seed + self.global_rank)

        self.noise_random_generator = torch.Generator(
            device="cpu").manual_seed(seed)
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(
            self.seed)
        self.validation_random_generator = torch.Generator(
            device="cpu").manual_seed(seed)

        logger.info("Initialized random seeds with seed: %s", seed)

        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()
        self._log_validation(self.transformer, self.training_args, 0)

        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            disable=self.local_rank > 0,
        )
        
        for step in range(self.init_steps + 1,
                        self.training_args.max_train_steps + 1):
            start_time = time.perf_counter()
            if vsa_available:
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
            generator_loss = training_batch.generator_loss
            fake_score_loss = training_batch.fake_score_loss
            grad_norm = training_batch.grad_norm

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "total_loss": f"{total_loss:.4f}",
                "generator_loss": f"{generator_loss:.4f}",
                "fake_score_loss": f"{fake_score_loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            })
            progress_bar.update(1)

            if self.global_rank == 0:
                # Prepare logging data
                log_data = {
                    "train_total_loss": total_loss,
                    "train_generator_loss": generator_loss,
                    "train_fake_score_loss": fake_score_loss,
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                    "VSA_train_sparsity": current_vsa_sparsity,
                }
                
                # Add DMD training metrics if available
                if hasattr(training_batch, 'dmd_log_dict') and training_batch.dmd_log_dict:
                    dmd_metrics = {
                        "dmd_gradient_norm": training_batch.dmd_log_dict.get("dmdtrain_gradient_norm", 0.0),
                        "dmd_timestep": training_batch.dmd_log_dict.get("timestep", 0.0).mean().item(),
                        "dmd_timestep_gen": training_batch.dmd_log_dict.get("dmd_timestep_gen", 0.0).mean().item()
                    }
                    log_data.update(dmd_metrics)
                
                # Add fake_score training metrics if available
                if hasattr(training_batch, 'fake_score_log_dict') and training_batch.fake_score_log_dict:
                    fake_score_metrics = {
                        "fake_score_timestep": training_batch.fake_score_log_dict.get("fake_score_timestep", 0.0).mean().item(),
                        "fake_score_timestep_gen": training_batch.fake_score_log_dict.get("fake_score_timestep_gen", 0.0).mean().item(),
                    }
                    log_data.update(fake_score_metrics)
                wandb.log(log_data, step=step)
                
            if step % self.training_args.checkpointing_steps == 0:
                print("rank", self.global_rank, "save checkpoint at step", step)
                save_checkpoint(self.transformer, self.global_rank, #TODO(yongqi)
                                self.training_args.output_dir, step,
                                self.optimizer, self.train_dataloader,
                                self.lr_scheduler, self.noise_random_generator)
                if self.transformer:
                    self.transformer.train()
                self.sp_group.barrier()
                
            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:


                self.add_visualization(training_batch.dmd_log_dict, training_batch.fake_score_log_dict, self.training_args, step)
                self._log_validation(self.transformer, self.training_args, step)



        wandb.finish()
        # save_checkpoint(self.transformer, self.global_rank,
        #                 self.training_args.output_dir,
        #                 self.training_args.max_train_steps, self.optimizer,
        #                 self.train_dataloader, self.lr_scheduler,
        #                 self.noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()


