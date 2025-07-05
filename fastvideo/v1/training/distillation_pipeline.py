# SPDX-License-Identifier: Apache-2.0
import gc
import math
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers.optimization import get_scheduler
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

import fastvideo.v1.envs as envs
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset import build_parquet_map_style_dataloader
from fastvideo.v1.distributed import (cleanup_dist_env_and_memory, get_sp_group,
                                      get_torch_device, get_world_group)
from fastvideo.v1.fastvideo_args import FastVideoArgs,TrainingArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import (ComposedPipelineBase, ForwardBatch,
                                    TrainingBatch)
from fastvideo.v1.training.training_pipeline import TrainingPipeline
from fastvideo.v1.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, get_sigmas, load_checkpoint,
    normalize_dit_input, save_checkpoint, shard_latents_across_sp, prepare_for_saving)
from fastvideo.v1.utils import set_random_seed
from fastvideo.v1.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchDiscreteScheduler
from fastvideo.v1.training.activation_checkpoint import (
    apply_activation_checkpointing)
import wandb  # isort: skip

logger = init_logger(__name__)


class DistillationPipeline(TrainingPipeline):
    """
    A distillation pipeline for training a student model using teacher model guidance.
    Inherits from TrainingPipeline to reuse training infrastructure.
    """
    _required_config_modules = ["scheduler", "transformer", "vae", "teacher_transformer", "critic_transformer"]
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
        
        # 1. Call parent initialization first
        super().initialize_training_pipeline(training_args)
        
        self.noise_scheduler = self.get_module("scheduler")
        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)
        # 2. Distillation-specific initialization
        # The parent class already sets self.transformer as the student model
        self.teacher_transformer = self.get_module("teacher_transformer")
        self.critic_transformer = self.get_module("critic_transformer")
        self.teacher_transformer.requires_grad_(False)
        self.teacher_transformer.eval()
        self.critic_transformer.requires_grad_(True)
        self.critic_transformer.train()
        
        if training_args.enable_gradient_checkpointing_type is not None:
            self.critic_transformer = apply_activation_checkpointing(
                self.critic_transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)

        # Initialize optimizers
        critic_params = list(filter(lambda p: p.requires_grad, self.critic_transformer.parameters()))
        self.critic_transformer_optimizer = torch.optim.AdamW(
            critic_params,
            lr=training_args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )
        
        self.critic_lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.critic_transformer_optimizer,
            num_warmup_steps=training_args.lr_warmup_steps * self.world_size,
            num_training_steps=training_args.max_train_steps * self.world_size,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            last_epoch=self.init_steps - 1,
        )
                
        logger.info("Distillation optimizers initialized: student and critic")

        self.student_critic_update_ratio = self.training_args.student_critic_update_ratio
        logger.info(f"Distillation pipeline initialized with student_critic_update_ratio={self.student_critic_update_ratio}")

        self.denoising_step_list = torch.tensor(
            self.training_args.denoising_step_list, dtype=torch.long, device=get_torch_device())
        logger.info(f"Distillation student model to {len(self.denoising_step_list)} denoising steps")
        self.num_train_timestep = self.noise_scheduler.config.num_train_timesteps
        # TODO(yongqi): hardcode for bidirectional distillation
        self.distill_task_type = "bidirectional_video"
        self.denoising_loss_type = 'flow'
        # TODO(yongqi): hardcode for causal distillation
        self.num_frame_per_block = 3

        self.timestep_shift = self.training_args.pipeline_config.flow_shift
        self.min_step = int(self.training_args.min_step_ratio * self.num_train_timestep)
        self.max_step = int(self.training_args.max_step_ratio * self.num_train_timestep)

        self.teacher_guidance_scale = self.training_args.teacher_guidance_scale
        self.denoising_loss_func = FlowPredLoss()



    @abstractmethod
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Initialize validation pipeline - must be implemented by subclasses."""
        raise NotImplementedError(
            "Distillation pipelines must implement this method") 

    def _prepare_distillation(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Prepare training environment for distillation."""
        self.transformer.requires_grad_(True)
        self.transformer.train()
        self.critic_transformer.requires_grad_(True)
        self.critic_transformer.train()
        self.optimizer.zero_grad()
        self.critic_transformer_optimizer.zero_grad()
         
        return training_batch
    
    def _process_timestep(self, timestep: torch.Tensor, type: str) -> torch.Tensor:
        """
        Pre-process the randomly generated timestep based on the generator's task type.
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.
            - type: a string indicating the type of the current model (image, bidirectional_video, or causal_video).
        Output Behavior:
            - image: check that the second dimension (num_frame) is 1.
            - bidirectional_video: broadcast the timestep to be the same for all frames.
            - causal_video: broadcast the timestep to be the same for all frames **in a block**.
        """
        if type == "image":
            assert timestep.shape[1] == 1
            return timestep
        elif type == "bidirectional_video":
            for index in range(timestep.shape[0]):
                timestep[index] = timestep[index, 0]
            return timestep
        elif type == "causal_video":
            # make the noise level the same within every motion block
            timestep = timestep.reshape(
                timestep.shape[0], -1, self.num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
        else:
            raise NotImplementedError("Unsupported model type {}".format(type))

    def _student_forward(self, conditional_dict: Dict[str, Any], unconditional_dict: Dict[str, Any], latents: torch.Tensor) -> torch.Tensor:
        """Forward pass through student transformer and compute student losses."""
        dtype = latents.dtype
        simulated_noisy_input = []
        for timestep in self.denoising_step_list:
            noise = torch.randn(
                self.video_latent_shape, device=self.device, dtype=dtype)

            noisy_timestep = timestep * torch.ones(
                [self.latent_shape_bs, self.latent_shape_t], device=self.device, dtype=torch.long)

            if timestep != 0:
                noisy_image = self.noise_scheduler.add_noise(
                    latents.flatten(0, 1),
                    noise.flatten(0, 1),
                    noisy_timestep.flatten(0, 1)
                ).unflatten(0, self.video_latent_shape[:2])
            else:
                noisy_image = latents

            simulated_noisy_input.append(noisy_image)

        simulated_noisy_input = torch.stack(simulated_noisy_input, dim=1)

        # Step 2: Randomly sample a timestep and pick the corresponding input
        index = torch.randint(0, len(self.denoising_step_list), [
                              self.latent_shape_bs, self.latent_shape_t], device=self.device, dtype=torch.long)

        index = self._process_timestep(index, type=self.distill_task_type)
        # torch.distributed.breakpoint()
        # select the corresponding timestep's noisy input from the stacked tensor [B, T, F, C, H, W]
        index_expanded = index.reshape(index.shape[0], 1, index.shape[1], 1, 1, 1)
        index_expanded= index_expanded.expand(-1, -1, -1, self.video_latent_shape[1], *self.video_latent_shape[3:]).permute(0, 1, 3, 2, 4, 5)
        noisy_input = torch.gather(
            simulated_noisy_input, dim=1,
            index=index_expanded
        ).squeeze(1)

        timestep = self.denoising_step_list[index]
        # TODO(yongqi)
        pred_video_noise = self.transformer(
            hidden_states=noisy_input,
            **conditional_dict,
            timestep=timestep[0][:1]
        )

        pred_video = self._convert_flow_pred_to_x0(
            flow_pred=pred_video_noise.flatten(0, 1),
            xt=noisy_input.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, pred_video_noise.shape[:2])
        
        pred_video = pred_video.type_as(noisy_input)

        return pred_video

    def _compute_kl_grad(
        self, noisy_video: torch.Tensor,
        estimated_clean_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:

        pred_fake_video_noise = self.critic_transformer(
            hidden_states=noisy_video,
            **conditional_dict,
            timestep=timestep[0][:1]
        )
        pred_fake_video = self._convert_flow_pred_to_x0(
            flow_pred=pred_fake_video_noise.flatten(0, 1),
            xt=noisy_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, pred_fake_video_noise.shape[:2])
                
        pred_real_video_cond_noise = self.teacher_transformer(
            hidden_states=noisy_video,
            **conditional_dict,
            timestep=timestep[0][:1]
        )
        pred_real_video_cond = self._convert_flow_pred_to_x0(
            flow_pred=pred_real_video_cond_noise.flatten(0, 1),
            xt=noisy_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, pred_real_video_cond_noise.shape[:2])

        pred_real_video_uncond_noise = self.teacher_transformer(
            hidden_states=noisy_video,
            **unconditional_dict,
            timestep=timestep[0][:1]
        )
        pred_real_video_uncond = self._convert_flow_pred_to_x0(
            flow_pred=pred_real_video_uncond_noise.flatten(0, 1),
            xt=noisy_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, pred_real_video_uncond_noise.shape[:2])

        pred_real_video = pred_real_video_cond + (
            pred_real_video_cond - pred_real_video_uncond
        ) * self.teacher_guidance_scale

        grad = (pred_fake_video - pred_real_video)

        if normalization:
            p_real = (estimated_clean_video - pred_real_video)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_latents": estimated_clean_video.detach(),
            "dmdtrain_noisy_latent": noisy_video.detach(),
            "dmdtrain_pred_real_video": pred_real_video.detach(),
            "dmdtrain_pred_fake_video": pred_fake_video.detach(),
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.float().detach()
        }

    def _compute_dmd_loss(self, pred_video: torch.Tensor, conditional_dict: Dict[str, Any], unconditional_dict: Dict[str, Any]) -> Tuple[torch.Tensor, dict]:
        """Compute DMD (Diffusion Model Distillation) loss."""
        
        original_latent = pred_video

        with torch.no_grad():
            timestep = torch.randint(
                0,
                self.num_train_timestep,
                [self.latent_shape_bs, self.latent_shape_t],
                device=self.device,
                dtype=torch.long,
                generator=self.timestep_generator
            )

            timestep = self._process_timestep(
                timestep, type=self.distill_task_type)

            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / self.num_train_timestep) / \
                    (1 + (self.timestep_shift - 1) * (timestep / self.num_train_timestep)) * self.num_train_timestep
            
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(pred_video)
            noisy_latent = self.noise_scheduler.add_noise(
                pred_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, self.video_latent_shape[:2])

            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_video=noisy_latent,
                estimated_clean_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        dmd_loss = 0.5 * F.mse_loss(original_latent.double(
        ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        
        return dmd_loss, dmd_log_dict

    def _student_forward_and_compute_dmd_loss(self, training_batch: TrainingBatch) -> Tuple[TrainingBatch, torch.Tensor, dict]:
        """Forward pass through student transformer and compute student losses."""
        assert training_batch.conditional_dict is not None
        assert training_batch.unconditional_dict is not None
        assert training_batch.latents is not None

        pred_video = self._student_forward(
            conditional_dict=training_batch.conditional_dict,
            unconditional_dict=training_batch.unconditional_dict,
            latents=training_batch.latents
        )

        dmd_loss, dmd_log_dict = self._compute_dmd_loss(
            pred_video=pred_video,
            conditional_dict=training_batch.conditional_dict,
            unconditional_dict=training_batch.unconditional_dict
        )
        
        return training_batch, dmd_loss, dmd_log_dict

    def _critic_forward_and_compute_loss(self, training_batch: TrainingBatch) -> Tuple[TrainingBatch, torch.Tensor, dict]:
        assert training_batch.conditional_dict is not None
        assert training_batch.unconditional_dict is not None
        assert training_batch.latents is not None

        with torch.no_grad():
            generated_video = self._student_forward(
                conditional_dict=training_batch.conditional_dict,
                unconditional_dict=training_batch.unconditional_dict,
                latents=training_batch.latents
            )

        critic_timestep = torch.randint(
            0,
            self.num_train_timestep,
            [self.latent_shape_bs, self.latent_shape_t],
            device=self.device,
            dtype=torch.long,
            generator=self.timestep_generator
        )
        critic_timestep = self._process_timestep(
            critic_timestep, type=self.distill_task_type)

        # TODO: Add timestep warping
        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / self.num_train_timestep) / (1 + (self.timestep_shift - 1) * (critic_timestep / self.num_train_timestep)) * self.num_train_timestep

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_video)
        noisy_generated_video = self.noise_scheduler.add_noise(
            generated_video.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, self.video_latent_shape[:2])


        pred_fake_video_noise = self.critic_transformer(
            hidden_states=noisy_generated_video,
            **training_batch.conditional_dict,
            timestep=critic_timestep[0][:1]
        )
        pred_fake_video = self._convert_flow_pred_to_x0(
            flow_pred=pred_fake_video_noise.flatten(0, 1),
            xt=noisy_generated_video.flatten(0, 1),
            timestep=critic_timestep.flatten(0, 1)
        ).unflatten(0, pred_fake_video_noise.shape[:2])

        # Step 3: Compute the denoising loss for the fake critic
        flow_pred = self._convert_x0_to_flow_pred(
            scheduler=self.noise_scheduler,
            x0_pred=pred_fake_video.flatten(0, 1),
            xt=noisy_generated_video.flatten(0, 1),
            timestep=critic_timestep.flatten(0, 1)
        )

        denoising_loss = self.denoising_loss_func(
            x=generated_video.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            flow_pred=flow_pred
        )

        critic_log_dict = {
            "critictrain_latent": generated_video.detach(),
            "critictrain_noisy_latent": noisy_generated_video.detach(),
            "critictrain_pred_video": pred_fake_video.detach(),
            "critic_timestep": critic_timestep.float().detach()
        }

        return training_batch, denoising_loss, critic_log_dict

    def _convert_x0_to_flow_pred(self,scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps] 
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(1, -1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.noise_scheduler.sigmas,
                                                        self.noise_scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(1, -1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)
        
    def _clip_grad_norm(self, training_batch: TrainingBatch, transformer) -> TrainingBatch:
        assert self.training_args is not None
        max_grad_norm = self.training_args.max_grad_norm

        # TODO(will): perhaps move this into transformer api so that we can do
        # the following:
        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        if max_grad_norm is not None:
            # Clip gradients for both student and critic models
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
        self.video_latent_shape = training_batch.latents.shape # [B, C, T, H, W]
        self.latent_shape_bs = training_batch.latents.shape[0]
        self.latent_shape_t = training_batch.latents.shape[2]

        return training_batch
    
    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Train one step with alternating student and critic updates."""
        assert self.training_args is not None
        training_batch = self._prepare_distillation(training_batch)
        TRAIN_STUDENT = self.current_trainstep % self.student_critic_update_ratio == 0
    # for _ in range(self.training_args.gradient_accumulation_steps):
        training_batch = self._get_next_batch(training_batch)

        training_batch = self._normalize_dit_input(training_batch)
        training_batch = self._prepare_dit_inputs(training_batch)
        
        training_batch = self._build_attention_metadata(training_batch)
        
        if TRAIN_STUDENT:
            training_batch, dmd_loss, dmd_log_dict = self._student_forward_and_compute_dmd_loss(training_batch)
            training_batch.dmd_log_dict = dmd_log_dict
            dmd_loss.backward()
            training_batch = self._clip_grad_norm(training_batch, self.transformer)
            self.optimizer.step()
            self.lr_scheduler.step()
            
            avg_dmd_loss = dmd_loss.detach().clone()
            world_group = get_world_group()
            world_group.all_reduce(avg_dmd_loss, op=torch.distributed.ReduceOp.AVG)
            
            training_batch.student_loss = avg_dmd_loss.item() 
                            
        training_batch, critic_loss, critic_log_dict = self._critic_forward_and_compute_loss(training_batch)
        training_batch.critic_log_dict = critic_log_dict
        critic_loss.backward()
        training_batch = self._clip_grad_norm(training_batch, self.critic_transformer)
        self.critic_transformer_optimizer.step()
        self.critic_lr_scheduler.step()
        
        avg_critic_loss = critic_loss.detach().clone()
        world_group = get_world_group()
        world_group.all_reduce(avg_critic_loss, op=torch.distributed.ReduceOp.AVG)

        # Record loss values for logging
            
        training_batch.critic_loss = avg_critic_loss.item()
        
        training_batch.total_loss = training_batch.student_loss + training_batch.critic_loss
        
        return training_batch

    def _resume_from_checkpoint(self) -> None: #TODO(yongqi)
        """Resume training from checkpoint with distillation models."""
        assert self.training_args is not None
        logger.info("Loading distillation checkpoint from %s",
                    self.training_args.resume_from_checkpoint)
        
        resumed_step = load_checkpoint(
            self.transformer, self.global_rank,
            self.training_args.resume_from_checkpoint, self.optimizer,
            self.train_dataloader, self.lr_scheduler,
            self.noise_random_generator)
            
        # TODO: Add checkpoint loading for critic and teacher models
        
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
        logger.info("  Student/Critic update ratio: %s", self.student_critic_update_ratio)
        assert isinstance(self.training_args, TrainingArgs)
        logger.info("  Max gradient norm: %s", self.training_args.max_grad_norm)
        assert self.teacher_transformer is not None
        logger.info("  Teacher transformer parameters: %s B",
                    sum(p.numel() for p in self.teacher_transformer.parameters()) / 1e9)
        assert self.critic_transformer is not None
        logger.info("  Critic transformer parameters: %s B",
                    sum(p.numel() for p in self.critic_transformer.parameters()) / 1e9)
        
    def add_visualization(self, generator_log_dict: Dict[str, Any], critic_log_dict: Dict[str, Any], training_args: TrainingArgs, step: int):
        """Add visualization data to wandb logging."""
        wandb_loss_dict = {}
        
        # Clear GPU cache before VAE decoding to prevent OOM
        torch.cuda.empty_cache()
        
        # # Use consistent decoding approach - use decode_stage for all
        # decode_stage = self.validation_pipeline._stages[-1]
        
        # Process critic training data
        critic_latents_name = ['critictrain_latent', 'critictrain_noisy_latent', 'critictrain_pred_video']
        # critic_latents_name = ['critictrain_pred_video']
        for latent_key in critic_latents_name:
            latents = critic_log_dict[latent_key]
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

        dmd_latents_name = ['dmdtrain_pred_fake_video', 'dmdtrain_pred_real_video', 'dmdtrain_latents', 'dmdtrain_noisy_latent']
        for latent_key in dmd_latents_name:
            latents = generator_log_dict[latent_key]
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

    def dmd_inference(self, transformer, training_args, batch) -> torch.Tensor:
        #TODO(yongqi): remove hardcode shape
        noise=torch.randn(
            1,16,16,56,104, generator=torch.Generator(device="cuda").manual_seed(42),
            dtype=torch.bfloat16, device="cuda"
        )
        
        conditional_dict = {
            "encoder_hidden_states": batch.prompt_embeds,
            "encoder_attention_mask": batch.prompt_attention_mask,
        }

        # initial point
        noisy_video = noise
        
        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones(noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
            with set_forward_context(
                    current_timestep=0, attn_metadata=None):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_video_noise = transformer(
                        hidden_states=noisy_video,
                        **conditional_dict,
                        timestep=timestep[0][:1]
                    )  # [B, F, C, H, W]
                
            pred_video = self._convert_flow_pred_to_x0(
                flow_pred=pred_video_noise.flatten(0, 1),
                xt=noisy_video.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, pred_video_noise.shape[:2])
                
            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device)

                noisy_video = self.noise_scheduler.add_noise(
                    pred_video.flatten(0, 1),
                    torch.randn_like(pred_video.flatten(0, 1)),
                    next_timestep.flatten(0, 1)
                ).unflatten(0, noise.shape[:2])
                
        if isinstance(self.vae.scaling_factor, torch.Tensor):
            pred_video = pred_video / self.vae.scaling_factor.to(
                pred_video.device, pred_video.dtype)
        else:
            pred_video = pred_video / self.vae.scaling_factor

        # Apply shifting if needed
        if (hasattr(self.vae, "shift_factor")
                and self.vae.shift_factor is not None):
            if isinstance(self.vae.shift_factor, torch.Tensor):
                pred_video += self.vae.shift_factor.to(pred_video.device,
                                                    pred_video.dtype)
            else:
                pred_video += self.vae.shift_factor
                
        video = self.vae.decode(pred_video)
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().float()
        
        return video

    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        assert training_args is not None
        training_args.inference_mode = True
        training_args.use_cpu_offload = False
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation")

        # Create sampling parameters if not provided
        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

        # Set deterministic seed for validation
        set_random_seed(self.seed)
        logger.info("Using validation seed: %s", self.seed)

        # Prepare validation prompts
        logger.info('fastvideo_args.validation_preprocessed_path: %s',
                    training_args.validation_preprocessed_path)
        validation_dataset, validation_dataloader = build_parquet_map_style_dataloader(
            training_args.validation_preprocessed_path,
            batch_size=1,
            parquet_schema=self.validation_dataset_schema,
            num_data_workers=0,
            cfg_rate=0.0,
            drop_last=False,
            drop_first_row=sampling_param.negative_prompt is not None)
        if sampling_param.negative_prompt:
            self.negative_prompt_embeds, self.negative_prompt_attention_mask, negative_prompt = validation_dataset.get_validation_negative_prompt(
            )
            logger.info("Using negative_prompt: %s", negative_prompt)

        transformer.eval()

        validation_steps = training_args.validation_sampling_steps.split(",")
        validation_steps = [int(step) for step in validation_steps]
        validation_steps = [step for step in validation_steps if step > 0]

        # Process each validation prompt for each validation step
        for num_inference_steps in validation_steps:
            step_videos: List[np.ndarray] = []
            step_captions: List[str] = []

            for validation_batch in validation_dataloader:
                batch = self._prepare_validation_inputs(
                    sampling_param, training_args, validation_batch,
                    num_inference_steps, self.negative_prompt_embeds,
                    self.negative_prompt_attention_mask)

                assert batch.prompt is not None and isinstance(
                    batch.prompt, str)
                step_captions.append(batch.prompt)

                # Run validation inference
                    #TODO add dmd inference
                    # output_batch = self.validation_pipeline.forward(
                    #     batch, training_args)
                    # samples = output_batch.output # [1, 3, 61, 448, 832]
                samples = self.dmd_inference(
                    transformer, training_args, batch)
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
                    for i, (video,
                            caption) in enumerate(zip(all_videos,
                                                      all_captions)):
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
                                video_filenames, all_captions)
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
        torch.cuda.empty_cache()
        
    def train(self) -> None:
        """Main training loop with distillation-specific logging."""
        assert self.training_args is not None

        assert self.training_args.seed is not None, "seed must be set"
        seed = self.training_args.seed
        set_random_seed(seed)

        self.noise_random_generator = torch.Generator(
            device="cpu").manual_seed(seed)
        self.timestep_generator = torch.Generator(device=get_torch_device()).manual_seed(seed)

        logger.info("Initialized random seeds with seed: %s", seed)

        self.noise_scheduler = FlowMatchDiscreteScheduler()

        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()
        self._log_validation(self.transformer, self.training_args, 1)

        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            disable=self.local_rank > 0,
        )
        
        for step in range(self.init_steps,
                          self.training_args.max_train_steps + 1):
            start_time = time.perf_counter()

            training_batch = TrainingBatch()
            self.current_trainstep = step
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with set_forward_context(
                    current_timestep=step, attn_metadata=None):
                    training_batch = self.train_one_step(training_batch)

            total_loss = training_batch.total_loss
            student_loss = training_batch.student_loss
            critic_loss = training_batch.critic_loss
            grad_norm = training_batch.grad_norm

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "total_loss": f"{total_loss:.4f}",
                "student_loss": f"{student_loss:.4f}",
                "critic_loss": f"{critic_loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            })
            progress_bar.update(1)

            if self.global_rank == 0:
                # Prepare logging data
                log_data = {
                    "train_total_loss": total_loss,
                    "train_student_loss": student_loss,
                    "train_critic_loss": critic_loss,
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                }
                
                # Add DMD training metrics if available
                if hasattr(training_batch, 'dmd_log_dict') and training_batch.dmd_log_dict:
                    dmd_metrics = {
                        "dmd_gradient_norm": training_batch.dmd_log_dict.get("dmdtrain_gradient_norm", 0.0),
                        "dmd_timestep": training_batch.dmd_log_dict.get("timestep", 0.0).mean().item() if isinstance(training_batch.dmd_log_dict.get("timestep"), torch.Tensor) else 0.0,
                    }
                    log_data.update(dmd_metrics)
                
                # Add critic training metrics if available
                if hasattr(training_batch, 'critic_log_dict') and training_batch.critic_log_dict:
                    critic_metrics = {
                        "critic_timestep": training_batch.critic_log_dict.get("critic_timestep", 0.0).mean().item() if isinstance(training_batch.critic_log_dict.get("critic_timestep"), torch.Tensor) else 0.0,
                    }
                    log_data.update(critic_metrics)
                
                wandb.log(log_data, step=step)
                
            # if step % self.training_args.checkpointing_steps == 0:
            #     print("rank", self.global_rank, "save checkpoint at step", step)
            #     save_checkpoint(self.transformer, self.global_rank, #TODO(yongqi)
            #                     self.training_args.output_dir, step,
            #                     self.optimizer, self.train_dataloader,
            #                     self.lr_scheduler, self.noise_random_generator)
            #     if self.transformer:
            #         self.transformer.train()
            #     self.sp_group.barrier()
                
            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
                logger.info("GPU memory usage before validation: %s MB",
                            gpu_memory_usage)
                self.add_visualization(training_batch.dmd_log_dict, training_batch.critic_log_dict, self.training_args, step)
                self._log_validation(self.transformer, self.training_args, step)
                gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
                logger.info("GPU memory usage after validation: %s MB",
                            gpu_memory_usage)

        wandb.finish()
        save_checkpoint(self.transformer, self.global_rank,
                        self.training_args.output_dir,
                        self.training_args.max_train_steps, self.optimizer,
                        self.train_dataloader, self.lr_scheduler,
                        self.noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()

class FlowPredLoss():
    def __call__(
        self, x: torch.Tensor,
        noise: torch.Tensor,
        flow_pred: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean((flow_pred - (noise - x)) ** 2)
