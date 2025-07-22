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
                                      get_local_torch_device, get_world_group, get_sp_parallel_rank, get_sp_world_size)
from fastvideo.v1.fastvideo_args import FastVideoArgs,TrainingArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import (ComposedPipelineBase, ForwardBatch,
                                    TrainingBatch)
from fastvideo.v1.training.training_pipeline import TrainingPipeline
from fastvideo.v1.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, get_sigmas, load_checkpoint,
    normalize_dit_input, save_checkpoint, prepare_for_saving)
from fastvideo.v1.utils import set_random_seed, is_vsa_available, maybe_download_model
from fastvideo.v1.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.v1.training.activation_checkpoint import (
    apply_activation_checkpointing)
from fastvideo.v1.dataset.validation_dataset import ValidationDataset
from fastvideo.v1.training.training_utils import DiffusionWrapper

import wandb  # isort: skip

vsa_available = is_vsa_available()

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

    def _prepare_component_model_metadata_for_loading(self, model_index: dict[str, Any], fastvideo_args: FastVideoArgs) -> dict[str, Any]:
        """
        For DMD distillation, we need to override model index to include the
        teacher and critic transformer model names.  By default, the teacher and
        critic transformer will be the same as the student transformer
        (transformer). However, if the user provides
        training_args.override_teacher_dit_model_name_or_path and
        training_args.override_critic_dit_model_name_or_path, we will override
        the teacher and critic transformer model names.
        """
        # some sanity checks
        assert len(
            model_index
        ) > 1, "model_index.json must contain at least one pipeline module"

        for module_name in self.required_config_modules:
            if module_name not in model_index:
                logger.warning(
                    f"model_index.json does not contain a {module_name} module, adding {module_name} to model_index")
                if 'transformer' in module_name:
                    model_index[module_name] = model_index['transformer']

        component_model_metadata= {}
        for module_name, (transformers_or_diffusers, architecture) in model_index.items():
            if 'transformer' in module_name:
                loading_module_name = module_name.split("_")[-1]
            else:
                loading_module_name = module_name
            component_model_path = os.path.join(self.model_path, loading_module_name)
            if not os.path.exists(component_model_path):
                raise ValueError(
                    f"Component model path {component_model_path} does not exist")
            component_model_metadata[module_name] = (transformers_or_diffusers, architecture, component_model_path)

        if fastvideo_args.override_teacher_dit_model_name_or_path:
            # we accept the following formats:
            # 1. a HF string to a diffusers model
            # 5. a local path to a directory for DiT model
            # 6. a local path to a diffusers model

            try:
                local_path = maybe_download_model(fastvideo_args.override_teacher_dit_model_name_or_path)
            except ValueError as e:
                raise ValueError(f"Failed to download teacher model {fastvideo_args.override_teacher_dit_model_name_or_path}: {e}")
            
            config_json_path = os.path.join(local_path, "config.json")
            model_index_json_path = os.path.join(local_path, "model_index.json")

            if os.path.exists(model_index_json_path):
                # This is a diffusers model directory
                logger.info(f"Teacher model path {local_path} contains model_index.json - treating as diffusers model")
                teacher_dit_path = os.path.join(local_path, "transformer")
            elif os.path.exists(config_json_path):
                # This is a DiT model directory
                logger.info(f"Teacher model path {local_path} contains config.json - treating as DiT model")
                teacher_dit_path = local_path
            else:
                raise ValueError(f"Teacher model directory {local_path} must contain either config.json or model_index.json")
            
            logger.info(f"Overriding teacher model path to {teacher_dit_path}")
            component_model_metadata['teacher_transformer'] = (component_model_metadata['transformer'][0], component_model_metadata['transformer'][1], teacher_dit_path)
        
        if fastvideo_args.override_critic_dit_model_name_or_path:
            try:
                local_path = maybe_download_model(fastvideo_args.override_critic_dit_model_name_or_path)
            except ValueError as e:
                raise ValueError(f"Failed to download critic model {fastvideo_args.override_critic_dit_model_name_or_path}: {e}")
            
            config_json_path = os.path.join(local_path, "config.json")
            model_index_json_path = os.path.join(local_path, "model_index.json")

            if os.path.exists(model_index_json_path):
                # This is a diffusers model directory
                logger.info(f"Critic model path {local_path} contains model_index.json - treating as diffusers model")
                critic_dit_path = os.path.join(local_path, "transformer")
            elif os.path.exists(config_json_path):
                # This is a DiT model directory
                logger.info(f"Critic model path {local_path} contains config.json - treating as DiT model")
                critic_dit_path = local_path
            else:
                raise ValueError(f"Critic model directory {local_path} must contain either config.json or model_index.json")
            
            logger.info(f"Overriding critic model path to {critic_dit_path}")
            component_model_metadata['critic_transformer'] = (component_model_metadata['transformer'][0], component_model_metadata['transformer'][1], critic_dit_path)

        return component_model_metadata

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
        
        self.timestep_shift = self.training_args.pipeline_config.flow_shift
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=self.timestep_shift)
        
        # 2. Distillation-specific initialization
        # The parent class already sets self.transformer as the student model
        self.student_transformer = DiffusionWrapper(self.transformer, self.noise_scheduler)
        self.teacher_transformer = DiffusionWrapper(self.get_module("teacher_transformer"), self.noise_scheduler)
        self.critic_transformer = DiffusionWrapper(self.get_module("critic_transformer"), self.noise_scheduler)
        # self.student_transformer.to(torch.bfloat16)
        # self.teacher_transformer.to(torch.bfloat16)
        # self.critic_transformer.to(torch.bfloat16)
        # torch.distributed.breakpoint()
        self.teacher_transformer.requires_grad_(False)
        self.teacher_transformer.eval()
        self.critic_transformer.requires_grad_(True)
        self.critic_transformer.train()

        if training_args.enable_gradient_checkpointing_type is not None:
            self.critic_transformer = apply_activation_checkpointing(
                self.critic_transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type)
            self.teacher_transformer = apply_activation_checkpointing(
                self.teacher_transformer,
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
            self.training_args.denoising_step_list, dtype=torch.long, device=get_local_torch_device())
        logger.info(f"Distillation student model to {len(self.denoising_step_list)} denoising steps")
        self.num_train_timestep = self.noise_scheduler.num_train_timesteps
        # TODO(yongqi): hardcode for bidirectional distillation
        self.distill_task_type = "bidirectional_video"
        self.denoising_loss_type = 'flow'
        # TODO(yongqi): hardcode for causal distillation
        self.num_frame_per_block = 3

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
        self.student_transformer.requires_grad_(True)
        self.student_transformer.train()
        self.critic_transformer.requires_grad_(True)
        self.critic_transformer.train()
         
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
            # Create a new tensor to avoid in-place operations
            new_timestep = timestep.clone()
            for index in range(timestep.shape[0]):
                new_timestep[index] = timestep[index, 0]
            return new_timestep
        elif type == "causal_video":
            # make the noise level the same within every motion block
            timestep = timestep.reshape(
                timestep.shape[0], -1, self.num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
        else:
            raise NotImplementedError("Unsupported model type {}".format(type))

    def _student_forward(self, training_batch: TrainingBatch) -> torch.Tensor:
        """Forward pass through student transformer and compute student losses."""
        latents = training_batch.latents
        dtype = latents.dtype
        simulated_noisy_input = []
        for timestep in self.denoising_step_list:
            # Use cross-codebase generator for reproducible noise generation
            noise = torch.randn(
                self.video_latent_shape, device=self.device, dtype=dtype)
            if self.sp_world_size > 1:
                noise = rearrange(noise,
                                    "b (n t) c h w -> b n t c h w",
                                    n=self.sp_world_size).contiguous()
                noise = noise[:, self.rank_in_sp_group, :, :, :, :]

            noisy_timestep = timestep * torch.ones(
                self.video_latent_shape_sp[:2], device=self.device, dtype=torch.long)

            if timestep != 0:
                noisy_video = self.noise_scheduler.add_noise(
                    latents.flatten(0, 1),
                    noise.flatten(0, 1),
                    noisy_timestep.flatten(0, 1)
                ).unflatten(0, self.video_latent_shape_sp[:2])
            else:
                noisy_video = latents

            simulated_noisy_input.append(noisy_video)

        simulated_noisy_input = torch.stack(simulated_noisy_input, dim=1)

        # Step 2: Randomly sample a timestep and pick the corresponding input
        # Use cross-codebase generator for reproducible index generation
        index = torch.randint(0, len(self.denoising_step_list), [
                              self.video_latent_shape_sp[0], self.video_latent_shape_sp[1]], device=self.device, dtype=torch.long)

        index = self._process_timestep(index, type=self.distill_task_type)

        # select the corresponding timestep's noisy input from the stacked tensor [B, T, F, C, H, W]

        noisy_input = torch.gather(
            simulated_noisy_input, dim=1,
            index=index.reshape(index.shape[0], 1, index.shape[1], 1, 1, 1).expand(
                -1, -1, -1, *self.video_latent_shape_sp[2:])
        ).squeeze(1)

        timestep = self.denoising_step_list[index]

        training_batch = self._build_input_kwargs(noisy_input, timestep, training_batch.conditional_dict, training_batch)

        pred_video = self.student_transformer(training_batch, timestep)

        pred_video = pred_video.type_as(noisy_input)
        return pred_video, timestep.float().detach()

    def _compute_kl_grad(
        self, noisy_video: torch.Tensor,
        estimated_clean_video: torch.Tensor,
        timestep: torch.Tensor,
        training_batch: TrainingBatch,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        # critic_transformer forward
        training_batch = self._build_input_kwargs(noisy_video, timestep, training_batch.conditional_dict, training_batch)
        pred_fake_video = self.critic_transformer(training_batch, timestep)
        
        # teacher_transformer cond forward
        training_batch = self._build_input_kwargs(noisy_video, timestep, training_batch.conditional_dict, training_batch)
        pred_real_video_cond = self.teacher_transformer(training_batch, timestep)
        
        # teacher_transformer uncond forward
        training_batch = self._build_input_kwargs(noisy_video, timestep, training_batch.unconditional_dict, training_batch)
        pred_real_video_uncond = self.teacher_transformer(training_batch, timestep)
        
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

    def _compute_dmd_loss(self, pred_video: torch.Tensor, training_batch: TrainingBatch) -> Tuple[torch.Tensor, dict]:
        """Compute DMD (Diffusion Model Distillation) loss."""
        
        original_latent = pred_video
        batch_size, latent_t = self.video_latent_shape_sp[:2]
        with torch.no_grad():
            # Use cross-codebase generator for reproducible timestep generation
            timestep = torch.randint(
                0,
                self.num_train_timestep,
                [batch_size, latent_t],
                device=self.device,
                dtype=torch.long
            )

            timestep = self._process_timestep(
                timestep, type=self.distill_task_type)

            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / self.num_train_timestep) / \
                    (1 + (self.timestep_shift - 1) * (timestep / self.num_train_timestep)) * self.num_train_timestep
            
            timestep = timestep.clamp(self.min_step, self.max_step)

            # Use cross-codebase generator for reproducible noise generation
            noise = torch.randn(
                self.video_latent_shape, device=self.device, dtype=pred_video.dtype)
            if self.sp_world_size > 1:
                noise = rearrange(noise,
                                    "b (n t) c h w -> b n t c h w",
                                    n=self.sp_world_size).contiguous()
                noise = noise[:, self.rank_in_sp_group, :, :, :, :]
                
            noisy_latent = self.noise_scheduler.add_noise(
                pred_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, latent_t))

            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_video=noisy_latent,
                estimated_clean_video=original_latent,
                timestep=timestep,
                training_batch=training_batch
            )

        dmd_loss = 0.5 * F.mse_loss(original_latent.double(
        ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        
        return dmd_loss, dmd_log_dict

    def _student_forward_and_compute_dmd_loss(self, training_batch: TrainingBatch) -> Tuple[TrainingBatch, torch.Tensor, dict]:
        """Forward pass through student transformer and compute student losses."""
        with set_forward_context(
                current_timestep=training_batch.timesteps, attn_metadata=training_batch.attn_metadata_vsa):
            pred_video, timestep_dmd = self._student_forward(training_batch)

        with set_forward_context(
                current_timestep=training_batch.timesteps, attn_metadata=training_batch.attn_metadata):
            dmd_loss, dmd_log_dict = self._compute_dmd_loss(
                pred_video=pred_video,
                training_batch=training_batch
            )

        dmd_log_dict['dmd_timestep_stu'] = timestep_dmd

        return training_batch, dmd_loss, dmd_log_dict

    def _critic_forward_and_compute_loss(self, training_batch: TrainingBatch) -> Tuple[TrainingBatch, torch.Tensor, dict]:
        with torch.no_grad():
            with set_forward_context(
                current_timestep=training_batch.timesteps, attn_metadata=training_batch.attn_metadata_vsa):
                generated_video, timestep_gen = self._student_forward(training_batch)
        
        critic_timestep = torch.randint(
            0,
            self.num_train_timestep,
            self.video_latent_shape_sp[:2],
            device=self.device,
            dtype=torch.long
        )
        critic_timestep = self._process_timestep(
            critic_timestep, type=self.distill_task_type)

        # TODO: Add timestep warping
        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / self.num_train_timestep) / (1 + (self.timestep_shift - 1) * (critic_timestep / self.num_train_timestep)) * self.num_train_timestep

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        # Use cross-codebase generator for reproducible noise generation
        critic_noise = torch.randn(
            self.video_latent_shape, device=self.device, dtype=generated_video.dtype)
        if self.sp_world_size > 1:
            critic_noise = rearrange(critic_noise,
                                    "b (n t) c h w -> b n t c h w",
                                    n=self.sp_world_size).contiguous()
            critic_noise = critic_noise[:, self.rank_in_sp_group, :, :, :, :]
        
        noisy_generated_video = self.noise_scheduler.add_noise(
            generated_video.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, self.video_latent_shape_sp[:2])

        with set_forward_context(
                current_timestep=training_batch.timesteps, attn_metadata=training_batch.attn_metadata):
            training_batch = self._build_input_kwargs(noisy_generated_video, critic_timestep, training_batch.conditional_dict, training_batch)
            
            pred_fake_video = self.critic_transformer(training_batch, critic_timestep)

        # # Step 3: Compute the denoising loss for the fake critic
        pred_fake_video_noise = DiffusionWrapper._convert_x0_to_flow_pred(
            x0_pred=pred_fake_video.flatten(0, 1),
            xt=noisy_generated_video.flatten(0, 1),
            timestep=critic_timestep.flatten(0, 1),
            scheduler=self.noise_scheduler
        )

        denoising_loss = self.denoising_loss_func(
            x=generated_video.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            flow_pred=pred_fake_video_noise
        )

        critic_log_dict = {
            "critictrain_latent": generated_video.detach(),
            "critictrain_noisy_latent": noisy_generated_video.detach(),
            "critictrain_pred_video": pred_fake_video.detach(),
            "critic_timestep": critic_timestep.float().detach(),
            "critic_timestep_stu": timestep_gen.float().detach(),
        }

        return training_batch, denoising_loss, critic_log_dict
        
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
        training_batch.latents = training_batch.latents.permute(0, 2, 1, 3, 4)
        self.video_latent_shape = training_batch.latents.shape # [B, C, T, H, W]
        training_batch.raw_latent_shape = training_batch.latents.shape

        if self.sp_world_size > 1:
            training_batch.latents = rearrange(training_batch.latents,
                                "b (n t) c h w -> b n t c h w",
                                n=self.sp_world_size).contiguous()
            training_batch.latents = training_batch.latents[:, self.rank_in_sp_group, :, :, :, :]

        self.video_latent_shape_sp = training_batch.latents.shape

        return training_batch

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Train one step with alternating student and critic updates, supporting gradient accumulation."""
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

        # Student accumulation
        self.optimizer.zero_grad()
        total_dmd_loss = 0.0
        total_dmd_log_dict = None
        if (self.current_trainstep % self.student_critic_update_ratio == 0):
            for batch in batches:
                batch_stu = copy.deepcopy(batch)
                batch_stu, dmd_loss, dmd_log_dict = self._student_forward_and_compute_dmd_loss(batch_stu)
                # Ensure backward is under the correct forward context
                with set_forward_context(
                    current_timestep=batch_stu.timesteps, attn_metadata=batch_stu.attn_metadata):
                    (dmd_loss / gradient_accumulation_steps).backward()
                total_dmd_loss += dmd_loss.detach().item()
                if total_dmd_log_dict is None:
                    total_dmd_log_dict = dmd_log_dict
                # Only keep the first log dict, ignore subsequent ones
            self._clip_grad_norm(batch_stu, self.student_transformer)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            avg_dmd_loss = torch.tensor(total_dmd_loss / gradient_accumulation_steps, device=self.device)
            world_group = get_world_group()
            world_group.all_reduce(avg_dmd_loss, op=torch.distributed.ReduceOp.AVG)
            training_batch.student_loss = avg_dmd_loss.item()
            training_batch.dmd_log_dict = total_dmd_log_dict if total_dmd_log_dict is not None else {}
        else:
            training_batch.student_loss = 0.0
            training_batch.dmd_log_dict = {}

        # Critic accumulation
        self.critic_transformer_optimizer.zero_grad()
        total_critic_loss = 0.0
        total_critic_log_dict = None
        for batch in batches:
            batch_critic = copy.deepcopy(batch)
            batch_critic, critic_loss, critic_log_dict = self._critic_forward_and_compute_loss(batch_critic)
            with set_forward_context(
                current_timestep=batch_critic.timesteps, attn_metadata=batch_critic.attn_metadata):
                (critic_loss / gradient_accumulation_steps).backward()
            total_critic_loss += critic_loss.detach().item()
            if total_critic_log_dict is None:
                total_critic_log_dict = critic_log_dict
            # Only keep the first log dict, ignore subsequent ones
        self._clip_grad_norm(batch_critic, self.critic_transformer)
        self.critic_transformer_optimizer.step()
        self.critic_lr_scheduler.step()
        self.critic_transformer_optimizer.zero_grad(set_to_none=True)
        avg_critic_loss = torch.tensor(total_critic_loss / gradient_accumulation_steps, device=self.device)
        world_group = get_world_group()
        world_group.all_reduce(avg_critic_loss, op=torch.distributed.ReduceOp.AVG)
        training_batch.critic_loss = avg_critic_loss.item()
        training_batch.critic_log_dict = total_critic_log_dict if total_critic_log_dict is not None else {}

        training_batch.total_loss = training_batch.student_loss + training_batch.critic_loss
        return training_batch

    def _resume_from_checkpoint(self) -> None: #TODO(yongqi)
        """Resume training from checkpoint with distillation models."""
        assert self.training_args is not None
        logger.info("Loading distillation checkpoint from %s",
                    self.training_args.resume_from_checkpoint)
        
        resumed_step = load_checkpoint(
            self.student_transformer.model, self.global_rank,
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
        """Add visualization data to wandb logging and save frames to disk."""
        wandb_loss_dict = {}
        
        # Clear GPU cache before VAE decoding to prevent OOM
        torch.cuda.empty_cache()
        
        # # Use consistent decoding approach - use decode_stage for all
        # decode_stage = self.validation_pipeline._stages[-1]
        
        # Process critic training data
        critic_latents_name = ['critictrain_latent', 'critictrain_noisy_latent', 'critictrain_pred_video']
        # critic_latents_name = ['critictrain_pred_video']

        for latent_key in critic_latents_name:
            latents = critic_log_dict[latent_key] # bs, t,c, h, w
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

        # Set deterministic seed for validation
        # set_random_seed(self.seed)
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

            # #     # Run validation inference
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

        self.validation_generator = torch.Generator(device=get_local_torch_device()).manual_seed(42)

        logger.info("Initialized random seeds with seed: %s", seed)

        if self.training_args.resume_from_checkpoint:
            self._resume_from_checkpoint()

        self.train_loader_iter = iter(self.train_dataloader)

        step_times: deque[float] = deque(maxlen=100)

        self._log_training_info()
        self._log_validation(self.student_transformer, self.training_args, 0)

        progress_bar = tqdm(
            range(0, self.training_args.max_train_steps),
            initial=self.init_steps,
            desc="Steps",
            disable=self.local_rank > 0,
        )
        
        for step in range(self.init_steps + 1,
                        self.training_args.max_train_steps + 1):
            start_time = time.perf_counter()
            current_vsa_sparsity = self.training_args.VSA_sparsity if vsa_available else 0.0
            
            training_batch = TrainingBatch()
            self.current_trainstep = step
            training_batch.current_vsa_sparsity = current_vsa_sparsity

            with torch.autocast("cuda", dtype=torch.bfloat16):
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
                        "dmd_timestep": training_batch.dmd_log_dict.get("timestep", 0.0).mean().item(),
                        "dmd_timestep_stu": training_batch.dmd_log_dict.get("dmd_timestep_stu", 0.0).mean().item()
                    }
                    log_data.update(dmd_metrics)
                
                # Add critic training metrics if available
                if hasattr(training_batch, 'critic_log_dict') and training_batch.critic_log_dict:
                    critic_metrics = {
                        "critic_timestep": training_batch.critic_log_dict.get("critic_timestep", 0.0).mean().item(),
                        "critic_timestep_stu": training_batch.critic_log_dict.get("critic_timestep_stu", 0.0).mean().item(),
                    }
                    log_data.update(critic_metrics)
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


                self.add_visualization(training_batch.dmd_log_dict, training_batch.critic_log_dict, self.training_args, step)
                self._log_validation(self.student_transformer, self.training_args, step)



        wandb.finish()
        # save_checkpoint(self.student_transformer.model, self.global_rank,
        #                 self.training_args.output_dir,
        #                 self.training_args.max_train_steps, self.optimizer,
        #                 self.train_dataloader, self.lr_scheduler,
        #                 self.noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()

class FlowPredLoss():
    def __call__(
        self, x: torch.Tensor,
        noise: torch.Tensor,
        flow_pred: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean((flow_pred - (noise - x)) ** 2)
