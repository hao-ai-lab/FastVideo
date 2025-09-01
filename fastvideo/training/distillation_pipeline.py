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
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines import (ComposedPipelineBase, ForwardBatch,
                                 TrainingBatch)
from fastvideo.training.activation_checkpoint import (
    apply_activation_checkpointing)
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases, count_trainable,
    get_scheduler, load_distillation_checkpoint, save_distillation_checkpoint,
    shift_timestep)
from fastvideo.utils import is_vsa_available, set_random_seed

import wandb  # isort: skip

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class DistillationPipeline(TrainingPipeline):
    """
    A distillation pipeline for training a 3 step model.
    Inherits from TrainingPipeline to reuse training infrastructure.
    """
    _required_config_modules = [
        "scheduler", "transformer", "vae", "real_score_transformer",
        "fake_score_transformer"
    ]
    _extra_config_module_map = {
        "real_score_transformer": "transformer",
        "fake_score_transformer": "transformer"
    }
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[dict[str, Any]]
    current_epoch: int = 0
    init_steps: int
    current_trainstep: int
    video_latent_shape: tuple[int, ...]
    video_latent_shape_sp: tuple[int, ...]
    real_score_transformer: torch.nn.Module
    fake_score_transformer: torch.nn.Module

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def set_trainable(self) -> None:
        super().set_trainable()
        self.modules["real_score_transformer"].requires_grad_(False)
        self.modules["vae"].requires_grad_(False)

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the distillation training pipeline with multiple models."""
        logger.info("Initializing distillation pipeline...")

        super().initialize_training_pipeline(training_args)

        self.noise_scheduler = self.get_module("scheduler")
        self.vae = self.get_module("vae")
        self.timestep_shift = self.training_args.pipeline_config.flow_shift
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=self.timestep_shift)

        # self.transformer is the generator model
        self.real_score_transformer = self.get_module("real_score_transformer")
        self.fake_score_transformer = self.get_module("fake_score_transformer")
        self.real_score_transformer.eval()
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
        fake_score_params = list(
            filter(lambda p: p.requires_grad,
                   self.fake_score_transformer.parameters()))

        # Use separate learning rate for fake_score_transformer if specified
        fake_score_lr = training_args.fake_score_learning_rate
        if fake_score_lr == 0.0:
            fake_score_lr = training_args.learning_rate

        self.fake_score_optimizer = torch.optim.AdamW(
            fake_score_params,
            lr=fake_score_lr,
            betas=(0.9, 0.999),
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )

        self.fake_score_lr_scheduler = get_scheduler(
            training_args.fake_score_lr_scheduler,
            optimizer=self.fake_score_optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )

        logger.info(
            "Distillation optimizers initialized: generator and fake_score")

        self.generator_update_interval = self.training_args.generator_update_interval
        logger.info(
            "Distillation pipeline initialized with generator_update_interval=%s",
            self.generator_update_interval)

        self.denoising_step_list = torch.tensor(
            self.training_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long,
            device=get_local_torch_device())
        logger.info("Distillation generator model to %s denoising steps",
                    len(self.denoising_step_list))
        self.num_train_timestep = self.noise_scheduler.num_train_timesteps

        self.min_timestep = int(self.training_args.min_timestep_ratio *
                                self.num_train_timestep)
        self.max_timestep = int(self.training_args.max_timestep_ratio *
                                self.num_train_timestep)

        self.real_score_guidance_scale = self.training_args.real_score_guidance_scale

    @abstractmethod
    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Initialize validation pipeline - must be implemented by subclasses."""
        raise NotImplementedError(
            "Distillation pipelines must implement this method")

    def _prepare_distillation(self,
                              training_batch: TrainingBatch) -> TrainingBatch:
        """Prepare training environment for distillation."""
        self.transformer.train()
        self.fake_score_transformer.train()

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

        latents = training_batch.latents
        dtype = latents.dtype
        index = torch.randint(0,
                              len(self.denoising_step_list), [1],
                              device=self.device,
                              dtype=torch.long)
        timestep = self.denoising_step_list[index]
        training_batch.dmd_latent_vis_dict["generator_timestep"] = timestep

        noise = torch.randn(self.video_latent_shape,
                            device=self.device,
                            dtype=dtype)
        if self.sp_world_size > 1:
            noise = rearrange(noise,
                              "b (n t) c h w -> b n t c h w",
                              n=self.sp_world_size).contiguous()
            noise = noise[:, self.rank_in_sp_group, :, :, :, :]
        noisy_latent = self.noise_scheduler.add_noise(latents.flatten(0, 1),
                                                      noise.flatten(0, 1),
                                                      timestep).unflatten(
                                                          0,
                                                          (1, latents.shape[1]))

        training_batch = self._build_distill_input_kwargs(
            noisy_latent, timestep, training_batch.conditional_dict,
            training_batch)
        pred_noise = self.transformer(**training_batch.input_kwargs).permute(
            0, 2, 1, 3, 4)
        pred_video = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_latent.flatten(0, 1),
            timestep=timestep,
            scheduler=self.noise_scheduler).unflatten(0, pred_noise.shape[:2])

        return pred_video

    def _generator_multi_step_simulation_forward(
            self, training_batch: TrainingBatch) -> torch.Tensor:
        """Forward pass through student transformer matching inference procedure."""
        latents = training_batch.latents
        dtype = latents.dtype

        # Step 1: Randomly sample a target timestep index from denoising_step_list
        target_timestep_idx = torch.randint(0,
                                            len(self.denoising_step_list), [1],
                                            device=self.device,
                                            dtype=torch.long)
        target_timestep_idx_int = target_timestep_idx.item()
        target_timestep = self.denoising_step_list[target_timestep_idx]

        # Step 2: Simulate the multi-step inference process up to the target timestep
        # Start from pure noise like in inference
        current_noise_latents = torch.randn(self.video_latent_shape,
                                            device=self.device,
                                            dtype=dtype)
        if self.sp_world_size > 1:
            current_noise_latents = rearrange(
                current_noise_latents,
                "b (n t) c h w -> b n t c h w",
                n=self.sp_world_size).contiguous()
            current_noise_latents = current_noise_latents[:, self.
                                                          rank_in_sp_group, :, :, :, :]
        current_noise_latents_copy = current_noise_latents.clone()

        # Only run intermediate steps if target_timestep_idx > 0
        max_target_idx = len(self.denoising_step_list) - 1
        noise_latents = []
        noise_latent_index = target_timestep_idx_int - 1
        if max_target_idx > 0:
            # Run student model for all steps before the target timestep
            with torch.no_grad():
                for step_idx in range(max_target_idx):
                    current_timestep = self.denoising_step_list[step_idx]
                    current_timestep_tensor = current_timestep * torch.ones(
                        1, device=self.device, dtype=torch.long)
                    # Run student model to get flow prediction
                    training_batch_temp = self._build_distill_input_kwargs(
                        current_noise_latents, current_timestep_tensor,
                        training_batch.conditional_dict, training_batch)
                    pred_flow = self.transformer(
                        **training_batch_temp.input_kwargs).permute(
                            0, 2, 1, 3, 4)
                    pred_clean = pred_noise_to_pred_video(
                        pred_noise=pred_flow.flatten(0, 1),
                        noise_input_latent=current_noise_latents.flatten(0, 1),
                        timestep=current_timestep_tensor,
                        scheduler=self.noise_scheduler).unflatten(
                            0, pred_flow.shape[:2])

                    # Add noise for the next timestep
                    next_timestep = self.denoising_step_list[step_idx + 1]
                    next_timestep_tensor = next_timestep * torch.ones(
                        1, device=self.device, dtype=torch.long)
                    noise = torch.randn(self.video_latent_shape,
                                        device=self.device,
                                        dtype=pred_clean.dtype)
                    if self.sp_world_size > 1:
                        noise = rearrange(noise,
                                          "b (n t) c h w -> b n t c h w",
                                          n=self.sp_world_size).contiguous()
                        noise = noise[:, self.rank_in_sp_group, :, :, :, :]
                    current_noise_latents = self.noise_scheduler.add_noise(
                        pred_clean.flatten(0, 1), noise.flatten(0, 1),
                        next_timestep_tensor).unflatten(0, pred_clean.shape[:2])
                    latent_copy = current_noise_latents.clone()
                    noise_latents.append(latent_copy)

        # Step 3: Use the simulated noisy input for the final training step
        # For timestep index 0, this is pure noise
        # For timestep index k > 0, this is the result after k denoising steps + noise at target level
        if noise_latent_index >= 0:
            assert noise_latent_index < len(
                self.denoising_step_list
            ) - 1, "noise_latent_index is out of bounds"
            noisy_input = noise_latents[noise_latent_index]
        else:
            noisy_input = current_noise_latents_copy

        # Step 4: Final student prediction (this is what we train on)
        training_batch = self._build_distill_input_kwargs(
            noisy_input, target_timestep, training_batch.conditional_dict,
            training_batch)
        pred_noise = self.transformer(**training_batch.input_kwargs).permute(
            0, 2, 1, 3, 4)
        pred_video = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_input.flatten(0, 1),
            timestep=target_timestep,
            scheduler=self.noise_scheduler).unflatten(0, pred_noise.shape[:2])
        training_batch.dmd_latent_vis_dict[
            "generator_timestep"] = target_timestep.float().detach()
        return pred_video

    def _dmd_forward(self, generator_pred_video: torch.Tensor,
                     training_batch: TrainingBatch) -> torch.Tensor:
        """Compute DMD (Diffusion Model Distillation) loss."""
        with torch.no_grad():
            timestep = torch.randint(0,
                                     self.num_train_timestep, [1],
                                     device=self.device,
                                     dtype=torch.long)

            timestep = shift_timestep(
                timestep,
                self.timestep_shift,  # type: ignore
                self.num_train_timestep)

            timestep = timestep.clamp(self.min_timestep, self.max_timestep)

            noise = torch.randn(self.video_latent_shape,
                                device=self.device,
                                dtype=generator_pred_video.dtype)
            if self.sp_world_size > 1:
                noise = rearrange(noise,
                                  "b (n t) c h w -> b n t c h w",
                                  n=self.sp_world_size).contiguous()
                noise = noise[:, self.rank_in_sp_group, :, :, :, :]

            noisy_latent = self.noise_scheduler.add_noise(
                generator_pred_video.flatten(0, 1), noise.flatten(0, 1),
                timestep).unflatten(0, (1, generator_pred_video.shape[1]))

            # fake_score_transformer forward
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.conditional_dict,
                training_batch)
            fake_score_pred_noise = self.fake_score_transformer(
                **training_batch.input_kwargs).permute(0, 2, 1, 3, 4)

            faker_score_pred_video = pred_noise_to_pred_video(
                pred_noise=fake_score_pred_noise.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, fake_score_pred_noise.shape[:2])

            # real_score_transformer cond forward
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.conditional_dict,
                training_batch)
            real_score_pred_noise_cond = self.real_score_transformer(
                **training_batch.input_kwargs).permute(0, 2, 1, 3, 4)

            pred_real_video_cond = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise_cond.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise_cond.shape[:2])

            # real_score_transformer uncond forward
            training_batch = self._build_distill_input_kwargs(
                noisy_latent, timestep, training_batch.unconditional_dict,
                training_batch)
            real_score_pred_noise_uncond = self.real_score_transformer(
                **training_batch.input_kwargs).permute(0, 2, 1, 3, 4)

            pred_real_video_uncond = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise_uncond.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise_uncond.shape[:2])

            real_score_pred_video = pred_real_video_cond + (
                pred_real_video_cond -
                pred_real_video_uncond) * self.real_score_guidance_scale

            grad = (faker_score_pred_video - real_score_pred_video) / torch.abs(
                generator_pred_video - real_score_pred_video).mean()
            grad = torch.nan_to_num(grad)

        dmd_loss = 0.5 * F.mse_loss(
            generator_pred_video.float(),
            (generator_pred_video.float() - grad.float()).detach())

        training_batch.dmd_latent_vis_dict.update({
            "training_batch_dmd_fwd_clean_latent":
            training_batch.latents,
            "generator_pred_video":
            generator_pred_video,
            "real_score_pred_video":
            real_score_pred_video,
            "faker_score_pred_video":
            faker_score_pred_video,
            "dmd_timestep":
            timestep,
        })

        return dmd_loss

    def faker_score_forward(
            self, training_batch: TrainingBatch
    ) -> tuple[TrainingBatch, torch.Tensor]:
        with torch.no_grad(), set_forward_context(
                current_timestep=training_batch.timesteps,
                attn_metadata=training_batch.attn_metadata_vsa):
            if self.training_args.simulate_generator_forward:
                generator_pred_video = self._generator_multi_step_simulation_forward(
                    training_batch)
            else:
                generator_pred_video = self._generator_forward(training_batch)

        fake_score_timestep = torch.randint(0,
                                            self.num_train_timestep, [1],
                                            device=self.device,
                                            dtype=torch.long)

        fake_score_timestep = shift_timestep(
            fake_score_timestep,
            self.timestep_shift,  # type: ignore
            self.num_train_timestep)

        fake_score_timestep = fake_score_timestep.clamp(self.min_timestep,
                                                        self.max_timestep)

        fake_score_noise = torch.randn(self.video_latent_shape,
                                       device=self.device,
                                       dtype=generator_pred_video.dtype)
        if self.sp_world_size > 1:
            fake_score_noise = rearrange(fake_score_noise,
                                         "b (n t) c h w -> b n t c h w",
                                         n=self.sp_world_size).contiguous()
            fake_score_noise = fake_score_noise[:, self.
                                                rank_in_sp_group, :, :, :, :]

        noisy_generator_pred_video = self.noise_scheduler.add_noise(
            generator_pred_video.flatten(0, 1), fake_score_noise.flatten(0, 1),
            fake_score_timestep).unflatten(0,
                                           (1, generator_pred_video.shape[1]))

        with set_forward_context(current_timestep=training_batch.timesteps,
                                 attn_metadata=training_batch.attn_metadata):
            training_batch = self._build_distill_input_kwargs(
                noisy_generator_pred_video, fake_score_timestep,
                training_batch.conditional_dict, training_batch)

            fake_score_pred_noise = self.fake_score_transformer(
                **training_batch.input_kwargs).permute(0, 2, 1, 3, 4)

        target = fake_score_noise - generator_pred_video
        flow_matching_loss = torch.mean((fake_score_pred_noise - target)**2)

        training_batch.fake_score_latent_vis_dict = {
            "training_batch_fakerscore_fwd_clean_latent":
            training_batch.latents,
            "generator_pred_video": generator_pred_video,
            "fake_score_timestep": fake_score_timestep,
        }

        return training_batch, flow_matching_loss

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
        gradient_accumulation_steps = getattr(self.training_args,
                                              'gradient_accumulation_steps', 1)
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
                batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore
            batches.append(batch)

        self.optimizer.zero_grad()
        total_dmd_loss = 0.0
        dmd_latent_vis_dict = {}
        fake_score_latent_vis_dict = {}
        if (self.current_trainstep % self.generator_update_interval == 0):
            for batch in batches:
                batch_gen = copy.deepcopy(batch)

                with set_forward_context(
                        current_timestep=batch_gen.timesteps,
                        attn_metadata=batch_gen.attn_metadata_vsa):
                    if self.training_args.simulate_generator_forward:
                        generator_pred_video = self._generator_multi_step_simulation_forward(
                            batch_gen)
                    else:
                        generator_pred_video = self._generator_forward(
                            batch_gen)

                with set_forward_context(current_timestep=batch_gen.timesteps,
                                         attn_metadata=batch_gen.attn_metadata):
                    dmd_loss = self._dmd_forward(
                        generator_pred_video=generator_pred_video,
                        training_batch=batch_gen)

                with set_forward_context(
                        current_timestep=batch_gen.timesteps,
                        attn_metadata=batch_gen.attn_metadata_vsa):
                    (dmd_loss / gradient_accumulation_steps).backward()
                total_dmd_loss += dmd_loss.detach().item()
            self._clip_model_grad_norm_(batch_gen, self.transformer)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            avg_dmd_loss = torch.tensor(total_dmd_loss /
                                        gradient_accumulation_steps,
                                        device=self.device)
            world_group = get_world_group()
            world_group.all_reduce(avg_dmd_loss,
                                   op=torch.distributed.ReduceOp.AVG)
            training_batch.generator_loss = avg_dmd_loss.item()
            dmd_latent_vis_dict = batch_gen.dmd_latent_vis_dict
        else:
            training_batch.generator_loss = 0.0

        self.fake_score_optimizer.zero_grad()
        total_fake_score_loss = 0.0
        for batch in batches:
            batch_fake = copy.deepcopy(batch)
            batch_fake, fake_score_loss = self.faker_score_forward(batch_fake)
            with set_forward_context(current_timestep=batch_fake.timesteps,
                                     attn_metadata=batch_fake.attn_metadata):
                (fake_score_loss / gradient_accumulation_steps).backward()
            total_fake_score_loss += fake_score_loss.detach().item()
            fake_score_latent_vis_dict.update(
                batch_fake.fake_score_latent_vis_dict)
        self._clip_model_grad_norm_(batch_fake, self.fake_score_transformer)
        self.fake_score_optimizer.step()
        self.fake_score_lr_scheduler.step()
        self.lr_scheduler.step()
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        avg_fake_score_loss = torch.tensor(total_fake_score_loss /
                                           gradient_accumulation_steps,
                                           device=self.device)
        world_group = get_world_group()
        world_group.all_reduce(avg_fake_score_loss,
                               op=torch.distributed.ReduceOp.AVG)
        training_batch.fake_score_loss = avg_fake_score_loss.item()
        training_batch.dmd_latent_vis_dict = dmd_latent_vis_dict
        training_batch.fake_score_latent_vis_dict = batch_fake.fake_score_latent_vis_dict

        training_batch.total_loss = training_batch.generator_loss + training_batch.fake_score_loss
        return training_batch

    def _resume_from_checkpoint(self) -> None:
        """Resume training from checkpoint with distillation models."""

        logger.info("Loading distillation checkpoint from %s",
                    self.training_args.resume_from_checkpoint)

        resumed_step = load_distillation_checkpoint(
            self.transformer, self.fake_score_transformer, self.global_rank,
            self.training_args.resume_from_checkpoint, self.optimizer,
            self.fake_score_optimizer, self.train_dataloader, self.lr_scheduler,
            self.fake_score_lr_scheduler, self.noise_random_generator)

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
        logger.info("  Generator update ratio: %s",
                    self.generator_update_interval)
        assert isinstance(self.training_args, TrainingArgs)
        logger.info("  Max gradient norm: %s", self.training_args.max_grad_norm)

        logger.info(
            "  Real score transformer parameters: %s B",
            sum(p.numel()
                for p in self.real_score_transformer.parameters()) / 1e9)

        logger.info(
            "  Fake score transformer parameters: %s B",
            sum(p.numel()
                for p in self.fake_score_transformer.parameters()) / 1e9)

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
        fake_score_latents_vis_dict = training_batch.fake_score_latent_vis_dict
        fake_score_log_keys = ['generator_pred_video']
        dmd_log_keys = ['faker_score_pred_video', 'real_score_pred_video']

        for latent_key in fake_score_log_keys:
            latents = fake_score_latents_vis_dict[latent_key]
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
            wandb_loss_dict[latent_key] = wandb.Video(
                video, fps=24, format="mp4")  # change to 16 for Wan2.1
            # Clean up references
            del video, latents

        # Process DMD training data if available - use decode_stage instead of self.vae.decode
        if 'generator_pred_video' in dmd_latents_vis_dict:
            for latent_key in dmd_log_keys:
                latents = dmd_latents_vis_dict[latent_key]
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
                        latents += self.vae.shift_factor.to(
                            latents.device, latents.dtype)
                    else:
                        latents += self.vae.shift_factor
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    video = self.vae.decode(latents)
                video = (video / 2 + 0.5).clamp(0, 1)
                video = video.cpu().float()
                video = video.permute(0, 2, 1, 3, 4)
                video = (video * 255).numpy().astype(np.uint8)
                wandb_loss_dict[latent_key] = wandb.Video(
                    video, fps=24, format="mp4")  # change to 16 for Wan2.1
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

        # Check trainable params
        num_trainable_generator = round(
            count_trainable(self.transformer) / 1e9, 3)
        num_trainable_critic = round(
            count_trainable(self.fake_score_transformer) / 1e9, 3)
        logger.info(
            "rank: %s: # of trainable params in generator: %sB, # of trainable params in critic: %sB",
            self.global_rank, num_trainable_generator, num_trainable_critic)
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
                    "train_total_loss":
                    total_loss,
                    "train_fake_score_loss":
                    fake_score_loss,
                    "learning_rate":
                    self.lr_scheduler.get_last_lr()[0],
                    "fake_score_learning_rate":
                    self.fake_score_lr_scheduler.get_last_lr()[0],
                    "step_time":
                    step_time,
                    "avg_step_time":
                    avg_step_time,
                    "grad_norm":
                    grad_norm,
                }
                # Only log generator loss when generator is actually trained
                if (step % self.generator_update_interval == 0):
                    log_data["train_generator_loss"] = generator_loss
                if use_vsa:
                    log_data["VSA_train_sparsity"] = current_vsa_sparsity

                if training_batch.dmd_latent_vis_dict:
                    dmd_additional_logs = {
                        "generator_timestep":
                        training_batch.
                        dmd_latent_vis_dict["generator_timestep"].item(),
                        "dmd_timestep":
                        training_batch.dmd_latent_vis_dict["dmd_timestep"].item(
                        ),
                    }
                    log_data.update(dmd_additional_logs)

                faker_score_additional_logs = {
                    "fake_score_timestep":
                    training_batch.
                    fake_score_latent_vis_dict["fake_score_timestep"].item(),
                }
                log_data.update(faker_score_additional_logs)

                wandb.log(log_data, step=step)

            # Save training state checkpoint (for resuming training)
            if (self.training_args.training_state_checkpointing_steps > 0
                    and step %
                    self.training_args.training_state_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save training state checkpoint at step", step)
                save_distillation_checkpoint(
                    self.transformer, self.fake_score_transformer,
                    self.global_rank, self.training_args.output_dir, step,
                    self.optimizer, self.fake_score_optimizer,
                    self.train_dataloader, self.lr_scheduler,
                    self.fake_score_lr_scheduler, self.noise_random_generator)

                if self.transformer:
                    self.transformer.train()
                self.sp_group.barrier()

            # Save weight-only checkpoint
            if (self.training_args.weight_only_checkpointing_steps > 0
                    and step %
                    self.training_args.weight_only_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save weight-only checkpoint at step", step)
                save_distillation_checkpoint(self.transformer,
                                             self.fake_score_transformer,
                                             self.global_rank,
                                             self.training_args.output_dir,
                                             f"{step}_weight_only",
                                             only_save_generator_weight=True)

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
        save_distillation_checkpoint(
            self.transformer, self.fake_score_transformer, self.global_rank,
            self.training_args.output_dir, self.training_args.max_train_steps,
            self.optimizer, self.fake_score_optimizer, self.train_dataloader,
            self.lr_scheduler, self.fake_score_lr_scheduler,
            self.noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()
