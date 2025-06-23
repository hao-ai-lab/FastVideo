# SPDX-License-Identifier: Apache-2.0
import gc
import math
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Union

import imageio
import numpy as np
import torch
import torchvision
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
import copy

import fastvideo.v1.envs as envs
from fastvideo.v1.attention.backends.video_sparse_attn import (
    VideoSparseAttentionMetadata)
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset import build_parquet_map_style_dataloader
from fastvideo.v1.distributed import (cleanup_dist_env_and_memory, get_sp_group,
                                      get_torch_device, get_world_group)
from fastvideo.v1.fastvideo_args import FastVideoArgs, DistillationArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import (ComposedPipelineBase, ForwardBatch,
                                    TrainingBatch)
from fastvideo.v1.training.training_pipeline import TrainingPipeline
from fastvideo.v1.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, get_sigmas, load_checkpoint,
    normalize_dit_input, save_checkpoint, shard_latents_across_sp)
from fastvideo.v1.utils import set_random_seed

import wandb  # isort: skip

logger = init_logger(__name__)


class DistillationPipeline(TrainingPipeline):
    """
    A distillation pipeline for training a student model using teacher model guidance.
    Inherits from TrainingPipeline to reuse training infrastructure.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      Dict[str, Any]]]
    current_epoch: int = 0

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def initialize_training_pipeline(self, distillation_args: DistillationArgs):
        """Initialize the distillation training pipeline with multiple models."""
        logger.info("Initializing distillation training pipeline...")
        
        # 1. Call parent initialization first
        super().initialize_training_pipeline(distillation_args)
        assert isinstance(self.training_args, DistillationArgs)
        
        # 2. Distillation-specific initialization
        # The parent class already sets self.transformer as the student model
        self.teacher_transformer = self._copy_model(self.transformer)
        assert self.teacher_transformer is not None, "Failed to create teacher model"
        self.critic_transformer = self._copy_model(self.transformer)
        assert self.critic_transformer is not None, "Failed to create critic model"
        self.teacher_transformer.requires_grad_(False)
        self.teacher_transformer.eval()
        self.critic_transformer.requires_grad_(True)
        self.critic_transformer.train()

        # Initialize optimizers
        student_params = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        self.student_transformer_optimizer = torch.optim.AdamW(
            student_params,
            lr=distillation_args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=distillation_args.weight_decay,
            eps=1e-8,
        )
        critic_params = list(filter(lambda p: p.requires_grad, self.critic_transformer.parameters()))
        self.critic_transformer_optimizer = torch.optim.AdamW(
            critic_params,
            lr=distillation_args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=distillation_args.weight_decay,
            eps=1e-8,
        )
        logger.info("Distillation optimizers initialized: student and critic")
        self.student_critic_update_ratio = self.training_args.student_critic_update_ratio
        logger.info(f"Distillation pipeline initialized with student_critic_update_ratio={self.student_critic_update_ratio}")

    @abstractmethod
    def initialize_validation_pipeline(self, distillation_args: DistillationArgs):
        """Initialize validation pipeline - must be implemented by subclasses."""
        raise NotImplementedError(
            "Distillation pipelines must implement this method") 

    def _copy_model(self, original_model):
        """Create a copy of a model using state dict for FSDP2 compatibility."""
        state_dict = original_model.state_dict()
        copied_model = type(original_model)()
        copied_model.load_state_dict(state_dict)
        return copied_model

    def _prepare_training(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Prepare training environment for distillation."""
        self.transformer.requires_grad_(True)
        self.transformer.train()
        self.critic_transformer.requires_grad_(True)
        self.critic_transformer.train()
        self.student_transformer_optimizer.zero_grad()
        self.critic_transformer_optimizer.zero_grad()
        
        training_batch.total_loss = 0.0
        setattr(training_batch, 'student_loss', 0.0)
        setattr(training_batch, 'critic_loss', 0.0)
        
        return training_batch

    def _compute_dmd_loss(self, training_batch) -> torch.Tensor:
        """Compute DMD (Diffusion Model Distillation) loss."""
        assert training_batch.latents is not None
        model_pred = getattr(training_batch, 'model_pred', None)
        assert model_pred is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.encoder_attention_mask is not None
        assert training_batch.timesteps is not None
        
        with torch.no_grad():
            assert self.teacher_transformer is not None
            teacher_clean_pred = self.teacher_transformer(
                hidden_states=training_batch.latents,
                encoder_hidden_states=training_batch.encoder_hidden_states,
                timestep=training_batch.timesteps,
                encoder_attention_mask=training_batch.encoder_attention_mask,
                return_dict=False
            )
        
        assert self.transformer is not None
        student_clean_pred = self.transformer(
            hidden_states=training_batch.latents,
            encoder_hidden_states=training_batch.encoder_hidden_states,
            timestep=training_batch.timesteps,
            encoder_attention_mask=training_batch.encoder_attention_mask,
            return_dict=False
        )
        
        dmd_loss = torch.mean((teacher_clean_pred.float() - student_clean_pred.float())**2)
        return dmd_loss

    def _compute_critic_loss(self, training_batch: TrainingBatch) -> torch.Tensor:
        """Compute critic loss for adversarial training."""
        assert self.critic_transformer is not None
        assert training_batch.latents is not None
        model_pred = getattr(training_batch, 'model_pred', None)
        assert model_pred is not None
        
        real_output = self.critic_transformer(training_batch.latents)
        real_loss = torch.mean((real_output - 1.0) ** 2)
        
        fake_output = self.critic_transformer(model_pred.detach())
        fake_loss = torch.mean(fake_output ** 2)
        
        critic_loss = real_loss + fake_loss
        return critic_loss

    def _student_forward_and_compute_loss(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Forward pass through student transformer and compute student losses."""
        assert self.transformer is not None
        assert training_batch.noisy_model_input is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.encoder_attention_mask is not None
        assert training_batch.timesteps is not None
        
        input_kwargs = {
            "hidden_states": training_batch.noisy_model_input,
            "encoder_hidden_states": training_batch.encoder_hidden_states,
            "encoder_hidden_states_neg": training_batch.encoder_hidden_states_neg,
            "timestep": training_batch.timesteps.to(get_torch_device(), dtype=torch.bfloat16),
            "encoder_attention_mask": training_batch.encoder_attention_mask,
            "encoder_attention_mask_neg": training_batch.encoder_attention_mask_neg,
            "return_dict": False,
        }
        with set_forward_context(
                current_timestep=training_batch.current_timestep,
                attn_metadata=training_batch.attn_metadata):
            model_output = self.transformer(**input_kwargs)
        setattr(training_batch, 'model_pred', model_output)
        
        dmd_loss = self._compute_dmd_loss(training_batch)
        
        dmd_loss.backward()
        
        return training_batch

    def _critic_forward_and_compute_loss(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Forward pass through critic transformer and compute critic loss."""
        assert self.critic_transformer is not None
        model_pred = getattr(training_batch, 'model_pred', None)
        assert model_pred is not None
        
        critic_loss = self._compute_critic_loss(training_batch)
        setattr(training_batch, 'critic_loss', critic_loss)
        training_batch.total_loss = float(getattr(training_batch, 'student_loss', 0.0) + critic_loss)
        
        critic_loss.backward()
        
        return training_batch


    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        assert self.training_args is not None
        max_grad_norm = self.training_args.max_grad_norm

        # TODO(will): perhaps move this into transformer api so that we can do
        # the following:
        # grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        if max_grad_norm is not None:
            # Clip gradients for both student and critic models
            model_parts = [self.transformer, self.critic_transformer]
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

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Train one step with alternating student and critic updates."""
        assert self.training_args is not None

        training_batch = self._prepare_training(training_batch)
        
        TRAIN_STUDENT = training_batch.current_timestep % self.student_critic_update_ratio == 0

        for _ in range(self.training_args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)

            # assert training_batch.latents is not None
            # training_batch.latents = shard_latents_across_sp(
            #     training_batch.latents,
            #     num_latent_t=self.training_args.num_latent_t)

            training_batch = self._normalize_dit_input(training_batch)
            training_batch = self._prepare_dit_inputs(training_batch)
            training_batch = self._build_attention_metadata(training_batch)

            if TRAIN_STUDENT:
                training_batch = self._student_forward_and_compute_loss(training_batch)

            training_batch = self._critic_forward_and_compute_loss(training_batch)

        # Clip gradients for both models
        training_batch = self._clip_grad_norm(training_batch)

        # Update student model only on certain steps
        if TRAIN_STUDENT and self.student_transformer_optimizer is not None:
            self.student_transformer_optimizer.step()
        
        # Always update critic model
        if self.critic_transformer_optimizer is not None:
            self.critic_transformer_optimizer.step()
        
        self.lr_scheduler.step()

        return training_batch

    def _resume_from_checkpoint(self) -> None:
        """Resume training from checkpoint with distillation models."""
        assert self.training_args is not None
        logger.info("Loading distillation checkpoint from %s",
                    self.training_args.resume_from_checkpoint)
        
        resumed_step = load_checkpoint(
            self.transformer, self.global_rank,
            self.training_args.resume_from_checkpoint, self.student_transformer_optimizer,
            self.train_dataloader, self.lr_scheduler,
            self.noise_random_generator)
            
        # TODO: Add checkpoint loading for critic and teacher models
        
        if resumed_step > 0:
            self.init_steps = resumed_step
            logger.info("Successfully resumed from step %s", resumed_step)
        else:
            logger.warning("Failed to load checkpoint, starting from step 0")
            self.init_steps = 0

    def _log_training_info(self) -> None:
        """Log distillation-specific training information."""
        # First call parent class method to get basic training info
        super()._log_training_info()
        
        # Then add distillation-specific information
        logger.info("Distillation-specific settings:")
        logger.info("  Student/Critic update ratio: %s", self.student_critic_update_ratio)
        assert isinstance(self.training_args, DistillationArgs)
        logger.info("  Max gradient norm: %s", self.training_args.max_grad_norm)
        assert self.teacher_transformer is not None
        logger.info("  Teacher transformer parameters: %s B",
                    sum(p.numel() for p in self.teacher_transformer.parameters()) / 1e9)
        assert self.critic_transformer is not None
        logger.info("  Critic transformer parameters: %s B",
                    sum(p.numel() for p in self.critic_transformer.parameters()) / 1e9)

    @torch.no_grad()
    def _log_validation(self, transformer, distillation_args, global_step) -> None:
        """Log validation results for distillation training."""
        assert distillation_args is not None
        distillation_args.inference_mode = True
        distillation_args.use_cpu_offload = False
        if not distillation_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation")

        sampling_param = SamplingParam.from_pretrained(distillation_args.model_path)

        validation_seed = distillation_args.seed if distillation_args.seed is not None else 42
        torch.manual_seed(validation_seed)
        torch.cuda.manual_seed_all(validation_seed)

        logger.info("Using validation seed: %s", validation_seed)

        logger.info('fastvideo_args.validation_preprocessed_path: %s',
                    distillation_args.validation_preprocessed_path)
        validation_dataset, validation_dataloader = build_parquet_map_style_dataloader(
            distillation_args.validation_preprocessed_path,
            batch_size=1,
            num_data_workers=0,
            drop_last=False,
            drop_first_row=sampling_param.negative_prompt is not None,
        )
        if sampling_param.negative_prompt:
            _, negative_prompt_embeds, negative_prompt_attention_mask, _ = validation_dataset.get_validation_negative_prompt(
            )

        if transformer:
            transformer.eval()

        validation_steps = distillation_args.validation_sampling_steps.split(",")
        validation_steps = [int(step) for step in validation_steps]
        validation_steps = [step for step in validation_steps if step > 0]

        for num_inference_steps in validation_steps:
            step_videos: List[np.ndarray] = []
            step_captions: List[str | None] = []

            for _, embeddings, masks, infos in validation_dataloader:
                step_captions.extend([None])
                prompt_embeds = embeddings.to(get_torch_device())
                prompt_attention_mask = masks.to(get_torch_device())

                latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                                sampling_param.height // 8,
                                sampling_param.width // 8]
                n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

                temporal_compression_factor = distillation_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
                num_frames = (distillation_args.num_latent_t -
                              1) * temporal_compression_factor + 1

                batch = ForwardBatch(
                    data_type="video",
                    latents=None,
                    seed=validation_seed,
                    generator=torch.Generator(
                        device="cpu").manual_seed(validation_seed),
                    prompt_embeds=[prompt_embeds],
                    prompt_attention_mask=[prompt_attention_mask],
                    negative_prompt_embeds=[negative_prompt_embeds],
                    negative_attention_mask=[negative_prompt_attention_mask],
                    height=distillation_args.num_height,
                    width=distillation_args.num_width,
                    num_frames=num_frames,
                    num_inference_steps=
                    num_inference_steps,
                    guidance_scale=sampling_param.guidance_scale,
                    n_tokens=n_tokens,
                    eta=0.0,
                )

                with torch.no_grad(), torch.autocast("cuda",
                                                     dtype=torch.bfloat16):
                    output_batch = self.validation_pipeline.forward(
                        batch, distillation_args)
                    samples = output_batch.output

                if self.rank_in_sp_group != 0:
                    continue

                video = rearrange(samples, "b c t h w -> t b c h w")
                frames: List[np.ndarray] = []
                for x in video:
                    x = torchvision.utils.make_grid(x, nrow=6)
                    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    frames.append((x * 255).numpy().astype(np.uint8))
                step_videos.append(np.array(frames))

            world_group = get_world_group()
            num_sp_groups = world_group.world_size // self.sp_group.world_size

            if self.rank_in_sp_group == 0:
                if self.global_rank == 0:
                    all_videos: List[np.ndarray] = []
                    all_videos.extend(step_videos)
                    all_captions = step_captions

                    for sp_group_idx in range(1, num_sp_groups):
                        src_rank = sp_group_idx * self.sp_world_size
                        recv_videos = world_group.recv_object(src=src_rank)
                        recv_captions = world_group.recv_object(src=src_rank)
                        all_videos.extend(recv_videos)
                        all_captions.extend(recv_captions)

                    video_filenames = []
                    for i, (video_frames,
                            caption) in enumerate(zip(all_videos,
                                                      all_captions)):
                        os.makedirs(distillation_args.output_dir, exist_ok=True)
                        filename = os.path.join(
                            distillation_args.output_dir,
                            f"validation_step_{global_step}_inference_steps_{num_inference_steps}_video_{i}.mp4"
                        )
                        imageio.mimsave(filename, list(video_frames), fps=sampling_param.fps)
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
                    world_group.send_object(step_videos, dst=0)
                    world_group.send_object(step_captions, dst=0)

        if transformer:
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

        logger.info("Initialized random seeds with seed: %s", seed)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler()

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
        
        for step in range(self.init_steps + 1,
                          self.training_args.max_train_steps + 1):
            start_time = time.perf_counter()

            training_batch = TrainingBatch()
            training_batch.current_timestep = step
            training_batch.visualize = step % self.training_args.validation_steps == 0
            training_batch = self.train_one_step(training_batch)

            total_loss = training_batch.total_loss
            student_loss = getattr(training_batch, 'student_loss', 0.0)
            student_loss = student_loss
            critic_loss = getattr(training_batch, 'critic_loss', 0.0)
            critic_loss = critic_loss
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
                wandb.log(
                    {
                        "train_total_loss": total_loss,
                        "train_student_loss": student_loss,
                        "train_critic_loss": critic_loss,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )
                
            if step % self.training_args.checkpointing_steps == 0:
                save_checkpoint(self.transformer, self.global_rank,
                                self.training_args.output_dir, step,
                                self.student_transformer_optimizer, self.train_dataloader,
                                self.lr_scheduler, self.noise_random_generator)
                if self.transformer:
                    self.transformer.train()
                self.sp_group.barrier()
                
            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                self._log_validation(self.transformer, self.training_args, step)
                gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
                logger.info("GPU memory usage after validation: %s MB",
                            gpu_memory_usage)

        wandb.finish()
        save_checkpoint(self.transformer, self.global_rank,
                        self.training_args.output_dir,
                        self.training_args.max_train_steps, self.student_transformer_optimizer,
                        self.train_dataloader, self.lr_scheduler,
                        self.noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()

