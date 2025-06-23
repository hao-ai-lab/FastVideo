# SPDX-License-Identifier: Apache-2.0
import gc
import math
import os
import time
from collections import deque
from typing import Any, Dict, Iterator, List, Optional

import imageio
import numpy as np
import torch
import torchvision
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

import fastvideo.v1.envs as envs
from fastvideo.v1.attention.backends.video_sparse_attn import (
    VideoSparseAttentionMetadata)
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset import build_parquet_map_style_dataloader
from fastvideo.v1.distributed import (cleanup_dist_env_and_memory, get_sp_group,
                                      get_torch_device, get_world_group)
from fastvideo.v1.fastvideo_args import (DistillationArgs, FastVideoArgs,
                                         TrainingArgs)
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import (ComposedPipelineBase, ForwardBatch,
                                    TrainingBatch)
from fastvideo.v1.training.training_pipeline import TrainingPipeline
from fastvideo.v1.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, get_sigmas, load_checkpoint,
    normalize_dit_input, save_checkpoint, shard_latents_across_sp)
from fastvideo.v1.utils import is_vsa_available, set_random_seed

import wandb  # isort: skip

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class DistillationPipeline(TrainingPipeline):
    """
    A distillation pipeline for training a student model using teacher model guidance.
    Inherits from TrainingPipeline to reuse training infrastructure.
    """
    _required_config_modules = ["scheduler", "transformer"]
    
    def __init__(self):
        super().__init__()
        self.student_transformer = None
        self.teacher_transformer = None
        self.critic_transformer = None
        self.student_transformer_optimizer = None
        self.critic_transformer_optimizer = None
        self.student_critic_update_ratio = 1  # How often to update student vs critic
        self.max_grad_norm = 10.0
        self.unconditional_dict = None  # Cache for unconditional embeddings

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the distillation training pipeline with multiple models."""
        logger.info("Initializing distillation training pipeline...")
        
        # Call parent initialization first
        super().initialize_training_pipeline(training_args)
        assert isinstance(self.training_args, DistillationArgs)
        
        # Rename transformer to student_transformer for clarity
        self.student_transformer = self.transformer
        
        # Initialize distillation-specific models
        self._initialize_distillation_models(training_args)
        
        # Initialize distillation-specific optimizers
        self._initialize_distillation_optimizers(training_args)
        
        # Set distillation-specific parameters
        self.student_critic_update_ratio = self.training_args.student_critic_update_ratio
        self.max_grad_norm = self.training_args.max_grad_norm
        
        logger.info(f"Distillation pipeline initialized with student_critic_update_ratio={self.student_critic_update_ratio}")

    def _initialize_distillation_models(self, training_args: TrainingArgs):
        """Initialize teacher and critic models by copying the student model."""
        assert self.student_transformer is not None, "Student model must be loaded first"
        
        # Create teacher model by copying student model
        self.teacher_transformer = self._copy_model(self.student_transformer)
        assert self.teacher_transformer is not None, "Failed to create teacher model"
        
        # Create critic model by copying student model
        self.critic_transformer = self._copy_model(self.student_transformer)
        assert self.critic_transformer is not None, "Failed to create critic model"
        
        # Set teacher model to eval mode and freeze parameters
        self.teacher_transformer.requires_grad_(False)
        self.teacher_transformer.eval()
        
        # Set critic model to train mode
        self.critic_transformer.requires_grad_(True)
        self.critic_transformer.train()
        
        logger.info("Distillation models initialized:")
        logger.info("  - Student transformer: %s B parameters (trainable)", 
                    sum(p.numel() for p in self.student_transformer.parameters()) / 1e9)
        logger.info("  - Teacher transformer: %s B parameters (frozen)", 
                    sum(p.numel() for p in self.teacher_transformer.parameters()) / 1e9)
        logger.info("  - Critic transformer: %s B parameters (trainable)", 
                    sum(p.numel() for p in self.critic_transformer.parameters()) / 1e9)

    def _copy_model(self, original_model):
        """Create a deep copy of a model."""
        import copy
        copied_model = copy.deepcopy(original_model)
        device = next(original_model.parameters()).device
        copied_model = copied_model.to(device)
        if hasattr(copied_model, 'reset_parameters'):
            copied_model.reset_parameters()
        return copied_model

    def _initialize_distillation_optimizers(self, training_args: TrainingArgs):
        """Initialize separate optimizers for student and critic models."""
        assert self.student_transformer is not None
        student_params = list(filter(lambda p: p.requires_grad, self.student_transformer.parameters()))
        self.student_transformer_optimizer = torch.optim.AdamW(
            student_params,
            lr=training_args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )
        
        assert self.critic_transformer is not None
        critic_params = list(filter(lambda p: p.requires_grad, self.critic_transformer.parameters()))
        self.critic_transformer_optimizer = torch.optim.AdamW(
            critic_params,
            lr=training_args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )
        
        logger.info("Distillation optimizers initialized: student and critic")

    def _prepare_training(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Prepare training environment for distillation."""
        assert self.student_transformer is not None
        self.student_transformer.requires_grad_(True)
        self.student_transformer.train()
        assert self.critic_transformer is not None
        self.critic_transformer.requires_grad_(True)
        self.critic_transformer.train()
        
        assert self.student_transformer_optimizer is not None
        self.student_transformer_optimizer.zero_grad()
        assert self.critic_transformer_optimizer is not None
        self.critic_transformer_optimizer.zero_grad()
        
        training_batch.total_loss = 0.0
        setattr(training_batch, 'student_transformer_loss', 0.0)
        setattr(training_batch, 'critic_transformer_loss', 0.0)
        
        return training_batch

    def _compute_teacher_outputs(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Compute teacher model outputs for distillation."""
        assert self.teacher_transformer is not None
        assert training_batch.noisy_model_input is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.encoder_attention_mask is not None
        assert training_batch.timesteps is not None
        
        with torch.no_grad():
            teacher_input_kwargs = {
                "hidden_states": training_batch.noisy_model_input,
                "encoder_hidden_states": training_batch.encoder_hidden_states,
                "timestep": training_batch.timesteps.to(get_torch_device(), dtype=torch.bfloat16),
                "encoder_attention_mask": training_batch.encoder_attention_mask,
                "return_dict": False,
            }
            
            teacher_output = self.teacher_transformer(**teacher_input_kwargs)
            setattr(training_batch, 'teacher_output', teacher_output)
            
        return training_batch

    def _compute_distillation_loss(self, training_batch: TrainingBatch) -> torch.Tensor:
        """Compute distillation loss between teacher and student outputs."""
        teacher_output = getattr(training_batch, 'teacher_output', None)
        assert teacher_output is not None
        model_pred = getattr(training_batch, 'model_pred', None)
        assert model_pred is not None
        
        distillation_loss = torch.mean((teacher_output.float() - model_pred.float())**2)
        return distillation_loss

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
        
        assert self.student_transformer is not None
        student_clean_pred = self.student_transformer(
            hidden_states=training_batch.latents,
            encoder_hidden_states=training_batch.encoder_hidden_states,
            timestep=training_batch.timesteps,
            encoder_attention_mask=training_batch.encoder_attention_mask,
            return_dict=False
        )
        
        dmd_loss = torch.mean((teacher_clean_pred.float() - student_clean_pred.float())**2)
        return dmd_loss

    def _compute_critic_transformer_loss(self, training_batch: TrainingBatch) -> torch.Tensor:
        """Compute critic loss for adversarial training."""
        assert self.critic_transformer is not None
        assert training_batch.latents is not None
        model_pred = getattr(training_batch, 'model_pred', None)
        assert model_pred is not None
        
        real_output = self.critic_transformer(training_batch.latents)
        real_loss = torch.mean((real_output - 1.0) ** 2)
        
        fake_output = self.critic_transformer(model_pred.detach())
        fake_loss = torch.mean(fake_output ** 2)
        
        critic_transformer_loss = real_loss + fake_loss
        return critic_transformer_loss

    def _compute_student_adversarial_loss(self, training_batch: TrainingBatch) -> torch.Tensor:
        """Compute student's adversarial loss."""
        assert self.critic_transformer is not None
        model_pred = getattr(training_batch, 'model_pred', None)
        assert model_pred is not None
        
        fake_output = self.critic_transformer(model_pred)
        adversarial_loss = torch.mean((fake_output - 1.0) ** 2)
        return adversarial_loss

    def _student_transformer_forward_and_compute_loss(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Forward pass through student and compute all losses."""
        assert self.student_transformer is not None
        assert training_batch.noisy_model_input is not None
        assert training_batch.encoder_hidden_states is not None
        assert training_batch.encoder_attention_mask is not None
        assert training_batch.timesteps is not None
        
        model_output = self.student_transformer(
            hidden_states=training_batch.noisy_model_input,
            encoder_hidden_states=training_batch.encoder_hidden_states,
            timestep=training_batch.timesteps.to(get_torch_device(), dtype=torch.bfloat16),
            encoder_attention_mask=training_batch.encoder_attention_mask,
            return_dict=False,
        )
        setattr(training_batch, 'model_pred', model_output)
        
        training_batch = self._compute_teacher_outputs(training_batch)
        
        task_loss = self._compute_task_loss(training_batch)
        distillation_loss = self._compute_distillation_loss(training_batch)
        dmd_loss = self._compute_dmd_loss(training_batch)
        adversarial_loss = self._compute_student_adversarial_loss(training_batch)
        critic_transformer_loss = self._compute_critic_transformer_loss(training_batch)
        
        assert isinstance(self.training_args, DistillationArgs)
        distillation_weight = self.training_args.distillation_weight
        dmd_weight = self.training_args.dmd_weight
        adversarial_weight = self.training_args.adversarial_weight
        
        student_transformer_loss = (task_loss + 
                         distillation_weight * distillation_loss + 
                         dmd_weight * dmd_loss + 
                         adversarial_weight * adversarial_loss)
        
        setattr(training_batch, 'task_loss', task_loss)
        setattr(training_batch, 'distillation_loss', distillation_loss)
        setattr(training_batch, 'dmd_loss', dmd_loss)
        setattr(training_batch, 'adversarial_loss', adversarial_loss)
        setattr(training_batch, 'critic_transformer_loss', critic_transformer_loss)
        setattr(training_batch, 'student_transformer_loss', student_transformer_loss)
        training_batch.total_loss = float(student_transformer_loss + critic_transformer_loss)
        
        student_transformer_loss.backward()
        critic_transformer_loss.backward()
        
        return training_batch

    def _compute_task_loss(self, training_batch: TrainingBatch) -> torch.Tensor:
        """Compute the main task loss (e.g., noise prediction loss)."""
        model_pred = getattr(training_batch, 'model_pred', None)
        assert model_pred is not None
        assert training_batch.noise is not None
        
        task_loss = torch.mean((model_pred - training_batch.noise) ** 2)
        return task_loss

    def _clip_grad_norm(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Clip gradients for both student and critic."""
        assert self.student_transformer is not None
        student_transformer_grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
            self.student_transformer, self.max_grad_norm)
        
        assert self.critic_transformer is not None
        critic_transformer_grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
            self.critic_transformer, self.max_grad_norm)
        
        setattr(training_batch, 'student_transformer_grad_norm', student_transformer_grad_norm)
        setattr(training_batch, 'critic_transformer_grad_norm', critic_transformer_grad_norm)
        training_batch.grad_norm = float(student_transformer_grad_norm) if student_transformer_grad_norm is not None else 0.0
        
        return training_batch

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """Train one step with alternating student and critic updates."""
        assert self.training_args is not None

        training_batch = self._prepare_training(training_batch)
        
        TRAIN_STUDENT_TRANSFORMER = training_batch.current_timestep % self.student_critic_update_ratio == 0

        for _ in range(self.training_args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)

            assert training_batch.latents is not None
            training_batch.latents = shard_latents_across_sp(
                training_batch.latents,
                num_latent_t=self.training_args.num_latent_t)

            training_batch = self._normalize_dit_input(training_batch)
            training_batch = self._prepare_dit_inputs(training_batch)
            training_batch = self._build_attention_metadata(training_batch)
            
            training_batch = self._student_transformer_forward_and_compute_loss(training_batch)

        training_batch = self._clip_grad_norm(training_batch)

        if TRAIN_STUDENT_TRANSFORMER and self.student_transformer_optimizer is not None:
            self.student_transformer_optimizer.step()
        if self.critic_transformer_optimizer is not None:
            self.critic_transformer_optimizer.step()
        
        if TRAIN_STUDENT_TRANSFORMER and hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return training_batch

    def _resume_from_checkpoint(self) -> None:
        """Resume training from checkpoint with distillation models."""
        assert self.training_args is not None
        logger.info("Loading distillation checkpoint from %s",
                    self.training_args.resume_from_checkpoint)
        
        resumed_step = load_checkpoint(
            self.student_transformer, self.global_rank,
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
        super()._log_training_info()
        
        logger.info("Distillation-specific settings:")
        logger.info("  Student/Critic update ratio: %s", self.student_critic_update_ratio)
        logger.info("  Max gradient norm: %s", self.max_grad_norm)
        assert self.teacher_transformer is not None
        logger.info("  Teacher transformer parameters: %s B",
                    sum(p.numel() for p in self.teacher_transformer.parameters()) / 1e9)
        assert self.critic_transformer is not None
        logger.info("  Critic transformer parameters: %s B",
                    sum(p.numel() for p in self.critic_transformer.parameters()) / 1e9)

    @torch.no_grad()
    def _log_validation(self, student_transformer, training_args, global_step) -> None:
        """Log validation results for distillation training."""
        assert training_args is not None
        training_args.inference_mode = True
        training_args.use_cpu_offload = False
        if not training_args.log_validation:
            return
        if self.validation_pipeline is None:
            raise ValueError("Validation pipeline is not set")

        logger.info("Starting validation")

        sampling_param = SamplingParam.from_pretrained(training_args.model_path)

        validation_seed = training_args.seed if training_args.seed is not None else 42
        torch.manual_seed(validation_seed)
        torch.cuda.manual_seed_all(validation_seed)

        logger.info("Using validation seed: %s", validation_seed)

        logger.info('fastvideo_args.validation_preprocessed_path: %s',
                    training_args.validation_preprocessed_path)
        validation_dataset, validation_dataloader = build_parquet_map_style_dataloader(
            training_args.validation_preprocessed_path,
            batch_size=1,
            num_data_workers=0,
            drop_last=False,
            drop_first_row=sampling_param.negative_prompt is not None,
        )
        if sampling_param.negative_prompt:
            _, negative_prompt_embeds, negative_prompt_attention_mask, _ = validation_dataset.get_validation_negative_prompt(
            )

        if self.student_transformer:
            self.student_transformer.eval()

        validation_steps = training_args.validation_sampling_steps.split(",")
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

                temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
                num_frames = (training_args.num_latent_t -
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
                    height=training_args.num_height,
                    width=training_args.num_width,
                    num_frames=num_frames,
                    num_inference_steps=
                    num_inference_steps,
                    guidance_scale=sampling_param.guidance_scale,
                    n_tokens=n_tokens,
                    eta=0.0,
                    VSA_sparsity=training_args.VSA_sparsity,
                )

                with torch.no_grad(), torch.autocast("cuda",
                                                     dtype=torch.bfloat16):
                    output_batch = self.validation_pipeline.forward(
                        batch, training_args)
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
                        os.makedirs(training_args.output_dir, exist_ok=True)
                        filename = os.path.join(
                            training_args.output_dir,
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

        if self.student_transformer:
            self.student_transformer.train()
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
        self._log_validation(self.student_transformer, self.training_args, 1)

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
                current_decay_times = min(step // vsa_decay_interval_steps,
                                          vsa_sparsity // vsa_decay_rate)
                current_vsa_sparsity = current_decay_times * vsa_decay_rate
            else:
                current_vsa_sparsity = 0.0

            training_batch = TrainingBatch()
            training_batch.current_timestep = step
            training_batch.current_vsa_sparsity = current_vsa_sparsity
            training_batch = self.train_one_step(training_batch)

            total_loss = training_batch.total_loss
            student_transformer_loss = getattr(training_batch, 'student_transformer_loss', 0.0)
            student_transformer_loss = student_transformer_loss
            critic_transformer_loss = getattr(training_batch, 'critic_transformer_loss', 0.0)
            critic_transformer_loss = critic_transformer_loss
            distillation_loss = getattr(training_batch, 'distillation_loss', 0.0)
            distillation_loss = distillation_loss
            task_loss = getattr(training_batch, 'task_loss', 0.0)
            task_loss = task_loss
            adversarial_loss = getattr(training_batch, 'adversarial_loss', 0.0)
            adversarial_loss = adversarial_loss
            grad_norm = training_batch.grad_norm

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "total_loss": f"{total_loss:.4f}",
                "student_loss": f"{student_transformer_loss:.4f}",
                "critic_loss": f"{critic_transformer_loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            })
            progress_bar.update(1)
            
            if self.global_rank == 0:
                wandb.log(
                    {
                        "train_total_loss": total_loss,
                        "train_student_transformer_loss": student_transformer_loss,
                        "train_critic_transformer_loss": critic_transformer_loss,
                        "train_distillation_loss": distillation_loss,
                        "train_task_loss": task_loss,
                        "train_adversarial_loss": adversarial_loss,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                        "vsa_sparsity": current_vsa_sparsity,
                    },
                    step=step,
                )
                
            if step % self.training_args.checkpointing_steps == 0:
                save_checkpoint(self.student_transformer, self.global_rank,
                                self.training_args.output_dir, step,
                                self.student_transformer_optimizer, self.train_dataloader,
                                self.lr_scheduler, self.noise_random_generator)
                if self.student_transformer:
                    self.student_transformer.train()
                self.sp_group.barrier()
                
            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                self._log_validation(self.student_transformer, self.training_args, step)
                gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
                logger.info("GPU memory usage after validation: %s MB",
                            gpu_memory_usage)

        wandb.finish()
        save_checkpoint(self.student_transformer, self.global_rank,
                        self.training_args.output_dir,
                        self.training_args.max_train_steps, self.student_transformer_optimizer,
                        self.train_dataloader, self.lr_scheduler,
                        self.noise_random_generator)

        if get_sp_group():
            cleanup_dist_env_and_memory()

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Initialize validation pipeline - must be implemented by subclasses."""
        raise NotImplementedError(
            "Distillation pipelines must implement this method") 