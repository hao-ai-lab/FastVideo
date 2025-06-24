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
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from einops import rearrange
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

import fastvideo.v1.envs as envs
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.dataset import build_parquet_map_style_dataloader
from fastvideo.v1.distributed import (cleanup_dist_env_and_memory, get_sp_group,
                                      get_torch_device, get_world_group)
from fastvideo.v1.fastvideo_args import FastVideoArgs, DistillationArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import (ComposedPipelineBase, ForwardBatch,
                                    TrainingBatch, DistillBatch)
from fastvideo.v1.training.training_pipeline import TrainingPipeline
from fastvideo.v1.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    compute_density_for_timestep_sampling, get_sigmas, load_checkpoint,
    normalize_dit_input, save_checkpoint, shard_latents_across_sp, prepare_for_saving)
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

    def _prepare_distillation(self, training_batch: DistillBatch) -> DistillBatch:
        """Prepare training environment for distillation."""
        self.transformer.requires_grad_(True)
        self.transformer.train()
        self.critic_transformer.requires_grad_(True)
        self.critic_transformer.train()
        self.student_transformer_optimizer.zero_grad()
        self.critic_transformer_optimizer.zero_grad()
        
        training_batch.total_loss = 0.0
        training_batch.student_loss = 0.0
        training_batch.critic_loss = 0.0

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

    def _student_forward(self, conditional_dict: Dict[str, Any], unconditional_dict: Dict[str, Any], clean_latent: torch.Tensor) -> torch.Tensor:
        """Forward pass through student transformer and compute student losses."""
        video_latent_shape = self.video_latent_shape
        dtype = clean_latent.dtype
        simulated_noisy_input = []
        for timestep in self.denoising_step_list:
            noise = torch.randn(
                video_latent_shape, device=self.device, dtype=dtype)

            noisy_timestep = timestep * torch.ones(
                video_latent_shape[:2], device=self.device, dtype=torch.long)

            if timestep != 0:
                noisy_image = self.noise_scheduler.add_noise(
                    clean_latent.flatten(0, 1),
                    noise.flatten(0, 1),
                    noisy_timestep.flatten(0, 1)
                ).unflatten(0, video_latent_shape[:2])
            else:
                noisy_image = clean_latent

            simulated_noisy_input.append(noisy_image)

        simulated_noisy_input = torch.stack(simulated_noisy_input, dim=1)

        # Step 2: Randomly sample a timestep and pick the corresponding input
        index = torch.randint(0, len(self.denoising_step_list), [
                              video_latent_shape[0], video_latent_shape[1]], device=self.device, dtype=torch.long)

        index = self._process_timestep(index, type=self.distill_task_type)

        # select the corresponding timestep's noisy input from the stacked tensor [B, T, F, C, H, W]
        noisy_input = torch.gather(
            simulated_noisy_input, dim=1,
            index=index.reshape(index.shape[0], 1, index.shape[1], 1, 1, 1).expand(
                -1, -1, -1, *video_latent_shape[2:])
        ).squeeze(1)

        timestep = self.denoising_step_list[index]

        # TODO(yongqi)
        pred_video = self.transformer(
            noisy_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        pred_video = pred_video.type_as(noisy_input)

        return pred_video

    def _compute_kl_grad(
        self, noisy_video: torch.Tensor,
        estimated_clean_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        pred_fake_video = self.critic_transformer(
            noisy_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        pred_real_video_cond = self.teacher_transformer(
            noisy_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        pred_real_video_uncond = self.teacher_transformer(
            noisy_video=noisy_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

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
            "dmdtrain_clean_latent": estimated_clean_video.detach(),
            "dmdtrain_noisy_latent": noisy_video.detach(),
            "dmdtrain_pred_real_video": pred_real_video.detach(),
            "dmdtrain_pred_fake_video": pred_fake_video.detach(),
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def _compute_dmd_loss(self, pred_video: torch.Tensor, conditional_dict: Dict[str, Any], unconditional_dict: Dict[str, Any]) -> Tuple[torch.Tensor, dict]:
        """Compute DMD (Diffusion Model Distillation) loss."""
        
        original_latent = pred_video

        batch_size, num_frame = pred_video.shape[:2]

        with torch.no_grad():
            timestep = torch.randint(
                0,
                self.num_train_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )

            timestep = self._process_timestep(
                timestep, type=self.distill_task_type)

            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(pred_video)
            noisy_latent = self.noise_scheduler.add_noise(
                pred_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

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

    def _compute_critic_loss(self, training_batch: DistillBatch) -> torch.Tensor:
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

    def _student_forward_and_compute_dmd_loss(self, training_batch: DistillBatch) -> Tuple[DistillBatch, torch.Tensor, dict]:
        """Forward pass through student transformer and compute student losses."""
        assert training_batch.conditional_dict is not None
        assert training_batch.unconditional_dict is not None
        assert training_batch.clean_latent is not None
        
        pred_video = self._student_forward(
            conditional_dict=training_batch.conditional_dict,
            unconditional_dict=training_batch.unconditional_dict,
            clean_latent=training_batch.clean_latent
        )

        dmd_loss, dmd_log_dict = self._compute_dmd_loss(
            pred_video=pred_video,
            conditional_dict=training_batch.conditional_dict,
            unconditional_dict=training_batch.unconditional_dict
        )
        
        return training_batch, dmd_loss, dmd_log_dict

    def _critic_forward_and_compute_loss(self, training_batch: DistillBatch) -> Tuple[DistillBatch, torch.Tensor, dict]:
        assert training_batch.conditional_dict is not None
        assert training_batch.unconditional_dict is not None
        assert training_batch.clean_latent is not None

        with torch.no_grad():
            generated_video = self._student_forward(
                conditional_dict=training_batch.conditional_dict,
                unconditional_dict=training_batch.unconditional_dict,
                clean_latent=training_batch.clean_latent
            )
        video_latent_shape = self.video_latent_shape
        critic_timestep = torch.randint(
            0,
            self.num_train_timestep,
            video_latent_shape[:2],
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

        critic_noise = torch.randn_like(generated_video)
        noisy_generated_video = self.noise_scheduler.add_noise(
            generated_video.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, video_latent_shape[:2])

        pred_fake_video = self.critic_transformer(
            noisy_video=noisy_generated_video,
            conditional_dict=training_batch.conditional_dict,
            timestep=critic_timestep
        )

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
            "critic_timestep": critic_timestep.detach()
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
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)
        
    def _clip_grad_norm(self, training_batch: DistillBatch) -> DistillBatch:
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

    def train_one_step(self, training_batch: DistillBatch) -> DistillBatch:
        """Train one step with alternating student and critic updates."""
        assert self.training_args is not None

        training_batch = self._prepare_distillation(training_batch)

        TRAIN_STUDENT = training_batch.current_timestep % self.student_critic_update_ratio == 0

        for _ in range(self.training_args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)
            training_batch = DistillBatch(**{k: v for k, v in training_batch.__dict__.items()})

            # assert training_batch.latents is not None
            # training_batch.latents = shard_latents_across_sp(
            #     training_batch.latents,
            #     num_latent_t=self.training_args.num_latent_t)

            training_batch = self._normalize_dit_input(training_batch)
            training_batch = self._prepare_dit_inputs(training_batch)
            training_batch = self._build_attention_metadata(training_batch)

            if TRAIN_STUDENT:
                training_batch, dmd_loss, dmd_log_dict = self._student_forward_and_compute_dmd_loss(training_batch)
                dmd_loss.backward()

            training_batch, critic_loss, critic_log_dict = self._critic_forward_and_compute_loss(training_batch)

        # Clip gradients for both models
        training_batch = self._clip_grad_norm(training_batch)

        # Update student model only on certain steps
        if TRAIN_STUDENT:
            self.student_transformer_optimizer.step()
        
        # Always update critic model
        critic_loss.backward()
        self.critic_transformer_optimizer.step()
        
        self.lr_scheduler.step()

        training_batch.dmd_log_dict = dmd_log_dict
        training_batch.critic_log_dict = critic_log_dict
        
        # Record loss values for logging
        training_batch.student_loss = dmd_loss.item() if TRAIN_STUDENT else 0.0
        training_batch.critic_loss = critic_loss.item()
        training_batch.total_loss = training_batch.student_loss + training_batch.critic_loss
        
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
            _, self.negative_prompt_embeds, self.negative_prompt_attention_mask, _ = validation_dataset.get_validation_negative_prompt(
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
                    negative_prompt_embeds=[self.negative_prompt_embeds],
                    negative_attention_mask=[self.negative_prompt_attention_mask],
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
        
    def add_visualization(self, generator_log_dict: Dict[str, Any], critic_log_dict: Dict[str, Any]):
        """Add visualization data to wandb logging."""
        wandb_loss_dict = {}
        
        # Process critic training data
        critictrain_latent, critictrain_noisy_latent, critictrain_pred_video = map(
            lambda x: self.transformer.vae.decode_to_pixel(x).squeeze(1),
            [critic_log_dict['critictrain_latent'], critic_log_dict['critictrain_noisy_latent'],
             critic_log_dict['critictrain_pred_video']]
        )

        wandb_loss_dict.update({
            "critictrain_latent": prepare_for_saving(critictrain_latent),
            "critictrain_noisy_latent": prepare_for_saving(critictrain_noisy_latent),
            "critictrain_pred_video": prepare_for_saving(critictrain_pred_video)
        })

        # Process DMD training data if available
        if "dmdtrain_clean_latent" in generator_log_dict:
            (dmdtrain_clean_latent, dmdtrain_noisy_latent, dmdtrain_pred_real_video, dmdtrain_pred_fake_video) = map(
                lambda x: self.transformer.vae.decode_to_pixel(x).squeeze(1),
                [generator_log_dict['dmdtrain_clean_latent'], generator_log_dict['dmdtrain_noisy_latent'],
                 generator_log_dict['dmdtrain_pred_real_video'], generator_log_dict['dmdtrain_pred_fake_video']]
            )

            wandb_loss_dict.update({
                "dmdtrain_clean_latent": prepare_for_saving(dmdtrain_clean_latent),
                "dmdtrain_noisy_latent": prepare_for_saving(dmdtrain_noisy_latent),
                "dmdtrain_pred_real_video": prepare_for_saving(dmdtrain_pred_real_video),
                "dmdtrain_pred_fake_video": prepare_for_saving(dmdtrain_pred_fake_video)
            })
        
        # Log to wandb
        if self.global_rank == 0:
            wandb.log(wandb_loss_dict)

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

            training_batch = DistillBatch()
            training_batch.current_timestep = step
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
                
            if step % self.training_args.checkpointing_steps == 0:
                save_checkpoint(self.transformer, self.global_rank,
                                self.training_args.output_dir, step,
                                self.student_transformer_optimizer, self.train_dataloader,
                                self.lr_scheduler, self.noise_random_generator)
                if self.transformer:
                    self.transformer.train()
                # self.sp_group.barrier()
                
            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                self.add_visualization(training_batch.dmd_log_dict, training_batch.critic_log_dict)
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

        # if get_sp_group():
        #     cleanup_dist_env_and_memory()

    def _get_next_batch(self, training_batch: DistillBatch) -> DistillBatch:
        """Override parent method to return DistillBatch."""
        result = super()._get_next_batch(training_batch)
        # Convert TrainingBatch to DistillBatch
        distill_batch = DistillBatch()
        for key, value in result.__dict__.items():
            setattr(distill_batch, key, value)
        return distill_batch

    def _normalize_dit_input(self, training_batch: DistillBatch) -> DistillBatch:
        """Override parent method to return DistillBatch."""
        result = super()._normalize_dit_input(training_batch)
        # Convert TrainingBatch to DistillBatch
        distill_batch = DistillBatch()
        for key, value in result.__dict__.items():
            setattr(distill_batch, key, value)
        return distill_batch

    def _prepare_dit_inputs(self, training_batch: DistillBatch) -> DistillBatch:
        """Override parent method to return DistillBatch."""
        result = super()._prepare_dit_inputs(training_batch)
        # Convert TrainingBatch to DistillBatch
        distill_batch = DistillBatch()
        for key, value in result.__dict__.items():
            setattr(distill_batch, key, value)
        conditional_dict = {
            "encoder_hidden_states": training_batch.encoder_hidden_states,
            "encoder_attention_mask": training_batch.encoder_attention_mask,
        }
        unconditional_dict = {
            "encoder_hidden_states_neg": self.negative_prompt_embeds,
            "encoder_attention_mask_neg": self.negative_prompt_attention_mask,
        }

        distill_batch.conditional_dict = conditional_dict
        distill_batch.unconditional_dict = unconditional_dict
        self.video_latent_shape = training_batch.latents.shape

        return distill_batch

    def _build_attention_metadata(self, training_batch: DistillBatch) -> DistillBatch:
        """Override parent method to return DistillBatch."""
        result = super()._build_attention_metadata(training_batch)
        # Convert TrainingBatch to DistillBatch
        distill_batch = DistillBatch()
        for key, value in result.__dict__.items():
            setattr(distill_batch, key, value)
        return distill_batch

class FlowPredLoss():
    def __call__(
        self, x: torch.Tensor,
        noise: torch.Tensor,
        flow_pred: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean((flow_pred - (noise - x)) ** 2)
