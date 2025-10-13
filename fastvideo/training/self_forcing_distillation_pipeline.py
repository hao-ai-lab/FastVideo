# SPDX-License-Identifier: Apache-2.0
import copy
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from tqdm.auto import tqdm

import fastvideo.envs as envs
from fastvideo.distributed import (cleanup_dist_env_and_memory,
                                   get_local_torch_device, get_sp_group,
                                   get_world_group)
from fastvideo.fastvideo_args import TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger

from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import SelfForcingFlowMatchScheduler
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.training.training_utils import (EMA_FSDP,
                                               save_distillation_checkpoint)
from fastvideo.utils import is_vsa_available, set_random_seed
from fastvideo.profiler import profile_region

logger = init_logger(__name__)

vsa_available = is_vsa_available()


class SelfForcingDistillationPipeline(DistillationPipeline):
    """
    A self-forcing distillation pipeline that alternates between training
    the generator and critic based on the self-forcing methodology.
    
    This implementation follows the self-forcing approach where:
    1. Generator and critic are trained in alternating steps
    2. Generator loss uses DMD-style loss with the critic as fake score
    3. Critic loss trains the fake score model to distinguish real vs fake
    """

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the self-forcing training pipeline."""
        logger.info("Initializing self-forcing distillation pipeline...")

        self.generator_ema: EMA_FSDP | None = None
        self.generator_ema_2: EMA_FSDP | None = None

        super().initialize_training_pipeline(training_args)
        try:
            logger.info("RANK: %s, entered initialize_training_pipeline",
                        self.global_rank,
                        local_main_process_only=False)
        except Exception:
            logger.info("Entered initialize_training_pipeline (rank unknown)")

        self.noise_scheduler = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            training=True)
        self.dfake_gen_update_ratio = getattr(training_args,
                                              'dfake_gen_update_ratio', 5)

        self.num_frame_per_block = getattr(training_args, 'num_frame_per_block',
                                           3)
        self.independent_first_frame = getattr(training_args,
                                               'independent_first_frame', False)
        self.same_step_across_blocks = getattr(training_args,
                                               'same_step_across_blocks', False)
        self.last_step_only = getattr(training_args, 'last_step_only', False)
        self.context_noise = getattr(training_args, 'context_noise', 0)

        self.kv_cache1: list[dict[str, Any]] | None = None
        self.crossattn_cache: list[dict[str, Any]] | None = None

        logger.info("Self-forcing generator update ratio: %s",
                    self.dfake_gen_update_ratio)
        logger.info("RANK: %s, exiting initialize_training_pipeline",
                    self.global_rank,
                    local_main_process_only=False)

    def generate_and_sync_list(self, num_blocks: int, num_denoising_steps: int,
                               device: torch.device) -> list[int]:
        """Generate and synchronize random exit flags across distributed processes."""
        logger.info(
            "RANK: %s, enter generate_and_sync_list blocks=%s steps=%s device=%s",
            self.global_rank,
            num_blocks,
            num_denoising_steps,
            str(device),
            local_main_process_only=False)
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(low=0,
                                    high=num_denoising_steps,
                                    size=(num_blocks, ),
                                    device=device)
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices,
                           src=0)  # Broadcast the random indices to all ranks
        flags = indices.tolist()
        logger.info(
            "RANK: %s, exit generate_and_sync_list flags_len=%s first=%s",
            self.global_rank,
            len(flags),
            flags[0] if len(flags) > 0 else None,
            local_main_process_only=False)
        return flags

    def generator_loss(
            self, training_batch: TrainingBatch
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute generator loss using DMD-style approach.
        The generator tries to fool the critic (fake_score_transformer).
        """
        with set_forward_context(
                current_timestep=training_batch.timesteps,
                attn_metadata=training_batch.attn_metadata_vsa):
            generator_pred_video = self._generator_multi_step_simulation_forward(
                training_batch)

        with set_forward_context(current_timestep=training_batch.timesteps,
                                 attn_metadata=training_batch.attn_metadata):
            dmd_loss = self._dmd_forward(
                generator_pred_video=generator_pred_video,
                training_batch=training_batch)

        log_dict = {
            "dmdtrain_gradient_norm": torch.tensor(0.0, device=self.device)
        }

        return dmd_loss, log_dict

    def critic_loss(
            self, training_batch: TrainingBatch
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute critic loss using flow matching between noise and generator output.
        The critic learns to predict the flow from noise to the generator's output.
        """
        updated_batch, flow_matching_loss = self.faker_score_forward(
            training_batch)
        training_batch.fake_score_latent_vis_dict = updated_batch.fake_score_latent_vis_dict
        log_dict: dict[str, Any] = {}

        return flow_matching_loss, log_dict

    def _generator_multi_step_simulation_forward(
            self,
            training_batch: TrainingBatch,
            return_sim_steps: bool = False) -> torch.Tensor:
        """Forward pass through student transformer matching inference procedure with KV cache management.
        
        This function is adapted from the reference self-forcing implementation's inference_with_trajectory
        and includes gradient masking logic for dynamic frame generation.
        """
        latents = training_batch.latents
        dtype = latents.dtype
        batch_size = latents.shape[0]
        initial_latent = getattr(training_batch, 'image_latent', None)

        # Dynamic frame generation logic (adapted from _run_generator)
        num_training_frames = getattr(self.training_args, 'num_latent_t', 21)

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.independent_first_frame else 21
        max_num_frames = num_training_frames - 1 if self.independent_first_frame else num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block

        # Sample number of blocks and sync across processes
        num_generated_blocks = torch.randint(min_num_blocks,
                                             max_num_blocks + 1, (1, ),
                                             device=self.device)
        if dist.is_initialized():
            dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1

        # Create noise with dynamic shape
        if initial_latent is not None:
            noise_shape = [
                batch_size, num_generated_frames - 1,
                *self.video_latent_shape[2:]
            ]
        else:
            noise_shape = [
                batch_size, num_generated_frames, *self.video_latent_shape[2:]
            ]

        noise = torch.randn(noise_shape, device=self.device, dtype=dtype)
        if self.sp_world_size > 1:
            noise = rearrange(noise,
                              "b (n t) c h w -> b n t c h w",
                              n=self.sp_world_size).contiguous()
            noise = noise[:, self.rank_in_sp_group, :, :, :, :]

        batch_size, num_frames, num_channels, height, width = noise.shape

        # Block size calculation
        if not self.independent_first_frame or (self.independent_first_frame
                                                and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[
            1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype)

        def get_model_device(model):
            if model is None:
                return "None"
            try:
                return next(model.parameters()).device
            except (StopIteration, AttributeError):
                return "Unknown"

        # Step 1: Initialize KV cache to all zeros
        cache_frames = num_generated_frames + num_input_frames
        self.kv_cache1, self.crossattn_cache = self._initialize_simulation_caches(
            batch_size, dtype, self.device, max_num_frames=cache_frames)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones(
                [batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            output[:, :1] = initial_latent
            with torch.no_grad():
                # Build input kwargs for initial latent
                training_batch_temp = self._build_distill_input_kwargs(
                    initial_latent, timestep * 0,
                    training_batch.conditional_dict, training_batch)
                # we process the image latent with self.transformer_2 (low-noise expert)
                current_model = self.transformer_2 if self.transformer_2 is not None else self.transformer
                current_model(
                    hidden_states=training_batch_temp.
                    input_kwargs['hidden_states'],
                    encoder_hidden_states=training_batch_temp.
                    input_kwargs['encoder_hidden_states'],
                    timestep=training_batch_temp.input_kwargs['timestep'],
                    encoder_hidden_states_image=training_batch_temp.
                    input_kwargs.get('encoder_hidden_states_image'),
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    start_frame=current_start_frame)
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames),
                                                 num_denoising_steps,
                                                 device=noise.device)
        start_gradient_frame_index = max(0, num_output_frames - 21)

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[:, current_start_frame -
                                num_input_frames:current_start_frame +
                                current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])

                timestep = torch.ones([batch_size, current_num_frames],
                                      device=noise.device,
                                      dtype=torch.int64) * current_timestep

                if self.boundary_timestep is not None and current_timestep < self.boundary_timestep and self.transformer_2 is not None:
                    current_model = self.transformer_2
                    self._enable_training(self.transformer_2, self.optimizer_2)
                    self._disable_training(self.transformer, self.optimizer)
                else:
                    current_model = self.transformer
                    self._enable_training(self.transformer, self.optimizer)
                    if self.boundary_timestep is not None and self.transformer_2 is not None:
                        self._disable_training(self.transformer_2,
                                               self.optimizer_2)

                if not exit_flag:
                    with torch.no_grad():
                        # Build input kwargs
                        training_batch_temp = self._build_distill_input_kwargs(
                            noisy_input, timestep,
                            training_batch.conditional_dict, training_batch)

                        pred_flow = current_model(
                            hidden_states=training_batch_temp.
                            input_kwargs['hidden_states'],
                            encoder_hidden_states=training_batch_temp.
                            input_kwargs['encoder_hidden_states'],
                            timestep=training_batch_temp.
                            input_kwargs['timestep'],
                            encoder_hidden_states_image=training_batch_temp.
                            input_kwargs.get('encoder_hidden_states_image'),
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame *
                            self.frame_seq_length,
                            start_frame=current_start_frame).permute(
                                0, 2, 1, 3, 4)

                        denoised_pred = pred_noise_to_pred_video(
                            pred_noise=pred_flow.flatten(0, 1),
                            noise_input_latent=noisy_input.flatten(0, 1),
                            timestep=timestep,
                            scheduler=self.noise_scheduler).unflatten(
                                0, pred_flow.shape[:2])

                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.noise_scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep *
                            torch.ones([batch_size * current_num_frames],
                                       device=noise.device,
                                       dtype=torch.long)).unflatten(
                                           0, denoised_pred.shape[:2])
                else:
                    # Final prediction with gradient control
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            training_batch_temp = self._build_distill_input_kwargs(
                                noisy_input, timestep,
                                training_batch.conditional_dict, training_batch)

                            pred_flow = current_model(
                                hidden_states=training_batch_temp.
                                input_kwargs['hidden_states'],
                                encoder_hidden_states=training_batch_temp.
                                input_kwargs['encoder_hidden_states'],
                                timestep=training_batch_temp.
                                input_kwargs['timestep'],
                                encoder_hidden_states_image=training_batch_temp.
                                input_kwargs.get('encoder_hidden_states_image'),
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame *
                                self.frame_seq_length,
                                start_frame=current_start_frame).permute(
                                    0, 2, 1, 3, 4)
                    else:
                        training_batch_temp = self._build_distill_input_kwargs(
                            noisy_input, timestep,
                            training_batch.conditional_dict, training_batch)

                        pred_flow = current_model(
                            hidden_states=training_batch_temp.
                            input_kwargs['hidden_states'],
                            encoder_hidden_states=training_batch_temp.
                            input_kwargs['encoder_hidden_states'],
                            timestep=training_batch_temp.
                            input_kwargs['timestep'],
                            encoder_hidden_states_image=training_batch_temp.
                            input_kwargs.get('encoder_hidden_states_image'),
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame *
                            self.frame_seq_length,
                            start_frame=current_start_frame).permute(
                                0, 2, 1, 3, 4)

                    denoised_pred = pred_noise_to_pred_video(
                        pred_noise=pred_flow.flatten(0, 1),
                        noise_input_latent=noisy_input.flatten(0, 1),
                        timestep=timestep,
                        scheduler=self.noise_scheduler).unflatten(
                            0, pred_flow.shape[:2])
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame +
                   current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            denoised_pred = self.noise_scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                training_batch_temp = self._build_distill_input_kwargs(
                    denoised_pred, context_timestep,
                    training_batch.conditional_dict, training_batch)

                # context_timestep is 0 so we use transformer_2
                current_model = self.transformer_2 if self.transformer_2 is not None else self.transformer
                current_model(
                    hidden_states=training_batch_temp.
                    input_kwargs['hidden_states'],
                    encoder_hidden_states=training_batch_temp.
                    input_kwargs['encoder_hidden_states'],
                    timestep=training_batch_temp.input_kwargs['timestep'],
                    encoder_hidden_states_image=training_batch_temp.
                    input_kwargs.get('encoder_hidden_states_image'),
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    start_frame=current_start_frame)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Handle last 21 frames logic
        pred_image_or_video = output
        if num_input_frames > 0:
            pred_image_or_video = output[:, num_input_frames:]

        # Slice last 21 frames if we generated more
        gradient_mask = None
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                # Re-encode to get image latent
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                # Decode to video
                latent_to_decode = latent_to_decode.permute(
                    0, 2, 1, 3, 4)  # [B, C, F, H, W]

                # Apply VAE scaling and shift factors
                if isinstance(self.vae.scaling_factor, torch.Tensor):
                    latent_to_decode = latent_to_decode / self.vae.scaling_factor.to(
                        latent_to_decode.device, latent_to_decode.dtype)
                else:
                    latent_to_decode = latent_to_decode / self.vae.scaling_factor

                if hasattr(
                        self.vae,
                        "shift_factor") and self.vae.shift_factor is not None:
                    if isinstance(self.vae.shift_factor, torch.Tensor):
                        latent_to_decode += self.vae.shift_factor.to(
                            latent_to_decode.device, latent_to_decode.dtype)
                    else:
                        latent_to_decode += self.vae.shift_factor

                # Decode to pixels
                pixels = self.vae.decode(latent_to_decode)
                frame = pixels[:, :, -1:, :, :].to(
                    dtype)  # Last frame [B, C, 1, H, W]

                # Encode frame back to get image latent
                image_latent = self.vae.encode(frame).to(dtype)
                image_latent = image_latent.permute(0, 2, 1, 3,
                                                    4)  # [B, F, C, H, W]

            pred_image_or_video_last_21 = torch.cat(
                [image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        # Set up gradient mask if we generated more than minimum frames
        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21,
                                            dtype=torch.bool)
            if self.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False

        # Apply gradient masking if needed
        final_output = pred_image_or_video_last_21.to(dtype)
        if gradient_mask is not None:
            # Apply gradient masking: detach frames that shouldn't contribute gradients
            final_output = torch.where(
                gradient_mask,
                pred_image_or_video_last_21,  # Keep original values where gradient_mask is True
                pred_image_or_video_last_21.detach(
                )  # Detach where gradient_mask is False
            )

        # Store visualization data
        training_batch.dmd_latent_vis_dict["generator_timestep"] = torch.tensor(
            self.denoising_step_list[exit_flags[0]],
            dtype=torch.float32,
            device=self.device)

        # Store gradient mask information for debugging
        if gradient_mask is not None:
            training_batch.dmd_latent_vis_dict[
                "gradient_mask"] = gradient_mask.float()
            training_batch.dmd_latent_vis_dict[
                "num_generated_frames"] = torch.tensor(num_generated_frames,
                                                       dtype=torch.float32,
                                                       device=self.device)
            training_batch.dmd_latent_vis_dict["min_num_frames"] = torch.tensor(
                min_num_frames, dtype=torch.float32, device=self.device)

        # Clean up caches
        assert self.kv_cache1 is not None
        assert self.crossattn_cache is not None
        self._reset_simulation_caches(self.kv_cache1, self.crossattn_cache)

        return final_output if gradient_mask is not None else pred_image_or_video

    def _initialize_simulation_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        max_num_frames: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Initialize KV cache and cross-attention cache for multi-step simulation."""
        num_transformer_blocks = len(self.transformer.blocks)
        latent_shape = self.video_latent_shape_sp
        _, num_frames, _, height, width = latent_shape

        _, p_h, p_w = self.transformer.patch_size
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        frame_seq_length = post_patch_height * post_patch_width
        self.frame_seq_length = frame_seq_length

        # Get model configuration parameters - handle FSDP wrapping
        num_attention_heads = getattr(self.transformer, 'num_attention_heads',
                                      None)
        attention_head_dim = getattr(self.transformer, 'attention_head_dim',
                                     None)
        text_len = getattr(self.transformer, 'text_len', None)

        if max_num_frames is None:
            max_num_frames = num_frames
        num_max_frames = max(max_num_frames, num_frames)
        kv_cache_size = num_max_frames * frame_seq_length

        kv_cache = []
        for _ in range(num_transformer_blocks):
            kv_cache.append({
                "k":
                torch.zeros([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device)
            })

        # Initialize cross-attention cache
        crossattn_cache = []
        for _ in range(num_transformer_blocks):
            crossattn_cache.append({
                "k":
                torch.zeros([
                    batch_size, text_len, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, text_len, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "is_init":
                False
            })

        return kv_cache, crossattn_cache

    def _reset_simulation_caches(self, kv_cache: list[dict[str, Any]],
                                 crossattn_cache: list[dict[str, Any]]) -> None:
        """Reset KV cache and cross-attention cache to clean state."""
        if kv_cache is not None:
            for cache_dict in kv_cache:
                cache_dict["global_end_index"].fill_(0)
                cache_dict["local_end_index"].fill_(0)
                cache_dict["k"].zero_()
                cache_dict["v"].zero_()

        if crossattn_cache is not None:
            for cache_dict in crossattn_cache:
                cache_dict["is_init"] = False
                cache_dict["k"].zero_()
                cache_dict["v"].zero_()

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        # latents, encoder_hidden_states, encoder_attention_mask, infos = batch
        encoder_hidden_states = batch['text_embedding']
        encoder_attention_mask = batch['text_attention_mask']
        infos = batch['info_list']

        batch_size = encoder_hidden_states.shape[0]
        vae_config = self.training_args.pipeline_config.vae_config.arch_config
        num_channels = vae_config.z_dim
        spatial_compression_ratio = vae_config.spatial_compression_ratio

        latent_height = self.training_args.num_height // spatial_compression_ratio
        latent_width = self.training_args.num_width // spatial_compression_ratio

        latents = torch.randn(batch_size, num_channels,
                              self.training_args.num_latent_t, latent_height,
                              latent_width).to(get_local_torch_device(),
                                               dtype=torch.bfloat16)

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = encoder_hidden_states.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.infos = infos
        return training_batch

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Self-forcing training step that alternates between generator and critic training.
        """
        gradient_accumulation_steps = getattr(self.training_args,
                                              'gradient_accumulation_steps', 1)
        train_generator = (self.current_trainstep %
                           self.dfake_gen_update_ratio == 0)

        batches = []
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

        training_batch.dmd_latent_vis_dict = {}
        training_batch.fake_score_latent_vis_dict = {}

        if train_generator:
            logger.debug("Training generator at step %s",
                         self.current_trainstep)
            self.optimizer.zero_grad()
            if self.transformer_2 is not None:
                self.optimizer_2.zero_grad()
            total_generator_loss = 0.0
            generator_log_dict = {}

            for batch in batches:
                # Create a new batch with detached tensors
                batch_gen = TrainingBatch()
                for key, value in batch.__dict__.items():
                    if isinstance(value, torch.Tensor):
                        setattr(batch_gen, key, value.detach().clone())
                    elif isinstance(value, dict):
                        setattr(
                            batch_gen, key, {
                                k:
                                v.detach().clone() if isinstance(
                                    v, torch.Tensor) else copy.deepcopy(v)
                                for k, v in value.items()
                            })
                    else:
                        setattr(batch_gen, key, copy.deepcopy(value))

                generator_loss, gen_log_dict = self.generator_loss(batch_gen)
                with set_forward_context(current_timestep=batch_gen.timesteps,
                                         attn_metadata=batch_gen.attn_metadata):
                    (generator_loss / gradient_accumulation_steps).backward()
                total_generator_loss += generator_loss.detach().item()
                generator_log_dict.update(gen_log_dict)
                # Store visualization data from generator training
                if hasattr(batch_gen, 'dmd_latent_vis_dict'):
                    training_batch.dmd_latent_vis_dict.update(
                        batch_gen.dmd_latent_vis_dict)

            # Only clip gradients and step optimizer for the model that is currently training
            if hasattr(
                    self, 'train_transformer_2'
            ) and self.train_transformer_2 and self.transformer_2 is not None:
                self._clip_model_grad_norm_(batch_gen, self.transformer_2)
                self.optimizer_2.step()
                self.lr_scheduler_2.step()
            else:
                self._clip_model_grad_norm_(batch_gen, self.transformer)
                self.optimizer.step()
                self.lr_scheduler.step()

            if self.generator_ema is not None:
                if hasattr(
                        self, 'train_transformer_2'
                ) and self.train_transformer_2 and self.transformer_2 is not None:
                    # Update EMA for transformer_2 when training it
                    if self.generator_ema_2 is not None:
                        self.generator_ema_2.update(self.transformer_2)
                else:
                    self.generator_ema.update(self.transformer)

            avg_generator_loss = torch.tensor(total_generator_loss /
                                              gradient_accumulation_steps,
                                              device=self.device)
            world_group = get_world_group()
            world_group.all_reduce(avg_generator_loss,
                                   op=torch.distributed.ReduceOp.AVG)
            training_batch.generator_loss = avg_generator_loss.item()
        else:
            training_batch.generator_loss = 0.0

        logger.debug("Training critic at step %s", self.current_trainstep)
        self.fake_score_optimizer.zero_grad()
        total_critic_loss = 0.0
        critic_log_dict = {}

        for batch in batches:
            # Create a new batch with detached tensors
            batch_critic = TrainingBatch()
            for key, value in batch.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(batch_critic, key, value.detach().clone())
                elif isinstance(value, dict):
                    setattr(
                        batch_critic, key, {
                            k:
                            v.detach().clone()
                            if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                            for k, v in value.items()
                        })
                else:
                    setattr(batch_critic, key, copy.deepcopy(value))

            critic_loss, crit_log_dict = self.critic_loss(batch_critic)
            with set_forward_context(current_timestep=batch_critic.timesteps,
                                     attn_metadata=batch_critic.attn_metadata):
                (critic_loss / gradient_accumulation_steps).backward()
            total_critic_loss += critic_loss.detach().item()
            critic_log_dict.update(crit_log_dict)
            # Store visualization data from critic training
            if hasattr(batch_critic, 'fake_score_latent_vis_dict'):
                training_batch.fake_score_latent_vis_dict.update(
                    batch_critic.fake_score_latent_vis_dict)

        if self.train_fake_score_transformer_2 and self.fake_score_transformer_2 is not None:
            self._clip_model_grad_norm_(batch_critic,
                                        self.fake_score_transformer_2)
            self.fake_score_optimizer_2.step()
            self.fake_score_lr_scheduler_2.step()
        else:
            self._clip_model_grad_norm_(batch_critic,
                                        self.fake_score_transformer)
            self.fake_score_optimizer.step()
            self.fake_score_lr_scheduler.step()

        avg_critic_loss = torch.tensor(total_critic_loss /
                                       gradient_accumulation_steps,
                                       device=self.device)
        world_group = get_world_group()
        world_group.all_reduce(avg_critic_loss,
                               op=torch.distributed.ReduceOp.AVG)
        training_batch.fake_score_loss = avg_critic_loss.item()

        training_batch.total_loss = training_batch.generator_loss + training_batch.fake_score_loss
        return training_batch

    def _log_training_info(self) -> None:
        """Log self-forcing specific training information."""
        super()._log_training_info()
        logger.info("Self-forcing specific settings:")
        logger.info("  Generator update ratio: %s", self.dfake_gen_update_ratio)

    def visualize_intermediate_latents(self, training_batch: TrainingBatch,
                                       training_args: TrainingArgs, step: int):
        """Add visualization data to tracker logging and save frames to disk."""
        tracker_loss_dict: dict[str, Any] = {}

        # Debug logging
        if hasattr(training_batch, 'dmd_latent_vis_dict'):
            logger.info("DMD latent keys: %s",
                        list(training_batch.dmd_latent_vis_dict.keys()))
        if hasattr(training_batch, 'fake_score_latent_vis_dict'):
            logger.info("Fake score latent keys: %s",
                        list(training_batch.fake_score_latent_vis_dict.keys()))

        # Process generator predictions if available
        if hasattr(
                training_batch,
                'dmd_latent_vis_dict') and training_batch.dmd_latent_vis_dict:
            dmd_latents_vis_dict = training_batch.dmd_latent_vis_dict
            dmd_log_keys = [
                'generator_pred_video', 'real_score_pred_video',
                'faker_score_pred_video'
            ]

            for latent_key in dmd_log_keys:
                if latent_key in dmd_latents_vis_dict:
                    logger.info("Processing DMD latent: %s", latent_key)
                    latents = dmd_latents_vis_dict[latent_key]
                    if not isinstance(latents, torch.Tensor):
                        logger.warning("Expected tensor for %s, got %s",
                                       latent_key, type(latents))
                        continue

                    latents = latents.detach()
                    latents = latents.permute(0, 2, 1, 3, 4)

                    if isinstance(self.vae.scaling_factor, torch.Tensor):
                        latents = latents / self.vae.scaling_factor.to(
                            latents.device, latents.dtype)
                    else:
                        latents = latents / self.vae.scaling_factor

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
                    video_artifact = self.tracker.video(video,
                                                        fps=24,
                                                        format="mp4")
                    if video_artifact is not None:
                        tracker_loss_dict[f"dmd_{latent_key}"] = video_artifact
                    del video, latents

        # Process critic predictions
        if hasattr(training_batch, 'fake_score_latent_vis_dict'
                   ) and training_batch.fake_score_latent_vis_dict:
            fake_score_latents_vis_dict = training_batch.fake_score_latent_vis_dict
            fake_score_log_keys = ['generator_pred_video']

            for latent_key in fake_score_log_keys:
                if latent_key in fake_score_latents_vis_dict:
                    logger.info("Processing critic latent: %s", latent_key)
                    latents = fake_score_latents_vis_dict[latent_key]
                    if not isinstance(latents, torch.Tensor):
                        logger.warning("Expected tensor for %s, got %s",
                                       latent_key, type(latents))
                        continue

                    latents = latents.detach()
                    latents = latents.permute(0, 2, 1, 3, 4)

                    if isinstance(self.vae.scaling_factor, torch.Tensor):
                        latents = latents / self.vae.scaling_factor.to(
                            latents.device, latents.dtype)
                    else:
                        latents = latents / self.vae.scaling_factor

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
                    video_artifact = self.tracker.video(video,
                                                        fps=24,
                                                        format="mp4")
                    if video_artifact is not None:
                        tracker_loss_dict[
                            f"critic_{latent_key}"] = video_artifact
                    del video, latents

        # Log metadata
        if hasattr(
                training_batch,
                'dmd_latent_vis_dict') and training_batch.dmd_latent_vis_dict:
            if "generator_timestep" in training_batch.dmd_latent_vis_dict:
                tracker_loss_dict[
                    "generator_timestep"] = training_batch.dmd_latent_vis_dict[
                        "generator_timestep"].item()
            if "dmd_timestep" in training_batch.dmd_latent_vis_dict:
                tracker_loss_dict[
                    "dmd_timestep"] = training_batch.dmd_latent_vis_dict[
                        "dmd_timestep"].item()

        if hasattr(
                training_batch, 'fake_score_latent_vis_dict'
        ) and training_batch.fake_score_latent_vis_dict and "fake_score_timestep" in training_batch.fake_score_latent_vis_dict:
            tracker_loss_dict[
                "fake_score_timestep"] = training_batch.fake_score_latent_vis_dict[
                    "fake_score_timestep"].item()

        # Log final dict contents
        logger.info("Final tracker_loss_dict keys: %s",
                    list(tracker_loss_dict.keys()))

        if self.global_rank == 0 and tracker_loss_dict:
            self.tracker.log_artifacts(tracker_loss_dict, step)

    @profile_region("profiler_region_training_train")
    def train(self) -> None:
        """Main training loop with self-forcing specific logging."""
        assert self.training_args.seed is not None, "seed must be set"
        seed = self.training_args.seed

        # Set the same seed within each SP group to ensure reproducibility
        if self.sp_world_size > 1:
            # Use the same seed for all processes within the same SP group
            sp_group_seed = seed + (self.global_rank // self.sp_world_size)
            set_random_seed(sp_group_seed)
        else:
            set_random_seed(seed + self.global_rank)

        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed)
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(
            self.seed)
        self.validation_random_generator = torch.Generator(
            device="cpu").manual_seed(self.seed)
        logger.info("Initialized random seeds with seed: %s", seed)

        self.current_trainstep = self.init_steps

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

            if (step >= self.training_args.ema_start_step) and \
                    (self.generator_ema is None) and (self.training_args.ema_decay > 0):
                self.generator_ema = EMA_FSDP(
                    self.transformer, decay=self.training_args.ema_decay)
                logger.info("Created generator EMA at step %s with decay=%s",
                            step, self.training_args.ema_decay)

                # Create EMA for transformer_2 if it exists
                if self.transformer_2 is not None and self.generator_ema_2 is None:
                    self.generator_ema_2 = EMA_FSDP(
                        self.transformer_2, decay=self.training_args.ema_decay)
                    logger.info(
                        "Created generator EMA_2 at step %s with decay=%s",
                        step, self.training_args.ema_decay)

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
                "total_loss":
                f"{total_loss:.4f}",
                "generator_loss":
                f"{generator_loss:.4f}",
                "fake_score_loss":
                f"{fake_score_loss:.4f}",
                "step_time":
                f"{step_time:.2f}s",
                "grad_norm":
                grad_norm,
                "ema":
                "✓" if (self.generator_ema is not None and self.is_ema_ready())
                else "✗",
                "ema2":
                "✓" if (self.generator_ema_2 is not None
                        and self.is_ema_ready()) else "✗",
            })
            progress_bar.update(1)

            if self.global_rank == 0:
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
                if (step % self.dfake_gen_update_ratio == 0):
                    log_data["train_generator_loss"] = generator_loss
                if use_vsa:
                    log_data["VSA_train_sparsity"] = current_vsa_sparsity

                if self.generator_ema is not None or self.generator_ema_2 is not None:
                    log_data["ema_enabled"] = self.generator_ema is not None
                    log_data["ema_2_enabled"] = self.generator_ema_2 is not None
                    log_data["ema_decay"] = self.training_args.ema_decay
                else:
                    log_data["ema_enabled"] = False
                    log_data["ema_2_enabled"] = False

                ema_stats = self.get_ema_stats()
                log_data.update(ema_stats)

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

                self.tracker.log(log_data, step)

                if self.training_args.log_validation and step % self.training_args.validation_steps == 0 and self.training_args.log_visualization:
                    self.visualize_intermediate_latents(training_batch,
                                                        self.training_args,
                                                        step)

            if (self.training_args.training_state_checkpointing_steps > 0
                    and step %
                    self.training_args.training_state_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save training state checkpoint at step", step)
                save_distillation_checkpoint(
                    self.transformer,
                    self.fake_score_transformer,
                    self.global_rank,
                    self.training_args.output_dir,
                    step,
                    self.optimizer,
                    self.fake_score_optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                    self.fake_score_lr_scheduler,
                    self.noise_random_generator,
                    self.generator_ema,
                    # MoE support
                    generator_transformer_2=getattr(self, 'transformer_2',
                                                    None),
                    real_score_transformer_2=getattr(
                        self, 'real_score_transformer_2', None),
                    fake_score_transformer_2=getattr(
                        self, 'fake_score_transformer_2', None),
                    generator_optimizer_2=getattr(self, 'optimizer_2', None),
                    fake_score_optimizer_2=getattr(self,
                                                   'fake_score_optimizer_2',
                                                   None),
                    generator_scheduler_2=getattr(self, 'lr_scheduler_2', None),
                    fake_score_scheduler_2=getattr(self,
                                                   'fake_score_lr_scheduler_2',
                                                   None),
                    generator_ema_2=getattr(self, 'generator_ema_2', None))

                if self.transformer:
                    self.transformer.train()
                self.sp_group.barrier()

            if (self.training_args.weight_only_checkpointing_steps > 0
                    and step %
                    self.training_args.weight_only_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save weight-only checkpoint at step", step)
                save_distillation_checkpoint(
                    self.transformer,
                    self.fake_score_transformer,
                    self.global_rank,
                    self.training_args.output_dir,
                    f"{step}_weight_only",
                    only_save_generator_weight=True,
                    generator_ema=self.generator_ema,
                    # MoE support
                    generator_transformer_2=getattr(self, 'transformer_2',
                                                    None),
                    real_score_transformer_2=getattr(
                        self, 'real_score_transformer_2', None),
                    fake_score_transformer_2=getattr(
                        self, 'fake_score_transformer_2', None),
                    generator_optimizer_2=getattr(self, 'optimizer_2', None),
                    fake_score_optimizer_2=getattr(self,
                                                   'fake_score_optimizer_2',
                                                   None),
                    generator_scheduler_2=getattr(self, 'lr_scheduler_2', None),
                    fake_score_scheduler_2=getattr(self,
                                                   'fake_score_lr_scheduler_2',
                                                   None),
                    generator_ema_2=getattr(self, 'generator_ema_2', None))

                if self.training_args.use_ema and self.is_ema_ready():
                    self.save_ema_weights(self.training_args.output_dir, step)

            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                self._log_validation(self.transformer, self.training_args, step)

        self.tracker.finish()

        print("rank", self.global_rank,
              "save final training state checkpoint at step",
              self.training_args.max_train_steps)
        save_distillation_checkpoint(
            self.transformer,
            self.fake_score_transformer,
            self.global_rank,
            self.training_args.output_dir,
            self.training_args.max_train_steps,
            self.optimizer,
            self.fake_score_optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.fake_score_lr_scheduler,
            self.noise_random_generator,
            self.generator_ema,
            # MoE support
            generator_transformer_2=getattr(self, 'transformer_2', None),
            real_score_transformer_2=getattr(self, 'real_score_transformer_2',
                                             None),
            fake_score_transformer_2=getattr(self, 'fake_score_transformer_2',
                                             None),
            generator_optimizer_2=getattr(self, 'optimizer_2', None),
            fake_score_optimizer_2=getattr(self, 'fake_score_optimizer_2',
                                           None),
            generator_scheduler_2=getattr(self, 'lr_scheduler_2', None),
            fake_score_scheduler_2=getattr(self, 'fake_score_lr_scheduler_2',
                                           None),
            generator_ema_2=getattr(self, 'generator_ema_2', None))

        if self.training_args.use_ema and self.is_ema_ready():
            self.save_ema_weights(self.training_args.output_dir,
                                  self.training_args.max_train_steps)

        if envs.FASTVIDEO_TORCH_PROFILER_DIR:
            logger.info("Stopping profiler...")
            self.profiler_controller.stop()
            logger.info("Profiler stopped.")

        if get_sp_group():
            cleanup_dist_env_and_memory()
