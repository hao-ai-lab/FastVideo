# SPDX-License-Identifier: Apache-2.0
import copy
import gc
import logging
from typing import Any
from collections import deque
import time

import torch
import torch.nn.functional as F
import wandb
from tqdm.auto import tqdm

from fastvideo.fastvideo_args import TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    EMA_FSDP,
    save_distillation_checkpoint,
)
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.distributed import get_world_group
import torch.distributed as dist
import numpy as np
from fastvideo.utils import set_random_seed, is_vsa_available
import fastvideo.envs as envs
from einops import rearrange

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

        super().initialize_training_pipeline(training_args)
        self.dfake_gen_update_ratio = getattr(training_args, 'dfake_gen_update_ratio', 5)

        logger.info(f"Self-forcing generator update ratio: {self.dfake_gen_update_ratio}")

    def generator_loss(self, training_batch: TrainingBatch) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute generator loss using DMD-style approach.
        The generator tries to fool the critic (fake_score_transformer).
        """
        with set_forward_context(
                current_timestep=training_batch.timesteps,
                attn_metadata=training_batch.attn_metadata_vsa):
            if self.training_args.simulate_generator_forward:
                generator_pred_video = self._generator_multi_step_simulation_forward(training_batch)
            else:
                generator_pred_video = self._generator_forward(training_batch)

        with set_forward_context(
                current_timestep=training_batch.timesteps,
                attn_metadata=training_batch.attn_metadata):
            dmd_loss = self._dmd_forward(
                generator_pred_video=generator_pred_video,
                training_batch=training_batch
            )

        log_dict = {
            "dmdtrain_gradient_norm": torch.tensor(0.0, device=self.device)
        }
        
        return dmd_loss, log_dict

    def critic_loss(self, training_batch: TrainingBatch) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute critic loss using flow matching between noise and generator output.
        The critic learns to predict the flow from noise to the generator's output.
        """
        updated_batch, flow_matching_loss = self.faker_score_forward(training_batch)
        training_batch.fake_score_latent_vis_dict = updated_batch.fake_score_latent_vis_dict
        log_dict = {}
        
        return flow_matching_loss, log_dict

    def _generator_forward(self, training_batch: TrainingBatch) -> torch.Tensor:
        """Forward pass through generator with KV cache support for causal generation."""
        latents = training_batch.latents
        dtype = latents.dtype
        batch_size = latents.shape[0]
        
        # Step 1: Sample a timestep from denoising_step_list
        index = torch.randint(0,
                              len(self.denoising_step_list), [1],
                              device=self.device,
                              dtype=torch.long)
        timestep = self.denoising_step_list[index]
        training_batch.dmd_latent_vis_dict["generator_timestep"] = timestep

        # Step 2: Initialize KV cache and cross-attention cache for causal generation
        kv_cache, crossattn_cache = self._initialize_simulation_caches(batch_size, dtype, self.device)
        
        if getattr(self.training_args, 'validate_cache_structure', False):
            self._validate_cache_structure(kv_cache, crossattn_cache, batch_size)

        # Step 3: Add noise to latents
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

        # Step 4: Build input kwargs with KV cache support
        training_batch = self._build_distill_input_kwargs(
            noisy_latent, timestep, training_batch.conditional_dict,
            training_batch)
        
        # Step 5: Forward pass with KV cache if available
        if hasattr(self.transformer, '_forward_inference'):
            # Use causal inference forward with KV cache
            pred_noise = self.transformer(
                hidden_states=training_batch.input_kwargs['hidden_states'],
                encoder_hidden_states=training_batch.input_kwargs['encoder_hidden_states'],
                timestep=training_batch.input_kwargs['timestep'],
                encoder_hidden_states_image=training_batch.input_kwargs.get('encoder_hidden_states_image'),
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=0,  # Start from beginning for single-step
                cache_start=0
            ).permute(0, 2, 1, 3, 4)
        else:
            # Fallback to regular forward
            pred_noise = self.transformer(**training_batch.input_kwargs).permute(
                0, 2, 1, 3, 4)
        
        # Step 6: Convert noise prediction to video prediction
        pred_video = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_latent.flatten(0, 1),
            timestep=timestep,
            scheduler=self.noise_scheduler).unflatten(0, pred_noise.shape[:2])

        # Step 7: Clean up caches to prevent memory leaks
        self._reset_simulation_caches(kv_cache, crossattn_cache)

        return pred_video

    def _generator_multi_step_simulation_forward(
            self, training_batch: TrainingBatch) -> torch.Tensor:
        """Forward pass through student transformer matching inference procedure with KV cache management."""
        latents = training_batch.latents
        dtype = latents.dtype
        batch_size = latents.shape[0]

        # Step 1: Randomly sample a target timestep index from denoising_step_list
        target_timestep_idx = torch.randint(0,
                                            len(self.denoising_step_list), [1],
                                            device=self.device,
                                            dtype=torch.long)
        target_timestep_idx_int = target_timestep_idx.item()
        target_timestep = self.denoising_step_list[target_timestep_idx]

        # Step 2: Initialize KV cache and cross-attention cache for causal generation
        kv_cache, crossattn_cache = self._initialize_simulation_caches(batch_size, dtype, self.device)
        
        # Validate cache structure (can be disabled in production)
        if getattr(self.training_args, 'validate_cache_structure', False):
            self._validate_cache_structure(kv_cache, crossattn_cache, batch_size)

        # Step 3: Simulate the multi-step inference process up to the target timestep
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

        # Step 4: Determine gradient masking for frame-level selective training
        num_frames = current_noise_latents.shape[1]
        num_frame_per_block = getattr(self.training_args, 'num_frame_per_block', 4)
        independent_first_frame = getattr(self.training_args, 'independent_first_frame', False)
        enable_gradient_masking = getattr(self.training_args, 'enable_gradient_masking', True)
        gradient_mask_last_n_frames = getattr(self.training_args, 'gradient_mask_last_n_frames', 21)
        
        gradient_mask = None
        if enable_gradient_masking:
            # Calculate which frames should have gradients enabled (last N frames)
            start_gradient_frame_index = max(0, num_frames - gradient_mask_last_n_frames)
            gradient_mask = torch.ones(batch_size, num_frames, dtype=torch.bool, device=self.device)
            
            if independent_first_frame:
                # First frame is independent, disable gradients for early frames
                gradient_mask[:, :max(1, start_gradient_frame_index)] = False
            else:
                # Disable gradients for early blocks
                num_early_blocks = start_gradient_frame_index // num_frame_per_block
                gradient_mask[:, :num_early_blocks * num_frame_per_block] = False

        # Step 5: Multi-step simulation with causal block processing
        # Get num_frames_per_block from transformer config like causal_denoising.py
        num_frames_per_block = getattr(self.training_args, 'num_frame_per_block', 4)
        
        t = num_frames
        if not independent_first_frame or (independent_first_frame and hasattr(training_batch, 'image_latent') and training_batch.image_latent is not None):
            if t % num_frames_per_block != 0:
                raise ValueError(
                    "num_frames must be divisible by num_frames_per_block for causal DMD denoising"
                )
            num_blocks = t // num_frames_per_block
            block_sizes = [num_frames_per_block] * num_blocks
        else:
            if (t - 1) % num_frames_per_block != 0:
                raise ValueError(
                    "(num_frames - 1) must be divisible by num_frame_per_block when independent_first_frame=True"
                )
            num_blocks = (t - 1) // num_frames_per_block
            block_sizes = [1] + [num_frames_per_block] * num_blocks

        max_target_idx = len(self.denoising_step_list) - 1
        noise_latents = []
        noise_latent_index = target_timestep_idx_int - 1
        
        if max_target_idx > 0:
            # Run student model for all steps before the target timestep with causal blocks
            with torch.no_grad():
                start_index = 0
                current_start_frame = 0
                
                # Process each causal block
                for current_num_frames in block_sizes:
                    # Extract current block from the full latents
                    block_latents = current_noise_latents[:, start_index:start_index + current_num_frames, :, :, :]
                    
                    # Process denoising timesteps for this block
                    for step_idx in range(max_target_idx):
                        current_timestep = self.denoising_step_list[step_idx]
                        current_timestep_tensor = current_timestep * torch.ones(
                            1, device=self.device, dtype=torch.long)
                        
                        # Build input kwargs with KV cache support for this block
                        training_batch_temp = self._build_distill_input_kwargs(
                            block_latents, current_timestep_tensor,
                            training_batch.conditional_dict, training_batch)
                        
                        # Add KV cache parameters for causal generation
                        if hasattr(self.transformer, '_forward_inference'):
                            # Use causal inference forward with KV cache
                            pred_flow = self.transformer(
                                hidden_states=training_batch_temp.input_kwargs['hidden_states'],
                                encoder_hidden_states=training_batch_temp.input_kwargs['encoder_hidden_states'],
                                timestep=training_batch_temp.input_kwargs['timestep'],
                                encoder_hidden_states_image=training_batch_temp.input_kwargs.get('encoder_hidden_states_image'),
                                kv_cache=kv_cache,
                                crossattn_cache=crossattn_cache,
                                current_start=current_start_frame * 1560,  # TODO: remove hardcode
                                cache_start=0
                            ).permute(0, 2, 1, 3, 4)
                        else:
                            # Fallback to regular forward
                            pred_flow = self.transformer(**training_batch_temp.input_kwargs).permute(0, 2, 1, 3, 4)
                        
                        pred_clean = pred_noise_to_pred_video(
                            pred_noise=pred_flow.flatten(0, 1),
                            noise_input_latent=block_latents.flatten(0, 1),
                            timestep=current_timestep_tensor,
                            scheduler=self.noise_scheduler).unflatten(
                                0, pred_flow.shape[:2])

                        # Add noise for the next timestep
                        if step_idx < max_target_idx - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
                            next_timestep_tensor = next_timestep * torch.ones(
                                1, device=self.device, dtype=torch.long)
                            
                            # Generate noise for this block size
                            block_noise_shape = (batch_size, current_num_frames, *self.video_latent_shape[2:])
                            noise = torch.randn(block_noise_shape,
                                              device=self.device,
                                              dtype=pred_clean.dtype)
                            if self.sp_world_size > 1:
                                noise = rearrange(noise,
                                                "b (n t) c h w -> b n t c h w",
                                                n=self.sp_world_size).contiguous()
                                noise = noise[:, self.rank_in_sp_group, :, :, :, :]
                            block_latents = self.noise_scheduler.add_noise(
                                pred_clean.flatten(0, 1), noise.flatten(0, 1),
                                next_timestep_tensor).unflatten(0, pred_clean.shape[:2])
                        else:
                            # Final step: use clean prediction
                            block_latents = pred_clean
                    
                    # Store the processed block result
                    latent_copy = block_latents.clone()
                    noise_latents.append(latent_copy)
                    
                    # Update indices for next block
                    current_start_frame += current_num_frames
                    start_index += current_num_frames
                
                # Reconstruct full latents from blocks
                if noise_latents:
                    current_noise_latents = torch.cat(noise_latents, dim=1)

        # Step 6: Use the simulated noisy input for the final training step
        if noise_latent_index >= 0:
            assert noise_latent_index < len(
                self.denoising_step_list
            ) - 1, "noise_latent_index is out of bounds"
            noisy_input = current_noise_latents
        else:
            noisy_input = current_noise_latents_copy

        # Step 7: Final student prediction with selective gradient computation
        training_batch = self._build_distill_input_kwargs(
            noisy_input, target_timestep, training_batch.conditional_dict,
            training_batch)
        
        # Apply gradient masking during final forward pass
        if gradient_mask is not None:
            # Create a custom forward function that applies gradient masking
            def masked_forward():
                pred_flow = self.transformer(**training_batch.input_kwargs).permute(0, 2, 1, 3, 4)
                pred_video = pred_noise_to_pred_video(
                    pred_noise=pred_flow.flatten(0, 1),
                    noise_input_latent=noisy_input.flatten(0, 1),
                    timestep=target_timestep,
                    scheduler=self.noise_scheduler).unflatten(0, pred_flow.shape[:2])
                
                # Apply gradient masking: detach frames that shouldn't contribute gradients
                masked_pred_video = pred_video.clone()
                masked_pred_video = torch.where(
                    gradient_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),  # Broadcast to [B, F, 1, 1, 1]
                    pred_video,  # Keep original values where gradient_mask is True
                    pred_video.detach()  # Detach where gradient_mask is False
                )
                
                return masked_pred_video
            
            pred_video = masked_forward()
        else:
            pred_flow = self.transformer(**training_batch.input_kwargs).permute(0, 2, 1, 3, 4)
            pred_video = pred_noise_to_pred_video(
                pred_noise=pred_flow.flatten(0, 1),
                noise_input_latent=noisy_input.flatten(0, 1),
                timestep=target_timestep,
                scheduler=self.noise_scheduler).unflatten(0, pred_flow.shape[:2])
        
        training_batch.dmd_latent_vis_dict[
            "generator_timestep"] = target_timestep.float().detach()
        
        # Store gradient mask information for debugging
        if gradient_mask is not None:
            training_batch.dmd_latent_vis_dict["gradient_mask"] = gradient_mask.float()
            start_gradient_frame_index = max(0, num_frames - gradient_mask_last_n_frames)
            training_batch.dmd_latent_vis_dict["start_gradient_frame_index"] = torch.tensor(
                start_gradient_frame_index, dtype=torch.float32, device=self.device)
        
        self._reset_simulation_caches(kv_cache, crossattn_cache)
        
        return pred_video

    def _initialize_simulation_caches(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Initialize KV cache and cross-attention cache for multi-step simulation."""
        # Get actual model parameters instead of hardcoded values
        num_transformer_blocks = len(self.transformer.blocks)
        
        # Calculate frame sequence length based on input dimensions and patch size
        # From the training batch, we can get the actual latent dimensions
        latent_shape = self.video_latent_shape_sp  # This is set in _prepare_dit_inputs
        batch_size_actual, num_frames, num_channels, height, width = latent_shape
        
        # Get patch size from transformer config
        p_t, p_h, p_w = self.transformer.patch_size
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        
        # Frame sequence length is the spatial sequence length per frame
        frame_seq_length = post_patch_height * post_patch_width
        
        # Get local attention size from transformer config
        local_attn_size = getattr(self.transformer, 'local_attn_size', -1)
        
        # Get model configuration parameters - handle FSDP wrapping
        if hasattr(self.transformer, 'config'):
            config = self.transformer.config
            num_attention_heads = config.num_attention_heads
            attention_head_dim = config.attention_head_dim
            text_len = config.text_len
        else:
            # Fallback to direct attribute access for non-FSDP models
            num_attention_heads = getattr(self.transformer, 'num_attention_heads', 40)
            attention_head_dim = getattr(self.transformer, 'attention_head_dim', 128)
            text_len = getattr(self.transformer, 'text_len', 512)
        
        num_max_frames = getattr(self.training_args, "num_frames", num_frames)
        kv_cache_size = num_max_frames * frame_seq_length

        kv_cache = []
        for _ in range(num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_attention_heads, attention_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_attention_heads, attention_head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        
        # Initialize cross-attention cache
        crossattn_cache = []
        for _ in range(num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, text_len, num_attention_heads, attention_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, text_len, num_attention_heads, attention_head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        
        return kv_cache, crossattn_cache

    def _reset_simulation_caches(self, kv_cache, crossattn_cache):
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

    def _validate_cache_structure(self, kv_cache, crossattn_cache, batch_size: int):
        """Validate that cache structures are correctly initialized."""
        num_transformer_blocks = len(self.transformer.blocks)
        
        # Get model configuration parameters - handle FSDP wrapping
        if hasattr(self.transformer, 'config'):
            config = self.transformer.config
            num_attention_heads = config.num_attention_heads
            attention_head_dim = config.attention_head_dim
            text_len = config.text_len
        else:
            # Fallback to direct attribute access for non-FSDP models
            num_attention_heads = getattr(self.transformer, 'num_attention_heads', 40)
            attention_head_dim = getattr(self.transformer, 'attention_head_dim', 128)
            text_len = getattr(self.transformer, 'text_len', 512)
        
        if kv_cache is not None:
            assert len(kv_cache) == num_transformer_blocks, f"Expected {num_transformer_blocks} transformer blocks, got {len(kv_cache)}"
            for i, cache_dict in enumerate(kv_cache):
                assert "k" in cache_dict and "v" in cache_dict, f"Missing k/v in kv_cache block {i}"
                assert "global_end_index" in cache_dict and "local_end_index" in cache_dict, f"Missing indices in kv_cache block {i}"
                assert cache_dict["k"].shape[0] == batch_size, f"Batch size mismatch in kv_cache block {i}"
                assert cache_dict["v"].shape[0] == batch_size, f"Batch size mismatch in kv_cache block {i}"
                assert cache_dict["k"].shape[2] == num_attention_heads, f"Attention heads mismatch in kv_cache block {i}"
                assert cache_dict["k"].shape[3] == attention_head_dim, f"Attention head dim mismatch in kv_cache block {i}"
        
        if crossattn_cache is not None:
            assert len(crossattn_cache) == num_transformer_blocks, f"Expected {num_transformer_blocks} transformer blocks, got {len(crossattn_cache)}"
            for i, cache_dict in enumerate(crossattn_cache):
                assert "k" in cache_dict and "v" in cache_dict, f"Missing k/v in crossattn_cache block {i}"
                assert "is_init" in cache_dict, f"Missing is_init in crossattn_cache block {i}"
                assert cache_dict["k"].shape[0] == batch_size, f"Batch size mismatch in crossattn_cache block {i}"
                assert cache_dict["v"].shape[0] == batch_size, f"Batch size mismatch in crossattn_cache block {i}"
                assert cache_dict["k"].shape[1] == text_len, f"Text length mismatch in crossattn_cache block {i}"
                assert cache_dict["k"].shape[2] == num_attention_heads, f"Attention heads mismatch in crossattn_cache block {i}"
                assert cache_dict["k"].shape[3] == attention_head_dim, f"Attention head dim mismatch in crossattn_cache block {i}"

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Self-forcing training step that alternates between generator and critic training.
        """
        gradient_accumulation_steps = getattr(self.training_args, 'gradient_accumulation_steps', 1)
        train_generator = (self.current_trainstep % self.dfake_gen_update_ratio == 0)
        
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

        # Initialize visualization dictionaries
        training_batch.dmd_latent_vis_dict = {}
        training_batch.fake_score_latent_vis_dict = {}

        if train_generator:
            logger.debug(f"Training generator at step {self.current_trainstep}")
            self.optimizer.zero_grad()
            total_generator_loss = 0.0
            generator_log_dict = {}
            
            for batch in batches:
                # Create a new batch with detached tensors
                batch_gen = TrainingBatch()
                for key, value in batch.__dict__.items():
                    if isinstance(value, torch.Tensor):
                        setattr(batch_gen, key, value.detach().clone())
                    elif isinstance(value, dict):
                        setattr(batch_gen, key, {k: v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in value.items()})
                    else:
                        setattr(batch_gen, key, copy.deepcopy(value))
                
                generator_loss, gen_log_dict = self.generator_loss(batch_gen)
                with set_forward_context(
                        current_timestep=batch_gen.timesteps,
                        attn_metadata=batch_gen.attn_metadata):
                    (generator_loss / gradient_accumulation_steps).backward()
                total_generator_loss += generator_loss.detach().item()
                generator_log_dict.update(gen_log_dict)
                # Store visualization data from generator training
                if hasattr(batch_gen, 'dmd_latent_vis_dict'):
                    training_batch.dmd_latent_vis_dict.update(batch_gen.dmd_latent_vis_dict)
            
            self._clip_model_grad_norm_(batch_gen, self.transformer)
            self.optimizer.step()
            self.lr_scheduler.step()
            
            if self.generator_ema is not None:
                self.generator_ema.update(self.transformer)
            
            avg_generator_loss = torch.tensor(
                total_generator_loss / gradient_accumulation_steps,
                device=self.device
            )
            world_group = get_world_group()
            world_group.all_reduce(avg_generator_loss, op=torch.distributed.ReduceOp.AVG)
            training_batch.generator_loss = avg_generator_loss.item()
        else:
            training_batch.generator_loss = 0.0

        logger.debug(f"Training critic at step {self.current_trainstep}")
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
                    setattr(batch_critic, key, {k: v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in value.items()})
                else:
                    setattr(batch_critic, key, copy.deepcopy(value))
            
            critic_loss, crit_log_dict = self.critic_loss(batch_critic)
            with set_forward_context(
                    current_timestep=batch_critic.timesteps,
                    attn_metadata=batch_critic.attn_metadata):
                (critic_loss / gradient_accumulation_steps).backward()
            total_critic_loss += critic_loss.detach().item()
            critic_log_dict.update(crit_log_dict)
            # Store visualization data from critic training
            if hasattr(batch_critic, 'fake_score_latent_vis_dict'):
                training_batch.fake_score_latent_vis_dict.update(batch_critic.fake_score_latent_vis_dict)
        
        self._clip_model_grad_norm_(batch_critic, self.fake_score_transformer)
        self.fake_score_optimizer.step()
        self.fake_score_lr_scheduler.step()
        
        avg_critic_loss = torch.tensor(
            total_critic_loss / gradient_accumulation_steps,
            device=self.device
        )
        world_group = get_world_group()
        world_group.all_reduce(avg_critic_loss, op=torch.distributed.ReduceOp.AVG)
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
        """Add visualization data to wandb logging and save frames to disk."""
        wandb_loss_dict = {}
        
        # Debug logging
        logger.info(f"Step {step}: Starting visualization")
        if hasattr(training_batch, 'dmd_latent_vis_dict'):
            logger.info(f"DMD latent keys: {list(training_batch.dmd_latent_vis_dict.keys())}")
        if hasattr(training_batch, 'fake_score_latent_vis_dict'):
            logger.info(f"Fake score latent keys: {list(training_batch.fake_score_latent_vis_dict.keys())}")
        
        # Process generator predictions if available
        if hasattr(training_batch, 'dmd_latent_vis_dict') and training_batch.dmd_latent_vis_dict:
            dmd_latents_vis_dict = training_batch.dmd_latent_vis_dict
            dmd_log_keys = ['generator_pred_video', 'real_score_pred_video', 'faker_score_pred_video']
            
            for latent_key in dmd_log_keys:
                if latent_key in dmd_latents_vis_dict:
                    logger.info(f"Processing DMD latent: {latent_key}")
                    latents = dmd_latents_vis_dict[latent_key]
                    if not isinstance(latents, torch.Tensor):
                        logger.warning(f"Expected tensor for {latent_key}, got {type(latents)}")
                        continue
                        
                    latents = latents.detach()
                    latents = latents.permute(0, 2, 1, 3, 4)

                    if isinstance(self.vae.scaling_factor, torch.Tensor):
                        latents = latents / self.vae.scaling_factor.to(latents.device, latents.dtype)
                    else:
                        latents = latents / self.vae.scaling_factor

                    if (hasattr(self.vae, "shift_factor") and self.vae.shift_factor is not None):
                        if isinstance(self.vae.shift_factor, torch.Tensor):
                            latents += self.vae.shift_factor.to(latents.device, latents.dtype)
                        else:
                            latents += self.vae.shift_factor

                    try:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            video = self.vae.decode(latents)
                        video = (video / 2 + 0.5).clamp(0, 1)
                        video = video.cpu().float()
                        video = video.permute(0, 2, 1, 3, 4)
                        video = (video * 255).numpy().astype(np.uint8)
                        wandb_loss_dict[f"dmd_{latent_key}"] = wandb.Video(video, fps=24, format="mp4")
                        logger.info(f"Successfully processed DMD latent: {latent_key}")
                    except Exception as e:
                        logger.error(f"Error processing DMD latent {latent_key}: {str(e)}")
                    del video, latents

        # Process critic predictions
        if hasattr(training_batch, 'fake_score_latent_vis_dict') and training_batch.fake_score_latent_vis_dict:
            fake_score_latents_vis_dict = training_batch.fake_score_latent_vis_dict
            fake_score_log_keys = ['generator_pred_video']
            
            for latent_key in fake_score_log_keys:
                if latent_key in fake_score_latents_vis_dict:
                    logger.info(f"Processing critic latent: {latent_key}")
                    latents = fake_score_latents_vis_dict[latent_key]
                    if not isinstance(latents, torch.Tensor):
                        logger.warning(f"Expected tensor for {latent_key}, got {type(latents)}")
                        continue
                        
                    latents = latents.detach()
                    latents = latents.permute(0, 2, 1, 3, 4)

                    if isinstance(self.vae.scaling_factor, torch.Tensor):
                        latents = latents / self.vae.scaling_factor.to(latents.device, latents.dtype)
                    else:
                        latents = latents / self.vae.scaling_factor

                    if (hasattr(self.vae, "shift_factor") and self.vae.shift_factor is not None):
                        if isinstance(self.vae.shift_factor, torch.Tensor):
                            latents += self.vae.shift_factor.to(latents.device, latents.dtype)
                        else:
                            latents += self.vae.shift_factor

                    try:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            video = self.vae.decode(latents)
                        video = (video / 2 + 0.5).clamp(0, 1)
                        video = video.cpu().float()
                        video = video.permute(0, 2, 1, 3, 4)
                        video = (video * 255).numpy().astype(np.uint8)
                        wandb_loss_dict[f"critic_{latent_key}"] = wandb.Video(video, fps=24, format="mp4")
                        logger.info(f"Successfully processed critic latent: {latent_key}")
                    except Exception as e:
                        logger.error(f"Error processing critic latent {latent_key}: {str(e)}")
                    del video, latents

        # Log metadata
        if hasattr(training_batch, 'dmd_latent_vis_dict') and training_batch.dmd_latent_vis_dict:
            if "generator_timestep" in training_batch.dmd_latent_vis_dict:
                wandb_loss_dict["generator_timestep"] = training_batch.dmd_latent_vis_dict["generator_timestep"].item()
            if "dmd_timestep" in training_batch.dmd_latent_vis_dict:
                wandb_loss_dict["dmd_timestep"] = training_batch.dmd_latent_vis_dict["dmd_timestep"].item()

        if hasattr(training_batch, 'fake_score_latent_vis_dict') and training_batch.fake_score_latent_vis_dict:
            if "fake_score_timestep" in training_batch.fake_score_latent_vis_dict:
                wandb_loss_dict["fake_score_timestep"] = training_batch.fake_score_latent_vis_dict["fake_score_timestep"].item()

        # Log final dict contents
        logger.info(f"Final wandb_loss_dict keys: {list(wandb_loss_dict.keys())}")

        if self.global_rank == 0:
            wandb.log(wandb_loss_dict, step=step)

    def train(self) -> None:
        """Main training loop with self-forcing specific logging."""
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

        step_times = deque(maxlen=100)

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
                self.generator_ema = EMA_FSDP(self.transformer, decay=self.training_args.ema_decay)
                logger.info(f"Created generator EMA at step {step} with decay={self.training_args.ema_decay}")

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
                "ema": "✓" if (self.generator_ema is not None and self.is_ema_ready()) else "✗",
            })
            progress_bar.update(1)

            if self.global_rank == 0:
                # Prepare logging data
                log_data = {
                    "train_total_loss": total_loss,
                    "train_fake_score_loss": fake_score_loss,
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "fake_score_learning_rate": self.fake_score_lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                }
                # Only log generator loss when generator is actually trained
                if (step % self.dfake_gen_update_ratio == 0):
                    log_data["train_generator_loss"] = generator_loss
                if use_vsa:
                    log_data["VSA_train_sparsity"] = current_vsa_sparsity

                if self.generator_ema is not None:
                    log_data["ema_enabled"] = True
                    log_data["ema_decay"] = self.training_args.ema_decay
                else:
                    log_data["ema_enabled"] = False

                ema_stats = self.get_ema_stats()
                log_data.update(ema_stats)

                if training_batch.dmd_latent_vis_dict:
                    dmd_additional_logs = {
                        "generator_timestep": training_batch.dmd_latent_vis_dict["generator_timestep"].item(),
                        "dmd_timestep": training_batch.dmd_latent_vis_dict["dmd_timestep"].item(),
                    }
                    log_data.update(dmd_additional_logs)

                faker_score_additional_logs = {
                    "fake_score_timestep": training_batch.fake_score_latent_vis_dict["fake_score_timestep"].item(),
                }
                log_data.update(faker_score_additional_logs)

                wandb.log(log_data, step=step)

                # Add visualization during validation
                if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
                    if self.training_args.log_visualization:
                        self.visualize_intermediate_latents(training_batch, self.training_args, step)

            # Save training state checkpoint (for resuming training)
            if (self.training_args.training_state_checkpointing_steps > 0
                    and step % self.training_args.training_state_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save training state checkpoint at step", step)
                save_distillation_checkpoint(
                    self.transformer, self.fake_score_transformer,
                    self.global_rank, self.training_args.output_dir, step,
                    self.optimizer, self.fake_score_optimizer,
                    self.train_dataloader, self.lr_scheduler,
                    self.fake_score_lr_scheduler, self.noise_random_generator,
                    self.generator_ema)

                if self.transformer:
                    self.transformer.train()
                self.sp_group.barrier()

            # Save weight-only checkpoint
            if (self.training_args.weight_only_checkpointing_steps > 0
                    and step % self.training_args.weight_only_checkpointing_steps == 0):
                print("rank", self.global_rank,
                      "save weight-only checkpoint at step", step)
                save_distillation_checkpoint(self.transformer,
                                             self.fake_score_transformer,
                                             self.global_rank,
                                             self.training_args.output_dir,
                                             f"{step}_weight_only",
                                             only_save_generator_weight=True,
                                             generator_ema=self.generator_ema)
                
                if self.training_args.use_ema and self.is_ema_ready():
                    self.save_ema_weights(self.training_args.output_dir, step)

            if self.training_args.log_validation and step % self.training_args.validation_steps == 0:
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
            self.noise_random_generator, self.generator_ema)

        if self.training_args.use_ema and self.is_ema_ready():
            self.save_ema_weights(self.training_args.output_dir, self.training_args.max_train_steps)

        if get_sp_group():
            cleanup_dist_env_and_memory() 