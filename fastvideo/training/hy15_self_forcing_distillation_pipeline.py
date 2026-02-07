# SPDX-License-Identifier: Apache-2.0
import copy
import os
import gc
import sys
from typing import Any
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fastvideo.distributed import (get_local_torch_device, get_world_group)
from fastvideo.fastvideo_args import TrainingArgs, FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger

from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.self_forcing_distillation_pipeline import SelfForcingDistillationPipeline
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.pipelines.basic.hunyuan15.hunyuan15_causal_dmd_pipeline import (
    Hy15CausalDMDPipeline)
from fastvideo.training.training_utils import (EMA_FSDP)
from fastvideo.utils import is_vsa_available

logger = init_logger(__name__)

vsa_available = is_vsa_available()


class Hy15SelfForcingDistillationPipeline(SelfForcingDistillationPipeline):
    """
    A self-forcing distillation pipeline for Hy15 that uses the self-forcing methodology
    with DMD for video generation.
    """
    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
        "scheduler",
        "transformer",
        "vae",
    ]

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the self-forcing training pipeline."""
        # Check if FSDP2 auto wrap is enabled - not supported for self-forcing distillation
        if os.environ.get("FASTVIDEO_FSDP2_AUTOWRAP", "0") == "1":
            raise NotImplementedError(
                "FASTVIDEO_FSDP2_AUTOWRAP is not implemented for self-forcing distillation. "
                "Please set FASTVIDEO_FSDP2_AUTOWRAP=0 or unset the environment variable."
            )

        logger.info("Initializing self-forcing distillation pipeline...")

        DistillationPipeline.initialize_training_pipeline(self, training_args)
        self.noise_scheduler.set_timesteps(num_inference_steps=1000,
                                           extra_one_step=True,
                                           device=get_local_torch_device())
        self.generator_ema: EMA_FSDP | None = None

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

        logger.info("Self-forcing generator update ratio: %s",
                    self.dfake_gen_update_ratio)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        validation_pipeline = Hy15CausalDMDPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=False)

        self.validation_pipeline = validation_pipeline

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        training_batch = super()._prepare_dit_inputs(training_batch)
        training_batch.timesteps = 0
        training_batch.image_embeds = [
            torch.zeros(1,
                        729,
                        1152,
                        device=get_local_torch_device(),
                        dtype=torch.bfloat16)
        ]
        raw_latent_shape = list(self.video_latent_shape)
        raw_latent_shape[2] = 1
        training_batch.video_latent = torch.zeros(
            tuple(raw_latent_shape),
            device=get_local_torch_device(),
            dtype=torch.bfloat16)
        return training_batch

    def _prepare_gt_inputs(self,
                           training_batch: TrainingBatch) -> TrainingBatch:
        trajectory_latents = training_batch.trajectory_latents

        start_timestep_index = torch.randint(1,
                                             len(self.denoising_step_list),
                                             (1, ),
                                             device=self.device)
        if dist.is_initialized():
            dist.broadcast(start_timestep_index, src=0)
        start_timestep_index = start_timestep_index.cpu().item()

        trajectory_latents = trajectory_latents[:, -1, :self.training_args.num_latent_t]
        training_batch.trajectory_latents = self.noise_scheduler.add_noise(
            trajectory_latents.flatten(0, 1),
            torch.randn_like(trajectory_latents.flatten(0, 1)),
            self.denoising_step_list[start_timestep_index] * torch.ones(
                [trajectory_latents.shape[0] * trajectory_latents.shape[1]],
                device=self.device,
                dtype=torch.long)).unflatten(
                    0,
                    (trajectory_latents.shape[0], trajectory_latents.shape[1]))
        del trajectory_latents
        training_batch.start_timestep_index = start_timestep_index
        return training_batch

    def _prepare_ode_init_inputs(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        trajectory_latents = training_batch.trajectory_latents

        training_batch.trajectory_latents = trajectory_latents[:, -1, :self.training_args.num_latent_t]
        # _cached_closest_idx_per_dmd = torch.tensor([0, 12, 24, 36, 50],
        #                                            dtype=torch.long).cpu()
        # training_batch.trajectory_latents = torch.index_select(
        #     trajectory_latents,
        #     dim=1,
        #     index=_cached_closest_idx_per_dmd.to(trajectory_latents.device))
        del trajectory_latents
        return training_batch

    def _prepare_context_forcing_inputs(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        trajectory_latents = training_batch.trajectory_latents

        total_num_chunks = self.training_args.num_latent_t // self.num_frame_per_block
        # Gradually increase maximum number of chunks
        # Increase max_num_chunks by 1 every 100 training steps until it reaches total_num_chunks
        max_num_chunks = min(total_num_chunks,
                             self.current_trainstep // 100 + 2)
        context_forcing_length = torch.randint(
            0, max_num_chunks,
            (1, ), device=get_local_torch_device()) * self.num_frame_per_block
        # broadcast context_forcing_length to all ranks
        if dist.is_initialized():
            dist.broadcast(context_forcing_length, src=0)
        context_forcing_length = context_forcing_length.cpu().item()
        assert context_forcing_length < self.training_args.num_latent_t, "Context forcing length is greater than the number of latent frames"
        if context_forcing_length > 0:
            training_batch.trajectory_latents = trajectory_latents[:, -1, :
                                                                   context_forcing_length]
        else:
            training_batch.trajectory_latents = None
        del trajectory_latents
        return training_batch

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

        min_num_frames = num_training_frames - 1 if self.independent_first_frame else num_training_frames

        num_generated_frames = num_training_frames
        if self.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1

        if training_batch.use_gt_trajectory and training_batch.trajectory_latents is not None:
            noise = training_batch.trajectory_latents.to(self.device,
                                                         dtype=dtype)
        else:
            noise = training_batch.latents.to(self.device, dtype=dtype)

        batch_size, num_frames, num_channels, height, width = noise.shape

        import math
        num_blocks = math.ceil(num_frames / self.num_frame_per_block)

        num_input_frames = initial_latent.shape[
            1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype)

        # Step 1: Initialize KV cache to all zeros
        cache_frames = num_generated_frames + num_input_frames
        kv_cache1, crossattn_cache = self._initialize_simulation_caches(
            batch_size, dtype, self.device, max_num_frames=cache_frames)

        current_start_frame = 0

        # Step 2: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        all_num_frames[0] = 1
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        if training_batch.use_gt_trajectory and training_batch.trajectory_latents is not None:
            start_timestep_index = training_batch.start_timestep_index
            end_timestep_index = len(self.denoising_step_list)
        else:
            start_timestep_index = 0
            end_timestep_index = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames),
                                                 start_timestep_index,
                                                 end_timestep_index,
                                                 device=noise.device)
        start_gradient_frame_index = max(0, num_output_frames - 31)

        logger.info("distillation step list: %s", self.denoising_step_list)

        # Initialize txt kv cache
        txt_kv_cache = self.transformer(
            txt_inference=True,
            vision_inference=False,
            encoder_hidden_states=training_batch.
            conditional_dict["encoder_hidden_states"],
            encoder_hidden_states_image=training_batch.image_embeds,
            encoder_attention_mask=training_batch.
            conditional_dict["encoder_attention_mask"],
            timestep=torch.zeros([batch_size], device=noise.device),
            cache_txt=True,
        )

        # Context forcing logic
        if self.training_args.use_context_forcing and training_batch.trajectory_latents is not None:
            gt_latents = training_batch.trajectory_latents
            context_forcing_length = gt_latents.shape[1]
            output[:, :context_forcing_length] = gt_latents.detach()
            num_residual_blocks = (
                self.training_args.num_latent_t -
                context_forcing_length) // self.num_frame_per_block
            assert num_residual_blocks > 0, "Number of residual blocks is less than 0"
            all_num_frames = [self.num_frame_per_block] * num_residual_blocks

            if training_batch.video_latent is not None:
                gt_latents = torch.cat([
                    gt_latents,
                    training_batch.
                    video_latent[:, current_start_frame:current_start_frame +
                                 context_forcing_length],
                    torch.zeros_like(gt_latents),
                ],
                                       dim=2)

            with torch.no_grad():
                training_batch_temp = self._build_distill_input_kwargs(
                    gt_latents,
                    torch.zeros([batch_size, context_forcing_length],
                                device=noise.device,
                                dtype=torch.int64),
                    training_batch.conditional_dict, training_batch)

                self.transformer(
                    txt_inference=False,
                    vision_inference=True,
                    hidden_states=training_batch_temp.
                    input_kwargs['hidden_states'],
                    timestep=training_batch_temp.input_kwargs['timestep'],
                    kv_cache=kv_cache1,
                    txt_kv_cache=txt_kv_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    rope_start_idx=current_start_frame)

            current_start_frame += context_forcing_length

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[:, current_start_frame -
                                num_input_frames:current_start_frame +
                                current_num_frames - num_input_frames]

            index = start_timestep_index
            current_timestep = self.denoising_step_list[index]

            assert index < len(
                self.denoising_step_list
            ), "Index is greater than the number of denoising steps"
            # Step 3.1: Spatial denoising loop
            while index < len(self.denoising_step_list):
                noisy_input_latent = noisy_input.clone()
                if training_batch.video_latent is not None:
                    noisy_input_latent = torch.cat([
                        noisy_input_latent,
                        training_batch.
                        video_latent[:,
                                     current_start_frame:current_start_frame +
                                     current_num_frames],
                        torch.zeros_like(noisy_input_latent),
                    ],
                                                   dim=2)

                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])

                timestep = torch.ones([batch_size, current_num_frames],
                                      device=noise.device,
                                      dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        # Build input kwargs
                        training_batch_temp = self._build_distill_input_kwargs(
                            noisy_input_latent, timestep,
                            training_batch.conditional_dict, training_batch)

                        pred_flow, kv_cache1 = self.transformer(
                            txt_inference=False,
                            vision_inference=True,
                            hidden_states=training_batch_temp.
                            input_kwargs['hidden_states'],
                            timestep=training_batch_temp.
                            input_kwargs['timestep'],
                            kv_cache=kv_cache1,
                            txt_kv_cache=txt_kv_cache,
                            current_start=current_start_frame *
                            self.frame_seq_length,
                            rope_start_idx=current_start_frame)
                        pred_flow = pred_flow.permute(0, 2, 1, 3, 4)

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
                                noisy_input_latent, timestep,
                                training_batch.conditional_dict, training_batch)

                            pred_flow, kv_cache1 = self.transformer(
                                txt_inference=False,
                                vision_inference=True,
                                hidden_states=training_batch_temp.
                                input_kwargs['hidden_states'],
                                timestep=training_batch_temp.
                                input_kwargs['timestep'],
                                kv_cache=kv_cache1,
                                txt_kv_cache=txt_kv_cache,
                                current_start=current_start_frame *
                                self.frame_seq_length,
                                rope_start_idx=current_start_frame)
                            pred_flow = pred_flow.permute(0, 2, 1, 3, 4)
                    else:
                        training_batch_temp = self._build_distill_input_kwargs(
                            noisy_input_latent, timestep,
                            training_batch.conditional_dict, training_batch)

                        pred_flow, kv_cache1 = self.transformer(
                            txt_inference=False,
                            vision_inference=True,
                            hidden_states=training_batch_temp.
                            input_kwargs['hidden_states'],
                            timestep=training_batch_temp.
                            input_kwargs['timestep'],
                            kv_cache=kv_cache1,
                            txt_kv_cache=txt_kv_cache,
                            current_start=current_start_frame *
                            self.frame_seq_length,
                            rope_start_idx=current_start_frame)
                        pred_flow = pred_flow.permute(0, 2, 1, 3, 4)

                    denoised_pred = pred_noise_to_pred_video(
                        pred_noise=pred_flow.flatten(0, 1),
                        noise_input_latent=noisy_input.flatten(0, 1),
                        timestep=timestep,
                        scheduler=self.noise_scheduler).unflatten(
                            0, pred_flow.shape[:2])
                    break

                index += 1
                current_timestep = self.denoising_step_list[index]

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
                if training_batch.video_latent is not None:
                    denoised_pred = torch.cat([
                        denoised_pred,
                        training_batch.
                        video_latent[:,
                                     current_start_frame:current_start_frame +
                                     current_num_frames],
                        torch.zeros_like(denoised_pred),
                    ],
                                              dim=2)

                training_batch_temp = self._build_distill_input_kwargs(
                    denoised_pred, context_timestep,
                    training_batch.conditional_dict, training_batch)

                _, kv_cache1 = self.transformer(
                    txt_inference=False,
                    vision_inference=True,
                    hidden_states=training_batch_temp.
                    input_kwargs['hidden_states'],
                    timestep=training_batch_temp.input_kwargs['timestep'],
                    kv_cache=kv_cache1,
                    txt_kv_cache=txt_kv_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    rope_start_idx=current_start_frame)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Handle last 21 frames logic
        pred_image_or_video = output
        if num_input_frames > 0:
            pred_image_or_video = output[:, num_input_frames:]

        # Slice last 21 frames if we generated more
        gradient_mask = None
        if pred_image_or_video.shape[1] > num_training_frames:
            with torch.no_grad():
                # Re-encode to get image latent
                latent_to_decode = pred_image_or_video[:, :-(
                    num_training_frames - 1), ...]
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

            pred_image_or_video_last_21 = torch.cat([
                image_latent,
                pred_image_or_video[:, -(num_training_frames - 1):, ...]
            ],
                                                    dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

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
        assert kv_cache1 is not None
        self._reset_simulation_caches(kv_cache1, crossattn_cache)

        assert pred_image_or_video.shape[
            1] == num_training_frames, f"pred_image_or_video must have {num_training_frames} frames"

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
        num_transformer_blocks = self.transformer.config.num_layers
        latent_shape = self.video_latent_shape_sp
        _, num_frames, _, height, width = latent_shape

        if isinstance(self.transformer.config.patch_size, tuple):
            ph, pw = self.transformer.config.patch_size[
                1], self.transformer.config.patch_size[2]
        elif isinstance(self.transformer.config.patch_size, int):
            ph, pw = self.transformer.config.patch_size, self.transformer.config.patch_size
        else:
            raise ValueError(
                f"Unsupported patch size type: {type(self.transformer.config.patch_size)}"
            )
        post_patch_height = height // ph
        post_patch_width = width // pw

        frame_seq_length = post_patch_height * post_patch_width
        self.frame_seq_length = frame_seq_length

        # Get model configuration parameters - handle FSDP wrapping
        num_attention_heads = self.transformer.config.num_attention_heads
        attention_head_dim = self.transformer.config.attention_head_dim

        if max_num_frames is None:
            max_num_frames = num_frames
        num_max_frames = max(max_num_frames, num_frames)
        local_attn_size = getattr(self.transformer.config, 'local_attn_size',
                                  -1)
        kv_cache_size = num_max_frames * frame_seq_length if local_attn_size == -1 else local_attn_size * frame_seq_length

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
        crossattn_cache = None

        return kv_cache, crossattn_cache

    def _step_predict_next_latent(
        self, training_batch: TrainingBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str,
                                                              torch.Tensor]]:
        latent_vis_dict: dict[str, torch.Tensor] = {}
        device = get_local_torch_device()
        target_latent = training_batch.trajectory_latents

        # Shapes: traj_latents [B, S, C, T, H, W], traj_timesteps [B, S]
        B, num_frames, num_channels, height, width = training_batch.trajectory_latents.shape

        # dmd_denoising_steps = torch.cat([
        #     self.denoising_step_list,
        #     torch.tensor([0], device=self.device, dtype=torch.long)
        # ])
        # indexes = self._get_timestep(  # [B, num_frames]
        #     0,
        #     len(dmd_denoising_steps),
        #     B,
        #     num_frames,
        #     3,
        #     uniform_timestep=False)
        # noisy_input = relevant_traj_latents[indexes]
        # latents = torch.gather(
        #     training_batch.trajectory_latents,
        #     dim=1,
        #     index=indexes.reshape(B, 1, num_frames, 1, 1,
        #                           1).expand(-1, -1, -1, num_channels, height,
        #                                     width).to(self.device)).squeeze(1)
        indexes = self._get_timestep(  # [B, num_frames]
            0,
            1000,
            B,
            num_frames,
            3,
            uniform_timestep=False)
        timestep = self.noise_scheduler.timesteps[indexes.cpu()].to(device)
        latents = self.noise_scheduler.add_noise(
            target_latent.flatten(0, 1),
            torch.randn_like(target_latent.flatten(0, 1)),
            timestep).unflatten(0, target_latent.shape[:2])
        noisy_input = torch.cat([
            latents,
            torch.zeros_like(latents),
            torch.zeros_like(latents[:, :, 0:1])
        ],
                                dim=2)
        # timestep = dmd_denoising_steps[indexes]

        logger.info("timestep: %s", timestep)
        txt_input_kwargs = {
            "txt_inference":
            True,
            "vision_inference":
            False,
            "encoder_hidden_states":
            training_batch.encoder_hidden_states,
            "encoder_hidden_states_image":
            training_batch.image_embeds,
            "encoder_attention_mask":
            training_batch.encoder_attention_mask,
            "timestep":
            torch.zeros([latents.shape[0]],
                        device=latents.device,
                        dtype=torch.bfloat16),
            "cache_txt":
            True,
        }
        with set_forward_context(current_timestep=timestep,
                                 attn_metadata=None,
                                 forward_batch=None):
            txt_kv_cache = self.transformer(**txt_input_kwargs)

        vision_input_kwargs = {
            "txt_inference": False,
            "vision_inference": True,
            "hidden_states": noisy_input.permute(0, 2, 1, 3, 4),
            "timestep": timestep.to(device, dtype=torch.bfloat16),
            "txt_kv_cache": txt_kv_cache,
        }
        # Predict noise and step the scheduler to obtain next latent
        with set_forward_context(current_timestep=timestep,
                                 attn_metadata=None,
                                 forward_batch=None):
            noise_pred, _ = self.transformer(**vision_input_kwargs)
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)

        from fastvideo.models.utils import pred_noise_to_pred_video
        pred_video = pred_noise_to_pred_video(
            pred_noise=noise_pred.flatten(0, 1),
            noise_input_latent=latents.flatten(0, 1),
            timestep=timestep.to(dtype=torch.bfloat16).flatten(0, 1),
            scheduler=self.noise_scheduler).unflatten(0, noise_pred.shape[:2])
        latent_vis_dict["pred_video"] = pred_video.permute(
            0, 2, 1, 3, 4).detach().clone().cpu()

        return pred_video, target_latent, timestep

    def _get_timestep(self,
                      min_timestep: int,
                      max_timestep: int,
                      batch_size: int,
                      num_frame: int,
                      num_frame_per_block: int,
                      uniform_timestep: bool = False) -> torch.Tensor:
        if uniform_timestep:
            timestep = torch.randint(min_timestep,
                                     max_timestep, [batch_size, 1],
                                     device=self.device,
                                     dtype=torch.long).repeat(1, num_frame)
            return timestep
        else:
            num_pad_frames = 0
            if num_frame % num_frame_per_block != 0:
                # Pad num_frame to be divisible by num_frame_per_block
                num_pad_frames = num_frame_per_block - (num_frame %
                                                        num_frame_per_block)
                num_frame += num_pad_frames
            timestep = torch.randint(min_timestep,
                                     max_timestep, [batch_size, num_frame],
                                     device=self.device,
                                     dtype=torch.long)
            # logger.info(f"individual timestep: {timestep}")
            # make the noise level the same within every block
            timestep = timestep.reshape(timestep.shape[0], -1,
                                        num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            if num_pad_frames > 0:
                timestep = timestep[:, num_pad_frames:]
            return timestep

    def _get_next_batch_2(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter_2, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            self.train_loader_iter_2 = iter(self.train_dataloader_2)
            batch = next(self.train_loader_iter_2)

        # Required fields from parquet (ODE trajectory schema)
        device = get_local_torch_device()
        encoder_hidden_states = batch['text_embedding'].to(
            device, dtype=torch.bfloat16).squeeze(0)
        encoder_hidden_states_2 = batch['text_embedding_2'].to(
            device, dtype=torch.bfloat16).squeeze(0)
        encoder_attention_mask = batch['text_mask'].to(
            device, dtype=torch.bfloat16).squeeze(0)
        encoder_attention_mask_2 = batch['text_mask_2'].to(
            device, dtype=torch.bfloat16).squeeze(0)
        infos = batch['info_list']

        if encoder_hidden_states.dim() < 3:
            prompt_embeds_list, prompt_masks_list = self.prompt_encoding_stage.encode_text(
                infos[0]["caption"],
                self.training_args,
                encoder_index=[0],
                return_attention_mask=True,
            )
            encoder_hidden_states = prompt_embeds_list[0].to(
                device, dtype=torch.bfloat16)
            encoder_attention_mask = prompt_masks_list[0].to(
                device, dtype=torch.bfloat16)

        # Trajectory tensors may include a leading singleton batch dim per row
        trajectory_latents = batch['trajectory_latents']
        if trajectory_latents.dim() == 7:
            # [B, 1, S, C, T, H, W] -> [B, S, C, T, H, W]
            trajectory_latents = trajectory_latents[:, 0]
        elif trajectory_latents.dim() == 6:
            # already [B, S, C, T, H, W]
            pass
        else:
            raise ValueError(
                f"Unexpected trajectory_latents dim: {trajectory_latents.dim()}"
            )

        trajectory_timesteps = batch['trajectory_timesteps']
        if trajectory_timesteps.dim() == 3:
            # [B, 1, S] -> [B, S]
            trajectory_timesteps = trajectory_timesteps[:, 0]
        elif trajectory_timesteps.dim() == 2:
            # [B, S]
            pass
        else:
            raise ValueError(
                f"Unexpected trajectory_timesteps dim: {trajectory_timesteps.dim()}"
            )
        # [B, S, C, T, H, W] -> [B, S, T, C, H, W] to match self-forcing
        trajectory_latents = trajectory_latents.permute(0, 1, 3, 2, 4, 5)

        batch_size = trajectory_latents.shape[0]
        vae_config = self.training_args.pipeline_config.vae_config.arch_config
        num_channels = vae_config.latent_channels
        spatial_compression_ratio = vae_config.spatial_compression_ratio

        latent_height = self.training_args.num_height // spatial_compression_ratio
        latent_width = self.training_args.num_width // spatial_compression_ratio
        training_batch.latents = torch.randn(
            batch_size, num_channels, self.training_args.num_latent_t,
            latent_height, latent_width).to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.trajectory_latents = trajectory_latents[:, :, :self.
                                                               training_args.
                                                               num_latent_t].to(
                                                                   device,
                                                                   dtype=torch.
                                                                   bfloat16)
        training_batch.trajectory_timesteps = trajectory_timesteps.to(device)
        training_batch.encoder_hidden_states = [
            encoder_hidden_states, encoder_hidden_states_2
        ]
        training_batch.encoder_attention_mask = [
            encoder_attention_mask, encoder_attention_mask_2
        ]
        training_batch.infos = infos
        return training_batch

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        # latents, encoder_hidden_states, encoder_attention_mask, infos = batch
        # encoder_hidden_states = batch['text_embedding']
        # encoder_attention_mask = batch['text_attention_mask']
        from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
        infos = batch['info_list']
        batch_prompt = ForwardBatch(
            data_type="video",
            prompt=infos[0]["caption"],
            prompt_embeds=[],
            prompt_attention_mask=[],
        )
        result_batch = self.validation_pipeline.prompt_encoding_stage(  # type: ignore
            batch_prompt, self.training_args)
        encoder_hidden_states, encoder_attention_mask = result_batch.prompt_embeds, result_batch.prompt_attention_mask
        for i in range(len(encoder_hidden_states)):
            encoder_hidden_states[i] = encoder_hidden_states[i].to(
                get_local_torch_device(), dtype=torch.bfloat16)
            encoder_attention_mask[i] = encoder_attention_mask[i].to(
                get_local_torch_device(), dtype=torch.bfloat16)

        batch_size = encoder_hidden_states[0].shape[0]
        vae_config = self.training_args.pipeline_config.vae_config.arch_config
        num_channels = vae_config.latent_channels
        spatial_compression_ratio = vae_config.spatial_compression_ratio

        latent_height = self.training_args.num_height // spatial_compression_ratio
        latent_width = self.training_args.num_width // spatial_compression_ratio

        latents = torch.randn(batch_size, num_channels,
                              self.training_args.num_latent_t, latent_height,
                              latent_width).to(get_local_torch_device(),
                                               dtype=torch.bfloat16)

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = encoder_hidden_states
        training_batch.encoder_attention_mask = encoder_attention_mask
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

        if getattr(self, "train_dataloader_2", None
                   ) is not None and not self.training_args.use_context_forcing:
            use_gt_trajectory = torch.rand(1, device=get_local_torch_device())
            if dist.is_initialized():
                dist.broadcast(use_gt_trajectory, src=0)
            training_batch.use_gt_trajectory = (use_gt_trajectory < 0.5).item()
        else:
            training_batch.use_gt_trajectory = False

        batches = []
        for _ in range(gradient_accumulation_steps):
            if training_batch.use_gt_trajectory:
                batch = self._get_next_batch_2(training_batch)
            else:
                if self.training_args.use_context_forcing:
                    batch = self._get_next_batch_2(training_batch)
                else:
                    batch = self._get_next_batch(training_batch)
            batch = self._prepare_dit_inputs(batch)
            if self.training_args.use_context_forcing:
                batch = self._prepare_context_forcing_inputs(batch)
            if training_batch.use_gt_trajectory:
                batch = self._prepare_gt_inputs(batch)
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
                if self.training_args.use_ode_init:
                    batch_gt = self._get_next_batch_2(training_batch)
                    batch_gt = self._prepare_dit_inputs(batch_gt)
                    batch_gt = self._prepare_ode_init_inputs(batch_gt)
                    batch_gt = self._build_attention_metadata(batch_gt)
                    batch_gt.attn_metadata_vsa = copy.deepcopy(
                        batch_gt.attn_metadata)
                    batch_gt_gen = TrainingBatch()
                    for key, value in batch_gt.__dict__.items():
                        if isinstance(value, torch.Tensor):
                            setattr(batch_gt_gen, key, value.detach().clone())
                        elif isinstance(value, dict):
                            setattr(
                                batch_gt_gen, key, {
                                    k:
                                    v.detach().clone() if isinstance(
                                        v, torch.Tensor) else copy.deepcopy(v)
                                    for k, v in value.items()
                                })
                        else:
                            setattr(batch_gt_gen, key, copy.deepcopy(value))

                    gt_pred_video, gt_target_latent, gt_timestep = self._step_predict_next_latent(
                        batch_gt_gen)
                    mask = gt_timestep != 0
                    # Compute loss
                    gt_loss = F.mse_loss(gt_pred_video[mask],
                                         gt_target_latent[mask],
                                         reduction="mean")
                    generator_loss += gt_loss

                    # Store visualization data from generator training
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

        if training_batch.current_timestep % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return training_batch


def main(args) -> None:
    logger.info("Starting Hy15 self-forcing distillation pipeline...")

    pipeline = Hy15SelfForcingDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)

    args = pipeline.training_args
    pipeline.train()
    logger.info("Hy15 self-forcing distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
