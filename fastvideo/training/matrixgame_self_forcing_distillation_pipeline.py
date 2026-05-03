# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_matrixgame)
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler)
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines import ComposedPipelineBase
from fastvideo.pipelines.basic.matrixgame.matrixgame_causal_dmd_pipeline import (
    MatrixGameCausalDMDPipeline)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.training.self_forcing_distillation_pipeline import (
    SelfForcingDistillationPipeline)
from fastvideo.training.training_utils import shift_timestep
from fastvideo.utils import is_vsa_available, shallow_asdict

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class MatrixGameSelfForcingDistillationPipeline(SelfForcingDistillationPipeline
                                                ):
    """
    A self-forcing distillation pipeline for MatrixGame that uses the self-forcing methodology
    with DMD for video generation.
    """
    _required_config_modules = [
        "scheduler",
        "transformer",
        "vae",
        "image_encoder",
        "image_processor",
    ]

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_matrixgame

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        modules = ComposedPipelineBase.load_modules(
            self,
            fastvideo_args,
            loaded_modules,
        )
        training_args = cast(TrainingArgs, fastvideo_args)
        old_override = training_args.override_transformer_cls_name
        training_args.override_transformer_cls_name = "MatrixGameWanModel"
        try:
            if loaded_modules is not None and "real_score_transformer" in loaded_modules:
                self.real_score_transformer = loaded_modules["real_score_transformer"]
            elif training_args.real_score_model_path:
                logger.info(
                    "Loading real score transformer from: %s",
                    training_args.real_score_model_path,
                )
                self.real_score_transformer = self.load_module_from_path(
                    training_args.real_score_model_path,
                    "transformer",
                    training_args,
                )
            else:
                raise ValueError(
                    "real_score_model_path is required for DMD distillation pipeline"
                )
            modules["real_score_transformer"] = self.real_score_transformer

            if loaded_modules is not None and "fake_score_transformer" in loaded_modules:
                self.fake_score_transformer = loaded_modules["fake_score_transformer"]
            elif training_args.fake_score_model_path:
                logger.info(
                    "Loading fake score transformer from: %s",
                    training_args.fake_score_model_path,
                )
                self.fake_score_transformer = self.load_module_from_path(
                    training_args.fake_score_model_path,
                    "transformer",
                    training_args,
                )
            else:
                raise ValueError(
                    "fake_score_model_path is required for DMD distillation pipeline"
                )
            modules["fake_score_transformer"] = self.fake_score_transformer
        finally:
            training_args.override_transformer_cls_name = old_override

        self.real_score_transformer_2 = None
        self.fake_score_transformer_2 = None
        return modules

    def _build_matrixgame_cond_concat(
        self,
        image_latents: torch.Tensor,
    ) -> torch.Tensor:
        if image_latents.ndim != 5:
            raise ValueError(
                "first_frame_latent must have shape [B, C, T, H, W], got "
                f"{tuple(image_latents.shape)}"
            )
        if image_latents.shape[1] != 16:
            raise ValueError(
                "MatrixGame cond conversion expects a 16-channel "
                f"first_frame_latent, got {image_latents.shape[1]} channels."
            )

        temporal_compression_ratio = (
            self.training_args.pipeline_config.vae_config.arch_config
            .temporal_compression_ratio
        )
        num_latent_t = image_latents.shape[2]
        num_frames = ((num_latent_t - 1) * temporal_compression_ratio + 1)
        batch_size, _, _, latent_height, latent_width = image_latents.shape

        mask_lat_size = torch.ones(
            batch_size,
            1,
            num_frames,
            latent_height,
            latent_width,
            device=image_latents.device,
            dtype=image_latents.dtype,
        )
        mask_lat_size[:, :, 1:] = 0

        first_frame_mask = mask_lat_size[:, :, :1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask,
            dim=2,
            repeats=temporal_compression_ratio,
        )
        mask_lat_size = torch.cat(
            [first_frame_mask, mask_lat_size[:, :, 1:]],
            dim=2,
        )
        mask_lat_size = mask_lat_size.view(
            batch_size,
            -1,
            temporal_compression_ratio,
            latent_height,
            latent_width,
        )
        mask_lat_size = mask_lat_size.transpose(1, 2).to(
            image_latents.device,
            dtype=image_latents.dtype,
        )

        return torch.cat([mask_lat_size, image_latents], dim=1)

    def _initialize_simulation_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
               list[dict[str, Any] | None], list[dict[str, Any] | None]]:
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

        # 1 CLS token + 256 patch tokens = 257
        text_len = 257

        action_config = getattr(self.transformer, 'action_config', {})
        action_blocks = action_config.get('blocks', []) if action_config else []
        local_attn_size = getattr(
            self.transformer, "local_attn_size",
            action_config.get('local_attn_size', 6) if action_config else 6)

        if local_attn_size <= 0:
            raise ValueError(
                "MatrixGame self-forcing requires transformer.local_attn_size > 0"
            )
        kv_cache_size = local_attn_size * frame_seq_length

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

        # Initialize action module KV caches
        action_heads_num = action_config.get('heads_num',
                                             16) if action_config else 16
        mouse_hidden_dim = action_config.get('mouse_hidden_dim',
                                             1024) if action_config else 1024
        keyboard_hidden_dim = action_config.get('keyboard_hidden_dim',
                                                1024) if action_config else 1024

        mouse_head_dim = mouse_hidden_dim // action_heads_num
        keyboard_head_dim = keyboard_hidden_dim // action_heads_num

        kv_cache_mouse = []
        kv_cache_keyboard = []
        for block_idx in range(num_transformer_blocks):
            if block_idx in action_blocks:
                kv_cache_mouse.append({
                    "k":
                    torch.zeros([
                        batch_size * frame_seq_length, local_attn_size,
                        action_heads_num, mouse_head_dim
                    ],
                                dtype=dtype,
                                device=device),
                    "v":
                    torch.zeros([
                        batch_size * frame_seq_length, local_attn_size,
                        action_heads_num, mouse_head_dim
                    ],
                                dtype=dtype,
                                device=device),
                    "global_end_index":
                    torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index":
                    torch.tensor([0], dtype=torch.long, device=device)
                })
                kv_cache_keyboard.append({
                    "k":
                    torch.zeros([
                        batch_size, local_attn_size, action_heads_num,
                        keyboard_head_dim
                    ],
                                dtype=dtype,
                                device=device),
                    "v":
                    torch.zeros([
                        batch_size, local_attn_size, action_heads_num,
                        keyboard_head_dim
                    ],
                                dtype=dtype,
                                device=device),
                    "global_end_index":
                    torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index":
                    torch.tensor([0], dtype=torch.long, device=device)
                })
            else:
                kv_cache_mouse.append(None)
                kv_cache_keyboard.append(None)

        return kv_cache, crossattn_cache, kv_cache_mouse, kv_cache_keyboard

    def _reset_simulation_caches(
            self, kv_cache: list[dict[str,
                                      Any]], crossattn_cache: list[dict[str,
                                                                        Any]],
            kv_cache_mouse: list[dict[str, Any]
                                 | None] | None,
            kv_cache_keyboard: list[dict[str, Any]
                                    | None] | None) -> None:
        """Reset KV cache, cross-attention cache, and action caches to clean state."""
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

        if kv_cache_mouse is not None:
            for cache_dict in kv_cache_mouse:
                if cache_dict is not None:
                    cache_dict["global_end_index"].fill_(0)
                    cache_dict["local_end_index"].fill_(0)
                    cache_dict["k"].zero_()
                    cache_dict["v"].zero_()

        if kv_cache_keyboard is not None:
            for cache_dict in kv_cache_keyboard:
                if cache_dict is not None:
                    cache_dict["global_end_index"].fill_(0)
                    cache_dict["local_end_index"].fill_(0)
                    cache_dict["k"].zero_()
                    cache_dict["v"].zero_()

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

        num_training_frames = getattr(self.training_args, 'num_latent_t', 21)
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
        if self.independent_first_frame:
            num_generated_frames += 1
            min_num_frames += 1

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

        if self.independent_first_frame:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        else:
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block

        num_output_frames = num_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype)

        # Step 1: Initialize KV cache to all zeros
        (self.kv_cache1, self.crossattn_cache, self.kv_cache_mouse,
         self.kv_cache_keyboard) = self._initialize_simulation_caches(
             batch_size, dtype, self.device)

        # Step 2: Temporal denoising loop
        current_start_frame = 0
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames),
                                                 num_denoising_steps,
                                                 device=noise.device)
        start_gradient_frame_index = max(0, num_output_frames - 21)

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[:, current_start_frame:current_start_frame +
                                current_num_frames]

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
                else:
                    current_model = self.transformer

                if not exit_flag:
                    with torch.no_grad():
                        # Build input kwargs
                        training_batch_temp = self._build_distill_input_kwargs(
                            noisy_input,
                            timestep,
                            training_batch.conditional_dict,
                            training_batch,
                            frame_start=current_start_frame,
                            frame_end=current_start_frame + current_num_frames,
                            num_frame_per_block=current_num_frames)

                        pred_flow = current_model(
                            **training_batch_temp.input_kwargs,
                            kv_cache=self.kv_cache1,
                            kv_cache_mouse=self.kv_cache_mouse,
                            kv_cache_keyboard=self.kv_cache_keyboard,
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
                                noisy_input,
                                timestep,
                                training_batch.conditional_dict,
                                training_batch,
                                frame_start=current_start_frame,
                                frame_end=current_start_frame +
                                current_num_frames,
                                num_frame_per_block=current_num_frames)

                            pred_flow = current_model(
                                **training_batch_temp.input_kwargs,
                                kv_cache=self.kv_cache1,
                                kv_cache_mouse=self.kv_cache_mouse,
                                kv_cache_keyboard=self.kv_cache_keyboard,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame *
                                self.frame_seq_length,
                                start_frame=current_start_frame).permute(
                                    0, 2, 1, 3, 4)
                    else:
                        training_batch_temp = self._build_distill_input_kwargs(
                            noisy_input,
                            timestep,
                            training_batch.conditional_dict,
                            training_batch,
                            frame_start=current_start_frame,
                            frame_end=current_start_frame + current_num_frames,
                            num_frame_per_block=current_num_frames)

                        pred_flow = current_model(
                            **training_batch_temp.input_kwargs,
                            kv_cache=self.kv_cache1,
                            kv_cache_mouse=self.kv_cache_mouse,
                            kv_cache_keyboard=self.kv_cache_keyboard,
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
                    denoised_pred,
                    context_timestep,
                    training_batch.conditional_dict,
                    training_batch,
                    frame_start=current_start_frame,
                    frame_end=current_start_frame + current_num_frames,
                    num_frame_per_block=current_num_frames)

                # context_timestep is 0 so we use transformer_2
                current_model = self.transformer_2 if self.transformer_2 is not None else self.transformer
                current_model(
                    **training_batch_temp.input_kwargs,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    start_frame=current_start_frame)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Handle last 21 frames logic
        pred_image_or_video = output

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
        training_batch.dmd_latent_vis_dict["generator_timestep"] = torch.as_tensor(
            self.denoising_step_list[exit_flags[0]],
            dtype=torch.float32,
            device=self.device,
        ).detach().clone()

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
        self._reset_simulation_caches(self.kv_cache1, self.crossattn_cache,
                                      self.kv_cache_mouse,
                                      self.kv_cache_keyboard)

        return final_output if gradient_mask is not None else pred_image_or_video

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True
        # Use the same flow-matching scheduler as training for consistent validation.
        validation_scheduler = SelfForcingFlowMatchScheduler(
            shift=args_copy.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True)
        validation_scheduler.set_timesteps(num_inference_steps=1000,
                                           training=True)
        # Warm start validation with current transformer
        self.validation_pipeline = MatrixGameCausalDMDPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
                "vae": self.get_module("vae"),
                "scheduler": validation_scheduler,
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True)

        if not hasattr(self.validation_pipeline, "prompt_encoding_stage"):
            # validation expects a prompt stage
            def _prompt_encoding_stage(batch: ForwardBatch,
                                       _args: TrainingArgs) -> ForwardBatch:
                if not batch.prompt_embeds:
                    batch.prompt_embeds = [None]
                if batch.prompt_attention_mask is None or not batch.prompt_attention_mask:
                    batch.prompt_attention_mask = [None]
                return batch

            self.validation_pipeline.prompt_encoding_stage = _prompt_encoding_stage  # type: ignore[attr-defined]

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        clip_feature = batch['clip_feature']
        first_frame_latent = batch['first_frame_latent']
        keyboard_cond = batch.get('keyboard_cond', None)
        mouse_cond = batch.get('mouse_cond', None)
        infos = batch['info_list']

        batch_size = clip_feature.shape[0]
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
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None
        training_batch.image_embeds = clip_feature.to(get_local_torch_device(),
                                                      dtype=torch.bfloat16)
        training_batch.image_latents = first_frame_latent.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        # Action conditioning
        if keyboard_cond is not None and keyboard_cond.numel() > 0:
            keyboard_cond_full = keyboard_cond.to(get_local_torch_device(),
                                                  dtype=torch.bfloat16)
            training_batch.keyboard_cond = keyboard_cond_full  # For Teacher/Critic (dim=6)
        else:
            training_batch.keyboard_cond = None
        if mouse_cond is not None and mouse_cond.numel() > 0:
            training_batch.mouse_cond = mouse_cond.to(get_local_torch_device(),
                                                      dtype=torch.bfloat16)
        else:
            training_batch.mouse_cond = None
        training_batch.infos = infos
        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        """Override to properly handle I2V concatenation - call parent first, then concatenate image conditioning."""
        # First, call parent method to prepare noise, timesteps, etc. for video latents
        training_batch = super()._prepare_dit_inputs(training_batch)

        assert isinstance(training_batch.image_latents, torch.Tensor)
        image_latents = training_batch.image_latents.to(
            get_local_torch_device(), dtype=torch.bfloat16)

        # cond_concat = [mask(4), image_latent(16)] with 20 channels.
        expected_cond_channels = 20
        if image_latents.shape[1] == 16:
            image_latents = self._build_matrixgame_cond_concat(image_latents)
        elif image_latents.shape[1] != expected_cond_channels:
            raise ValueError(
                "Unexpected first_frame_latent channels, "
                "Expected {expected_cond_channels} (cond_concat), got {image_latents.shape[1]}."
            )

        if self.sp_world_size > 1:
            total_frames = image_latents.shape[2]
            # Split cond latents to local SP shard only when tensor is still global.
            if total_frames == self.training_args.num_latent_t:
                if total_frames % self.sp_world_size != 0:
                    raise ValueError(
                        "image_latents temporal dim is not divisible by SP world size: "
                        f"frames={total_frames}, sp_world_size={self.sp_world_size}"
                    )
                image_latents = rearrange(image_latents,
                                          "b c (n t) h w -> b c n t h w",
                                          n=self.sp_world_size).contiguous()
                image_latents = image_latents[:, :, self.rank_in_sp_group, :, :,
                                              :]

        training_batch.image_latents = image_latents

        return training_batch

    def _build_distill_input_kwargs(
            self,
            noise_input: torch.Tensor,
            timestep: torch.Tensor,
            text_dict: dict[str, torch.Tensor] | None,
            training_batch: TrainingBatch,
            frame_start: int | None = None,
            frame_end: int | None = None,
            num_frame_per_block: int | None = None) -> TrainingBatch:
        # Image Embeds for conditioning
        image_embeds = training_batch.image_embeds
        assert image_embeds is not None
        assert torch.isnan(image_embeds).sum() == 0
        image_embeds = image_embeds.to(get_local_torch_device(),
                                       dtype=torch.bfloat16)
        image_embeds = [image_embeds]

        image_latents = training_batch.image_latents
        if frame_start is not None and frame_end is not None:
            image_latents = image_latents[:, :, frame_start:frame_end, :, :]

        vae_temporal_compression_ratio = 4
        if frame_end is not None:
            action_frame_end = (frame_end -
                                1) * vae_temporal_compression_ratio + 1
            keyboard_cond_sliced = training_batch.keyboard_cond[:, :
                                                                action_frame_end, :] if training_batch.keyboard_cond is not None else None
            mouse_cond_sliced = training_batch.mouse_cond[:, :
                                                          action_frame_end, :] if training_batch.mouse_cond is not None else None
        else:
            keyboard_cond_sliced = training_batch.keyboard_cond
            mouse_cond_sliced = training_batch.mouse_cond

        noisy_model_input = torch.cat(
            [noise_input, image_latents.permute(0, 2, 1, 3, 4)], dim=2)

        training_batch.input_kwargs = {
            "hidden_states": noisy_model_input.permute(0, 2, 1, 3,
                                                       4),  # bs, c, t, h, w
            "encoder_hidden_states": None,
            "timestep": timestep,
            "encoder_hidden_states_image": image_embeds,
            "keyboard_cond": keyboard_cond_sliced,
            "mouse_cond": mouse_cond_sliced,
            "num_frame_per_block": num_frame_per_block if num_frame_per_block is not None else self.num_frame_per_block,
        }
        training_batch.noise_latents = noise_input

        return training_batch

    def _dmd_forward(self, generator_pred_video: torch.Tensor,
                     training_batch: TrainingBatch) -> torch.Tensor:
        """Compute DMD (Diffusion Model Distillation) loss for MatrixGame."""
        original_latent = generator_pred_video
        with torch.no_grad():
            timestep = torch.randint(0,
                                     self.num_train_timestep, [1],
                                     device=self.device,
                                     dtype=torch.long)

            timestep = shift_timestep(timestep, self.timestep_shift,
                                      self.num_train_timestep)

            timestep = timestep.clamp(self.min_timestep, self.max_timestep)

            noise = torch.randn(self.video_latent_shape,
                                device=self.device,
                                dtype=generator_pred_video.dtype)

            noisy_latent = self.noise_scheduler.add_noise(
                generator_pred_video.flatten(0, 1), noise.flatten(0, 1),
                timestep).detach().unflatten(0, (generator_pred_video.shape[0],
                                                 generator_pred_video.shape[1]))

            # Non-causal models expect 1D timestep (batch_size,)
            critic_timestep = timestep.expand(noisy_latent.shape[0])

            self._build_distill_input_kwargs(
                noisy_latent, critic_timestep, None, training_batch
            )

            # fake_score_transformer forward
            current_fake_score_transformer = self._get_fake_score_transformer(
                timestep)
            fake_score_pred_noise = current_fake_score_transformer(
                **training_batch.input_kwargs
            ).permute(0, 2, 1, 3, 4)

            faker_score_pred_video = pred_noise_to_pred_video(
                pred_noise=fake_score_pred_noise.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, fake_score_pred_noise.shape[:2])

            # real_score_transformer forward
            current_real_score_transformer = self._get_real_score_transformer(
                timestep)
            real_score_pred_noise = current_real_score_transformer(
                **training_batch.input_kwargs
            ).permute(0, 2, 1, 3, 4)

            real_score_pred_video = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise.shape[:2])

            # No CFG for MatrixGame - use real_score_pred_video directly
            grad = (faker_score_pred_video - real_score_pred_video) / torch.abs(
                original_latent - real_score_pred_video).mean()
            grad = torch.nan_to_num(grad)

        dmd_loss = 0.5 * F.mse_loss(
            original_latent.float(),
            (original_latent.float() - grad.float()).detach())

        training_batch.dmd_latent_vis_dict.update({
            "training_batch_dmd_fwd_clean_latent":
            training_batch.latents,
            "generator_pred_video":
            original_latent.detach(),
            "real_score_pred_video":
            real_score_pred_video.detach(),
            "faker_score_pred_video":
            faker_score_pred_video.detach(),
            "dmd_timestep":
            timestep.detach(),
        })

        return dmd_loss

    def faker_score_forward(
            self, training_batch: TrainingBatch
    ) -> tuple[TrainingBatch, torch.Tensor]:
        """Forward pass for critic training with MatrixGame action conditioning."""
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

        fake_score_timestep = shift_timestep(fake_score_timestep,
                                             self.timestep_shift,
                                             self.num_train_timestep)

        fake_score_timestep = fake_score_timestep.clamp(self.min_timestep,
                                                        self.max_timestep)

        fake_score_noise = torch.randn(self.video_latent_shape,
                                       device=self.device,
                                       dtype=generator_pred_video.dtype)

        noisy_generator_pred_video = self.noise_scheduler.add_noise(
            generator_pred_video.flatten(0, 1),
            fake_score_noise.flatten(0, 1), fake_score_timestep).unflatten(
                0,
                (generator_pred_video.shape[0], generator_pred_video.shape[1]))

        # Non-causal critic expects 1D timestep (batch_size,), not 2D (batch_size, num_frames).
        expanded_fake_score_timestep = fake_score_timestep.expand(
            noisy_generator_pred_video.shape[0])

        self._build_distill_input_kwargs(
            noisy_generator_pred_video, expanded_fake_score_timestep, None, training_batch
        )

        with set_forward_context(current_timestep=training_batch.timesteps,
                                 attn_metadata=training_batch.attn_metadata):
            current_fake_score_transformer = self._get_fake_score_transformer(fake_score_timestep)
            fake_score_pred_noise = current_fake_score_transformer(
                **training_batch.input_kwargs
            ).permute(0, 2, 1, 3, 4)

        target = fake_score_noise - generator_pred_video
        flow_matching_loss = torch.mean((fake_score_pred_noise - target)**2)

        training_batch.fake_score_latent_vis_dict = {
            "training_batch_fakerscore_fwd_clean_latent":
            training_batch.latents,
            "generator_pred_video": generator_pred_video,
            "fake_score_timestep": fake_score_timestep,
        }

        return training_batch, flow_matching_loss

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict[str, Any],
                                  num_inference_steps: int) -> ForwardBatch:
        sampling_param.prompt = validation_batch['prompt']
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.image_path = validation_batch.get(
            'image_path') or validation_batch.get('video_path')
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        assert self.seed is not None
        sampling_param.seed = self.seed

        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t -
                      1) * temporal_compression_factor + 1
        sampling_param.num_frames = num_frames
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )
        if "image" in validation_batch and validation_batch["image"] is not None:
            batch.pil_image = validation_batch["image"]

        if "keyboard_cond" in validation_batch and validation_batch[
                "keyboard_cond"] is not None:
            keyboard_cond = validation_batch["keyboard_cond"]
            keyboard_cond = torch.tensor(keyboard_cond, dtype=torch.bfloat16)
            keyboard_cond = keyboard_cond.unsqueeze(0)
            batch.keyboard_cond = keyboard_cond

        if "mouse_cond" in validation_batch and validation_batch[
                "mouse_cond"] is not None:
            mouse_cond = validation_batch["mouse_cond"]
            mouse_cond = torch.tensor(mouse_cond, dtype=torch.bfloat16)
            mouse_cond = mouse_cond.unsqueeze(0)
            batch.mouse_cond = mouse_cond

        return batch


def main(args) -> None:
    logger.info("Starting MatrixGame self-forcing distillation pipeline...")

    pipeline = MatrixGameSelfForcingDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)

    args = pipeline.training_args
    pipeline.train()
    logger.info("MatrixGame self-forcing distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)