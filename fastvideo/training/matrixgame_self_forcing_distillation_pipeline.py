# SPDX-License-Identifier: Apache-2.0
import sys
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterator, cast

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_matrixgame,
    pyarrow_schema_matrixgame_ode_trajectory)
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.matrixgame.kv_cache import KVCache
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler)
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
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
        # self.train_dataset_schema = pyarrow_schema_matrixgame_ode_trajectory

    @contextmanager
    def _temporary_teacher_transformer_override(
        self,
        training_args: TrainingArgs,
        cls_name: str = "MatrixGameWanModel",
    ) -> Iterator[None]:
        old_override = training_args.override_transformer_cls_name
        training_args.override_transformer_cls_name = cls_name
        try:
            yield
        finally:
            training_args.override_transformer_cls_name = old_override

    def _load_non_causal_score_transformer(
        self,
        model_path: str,
        module_key: str,
        training_args: TrainingArgs,
    ) -> torch.nn.Module:
        with self._temporary_teacher_transformer_override(training_args):
            module = self.load_module_from_path(
                model_path, "transformer", training_args
            )
        logger.info(
            "Loaded %s as %s",
            module_key,
            module.__class__.__name__,
        )
        return module

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        modules = ComposedPipelineBase.load_modules(
            self, fastvideo_args, loaded_modules
        )
        training_args = cast(TrainingArgs, fastvideo_args)

        if loaded_modules is not None and "real_score_transformer" in loaded_modules:
            self.real_score_transformer = loaded_modules["real_score_transformer"]
        elif training_args.real_score_model_path:
            logger.info(
                "Loading real score transformer from: %s",
                training_args.real_score_model_path,
            )
            self.real_score_transformer = self._load_non_causal_score_transformer(
                training_args.real_score_model_path,
                "real_score_transformer",
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
            self.fake_score_transformer = self._load_non_causal_score_transformer(
                training_args.fake_score_model_path,
                "fake_score_transformer",
                training_args,
            )
        else:
            raise ValueError(
                "fake_score_model_path is required for DMD distillation pipeline"
            )
        modules["fake_score_transformer"] = self.fake_score_transformer

        self.real_score_transformer_2 = None
        self.fake_score_transformer_2 = None
        return modules

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)

        self.denoising_step_list = torch.tensor(
            training_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long,
            device=get_local_torch_device(),
        )
        if training_args.warp_denoising_step:
            timesteps = torch.cat(
                (
                    self.noise_scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            ).to(get_local_torch_device())
            self.denoising_step_list = timesteps[
                1000 - self.denoising_step_list
            ]

        self.action_config = getattr(self.transformer, "action_config", {}) or {}
        self.use_action_module = len(self.action_config) > 0
        self.local_attn_size = getattr(self.transformer, "local_attn_size", -1)
        if self.local_attn_size <= 0:
            raise ValueError(
                "MatrixGame self-forcing requires transformer.local_attn_size > 0"
            )
        self.vae_time_compression_ratio = int(
            self.action_config.get("vae_time_compression_ratio", 4)
        )
        self.keyboard_dim_in = int(self.action_config.get("keyboard_dim_in", 0))
        self.mouse_dim_in = int(self.action_config.get("mouse_dim_in", 0))

    def _assert_matrixgame_i2v_inputs(
        self,
        training_batch: TrainingBatch,
        *,
        frame_start: int | None = None,
        frame_end: int | None = None,
        noise_input: torch.Tensor | None = None,
    ) -> None:
        image_embeds = training_batch.image_embeds
        image_latents = training_batch.image_latents
        if image_embeds is None or not isinstance(image_embeds, torch.Tensor):
            raise ValueError("training_batch.image_embeds must be a tensor")
        if image_embeds.ndim != 3:
            raise ValueError(
                "training_batch.image_embeds must have shape [B, S, D], "
                f"got {tuple(image_embeds.shape)}"
            )
        if image_latents is None or not isinstance(image_latents, torch.Tensor):
            raise ValueError("training_batch.image_latents must be a tensor")
        if image_latents.ndim != 5:
            raise ValueError(
                "training_batch.image_latents must have shape [B, C, T, H, W], "
                f"got {tuple(image_latents.shape)}"
            )
        if image_latents.shape[1] != 20:
            raise ValueError(
                "MatrixGame first_frame_latent must have 20 channels "
                f"(4 mask + 16 latent), got {image_latents.shape[1]}"
            )
        if noise_input is not None:
            if noise_input.ndim != 5:
                raise ValueError(
                    "noise_input must have shape [B, F, C, H, W], "
                    f"got {tuple(noise_input.shape)}"
                )
            if image_latents.shape[0] != noise_input.shape[0]:
                raise ValueError(
                    "Batch mismatch between image_latents and noise_input: "
                    f"{image_latents.shape[0]} vs {noise_input.shape[0]}"
                )
            if image_latents.shape[-2:] != noise_input.shape[-2:]:
                raise ValueError(
                    "Spatial mismatch between image_latents and noise_input: "
                    f"{tuple(image_latents.shape[-2:])} vs "
                    f"{tuple(noise_input.shape[-2:])}"
                )
        if frame_start is not None and frame_end is not None:
            if frame_start < 0 or frame_end <= frame_start:
                raise ValueError(
                    f"Invalid frame slice [{frame_start}, {frame_end})"
                )
            if frame_end > image_latents.shape[2]:
                raise ValueError(
                    "Requested image_latents slice exceeds available frames: "
                    f"end={frame_end}, available={image_latents.shape[2]}"
                )

    def _assert_action_inputs(
        self,
        training_batch: TrainingBatch,
        *,
        frame_end: int | None = None,
    ) -> None:
        if not self.use_action_module:
            return

        keyboard_cond = training_batch.keyboard_cond
        mouse_cond = training_batch.mouse_cond
        if keyboard_cond is None or mouse_cond is None:
            raise ValueError(
                "MatrixGame action-conditioned self-forcing requires both "
                "keyboard_cond and mouse_cond."
            )
        if keyboard_cond.ndim != 3 or mouse_cond.ndim != 3:
            raise ValueError(
                "Action tensors must have shape [B, T, D], got "
                f"keyboard={tuple(keyboard_cond.shape)}, "
                f"mouse={tuple(mouse_cond.shape)}"
            )
        if keyboard_cond.shape[0] != mouse_cond.shape[0]:
            raise ValueError(
                "keyboard_cond and mouse_cond batch sizes must match"
            )
        if self.keyboard_dim_in > 0 and keyboard_cond.shape[2] != self.keyboard_dim_in:
            raise ValueError(
                "keyboard_cond feature dim mismatch: "
                f"expected {self.keyboard_dim_in}, got {keyboard_cond.shape[2]}"
            )
        if self.mouse_dim_in > 0 and mouse_cond.shape[2] != self.mouse_dim_in:
            raise ValueError(
                "mouse_cond feature dim mismatch: "
                f"expected {self.mouse_dim_in}, got {mouse_cond.shape[2]}"
            )
        if frame_end is not None:
            required_action_frames = (
                (frame_end - 1) * self.vae_time_compression_ratio + 1
            )
            if keyboard_cond.shape[1] < required_action_frames:
                raise ValueError(
                    "keyboard_cond length is shorter than required for "
                    "latent frames: "
                    f"got={keyboard_cond.shape[1]}, "
                    f"required>={required_action_frames}"
                )
            if mouse_cond.shape[1] < required_action_frames:
                raise ValueError(
                    "mouse_cond length is shorter than required for "
                    "latent frames: "
                    f"got={mouse_cond.shape[1]}, "
                    f"required>={required_action_frames}"
                )

    def _assert_causal_timestep(
        self,
        timestep: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        name: str,
    ) -> None:
        if timestep.ndim != 2:
            raise ValueError(
                f"{name} must have shape [B, F] for MatrixGame causal rollout, "
                f"got {tuple(timestep.shape)}"
            )
        if timestep.shape != (batch_size, num_frames):
            raise ValueError(
                f"{name} shape mismatch: expected {(batch_size, num_frames)}, "
                f"got {tuple(timestep.shape)}"
            )
        if not torch.all(timestep == timestep[:, :1]):
            raise ValueError(
                f"{name} must be constant within each chunk for chunkwise rollout"
            )

    def _get_self_forcing_timestep_bounds(
        self, exit_flag: int
    ) -> tuple[int | None, int | None]:
        scheduler_timesteps = self.noise_scheduler.timesteps.to(self.device)
        denoising_timestep = torch.as_tensor(
            self.denoising_step_list[exit_flag],
            device=self.device,
            dtype=scheduler_timesteps.dtype,
        )
        denoised_timestep_from = self.num_train_timestep - torch.argmin(
            (scheduler_timesteps - denoising_timestep).abs(), dim=0
        ).item()
        if exit_flag == len(self.denoising_step_list) - 1:
            return denoised_timestep_from, 0

        next_denoising_timestep = torch.as_tensor(
            self.denoising_step_list[exit_flag + 1],
            device=self.device,
            dtype=scheduler_timesteps.dtype,
        )
        denoised_timestep_to = self.num_train_timestep - torch.argmin(
            (scheduler_timesteps - next_denoising_timestep).abs(), dim=0
        ).item()
        return denoised_timestep_from, denoised_timestep_to

    def _sample_restricted_batch_timesteps(
        self,
        batch_size: int,
        *,
        denoised_timestep_from: int | None,
        denoised_timestep_to: int | None,
    ) -> torch.Tensor:
        min_timestep = (
            denoised_timestep_to
            if denoised_timestep_to is not None
            else 0
        )
        max_timestep = (
            denoised_timestep_from
            if denoised_timestep_from is not None
            else self.num_train_timestep
        )
        if max_timestep <= min_timestep:
            max_timestep = min_timestep + 1

        timestep = torch.randint(
            min_timestep,
            max_timestep,
            [batch_size],
            device=self.device,
            dtype=torch.long,
        )
        timestep = shift_timestep(
            timestep, self.timestep_shift, self.num_train_timestep
        )
        return timestep.clamp(self.min_timestep, self.max_timestep)

    def _get_image_embeds(self, training_batch: TrainingBatch) -> torch.Tensor:
        image_embeds = training_batch.image_embeds
        if image_embeds is None:
            raise ValueError("training_batch.image_embeds must be set")
        if torch.isnan(image_embeds).any():
            raise ValueError("training_batch.image_embeds contains NaN values")
        return image_embeds.to(get_local_torch_device(), dtype=torch.bfloat16)

    def _concat_matrixgame_hidden_states(
        self,
        noise_input: torch.Tensor,
        image_latents: torch.Tensor,
    ) -> torch.Tensor:
        noisy_model_input = torch.cat(
            [noise_input, image_latents.permute(0, 2, 1, 3, 4)], dim=2
        )
        expected_channels = noise_input.shape[2] + image_latents.shape[1]
        if noisy_model_input.shape[2] != expected_channels:
            raise ValueError(
                "Unexpected MatrixGame noisy_model_input channel count: "
                f"got {noisy_model_input.shape[2]}, expected {expected_channels}"
            )
        return noisy_model_input.permute(0, 2, 1, 3, 4)

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
        num_frames = (
            (num_latent_t - 1) * temporal_compression_ratio + 1
        )
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

    def _build_generator_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        training_batch: TrainingBatch,
        *,
        frame_start: int,
        frame_end: int,
        num_frame_per_block: int,
    ) -> dict[str, torch.Tensor | None | int]:
        self._assert_matrixgame_i2v_inputs(
            training_batch,
            frame_start=frame_start,
            frame_end=frame_end,
            noise_input=noise_input,
        )
        self._assert_action_inputs(training_batch, frame_end=frame_end)
        self._assert_causal_timestep(
            timestep,
            batch_size=noise_input.shape[0],
            num_frames=noise_input.shape[1],
            name="generator_timestep",
        )

        image_embeds = self._get_image_embeds(training_batch)
        assert training_batch.image_latents is not None
        image_latents = training_batch.image_latents[
            :, :, frame_start:frame_end, :, :
        ]
        action_frame_end = (
            (frame_end - 1) * self.vae_time_compression_ratio + 1
        )
        keyboard_cond = (
            training_batch.keyboard_cond[:, :action_frame_end, :]
            if training_batch.keyboard_cond is not None
            else None
        )
        mouse_cond = (
            training_batch.mouse_cond[:, :action_frame_end, :]
            if training_batch.mouse_cond is not None
            else None
        )
        return {
            "hidden_states": self._concat_matrixgame_hidden_states(
                noise_input, image_latents
            ),
            "encoder_hidden_states": None,
            "timestep": timestep,
            "encoder_hidden_states_image": image_embeds,
            "keyboard_cond": keyboard_cond,
            "mouse_cond": mouse_cond,
            "num_frame_per_block": num_frame_per_block,
        }

    def _build_score_model_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        training_batch: TrainingBatch,
    ) -> dict[str, torch.Tensor | None]:
        self._assert_matrixgame_i2v_inputs(
            training_batch,
            noise_input=noise_input,
        )
        self._assert_action_inputs(
            training_batch, frame_end=noise_input.shape[1]
        )
        if timestep.ndim != 1 or timestep.shape[0] != noise_input.shape[0]:
            raise ValueError(
                "MatrixGame score-model timestep must have shape [B], got "
                f"{tuple(timestep.shape)}"
            )

        image_embeds = self._get_image_embeds(training_batch)
        assert training_batch.image_latents is not None
        return {
            "hidden_states": self._concat_matrixgame_hidden_states(
                noise_input, training_batch.image_latents
            ),
            "encoder_hidden_states": None,
            "timestep": timestep,
            "encoder_hidden_states_image": image_embeds,
            "keyboard_cond": training_batch.keyboard_cond,
            "mouse_cond": training_batch.mouse_cond,
        }

    def _initialize_simulation_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        max_num_frames: int | None = None,
    ) -> tuple[
        list[KVCache],
        list[dict[str, Any]],
        list[KVCache | None],
        list[KVCache | None],
    ]:
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

        if max_num_frames is None:
            max_num_frames = num_frames
        num_max_frames = max(max_num_frames, num_frames)
        kv_cache_size = self.local_attn_size * frame_seq_length

        kv_cache = []
        for _ in range(num_transformer_blocks):
            kv_cache.append(
                KVCache.zeros(
                    batch_size=batch_size,
                    cache_size=kv_cache_size,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    dtype=dtype,
                    device=device,
                ))

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
        action_config = getattr(self.transformer, 'action_config', {})
        action_blocks = action_config.get('blocks', []) if action_config else []

        action_heads_num = action_config.get('heads_num',
                                             16) if action_config else 16
        mouse_hidden_dim = action_config.get('mouse_hidden_dim',
                                             1024) if action_config else 1024
        keyboard_hidden_dim = action_config.get('keyboard_hidden_dim',
                                                1024) if action_config else 1024
        local_attn_size = getattr(
            self.transformer, "local_attn_size",
            action_config.get('local_attn_size', 6) if action_config else 6)

        mouse_head_dim = mouse_hidden_dim // action_heads_num
        keyboard_head_dim = keyboard_hidden_dim // action_heads_num

        action_cache_size = self.local_attn_size
        kv_cache_mouse = []
        kv_cache_keyboard = []
        for block_idx in range(num_transformer_blocks):
            if block_idx in action_blocks:
                kv_cache_mouse.append(
                    KVCache.zeros(
                        batch_size=batch_size * frame_seq_length,
                        cache_size=action_cache_size,
                        num_heads=action_heads_num,
                        head_dim=mouse_head_dim,
                        dtype=dtype,
                        device=device,
                    ))
                kv_cache_keyboard.append(
                    KVCache.zeros(
                        batch_size=batch_size,
                        cache_size=action_cache_size,
                        num_heads=action_heads_num,
                        head_dim=keyboard_head_dim,
                        dtype=dtype,
                        device=device,
                    ))
            else:
                kv_cache_mouse.append(None)
                kv_cache_keyboard.append(None)

        return kv_cache, crossattn_cache, kv_cache_mouse, kv_cache_keyboard

    def _reset_simulation_caches(
            self,
            kv_cache: list[KVCache],
            crossattn_cache: list[dict[str, Any]],
            kv_cache_mouse: list[KVCache | None] | None,
            kv_cache_keyboard: list[KVCache | None] | None) -> None:
        """Reset KV cache, cross-attention cache, and action caches to clean state."""
        if kv_cache is not None:
            for cache in kv_cache:
                cache.length.fill_(0)
                cache.k.zero_()
                cache.v.zero_()

        if crossattn_cache is not None:
            for cache_dict in crossattn_cache:
                cache_dict["is_init"] = False
                cache_dict["k"].zero_()
                cache_dict["v"].zero_()

        if kv_cache_mouse is not None:
            for cache in kv_cache_mouse:
                if cache is not None:
                    cache.length.fill_(0)
                    cache.k.zero_()
                    cache.v.zero_()

        if kv_cache_keyboard is not None:
            for cache in kv_cache_keyboard:
                if cache is not None:
                    cache.length.fill_(0)
                    cache.k.zero_()
                    cache.v.zero_()

    def _generator_multi_step_simulation_forward(
            self,
            training_batch: TrainingBatch) -> torch.Tensor:
        """Forward pass through student transformer matching inference procedure with KV cache management.
        
        This function is adapted from the reference self-forcing implementation's inference_with_trajectory
        and includes gradient masking logic for dynamic frame generation.
        """
        latents = training_batch.latents
        if latents is None:
            raise ValueError("training_batch.latents must be set")
        dtype = latents.dtype
        batch_size = latents.shape[0]
        self._assert_matrixgame_i2v_inputs(training_batch)
        self._assert_action_inputs(
            training_batch, frame_end=training_batch.image_latents.shape[2]
        )

        num_training_frames = getattr(self.training_args, 'num_latent_t', 21)
        if num_training_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "num_latent_t must be divisible by num_frame_per_block for "
                "MatrixGame self-forcing"
            )
        num_generated_frames = num_training_frames

        # Create noise with dynamic shape
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
        if training_batch.image_latents.shape[2] < num_frames:
            raise ValueError(
                "image_latents must cover generated latent frames: "
                f"available={training_batch.image_latents.shape[2]}, "
                f"required={num_frames}"
            )

        num_blocks = num_frames // self.num_frame_per_block
        num_output_frames = num_frames
        output = torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype)

        # Step 1: Initialize KV cache to all zeros
        cache_frames = num_generated_frames
        (self.kv_cache1, self.crossattn_cache, self.kv_cache_mouse,
         self.kv_cache_keyboard) = self._initialize_simulation_caches(
             batch_size, dtype, self.device, max_num_frames=cache_frames)

        # Step 2: Temporal denoising loop
        current_start_frame = 0
        all_num_frames = [self.num_frame_per_block] * num_blocks
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames),
                                                 num_denoising_steps,
                                                 device=noise.device)
        start_gradient_frame_index = max(0, num_output_frames - 21)

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames
            ]

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                exit_flag = (index == exit_flags[0])
                timestep = torch.ones([batch_size, current_num_frames],
                                      device=noise.device,
                                      dtype=torch.int64) * current_timestep
                self._assert_causal_timestep(
                    timestep,
                    batch_size=batch_size,
                    num_frames=current_num_frames,
                    name="rollout_timestep",
                )

                current_model = self.transformer

                if not exit_flag:
                    with torch.no_grad():
                        input_kwargs = self._build_generator_input_kwargs(
                            noisy_input,
                            timestep,
                            training_batch,
                            frame_start=current_start_frame,
                            frame_end=current_start_frame + current_num_frames,
                            num_frame_per_block=current_num_frames,
                        )
                        training_batch.noise_latents = noisy_input

                        pred_flow = current_model(
                            **input_kwargs,
                            kv_cache=self.kv_cache1,
                            kv_cache_mouse=self.kv_cache_mouse,
                            kv_cache_keyboard=self.kv_cache_keyboard,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame *
                            self.frame_seq_length,
                            start_frame=current_start_frame,
                            update_kv_cache=False).permute(
                                0, 2, 1, 3, 4)

                    denoised_pred = pred_noise_to_pred_video(
                        pred_noise=pred_flow.flatten(0, 1),
                        noise_input_latent=noisy_input.flatten(0, 1),
                        timestep=timestep,
                        scheduler=self.noise_scheduler,
                    ).unflatten(0, pred_flow.shape[:2])

                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.noise_scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Final prediction with gradient control
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            input_kwargs = self._build_generator_input_kwargs(
                                noisy_input,
                                timestep,
                                training_batch,
                                frame_start=current_start_frame,
                                frame_end=current_start_frame +
                                current_num_frames,
                                num_frame_per_block=current_num_frames,
                            )
                            training_batch.noise_latents = noisy_input

                            pred_flow = current_model(
                                **input_kwargs,
                                kv_cache=self.kv_cache1,
                                kv_cache_mouse=self.kv_cache_mouse,
                                kv_cache_keyboard=self.kv_cache_keyboard,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame *
                                self.frame_seq_length,
                                start_frame=current_start_frame,
                                update_kv_cache=False).permute(
                                    0, 2, 1, 3, 4)
                    else:
                        input_kwargs = self._build_generator_input_kwargs(
                            noisy_input,
                            timestep,
                            training_batch,
                            frame_start=current_start_frame,
                            frame_end=current_start_frame + current_num_frames,
                            num_frame_per_block=current_num_frames,
                        )
                        training_batch.noise_latents = noisy_input

                        pred_flow = current_model(
                            **input_kwargs,
                            kv_cache=self.kv_cache1,
                            kv_cache_mouse=self.kv_cache_mouse,
                            kv_cache_keyboard=self.kv_cache_keyboard,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame *
                            self.frame_seq_length,
                            start_frame=current_start_frame,
                            update_kv_cache=False).permute(
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
            self._assert_causal_timestep(
                context_timestep,
                batch_size=batch_size,
                num_frames=current_num_frames,
                name="context_timestep",
            )
            denoised_pred = self.noise_scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                input_kwargs = self._build_generator_input_kwargs(
                    denoised_pred,
                    context_timestep,
                    training_batch,
                    frame_start=current_start_frame,
                    frame_end=current_start_frame + current_num_frames,
                    num_frame_per_block=current_num_frames,
                )
                training_batch.noise_latents = denoised_pred

                self.transformer(
                    **input_kwargs,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    start_frame=current_start_frame,
                    update_kv_cache=True)

            # Step 2.4: update the start and end frame indices
            current_start_frame += current_num_frames

        denoised_timestep_from, denoised_timestep_to = (
            self._get_self_forcing_timestep_bounds(exit_flags[0])
        )
        training_batch.denoised_timestep_from = denoised_timestep_from
        training_batch.denoised_timestep_to = denoised_timestep_to

        final_output = output.to(dtype)

        # Store visualization data
        training_batch.dmd_latent_vis_dict["generator_timestep"] = torch.tensor(
            self.denoising_step_list[exit_flags[0]],
            dtype=torch.float32,
            device=self.device)

        # Clean up caches
        assert self.kv_cache1 is not None
        assert self.crossattn_cache is not None
        self._reset_simulation_caches(self.kv_cache1, self.crossattn_cache,
                                      self.kv_cache_mouse,
                                      self.kv_cache_keyboard)

        return final_output

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
        if self.get_module("image_encoder") is None:
            raise ValueError(
                "MatrixGame validation requires image_encoder to be loaded"
            )
        if self.get_module("image_processor") is None:
            raise ValueError(
                "MatrixGame validation requires image_processor to be loaded"
            )
        # Warm start validation with current transformer
        self.validation_pipeline = MatrixGameCausalDMDPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
                "vae": self.get_module("vae"),
                "scheduler": validation_scheduler,
                "image_encoder": self.get_module("image_encoder"),
                "image_processor": self.get_module("image_processor"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True)

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
        # Normal MatrixGame parquet stores the raw 16-channel first-frame
        # latent. `_prepare_dit_inputs` upgrades it to cond_concat [mask(4),
        # image_latent(16)] for both causal student rollout and non-causal
        # teacher/critic forwards.
        training_batch.image_latents = first_frame_latent.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        if training_batch.image_latents.ndim != 5:
            raise ValueError(
                "first_frame_latent must have shape [B, C, T, H, W], got "
                f"{tuple(training_batch.image_latents.shape)}"
            )
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

        if image_latents.shape[1] == 16:
            image_latents = self._build_matrixgame_cond_concat(image_latents)
        elif image_latents.shape[1] != 20:
            raise ValueError(
                "MatrixGame self-forcing expects either a 16-channel "
                "first_frame_latent from pyarrow_schema_matrixgame or a "
                "20-channel MatrixGame cond_concat tensor, got "
                f"{image_latents.shape[1]} channels."
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
        self._assert_matrixgame_i2v_inputs(training_batch)

        return training_batch

    def _build_distill_input_kwargs(
            self,
            noise_input: torch.Tensor,
            timestep: torch.Tensor,
            training_batch: TrainingBatch,
            frame_start: int | None = None,
            frame_end: int | None = None,
            num_frame_per_block: int | None = None) -> TrainingBatch:
        if frame_start is not None and frame_end is not None:
            if num_frame_per_block is None:
                num_frame_per_block = frame_end - frame_start
            input_kwargs = self._build_generator_input_kwargs(
                noise_input,
                timestep,
                training_batch,
                frame_start=frame_start,
                frame_end=frame_end,
                num_frame_per_block=num_frame_per_block,
            )
        else:
            input_kwargs = self._build_score_model_input_kwargs(
                noise_input,
                timestep,
                training_batch,
            )
        training_batch.input_kwargs = input_kwargs
        training_batch.noise_latents = noise_input

        return training_batch

    def _dmd_forward(self, generator_pred_video: torch.Tensor,
                     training_batch: TrainingBatch) -> torch.Tensor:
        """Compute DMD (Diffusion Model Distillation) loss for MatrixGame."""
        original_latent = generator_pred_video
        batch_size, num_frames = generator_pred_video.shape[:2]
        denoised_timestep_from = training_batch.denoised_timestep_from
        denoised_timestep_to = training_batch.denoised_timestep_to
        with torch.no_grad():
            critic_timestep = self._sample_restricted_batch_timesteps(
                batch_size,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
            )
            scheduler_timestep = critic_timestep.repeat_interleave(num_frames)
            noise = torch.randn_like(generator_pred_video)

            noisy_latent = self.noise_scheduler.add_noise(
                generator_pred_video.flatten(0, 1),
                noise.flatten(0, 1),
                scheduler_timestep,
            ).detach().unflatten(0, (batch_size, num_frames))

            score_input_kwargs = self._build_score_model_input_kwargs(
                noisy_latent, critic_timestep, training_batch
            )
            training_batch.noise_latents = noisy_latent

            # fake_score_transformer forward
            current_fake_score_transformer = self._get_fake_score_transformer(
                critic_timestep[:1])
            fake_score_pred_noise = current_fake_score_transformer(
                **score_input_kwargs
            ).permute(0, 2, 1, 3, 4)

            faker_score_pred_video = pred_noise_to_pred_video(
                pred_noise=fake_score_pred_noise.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=scheduler_timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, fake_score_pred_noise.shape[:2])

            # real_score_transformer forward
            current_real_score_transformer = self._get_real_score_transformer(
                critic_timestep[:1])
            real_score_pred_noise = current_real_score_transformer(
                **score_input_kwargs
            ).permute(0, 2, 1, 3, 4)

            real_score_pred_video = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise.flatten(0, 1),
                noise_input_latent=noisy_latent.flatten(0, 1),
                timestep=scheduler_timestep,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise.shape[:2])

            # No CFG for MatrixGame - use real_score_pred_video directly
            grad = (faker_score_pred_video - real_score_pred_video) / torch.abs(
                original_latent - real_score_pred_video).mean()
            grad = torch.nan_to_num(grad)

        dmd_target = (original_latent.float() - grad.float()).detach()
        dmd_loss = 0.5 * F.mse_loss(
            original_latent.float(), dmd_target, reduction="mean"
        )

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
            critic_timestep.detach(),
        })

        return dmd_loss

    def faker_score_forward(
            self, training_batch: TrainingBatch
    ) -> tuple[TrainingBatch, torch.Tensor]:
        """Forward pass for critic training with MatrixGame action conditioning."""
        with torch.no_grad(), set_forward_context(
                current_timestep=training_batch.timesteps,
                attn_metadata=training_batch.attn_metadata_vsa):
            generator_pred_video = self._generator_multi_step_simulation_forward(
                training_batch)

        batch_size, num_frames = generator_pred_video.shape[:2]
        fake_score_timestep = self._sample_restricted_batch_timesteps(
            batch_size,
            denoised_timestep_from=training_batch.denoised_timestep_from,
            denoised_timestep_to=training_batch.denoised_timestep_to,
        )
        scheduler_timestep = fake_score_timestep.repeat_interleave(num_frames)
        fake_score_noise = torch.randn_like(generator_pred_video)

        noisy_generator_pred_video = self.noise_scheduler.add_noise(
            generator_pred_video.flatten(0, 1),
            fake_score_noise.flatten(0, 1),
            scheduler_timestep,
        ).unflatten(0, (batch_size, num_frames))

        score_input_kwargs = self._build_score_model_input_kwargs(
            noisy_generator_pred_video,
            fake_score_timestep,
            training_batch,
        )
        training_batch.noise_latents = noisy_generator_pred_video

        with set_forward_context(current_timestep=training_batch.timesteps,
                                 attn_metadata=training_batch.attn_metadata):
            current_fake_score_transformer = self._get_fake_score_transformer(
                fake_score_timestep[:1]
            )
            fake_score_pred_noise = current_fake_score_transformer(
                **score_input_kwargs
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
        sampling_param.prompt = ""
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

    def _post_process_validation_frames(
            self, frames: list[np.ndarray],
            batch: ForwardBatch) -> list[np.ndarray]:
        """Apply action overlay to validation frames for WanGame.
        
        Draws keyboard (WASD) and mouse (pitch/yaw) indicators on the video frames.
        """
        # Check if action data is available
        keyboard_cond = getattr(batch, 'keyboard_cond', None)
        mouse_cond = getattr(batch, 'mouse_cond', None)

        if keyboard_cond is None and mouse_cond is None:
            return frames

        # Import overlay functions
        from fastvideo.models.dits.matrixgame.utils import (draw_keys_on_frame,
                                                            draw_mouse_on_frame)

        # Convert tensors to numpy if needed (bfloat16 -> float32 -> numpy)
        if keyboard_cond is not None:
            keyboard_cond = keyboard_cond.squeeze(
                0).cpu().float().numpy()  # (T, 6)
        if mouse_cond is not None:
            mouse_cond = mouse_cond.squeeze(0).cpu().float().numpy()  # (T, 2)

        # MatrixGame convention: keyboard [W, S, A, D, left, right], mouse [Pitch, Yaw]
        key_names = ["W", "S", "A", "D", "left", "right"]

        processed_frames = []
        for frame_idx, frame in enumerate(frames):
            frame = np.ascontiguousarray(frame.copy())

            # Draw keyboard overlay
            if keyboard_cond is not None and frame_idx < len(keyboard_cond):
                keys = {
                    key_names[i]: bool(keyboard_cond[frame_idx, i])
                    for i in range(min(len(key_names), keyboard_cond.shape[1]))
                }
                draw_keys_on_frame(frame, keys, mode='universal')

            # Draw mouse overlay
            if mouse_cond is not None and frame_idx < len(mouse_cond):
                pitch = float(mouse_cond[frame_idx, 0])
                yaw = float(mouse_cond[frame_idx, 1])
                draw_mouse_on_frame(frame, pitch, yaw)

            processed_frames.append(frame)

        return processed_frames


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