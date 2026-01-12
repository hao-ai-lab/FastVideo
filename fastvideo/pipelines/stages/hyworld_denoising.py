# SPDX-License-Identifier: Apache-2.0
"""
HyWorld denoising stage for chunk-based video generation with context frame selection.

This stage implements the bi_rollout denoising logic from HyWorld, which processes
video generation in chunks with camera-aware context frame selection for temporal consistency.
"""

import inspect
import weakref
from collections.abc import Iterable
from typing import Any

import torch
from einops import repeat
from tqdm.auto import tqdm

from fastvideo.distributed import get_local_torch_device, get_world_group
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import dict_to_3d_list
from fastvideo.models.dits.hyworld.retrieval_context import (
    generate_points_in_sphere,
    select_aligned_memory_frames
)


logger = init_logger(__name__)


class HyWorldDenoisingStage(DenoisingStage):
    """
    Denoising stage for HyWorld-style chunk-based video generation.
    
    This stage implements bi_rollout denoising with:
    - Chunk-based processing (generates video in chunks, e.g., 4 frames at a time)    - Context frame selection based on camera view alignment
    - 3D-aware generation using view matrices and camera intrinsics
    - Support for action conditioning
    - Dual timestep handling (context frames use different timestep than current frames)    - Context frame selection based on camera view alignment
    """

    def __init__(
        self,
        transformer,
        scheduler,
        pipeline=None,
        transformer_2=None,
        vae=None,
    ) -> None:
        super().__init__(transformer, scheduler, pipeline, transformer_2, vae)

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Run the chunk-based denoising loop with context frame selection.
        
        Args:
            batch: The current batch information. Must contain:
                - viewmats: torch.Tensor | None - Camera view matrices (B, T, 4, 4)
                - Ks: torch.Tensor | None - Camera intrinsics (B, T, 3, 3)
                - action: torch.Tensor | None - Action conditioning (B, T)
                - chunk_latent_frames: int - Number of frames per chunk (default: 4)
                These can be passed via batch.extra dict or as direct attributes.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with denoised latents.
        """

        pipeline = self.pipeline() if self.pipeline else None
        if not fastvideo_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(
                fastvideo_args.model_paths["transformer"], fastvideo_args
            )
            if pipeline:
                pipeline.add_module("transformer", self.transformer)
            fastvideo_args.model_loaded["transformer"] = True

        # Extract HyWorld-specific parameters from batch.extra or batch attributes
        viewmats = getattr(batch, "viewmats", None) or batch.extra.get("viewmats", None)
        Ks = getattr(batch, "Ks", None) or batch.extra.get("Ks", None)
        action = getattr(batch, "action", None) or batch.extra.get("action", None)
        chunk_latent_frames = (
            getattr(batch, "chunk_latent_frames", None)
            or batch.extra.get("chunk_latent_frames", 4)
        )
        stabilization_level = 15
        points_local = (
            getattr(batch, "points_local", None)
            or batch.extra.get("points_local", None)
        )

        if viewmats is None or Ks is None:
            raise ValueError(
                "viewmats and Ks are required for HyWorld denoising. "
                "Please provide them in batch.extra['viewmats'] and batch.extra['Ks']"
            )

        # Prepare extra step kwargs for scheduler
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {
                "generator": batch.generator,
                "eta": batch.eta,
            },
        )

        # Setup precision and autocast settings
        target_dtype = torch.bfloat16
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not fastvideo_args.disable_autocast

        # Get timesteps and calculate warmup steps
        timesteps = batch.timesteps
        if timesteps is None:
            raise ValueError("Timesteps must be provided")
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.scheduler.order
        )

        # Prepare image latents and embeddings for I2V generation
        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert not torch.isnan(image_embeds[0]).any(), "image_embeds contains nan"
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        image_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_image": image_embeds,
                "mask_strategy": dict_to_3d_list(
                    None, t_max=50, l_max=60, h_max=24),
            },
        )

        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

        neg_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_2": batch.clip_embedding_neg,
                "encoder_attention_mask": batch.negative_attention_mask,
            },
        )

        action_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "mouse_cond": batch.mouse_cond,
                "keyboard_cond": batch.keyboard_cond,
            },
        )

        # Get latents and embeddings
        latents = batch.latents
        prompt_embeds = batch.prompt_embeds
        assert not torch.isnan(prompt_embeds[0]).any(), "prompt_embeds contains nan"
        if batch.do_classifier_free_guidance:
            neg_prompt_embeds = batch.negative_prompt_embeds
            assert neg_prompt_embeds is not None
            assert not torch.isnan(
                neg_prompt_embeds[0]
            ).any(), "neg_prompt_embeds contains nan"

        latent_model_input = latents.to(target_dtype)
        assert latent_model_input.shape[0] == 1, "only support batch size 1"
        device = get_local_torch_device()

        # Generate local points if not provided
        if points_local is None:
            points_local = generate_points_in_sphere(50000, 8.0).to(device)
        else:
            points_local = points_local.to(device)

        # Prepare conditional latents if needed (for i2v)
        latent_frames = latents.shape[2]
        cond_latents = batch.image_latent.repeat(1, 1, latent_frames, 1, 1)
        cond_latents[:, :, 1:, :, :] = 0.0 if cond_latents is not None else None

        # Calculate chunk configuration
        latent_frames = latents.shape[2]
        chunk_num = latent_frames // chunk_latent_frames

        # Initialize lists for ODE trajectory
        trajectory_timesteps: list[torch.Tensor] = []
        trajectory_latents: list[torch.Tensor] = []

        # Main chunk processing loop
        for chunk_i in range(chunk_num):
            if chunk_i > 0:
                # Select context frames based on camera alignment
                current_frame_idx = chunk_i * chunk_latent_frames

                selected_frame_indices = []
                for chunk_start_idx in range(
                    current_frame_idx,
                    current_frame_idx + chunk_latent_frames,
                    4,  # Process every 4 frames
                ):
                    selected_history_frame_id = select_aligned_memory_frames(
                        viewmats[0].cpu().detach().numpy(),
                        chunk_start_idx,
                        memory_frames=20,
                        temporal_context_size=12,
                        pred_latent_size=4,
                        points_local=points_local,
                        device=device,
                    )
                    selected_frame_indices.extend(selected_history_frame_id)

                selected_frame_indices = sorted(list(set(selected_frame_indices)))
                # Remove current chunk frames from context
                to_remove = list(
                    range(current_frame_idx, current_frame_idx + chunk_latent_frames)
                )
                selected_frame_indices = [
                    x for x in selected_frame_indices if x not in to_remove
                ]

                # Extract context frames
                context_latents = latents[:, :, selected_frame_indices]
                context_w2c = viewmats[:, selected_frame_indices]
                context_Ks = Ks[:, selected_frame_indices]
                context_action = action[:, selected_frame_indices]
                
            self.scheduler.set_timesteps(num_inference_steps, device=device)

            # Define chunk boundaries
            start_idx = chunk_i * chunk_latent_frames
            end_idx = chunk_i * chunk_latent_frames + chunk_latent_frames

            # Denoising loop for this chunk
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):

                    if chunk_i == 0:
                        # First chunk: standard processing
                        timestep_input = torch.full(
                            (chunk_latent_frames,),
                            t.item(),
                            device=device,
                            dtype=timesteps.dtype,
                        )
                        latent_model_input = latents[:, :, :chunk_latent_frames]
                        cond_latents_input = cond_latents[
                            :, :, :chunk_latent_frames
                        ]
                    else:
                        # Subsequent chunks: use context frames with different timesteps
                        t_now = torch.full(
                            (chunk_latent_frames,),
                            t.item(),
                            device=device,
                            dtype=timesteps.dtype,
                        )
                        t_ctx = torch.full(
                            (len(selected_frame_indices),),
                            stabilization_level - 1,
                            device=device,
                            dtype=timesteps.dtype,
                        )
                        timestep_input = torch.cat([t_ctx, t_now], dim=0)

                        latents_model_now = latents[:, :, start_idx:end_idx]
                        latent_model_input = torch.cat(
                            [context_latents, latents_model_now], dim=2
                        )
                        cond_latents_input = cond_latents[
                            :, :, :latent_model_input.shape[2]
                        ]

                    # Prepare viewmats, Ks, action for current chunk
                    viewmats_input = viewmats[:, start_idx:end_idx]
                    Ks_input = Ks[:, start_idx:end_idx]
                    action_input = (
                        action[:, start_idx:end_idx] if action is not None else None
                    )

                    if chunk_i > 0:
                        viewmats_input = torch.cat([context_w2c, viewmats_input], dim=1)
                        Ks_input = torch.cat([context_Ks, Ks_input], dim=1)
                        action_input = torch.cat([context_action, action_input], dim=1)


                    # Prepare latent input (concatenate with cond_latents if needed)
                    latents_concat = torch.concat(
                        [latent_model_input, cond_latents_input], dim=1
                    )
                    
                    # Note: Unlike some other pipelines, HyWorld runs CFG sequentially (two passes)
                    # rather than batching pos/neg together, following the original implementation
                    latents_concat = self.scheduler.scale_model_input(latents_concat, t)

                    # Keep batch size 1 for sequential CFG
                    t_expand_txt = t.unsqueeze(0)
                    t_expand = timestep_input
                    viewmats_input = viewmats_input.to(device)
                    Ks_input = Ks_input.to(device)
                    action_input = action_input.reshape(-1).to(device)

                    with torch.autocast(
                        device_type="cuda",
                        dtype=target_dtype,
                        enabled=autocast_enabled,
                    ):
                        current_model = self.transformer
                        batch.is_cfg_negative = False

                        # Prepare transformer kwargs with HyWorld-specific inputs
                        # Note: batch size 1 for sequential CFG (matching original HY-WorldPlay)
                        transformer_kwargs = {
                            **image_kwargs,
                            **pos_cond_kwargs,
                            **action_kwargs,
                            "timestep": t_expand,  # [num_frames] num_frames = 16
                            "timestep_txt": t_expand_txt,  # [1]
                            "viewmats": viewmats_input.to(target_dtype),  # [1, num_frames, 4, 4]
                            "Ks": Ks_input.to(target_dtype),  # [1, num_frames, 3, 3]
                        }
                        # ref_image_tensor.shape: torch.Size([1, 3, 1, 480, 832])
                        # cond_latents.shape: torch.Size([1, 32, 1, 30, 52])
                        # encoder_hidden_states_image: torch.Size([1, 729, 1152])
                        # encoder_attention_mask: torch.Size([1, 1000])
                        for key, value in transformer_kwargs.items():
                            if key != "encoder_hidden_states_image" and key != "encoder_attention_mask":
                                print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")
                            else:
                                print(f"{key}: {value[0].shape}")

                        if action_input is not None:
                            transformer_kwargs["action"] = action_input.to(target_dtype)

                        with set_forward_context(
                            current_timestep=i,
                            attn_metadata=None,
                            forward_batch=batch,
                        ):
                            noise_pred = current_model(
                                latents_concat,
                                prompt_embeds,
                                **transformer_kwargs,
                            )
                            
                        print(f"noise_pred.shape: {noise_pred.shape}")
                        print("batch.guidance_scale", batch.guidance_scale)
                        print("batch.do_classifier_free_guidance", batch.do_classifier_free_guidance)
                        print("batch.guidance_rescale", batch.guidance_rescale)

                        if batch.do_classifier_free_guidance:
                            batch.is_cfg_negative = True
                            with set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                            ):
                                noise_pred_uncond = current_model(
                                    latents_concat,
                                    neg_prompt_embeds,
                                    **transformer_kwargs,
                                )

                            noise_pred_text = noise_pred
                            noise_pred = noise_pred_uncond + batch.guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )

                            # Apply guidance rescale if needed
                            if batch.guidance_rescale > 0.0:
                                noise_pred = self.rescale_noise_cfg(
                                    noise_pred,
                                    noise_pred_text,
                                    guidance_rescale=batch.guidance_rescale,
                                )

                    # Step scheduler - update only the current chunk's latents
                    latent_model_input = self.scheduler.step(
                        noise_pred, t, latent_model_input, **extra_step_kwargs, return_dict=False
                    )[0]

                    # Update only the current chunk's latents
                    latents[:, :, start_idx:end_idx] = latent_model_input[
                        :, :, -chunk_latent_frames:
                    ]

                    # Save trajectory latents if needed
                    if batch.return_trajectory_latents:
                        trajectory_timesteps.append(t)
                        trajectory_latents.append(latents.clone())

                    # Update progress bar
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                        and progress_bar is not None
                    ):
                        progress_bar.update()

        # Handle trajectory output
        trajectory_tensor: torch.Tensor | None = None
        if trajectory_latents:
            trajectory_tensor = torch.stack(trajectory_latents, dim=1)
            trajectory_timesteps_tensor = torch.stack(trajectory_timesteps, dim=0)
        else:
            trajectory_tensor = None
            trajectory_timesteps_tensor = None

        if trajectory_tensor is not None and trajectory_timesteps_tensor is not None:
            batch.trajectory_timesteps = trajectory_timesteps_tensor.cpu()
            batch.trajectory_latents = trajectory_tensor.cpu()

        # Update batch with final latents
        batch.latents = latents

        return batch

    def verify_input(
        self, batch: ForwardBatch, fastvideo_args: FastVideoArgs
    ) -> VerificationResult:
        """Verify HyWorld denoising stage inputs."""
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)

        # Check for HyWorld-specific inputs
        viewmats = getattr(batch, "viewmats", None) or batch.extra.get("viewmats", None)
        Ks = getattr(batch, "Ks", None) or batch.extra.get("Ks", None)

        if viewmats is None:
            result.add_failure(
                "viewmats",
                "viewmats must be provided in batch.extra['viewmats'] or as batch.viewmats",
            )
        else:
            result.add_check("viewmats", viewmats, V.is_tensor)

        if Ks is None:
            result.add_failure(
                "Ks", "Ks must be provided in batch.extra['Ks'] or as batch.Ks"
            )
        else:
            result.add_check("Ks", Ks, V.is_tensor)

        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        result.add_check("guidance_scale", batch.guidance_scale, V.positive_float)

        if batch.do_classifier_free_guidance:
            result.add_check(
                "negative_prompt_embeds",
                batch.negative_prompt_embeds,
                V.list_not_empty,
            )

        return result

    def verify_output(
        self, batch: ForwardBatch, fastvideo_args: FastVideoArgs
    ) -> VerificationResult:
        """Verify HyWorld denoising stage outputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result
