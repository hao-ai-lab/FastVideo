# SPDX-License-Identifier: Apache-2.0
"""
GameCraft denoising stage for camera/action-conditioned video generation.

This stage implements the denoising loop for HunyuanGameCraft, which generates
game-like videos with camera and action conditioning via:
1. CameraNet - Encodes Plücker coordinates into features added to image embeddings
2. Concatenated input - 33 channels (16 latent + 16 gt_latent + 1 mask)
3. Mask-based conditioning for autoregressive generation
"""

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class GameCraftDenoisingStage(DenoisingStage):
    """
    Denoising stage for HunyuanGameCraft with camera/action conditioning.
    
    This stage handles:
    - Camera state encoding via CameraNet (Plücker coordinates)
    - Concatenation of latents with gt_latents and mask (33 channels)
    - Flow matching denoising with camera conditioning
    - Support for autoregressive generation with history frames
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
        Run the denoising loop with camera/action conditioning.
        
        Args:
            batch: The current batch information. Must contain:
                - latents: Noise latents [B, 16, T, H, W]
                - camera_states: Plücker coordinates [B, T_video, 6, H_video, W_video]
                - gt_latents (optional): Ground truth latents for conditioning [B, 16, T, H, W]
                - conditioning_mask (optional): Mask for conditioning [B, 1, T, H, W]
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

        # Extract GameCraft-specific parameters
        camera_states = getattr(batch, "camera_states", None)
        if camera_states is None:
            camera_states = batch.extra.get("camera_states", None)
        
        gt_latents = getattr(batch, "gt_latents", None)
        if gt_latents is None:
            gt_latents = batch.extra.get("gt_latents", None)
        
        conditioning_mask = getattr(batch, "conditioning_mask", None)
        if conditioning_mask is None:
            conditioning_mask = batch.extra.get("conditioning_mask", None)

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

        # Get timesteps
        timesteps = batch.timesteps
        if timesteps is None:
            raise ValueError("Timesteps must be provided")
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Prepare image embeddings for I2V generation (if any)
        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert not torch.isnan(image_embeds[0]).any(), "image_embeds contains nan"
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        image_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {"encoder_hidden_states_image": image_embeds},
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

        # Prepare gt_latents and mask for concatenation
        # If not provided, use zeros (for unconditional generation)
        if gt_latents is None:
            gt_latents = torch.zeros_like(latents)
        else:
            gt_latents = gt_latents.to(target_dtype)

        if conditioning_mask is None:
            # Default mask: all zeros (generate everything)
            conditioning_mask = torch.zeros(
                latents.shape[0], 1, *latents.shape[2:],
                device=latents.device,
                dtype=target_dtype,
            )
        else:
            conditioning_mask = conditioning_mask.to(target_dtype)

        # Move camera states to device if provided
        if camera_states is not None:
            camera_states = camera_states.to(device=latents.device, dtype=target_dtype)
            
        # Debug logging
        logger.info(f"[GameCraft DEBUG] latents shape: {latents.shape}, min/max: {latents.min():.4f}/{latents.max():.4f}")
        logger.info(f"[GameCraft DEBUG] camera_states: {camera_states.shape if camera_states is not None else None}")
        logger.info(f"[GameCraft DEBUG] prompt_embeds[0] shape: {prompt_embeds[0].shape}")

        # Initialize lists for trajectory
        trajectory_timesteps: list[torch.Tensor] = []
        trajectory_latents: list[torch.Tensor] = []

        # Run denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Skip if interrupted
                if hasattr(self, "interrupt") and self.interrupt:
                    continue

                current_model = self.transformer
                current_guidance_scale = batch.guidance_scale

                # Prepare model input: concatenate latents, gt_latents, and mask
                # [B, 33, T, H, W] = [B, 16, T, H, W] + [B, 16, T, H, W] + [B, 1, T, H, W]
                latent_model_input = torch.cat(
                    [latents.to(target_dtype), gt_latents, conditioning_mask],
                    dim=1,
                )

                assert not torch.isnan(
                    latent_model_input
                ).any(), "latent_model_input contains nan"

                t_expand = t.repeat(latent_model_input.shape[0])

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # Official GameCraft does NOT use embedded guidance (guidance=None)
                # It uses standard CFG with guidance_scale instead
                guidance_expand = None

                # Run transformer with camera conditioning
                with torch.autocast(
                    device_type="cuda",
                    dtype=target_dtype,
                    enabled=autocast_enabled,
                ):
                    batch.is_cfg_negative = False
                    with set_forward_context(
                        current_timestep=i,
                        attn_metadata=None,
                        forward_batch=batch,
                    ):
                        noise_pred = current_model(
                            latent_model_input,
                            prompt_embeds,
                            t_expand,
                            camera_states=camera_states,
                            guidance=guidance_expand,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )
                        
                        # Debug: log first step output
                        if i == 0:
                            logger.info(f"[GameCraft DEBUG] Step 0 noise_pred: shape={noise_pred.shape}, min/max={noise_pred.min():.4f}/{noise_pred.max():.4f}")

                    # Classifier-free guidance
                    if batch.do_classifier_free_guidance:
                        batch.is_cfg_negative = True
                        with set_forward_context(
                            current_timestep=i,
                            attn_metadata=None,
                            forward_batch=batch,
                        ):
                            noise_pred_uncond = current_model(
                                latent_model_input,
                                neg_prompt_embeds,
                                t_expand,
                                camera_states=camera_states,
                                guidance=guidance_expand,
                                **image_kwargs,
                                **neg_cond_kwargs,
                            )
                        
                        noise_pred = noise_pred_uncond + current_guidance_scale * (
                            noise_pred - noise_pred_uncond
                        )

                # Compute the previous noisy sample
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

                # Store trajectory if requested
                if batch.return_trajectory_latents:
                    trajectory_timesteps.append(t.clone())
                    trajectory_latents.append(latents.clone())

                # Update progress bar
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # Debug: log final latents
        logger.info(f"[GameCraft DEBUG] Final latents: shape={latents.shape}, min/max={latents.min():.4f}/{latents.max():.4f}")
        
        # Store final latents and trajectory
        batch.latents = latents
        if batch.return_trajectory_latents:
            batch.trajectory_timesteps = trajectory_timesteps
            batch.trajectory_latents = torch.stack(trajectory_latents, dim=0)

        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify that required inputs are present."""
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify that outputs are properly set."""
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result
