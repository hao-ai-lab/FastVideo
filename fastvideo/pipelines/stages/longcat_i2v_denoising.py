# SPDX-License-Identifier: Apache-2.0
"""
LongCat I2V Denoising Stage with conditioning support.

This stage implements Tier 3 I2V denoising:
1. Per-frame timestep masking (timestep[:, :num_cond_latents] = 0)
2. Passes num_cond_latents to transformer (for RoPE skipping)
3. Selective denoising (only updates non-conditioned frames)
4. CFG-zero optimized guidance
"""

import time
import torch
from tqdm import tqdm

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.longcat_denoising import LongCatDenoisingStage

logger = init_logger(__name__)


class LongCatI2VDenoisingStage(LongCatDenoisingStage):
    """
    LongCat denoising with I2V conditioning support.
    
    Key modifications from base LongCat denoising:
    1. Sets timestep=0 for conditioning frames
    2. Passes num_cond_latents to transformer
    3. Only applies scheduler step to non-conditioned frames
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Run denoising loop with I2V conditioning."""

        # Load transformer if needed
        if not fastvideo_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(
                fastvideo_args.model_paths["transformer"], fastvideo_args)
            fastvideo_args.model_loaded["transformer"] = True

        # Setup
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        latents = batch.latents
        timesteps = batch.timesteps
        prompt_embeds = batch.prompt_embeds[0]
        prompt_attention_mask = (batch.prompt_attention_mask[0]
                                 if batch.prompt_attention_mask else None)
        guidance_scale = batch.guidance_scale
        do_classifier_free_guidance = batch.do_classifier_free_guidance

        # Get num_cond_latents from batch
        num_cond_latents = getattr(batch, 'num_cond_latents', 0)

        if num_cond_latents > 0:
            logger.info(f"I2V Denoising: num_cond_latents={num_cond_latents}, "
                        f"latent_shape={latents.shape}")

        # DEBUG: Check devices
        logger.info(
            f"DEBUG devices: latents={latents.device}, timesteps={timesteps.device}, "
            f"prompt_embeds={prompt_embeds.device}")

        # Prepare negative prompts for CFG
        if do_classifier_free_guidance:
            negative_prompt_embeds = batch.negative_prompt_embeds[0]
            negative_prompt_attention_mask = (batch.negative_attention_mask[0]
                                              if batch.negative_attention_mask
                                              else None)

            prompt_embeds_combined = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0)
            if prompt_attention_mask is not None:
                prompt_attention_mask_combined = torch.cat(
                    [negative_prompt_attention_mask, prompt_attention_mask],
                    dim=0)
            else:
                prompt_attention_mask_combined = None
        else:
            prompt_embeds_combined = prompt_embeds
            prompt_attention_mask_combined = prompt_attention_mask

        # Denoising loop
        num_inference_steps = len(timesteps)
        step_times = []

        with tqdm(total=num_inference_steps,
                  desc="I2V Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                step_start = time.time()

                # 1. Expand latents for CFG
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                latent_model_input = latent_model_input.to(target_dtype)

                # 2. Expand timestep to match batch size
                timestep = t.expand(
                    latent_model_input.shape[0]).to(target_dtype)

                # 3. CRITICAL: Expand timestep to temporal dimension
                # and set conditioning frames to timestep=0
                timestep = timestep.unsqueeze(-1).repeat(
                    1, latent_model_input.shape[2])

                # Mark conditioning frames as clean (timestep=0)
                if num_cond_latents > 0:
                    timestep[:, :num_cond_latents] = 0

                # 4. Run transformer with num_cond_latents
                batch.is_cfg_negative = False
                transformer_start = time.time()
                with set_forward_context(
                        current_timestep=i,
                        attn_metadata=None,
                        forward_batch=batch,
                ), torch.autocast(device_type='cuda',
                                  dtype=target_dtype,
                                  enabled=autocast_enabled):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds_combined,
                        timestep=timestep,
                        encoder_attention_mask=prompt_attention_mask_combined,
                        num_cond_latents=
                        num_cond_latents,  # NEW: for RoPE skipping
                    )
                torch.cuda.synchronize()
                transformer_time = time.time() - transformer_start

                # 5. Apply CFG with optimized scale
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                    B = noise_pred_cond.shape[0]
                    positive = noise_pred_cond.reshape(B, -1)
                    negative = noise_pred_uncond.reshape(B, -1)

                    # CFG-zero optimized scale
                    st_star = self.optimized_scale(positive, negative)
                    st_star = st_star.view(B, 1, 1, 1, 1)

                    noise_pred = (
                        noise_pred_uncond * st_star + guidance_scale *
                        (noise_pred_cond - noise_pred_uncond * st_star))

                # 6. CRITICAL: Negate for flow matching scheduler
                noise_pred = -noise_pred

                # 7. CRITICAL: Only update non-conditioned frames
                # The conditioning frames stay FIXED throughout denoising
                if num_cond_latents > 0:
                    latents[:, :, num_cond_latents:] = self.scheduler.step(
                        noise_pred[:, :, num_cond_latents:],
                        t,
                        latents[:, :, num_cond_latents:],
                        return_dict=False)[0]
                else:
                    # No conditioning, update all frames
                    latents = self.scheduler.step(noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)[0]

                step_time = time.time() - step_start
                step_times.append(step_time)

                # Log timing for first few steps
                if i < 3:
                    logger.info(
                        f"Step {i}: total={step_time:.2f}s, transformer={transformer_time:.2f}s"
                    )
                    logger.info(
                        f"  Devices: latent_input={latent_model_input.device}, timestep={timestep.device}"
                    )

                progress_bar.update()

        # Log average timing
        avg_time = sum(step_times) / len(step_times)
        logger.info(
            f"Average step time: {avg_time:.2f}s (total: {sum(step_times):.1f}s)"
        )

        # Update batch with denoised latents
        batch.latents = latents
        return batch
