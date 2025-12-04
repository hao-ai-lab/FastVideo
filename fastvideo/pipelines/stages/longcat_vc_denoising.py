# SPDX-License-Identifier: Apache-2.0
"""
LongCat VC Denoising Stage with KV cache support.

This stage extends the I2V denoising stage to support:
1. KV cache for conditioning frames
2. Video continuation with multiple conditioning frames
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


class LongCatVCDenoisingStage(LongCatDenoisingStage):
    """
    LongCat denoising with Video Continuation and KV cache support.
    
    Key differences from I2V denoising:
    - Supports KV cache (reuses cached K/V from conditioning frames)
    - Handles larger num_cond_latents
    - Concatenates conditioning latents back after denoising
    
    When use_kv_cache=True:
    - batch.latents contains ONLY noise frames (cond removed by KV cache init)
    - batch.kv_cache_dict contains cached K/V
    - batch.cond_latents contains conditioning latents for post-concat
    
    When use_kv_cache=False:
    - batch.latents contains ALL frames (cond + noise)
    - Timestep masking: timestep[:, :num_cond_latents] = 0
    - Selective denoising: only update noise frames
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Run denoising loop with VC conditioning and optional KV cache."""

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

        # Get VC-specific parameters
        num_cond_latents = getattr(batch, 'num_cond_latents', 0)
        use_kv_cache = getattr(batch, 'use_kv_cache', False)
        kv_cache_dict = getattr(batch, 'kv_cache_dict', {})

        logger.info(
            "VC Denoising: num_cond_latents=%d, use_kv_cache=%s, latent_shape=%s",
            num_cond_latents, use_kv_cache, latents.shape)

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
                  desc="VC Denoising") as progress_bar:
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

                # 3. Expand timestep to temporal dimension
                timestep = timestep.unsqueeze(-1).repeat(
                    1, latent_model_input.shape[2])

                # 4. Timestep masking (only when NOT using KV cache)
                if not use_kv_cache and num_cond_latents > 0:
                    timestep[:, :num_cond_latents] = 0

                # 5. Prepare transformer kwargs
                # IMPORTANT: num_cond_latents is ALWAYS passed - needed for RoPE position offset
                transformer_kwargs = {
                    'num_cond_latents': num_cond_latents,
                }
                if use_kv_cache:
                    transformer_kwargs['kv_cache_dict'] = kv_cache_dict

                # 6. Run transformer
                batch.is_cfg_negative = False
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
                        **transformer_kwargs,
                    )

                # 7. Apply CFG with optimized scale (CFG-zero)
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                    B = noise_pred_cond.shape[0]
                    positive = noise_pred_cond.reshape(B, -1)
                    negative = noise_pred_uncond.reshape(B, -1)

                    st_star = self.optimized_scale(positive, negative)
                    st_star = st_star.view(B, 1, 1, 1, 1)

                    noise_pred = (
                        noise_pred_uncond * st_star + guidance_scale *
                        (noise_pred_cond - noise_pred_uncond * st_star))

                # 8. Negate for flow matching scheduler
                noise_pred = -noise_pred

                # 9. Scheduler step
                if use_kv_cache:
                    # All latents are noise frames (conditioning is in cache)
                    latents = self.scheduler.step(noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)[0]
                else:
                    # Only update noise frames (skip conditioning)
                    if num_cond_latents > 0:
                        latents[:, :, num_cond_latents:] = self.scheduler.step(
                            noise_pred[:, :, num_cond_latents:],
                            t,
                            latents[:, :, num_cond_latents:],
                            return_dict=False,
                        )[0]
                    else:
                        latents = self.scheduler.step(noise_pred,
                                                      t,
                                                      latents,
                                                      return_dict=False)[0]

                step_time = time.time() - step_start
                step_times.append(step_time)

                # Log timing for first few steps
                if i < 3:
                    logger.info("Step %d: %.2fs", i, step_time)

                progress_bar.update()

        # 10. If using KV cache, concatenate conditioning latents back
        if use_kv_cache and hasattr(
                batch, 'cond_latents') and batch.cond_latents is not None:
            latents = torch.cat([batch.cond_latents, latents], dim=2)
            logger.info(
                "Concatenated conditioning latents back, final shape: %s",
                latents.shape)

        # Log average timing
        avg_time = sum(step_times) / len(step_times)
        logger.info(
            "Average step time: %.2fs (total: %.1fs)",
            avg_time, sum(step_times))

        # Update batch with denoised latents
        batch.latents = latents
        return batch
