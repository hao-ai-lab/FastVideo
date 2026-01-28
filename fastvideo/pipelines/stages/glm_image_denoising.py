# SPDX-License-Identifier: Apache-2.0
"""
GLM-Image denoising stage.

This stage performs the denoising loop for GLM-Image generation.
"""

import torch
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.distributed import get_local_torch_device
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.attention.selector import backend_name_to_enum


class GlmImageDenoisingStage(DenoisingStage):
    """Denoising stage for GLM-Image pipeline."""

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify denoising stage inputs."""
        result = VerificationResult()
        # Basic checks
        result.add_check("timesteps", batch.timesteps,
                         [V.is_tensor, V.min_dims(1)])
        # Use latents or latent
        latents = getattr(batch, 'latent', getattr(batch, 'latents', None))
        result.add_check("latents", latents, [V.is_tensor, V.with_dims(5)])

        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)

        # We handle prompt_embeds as concatenated [neg, pos] for CFG
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)

        # Explicitly skip negative_prompt_embeds logic from base class since we pre-concat

        return result

    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = get_local_torch_device()
        dtype = torch.bfloat16

        # Parameters
        num_inference_steps = batch.num_inference_steps
        guidance_scale = getattr(batch, 'guidance_scale', 1.5)

        # Get latents from batch
        latents = getattr(batch, 'latent', getattr(batch, 'latents', None))
        if latents is None:
            raise ValueError("No latents found in batch.")

        # GLM-Image is 2D, squeeze temporal dimension if present [B, C, T, H, W] -> [B, C, H, W]
        if latents.dim() == 5:
            latents = latents.squeeze(2)

        timesteps = batch.timesteps
        prompt_embeds = batch.prompt_embeds[0]  # (2, L, D) for CFG
        text_attention_mask = getattr(batch, 'attention_mask', None)

        # Prepare target_size and crop_coords
        bs = 2 if guidance_scale > 1.0 else 1
        target_size = torch.tensor([[batch.height, batch.width]],
                                   device=device,
                                   dtype=torch.long).repeat(bs, 1)
        crop_coords = torch.zeros((bs, 2), device=device, dtype=torch.long)

        # Prepare prior tokens for CFG
        prior_token_id = batch.prior_token_id
        if guidance_scale > 1.0 and prior_token_id.shape[0] == 1:
            prior_token_id = prior_token_id.repeat(2, 1)
            # Create drop mask: [False] for cond, [True] for uncond (batch dimension indexing)
            prior_token_drop = torch.tensor([False, True], device=device)
        else:
            prior_token_drop = getattr(batch, 'prior_token_drop',
                                       torch.tensor([False], device=device))

        # Compute image sequence length from latents
        # Get patch_size from transformer config (patch_size is in latent space)
        patch_size = self.transformer.patch_size
        _, _, h, w = latents.shape
        image_seq_length = (h // patch_size) * (w // patch_size)

        # Compute text sequence length from prompt_embeds
        text_seq_length = prompt_embeds.shape[1] if prompt_embeds.dim() >= 2 else 0

        # The mask will be set in forward context before each transformer call
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Prepare model input (CFG pass)
                if guidance_scale > 1.0:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # Scale model input for flow matching / diffusion
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                latent_model_input = latent_model_input.to(dtype)
                t_expand = t.expand(latent_model_input.shape[0])

                # Create combined attention mask for [text, image] sequence
                attn_metadata = None
                if text_attention_mask is not None:
                    first_block = self.transformer.transformer_blocks[0]
                    attn_backend_name = first_block.attn1.attn.backend.name if hasattr(first_block.attn1.attn, 'backend') else None
                    
                    if attn_backend_name == "SDPA" or backend_name_to_enum(attn_backend_name) == AttentionBackendEnum.TORCH_SDPA:
                        batch_size = latent_model_input.shape[0]
                        # Create combined mask: text part uses provided mask, image part is always 1
                        mix_attn_mask = torch.ones(
                            (batch_size, text_seq_length + image_seq_length),
                            device=device,
                            dtype=torch.float32
                        )
                        # Expand text mask to batch size if needed
                        if text_attention_mask.shape[0] == 1 and batch_size > 1:
                            text_attention_mask = text_attention_mask.repeat(batch_size, 1)
                        # Set text part of mask
                        mix_attn_mask[:, :text_seq_length] = text_attention_mask.float().to(device)
                        
                        # Convert to SDPA format: (B, 1, 1, L) for key-padding style mask
                        # True = attend, False = ignore (will be converted to additive mask by SDPA)
                        attention_mask_kv = (mix_attn_mask > 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
                        
                        # Create SDPAMetadata with the mask
                        attn_metadata = SDPAMetadata(
                            current_timestep=i,
                            attn_mask=attention_mask_kv
                        )
                    # If using Flash Attention, masks are not supported - attn_metadata remains None

                with torch.no_grad(), set_forward_context(current_timestep=i,
                                                          attn_metadata=attn_metadata,
                                                          forward_batch=batch):
                    # Predict noise
                    noise_pred = self.transformer(latent_model_input,
                                                  prompt_embeds,
                                                  t_expand,
                                                  prior_token_id,
                                                  prior_token_drop,
                                                  target_size,
                                                  crop_coords)

                    # Apply Classifier-Free Guidance
                    if guidance_scale > 1.0:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_cond - noise_pred_uncond)

                        # Apply guidance rescale if requested
                        guidance_rescale = getattr(batch, 'guidance_rescale',
                                                   0.0)
                        if guidance_rescale > 0.0:
                            std_text = noise_pred_cond.std(dim=list(
                                range(1, noise_pred_cond.ndim)),
                                                           keepdim=True)
                            std_cfg = noise_pred.std(dim=list(
                                range(1, noise_pred.ndim)),
                                                     keepdim=True)
                            noise_pred_rescaled = noise_pred * (std_text /
                                                                std_cfg)
                            noise_pred = guidance_rescale * noise_pred_rescaled + (
                                1 - guidance_rescale) * noise_pred

                    # Step scheduler
                    latents = self.scheduler.step(noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)[0]

                progress_bar.update()

        # Store result in batch (add temporal dim back for video-compatible decoding)
        batch.latents = latents.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
        return batch
