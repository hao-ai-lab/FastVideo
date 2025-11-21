# SPDX-License-Identifier: Apache-2.0
"""
LongCat-specific denoising stage implementing CFG-zero optimized guidance.
"""

import torch
from tqdm import tqdm

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.forward_context import set_forward_context

logger = init_logger(__name__)


class LongCatDenoisingStage(DenoisingStage):
    """
    LongCat denoising stage with CFG-zero optimized guidance scale.
    
    Implements:
    1. Optimized CFG scale from CFG-zero paper
    2. Negation of noise prediction before scheduler step (flow matching convention)
    3. Batched CFG computation (unlike standard FastVideo separate passes)
    """

    def optimized_scale(self, positive_flat, negative_flat) -> torch.Tensor:
        """
        Calculate optimized scale from CFG-zero paper.
        
        st_star = (v_cond^T * v_uncond) / ||v_uncond||^2
        
        Args:
            positive_flat: Conditional prediction, flattened [B, -1]
            negative_flat: Unconditional prediction, flattened [B, -1]
        
        Returns:
            st_star: Optimized scale [B, 1]
        """
        # Calculate dot product
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        # Squared norm of uncondition
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        # st_star = v_cond^T * v_uncond / ||v_uncond||^2
        st_star = dot_product / squared_norm
        return st_star

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Run LongCat denoising loop with optimized CFG.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with denoised latents.
        """
        if not fastvideo_args.model_loaded["transformer"]:
            from fastvideo.models.model_loader import TransformerLoader
            loader = TransformerLoader()
            self.transformer = loader.load(
                fastvideo_args.model_paths["transformer"], fastvideo_args)
            pipeline = self.pipeline() if self.pipeline else None
            if pipeline:
                pipeline.add_module("transformer", self.transformer)
            fastvideo_args.model_loaded["transformer"] = True

        # Get transformer dtype
        if hasattr(self.transformer, 'module'):
            transformer_dtype = next(self.transformer.module.parameters()).dtype
        else:
            transformer_dtype = next(self.transformer.parameters()).dtype
        
        target_dtype = transformer_dtype
        autocast_enabled = (target_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # Extract batch parameters
        latents = batch.latents
        timesteps = batch.timesteps
        prompt_embeds = batch.prompt_embeds[0]  # LongCat uses single encoder
        prompt_attention_mask = batch.prompt_attention_mask[0] if batch.prompt_attention_mask else None
        guidance_scale = batch.guidance_scale
        do_classifier_free_guidance = batch.do_classifier_free_guidance

        # Get negative prompts if doing CFG
        if do_classifier_free_guidance:
            negative_prompt_embeds = batch.negative_prompt_embeds[0]
            negative_prompt_attention_mask = (
                batch.negative_attention_mask[0] if batch.negative_attention_mask else None
            )
            # Concatenate for batched processing
            prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            if prompt_attention_mask is not None:
                prompt_attention_mask_combined = torch.cat(
                    [negative_prompt_attention_mask, prompt_attention_mask], dim=0
                )
            else:
                prompt_attention_mask_combined = None
        else:
            prompt_embeds_combined = prompt_embeds
            prompt_attention_mask_combined = prompt_attention_mask

        # Denoising loop
        num_inference_steps = len(timesteps)
        with tqdm(total=num_inference_steps, desc="LongCat Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for CFG
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents
                
                latent_model_input = latent_model_input.to(target_dtype)
                
                # Expand timestep to match batch size
                timestep = t.expand(latent_model_input.shape[0]).to(target_dtype)
                
                # Run transformer with context
                batch.is_cfg_negative = False
                with set_forward_context(
                    current_timestep=i,
                    attn_metadata=None,
                    forward_batch=batch,
                ):
                    with torch.autocast(device_type='cuda', dtype=target_dtype, enabled=autocast_enabled):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds_combined,
                            timestep=timestep,
                            encoder_attention_mask=prompt_attention_mask_combined,
                        )
                
                # Apply CFG with optimized scale
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    
                    B = noise_pred_cond.shape[0]
                    positive = noise_pred_cond.reshape(B, -1)
                    negative = noise_pred_uncond.reshape(B, -1)
                    
                    # Calculate optimized scale (CFG-zero)
                    st_star = self.optimized_scale(positive, negative)
                    
                    # Reshape for broadcasting
                    st_star = st_star.view(B, 1, 1, 1, 1)
                    
                    # Apply optimized CFG formula
                    noise_pred = (
                        noise_pred_uncond * st_star + 
                        guidance_scale * (noise_pred_cond - noise_pred_uncond * st_star)
                    )
                
                # CRITICAL: Negate noise prediction for flow matching scheduler
                noise_pred = -noise_pred
                
                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, 
                    t, 
                    latents, 
                    return_dict=False
                )[0]
                
                progress_bar.update()

        # Update batch with denoised latents
        batch.latents = latents
        return batch

