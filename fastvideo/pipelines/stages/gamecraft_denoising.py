# SPDX-License-Identifier: Apache-2.0
"""
GameCraft-specific denoising stage for hybrid history conditioning.

HunyuanGameCraft uses a 33-channel input format:
- 16 channels: noisy latents
- 16 channels: conditioning/history latents  
- 1 channel: mask indicating history vs predicted frames

The model outputs 16-channel noise predictions which are used to update
only the non-history portion of the latents.
"""

import inspect
import weakref

import torch
from tqdm.auto import tqdm

from fastvideo.attention import get_attn_backend
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.platforms import AttentionBackendEnum

logger = init_logger(__name__)


class GameCraftDenoisingStage(PipelineStage):
    """
    Denoising stage for HunyuanGameCraft with hybrid history conditioning.
    
    This stage handles the special 33-channel input format required by GameCraft:
    - Concatenates latents (16ch) + conditioning (16ch) + mask (1ch)
    - Passes 33-channel tensor to transformer
    - Gets 16-channel noise prediction
    - Updates latents via scheduler step
    
    For text-to-video (no history), conditioning is zeros and mask is ones.
    For video continuation, conditioning contains history frames and mask
    indicates which frames are history (0) vs predicted (1).
    """

    def __init__(
        self,
        transformer,
        scheduler,
        pipeline=None,
        vae=None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.pipeline = weakref.ref(pipeline) if pipeline else None
        
        # Get attention backend
        attn_head_size = self.transformer.hidden_size // self.transformer.num_attention_heads
        self.attn_backend = get_attn_backend(
            head_size=attn_head_size,
            dtype=torch.float16,
            supported_attention_backends=(
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
            )
        )

    def _prepare_conditioning_latents(
        self,
        latents: torch.Tensor,
        history_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare conditioning latents and mask for GameCraft.
        
        Args:
            latents: Current noisy latents [B, 16, T, H, W]
            history_latents: Optional history latents for video continuation
            
        Returns:
            conditioning_latents: [B, 16, T, H, W] - zeros for t2v, history for continuation
            mask: [B, 1, T, H, W] - ones for t2v, mixed for continuation
        """
        batch_size, channels, num_frames, height, width = latents.shape
        device = latents.device
        dtype = latents.dtype
        
        if history_latents is None:
            # Text-to-video mode: no history conditioning
            # For pure T2V: all zeros conditioning, all zeros mask (generate everything)
            conditioning_latents = torch.zeros(
                batch_size, channels, num_frames, height, width,
                device=device, dtype=dtype
            )
            # All zeros mask = generate all frames
            mask = torch.zeros(
                batch_size, 1, num_frames, height, width,
                device=device, dtype=dtype
            )
        else:
            # Video continuation mode: use history as conditioning
            conditioning_latents = history_latents.clone()
            
            # Create mask: 0 for history frames, 1 for frames to generate
            # Typically first half are history, second half are generated
            mask = torch.ones(
                batch_size, 1, num_frames, height, width,
                device=device, dtype=dtype
            )
            
            # For continuation, first frame(s) come from history
            if num_frames > 1:
                # First half is history (mask=0), second half is predicted (mask=1)
                num_history = num_frames // 2
                mask[:, :, :num_history, :, :] = 0.0
                conditioning_latents[:, :, num_history:, :, :] = 0.0
            else:
                # Single frame: use first frame from history
                mask[:, :, 0, :, :] = 0.0
        
        return conditioning_latents, mask

    def prepare_extra_func_kwargs(self, func, kwargs):
        """Filter kwargs to only include parameters the function accepts."""
        # Get the function's signature
        sig = inspect.signature(func)
        valid_keys = set(sig.parameters.keys())
        
        # Filter to only valid kwargs
        return {k: v for k, v in kwargs.items() if k in valid_keys and v is not None}

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Run the GameCraft denoising loop with hybrid conditioning.
        
        Args:
            batch: The current batch information
            fastvideo_args: The inference arguments
            
        Returns:
            batch: Updated batch with denoised latents
        """
        device = get_local_torch_device()
        target_dtype = torch.bfloat16
        
        # Get batch data
        latents = batch.latents  # [B, 16, T, H, W]
        timesteps = batch.timesteps
        
        # Get prompt embeddings from batch (it's a list for multiple encoders)
        # Convert to target dtype
        prompt_embeds = [
            emb.to(target_dtype) if emb is not None else None 
            for emb in batch.prompt_embeds
        ]
        neg_prompt_embeds = [
            emb.to(target_dtype) if emb is not None else None 
            for emb in (batch.negative_prompt_embeds or [])
        ]
        
        # Get CLIP pooled embeddings if available
        clip_embedding_pos = getattr(batch, 'clip_embedding_pos', None)
        if clip_embedding_pos is not None:
            clip_embedding_pos = clip_embedding_pos.to(target_dtype)
        clip_embedding_neg = getattr(batch, 'clip_embedding_neg', None)
        if clip_embedding_neg is not None:
            clip_embedding_neg = clip_embedding_neg.to(target_dtype)
        
        # Get history latents if available (for video continuation)
        history_latents = getattr(batch, 'history_latents', None)
        
        # Prepare conditioning latents and mask
        conditioning_latents, mask = self._prepare_conditioning_latents(
            latents, history_latents
        )
        
        # Keep a copy of conditioning for each step
        gt_latents_concat = conditioning_latents.clone()
        mask_concat = mask
        
        # Prepare extra step kwargs for scheduler
        extra_step_kwargs = {}
        if "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        ):
            extra_step_kwargs["generator"] = batch.generator
        
        # Get camera states for CameraNet conditioning
        # Convert camera_states to target dtype and device
        camera_states = batch.camera_states
        if camera_states is not None:
            camera_states = camera_states.to(device=device, dtype=target_dtype)
        camera_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {"camera_states": camera_states},
        )
        
        # Prepare positional conditioning kwargs
        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_2": clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )
        
        neg_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_2": clip_embedding_neg,
                "encoder_attention_mask": batch.negative_attention_mask,
            },
        )
        
        # Denoising loop
        num_inference_steps = len(timesteps)
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        
        with tqdm(total=num_inference_steps, desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                # Debug: print timestep and latent stats on first step
                if i == 0:
                    logger.info(f"[DEBUG] Timestep 0: t={t.item() if hasattr(t, 'item') else t}")
                    logger.info(f"[DEBUG] Latents shape: {latents.shape}, mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")
                    logger.info(f"[DEBUG] Mask shape: {mask_concat.shape}, unique values: {mask_concat.unique().tolist()}")
                
                # Concatenate latents with conditioning for 33-channel input
                # [B, 16, T, H, W] + [B, 16, T, H, W] + [B, 1, T, H, W] = [B, 33, T, H, W]
                latents_concat = torch.cat(
                    [latents, gt_latents_concat, mask_concat], dim=1
                )
                
                latent_model_input = latents_concat.to(target_dtype)
                
                # Scale model input
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                
                # Expand timestep
                t_expand = t.unsqueeze(0) if t.dim() == 0 else t
                
                # Get primary text embeddings (LLaMA)
                current_prompt_embeds = prompt_embeds[0] if len(prompt_embeds) > 0 else None
                
                # Run transformer forward pass with positive conditioning
                batch.is_cfg_negative = False
                with set_forward_context(
                    current_timestep=i,
                    attn_metadata=None,
                    forward_batch=batch,
                ):
                    noise_pred = self.transformer(
                        latent_model_input,
                        current_prompt_embeds,
                        t_expand,
                        guidance=None,
                        **pos_cond_kwargs,
                        **camera_kwargs,
                    )
                
                # Debug: print noise prediction stats on first step
                if i == 0:
                    logger.info(f"[DEBUG] Noise pred shape: {noise_pred.shape}, mean: {noise_pred.mean().item():.4f}, std: {noise_pred.std().item():.4f}")
                
                # Apply classifier-free guidance if enabled
                if batch.do_classifier_free_guidance:
                    # Get negative embeddings
                    current_neg_embeds = neg_prompt_embeds[0] if neg_prompt_embeds and len(neg_prompt_embeds) > 0 else None
                    
                    batch.is_cfg_negative = True
                    with set_forward_context(
                        current_timestep=i,
                        attn_metadata=None,
                        forward_batch=batch,
                    ):
                        noise_pred_uncond = self.transformer(
                            latent_model_input,
                            current_neg_embeds,
                            t_expand,
                            guidance=None,
                            **neg_cond_kwargs,
                            **camera_kwargs,
                        )
                    
                    noise_pred = noise_pred_uncond + batch.guidance_scale * (
                        noise_pred - noise_pred_uncond
                    )
                
                # Debug: print noise prediction after CFG on first step
                if i == 0:
                    logger.info(f"[DEBUG] After CFG - Noise pred mean: {noise_pred.mean().item():.4f}, std: {noise_pred.std().item():.4f}")
                
                # Scheduler step
                # noise_pred is [B, 16, T, H, W], latents is [B, 16, T, H, W]
                num_frames = latents.shape[2]
                
                if history_latents is not None and num_frames > 1:
                    # Video continuation mode: only update second half of frames
                    num_history = num_frames // 2
                    latents[:, :, num_history:, :, :] = self.scheduler.step(
                        noise_pred[:, :, num_history:, :, :],
                        t,
                        latents[:, :, num_history:, :, :],
                        **extra_step_kwargs,
                        return_dict=False
                    )[0]
                else:
                    # T2V mode: update all frames
                    latents = self.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False
                    )[0]
                
                # Debug: print latent stats after scheduler step
                if i == 0 or i == len(timesteps) - 1:
                    logger.info(f"[DEBUG] Step {i} - Latents after step: mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")
                
                # Update progress bar
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and
                    (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
        
        # Update batch with final latents
        batch.latents = latents
        
        return batch
