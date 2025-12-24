# SPDX-License-Identifier: Apache-2.0
"""
GRPO utilities for Wan model in FastVideo.

This module ports the SDE step and pipeline functions from FlowGRPO to work with
FastVideo's scheduler and pipeline interfaces.

Ported from:
- flow_grpo/flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py

Key adaptations:
1. Uses FastVideo's FlowUniPCMultistepScheduler instead of diffusers' UniPCMultistepScheduler
2. Works with FastVideo's WanPipeline (ComposedPipelineBase) instead of diffusers' WanPipeline
3. Direct module access via pipeline.get_module() instead of pipeline attributes
4. Simplified prompt encoding (direct text encoder usage instead of pipeline stages)
"""

import math
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler
)
from fastvideo.pipelines.basic.wan.wan_pipeline import WanPipeline

logger = init_logger(__name__)


def sde_step_with_logprob(
    scheduler: FlowUniPCMultistepScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
    return_pixel_log_prob: bool = False,
    return_dt_and_std_dev_t: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ...]:
    """
    Predict the sample from the previous timestep by reversing the SDE. 
    This function propagates the flow process from the learned model outputs 
    (most often the predicted velocity) and computes log probabilities.

    Ported from FlowGRPO's sde_step_with_logprob to work with FastVideo's
    FlowUniPCMultistepScheduler.

    Args:
        scheduler: FastVideo FlowUniPCMultistepScheduler instance
        model_output: The direct output from learned flow model
        timestep: The current discrete timestep in the diffusion chain
        sample: A current instance of a sample created by the diffusion process
        prev_sample: Optional previous sample (if provided, used instead of sampling)
        generator: Optional random number generator
        determistic: If True, no noise is added (deterministic sampling)
        return_pixel_log_prob: If True, return pixel-level log probabilities (not used)
        return_dt_and_std_dev_t: If True, return dt and std_dev_t separately

    Returns:
        If return_dt_and_std_dev_t=True:
            (prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt)
        Otherwise:
            (prev_sample, log_prob, prev_sample_mean, std_dev_t * sqrt_dt)
    """
    # Convert all variables to fp32 for numerical stability
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    # Get step indices for current and previous timesteps
    # Handle both single timestep and batch of timesteps
    if isinstance(timestep, torch.Tensor):
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        step_indices = [scheduler.index_for_timestep(t.item()) for t in timestep]
    else:
        step_indices = [scheduler.index_for_timestep(timestep)]

    prev_step_indices = [step + 1 for step in step_indices]

    # Move sigmas to sample device
    sigmas = scheduler.sigmas.to(sample.device)
    
    # Get sigma values for current and previous steps
    sigma = sigmas[step_indices].view(-1, 1, 1, 1, 1)
    sigma_prev = sigmas[prev_step_indices].view(-1, 1, 1, 1, 1)
    sigma_max = sigmas[0].item()  # First sigma (highest)
    sigma_min = sigmas[-1].item()  # Last sigma (lowest)
    
    dt = sigma_prev - sigma

    # Compute std_dev_t and prev_sample_mean using SDE formulation
    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt) +
        model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    # Sample prev_sample if not provided
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        sqrt_dt = torch.sqrt(-1 * dt)  # dt is negative (going backwards)
        prev_sample = prev_sample_mean + std_dev_t * sqrt_dt * variance_noise
    else:
        sqrt_dt = torch.sqrt(-1 * dt)

    # No noise is added during evaluation (deterministic)
    if determistic:
        prev_sample = sample + dt * model_output
        sqrt_dt = torch.sqrt(-1 * dt)

    # Compute log probability: log p(prev_sample | sample, model_output)
    # Assuming Gaussian distribution: N(prev_sample_mean, (std_dev_t * sqrt_dt)^2)
    std_dev_sqrt_dt = std_dev_t * sqrt_dt
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_sqrt_dt**2))
        - torch.log(std_dev_sqrt_dt + 1e-8)  # Add small epsilon for numerical stability
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=sample.device)))
    )

    # Mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_dt_and_std_dev_t:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * sqrt_dt


def wan_pipeline_with_logprob(
    pipeline: WanPipeline,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pt",
    return_dict: bool = False,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    determistic: bool = False,
    kl_reward: float = 0.0,
    return_pixel_log_prob: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Wan pipeline with log probability computation for GRPO training.

    Ported from FlowGRPO's wan_pipeline_with_logprob to work with FastVideo's WanPipeline.
    This function generates videos and computes log probabilities at each denoising step.

    Args:
        pipeline: FastVideo WanPipeline instance
        prompt: Text prompt(s) for generation
        negative_prompt: Negative prompt(s) for classifier-free guidance
        height: Height of generated video
        width: Width of generated video
        num_frames: Number of frames in generated video
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        num_videos_per_prompt: Number of videos to generate per prompt
        generator: Random generator for reproducibility
        latents: Optional initial latents
        prompt_embeds: Optional pre-computed prompt embeddings
        negative_prompt_embeds: Optional pre-computed negative prompt embeddings
        output_type: Output type ("pt" for PyTorch tensor, "np" for numpy, "latent" for latents only)
        return_dict: Whether to return dict (not used, always returns tuple)
        attention_kwargs: Optional attention kwargs
        max_sequence_length: Maximum sequence length for text encoding
        determistic: If True, use deterministic sampling (no noise)
        kl_reward: KL reward coefficient (if > 0, computes KL divergence)
        return_pixel_log_prob: If True, return pixel-level log probabilities (not used)

    Returns:
        Tuple of:
            - video: Generated video tensor [B, C, T, H, W] or latents if output_type="latent"
            - all_latents: List of latents at each step [num_steps+1] of shape [B, C, T, H, W]
            - all_log_probs: List of log probabilities at each step [num_steps] of shape [B]
            - all_kl: List of KL divergences at each step [num_steps] of shape [B] (if kl_reward > 0)
    """
    # Get device from transformer
    transformer = pipeline.get_module("transformer")
    device = next(transformer.parameters()).device
    
    # Get scheduler and other modules
    scheduler = pipeline.get_module("scheduler")
    vae = pipeline.get_module("vae")

    # Determine batch size
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    elif prompt_embeds is not None:
        batch_size = prompt_embeds.shape[0]
    else:
        raise ValueError("Either prompt or prompt_embeds must be provided")

    # Encode prompts if not provided
    if prompt_embeds is None:
        # Encode prompts directly using text encoder and tokenizer
        # This is a simplified encoding - for full pipeline encoding, use TextEncodingStage
        text_encoder = pipeline.get_module("text_encoder")
        tokenizer = pipeline.get_module("tokenizer")
        
        # Normalize to list
        if isinstance(prompt, str):
            prompts_list = [prompt]
        else:
            prompts_list = prompt
        
        # Tokenize prompts
        text_inputs = tokenizer(
            prompts_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Encode with text encoder
        with torch.no_grad():
            outputs = text_encoder(
                text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                output_hidden_states=True,
            )
            # Get last hidden state (Wan typically uses last hidden state)
            prompt_embeds = outputs.last_hidden_state
        
        # Encode negative prompts if CFG is enabled
        if guidance_scale > 1.0:
            if negative_prompt is None:
                negative_prompt = [""] * len(prompts_list)
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            
            neg_text_inputs = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                neg_outputs = text_encoder(
                    neg_text_inputs["input_ids"],
                    attention_mask=neg_text_inputs["attention_mask"],
                    output_hidden_states=True,
                )
                negative_prompt_embeds = neg_outputs.last_hidden_state
        else:
            negative_prompt_embeds = None

    # logger.info("wan_pipeline_with_logprob's transformer class type: %s", type(transformer))
    # transformer_dtype = transformer.dtype
    # prompt_embeds = prompt_embeds.to(transformer_dtype)
    # if negative_prompt_embeds is not None:
    #     negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    # Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Prepare latent variables
    num_channels_latents = transformer.config.in_channels
    vae = pipeline.get_module("vae")
    # Get VAE scale factors
    vae_scale_factor_spatial = vae.spatial_compression_ratio
    vae_scale_factor_temporal = vae.temporal_compression_ratio
    
    if latents is None:
        # Generate random latents
        # Note: num_frames in latents accounts for temporal compression
        num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        latents_shape = (
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_latent_frames,
            height // vae_scale_factor_spatial,
            width // vae_scale_factor_spatial,
        )
        if generator is not None:
            if isinstance(generator, list):
                latents = [
                    torch.randn(
                        latents_shape[1:],
                        generator=gen,
                        device=device,
                        dtype=torch.float32,
                    )
                    for gen in generator
                ]
                latents = torch.stack(latents, dim=0)
            else:
                latents = torch.randn(
                    latents_shape,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
        else:
            latents = torch.randn(latents_shape, device=device, dtype=torch.float32)
    else:
        latents = latents.to(device=device, dtype=torch.float32)

    all_latents = [latents]
    all_log_probs = []
    all_kl = []

    # Denoising loop
    do_classifier_free_guidance = guidance_scale > 1.0

    for i, t in enumerate(timesteps):
        latents_ori = latents.clone()
        latent_model_input = latents.to(transformer_dtype)
        timestep = t.expand(latents.shape[0]) if isinstance(t, torch.Tensor) else torch.tensor([t] * latents.shape[0], device=device)

        # Predict noise with transformer
        noise_pred = transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.to(prompt_embeds.dtype)

        # Classifier-free guidance
        if do_classifier_free_guidance:
            noise_uncond = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        # SDE step with log probability
        latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            scheduler,
            noise_pred.float(),
            t.unsqueeze(0) if isinstance(t, torch.Tensor) else t,
            latents.float(),
            determistic=determistic,
            return_pixel_log_prob=return_pixel_log_prob
        )
        prev_latents = latents.clone()

        all_latents.append(latents)
        all_log_probs.append(log_prob)

        # Compute KL divergence if kl_reward > 0 (for KL reward in sampling)
        if kl_reward > 0 and not determistic:
            # Use reference model (disable adapter if using LoRA)
            latent_model_input_ref = torch.cat([latents_ori] * 2) if do_classifier_free_guidance else latents_ori
            with transformer.disable_adapter() if hasattr(transformer, 'disable_adapter') else torch.no_grad():
                noise_pred_ref = transformer(
                    hidden_states=latent_model_input_ref,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            noise_pred_ref = noise_pred_ref.to(prompt_embeds.dtype)
            
            # Perform guidance for reference model
            if do_classifier_free_guidance:
                noise_pred_uncond_ref, noise_pred_text_ref = noise_pred_ref.chunk(2)
                noise_pred_ref = noise_pred_uncond_ref + guidance_scale * (noise_pred_text_ref - noise_pred_uncond_ref)

            # Compute reference log prob
            _, ref_log_prob, ref_prev_latents_mean, ref_std_dev_t = sde_step_with_logprob(
                scheduler,
                noise_pred_ref.float(),
                t.unsqueeze(0) if isinstance(t, torch.Tensor) else t,
                latents_ori.float(),
                prev_sample=prev_latents.float(),
                determistic=determistic,
            )
            
            # Compute KL divergence: KL = (mean_diff)^2 / (2 * std^2)
            assert torch.allclose(std_dev_t, ref_std_dev_t), "std_dev_t should match between current and reference"
            kl = (prev_latents_mean - ref_prev_latents_mean) ** 2 / (2 * std_dev_t ** 2)
            kl = kl.mean(dim=tuple(range(1, kl.ndim)))
            all_kl.append(kl)
        else:
            # No KL reward, set to zero
            all_kl.append(torch.zeros(len(latents), device=latents.device))

    # Decode latents to video if needed
    if output_type != "latent":
        latents = latents.to(vae.dtype)
        
        # Apply VAE normalization (Wan VAE specific)
        # Wan VAE requires denormalization before decoding
        if hasattr(vae, 'config') and hasattr(vae.config, 'latents_mean') and hasattr(vae.config, 'latents_std'):
            # Get z_dim from config or VAE
            z_dim = getattr(vae.config, 'z_dim', latents.shape[1])
            latents_mean = (
                torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
                .view(1, z_dim, 1, 1, 1)
            )
            latents_std = (
                1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
                .view(1, z_dim, 1, 1, 1)
            )
            latents = latents / latents_std + latents_mean
        elif hasattr(vae, 'latents_mean') and hasattr(vae, 'latents_std'):
            # Alternative: check if latents_mean/std are direct attributes
            z_dim = latents.shape[1]
            latents_mean = (
                torch.tensor(vae.latents_mean, device=latents.device, dtype=latents.dtype)
                .view(1, z_dim, 1, 1, 1)
            )
            latents_std = (
                1.0 / torch.tensor(vae.latents_std, device=latents.device, dtype=latents.dtype)
                .view(1, z_dim, 1, 1, 1)
            )
            latents = latents / latents_std + latents_mean
        
        # Decode using VAE
        with torch.no_grad():
            video = vae.decode(latents)
            # VAE.decode returns tensor directly (not tuple)
            
            # Postprocess video: convert from [-1, 1] to [0, 1]
            # FastVideo VAE typically outputs in [-1, 1] range
            video = (video / 2 + 0.5).clamp(0, 1)
    else:
        video = latents

    return video, all_latents, all_log_probs, all_kl
