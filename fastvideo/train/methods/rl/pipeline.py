# SPDX-License-Identifier: Apache-2.0
"""Multi-step denoising with log-probability tracking.

Replaces diffusers' WanPipeline.__call__ for RL training.
Uses WanModel's forward_transformer_raw() instead of
the diffusers pipeline.
"""

from __future__ import annotations

import contextlib
import random
from typing import Any

import torch

from fastvideo.train.methods.rl.sde import (
    sde_step_with_logprob,
)


def wan_denoising_with_logprob(
    model,
    scheduler,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    generator: torch.Generator | None = None,
    noise_level: float = 0.7,
    sde_type: str = "flow_sde",
    deterministic: bool = False,
    diffusion_clip: bool = False,
    diffusion_clip_value: float = 0.45,
    sde_window_size: int = 0,
    sde_window_range: tuple[int, int] | None = None,
    kl_reward: float = 0.0,
    ref_transformer: torch.nn.Module | None = None,
    lora_model: Any | None = None,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Run full denoising loop, collecting latent
    trajectories and log-probabilities at each step.

    Args:
        model: WanModel (or GenRLWanModel) with
            forward_transformer_raw and vae.
        scheduler: Noise scheduler (UniPC/Euler).
        prompt_embeds: (B, L, D) text embeddings.
        negative_prompt_embeds: (B, L, D) or None.
        num_inference_steps: Number of denoising steps.
        guidance_scale: CFG scale.
        height, width, num_frames: Video dimensions.
        generator: RNG for reproducibility.
        noise_level: SDE noise level.
        sde_type: 'flow_sde' or 'flow_cps'.
        deterministic: If True, no SDE noise.
        diffusion_clip: Clip SDE variance.
        diffusion_clip_value: Clip threshold.
        sde_window_size: Window size for SDE training.
        sde_window_range: (start, end) range for window.
        kl_reward: KL penalty weight (>0 enables KL).
        ref_transformer: Reference model for KL.
        lora_model: LoRA model with disable_adapter().

    Returns:
        (videos, all_latents, all_log_probs,
         all_kl, all_timesteps)
    """
    device = model.device
    batch_size = prompt_embeds.shape[0]
    dtype = prompt_embeds.dtype

    do_cfg = (
        guidance_scale > 1.0
        and negative_prompt_embeds is not None
    )

    # Prepare initial noise.
    vae_config = model.vae.config
    vae_scale_temporal = getattr(
        vae_config, "temporal_compression_ratio", 4
    )
    vae_scale_spatial = getattr(
        vae_config, "spatial_compression_ratio", 8
    )
    num_channels = getattr(vae_config, "z_dim", 16)

    latent_frames = (num_frames - 1) // vae_scale_temporal + 1
    latent_h = height // vae_scale_spatial
    latent_w = width // vae_scale_spatial

    latents = torch.randn(
        batch_size,
        num_channels,
        latent_frames,
        latent_h,
        latent_w,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    # Setup scheduler.
    scheduler.set_timesteps(
        num_inference_steps, device=device
    )
    timesteps = scheduler.timesteps

    # Window setup.
    use_window = (
        sde_window_size > 0
        and sde_window_range is not None
    )
    if use_window:
        if (
            sde_window_range[1] - sde_window_range[0]
            < sde_window_size
        ):
            msg = (
                f"sde_window_range span "
                f"({sde_window_range[1] - sde_window_range[0]}) "
                f"must be >= sde_window_size "
                f"({sde_window_size})"
            )
            raise ValueError(msg)
        if generator is not None:
            gen = (
                generator[0]
                if isinstance(generator, list)
                else generator
            )
            max_start = (
                sde_window_range[1] - sde_window_size
            )
            start = torch.randint(
                sde_window_range[0],
                max_start + 1,
                (1,),
                generator=gen,
                device=device,
            ).item()
        else:
            start = random.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size,
            )
        end = start + sde_window_size
        sde_window = (start, end)
        all_latents: list[torch.Tensor] = []
    else:
        sde_window = None
        all_latents: list[torch.Tensor] = [latents]

    all_log_probs: list[torch.Tensor] = []
    all_kl: list[torch.Tensor] = []
    all_timesteps: list[torch.Tensor] = []

    for i, t in enumerate(timesteps):
        latents_ori = latents.clone()
        timestep = t.expand(batch_size)

        # Conditional prediction.
        noise_pred = model.forward_transformer_raw(
            latents.to(dtype),
            timestep,
            prompt_embeds,
        )
        noise_pred = noise_pred.to(dtype)

        # CFG.
        if do_cfg:
            noise_uncond = model.forward_transformer_raw(
                latents.to(dtype),
                timestep,
                negative_prompt_embeds,
            )
            noise_pred = noise_uncond + guidance_scale * (
                noise_pred - noise_uncond
            )

        # Determine noise level for this step.
        if use_window:
            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents)
            elif sde_window[0] < i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0
        else:
            cur_noise_level = noise_level

        # SDE step.
        (
            latents,
            log_prob,
            prev_latents_mean,
            std_dev_t,
            sigma,
            sigma_max,
        ) = sde_step_with_logprob(
            scheduler,
            noise_pred.float(),
            t.unsqueeze(0),
            latents.float(),
            noise_level=cur_noise_level,
            sde_type=sde_type,
            deterministic=deterministic,
            diffusion_clip=diffusion_clip,
            diffusion_clip_value=diffusion_clip_value,
        )
        prev_latents = latents.clone()

        # Record.
        in_window = (
            use_window
            and sde_window[0] <= i < sde_window[1]
        )
        should_record = (not use_window) or in_window

        if should_record:
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_timesteps.append(t)

        # KL computation.
        if should_record and kl_reward > 0 and not deterministic:
            ref_model = ref_transformer
            ref_ctx: Any = contextlib.nullcontext()
            if ref_model is None and lora_model is not None:
                ref_model = lora_model
                ref_ctx = lora_model.disable_adapter()

            if ref_model is not None:
                with ref_ctx:
                    ref_noise = ref_model(
                        hidden_states=latents_ori.to(dtype),
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )
                ref_noise = ref_noise.to(dtype)
                if do_cfg:
                    with ref_ctx:
                        ref_uncond = ref_model(
                            hidden_states=latents_ori.to(
                                dtype
                            ),
                            timestep=timestep,
                            encoder_hidden_states=(
                                negative_prompt_embeds
                            ),
                            return_dict=False,
                        )
                    ref_noise = (
                        ref_uncond
                        + guidance_scale
                        * (ref_noise - ref_uncond)
                    )

                (
                    _,
                    _ref_log_prob,
                    ref_prev_mean,
                    ref_std,
                    _ref_sigma,
                    _ref_sigma_max,
                ) = sde_step_with_logprob(
                    scheduler,
                    ref_noise.float(),
                    t.unsqueeze(0),
                    latents_ori.float(),
                    noise_level=noise_level,
                    sde_type=sde_type,
                    prev_sample=prev_latents.float(),
                    deterministic=deterministic,
                    diffusion_clip=diffusion_clip,
                    diffusion_clip_value=diffusion_clip_value,
                )
                kl = (prev_latents_mean - ref_prev_mean) ** 2 / (
                    2 * std_dev_t**2
                )
                kl = kl.mean(
                    dim=tuple(range(1, kl.ndim))
                )
                all_kl.append(kl)
            else:
                all_kl.append(
                    torch.zeros(
                        batch_size, device=device
                    )
                )
        elif should_record:
            all_kl.append(
                torch.zeros(batch_size, device=device)
            )

    # Decode to video.
    videos = model.decode_latents(latents)

    return (
        videos,
        all_latents,
        all_log_probs,
        all_kl,
        all_timesteps,
    )
