# SPDX-License-Identifier: Apache-2.0
"""Sampling epoch for RL training — generate videos and
compute rewards."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.rl.embeddings import (
    compute_text_embeddings,
)
from fastvideo.train.methods.rl.pipeline import (
    wan_denoising_with_logprob,
)

logger = init_logger(__name__)

SEED_EPOCH_STRIDE = 10_000


def create_generator(
    prompts: list[str],
    base_seed: int,
    device: torch.device,
) -> list[torch.Generator]:
    """Create deterministic generators seeded by prompt."""
    generators = []
    for prompt in prompts:
        g = torch.Generator(device=device)
        g.manual_seed(base_seed + hash(prompt) % (2**31))
        generators.append(g)
    return generators


def sample_epoch(
    model,
    scheduler,
    train_sampler,
    train_iter,
    reward_fn: Callable,
    sample_neg_prompt_embeds: torch.Tensor,
    text_encoder,
    tokenizer,
    executor,
    epoch: int,
    global_step: int,
    *,
    # Config values passed explicitly.
    sample_batch_size: int,
    num_batches_per_epoch: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    num_frames: int,
    noise_level: float,
    sde_type: str,
    diffusion_clip: bool,
    diffusion_clip_value: float,
    sde_window_size: int,
    sde_window_range: tuple[int, int] | None,
    kl_reward: float,
    same_latent: bool,
    seed: int,
    device: torch.device,
    is_main_process: bool,
    ref_transformer: torch.nn.Module | None = None,
    lora_model: Any | None = None,
    tracker: Any | None = None,
) -> list[dict[str, Any]]:
    """Run one sampling epoch: generate videos, compute
    rewards asynchronously.

    Returns:
        List of sample dicts with prompt_ids,
        prompt_embeds, latents, log_probs, kl,
        timesteps, rewards.
    """
    samples = []

    for i in range(num_batches_per_epoch):
        current_epoch_tag = (
            epoch * num_batches_per_epoch + i
        )
        train_sampler.set_epoch(current_epoch_tag)

        # Drain until epoch tag matches.
        while True:
            epoch_tag, prompts, prompt_metadata = next(
                train_iter
            )
            if epoch_tag == current_epoch_tag:
                break

        prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoder,
            tokenizer,
            max_sequence_length=512,
            device=device,
        )
        prompt_ids = tokenizer(
            prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        # Generator setup.
        if same_latent:
            gen = create_generator(
                prompts,
                base_seed=epoch * SEED_EPOCH_STRIDE + i,
                device=device,
            )
        else:
            gen = torch.Generator(device=device)
            gen.manual_seed(
                seed + epoch * SEED_EPOCH_STRIDE + i
            )

        with torch.no_grad():
            (
                videos,
                latents_list,
                log_probs_list,
                kls_list,
                timesteps_list,
            ) = wan_denoising_with_logprob(
                model,
                scheduler,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=(
                    sample_neg_prompt_embeds
                ),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_frames=num_frames,
                generator=gen,
                noise_level=noise_level,
                sde_type=sde_type,
                diffusion_clip=diffusion_clip,
                diffusion_clip_value=diffusion_clip_value,
                sde_window_size=sde_window_size,
                sde_window_range=sde_window_range,
                kl_reward=kl_reward,
                ref_transformer=ref_transformer,
                lora_model=lora_model,
            )

        latents = torch.stack(latents_list, dim=1)
        log_probs = torch.stack(log_probs_list, dim=1)
        kls = torch.stack(kls_list, dim=1)
        kl = kls.detach()

        timesteps = (
            torch.stack(timesteps_list)
            .unsqueeze(0)
            .repeat(sample_batch_size, 1)
        )

        # Async reward computation.
        rewards_future = executor.submit(
            reward_fn,
            videos,
            prompts,
            prompt_metadata,
            True,
        )
        time.sleep(0)

        samples.append(
            {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": (
                    sample_neg_prompt_embeds
                ),
                "timesteps": timesteps,
                "latents": latents[:, :-1],
                "next_latents": latents[:, 1:],
                "log_probs": log_probs,
                "kl": kl,
                "rewards": rewards_future,
            }
        )

    # Wait for all rewards.
    for sample in samples:
        rewards, _ = sample["rewards"].result()
        sample["rewards"] = {
            key: torch.as_tensor(value, device=device).float()
            for key, value in rewards.items()
        }

    return samples
