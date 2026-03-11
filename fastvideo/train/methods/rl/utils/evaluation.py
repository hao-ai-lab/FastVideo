# SPDX-License-Identifier: Apache-2.0
"""Evaluation loop for RL training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.rl.utils.embeddings import (
    compute_text_embeddings,
)
from fastvideo.train.methods.rl.utils.pipeline import (
    wan_denoising_with_logprob,
)

logger = init_logger(__name__)


def eval_once(
    model,
    scheduler,
    test_dataloader,
    text_encoder,
    tokenizer,
    sample_neg_prompt_embeds: torch.Tensor,
    eval_reward_fn: Callable,
    global_step: int,
    ema_callback,
    *,
    eval_num_steps: int,
    eval_guidance_scale: float,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device,
    world_size: int,
    rank: int,
    is_main_process: bool,
    tracker: Any | None = None,
) -> dict[str, float]:
    """Run evaluation on test set.

    Args:
        ema_callback: An ``EMACallback`` instance (or
            ``None``).  Used to temporarily swap EMA
            weights into the transformer for evaluation.

    Returns:
        Dict of aggregated eval metrics.
    """
    model.transformer.eval()
    all_rewards: dict[str, list[float]] = {}

    # Use EMA context manager if available.
    if ema_callback is not None:
        ctx = ema_callback.ema_context(
            model.transformer
        )
    else:
        from contextlib import nullcontext

        ctx = nullcontext()

    with ctx:
        for batch_idx, (
            _epoch_tag,
            prompts,
            metadata,
        ) in enumerate(test_dataloader):
            prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoder,
                tokenizer,
                max_sequence_length=512,
                device=device,
            )

            with torch.no_grad():
                (
                    videos,
                    _latents,
                    _log_probs,
                    _kls,
                    _timesteps,
                ) = wan_denoising_with_logprob(
                    model,
                    scheduler,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=(
                        sample_neg_prompt_embeds
                    ),
                    num_inference_steps=eval_num_steps,
                    guidance_scale=eval_guidance_scale,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    deterministic=True,
                    sde_type="flow_sde",
                )

            rewards, _ = eval_reward_fn(
                videos, prompts, metadata
            )
            for key, val in rewards.items():
                if key not in all_rewards:
                    all_rewards[key] = []
                if isinstance(val, torch.Tensor):
                    all_rewards[key].extend(
                        val.detach().cpu().tolist()
                    )
                else:
                    all_rewards[key].append(float(val))

    # Aggregate metrics.
    metrics = {}
    for key, vals in all_rewards.items():
        avg = sum(vals) / max(len(vals), 1)
        metrics[f"eval_{key}"] = avg

    if is_main_process and tracker is not None:
        tracker.log(metrics, global_step)

    return metrics
