# SPDX-License-Identifier: Apache-2.0
"""Streaming block-autoregressive DMD sampler for the MLX causal Wan runtime.

Track C, rung 5. Drives ``MLXCausalWanDiT`` the way the torch causal DMD stage
(``fastvideo/pipelines/stages/causal_denoising.py``) does: one frame-block at a
time, each block denoised over the few-step DMD schedule while the KV cache holds
the *clean* latents of every earlier block. After a block's few steps, a
context-update forward at ``timestep=context_noise`` rewrites that block's K/V
with its clean values before the next block starts — so subsequent blocks attend
to clean history. Yields each block's latents as it finalizes, which is what
makes the preview stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Iterator

from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, dmd_step, pred_noise_to_pred_video

if TYPE_CHECKING:
    import mlx.core as mx

    from fastvideo.mlx_runtime.causal_dit import MLXCausalWanDiT


def build_dmd_schedule(
    dmd_denoising_steps: list[int],
    *,
    flow_shift: float = 8.0,
    warp_denoising_step: bool = True,
) -> tuple[MLXDMDSchedule, list[float]]:
    """Return ``(schedule, timesteps)`` for causal DMD.

    Mirrors the torch stage: the raw denoising steps (e.g. ``[1000, 750, 500,
    250]``) index into a warped 1000-step flow-match schedule when
    ``warp_denoising_step`` is set.
    """
    import torch

    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    scheduler.set_timesteps(1000, device="cpu")
    steps = torch.tensor(dmd_denoising_steps, dtype=torch.long)
    if warp_denoising_step:
        warped = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        timesteps = warped[1000 - steps]
    else:
        timesteps = steps.to(torch.float32)
    schedule = MLXDMDSchedule.from_torch_scheduler(scheduler)
    return schedule, [float(t) for t in timesteps]


def stream_causal_latents(
    model: MLXCausalWanDiT,
    encoder_hidden_states: mx.array,
    noise_latents: mx.array,
    cos_full: mx.array,
    sin_full: mx.array,
    schedule: MLXDMDSchedule,
    timesteps: list[float],
    *,
    frame_seqlen: int,
    context_noise: float = 0.0,
    seed: int = 0,
) -> Iterator[tuple[int, mx.array]]:
    """Yield ``(block_index, clean_block_latents)`` as each block finalizes.

    ``noise_latents`` is ``[B, C, T, H, W]`` pure noise; ``cos_full``/``sin_full``
    are the rotary tables for the whole clip (frame-major), sliced per block.
    """
    import mlx.core as mx

    mx.random.seed(seed)
    dtype = noise_latents.dtype
    batch, _, total_frames, _, _ = noise_latents.shape
    nfb = model.num_frames_per_block
    if total_frames % nfb != 0:
        raise ValueError(f"total latent frames {total_frames} not divisible by num_frames_per_block {nfb}")

    kv_caches, crossattn_caches = model.allocate_caches(batch=batch, frame_seqlen=frame_seqlen, dtype=dtype)
    block_tokens = nfb * frame_seqlen
    last = len(timesteps) - 1

    for block_index in range(total_frames // nfb):
        start = block_index * nfb
        current_start = start * frame_seqlen
        cos_blk = cos_full[current_start:current_start + block_tokens]
        sin_blk = sin_full[current_start:current_start + block_tokens]
        current = noise_latents[:, :, start:start + nfb]

        for i, timestep in enumerate(timesteps):
            ts = mx.full((batch, nfb), timestep, dtype=mx.float32)
            pred_noise = model.forward_chunk(current,
                                             encoder_hidden_states,
                                             ts,
                                             cos_blk,
                                             sin_blk,
                                             kv_caches,
                                             crossattn_caches,
                                             current_start=current_start)
            noise_input = current.astype(mx.float32)
            pred = pred_noise.astype(mx.float32)
            if i < last:
                renoise = mx.random.normal(current.shape).astype(mx.float32)
                current = dmd_step(latents=noise_input,
                                   noise_input_latent=noise_input,
                                   pred_noise=pred,
                                   schedule=schedule,
                                   timestep=timestep,
                                   next_timestep=timesteps[i + 1],
                                   noise=renoise).astype(dtype)
            else:
                current = pred_noise_to_pred_video(pred, noise_input, schedule.sigma_for(timestep)).astype(dtype)

        # Context update: rewrite this block's K/V from its clean latents so later
        # blocks attend to clean history (output discarded).
        ts_ctx = mx.full((batch, nfb), context_noise, dtype=mx.float32)
        model.forward_chunk(current,
                            encoder_hidden_states,
                            ts_ctx,
                            cos_blk,
                            sin_blk,
                            kv_caches,
                            crossattn_caches,
                            current_start=current_start)
        mx.eval(current)
        yield block_index, current
