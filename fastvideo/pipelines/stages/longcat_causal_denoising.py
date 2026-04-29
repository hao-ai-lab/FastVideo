# SPDX-License-Identifier: Apache-2.0
"""LongCat causal multi-step denoising stage.

Block-by-block causal inference with KV caching, using the full
scheduler timestep schedule (40-50 steps) rather than DMD few-step.
This is the right inference pipeline for DFSFT-trained LongCat models.

Mirrors :class:`CausalDenoisingStage` (Wan) but adapted to LongCat's
``kv_cache_dict`` interface — same buffer-based KV cache as
:class:`LongCatCausalDMDDenoisingStage` (which we subclass for the
``_cache_view`` / ``_write_chunk_to_buffer`` helpers and constructor).
The inner loop swaps DMD's ``pred_noise_to_pred_video`` for the
standard ``scheduler.step``, with the LongCat-convention noise
negation (matches :class:`LongCatDenoisingStage`).
"""
from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.longcat_causal_dmd_denoising import (
    LongCatCausalDMDDenoisingStage, )

logger = init_logger(__name__)


class LongCatCausalDenoisingStage(LongCatCausalDMDDenoisingStage):
    """Multi-step block-by-block denoising for LongCat with KV cache.

    Each block is fully denoised through ``num_inference_steps``
    scheduler steps before moving to the next block. After each block
    is denoised, one extra forward with ``return_kv=True`` is run on
    the clean denoised context to refresh the KV cache (same pattern
    as DMD).
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = ((target_dtype != torch.float32)
                            and not fastvideo_args.disable_autocast)

        latents = batch.latents
        assert latents is not None
        b, c, t, h, w = latents.shape
        device = latents.device

        if t % self.chunk_size != 0:
            raise ValueError(
                f"num_latent_frames ({t}) must be divisible by chunk_size "
                f"({self.chunk_size}) for LongCat causal denoising")
        num_blocks = t // self.chunk_size

        prompt_embeds = batch.prompt_embeds[0]
        prompt_attention_mask = (batch.prompt_attention_mask[0]
                                 if batch.prompt_attention_mask else None)
        assert torch.isnan(prompt_embeds).sum() == 0

        num_inference_steps = batch.num_inference_steps
        if num_inference_steps is None or num_inference_steps <= 0:
            raise ValueError(
                "num_inference_steps must be a positive int for LongCat "
                f"causal multi-step denoising; got {num_inference_steps}")

        context_noise = int(
            getattr(fastvideo_args.pipeline_config, "context_noise", 0))

        cache_state: dict | None = None
        start_index = 0

        with self.progress_bar(total=num_blocks *
                               num_inference_steps) as progress_bar:
            for _ in range(num_blocks):
                current_latents = latents[:, :,
                                          start_index:start_index +
                                          self.chunk_size, :, :].clone()

                # Reset scheduler per block so its multi-step history
                # starts fresh for the new chunk.
                self.scheduler.set_timesteps(num_inference_steps,
                                             device=device)
                timesteps = self.scheduler.timesteps

                for i, t_cur in enumerate(timesteps):
                    t_expand = t_cur.expand(b)

                    kv_view, cached_frames = self._cache_view(cache_state)

                    # Transformer expects BCTHW input.
                    latent_input_bcthw = current_latents.to(target_dtype)

                    with torch.autocast(
                            device_type="cuda",
                            dtype=target_dtype,
                            enabled=autocast_enabled,
                    ), set_forward_context(
                            current_timestep=i,
                            attn_metadata=None,
                            forward_batch=batch,
                    ):
                        pred_noise_bcthw = self.transformer(
                            hidden_states=latent_input_bcthw,
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            timestep=t_expand,
                            num_cond_latents=cached_frames,
                            kv_cache_dict=kv_view,
                            kv_cache_start_frame=0,
                        )
                    if isinstance(pred_noise_bcthw, tuple):
                        raise RuntimeError(
                            "LongCat transformer returned a tuple when "
                            "return_kv=False")

                    # LongCat scheduler convention: negate noise pred
                    # before scheduler.step (matches LongCatDenoisingStage,
                    # the regular bidirectional pipeline).
                    pred_noise_bcthw = -pred_noise_bcthw

                    # scheduler.step works on flat [B*T, C, H, W] tensors
                    # with BTCHW ordering, so permute first.
                    pred_noise_btchw = pred_noise_bcthw.permute(
                        0, 2, 1, 3, 4)
                    latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
                    nf = self.chunk_size
                    pred_noise_flat = pred_noise_btchw.flatten(0, 1)
                    latents_flat = latents_btchw.flatten(0, 1)
                    updated_flat = self.scheduler.step(
                        pred_noise_flat,
                        t_cur,
                        latents_flat,
                        return_dict=False,
                    )[0]
                    current_latents = updated_flat.unflatten(
                        0, (b, nf)).permute(0, 2, 1, 3, 4).contiguous()

                    if progress_bar is not None:
                        progress_bar.update()

                # Write the denoised block back.
                latents[:, :, start_index:start_index +
                        self.chunk_size, :, :] = current_latents

                # Refresh KV cache with clean denoised context.
                t_context = torch.full(
                    (b, ),
                    context_noise,
                    device=device,
                    dtype=torch.long,
                )
                context_bcthw = current_latents.to(target_dtype)
                _, cached_frames_for_write = self._cache_view(cache_state)

                with torch.autocast(
                        device_type="cuda",
                        dtype=target_dtype,
                        enabled=autocast_enabled,
                ), set_forward_context(
                        current_timestep=0,
                        attn_metadata=None,
                        forward_batch=batch,
                ):
                    out = self.transformer(
                        hidden_states=context_bcthw,
                        encoder_hidden_states=prompt_embeds,
                        timestep=t_context,
                        num_cond_latents=cached_frames_for_write,
                        return_kv=True,
                        skip_crs_attn=True,
                    )
                if not isinstance(out, tuple) or len(out) != 2:
                    raise RuntimeError(
                        "LongCat transformer did not return (output, kv) "
                        "with return_kv=True")
                _, new_chunk = out
                if not new_chunk:
                    raise RuntimeError(
                        "LongCat transformer returned no KV cache when "
                        "return_kv=True")

                cache_state = self._write_chunk_to_buffer(
                    cache_state=cache_state,
                    new_chunk=new_chunk,
                    new_frames=self.chunk_size,
                    num_frames_total=t,
                    device=device,
                )

                start_index += self.chunk_size

        batch.latents = latents
        return batch
