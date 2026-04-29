# SPDX-License-Identifier: Apache-2.0
"""LongCat causal DMD denoising stage.

Mirrors :class:`CausalDMDDenosingStage` (Wan) but adapted to LongCat's
transformer interface, which exposes the streaming KV cache as a
``{block_idx: (k, v)}`` dict instead of Wan's pre-allocated ``kv_cache``
list with index tensors.

For each causal block (``chunk_size`` latent frames):

1. Run ``len(dmd_denoising_steps)`` few-step DMD denoising passes,
   reading from the cached K/V of all previously denoised blocks.
2. After the block is fully denoised, run one more forward with
   ``return_kv=True`` and a clean ``context_noise`` timestep to
   produce K/V for the cache, and write them into a pre-allocated
   buffer (Wan-style).

Buffer management mirrors the training-side
:class:`LongCatCausalModel` (fixed-size buffer per layer +
``write_idx`` pointer) but is inlined here to keep the inference and
training paths independent.
"""
from __future__ import annotations

from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class LongCatCausalDMDDenoisingStage(DenoisingStage):
    """Block-by-block DMD few-step denoising with LongCat KV cache."""

    def __init__(
        self,
        transformer,
        scheduler,
        vae=None,
        chunk_size: int = 3,
    ) -> None:
        super().__init__(transformer, scheduler)
        self.vae = vae
        # Latent frames per causal block. Defaults to the value used by
        # the training-side `SelfForcingMethod` config; can be overridden
        # by the caller if a different rollout shape is needed.
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

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
                f"({self.chunk_size}) for LongCat causal DMD denoising")
        num_blocks = t // self.chunk_size

        # LongCat uses a single text encoder; prompt_embeds is a list.
        prompt_embeds = batch.prompt_embeds[0]
        prompt_attention_mask = (batch.prompt_attention_mask[0]
                                 if batch.prompt_attention_mask else None)
        assert torch.isnan(prompt_embeds).sum() == 0

        # DMD few-step timesteps. Optionally remap through the scheduler
        # if warp_denoising_step is on (mirrors Wan's CausalDMDDenosingStage).
        dmd_timesteps = torch.tensor(
            fastvideo_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long,
        ).cpu()
        if getattr(fastvideo_args.pipeline_config, "warp_denoising_step",
                   False):
            scheduler_timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(),
                 torch.tensor([0], dtype=torch.float32)))
            dmd_timesteps = scheduler_timesteps[1000 - dmd_timesteps]
        dmd_timesteps = dmd_timesteps.to(device)

        context_noise = int(
            getattr(fastvideo_args.pipeline_config, "context_noise", 0))

        # Lazy-allocated KV buffer state.
        cache_state: dict[str, Any] | None = None
        start_index = 0

        with self.progress_bar(total=num_blocks *
                               len(dmd_timesteps)) as progress_bar:
            for _ in range(num_blocks):
                current_latents = latents[:, :,
                                          start_index:start_index +
                                          self.chunk_size, :, :].clone()
                # Track the original noisy latents (BTCHW) for DMD
                # noise->video conversion across inner steps.
                noise_latents_btchw = current_latents.permute(
                    0, 2, 1, 3, 4).contiguous()
                video_raw_latent_shape = noise_latents_btchw.shape

                for i, t_cur in enumerate(dmd_timesteps):
                    t_expand = t_cur.repeat(b)

                    kv_view, cached_frames = self._cache_view(cache_state)

                    # Transformer expects BCTHW (Conv3d patch_embed); leave
                    # input as-is. Output is BCTHW too — permute to BTCHW
                    # afterwards for the DMD scheduler routines.
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
                    pred_noise_btchw = pred_noise_bcthw.permute(0, 2, 1, 3,
                                                                 4)

                    # DMD few-step: convert pred_noise to pred_video
                    # using the FlowMatch scheduler.
                    pred_video_btchw = pred_noise_to_pred_video(
                        pred_noise=pred_noise_btchw.flatten(0, 1),
                        noise_input_latent=noise_latents_btchw.flatten(0, 1),
                        timestep=t_expand,
                        scheduler=self.scheduler,
                    ).unflatten(0, pred_noise_btchw.shape[:2])

                    if i < len(dmd_timesteps) - 1:
                        next_t = dmd_timesteps[i + 1] * torch.ones(
                            [1], dtype=torch.long, device=device)
                        gen = (batch.generator[0] if isinstance(
                            batch.generator, list) else batch.generator)
                        noise = torch.randn(
                            video_raw_latent_shape,
                            dtype=pred_video_btchw.dtype,
                            generator=gen,
                        ).to(device)
                        noise_latents_btchw = self.scheduler.add_noise(
                            pred_video_btchw.flatten(0, 1),
                            noise.flatten(0, 1),
                            next_t,
                        ).unflatten(0, pred_video_btchw.shape[:2])
                        current_latents = noise_latents_btchw.permute(
                            0, 2, 1, 3, 4).contiguous()
                    else:
                        current_latents = pred_video_btchw.permute(
                            0, 2, 1, 3, 4).contiguous()

                    if progress_bar is not None:
                        progress_bar.update()

                # Write the denoised block back to the full latent tensor.
                latents[:, :, start_index:start_index +
                        self.chunk_size, :, :] = current_latents

                # Refresh KV cache with the clean denoised context so the
                # next block can attend to it. Mirrors the training-side
                # `LongCatCausalModel.predict_noise_streaming` store_kv
                # path (return_kv=True + skip_crs_attn).
                t_context = torch.full(
                    (b, ),
                    context_noise,
                    device=device,
                    dtype=torch.long,
                )
                # Transformer wants BCTHW; current_latents already is.
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

    # ------------------------------------------------------------------
    # KV cache buffer (inlined from LongCatCausalModel — same pattern,
    # but no grad-checkpoint / autograd considerations needed here)
    # ------------------------------------------------------------------

    def _cache_view(
        self,
        cache_state: dict[str, Any] | None,
    ) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]] | None, int]:
        if cache_state is None:
            return None, 0
        widx = int(cache_state["write_idx"].item())
        if widx == 0:
            return None, 0
        view = {
            idx: (buf["k"][:, :, :widx, :], buf["v"][:, :, :widx, :])
            for idx, buf in cache_state["buffers"].items()
        }
        cached_frames = widx // cache_state["tokens_per_frame"]
        return view, cached_frames

    def _write_chunk_to_buffer(
        self,
        *,
        cache_state: dict[str, Any] | None,
        new_chunk: dict[int, tuple[torch.Tensor, torch.Tensor]],
        new_frames: int,
        num_frames_total: int,
        device: torch.device,
    ) -> dict[str, Any]:
        sample_k, _ = next(iter(new_chunk.values()))
        if sample_k.ndim != 4:
            raise ValueError(
                "Unexpected LongCat KV cache shape; expected [B, H, N, D], "
                f"got ndim={sample_k.ndim}")
        new_tokens = int(sample_k.shape[2])
        if new_tokens % new_frames != 0:
            raise ValueError(
                "LongCat KV cache token count is not divisible by the number "
                f"of frames in the cached chunk: {new_tokens} vs {new_frames}")
        tokens_per_frame = new_tokens // new_frames

        if cache_state is None:
            max_tokens = tokens_per_frame * int(num_frames_total)
            B = int(sample_k.shape[0])
            H = int(sample_k.shape[1])
            D = int(sample_k.shape[3])
            buffers: dict[int, dict[str, torch.Tensor]] = {}
            for idx, (k, v) in new_chunk.items():
                buffers[idx] = {
                    "k":
                    torch.zeros(
                        (B, H, max_tokens, D),
                        dtype=k.dtype,
                        device=k.device,
                    ),
                    "v":
                    torch.zeros(
                        (B, H, max_tokens, D),
                        dtype=v.dtype,
                        device=v.device,
                    ),
                }
            cache_state = {
                "buffers": buffers,
                "write_idx": torch.zeros((),
                                         dtype=torch.long,
                                         device=device),
                "tokens_per_frame": tokens_per_frame,
                "max_tokens": max_tokens,
            }
        else:
            if cache_state["tokens_per_frame"] != tokens_per_frame:
                raise ValueError(
                    "LongCat KV cache token density changed between blocks: "
                    f"{cache_state['tokens_per_frame']} vs {tokens_per_frame}"
                )

        widx = int(cache_state["write_idx"].item())
        new_widx = widx + new_tokens
        if new_widx > cache_state["max_tokens"]:
            raise ValueError(
                "LongCat KV cache buffer overflow: tried to write up to "
                f"{new_widx} tokens, capacity is {cache_state['max_tokens']}")

        for idx, (k, v) in new_chunk.items():
            buf = cache_state["buffers"].get(idx)
            if buf is None:
                raise ValueError(
                    f"LongCat returned new K/V for layer {idx} not present "
                    "in the pre-allocated buffer; the layer set must be "
                    "consistent across chunks")
            buf["k"][:, :, widx:new_widx, :].copy_(k.detach())
            buf["v"][:, :, widx:new_widx, :].copy_(v.detach())

        cache_state["write_idx"].fill_(new_widx)
        return cache_state

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds,
                         V.list_not_empty)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        return result
