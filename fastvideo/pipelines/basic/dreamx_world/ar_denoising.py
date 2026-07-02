# SPDX-License-Identifier: Apache-2.0
"""DreamX-World autoregressive causal denoising stage."""

from __future__ import annotations

from typing import Any

import torch
from tqdm.auto import tqdm

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.dreamx_world.stages import DREAMX_Y_CAMERA_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage


class DreamXWorldARCausalDenoisingStage(DenoisingStage):
    """Official DreamX AR-forcing denoising loop with KV cache."""

    _AR_NOISE_SEED_OFFSET = 1_000_003

    def __init__(self, transformer, scheduler, pipeline=None, vae=None) -> None:
        super().__init__(transformer=transformer, scheduler=scheduler, pipeline=pipeline, vae=vae)
        self.num_transformer_blocks = len(self.transformer.blocks)
        self.num_frame_per_block = int(getattr(self.transformer, "num_frame_per_block", 3))
        self.local_attn_size = int(getattr(self.transformer, "local_attn_size", 12))

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.latents is not None, "latents must be prepared before DreamX AR denoising"
        assert batch.prompt_embeds, "prompt embeds must be prepared before DreamX AR denoising"
        latents = batch.latents
        device = latents.device
        target_dtype = torch.bfloat16
        autocast_enabled = device.type == "cuda" and not fastvideo_args.disable_autocast

        frame_seq_length = (latents.shape[-2] // self.transformer.patch_size[1]) * (latents.shape[-1] //
                                                                                    self.transformer.patch_size[2])
        timesteps = torch.tensor(
            tuple(getattr(fastvideo_args.pipeline_config, "dmd_denoising_steps", (1000, 750, 500, 250))),
            dtype=torch.long,
        ).cpu()
        if getattr(fastvideo_args.pipeline_config, "warp_denoising_step", True):
            self.scheduler.set_timesteps(1000)
            scheduler_timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(device)

        if latents.shape[2] % self.num_frame_per_block != 0:
            raise ValueError("DreamX AR latent frames must be divisible by num_frame_per_block")

        y_camera = batch.extra.get(DREAMX_Y_CAMERA_KEY, batch.extra.get("y_camera"))
        if isinstance(y_camera, dict):
            y_camera = {
                k: v.to(device=device, dtype=target_dtype) if torch.is_tensor(v) else v
                for k, v in y_camera.items()
            }

        if batch.image_latent is not None and batch.image_latent.shape[1] == latents.shape[1]:
            latents[:, :, :batch.image_latent.shape[2]] = batch.image_latent.to(device=device, dtype=latents.dtype)

        kv_cache = self._initialize_kv_cache(latents.shape[0], target_dtype, device, frame_seq_length)
        crossattn_cache = self._initialize_crossattn_cache(latents.shape[0], target_dtype, device)
        prompt = batch.prompt_embeds[0]
        if torch.is_tensor(prompt):
            prompt = prompt.to(device=device, dtype=target_dtype)
            context = [sample for sample in prompt]
        else:
            context = prompt

        num_blocks = latents.shape[2] // self.num_frame_per_block
        start = 0
        first_frame_mask = torch.ones_like(latents)
        first_frame_mask[:, :, 0] = 0
        base_generator = batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        noise_generator = self._make_noise_generator(base_generator, device)

        with tqdm(total=num_blocks * len(timesteps), desc="DreamX AR denoising", leave=False) as progress:
            for _ in range(num_blocks):
                current_num_frames = self.num_frame_per_block
                block_latents = latents[:, :, start:start + current_num_frames]
                noisy_input = block_latents.clone()
                mask_block = first_frame_mask[:, :, start:start + current_num_frames]
                camera_block = self._slice_camera(y_camera, start, current_num_frames)

                for idx, current_timestep in enumerate(timesteps):
                    timestep = torch.full(
                        (latents.shape[0], current_num_frames * frame_seq_length),
                        int(current_timestep.item()),
                        device=device,
                        dtype=torch.long,
                    )
                    if start == 0:
                        timestep[:, :frame_seq_length] = 0
                    with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                        denoised = self.transformer(
                            hidden_states=block_latents.to(target_dtype),
                            encoder_hidden_states=torch.stack(context).to(target_dtype),
                            timestep=timestep,
                            y_camera=camera_block,
                            kv_cache=kv_cache,
                            crossattn_cache=crossattn_cache,
                            current_start=start * frame_seq_length,
                        )
                    denoised = denoised.to(latents.dtype)
                    if idx < len(timesteps) - 1:
                        next_timestep = torch.full((latents.shape[0], current_num_frames),
                                                   int(timesteps[idx + 1].item()),
                                                   device=device,
                                                   dtype=torch.long)
                        noise_kwargs = {"device": device, "dtype": denoised.dtype}
                        if noise_generator is not None:
                            noise_kwargs["generator"] = noise_generator
                        noise = torch.randn(denoised.permute(0, 2, 1, 3, 4).shape, **noise_kwargs)
                        block_btchw = self.scheduler.add_noise(
                            denoised.permute(0, 2, 1, 3, 4).flatten(0, 1),
                            noise.flatten(0, 1),
                            next_timestep.flatten(),
                        ).unflatten(0, (latents.shape[0], current_num_frames))
                        block_latents = block_btchw.permute(0, 2, 1, 3, 4)
                        block_latents = block_latents * mask_block + noisy_input * (1 - mask_block)
                    else:
                        block_latents = denoised * mask_block + noisy_input * (1 - mask_block)
                    progress.update()

                latents[:, :, start:start + current_num_frames] = block_latents
                self._update_context_cache(block_latents, context, camera_block, kv_cache, crossattn_cache, start,
                                           frame_seq_length, target_dtype, autocast_enabled,
                                           float(getattr(fastvideo_args.pipeline_config, "context_noise", 0.1)))
                start += current_num_frames

        batch.latents = latents
        return batch

    def _make_noise_generator(self, generator: torch.Generator | None, device: torch.device) -> torch.Generator | None:
        if generator is None:
            return None
        if getattr(generator, "device", None) == device:
            return generator
        seed = int(generator.initial_seed()) + self._AR_NOISE_SEED_OFFSET
        return torch.Generator(device=device).manual_seed(seed)

    @staticmethod
    def _context_noise_timestep(context_noise: float) -> int:
        if 0.0 < context_noise <= 1.0:
            return int(context_noise * 1000)
        return int(context_noise)

    def _slice_camera(self, y_camera: Any, start: int, num_frames: int):
        if not isinstance(y_camera, dict):
            return y_camera
        return {
            "viewmats": y_camera["viewmats"][:, start:start + num_frames],
            "K": y_camera["K"][:, start:start + num_frames],
        }

    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device,
                             frame_seq_length: int) -> list[dict[str, Any]]:
        size = self.local_attn_size * frame_seq_length if self.local_attn_size != -1 else 18480
        heads = self.transformer.num_attention_heads
        head_dim = self.transformer.attention_head_dim
        cam_self_attn = next(
            (getattr(block, "cam_self_attn", None)
             for block in self.transformer.blocks if getattr(block, "cam_self_attn", None) is not None),
            None,
        )

        caches = []
        for _ in range(self.num_transformer_blocks):
            cache = {
                "k": torch.zeros(batch_size, size, heads, head_dim, dtype=dtype, device=device),
                "v": torch.zeros(batch_size, size, heads, head_dim, dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            if cam_self_attn is not None:
                cam_heads = int(cam_self_attn.num_heads)
                cam_head_dim = int(cam_self_attn.head_dim)
                cache.update({
                    "prope_k":
                    torch.zeros(batch_size, size, cam_heads, cam_head_dim, dtype=dtype, device=device),
                    "prope_v":
                    torch.zeros(batch_size, size, cam_heads, cam_head_dim, dtype=dtype, device=device),
                    "prope_global_end_index":
                    torch.tensor([0], dtype=torch.long, device=device),
                    "prope_local_end_index":
                    torch.tensor([0], dtype=torch.long, device=device),
                })
            caches.append(cache)
        return caches

    def _initialize_crossattn_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        heads = self.transformer.num_attention_heads
        head_dim = self.transformer.attention_head_dim
        return [{
            "k": torch.zeros(batch_size, 512, heads, head_dim, dtype=dtype, device=device),
            "v": torch.zeros(batch_size, 512, heads, head_dim, dtype=dtype, device=device),
            "is_init": False,
        } for _ in range(self.num_transformer_blocks)]

    def _update_context_cache(self, block_latents: torch.Tensor, context: Any, camera_block: Any,
                              kv_cache: list[dict[str, Any]], crossattn_cache: list[dict[str, Any]], start: int,
                              frame_seq_length: int, target_dtype: torch.dtype, autocast_enabled: bool,
                              context_noise: float) -> None:
        timestep = torch.full(
            (block_latents.shape[0], block_latents.shape[2] * frame_seq_length),
            self._context_noise_timestep(context_noise),
            device=block_latents.device,
            dtype=torch.long,
        )
        with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
            self.transformer(
                hidden_states=block_latents.to(target_dtype),
                encoder_hidden_states=torch.stack(context).to(target_dtype),
                timestep=timestep,
                y_camera=camera_block,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=start * frame_seq_length,
            )
