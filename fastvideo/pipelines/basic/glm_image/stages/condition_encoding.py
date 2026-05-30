# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.image_processor import ImageProcessor
from fastvideo.models.dits.glm_image import GlmImageKVCache
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage

_CONDITION_MULTIPLE_OF = 16  # vae_scale_factor (8) * DiT patch_size (2)


class GlmImageConditionEncodingStage(PipelineStage):

    def __init__(self, vae, transformer) -> None:
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.image_processor = ImageProcessor(vae_scale_factor=_CONDITION_MULTIPLE_OF)

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.pil_image is None:
            return batch

        device = get_local_torch_device()
        dtype = torch.bfloat16
        self.vae.to(device)

        prior_ids = batch.extra["glm_prior_token_image_ids"].to(device)
        if prior_ids.dim() == 1:
            prior_ids = prior_ids.unsqueeze(0)

        # Latent patch count must match the source prior tokens; mismatch is fatal.
        src_grid = batch.extra["glm_source_image_grid_thw"][0]
        cond_h = int(src_grid[1]) * _CONDITION_MULTIPLE_OF
        cond_w = int(src_grid[2]) * _CONDITION_MULTIPLE_OF
        cond_img = self.image_processor.preprocess(batch.pil_image, cond_h, cond_w).to(device=device,
                                                                                       dtype=torch.float32)
        latent = self.vae.encode(cond_img).latent_dist.mode()
        cfg = self.vae.config
        mean = torch.tensor(cfg.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(cfg.latents_std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        latent = ((latent - mean) / std).to(dtype)

        kv_caches = GlmImageKVCache(num_layers=self.transformer.num_layers)
        empty_text = batch.prompt_embeds[0][:1, :0, :].to(device=device, dtype=dtype)
        with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=batch):
            self.transformer(
                hidden_states=latent,
                encoder_hidden_states=empty_text,
                prior_token_id=prior_ids,
                prior_token_drop=torch.zeros((prior_ids.shape[0], ), dtype=torch.bool, device=device),
                timestep=torch.zeros((1, ), device=device),
                target_size=torch.tensor([tuple(cond_img.shape[-2:])], device=device, dtype=torch.long),
                crop_coords=torch.zeros((1, 2), device=device, dtype=torch.long),
                kv_caches=kv_caches,
                kv_caches_mode="write",
            )
        batch.extra["glm_kv_caches"] = kv_caches
        return batch
