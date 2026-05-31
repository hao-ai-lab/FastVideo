# SPDX-License-Identifier: Apache-2.0
"""CFG convention (both denoise paths): row 0 conditional (positive), row 1 unconditional."""
from __future__ import annotations

import torch

from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.platforms import AttentionBackendEnum


class GlmImageDenoisingStage(DenoisingStage):

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        latents = getattr(batch, "latent", getattr(batch, "latents", None))
        result.add_check("latents", latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        return result

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = get_local_torch_device()
        dtype = torch.bfloat16
        guidance_scale = getattr(batch, "guidance_scale", 1.5)
        do_cfg = guidance_scale > 1.0

        latents = getattr(batch, "latent", getattr(batch, "latents", None))
        if latents is None:
            raise ValueError("No latents found in batch.")
        if latents.dim() == 5:
            latents = latents.squeeze(2)

        prompt_embeds = batch.prompt_embeds[0]
        text_attention_mask = getattr(batch, "attention_mask", None)
        timesteps = batch.timesteps

        patch_size = self.transformer.patch_size
        _, _, h, w = latents.shape
        image_seq_length = (h // patch_size) * (w // patch_size)
        text_seq_length = prompt_embeds.shape[1] if prompt_embeds.dim() >= 2 else 0

        first_block = self.transformer.transformer_blocks[0]
        backend = getattr(first_block.attn1.attn, "backend", None)
        sdpa = backend == AttentionBackendEnum.TORCH_SDPA and text_attention_mask is not None

        kv_caches = batch.extra.get("glm_kv_caches")
        if kv_caches is None:
            self._denoise_t2i(batch, latents, prompt_embeds, text_attention_mask, timesteps, do_cfg, guidance_scale,
                              text_seq_length, image_seq_length, sdpa, device, dtype)
        else:
            self._denoise_i2i(batch, latents, prompt_embeds, text_attention_mask, timesteps, do_cfg, guidance_scale,
                              text_seq_length, image_seq_length, sdpa, kv_caches, device, dtype)
        return batch

    def _denoise_t2i(self, batch, latents, prompt_embeds, text_attention_mask, timesteps, do_cfg, guidance_scale,
                     text_seq_length, image_seq_length, sdpa, device, dtype) -> None:
        num_inference_steps = batch.num_inference_steps
        bs = 2 if do_cfg else 1
        target_size = torch.tensor([[batch.height, batch.width]], device=device, dtype=torch.long).repeat(bs, 1)
        crop_coords = torch.zeros((bs, 2), device=device, dtype=torch.long)

        prior_token_id = batch.prior_token_id
        if do_cfg and prior_token_id.shape[0] == 1:
            prior_token_id = prior_token_id.repeat(2, 1)
        if do_cfg:
            prior_token_drop = torch.tensor([False, True], device=device)
        else:
            prior_token_drop = getattr(batch, "prior_token_drop", torch.tensor([False], device=device))

        attention_mask_kv = None
        if sdpa:
            if (text_attention_mask.shape[0] == 1 and bs > 1):
                text_attention_mask = text_attention_mask.repeat(bs, 1)
            mix_attn_mask = torch.ones((bs, text_seq_length + image_seq_length), device=device, dtype=torch.float32)
            mix_attn_mask[:, :text_seq_length] = (text_attention_mask.float().to(device))
            attention_mask_kv = (mix_attn_mask > 0).unsqueeze(1).unsqueeze(2)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).to(dtype)
                t_expand = t.expand(latent_model_input.shape[0]) - 1

                attn_metadata = (SDPAMetadata(current_timestep=i, attn_mask=attention_mask_kv)
                                 if attention_mask_kv is not None else None)

                with torch.no_grad(), set_forward_context(current_timestep=i,
                                                          attn_metadata=attn_metadata,
                                                          forward_batch=batch):
                    noise_pred = self.transformer(
                        latent_model_input,
                        prompt_embeds,
                        prior_token_id,
                        prior_token_drop,
                        t_expand,
                        target_size,
                        crop_coords,
                    )

                    if do_cfg:
                        noise_pred_cond, noise_pred_uncond = noise_pred.float().chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        guidance_rescale = getattr(batch, "guidance_rescale", 0.0)
                        if guidance_rescale > 0.0:
                            dims = list(range(1, noise_pred_cond.ndim))
                            std_text = noise_pred_cond.std(dim=dims, keepdim=True)
                            std_cfg = noise_pred.std(dim=dims, keepdim=True)
                            rescaled = noise_pred * (std_text / std_cfg)
                            noise_pred = (guidance_rescale * rescaled + (1 - guidance_rescale) * noise_pred)

                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()

        batch.latents = latents.unsqueeze(2)

    def _denoise_i2i(self, batch, latents, prompt_embeds, text_attention_mask, timesteps, do_cfg, guidance_scale,
                     text_seq_length, image_seq_length, sdpa, kv_caches, device, dtype) -> None:
        """Two separate transformer calls (cond reads the cache, uncond skips it):
        the cache mode is one global flag with batch-1 k/v, so a 2-row CFG call
        cannot express both."""
        num_inference_steps = batch.num_inference_steps
        target_size = torch.tensor([[batch.height, batch.width]], device=device, dtype=torch.long)
        crop_coords = torch.zeros((1, 2), device=device, dtype=torch.long)
        prior_token_id = batch.prior_token_id[:1]
        drop_keep = torch.zeros((1, ), dtype=torch.bool, device=device)
        drop_all = torch.ones((1, ), dtype=torch.bool, device=device)
        cache_len = kv_caches[0].k_cache.shape[1] if kv_caches[0].k_cache is not None else 0

        def _mask(row: int, with_cache: bool):
            if not sdpa:
                return None
            prefix = cache_len if with_cache else 0
            m = torch.ones((1, prefix + text_seq_length + image_seq_length), device=device, dtype=torch.float32)
            m[:, prefix:prefix + text_seq_length] = text_attention_mask[row:row + 1].float().to(device)
            return (m > 0).unsqueeze(1).unsqueeze(2)

        cond_mask = _mask(0, with_cache=True)
        uncond_mask = _mask(1, with_cache=False) if do_cfg else None
        pos_embeds = prompt_embeds[:1]
        neg_embeds = prompt_embeds[1:2] if do_cfg else None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(latents, t).to(dtype)
                t_expand = t.expand(1) - 1

                cond_meta = (SDPAMetadata(current_timestep=i, attn_mask=cond_mask) if cond_mask is not None else None)
                with torch.no_grad(), set_forward_context(current_timestep=i,
                                                          attn_metadata=cond_meta,
                                                          forward_batch=batch):
                    noise_pred = self.transformer(latent_model_input,
                                                  pos_embeds,
                                                  prior_token_id,
                                                  drop_keep,
                                                  t_expand,
                                                  target_size,
                                                  crop_coords,
                                                  kv_caches=kv_caches,
                                                  kv_caches_mode="read")

                if do_cfg:
                    uncond_meta = (SDPAMetadata(current_timestep=i, attn_mask=uncond_mask)
                                   if uncond_mask is not None else None)
                    with torch.no_grad(), set_forward_context(current_timestep=i,
                                                              attn_metadata=uncond_meta,
                                                              forward_batch=batch):
                        noise_pred_uncond = self.transformer(latent_model_input,
                                                             neg_embeds,
                                                             prior_token_id,
                                                             drop_all,
                                                             t_expand,
                                                             target_size,
                                                             crop_coords,
                                                             kv_caches=kv_caches,
                                                             kv_caches_mode="skip")
                    noise_pred = noise_pred_uncond.float() + guidance_scale * (noise_pred.float() -
                                                                               noise_pred_uncond.float())

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()

        kv_caches.clear()
        batch.latents = latents.unsqueeze(2)
