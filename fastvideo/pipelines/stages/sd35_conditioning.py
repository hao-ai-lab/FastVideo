# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.utils import PRECISION_TO_TYPE


class SD35LatentPreparationStage(PipelineStage):

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.height is None or batch.width is None:
            raise ValueError(
                "height/width must be set for SD35LatentPreparationStage")

        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            if not batch.prompt_embeds:
                raise ValueError("prompt or prompt_embeds must be provided")
            batch_size = batch.prompt_embeds[0].shape[0]

        batch_size *= batch.num_videos_per_prompt

        dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        device = get_local_torch_device()

        if isinstance(batch.generator, list) and len(
                batch.generator) != batch_size:
            raise ValueError(
                f"generator list length {len(batch.generator)} does not match batch_size {batch_size}"
            )

        in_channels = int(
            fastvideo_args.pipeline_config.dit_config.arch_config.in_channels)
        spatial_ratio = int(fastvideo_args.pipeline_config.vae_config.
                            arch_config.spatial_compression_ratio)
        h_lat = int(batch.height) // spatial_ratio
        w_lat = int(batch.width) // spatial_ratio
        shape = (batch_size, in_channels, 1, h_lat, w_lat)

        latents = batch.latents
        if latents is None:
            latents = randn_tensor(
                shape,
                generator=batch.generator,
                device=device,
                dtype=dtype,
            )
            if hasattr(self.scheduler, "init_noise_sigma"):
                latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(device=device, dtype=dtype)
            if hasattr(self.scheduler, "init_noise_sigma"):
                latents = latents * self.scheduler.init_noise_sigma

        batch.latents = latents
        batch.raw_latent_shape = shape
        return batch


class SD35ConditioningStage(PipelineStage):

    def __init__(self, text_encoders, tokenizers) -> None:
        super().__init__()
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

    @staticmethod
    def _tokenize(
        tokenizer: Any,
        prompts: str | list[str],
        tok_kwargs: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [prompts] if isinstance(prompts, str) else prompts
        enc = tokenizer(texts, **tok_kwargs)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        return input_ids, attention_mask

    @torch.no_grad()
    def _clip_pooled(
        self,
        text_encoder,
        tokenizer,
        prompts: str | list[str],
        tok_kwargs: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        input_ids, attention_mask = self._tokenize(tokenizer, prompts,
                                                   tok_kwargs, device)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            out = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            raise RuntimeError(
                "CLIP pooled output is required for SD3.5 conditioning")
        return pooled.to(dtype=dtype)

    @torch.no_grad()
    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if len(batch.prompt_embeds) < 3:
            raise ValueError(
                f"SD35ConditioningStage expects 3 prompt_embeds entries (2x CLIP + 1x T5), got {len(batch.prompt_embeds)}"
            )

        device = get_local_torch_device()
        target_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.dit_precision]

        clip_1 = batch.prompt_embeds[0].to(device=device, dtype=target_dtype)
        clip_2 = batch.prompt_embeds[1].to(device=device, dtype=target_dtype)
        t5 = batch.prompt_embeds[2].to(device=device, dtype=target_dtype)

        clip_prompt = torch.cat([clip_1, clip_2], dim=-1)
        if clip_prompt.shape[-1] > t5.shape[-1]:
            raise ValueError(
                f"CLIP prompt dim {clip_prompt.shape[-1]} exceeds T5 dim {t5.shape[-1]}"
            )
        clip_prompt = F.pad(clip_prompt,
                            (0, t5.shape[-1] - clip_prompt.shape[-1]))
        prompt_embeds = torch.cat([clip_prompt, t5], dim=-2)

        te_cfgs = fastvideo_args.pipeline_config.text_encoder_configs
        clip_tok_kwargs_1 = dict(getattr(te_cfgs[0], "tokenizer_kwargs", {}))
        clip_tok_kwargs_2 = dict(getattr(te_cfgs[1], "tokenizer_kwargs", {}))
        clip_tok_kwargs_1.setdefault("padding", "max_length")
        clip_tok_kwargs_2.setdefault("padding", "max_length")
        clip_tok_kwargs_1.setdefault("max_length", 77)
        clip_tok_kwargs_2.setdefault("max_length", 77)
        clip_tok_kwargs_1.setdefault("truncation", True)
        clip_tok_kwargs_2.setdefault("truncation", True)
        clip_tok_kwargs_1.setdefault("return_tensors", "pt")
        clip_tok_kwargs_2.setdefault("return_tensors", "pt")

        pooled_1 = self._clip_pooled(self.text_encoders[0], self.tokenizers[0],
                                     batch.prompt, clip_tok_kwargs_1, device,
                                     target_dtype)
        pooled_2 = self._clip_pooled(self.text_encoders[1], self.tokenizers[1],
                                     batch.prompt, clip_tok_kwargs_2, device,
                                     target_dtype)
        pooled = torch.cat([pooled_1, pooled_2], dim=-1)

        batch.extra["sd35_encoder_hidden_states"] = prompt_embeds
        batch.extra["sd35_pooled_projections"] = pooled

        if batch.do_classifier_free_guidance:
            if batch.negative_prompt_embeds is None or len(
                    batch.negative_prompt_embeds) < 3:
                raise ValueError(
                    "negative_prompt_embeds must contain 3 entries when CFG is enabled"
                )

            neg_clip_1 = batch.negative_prompt_embeds[0].to(device=device,
                                                            dtype=target_dtype)
            neg_clip_2 = batch.negative_prompt_embeds[1].to(device=device,
                                                            dtype=target_dtype)
            neg_t5 = batch.negative_prompt_embeds[2].to(device=device,
                                                        dtype=target_dtype)

            neg_clip_prompt = torch.cat([neg_clip_1, neg_clip_2], dim=-1)
            neg_clip_prompt = F.pad(
                neg_clip_prompt,
                (0, neg_t5.shape[-1] - neg_clip_prompt.shape[-1]))
            neg_prompt_embeds = torch.cat([neg_clip_prompt, neg_t5], dim=-2)

            negative_pooled_1 = self._clip_pooled(
                self.text_encoders[0],
                self.tokenizers[0],
                batch.negative_prompt if isinstance(batch.negative_prompt, str
                                                    | list) else "",
                clip_tok_kwargs_1,
                device,
                target_dtype,
            )
            negative_pooled_2 = self._clip_pooled(
                self.text_encoders[1],
                self.tokenizers[1],
                batch.negative_prompt if isinstance(batch.negative_prompt, str
                                                    | list) else "",
                clip_tok_kwargs_2,
                device,
                target_dtype,
            )
            neg_pooled = torch.cat([negative_pooled_1, negative_pooled_2],
                                   dim=-1)

            batch.extra[
                "sd35_negative_encoder_hidden_states"] = neg_prompt_embeds
            batch.extra["sd35_negative_pooled_projections"] = neg_pooled

        return batch


class SD35DenoisingStage(PipelineStage):
    """Denoising loop for SD3.5 (2D transformer + FlowMatch scheduler)."""

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    @staticmethod
    def _prepare_extra_func_kwargs(func, kwargs) -> dict[str, Any]:
        extra_kwargs: dict[str, Any] = {}
        sig = inspect.signature(func)
        for k, v in kwargs.items():
            if k in sig.parameters:
                extra_kwargs[k] = v
        return extra_kwargs

    @torch.no_grad()
    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.timesteps is None:
            raise ValueError("timesteps must be set before SD35DenoisingStage")
        if batch.latents is None:
            raise ValueError("latents must be set before SD35DenoisingStage")

        prompt_embeds: torch.Tensor = batch.extra["sd35_encoder_hidden_states"]
        pooled: torch.Tensor = batch.extra["sd35_pooled_projections"]

        neg_prompt_embeds: torch.Tensor | None = batch.extra.get(
            "sd35_negative_encoder_hidden_states")
        neg_pooled: torch.Tensor | None = batch.extra.get(
            "sd35_negative_pooled_projections")

        timesteps = batch.timesteps
        latents = batch.latents
        guidance_scale = float(batch.guidance_scale)

        target_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.dit_precision]
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        extra_step_kwargs = self._prepare_extra_func_kwargs(
            self.scheduler.step, {
                "generator":
                batch.generator[0]
                if isinstance(batch.generator, list) else batch.generator
            })

        for t in timesteps:
            latents_4d = latents.squeeze(2)

            if batch.do_classifier_free_guidance:
                if neg_prompt_embeds is None or neg_pooled is None:
                    raise ValueError(
                        "Missing negative conditioning tensors for CFG")
                latent_model_input = torch.cat([latents_4d] * 2, dim=0)
                cond_embeds = torch.cat([neg_prompt_embeds, prompt_embeds],
                                        dim=0)
                cond_pooled = torch.cat([neg_pooled, pooled], dim=0)
            else:
                latent_model_input = latents_4d
                cond_embeds = prompt_embeds
                cond_pooled = pooled

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            timestep = t.expand(latent_model_input.shape[0])

            with torch.autocast(
                    device_type="cuda",
                    dtype=target_dtype,
                    enabled=autocast_enabled
                    and (get_local_torch_device().type == "cuda"),
            ):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=cond_embeds,
                    pooled_projections=cond_pooled,
                    return_dict=False,
                )[0]

            if batch.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            noise_pred_5d = noise_pred.unsqueeze(2)
            latents = self.scheduler.step(noise_pred_5d,
                                          t,
                                          latents,
                                          return_dict=False,
                                          **extra_step_kwargs)[0]

        batch.latents = latents
        return batch


class SD35DecodingStage(PipelineStage):

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    @staticmethod
    def _denormalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
        # Prefer config fields to avoid diffusers deprecation warnings for direct
        # attribute access (vae.scaling_factor / vae.shift_factor).
        cfg = getattr(vae, "config", None)
        sf = getattr(cfg, "scaling_factor", None) if cfg is not None else None
        sh = getattr(cfg, "shift_factor", None) if cfg is not None else None

        if sf is None and hasattr(vae, "scaling_factor"):
            sf = vae.scaling_factor
        if sh is None and hasattr(vae, "shift_factor"):
            sh = vae.shift_factor

        if sf is not None:
            latents = latents / (sf.to(latents.device, latents.dtype)
                                 if isinstance(sf, torch.Tensor) else sf)
            if sh is not None:
                latents = latents + (sh.to(latents.device, latents.dtype)
                                     if isinstance(sh, torch.Tensor) else sh)
        return latents

    @torch.no_grad()
    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.latents is None:
            raise ValueError("latents must be set before SD35DecodingStage")

        device = get_local_torch_device()
        latents_5d = batch.latents.to(device)
        latents_4d = latents_5d.squeeze(2)

        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        latents_4d = self._denormalize_latents(latents_4d, self.vae)

        with torch.autocast(
                device_type="cuda",
                dtype=vae_dtype,
                enabled=autocast_enabled and (device.type == "cuda"),
        ):
            if not autocast_enabled:
                latents_4d = latents_4d.to(dtype=vae_dtype)
            dec = self.vae.decode(latents_4d)
            image = dec.sample if hasattr(dec, "sample") else dec[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        batch.output = image.unsqueeze(2).detach().float().cpu()
        return batch
