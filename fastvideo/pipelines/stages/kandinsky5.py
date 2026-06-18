# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib

import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import TransformerLoader, VAELoader
from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.decoding import DecodingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE


class Kandinsky5LatentPreparationStage(PipelineStage):

    def __init__(self, scheduler, transformer) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.transformer = transformer

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.height is None or batch.width is None:
            raise ValueError("height and width must be provided for Kandinsky5.")
        height = int(batch.height)
        width = int(batch.width)
        num_frames = int(batch.num_frames)

        temporal_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        spatial_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        patch_size = fastvideo_args.pipeline_config.dit_config.arch_config.patch_size

        if num_frames % temporal_ratio != 1:
            num_frames = num_frames // temporal_ratio * temporal_ratio + 1
            batch.num_frames = num_frames

        required_divisor = spatial_ratio * patch_size[1]
        if height % required_divisor != 0 or width % required_divisor != 0:
            raise ValueError(f"Kandinsky5 height/width must be divisible by {required_divisor}; "
                             f"got height={height}, width={width}.")

        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds[0].shape[0]
        batch_size *= batch.num_videos_per_prompt

        dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        device = get_local_torch_device()
        num_latent_frames = (num_frames - 1) // temporal_ratio + 1
        num_channels = getattr(
            self.transformer,
            "in_visual_dim",
            fastvideo_args.pipeline_config.dit_config.arch_config.in_visual_dim,
        )
        shape = (
            batch_size,
            num_latent_frames,
            height // spatial_ratio,
            width // spatial_ratio,
            num_channels,
        )

        if isinstance(batch.generator, list) and len(batch.generator) != batch_size:
            raise ValueError(f"generator list length {len(batch.generator)} does not match batch size {batch_size}.")

        if batch.latents is None:
            latents = randn_tensor(shape, generator=batch.generator, device=device, dtype=dtype)
            if hasattr(self.scheduler, "init_noise_sigma"):
                latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = batch.latents.to(device=device, dtype=dtype)

        if getattr(self.transformer, "visual_cond", False) and latents.shape[-1] == num_channels:
            visual_cond = torch.zeros_like(latents)
            visual_cond_mask = torch.zeros(
                (*latents.shape[:-1], 1),
                device=latents.device,
                dtype=latents.dtype,
            )
            latents = torch.cat([latents, visual_cond, visual_cond_mask], dim=-1)

        batch.latents = latents
        batch.raw_latent_shape = (
            batch_size,
            num_channels,
            num_latent_frames,
            height // spatial_ratio,
            width // spatial_ratio,
        )
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result


class Kandinsky5DenoisingStage(PipelineStage):

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    @staticmethod
    def _scale_factor(height: int, width: int) -> tuple[float, float, float]:
        if 480 <= height <= 854 and 480 <= width <= 854:
            return (1.0, 2.0, 2.0)
        return (1.0, 3.16, 3.16)

    @staticmethod
    def _text_rope_pos(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        seq_len = int(mask.sum(1).max().item())
        return torch.arange(seq_len, device=device)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.timesteps is None:
            raise ValueError("timesteps must be prepared before Kandinsky5 denoising.")
        if batch.latents is None:
            raise ValueError("latents must be prepared before Kandinsky5 denoising.")
        if not fastvideo_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(fastvideo_args.model_paths["transformer"], fastvideo_args)
            fastvideo_args.model_loaded["transformer"] = True

        device = get_local_torch_device()
        target_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        autocast_enabled = target_dtype != torch.float32 and not fastvideo_args.disable_autocast
        latents = batch.latents
        num_channels = getattr(
            self.transformer,
            "in_visual_dim",
            fastvideo_args.pipeline_config.dit_config.arch_config.in_visual_dim,
        )

        prompt_embeds = batch.prompt_embeds[0].to(device=device, dtype=target_dtype)
        pooled = batch.prompt_embeds[1].to(device=device, dtype=target_dtype)
        if batch.prompt_attention_mask is None or not batch.prompt_attention_mask:
            raise ValueError("Kandinsky5 requires Qwen prompt attention masks.")
        text_rope_pos = self._text_rope_pos(batch.prompt_attention_mask[0].to(device), device)

        neg_prompt_embeds = None
        neg_pooled = None
        negative_text_rope_pos = None
        if batch.do_classifier_free_guidance and batch.negative_prompt_embeds:
            neg_prompt_embeds = batch.negative_prompt_embeds[0].to(device=device, dtype=target_dtype)
            neg_pooled = batch.negative_prompt_embeds[1].to(device=device, dtype=target_dtype)
            if batch.negative_attention_mask is None or not batch.negative_attention_mask:
                raise ValueError("Kandinsky5 requires Qwen negative attention masks for CFG.")
            negative_text_rope_pos = self._text_rope_pos(batch.negative_attention_mask[0].to(device), device)

        height = int(batch.height)
        width = int(batch.width)
        temporal_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        spatial_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        num_latent_frames = (int(batch.num_frames) - 1) // temporal_ratio + 1
        visual_rope_pos = [
            torch.arange(num_latent_frames, device=device),
            torch.arange(height // spatial_ratio // 2, device=device),
            torch.arange(width // spatial_ratio // 2, device=device),
        ]
        scale_factor = self._scale_factor(height, width)

        with tqdm(total=batch.num_inference_steps, desc="Kandinsky5 Denoising") as progress_bar:
            for i, timestep in enumerate(batch.timesteps):
                if hasattr(self, "interrupt") and self.interrupt:
                    continue

                t_expand = timestep.unsqueeze(0).repeat(latents.shape[0]).to(device=device, dtype=target_dtype)
                autocast_ctx = (torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled)
                                if device.type == "cuda" else contextlib.nullcontext())
                with autocast_ctx:
                    pred_velocity = self.transformer(
                        hidden_states=latents.to(dtype=target_dtype),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        timestep=t_expand,
                        visual_rope_pos=visual_rope_pos,
                        text_rope_pos=text_rope_pos,
                        scale_factor=scale_factor,
                        sparse_params=None,
                        return_dict=True,
                    ).sample

                    if neg_prompt_embeds is not None and neg_pooled is not None:
                        uncond_pred_velocity = self.transformer(
                            hidden_states=latents.to(dtype=target_dtype),
                            encoder_hidden_states=neg_prompt_embeds,
                            pooled_projections=neg_pooled,
                            timestep=t_expand,
                            visual_rope_pos=visual_rope_pos,
                            text_rope_pos=negative_text_rope_pos,
                            scale_factor=scale_factor,
                            sparse_params=None,
                            return_dict=True,
                        ).sample
                        pred_velocity = uncond_pred_velocity + batch.guidance_scale * (pred_velocity -
                                                                                       uncond_pred_velocity)

                latents[:, :, :, :, :num_channels] = self.scheduler.step(
                    pred_velocity,
                    timestep,
                    latents[:, :, :, :, :num_channels],
                    return_dict=False,
                )[0]

                if i == len(batch.timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        batch.latents = latents[:, :, :, :, :num_channels]
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        return result


class Kandinsky5DecodingStage(DecodingStage):

    def __init__(self, vae: ParallelTiledVAE, pipeline=None) -> None:
        super().__init__(vae=vae, pipeline=pipeline)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if not fastvideo_args.model_loaded["vae"]:
            loader = VAELoader()
            self.vae = loader.load(fastvideo_args.model_paths["vae"], fastvideo_args)
            fastvideo_args.model_loaded["vae"] = True

        if batch.latents is None:
            raise ValueError("latents must be available before Kandinsky5 decoding.")

        if fastvideo_args.output_type == "latent":
            frames = batch.latents.permute(0, 4, 1, 2, 3).contiguous()
        else:
            frames = self.decode(batch.latents.permute(0, 4, 1, 2, 3).contiguous(), fastvideo_args)

        batch.output = frames.to(torch.float32)
        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return batch
