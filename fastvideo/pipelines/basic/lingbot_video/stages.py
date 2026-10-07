# SPDX-License-Identifier: Apache-2.0
"""LingBot-Video stages whose contracts differ from shared Wan behavior."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.input_validation import InputValidationStage

LINGBOT_VIDEO_REFINER_TAIL_STEPS = 2


def _compute_refiner_sigmas(
    sigma_max: float,
    sigma_min: float,
    num_inference_steps: int,
    shift: float,
    t_thresh: float,
) -> np.ndarray:
    """Build the released truncated schedule plus its two-step low-noise tail."""
    if not 0.0 < t_thresh <= 1.0:
        raise ValueError(f"LingBot-Video refiner t_thresh must be in (0, 1], got {t_thresh}")
    if num_inference_steps < 1:
        raise ValueError("LingBot-Video refiner requires at least one inference step")
    base = np.linspace(sigma_max, sigma_min, num_inference_steps + 1).copy()[:-1]
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    sigmas = shifted[shifted <= t_thresh + 1e-6]
    if sigmas.size == 0 or abs(float(sigmas[0]) - t_thresh) > 1e-6:
        sigmas = np.concatenate(([t_thresh], sigmas))
    tail = np.linspace(
        float(sigmas[-1]),
        min(sigma_min, float(sigmas[-1])),
        LINGBOT_VIDEO_REFINER_TAIL_STEPS + 2,
    )[1:-1]
    sigmas = np.concatenate((sigmas, tail))
    if sigmas.size > 1 and not np.all(np.diff(sigmas) < 0.0):
        raise ValueError(f"LingBot-Video refiner sigmas must descend strictly, got {sigmas.tolist()}")
    return sigmas.astype(np.float32)


class LingBotVideoInputValidationStage(InputValidationStage):
    """Validate released shape constraints and construct the official CUDA RNG."""

    def __init__(self, refiner_enabled: bool = False) -> None:
        self.refiner_enabled = refiner_enabled

    def _generate_seeds(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> None:
        """Use one device-local generator, matching the official production runner."""
        del fastvideo_args
        if batch.seed is None:
            raise ValueError("LingBot-Video requires a seed")
        if batch.num_videos_per_prompt != 1:
            raise ValueError("LingBot-Video currently supports one video per prompt")
        batch.seeds = [batch.seed]
        batch.generator = torch.Generator(device=get_local_torch_device()).manual_seed(batch.seed)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Run shared validation, then enforce LingBot temporal and spatial geometry."""
        batch = super().forward(batch, fastvideo_args)
        if not isinstance(batch.num_frames, int):
            raise TypeError("LingBot-Video num_frames must be an integer")
        if batch.num_frames != 1 and (batch.num_frames - 1) % 4 != 0:
            raise ValueError(f"num_frames must be 1 or 4n+1, got {batch.num_frames}")
        if not isinstance(batch.height, int) or not isinstance(batch.width, int):
            raise TypeError("LingBot-Video height and width must be integers")
        if batch.height % 16 != 0 or batch.width % 16 != 0:
            raise ValueError(f"height and width must be divisible by 16, got {batch.height}x{batch.width}")
        if isinstance(batch.prompt, list) and len(batch.prompt) != 1:
            raise ValueError("LingBot-Video currently supports prompt batch size one")
        if self.refiner_enabled and fastvideo_args.output_type == "latent":
            raise ValueError("LingBot-Video refinement requires decoded pixel output")
        return batch


class LingBotVideoLatentPreparationStage(PipelineStage):
    """Prepare fp32 latents in the released 4x temporal and 8x spatial geometry."""

    def __init__(self, transformer) -> None:
        self.transformer = transformer

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Generate or validate one normalized fp32 latent video."""
        del fastvideo_args
        if not all(isinstance(value, int) for value in (batch.num_frames, batch.height, batch.width)):
            raise TypeError("latent geometry must contain integer frames, height, and width")
        shape = (
            1,
            self.transformer.num_channels_latents,
            (batch.num_frames - 1) // 4 + 1,
            batch.height // 8,
            batch.width // 8,
        )
        device = get_local_torch_device()
        if batch.latents is None:
            batch.latents = torch.randn(
                shape,
                generator=batch.generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            if tuple(batch.latents.shape) != shape:
                raise ValueError(f"supplied latent shape {tuple(batch.latents.shape)} does not match {shape}")
            batch.latents = batch.latents.to(device=device, dtype=torch.float32)
        batch.raw_latent_shape = shape
        return batch


class LingBotVideoRefinerPreparationStage(PipelineStage):
    """Resize and encode the base video, then initialize the released refiner state."""

    performance_component_metric = "vae_encode_time_s"

    def __init__(self, vae, scheduler) -> None:
        self.vae = vae
        self.scheduler = scheduler

    @staticmethod
    def _resize_video(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Bicubic-resize every decoded frame using the released tensor layout."""
        batch, channels, frames, source_height, source_width = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, source_height, source_width)
        resized = F.interpolate(flat, size=(height, width), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        return resized.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()

    def _encode_video(
        self,
        video: torch.Tensor,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode `[0,1]` pixels and convert Wan VAE latents to normalized DiT space."""
        video = video.to(device=device, dtype=torch.float32).mul(2.0).sub(1.0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoded = self.vae.encode(video)
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.sample(generator)
        elif hasattr(encoded, "sample") and callable(encoded.sample):
            latents = encoded.sample(generator)
        elif isinstance(encoded, tuple | list):
            latents = encoded[0]
        else:
            latents = encoded
        mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        return ((latents.float() - mean) / std).to(latents)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Prepare high-resolution refiner latents and its exact truncated sigma schedule."""
        if batch.output is None or batch.output.ndim != 5:
            raise ValueError("LingBot-Video refinement requires a decoded base video")
        if not isinstance(batch.height_sr, int) or not isinstance(batch.width_sr, int):
            raise TypeError("LingBot-Video refinement requires integer height_sr and width_sr")
        if batch.height_sr % 16 != 0 or batch.width_sr % 16 != 0:
            raise ValueError("LingBot-Video refiner height_sr and width_sr must be divisible by 16")
        if batch.seed is None:
            raise ValueError("LingBot-Video refinement requires a seed")
        device = get_local_torch_device()
        if isinstance(self.vae, torch.nn.Module):
            self.vae.to(device)
        generator = torch.Generator(device=device).manual_seed(batch.seed)
        resized = self._resize_video(batch.output, batch.height_sr, batch.width_sr)
        encoded = self._encode_video(resized, generator, device)
        noise = torch.randn(encoded.shape, generator=generator, device=device, dtype=encoded.dtype)
        batch.latents = ((1.0 - batch.t_thresh) * encoded + batch.t_thresh * noise).float()
        batch.generator = generator
        batch.height = batch.height_sr
        batch.width = batch.width_sr
        batch.raw_latent_shape = tuple(batch.latents.shape)
        batch.extra["lingbot_video_base_shape"] = tuple(batch.output.shape)
        batch.output = None

        shift = fastvideo_args.pipeline_config.flow_shift
        if shift is None:
            raise ValueError("LingBot-Video refinement requires a flow shift")
        sigmas = _compute_refiner_sigmas(
            float(self.scheduler.sigma_max),
            float(self.scheduler.sigma_min),
            batch.num_inference_steps_sr,
            float(shift),
            float(batch.t_thresh),
        )
        self.scheduler.set_timesteps(len(sigmas), device=device, sigmas=sigmas, shift=1.0)
        batch.timesteps = self.scheduler.timesteps
        if getattr(fastvideo_args, "vae_cpu_offload", False):
            self.vae.to("cpu")
        return batch


class LingBotVideoDenoisingStage(PipelineStage):
    """Run the released batched-CFG bf16 DiT loop with fp32 scheduler state."""

    performance_component_metric = "dit_time_s"

    def __init__(self, transformer, scheduler, refiner: bool = False) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.refiner = refiner

    @staticmethod
    def _pad_condition(
        embeds: torch.Tensor,
        mask: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad one condition stream to a shared batched-CFG length."""
        pad_length = length - embeds.shape[1]
        if pad_length < 0 or embeds.shape[:2] != mask.shape:
            raise ValueError("invalid LingBot-Video prompt embedding/mask shapes")
        if pad_length == 0:
            return embeds, mask
        embed_padding = embeds.new_zeros(embeds.shape[0], pad_length, embeds.shape[2])
        mask_padding = mask.new_zeros(mask.shape[0], pad_length)
        return (
            torch.cat((embeds, embed_padding), dim=1),
            torch.cat((mask, mask_padding), dim=1),
        )

    @staticmethod
    def _transformer_timestep(timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Reproduce the official divide-cast-multiply timestep rounding."""
        sigma = timestep.float() / 1000.0
        if dtype in (torch.bfloat16, torch.float16):
            sigma = sigma.to(dtype)
        return (sigma * 1000.0).float()

    def _prepare_conditions(
        self,
        batch: ForwardBatch,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack conditional then unconditional text streams for one batched CFG call."""
        prompt = batch.prompt_embeds[0].to(device=device, dtype=dtype)
        if batch.prompt_attention_mask is None or not batch.prompt_attention_mask:
            raise ValueError("LingBot-Video requires a prompt attention mask")
        prompt_mask = batch.prompt_attention_mask[0].to(device=device)
        if not self._uses_cfg(batch) or not batch.batch_cfg:
            return prompt, prompt_mask
        negative, negative_mask = self._negative_condition(batch, prompt, prompt_mask, dtype, device)
        target_length = max(prompt.shape[1], negative.shape[1])
        prompt, prompt_mask = self._pad_condition(prompt, prompt_mask, target_length)
        negative, negative_mask = self._pad_condition(negative, negative_mask, target_length)
        return (
            torch.cat((prompt, negative), dim=0),
            torch.cat((prompt_mask, negative_mask), dim=0),
        )

    def _negative_condition(
        self,
        batch: ForwardBatch,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the refiner's zero-cloned null condition or the encoded negative prompt."""
        if self.refiner:
            return torch.zeros_like(prompt), prompt_mask.clone()
        if batch.negative_prompt_embeds is None or not batch.negative_prompt_embeds:
            raise ValueError("LingBot-Video CFG requires negative prompt embeddings")
        if batch.negative_attention_mask is None or not batch.negative_attention_mask:
            raise ValueError("LingBot-Video CFG requires a negative prompt mask")
        return (
            batch.negative_prompt_embeds[0].to(device=device, dtype=dtype),
            batch.negative_attention_mask[0].to(device=device),
        )

    def _uses_cfg(self, batch: ForwardBatch) -> bool:
        """Enable guidance independently for the base or refiner scale."""
        scale = batch.guidance_scale_2 if self.refiner else batch.guidance_scale
        return scale is not None and scale > 1.0

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Denoise latents while keeping scheduler samples and predictions in fp32."""
        if batch.latents is None or batch.timesteps is None:
            raise ValueError("LingBot-Video denoising requires latents and timesteps")
        device = get_local_torch_device()
        transformer_dtype = next(self.transformer.parameters()).dtype
        condition, condition_mask = self._prepare_conditions(batch, transformer_dtype, device)
        latents = batch.latents.to(device=device, dtype=torch.float32)
        do_cfg = self._uses_cfg(batch)
        negative = negative_mask = None
        if do_cfg and not batch.batch_cfg:
            negative, negative_mask = self._negative_condition(batch, condition, condition_mask, transformer_dtype,
                                                               device)
        trajectory: list[torch.Tensor] = []
        trajectory_timesteps: list[torch.Tensor] = []
        for timestep in batch.timesteps:
            timestep_batch = self._transformer_timestep(timestep, transformer_dtype).expand(1).to(device)
            latent_input = latents
            if do_cfg and batch.batch_cfg:
                latent_input = torch.cat((latents, latents), dim=0)
                timestep_batch = torch.cat((timestep_batch, timestep_batch), dim=0)
            autocast_enabled = device.type == "cuda" and transformer_dtype != torch.float32
            with torch.autocast(device_type=device.type, dtype=transformer_dtype, enabled=autocast_enabled):
                prediction = self.transformer(
                    latent_input,
                    timestep_batch,
                    condition,
                    encoder_attention_mask=condition_mask,
                    return_dict=False,
                )[0].float()
            if do_cfg:
                if batch.batch_cfg:
                    conditional, unconditional = prediction.chunk(2, dim=0)
                else:
                    with torch.autocast(
                            device_type=device.type,
                            dtype=transformer_dtype,
                            enabled=autocast_enabled,
                    ):
                        unconditional = self.transformer(
                            latents,
                            timestep_batch,
                            negative,
                            encoder_attention_mask=negative_mask,
                            return_dict=False,
                        )[0].float()
                    conditional = prediction
                guidance_scale = batch.guidance_scale_2 if self.refiner else batch.guidance_scale
                if guidance_scale is None:
                    raise ValueError("LingBot-Video CFG requires a guidance scale")
                prediction = unconditional + guidance_scale * (conditional - unconditional)
            latents = self.scheduler.step(
                prediction,
                timestep,
                latents,
                return_dict=False,
                generator=batch.generator,
            )[0].float()
            if batch.return_trajectory_latents:
                trajectory.append(latents.detach().cpu())
                trajectory_timesteps.append(timestep.detach().cpu())
        batch.latents = latents
        if trajectory:
            batch.trajectory_latents = torch.stack(trajectory, dim=1)
            batch.trajectory_timesteps = trajectory_timesteps
        return batch
