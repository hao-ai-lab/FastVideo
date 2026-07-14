# SPDX-License-Identifier: Apache-2.0
"""Pipeline stages for the native Z-Image text-to-image path."""

from __future__ import annotations

import inspect

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.hooks.activation_trace import trace_step
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.utils import PRECISION_TO_TYPE


class ZImageInputValidationStage(InputValidationStage):
    """Validate the image geometry and reproduce the official device RNG."""

    def _generate_seeds(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> None:
        del fastvideo_args
        assert batch.seed is not None
        batch.seeds = [batch.seed]
        device = get_local_torch_device()
        batch.generator = torch.Generator(device=device).manual_seed(batch.seed)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.do_classifier_free_guidance and batch.negative_prompt is None and not batch.negative_prompt_embeds:
            batch.negative_prompt = ""
        batch = super().forward(batch, fastvideo_args)
        if batch.num_frames != 1:
            raise ValueError(f"Z-Image is text-to-image and requires num_frames=1, got {batch.num_frames}")
        if batch.height is None or batch.width is None:
            raise ValueError("Z-Image requires height and width")
        if batch.height % 16 or batch.width % 16:
            raise ValueError("Z-Image height and width must be divisible by 16; "
                             f"got {batch.height}x{batch.width}")
        return batch


class ZImageConditioningStage(PipelineStage):
    """Trim padded Qwen states and materialize variable-length CFG streams."""

    @staticmethod
    def _trim_embeddings(
        embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if attention_mask is None:
            return list(embeds.unbind(0))
        return [
            sample[mask.to(device=sample.device, dtype=torch.bool)]
            for sample, mask in zip(embeds, attention_mask, strict=True)
        ]

    @staticmethod
    def _repeat(items: list[torch.Tensor], count: int) -> list[torch.Tensor]:
        return [item for item in items for _ in range(count)]

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del fastvideo_args
        if len(batch.prompt_embeds) != 1:
            raise ValueError(f"Z-Image expects one text encoder, got {len(batch.prompt_embeds)}")

        prompt_mask = batch.prompt_attention_mask[0] if batch.prompt_attention_mask else None
        prompt_embeds = self._trim_embeddings(batch.prompt_embeds[0], prompt_mask)
        batch.extra["zimage_prompt_embeds"] = self._repeat(prompt_embeds, batch.num_videos_per_prompt)

        if batch.do_classifier_free_guidance:
            if not batch.negative_prompt_embeds:
                raise ValueError("Z-Image CFG requires negative prompt embeddings")
            negative_mask = batch.negative_attention_mask[0] if batch.negative_attention_mask else None
            negative_embeds = self._trim_embeddings(batch.negative_prompt_embeds[0], negative_mask)
            batch.extra["zimage_negative_prompt_embeds"] = self._repeat(
                negative_embeds,
                batch.num_videos_per_prompt,
            )
        else:
            batch.extra["zimage_negative_prompt_embeds"] = []
        return batch


class ZImageLatentPreparationStage(PipelineStage):
    """Create the official fp32 image latents on the transformer device."""

    def __init__(self, transformer) -> None:
        self.transformer = transformer

    @staticmethod
    def _randn(
        shape: tuple[int, ...],
        generators: torch.Generator | list[torch.Generator] | None,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(generators, list):
            if len(generators) != shape[0]:
                raise ValueError(f"generator list length {len(generators)} does not match batch size {shape[0]}")
            sample_shape = (1, *shape[1:])
            return torch.cat([
                torch.randn(sample_shape, generator=generator, device=device, dtype=torch.float32)
                for generator in generators
            ])
        return torch.randn(shape, generator=generators, device=device, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.height is None or batch.width is None:
            raise ValueError("Z-Image requires height and width before latent preparation")

        prompt_embeds = batch.extra.get("zimage_prompt_embeds")
        if not isinstance(prompt_embeds, list) or not prompt_embeds:
            raise ValueError("Z-Image conditioning must run before latent preparation")

        channels = int(getattr(self.transformer, "in_channels", 16))
        spatial_ratio = int(fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio)
        shape = (
            len(prompt_embeds),
            channels,
            1,
            batch.height // spatial_ratio,
            batch.width // spatial_ratio,
        )
        device = get_local_torch_device()
        if batch.latents is None:
            latents = self._randn(shape, batch.generator, device)
        else:
            latents = batch.latents
            if latents.ndim == 4:
                latents = latents.unsqueeze(2)
            if tuple(latents.shape) != shape:
                raise ValueError(f"Expected Z-Image latents with shape {shape}, got {tuple(latents.shape)}")
            latents = latents.to(device=device, dtype=torch.float32)

        batch.latents = latents
        batch.raw_latent_shape = shape
        return batch


class ZImageTimestepPreparationStage(PipelineStage):
    """Apply the native scheduler's zero endpoint and discrete schedule."""

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    @staticmethod
    def _calculate_shift(
        image_seq_len: int,
        base_seq_len: int,
        max_seq_len: int,
        base_shift: float,
        max_shift: float,
    ) -> float:
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        return image_seq_len * slope + base_shift - slope * base_seq_len

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.latents is None:
            raise ValueError("Z-Image latents must be prepared before timesteps")
        if batch.timesteps is not None and batch.sigmas is not None:
            raise ValueError("Only one of timesteps or sigmas may be supplied")

        scheduler = self.scheduler
        sigma_min = float(fastvideo_args.pipeline_config.scheduler_sigma_min)
        use_reference_timesteps = bool(fastvideo_args.pipeline_config.scheduler_use_reference_discrete_timesteps)
        scheduler.sigma_min = sigma_min
        scheduler.register_to_config(
            sigma_min=sigma_min,
            use_reference_discrete_timesteps=use_reference_timesteps,
        )
        config = scheduler.config
        image_seq_len = (batch.latents.shape[-2] // 2) * (batch.latents.shape[-1] // 2)
        mu = self._calculate_shift(
            image_seq_len,
            int(config.get("base_image_seq_len", 256)),
            int(config.get("max_image_seq_len", 4096)),
            float(config.get("base_shift", 0.5)),
            float(config.get("max_shift", 1.15)),
        )
        device = get_local_torch_device()

        if batch.timesteps is not None:
            if "timesteps" not in inspect.signature(scheduler.set_timesteps).parameters:
                raise ValueError(f"{type(scheduler).__name__} does not accept custom timesteps")
            scheduler.set_timesteps(timesteps=batch.timesteps, device=device, mu=mu)
        elif batch.sigmas is not None:
            if "sigmas" not in inspect.signature(scheduler.set_timesteps).parameters:
                raise ValueError(f"{type(scheduler).__name__} does not accept custom sigmas")
            scheduler.set_timesteps(sigmas=batch.sigmas, device=device, mu=mu)
        else:
            scheduler.set_timesteps(batch.num_inference_steps, device=device, mu=mu)

        batch.timesteps = scheduler.timesteps
        batch.num_inference_steps = len(batch.timesteps)
        return batch


class ZImageDenoisingStage(PipelineStage):
    """Run the native Z-Image flow-matching loop."""

    performance_component_metric = "transformer_time_s"

    def __init__(self, transformer, scheduler) -> None:
        self.transformer = transformer
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.latents is None or batch.timesteps is None:
            raise ValueError("Z-Image denoising requires latents and timesteps")

        latents = batch.latents.float()
        positive = batch.extra.get("zimage_prompt_embeds")
        negative = batch.extra.get("zimage_negative_prompt_embeds", [])
        if not isinstance(positive, list) or not positive:
            raise ValueError("Z-Image denoising requires prompt embeddings")

        device = get_local_torch_device()
        target_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        batch_size = latents.shape[0]

        for index, timestep_value in enumerate(batch.timesteps):
            if timestep_value.item() == 0 and index == len(batch.timesteps) - 1:
                continue

            timestep = timestep_value.expand(batch_size).to(device=device, dtype=torch.float32)
            timestep = (1000.0 - timestep) / 1000.0
            current_guidance_scale = float(batch.guidance_scale)
            if (batch.do_classifier_free_guidance and batch.cfg_truncation is not None and batch.cfg_truncation <= 1.0
                    and timestep[0].item() > batch.cfg_truncation):
                current_guidance_scale = 0.0
            apply_cfg = batch.do_classifier_free_guidance and current_guidance_scale > 0.0

            if apply_cfg:
                if not isinstance(negative, list) or len(negative) != batch_size:
                    raise ValueError("Z-Image CFG requires one negative embedding per image")
                model_latents = latents.to(target_dtype).repeat(2, 1, 1, 1, 1)
                model_embeddings = positive + negative
                model_timestep = timestep.repeat(2)
            else:
                model_latents = latents.to(target_dtype)
                model_embeddings = positive
                model_timestep = timestep

            with (
                    torch.autocast(
                        device_type=device.type,
                        enabled=False,
                    ),
                    trace_step(index),
                    set_forward_context(
                        current_timestep=index,
                        attn_metadata=None,
                        forward_batch=batch,
                    ),
            ):
                model_outputs = self.transformer(
                    hidden_states=model_latents,
                    encoder_hidden_states=model_embeddings,
                    timestep=model_timestep,
                )[0]

            if apply_cfg:
                positive_outputs = model_outputs[:batch_size]
                negative_outputs = model_outputs[batch_size:]
                guided_outputs = []
                for positive_output, negative_output in zip(
                        positive_outputs,
                        negative_outputs,
                        strict=True,
                ):
                    positive_fp32 = positive_output.float()
                    prediction = positive_fp32 + current_guidance_scale * (positive_fp32 - negative_output.float())
                    if batch.cfg_normalization:
                        positive_norm = torch.linalg.vector_norm(positive_fp32)
                        prediction_norm = torch.linalg.vector_norm(prediction)
                        if prediction_norm > positive_norm:
                            prediction = prediction * (positive_norm / prediction_norm)
                    guided_outputs.append(prediction)
                noise_pred = torch.stack(guided_outputs)
            else:
                noise_pred = torch.stack([output.float() for output in model_outputs])

            noise_pred = -noise_pred
            latents = self.scheduler.step(
                noise_pred,
                timestep_value,
                latents,
                return_dict=False,
            )[0].float()

        batch.latents = latents
        return batch


class ZImageDecodingStage(PipelineStage):
    """Apply the official latent transform and decode one image frame."""

    performance_component_metric = "vae_decode_time_s"

    def __init__(self, vae) -> None:
        self.vae = vae

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.latents is None:
            raise ValueError("Z-Image decoding requires latents")
        if fastvideo_args.output_type == "latent":
            # FastVideo standardizes image and video latents as [B,C,T,H,W].
            # Tongyi's image-only API returns the equivalent tensor with T squeezed.
            batch.output = batch.latents
            return batch

        latents = batch.latents
        if latents.ndim != 5 or latents.shape[2] != 1:
            raise ValueError(f"Expected Z-Image latents [B,C,1,H,W], got {tuple(latents.shape)}")

        device = get_local_torch_device()
        self.vae = self.vae.to(device)
        vae_dtype = getattr(self.vae, "dtype", None)
        if vae_dtype is None:
            vae_dtype = next(self.vae.parameters()).dtype
        config = self.vae.config
        scaling_factor = float(config.scaling_factor)
        shift_factor = float(config.shift_factor or 0.0)
        latents_2d = latents.squeeze(2).to(device=device, dtype=vae_dtype)
        latents_2d = latents_2d / scaling_factor + shift_factor
        decoded = self.vae.decode(latents_2d, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        batch.output = decoded.unsqueeze(2).float()

        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return batch
