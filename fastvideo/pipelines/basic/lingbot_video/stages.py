# SPDX-License-Identifier: Apache-2.0
"""LingBot-Video stages whose contracts differ from shared Wan behavior."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.distributed.tensor import DTensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage

LINGBOT_VIDEO_REFINER_TAIL_STEPS = 2
LINGBOT_VIDEO_IMAGE_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"
LINGBOT_VIDEO_IMAGE_MIN_TOKENS = 4
LINGBOT_VIDEO_IMAGE_MAX_TOKENS = 16384
LINGBOT_VIDEO_IMAGE_MAX_RATIO = 200


def _smart_resize_image(height: int, width: int, factor: int) -> tuple[int, int]:
    """Match Qwen3-VL's bounded patch-grid resize used by LingBot-Video."""
    if max(height, width) / min(height, width) > LINGBOT_VIDEO_IMAGE_MAX_RATIO:
        raise ValueError(f"absolute image aspect ratio must be smaller than {LINGBOT_VIDEO_IMAGE_MAX_RATIO}")
    min_pixels = LINGBOT_VIDEO_IMAGE_MIN_TOKENS * factor**2
    max_pixels = LINGBOT_VIDEO_IMAGE_MAX_TOKENS * factor**2
    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / beta / factor) * factor
        resized_width = math.floor(width / beta / factor) * factor
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


def _preprocess_condition_image(image: Image.Image, height: int, width: int) -> torch.Tensor:
    """Resize and center-crop the clean condition with released uint8 arithmetic."""
    raw = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).contiguous()
    source_height, source_width = raw.shape[-2:]
    scale = max(height / source_height, width / source_width)
    resized_height = max(math.ceil(source_height * scale), height)
    resized_width = max(math.ceil(source_width * scale), width)
    resized = F.interpolate(raw, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    top = int(round((resized_height - height) / 2.0))
    left = int(round((resized_width - width) / 2.0))
    return resized[:, :, top:top + height, left:left + width].float().div(255.0).unsqueeze(2)


def _condition_tensor_to_vlm_image(pixel: torch.Tensor, patch_size: int) -> Image.Image:
    """Convert the base condition tensor to the processor's exact RGB image."""
    frame = pixel[0, :, 0].detach().cpu().clamp(0, 1)
    array = frame.permute(1, 2, 0).mul(255).byte().numpy()
    image = Image.fromarray(array, mode="RGB")
    target_height, target_width = _smart_resize_image(image.height, image.width, patch_size * 2)
    return image.resize((target_width, target_height))


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


class LingBotVideoImageInputValidationStage(LingBotVideoInputValidationStage):
    """Validate TI2V input and prepare its shared base-resolution condition."""

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Load one RGB image and preserve both raw and preprocessed representations."""
        batch = super().forward(batch, fastvideo_args)
        if not isinstance(batch.pil_image, Image.Image):
            raise ValueError("LingBot-Video TI2V requires image_path or one PIL image")
        if not isinstance(batch.height, int) or not isinstance(batch.width, int):
            raise TypeError("LingBot-Video TI2V requires integer height and width")
        batch.preprocessed_image = _preprocess_condition_image(batch.pil_image, batch.height, batch.width)
        return batch


class LingBotVideoImagePromptEncodingStage(PipelineStage):
    """Encode positive and negative TI2V prompts with the same Qwen3-VL image."""

    performance_component_metric = "text_encoder_time_s"

    def __init__(self, text_encoder, processor) -> None:
        self.text_encoder = text_encoder
        self.processor = processor

    def _encode(
        self,
        text: str,
        image: Image.Image,
        fastvideo_args: FastVideoArgs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the released processor call and native compound encoder once."""
        encoder_config = fastvideo_args.pipeline_config.text_encoder_configs[0]
        processed_text = fastvideo_args.pipeline_config.preprocess_text_funcs[0](LINGBOT_VIDEO_IMAGE_TEMPLATE + text)
        target_device = get_local_torch_device()
        first_parameter = next(self.text_encoder.parameters(), None)
        encoder_device = first_parameter.device if first_parameter is not None else target_device
        moved_for_forward = False
        if (first_parameter is not None and not isinstance(first_parameter, DTensor)
                and encoder_device.type != target_device.type):
            self.text_encoder.to(target_device)
            encoder_device = target_device
            moved_for_forward = True
        inputs = self.processor(
            text=[processed_text],
            images=[image],
            videos=None,
            video_metadata=None,
            do_resize=False,
            truncation=True,
            max_length=encoder_config.text_len,
            padding="longest",
            return_tensors="pt",
        ).to(encoder_device)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.text_encoder(**inputs, output_hidden_states=True)
        prompt_embeds, prompt_mask = fastvideo_args.pipeline_config.postprocess_text_funcs[0](
            outputs,
            inputs["attention_mask"],
        )
        if moved_for_forward and fastvideo_args.text_encoder_cpu_offload:
            self.text_encoder.to("cpu")
        return prompt_embeds.to(target_device), prompt_mask.to(target_device)

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Populate both CFG branches with image-conditioned prompt embeddings."""
        if not isinstance(batch.prompt, str) or not isinstance(batch.negative_prompt, str):
            raise TypeError("LingBot-Video TI2V requires string positive and negative prompts")
        if batch.preprocessed_image is None:
            raise ValueError("LingBot-Video TI2V prompt encoding requires the preprocessed image")
        vision_config = fastvideo_args.pipeline_config.text_encoder_configs[0].vision_config
        patch_size = int(vision_config["patch_size"])
        vlm_image = _condition_tensor_to_vlm_image(batch.preprocessed_image, patch_size)
        prompt_embeds, prompt_mask = self._encode(batch.prompt, vlm_image, fastvideo_args)
        batch.prompt_embeds.append(prompt_embeds)
        if batch.prompt_attention_mask is not None:
            batch.prompt_attention_mask.append(prompt_mask)
        if batch.do_classifier_free_guidance:
            negative_embeds, negative_mask = self._encode(batch.negative_prompt, vlm_image, fastvideo_args)
            if batch.negative_prompt_embeds is not None:
                batch.negative_prompt_embeds.append(negative_embeds)
            if batch.negative_attention_mask is not None:
                batch.negative_attention_mask.append(negative_mask)
        return batch


class LingBotVideoImageLatentPreparationStage(PipelineStage):
    """Encode the clean first frame before base diffusion noise is generated."""

    performance_component_metric = "vae_encode_time_s"

    def __init__(self, vae) -> None:
        self.vae = vae

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Sample and normalize the released VAE posterior with the shared RNG."""
        if batch.preprocessed_image is None or not isinstance(batch.generator, torch.Generator):
            raise ValueError("LingBot-Video TI2V requires a condition image and device generator")
        device = get_local_torch_device()
        if isinstance(self.vae, torch.nn.Module):
            self.vae.to(device)
        pixels = batch.preprocessed_image.to(device=device, dtype=torch.float32)
        normalized = (pixels - 0.5) / 0.5
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoded = self.vae.encode(normalized)
        distribution = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded
        if not hasattr(distribution, "sample") or not callable(distribution.sample):
            raise TypeError("LingBot-Video TI2V requires a VAE posterior distribution")
        latents = distribution.sample(batch.generator)
        mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std_inverse = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1, 1)
        batch.image_latent = (latents.float() - mean) * std_inverse
        if getattr(fastvideo_args, "vae_cpu_offload", False):
            self.vae.to("cpu")
        return batch


class LingBotVideoRefinerTextEncodingStage(PipelineStage):
    """Replace image-conditioned embeddings with original text-only refiner input."""

    performance_component_metric = "text_encoder_time_s"

    def __init__(self, text_encoder, processor) -> None:
        self.text_stage = TextEncodingStage([text_encoder], [processor])

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Re-encode only the positive prompt; refiner CFG clones it as zeros."""
        if not isinstance(batch.prompt, str):
            raise TypeError("LingBot-Video refiner requires one string prompt")
        embeds, masks = self.text_stage.encode_text(
            batch.prompt,
            fastvideo_args,
            encoder_index=0,
            return_attention_mask=True,
        )
        batch.prompt_embeds = embeds
        batch.prompt_attention_mask = masks
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
        if batch.image_latent is not None:
            condition_frames = batch.image_latent.shape[2]
            batch.latents[:, :, :condition_frames] = batch.image_latent.float()
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

    @staticmethod
    def _resize_clean_condition(
        image: Image.Image,
        target_height: int,
        target_width: int,
        geometry_height: int,
        geometry_width: int,
    ) -> torch.Tensor:
        """Center-crop to base aspect before the released bicubic refiner resize."""
        image = image.convert("RGB")
        image_width, image_height = image.size
        geometry_aspect = float(geometry_width) / float(geometry_height)
        image_aspect = float(image_width) / float(image_height)
        if image_aspect > geometry_aspect:
            crop_height = image_height
            crop_width = max(1, int(round(crop_height * geometry_aspect)))
            left, top = int(round((image_width - crop_width) / 2.0)), 0
        else:
            crop_width = image_width
            crop_height = max(1, int(round(crop_width / geometry_aspect)))
            left, top = 0, int(round((image_height - crop_height) / 2.0))
        crop = image.crop((left, top, left + crop_width, top + crop_height))
        crop = crop.resize((target_width, target_height), resample=Image.BICUBIC)
        frame = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return frame.contiguous().permute(1, 0, 2, 3).unsqueeze(0)

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
        std_inverse = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1, 1)
        return ((latents.float() - mean) * std_inverse).to(latents)

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
        if batch.preprocessed_image is not None:
            if not isinstance(batch.pil_image, Image.Image):
                raise TypeError("LingBot-Video TI2V refiner requires the original PIL condition")
            clean_pixels = self._resize_clean_condition(
                batch.pil_image,
                batch.height_sr,
                batch.width_sr,
                batch.height,
                batch.width,
            )
            clean_latent = self._encode_video(clean_pixels, generator, device)[:, :, :1].contiguous()
            encoded[:, :, :1] = clean_latent.to(encoded.dtype)
            batch.image_latent = clean_latent.float()
        else:
            batch.image_latent = None
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
        if batch.image_latent is not None:
            condition_frames = batch.image_latent.shape[2]
            latents[:, :, :condition_frames] = batch.image_latent.float()
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
            if batch.image_latent is not None:
                condition_frames = batch.image_latent.shape[2]
                latents[:, :, :condition_frames] = batch.image_latent.float()
            if batch.return_trajectory_latents:
                trajectory.append(latents.detach().cpu())
                trajectory_timesteps.append(timestep.detach().cpu())
        batch.latents = latents
        if trajectory:
            batch.trajectory_latents = torch.stack(trajectory, dim=1)
            batch.trajectory_timesteps = trajectory_timesteps
        return batch
