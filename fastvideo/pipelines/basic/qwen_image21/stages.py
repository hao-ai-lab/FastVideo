# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Adapted from Diffusers' Apache-2.0 QwenImage21Pipeline.
"""Separate preparation, encoding, scheduling, denoising and decoding stages."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.distributed.tensor import DTensor
from tqdm.auto import tqdm

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.hooks.hooks import ModuleHookManager
from fastvideo.logger import init_logger
from fastvideo.models import pinned_offload
from fastvideo.models.dits.qwen_image21 import QwenImage21KVCache, QwenImage21KVCacheAllocationError
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import V, VerificationResult

logger = init_logger(__name__)

STATE_KEY = "qwen_image21"
SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


@dataclass
class QwenImage21State:
    references: list[Image.Image] = field(default_factory=list)
    img_shapes: list[list[tuple[int, int, int]]] = field(default_factory=list)
    image_pad_mask: torch.Tensor | None = None
    negative_image_pad_mask: torch.Tensor | None = None


def load_reference(value: Any) -> Image.Image:
    """Detach a local image from its file and preserve alpha for the VAE."""
    if isinstance(value, str | Path):
        with Image.open(value) as image:
            return ImageOps.exif_transpose(image).convert("RGBA")
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGBA")
    if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim in (2, 3):
        return Image.fromarray(value).convert("RGBA")
    raise TypeError("Each reference must be a local image path, PIL image, or uint8 image array")


def collect_references(references: list[Any] | None, image_path: Any, pil_image: Any) -> list[Image.Image]:
    if references is not None and not isinstance(references, list | tuple):
        raise TypeError("references must be an ordered list of images")
    shortcuts = [value for value in (image_path, pil_image) if value is not None]
    if len(shortcuts) > 1 or (references and shortcuts):
        raise ValueError("Supply references, image_path, or pil_image as a single source of reference images")
    values = list(references or shortcuts)
    if len(values) > 10:
        raise ValueError("Qwen-Image-2.1 supports at most 10 ordered reference images")
    return [load_reference(value) for value in values]


def reference_dimensions(width: int, height: int, resolution: int) -> tuple[int, int]:
    if min(width, height, resolution) <= 0:
        raise ValueError("Image dimensions and reference_resolution must be positive")
    ratio = width / height
    width_float = math.sqrt(resolution**2 * ratio)
    height_float = width_float / ratio
    new_width = round(width_float / 32) * 32
    new_height = round(height_float / 32) * 32
    if min(new_width, new_height) < 32:
        raise ValueError("Reference aspect ratio is too extreme for a 32-pixel latent grid")
    return new_width, new_height


def resize_reference(image: Image.Image, resolution: int) -> Image.Image:
    return image.resize(reference_dimensions(image.width, image.height, resolution), resample=Image.Resampling.LANCZOS)


def vision_reference(image: Image.Image) -> Image.Image:
    white = Image.new("RGB", image.size, (255, 255, 255))
    white.paste(image, mask=image.getchannel("A"))
    return white


def reference_pixels(image: Image.Image) -> torch.Tensor:
    # [B, RGBA, one frame, H, W], with all four channels normalized to [-1, 1].
    # Build the batch axis before permuting, as upstream's numpy_to_pt does.
    # Even a singleton batch stride affects BF16 convolution kernel rounding.
    pixels = torch.from_numpy(np.array(image, dtype=np.float32)[None]).permute(0, 3, 1, 2)
    return (pixels / 127.5 - 1).unsqueeze(2)


def prompt_template(prompt: str, num_references: int) -> str:
    if not 0 <= num_references <= 10:
        raise ValueError("Qwen-Image-2.1 supports 0–10 references")
    prefix = " ".join(f"<image{index + 1}>{IMAGE_PLACEHOLDER}" for index in range(num_references))
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n")


def extract_prompt(hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, drop_tokens: int,
                   image_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, image_rows = [], []
    for embeddings, ids, mask in zip(hidden, input_ids, attention_mask, strict=True):
        valid = mask.bool()
        rows.append(embeddings[valid][drop_tokens:])
        image_rows.append((ids[valid] == image_token_id)[drop_tokens:])
    if not rows or min(row.shape[0] for row in rows) == 0:
        raise ValueError("The processor produced an empty prompt after removing system tokens")
    length = max(row.shape[0] for row in rows)
    embeds = torch.stack([torch.cat((row, row.new_zeros(length - row.shape[0], row.shape[1]))) for row in rows])
    masks = torch.stack([torch.arange(length, device=row.device) < row.shape[0] for row in rows])
    image_masks = torch.stack([torch.cat((row, row.new_zeros(length - row.shape[0]))) for row in image_rows])
    return embeds, masks, image_masks


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack [B, C, 1, H, W] into [B, H*W, C], with width varying fastest."""
    if latents.ndim != 5 or latents.shape[2] != 1:
        raise ValueError("Qwen-Image-2.1 latents must have shape [B, C, 1, H, W]")
    return latents.flatten(2).transpose(1, 2)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Restore [B, C, 1, H/16, W/16] from tokens containing all C channels."""
    if latents.ndim != 3 or latents.shape[1] != (height // 16) * (width // 16):
        raise ValueError("Packed latent length does not match the requested image dimensions")
    return latents.transpose(1, 2).reshape(latents.shape[0], latents.shape[2], 1, height // 16, width // 16)


def normalize_latents(latents: torch.Tensor, means: Any, stds: Any, *, inverse: bool = False) -> torch.Tensor:
    mean = torch.as_tensor(means, device=latents.device, dtype=latents.dtype).reshape(1, -1, 1, 1, 1)
    std = torch.as_tensor(stds, device=latents.device, dtype=latents.dtype).reshape(1, -1, 1, 1, 1)
    if mean.shape[1] != latents.shape[1] or std.shape[1] != latents.shape[1] or bool((std <= 0).any()):
        raise ValueError("VAE latent statistics must match its channels and have positive standard deviations")
    return latents * std + mean if inverse else (latents - mean) / std


def schedule_shift(target_tokens: int, config: Any) -> float:
    base_len, max_len = config.get("base_image_seq_len", 256), config.get("max_image_seq_len", 8192)
    base_shift, max_shift = config.get("base_shift", 0.5), config.get("max_shift", 0.9)
    slope = (max_shift - base_shift) / (max_len - base_len)
    return base_shift + slope * (target_tokens - base_len)


def validate_request(batch: Any, args: Any) -> None:
    if args.tp_size != 1 or args.sp_size != 1 or args.num_gpus != 1:
        raise ValueError("The Qwen-Image-2.1 port currently supports one GPU with tp_size=sp_size=1")
    if batch.num_frames != 1 or batch.num_videos_per_prompt != 1:
        raise ValueError("Qwen-Image-2.1 requires num_frames=1 and num_videos_per_prompt=1")
    if not isinstance(batch.prompt, str) and not (isinstance(batch.prompt, list) and len(batch.prompt) == 1
                                                  and isinstance(batch.prompt[0], str)):
        raise ValueError("Qwen-Image-2.1 requires one text prompt per request")
    for name, value in (("height", batch.height), ("width", batch.width), ("reference_resolution",
                                                                           args.pipeline_config.reference_resolution)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value % 32:
            raise ValueError(f"{name} must be a positive multiple of 32")
    if not isinstance(batch.num_inference_steps, int) or batch.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    if not math.isfinite(batch.true_cfg_scale) or batch.true_cfg_scale < 1:
        raise ValueError("true_cfg_scale must be finite and at least 1")
    if batch.guidance_scale != 1:
        raise ValueError("Qwen-Image-2.1 has no embedded guidance; use true_cfg_scale with negative_prompt")
    if batch.video_path is not None or getattr(batch, "conditioning_mask", None) is not None:
        raise ValueError("Pass annotations or mask images as references; this pipeline takes single-frame images")
    if batch.prompt_embeds or batch.negative_prompt_embeds:
        raise ValueError("Externally supplied prompt embeddings are not supported by this Qwen-Image-2.1 port")
    if batch.negative_prompt is not None and not isinstance(batch.negative_prompt, str):
        raise ValueError("negative_prompt must be a string or None")


def _state(batch: ForwardBatch) -> QwenImage21State:
    state = batch.extra.get(STATE_KEY)
    if not isinstance(state, QwenImage21State):
        raise ValueError("Qwen-Image-2.1 input preparation must precede this stage")
    return state


class QwenImage21InputStage(PipelineStage):

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("height", batch.height, V.positive_int_divisible(32))
        result.add_check("width", batch.width, V.positive_int_divisible(32))
        result.add_check("num_frames", batch.num_frames, lambda value: value == 1)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check(STATE_KEY, batch.extra.get(STATE_KEY),
                                              lambda value: isinstance(value, QwenImage21State))

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        validate_request(batch, fastvideo_args)
        references = collect_references(batch.references, batch.image_path, batch.pil_image)
        references = [
            resize_reference(image, fastvideo_args.pipeline_config.reference_resolution) for image in references
        ]
        shapes = [(1, image.height // 16, image.width // 16) for image in references]
        shapes.append((1, batch.height // 16, batch.width // 16))
        batch.extra[STATE_KEY] = QwenImage21State(references=references, img_shapes=[shapes])
        batch.batch_size = 1
        batch.do_classifier_free_guidance = batch.true_cfg_scale > 1 and batch.negative_prompt is not None
        if batch.true_cfg_scale > 1 and not batch.do_classifier_free_guidance:
            logger.warning("true_cfg_scale > 1 requires an explicit negative_prompt; guidance is disabled")
        return batch


class QwenImage21TextEncodingStage(PipelineStage):
    performance_component_metric = "text_encoder_time_s"

    def __init__(self, text_encoder, processor) -> None:
        self.text_encoder = text_encoder
        self.processor = processor

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check(STATE_KEY, batch.extra.get(STATE_KEY),
                                              lambda value: isinstance(value, QwenImage21State))

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("prompt_embeds", batch.prompt_embeds,
                                              lambda value: len(value) == 1 and value[0].ndim == 3)

    def _encode(self, prompt: str, state: QwenImage21State, device: torch.device):
        kwargs = dict(text=[prompt_template(prompt, len(state.references))],
                      padding=True,
                      padding_side="left",
                      return_tensors="pt")
        if state.references:
            kwargs["images"] = [vision_reference(image) for image in state.references]
        model_inputs = self.processor(**kwargs)
        inputs = {key: value.to(device) for key, value in model_inputs.items() if isinstance(value, torch.Tensor)}
        hidden = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )
        system = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        system_ids = self.processor.apply_chat_template(system, tokenize=True, return_dict=False)
        drop_tokens = len(system_ids[0])
        image_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]
        embeds, mask, image_mask = extract_prompt(hidden, inputs["input_ids"], inputs["attention_mask"], drop_tokens,
                                                  image_token_id)
        return embeds, None if bool(mask.all()) else mask, image_mask

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state = _state(batch)
        device = get_local_torch_device()
        first_param = next(self.text_encoder.parameters(), None)
        # The loader's FSDP2 CPU offload already streams individual text layers.
        move = (fastvideo_args.text_encoder_cpu_offload and first_param is not None
                and not isinstance(first_param, DTensor))
        try:
            if move:
                pinned_offload.load(self.text_encoder, device, pin=fastvideo_args.pin_cpu_memory)
            prompt = batch.prompt[0] if isinstance(batch.prompt, list) else batch.prompt
            if prompt is None:
                raise ValueError("A text prompt is required")
            embeds, mask, state.image_pad_mask = self._encode(prompt, state, device)
            batch.prompt_embeds = [embeds]
            batch.prompt_attention_mask = [mask]
            if batch.do_classifier_free_guidance:
                if batch.negative_prompt is None:
                    raise ValueError("True CFG requires an explicit negative prompt")
                embeds, mask, state.negative_image_pad_mask = self._encode(batch.negative_prompt, state, device)
                batch.negative_prompt_embeds = [embeds]
                batch.negative_attention_mask = [mask]
        finally:
            if move:
                pinned_offload.unload(self.text_encoder)
        return batch


class QwenImage21LatentStage(PipelineStage):
    performance_component_metric = "vae_encode_time_s"

    def __init__(self, vae) -> None:
        self.vae = vae

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("prompt_embeds", batch.prompt_embeds, lambda value: len(value) == 1)

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("latents", batch.latents, V.with_dims(3))

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state = _state(batch)
        device, dtype = get_local_torch_device(), batch.prompt_embeds[0].dtype
        channels = fastvideo_args.pipeline_config.dit_config.arch_config.in_channels
        shape = (1, channels, 1, batch.height // 16, batch.width // 16)
        if batch.latents is None:
            generator = batch.generator
            if isinstance(generator, list):
                if len(generator) != 1:
                    raise ValueError("Qwen-Image-2.1 requires one generator")
                generator = generator[0]
            if generator is None:
                generator = torch.Generator(device="cpu").manual_seed(batch.seed or 0)
            noise_device = generator.device
            batch.latents = pack_latents(
                torch.randn(shape, generator=generator, device=noise_device, dtype=dtype).to(device))
        else:
            if batch.latents.ndim == 5:
                batch.latents = pack_latents(batch.latents)
            if tuple(batch.latents.shape) != (1, shape[3] * shape[4], channels):
                raise ValueError("Supplied latents must match [1, target tokens, latent channels]")
            batch.latents = batch.latents.to(device=device, dtype=dtype)
        batch.raw_latent_shape = shape
        if state.references:
            try:
                pinned_offload.load(self.vae, device, pin=fastvideo_args.pin_cpu_memory)
                if fastvideo_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
                else:
                    self.vae.disable_tiling()
                vae_dtype = next(self.vae.parameters()).dtype
                rows = []
                for image in state.references:
                    pixels = reference_pixels(image).to(device=device, dtype=vae_dtype)
                    latents = self.vae.encode(pixels).latent_dist.mode()
                    latents = normalize_latents(latents, self.vae.config.latents_mean, self.vae.config.latents_std)
                    rows.append(pack_latents(latents).to(dtype))
                batch.image_latent = torch.cat(rows, dim=1)
                expected = int(state.image_pad_mask.sum().item()) * 4
                if expected != batch.image_latent.shape[1]:
                    raise ValueError("The processor's image slots do not match the VAE reference grid. "
                                     "Use the checkpoint's processor and a reference_resolution "
                                     "within its image budget.")
            finally:
                if fastvideo_args.vae_cpu_offload:
                    pinned_offload.unload(self.vae)
        # One target mask slot stands for four unpatched latent tokens.
        slots = state.image_pad_mask.new_ones(1, batch.latents.shape[1] // 4)
        state.image_pad_mask = torch.cat((state.image_pad_mask, slots), dim=1)
        if state.negative_image_pad_mask is not None:
            slots = state.negative_image_pad_mask.new_ones(1, batch.latents.shape[1] // 4)
            state.negative_image_pad_mask = torch.cat((state.negative_image_pad_mask, slots), dim=1)
        return batch


class QwenImage21ScheduleStage(PipelineStage):

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("latents", batch.latents, V.with_dims(3))

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("timesteps", batch.timesteps, V.with_dims(1))

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        sigmas = batch.sigmas.tolist() if isinstance(batch.sigmas, torch.Tensor) else batch.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / batch.num_inference_steps, batch.num_inference_steps).tolist()
        if not sigmas or any(not np.isfinite(sigma) or sigma <= 0 or sigma > 1 for sigma in sigmas):
            raise ValueError("sigmas must be finite values in (0, 1]")
        if any(first <= second for first, second in zip(sigmas, sigmas[1:], strict=False)):
            raise ValueError("sigmas must be strictly decreasing")
        self.scheduler.set_timesteps(sigmas=sigmas,
                                     device=get_local_torch_device(),
                                     mu=schedule_shift(batch.latents.shape[1], self.scheduler.config))
        self.scheduler.set_begin_index(0)
        batch.timesteps = self.scheduler.timesteps
        batch.num_inference_steps = len(batch.timesteps)
        return batch


class QwenImage21DenoisingStage(PipelineStage):
    performance_component_metric = "dit_time_s"

    def __init__(self, transformer, scheduler) -> None:
        self.transformer, self.scheduler = transformer, scheduler

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("latents", batch.latents,
                                              V.with_dims(3)).add_check("timesteps", batch.timesteps, V.with_dims(1))

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("latents", batch.latents, V.with_dims(3))

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state, device = _state(batch), get_local_torch_device()
        cache_device = fastvideo_args.pipeline_config.kv_cache_device
        if cache_device not in ("cpu", "cuda"):
            raise ValueError("kv_cache_device must be cpu or cuda; use_kv_cache=False disables caching")
        cache_enabled = fastvideo_args.pipeline_config.use_kv_cache and self.transformer.config.causal_condition
        caches = []
        if cache_enabled:
            caches = [QwenImage21KVCache(len(self.transformer.transformer_blocks), storage_device=cache_device)]
            if batch.do_classifier_free_guidance:
                caches.append(QwenImage21KVCache(len(self.transformer.transformer_blocks), storage_device=cache_device))
        move = (fastvideo_args.dit_cpu_offload and not fastvideo_args.dit_layerwise_offload
                and not fastvideo_args.use_fsdp_inference)
        try:
            if move:
                pinned_offload.load(self.transformer, device, pin=fastvideo_args.pin_cpu_memory)
            for index, timestep in enumerate(tqdm(batch.timesteps, desc="Denoising", total=len(batch.timesteps))):
                if getattr(self, "interrupt", False):
                    break
                latent_input = batch.latents
                if batch.image_latent is not None:
                    latent_input = torch.cat((batch.image_latent, latent_input), dim=1)
                mode = "extract" if cache_enabled and index == 0 else ("cached" if cache_enabled else None)

                def predict(embeds: torch.Tensor, mask: torch.Tensor | None, image_mask: torch.Tensor,
                            cache: QwenImage21KVCache | None, latent_input: torch.Tensor, timestep: torch.Tensor,
                            mode: str | None) -> torch.Tensor:
                    result = self.transformer(
                        hidden_states=latent_input,
                        timestep=timestep.expand(1).to(batch.latents.dtype) / 1000,
                        encoder_hidden_states=embeds,
                        encoder_hidden_states_mask=mask,
                        img_shapes=state.img_shapes,
                        img_mask=image_mask,
                        kv_cache=cache,
                        kv_cache_mode=mode,
                        return_dict=False,
                    )[0]
                    return result[:, -batch.latents.shape[1]:]

                while True:
                    try:
                        positive = predict(batch.prompt_embeds[0], batch.prompt_attention_mask[0], state.image_pad_mask,
                                           caches[0] if caches else None, latent_input, timestep, mode)
                        noise = positive
                        if batch.do_classifier_free_guidance:
                            negative = predict(batch.negative_prompt_embeds[0], batch.negative_attention_mask[0],
                                               state.negative_image_pad_mask, caches[1] if caches else None,
                                               latent_input, timestep, mode)
                            noise = negative + batch.true_cfg_scale * (positive - negative)
                        break
                    except QwenImage21KVCacheAllocationError:
                        if not cache_enabled:
                            raise
                        logger.warning("Host prefix-cache allocation failed; retrying this request without KV caching")
                        for cache in caches:
                            cache.clear()
                        caches = []
                        cache_enabled, mode = False, None
                        # An aborted layer may retain the next layer's prefetch.
                        self._release_layerwise()
                batch.latents = self.scheduler.step(noise, timestep, batch.latents, return_dict=False)[0]
                batch.step_index, batch.timestep = index, timestep
        finally:
            for cache in caches:
                cache.clear()
            if fastvideo_args.dit_layerwise_offload:
                self._release_layerwise()
            if move:
                pinned_offload.unload(self.transformer)
        return batch

    def _release_layerwise(self) -> None:
        # The shared offloader links the last block back to the first, leaving
        # its next-step prefetch resident. Also handle an interrupted block.
        for block in self.transformer.transformer_blocks:
            manager = ModuleHookManager.get_from(block)
            hook = manager.get_forward_hook("LayerwiseOffloadHook") if manager is not None else None
            if hook is not None and hook.state.gpu_named_parameters:
                torch.cuda.current_stream().wait_stream(hook.state.async_copy_stream)
                hook.state.release_gpu_params()


class QwenImage21DecodingStage(PipelineStage):
    performance_component_metric = "vae_decode_time_s"

    def __init__(self, vae) -> None:
        self.vae = vae

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("latents", batch.latents, V.with_dims(3))

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("output", batch.output, V.is_tensor)

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if fastvideo_args.output_type == "latent":
            batch.output = batch.latents
            return batch
        device = get_local_torch_device()
        try:
            pinned_offload.load(self.vae, device, pin=fastvideo_args.pin_cpu_memory)
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            else:
                self.vae.disable_tiling()
            dtype = next(self.vae.parameters()).dtype
            latents = unpack_latents(batch.latents, batch.height, batch.width).to(device=device, dtype=dtype)
            latents = normalize_latents(latents,
                                        self.vae.config.latents_mean,
                                        self.vae.config.latents_std,
                                        inverse=True)
            # Keep alpha as the fourth channel all the way to the PNG exporter.
            batch.output = self.vae.decode(latents, return_dict=False)[0].div(2).add(0.5).clamp(0, 1).float()
        finally:
            if fastvideo_args.vae_cpu_offload:
                pinned_offload.unload(self.vae)
        return batch
