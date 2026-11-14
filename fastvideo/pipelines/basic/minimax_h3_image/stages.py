# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image stages: resize, Qwen3-VL encode, PDD denoise, 2D decode."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from collections.abc import Iterator

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.distributed.tensor import DTensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models import pinned_offload
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_IMAGE_PAD_TOKEN,
    MINIMAX_H3_VISION_END_TOKEN,
    MINIMAX_H3_VISION_START_TOKEN,
)
from fastvideo.pipelines.basic.minimax_h3_image.packing import (
    MINIMAX_H3_IMAGE_CANVAS_MULTIPLE,
    MINIMAX_H3_IMAGE_MAX_REFERENCES,
    MINIMAX_H3_IMAGE_MAX_SEQUENCE_LENGTH,
    MINIMAX_H3_IMAGE_NUM_STEPS,
    MINIMAX_H3_IMAGE_REFERENCE_NOISE_AUG,
    MINIMAX_H3_IMAGE_TAG,
    MINIMAX_H3_IMAGE_TEXT_TAG,
    OUTPUT_HEIGHT_KEY,
    OUTPUT_WIDTH_KEY,
    REFERENCE_IMAGES_KEY,
    REFERENCE_MAX_SHORT_EDGE,
    REFERENCE_MIN_SHORT_EDGE,
    TEXT_TOKEN_TAGS_KEY,
    VISION_IMAGES_KEY,
    VISION_MAX_SHORT_EDGE,
    VISION_MIN_SHORT_EDGE,
    bound_short_edge,
    build_image_packed_sequence,
    edit_label,
    get_image_canvas_size,
    patchify_video_latents,
    rounded_sample_sigmas,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage


def _as_int(value: int | list[int] | None, name: str) -> int:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"MiniMax-H3-Image expects one {name}, got {value}.")
        value = value[0]
    if value is None:
        raise ValueError(f"MiniMax-H3-Image requires {name}.")
    return int(value)


def _load_rgb(image: Any) -> Image.Image:
    loaded = image if isinstance(image, Image.Image) else Image.open(image)
    transposed = ImageOps.exif_transpose(loaded)
    if transposed is None:
        transposed = loaded
    return transposed.convert("RGB")


def _resize(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.size == (width, height):
        return image
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((width, height), resampling)


def _request_generator(batch: ForwardBatch) -> torch.Generator:
    generator = batch.generator
    if isinstance(generator, list):
        if len(generator) != 1:
            raise ValueError("MiniMax-H3-Image generates one image per request.")
        generator = generator[0]
    if isinstance(generator, torch.Generator):
        return generator
    seed = 0 if batch.seed is None else int(batch.seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch.generator = generator
    return generator


def _randn(shape: tuple[int, ...], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    noise = torch.randn(shape, generator=generator, device=generator.device, dtype=torch.float32)
    return noise.to(device=device)


class MiniMaxH3ImageInputStage(PipelineStage):
    """Validate the fixed 6-step contract and resize edit references."""

    def __init__(self, min_generation_area: int) -> None:
        self.min_generation_area = int(min_generation_area)

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del fastvideo_args
        if not isinstance(batch.prompt, str) or not batch.prompt.strip():
            raise ValueError("MiniMax-H3-Image requires one non-empty prompt.")
        if _as_int(batch.num_frames, "num_frames") != 1:
            raise ValueError("MiniMax-H3-Image generates a single frame.")
        if int(batch.num_videos_per_prompt or 1) != 1:
            raise ValueError("MiniMax-H3-Image generates one image per request.")
        steps = int(batch.num_inference_steps)
        if steps != MINIMAX_H3_IMAGE_NUM_STEPS:
            raise ValueError(f"MiniMax-H3-Image only supports num_inference_steps="
                             f"{MINIMAX_H3_IMAGE_NUM_STEPS}, got {steps}.")
        guidance = float(batch.guidance_scale)
        if guidance != 1.0:
            raise ValueError(f"MiniMax-H3-Image is guidance-distilled; guidance_scale must be 1, got {guidance}.")
        negative = batch.negative_prompt or ""
        if negative.strip():
            raise ValueError("MiniMax-H3-Image does not take a negative prompt.")

        height = _as_int(batch.height, "height")
        width = _as_int(batch.width, "width")
        batch.extra[OUTPUT_HEIGHT_KEY] = height
        batch.extra[OUTPUT_WIDTH_KEY] = width
        _request_generator(batch)

        images = _reference_inputs(batch)
        if len(images) > MINIMAX_H3_IMAGE_MAX_REFERENCES:
            raise ValueError(f"MiniMax-H3-Image accepts at most {MINIMAX_H3_IMAGE_MAX_REFERENCES} references, "
                             f"got {len(images)}.")
        multiple = MINIMAX_H3_IMAGE_CANVAS_MULTIPLE
        reference_images = []
        vision_images = []
        for image in images:
            rgb = _load_rgb(image)
            if min(rgb.size) < multiple:
                raise ValueError(f"A reference image needs at least {multiple} pixels on each side, got {rgb.size}.")
            vae_size = bound_short_edge(rgb.size[0], rgb.size[1], REFERENCE_MIN_SHORT_EDGE, REFERENCE_MAX_SHORT_EDGE,
                                        multiple)
            vision_size = bound_short_edge(rgb.size[0], rgb.size[1], VISION_MIN_SHORT_EDGE, VISION_MAX_SHORT_EDGE,
                                           multiple)
            reference_images.append(_resize(rgb, vae_size[0], vae_size[1]))
            vision_images.append(_resize(rgb, vision_size[0], vision_size[1]))
        batch.extra[REFERENCE_IMAGES_KEY] = reference_images
        batch.extra[VISION_IMAGES_KEY] = vision_images
        return batch


def _reference_inputs(batch: ForwardBatch) -> list[Any]:
    if batch.image_path:
        paths = batch.image_path if isinstance(batch.image_path, list) else [batch.image_path]
        return list(paths)
    if batch.pil_image is None:
        return []
    if isinstance(batch.pil_image, list):
        return list(batch.pil_image)
    return [batch.pil_image]


class MiniMaxH3ImageConditioningStage(PipelineStage):
    """Encode a verbatim prompt, or ``Picture i: `` labels plus vision tokens for an edit."""

    def __init__(self, conditioner: Any, tokenizer: Any, processor: Any) -> None:
        self.conditioner = conditioner
        self.tokenizer = tokenizer
        self.processor = processor

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if self.conditioner is None or self.tokenizer is None:
            raise RuntimeError("MiniMax-H3-Image text encoder is not loaded.")
        device = get_local_torch_device()
        first_param = next(self.conditioner.parameters(), None)
        moved_for_forward = (fastvideo_args.text_encoder_cpu_offload and first_param is not None
                             and not isinstance(first_param, DTensor))
        if moved_for_forward:
            self.conditioner.to(device)
        try:
            prompt_embeds, token_tags = self._encode(batch, device)
        finally:
            if moved_for_forward:
                self.conditioner.to("cpu")
        batch.prompt_embeds = [prompt_embeds]
        batch.extra[TEXT_TOKEN_TAGS_KEY] = token_tags
        return batch

    def _encode(self, batch: ForwardBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(batch.prompt, str)
        vision_images = list(batch.extra.get(VISION_IMAGES_KEY) or [])
        token_ids, token_tags = self._presentation(batch.prompt, vision_images)
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        vision: dict[str, torch.Tensor | None] = {"pixel_values": None, "image_grid_thw": None}
        if vision_images:
            processed = self.processor.image_processor(images=vision_images, return_tensors="pt")
            dtype = self.conditioner.dtype
            vision = {
                "pixel_values": processed["pixel_values"].to(device=device, dtype=dtype),
                "image_grid_thw": processed["image_grid_thw"].to(device=device),
            }
        prompt_embeds = self.conditioner(input_ids, **vision)
        if prompt_embeds.ndim != 2 or prompt_embeds.shape[0] != len(token_ids):
            raise ValueError(f"MiniMax-H3-Image text encoder returned shape {tuple(prompt_embeds.shape)}.")
        return prompt_embeds.unsqueeze(0), torch.tensor(token_tags, dtype=torch.long)

    def _presentation(self, prompt: str, vision_images: list[Image.Image]) -> tuple[list[int], list[int]]:
        tokenizer = self.tokenizer
        max_length = MINIMAX_H3_IMAGE_MAX_SEQUENCE_LENGTH
        token_ids: list[int] = []
        token_tags: list[int] = []
        if vision_images:
            processed = self.processor.image_processor(images=vision_images, return_tensors="pt")
            merge_size = int(self.processor.image_processor.merge_size)**2
            for index, grid in enumerate(processed["image_grid_thw"]):
                label_ids = _token_ids(tokenizer(edit_label(index + 1), add_special_tokens=False))[:max_length]
                vision_ids = ([int(tokenizer.convert_tokens_to_ids(MINIMAX_H3_VISION_START_TOKEN))] +
                              [int(tokenizer.convert_tokens_to_ids(MINIMAX_H3_IMAGE_PAD_TOKEN))] *
                              (int(grid.prod()) // merge_size) +
                              [int(tokenizer.convert_tokens_to_ids(MINIMAX_H3_VISION_END_TOKEN))])
                token_ids.extend(label_ids)
                token_ids.extend(vision_ids)
                token_tags.extend([MINIMAX_H3_IMAGE_TEXT_TAG] * len(label_ids))
                token_tags.extend([MINIMAX_H3_IMAGE_TAG] * len(vision_ids))
        prompt_ids = _token_ids(tokenizer(prompt, add_special_tokens=False))[:max_length]
        token_ids.extend(prompt_ids)
        token_tags.extend([MINIMAX_H3_IMAGE_TEXT_TAG] * len(prompt_ids))
        if not token_ids:
            raise ValueError("MiniMax-H3-Image prompt tokenized to an empty sequence.")
        return token_ids, token_tags


def _token_ids(tokenized: Any) -> list[int]:
    input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


class MiniMaxH3ImageSamplingStage(PipelineStage):
    """Encode references, denoise with the per-step PDD head, and decode one image."""

    def __init__(self, vae: Any, transformer: Any, scheduler: Any, min_generation_area: int) -> None:
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.min_generation_area = int(min_generation_area)

    @contextmanager
    def _vae_on_device(self, fastvideo_args: FastVideoArgs, device: torch.device) -> Iterator[None]:
        offload = bool(fastvideo_args.vae_cpu_offload)
        if offload:
            pinned_offload.load(self.vae, device, pin=fastvideo_args.pin_cpu_memory)
        try:
            yield
        finally:
            if offload:
                pinned_offload.unload(self.vae)

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if self.vae is None or self.transformer is None or self.scheduler is None:
            raise RuntimeError("MiniMax-H3-Image denoise modules are not loaded.")
        device = next(self.transformer.parameters()).device
        generator = _request_generator(batch)
        output_height = int(batch.extra[OUTPUT_HEIGHT_KEY])
        output_width = int(batch.extra[OUTPUT_WIDTH_KEY])
        canvas_height, canvas_width = get_image_canvas_size(
            output_height,
            output_width,
            self.min_generation_area,
            MINIMAX_H3_IMAGE_CANVAS_MULTIPLE,
        )
        ratio = int(self.vae.spatial_compression_ratio)
        patch_values = tuple(int(value) for value in self.transformer.patch_size)
        if len(patch_values) != 3:
            raise ValueError(f"MiniMax-H3-Image patch_size must have three axes, got {patch_values}.")
        patch_size = (patch_values[0], patch_values[1], patch_values[2])
        if canvas_height % ratio or canvas_width % ratio:
            raise ValueError(f"Canvas {canvas_height}x{canvas_width} is not divisible by the VAE ratio {ratio}.")
        latent_height = canvas_height // ratio
        latent_width = canvas_width // ratio
        with self._vae_on_device(fastvideo_args, device):
            reference_latents = self._encode_references(batch, device, generator, ratio, patch_size)
        reference_grids = [(int(latent.shape[-2]), int(latent.shape[-1])) for latent in reference_latents]

        text_tags = batch.extra[TEXT_TOKEN_TAGS_KEY]
        position_ids, token_tags, image_indices, text_indices, num_reference_rows = build_image_packed_sequence(
            text_tags.detach().cpu(),
            reference_grids,
            latent_height,
            latent_width,
            patch_size,
        )
        position_ids = position_ids.to(device)
        token_tags = token_tags.to(device)
        image_indices = image_indices.to(device)
        text_indices = text_indices.to(device)

        prompt_embeds = batch.prompt_embeds[0].to(device=device)
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        target = _randn((1, self.vae.latent_channels, 1, latent_height, latent_width), generator, device)
        rows = [patchify_video_latents(target, patch_size)]
        condition_rows = []
        for latent in reference_latents:
            noise = _randn(tuple(latent.shape), generator, device)
            noised = self.scheduler.scale_noise(latent, torch.tensor(MINIMAX_H3_IMAGE_REFERENCE_NOISE_AUG), noise)
            condition_rows.append(patchify_video_latents(noised[:, :, None], patch_size))
        if condition_rows:
            rows = condition_rows + rows
        latents = torch.cat(rows, dim=0)
        if num_reference_rows and latents.shape[0] != image_indices.numel():
            raise RuntimeError("Packed reference rows do not match the layout.")

        sigmas = rounded_sample_sigmas().to(device)
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        sequence_length = int(image_indices.numel() + text_indices.numel())
        audio_hidden = latents.new_empty((1, 0, 1))
        audio_indices = torch.empty(0, dtype=torch.long, device=device)
        for step_index, timestep in enumerate(self.scheduler.timesteps):
            row_timesteps = torch.full((sequence_length, ), float(timestep), dtype=torch.float32, device=device)
            if num_reference_rows:
                row_timesteps[image_indices[:num_reference_rows]] = MINIMAX_H3_IMAGE_REFERENCE_NOISE_AUG
            unique_timesteps, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
            with set_forward_context(current_timestep=step_index, attn_metadata=None, forward_batch=batch):
                prediction, _ = self.transformer(
                    hidden_states=latents.unsqueeze(0),
                    audio_hidden_states=audio_hidden,
                    encoder_hidden_states=prompt_embeds,
                    timestep=unique_timesteps,
                    timestep_indices=timestep_indices,
                    token_tags=token_tags,
                    position_ids=position_ids,
                    video_indices=image_indices,
                    audio_indices=audio_indices,
                    text_indices=text_indices,
                    pdd_step_index=step_index,
                )
            updated = self.scheduler.step(
                prediction[0, num_reference_rows:].float(),
                timestep,
                latents[num_reference_rows:],
                return_dict=False,
            )[0]
            latents = latents.clone()
            latents[num_reference_rows:] = updated

        with self._vae_on_device(fastvideo_args, device):
            decoded = self._decode(latents[num_reference_rows:], latent_height, latent_width, patch_size, output_height,
                                   output_width)
        batch.output = decoded
        return batch

    def _encode_references(
        self,
        batch: ForwardBatch,
        device: torch.device,
        generator: torch.Generator,
        ratio: int,
        patch_size: tuple[int, int, int],
    ) -> list[torch.Tensor]:
        images = list(batch.extra.get(REFERENCE_IMAGES_KEY) or [])
        if not images:
            return []
        _, patch_h, patch_w = patch_size
        encoded = []
        for image in images:
            pixels = torch.from_numpy(np.asarray(image)).to(device).permute(2, 0, 1)[None]
            pixels = pixels[..., :pixels.shape[-2] // ratio * ratio, :pixels.shape[-1] // ratio * ratio]
            pixels = self.vae.normalize_pixels(pixels.to(torch.float32).div(255.0))
            posterior = self.vae.encode(pixels, return_dict=False)[0]
            latents = posterior.sample(generator=generator).to(torch.float16).float()
            latents = latents[..., :latents.shape[-2] // patch_h * patch_h, :latents.shape[-1] // patch_w * patch_w]
            encoded.append(self.vae.normalize_latents(latents))
        return encoded

    def _decode(
        self,
        rows: torch.Tensor,
        latent_height: int,
        latent_width: int,
        patch_size: tuple[int, int, int],
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        _, patch_h, patch_w = patch_size
        channels = int(self.vae.latent_channels)
        latents = rows.reshape(-1, latent_height // patch_h, latent_width // patch_w, channels, patch_h, patch_w)
        latents = latents.permute(0, 3, 1, 4, 2, 5).reshape(-1, channels, latent_height, latent_width).contiguous()
        latents = self.vae.denormalize_latents(latents)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.vae.denormalize_pixels(image.float()).clamp(0, 1)
        image = (image * 255).clamp(0, 255).to(torch.uint8)
        if (output_height, output_width) != tuple(image.shape[-2:]):
            array = image[0].permute(1, 2, 0).cpu().numpy()
            resized = _resize(Image.fromarray(array), output_width, output_height)
            image = torch.from_numpy(np.asarray(resized)).permute(2, 0, 1).unsqueeze(0)
        return image.unsqueeze(2).contiguous()
