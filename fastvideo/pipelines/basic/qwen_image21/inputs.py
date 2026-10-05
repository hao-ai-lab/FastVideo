# SPDX-License-Identifier: Apache-2.0
# Adapted from Hugging Face Diffusers' QwenImage21Pipeline (Apache-2.0).
"""Image, prompt and latent conventions shared by Qwen-Image-2.1 stages.

This module intentionally has no FastVideo runtime imports so input preparation
and layout checks also run on a CPU-only development machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

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
    if latents.ndim != 5 or latents.shape[2] != 1:
        raise ValueError("Qwen-Image-2.1 latents must have shape [B, C, 1, H, W]")
    return latents.flatten(2).transpose(1, 2)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
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
    for name in ("height", "width", "reference_resolution"):
        value = getattr(batch, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value % 32:
            raise ValueError(f"{name} must be a positive multiple of 32")
    if not isinstance(batch.num_inference_steps, int) or batch.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    if not math.isfinite(batch.true_cfg_scale) or batch.true_cfg_scale < 1:
        raise ValueError("true_cfg_scale must be finite and at least 1")
    if batch.guidance_scale != 1:
        raise ValueError("Qwen-Image-2.1 has no embedded guidance; use true_cfg_scale with negative_prompt")
    if batch.video_path is not None or batch.conditioning_mask is not None:
        raise ValueError("Pass annotations or mask images as references; this pipeline takes single-frame images")
    if batch.prompt_embeds or batch.negative_prompt_embeds:
        raise ValueError("Externally supplied prompt embeddings are not supported by this Qwen-Image-2.1 port")
    if batch.negative_prompt is not None and not isinstance(batch.negative_prompt, str):
        raise ValueError("negative_prompt must be a string or None")
