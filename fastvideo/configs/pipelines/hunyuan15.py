# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
import re

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import HunyuanVideo15Config
from fastvideo.configs.models.encoders import (BaseEncoderOutput,
                                               Qwen2_5_VLConfig, T5Config)
from fastvideo.configs.models.vaes import Hunyuan15VAEConfig
from fastvideo.configs.models.upsamplers import SRTo1080pUpsamplerConfig
from fastvideo.configs.pipelines.base import PipelineConfig, UpsamplerConfig

PROMPT_TEMPLATE_TOKEN_LENGTH = 108

PROMPT_TEMPLATE_ENCODE_VIDEO = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."


def extract_glyph_texts(prompt: str) -> str | None:
    """
    Extract glyph texts from prompt using regex pattern.

    Args:
        prompt: Input prompt string

    Returns:
        List of extracted glyph texts
    """
    pattern = r"\"(.*?)\"|“(.*?)”"
    matches = re.findall(pattern, prompt)
    result = [match[0] or match[1] for match in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result

    if result:
        formatted_result = ". ".join([f'Text "{text}"'
                                      for text in result]) + ". "
    else:
        formatted_result = None

    return formatted_result


def format_text_input(prompt: str, system_message: str) -> list[dict[str, Any]]:
    """
    Apply text to template.

    Args:
        prompt (List[str]): Input text.
        system_message (str): System message.

    Returns:
        List[Dict[str, Any]]: List of chat conversation.
    """

    template = [{
        "role": "system",
        "content": system_message
    }, {
        "role": "user",
        "content": prompt if prompt else " "
    }]

    return template


def qwen_preprocess_text(prompt: str) -> list[dict[str, Any]]:
    output = format_text_input(prompt, PROMPT_TEMPLATE_ENCODE_VIDEO)
    return output


def qwen_postprocess_text(
        outputs: BaseEncoderOutput,
        mask: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
    assert outputs.hidden_states is not None
    output = outputs.hidden_states[-3]
    output = output[:, PROMPT_TEMPLATE_TOKEN_LENGTH:]
    mask = mask[:, PROMPT_TEMPLATE_TOKEN_LENGTH:]
    return output, mask


def byt5_preprocess_text(prompt: str) -> str | None:
    prompts = [prompt] if isinstance(prompt, str) else prompt
    glyph_texts = [extract_glyph_texts(p) for p in prompts]
    return glyph_texts[0]


def byt5_postprocess_text(outputs: BaseEncoderOutput) -> torch.tensor:
    return outputs.last_hidden_state


@dataclass
class Hunyuan15T2V480PConfig(PipelineConfig):
    """Base configuration for HunYuan pipeline architecture."""

    # HunyuanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=HunyuanVideo15Config)
    # VAE
    vae_config: VAEConfig = field(default_factory=Hunyuan15VAEConfig)
    # Denoising stage
    flow_shift: int = 5

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen2_5_VLConfig(), T5Config()))
    preprocess_text_funcs: tuple[Callable[[Any], Any], ...] = field(
        default_factory=lambda: (qwen_preprocess_text, byt5_preprocess_text))
    postprocess_text_funcs: tuple[Callable[..., Any], ...] = field(
        default_factory=lambda: (qwen_postprocess_text, byt5_postprocess_text))

    # Precision for each component
    dit_precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", "fp32"))
    text_encoder_crop_start: int = PROMPT_TEMPLATE_TOKEN_LENGTH
    text_encoder_max_lengths: tuple[int, ...] = field(
        default_factory=lambda: (1000 + PROMPT_TEMPLATE_TOKEN_LENGTH, 256))

    vae_tiling: bool = True

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class Hunyuan15T2V720PConfig(Hunyuan15T2V480PConfig):
    """Base configuration for HunYuan pipeline architecture."""

    # HunyuanConfig-specific parameters with defaults
    flow_shift: int = 9


@dataclass
class Hunyuan15I2V720PConfig(Hunyuan15T2V720PConfig):
    """Base configuration for HunYuan pipeline architecture."""

    # HunyuanConfig-specific parameters with defaults
    flow_shift: int = 7


@dataclass
class Hunyuan15SR1080PConfig(Hunyuan15T2V720PConfig):
    """Base configuration for HunYuan pipeline architecture."""

    # HunyuanConfig-specific parameters with defaults
    flow_shift: int = 9
    flow_shift_sr: int = 2
    upsampler_config: UpsamplerConfig = field(
        default_factory=SRTo1080pUpsamplerConfig)
    upsampler_precision: str = "fp32"
