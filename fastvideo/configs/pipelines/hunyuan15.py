# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypedDict, List, Dict, Any
import re

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import HunyuanVideo15Config
from fastvideo.configs.models.encoders import (BaseEncoderOutput,
                                               Qwen2_5_VLConfig, T5Config)
from fastvideo.configs.models.vaes import Hunyuan15VAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig

PROMPT_TEMPLATE_ENCODE_VIDEO = "You are a helpful assistant. Describe the video by detailing the following aspects: \
1. The main content and theme of the video. \
2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
4. background environment, light, style and atmosphere. \
5. camera angles, movements, and transitions used in the video."

def extract_glyph_texts(prompt: str) -> List[str]:
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
        formatted_result = ". ".join([f'Text "{text}"' for text in result]) + ". "
    else:
        formatted_result = None

    return formatted_result

def format_text_input(prompt: List[str], system_message: str) -> List[Dict[str, Any]]:
    """
    Apply text to template.

    Args:
        prompt (List[str]): Input text.
        system_message (str): System message.

    Returns:
        List[Dict[str, Any]]: List of chat conversation.
    """

    template = [
        [{"role": "system", "content": system_message}, {"role": "user", "content": p if p else " "}] for p in prompt
    ]

    return template


def qwen_preprocess_text(prompt: str) -> str:
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = format_text_input(prompt, PROMPT_TEMPLATE_ENCODE_VIDEO)
    return prompt


def qwen_postprocess_text(outputs: BaseEncoderOutput) -> torch.tensor:
    assert outputs.hidden_states is not None
    output = outputs.hidden_states[-3]
    output = output[:, 108:]
    return output


def t5_preprocess_text(prompt: str) -> str:
    prompt = [prompt] if isinstance(prompt, str) else prompt
    glyph_texts = [extract_glyph_texts(p) for p in prompt]
    return prompt


def clip_postprocess_text(outputs: BaseEncoderOutput) -> torch.tensor:
    pooler_output: torch.tensor = outputs.pooler_output
    return pooler_output


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
        default_factory=lambda: (T5Config(), Qwen2_5_VLConfig()))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (qwen_preprocess_text, clip_preprocess_text))
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput], torch.tensor],
        ...] = field(default_factory=lambda:
                     (qwen_postprocess_text, clip_postprocess_text))

    # Precision for each component
    dit_precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", "fp32"))
    text_encoder_crop_start: int = 108

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True
