# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Ovis-Image text-to-image model."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig
from fastvideo.configs.models.dits import OvisImageTransformer2DModelConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, Qwen3Config
from fastvideo.configs.pipelines.base import PipelineConfig

# System prompt from the Diffusers OvisImagePipeline
OVIS_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, quantity, text, shape, size, "
    "texture, spatial relationships of the objects and background: ")
# Number of tokens the system prompt + chat template special tokens occupy
USER_PROMPT_BEGIN_ID = 28


def qwen3_preprocess_text(prompt: str) -> list[dict[str, Any]]:
    """Format prompt as a chat message with system prompt for Qwen3.

    The Ovis-Image pipeline prepends a system prompt to guide text encoding,
    formatted as a single user message for the chat template.
    """
    return [{"role": "user", "content": OVIS_SYSTEM_PROMPT + prompt}]


def qwen3_postprocess_text(
        outputs: BaseEncoderOutput,
        mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Post-process Qwen3 encoder output for Ovis-Image.

    Applies attention mask to zero out padding tokens, then slices off
    the system prompt tokens (first 28 tokens) from both embeddings and mask.
    """
    prompt_embeds = outputs.last_hidden_state
    # Zero out padding tokens
    prompt_embeds = prompt_embeds * mask[..., None]
    # Slice off system prompt tokens
    prompt_embeds = prompt_embeds[:, USER_PROMPT_BEGIN_ID:]
    mask = mask[:, USER_PROMPT_BEGIN_ID:]
    return prompt_embeds, mask


@dataclass
class OvisImageT2IConfig(PipelineConfig):
    """
    Configuration for Ovis-Image-7B text-to-image pipeline.

    Ovis-Image is optimized for high-quality text rendering in generated images.
    This config uses Qwen3 (Ovis2.5-2B) as the text encoder.
    """

    # Denoising stage
    embedded_cfg_scale: float = 5.0
    flow_shift: float = 3.0

    # DiT configuration
    dit_config: DiTConfig = field(
        default_factory=OvisImageTransformer2DModelConfig)

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen3Config(), ))
    preprocess_text_funcs: tuple[Callable[[str], list[dict[str, Any]]],
                                 ...] = field(default_factory=lambda:
                                              (qwen3_preprocess_text, ))
    postprocess_text_funcs: tuple[Callable[[Any, Any], tuple[Any, Any]],
                                  ...] = field(default_factory=lambda:
                                               (qwen3_postprocess_text, ))

    # Precision for each component
    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", ))

    def __post_init__(self):
        """Configure VAE for decoder-only mode."""
        # Since VAEConfig may be set via kwargs, check and configure
        if hasattr(self, 'vae_config') and self.vae_config is not None:
            self.vae_config.load_encoder = False
            self.vae_config.load_decoder = True
