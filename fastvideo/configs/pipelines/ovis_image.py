# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig
from fastvideo.configs.models.dits import OvisImageTransformer2DModelConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, Qwen3Config
from fastvideo.configs.pipelines.base import PipelineConfig

OVIS_SYSTEM_PROMPT = ("Describe the image by detailing the color, quantity, text, shape, size, "
                      "texture, spatial relationships of the objects and background: ")
# Tokens the system prompt + chat-template specials occupy, sliced off after encoding.
USER_PROMPT_BEGIN_ID = 28


def qwen3_preprocess_text(prompt: str) -> list[dict[str, Any]]:
    """Wrap the prompt as a system-prefixed Qwen3 chat message."""
    return [{"role": "user", "content": OVIS_SYSTEM_PROMPT + prompt}]


def qwen3_postprocess_text(outputs: BaseEncoderOutput, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask padding tokens, then drop the leading system-prompt tokens."""
    prompt_embeds = outputs.last_hidden_state
    prompt_embeds = prompt_embeds * mask[..., None]
    prompt_embeds = prompt_embeds[:, USER_PROMPT_BEGIN_ID:]
    mask = mask[:, USER_PROMPT_BEGIN_ID:]
    return prompt_embeds, mask


@dataclass
class OvisImageT2IConfig(PipelineConfig):
    """Ovis-Image-7B T2I config (Qwen3 / Ovis2.5-2B text encoder)."""

    embedded_cfg_scale: float | None = None
    flow_shift: float = 3.0

    dit_config: DiTConfig = field(default_factory=OvisImageTransformer2DModelConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (Qwen3Config(), ))
    preprocess_text_funcs: tuple[Callable[[str], list[dict[str, Any]]],
                                 ...] = field(default_factory=lambda: (qwen3_preprocess_text, ))
    postprocess_text_funcs: tuple[Callable[[Any, Any], tuple[Any, Any]],
                                  ...] = field(default_factory=lambda: (qwen3_postprocess_text, ))

    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))

    def __post_init__(self):
        """Configure VAE for decoder-only mode.

        `PipelineConfig` (the base) has no `__post_init__`; it invokes this hook
        itself via `hasattr`, so there is intentionally no `super()` call.
        """
        # T2I only decodes, so skip loading the VAE encoder weights.
        if hasattr(self, 'vae_config') and self.vae_config is not None:
            self.vae_config.load_encoder = False
            self.vae_config.load_decoder = True
