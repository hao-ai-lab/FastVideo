# SPDX-License-Identifier: Apache-2.0
"""Dense LingBot-Video T2V pipeline configuration."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.lingbot_video import LingBotVideoConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.lingbot_video import LingBotVideoQwen3VLTextConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig

PROMPT_CROP_START = 140
PROMPT_TEMPLATE = ("<|im_start|>system\nGiven a user input that may include a text prompt alone, "
                   "a text prompt with an image reference, or a text prompt with a video reference "
                   'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
                   "visual descriptions suitable for video generation. Evaluate the level of detail "
                   "in the user's input: if it is simple, enrich it by adding specifics about colors, "
                   "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
                   "progression, and spatial relationships to create vivid, concrete, and temporally "
                   "coherent scenes to create vivid and concrete scenes. Please generate only the "
                   "enhanced description for the prompt below and avoid including any additional "
                   "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
                   "<|im_start|>assistant\n")


def preprocess_lingbot_video_prompt(prompt: str) -> str:
    """Apply the released T2V system/user/assistant prompt template."""
    return PROMPT_TEMPLATE.format(prompt)


def postprocess_lingbot_video_text(
    outputs: BaseEncoderOutput,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the final hidden state, crop the template, and trim batch-one padding."""
    if outputs.hidden_states is None:
        raise ValueError("LingBot-Video requires text-encoder hidden states")
    prompt_embeds = outputs.hidden_states[-1][:, PROMPT_CROP_START:]
    prompt_mask = attention_mask[:, PROMPT_CROP_START:]
    if prompt_embeds.shape[0] == 1:
        true_length = int(prompt_mask[0].sum().item())
        prompt_embeds = prompt_embeds[:, :true_length]
        prompt_mask = prompt_mask[:, :true_length]
    return prompt_embeds, prompt_mask


@dataclass
class LingBotVideoT2VConfig(PipelineConfig):
    """Released Dense T2V component wiring and numerical precision policy."""

    dit_config: DiTConfig = field(default_factory=LingBotVideoConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (LingBotVideoQwen3VLTextConfig(), ))
    preprocess_text_funcs: tuple[Callable, ...] = field(default_factory=lambda: (preprocess_lingbot_video_prompt, ))
    postprocess_text_funcs: tuple[Callable, ...] = field(default_factory=lambda: (postprocess_lingbot_video_text, ))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))
    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    vae_decode_precision: str | None = "fp32"
    vae_tiling: bool = False
    vae_sp: bool = False
    flow_shift: float | None = 3.0
    embedded_cfg_scale: float | None = None
    scheduler_step_in_fp32: bool = True

    def __post_init__(self) -> None:
        """Load only the VAE decoder for the T2V workload."""
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True
