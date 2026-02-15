# SPDX-License-Identifier: Apache-2.0
"""
Pipeline configuration for HunyuanGameCraft.

HunyuanGameCraft extends HunyuanVideo with:
1. CameraNet for camera/action conditioning (Plücker coordinates)
2. Mask-based conditioning for autoregressive generation
3. 33 input channels (16 latent + 16 gt_latent + 1 mask)

Text encoders are the same as HunyuanVideo:
- LLaVA-LLaMA-3-8B for primary text encoding (4096 dim)
- CLIP ViT-L/14 for secondary pooled embeddings (768 dim)
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypedDict

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import HunyuanGameCraftConfig
from fastvideo.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
    LlamaConfig,
)
from fastvideo.configs.models.vaes import GameCraftVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig

# GameCraft uses the same prompt template as HunyuanVideo
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>")


class PromptTemplate(TypedDict):
    template: str
    crop_start: int


prompt_template_video: PromptTemplate = {
    "template": PROMPT_TEMPLATE_ENCODE_VIDEO,
    "crop_start": 95,
}


def llama_preprocess_text(prompt: str) -> str:
    """Apply prompt template for LLaMA encoder."""
    return prompt_template_video["template"].format(prompt)


def llama_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Extract hidden states from LLaMA output, skipping instruction tokens."""
    hidden_state_skip_layer = 2
    assert outputs.hidden_states is not None
    hidden_states: tuple[torch.Tensor, ...] = outputs.hidden_states
    last_hidden_state: torch.Tensor = hidden_states[-(hidden_state_skip_layer +
                                                      1)]
    crop_start = prompt_template_video.get("crop_start", -1)
    last_hidden_state = last_hidden_state[:, crop_start:]
    return last_hidden_state


def clip_preprocess_text(prompt: str) -> str:
    """No preprocessing for CLIP encoder."""
    return prompt


def clip_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Extract pooled output from CLIP encoder."""
    pooler_output: torch.Tensor = outputs.pooler_output
    return pooler_output


@dataclass
class HunyuanGameCraftPipelineConfig(PipelineConfig):
    """Configuration for HunyuanGameCraft pipeline.
    
    Inherits text encoding from HunyuanVideo but uses:
    - GameCraft DiT with CameraNet
    - Same VAE (HunyuanVAE)
    - Same text encoders (LLaMA + CLIP)
    """

    # DiT config - uses GameCraft config (33 input channels)
    dit_config: DiTConfig = field(default_factory=HunyuanGameCraftConfig)

    # VAE config - GameCraft VAE (mid_block_causal_attn=True, etc.)
    vae_config: VAEConfig = field(default_factory=GameCraftVAEConfig)

    # Denoising parameters
    # Official GameCraft does NOT use embedded guidance (passes guidance=None)
    # It uses standard CFG with guidance_scale=6.0 instead
    embedded_cfg_scale = None
    flow_shift: int = 5  # Official GameCraft uses flow_shift=5.0

    # Text encoding stage - same as HunyuanVideo
    # Uses LLaMA-3-8B (via LLaVA) + CLIP
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (LlamaConfig(), CLIPTextConfig()))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (llama_preprocess_text, clip_preprocess_text))
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput], torch.Tensor],
        ...] = field(default_factory=lambda:
                     (llama_postprocess_text, clip_postprocess_text))

    # Precision for each component
    dit_precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp16", "fp16"))

    def __post_init__(self):
        # VAE only needs decoder for inference
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True
