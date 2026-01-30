# SPDX-License-Identifier: Apache-2.0
"""
Pipeline configuration for Hunyuan-GameCraft.

This extends the HunyuanVideo pipeline with support for camera pose conditioning
via CameraNet using Plücker coordinate representation.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypedDict

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.hunyuangamecraft import HunyuanGameCraftConfig
from fastvideo.configs.models.encoders import (BaseEncoderOutput,
                                               CLIPTextConfig, LlamaConfig)
from fastvideo.configs.models.vaes import HunyuanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


# Prompt template for game video generation
PROMPT_TEMPLATE_GAMECRAFT = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)


class PromptTemplate(TypedDict):
    template: str
    crop_start: int


prompt_template_gamecraft: PromptTemplate = {
    "template": PROMPT_TEMPLATE_GAMECRAFT,
    "crop_start": 95,
}


def llama_preprocess_text(prompt: str) -> str:
    """Preprocess text prompt for LLaMA encoder."""
    return prompt_template_gamecraft["template"].format(prompt)


def llama_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Postprocess LLaMA encoder outputs."""
    hidden_state_skip_layer = 2
    assert outputs.hidden_states is not None
    hidden_states: tuple[torch.Tensor, ...] = outputs.hidden_states
    last_hidden_state: torch.Tensor = hidden_states[-(hidden_state_skip_layer + 1)]
    crop_start = prompt_template_gamecraft.get("crop_start", -1)
    last_hidden_state = last_hidden_state[:, crop_start:]
    return last_hidden_state


def clip_preprocess_text(prompt: str) -> str:
    """Preprocess text prompt for CLIP encoder."""
    return prompt


def clip_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Postprocess CLIP encoder outputs."""
    pooler_output: torch.Tensor = outputs.pooler_output
    return pooler_output


@dataclass
class HunyuanGameCraftPipelineConfig(PipelineConfig):
    """
    Pipeline configuration for Hunyuan-GameCraft.
    
    This configures the full video generation pipeline including:
    - HunyuanGameCraft DiT transformer with CameraNet
    - HunyuanVideo VAE for latent encoding/decoding
    - LLaMA + CLIP text encoders (same as HunyuanVideo)
    
    The camera conditioning uses Plücker coordinates (6D ray representation)
    which are processed through CameraNet and added to the image embeddings.
    """
    
    # DiT configuration (HunyuanGameCraft with CameraNet)
    dit_config: DiTConfig = field(default_factory=HunyuanGameCraftConfig)
    
    # VAE configuration (same as HunyuanVideo - causal 3D VAE)
    vae_config: VAEConfig = field(default_factory=HunyuanVAEConfig)
    
    # Denoising parameters
    embedded_cfg_scale: int = 6
    flow_shift: int = 7
    
    # Text encoder configurations (LLaMA + CLIP)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (LlamaConfig(), CLIPTextConfig())
    )
    
    # Text preprocessing functions
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (llama_preprocess_text, clip_preprocess_text)
    )
    
    # Text postprocessing functions
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput], torch.Tensor], ...
    ] = field(
        default_factory=lambda: (llama_postprocess_text, clip_postprocess_text)
    )
    
    # Precision settings for each component
    dit_precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp16", "fp16")
    )
    
    def __post_init__(self):
        """Post-initialization configuration."""
        # Only load VAE decoder by default (encoder not needed for inference)
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class HunyuanGameCraftT2VConfig(HunyuanGameCraftPipelineConfig):
    """
    Text-to-Video configuration for Hunyuan-GameCraft.
    
    Standard T2V generation with camera pose conditioning.
    """
    pass


@dataclass
class HunyuanGameCraftI2VConfig(HunyuanGameCraftPipelineConfig):
    """
    Image-to-Video configuration for Hunyuan-GameCraft.
    
    I2V generation with camera pose conditioning.
    This configuration would typically include an image encoder
    for processing the initial frame.
    """
    
    def __post_init__(self):
        super().__post_init__()
        # Enable VAE encoder for encoding the initial image
        self.vae_config.load_encoder = True
