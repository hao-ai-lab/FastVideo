# SPDX-License-Identifier: Apache-2.0
"""GLM-Image pipeline configuration.

GLM-Image is a hybrid model combining:
- 9B autoregressive (AR) vision-language encoder for semantic/layout token generation
- 7B diffusion transformer (DiT) decoder for high-fidelity image generation

The pipeline supports:
- Text-to-image (T2I) generation
- Image-to-image (I2I) editing/style transfer
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.glm_image import GlmImageDiTConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.configs.models.vaes.glm_image import GlmImageVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def glm_image_t5_postprocess(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Postprocess T5 encoder outputs for GLM-Image.
    
    GLM-Image uses T5 as a glyph/text encoder for text rendering capabilities.
    """
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()

    assert torch.isnan(hidden_state).sum() == 0, "T5 hidden states contain NaN"

    # Pack embeddings with proper padding
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    max_len = 512  # GLM-Image uses 512 max sequence length
    prompt_embeds_tensor: torch.Tensor = torch.stack([
        torch.cat([u, u.new_zeros(max_len - u.size(0), u.size(1))])
        for u in prompt_embeds
    ],
                                                     dim=0)

    return prompt_embeds_tensor


@dataclass
class GlmImageConfig(PipelineConfig):
    """Configuration for GLM-Image T2I/I2I pipeline.
    
    This configuration sets up the complete GLM-Image pipeline including:
    - T5 text encoder (glyph encoder for text rendering)
    - Vision-language encoder (AR model for semantic tokens)
    - DiT diffusion decoder
    - VAE for image encoding/decoding
    """

    # Image generation flag (not video)
    is_video_pipeline: bool = False

    # DiT configuration
    dit_config: DiTConfig = field(default_factory=GlmImageDiTConfig)
    dit_precision: str = "bf16"

    # VAE configuration
    vae_config: VAEConfig = field(default_factory=GlmImageVAEConfig)
    vae_precision: str = "fp32"
    vae_tiling: bool = True
    vae_sp: bool = False  # No spatial parallelism for image VAE

    # Text encoder (T5 for glyph/text rendering)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(), ))
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp32", ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda:
                                               (glm_image_t5_postprocess, ))

    # Flow matching scheduler settings
    flow_shift: float | None = 1.0
    embedded_cfg_scale: float = 7.5

    # AR model sampling parameters
    ar_temperature: float = 0.95
    ar_top_p: float = 0.75
    ar_do_sample: bool = True
    ar_max_new_tokens: int = 1024

    # Image resolution constraints (must be divisible by 32)
    default_height: int = 1024
    default_width: int = 1024
    resolution_multiple: int = 32

    # Post-decoding hook for custom processing
    post_decoding: Callable | None = None

    def __post_init__(self):
        # GLM-Image needs both encoder and decoder
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def validate_resolution(self, height: int, width: int) -> tuple[int, int]:
        """Ensure resolution is divisible by 32 as required by GLM-Image."""
        h = (height // self.resolution_multiple) * self.resolution_multiple
        w = (width // self.resolution_multiple) * self.resolution_multiple
        return h, w


@dataclass
class GlmImageI2IConfig(GlmImageConfig):
    """Configuration for GLM-Image image-to-image pipeline.
    
    Inherits from GlmImageConfig with settings optimized for I2I tasks
    like style transfer, editing, and inpainting.
    """

    # Lower CFG for I2I to preserve input structure
    embedded_cfg_scale: float = 5.0

    # Strength for I2I (how much of the original to preserve)
    strength: float = 0.8
