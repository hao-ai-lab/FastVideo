# SPDX-License-Identifier: Apache-2.0
"""GLM-Image DiT configuration.

GLM-Image uses a 7B diffusion transformer (DiT) decoder that expands tokens
from the autoregressive vision-language encoder into high-resolution images.
"""
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "transformer_blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class GlmImageDiTArchConfig(DiTArchConfig):
    """Architecture config for GlmImageTransformer2DModel."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    # GLM-Image DiT settings (7B model)
    # hidden_size = num_attention_heads * attention_head_dim = 64 * 64 = 4096
    hidden_size: int = 4096
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 30

    # Text and condition dims
    text_embed_dim: int = 1472
    time_embed_dim: int = 512
    condition_dim: int = 256

    # VQ settings for AR tokens
    prior_vq_quantizer_codebook_size: int = 16384

    # Patch embedding
    patch_size: int = 2

    # Positional embedding max resolution
    max_height: int = 2048
    max_width: int = 2048

    # QK normalization
    qk_norm: str = "layer_norm"
    eps: float = 1e-5

    # LoRA exclusions
    exclude_lora_layers: list[str] = field(
        default_factory=lambda:
        ["image_projector", "glyph_projector", "prior_token_embedding"])

    # Param name mappings for weight loading (HF -> custom)
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Projectors (mapped from FeedForward net.0.proj -> fc_in, net.2 -> fc_out)
            r"^image_projector\.net\.0\.proj\.(.*)$":
            r"image_projector.fc_in.\1",
            r"^image_projector\.net\.2\.(.*)$": r"image_projector.fc_out.\1",
            r"^glyph_projector\.net\.0\.proj\.(.*)$":
            r"glyph_projector.fc_in.\1",
            r"^glyph_projector\.net\.2\.(.*)$": r"glyph_projector.fc_out.\1",
            r"^prior_projector\.net\.0\.proj\.(.*)$":
            r"prior_projector.fc_in.\1",
            r"^prior_projector\.net\.2\.(.*)$": r"prior_projector.fc_out.\1",
            r"^prior_token_embedding\.(.*)$": r"prior_token_embedding.\1",

            # Transformer blocks
            r"^transformer_blocks\.(\d+)\.norm1\.(.*)$":
            r"transformer_blocks.\1.norm1.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.to_q\.(.*)$":
            r"transformer_blocks.\1.attn1.to_q.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.to_k\.(.*)$":
            r"transformer_blocks.\1.attn1.to_k.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.to_v\.(.*)$":
            r"transformer_blocks.\1.attn1.to_v.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$":
            r"transformer_blocks.\1.attn1.to_out.0.\2",

            # FeedForward in blocks (net.0.proj -> fc_in, net.2 -> fc_out)
            r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.(.*)$":
            r"transformer_blocks.\1.ff.fc_in.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.2\.(.*)$":
            r"transformer_blocks.\1.ff.fc_out.\2",

            # Output
            r"^norm_out\.(.*)$": r"norm_out.\1",
            r"^proj_out\.(.*)$": r"proj_out.\1",
        })

    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.num_channels_latents = self.out_channels


@dataclass
class GlmImageDiTConfig(DiTConfig):
    """Configuration for GLM-Image DiT model."""

    arch_config: DiTArchConfig = field(default_factory=GlmImageDiTArchConfig)
    prefix: str = "GlmImage"
