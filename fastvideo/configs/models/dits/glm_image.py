# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "transformer_blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class GlmImageDiTArchConfig(DiTArchConfig):

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    hidden_size: int = 4096
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 30

    text_embed_dim: int = 1472
    time_embed_dim: int = 512
    condition_dim: int = 256

    prior_vq_quantizer_codebook_size: int = 16384

    patch_size: int = 2

    max_height: int = 2048
    max_width: int = 2048

    qk_norm: str = "layer_norm"
    eps: float = 1e-5

    exclude_lora_layers: list[str] = field(
        default_factory=lambda: ["image_projector", "glyph_projector", "prior_token_embedding"])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^glyph_projector\.net\.0\.proj\.(.*)$": r"glyph_projector.fc_in.\1",
            r"^glyph_projector\.net\.2\.(.*)$": r"glyph_projector.fc_out.\1",
            r"^prior_projector\.net\.0\.proj\.(.*)$": r"prior_projector.fc_in.\1",
            r"^prior_projector\.net\.2\.(.*)$": r"prior_projector.fc_out.\1",
            r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.(.*)$": r"transformer_blocks.\1.ff.fc_in.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.2\.(.*)$": r"transformer_blocks.\1.ff.fc_out.\2",
        })

    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.num_channels_latents = self.out_channels


@dataclass
class GlmImageDiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=GlmImageDiTArchConfig)
    prefix: str = "GlmImage"
