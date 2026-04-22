# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_transformer_layer(n: str, m) -> bool:
    return "layers" in n and str.isdigit(n.split(".")[-1])


@dataclass
class DaVinciArchConfig(DiTArchConfig):
    """Configuration for daVinci-MagiHuman DiT architecture."""

    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_transformer_layer])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Adapter: official uses *_embedder, FastVideo uses *_proj
            r"^adapter\.video_embedder\.(.*)$":
            r"adapter.video_proj.\1",
            r"^adapter\.text_embedder\.(.*)$":
            r"adapter.text_proj.\1",
            r"^adapter\.audio_embedder\.(.*)$":
            r"adapter.audio_proj.\1",

            # Transformer layers: block.layers.N -> layers.N
            # All sub-keys (attention, mlp, norms) pass through unchanged
            r"^block\.layers\.(\d+)\.(.*)$":
            r"layers.\1.\2",

            # Final norms: pass through (same names)
            r"^(final_norm_video\..*)$":
            r"\1",
            r"^(final_norm_audio\..*)$":
            r"\1",

            # Final projections: official uses linear, FastVideo uses proj
            r"^final_linear_video\.(.*)$":
            r"final_proj_video.\1",
            r"^final_linear_audio\.(.*)$":
            r"final_proj_audio.\1",
        })

    # Architecture parameters (from inference/common/config.py ModelConfig)
    num_layers: int = 40
    hidden_size: int = 5120
    head_dim: int = 128
    num_heads_q: int = 40
    num_heads_kv: int = 8
    num_channels_latents: int = 48
    patch_size: tuple = (1, 2, 2)
    video_in_channels: int = 192  # 48 * 1 * 2 * 2
    audio_in_channels: int = 64
    text_in_channels: int = 3584  # t5gemma-9b hidden dim

    def __post_init__(self):
        super().__post_init__()
        self.num_attention_heads = self.num_heads_q


@dataclass
class DaVinciDiTConfig(DiTConfig):
    """Full pipeline config for daVinci-MagiHuman."""
    arch_config: DiTArchConfig = field(default_factory=DaVinciArchConfig)
    prefix: str = "DaVinci"