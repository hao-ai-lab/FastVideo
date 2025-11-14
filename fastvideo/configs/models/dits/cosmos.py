# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_transformer_blocks(n: str, m) -> bool:
    return "transformer_blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class CosmosArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_transformer_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embed\.(.*)$": r"patch_embed.\1",
            r"^time_embed\.time_proj\.(.*)$": r"time_embed.time_proj.\1",
            r"^time_embed\.t_embedder\.(.*)$": r"time_embed.t_embedder.\1",
            r"^time_embed\.norm\.(.*)$": r"time_embed.norm.\1",
            r"^transformer_blocks\.(\d+)\.attn1\.to_q\.(.*)$":
            r"transformer_blocks.\1.attn1.to_q.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.to_k\.(.*)$":
            r"transformer_blocks.\1.attn1.to_k.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.to_v\.(.*)$":
            r"transformer_blocks.\1.attn1.to_v.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$":
            r"transformer_blocks.\1.attn1.to_out.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.norm_q\.(.*)$":
            r"transformer_blocks.\1.attn1.norm_q.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.norm_k\.(.*)$":
            r"transformer_blocks.\1.attn1.norm_k.\2",
            r"^transformer_blocks\.(\d+)\.attn2\.to_q\.(.*)$":
            r"transformer_blocks.\1.attn2.to_q.\2",
            r"^transformer_blocks\.(\d+)\.attn2\.to_k\.(.*)$":
            r"transformer_blocks.\1.attn2.to_k.\2",
            r"^transformer_blocks\.(\d+)\.attn2\.to_v\.(.*)$":
            r"transformer_blocks.\1.attn2.to_v.\2",
            r"^transformer_blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$":
            r"transformer_blocks.\1.attn2.to_out.\2",
            r"^transformer_blocks\.(\d+)\.attn2\.norm_q\.(.*)$":
            r"transformer_blocks.\1.attn2.norm_q.\2",
            r"^transformer_blocks\.(\d+)\.attn2\.norm_k\.(.*)$":
            r"transformer_blocks.\1.attn2.norm_k.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.(.*)$":
            r"transformer_blocks.\1.ff.fc_in.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.2\.(.*)$":
            r"transformer_blocks.\1.ff.fc_out.\2",
            r"^norm_out\.(.*)$": r"norm_out.\1",
            r"^proj_out\.(.*)$": r"proj_out.\1",
        })

    lora_param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^transformer_blocks\.(\d+)\.attn1\.to_q\.(.*)$":
            r"transformer_blocks.\1.attn1.to_q.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.to_k\.(.*)$":
            r"transformer_blocks.\1.attn1.to_k.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.to_v\.(.*)$":
            r"transformer_blocks.\1.attn1.to_v.\2",
            r"^transformer_blocks\.(\d+)\.attn1\.to_out\.(.*)$":
            r"transformer_blocks.\1.attn1.to_out.\2",
            r"^transformer_blocks\.(\d+)\.attn2\.to_q\.(.*)$":
            r"transformer_blocks.\1.attn2.to_q.\2",
            r"^transformer_blocks\.(\d+)\.attn2\.to_k\.(.*)$":
            r"transformer_blocks.\1.attn2.to_k.\2",
            r"^transformer_blocks\.(\d+)\.attn2\.to_v\.(.*)$":
            r"transformer_blocks.\1.attn2.to_v.\2",
            r"^transformer_blocks\.(\d+)\.attn2\.to_out\.(.*)$":
            r"transformer_blocks.\1.attn2.to_out.\2",
            r"^transformer_blocks\.(\d+)\.ff\.(.*)$":
            r"transformer_blocks.\1.ff.\2",
        })

    # Cosmos-specific config parameters based on transformer_cosmos.py
    in_channels: int = 16
    out_channels: int = 16
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    num_layers: int = 28
    mlp_ratio: float = 4.0
    text_embed_dim: int = 1024
    adaln_lora_dim: int = 256
    max_size: tuple[int, int, int] = (128, 240, 240)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    rope_scale: tuple[float, float, float] = (1.0, 3.0, 3.0)
    concat_padding_mask: bool = True
    extra_pos_embed_type: str | None = None
    qk_norm: str = "rms_norm"
    eps: float = 1e-6
    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.in_channels


@dataclass
class CosmosVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=CosmosArchConfig)
    prefix: str = "Cosmos"
