# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class LTXVideoArchConfig(DiTArchConfig):
    fsdp_shard_conditions: list = field(
        default_factory=lambda:
        [lambda n, m: "transformer_blocks" in n and n.split(".")[-1].isdigit()])

    # Parameter name mappings for loading pretrained weights from HuggingFace/Diffusers
    # Maps from source model parameter names to FastVideo LTX implementation names
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # todo: double check all of this
            r"^transformer_blocks\.(\d+)\.norm1\.weight$":
            r"transformer_blocks.\1.norm1.weight",
            r"^transformer_blocks\.(\d+)\.norm2\.weight$":
            r"transformer_blocks.\1.norm2.weight",

            # FeedForward network mappings (double check)
            r"^transformer_blocks\.(\d+)\.ff\.net\.0\.weight$":
            r"transformer_blocks.\1.ff.net.0.weight",
            r"^transformer_blocks\.(\d+)\.ff\.net\.0\.bias$":
            r"transformer_blocks.\1.ff.net.0.bias",
            r"^transformer_blocks\.(\d+)\.ff\.net\.3\.weight$":
            r"transformer_blocks.\1.ff.net.3.weight",
            r"^transformer_blocks\.(\d+)\.ff\.net\.3\.bias$":
            r"transformer_blocks.\1.ff.net.3.bias",

            # Scale-shift table for adaptive layer norm
            r"^transformer_blocks\.(\d+)\.scale_shift_table$":
            r"transformer_blocks.\1.scale_shift_table",

            # Time embedding mappings
            r"^time_embed\.emb\.timestep_embedder\.linear_1\.(weight|bias)$":
            r"time_embed.emb.mlp.fc_in.\1",
            r"^time_embed\.emb\.timestep_embedder\.linear_2\.(weight|bias)$":
            r"time_embed.emb.mlp.fc_out.\1",

            # Caption projection mappings
            r"^caption_projection\.linear_1\.(weight|bias)$":
            r"caption_projection.fc_in.\1",
            r"^caption_projection\.linear_2\.(weight|bias)$":
            r"caption_projection.fc_out.\1",

            # Output normalization (FP32LayerNorm)
            r"^norm_out\.weight$": r"norm_out.weight",

            # Global scale-shift table
            r"^scale_shift_table$": r"scale_shift_table",
        })

    num_attention_heads: int = 32
    attention_head_dim: int = 64
    in_channels: int = 128
    out_channels: int | None = 128
    num_layers: int = 28
    dropout: float = 0.0
    patch_size: int = 1
    patch_size_t: int = 1
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-6
    activation_fn: str = "gelu-approximate"
    attention_bias: bool = True
    attention_out_bias: bool = True
    caption_channels: int | list[int] | tuple[int, ...] | None = 4096
    cross_attention_dim: int = 2048
    qk_norm: str = "rms_norm_across_heads"
    attention_type: str | None = "torch"
    use_additional_conditions: bool | None = False
    exclude_lora_layers: list[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.out_channels = self.in_channels if self.out_channels is None else self.out_channels
        self.num_channels_latents = self.out_channels


@dataclass
class LTXVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LTXVideoArchConfig)
    prefix: str = "LTXVideo"
