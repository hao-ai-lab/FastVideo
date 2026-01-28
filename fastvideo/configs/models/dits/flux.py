# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Tuple

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class FluxArchConfig(DiTArchConfig):
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^norm_out\.linear\.(weight|bias)$":
            r"final_layer.linear.\1",
            r"^norm_out\.adaLN_modulation\.(weight|bias)$":
            r"final_layer.adaLN_modulation.1.\1",
            r"^proj_out\.(weight|bias)$":
            r"final_layer.linear.\1",
            r"^single_transformer_blocks\.(\d+)\.attn\.norm_q\.weight$":
            r"single_blocks.\1.norm.query_norm.weight",
            r"^single_transformer_blocks\.(\d+)\.attn\.norm_k\.weight$":
            r"single_blocks.\1.norm.key_norm.weight",
            r"^single_transformer_blocks\.(\d+)\.norm\.linear\.(weight|bias)$":
            r"single_blocks.\1.modulation.lin.\2",
            r"^single_transformer_blocks\.(\d+)\.attn\.to_q\.(weight|bias)$":
            (r"single_blocks.\1.linear1.\2", 0, 4),
            r"^single_transformer_blocks\.(\d+)\.attn\.to_k\.(weight|bias)$":
            (r"single_blocks.\1.linear1.\2", 1, 4),
            r"^single_transformer_blocks\.(\d+)\.attn\.to_v\.(weight|bias)$":
            (r"single_blocks.\1.linear1.\2", 2, 4),
            r"^single_transformer_blocks\.(\d+)\.proj_mlp\.(weight|bias)$":
            (r"single_blocks.\1.linear1.\2", 3, 4),
            r"^single_transformer_blocks\.(\d+)\.proj_out\.(weight|bias)$":
            r"single_blocks.\1.linear2.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_q\.weight$":
            r"double_blocks.\1.img_attn.norm.query_norm.weight",
            r"^transformer_blocks\.(\d+)\.attn\.norm_k\.weight$":
            r"double_blocks.\1.img_attn.norm.key_norm.weight",
            r"^transformer_blocks\.(\d+)\.attn\.norm_added_q\.weight$":
            r"double_blocks.\1.txt_attn.norm.query_norm.weight",
            r"^transformer_blocks\.(\d+)\.attn\.norm_added_k\.weight$":
            r"double_blocks.\1.txt_attn.norm.key_norm.weight",
            r"^transformer_blocks\.(\d+)\.attn\.to_q\.(weight|bias)$":
            (r"double_blocks.\1.img_attn.qkv.\2", 0, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_k\.(weight|bias)$":
            (r"double_blocks.\1.img_attn.qkv.\2", 1, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_v\.(weight|bias)$":
            (r"double_blocks.\1.img_attn.qkv.\2", 2, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_q_proj\.(weight|bias)$":
            (r"double_blocks.\1.txt_attn.qkv.\2", 0, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_k_proj\.(weight|bias)$":
            (r"double_blocks.\1.txt_attn.qkv.\2", 1, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_v_proj\.(weight|bias)$":
            (r"double_blocks.\1.txt_attn.qkv.\2", 2, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.(weight|bias)$":
            r"double_blocks.\1.img_attn.proj.\2",
            r"^transformer_blocks\.(\d+)\.attn\.to_add_out\.(weight|bias)$":
            r"double_blocks.\1.txt_attn.proj.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.(weight|bias)$":
            r"double_blocks.\1.img_mlp.fc_in.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.2\.(weight|bias)$":
            r"double_blocks.\1.img_mlp.fc_out.\2",
            r"^transformer_blocks\.(\d+)\.ff_context\.net\.0\.proj\.(weight|bias)$":
            r"double_blocks.\1.txt_mlp.fc_in.\2",
            r"^transformer_blocks\.(\d+)\.ff_context\.net\.2\.(weight|bias)$":
            r"double_blocks.\1.txt_mlp.fc_out.\2",
            r"^transformer_blocks\.(\d+)\.norm1\.linear\.(weight|bias)$":
            r"double_blocks.\1.img_mod.lin.\2",
            r"^transformer_blocks\.(\d+)\.norm1_context\.linear\.(weight|bias)$":
            r"double_blocks.\1.txt_mod.lin.\2",
            r"^context_embedder\.(weight|bias)$":
            r"txt_in.\1",
            r"^time_text_embed\.time_embedder\.linear_1\.(weight|bias)$":
            r"time_in.in_layer.\1",
            r"^time_text_embed\.time_embedder\.linear_2\.(weight|bias)$":
            r"time_in.out_layer.\1",
            r"^time_text_embed\.timestep_embedder\.linear_1\.(weight|bias)$":
            r"time_in.in_layer.\1",
            r"^time_text_embed\.timestep_embedder\.linear_2\.(weight|bias)$":
            r"time_in.out_layer.\1",
            r"^time_text_embed\.text_embedder\.linear_1\.(weight|bias)$":
            r"vector_in.in_layer.\1",
            r"^time_text_embed\.text_embedder\.linear_2\.(weight|bias)$":
            r"vector_in.out_layer.\1",
            r"^time_text_embed\.guidance_embedder\.linear_1\.(weight|bias)$":
            r"guidance_in.in_layer.\1",
            r"^time_text_embed\.guidance_embedder\.linear_2\.(weight|bias)$":
            r"guidance_in.out_layer.\1",
            r"^x_embedder\.(weight|bias)$":
            r"img_in.\1",
        })
    # Diffusers config fields
    attention_head_dim: int = 128
    guidance_embeds: bool = True
    in_channels: int = 64
    joint_attention_dim: int = 4096
    num_attention_heads: int = 24
    num_layers: int = 19
    num_single_layers: int = 38
    patch_size: int = 1
    pooled_projection_dim: int = 768
    qkv_bias: bool = True
    _name_or_path: str | None = None

    # FastVideo-specific defaults
    mlp_ratio: float = 4.0
    rope_axes_dim: Tuple[int, int] = (64, 64)
    rope_theta: float = 10000.0
    out_channels: int | None = None

    def __post_init__(self) -> None:
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        if sum(self.rope_axes_dim) != self.attention_head_dim:
            half = self.attention_head_dim // 2
            self.rope_axes_dim = (half, self.attention_head_dim - half)
        if self.out_channels is None:
            self.out_channels = self.in_channels
        self.num_channels_latents = self.out_channels


@dataclass
class FluxConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=FluxArchConfig)
    prefix: str = "flux"
