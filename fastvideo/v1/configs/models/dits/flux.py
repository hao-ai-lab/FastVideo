from dataclasses import dataclass, field
from typing import Optional, Tuple

from fastvideo.v1.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class FluxImageArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    _param_names_mapping: dict = field(
        default_factory=lambda: {
            # 1. context_embedder to txt_in mapping:
            r"^context_embedder\.(.*)$":
            r"txt_in.\1",

            # 2. x_embedder to img_in mapping:
            r"^x_embedder\.(.*)$":
            r"img_in.\1",

            # 3. Top-level time_text_embed mappings:
            r"^time_text_embed\.timestep_embedder\.linear_1\.(.*)$":
            r"time_in.mlp.fc_in.\1",
            r"^time_text_embed\.timestep_embedder\.linear_2\.(.*)$":
            r"time_in.mlp.fc_out.\1",
            r"^time_text_embed\.guidance_embedder\.linear_1\.(.*)$":
            r"guidance_in.mlp.fc_in.\1",
            r"^time_text_embed\.guidance_embedder\.linear_2\.(.*)$":
            r"guidance_in.mlp.fc_out.\1",
            r"^time_text_embed\.text_embedder\.linear_1\.(.*)$":
            r"txt2_in.fc_in.\1",
            r"^time_text_embed\.text_embedder\.linear_2\.(.*)$":
            r"txt2_in.fc_out.\1",

            # 4. transformer_blocks mapping:
            r"^transformer_blocks\.(\d+)\.norm1\.linear\.(.*)$":
            r"double_blocks.\1.img_mod.linear.\2",
            r"^transformer_blocks\.(\d+)\.norm1_context\.linear\.(.*)$":
            r"double_blocks.\1.txt_mod.linear.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_q\.(.*)$":
            r"double_blocks.\1.img_attn_q_norm.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_k\.(.*)$":
            r"double_blocks.\1.img_attn_k_norm.\2",
            r"^transformer_blocks\.(\d+)\.attn\.to_q\.(.*)$":
            (r"double_blocks.\1.img_attn_qkv.\2", 0, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_k\.(.*)$":
            (r"double_blocks.\1.img_attn_qkv.\2", 1, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_v\.(.*)$":
            (r"double_blocks.\1.img_attn_qkv.\2", 2, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_q_proj\.(.*)$":
            (r"double_blocks.\1.txt_attn_qkv.\2", 0, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_k_proj\.(.*)$":
            (r"double_blocks.\1.txt_attn_qkv.\2", 1, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_v_proj\.(.*)$":
            (r"double_blocks.\1.txt_attn_qkv.\2", 2, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.(.*)$":
            r"double_blocks.\1.img_attn_proj.\2",
            # Corrected: merge attn.to_add_out into the main projection.
            r"^transformer_blocks\.(\d+)\.attn\.to_add_out\.(.*)$":
            r"double_blocks.\1.txt_attn_proj.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_added_q\.(.*)$":
            r"double_blocks.\1.txt_attn_q_norm.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_added_k\.(.*)$":
            r"double_blocks.\1.txt_attn_k_norm.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.0(?:\.proj)?\.(.*)$":
            r"double_blocks.\1.img_mlp.fc_in.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.2(?:\.proj)?\.(.*)$":
            r"double_blocks.\1.img_mlp.fc_out.\2",
            r"^transformer_blocks\.(\d+)\.ff_context\.net\.0(?:\.proj)?\.(.*)$":
            r"double_blocks.\1.txt_mlp.fc_in.\2",
            r"^transformer_blocks\.(\d+)\.ff_context\.net\.2(?:\.proj)?\.(.*)$":
            r"double_blocks.\1.txt_mlp.fc_out.\2",

            # 5. single_transformer_blocks mapping:
            r"^single_transformer_blocks\.(\d+)\.attn\.norm_q\.(.*)$":
            r"single_blocks.\1.q_norm.\2",
            r"^single_transformer_blocks\.(\d+)\.attn\.norm_k\.(.*)$":
            r"single_blocks.\1.k_norm.\2",
            r"^single_transformer_blocks\.(\d+)\.attn\.to_q\.(.*)$":
            (r"single_blocks.\1.linear1.\2", 0, 4),
            r"^single_transformer_blocks\.(\d+)\.attn\.to_k\.(.*)$":
            (r"single_blocks.\1.linear1.\2", 1, 4),
            r"^single_transformer_blocks\.(\d+)\.attn\.to_v\.(.*)$":
            (r"single_blocks.\1.linear1.\2", 2, 4),
            r"^single_transformer_blocks\.(\d+)\.proj_mlp\.(.*)$":
            (r"single_blocks.\1.linear1.\2", 3, 4),
            # Corrected: map proj_out to modulation.linear rather than a separate proj_out branch.
            r"^single_transformer_blocks\.(\d+)\.proj_out\.(.*)$":
            r"single_blocks.\1.linear2.\2",
            r"^single_transformer_blocks\.(\d+)\.norm\.linear\.(.*)$":
            r"single_blocks.\1.modulation.linear.\2",

            # 6. Final layers mapping:
            r"^norm_out\.linear\.(.*)$":
            r"final_layer.adaLN_modulation.linear.\1",
            r"^proj_out\.(.*)$":
            r"final_layer.linear.\1",
        })

    patch_size: int = 1
    in_channels: int = 64
    out_channels: Optional[int] = None
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = False
    axes_dims_rope: Tuple[int] = (16, 56, 56)
    repo_theta: int = 10000

    def __post_init__(self):
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.in_channels


@dataclass
class FluxImageConfig(DiTConfig):
    arch_config: DiTArchConfig = FluxImageArchConfig()

    prefix: str = "Flux"
