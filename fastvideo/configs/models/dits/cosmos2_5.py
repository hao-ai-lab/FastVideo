# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_transformer_blocks(n: str, m) -> bool:
    return "transformer_blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class Cosmos25ArchConfig(DiTArchConfig):
    """Configuration for Cosmos 2.5 architecture (MiniTrainDIT)."""

    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_transformer_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Remove "net." prefix and map official structure to FastVideo
            # Patch embedding: net.x_embedder.proj.1.weight -> patch_embed.proj.weight
            r"^net\.x_embedder\.proj\.1\.(.*)$":
            r"patch_embed.proj.\1",

            # Time embedding: net.t_embedder.1.linear_1.weight -> time_embed.t_embedder.linear_1.weight
            r"^net\.t_embedder\.1\.linear_1\.(.*)$":
            r"time_embed.t_embedder.linear_1.\1",
            r"^net\.t_embedder\.1\.linear_2\.(.*)$":
            r"time_embed.t_embedder.linear_2.\1",
            # Time embedding norm: net.t_embedding_norm.weight -> time_embed.norm.weight
            # Note: This also handles _extra_state if present
            r"^net\.t_embedding_norm\.(.*)$":
            r"time_embed.norm.\1",

            # Cross-attention projection (optional): net.crossattn_proj.0.weight -> crossattn_proj.0.weight
            r"^net\.crossattn_proj\.0\.weight$":
            r"crossattn_proj.0.weight",
            r"^net\.crossattn_proj\.0\.bias$":
            r"crossattn_proj.0.bias",

            # Transformer blocks: net.blocks.N -> transformer_blocks.N
            # Self-attention (self_attn -> attn1)
            r"^net\.blocks\.(\d+)\.self_attn\.q_proj\.(.*)$":
            r"transformer_blocks.\1.attn1.to_q.\2",
            r"^net\.blocks\.(\d+)\.self_attn\.k_proj\.(.*)$":
            r"transformer_blocks.\1.attn1.to_k.\2",
            r"^net\.blocks\.(\d+)\.self_attn\.v_proj\.(.*)$":
            r"transformer_blocks.\1.attn1.to_v.\2",
            r"^net\.blocks\.(\d+)\.self_attn\.output_proj\.(.*)$":
            r"transformer_blocks.\1.attn1.to_out.\2",
            r"^net\.blocks\.(\d+)\.self_attn\.q_norm\.weight$":
            r"transformer_blocks.\1.attn1.norm_q.weight",
            r"^net\.blocks\.(\d+)\.self_attn\.k_norm\.weight$":
            r"transformer_blocks.\1.attn1.norm_k.weight",
            # RMSNorm _extra_state keys (internal PyTorch state, will be recomputed automatically)
            r"^net\.blocks\.(\d+)\.self_attn\.q_norm\._extra_state$":
            r"transformer_blocks.\1.attn1.norm_q._extra_state",
            r"^net\.blocks\.(\d+)\.self_attn\.k_norm\._extra_state$":
            r"transformer_blocks.\1.attn1.norm_k._extra_state",

            # Cross-attention (cross_attn -> attn2)
            r"^net\.blocks\.(\d+)\.cross_attn\.q_proj\.(.*)$":
            r"transformer_blocks.\1.attn2.to_q.\2",
            r"^net\.blocks\.(\d+)\.cross_attn\.k_proj\.(.*)$":
            r"transformer_blocks.\1.attn2.to_k.\2",
            r"^net\.blocks\.(\d+)\.cross_attn\.v_proj\.(.*)$":
            r"transformer_blocks.\1.attn2.to_v.\2",
            r"^net\.blocks\.(\d+)\.cross_attn\.output_proj\.(.*)$":
            r"transformer_blocks.\1.attn2.to_out.\2",
            r"^net\.blocks\.(\d+)\.cross_attn\.q_norm\.weight$":
            r"transformer_blocks.\1.attn2.norm_q.weight",
            r"^net\.blocks\.(\d+)\.cross_attn\.k_norm\.weight$":
            r"transformer_blocks.\1.attn2.norm_k.weight",
            # RMSNorm _extra_state keys for cross-attention
            r"^net\.blocks\.(\d+)\.cross_attn\.q_norm\._extra_state$":
            r"transformer_blocks.\1.attn2.norm_q._extra_state",
            r"^net\.blocks\.(\d+)\.cross_attn\.k_norm\._extra_state$":
            r"transformer_blocks.\1.attn2.norm_k._extra_state",

            # MLP: net.blocks.N.mlp.layer1 -> transformer_blocks.N.mlp.fc_in
            r"^net\.blocks\.(\d+)\.mlp\.layer1\.(.*)$":
            r"transformer_blocks.\1.mlp.fc_in.\2",
            r"^net\.blocks\.(\d+)\.mlp\.layer2\.(.*)$":
            r"transformer_blocks.\1.mlp.fc_out.\2",

            # AdaLN-LoRA modulations: net.blocks.N.adaln_modulation_* -> transformer_blocks.N.adaln_modulation_*
            # These are now at the block level, not inside norm layers
            r"^net\.blocks\.(\d+)\.adaln_modulation_self_attn\.1\.(.*)$":
            r"transformer_blocks.\1.adaln_modulation_self_attn.1.\2",
            r"^net\.blocks\.(\d+)\.adaln_modulation_self_attn\.2\.(.*)$":
            r"transformer_blocks.\1.adaln_modulation_self_attn.2.\2",
            r"^net\.blocks\.(\d+)\.adaln_modulation_cross_attn\.1\.(.*)$":
            r"transformer_blocks.\1.adaln_modulation_cross_attn.1.\2",
            r"^net\.blocks\.(\d+)\.adaln_modulation_cross_attn\.2\.(.*)$":
            r"transformer_blocks.\1.adaln_modulation_cross_attn.2.\2",
            r"^net\.blocks\.(\d+)\.adaln_modulation_mlp\.1\.(.*)$":
            r"transformer_blocks.\1.adaln_modulation_mlp.1.\2",
            r"^net\.blocks\.(\d+)\.adaln_modulation_mlp\.2\.(.*)$":
            r"transformer_blocks.\1.adaln_modulation_mlp.2.\2",

            # Layer norms: net.blocks.N.layer_norm_* -> transformer_blocks.N.norm*.norm
            r"^net\.blocks\.(\d+)\.layer_norm_self_attn\._extra_state$":
            r"transformer_blocks.\1.norm1.norm._extra_state",
            r"^net\.blocks\.(\d+)\.layer_norm_cross_attn\._extra_state$":
            r"transformer_blocks.\1.norm2.norm._extra_state",
            r"^net\.blocks\.(\d+)\.layer_norm_mlp\._extra_state$":
            r"transformer_blocks.\1.norm3.norm._extra_state",

            # Final layer: net.final_layer.linear -> final_layer.proj_out
            r"^net\.final_layer\.linear\.(.*)$":
            r"final_layer.proj_out.\1",
            # Final layer AdaLN-LoRA: net.final_layer.adaln_modulation -> final_layer.linear_*
            r"^net\.final_layer\.adaln_modulation\.1\.(.*)$":
            r"final_layer.linear_1.\1",
            r"^net\.final_layer\.adaln_modulation\.2\.(.*)$":
            r"final_layer.linear_2.\1",

            # Note: The following keys from official checkpoint are NOT mapped and can be safely ignored:
            # - net.pos_embedder.* (seq, dim_spatial_range, dim_temporal_range) - These are computed dynamically
            #   in FastVideo's Cosmos25RotaryPosEmbed forward() method, so they don't need to be loaded.
            # - net.accum_* keys (training metadata) - These are skipped during checkpoint loading.
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
            r"^transformer_blocks\.(\d+)\.mlp\.(.*)$":
            r"transformer_blocks.\1.mlp.\2",
        })

    # Cosmos 2.5 specific config parameters
    in_channels: int = 16
    out_channels: int = 16
    num_attention_heads: int = 16
    attention_head_dim: int = 128  # 2048 / 16
    num_layers: int = 28
    mlp_ratio: float = 4.0
    text_embed_dim: int = 1024
    adaln_lora_dim: int = 256
    use_adaln_lora: bool = True
    max_size: tuple[int, int, int] = (128, 240, 240)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    rope_scale: tuple[float, float, float] = (1.0, 3.0, 3.0)  # T, H, W scaling
    concat_padding_mask: bool = True
    extra_pos_embed_type: str | None = None  # "learnable" or None
    # Note: Official checkpoint has use_crossattn_projection=True with 100K-dim input from Qwen 7B.
    # When enabled, must provide 100,352-dim embeddings to match the projection layer in checkpoint.
    use_crossattn_projection: bool = False
    crossattn_proj_in_channels: int = 100352  # Qwen 7B embedding dimension
    rope_enable_fps_modulation: bool = True
    qk_norm: str = "rms_norm"
    eps: float = 1e-6
    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.in_channels


@dataclass
class Cosmos25VideoConfig(DiTConfig):
    """Configuration for Cosmos 2.5 video generation model."""
    arch_config: DiTArchConfig = field(default_factory=Cosmos25ArchConfig)
    prefix: str = "Cosmos25"
