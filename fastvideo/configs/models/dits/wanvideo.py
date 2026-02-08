# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class WanVideoArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Official Wan/LingBot checkpoint naming.
            r"^patch_embedding\.(.*)$":
            r"patch_embedding.proj.\1",
            r"^text_embedding\.0\.(.*)$":
            r"condition_embedder.text_embedder.fc_in.\1",
            r"^text_embedding\.2\.(.*)$":
            r"condition_embedder.text_embedder.fc_out.\1",
            r"^time_embedding\.0\.(.*)$":
            r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^time_embedding\.2\.(.*)$":
            r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^time_projection\.1\.(.*)$":
            r"condition_embedder.time_modulation.linear.\1",
            r"^head\.modulation$":
            r"scale_shift_table",
            r"^head\.head\.(.*)$":
            r"proj_out.\1",
            r"^blocks\.(\d+)\.modulation$":
            r"blocks.\1.scale_shift_table",
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$":
            r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$":
            r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$":
            r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$":
            r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$":
            r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$":
            r"blocks.\1.norm_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$":
            r"blocks.\1.attn2.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$":
            r"blocks.\1.attn2.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.k_img\.(.*)$":
            r"blocks.\1.attn2.add_k_proj.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$":
            r"blocks.\1.attn2.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.v_img\.(.*)$":
            r"blocks.\1.attn2.add_v_proj.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$":
            r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$":
            r"blocks.\1.attn2.norm_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$":
            r"blocks.\1.attn2.norm_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_q_img\.(.*)$":
            r"blocks.\1.attn2.norm_added_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_k_img\.(.*)$":
            r"blocks.\1.attn2.norm_added_k.\2",
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$":
            r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$":
            r"blocks.\1.ffn.fc_out.\2",
            r"^blocks\.(\d+)\.norm3\.(.*)$":
            r"blocks.\1.self_attn_residual_norm.norm.\2",

            # Diffusers-style naming.
            r"^condition_embedder\.text_embedder\.linear_1\.(.*)$":
            r"condition_embedder.text_embedder.fc_in.\1",
            r"^condition_embedder\.text_embedder\.linear_2\.(.*)$":
            r"condition_embedder.text_embedder.fc_out.\1",
            r"^condition_embedder\.time_embedder\.linear_1\.(.*)$":
            r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^condition_embedder\.time_embedder\.linear_2\.(.*)$":
            r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^condition_embedder\.time_proj\.(.*)$":
            r"condition_embedder.time_modulation.linear.\1",
            r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj\.(.*)$":
            r"condition_embedder.image_embedder.ff.fc_in.\1",
            r"^condition_embedder\.image_embedder\.ff\.net\.2\.(.*)$":
            r"condition_embedder.image_embedder.ff.fc_out.\1",
            r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$":
            r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$":
            r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$":
            r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$":
            r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$":
            r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$":
            r"blocks.\1.norm_k.\2",
            r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$":
            r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$":
            r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$":
            r"blocks.\1.ffn.fc_out.\2",
            r"^blocks\.(\d+)\.norm2\.(.*)$":
            r"blocks.\1.self_attn_residual_norm.norm.\2",
        })

    # Reverse mapping for saving checkpoints: custom -> hf
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    # Some LoRA adapters use the original official layer names instead of hf layer names,
    # so apply this before the param_names_mapping
    lora_param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.attn1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$":
            r"blocks.\1.attn1.to_out.0.\2",
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$":
            r"blocks.\1.attn2.to_out.0.\2",
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
        })

    patch_size: tuple[int, int, int] = (1, 2, 2)
    dim: int | None = None
    text_len = 512
    num_heads: int | None = None
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_dim: int | None = None
    in_channels: int = 16
    out_dim: int | None = None
    out_channels: int = 16
    model_type: str | None = None
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int | None = None
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: int | None = None
    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])

    # Wan MoE
    boundary_ratio: float | None = None

    # Causal Wan
    local_attn_size: int = -1  # Window size for temporal local attention (-1 indicates global attention)
    sink_size: int = 0  # Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
    num_frames_per_block: int = 3
    sliding_window_num_frames: int = 21

    def __post_init__(self):
        if self.num_heads is not None:
            self.num_attention_heads = self.num_heads
        if self.in_dim is not None:
            self.in_channels = self.in_dim
        if self.out_dim is not None:
            self.out_channels = self.out_dim
        if self.dim is not None and self.num_attention_heads > 0:
            self.attention_head_dim = self.dim // self.num_attention_heads
            self.hidden_size = self.dim
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class WanVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=WanVideoArchConfig)

    prefix: str = "Wan"
