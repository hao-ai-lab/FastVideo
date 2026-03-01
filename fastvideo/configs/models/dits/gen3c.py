# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_transformer_blocks(n: str, m) -> bool:
    return "transformer_blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class Gen3CArchConfig(DiTArchConfig):
    """Configuration for GEN3C architecture (VideoExtendGeneralDIT)."""

    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_transformer_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Official GEN3C checkpoint key naming to FastVideo mapping.
            # The official checkpoint uses nn.Sequential patterns like attn.to_q.0 (Linear)
            # and attn.to_q.1 (RMSNorm), and layer1/layer2 for MLP.
            #
            # Patch embedding: net.x_embedder.proj.1.weight -> patch_embed.proj.weight
            r"^net\.x_embedder\.proj\.1\.(.*)$":
            r"patch_embed.proj.\1",

            # Time embedding: net.t_embedder.1.linear_*.weight -> time_embed.t_embedder.linear_*.weight
            r"^net\.t_embedder\.0\.(.*)$":
            r"time_embed.time_proj.\1",
            r"^net\.t_embedder\.1\.linear_1\.(.*)$":
            r"time_embed.t_embedder.linear_1.\1",
            r"^net\.t_embedder\.1\.linear_2\.(.*)$":
            r"time_embed.t_embedder.linear_2.\1",

            # Augment sigma embedding (GEN3C-specific)
            r"^net\.augment_sigma_embedder\.0\.(.*)$":
            r"augment_sigma_embed.time_proj.\1",
            r"^net\.augment_sigma_embedder\.1\.linear_1\.(.*)$":
            r"augment_sigma_embed.t_embedder.linear_1.\1",
            r"^net\.augment_sigma_embedder\.1\.linear_2\.(.*)$":
            r"augment_sigma_embed.t_embedder.linear_2.\1",

            # Affine embedding norm: net.affline_norm.weight -> affine_norm.weight
            # Note: "affline" is a typo in the official GEN3C checkpoint (should be "affine")
            r"^net\.affline_norm\.(.*)$":
            r"affine_norm.\1",

            # Extra positional embeddings (learnable per-axis)
            r"^net\.extra_pos_embedder\.pos_emb_t$":
            r"learnable_pos_embed.pos_emb_t",
            r"^net\.extra_pos_embedder\.pos_emb_h$":
            r"learnable_pos_embed.pos_emb_h",
            r"^net\.extra_pos_embedder\.pos_emb_w$":
            r"learnable_pos_embed.pos_emb_w",

            # Transformer blocks: net.blocks.blockN -> transformer_blocks.N
            # Official uses: block.attn.to_q.0 (Linear), block.attn.to_q.1 (QK RMSNorm)
            #
            # Self-attention (block index 0)
            r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_q\.0\.(.*)$":
            r"transformer_blocks.\1.attn1.to_q.\2",
            r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_q\.1\.(.*)$":
            r"transformer_blocks.\1.attn1.norm_q.\2",
            r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_k\.0\.(.*)$":
            r"transformer_blocks.\1.attn1.to_k.\2",
            r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_k\.1\.(.*)$":
            r"transformer_blocks.\1.attn1.norm_k.\2",
            r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_v\.0\.(.*)$":
            r"transformer_blocks.\1.attn1.to_v.\2",
            r"^net\.blocks\.block(\d+)\.blocks\.0\.block\.attn\.to_out\.0\.(.*)$":
            r"transformer_blocks.\1.attn1.to_out.\2",
            # AdaLN modulation for self-attention
            r"^net\.blocks\.block(\d+)\.blocks\.0\.adaLN_modulation\.(.*)$":
            r"transformer_blocks.\1.adaln_modulation_self_attn.\2",

            # Cross-attention (block index 1)
            r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_q\.0\.(.*)$":
            r"transformer_blocks.\1.attn2.to_q.\2",
            r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_q\.1\.(.*)$":
            r"transformer_blocks.\1.attn2.norm_q.\2",
            r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_k\.0\.(.*)$":
            r"transformer_blocks.\1.attn2.to_k.\2",
            r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_k\.1\.(.*)$":
            r"transformer_blocks.\1.attn2.norm_k.\2",
            r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_v\.0\.(.*)$":
            r"transformer_blocks.\1.attn2.to_v.\2",
            r"^net\.blocks\.block(\d+)\.blocks\.1\.block\.attn\.to_out\.0\.(.*)$":
            r"transformer_blocks.\1.attn2.to_out.\2",
            # AdaLN modulation for cross-attention
            r"^net\.blocks\.block(\d+)\.blocks\.1\.adaLN_modulation\.(.*)$":
            r"transformer_blocks.\1.adaln_modulation_cross_attn.\2",

            # MLP (block index 2): layer1 -> fc_in, layer2 -> fc_out
            r"^net\.blocks\.block(\d+)\.blocks\.2\.block\.layer1\.(.*)$":
            r"transformer_blocks.\1.mlp.fc_in.\2",
            r"^net\.blocks\.block(\d+)\.blocks\.2\.block\.layer2\.(.*)$":
            r"transformer_blocks.\1.mlp.fc_out.\2",
            # AdaLN modulation for MLP
            r"^net\.blocks\.block(\d+)\.blocks\.2\.adaLN_modulation\.(.*)$":
            r"transformer_blocks.\1.adaln_modulation_mlp.\2",

            # Final layer: net.final_layer.linear -> final_layer.proj_out
            r"^net\.final_layer\.linear\.(.*)$":
            r"final_layer.proj_out.\1",
            # Final layer AdaLN: net.final_layer.adaLN_modulation -> final_layer.adaln_modulation
            r"^net\.final_layer\.adaLN_modulation\.(.*)$":
            r"final_layer.adaln_modulation.\1",

            # Note: The following keys from official checkpoint are NOT mapped and can be safely ignored:
            # - net.pos_embedder.* (rope position embeddings computed dynamically)
            # - net.accum_* keys (training metadata)
            # - logvar.* (training-only module, not used in inference)
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

    # GEN3C architecture parameters
    # Base VAE latent channels
    in_channels: int = 16
    out_channels: int = 16

    # Channels per 3D cache buffer: 16 (warped frame latent) + 16 (warped mask latent)
    CHANNELS_PER_BUFFER: int = 32

    # Number of 3D cache buffers
    frame_buffer_max: int = 2

    # Attention configuration (7B model: 32 heads x 128 dim = 4096 hidden)
    num_attention_heads: int = 32
    attention_head_dim: int = 128  # 4096 / 32
    num_layers: int = 28
    mlp_ratio: float = 4.0

    # Text encoder configuration
    text_embed_dim: int = 1024

    # AdaLN-LoRA configuration
    adaln_lora_dim: int = 256
    use_adaln_lora: bool = True

    # GEN3C-specific: augment sigma embedding for conditioning noise augmentation
    # Note: The official GEN3C-Cosmos-7B checkpoint was trained without this
    add_augment_sigma_embedding: bool = False

    # Position embedding configuration
    max_size: tuple[int, int, int] = (128, 240, 240)  # T, H, W
    patch_size: tuple[int, int, int] = (1, 2, 2)
    rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0)  # T, H, W scaling

    # GEN3C uses learnable positional embeddings in addition to RoPE
    extra_pos_embed_type: str = "learnable"

    # Padding mask handling
    concat_padding_mask: bool = True

    # Cross-attention projection (not used in GEN3C 7B)
    use_crossattn_projection: bool = False

    # RoPE FPS modulation
    rope_enable_fps_modulation: bool = True

    # QK normalization
    qk_norm: str = "rms_norm"
    eps: float = 1e-6

    # Affine embedding normalization
    affine_emb_norm: bool = True

    # Block format (THWBD for GEN3C compatibility)
    block_x_format: str = "THWBD"

    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.in_channels

        # Calculate total input channels for patch embedding:
        # - in_channels (16): VAE latent
        # - condition_video_input_mask (1): Binary mask for conditioning frames
        # - condition_video_pose (frame_buffer_max * 32): 3D cache buffers
        # - padding_mask (1 if concat_padding_mask): Padding mask
        self.buffer_channels = self.frame_buffer_max * self.CHANNELS_PER_BUFFER
        self.total_input_channels = (
            self.in_channels +  # 16: VAE latent
            1 +  # 1: condition_video_input_mask
            self.buffer_channels  # 64: 3D cache buffers (2 * 32)
        )
        # padding_mask is added in build_patch_embed if concat_padding_mask=True


@dataclass
class Gen3CVideoConfig(DiTConfig):
    """Configuration for GEN3C video generation model."""
    arch_config: DiTArchConfig = field(default_factory=Gen3CArchConfig)
    prefix: str = "Gen3C"
