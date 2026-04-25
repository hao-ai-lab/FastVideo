# SPDX-License-Identifier: Apache-2.0
# Config for daVinci-MagiHuman DiT port.
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


@dataclass
class DaVinciMagiHumanArchConfig(DiTArchConfig):
    """Architecture configuration for daVinci-MagiHuman.

    Derived from ModelConfig in inference/common/config.py.
    Verified against the official checkpoint structure.
    """

    # ── Model dimensions ──────────────────────────────────────────────────────
    hidden_size: int = 5120
    head_dim: int = 128
    # Computed: hidden_size // head_dim = 40
    num_heads_q: int = 40
    # Grouped-query attention: num_query_groups = 8
    num_heads_kv: int = 8
    num_layers: int = 40

    # ── Latent / channel dimensions ───────────────────────────────────────────
    # z_dim = VAE latent channels (Wan2.2-TI2V-5B)
    z_dim: int = 48
    # video_in_channels = z_dim * spatial_patch (2×2 = 4)
    video_in_channels: int = 192
    audio_in_channels: int = 64
    # t5gemma-9b hidden size
    text_in_channels: int = 3584

    # ── Sandwich architecture ─────────────────────────────────────────────────
    # First 4 and last 4 layers: per-modality MoE projections (num_experts=3).
    # Middle 32 layers (4-35): shared projections across all modalities.
    mm_layers: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 36, 37, 38, 39])

    # ── Activation variants ───────────────────────────────────────────────────
    # Layers 0-3: GELU7 (non-gated); layers 4-39: SwiGLU7 (gated).
    gelu7_layers: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3])

    # ── Post-norm layers (empty by default, used for SR variant) ─────────────
    post_norm_layers: list[int] = field(default_factory=list)

    # ── Attention options ─────────────────────────────────────────────────────
    enable_attn_gating: bool = True

    # ── FastVideo BaseDiT required fields ─────────────────────────────────────
    # num_attention_heads mirrors num_heads_q for BaseDiT compatibility
    num_attention_heads: int = 40
    # num_channels_latents = z_dim (VAE latent channels, not post-patch)
    num_channels_latents: int = 48
    # in_channels / out_channels = video_in_channels (post-patch)
    in_channels: int = 192
    out_channels: int = 192

    # ── FSDP / compile sharding ───────────────────────────────────────────────
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: ["layers"])
    _compile_conditions: list = field(
        default_factory=lambda: ["layers"])

    # ── Attention backends ────────────────────────────────────────────────────
    # daVinci uses full self-attention; all standard backends are valid.
    # Set FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA for alignment tests.
    _supported_attention_backends: tuple = (
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.SAGE_ATTN,
    )

    # ── Checkpoint key mapping ────────────────────────────────────────────────
    # Maps official state_dict keys → FastVideo keys (applied in order).
    #
    # Official structure:            FastVideo structure:
    #   block.layers.N.*         →   layers.N.*
    #   adapter.video_embedder.* →   adapter.video_proj.*
    #   adapter.text_embedder.*  →   adapter.text_proj.*
    #   adapter.audio_embedder.* →   adapter.audio_proj.*
    #   final_linear_video.*     →   final_proj_video.*
    #   final_linear_audio.*     →   final_proj_audio.*
    #
    # All other keys are pass-through (e.g. final_norm_video.*, adapter.rope.*).
    param_names_mapping: dict = field(default_factory=lambda: {
        r"^block\.layers\.(\d+)\.(.*)$":    r"layers.\1.\2",
        r"^adapter\.video_embedder\.(.*)$": r"adapter.video_proj.\1",
        r"^adapter\.text_embedder\.(.*)$":  r"adapter.text_proj.\1",
        r"^adapter\.audio_embedder\.(.*)$": r"adapter.audio_proj.\1",
        r"^final_linear_video\.(.*)$":      r"final_proj_video.\1",
        r"^final_linear_audio\.(.*)$":      r"final_proj_audio.\1",
    })

    # reverse_param_names_mapping is populated automatically from the above
    # by the weight loading utilities; no need to set manually.


@dataclass
class DaVinciMagiHumanConfig(DiTConfig):
    """Top-level config wrapping DaVinciMagiHumanArchConfig."""

    arch_config: DaVinciMagiHumanArchConfig = field(
        default_factory=DaVinciMagiHumanArchConfig)
