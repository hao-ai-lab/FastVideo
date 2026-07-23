# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 VFM Transformer FastVideo dataclass configs.

Architecture is 1:1 with the published ``nvidia/Cosmos3-Nano`` checkpoint
(``transformer/config.json``; class ``Cosmos3OmniTransformer`` / framework
``Cosmos3VFMNetwork``). Field values match that config so the FastVideo native
DiT builds a parameter tree matching the checkpoint's state-dict surface
(814 tensors / 44 patterns, validated 2026-06-06).

Reference of record: ``cosmos-framework`` (NVIDIA). The checkpoint is a single
``layers`` ModuleList of dual-pathway (understanding/text + generation/vision)
decoder blocks; per layer: ``self_attn`` with und (``to_{q,k,v}``/``to_out``)
and gen (``add_{q,k,v}_proj``/``to_add_out``) projections + QK-norms, plus
``mlp`` (und) and ``mlp_moe_gen`` (gen), and four RMSNorms. Top level adds
``embed_tokens``/``norm``/``norm_moe_gen``/``lm_head``/``proj_in``/``proj_out``/
``time_embedder`` and dormant ``action_*``/``audio_*`` heads. The checkpoint
remap lives in ``scripts/checkpoint_conversion/cosmos3_convert.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_cosmos3_transformer_block(name: str, module) -> bool:
    """FSDP shard boundary: the dual-pathway decoder blocks ``layers.{i}``."""
    del module
    parts = name.split(".")
    return "layers" in parts and parts[-1].isdigit()


@dataclass
class Cosmos3ArchConfig(DiTArchConfig):
    """Architecture config for the Cosmos3 omni DiT (Cosmos3-Nano).

    1:1 with ``transformer/config.json``. The action/sound heads ship in the
    checkpoint, so they are constructed for strict-load parity even though the
    PR1 video path (T2V/I2V/T2I) leaves them dormant.
    """

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_cosmos3_transformer_block])

    # Conversion is owned by scripts/checkpoint_conversion/cosmos3_convert.py;
    # the native module tree is the source of truth for parameter names.
    param_names_mapping: dict = field(default_factory=dict)

    # ---- Backbone (Qwen3-VL-text) ----
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA (4 query groups)
    head_dim: int = 128
    intermediate_size: int = 12288
    hidden_act: str = "silu"
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    qk_norm_for_diffusion: bool = True
    qk_norm_for_text: bool = True
    use_moe: bool = True  # dual-pathway weights; sparse routing unused
    joint_attn_implementation: str = "two_way"
    freeze_und: bool = False

    # ---- Position embedding (unified 3D MRoPE) ----
    position_embedding_type: str = "unified_3d_mrope"
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    mrope_interleaved: bool = True
    unified_3d_mrope_reset_spatial_ids: bool = True
    temporal_modality_margin: int = 15000  # unified_3d_mrope_temporal_modality_margin

    # ---- VAE / patch geometry ----
    latent_patch_size: int = 2
    latent_channel: int = 48
    patch_latent_dim: int = 192  # latent_patch_size**2 * latent_channel

    # ---- Diffusion conditioning ----
    timestep_scale: float = 0.001

    # ---- Temporal / FPS modulation ----
    base_fps: float = 24.0
    temporal_compression_factor: int = 4
    enable_fps_modulation: bool = True
    video_temporal_causal: bool = False

    # ---- Action generation head (dormant in PR1 video path) ----
    action_gen: bool = True
    action_dim: int = 64
    max_action_dim: int = 64
    num_embodiment_domains: int = 32

    # ---- Sound generation head (dormant in PR1 video path) ----
    sound_gen: bool = True
    sound_dim: int = 64
    sound_latent_fps: float = 25.0
    temporal_compression_factor_sound: int = 1

    # ---- BaseDiT bookkeeping ----
    in_channels: int = 48
    out_channels: int = 48

    def __post_init__(self) -> None:
        super().__post_init__()
        # Video DiT contract: latent channels == VAE z_dim.
        self.num_channels_latents = self.latent_channel
        if not self.out_channels:
            self.out_channels = self.in_channels
        # Derived: patchify packs latent_patch_size**2 spatial patches * channels.
        self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel


@dataclass
class Cosmos3VideoConfig(DiTConfig):
    """Pipeline-level Cosmos3 DiT config (T2V / I2V / T2I share this surface)."""

    arch_config: DiTArchConfig = field(default_factory=Cosmos3ArchConfig)
    prefix: str = "Cosmos3"
