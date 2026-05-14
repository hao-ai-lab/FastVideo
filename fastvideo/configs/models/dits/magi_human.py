# SPDX-License-Identifier: Apache-2.0
"""Architecture / model config for the daVinci-MagiHuman DiT.

The MagiHuman base DiT is a 15B-parameter single-stream transformer that
jointly denoises video, audio, and text tokens in one flat sequence. Layout
details verified against GAIR/daVinci-MagiHuman's base/ shards (2026-04-24).

This file captures only configuration. The module implementation lives in
fastvideo/models/dits/magi_human.py and the pipeline wiring in
fastvideo/pipelines/basic/magi_human/.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def _is_block_layer(n: str, m) -> bool:
    # Match "block.layers.<idx>" — the FSDP shard boundary for MagiHuman.
    parts = n.split(".")
    return (len(parts) >= 3 and parts[0] == "block" and parts[1] == "layers" and str.isdigit(parts[2]))


@dataclass
class MagiHumanArchConfig(DiTArchConfig):
    """MagiHuman base DiT architecture constants.

    **Scope contract:** fields here must match the `transformer/config.json`
    emitted by `scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py`
    1:1, and both are sourced from the upstream Python reference
    `inference/common/config.py::ModelConfig` (the HF root `config.json`
    is empty so the Python source is canonical). Pipeline-level knobs
    (VAE stride, fps, num_inference_steps, CFG scales, flow_shift,
    t5_gemma_target_length) and data-proxy knobs (coords_style,
    frame_receptive_field, ref_audio_offset, text_offset) live on
    `MagiHumanBaseConfig`, NOT here.

    `param_names_mapping` is intentionally empty: the FastVideo implementation
    keeps the same module tree as the reference (`adapter.*`,
    `block.layers.<i>.*`, `final_linear_{video,audio}.*`,
    `final_norm_{video,audio}.*`), so converted weights load directly.
    """

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_block_layer])

    # No renames needed — the FastVideo module mirrors the reference names.
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    # --- transformer shape ---
    num_layers: int = 40
    hidden_size: int = 5120
    head_dim: int = 128
    num_query_groups: int = 8  # num_heads_kv (GQA)

    # --- modality channels ---
    # video_in_channels = z_dim (48) * patch_size product (1*2*2=4), so the
    # embedder receives 192 per token. text_in_channels is T5Gemma-9B's
    # encoder hidden size.
    video_in_channels: int = 192
    audio_in_channels: int = 64
    text_in_channels: int = 3584

    # --- block-level architecture switches ---
    # Sandwich MoE: first and last 4 layers have per-modality experts
    # (video/audio/text), middle layers share a single set of weights.
    mm_layers: tuple[int, ...] = (0, 1, 2, 3, 36, 37, 38, 39)
    local_attn_layers: tuple[int, ...] = ()
    gelu7_layers: tuple[int, ...] = (0, 1, 2, 3)
    post_norm_layers: tuple[int, ...] = ()
    enable_attn_gating: bool = True
    activation_type: str = "swiglu7"

    # --- DiT patching (upstream `ModelConfig`-equivalent; NOT the VAE
    #     stride, which is pipeline-level). ---
    patch_size: tuple[int, int, int] = (1, 2, 2)
    spatial_rope_interpolation: str = "extra"

    # --- TReAD (token routing + early drop). Flattened from the upstream
    #     nested `tread_config` dict so it round-trips through
    #     `update_model_arch` cleanly. ---
    tread_selection_rate: float = 0.5
    tread_start_layer_idx: int = 2
    tread_end_layer_idx: int = 25

    # --- derived fields (populated in __post_init__) ---
    num_attention_heads: int = 0  # hidden_size / head_dim
    num_heads_kv: int = 0  # == num_query_groups
    in_channels: int = 0  # mirror of video_in_channels (FastVideo contract)
    out_channels: int = 0  # mirror of video_in_channels

    def __post_init__(self) -> None:
        super().__post_init__()
        self.num_attention_heads = self.hidden_size // self.head_dim
        self.num_heads_kv = self.num_query_groups
        self.in_channels = self.video_in_channels
        self.out_channels = self.video_in_channels
        # num_channels_latents is the VAE latent z_dim (48 for Wan 2.2 TI2V-5B).
        # We don't declare z_dim on the arch config (it's a VAE property),
        # but we still set num_channels_latents for the BaseDiT contract.
        self.num_channels_latents = 48


@dataclass
class MagiHumanVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=MagiHumanArchConfig)

    prefix: str = "magi_human"
