# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 VFM Transformer FastVideo dataclass configs (Phase 2b.1 scaffold).

Mirrors the architectural defaults of
``vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py::Cosmos3VFMTransformer``
(see lines 910-977 of the vllm-omni reference at HEAD ``8536f5b1421f``).

Phase 2b.1 ships only the dataclass surface and skeleton tensor shapes that
``Cosmos3VFMTransformer.__init__`` needs to build a parameter-named module
tree. The ``param_names_mapping`` table is left empty here; the checkpoint
remap lives in ``scripts/checkpoint_conversion/cosmos3_convert.py`` (Phase 5)
and the FastVideo pipeline's ``_remap_ckpt_key`` (Phase 2b.2+).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_cosmos3_transformer_block(name: str, module) -> bool:
    """FSDP shard boundary for Cosmos3 UND/GEN decoder layers.

    Mirrors ``Cosmos3VFMTransformer._is_transformer_block`` (reference line
    870). Matches ``gen_layers.{i}`` and ``language_model.layers.{i}`` where
    ``{i}`` is a non-negative integer.
    """
    del module
    if "gen_layers" not in name and "language_model.layers" not in name:
        return False
    return name.split(".")[-1].isdigit()


@dataclass
class Cosmos3ArchConfig(DiTArchConfig):
    """Architecture config for Cosmos3 VFM Transformer (UND + GEN pathways).

    Defaults match the upstream Cosmos3-Nano constructor defaults at
    ``transformer_cosmos3.py`` lines 910-934.
    """

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_cosmos3_transformer_block])

    # ``param_names_mapping`` is intentionally empty for Phase 2b.1. The
    # canonical Cosmos3 checkpoint remap is owned by the pipeline's
    # ``_remap_ckpt_key`` (vllm-omni reference at pipeline_cosmos3.py:319-409)
    # and is ported to FastVideo in a later phase (the conversion script under
    # ``scripts/checkpoint_conversion/``). Leaving this empty makes the DiT
    # module tree the single source of truth for parameter names during
    # parity-test development.
    param_names_mapping: dict = field(default_factory=dict)

    # ---- Backbone ----
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])

    # ---- VAE / patch geometry ----
    latent_patch_size: int = 2
    latent_channel: int = 48

    # ---- Diffusion conditioning ----
    timestep_scale: float = 0.001

    # ---- Temporal modulation ----
    base_fps: float = 24.0
    temporal_compression_factor: int = 4
    enable_fps_modulation: bool = True
    temporal_modality_margin: int = 15000

    # ---- BaseDiT bookkeeping ----
    in_channels: int = 48
    out_channels: int = 48

    def __post_init__(self) -> None:
        super().__post_init__()
        # Mirror the BaseDiT contract: ``num_channels_latents`` matches the
        # VAE latent channel count for video DiTs.
        self.num_channels_latents = self.latent_channel
        if not self.out_channels:
            self.out_channels = self.in_channels


@dataclass
class Cosmos3VideoConfig(DiTConfig):
    """Pipeline-level Cosmos3 DiT config (T2V / I2V / T2I share this surface)."""

    arch_config: DiTArchConfig = field(default_factory=Cosmos3ArchConfig)
    prefix: str = "Cosmos3"
