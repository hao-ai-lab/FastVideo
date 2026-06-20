# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Transformer configuration for native FastVideo integration.
"""

from dataclasses import dataclass, field

from v2._vendor.configs.models.dits.base import DiTArchConfig, DiTConfig

import re


def is_ltx2_blocks(name: str, _module) -> bool:
    res = re.search(r"(?:^|\.)transformer_blocks\.\d+$", name) is not None
    return res


@dataclass
class LTX2VideoArchConfig(DiTArchConfig):
    """Architecture configuration for LTX-2 video transformer."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_ltx2_blocks])
    _compile_conditions: list = field(default_factory=lambda: [is_ltx2_blocks])

    # Parameter name mapping for weight conversion (hf/comfy -> FastVideo).
    # The ``to_gate_compress`` -> ``to_gate_logits`` rules for the LTX-2.3
    # gated-attention path are inserted at the front of this dict in
    # ``__post_init__`` only when ``apply_gated_attention=True``.  Without
    # that flag the target model has no ``to_gate_logits`` slot, *and* the
    # same-named ``to_gate_compress`` already lives on the LTX-2.0 VSA-QAT
    # gate path (plus it is in the default ``lora_target_modules`` list).
    # Applying the rename unconditionally silently breaks LTX-2.0 VSA
    # checkpoints and default-target LoRAs.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^model\.diffusion_model\.(.*)$": r"model.\1",
            r"^diffusion_model\.(.*)$": r"model.\1",
            r"^model\.(.*)$": r"model.\1",
            r"^(.*)$": r"model.\1",
        })

    reverse_param_names_mapping: dict = field(default_factory=lambda: {})
    lora_param_names_mapping: dict = field(default_factory=lambda: {})

    # Core transformer settings (defaults from LTX-2 metadata)
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    num_layers: int = 48
    cross_attention_dim: int = 4096
    caption_channels: int = 3840
    norm_eps: float = 1e-6
    attention_type: str = "default"
    rope_type: str = "split"
    double_precision_rope: bool = True
    # LTX-2.3 gated extensions. All default OFF == LTX-2.0 behavior.
    cross_attention_adaln: bool = False
    caption_proj_before_connector: bool = False

    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: list[int] = field(default_factory=lambda: [20, 2048, 2048])
    timestep_scale_multiplier: int = 1000
    use_middle_indices_grid: bool = True

    # Patchification (video-only path)
    patch_size: tuple[int, int, int] = (1, 1, 1)
    num_channels_latents: int = 128
    in_channels: int | None = None
    out_channels: int | None = None

    # Audio defaults (reserved for joint AV ports)
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_cross_attention_dim: int = 2048
    audio_positional_embedding_max_pos: list[int] = field(default_factory=lambda: [20])
    av_ca_timestep_scale_multiplier: int = 1
    # LTX-2.3 gated self-attention (distinct from the VSA-QAT to_gate_compress
    # gate). Default OFF == LTX-2.0 behavior.
    apply_gated_attention: bool = False

    # Text connector/feature extractor compatibility fields carried in some
    # transformer configs (used by the LTX-2.3 text stack). Defaults match
    # the LTX-2.0 connector layout.
    caption_projection_first_linear: bool = True
    caption_proj_input_norm: bool = True
    caption_projection_second_linear: bool = True
    connector_num_attention_heads: int = 30
    connector_attention_head_dim: int = 128
    connector_num_layers: int = 2
    audio_connector_num_attention_heads: int = 30
    audio_connector_attention_head_dim: int = 128
    audio_connector_num_layers: int = 2

    # STG perturbation block index differs across model versions.
    # LTX-2.0 defaults to block 29; LTX-2.3 (caption_proj_before_connector)
    # uses block 28. ``None`` resolves in __post_init__.
    stg_block_idx: int | None = None

    def __post_init__(self):
        super().__post_init__()
        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        if self.in_channels is None:
            self.in_channels = self.num_channels_latents * patch_volume
        if self.out_channels is None:
            self.out_channels = self.in_channels
        if self.stg_block_idx is None:
            self.stg_block_idx = 28 if self.caption_proj_before_connector else 29

        # LTX-2.3 stores the gated-attention weight under ``to_gate_compress``
        # upstream; FastVideo's internal name is ``to_gate_logits``.  Only
        # enable the rename when the gated path is actually configured: the
        # LTX-2.0 attention module's own ``to_gate_compress`` parameter
        # (created when the backend is ``VIDEO_SPARSE_ATTN``) and the default
        # ``to_gate_compress`` LoRA target both share the upstream name, so
        # an unconditional rename would silently retarget them.  Inserted at
        # the front so first-match-wins matching fires the rename before the
        # generic prefix-strip rules.
        if self.apply_gated_attention:
            gate_rules = {
                r"^model\.diffusion_model\.(.*)\.to_gate_compress\.(.*)$": r"model.\1.to_gate_logits.\2",
                r"^diffusion_model\.(.*)\.to_gate_compress\.(.*)$": r"model.\1.to_gate_logits.\2",
                r"^model\.(.*)\.to_gate_compress\.(.*)$": r"model.\1.to_gate_logits.\2",
                r"^(.*)\.to_gate_compress\.(.*)$": r"model.\1.to_gate_logits.\2",
            }
            self.param_names_mapping = {
                **gate_rules,
                **self.param_names_mapping,
            }


@dataclass
class LTX2VideoConfig(DiTConfig):
    """Main configuration for LTX-2 transformer."""

    arch_config: DiTArchConfig = field(default_factory=LTX2VideoArchConfig)
    prefix: str = "ltx2"
