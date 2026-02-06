# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from diffusers import SD3Transformer2DModel as _DiffusersSD3Transformer2DModel

from fastvideo.configs.models import DiTConfig


class SD3Transformer2DModel(_DiffusersSD3Transformer2DModel):

    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []
    param_names_mapping: dict[str, Any] = {}
    reverse_param_names_mapping: dict[str, Any] = {}
    lora_param_names_mapping: dict[str, Any] = {}

    def __init__(self, config: DiTConfig, hf_config: dict[str, Any], **kwargs):
        self.fastvideo_config = config
        self.hf_config = hf_config

        arch = config.arch_config
        dual_layers = getattr(arch, "dual_attention_layers", ())
        if isinstance(dual_layers, list):
            dual_layers = tuple(dual_layers)

        super().__init__(
            sample_size=arch.sample_size,
            patch_size=arch.patch_size,
            in_channels=arch.in_channels,
            num_layers=arch.num_layers,
            attention_head_dim=arch.attention_head_dim,
            num_attention_heads=arch.num_attention_heads,
            joint_attention_dim=arch.joint_attention_dim,
            caption_projection_dim=arch.caption_projection_dim,
            pooled_projection_dim=arch.pooled_projection_dim,
            out_channels=arch.out_channels,
            pos_embed_max_size=arch.pos_embed_max_size,
            dual_attention_layers=dual_layers,
            qk_norm=arch.qk_norm,
            **kwargs,
        )
