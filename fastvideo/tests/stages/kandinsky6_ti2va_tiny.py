# SPDX-License-Identifier: Apache-2.0
"""Tiny random Kandinsky6 TI2VA DiT (video+audio, text+pooled conditioning) for CPU tests.

Small enough to run forward passes in milliseconds: 2 fused decoder blocks, 1 text block per tower,
axes_dims (8, 4, 4) -> head_dim 16, model_dim 32.
"""
from __future__ import annotations

import copy
from typing import Any

import torch

TINY_DIT_CFG: dict[str, Any] = dict(
    in_visual_dim=4,
    out_visual_dim=4,
    in_text_dim=8,
    in_text_dim2=8,
    time_dim=16,
    patch_size=(1, 2, 2),
    model_dim=32,
    ff_dim=32,
    num_text_blocks=1,
    num_visual_blocks=2,
    axes_dims=(8, 4, 4),
    visual_cond=True,
    is_multimodal=True,
    in_audio_dim=6,
    # Set explicitly (not left to the `x or time_dim` default-resolution in Kandinsky6ArchConfig.
    # __post_init__): update_model_arch calls __post_init__ twice -- once via the arch_config field's
    # default_factory (which already resolves *_a fields from the *default* time_dim/model_dim/ff_dim/
    # axes_dims, not this dict's), and again after applying this dict -- so an unset *_a field here
    # would silently keep the stale default-resolved value instead of picking up this dict's values.
    model_dim_a=32,
    time_dim_a=16,
    ff_dim_a=32,
    axes_dims_a=(8, 4, 4),
    visual_token_type_num_embeddings=2,
    attention_engine="auto",
)


def dit_config_dict(dit_cfg: dict | None = None) -> dict[str, Any]:
    return copy.deepcopy(dit_cfg or TINY_DIT_CFG)


def randomize(module: torch.nn.Module, seed: int, std: float = 0.05) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * std)
    return module


def build_dit(seed: int = 0, dit_cfg: dict | None = None):
    from fastvideo.configs.models.dits.kandinsky6 import Kandinsky6VideoAudioConfig
    from fastvideo.models.dits.kandinsky6 import Kandinsky6Transformer3DModel

    cfg = dit_config_dict(dit_cfg)
    config = Kandinsky6VideoAudioConfig()
    config.update_model_arch(cfg)
    return randomize(Kandinsky6Transformer3DModel(config, cfg), seed + 2).eval()
