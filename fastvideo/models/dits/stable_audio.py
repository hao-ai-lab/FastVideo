# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio DiT model for FastVideo.

Wraps stable-audio-tools DiffusionTransformer for loading from unified
model.safetensors checkpoint.
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn

from fastvideo.configs.models.dits.stable_audio import (
    StableAudioDiTArchConfig,
    StableAudioDiTConfig,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _get_stable_audio_tools_transformer():
    """Import DiffusionTransformer from stable-audio-tools."""
    try:
        from stable_audio_tools.models.dit import DiffusionTransformer
        return DiffusionTransformer
    except ImportError:
        pass
    # Try project-local stable-audio-tools
    import sys
    proj_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    sat_path = os.path.join(proj_root, "stable-audio-tools")
    if os.path.isdir(sat_path) and sat_path not in sys.path:
        sys.path.insert(0, sat_path)
    from stable_audio_tools.models.dit import DiffusionTransformer
    return DiffusionTransformer


def _hf_config_to_dit_kwargs(config: dict) -> dict:
    """Map HF transformer config to stable-audio-tools DiffusionTransformer kwargs."""
    embed_dim = config.get("global_states_input_dim", 1536)
    return {
        "io_channels": config.get("in_channels", 64),
        "embed_dim": embed_dim,
        "cond_token_dim": config.get("cross_attention_dim", 768),
        "project_cond_tokens": False,
        "global_cond_dim": embed_dim,
        "project_global_cond": False,
        "depth": config.get("num_layers", 24),
        "num_heads": config.get("num_attention_heads", 24),
        "patch_size": 1,
    }


class StableAudioDiTModel(nn.Module):
    """
    FastVideo wrapper for Stable Audio DiffusionTransformer.

    Loads from unified model.safetensors using param_names_mapping to strip
    the model.model. prefix. Compatible with stable-audio-tools checkpoints.
    """

    param_names_mapping = StableAudioDiTConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = (
        StableAudioDiTConfig().arch_config.reverse_param_names_mapping
    )
    lora_param_names_mapping = (
        StableAudioDiTConfig().arch_config.lora_param_names_mapping
    )

    def __init__(self, config: dict | StableAudioDiTConfig, hf_config: dict | None = None):
        super().__init__()
        if isinstance(config, StableAudioDiTConfig):
            arch = config.arch_config
            embed_dim = getattr(arch, "global_states_input_dim", None) or getattr(
                arch, "hidden_size", 1536
            )
            kwargs = {
                "io_channels": getattr(arch, "in_channels", 64),
                "embed_dim": embed_dim,
                "cond_token_dim": getattr(arch, "cross_attention_dim", 768),
                "project_cond_tokens": False,
                "global_cond_dim": embed_dim,
                "project_global_cond": False,
                "depth": getattr(arch, "num_layers", 24),
                "num_heads": getattr(arch, "num_attention_heads", 24),
                "patch_size": 1,
            }
        else:
            kwargs = _hf_config_to_dit_kwargs(config)
        DiffusionTransformer = _get_stable_audio_tools_transformer()
        self.model = DiffusionTransformer(**kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: dict | None = None,
        **kwargs,
    ) -> "StableAudioDiTModel":
        """Create from HF-style model path."""
        config_path = os.path.join(model_path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)
        config = config or {}
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        return cls(config=config, **kwargs)


EntryClass = StableAudioDiTModel
