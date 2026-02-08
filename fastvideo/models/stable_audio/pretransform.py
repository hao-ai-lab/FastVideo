# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio Oobleck VAE pretransform for FastVideo.

Loads from unified model.safetensors using pretransform.model.* prefix.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import torch
import torch.nn as nn

from fastvideo.logger import init_logger

logger = init_logger(__name__)

PRETRANSFORM_KEY_PREFIX = "pretransform.model."


def _get_stable_audio_tools():
    """Ensure stable-audio-tools is importable."""
    proj_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    sat_path = os.path.join(proj_root, "stable-audio-tools")
    if os.path.isdir(sat_path) and sat_path not in sys.path:
        sys.path.insert(0, sat_path)


def _create_pretransform_from_config(model_config: dict) -> nn.Module:
    """Create pretransform from stable-audio-tools model_config.json structure."""
    _get_stable_audio_tools()
    from stable_audio_tools.models.factory import create_pretransform_from_config

    pretransform_cfg = model_config.get("model", model_config).get("pretransform")
    if pretransform_cfg is None:
        raise ValueError("model_config must contain model.pretransform")
    sample_rate = model_config.get("sample_rate", 44100)
    return create_pretransform_from_config(pretransform_cfg, sample_rate)


class StableAudioPretransform(nn.Module):
    """
    FastVideo wrapper for Stable Audio Oobleck VAE pretransform.

    Loads from unified model.safetensors using pretransform.model.* prefix.
    Delegates to stable-audio-tools AutoencoderPretransform.
    """

    def __init__(
        self,
        model_config: dict | str | None = None,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        if isinstance(model_config, str):
            with open(model_config) as f:
                model_config = json.load(f)
        if model_config is None:
            model_config = {}
        self._pretransform = _create_pretransform_from_config(model_config)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_from_unified_checkpoint(checkpoint_path)

    def load_from_unified_checkpoint(self, checkpoint_path: str) -> None:
        """Load pretransform weights from unified model.safetensors or model.ckpt."""
        from fastvideo.models.loader.weight_utils import unified_checkpoint_weights_iterator

        state_dict = {}
        for key, tensor in unified_checkpoint_weights_iterator(
            checkpoint_path, to_cpu=True, key_prefix=PRETRANSFORM_KEY_PREFIX
        ):
            # Inner model in AutoencoderPretransform is self.model
            state_dict[key] = tensor

        # AutoencoderPretransform.load_state_dict passes to self.model
        self._pretransform.load_state_dict(state_dict, strict=True)
        logger.info("Loaded pretransform from %s (%d keys)", checkpoint_path, len(state_dict))

    @property
    def downsampling_ratio(self) -> int:
        return self._pretransform.downsampling_ratio

    @property
    def encoded_channels(self) -> int:
        return self._pretransform.encoded_channels

    @property
    def io_channels(self) -> int:
        return self._pretransform.io_channels

    @property
    def scale(self) -> float:
        return getattr(self._pretransform, "scale", 1.0)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._pretransform.encode(x, **kwargs)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._pretransform.decode(z, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode for compatibility."""
        return self.encode(x, **kwargs)
