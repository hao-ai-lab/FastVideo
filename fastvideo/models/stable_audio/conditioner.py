# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio conditioner: T5 (prompt) + Number (seconds_start, seconds_total).

Uses inlined conditioners (t5 + number only). Loads weights from unified
model.safetensors using conditioner.conditioners.*
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from fastvideo.logger import init_logger
from fastvideo.models.stable_audio.conditioners_inline import (
    create_multi_conditioner_from_conditioning_config,
)

logger = init_logger(__name__)

CONDITIONER_KEY_PREFIX = "conditioner.conditioners."


def _create_multi_conditioner_from_config(model_config: dict) -> nn.Module:
    """Create MultiConditioner from model_config (t5 + number only, no clone)."""
    model_cfg = model_config.get("model", model_config)
    conditioning_config = model_cfg.get("conditioning")
    if conditioning_config is None:
        raise ValueError("model_config must contain model.conditioning")
    return create_multi_conditioner_from_conditioning_config(conditioning_config)


class StableAudioConditioner(nn.Module):
    """
    FastVideo wrapper for Stable Audio conditioner.

    - prompt: T5-base (external, from HuggingFace)
    - seconds_start, seconds_total: NumberEmbedder (loaded from checkpoint)

    Loads conditioner weights from unified model.safetensors.
    """

    def __init__(
        self,
        model_config: dict | str | None = None,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        if isinstance(model_config, str):
            import json
            with open(model_config) as f:
                model_config = json.load(f)
        if model_config is None:
            model_config = {}
        self._conditioner = _create_multi_conditioner_from_config(model_config)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_from_unified_checkpoint(checkpoint_path)

    def load_from_unified_checkpoint(self, checkpoint_path: str) -> None:
        """Load conditioner weights (seconds_start, seconds_total) from unified checkpoint."""
        from fastvideo.models.loader.weight_utils import unified_checkpoint_weights_iterator

        state_dict = {}
        for key, tensor in unified_checkpoint_weights_iterator(
            checkpoint_path, to_cpu=True, key_prefix=CONDITIONER_KEY_PREFIX
        ):
            # Checkpoint: conditioner.conditioners.seconds_start.embedder.xxx
            # After strip: seconds_start.embedder.xxx
            # MultiConditioner has self.conditioners = ModuleDict, so state_dict keys need "conditioners." prefix
            state_dict["conditioners." + key] = tensor

        self._conditioner.load_state_dict(state_dict, strict=False)
        loaded = len(state_dict)
        logger.info("Loaded conditioner from %s (%d keys)", checkpoint_path, loaded)

    def forward(
        self,
        batch_metadata: list[dict[str, Any]],
        device: torch.device | str,
    ) -> dict[str, Any]:
        """Run conditioner on batch metadata. Same interface as stable-audio-tools."""
        return self._conditioner(batch_metadata, device)

    def __call__(
        self,
        batch_metadata: list[dict[str, Any]],
        device: torch.device | str,
    ) -> dict[str, Any]:
        return self.forward(batch_metadata, device)
