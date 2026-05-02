# SPDX-License-Identifier: Apache-2.0
"""Config for the Stable Audio Open 1.0 multi-conditioner.

The conditioner bundles three sub-conditioners — a T5 text encoder
(prompt) and two NumberConditioners (`seconds_start` / `seconds_total`)
— into the (cross_attn_cond, cross_attn_mask, global_embed) triple the
DiT consumes. The architecture is fully specified by the official
`stable_audio_tools` `MultiConditioner` config; the constants here
mirror that.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.base import ArchConfig
from fastvideo.configs.models.encoders.base import (EncoderArchConfig, EncoderConfig)


def _default_configs() -> list[dict]:
    """Default = `stable-audio-open-1.0`'s three sub-conditioners."""
    return [
        {
            "id": "prompt",
            "type": "t5",
            "config": {
                "t5_model_name": "t5-base",
                "max_length": 128
            }
        },
        {
            "id": "seconds_start",
            "type": "number",
            "config": {
                "min_val": 0,
                "max_val": 512
            }
        },
        {
            "id": "seconds_total",
            "type": "number",
            "config": {
                "min_val": 0,
                "max_val": 512
            }
        },
    ]


@dataclass
class StableAudioConditionerArchConfig(EncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["StableAudioMultiConditioner"])

    # Shared embedding width across all sub-conditioners (T5 last-hidden
    # dim and NumberEmbedder feature dim both = `cond_dim`).
    cond_dim: int = 768

    # Sub-conditioner identifiers. Order in `cross_attention_cond_ids`
    # is the concat order for the cross-attn token sequence; order in
    # `global_cond_ids` is the concat order for the global FiLM-style
    # embedding.
    cross_attention_cond_ids: tuple[str, ...] = ("prompt", "seconds_start", "seconds_total")
    global_cond_ids: tuple[str, ...] = ("seconds_start", "seconds_total")

    # Per-sub-conditioner spec list (mirrors upstream
    # `model_config.json.model.conditioning.configs`). Each entry is
    # `{"id": ..., "type": "t5"|"number", "config": {...}}`. The default
    # matches `stable-audio-open-1.0`; SA-small overrides via the
    # `conditioner/config.json` shipped in the converted repo.
    configs: list = field(default_factory=_default_configs)

    # Match official `stable_audio_tools/models/conditioners.py:334`:
    # T5 is loaded directly in fp16.
    t5_dtype: str = "float16"


@dataclass
class StableAudioConditionerConfig(EncoderConfig):
    arch_config: ArchConfig = field(default_factory=StableAudioConditionerArchConfig)

    prefix: str = "stable_audio_conditioner"
