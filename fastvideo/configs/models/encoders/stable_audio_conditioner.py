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


@dataclass
class StableAudioConditionerArchConfig(EncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["StableAudioMultiConditioner"])

    # Shared embedding width across all sub-conditioners (T5 last-hidden
    # dim and NumberEmbedder feature dim both = `cond_dim`).
    cond_dim: int = 768

    # Sub-conditioner identifiers. Order in `cross_attention_cond_ids`
    # is the concat order for the cross-attn token sequence; order in
    # `global_cond_ids` is the concat order for the global FiLM-style
    # embedding. Renaming a key here requires the matching change in
    # `StableAudioConditioningStage` (which builds the `cond_meta`
    # dicts).
    cross_attention_cond_ids: tuple[str, ...] = ("prompt", "seconds_start", "seconds_total")
    global_cond_ids: tuple[str, ...] = ("seconds_start", "seconds_total")

    # T5 prompt encoder. `t5_max_length=128` matches the official repo's
    # tokenizer truncation (NOT the standard 512); changing this breaks
    # numerical parity with upstream.
    t5_model_name: str = "t5-base"
    t5_max_length: int = 128
    # Match official `stable_audio_tools/models/conditioners.py:334`:
    # T5 is loaded directly in fp16.
    t5_dtype: str = "float16"

    # NumberConditioner clamping range for `seconds_start` and
    # `seconds_total` (max ~47.5s for the published 1.0 model, but the
    # range allows up-to-`max_val`-second clamping so out-of-band values
    # don't wrap).
    number_min_val: float = 0.0
    number_max_val: float = 512.0


@dataclass
class StableAudioConditionerConfig(EncoderConfig):
    arch_config: ArchConfig = field(default_factory=StableAudioConditionerArchConfig)

    prefix: str = "stable_audio_conditioner"
