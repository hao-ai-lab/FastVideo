# SPDX-License-Identifier: Apache-2.0
"""Typed override surfaces for the LTX-2 two-stage refine flow.

* ``preset_overrides.refine`` — init-time knobs (see
  :class:`LTX2RefinePresetOverride`).
* ``stage_overrides.refine`` — per-request knobs (see
  :class:`LTX2RefineStageOverride`).

Asset paths live on :class:`~fastvideo.api.schema.ComponentConfig`
(``upsampler_weights`` and ``lora_path``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class LTX2RefinePresetOverride:
    """Init-time refine wiring under ``preset_overrides.refine``."""

    enabled: bool | None = None
    add_noise: bool | None = None


@dataclass
class LTX2RefineStageOverride:
    """Per-request refine tuning under ``stage_overrides.refine``."""

    # Stage-2 refine only validates 2 (reduced) and 3 (official distilled)
    # sigma schedules; other values raise at pipeline construction.
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    image_crf: int | None = None
    video_position_offset_sec: float | None = None


def refine_override_to_dict(override: LTX2RefinePresetOverride | LTX2RefineStageOverride, ) -> dict[str, Any]:
    """Serialise a refine override, dropping ``None`` entries so only
    user-set fields reach ``preset_overrides.refine`` or
    ``stage_overrides.refine``."""
    return {k: v for k, v in asdict(override).items() if v is not None}


def refine_preset_override_fields() -> frozenset[str]:
    return frozenset(f.name for f in fields(LTX2RefinePresetOverride))


def refine_stage_override_fields() -> frozenset[str]:
    return frozenset(f.name for f in fields(LTX2RefineStageOverride))


__all__ = [
    "LTX2RefinePresetOverride",
    "LTX2RefineStageOverride",
    "refine_override_to_dict",
    "refine_preset_override_fields",
    "refine_stage_override_fields",
]
