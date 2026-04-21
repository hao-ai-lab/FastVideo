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

    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    image_crf: int | None = None
    video_position_offset_sec: float | None = None


def refine_override_to_dict(override: LTX2RefinePresetOverride | LTX2RefineStageOverride, ) -> dict[str, Any]:
    """Serialise a refine override, dropping ``None`` entries so only
    user-set fields reach ``preset_overrides.refine`` or
    ``stage_overrides.refine``."""
    return {k: v for k, v in asdict(override).items() if v is not None}


REFINE_PRESET_OVERRIDE_FIELDS: frozenset[str] = frozenset(f.name for f in fields(LTX2RefinePresetOverride))
REFINE_STAGE_OVERRIDE_FIELDS: frozenset[str] = frozenset(f.name for f in fields(LTX2RefineStageOverride))
REFINE_FLAT_KEYS: frozenset[str] = (REFINE_PRESET_OVERRIDE_FIELDS | REFINE_STAGE_OVERRIDE_FIELDS)

__all__ = [
    "LTX2RefinePresetOverride",
    "LTX2RefineStageOverride",
    "REFINE_FLAT_KEYS",
    "REFINE_PRESET_OVERRIDE_FIELDS",
    "REFINE_STAGE_OVERRIDE_FIELDS",
    "refine_override_to_dict",
]
