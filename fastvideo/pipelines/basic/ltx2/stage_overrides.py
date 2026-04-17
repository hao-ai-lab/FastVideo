# SPDX-License-Identifier: Apache-2.0
"""Typed override surfaces for the LTX-2 two-stage refine flow.

LTX-2 exposes two distinct override paths:

* ``generator.pipeline.preset_overrides.refine`` — **init-time** knobs
  that affect pipeline topology and module loading. Consumed once when
  :class:`fastvideo.entrypoints.video_generator.VideoGenerator` is
  constructed. Rebinding a generator is required to change these.
* ``request.stage_overrides.refine`` — **per-request** tuning knobs
  applied when a :class:`~fastvideo.api.schema.GenerationRequest` is
  normalised. Some of these are still baked into the pipeline during
  :meth:`~fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.create_pipeline_stages`
  today (e.g. stage-2 sigma schedule derives from
  ``num_inference_steps``); they are exposed here so downstream tooling
  can depend on the shape, and so the compat layer has a typed target
  for the legacy ``ltx2_refine_*`` kwargs emitted by the
  FastVideo-internal ltx2-streaming ``gpu_pool``.

Asset wiring lives on :class:`~fastvideo.api.schema.ComponentConfig`
instead: ``components.upsampler_weights`` replaces
``ltx2_refine_upsampler_path`` and ``components.lora_path`` replaces
``ltx2_refine_lora_path``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class LTX2RefinePresetOverride:
    """Init-time refine wiring under ``preset_overrides.refine``.

    These fields control whether the second-stage refine pipeline is
    assembled at all, and how it mixes stage-1 latents into stage-2 via
    Gaussian noise.
    """

    enabled: bool | None = None
    """Toggle the refine stages. Must be ``True`` when combined with the
    ``ltx2_two_stage`` preset for refinement to run; ``None`` / ``False``
    falls back to the stage-1-only flow."""

    add_noise: bool | None = None
    """Whether :class:`LTX2UpsampleStage` mixes noise into the
    upsampled latents before stage-2 denoising. The distilled refine
    path expects ``True``; set ``False`` to pass clean latents straight
    through."""


@dataclass
class LTX2RefineStageOverride:
    """Per-request refine tuning under ``stage_overrides.refine``.

    Fields mirror :class:`PresetStageSpec.allowed_overrides` on the
    ``refine`` stage of the ``ltx2_two_stage`` preset. Unknown fields
    are rejected by
    :func:`fastvideo.api.presets.validate_stage_overrides`.
    """

    num_inference_steps: int | None = None
    """Stage-2 denoising step count. The LTX-2 refine path currently
    only validates ``2`` (reduced schedule) and ``3`` (official
    distilled schedule); other values raise at pipeline construction."""

    guidance_scale: float | None = None
    """Classifier-free guidance scale for stage-2 denoising. Passed as
    ``force_guidance_scale`` to the stage-2 denoiser."""

    image_crf: int | None = None
    """Image-encoding CRF hint for the refine output."""

    video_position_offset_sec: float | None = None
    """Seconds to shift video RoPE positions relative to audio. Used
    by the continuation flow in ltx2-streaming when the audio
    conditioning extends before video t=0."""


def refine_preset_override_to_dict(override: LTX2RefinePresetOverride, ) -> dict[str, Any]:
    """Serialise, dropping ``None`` entries so only user-set fields
    reach ``preset_overrides.refine``."""
    return {k: v for k, v in asdict(override).items() if v is not None}


def refine_stage_override_to_dict(override: LTX2RefineStageOverride, ) -> dict[str, Any]:
    """Serialise, dropping ``None`` entries so only user-set fields
    reach ``stage_overrides.refine``."""
    return {k: v for k, v in asdict(override).items() if v is not None}


def refine_preset_override_fields() -> frozenset[str]:
    """Field names on :class:`LTX2RefinePresetOverride`; used by the
    compat layer to validate flat-kwarg translations."""
    return frozenset(f.name for f in fields(LTX2RefinePresetOverride))


def refine_stage_override_fields() -> frozenset[str]:
    """Field names on :class:`LTX2RefineStageOverride`; used by the
    ``ltx2_two_stage`` preset's stage schema and the compat layer."""
    return frozenset(f.name for f in fields(LTX2RefineStageOverride))


__all__ = [
    "LTX2RefinePresetOverride",
    "LTX2RefineStageOverride",
    "refine_preset_override_fields",
    "refine_preset_override_to_dict",
    "refine_stage_override_fields",
    "refine_stage_override_to_dict",
]
