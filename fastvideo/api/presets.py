# SPDX-License-Identifier: Apache-2.0
"""Pipeline preset registry.

A *preset* is a named inference preset for a model family. It bundles:

  * ``defaults`` — sampling values applied when the user does not
    override them (consumed at runtime via ``SamplingParam.from_pretrained``);
  * ``stage_schemas`` — **validation-only** metadata describing which
    user-facing stage names (``"denoise"``, ``"sr"``) the preset recognises
    and which ``stage_overrides`` keys each stage accepts.

The ``stage_schemas`` tuple does **not** drive pipeline execution. The
concrete execution DAG (text encoding, denoising, VAE decoding, …) is
hard-coded per-pipeline in ``create_pipeline_stages()``. Schemas exist
purely so that ``PipelineSelection.preset`` and
``GenerationRequest.stage_overrides`` can be type-checked up front
without touching the pipeline.

Preset base types and the registry API live here (public API surface).
Preset *instances* are defined in pipeline-local ``presets.py`` files
(e.g. ``fastvideo/pipelines/basic/wan/presets.py``) and registered
explicitly from :func:`_register_presets` in ``fastvideo/registry.py``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fastvideo.api.errors import ConfigValidationError

# -------------------------------------------------------------------
# Types
# -------------------------------------------------------------------


@dataclass(frozen=True)
class PresetStageSpec:
    """A user-facing stage name within a preset, used only to validate
    ``stage_overrides`` keys. Not read by pipeline execution — the real
    execution DAG lives in each pipeline's ``create_pipeline_stages()``.
    """

    name: str
    """Short user-facing name, e.g. ``"denoise"``, ``"sr"``."""

    kind: str
    """Semantic kind, e.g. ``"denoising"``, ``"super_resolution"``."""

    description: str = ""

    allowed_overrides: frozenset[str] = field(default_factory=frozenset)
    """Keys that may appear in ``stage_overrides[name]``."""


@dataclass(frozen=True)
class InferencePreset:
    """A named inference preset for a model family."""

    name: str
    """Preset name, e.g. ``"wan_t2v_1_3b"``."""

    version: int
    """Preset schema version; bump on breaking schema changes."""

    model_family: str
    """Model family key, e.g. ``"wan"``, ``"ltx2"``."""

    description: str = ""

    workload_type: str | None = None
    """Optional workload hint: ``"t2v"``, ``"i2v"``, etc."""

    stage_schemas: tuple[PresetStageSpec, ...] = ()
    """User-facing stage names for ``stage_overrides`` validation.

    Validation-only: this tuple is consumed by
    :func:`validate_stage_overrides` and is **not** used to drive
    pipeline execution. Omit or leave empty if the preset exposes no
    per-stage override surface.
    """

    defaults: dict[str, Any] = field(default_factory=dict)
    """Preset-level default sampling/runtime values."""

    stage_defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-stage default overrides, keyed by stage name."""


# -------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------

# Keyed by (model_family, name, version).
_PRESET_REGISTRY: dict[tuple[str, str, int], InferencePreset] = {}


def register_preset(preset: InferencePreset) -> None:
    """Register a preset definition.

    Raises :class:`ValueError` on duplicate
    ``(model_family, name, version)`` keys.
    """
    key = (preset.model_family, preset.name, preset.version)
    if key in _PRESET_REGISTRY:
        raise ValueError(f"Duplicate preset registration: "
                         f"model_family={key[0]!r}, name={key[1]!r}, "
                         f"version={key[2]!r}")
    _PRESET_REGISTRY[key] = preset


def get_preset(
    name: str,
    model_family: str,
    version: int | None = None,
) -> InferencePreset:
    """Look up a registered preset.

    When *version* is ``None`` the highest registered version for the
    given *(model_family, name)* pair is returned.

    Raises :class:`~fastvideo.api.errors.ConfigValidationError` when the
    preset cannot be found.
    """
    if version is not None:
        key = (model_family, name, version)
        preset = _PRESET_REGISTRY.get(key)
        if preset is not None:
            return preset
        raise ConfigValidationError(
            "pipeline.preset",
            f"unknown preset {name!r} version {version!r} "
            f"for model family {model_family!r}; "
            f"registered: {_format_registered(model_family)}",
        )

    # Find the highest version for (model_family, name).
    candidates = [prof for (fam, n, _v), prof in _PRESET_REGISTRY.items() if fam == model_family and n == name]
    if not candidates:
        raise ConfigValidationError(
            "pipeline.preset",
            f"unknown preset {name!r} for model family "
            f"{model_family!r}; "
            f"registered: {_format_registered(model_family)}",
        )
    return max(candidates, key=lambda p: p.version)


def get_presets_for_family(model_family: str, ) -> list[InferencePreset]:
    """Return all presets registered for *model_family*."""
    return [prof for (fam, _n, _v), prof in _PRESET_REGISTRY.items() if fam == model_family]


def get_all_preset_names() -> list[str]:
    """Return the sorted list of all registered preset names."""
    return sorted({prof.name for prof in _PRESET_REGISTRY.values()})


# -------------------------------------------------------------------
# Validation helpers
# -------------------------------------------------------------------


def validate_stage_names(
    preset: InferencePreset,
    stage_overrides: Mapping[str, Any],
) -> None:
    """Check that *stage_overrides* keys are valid stage names.

    Raises :class:`~fastvideo.api.errors.ConfigValidationError` with a
    path-qualified message for unknown stage names.
    """
    valid_names = {stage.name for stage in preset.stage_schemas}
    for stage_name in stage_overrides:
        if stage_name not in valid_names:
            raise ConfigValidationError(
                f"stage_overrides.{stage_name}",
                f"unknown stage for preset {preset.name!r}; "
                f"valid stages: {sorted(valid_names)}",
            )


def validate_stage_overrides(
    preset: InferencePreset,
    stage_overrides: Mapping[str, Any],
) -> None:
    """Validate stage override keys against the preset.

    Calls :func:`validate_stage_names` first, then checks that each
    override key is in the stage's ``allowed_overrides``.
    """
    validate_stage_names(preset, stage_overrides)
    stages_by_name = {stage.name: stage for stage in preset.stage_schemas}
    for stage_name, overrides in stage_overrides.items():
        if not isinstance(overrides, Mapping):
            raise ConfigValidationError(
                f"stage_overrides.{stage_name}",
                "must be a mapping",
            )
        stage_spec = stages_by_name[stage_name]
        if not stage_spec.allowed_overrides:
            if overrides:
                raise ConfigValidationError(
                    f"stage_overrides.{stage_name}",
                    f"stage {stage_name!r} does not accept "
                    f"overrides",
                )
            continue
        for key in overrides:
            if key not in stage_spec.allowed_overrides:
                raise ConfigValidationError(
                    f"stage_overrides.{stage_name}.{key}",
                    f"not an allowed override for stage "
                    f"{stage_name!r}; allowed: "
                    f"{sorted(stage_spec.allowed_overrides)}",
                )


def validate_preset_selection(
    preset_name: str | None,
    model_family: str,
    *,
    preset_version: int | None = None,
    stage_overrides: Mapping[str, Any] | None = None,
) -> InferencePreset | None:
    """Resolve and validate a preset selection end-to-end.

    Returns the resolved :class:`InferencePreset`, or ``None`` if
    *preset_name* is ``None`` (no preset requested).
    """
    if preset_name is None:
        return None
    preset = get_preset(preset_name, model_family, version=preset_version)
    if stage_overrides:
        validate_stage_overrides(preset, stage_overrides)
    return preset


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _format_registered(model_family: str) -> str:
    names = sorted({prof.name for (fam, _n, _v), prof in _PRESET_REGISTRY.items() if fam == model_family})
    if not names:
        return "(none)"
    return ", ".join(repr(n) for n in names)


__all__ = [
    "InferencePreset",
    "PresetStageSpec",
    "get_all_preset_names",
    "get_preset",
    "get_presets_for_family",
    "register_preset",
    "validate_preset_selection",
    "validate_stage_names",
    "validate_stage_overrides",
]
