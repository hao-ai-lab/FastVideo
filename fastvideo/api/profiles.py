# SPDX-License-Identifier: Apache-2.0
"""Pipeline profile registry.

A *profile* is a named inference preset for a model family.  It declares
the user-facing stage topology and which override keys each stage accepts,
enabling typed validation of ``PipelineSelection.profile`` and
``GenerationRequest.stage_overrides`` without touching pipeline execution.

Profile base types and the registry API live here (public API surface).
Profile *instances* are defined in pipeline-local ``profiles.py`` files
(e.g. ``fastvideo/pipelines/basic/wan/profiles.py``) and registered
explicitly from :func:`_register_profiles` in ``fastvideo/registry.py``.
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
class ProfileStageSpec:
    """A user-facing named stage within a profile."""

    name: str
    """Short user-facing name, e.g. ``"denoise"``, ``"sr"``."""

    kind: str
    """Semantic kind, e.g. ``"denoising"``, ``"super_resolution"``."""

    description: str = ""

    allowed_overrides: frozenset[str] = field(default_factory=frozenset)
    """Keys that may appear in ``stage_overrides[name]``."""


@dataclass(frozen=True)
class PipelineProfile:
    """A named inference preset for a model family."""

    name: str
    """Profile name, e.g. ``"wan_t2v_1_3b"``."""

    version: str
    """Profile version string, e.g. ``"1"``."""

    model_family: str
    """Model family key, e.g. ``"wan"``, ``"ltx2"``."""

    description: str = ""

    workload_type: str | None = None
    """Optional workload hint: ``"t2v"``, ``"i2v"``, etc."""

    stages: tuple[ProfileStageSpec, ...] = ()
    """Ordered user-facing stage definitions."""

    defaults: dict[str, Any] = field(default_factory=dict)
    """Profile-level default sampling/runtime values."""

    stage_defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-stage default overrides, keyed by stage name."""


# -------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------

# Keyed by (model_family, name, version).
_PROFILE_REGISTRY: dict[tuple[str, str, str], PipelineProfile] = {}


def register_profile(profile: PipelineProfile) -> None:
    """Register a profile definition.

    Raises :class:`ValueError` on duplicate
    ``(model_family, name, version)`` keys.
    """
    key = (profile.model_family, profile.name, profile.version)
    if key in _PROFILE_REGISTRY:
        raise ValueError(f"Duplicate profile registration: "
                         f"model_family={key[0]!r}, name={key[1]!r}, "
                         f"version={key[2]!r}")
    _PROFILE_REGISTRY[key] = profile


def get_profile(
    name: str,
    model_family: str,
    version: str | None = None,
) -> PipelineProfile:
    """Look up a registered profile.

    When *version* is ``None`` the highest registered version for the
    given *(model_family, name)* pair is returned.

    Raises :class:`~fastvideo.api.errors.ConfigValidationError` when the
    profile cannot be found.
    """
    if version is not None:
        key = (model_family, name, version)
        profile = _PROFILE_REGISTRY.get(key)
        if profile is not None:
            return profile
        raise ConfigValidationError(
            "pipeline.profile",
            f"unknown profile {name!r} version {version!r} "
            f"for model family {model_family!r}; "
            f"registered: {_format_registered(model_family)}",
        )

    # Find the highest version for (model_family, name).
    candidates = [prof for (fam, n, _v), prof in _PROFILE_REGISTRY.items() if fam == model_family and n == name]
    if not candidates:
        raise ConfigValidationError(
            "pipeline.profile",
            f"unknown profile {name!r} for model family "
            f"{model_family!r}; "
            f"registered: {_format_registered(model_family)}",
        )
    candidates.sort(key=lambda p: [int(x) if x.isdigit() else x for x in p.version.split(".")])
    return candidates[-1]


def get_profiles_for_family(model_family: str, ) -> list[PipelineProfile]:
    """Return all profiles registered for *model_family*."""
    return [prof for (fam, _n, _v), prof in _PROFILE_REGISTRY.items() if fam == model_family]


def get_all_profile_names() -> list[str]:
    """Return the sorted list of all registered profile names."""
    return sorted({prof.name for prof in _PROFILE_REGISTRY.values()})


# -------------------------------------------------------------------
# Validation helpers
# -------------------------------------------------------------------


def validate_stage_names(
    profile: PipelineProfile,
    stage_overrides: Mapping[str, Any],
) -> None:
    """Check that *stage_overrides* keys are valid stage names.

    Raises :class:`~fastvideo.api.errors.ConfigValidationError` with a
    path-qualified message for unknown stage names.
    """
    valid_names = {stage.name for stage in profile.stages}
    for stage_name in stage_overrides:
        if stage_name not in valid_names:
            raise ConfigValidationError(
                f"stage_overrides.{stage_name}",
                f"unknown stage for profile {profile.name!r}; "
                f"valid stages: {sorted(valid_names)}",
            )


def validate_stage_overrides(
    profile: PipelineProfile,
    stage_overrides: Mapping[str, Any],
) -> None:
    """Validate stage override keys against the profile.

    Calls :func:`validate_stage_names` first, then checks that each
    override key is in the stage's ``allowed_overrides``.
    """
    validate_stage_names(profile, stage_overrides)
    stages_by_name = {stage.name: stage for stage in profile.stages}
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


def validate_profile_selection(
    profile_name: str | None,
    model_family: str,
    *,
    profile_version: str | None = None,
    stage_overrides: Mapping[str, Any] | None = None,
) -> PipelineProfile | None:
    """Resolve and validate a profile selection end-to-end.

    Returns the resolved :class:`PipelineProfile`, or ``None`` if
    *profile_name* is ``None`` (no profile requested).
    """
    if profile_name is None:
        return None
    profile = get_profile(profile_name, model_family, version=profile_version)
    if stage_overrides:
        validate_stage_overrides(profile, stage_overrides)
    return profile


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _format_registered(model_family: str) -> str:
    names = sorted({prof.name for (fam, _n, _v), prof in _PROFILE_REGISTRY.items() if fam == model_family})
    if not names:
        return "(none)"
    return ", ".join(repr(n) for n in names)


def _clear_registry() -> None:
    """Reset the registry.  **Test-only.**"""
    _PROFILE_REGISTRY.clear()


__all__ = [
    "PipelineProfile",
    "ProfileStageSpec",
    "get_all_profile_names",
    "get_profile",
    "get_profiles_for_family",
    "register_profile",
    "validate_profile_selection",
    "validate_stage_names",
    "validate_stage_overrides",
]
