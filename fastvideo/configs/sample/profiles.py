# SPDX-License-Identifier: Apache-2.0
"""Pipeline profile registry for SamplingParam defaults.

A *profile* is a named collection of default field values that can be
applied to a base ``SamplingParam`` instance, replacing the need for
model-specific ``SamplingParam`` subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineProfile:
    """Named set of default overrides for SamplingParam."""

    name: str
    defaults: dict[str, Any]


_PROFILE_REGISTRY: dict[str, PipelineProfile] = {}


def register_profile(profile: PipelineProfile) -> None:
    """Register a profile by name."""
    _PROFILE_REGISTRY[profile.name] = profile


def get_profile(name: str) -> PipelineProfile | None:
    """Look up a profile by name."""
    return _PROFILE_REGISTRY.get(name)
