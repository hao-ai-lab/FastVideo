# SPDX-License-Identifier: Apache-2.0
"""
Profile-based defaults for SamplingParam.

A ModelProfile captures the recommended sampling defaults for a specific
model variant (resolution, fps, guidance scale, etc.) without requiring
a dedicated SamplingParam subclass.  ``SamplingParam.from_pretrained``
resolves the profile via the registry and applies its ``defaults`` dict
with simple ``setattr`` calls on a base ``SamplingParam`` instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Global registry: profile name -> ModelProfile
_PROFILE_REGISTRY: dict[str, ModelProfile] = {}


@dataclass
class ModelProfile:
    """Declarative bag of sampling defaults for one model variant."""

    name: str
    defaults: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _PROFILE_REGISTRY[self.name] = self


def get_profile(name: str) -> ModelProfile | None:
    """Look up a registered profile by name."""
    return _PROFILE_REGISTRY.get(name)
