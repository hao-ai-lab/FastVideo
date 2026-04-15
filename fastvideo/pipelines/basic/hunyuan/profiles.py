# SPDX-License-Identifier: Apache-2.0
"""Pipeline profiles for Hunyuan model family.

Each profile defines default sampling parameters that differ from the
base ``SamplingParam`` defaults.  The registry points a model to its
``default_profile`` name, and ``SamplingParam._from_profile`` applies
the profile's ``defaults`` dict onto a freshly-constructed base
instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProfileEntry:
    """Immutable description of a pipeline profile."""

    defaults: dict[str, Any]


# Hunyuan base: all fields match SamplingParam defaults, so the
# defaults dict is empty.  The profile still exists so that the
# registry can reference it.
HUNYUAN_T2V = ProfileEntry(defaults={})

# FastHunyuan: only num_inference_steps differs from base.
FAST_HUNYUAN_T2V = ProfileEntry(defaults={
    "num_inference_steps": 6,
})

# Name -> ProfileEntry lookup used by SamplingParam._from_profile.
PROFILES: dict[str, ProfileEntry] = {
    "hunyuan_t2v": HUNYUAN_T2V,
    "fast_hunyuan_t2v": FAST_HUNYUAN_T2V,
}
