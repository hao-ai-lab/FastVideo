"""Slim vendored API config surface for the v2 VideoGenerator.

Only the inference-config dataclasses (schema) + result types are vendored. The fastvideo
parser / presets / overrides modules are intentionally NOT vendored — they pull the fastvideo
pipeline runtime, which v2 replaces. See v2/README.md (vendoring)."""
from __future__ import annotations

from v2.api.results import GenerationResult
from v2.api.schema import (
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    OffloadConfig,
    OutputConfig,
    SamplingConfig,
)

__all__ = [
    "EngineConfig", "GenerationRequest", "GeneratorConfig", "OffloadConfig", "OutputConfig", "SamplingConfig",
    "GenerationResult",
]
