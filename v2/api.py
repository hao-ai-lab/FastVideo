# STUB: re-exports fastvideo until vendored (see memory: v2-vendoring-approach).
"""``api`` facade — the typed config dataclasses the VideoGenerator consumes. Re-exported so v2 code
imports ``v2.api`` instead of ``fastvideo.api``; a vendored cutover will replace these with v2-native configs."""
from fastvideo.api import (  # noqa: F401
    EngineConfig, GenerationRequest, GenerationResult, GeneratorConfig, OffloadConfig, OutputConfig, SamplingConfig,
)
