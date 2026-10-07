"""Shared contract between DreamVerse generation backends and GPU workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class StepResult:
    """Decoded media and stream-trimming metadata for one DreamVerse segment."""

    frames: list
    audio: Any
    audio_sample_rate: int | None
    timings: dict[str, float]
    head_trim_frames: int
    head_trim_audio_frames: int


class GenerationBackend(Protocol):
    """Model-owned generation operations used by one GPU worker process."""

    def initialize(self, model_config: dict | None = None) -> None:
        ...

    def shutdown(self) -> None:
        ...

    def clear_conditioning(self) -> None:
        ...

    def generate_step(
        self,
        prompt: str,
        segment_idx: int,
        image_path: str | None,
        reset_conditioning: bool,
    ) -> StepResult:
        ...

    def warmup(self, prompt: str) -> dict[str, float]:
        ...

    def apply_lora_stack(self, stack: list[tuple[str, float]]) -> tuple[str | None, str | None]:
        ...
