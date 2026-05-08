# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping

from fastvideo.api.schema import ContinuationState


@dataclass
class GenerationResult:
    prompt: str | None = None
    prompt_index: int | None = None
    samples: Any | None = None
    frames: Any | None = None
    audio: Any | None = None
    audio_sample_rate: int | None = None
    size: tuple[int, int, int] | None = None
    generation_time: float | None = None
    logging_info: Any | None = None
    trajectory: Any | None = None
    trajectory_timesteps: Any | None = None
    trajectory_decoded: Any | None = None
    video_path: str | None = None
    peak_memory_mb: float | None = None
    state: ContinuationState | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy_result(
        cls,
        result: Mapping[str, Any],
    ) -> GenerationResult:
        prompt = result.get("prompt")
        if prompt is None:
            prompt = result.get("prompts")

        extra = {
            key: value
            for key, value in result.items() if key not in {
                "prompt",
                "prompt_index",
                "prompts",
                "samples",
                "frames",
                "audio",
                "audio_sample_rate",
                "size",
                "generation_time",
                "logging_info",
                "trajectory",
                "trajectory_timesteps",
                "trajectory_decoded",
                "video_path",
                "peak_memory_mb",
                "state",
            }
        }

        return cls(
            prompt=prompt,
            prompt_index=result.get("prompt_index"),
            samples=result.get("samples"),
            frames=result.get("frames"),
            audio=result.get("audio"),
            audio_sample_rate=result.get("audio_sample_rate"),
            size=result.get("size"),
            generation_time=result.get("generation_time"),
            logging_info=result.get("logging_info"),
            trajectory=result.get("trajectory"),
            trajectory_timesteps=result.get("trajectory_timesteps"),
            trajectory_decoded=result.get("trajectory_decoded"),
            video_path=result.get("video_path"),
            peak_memory_mb=result.get("peak_memory_mb"),
            state=result.get("state"),
            extra=extra,
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        result = {
            "prompts": self.prompt,
            "samples": self.samples,
            "frames": self.frames,
            "audio": self.audio,
            "audio_sample_rate": self.audio_sample_rate,
            "size": self.size,
            "generation_time": self.generation_time,
            "logging_info": self.logging_info,
            "trajectory": self.trajectory,
            "trajectory_timesteps": self.trajectory_timesteps,
            "trajectory_decoded": self.trajectory_decoded,
            "video_path": self.video_path,
            "peak_memory_mb": self.peak_memory_mb,
        }
        if self.prompt_index is not None:
            result["prompt_index"] = self.prompt_index
            result["prompt"] = self.prompt
        if self.state is not None:
            result["state"] = self.state
        result.update(self.extra)
        return result


__all__ = ["GenerationResult"]
