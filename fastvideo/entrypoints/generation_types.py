# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class OutputOptions:
    """Controls what `VideoGenerator.generate_video()` returns and saves.

    Notes:
    - This is intentionally separate from `SamplingParam` (sampling settings).
    - When `return_format="legacy"`, `return_frames=True` preserves the old
      behavior of returning a list of frames instead of a result object.
    - When `return_format="dataclass"`, `return_frames` only controls whether
      frames are produced; the return type is still `GenerationResult`.
    """

    save_video: bool | None = None
    return_frames: bool = False
    include_frames: bool = False
    include_samples: bool = False
    include_trajectory_latents: bool = False
    include_trajectory_decoded: bool = False


@dataclass
class GenerationResult:
    output_path: str
    prompt: str
    size: tuple[int, int, int]
    generation_time: float
    logging_info: Any

    frames: list[np.ndarray] | None = None
    samples: torch.Tensor | None = None
    trajectory_latents: torch.Tensor | None = None
    trajectory_timesteps: list[torch.Tensor] | None = None
    trajectory_decoded: list[torch.Tensor] | None = None

    def to_legacy_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "output_path": self.output_path,
            "prompt": self.prompt,
            "prompts": self.prompt,
            "size": self.size,
            "generation_time": self.generation_time,
            "logging_info": self.logging_info,
        }
        if self.frames is not None:
            out["frames"] = self.frames
        if self.samples is not None:
            out["samples"] = self.samples
        if self.trajectory_latents is not None:
            out["trajectory"] = self.trajectory_latents
            out["trajectory_timesteps"] = self.trajectory_timesteps
        if self.trajectory_decoded is not None:
            out["trajectory_decoded"] = self.trajectory_decoded
        return out
