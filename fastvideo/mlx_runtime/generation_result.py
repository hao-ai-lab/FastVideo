# SPDX-License-Identifier: Apache-2.0
"""Generation output shared by native MLX pipelines."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GenerationResult:
    video_path: str | None
    frames: np.ndarray | None = None
    waveform: np.ndarray | None = None
    sample_rate: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    peak_memory_gib: dict[str, float] = field(default_factory=dict)
    vsa: dict[str, Any] = field(default_factory=dict)
    video_decode_backend: str = "h3-vae"
