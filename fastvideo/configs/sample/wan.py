# SPDX-License-Identifier: Apache-2.0
"""Wan-family sampling parameter classes.

Wan model-specific SamplingParam subclasses have been removed.
Defaults are now provided by pipeline profiles in
``fastvideo/pipelines/basic/wan/profiles.py``.

MatrixGame2_SamplingParam remains here temporarily because the
matrixgame model family is not yet profile-migrated.
"""
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class MatrixGame2_SamplingParam(SamplingParam):
    height: int = 352
    width: int = 640
    num_frames: int = 57
    fps: int = 25
    guidance_scale: float = 1.0
    num_inference_steps: int = 3
    negative_prompt: str | None = None
