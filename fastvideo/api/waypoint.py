# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for Waypoint-1-Small world model."""

from dataclasses import dataclass

from fastvideo.api.sampling_param import SamplingParam


@dataclass
class WaypointSamplingParam(SamplingParam):
    """Waypoint-1-Small sampling defaults."""

    height: int = 360
    width: int = 640
    num_frames: int = 240
    fps: int = 60

    num_inference_steps: int = 4
    guidance_scale: float = 1.0

    negative_prompt: str | None = None
    video_quality: int = 8
